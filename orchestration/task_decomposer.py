"""Multimodal task decomposition for base/arm runtime orchestration.

This module turns a natural-language user instruction into a validated serial
task list compatible with ``deploy_v7_dual.task_sequence``.

Design goals:
- Reuse the existing arm/base execution capabilities instead of inventing new
  executors.
- Let the program compress navigation/map context before asking an LLM to make
  semantic decisions.
- Reject partial plans: if one step is not executable, the whole task is
  rejected and only diagnostics are returned.
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import logging
import math
import mimetypes
import os
import importlib.util
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import dashscope
import networkx as nx
import numpy as np


LOGGER = logging.getLogger("task_decomposer")

SUPPORTED_EXECUTORS = {"arm", "base"}
DEFAULT_ARM_CAPABILITIES = [
    "Operate only when the mobile base is docked at the workbench.",
    "Perform tabletop pick/place/load actions on cups, plates, trays, and similar objects.",
    "Load target objects onto the mobile base tray for cross-region transport.",
]
DEFAULT_ARM_LIMITATIONS = [
    "Cannot operate outside the workbench region.",
    "Cannot perform cross-region transport by itself.",
    "Uses instruction groups as normalization references rather than an exhaustive natural-language task list.",
]
DEFAULT_BASE_CAPABILITIES = [
    "Navigate to a semantically matched region or target area.",
    "Move the mobile base to a representative docking position near that area.",
    "Transport objects that have already been loaded onto the base tray at the workbench.",
]
DEFAULT_BASE_LIMITATIONS = [
    "Cannot open doors or drawers.",
    "Cannot directly grasp or manipulate objects without a separate arm step at the workbench.",
    "Cannot independently load objects onto or unload objects from the base tray.",
]

EXECUTOR_DESCRIPTIONS = [
    "arm — tabletop manipulator available only in the workbench region; performs pick/place/load actions on workbench and tray objects",
    "base — mobile base that navigates across regions and transports objects that have already been loaded onto its tray",
]
DEFAULT_ARM_CAPABILITY_SUMMARY_SHORT = "\n".join([
    "- arm actions are allowed only in the workbench region.",
    "- the arm handles high-level tabletop actions at the workbench, such as loading an object from the table onto the tray.",
    "- the arm does not perform cross-region transport.",
])
DEFAULT_BASE_CAPABILITY_SUMMARY_SHORT = "\n".join([
    "- the base handles navigation across rooms and regions.",
    "- the base can transport objects that have already been loaded onto its tray by the arm.",
    "- the base cannot grasp, load, or unload objects by itself.",
])


def _load_instruction_groups() -> Dict[str, Any]:
    module_path = Path(__file__).resolve().parents[1] / "mujoco_env" / "instruction_utils.py"
    spec = importlib.util.spec_from_file_location("_task_decomposer_instruction_utils", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load instruction_utils.py: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instruction_groups = getattr(module, "INSTRUCTION_GROUPS", None)
    if not isinstance(instruction_groups, dict):
        raise RuntimeError("instruction_utils.py is missing a valid INSTRUCTION_GROUPS mapping")
    return instruction_groups


INSTRUCTION_GROUPS = _load_instruction_groups()


class DecompositionBackend(Protocol):
    """LLM-facing backend used by the decomposer."""

    def split_instruction(
        self,
        *,
        user_instruction: str,
        executor_hints: Sequence[str],
        arm_capability_summary_short: str,
        base_capability_summary_short: str,
    ) -> List[Dict[str, Any]]:
        """Split the user instruction into ordered executor-tagged clauses."""

    def assess_arm_steps_batch(
        self,
        *,
        user_instruction: str,
        raw_clauses: Sequence[str],
        agent_image_path: str,
        arm_capability_summary: str,
    ) -> List[Dict[str, Any]]:
        """Batch-judge whether multiple arm clauses are executable."""

    def assess_base_steps_batch(
        self,
        *,
        user_instruction: str,
        raw_clauses: Sequence[str],
        navigation_context: Dict[str, Any],
        base_capability_summary: str,
        per_task_candidates: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """Batch-judge whether multiple base clauses are executable."""


@dataclass
class StepDraft:
    step_index: int
    raw_clause: str
    executor: str
    planning_note: str = ""
    depends_on: List[int] = field(default_factory=list)
    skip_if: str = ""


@dataclass
class StepEvaluation:
    step_index: int
    raw_clause: str
    executor: str
    feasible: bool
    normalized_task: Optional[Dict[str, Any]] = None
    reason_code: str = ""
    reason_text: str = ""
    model_output: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskDecompositionResult:
    feasible: bool
    user_instruction: str
    normalized_tasks: List[Dict[str, Any]]
    diagnostics: List[Dict[str, Any]]
    summary_for_user: str
    step_evaluations: List[StepEvaluation] = field(default_factory=list)
    llm_debug_records: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PlannerState:
    base_region_id: str = ""
    base_region_caption: str = ""


@dataclass
class CandidateRegion:
    cluster_id: str
    cluster_caption: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "cluster_caption": self.cluster_caption,
        }


@dataclass
class NavigationSceneContext:
    current_pose_text: str
    nearest_topology_node: str
    current_region: Dict[str, Any]
    candidate_regions: List[CandidateRegion]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_pose_text": self.current_pose_text,
            "nearest_topology_node": self.nearest_topology_node,
            "current_region": dict(self.current_region),
            "candidate_regions": [item.to_dict() for item in self.candidate_regions],
        }


def _safe_json_loads(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        raise RuntimeError("LLM returned an empty response")
    start = text.find("{")
    end = text.rfind("}")
    snippet = text[start : end + 1] if start >= 0 and end >= start else text
    try:
        payload = json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse LLM JSON output: {text}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"LLM output is not a JSON object: {text}")
    return payload


def _image_path_to_data_uri(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        mime_type = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _normalize_executor(value: Any) -> str:
    executor = str(value or "").strip().lower()
    if executor not in SUPPORTED_EXECUTORS:
        return ""
    return executor


def _bool_from_payload(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _reason_code_from_payload(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _make_task_name(executor: str, step_index: int, raw_clause: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in raw_clause.lower()).strip("_")
    slug = "_".join(part for part in slug.split("_") if part)[:40]
    if not slug:
        slug = f"{executor}_step"
    return f"{executor}_{step_index + 1}_{slug}"


def _text_mentions_workbench(text: str) -> bool:
    content = str(text or "").strip()
    content_lower = content.lower()
    keywords = ("workbench", "work bench", "bench", "table")
    return any(keyword in content_lower or keyword in content for keyword in keywords)


def _normalize_raw_clause_text(text: str, executor: str) -> str:
    clause = str(text or "").strip()
    if not clause:
        return ""

    replacements = {
        "move the base to": "go to",
        "move the robot base to": "go to",
        "move the mobile base to": "go to",
        "drive the base to": "go to",
    }
    for src, dst in replacements.items():
        clause = clause.replace(src, dst)

    if executor == "base":
        if clause.startswith("go to go to"):
            clause = "go to" + clause[len("go to go to"):]
        if clause.startswith("go go to"):
            clause = "go to" + clause[len("go go to"):]

    return " ".join(clause.split())


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    content = str(text or "").lower()
    return any(keyword in content for keyword in keywords)


def _split_composite_arm_clause(raw_clause: str) -> List[str]:
    clause = " ".join(str(raw_clause or "").split())
    if not clause:
        return []

    separators = [
        r", then ",
        r" then ",
        r" and then ",
    ]
    first_action_hints = (
        "unload",
        "place",
        "put",
        "set down",
        "drop",
        "unload",
    )
    second_action_hints = (
        "pick",
        "pick up",
        "take",
        "grasp",
        "grab",
        "load",
    )
    tray_hints = ("tray",)
    station_hints = ("workbench", "work bench", "table", "bench")

    for separator in separators:
        parts = re.split(separator, clause, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2:
            continue
        first, second = (part.strip(" ,，。.;；") for part in parts)
        if not first or not second:
            continue
        if not _contains_any(first, first_action_hints):
            continue
        if not _contains_any(second, second_action_hints):
            continue
        if not (_contains_any(clause, tray_hints) and _contains_any(clause, station_hints)):
            continue
        return [f"{first}.", f"{second}."]
    return [clause]


def _expand_split_steps(raw_steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    old_to_new: Dict[int, List[int]] = {}

    for old_idx, item in enumerate(raw_steps, start=1):
        executor = _normalize_executor(item.get("executor"))
        split_clauses = [str(item.get("raw_clause", "")).strip()]
        if executor == "arm":
            candidate = _split_composite_arm_clause(item.get("raw_clause", ""))
            if len(candidate) > 1:
                split_clauses = candidate

        mapped_indices: List[int] = []
        for split_pos, clause in enumerate(split_clauses):
            new_item = dict(item)
            new_item["raw_clause"] = clause
            new_item["_origin_step"] = old_idx
            new_item["_split_pos"] = split_pos
            expanded.append(new_item)
            mapped_indices.append(len(expanded))
        old_to_new[old_idx] = mapped_indices

    remapped: List[Dict[str, Any]] = []
    for item in expanded:
        original_depends = item.get("depends_on", [])
        new_depends: List[int] = []
        if isinstance(original_depends, list):
            for dep in original_depends:
                if isinstance(dep, int) and dep > 0 and dep in old_to_new:
                    new_depends.append(old_to_new[dep][-1])
        if int(item.get("_split_pos", 0)) > 0:
            origin_indices = old_to_new.get(int(item.get("_origin_step", 0)), [])
            if origin_indices:
                new_depends.append(origin_indices[int(item["_split_pos"]) - 1])
        deduped_depends: List[int] = []
        for dep in new_depends:
            if dep not in deduped_depends:
                deduped_depends.append(dep)
        clean_item = {
            key: value
            for key, value in item.items()
            if key not in {"_origin_step", "_split_pos"}
        }
        clean_item["depends_on"] = deduped_depends
        remapped.append(clean_item)
    return remapped


def build_arm_capability_summary(
    instruction_groups: Optional[Sequence[Dict[str, Any]]] = None,
    capabilities: Optional[Sequence[str]] = None,
    limitations: Optional[Sequence[str]] = None,
) -> str:
    groups = list(instruction_groups or INSTRUCTION_GROUPS.get("arm", []))
    caps = list(capabilities or DEFAULT_ARM_CAPABILITIES)
    limits = list(limitations or DEFAULT_ARM_LIMITATIONS)
    lines: List[str] = ["Available arm capabilities:"]
    lines.extend(f"- {item}" for item in caps)
    lines.append("Known arm limitations:")
    lines.extend(f"- {item}" for item in limits)
    lines.append("Arm instruction groups (examples / normalization references):")
    for group in groups:
        group_name = str(group.get("name", "arm_group"))
        instructions = [str(item).strip() for item in group.get("instructions", []) if str(item).strip()]
        if not instructions:
            continue
        lines.append(f"[{group_name}]")
        lines.extend(f"- {item}" for item in instructions)
    return "\n".join(lines)


def build_base_capability_summary(
    capabilities: Optional[Sequence[str]] = None,
    limitations: Optional[Sequence[str]] = None,
) -> str:
    caps = list(capabilities or DEFAULT_BASE_CAPABILITIES)
    limits = list(limitations or DEFAULT_BASE_LIMITATIONS)
    sections = ["Available base capabilities:"]
    sections.extend(f"- {item}" for item in caps)
    sections.append("Known base limitations:")
    sections.extend(f"- {item}" for item in limits)
    return "\n".join(sections)


class DashScopeTaskDecompositionBackend:
    """Default LLM/VLM backend for task decomposition."""

    def __init__(
        self,
        *,
        text_model: str = "qwen3.5-plus-2026-02-15",
        vision_model: str = "qwen-vl-max-latest",
        stream_output: bool = True,
        require_api_key: bool = True,
    ):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if require_api_key and not api_key:
            raise EnvironmentError("Missing environment variable DASHSCOPE_API_KEY; export it before running")
        if api_key:
            dashscope.api_key = api_key
        self.text_model = str(text_model)
        self.vision_model = str(vision_model)
        self.stream_output = bool(stream_output)
        self._debug_records: List[Dict[str, Any]] = []

    def reset_debug_records(self) -> None:
        self._debug_records = []

    def get_debug_records(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._debug_records]

    def _append_debug_record(
        self,
        *,
        stage: str,
        model: str,
        multimodal: bool,
        raw_text: str,
        parsed_payload: Dict[str, Any],
    ) -> None:
        self._debug_records.append(
            {
                "stage": stage,
                "model": model,
                "multimodal": bool(multimodal),
                "raw_text": raw_text,
                "parsed_payload": dict(parsed_payload),
            }
        )

    def split_instruction(
        self,
        *,
        user_instruction: str,
        executor_hints: Sequence[str],
        arm_capability_summary_short: str,
        base_capability_summary_short: str,
    ) -> List[Dict[str, Any]]:
        prompt = (
            "You are a high-level robot task planner. Decompose the user task into a serial plan sketch for downstream reviewers.\n"
            "Do high-level orchestration only. Do not perform scene inspection and do not emit perception-only micro-steps.\n"
            "Planning input summary:\n"
            "arm summary:\n%s\n\n"
            "base summary:\n%s\n\n"
            "Allowed executors:\n%s\n\n"
            "Planning rules:\n"
            "1. Each step must be assigned to exactly one executor: arm or base.\n"
            "2. If the task requires multiple executors, prefer a collaborative chain instead of failing early.\n"
            "3. Generic object references such as 'a cup' mean any instance of that category by default.\n"
            "4. The arm is available only at the workbench. Any high-level arm action must be preceded by a base step that reaches the workbench.\n"
            "5. Do not split 'find / identify / locate an object' into a standalone step. Absorb it into the following arm action.\n"
            "6. Consecutive arm micro-actions on the same object flow may be merged into one high-level step. For example, 'pick up a cup and place it onto the tray' should be treated as one arm step.\n"
            "7. If the task involves both an object already carried on the robot / already on the tray and another object of the same category to be handled afterward, plan the two object flows as separate steps. Do not merge 'unload the carried object' with 'pick/load another object'. Do not omit unloading, transferring, or processing the existing carried object.\n"
            "8. Treat phrases such as 'the cup on the tray' and 'a cup on the workbench' as different instances by default, even if no color is given.\n"
            "9. If an existing carried object has no color or identity specified, interpret it as one existing instance that still needs to be handled. Do not delete that step merely because it is underspecified. Keep ambiguity only if precise disambiguation is truly required and planning cannot proceed.\n"
            "10. If the destination region is not the workbench, do not generate an extra destination-side arm placement step there. Transportation tasks usually finish when the base reaches the destination while carrying the object on the tray.\n"
            "11. raw_clause must be concise and natural. Base steps should prefer short phrases such as 'Go to the workbench.' or 'Go to the bedroom.'\n"
            "12. depends_on uses 1-based step indices and may refer only to earlier steps. Use an empty array when there is no dependency.\n"
            "Do not invent new capabilities. If the executor is unclear, keep the step but set executor to an empty string.\n"
            "Output JSON only. Do not output any extra text.\n"
            "JSON schema:\n"
            "{\n"
            '  "steps": [\n'
            "    {\n"
            '      "raw_clause": "short core action phrase for this step",\n'
            '      "executor": "arm" | "base" | "",\n'
            '      "planning_note": "natural-language note for the stage-2 reviewer; may include hidden preconditions, manipulated object, source and destination, or why the order is chosen",\n'
            '      "depends_on": [1, 2],\n'
            '      "skip_if": "write a skip condition if the step can be skipped when its precondition is already satisfied; otherwise use an empty string"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "User task:\n%s\n"
        ) % (
            arm_capability_summary_short,
            base_capability_summary_short,
            "\n".join(f"- {item}" for item in executor_hints),
            user_instruction,
        )
        payload = self._call_text_json(prompt, stage="split_instruction")
        steps = payload.get("steps")
        if not isinstance(steps, list) or not steps:
            raise RuntimeError("Plan sketch returned an empty steps list")
        normalized: List[Dict[str, Any]] = []
        for item in steps:
            if not isinstance(item, dict):
                continue
            depends_on_raw = item.get("depends_on", [])
            depends_on: List[int] = []
            if isinstance(depends_on_raw, list):
                for dep in depends_on_raw:
                    try:
                        dep_idx = int(dep)
                    except (TypeError, ValueError):
                        continue
                    if dep_idx > 0:
                        depends_on.append(dep_idx)
            normalized.append(
                {
                    "raw_clause": str(item.get("raw_clause", "")).strip(),
                    "executor": _normalize_executor(item.get("executor")),
                    "planning_note": str(item.get("planning_note", "")).strip(),
                    "depends_on": depends_on,
                    "skip_if": str(item.get("skip_if", "")).strip(),
                }
            )
        normalized = [item for item in normalized if item["raw_clause"]]
        if not normalized:
            raise RuntimeError("Plan sketch did not return any valid steps")
        return normalized

    def assess_arm_steps_batch(
        self,
        *,
        user_instruction: str,
        raw_clauses: Sequence[str],
        agent_image_path: str,
        arm_capability_summary: str,
    ) -> List[Dict[str, Any]]:
        clauses_text = "\n".join(
            f'{i + 1}. "{clause}"' for i, clause in enumerate(raw_clauses)
        )
        prompt = (
            "You are a batch feasibility reviewer for tabletop arm tasks. "
            "You will receive the full user task, the current tabletop agent_view image, and all arm plan sketches extracted from that task. "
            "Evaluate each arm subtask in order: whether its execution preconditions are satisfied in the current situation and whether it can be mapped to current arm capabilities.\n"
            "Decision rules:\n"
            "1. In the current environment, arm actions are allowed only in the workbench region.\n"
            "2. First judge whether each arm step is a valid tabletop action within a multi-executor collaboration chain. Do not incorrectly force cross-region transport responsibility onto the arm alone.\n"
            "3. For generic object references such as 'a cup' or 'a plate', allow any instance of that category by default. Do not reject only because color, index, or unique identity is not specified.\n"
            "4. Use ambiguous_argument only when the current scene truly contains multiple candidates that must be distinguished and the step requires a unique target.\n"
            "5. For cross-region transport tasks, the arm usually handles workbench-side pick, load-to-tray, or other tabletop pick/place actions. Do not mark an arm step out of scope just because the overall task ends in another room.\n"
            "6. The current arm instruction catalog is both a capability example and a normalization reference, not an exhaustive list of all valid surface forms. If the step is a legal workbench tabletop action, you may mark it feasible and return a normalized_query, leaving exact normalization to later query/instruction matching.\n"
            "7. The input image is only a local agent_view near the arm. It may not show distant or occluded tray contents. Therefore, do not mark scene_precondition_failed merely because a tray object is not directly visible in the current image when the main precondition comes from task context.\n"
            "Output results in the original order. The number of items in results must exactly match the input.\n"
            "Output JSON only. Do not output any extra text.\n"
            "JSON schema:\n"
            "{\n"
            '  "results": [\n'
            "    {\n"
            '      "feasible": true | false,\n'
            '      "reason_code": "scene_precondition_failed | ambiguous_argument | unsupported_capability | normalization_failed | ok",\n'
            '      "reason_text": "one-sentence explanation",\n'
            '      "normalized_query": "normalized arm query; may be empty on failure",\n'
            '      "normalized_instruction": "fill only if a standard instruction can already be determined directly; otherwise use an empty string"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Current arm capability summary:\n%s\n\n"
            "Full user task:\n%s\n\n"
            "Arm subtasks to review:\n%s\n"
        ) % (arm_capability_summary, user_instruction, clauses_text)
        payload = self._call_multimodal_json(
            prompt=prompt,
            image_paths=[agent_image_path],
            stage="assess_arm_steps_batch",
        )
        results = payload.get("results")
        if not isinstance(results, list):
            raise RuntimeError("Arm feasibility review did not return a results array")
        return [item if isinstance(item, dict) else {} for item in results]

    def assess_base_steps_batch(
        self,
        *,
        user_instruction: str,
        raw_clauses: Sequence[str],
        navigation_context: Dict[str, Any],
        base_capability_summary: str,
        per_task_candidates: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        # Build the task listing — either with per-task top-K candidates
        # or with a shared global navigation context.
        if per_task_candidates is not None and len(per_task_candidates) == len(raw_clauses):
            task_lines: List[str] = []
            for i, clause in enumerate(raw_clauses):
                cands = per_task_candidates[i]
                cand_lines = "\n".join(
                    f'   - {c.get("cluster_id", "?")}: '
                    f'{c.get("cluster_caption", "")} '
                    f'(match score: {c.get("retrieval_score", "?")})'
                    for c in cands
                ) if cands else "   (no matched candidate region)"
                task_lines.append(f'{i + 1}. "{clause}"\n   Candidate regions:\n{cand_lines}')
            clauses_section = "\n\n".join(task_lines)
            context_note = "The program has already pre-filtered the most relevant candidate regions for each subtask, sorted by semantic match score."
        else:
            clauses_section = "\n".join(
                f'{i + 1}. "{clause}"' for i, clause in enumerate(raw_clauses)
            )
            context_note = "The program has already compressed the map into the current pose and the full list of semantic regions."

        # Global scene info (current pose + current region).
        scene_global = {
            "current_pose_text": navigation_context.get("current_pose_text", ""),
            "nearest_topology_node": navigation_context.get("nearest_topology_node", ""),
            "current_region": navigation_context.get("current_region", {}),
        }

        prompt = (
            "You are a batch feasibility reviewer for mobile-robot navigation tasks. "
            "%s You do not need to perform graph search. "
            "Judge each navigation plan sketch in order: whether it is clear and whether the target exists in the scene.\n"
            "Decision rules:\n"
            "1. Prefer judging whether each navigation step is a valid movement stage within a multi-executor collaboration chain.\n"
            "2. Do not reject a base step merely because later steps may still require arm participation.\n"
            "3. For generic object references, do not treat missing object attributes as a navigation failure reason. Navigation only needs the movement target and region to be clear.\n"
            "4. If an object has already been loaded onto the robot tray at the workbench, or will be loaded there by the arm, the base may transport that tray load across regions. Do not misclassify such collaborative transport as unsupported_capability.\n"
            "Output results in the original order. The number of items in results must exactly match the input.\n"
            "Output JSON only. Do not output any extra text.\n"
            "JSON schema:\n"
            "{\n"
            '  "results": [\n'
            "    {\n"
            '      "feasible": true | false,\n'
            '      "reason_code": "ambiguous_argument | unsupported_capability | normalization_failed | scene_precondition_failed | ok",\n'
            '      "reason_text": "one-sentence explanation",\n'
            '      "normalized_query": "normalized base query; may be empty on failure",\n'
            '      "goal_region": "Cluster_x or an empty string"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Current base capabilities and limitations:\n%s\n\n"
            "Full user task:\n%s\n\n"
            "Current location:\n%s\n\n"
            "Base subtasks to review:\n%s\n"
        ) % (
            context_note,
            base_capability_summary,
            user_instruction,
            json.dumps(scene_global, ensure_ascii=False, indent=2),
            clauses_section,
        )
        payload = self._call_text_json(prompt, stage="assess_base_steps_batch")
        results = payload.get("results")
        if not isinstance(results, list):
            raise RuntimeError("Base feasibility review did not return a results array")
        return [item if isinstance(item, dict) else {} for item in results]

    def _call_text_json(self, prompt: str, *, stage: str) -> Dict[str, Any]:
        response = dashscope.Generation.call(
            model=self.text_model,
            messages=[
                {"role": "system", "content": "You are a structured robot task analyzer."},
                {"role": "user", "content": prompt},
            ],
            stream=self.stream_output,
        )
        text = self._collect_text_response(response, multimodal=False, stream_label="[TASK-DECOMP][STREAM]")
        payload = _safe_json_loads(text)
        self._append_debug_record(
            stage=stage,
            model=self.text_model,
            multimodal=False,
            raw_text=text,
            parsed_payload=payload,
        )
        return payload

    def _call_multimodal_json(self, *, prompt: str, image_paths: Sequence[str], stage: str) -> Dict[str, Any]:
        user_content: List[Dict[str, str]] = []
        for idx, image_path in enumerate(image_paths, start=1):
            path = Path(str(image_path)).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Image does not exist: {path}")
            user_content.append({"text": f"Image {idx}"})
            # Use data URIs instead of file:// paths because the current
            # DashScope endpoint may reject local file URLs with url error.
            user_content.append({"image": _image_path_to_data_uri(path)})
        user_content.append({"text": prompt})
        response = dashscope.MultiModalConversation.call(
            model=self.vision_model,
            messages=[
                {"role": "system", "content": [{"text": "You are a structured robot task analyzer."}]},
                {"role": "user", "content": user_content},
            ],
            stream=self.stream_output,
        )
        text = self._collect_text_response(response, multimodal=True, stream_label="[TASK-DECOMP][STREAM]")
        payload = _safe_json_loads(text)
        self._append_debug_record(
            stage=stage,
            model=self.vision_model,
            multimodal=True,
            raw_text=text,
            parsed_payload=payload,
        )
        return payload

    def _collect_text_response(self, response: Any, *, multimodal: bool, stream_label: str) -> str:
        if not self.stream_output:
            extractor = self._extract_mm_text if multimodal else self._extract_generation_text
            return extractor(response)

        if not hasattr(response, "__iter__") or isinstance(response, (dict, str, bytes)):
            extractor = self._extract_mm_text if multimodal else self._extract_generation_text
            return extractor(response)

        extractor = self._extract_mm_text if multimodal else self._extract_generation_text
        chunks: List[str] = []
        last_error: Optional[str] = None
        printed_prefix = ""
        print(f"\n{stream_label} start")
        for part in response:
            try:
                text = extractor(part)
            except Exception as exc:
                last_error = str(exc)
                continue
            if not text:
                continue
            chunks.append(text)
            delta = text
            if printed_prefix and text.startswith(printed_prefix):
                delta = text[len(printed_prefix):]
            elif chunks[:-1]:
                prev = chunks[-2]
                if text.startswith(prev):
                    delta = text[len(prev):]
            if delta:
                print(delta, end="", flush=True)
            printed_prefix = text
        print(f"\n{stream_label} done")
        if not chunks:
            if last_error:
                raise RuntimeError(f"Streaming output did not yield parseable text; last response error: {last_error}")
            raise RuntimeError("Streaming output did not return any text content")
        return chunks[-1]

    @staticmethod
    def _extract_generation_text(response: Any) -> str:
        response_data = response if isinstance(response, dict) else getattr(response, "to_dict", lambda: response)()
        if isinstance(response_data, dict):
            status_code = response_data.get("status_code")
            if status_code not in (None, 200):
                code = str(response_data.get("code") or "").strip()
                message = str(response_data.get("message") or "").strip()
                detail = ", ".join(part for part in [f"status_code={status_code}", f"code={code}" if code else "", f"message={message}" if message else ""] if part)
                raise RuntimeError(f"DashScope request failed: {detail or response_data}")
            output = response_data.get("output")
            if isinstance(output, dict):
                choices = output.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        message = first.get("message")
                        if isinstance(message, dict):
                            content = message.get("content")
                            # content 可能是 str（Generation 文本模型）
                            # 也可能是 list（MultiModal 多模态模型）
                            if isinstance(content, str) and content.strip():
                                return content.strip()
                            if isinstance(content, list):
                                for item in content:
                                    text = item.get("text") if isinstance(item, dict) else None
                                    if isinstance(text, str) and text.strip():
                                        return text.strip()
                        text = first.get("text")
                        if isinstance(text, str) and text.strip():
                            return text.strip()
                text = output.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        raise RuntimeError(f"Failed to extract text from DashScope Generation response: {response}")

    @classmethod
    def _extract_mm_text(cls, response: Any) -> str:
        return cls._extract_generation_text(response)


class TaskDecomposer:
    """High-level coordinator that validates and publishes serial tasks."""

    def __init__(
        self,
        *,
        backend: Optional[DecompositionBackend] = None,
        arm_instruction_groups: Optional[Sequence[Dict[str, Any]]] = None,
        base_capabilities: Optional[Sequence[str]] = None,
        base_limitations: Optional[Sequence[str]] = None,
        nav_top_k: int = 4,
        embed_query_fn: Optional[Callable[[str, str], np.ndarray]] = None,
        embedding_model: str = "",
    ):
        self.backend = backend or DashScopeTaskDecompositionBackend()
        self.arm_capability_summary = build_arm_capability_summary(arm_instruction_groups)
        self.arm_capability_summary_short = DEFAULT_ARM_CAPABILITY_SUMMARY_SHORT
        self.base_capability_summary = build_base_capability_summary(
            capabilities=base_capabilities,
            limitations=base_limitations,
        )
        self.base_capability_summary_short = DEFAULT_BASE_CAPABILITY_SUMMARY_SHORT
        self.nav_top_k = max(1, int(nav_top_k))
        self._embed_query_fn = embed_query_fn
        self._embedding_model = str(embedding_model).strip()

    @staticmethod
    def _print_progress(message: str) -> None:
        print(f"\n[TASK-DECOMP] {message}", flush=True)

    def decompose_user_task(
        self,
        user_instruction: str,
        *,
        agent_image_path: str | None = None,
        navigation_context: Dict[str, Any] | None = None,
    ) -> TaskDecompositionResult:
        """Two-stage task decomposition.

        Stage 1 — **plan sketch** (text-only LLM, lightweight):
            Break the user instruction into ordered sub-tasks with executor
            assignment plus lightweight planning notes / dependencies /
            skip conditions. Only high-level executor descriptions are
            provided; no image, no cluster list, no instruction catalog.

        Stage 2 — **batch feasibility** (per-executor, parallel):
            * arm steps  → one VLM call with the agent-view image + arm
              instruction catalog, using the stage-1 plan sketch as input.
            * base steps → per-task embedding top-K cluster filtering, then
              one text LLM call with per-task candidates + current pose,
              also using the stage-1 plan sketch as input.
            The two calls are issued in parallel via a thread pool.
        """
        instruction = str(user_instruction).strip()
        if not instruction:
            raise ValueError("user_instruction must not be empty")
        if hasattr(self.backend, "reset_debug_records"):
            self.backend.reset_debug_records()
        self._print_progress(f"Start planning for user instruction: {instruction}")

        # ── Stage 1: task decomposition (text-only LLM) ──
        self._print_progress("Stage 1/3: generating plan sketch...")
        drafts = self._build_step_drafts(instruction)
        self._print_progress(f"Stage 1/3 done: {len(drafts)} draft step(s) after sanitization.")

        # ── Pre-check & group by executor ──
        self._print_progress("Stage 2/3: running local pre-checks and grouping by executor...")
        arm_drafts: List[StepDraft] = []
        base_drafts: List[StepDraft] = []
        pre_evaluations: Dict[int, StepEvaluation] = {}

        for draft in drafts:
            if not draft.raw_clause:
                pre_evaluations[draft.step_index] = StepEvaluation(
                    step_index=draft.step_index,
                    raw_clause="",
                    executor=draft.executor,
                    feasible=False,
                    reason_code="normalization_failed",
                    reason_text="The decomposer did not return a valid task clause.",
                )
            elif not draft.executor:
                pre_evaluations[draft.step_index] = StepEvaluation(
                    step_index=draft.step_index,
                    raw_clause=draft.raw_clause,
                    executor="",
                    feasible=False,
                    reason_code="missing_executor",
                    reason_text="Could not determine whether this step belongs to the arm or the base.",
                )
            elif draft.executor == "arm":
                if not agent_image_path:
                    pre_evaluations[draft.step_index] = StepEvaluation(
                        step_index=draft.step_index,
                        raw_clause=draft.raw_clause,
                        executor="arm",
                        feasible=False,
                        reason_code="scene_precondition_failed",
                        reason_text="Missing current tabletop agent view, so the tabletop step cannot be validated.",
                    )
                else:
                    arm_drafts.append(draft)
            elif draft.executor == "base":
                if navigation_context is None:
                    pre_evaluations[draft.step_index] = StepEvaluation(
                        step_index=draft.step_index,
                        raw_clause=draft.raw_clause,
                        executor="base",
                        feasible=False,
                        reason_code="scene_precondition_failed",
                        reason_text="Missing navigation scene context, so the navigation step cannot be validated.",
                    )
                else:
                    base_drafts.append(draft)
            else:
                pre_evaluations[draft.step_index] = StepEvaluation(
                    step_index=draft.step_index,
                    raw_clause=draft.raw_clause,
                    executor=draft.executor,
                    feasible=False,
                    reason_code="unsupported_capability",
                    reason_text=f"Unsupported executor: {draft.executor}",
                )
        rejected_precheck = sum(1 for item in pre_evaluations.values() if not item.feasible)
        self._print_progress(
            "Stage 2/3 pre-check done: "
            f"arm={len(arm_drafts)}, base={len(base_drafts)}, "
            f"rejected_early={rejected_precheck}."
        )

        # ── Stage 2: batch feasibility assessment (parallel) ──
        arm_results: List[Dict[str, Any]] = []
        base_results: List[Dict[str, Any]] = []

        def _assess_arm_batch() -> List[Dict[str, Any]]:
            if not arm_drafts:
                self._print_progress("Stage 2/3 ARM review skipped: no arm draft.")
                return []
            self._print_progress(f"Stage 2/3 ARM review started: {len(arm_drafts)} draft(s).")
            return self.backend.assess_arm_steps_batch(
                user_instruction=instruction,
                raw_clauses=[self._render_draft_for_assessment(d) for d in arm_drafts],
                agent_image_path=str(agent_image_path),
                arm_capability_summary=self.arm_capability_summary,
            )

        def _assess_base_batch() -> List[Dict[str, Any]]:
            if not base_drafts:
                self._print_progress("Stage 2/3 BASE review skipped: no base draft.")
                return []
            self._print_progress(f"Stage 2/3 BASE review started: {len(base_drafts)} draft(s).")
            base_clause_queries = [d.raw_clause for d in base_drafts]
            base_clause_sketches = [self._render_draft_for_assessment(d) for d in base_drafts]
            all_candidates = (navigation_context or {}).get("candidate_regions", [])
            per_task_cands = self._select_per_task_candidates(base_clause_queries, all_candidates)
            return self.backend.assess_base_steps_batch(
                user_instruction=instruction,
                raw_clauses=base_clause_sketches,
                navigation_context=navigation_context,  # type: ignore[arg-type]
                base_capability_summary=self.base_capability_summary,
                per_task_candidates=per_task_cands,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            arm_future = pool.submit(_assess_arm_batch)
            base_future = pool.submit(_assess_base_batch)
            arm_results = arm_future.result()
            base_results = base_future.result()
        self._print_progress(
            "Stage 2/3 review done: "
            f"arm_results={len(arm_results)}, base_results={len(base_results)}."
        )

        # ── Merge batch results back into evaluations ──
        self._print_progress("Stage 3/3: merging review results and publishing task queue...")
        for i, draft in enumerate(arm_drafts):
            payload = arm_results[i] if i < len(arm_results) else {}
            pre_evaluations[draft.step_index] = self._evaluation_from_arm_payload(draft, payload)

        for i, draft in enumerate(base_drafts):
            payload = base_results[i] if i < len(base_results) else {}
            pre_evaluations[draft.step_index] = self._evaluation_from_base_payload(draft, payload)

        # ── Build final result in original step order ──
        evaluations = [pre_evaluations[draft.step_index] for draft in drafts]
        feasible = all(e.feasible for e in evaluations)
        diagnostics = [self._diagnostic_from_evaluation(e) for e in evaluations]
        normalized_tasks = (
            self._expand_normalized_tasks(
                evaluations=evaluations,
                navigation_context=navigation_context,
            )
            if feasible
            else []
        )
        summary = self._build_summary(
            feasible=feasible,
            evaluations=evaluations,
            published_task_count=len(normalized_tasks),
        )
        self._print_progress(
            "Finished: "
            f"feasible={feasible}, published_tasks={len(normalized_tasks)}, "
            f"diagnostics={len(diagnostics)}."
        )
        return TaskDecompositionResult(
            feasible=feasible,
            user_instruction=instruction,
            normalized_tasks=normalized_tasks,
            diagnostics=diagnostics,
            summary_for_user=summary,
            step_evaluations=evaluations,
            llm_debug_records=(
                self.backend.get_debug_records()
                if hasattr(self.backend, "get_debug_records")
                else []
            ),
        )

    def _get_embed_fn(self) -> Optional[Callable[[str, str], np.ndarray]]:
        """Return the embedding function, lazily creating the default if needed."""
        if self._embed_query_fn is not None:
            return self._embed_query_fn
        if not self._embedding_model:
            return None
        # Ensure API key is set for the default embed function.
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            return None
        dashscope.api_key = api_key
        return _default_embed_query

    def _select_per_task_candidates(
        self,
        base_clauses: Sequence[str],
        all_candidates: Sequence[Dict[str, Any]],
    ) -> Optional[List[List[Dict[str, Any]]]]:
        """Embedding top-K filtering: select the most relevant clusters per
        base sub-task.

        Returns ``None`` when embedding is unavailable (caller should fall
        back to the full candidate list).
        """
        embed_fn = self._get_embed_fn()
        if embed_fn is None or not all_candidates:
            return None

        model = self._embedding_model
        top_k = self.nav_top_k

        # Embed all cluster captions once (shared across tasks).
        caption_embeddings: List[tuple[Dict[str, Any], np.ndarray]] = []
        for cand in all_candidates:
            caption = str(cand.get("cluster_caption", "")).strip()
            if not caption:
                continue
            try:
                vec = embed_fn(caption, model)
            except Exception:
                LOGGER.warning("Embedding cluster caption failed: %s", caption, exc_info=True)
                continue
            caption_embeddings.append((cand, vec))

        if not caption_embeddings:
            return None

        per_task: List[List[Dict[str, Any]]] = []
        for clause in base_clauses:
            try:
                query_vec = embed_fn(str(clause), model)
            except Exception:
                LOGGER.warning("Embedding base clause failed: %s", clause, exc_info=True)
                # Fall back: give this task all candidates.
                per_task.append(list(all_candidates))
                continue

            scored = [
                (cand, _cosine_similarity(query_vec, cvec))
                for cand, cvec in caption_embeddings
            ]
            scored.sort(key=lambda item: item[1], reverse=True)
            selected = scored[:top_k]
            per_task.append([
                {**cand, "retrieval_score": round(score, 4)}
                for cand, score in selected
            ])
        return per_task

    def _build_step_drafts(self, instruction: str) -> List[StepDraft]:
        raw_steps = self.backend.split_instruction(
            user_instruction=instruction,
            executor_hints=EXECUTOR_DESCRIPTIONS,
            arm_capability_summary_short=self.arm_capability_summary_short,
            base_capability_summary_short=self.base_capability_summary_short,
        )
        raw_steps = _expand_split_steps(raw_steps)
        drafts: List[StepDraft] = []
        for idx, item in enumerate(raw_steps):
            executor = _normalize_executor(item.get("executor"))
            raw_clause = _normalize_raw_clause_text(item.get("raw_clause", ""), executor)
            planning_note = str(item.get("planning_note", "")).strip()
            skip_if = str(item.get("skip_if", "")).strip()
            depends_on_raw = item.get("depends_on", [])
            depends_on = [int(dep) for dep in depends_on_raw if isinstance(dep, int) and dep > 0]
            drafts.append(
                StepDraft(
                    step_index=idx,
                    raw_clause=raw_clause,
                    executor=executor,
                    planning_note=planning_note,
                    depends_on=depends_on,
                    skip_if=skip_if,
                )
            )
        if not drafts:
            raise RuntimeError("Task decomposition did not return any valid steps")
        return drafts

    @staticmethod
    def _render_draft_for_assessment(draft: StepDraft) -> str:
        lines = [f"Core step: {draft.raw_clause}"]
        if draft.planning_note:
            lines.append(f"Planning note: {draft.planning_note}")
        if draft.depends_on:
            deps = ", ".join(str(dep) for dep in draft.depends_on)
            lines.append(f"Depends on: {deps}")
        if draft.skip_if:
            lines.append(f"Skip if: {draft.skip_if}")
        return "\n".join(lines)

    def _evaluation_from_arm_payload(
        self,
        draft: StepDraft,
        payload: Dict[str, Any],
    ) -> StepEvaluation:
        feasible = _bool_from_payload(payload.get("feasible"))
        reason_code = _reason_code_from_payload(payload.get("reason_code"), "normalization_failed" if not feasible else "ok")
        reason_text = str(payload.get("reason_text", "")).strip()
        normalized_query = str(payload.get("normalized_query", "")).strip()

        task: Optional[Dict[str, Any]] = None
        if feasible:
            # ARM 执行阶段必须统一通过 ARM-RAG 选取库内标准指令，
            # 不允许直接信任 VLM 产出的自由文本 instruction。
            task = {
                "name": _make_task_name("arm", draft.step_index, draft.raw_clause),
                "executor": "arm",
                "instruction": "",
                "query": normalized_query or draft.raw_clause,
                "reinitialize_arm": True,
            }
        return StepEvaluation(
            step_index=draft.step_index,
            raw_clause=draft.raw_clause,
            executor="arm",
            feasible=feasible,
            normalized_task=task,
            reason_code=reason_code,
            reason_text=reason_text,
            model_output=dict(payload),
        )

    def _evaluation_from_base_payload(
        self,
        draft: StepDraft,
        payload: Dict[str, Any],
    ) -> StepEvaluation:
        feasible = _bool_from_payload(payload.get("feasible"))
        reason_code = _reason_code_from_payload(payload.get("reason_code"), "normalization_failed" if not feasible else "ok")
        reason_text = str(payload.get("reason_text", "")).strip()
        normalized_query = str(payload.get("normalized_query", "")).strip()
        goal_region = str(payload.get("goal_region", "")).strip()
        task: Optional[Dict[str, Any]] = None
        if feasible:
            task = {
                "name": _make_task_name("base", draft.step_index, draft.raw_clause),
                "executor": "base",
                "query": normalized_query or draft.raw_clause,
                "goal_region": goal_region,
            }
        return StepEvaluation(
            step_index=draft.step_index,
            raw_clause=draft.raw_clause,
            executor="base",
            feasible=feasible,
            normalized_task=task,
            reason_code=reason_code,
            reason_text=reason_text,
            model_output=dict(payload),
        )

    @staticmethod
    def _diagnostic_from_evaluation(evaluation: StepEvaluation) -> Dict[str, Any]:
        return {
            "step_index": int(evaluation.step_index),
            "raw_clause": evaluation.raw_clause,
            "executor": evaluation.executor,
            "status": "ok" if evaluation.feasible else "rejected",
            "reason_code": evaluation.reason_code,
            "reason_text": evaluation.reason_text,
        }

    def _expand_normalized_tasks(
        self,
        *,
        evaluations: Sequence[StepEvaluation],
        navigation_context: Dict[str, Any] | None,
    ) -> List[Dict[str, Any]]:
        state = self._initial_planner_state(navigation_context)
        workbench_region = self._resolve_workbench_region(navigation_context)
        if any(
            evaluation.normalized_task and str(evaluation.normalized_task.get("executor", "")).strip().lower() == "arm"
            for evaluation in evaluations
        ) and workbench_region is None:
            raise RuntimeError("Found arm tasks, but no workbench region exists in the navigation context")
        expanded_tasks: List[Dict[str, Any]] = []
        inserted_counter = 0

        for evaluation in evaluations:
            if not evaluation.normalized_task:
                continue
            task = dict(evaluation.normalized_task)
            executor = str(task.get("executor", "")).strip().lower()

            if executor == "arm" and workbench_region is not None:
                if state.base_region_id != workbench_region["cluster_id"]:
                    inserted_counter += 1
                    move_task = {
                        "name": f"base_pre_arm_workbench_{inserted_counter}",
                        "executor": "base",
                        "query": "Go to the workbench.",
                        "goal_region": workbench_region["cluster_id"],
                        "cluster_caption": workbench_region["cluster_caption"],
                    }
                    expanded_tasks.append(
                        self._annotate_task_with_state(
                            move_task,
                            state=state,
                            next_region_id=workbench_region["cluster_id"],
                            next_region_caption=workbench_region["cluster_caption"],
                            required_region_id="",
                            required_region_caption="",
                        )
                    )
                    state.base_region_id = workbench_region["cluster_id"]
                    state.base_region_caption = workbench_region["cluster_caption"]

                expanded_tasks.append(
                    self._annotate_task_with_state(
                        task,
                        state=state,
                        next_region_id=state.base_region_id,
                        next_region_caption=state.base_region_caption,
                        required_region_id=workbench_region["cluster_id"],
                        required_region_caption=workbench_region["cluster_caption"],
                    )
                )
                continue

            if executor == "base":
                goal_region = str(task.get("goal_region", "")).strip()
                next_region_caption = self._lookup_region_caption(goal_region, navigation_context)
                if next_region_caption and not str(task.get("cluster_caption", "")).strip():
                    task["cluster_caption"] = next_region_caption
                expanded_tasks.append(
                    self._annotate_task_with_state(
                        task,
                        state=state,
                        next_region_id=goal_region,
                        next_region_caption=next_region_caption,
                        required_region_id="",
                        required_region_caption="",
                    )
                )
                if goal_region:
                    state.base_region_id = goal_region
                    state.base_region_caption = next_region_caption
                else:
                    state.base_region_id = ""
                    state.base_region_caption = ""
                continue

            expanded_tasks.append(
                self._annotate_task_with_state(
                    task,
                    state=state,
                    next_region_id=state.base_region_id,
                    next_region_caption=state.base_region_caption,
                    required_region_id="",
                    required_region_caption="",
                )
            )

        return expanded_tasks

    @staticmethod
    def _initial_planner_state(navigation_context: Dict[str, Any] | None) -> PlannerState:
        current_region = (navigation_context or {}).get("current_region", {})
        return PlannerState(
            base_region_id=str(current_region.get("cluster_id", "")).strip(),
            base_region_caption=str(current_region.get("cluster_caption", "")).strip(),
        )

    @staticmethod
    def _lookup_region_caption(
        cluster_id: str,
        navigation_context: Dict[str, Any] | None,
    ) -> str:
        if not cluster_id:
            return ""
        current_region = (navigation_context or {}).get("current_region", {})
        if str(current_region.get("cluster_id", "")).strip() == cluster_id:
            return str(current_region.get("cluster_caption", "")).strip()
        for item in (navigation_context or {}).get("candidate_regions", []):
            if not isinstance(item, dict):
                continue
            if str(item.get("cluster_id", "")).strip() == cluster_id:
                return str(item.get("cluster_caption", "")).strip()
        return ""

    @staticmethod
    def _resolve_workbench_region(
        navigation_context: Dict[str, Any] | None,
    ) -> Optional[Dict[str, str]]:
        candidates = list((navigation_context or {}).get("candidate_regions", []))
        current_region = (navigation_context or {}).get("current_region", {})
        if isinstance(current_region, dict):
            candidates.append(current_region)

        keywords = ("workbench", "work bench", "bench", "table")
        for item in candidates:
            if not isinstance(item, dict):
                continue
            caption = str(item.get("cluster_caption", "")).strip()
            caption_lower = caption.lower()
            if any(keyword in caption_lower or keyword in caption for keyword in keywords):
                cluster_id = str(item.get("cluster_id", "")).strip()
                if cluster_id:
                    return {
                        "cluster_id": cluster_id,
                        "cluster_caption": caption,
                    }
        return None

    @staticmethod
    def _annotate_task_with_state(
        task: Dict[str, Any],
        *,
        state: PlannerState,
        next_region_id: str,
        next_region_caption: str,
        required_region_id: str,
        required_region_caption: str,
    ) -> Dict[str, Any]:
        annotated = dict(task)
        annotated["preconditions"] = {
            "base_region_before": state.base_region_id or "UNKNOWN",
            "base_region_before_caption": state.base_region_caption,
        }
        if required_region_id:
            annotated["preconditions"]["required_base_region"] = required_region_id
            annotated["preconditions"]["required_base_region_caption"] = required_region_caption
        annotated["effects"] = {
            "base_region_after": next_region_id or "UNKNOWN",
            "base_region_after_caption": next_region_caption,
        }
        return annotated

    @staticmethod
    def _build_summary(
        *,
        feasible: bool,
        evaluations: Sequence[StepEvaluation],
        published_task_count: int = 0,
    ) -> str:
        if feasible:
            count = published_task_count or len(evaluations)
            return f"Task is executable. Generated {count} serial subtask(s)."
        first_failed = next((item for item in evaluations if not item.feasible), None)
        if first_failed is None:
            return "Task is not executable."
        return (
            "Task is not executable: "
            f"step {first_failed.step_index + 1} was rejected because {first_failed.reason_text or first_failed.reason_code}."
        )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return -1.0
    return float(np.dot(a, b) / denominator)


def _default_embed_query(text: str, model_name: str) -> np.ndarray:
    """Call DashScope TextEmbedding to convert *text* into a vector."""
    response = dashscope.TextEmbedding.call(model=model_name, input=text)
    output_obj = getattr(response, "output", None)
    if output_obj is None and isinstance(response, dict):
        output_obj = response.get("output")
    embeddings = getattr(output_obj, "embeddings", None)
    if embeddings is None and isinstance(output_obj, dict):
        embeddings = output_obj.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings:
        raise RuntimeError(f"TextEmbedding response is missing embeddings: {response}")
    first = embeddings[0]
    vector = getattr(first, "embedding", None)
    if vector is None and isinstance(first, dict):
        vector = first.get("embedding")
    if not isinstance(vector, list) or not vector:
        raise RuntimeError(f"TextEmbedding response is missing an embedding vector: {response}")
    return np.asarray(vector, dtype=np.float64)


def _node_pose_xy(node_attr: Dict[str, Any]) -> Optional[np.ndarray]:
    pose = node_attr.get("pose")
    if not isinstance(pose, (list, tuple)) or len(pose) < 2:
        return None
    return np.asarray([float(pose[0]), float(pose[1])], dtype=np.float64)


def _format_pose_text(current_pose: Sequence[float], current_region: Dict[str, Any]) -> str:
    x = float(current_pose[0])
    y = float(current_pose[1])
    yaw = float(current_pose[2]) if len(current_pose) >= 3 else 0.0
    yaw_deg = math.degrees(yaw)
    return (
        "Robot pose: "
        f"x={x:.3f}, y={y:.3f}, yaw={yaw_deg:.1f}deg. "
        f"Current semantic region: {current_region.get('cluster_id', 'UNKNOWN')} "
        f"({current_region.get('cluster_caption', '')})."
    )


def _find_nearest_leaf_node(graph: nx.Graph, current_xy: np.ndarray) -> str:
    nearest_node = None
    nearest_dist = float("inf")
    for node_id, node_attr in graph.nodes(data=True):
        if node_attr.get("node_type") == "cluster_parent":
            continue
        pose_xy = _node_pose_xy(node_attr)
        if pose_xy is None:
            continue
        dist = float(np.linalg.norm(pose_xy - current_xy))
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_node = str(node_id)
    if nearest_node is None:
        raise RuntimeError("Could not find any leaf node with a pose in the topology graph")
    return nearest_node


def _find_parent_cluster(graph: nx.Graph, node_id: str) -> Dict[str, Any]:
    if node_id not in graph:
        raise KeyError(f"Node does not exist: {node_id}")
    for neighbor in graph.neighbors(node_id):
        edge_attr = graph.get_edge_data(node_id, neighbor) or {}
        if edge_attr.get("edge_type") == "is_child_of":
            node_attr = graph.nodes[neighbor]
            return {
                "cluster_id": str(neighbor),
                "cluster_caption": str(node_attr.get("semantic_caption", "")),
            }
    return {
        "cluster_id": "UNKNOWN",
        "cluster_caption": "",
    }


def build_navigation_scene_context(
    current_pose: Sequence[float],
    rag_navigator: Any,
) -> NavigationSceneContext:
    """Build a compact navigation context for LLM consumption.

    This function provides only the information the VLM needs for a **coarse
    feasibility check**: current robot location + the *full* list of semantic
    cluster regions available on the map.  The VLM decides which region(s) are
    relevant as part of its single-pass task decomposition – no embedding-based
    pre-filtering is needed at this stage.

    Fine-grained target-node resolution and path planning are deferred to the
    execution stage (``task_sequence._start_base_task``).

    ``rag_navigator`` is expected to be an initialized ``RAGNavigator``-like
    object with ``graph`` and ``cluster_info`` attributes.
    """

    pose_arr = np.asarray(current_pose, dtype=np.float64).reshape(-1)
    if pose_arr.shape[0] < 2:
        raise ValueError("current_pose must contain at least x and y")

    graph = rag_navigator.graph
    current_xy = pose_arr[:2]
    nearest_node = _find_nearest_leaf_node(graph, current_xy)
    current_region = _find_parent_cluster(graph, nearest_node)

    # Pass the full cluster list to the VLM so it can decide feasibility
    # for each navigation sub-task in a single call.
    candidate_regions: List[CandidateRegion] = []
    for cluster_id, caption in rag_navigator.cluster_info.items():
        caption_text = str(caption).strip()
        if not caption_text:
            continue
        candidate_regions.append(
            CandidateRegion(
                cluster_id=str(cluster_id),
                cluster_caption=caption_text,
            )
        )

    return NavigationSceneContext(
        current_pose_text=_format_pose_text(pose_arr, current_region),
        nearest_topology_node=str(nearest_node),
        current_region=current_region,
        candidate_regions=candidate_regions,
    )


def decompose_user_task(
    user_instruction: str,
    *,
    agent_image_path: str | None = None,
    navigation_context: Dict[str, Any] | None = None,
    backend: Optional[DecompositionBackend] = None,
) -> TaskDecompositionResult:
    """Convenience wrapper around :class:`TaskDecomposer`."""

    decomposer = TaskDecomposer(
        backend=backend,
    )
    return decomposer.decompose_user_task(
        user_instruction,
        agent_image_path=agent_image_path,
        navigation_context=navigation_context,
    )
