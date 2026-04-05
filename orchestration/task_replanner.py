"""Incremental failed-task replanning for the runtime task queue.

This module is intentionally side-effect free:
- It does not modify the existing task decomposition flow.
- It does not hook into the current task queue automatically.
- It only exposes a standalone replanning API for explicit future use.

Replanning input can combine:
- the original task queue produced by the first plan,
- the failed task plus runtime failure diagnostics,
- the current multimodal observations after queue execution reaches a failure,
- a semantic-topology map adapter for navigation-side grounding.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

import dashscope


def _safe_json_loads(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        raise RuntimeError("LLM returned an empty response")
    start = text.find("{")
    end = text.rfind("}")
    snippet = text[start : end + 1] if start >= 0 and end >= start else text
    payload = json.loads(snippet)
    if not isinstance(payload, dict):
        raise RuntimeError("LLM output is not a JSON object")
    return payload


def _image_path_to_data_uri(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        mime_type = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _normalize_executor(value: Any) -> str:
    executor = str(value or "").strip().lower()
    return executor if executor in {"arm", "base"} else ""


def _normalize_task(task: Dict[str, Any], *, fallback_name: str) -> Dict[str, Any]:
    normalized = dict(task)
    executor = _normalize_executor(normalized.get("executor"))
    if not executor:
        raise ValueError(f"Unsupported executor in replanned task: {task}")
    normalized["executor"] = executor
    normalized["name"] = str(normalized.get("name") or fallback_name).strip() or fallback_name
    if executor == "arm":
        normalized["instruction"] = str(normalized.get("instruction", "")).strip()
        normalized["query"] = str(normalized.get("query", "")).strip()
        normalized["reinitialize_arm"] = bool(normalized.get("reinitialize_arm", True))
        if not normalized["instruction"] and not normalized["query"]:
            raise ValueError("Replanned arm task requires instruction or query")
    else:
        normalized["query"] = str(normalized.get("query", "")).strip()
        normalized["target_node"] = str(normalized.get("target_node", "")).strip()
        normalized["goal_region"] = str(normalized.get("goal_region", "")).strip()
        normalized["cluster_caption"] = str(normalized.get("cluster_caption", "")).strip()
        normalized["target_caption"] = str(normalized.get("target_caption", "")).strip()
        if not normalized["query"] and not normalized["target_node"]:
            raise ValueError("Replanned base task requires query or target_node")
    return normalized


class SemanticTopologyAdapter(Protocol):
    """Abstract semantic-map interface consumed by the replanner."""

    def describe_current_region(self) -> Dict[str, Any]:
        """Return the robot's current semantic region."""

    def list_candidate_regions(self) -> List[Dict[str, Any]]:
        """Return known semantic regions usable by base navigation."""

    def retrieve_navigation_candidates(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return semantically matched navigation candidates for *query*."""


@dataclass
class RuntimeObservationImage:
    label: str
    image_path: str
    note: str = ""

    def as_prompt_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "image_path": self.image_path,
            "note": self.note,
        }


@dataclass
class RuntimeObservation:
    """Current multimodal state snapshot when replanning is requested."""

    current_region: Dict[str, Any] = field(default_factory=dict)
    textual_state: str = ""
    images: List[RuntimeObservationImage] = field(default_factory=list)
    extra_state: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "current_region": dict(self.current_region),
            "textual_state": self.textual_state,
            "images": [item.as_prompt_dict() for item in self.images],
            "extra_state": dict(self.extra_state),
        }


@dataclass
class FailedTaskContext:
    """Failure-side context from the existing queue execution."""

    original_user_instruction: str
    original_plan: List[Dict[str, Any]]
    failed_task: Dict[str, Any]
    failure_result: Dict[str, Any]
    completed_tasks: List[Dict[str, Any]] = field(default_factory=list)
    remaining_tasks: List[Dict[str, Any]] = field(default_factory=list)
    task_queue_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "original_user_instruction": self.original_user_instruction,
            "original_plan": [dict(item) for item in self.original_plan],
            "failed_task": dict(self.failed_task),
            "failure_result": dict(self.failure_result),
            "completed_tasks": [dict(item) for item in self.completed_tasks],
            "remaining_tasks": [dict(item) for item in self.remaining_tasks],
            "task_queue_results": [dict(item) for item in self.task_queue_results],
        }


@dataclass
class ReplanRequest:
    failure_context: FailedTaskContext
    observation: RuntimeObservation
    semantic_candidates: List[Dict[str, Any]] = field(default_factory=list)
    max_new_tasks: int = 5

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "failure_context": self.failure_context.to_prompt_dict(),
            "observation": self.observation.to_prompt_dict(),
            "semantic_candidates": [dict(item) for item in self.semantic_candidates],
            "max_new_tasks": int(self.max_new_tasks),
        }


@dataclass
class ReplanResult:
    feasible: bool
    failure_mode: str
    rationale: str
    replacement_tasks: List[Dict[str, Any]]
    merged_plan: List[Dict[str, Any]]
    raw_model_output: Dict[str, Any] = field(default_factory=dict)


class ReplanningBackend(Protocol):
    def replan_failed_task(self, request: ReplanRequest) -> Dict[str, Any]:
        """Return a JSON-like payload for the failed-task replacement."""


class DashScopeTaskReplanningBackend:
    """DashScope-based multimodal backend for failed-task replanning."""

    def __init__(
        self,
        *,
        model: str = "qwen-vl-max-latest",
        require_api_key: bool = True,
    ):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if require_api_key and not api_key:
            raise EnvironmentError("Missing environment variable DASHSCOPE_API_KEY; export it before running")
        if api_key:
            dashscope.api_key = api_key
        self.model = str(model)

    def replan_failed_task(self, request: ReplanRequest) -> Dict[str, Any]:
        prompt = (
            "You are a robot task replanner for a mobile manipulation system.\n"
            "You are given:\n"
            "1. the original task queue from the first planning pass,\n"
            "2. the failed task and runtime error information,\n"
            "3. the tasks already completed before the failure,\n"
            "4. the tasks still remaining after the failed task,\n"
            "5. the current environment observation after the failure,\n"
            "6. semantic-topology map candidates for base navigation.\n\n"
            "Your job is to replace only the failed task with a new short serial repair plan.\n"
            "Do not rewrite or reorder already completed tasks.\n"
            "Do not mutate the remaining queued tasks unless the replacement plan must explicitly bridge into them.\n"
            "Keep the result incremental and minimally invasive.\n\n"
            "Execution constraints:\n"
            "1. Only executors 'arm' and 'base' are allowed.\n"
            "2. Arm tasks must stay local tabletop manipulation tasks.\n"
            "3. Base tasks may use semantic navigation targets from the candidate list.\n"
            "4. Prefer a small repair plan instead of replanning the entire mission.\n"
            "5. If the failed task is no longer needed because the goal is already satisfied, return an empty replacement_tasks list and explain why.\n"
            "6. If the failure cannot be repaired safely from the current observation, set feasible=false and return no replacement tasks.\n"
            "7. Preserve queue-task field compatibility with the current runtime: arm tasks use instruction or query; base tasks use query or target_node.\n"
            "8. When possible, use goal_region / cluster_caption / target_caption to preserve navigation grounding.\n"
            "9. Respect the current observation. If an object is already on the tray / already delivered / already grasped, account for that.\n"
            "10. The output must replace only the failed task, not the whole mission.\n\n"
            "Output JSON only.\n"
            "JSON schema:\n"
            "{\n"
            '  "feasible": true,\n'
            '  "failure_mode": "goal_already_satisfied | transient_execution_failure | scene_changed | navigation_target_changed | unrecoverable",\n'
            '  "rationale": "short explanation",\n'
            '  "replacement_tasks": [\n'
            "    {\n"
            '      "name": "optional_task_name",\n'
            '      "executor": "arm" | "base",\n'
            '      "instruction": "optional for arm",\n'
            '      "query": "optional for arm/base",\n'
            '      "target_node": "optional for base",\n'
            '      "goal_region": "optional for base",\n'
            '      "cluster_caption": "optional for base",\n'
            '      "target_caption": "optional for base",\n'
            '      "reinitialize_arm": true\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Structured input:\n"
            f"{json.dumps(request.to_prompt_dict(), ensure_ascii=False, indent=2)}\n"
        )

        content: List[Dict[str, Any]] = [{"text": prompt}]
        for image in request.observation.images:
            path = Path(image.image_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Observation image does not exist: {path}")
            content.append({"image": _image_path_to_data_uri(path)})

        response = dashscope.MultiModalConversation.call(
            model=self.model,
            messages=[{"role": "user", "content": content}],
        )
        text = self._extract_text(response)
        return _safe_json_loads(text)

    @staticmethod
    def _extract_text(response: Any) -> str:
        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output")
        choices = getattr(output, "choices", None)
        if choices is None and isinstance(output, dict):
            choices = output.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"DashScope response is missing choices: {response}")
        first = choices[0]
        message = getattr(first, "message", None)
        if message is None and isinstance(first, dict):
            message = first.get("message")
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if not isinstance(content, list):
            raise RuntimeError(f"DashScope response is missing content: {response}")
        texts: List[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
            else:
                value = getattr(item, "text", None)
                if value is not None:
                    texts.append(str(value))
        merged = "\n".join(part for part in texts if str(part).strip()).strip()
        if not merged:
            raise RuntimeError(f"Failed to extract text from DashScope response: {response}")
        return merged


class TaskReplanner:
    """Standalone failed-task replanner.

    This class is intentionally not imported by the existing runtime flow.
    A caller must explicitly create it and invoke ``replan_failed_task``.
    """

    def __init__(
        self,
        *,
        backend: Optional[ReplanningBackend] = None,
        semantic_topology_adapter: Optional[SemanticTopologyAdapter] = None,
    ):
        self.backend = backend or DashScopeTaskReplanningBackend()
        self.semantic_topology_adapter = semantic_topology_adapter

    def build_request(
        self,
        *,
        original_user_instruction: str,
        original_plan: Sequence[Dict[str, Any]],
        failed_task: Dict[str, Any],
        failure_result: Dict[str, Any],
        completed_tasks: Optional[Sequence[Dict[str, Any]]] = None,
        remaining_tasks: Optional[Sequence[Dict[str, Any]]] = None,
        task_queue_results: Optional[Sequence[Dict[str, Any]]] = None,
        observation_images: Optional[Sequence[RuntimeObservationImage]] = None,
        observation_text: str = "",
        observation_extra_state: Optional[Dict[str, Any]] = None,
        semantic_candidates: Optional[Sequence[Dict[str, Any]]] = None,
        max_new_tasks: int = 5,
    ) -> ReplanRequest:
        adapter_current_region: Dict[str, Any] = {}
        adapter_candidates: List[Dict[str, Any]] = []
        if self.semantic_topology_adapter is not None:
            adapter_current_region = dict(self.semantic_topology_adapter.describe_current_region() or {})
            adapter_candidates = list(self.semantic_topology_adapter.list_candidate_regions() or [])

        merged_candidates: List[Dict[str, Any]] = []
        for item in list(semantic_candidates or []) + adapter_candidates:
            if not isinstance(item, dict):
                continue
            candidate = dict(item)
            key = (str(candidate.get("cluster_id", "")), str(candidate.get("target_node", "")))
            if key in {
                (str(existing.get("cluster_id", "")), str(existing.get("target_node", "")))
                for existing in merged_candidates
            }:
                continue
            merged_candidates.append(candidate)

        return ReplanRequest(
            failure_context=FailedTaskContext(
                original_user_instruction=str(original_user_instruction).strip(),
                original_plan=[dict(item) for item in original_plan],
                failed_task=dict(failed_task),
                failure_result=dict(failure_result),
                completed_tasks=[dict(item) for item in (completed_tasks or [])],
                remaining_tasks=[dict(item) for item in (remaining_tasks or [])],
                task_queue_results=[dict(item) for item in (task_queue_results or [])],
            ),
            observation=RuntimeObservation(
                current_region=adapter_current_region,
                textual_state=str(observation_text).strip(),
                images=[item for item in (observation_images or [])],
                extra_state=dict(observation_extra_state or {}),
            ),
            semantic_candidates=merged_candidates,
            max_new_tasks=max(0, int(max_new_tasks)),
        )

    def replan_failed_task(self, request: ReplanRequest) -> ReplanResult:
        payload = self.backend.replan_failed_task(request)
        feasible = bool(payload.get("feasible", False))
        failure_mode = str(payload.get("failure_mode", "")).strip() or "unrecoverable"
        rationale = str(payload.get("rationale", "")).strip()

        raw_replacements = payload.get("replacement_tasks", [])
        if not isinstance(raw_replacements, list):
            raise RuntimeError("replacement_tasks must be a list")

        replacement_tasks: List[Dict[str, Any]] = []
        max_new_tasks = request.max_new_tasks
        for idx, item in enumerate(raw_replacements[:max_new_tasks], start=1):
            if not isinstance(item, dict):
                continue
            replacement_tasks.append(
                _normalize_task(item, fallback_name=f"replanned_task_{idx}")
            )

        if not feasible:
            replacement_tasks = []

        merged_plan = (
            [dict(item) for item in request.failure_context.completed_tasks]
            + replacement_tasks
            + [dict(item) for item in request.failure_context.remaining_tasks]
        )

        return ReplanResult(
            feasible=feasible,
            failure_mode=failure_mode,
            rationale=rationale,
            replacement_tasks=replacement_tasks,
            merged_plan=merged_plan,
            raw_model_output=dict(payload),
        )


def build_replan_request(
    *,
    original_user_instruction: str,
    original_plan: Sequence[Dict[str, Any]],
    failed_task: Dict[str, Any],
    failure_result: Dict[str, Any],
    completed_tasks: Optional[Sequence[Dict[str, Any]]] = None,
    remaining_tasks: Optional[Sequence[Dict[str, Any]]] = None,
    task_queue_results: Optional[Sequence[Dict[str, Any]]] = None,
    observation_images: Optional[Sequence[RuntimeObservationImage]] = None,
    observation_text: str = "",
    observation_extra_state: Optional[Dict[str, Any]] = None,
    semantic_candidates: Optional[Sequence[Dict[str, Any]]] = None,
    max_new_tasks: int = 5,
    backend: Optional[ReplanningBackend] = None,
    semantic_topology_adapter: Optional[SemanticTopologyAdapter] = None,
) -> ReplanRequest:
    replanner = TaskReplanner(
        backend=backend,
        semantic_topology_adapter=semantic_topology_adapter,
    )
    return replanner.build_request(
        original_user_instruction=original_user_instruction,
        original_plan=original_plan,
        failed_task=failed_task,
        failure_result=failure_result,
        completed_tasks=completed_tasks,
        remaining_tasks=remaining_tasks,
        task_queue_results=task_queue_results,
        observation_images=observation_images,
        observation_text=observation_text,
        observation_extra_state=observation_extra_state,
        semantic_candidates=semantic_candidates,
        max_new_tasks=max_new_tasks,
    )


def replan_failed_task(
    *,
    original_user_instruction: str,
    original_plan: Sequence[Dict[str, Any]],
    failed_task: Dict[str, Any],
    failure_result: Dict[str, Any],
    completed_tasks: Optional[Sequence[Dict[str, Any]]] = None,
    remaining_tasks: Optional[Sequence[Dict[str, Any]]] = None,
    task_queue_results: Optional[Sequence[Dict[str, Any]]] = None,
    observation_images: Optional[Sequence[RuntimeObservationImage]] = None,
    observation_text: str = "",
    observation_extra_state: Optional[Dict[str, Any]] = None,
    semantic_candidates: Optional[Sequence[Dict[str, Any]]] = None,
    max_new_tasks: int = 5,
    backend: Optional[ReplanningBackend] = None,
    semantic_topology_adapter: Optional[SemanticTopologyAdapter] = None,
) -> ReplanResult:
    replanner = TaskReplanner(
        backend=backend,
        semantic_topology_adapter=semantic_topology_adapter,
    )
    request = replanner.build_request(
        original_user_instruction=original_user_instruction,
        original_plan=original_plan,
        failed_task=failed_task,
        failure_result=failure_result,
        completed_tasks=completed_tasks,
        remaining_tasks=remaining_tasks,
        task_queue_results=task_queue_results,
        observation_images=observation_images,
        observation_text=observation_text,
        observation_extra_state=observation_extra_state,
        semantic_candidates=semantic_candidates,
        max_new_tasks=max_new_tasks,
    )
    return replanner.replan_failed_task(request)

