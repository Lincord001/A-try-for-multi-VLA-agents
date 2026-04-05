"""VLM-based task verification for post-VLA checks.

This module is intended to run after:
- arm tail-event detection, or
- execution timeout / anomaly checks.

It uses the same DashScope/Qwen multimodal calling pattern as the existing
RAG semantic annotation pipeline, but returns a structured task verdict.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import dashscope
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter


LOGGER = logging.getLogger("vlm_verifier")

VERDICTS = {
    "success",
    "recoverable_failure",
    "irrecoverable_failure",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a robot task-execution verifier. You will inspect images captured after an arm pick/place task. "
    "Your job is not to restate the image. Your job is to judge whether the task has succeeded, or failed but is still recoverable."
)


@dataclass
class VLMVerifierConfig:
    model: str = "qwen3.5-plus-2026-02-15"
    max_retry: int = 3
    log_level: str = "INFO"
    require_api_key: bool = True


@dataclass
class VLMVerificationResult:
    verdict: str
    confidence: float
    rationale: str
    recoverable: bool
    target_identified: bool
    raw_text: str
    raw_response: Any


class VLMVerifier:
    """Structured VLM verifier built on DashScope MultiModalConversation."""

    def __init__(self, config: Optional[VLMVerifierConfig] = None):
        self.config = config or VLMVerifierConfig()
        logging.basicConfig(
            level=getattr(logging, str(self.config.log_level).upper(), logging.INFO),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if self.config.require_api_key and not api_key:
            raise EnvironmentError("缺少环境变量 DASHSCOPE_API_KEY，请先 export DASHSCOPE_API_KEY=...")
        if api_key:
            dashscope.api_key = api_key

    def verify_arm_task(
        self,
        *,
        agent_image_path: str,
        wrist_image_path: str,
        instruction: str,
        trigger_reason: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> VLMVerificationResult:
        verify_t0 = time.perf_counter()
        print(
            "[ARM-VLM][timing] verify_start "
            f"trigger={trigger_reason} instruction={instruction} "
            f"agent={agent_image_path} wrist={wrist_image_path}"
        )
        prompt = self._build_arm_prompt(
            instruction=instruction,
            trigger_reason=trigger_reason,
            trigger_reason_explanation=self._describe_trigger_reason(trigger_reason),
            extra_context=extra_context or {},
        )
        call_t0 = time.perf_counter()
        response = self._call_multimodal_conversation(
            image_items=[
                {"label": "agent_view", "path": agent_image_path},
                {"label": "wrist_view", "path": wrist_image_path},
            ],
            user_prompt=prompt,
        )
        call_dt = time.perf_counter() - call_t0
        print(f"[ARM-VLM][timing] request_done dt={call_dt:.3f}s")
        parse_t0 = time.perf_counter()
        raw_text = self._extract_text(response)
        parsed = self._parse_json_response(raw_text)
        verdict = str(parsed.get("verdict", "")).strip()
        if verdict not in VERDICTS:
            raise RuntimeError("VLM 返回了未知 verdict: %r | raw=%s" % (verdict, raw_text))

        confidence = float(parsed.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        rationale = str(parsed.get("rationale", "")).strip()
        target_identified = self._parse_target_identified(parsed.get("target_identified", None))
        if not target_identified and verdict == "recoverable_failure":
            verdict = "irrecoverable_failure"
            rationale = (
                (rationale + " ") if rationale else ""
            ) + "模型未能可靠识别任务目标物体，因此按更保守的不可恢复失败处理。"
        recoverable = verdict == "recoverable_failure"
        parse_dt = time.perf_counter() - parse_t0
        total_dt = time.perf_counter() - verify_t0
        print(
            "[ARM-VLM][timing] verify_done "
            f"total={total_dt:.3f}s request={call_dt:.3f}s parse={parse_dt:.3f}s "
            f"target_identified={target_identified} verdict={verdict} confidence={confidence:.3f}"
        )
        return VLMVerificationResult(
            verdict=verdict,
            confidence=confidence,
            rationale=rationale,
            recoverable=recoverable,
            target_identified=target_identified,
            raw_text=raw_text,
            raw_response=response,
        )

    def _build_arm_prompt(
        self,
        *,
        instruction: str,
        trigger_reason: str,
        trigger_reason_explanation: str,
        extra_context: Dict[str, Any],
    ) -> str:
        context_lines = []
        for key, value in extra_context.items():
            context_lines.append("- %s: %s" % (key, value))
        context_block = "\n".join(context_lines) if context_lines else "- none"

        return (
            "Judge the outcome of the current arm pick/place task from the images.\n"
            "You will receive two images with fixed roles:\n"
            "- Image 1 is agent_view: the wider scene view, mainly for checking the target area and whether the object has reached the correct place.\n"
            "- Image 2 is wrist_view: the close-up gripper view, mainly for checking whether the gripper still holds the object and whether release has finished.\n"
            "Usually use agent_view for the overall layout and wrist_view for local gripper details. "
            "However, if wrist_view clearly shows that the target object has fallen, left the workbench or tray, is stuck at an edge, or is in another obviously abnormal state, that anomaly evidence must take priority even if agent_view is unclear.\n\n"
            "Task instruction:\n"
            "%s\n\n"
            "Trigger reason:\n"
            "%s\n"
            "Meaning:\n"
            "%s\n\n"
            "Extra context:\n"
            "%s\n\n"
            "You must output exactly one JSON object and nothing else.\n"
            "JSON schema:\n"
            "{\n"
            '  "target_identified": "yes" | "no",\n'
            '  "verdict": "success" | "recoverable_failure" | "irrecoverable_failure",\n'
            '  "confidence": 0.0-1.0,\n'
            '  "rationale": "one or two short sentences"\n'
            "}\n\n"
            "Judgment criteria:\n"
            "- First identify the target object and target relation from the task instruction. Do not get distracted by more salient but irrelevant objects in the image.\n"
            "- You must explicitly decide target_identified as yes or no first. Output yes only when the target object can be reliably matched to the task instruction from the images.\n"
            "- Check whether the visible object matches the target attributes in the instruction, such as color, category, and target region. Only an attribute-consistent object can be treated as the target.\n"
            "- If the images show only similar but non-matching objects, such as wrong color, wrong category, or wrong spatial relation, do not treat them as the target and do not use them to justify recoverable_failure.\n"
            "- If the target object cannot be reliably identified in either image, handle it explicitly as target not identifiable instead of defaulting to some other salient object.\n"
            "- success: the target object has reached the correct region or target placement specified by the instruction, and the arm or gripper is no longer holding it.\n"
            "- recoverable_failure: the task has not succeeded yet, but the target object is still within the operable area, reachable, not dropped to the floor, and not clearly outside the task scene, so the arm could retry.\n"
            "- irrecoverable_failure: the scene is clearly not directly recoverable, for example the target object has fallen to the floor, left the workbench or tray surface, moved to an unreachable area, suffered a severe collision, tipped into a state outside direct retry, or otherwise left the task scene.\n"
            "- If either view clearly shows that the target object has fallen to the floor or left the operable surface, prefer irrecoverable_failure over recoverable_failure.\n"
            "- For timeout_without_tail_event checks, if the target object is not reliably identifiable or only mismatching objects are visible, do not default to recoverable_failure; prefer irrecoverable_failure.\n"
            "- If the two images conflict, judge conservatively. Do not claim success hastily, and prioritize anomaly evidence directly tied to the target object.\n"
            "- Do not fail the task just because the arm is still visible in the image. Focus on the relation between the task target, the target area, and the operable area.\n"
            "- You must reason from both the instruction and the images. Do not answer only because the scene looks normal.\n"
        ) % (instruction, trigger_reason, trigger_reason_explanation, context_block)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_multimodal_conversation(
        self,
        *,
        image_items: Sequence[Dict[str, str]],
        user_prompt: str,
    ) -> Any:
        resolved_items = []
        for item in image_items:
            label = str(item["label"])
            path = Path(str(item["path"])).expanduser().resolve()
            resolved_items.append({"label": label, "path": path})
        resolved = [item["path"] for item in resolved_items]
        if not resolved:
            raise ValueError("image_items 不能为空")
        missing = [str(p) for p in resolved if not p.exists()]
        if missing:
            raise FileNotFoundError("以下图像不存在: %s" % missing)

        user_content: List[Dict[str, str]] = []
        for item in resolved_items:
            user_content.append({"text": "The label of the following image is: %s" % item["label"]})
            user_content.append({"image": item["path"].as_uri()})
        user_content.append({"text": user_prompt})

        try:
            return dashscope.MultiModalConversation.call(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": [{"text": DEFAULT_SYSTEM_PROMPT}]},
                    {"role": "user", "content": user_content},
                ],
            )
        except Exception as exc:
            LOGGER.warning("VLM 调用异常，将重试: images=%s err=%s", [str(p) for p in resolved], exc)
            raise

    @staticmethod
    def _describe_trigger_reason(trigger_reason: str) -> str:
        mapping = {
            "arm_tail_event_detected": (
                "The monitor detected a likely tail event: the arm was holding an object, then released at a relatively low position, and then lifted after release. "
                "This usually means the VLA may have completed a placement, so visual confirmation is needed."
            ),
            "arm_timeout_without_tail_event": (
                "The monitor did not observe a clear tail placement event within the time limit. "
                "This usually means the task may be stuck, failed, or already in an abnormal state, so vision must decide whether the current scene is success, recoverable failure, or irrecoverable failure."
            ),
            "manual_check": (
                "This is a manually triggered visual review. Judge the final state directly from the images and the task instruction."
            ),
        }
        return mapping.get(
            str(trigger_reason),
            "This is a runtime visual status check. Use the task instruction and the images to judge whether the task succeeded or remains recoverable.",
        )

    @staticmethod
    def _to_plain_data(payload: Any) -> Any:
        if payload is None or isinstance(payload, (str, int, float, bool)):
            return payload
        if isinstance(payload, dict):
            return {k: VLMVerifier._to_plain_data(v) for k, v in payload.items()}
        if isinstance(payload, (list, tuple)):
            return [VLMVerifier._to_plain_data(v) for v in payload]

        for attr_name in ("model_dump", "to_dict"):
            method = getattr(payload, attr_name, None)
            if callable(method):
                try:
                    return VLMVerifier._to_plain_data(method())
                except Exception:
                    pass

        obj_dict = getattr(payload, "__dict__", None)
        if isinstance(obj_dict, dict) and obj_dict:
            return {k: VLMVerifier._to_plain_data(v) for k, v in obj_dict.items() if not k.startswith("_")}
        return payload

    @staticmethod
    def _extract_text_from_content(content: Any) -> str:
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            texts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
                elif isinstance(item, str) and item.strip():
                    texts.append(item.strip())
            if texts:
                return " ".join(texts)
        return ""

    @staticmethod
    def _collect_texts(payload: Any) -> List[str]:
        texts: List[str] = []
        if isinstance(payload, str):
            stripped = payload.strip()
            if stripped:
                texts.append(stripped)
            return texts
        if isinstance(payload, dict):
            val = payload.get("text")
            if isinstance(val, str) and val.strip():
                texts.append(val.strip())
            for sub in payload.values():
                texts.extend(VLMVerifier._collect_texts(sub))
            return texts
        if isinstance(payload, (list, tuple)):
            for item in payload:
                texts.extend(VLMVerifier._collect_texts(item))
        return texts

    def _extract_text(self, response: Any) -> str:
        if response is None:
            raise RuntimeError("DashScope 返回为空")

        status_code = getattr(response, "status_code", None)
        if status_code is not None and int(status_code) != 200:
            message = getattr(response, "message", None) or str(response)
            raise RuntimeError("DashScope 调用失败: status_code=%s, message=%s" % (status_code, message))

        response_data = self._to_plain_data(response)
        output_obj = response_data.get("output") if isinstance(response_data, dict) else None

        if isinstance(output_obj, dict):
            choices = output_obj.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message")
                    if isinstance(message, dict):
                        text = self._extract_text_from_content(message.get("content"))
                        if text:
                            return text

        texts = self._collect_texts(output_obj if output_obj is not None else response_data)
        if not texts:
            raise RuntimeError("无法从 DashScope 响应中提取文本: %s" % response)

        for candidate in texts:
            lowered = candidate.strip().lower()
            if lowered and lowered not in {"stop", "length", "tool_calls", "content_filter", "null"}:
                return candidate
        return texts[0]

    @staticmethod
    def _extract_json_snippet(text: str) -> str:
        text = str(text).strip()
        if not text:
            raise RuntimeError("VLM 返回空文本")
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise RuntimeError("VLM 返回中未找到 JSON 对象: %s" % text)
        return text[start : end + 1]

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        snippet = self._extract_json_snippet(text)
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError as exc:
            raise RuntimeError("VLM 返回 JSON 解析失败: %s | text=%s" % (exc, text))
        if not isinstance(parsed, dict):
            raise RuntimeError("VLM 返回 JSON 不是对象: %s" % text)
        return parsed

    @staticmethod
    def _parse_target_identified(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"yes", "true", "1"}:
            return True
        if text in {"no", "false", "0", "", "none", "null"}:
            return False
        return False


__all__ = [
    "VERDICTS",
    "VLMVerifier",
    "VLMVerifierConfig",
    "VLMVerificationResult",
]
