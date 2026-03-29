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
    "你是机器人任务执行验证器。你会看到一个机械臂执行 pick/place 任务后的现场图像。"
    "你的职责不是复述画面，而是判断任务是否已经成功，或者是否失败但仍可恢复。"
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
            "请根据图像判断当前机械臂 pick/place 任务的执行结果。\n"
            "你将收到两张图像，并且它们的语义固定如下：\n"
            "- 第1张图是 agent_view：远景主视角，主要用于观察目标区域、物体是否已放到正确位置。\n"
            "- 第2张图是 wrist_view：夹爪近景视角，主要用于观察夹爪是否仍然持物、是否已经完成释放。\n"
            "判断时通常以 agent_view 观察整体布局，以 wrist_view 观察夹爪附近细节。"
            "但如果 wrist_view 清楚显示目标物体已经掉落、离开工作台面/托盘、卡在边缘或处于明显异常位置，"
            "则必须优先使用该异常证据，不要因为 agent_view 中看不清就忽略。\n\n"
            "任务指令:\n"
            "%s\n\n"
            "触发检查原因:\n"
            "%s\n"
            "含义解释:\n"
            "%s\n\n"
            "补充上下文:\n"
            "%s\n\n"
            "你必须只输出一个 JSON 对象，不要输出任何额外文本。\n"
            "JSON schema:\n"
            "{\n"
            '  "target_identified": "yes" | "no",\n'
            '  "verdict": "success" | "recoverable_failure" | "irrecoverable_failure",\n'
            '  "confidence": 0.0-1.0,\n'
            '  "rationale": "一句到两句中文简述"\n'
            "}\n\n"
            "判断标准:\n"
            "- 第一步，先从任务指令中识别目标物体及目标关系，不要被画面里其他更显眼但不相关的物体干扰。\n"
            "- 你必须先显式判断 target_identified 是 yes 还是 no。只有在你能根据图像可靠确认目标物体与任务指令匹配时，才能输出 yes。\n"
            "- 你必须先检查图中看到的物体是否与任务指令中的目标属性一致，例如颜色、类别、目标区域等。只有属性一致的物体才能当作目标物体。\n"
            "- 如果图中只看到了与指令不匹配的相似物体，例如颜色不对、类别不对、位置关系不对，不能把它当作目标物体，也不能据此判断任务可恢复。\n"
            "- 如果两张图里都无法可靠识别目标物体，必须明确按“目标物体不可确认”处理，而不是默认把别的显眼物体当成目标。\n"
            "- success: 目标物体已经按指令到达正确区域或目标位置，且机械臂/夹爪没有继续持有它。\n"
            "- recoverable_failure: 当前未成功，但目标物体仍在可操作区域内、可达、未掉落到地面、未明显脱离任务场景，机械臂可以重新尝试。\n"
            "- irrecoverable_failure: 当前场景已明显不可恢复，例如目标物体掉到地面、离开工作台/托盘等可操作平面、掉到不可达区域、严重碰撞、明显倾覆并脱离可直接重试状态，或完全脱离任务场景。\n"
            "- 如果任一视角清楚显示目标物体已经掉落到地面或离开可操作平面，应优先判为 irrecoverable_failure，而不是 recoverable_failure。\n"
            "- 对于 timeout_without_tail_event 这种超时检查，如果目标物体在图中不可可靠识别，或者只能看到与指令不匹配的其他物体，默认不要判为 recoverable_failure；应优先考虑 irrecoverable_failure。\n"
            "- 如果两张图信息冲突，请保守判断，不要贸然判 success；同时优先关注与目标物体直接相关的异常证据。\n"
            "- 不要因为机械臂还停在画面里就判失败，重点看任务目标物与目标区域、可操作区域之间的关系。\n"
            "- 你必须结合任务指令判断，不要只基于“画面看起来正常”来回答。\n"
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
            user_content.append({"text": "下面这张图的标签是: %s" % item["label"]})
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
                "监视器检测到一个疑似任务尾部动作：先持物/闭夹，随后在较低位置释放，并在释放后上抬。"
                "这通常意味着 VLA 可能已经完成了放置动作，因此需要视觉确认是否真的成功。"
            ),
            "arm_timeout_without_tail_event": (
                "监视器在预设时限内没有看到明显的尾部放置动作。"
                "这通常意味着任务可能卡住、失败，或者已经处于异常状态，因此需要视觉判断当前场景是成功、可恢复失败还是不可恢复失败。"
            ),
            "manual_check": (
                "这是一次人工触发的视觉复核，请直接根据图像和任务指令判断最终状态。"
            ),
        }
        return mapping.get(
            str(trigger_reason),
            "这是一次运行时视觉状态检查，请结合任务指令和图像判断任务是否成功，或者是否仍可恢复。",
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
