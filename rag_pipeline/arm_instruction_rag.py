#!/usr/bin/env python3
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import dashscope
import numpy as np


LOGGER = logging.getLogger("arm_instruction_rag")

SELECT_PROMPT_TEMPLATE = (
    "You are a normalization assistant for arm-task instructions.\n"
    "The user's high-level task is: '{query}'.\n"
    "Below are the candidate standard arm control instructions currently available in the system.\n"
    "Your job is not to rewrite the task. Your job is to decide which candidate standard instruction is the closest semantic match to the user's intent.\n"
    "Decision rules:\n"
    "1. First determine which high-level skill family the user task belongs to: tabletop pick, tabletop place, load onto a receiving surface, retrieve from a receiving surface, or other tabletop transfer.\n"
    "2. If a candidate belongs to the same high-level skill family as the user task, prefer selecting the closest one instead of returning NO_MATCH merely because color, container name, or source/destination wording is not exactly the same.\n"
    "3. For generic references such as 'a cup' or 'a plate', match any candidate that fits that category by default. Do not reject only because color or identity is missing.\n"
    "4. Historical training labels for locations or containers may be imperfect. Terms such as plate, tray, and table may sometimes be approximate aliases. If the user's intent is clearly a workbench-side loading, placing, or transfer action, prefer approximate matching based on training semantics rather than strict literal token matching.\n"
    "5. If the user's task is to place an object onto a receiving container, carrying surface, or tray, and a candidate such as 'place the mug on the plate' expresses a near-equivalent receiving-surface placement action, prefer treating it as the same skill family rather than returning NO_MATCH immediately.\n"
    "6. Return NO_MATCH only if the user's intent is clearly inconsistent with every candidate at the high-level action-type level.\n"
    "7. Do not invent a new instruction. You must either choose one candidate or return NO_MATCH.\n"
    "Reply strictly in the following format, and both lines must be present:\n"
    "DECISION: CANDIDATE_2\n"
    "REASON: Explain in one sentence why it matches; if no candidate fits, explain why it is NO_MATCH.\n"
    "DECISION must be either one candidate ID or NO_MATCH.\n"
    "Candidate list:\n"
    "{candidate_lines}\n"
)


@dataclass
class ArmInstructionRAGArgs:
    instruction_groups: List[Dict[str, Any]]
    embedding_model: str
    selection_model: str
    cache_json: str
    top_k: int
    max_retry: int
    retry_wait: float


class ArmInstructionRAG:
    """Text retrieval + semantic normalization for arm instructions."""

    def __init__(self, args: ArmInstructionRAGArgs):
        self.args = args
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise EnvironmentError("缺少环境变量 DASHSCOPE_API_KEY，请先 export DASHSCOPE_API_KEY=...")
        dashscope.api_key = api_key

        self.cache_path = Path(args.cache_json).resolve()
        self.entries = self._flatten_instruction_groups(args.instruction_groups)
        if not self.entries:
            raise ValueError("instruction_groups 中没有可用的 ARM 指令。")
        self._ensure_entry_embeddings()

    @staticmethod
    def _to_plain_data(payload: Any) -> Any:
        if payload is None or isinstance(payload, (str, int, float, bool)):
            return payload
        if isinstance(payload, dict):
            return {k: ArmInstructionRAG._to_plain_data(v) for k, v in payload.items()}
        if isinstance(payload, (list, tuple)):
            return [ArmInstructionRAG._to_plain_data(v) for v in payload]
        for attr_name in ("model_dump", "to_dict"):
            method = getattr(payload, attr_name, None)
            if callable(method):
                try:
                    return ArmInstructionRAG._to_plain_data(method())
                except Exception:
                    pass
        obj_dict = getattr(payload, "__dict__", None)
        if isinstance(obj_dict, dict) and obj_dict:
            return {k: ArmInstructionRAG._to_plain_data(v) for k, v in obj_dict.items() if not k.startswith("_")}
        return payload

    @staticmethod
    def _extract_text_from_content(content: Any) -> str:
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            texts: List[str] = []
            for item in content:
                if isinstance(item, str) and item.strip():
                    texts.append(item.strip())
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
            if texts:
                return " ".join(texts)
        return ""

    def _extract_generation_text(self, response: Any) -> str:
        if response is None:
            raise RuntimeError("DashScope Generation 返回为空")
        status_code = getattr(response, "status_code", None)
        if status_code is not None and int(status_code) != 200:
            message = getattr(response, "message", None) or str(response)
            raise RuntimeError(f"DashScope Generation 调用失败: status_code={status_code}, message={message}")

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
                    text = first_choice.get("text")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
            text = output_obj.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
        raise RuntimeError(f"无法从 DashScope Generation 响应中提取文本: {response}")

    def _extract_text_embedding(self, response: Any) -> np.ndarray:
        if response is None:
            raise RuntimeError("DashScope TextEmbedding 返回为空")
        status_code = getattr(response, "status_code", None)
        if status_code is not None and int(status_code) != 200:
            message = getattr(response, "message", None) or str(response)
            raise RuntimeError(f"DashScope TextEmbedding 调用失败: status_code={status_code}, message={message}")

        output_obj = getattr(response, "output", None)
        if output_obj is None and isinstance(response, dict):
            output_obj = response.get("output")
        embeddings = getattr(output_obj, "embeddings", None)
        if embeddings is None and isinstance(output_obj, dict):
            embeddings = output_obj.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise RuntimeError(f"DashScope TextEmbedding 响应缺少 embeddings: {response}")

        first_item = embeddings[0]
        vector = getattr(first_item, "embedding", None)
        if vector is None and isinstance(first_item, dict):
            vector = first_item.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise RuntimeError(f"DashScope TextEmbedding 响应缺少 embedding 向量: {response}")
        return np.asarray(vector, dtype=np.float64)

    @staticmethod
    def _flatten_instruction_groups(instruction_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for group_idx, group in enumerate(instruction_groups):
            group_name = str(group.get("name", f"arm_group_{group_idx + 1}"))
            for inst_idx, instruction in enumerate(group.get("instructions", [])):
                text = str(instruction).strip()
                if not text:
                    continue
                entries.append(
                    {
                        "entry_id": f"{group_name}:{inst_idx}",
                        "group_index": int(group_idx),
                        "group_name": group_name,
                        "instruction": text,
                    }
                )
        return entries

    def _fingerprint(self) -> str:
        payload = {
            "embedding_model": self.args.embedding_model,
            "entries": [
                {
                    "entry_id": entry["entry_id"],
                    "group_name": entry["group_name"],
                    "instruction": entry["instruction"],
                }
                for entry in self.entries
            ],
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def _load_cache(self) -> Dict[str, Any] | None:
        if not self.cache_path.exists():
            return None
        with self.cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("fingerprint") != self._fingerprint():
            LOGGER.info("ARM instruction embedding cache fingerprint mismatch, rebuilding.")
            return None
        return payload

    def _save_cache(self, payload: Dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _generate_text_embedding(self, text: str) -> np.ndarray:
        last_error: Exception | None = None
        for attempt in range(1, self.args.max_retry + 1):
            try:
                LOGGER.info(
                    "生成 ARM 指令向量: attempt=%d/%d model=%s",
                    attempt,
                    self.args.max_retry,
                    self.args.embedding_model,
                )
                response = dashscope.TextEmbedding.call(model=self.args.embedding_model, input=text)
                return self._extract_text_embedding(response)
            except Exception as exc:
                last_error = exc
                LOGGER.warning("ARM 指令向量化失败: attempt=%d err=%s", attempt, exc)
                if attempt < self.args.max_retry:
                    time.sleep(self.args.retry_wait)
        if last_error is not None:
            raise last_error
        raise RuntimeError("ARM 指令向量化失败，且未捕获到具体异常。")

    def _ensure_entry_embeddings(self) -> None:
        cache_payload = self._load_cache()
        if cache_payload is not None:
            cached_entries = {
                str(item["entry_id"]): item
                for item in cache_payload.get("entries", [])
                if isinstance(item, dict) and "entry_id" in item and "embedding" in item
            }
            complete = True
            for entry in self.entries:
                cached = cached_entries.get(entry["entry_id"])
                if cached is None:
                    complete = False
                    break
                entry["embedding"] = np.asarray(cached["embedding"], dtype=np.float64)
            if complete:
                LOGGER.info("Loaded ARM instruction embedding cache: %s", self.cache_path)
                return

        LOGGER.info("Building ARM instruction embedding cache: entries=%d", len(self.entries))
        for entry in self.entries:
            entry["embedding"] = self._generate_text_embedding(entry["instruction"])
        self._save_cache(
            {
                "fingerprint": self._fingerprint(),
                "embedding_model": self.args.embedding_model,
                "entries": [
                    {
                        "entry_id": entry["entry_id"],
                        "group_index": entry["group_index"],
                        "group_name": entry["group_name"],
                        "instruction": entry["instruction"],
                        "embedding": entry["embedding"].astype(float).tolist(),
                    }
                    for entry in self.entries
                ],
            }
        )

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator <= 1e-12:
            return -1.0
        return float(np.dot(a, b) / denominator)

    def _retrieve_top_k(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        scored: List[Dict[str, Any]] = []
        for entry in self.entries:
            embedding = entry.get("embedding")
            if not isinstance(embedding, np.ndarray):
                continue
            if embedding.shape != query_embedding.shape:
                continue
            candidate = dict(entry)
            candidate["score"] = self._cosine_similarity(query_embedding, embedding)
            candidate.pop("embedding", None)
            scored.append(candidate)
        if not scored:
            raise RuntimeError("ARM instruction retrieval failed: no candidate has a usable embedding.")
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: max(1, int(self.args.top_k))]

    def _build_selection_prompt(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        lines = []
        for idx, candidate in enumerate(candidates, start=1):
            lines.append(f"[CANDIDATE_{idx}] {candidate['instruction']}")
        return SELECT_PROMPT_TEMPLATE.format(query=query, candidate_lines="\n".join(lines))

    def _call_selection_llm(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        prompt = self._build_selection_prompt(query, candidates)
        last_error: Exception | None = None
        for attempt in range(1, self.args.max_retry + 1):
            try:
                LOGGER.info(
                    "ARM 指令标准化调用 LLM: attempt=%d/%d model=%s",
                    attempt,
                    self.args.max_retry,
                    self.args.selection_model,
                )
                response = dashscope.Generation.call(
                    model=self.args.selection_model,
                    messages=[
                        {"role": "system", "content": "You are a normalization assistant for arm-task instructions."},
                        {"role": "user", "content": prompt},
                    ],
                )
                return self._extract_generation_text(response)
            except Exception as exc:
                last_error = exc
                LOGGER.warning("ARM 指令标准化失败: attempt=%d err=%s", attempt, exc)
                if attempt < self.args.max_retry:
                    time.sleep(self.args.retry_wait)
        if last_error is not None:
            raise last_error
        raise RuntimeError("ARM 指令标准化失败，且未捕获到具体异常。")

    @staticmethod
    def _extract_candidate_index(raw_text: str, num_candidates: int) -> int | None:
        text = str(raw_text or "").strip().upper()
        for line in text.splitlines():
            if line.strip().startswith("DECISION:"):
                decision_text = line.split(":", 1)[1].strip()
                if "NO_MATCH" in decision_text:
                    return None
                for idx in range(1, num_candidates + 1):
                    if f"CANDIDATE_{idx}" in decision_text:
                        return idx - 1
        if "NO_MATCH" in text:
            return None
        for idx in range(1, num_candidates + 1):
            if f"CANDIDATE_{idx}" in text:
                return idx - 1
        raise RuntimeError(f"无法从 LLM 回复中提取有效候选编号或 NO_MATCH，原始内容: {raw_text}")

    @staticmethod
    def _extract_reason(raw_text: str) -> str:
        text = str(raw_text or "").strip()
        if not text:
            return ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("REASON:"):
                return stripped.split(":", 1)[1].strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) >= 2:
            return lines[1]
        return text

    def retrieve_instruction(self, query: str) -> Dict[str, Any]:
        query = str(query).strip()
        if not query:
            raise ValueError("query 不能为空")

        query_embedding = self._generate_text_embedding(query)
        candidates = self._retrieve_top_k(query_embedding)
        llm_raw = self._call_selection_llm(query, candidates)
        llm_reason = self._extract_reason(llm_raw)
        selected_idx = self._extract_candidate_index(llm_raw, len(candidates))

        if selected_idx is None:
            return {
                "query": query,
                "instruction": None,
                "group_index": -1,
                "group_name": "NO_MATCH",
                "score": 0.0,
                "matched": False,
                "llm_reason": llm_reason,
                "selection_model": self.args.selection_model,
                "embedding_model": self.args.embedding_model,
                "selected_candidate_index": -1,
                "llm_raw": llm_raw,
                "top_k": candidates,
            }

        selected = dict(candidates[selected_idx])
        return {
            "query": query,
            "instruction": str(selected["instruction"]),
            "group_index": int(selected["group_index"]),
            "group_name": str(selected["group_name"]),
            "score": float(selected["score"]),
            "matched": True,
            "llm_reason": llm_reason,
            "selection_model": self.args.selection_model,
            "embedding_model": self.args.embedding_model,
            "selected_candidate_index": int(selected_idx),
            "llm_raw": llm_raw,
            "top_k": candidates,
        }
