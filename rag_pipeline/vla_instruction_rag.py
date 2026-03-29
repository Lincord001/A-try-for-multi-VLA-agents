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


LOGGER = logging.getLogger("vla_instruction_rag")

SELECT_PROMPT_TEMPLATE = (
    "你是一个移动机器人控制指令检索助手。\n"
    "用户原始指令是：'{query}'。\n"
    "下面是已经通过向量召回得到的候选 VLA 控制指令，请从中选择最适合用于最后一段微调导航的一条。\n"
    "你必须且只能回复候选编号本身，例如 CANDIDATE_2。\n"
    "候选列表如下：\n"
    "{candidate_lines}\n"
)


@dataclass
class VLAInstructionRAGArgs:
    instruction_groups: List[Dict[str, Any]]
    embedding_model: str
    selection_model: str
    cache_json: str
    top_k: int
    query_weight: float
    cluster_weight: float
    target_weight: float
    max_retry: int
    retry_wait: float


class VLAInstructionRAG:
    """Embedding + LLM-selection RAG for base VLA instruction retrieval."""

    def __init__(self, args: VLAInstructionRAGArgs):
        self.args = args
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise EnvironmentError("缺少环境变量 DASHSCOPE_API_KEY，请先 export DASHSCOPE_API_KEY=...")
        dashscope.api_key = api_key

        self.cache_path = Path(args.cache_json).resolve()
        self.entries = self._flatten_instruction_groups(args.instruction_groups)
        if not self.entries:
            raise ValueError("instruction_groups 中没有可用的 VLA 指令。")
        self._ensure_entry_embeddings()

    @staticmethod
    def _to_plain_data(payload: Any) -> Any:
        if payload is None or isinstance(payload, (str, int, float, bool)):
            return payload
        if isinstance(payload, dict):
            return {k: VLAInstructionRAG._to_plain_data(v) for k, v in payload.items()}
        if isinstance(payload, (list, tuple)):
            return [VLAInstructionRAG._to_plain_data(v) for v in payload]
        for attr_name in ("model_dump", "to_dict"):
            method = getattr(payload, attr_name, None)
            if callable(method):
                try:
                    return VLAInstructionRAG._to_plain_data(method())
                except Exception:
                    pass
        obj_dict = getattr(payload, "__dict__", None)
        if isinstance(obj_dict, dict) and obj_dict:
            return {k: VLAInstructionRAG._to_plain_data(v) for k, v in obj_dict.items() if not k.startswith("_")}
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
        if not all(isinstance(x, (int, float)) for x in vector):
            raise RuntimeError(f"DashScope TextEmbedding 向量元素类型非法: {response}")
        return np.asarray(vector, dtype=np.float64)

    @staticmethod
    def _flatten_instruction_groups(instruction_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for group_idx, group in enumerate(instruction_groups):
            group_name = str(group.get("name", f"base_group_{group_idx + 1}"))
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
            LOGGER.info("VLA instruction embedding cache fingerprint mismatch, rebuilding.")
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
                    "生成 VLA 指令向量: attempt=%d/%d model=%s",
                    attempt,
                    self.args.max_retry,
                    self.args.embedding_model,
                )
                response = dashscope.TextEmbedding.call(model=self.args.embedding_model, input=text)
                return self._extract_text_embedding(response)
            except Exception as exc:
                last_error = exc
                LOGGER.warning("VLA 指令向量化失败: attempt=%d err=%s", attempt, exc)
                if attempt < self.args.max_retry:
                    time.sleep(self.args.retry_wait)
        if last_error is not None:
            raise last_error
        raise RuntimeError("VLA 指令向量化失败，且未捕获到具体异常。")

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
                LOGGER.info("Loaded VLA instruction embedding cache: %s", self.cache_path)
                return

        LOGGER.info("Building VLA instruction embedding cache: entries=%d", len(self.entries))
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

    def _retrieve_top_k_multi(
        self,
        query_embedding: np.ndarray | None,
        cluster_embedding: np.ndarray | None,
        target_embedding: np.ndarray | None,
    ) -> List[Dict[str, Any]]:
        scored: List[Dict[str, Any]] = []
        for entry in self.entries:
            embedding = entry.get("embedding")
            if not isinstance(embedding, np.ndarray):
                continue
            component_scores: Dict[str, float] = {}
            total_score = 0.0
            total_weight = 0.0

            def _accumulate(name: str, vec: np.ndarray | None, weight: float) -> None:
                nonlocal total_score, total_weight
                if vec is None or weight <= 1e-12:
                    return
                if embedding.shape != vec.shape:
                    LOGGER.warning(
                        "Skip %s score due to embedding dim mismatch: entry=%s entry_dim=%s query_dim=%s",
                        name,
                        entry["entry_id"],
                        embedding.shape,
                        vec.shape,
                    )
                    return
                sim = self._cosine_similarity(vec, embedding)
                component_scores[name] = float(sim)
                total_score += float(weight) * float(sim)
                total_weight += float(weight)

            _accumulate("query", query_embedding, self.args.query_weight)
            _accumulate("cluster", cluster_embedding, self.args.cluster_weight)
            _accumulate("target", target_embedding, self.args.target_weight)
            if total_weight <= 1e-12:
                continue

            candidate = dict(entry)
            candidate["score"] = float(total_score / total_weight)
            candidate["component_scores"] = component_scores
            candidate.pop("embedding", None)
            scored.append(candidate)
        if not scored:
            raise RuntimeError("VLA instruction retrieval failed: no candidate has a usable embedding.")
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: max(1, int(self.args.top_k))]

    def _build_selection_prompt(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        lines = []
        for idx, candidate in enumerate(candidates, start=1):
            lines.append(
                f"[CANDIDATE_{idx}] group={candidate['group_name']} | "
                f"score={candidate['score']:.6f} | instruction={candidate['instruction']}"
            )
        return SELECT_PROMPT_TEMPLATE.format(query=query, candidate_lines="\n".join(lines))

    def _call_selection_llm(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        prompt = self._build_selection_prompt(query, candidates)
        last_error: Exception | None = None
        for attempt in range(1, self.args.max_retry + 1):
            try:
                LOGGER.info(
                    "VLA 指令候选重排调用 LLM: attempt=%d/%d model=%s",
                    attempt,
                    self.args.max_retry,
                    self.args.selection_model,
                )
                response = dashscope.Generation.call(
                    model=self.args.selection_model,
                    messages=[
                        {"role": "system", "content": "你是具身移动机器人控制指令检索助手。"},
                        {"role": "user", "content": prompt},
                    ],
                )
                return self._extract_generation_text(response)
            except Exception as exc:
                last_error = exc
                LOGGER.warning("VLA 指令候选重排失败: attempt=%d err=%s", attempt, exc)
                if attempt < self.args.max_retry:
                    time.sleep(self.args.retry_wait)
        if last_error is not None:
            raise last_error
        raise RuntimeError("VLA 指令候选重排失败，且未捕获到具体异常。")

    @staticmethod
    def _extract_candidate_index(raw_text: str, num_candidates: int) -> int:
        text = str(raw_text or "").strip().upper()
        for idx in range(1, num_candidates + 1):
            if f"CANDIDATE_{idx}" in text:
                return idx - 1
        raise RuntimeError(f"无法从 LLM 回复中提取有效候选编号，原始内容: {raw_text}")

    def retrieve_instruction(
        self,
        query: str,
        cluster_caption: str | None = None,
        target_caption: str | None = None,
    ) -> Dict[str, Any]:
        query = str(query).strip()
        if not query:
            raise ValueError("query 不能为空")

        query_embedding = self._generate_text_embedding(query)
        cluster_caption = str(cluster_caption or "").strip()
        target_caption = str(target_caption or "").strip()
        cluster_embedding = self._generate_text_embedding(cluster_caption) if cluster_caption else None
        target_embedding = self._generate_text_embedding(target_caption) if target_caption else None
        candidates = self._retrieve_top_k_multi(
            query_embedding=query_embedding,
            cluster_embedding=cluster_embedding,
            target_embedding=target_embedding,
        )
        llm_raw = self._call_selection_llm(query, candidates)
        selected_idx = self._extract_candidate_index(llm_raw, len(candidates))
        selected = dict(candidates[selected_idx])

        return {
            "query": query,
            "cluster_caption": cluster_caption,
            "target_caption": target_caption,
            "instruction": str(selected["instruction"]),
            "group_index": int(selected["group_index"]),
            "group_name": str(selected["group_name"]),
            "score": float(selected["score"]),
            "selection_model": self.args.selection_model,
            "embedding_model": self.args.embedding_model,
            "weights": {
                "query": float(self.args.query_weight),
                "cluster": float(self.args.cluster_weight),
                "target": float(self.args.target_weight),
            },
            "selected_candidate_index": int(selected_idx),
            "llm_raw": llm_raw,
            "top_k": candidates,
        }
