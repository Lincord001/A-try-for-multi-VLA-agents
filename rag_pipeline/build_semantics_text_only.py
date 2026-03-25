#!/usr/bin/env python3
import os

# 在导入可能访问 HuggingFace 的依赖前设置镜像端点，避免默认访问 huggingface.co 超时
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

"""
build_semantics.py
------------------
Embodied-RAG 第二阶段：语义赋能脚本。

功能：
1) 读取 topological_map.json（NetworkX node-link JSON）
2) 对每个节点对应全景图调用 Qwen-VL 生成语义描述
3) 使用 DashScope 多模态向量模型生成图文融合 embedding
4) 将 semantic_caption + embedding 写回图节点
5) 支持断点续传与每 N 个节点增量落盘
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import dashscope
import networkx as nx
from networkx.readwrite import json_graph
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter
from tqdm import tqdm


LOGGER = logging.getLogger("build_semantics")

SEMANTIC_PROMPT = (
    "你是一个移动机器人的视觉中枢。这是一张由机器人左、前、右三个视角水平拼接而成的全景图。"
    "请用一到两句话，简明扼要地描述你当前所在的空间类型（如走廊、卧室、厨房、工作区等）"
    "以及你看到的标志性家具或物品。不要描述图片拼接的缝隙，直接描述环境。"
)


@dataclass
class SemanticArgs:
    input_json: str
    output_json: str
    caption_output_json: str
    dashscope_model: str
    save_every: int
    log_level: str


def parse_args() -> SemanticArgs:
    parser = argparse.ArgumentParser(description="Embodied-RAG 第二阶段：拓扑图语义赋能")
    parser.add_argument(
        "--input_json",
        type=str,
        default="topology_output/topological_map.json",
        help="输入拓扑图 JSON 路径（node-link 格式）",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="topology_output/semantic_topological_map_text_only.json",
        help="输出语义拓扑图 JSON 路径",
    )
    parser.add_argument(
        "--caption_output_json",
        type=str,
        default="topology_output/semantic_topological_map_text_only_captions_only.json",
        help="仅包含节点描述文本的输出 JSON 路径",
    )
    parser.add_argument(
        "--dashscope_model",
        type=str,
        default="qwen3.5-plus-2026-02-15",
        help="DashScope 多模态模型名，如 qwen3.5-plus-2026-02-15",
    )
    parser.add_argument("--save_every", type=int, default=10, help="每处理多少个节点进行一次增量保存")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别，如 INFO/DEBUG")
    ns = parser.parse_args()
    return SemanticArgs(
        input_json=ns.input_json,
        output_json=ns.output_json,
        caption_output_json=ns.caption_output_json,
        dashscope_model=ns.dashscope_model,
        save_every=max(1, int(ns.save_every)),
        log_level=str(ns.log_level).upper(),
    )


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class SemanticAnnotator:
    """负责图谱语义标注与向量化。"""

    def __init__(self, args: SemanticArgs):
        self.args = args
        self.input_json = Path(args.input_json).resolve()
        self.output_json = Path(args.output_json).resolve()
        self.caption_output_json = Path(args.caption_output_json).resolve()
        self.map_base_dir = self.input_json.parent

        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise EnvironmentError("缺少环境变量 DASHSCOPE_API_KEY，请先 export DASHSCOPE_API_KEY=...")
        dashscope.api_key = api_key

        LOGGER.info("使用 DashScope 文本向量模型: text-embedding-v4")
        self.graph = self._load_graph()

        self.stats: Dict[str, int] = {
            "total_nodes": 0,
            "processed_nodes": 0,
            "skipped_nodes": 0,
            "failed_nodes": 0,
            "save_times": 0,
        }

    def _load_graph(self) -> nx.Graph:
        # 优先读取 output_json（若已存在）以支持中断后直接续跑；否则读取 input_json。
        source_json = self.output_json if self.output_json.exists() else self.input_json
        if not source_json.exists():
            raise FileNotFoundError(f"拓扑图 JSON 不存在: {source_json}")

        self.map_base_dir = source_json.parent
        LOGGER.info("加载图谱: %s", source_json)
        with source_json.open("r", encoding="utf-8") as f:
            graph_dict = json.load(f)
        graph = json_graph.node_link_graph(graph_dict)
        LOGGER.info("图谱加载完成: nodes=%d edges=%d", graph.number_of_nodes(), graph.number_of_edges())
        return graph

    def _save_graph(self) -> None:
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        graph_dict = json_graph.node_link_data(self.graph)
        with self.output_json.open("w", encoding="utf-8") as f:
            json.dump(graph_dict, f, ensure_ascii=False, indent=2)
        self._save_caption_only_json()
        self.stats["save_times"] += 1

    def _save_caption_only_json(self) -> None:
        self.caption_output_json.parent.mkdir(parents=True, exist_ok=True)
        captions = []
        for node_id, node_attr in self.graph.nodes(data=True):
            caption = node_attr.get("semantic_caption")
            if self._is_valid_caption(caption):
                captions.append({"node_id": node_id, "semantic_caption": caption.strip()})

        payload = {
            "caption_model": self.args.dashscope_model,
            "captions": captions,
        }
        with self.caption_output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _is_valid_caption(caption: Any) -> bool:
        if not isinstance(caption, str) or not caption.strip():
            return False
        if caption.strip().lower() in {"stop", "length", "tool_calls", "content_filter", "null"}:
            return False
        return True

    @staticmethod
    def _already_annotated(node_attr: Dict[str, Any]) -> bool:
        caption = node_attr.get("semantic_caption")
        embedding = node_attr.get("embedding")
        if not SemanticAnnotator._is_valid_caption(caption):
            return False
        if not isinstance(embedding, list) or len(embedding) == 0:
            return False
        return True

    def _resolve_panorama_path(self, panorama_path: str) -> Path:
        path_obj = Path(panorama_path)
        if path_obj.is_absolute():
            return path_obj
        return (self.map_base_dir / path_obj).resolve()

    @staticmethod
    def _to_plain_data(payload: Any) -> Any:
        """将 DashScope SDK 对象递归转为 Python 基础类型。"""
        if payload is None or isinstance(payload, (str, int, float, bool)):
            return payload
        if isinstance(payload, dict):
            return {k: SemanticAnnotator._to_plain_data(v) for k, v in payload.items()}
        if isinstance(payload, (list, tuple)):
            return [SemanticAnnotator._to_plain_data(v) for v in payload]

        # DashScope SDK 对象常见暴露方式：model_dump()/to_dict()/__dict__
        for attr_name in ("model_dump", "to_dict"):
            method = getattr(payload, attr_name, None)
            if callable(method):
                try:
                    return SemanticAnnotator._to_plain_data(method())
                except Exception:  # pylint: disable=broad-except
                    pass

        obj_dict = getattr(payload, "__dict__", None)
        if isinstance(obj_dict, dict) and obj_dict:
            return {k: SemanticAnnotator._to_plain_data(v) for k, v in obj_dict.items() if not k.startswith("_")}

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
                texts.extend(SemanticAnnotator._collect_texts(sub))
            return texts
        if isinstance(payload, (list, tuple)):
            for item in payload:
                texts.extend(SemanticAnnotator._collect_texts(item))
        return texts

    def _extract_caption(self, response: Any) -> str:
        if response is None:
            raise RuntimeError("DashScope 返回为空")

        status_code = getattr(response, "status_code", None)
        if status_code is not None and int(status_code) != 200:
            message = getattr(response, "message", None) or str(response)
            raise RuntimeError(f"DashScope 调用失败: status_code={status_code}, message={message}")

        response_data = self._to_plain_data(response)
        output_obj = response_data.get("output") if isinstance(response_data, dict) else None

        # 优先走 OpenAI 兼容结构，避免误提取 finish_reason=stop 之类字段。
        if isinstance(output_obj, dict):
            choices = output_obj.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message")
                    if isinstance(message, dict):
                        caption = self._extract_text_from_content(message.get("content"))
                        if caption:
                            return caption

        texts = self._collect_texts(output_obj if output_obj is not None else response_data)
        if not texts:
            raise RuntimeError(f"无法从 DashScope 响应中提取文本: {response}")

        for candidate in texts:
            lowered = candidate.strip().lower()
            if lowered and lowered not in {"stop", "length", "tool_calls", "content_filter", "null"}:
                return candidate
        return texts[0]

    def _extract_text_embedding(self, response: Any) -> List[float]:
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
        return [float(x) for x in vector]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _generate_caption(self, image_path: Path) -> str:
        # 每次调用都做异常透传，交给 Tenacity 重试控制。
        try:
            response = dashscope.MultiModalConversation.call(
                model=self.args.dashscope_model,
                messages=[
                    {"role": "system", "content": [{"text": SEMANTIC_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"image": image_path.resolve().as_uri()},
                            {"text": SEMANTIC_PROMPT},
                        ],
                    },
                ],
            )
            return self._extract_caption(response)
        except Exception as exc:
            LOGGER.warning("VLM 调用异常，将重试: image=%s err=%s", image_path, exc)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _generate_text_embedding(self, caption: str) -> List[float]:
        # 每次调用都做异常透传，交给 Tenacity 重试控制。
        try:
            response = dashscope.TextEmbedding.call(model="text-embedding-v4", input=caption)
            return self._extract_text_embedding(response)
        except Exception as exc:
            LOGGER.warning("文本 embedding 调用异常，将重试: caption=%s err=%s", caption, exc)
            raise

    def run(self) -> None:
        nodes_data = list(self.graph.nodes(data=True))
        self.stats["total_nodes"] = len(nodes_data)
        LOGGER.info("开始语义赋能: total_nodes=%d model=%s", self.stats["total_nodes"], self.args.dashscope_model)

        success_since_last_save = 0
        for node_id, node_attr in tqdm(nodes_data, desc="SemanticAnnotating", unit="node"):
            if self._already_annotated(node_attr):
                self.stats["skipped_nodes"] += 1
                continue

            try:
                panorama_rel = node_attr.get("panorama_path")
                if not isinstance(panorama_rel, str) or not panorama_rel.strip():
                    raise ValueError("节点缺少 panorama_path")

                panorama_abs = self._resolve_panorama_path(panorama_rel)
                if not panorama_abs.exists():
                    raise FileNotFoundError(f"全景图不存在: {panorama_abs}")

                caption = self._generate_caption(panorama_abs)
                embedding = self._generate_text_embedding(caption)

                node_attr["semantic_caption"] = caption
                node_attr["embedding"] = embedding

                self.stats["processed_nodes"] += 1
                success_since_last_save += 1

                if success_since_last_save >= self.args.save_every:
                    self._save_graph()
                    LOGGER.info(
                        "已增量保存: processed=%d skipped=%d failed=%d output=%s",
                        self.stats["processed_nodes"],
                        self.stats["skipped_nodes"],
                        self.stats["failed_nodes"],
                        self.output_json,
                    )
                    success_since_last_save = 0

            except Exception as exc:  # pylint: disable=broad-except
                self.stats["failed_nodes"] += 1
                LOGGER.error("节点处理失败: node_id=%s err=%s", node_id, exc)

        # 最终保存，确保最后不足 save_every 的部分也落盘。
        self._save_graph()
        self._log_summary()

    def _log_summary(self) -> None:
        LOGGER.info("语义赋能完成")
        LOGGER.info("输出 JSON: %s", self.output_json)
        LOGGER.info("描述输出 JSON: %s", self.caption_output_json)
        LOGGER.info("总节点数: %d", self.stats["total_nodes"])
        LOGGER.info("本次成功节点数: %d", self.stats["processed_nodes"])
        LOGGER.info("跳过节点数: %d", self.stats["skipped_nodes"])
        LOGGER.info("失败节点数: %d", self.stats["failed_nodes"])
        LOGGER.info("保存次数: %d", self.stats["save_times"])


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    LOGGER.info("运行参数: %s", args)
    annotator = SemanticAnnotator(args)
    annotator.run()


if __name__ == "__main__":
    main()
