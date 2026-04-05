#!/usr/bin/env python3
import os

# 在导入可能访问 HuggingFace 的依赖前设置镜像端点
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dashscope
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph


LOGGER = logging.getLogger("rag_navigator")
CLUSTER_ID_PATTERN = re.compile(r"Cluster_\d+")

MACRO_PROMPT_TEMPLATE = (
    "You are the navigation planner for an embodied robot. The user's request is: '{query}'.\n"
    "The environment currently contains the following candidate regions:\n"
    "{cluster_lines}\n"
    "Reason about which region is the most likely target of the user's request. "
    "You must reply with the selected region ID only (for example: Cluster_1). "
    "Do not output any explanation or punctuation."
)


@dataclass
class NavigatorArgs:
    query: str
    input_json: str
    macro_model: str
    embedding_model: str
    max_retry: int
    retry_wait: float
    log_level: str


def parse_args() -> NavigatorArgs:
    parser = argparse.ArgumentParser(description="Embodied-RAG 第四阶段：自上而下检索导航")
    parser.add_argument("--query", type=str, required=True, help='用户自然语言指令，例如 "我要去主卧拿衣服"')
    parser.add_argument(
        "--input_json",
        type=str,
        default="topology_output/forest_topological_map.json",
        help="输入森林拓扑图 JSON 路径（node-link 格式）",
    )
    parser.add_argument("--macro_model", type=str, default="qwen-max", help="宏观区域决策模型名")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-v4",
        help="用户指令向量化模型名",
    )
    parser.add_argument("--max_retry", type=int, default=3, help="DashScope 调用最大重试次数")
    parser.add_argument("--retry_wait", type=float, default=1.5, help="失败重试等待秒数")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别，如 INFO/DEBUG")
    ns = parser.parse_args()
    query = str(ns.query).strip()
    if not query:
        raise ValueError("--query 不能为空")
    return NavigatorArgs(
        query=query,
        input_json=ns.input_json,
        macro_model=ns.macro_model,
        embedding_model=ns.embedding_model,
        max_retry=max(1, int(ns.max_retry)),
        retry_wait=max(0.1, float(ns.retry_wait)),
        log_level=str(ns.log_level).upper(),
    )


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class RAGNavigator:
    """Embodied-RAG 第四阶段：先选父区域，再在子节点内做向量检索。"""

    def __init__(self, args: NavigatorArgs):
        self.args = args
        self.input_json = Path(args.input_json).resolve()
        self.map_base_dir = self.input_json.parent

        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise EnvironmentError("缺少环境变量 DASHSCOPE_API_KEY，请先 export DASHSCOPE_API_KEY=...")
        dashscope.api_key = api_key

        self.graph = self._load_graph()
        self.cluster_info = self._collect_cluster_parents()

    def _load_graph(self) -> nx.Graph:
        if not self.input_json.exists():
            raise FileNotFoundError(f"输入图谱 JSON 不存在: {self.input_json}")
        LOGGER.info("加载森林图谱: %s", self.input_json)
        with self.input_json.open("r", encoding="utf-8") as f:
            graph_dict = json.load(f)
        graph = json_graph.node_link_graph(graph_dict, edges="links")
        LOGGER.info("图谱加载完成: nodes=%d edges=%d", graph.number_of_nodes(), graph.number_of_edges())
        return graph

    @staticmethod
    def _to_plain_data(payload: Any) -> Any:
        if payload is None or isinstance(payload, (str, int, float, bool)):
            return payload
        if isinstance(payload, dict):
            return {k: RAGNavigator._to_plain_data(v) for k, v in payload.items()}
        if isinstance(payload, (list, tuple)):
            return [RAGNavigator._to_plain_data(v) for v in payload]
        for attr_name in ("model_dump", "to_dict"):
            method = getattr(payload, attr_name, None)
            if callable(method):
                try:
                    return RAGNavigator._to_plain_data(method())
                except Exception:  # pylint: disable=broad-except
                    pass
        obj_dict = getattr(payload, "__dict__", None)
        if isinstance(obj_dict, dict) and obj_dict:
            return {k: RAGNavigator._to_plain_data(v) for k, v in obj_dict.items() if not k.startswith("_")}
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

    def _collect_cluster_parents(self) -> Dict[str, str]:
        cluster_info: Dict[str, str] = {}
        for node_id, node_attr in self.graph.nodes(data=True):
            if node_attr.get("node_type") == "cluster_parent":
                caption = node_attr.get("semantic_caption")
                if isinstance(caption, str) and caption.strip():
                    cluster_info[str(node_id)] = caption.strip()
                else:
                    cluster_info[str(node_id)] = "unknown area"
        if not cluster_info:
            raise RuntimeError("图中没有可用的 cluster_parent 节点。")
        LOGGER.info("候选父区域数量: %d", len(cluster_info))
        return cluster_info

    @staticmethod
    def _cluster_sort_key(cluster_id: str) -> Tuple[int, str]:
        match = re.search(r"(\d+)$", cluster_id)
        if match:
            return (int(match.group(1)), cluster_id)
        return (10**9, cluster_id)

    def _build_macro_prompt(self, query: str) -> str:
        lines = [f"[{cluster_id}]: {caption}" for cluster_id, caption in sorted(self.cluster_info.items(), key=lambda x: self._cluster_sort_key(x[0]))]
        return MACRO_PROMPT_TEMPLATE.format(query=query, cluster_lines="\n".join(lines))

    def _call_macro_llm(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.args.max_retry + 1):
            try:
                LOGGER.info("宏观区域抉择调用 LLM: attempt=%d/%d model=%s", attempt, self.args.max_retry, self.args.macro_model)
                response = dashscope.Generation.call(
                    model=self.args.macro_model,
                    messages=[
                        {"role": "system", "content": "You are the navigation planner for an embodied robot."},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = self._extract_generation_text(response)
                LOGGER.debug("宏观决策原始回复: %s", text)
                return text
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                LOGGER.warning("宏观区域抉择失败: attempt=%d err=%s", attempt, exc)
                if attempt < self.args.max_retry:
                    time.sleep(self.args.retry_wait)
        if last_error is not None:
            raise last_error
        raise RuntimeError("宏观区域抉择失败，且未捕获到具体异常。")

    def _extract_cluster_id(self, raw_text: str) -> str:
        matches = CLUSTER_ID_PATTERN.findall(raw_text or "")
        for candidate in matches:
            if candidate in self.cluster_info:
                return candidate

        # 若模型回答了不存在的 Cluster_x，则回退到第一个可识别 ID（后续可据需要继续增强）。
        if matches:
            LOGGER.warning("LLM 返回的 Cluster ID 不在图中: %s", matches[0])

        raise RuntimeError(f"无法从 LLM 回复中提取有效 Cluster ID，原始内容: {raw_text}")

    def _generate_query_embedding(self, query: str) -> np.ndarray:
        last_error: Exception | None = None
        for attempt in range(1, self.args.max_retry + 1):
            try:
                LOGGER.info("生成查询向量: attempt=%d/%d model=%s", attempt, self.args.max_retry, self.args.embedding_model)
                response = dashscope.TextEmbedding.call(model=self.args.embedding_model, input=query)
                return self._extract_text_embedding(response)
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                LOGGER.warning("查询向量化失败: attempt=%d err=%s", attempt, exc)
                if attempt < self.args.max_retry:
                    time.sleep(self.args.retry_wait)
        if last_error is not None:
            raise last_error
        raise RuntimeError("查询向量化失败，且未捕获到具体异常。")

    def _collect_children_of_cluster(self, cluster_id: str) -> List[Any]:
        if cluster_id not in self.graph:
            raise KeyError(f"父节点不存在: {cluster_id}")
        children: List[Any] = []
        for _, neighbor_id, edge_attr in self.graph.edges(cluster_id, data=True):
            if edge_attr.get("edge_type") == "is_child_of":
                children.append(neighbor_id)
        if not children:
            raise RuntimeError(f"父节点 {cluster_id} 未找到任何 is_child_of 子节点。")
        LOGGER.info("目标父区域子节点数: cluster=%s children=%d", cluster_id, len(children))
        return children

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator <= 1e-12:
            return -1.0
        return float(np.dot(a, b) / denominator)

    def _micro_retrieve_top1(self, query_embedding: np.ndarray, child_node_ids: List[Any]) -> Tuple[Any, Dict[str, Any], float]:
        best_node_id: Any = None
        best_attr: Dict[str, Any] = {}
        best_score = -2.0

        for node_id in child_node_ids:
            node_attr = self.graph.nodes[node_id]
            embedding = node_attr.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                LOGGER.debug("跳过子节点（缺少 embedding）: %s", node_id)
                continue
            if not all(isinstance(x, (int, float)) for x in embedding):
                LOGGER.debug("跳过子节点（embedding 元素类型非法）: %s", node_id)
                continue

            node_embedding = np.asarray(embedding, dtype=np.float64)
            if node_embedding.shape != query_embedding.shape:
                LOGGER.debug(
                    "跳过子节点（向量维度不一致）: %s node_dim=%s query_dim=%s",
                    node_id,
                    node_embedding.shape,
                    query_embedding.shape,
                )
                continue

            score = self._cosine_similarity(query_embedding, node_embedding)
            if score > best_score:
                best_node_id = node_id
                best_attr = dict(node_attr)
                best_score = score

        if best_node_id is None:
            raise RuntimeError("微观检索失败：目标父区域下没有可用 embedding 的子节点。")
        return best_node_id, best_attr, best_score

    @staticmethod
    def _extract_xy_pose(node_attr: Dict[str, Any]) -> Tuple[float, float]:
        pose = node_attr.get("pose")
        if not isinstance(pose, list) or len(pose) < 2:
            raise RuntimeError(f"目标叶子节点 pose 缺失或格式非法: {pose}")
        return float(pose[0]), float(pose[1])

    def _resolve_panorama_path(self, panorama_path: str | None) -> str | None:
        if not panorama_path:
            return None
        path_obj = Path(str(panorama_path))
        if path_obj.is_absolute():
            return str(path_obj.resolve())
        return str((self.map_base_dir / path_obj).resolve())

    def retrieve_top_leaf(self, query: str) -> Dict[str, Any]:
        """对外复用接口：输入文本指令，返回 top-1 叶子节点检索结果。"""
        query = str(query).strip()
        if not query:
            raise ValueError("query 不能为空")

        macro_prompt = self._build_macro_prompt(query)
        macro_raw = self._call_macro_llm(macro_prompt)
        selected_cluster = self._extract_cluster_id(macro_raw)
        selected_cluster_caption = self.cluster_info[selected_cluster]
        LOGGER.info("宏观区域决策完成: cluster=%s caption=%s", selected_cluster, selected_cluster_caption)

        query_embedding = self._generate_query_embedding(query)
        children = self._collect_children_of_cluster(selected_cluster)
        top_leaf_id, top_leaf_attr, top_score = self._micro_retrieve_top1(query_embedding, children)
        leaf_caption = str(top_leaf_attr.get("semantic_caption", ""))
        x, y = self._extract_xy_pose(top_leaf_attr)
        panorama_path = self._resolve_panorama_path(top_leaf_attr.get("panorama_path"))

        return {
            "query": query,
            "cluster_id": str(selected_cluster),
            "cluster_caption": selected_cluster_caption,
            "target_node": str(top_leaf_id),
            "target_caption": leaf_caption,
            "target_image_path": panorama_path,
            "score": float(top_score),
            "target_xy": [float(x), float(y)],
        }

    def run(self) -> None:
        LOGGER.info("开始执行 Embodied-RAG 第四阶段检索流程")
        LOGGER.info("用户指令: %s", self.args.query)
        result = self.retrieve_top_leaf(self.args.query)

        print("\n================ Embodied-RAG Retrieval Report ================")
        print(f"[User Query] {result['query']}")
        print(f"[Macro Decision] {result['cluster_id']} | Region Summary: {result['cluster_caption']}")
        print(
            f"[Micro Match] {result['target_node']} | Caption: {result['target_caption']} "
            f"| Cosine Similarity: {result['score']:.6f}"
        )
        print(
            f"[Final Navigation Coordinates] (x, y) = "
            f"({result['target_xy'][0]:.6f}, {result['target_xy'][1]:.6f})"
        )
        print("======================================================\n")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    LOGGER.info("运行参数: %s", args)
    navigator = RAGNavigator(args)
    navigator.run()


if __name__ == "__main__":
    main()
