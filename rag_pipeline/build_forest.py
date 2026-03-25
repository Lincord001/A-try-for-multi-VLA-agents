#!/usr/bin/env python3
import os

# 在导入可能访问 HuggingFace 的依赖前设置镜像端点
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

"""
build_forest.py
---------------
Embodied-RAG 第三阶段：语义森林构建脚本。

功能：
1) 读取第二阶段输出的语义拓扑图（含 pose / semantic_caption / embedding）
2) 基于空间与语义信息构建混合距离矩阵，并执行层次聚类
3) 对每个簇调用大模型生成短摘要，作为高层语义标签
4) 在图中新增 Cluster 父节点并连接子节点，形成两层语义树结构
5) 将最终森林图保存为 forest_topological_map.json（或自定义输出路径）
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dashscope
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter


LOGGER = logging.getLogger("build_forest")

SUMMARY_PROMPT_TEMPLATE = (
    "你是一个室内空间的智能助手。这里有几段对同一个小区域内不同视角的描述：\n"
    "{captions}\n"
    "请你用一句话（不超过15个字）概括这个区域的核心功能或主要家具，例如"
    "'厨房的烹饪与洗涤区'或'包含双人床的主卧'。直接输出摘要，不要任何废话。"
)


@dataclass
class ForestArgs:
    input_json: str
    output_json: str
    alpha: float
    theta: float
    cluster_threshold: float
    summary_model: str
    max_retry: int
    log_level: str


def parse_args() -> ForestArgs:
    parser = argparse.ArgumentParser(description="Embodied-RAG 第三阶段：构建语义森林")
    parser.add_argument(
        "--input_json",
        type=str,
        default="topology_output/semantic_topological_map_text_only.json",
        help="输入语义拓扑图 JSON 路径（node-link 格式）",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="topology_output/forest_topological_map.json",
        help="输出森林拓扑图 JSON 路径",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="混合相似度中物理相似度权重 alpha，范围建议 [0,1]",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=1.0,
        help="空间指数衰减参数 theta（米），建议室内小空间取 1.0 左右",
    )
    parser.add_argument(
        "--cluster_threshold",
        type=float,
        default=0.6,
        help="层次聚类距离阈值 t（fcluster, criterion=distance）",
    )
    parser.add_argument(
        "--summary_model",
        type=str,
        default="qwen3-max-2026-01-23",
        help="DashScope 文本摘要模型名",
    )
    parser.add_argument("--max_retry", type=int, default=3, help="LLM 调用最大重试次数")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别，如 INFO/DEBUG")
    ns = parser.parse_args()
    return ForestArgs(
        input_json=ns.input_json,
        output_json=ns.output_json,
        alpha=float(ns.alpha),
        theta=max(float(ns.theta), 1e-6),
        cluster_threshold=float(ns.cluster_threshold),
        summary_model=str(ns.summary_model),
        max_retry=max(1, int(ns.max_retry)),
        log_level=str(ns.log_level).upper(),
    )


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class ForestBuilder:
    """根据语义图构建层级语义森林。"""

    def __init__(self, args: ForestArgs):
        self.args = args
        self.input_json = Path(args.input_json).resolve()
        self.output_json = Path(args.output_json).resolve()

        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise EnvironmentError("缺少环境变量 DASHSCOPE_API_KEY，请先 export DASHSCOPE_API_KEY=...")
        dashscope.api_key = api_key

        self.graph = self._load_graph()
        self.stats: Dict[str, Any] = {
            "leaf_nodes": 0,
            "cluster_count": 0,
            "cluster_sizes": {},
            "parent_nodes_created": 0,
            "hierarchy_edges_created": 0,
        }

    def _load_graph(self) -> nx.Graph:
        if not self.input_json.exists():
            raise FileNotFoundError(f"输入图谱 JSON 不存在: {self.input_json}")
        LOGGER.info("加载语义图谱: %s", self.input_json)
        with self.input_json.open("r", encoding="utf-8") as f:
            graph_dict = json.load(f)
        graph = json_graph.node_link_graph(graph_dict)
        LOGGER.info("图谱加载完成: nodes=%d edges=%d", graph.number_of_nodes(), graph.number_of_edges())
        return graph

    def _save_graph(self) -> None:
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        graph_dict = json_graph.node_link_data(self.graph)
        with self.output_json.open("w", encoding="utf-8") as f:
            json.dump(graph_dict, f, ensure_ascii=False, indent=2)
        LOGGER.info("森林图已保存: %s", self.output_json)

    @staticmethod
    def _is_valid_embedding(embedding: Any) -> bool:
        return (
            isinstance(embedding, list)
            and len(embedding) > 0
            and all(isinstance(x, (int, float)) for x in embedding)
        )

    @staticmethod
    def _is_leaf_node(node_id: Any, node_attr: Dict[str, Any]) -> bool:
        if isinstance(node_id, str) and node_id.startswith("Cluster_"):
            return False
        if node_attr.get("node_type") in {"cluster_parent", "cluster"}:
            return False
        caption = node_attr.get("semantic_caption")
        pose = node_attr.get("pose")
        embedding = node_attr.get("embedding")
        if not isinstance(caption, str) or not caption.strip():
            return False
        if not isinstance(pose, list) or len(pose) < 2:
            return False
        if not ForestBuilder._is_valid_embedding(embedding):
            return False
        return True

    def _collect_leaf_nodes(self) -> List[Tuple[Any, Dict[str, Any]]]:
        leaf_nodes: List[Tuple[Any, Dict[str, Any]]] = []
        for node_id, node_attr in self.graph.nodes(data=True):
            if self._is_leaf_node(node_id, node_attr):
                leaf_nodes.append((node_id, node_attr))
        if not leaf_nodes:
            raise RuntimeError("未找到可用于聚类的叶子节点，请检查节点的 pose/semantic_caption/embedding 字段。")
        self.stats["leaf_nodes"] = len(leaf_nodes)
        LOGGER.info("叶子节点数量: %d", len(leaf_nodes))
        return leaf_nodes

    def _build_hybrid_condensed_distance(self, leaf_nodes: List[Tuple[Any, Dict[str, Any]]]) -> np.ndarray:
        n = len(leaf_nodes)
        if n < 2:
            return np.array([], dtype=np.float64)

        poses = np.asarray([[float(n_attr["pose"][0]), float(n_attr["pose"][1])] for _, n_attr in leaf_nodes], dtype=np.float64)
        embeddings = np.asarray([n_attr["embedding"] for _, n_attr in leaf_nodes], dtype=np.float64)

        # 1) 空间相似度：S_spatial = exp(-d/theta)
        spatial_delta = poses[:, None, :] - poses[None, :, :]
        spatial_dist = np.linalg.norm(spatial_delta, axis=-1)
        s_spatial = np.exp(-spatial_dist / self.args.theta)

        # 2) 语义相似度：余弦相似度
        emb_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_norm = np.where(emb_norm == 0.0, 1e-12, emb_norm)
        embeddings_unit = embeddings / emb_norm
        s_semantic = embeddings_unit @ embeddings_unit.T
        s_semantic = np.clip(s_semantic, -1.0, 1.0)

        # 3) 混合距离：D_hybrid = 1 - S_hybrid
        s_hybrid = self.args.alpha * s_spatial + (1.0 - self.args.alpha) * s_semantic
        d_hybrid = 1.0 - s_hybrid
        d_hybrid = np.maximum(d_hybrid, 0.0)
        np.fill_diagonal(d_hybrid, 0.0)

        condensed = squareform(d_hybrid, checks=False)
        LOGGER.info(
            "混合距离矩阵构建完成: n=%d condensed_len=%d alpha=%.3f theta=%.3f",
            n,
            condensed.shape[0],
            self.args.alpha,
            self.args.theta,
        )
        return condensed

    def _cluster_leaf_nodes(self, condensed: np.ndarray, leaf_nodes: List[Tuple[Any, Dict[str, Any]]]) -> Dict[int, List[Any]]:
        if len(leaf_nodes) == 1:
            return {1: [leaf_nodes[0][0]]}

        z = linkage(condensed, method="complete")
        labels = fcluster(z, t=self.args.cluster_threshold, criterion="distance")

        clusters: Dict[int, List[Any]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(leaf_nodes[idx][0])
        self.stats["cluster_count"] = len(clusters)
        self.stats["cluster_sizes"] = {f"cluster_{k}": len(v) for k, v in clusters.items()}
        LOGGER.info(
            "层次聚类完成: clusters=%d threshold=%.3f method=complete",
            len(clusters),
            self.args.cluster_threshold,
        )
        for k, node_ids in sorted(clusters.items(), key=lambda x: x[0]):
            LOGGER.info("  - cluster_%d: %d 个子节点", k, len(node_ids))
        return clusters

    @staticmethod
    def _to_plain_data(payload: Any) -> Any:
        if payload is None or isinstance(payload, (str, int, float, bool)):
            return payload
        if isinstance(payload, dict):
            return {k: ForestBuilder._to_plain_data(v) for k, v in payload.items()}
        if isinstance(payload, (list, tuple)):
            return [ForestBuilder._to_plain_data(v) for v in payload]
        for attr_name in ("model_dump", "to_dict"):
            method = getattr(payload, attr_name, None)
            if callable(method):
                try:
                    return ForestBuilder._to_plain_data(method())
                except Exception:  # pylint: disable=broad-except
                    pass
        obj_dict = getattr(payload, "__dict__", None)
        if isinstance(obj_dict, dict) and obj_dict:
            return {k: ForestBuilder._to_plain_data(v) for k, v in obj_dict.items() if not k.startswith("_")}
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
                    txt = item.get("text")
                    if isinstance(txt, str) and txt.strip():
                        texts.append(txt.strip())
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

    def _summarize_cluster_by_llm(self, captions: List[str]) -> str:
        prompt = SUMMARY_PROMPT_TEMPLATE.format(captions="\n".join(captions))
        last_exc: Exception | None = None
        for attempt in Retrying(
            stop=stop_after_attempt(self.args.max_retry),
            wait=wait_exponential_jitter(initial=1, max=8),
            retry=retry_if_exception_type(Exception),
            reraise=False,
        ):
            with attempt:
                try:
                    response = dashscope.Generation.call(
                        model=self.args.summary_model,
                        messages=[
                            {"role": "system", "content": "你是室内空间语义抽象专家。"},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    return self._extract_generation_text(response)
                except Exception as exc:  # pylint: disable=broad-except
                    last_exc = exc
                    LOGGER.warning("LLM 摘要调用失败，第 %d 次重试: err=%s", attempt.retry_state.attempt_number, exc)
                    raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM 摘要调用失败，且未捕获到具体异常。")

    @staticmethod
    def _fallback_summary(captions: List[str]) -> str:
        if not captions:
            return "未知区域"
        text = captions[0].strip()
        if not text:
            return "未知区域"
        return text[:15]

    def _next_cluster_node_id(self) -> str:
        index = 0
        while True:
            candidate = f"Cluster_{index}"
            if candidate not in self.graph:
                return candidate
            index += 1

    def _cluster_center_pose(self, child_node_ids: List[Any]) -> List[float]:
        xy = []
        for node_id in child_node_ids:
            pose = self.graph.nodes[node_id].get("pose")
            if isinstance(pose, list) and len(pose) >= 2:
                xy.append([float(pose[0]), float(pose[1])])
        if not xy:
            return [0.0, 0.0]
        arr = np.asarray(xy, dtype=np.float64)
        center = arr.mean(axis=0)
        return [float(center[0]), float(center[1])]

    def _rebuild_forest(self, clusters: Dict[int, List[Any]]) -> None:
        for cluster_label, child_node_ids in sorted(clusters.items(), key=lambda x: x[0]):
            captions = []
            for node_id in child_node_ids:
                caption = self.graph.nodes[node_id].get("semantic_caption")
                if isinstance(caption, str) and caption.strip():
                    captions.append(caption.strip())

            try:
                summary = self._summarize_cluster_by_llm(captions)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.error("簇摘要失败，使用降级摘要: cluster_%s err=%s", cluster_label, exc)
                summary = self._fallback_summary(captions)

            parent_id = self._next_cluster_node_id()
            center_pose = self._cluster_center_pose(child_node_ids)
            self.graph.add_node(
                parent_id,
                node_type="cluster_parent",
                semantic_caption=summary,
                pose=center_pose,
                cluster_label=int(cluster_label),
                child_count=len(child_node_ids),
            )
            self.stats["parent_nodes_created"] += 1

            for child_id in child_node_ids:
                self.graph.add_edge(parent_id, child_id, edge_type="is_child_of")
                self.stats["hierarchy_edges_created"] += 1
                self.graph.nodes[child_id]["parent_cluster"] = parent_id

    def run(self) -> None:
        leaf_nodes = self._collect_leaf_nodes()
        condensed = self._build_hybrid_condensed_distance(leaf_nodes)
        clusters = self._cluster_leaf_nodes(condensed, leaf_nodes)
        self._rebuild_forest(clusters)
        self._save_graph()
        self._log_summary()

    def _log_summary(self) -> None:
        LOGGER.info("语义森林构建完成")
        LOGGER.info("输出 JSON: %s", self.output_json)
        LOGGER.info("叶子节点数: %d", self.stats["leaf_nodes"])
        LOGGER.info("父节点数: %d", self.stats["parent_nodes_created"])
        LOGGER.info("层级边数: %d", self.stats["hierarchy_edges_created"])
        LOGGER.info("聚类数: %d", self.stats["cluster_count"])
        for cluster_name, size in sorted(self.stats["cluster_sizes"].items()):
            LOGGER.info("  - %s: %d 个子节点", cluster_name, size)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    LOGGER.info("运行参数: %s", args)
    builder = ForestBuilder(args)
    builder.run()


if __name__ == "__main__":
    main()
