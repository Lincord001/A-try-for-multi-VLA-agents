#!/usr/bin/env python3
"""
build_topology.py
-----------------
Embodied-RAG 第一阶段：拓扑记忆图谱构建脚本。

功能：
1) 读取 base 模式 LeRobot 高频轨迹数据
2) 执行三重过滤（运动模糊 / 位移阈值 / 可选 CLIP 语义）
3) 生成关键帧节点 + 面包屑轨迹边属性的 NetworkX 图
4) 执行跨 Episode 自动缝合
5) 导出 topological_map.json 与全景图文件
"""

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# 与部署脚本保持一致：强制走 Hugging Face 镜像。
# 必须在任何可能触发 Hugging Face 初始化的三方库 import 前设置。
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"

import networkx as nx
import numpy as np
from PIL import Image
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from networkx.readwrite import json_graph


LOGGER = logging.getLogger("build_topology")


@dataclass
class TopologyArgs:
    dataset_dir: str
    output_dir: str
    repo_name: Optional[str]
    enable_clip: bool
    clip_model: str
    clip_device: str
    clip_similarity_thresh: float
    turn_diff_thresh: float
    min_distance: float
    min_angle_deg: float
    intra_stitch_dist_thresh: float
    stitch_dist_thresh: float
    max_episodes: Optional[int]
    log_level: str


def parse_args() -> TopologyArgs:
    parser = argparse.ArgumentParser(description="构建 Embodied-RAG 拓扑记忆图谱")
    parser.add_argument("--dataset_dir", type=str, required=True, help="LeRobot 数据集根目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--repo_name", type=str, default=None, help="可选：LeRobotDataset 的 repo_name")
    parser.add_argument("--enable_clip", action="store_true", help="启用 CLIP 语义过滤")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="CLIP 模型名")
    parser.add_argument("--clip_device", type=str, default="cuda", help="CLIP 设备，例: cuda/cpu")
    parser.add_argument("--clip_similarity_thresh", type=float, default=0.95, help="CLIP 余弦相似度阈值")
    parser.add_argument("--turn_diff_thresh", type=float, default=4.0, help="A 过滤器阈值 |v_left-v_right|")
    parser.add_argument("--min_distance", type=float, default=0.3, help="B 过滤器最小距离阈值 (m)")
    parser.add_argument("--min_angle_deg", type=float, default=30.0, help="B 过滤器最小角度阈值 (deg)")
    parser.add_argument(
        "--intra_stitch_dist_thresh",
        type=float,
        default=0.1,
        help="Episode 内空间近邻补边阈值 (m)",
    )
    parser.add_argument("--stitch_dist_thresh", type=float, default=0.1, help="跨 Episode 缝合阈值 (m)")
    parser.add_argument("--max_episodes", type=int, default=None, help="可选：仅处理前 N 个 Episode")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别，如 INFO/DEBUG")
    ns = parser.parse_args()
    return TopologyArgs(
        dataset_dir=ns.dataset_dir,
        output_dir=ns.output_dir,
        repo_name=ns.repo_name,
        enable_clip=bool(ns.enable_clip),
        clip_model=ns.clip_model,
        clip_device=ns.clip_device,
        clip_similarity_thresh=float(ns.clip_similarity_thresh),
        turn_diff_thresh=float(ns.turn_diff_thresh),
        min_distance=float(ns.min_distance),
        min_angle_deg=float(ns.min_angle_deg),
        intra_stitch_dist_thresh=float(ns.intra_stitch_dist_thresh),
        stitch_dist_thresh=float(ns.stitch_dist_thresh),
        max_episodes=ns.max_episodes,
        log_level=str(ns.log_level).upper(),
    )


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_output_dirs(output_dir: Path) -> Tuple[Path, Path]:
    panoramas_dir = output_dir / "panoramas"
    output_dir.mkdir(parents=True, exist_ok=True)
    panoramas_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, panoramas_dir


def to_numpy(value) -> np.ndarray:
    if value is None:
        return np.array([])
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def wrap_angle_rad(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def pose_xy_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


class DatasetLoader:
    """负责 LeRobot 数据集加载与按 Episode/帧迭代。"""

    def __init__(self, dataset_dir: Path, repo_name: Optional[str]):
        self.dataset_dir = dataset_dir
        self.repo_name = repo_name
        self.dataset = self._load_dataset()
        self.num_episodes = int(self.dataset.num_episodes)
        self.total_frames = int(len(self.dataset))
        self.episode_data_index = self.dataset.episode_data_index

    def _build_repo_candidates(self) -> List[str]:
        candidates: List[str] = []
        if self.repo_name:
            candidates.append(self.repo_name)
        base = self.dataset_dir.name
        if base:
            candidates.append(base)
            if base.startswith("demo_data_"):
                candidates.append(base.replace("demo_data_", "omy_", 1))

        dedup: List[str] = []
        for name in candidates:
            if name and name not in dedup:
                dedup.append(name)
        return dedup

    def _load_dataset(self) -> LeRobotDataset:
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"dataset_dir 不存在: {self.dataset_dir}")

        candidates = self._build_repo_candidates()
        if not candidates:
            raise ValueError("没有可用的 repo_name 候选，请显式传入 --repo_name")

        last_error: Optional[Exception] = None
        for cand in candidates:
            try:
                LOGGER.info("尝试加载数据集: repo_name=%s root=%s", cand, self.dataset_dir)
                ds = LeRobotDataset(cand, root=str(self.dataset_dir))
                LOGGER.info("数据集加载成功: repo_name=%s episodes=%d frames=%d", cand, ds.num_episodes, len(ds))
                return ds
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                LOGGER.warning("加载失败: repo_name=%s, err=%s", cand, exc)

        raise RuntimeError(f"无法加载 LeRobotDataset，最后错误: {last_error}")

    def get_episode_range(self, ep_idx: int) -> Tuple[int, int]:
        from_idx = int(self.episode_data_index["from"][ep_idx].item())
        to_idx = int(self.episode_data_index["to"][ep_idx].item())
        return from_idx, to_idx

    def iter_episode_frames(self, ep_idx: int) -> Iterable[Tuple[int, int, Dict]]:
        start, end = self.get_episode_range(ep_idx)
        for local_idx, abs_idx in enumerate(range(start, end)):
            yield local_idx, abs_idx, self.dataset[abs_idx]


class ImageProcessor:
    """负责图像解码、全景拼接与保存。"""

    CAM_ORDER = ("left", "front", "right")

    def __init__(self, panoramas_dir: Path):
        self.panoramas_dir = panoramas_dir

    @staticmethod
    def _to_uint8_hwc(img) -> Optional[np.ndarray]:
        arr = to_numpy(img)
        if arr.size == 0:
            return None
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3:
            return None
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.dtype != np.uint8:
            max_val = float(np.max(arr)) if arr.size else 0.0
            if max_val <= 1.0:
                arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return arr

    @staticmethod
    def _resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
        if img.shape[0] == h:
            return img
        w = max(1, int(img.shape[1] * h / img.shape[0]))
        return np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))

    def build_panorama(self, frame_item: Dict) -> Optional[np.ndarray]:
        images = []
        for cam in self.CAM_ORDER:
            key = f"observation.images.{cam}"
            if key not in frame_item:
                return None
            img = self._to_uint8_hwc(frame_item[key])
            if img is None:
                return None
            images.append(img)

        target_h = min(im.shape[0] for im in images)
        aligned = [self._resize_to_height(im, target_h) for im in images]
        return np.concatenate(aligned, axis=1)

    def save_panorama(self, panorama: np.ndarray, ep_idx: int, frame_idx: int) -> Path:
        filename = f"ep{ep_idx}_frame{frame_idx}.jpg"
        out_path = self.panoramas_dir / filename
        Image.fromarray(panorama).save(out_path, format="JPEG", quality=95)
        return out_path


class SemanticFilter:
    """可选 CLIP 语义过滤器。"""

    def __init__(self, enabled: bool, model_name: str, device: str, threshold: float):
        self.enabled = enabled
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self.last_embedding = None
        self.model = None
        self.processor = None
        self.torch = None
        self._init_model_if_needed()

    def _init_model_if_needed(self) -> None:
        if not self.enabled:
            return
        try:
            import torch  # pylint: disable=import-outside-toplevel
            from transformers import CLIPModel, CLIPProcessor  # pylint: disable=import-outside-toplevel

            self.torch = torch
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            LOGGER.info("CLIP 已启用: model=%s device=%s", self.model_name, self.device)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("CLIP 初始化失败，自动降级为仅物理过滤: %s", exc)
            self.enabled = False
            self.model = None
            self.processor = None
            self.torch = None

    def reset_episode(self) -> None:
        self.last_embedding = None

    def _encode_image(self, panorama: np.ndarray):
        pil_image = Image.fromarray(panorama)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = self.torch.nn.functional.normalize(features, dim=-1)
        return features

    def should_skip(self, panorama: np.ndarray) -> Tuple[bool, Optional[float], Optional[object]]:
        if not self.enabled:
            return False, None, None

        current_embedding = self._encode_image(panorama)
        if self.last_embedding is None:
            return False, None, current_embedding

        similarity = float(
            self.torch.nn.functional.cosine_similarity(self.last_embedding, current_embedding, dim=-1).item()
        )
        skip = similarity > self.threshold
        return skip, similarity, current_embedding

    def set_reference_embedding(self, embedding) -> None:
        self.last_embedding = embedding


class TopologyBuilder:
    """执行核心拓扑构建流程。"""

    def __init__(self, args: TopologyArgs):
        self.args = args
        self.output_dir = Path(args.output_dir).resolve()
        _, panoramas_dir = ensure_output_dirs(self.output_dir)
        self.loader = DatasetLoader(Path(args.dataset_dir).resolve(), args.repo_name)
        self.image_processor = ImageProcessor(panoramas_dir)
        self.semantic_filter = SemanticFilter(
            enabled=args.enable_clip,
            model_name=args.clip_model,
            device=args.clip_device,
            threshold=args.clip_similarity_thresh,
        )
        self.graph = nx.Graph()
        self.node_pose_map: Dict[str, np.ndarray] = {}
        self.nodes_by_episode: Dict[int, List[str]] = {}
        self.stats = {
            "total_episodes": 0,
            "total_frames": 0,
            "skipped_missing_fields": 0,
            "skipped_a_motion_blur": 0,
            "skipped_b_physical_small_move": 0,
            "skipped_c_semantic": 0,
            "keyframes": 0,
            "intra_episode_edges": 0,
            "intra_episode_spatial_edges": 0,
            "cross_episode_edges": 0,
        }

    @staticmethod
    def _extract_pose(frame_item: Dict) -> Optional[np.ndarray]:
        if "base_pose" not in frame_item:
            return None
        pose = to_numpy(frame_item["base_pose"]).astype(np.float32).reshape(-1)
        if pose.shape[0] < 3:
            return None
        return pose[:3]

    @staticmethod
    def _extract_action(frame_item: Dict) -> Optional[np.ndarray]:
        if "action" not in frame_item:
            return None
        action = to_numpy(frame_item["action"]).astype(np.float32).reshape(-1)
        if action.shape[0] < 2:
            return None
        return action[:2]

    def _interceptor_a(self, action: np.ndarray) -> bool:
        return abs(float(action[0] - action[1])) > self.args.turn_diff_thresh

    def _interceptor_b(self, pose: np.ndarray, prev_key_pose: np.ndarray) -> bool:
        delta_d = pose_xy_distance(pose, prev_key_pose)
        delta_theta = abs(wrap_angle_rad(float(pose[2] - prev_key_pose[2])))
        theta_thresh_rad = math.radians(self.args.min_angle_deg)
        return (delta_d < self.args.min_distance) and (delta_theta < theta_thresh_rad)

    def _add_node(
        self,
        node_id: str,
        pose: np.ndarray,
        panorama_path: Path,
        episode_id: int,
        frame_id: int,
        dataset_index: int,
    ) -> None:
        rel_path = panorama_path.relative_to(self.output_dir).as_posix()
        self.graph.add_node(
            node_id,
            pose=[float(x) for x in pose.tolist()],
            panorama_path=rel_path,
            episode_id=int(episode_id),
            frame_id=int(frame_id),
            dataset_index=int(dataset_index),
        )
        self.node_pose_map[node_id] = pose.copy()
        self.nodes_by_episode.setdefault(episode_id, []).append(node_id)
        self.stats["keyframes"] += 1

    def _add_intra_edge(self, node_a: str, node_b: str, breadcrumbs: List[List[float]]) -> None:
        pose_a = self.node_pose_map[node_a]
        pose_b = self.node_pose_map[node_b]
        weight = pose_xy_distance(pose_a, pose_b)
        self.graph.add_edge(
            node_a,
            node_b,
            weight=float(weight),
            raw_trajectory=[list(p) for p in breadcrumbs],
            edge_type="intra_episode",
        )
        self.stats["intra_episode_edges"] += 1

    def process_episodes(self) -> None:
        max_eps = self.loader.num_episodes if self.args.max_episodes is None else min(self.args.max_episodes, self.loader.num_episodes)
        self.stats["total_episodes"] = int(max_eps)
        LOGGER.info("开始处理 Episodes: %d / %d", max_eps, self.loader.num_episodes)

        for ep_idx in tqdm(range(max_eps), desc="Episodes", unit="ep"):
            start, end = self.loader.get_episode_range(ep_idx)
            frame_count = max(0, end - start)
            trajectory_breadcrumbs: List[List[float]] = []
            prev_key_node: Optional[str] = None
            prev_key_pose: Optional[np.ndarray] = None
            self.semantic_filter.reset_episode()

            frame_iter = self.loader.iter_episode_frames(ep_idx)
            for local_idx, abs_idx, frame_item in tqdm(
                frame_iter,
                total=frame_count,
                desc=f"Ep{ep_idx} Frames",
                leave=False,
                unit="frame",
            ):
                self.stats["total_frames"] += 1
                pose = self._extract_pose(frame_item)
                action = self._extract_action(frame_item)
                if pose is None or action is None:
                    self.stats["skipped_missing_fields"] += 1
                    LOGGER.debug("跳过缺字段帧: ep=%d frame=%d abs=%d", ep_idx, local_idx, abs_idx)
                    continue

                is_first_keyframe = prev_key_node is None

                if not is_first_keyframe:
                    if self._interceptor_a(action):
                        trajectory_breadcrumbs.append([float(x) for x in pose.tolist()])
                        self.stats["skipped_a_motion_blur"] += 1
                        continue

                    if self._interceptor_b(pose, prev_key_pose):
                        trajectory_breadcrumbs.append([float(x) for x in pose.tolist()])
                        self.stats["skipped_b_physical_small_move"] += 1
                        continue

                panorama = self.image_processor.build_panorama(frame_item)
                if panorama is None:
                    self.stats["skipped_missing_fields"] += 1
                    if not is_first_keyframe:
                        trajectory_breadcrumbs.append([float(x) for x in pose.tolist()])
                    LOGGER.debug("跳过缺图像帧: ep=%d frame=%d abs=%d", ep_idx, local_idx, abs_idx)
                    continue

                clip_embedding = None
                if not is_first_keyframe and self.semantic_filter.enabled:
                    skip_semantic, similarity, clip_embedding = self.semantic_filter.should_skip(panorama)
                    if skip_semantic:
                        trajectory_breadcrumbs.append([float(x) for x in pose.tolist()])
                        self.stats["skipped_c_semantic"] += 1
                        LOGGER.debug(
                            "语义过滤跳过: ep=%d frame=%d similarity=%.4f",
                            ep_idx,
                            local_idx,
                            similarity if similarity is not None else -1.0,
                        )
                        continue

                node_id = f"ep{ep_idx}_frame{local_idx}"
                pano_path = self.image_processor.save_panorama(panorama, ep_idx, local_idx)
                self._add_node(node_id, pose, pano_path, ep_idx, local_idx, abs_idx)

                if prev_key_node is not None:
                    self._add_intra_edge(prev_key_node, node_id, trajectory_breadcrumbs)
                    trajectory_breadcrumbs.clear()

                prev_key_node = node_id
                prev_key_pose = pose

                if self.semantic_filter.enabled:
                    if clip_embedding is None:
                        _, _, clip_embedding = self.semantic_filter.should_skip(panorama)
                    self.semantic_filter.set_reference_embedding(clip_embedding)

    def stitch_across_episodes(self) -> None:
        episodes = sorted(self.nodes_by_episode.keys())
        if len(episodes) < 2:
            return

        episode_pairs = []
        for i, ep_i in enumerate(episodes):
            for ep_j in episodes[i + 1 :]:
                episode_pairs.append((ep_i, ep_j))

        LOGGER.info("开始跨 Episode 缝合: 待比较 episode 对数=%d", len(episode_pairs))
        for ep_i, ep_j in tqdm(episode_pairs, desc="CrossEpisodeStitch", unit="pair"):
            nodes_i = self.nodes_by_episode.get(ep_i, [])
            nodes_j = self.nodes_by_episode.get(ep_j, [])
            for node_i in nodes_i:
                pose_i = self.node_pose_map[node_i]
                for node_j in nodes_j:
                    if self.graph.has_edge(node_i, node_j):
                        continue
                    pose_j = self.node_pose_map[node_j]
                    dist = pose_xy_distance(pose_i, pose_j)
                    if dist < self.args.stitch_dist_thresh:
                        self.graph.add_edge(
                            node_i,
                            node_j,
                            weight=float(dist),
                            raw_trajectory=[],
                            edge_type="cross_episode",
                        )
                        self.stats["cross_episode_edges"] += 1

    def stitch_within_episodes(self) -> None:
        episodes = sorted(self.nodes_by_episode.keys())
        if not episodes:
            return

        LOGGER.info("开始 Episode 内空间近邻补边: episode 数=%d", len(episodes))
        for ep in tqdm(episodes, desc="IntraEpisodeStitch", unit="ep"):
            nodes = self.nodes_by_episode.get(ep, [])
            if len(nodes) < 2:
                continue
            for i, node_i in enumerate(nodes):
                pose_i = self.node_pose_map[node_i]
                for node_j in nodes[i + 1 :]:
                    if self.graph.has_edge(node_i, node_j):
                        continue
                    pose_j = self.node_pose_map[node_j]
                    dist = pose_xy_distance(pose_i, pose_j)
                    if dist < self.args.intra_stitch_dist_thresh:
                        self.graph.add_edge(
                            node_i,
                            node_j,
                            weight=float(dist),
                            raw_trajectory=[],
                            edge_type="intra_episode_spatial",
                        )
                        self.stats["intra_episode_spatial_edges"] += 1

    def export_json(self) -> Path:
        out_file = self.output_dir / "topological_map.json"
        graph_dict = json_graph.node_link_data(self.graph)
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(graph_dict, f, ensure_ascii=False, indent=2)
        return out_file

    def log_summary(self, output_json: Path) -> None:
        total_frames = self.stats["total_frames"]
        keyframes = self.stats["keyframes"]
        compression = 0.0
        if total_frames > 0:
            compression = (1.0 - (keyframes / float(total_frames))) * 100.0

        LOGGER.info("拓扑构建完成")
        LOGGER.info("输出 JSON: %s", output_json)
        LOGGER.info("节点数: %d", self.graph.number_of_nodes())
        LOGGER.info("边数: %d", self.graph.number_of_edges())
        LOGGER.info("总帧数: %d", total_frames)
        LOGGER.info("关键帧数: %d", keyframes)
        LOGGER.info("压缩率: %.2f%%", compression)
        LOGGER.info("A 过滤命中: %d", self.stats["skipped_a_motion_blur"])
        LOGGER.info("B 过滤命中: %d", self.stats["skipped_b_physical_small_move"])
        LOGGER.info("C 过滤命中: %d", self.stats["skipped_c_semantic"])
        LOGGER.info("缺字段跳过: %d", self.stats["skipped_missing_fields"])
        LOGGER.info("Episode 内边数: %d", self.stats["intra_episode_edges"])
        LOGGER.info("Episode 内空间补边数: %d", self.stats["intra_episode_spatial_edges"])
        LOGGER.info("跨 Episode 边数: %d", self.stats["cross_episode_edges"])


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    LOGGER.info(
        "HF endpoint: %s",
        os.environ.get("HUGGINGFACE_HUB_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://huggingface.co")),
    )
    LOGGER.info("构建参数: %s", args)
    builder = TopologyBuilder(args)
    builder.process_episodes()
    builder.stitch_within_episodes()
    builder.stitch_across_episodes()
    output_json = builder.export_json()
    builder.log_summary(output_json)


if __name__ == "__main__":
    main()
