#!/usr/bin/env python3
"""
Embodied-RAG 第五阶段：全局规划与局部执行（轨迹解压版）

核心能力：
1) 从 topological_map.json 中做拓扑级最短路。
2) 从每条边解压 raw_trajectory 并按方向拼接为高密度航点。
3) 导出 dense_waypoints_output.json，供后续执行器消费。
4) 提供可被部署脚本复用的 TrajectoryExecutor 类接口。
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph


LOGGER = logging.getLogger("rag_executor")


@dataclass
class ExecutorArgs:
    start_pose: str
    target_node: str
    input_json: str
    output_json: str
    log_level: str


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> ExecutorArgs:
    parser = argparse.ArgumentParser(description="Embodied-RAG 第五阶段：全局规划与局部执行")
    parser.add_argument(
        "--start_pose",
        type=str,
        required=True,
        help='当前小车真实坐标，格式 "x,y,yaw"，例如 "1.2,0.5,0.0"',
    )
    parser.add_argument(
        "--target_node",
        type=str,
        required=True,
        help='目标叶子节点 ID，例如 "ep2_frame256"',
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="topology_output/topological_map.json",
        help="物理拓扑图 JSON 路径（node-link 格式）",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="dense_waypoints_output.json",
        help="导出的密集航点文件路径",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="日志级别，如 INFO/DEBUG/WARNING",
    )
    ns = parser.parse_args()
    return ExecutorArgs(
        start_pose=str(ns.start_pose).strip(),
        target_node=str(ns.target_node).strip(),
        input_json=str(ns.input_json).strip(),
        output_json=str(ns.output_json).strip(),
        log_level=str(ns.log_level).strip().upper(),
    )


class TrajectoryExecutor:
    """负责拓扑寻路 + raw_trajectory 解压拼接。"""

    def __init__(self, input_json: str):
        self.input_json = Path(input_json).resolve()
        self.graph = self._load_graph()

    def _load_graph(self) -> nx.Graph:
        if not self.input_json.exists():
            raise FileNotFoundError(f"输入图谱不存在: {self.input_json}")
        try:
            with self.input_json.open("r", encoding="utf-8") as f:
                graph_dict = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 解析失败: {self.input_json} | {exc}") from exc

        graph = json_graph.node_link_graph(graph_dict, edges="links")
        if graph.number_of_nodes() == 0:
            raise ValueError(f"图谱为空，无法规划: {self.input_json}")
        if not isinstance(graph, nx.Graph):
            graph = nx.Graph(graph)
        LOGGER.info(
            "图谱加载完成: path=%s nodes=%d edges=%d",
            self.input_json,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph

    @staticmethod
    def parse_start_pose(start_pose: str) -> np.ndarray:
        parts = [x.strip() for x in str(start_pose).split(",")]
        if len(parts) != 3:
            raise ValueError(f"--start_pose 格式错误，应为 x,y,yaw，当前: {start_pose}")
        try:
            xyz = np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"--start_pose 含非数字字段: {start_pose}") from exc
        return xyz

    def _node_pose_xy(self, node_id: Any) -> np.ndarray:
        node_attr = self.graph.nodes[node_id]
        pose = node_attr.get("pose")
        if not isinstance(pose, (list, tuple)) or len(pose) < 2:
            raise ValueError(f"节点 {node_id} 缺少合法 pose 字段: {pose}")
        return np.asarray([float(pose[0]), float(pose[1])], dtype=np.float64)

    def find_nearest_start_node(self, start_xy: np.ndarray) -> Tuple[Any, float]:
        nearest_node = None
        nearest_dist = float("inf")
        for node_id in self.graph.nodes:
            try:
                node_xy = self._node_pose_xy(node_id)
            except Exception as exc:
                LOGGER.warning("跳过 pose 非法节点 %s: %s", node_id, exc)
                continue
            dist = float(np.linalg.norm(node_xy - start_xy))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_node = node_id

        if nearest_node is None:
            raise RuntimeError("未找到可用起始锚点（所有节点 pose 都不可用）。")
        return nearest_node, nearest_dist

    @staticmethod
    def _edge_cost(_u: Any, _v: Any, edge_data: Dict[str, Any]) -> float:
        """最短路代价：优先走有 raw_trajectory 的边，空轨迹边附加惩罚。"""
        base = float(edge_data.get("weight", 1.0))
        raw = edge_data.get("raw_trajectory", [])
        if isinstance(raw, list) and len(raw) > 0:
            return base
        return base + 2.0

    def shortest_node_path(self, start_node: Any, target_node: str) -> List[Any]:
        if target_node not in self.graph:
            raise KeyError(f"目标节点不存在: {target_node}")
        try:
            path = nx.shortest_path(
                self.graph,
                source=start_node,
                target=target_node,
                weight=self._edge_cost,
            )
        except nx.NetworkXNoPath as exc:
            raise nx.NetworkXNoPath(f"起点 {start_node} 与目标 {target_node} 在图中不连通。") from exc
        return list(path)

    @staticmethod
    def _xy_dist(a: Sequence[float], b: Sequence[float]) -> float:
        aa = np.asarray([float(a[0]), float(a[1])], dtype=np.float64)
        bb = np.asarray([float(b[0]), float(b[1])], dtype=np.float64)
        return float(np.linalg.norm(aa - bb))

    @staticmethod
    def _normalize_traj_point(pt: Sequence[float]) -> List[float]:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            raise ValueError(f"raw_trajectory 点格式非法: {pt}")
        yaw = float(pt[2]) if len(pt) >= 3 else 0.0
        return [float(pt[0]), float(pt[1]), yaw]

    def _orient_segment_u_to_v(self, u: Any, v: Any, raw_trajectory: Sequence[Sequence[float]]) -> List[List[float]]:
        if not raw_trajectory:
            raise ValueError(f"边 ({u}, {v}) 的 raw_trajectory 为空，无法安全解压。")
        segment = [self._normalize_traj_point(p) for p in raw_trajectory]

        u_pose = self.graph.nodes[u].get("pose")
        v_pose = self.graph.nodes[v].get("pose")
        if not isinstance(u_pose, (list, tuple)) or len(u_pose) < 2:
            raise ValueError(f"节点 {u} pose 非法: {u_pose}")
        if not isinstance(v_pose, (list, tuple)) or len(v_pose) < 2:
            raise ValueError(f"节点 {v} pose 非法: {v_pose}")

        first_pt = segment[0]
        last_pt = segment[-1]
        first_to_u = self._xy_dist(first_pt, u_pose)
        first_to_v = self._xy_dist(first_pt, v_pose)
        last_to_u = self._xy_dist(last_pt, u_pose)
        last_to_v = self._xy_dist(last_pt, v_pose)

        should_reverse = (first_to_u > first_to_v) or (last_to_u < last_to_v)
        if should_reverse:
            segment = segment[::-1]
        return segment

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi

    def _synthesize_segment_u_to_v(self, u: Any, v: Any, max_step_xy: float = 0.05) -> List[List[float]]:
        """当边缺少 raw_trajectory 时，用两端节点 pose 线性插值生成稠密连接段。"""
        u_pose = self.graph.nodes[u].get("pose")
        v_pose = self.graph.nodes[v].get("pose")
        if not isinstance(u_pose, (list, tuple)) or len(u_pose) < 2:
            raise ValueError(f"节点 {u} pose 非法: {u_pose}")
        if not isinstance(v_pose, (list, tuple)) or len(v_pose) < 2:
            raise ValueError(f"节点 {v} pose 非法: {v_pose}")

        x0, y0 = float(u_pose[0]), float(u_pose[1])
        x1, y1 = float(v_pose[0]), float(v_pose[1])
        dxy = float(np.linalg.norm(np.array([x1 - x0, y1 - y0], dtype=np.float64)))
        steps = max(2, int(np.ceil(dxy / max_step_xy)) + 1)

        # yaw 优先使用节点 yaw，若缺失则使用连线方向
        if len(u_pose) >= 3 and len(v_pose) >= 3:
            yaw0 = float(u_pose[2])
            yaw1 = float(v_pose[2])
            delta = self._wrap_to_pi(yaw1 - yaw0)
            yaws = [self._wrap_to_pi(yaw0 + delta * (i / (steps - 1))) for i in range(steps)]
        else:
            yaw_line = float(np.arctan2(y1 - y0, x1 - x0))
            yaws = [yaw_line] * steps

        xs = np.linspace(x0, x1, num=steps)
        ys = np.linspace(y0, y1, num=steps)
        return [[float(xs[i]), float(ys[i]), float(yaws[i])] for i in range(steps)]

    @staticmethod
    def _append_segment_wo_duplicate(dst: List[List[float]], seg: List[List[float]], eps: float = 1e-6) -> None:
        if not seg:
            return
        if not dst:
            dst.extend(seg)
            return
        tail = dst[-1]
        head = seg[0]
        if abs(float(tail[0]) - float(head[0])) <= eps and abs(float(tail[1]) - float(head[1])) <= eps:
            dst.extend(seg[1:])
        else:
            dst.extend(seg)

    def unroll_dense_waypoints(self, node_path: Sequence[Any]) -> List[List[float]]:
        if len(node_path) < 2:
            raise ValueError(f"节点路径长度不足: {len(node_path)}，无法展开边轨迹。")

        dense_waypoints: List[List[float]] = []
        self.last_sparse_connector_count = 0
        for idx in range(len(node_path) - 1):
            u = node_path[idx]
            v = node_path[idx + 1]
            if not self.graph.has_edge(u, v):
                raise KeyError(f"图中不存在边: ({u}, {v})")
            edge_data = self.graph.get_edge_data(u, v) or {}
            raw_trajectory = edge_data.get("raw_trajectory", [])
            if isinstance(raw_trajectory, list) and len(raw_trajectory) > 0:
                segment = self._orient_segment_u_to_v(u, v, raw_trajectory)
            else:
                self.last_sparse_connector_count += 1
                segment = self._synthesize_segment_u_to_v(u, v)
            self._append_segment_wo_duplicate(dense_waypoints, segment)
        if not dense_waypoints:
            raise RuntimeError("轨迹解压后为空，请检查图谱边数据。")
        return dense_waypoints

    def plan_dense_waypoints(
        self,
        start_pose: Sequence[float],
        target_node: str,
    ) -> Dict[str, Any]:
        start_pose_arr = np.asarray(start_pose, dtype=np.float64).reshape(-1)
        if start_pose_arr.shape[0] < 2:
            raise ValueError(f"start_pose 至少应包含 x,y，当前: {start_pose}")
        start_xy = start_pose_arr[:2]

        nearest_node, nearest_dist = self.find_nearest_start_node(start_xy)
        node_path = self.shortest_node_path(nearest_node, target_node)
        dense_waypoints = self.unroll_dense_waypoints(node_path)

        result = {
            "start_pose": [float(x) for x in start_pose_arr.tolist()],
            "matched_start_node": str(nearest_node),
            "matched_start_node_distance": float(nearest_dist),
            "target_node": str(target_node),
            "path_nodes": [str(n) for n in node_path],
            "dense_waypoints_count": int(len(dense_waypoints)),
            "sparse_connector_edges_used": int(getattr(self, "last_sparse_connector_count", 0)),
            "dense_waypoints": dense_waypoints,
        }
        return result

    @staticmethod
    def save_result(result: Dict[str, Any], output_json: str) -> Path:
        output_path = Path(output_json).resolve()
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return output_path


def execute_trajectory(waypoints: Sequence[Sequence[float]]) -> None:
    """
    Mock 执行接口（供接入 MuJoCo `step_base_auto` 逻辑时参考）。

    你可以按以下方式把 waypoints 融入底盘控制循环：
    1) 在每个控制周期（20Hz）读取当前底盘位姿 (x, y, yaw)。
    2) 在 waypoints 中基于当前索引选一个 Lookahead Point（前视点），
       常见做法是：在当前点之后找第一个与当前机器人距离 >= lookahead_dist 的点。
    3) 计算机器人朝向与目标方向的误差 heading_error：
       desired_yaw = atan2(y_lh - y, x_lh - x)，
       heading_error = wrap_to_pi(desired_yaw - yaw)。
    4) 将误差转换为左右轮速度：
       - v_forward 随与目标点距离变化（远时快，近时慢）
       - v_turn 由 heading_error 的比例控制得到
       - wheel_left = v_forward - v_turn
       - wheel_right = v_forward + v_turn
       最后对 wheel_left/right 做限幅，避免过大指令。
    5) 当机器人接近当前 waypoint（距离 < arrive_threshold）时推进索引；
       当索引到达末尾时，输出 [0, 0] 停车并结束导航。
    6) 若你同时启用了视觉策略（pi0），建议在“导航模式”下暂时关闭 base 自动推理，
       防止模型输出与导航控制器同时写轮速导致互相干扰。
    """
    if not waypoints:
        LOGGER.warning("execute_trajectory 收到空 waypoints，跳过执行。")
        return
    LOGGER.info("Mock execute_trajectory called, waypoints=%d", len(waypoints))


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    try:
        executor = TrajectoryExecutor(args.input_json)
        start_pose = executor.parse_start_pose(args.start_pose)
        LOGGER.info("起点坐标: x=%.6f y=%.6f yaw=%.6f", start_pose[0], start_pose[1], start_pose[2])
        result = executor.plan_dense_waypoints(start_pose=start_pose, target_node=args.target_node)
        LOGGER.info(
            "匹配起点锚点: %s (dist=%.4f m) | 途经节点数: %d | 密集航点数: %d",
            result["matched_start_node"],
            result["matched_start_node_distance"],
            len(result["path_nodes"]),
            result["dense_waypoints_count"],
        )
        output_path = executor.save_result(result, args.output_json)
        LOGGER.info("已导出密集航点: %s", output_path)
        execute_trajectory(result["dense_waypoints"])
    except Exception as exc:
        LOGGER.error("轨迹规划失败: %s", exc)
        raise


if __name__ == "__main__":
    main()
