#!/usr/bin/env python3
"""
RAG pipeline unified entry.

Provides three subcommands:
1) build   : Stage1/2/3 offline build (+ optional plotting)
2) query   : Stage4 retrieval + Stage5 dense waypoint planning
3) run_all : build then query in one command
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


# Keep behavior consistent with existing scripts that rely on HF mirrors.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", "https://hf-mirror.com")


LOGGER = logging.getLogger("rag_main")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def require_dashscope_api_key() -> None:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise EnvironmentError("缺少环境变量 DASHSCOPE_API_KEY，请先 export DASHSCOPE_API_KEY=...")


def parse_start_pose_text(start_pose_text: str) -> Sequence[float]:
    parts = [x.strip() for x in str(start_pose_text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"--start_pose 格式错误，应为 x,y,yaw，当前: {start_pose_text}")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def print_banner() -> None:
    print("\n" + "=" * 72)
    print("  Embodied-RAG Unified Entry")
    print("  build | query | run_all")
    print("=" * 72 + "\n")


def _build_stage1(args: argparse.Namespace) -> Path:
    try:
        from .build_topology import TopologyArgs, TopologyBuilder
    except ImportError:
        from rag_pipeline.build_topology import TopologyArgs, TopologyBuilder

    LOGGER.info("[Stage1] Build topology: start")
    topology_args = TopologyArgs(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        repo_name=args.repo_name,
        enable_clip=bool(args.enable_clip),
        clip_model=args.clip_model,
        clip_device=args.clip_device,
        clip_similarity_thresh=float(args.clip_similarity_thresh),
        turn_diff_thresh=float(args.turn_diff_thresh),
        min_distance=float(args.min_distance),
        min_angle_deg=float(args.min_angle_deg),
        intra_stitch_dist_thresh=float(args.intra_stitch_dist_thresh),
        stitch_dist_thresh=float(args.stitch_dist_thresh),
        max_episodes=args.max_episodes,
        log_level=args.log_level.upper(),
    )
    builder = TopologyBuilder(topology_args)
    builder.process_episodes()
    builder.stitch_within_episodes()
    builder.stitch_across_episodes()
    topological_json = builder.export_json()
    builder.log_summary(topological_json)
    LOGGER.info("[Stage1] Build topology: done -> %s", topological_json)
    return topological_json


def _build_stage2_text_only(args: argparse.Namespace, input_json: Path) -> Path:
    try:
        from .build_semantics_text_only import SemanticArgs, SemanticAnnotator
    except ImportError:
        from rag_pipeline.build_semantics_text_only import SemanticArgs, SemanticAnnotator

    require_dashscope_api_key()
    LOGGER.info("[Stage2] Build semantics (text-only): start")
    semantics_json = Path(args.semantic_output_json).resolve()
    captions_json = Path(args.semantic_caption_output_json).resolve()
    semantic_args = SemanticArgs(
        input_json=str(input_json),
        output_json=str(semantics_json),
        caption_output_json=str(captions_json),
        dashscope_model=args.semantic_model,
        save_every=int(args.semantic_save_every),
        log_level=args.log_level.upper(),
    )
    annotator = SemanticAnnotator(semantic_args)
    annotator.run()
    LOGGER.info("[Stage2] Build semantics (text-only): done -> %s", semantics_json)
    return semantics_json


def _build_stage3_forest(args: argparse.Namespace, input_json: Path) -> Path:
    try:
        from .build_forest import ForestArgs, ForestBuilder
    except ImportError:
        from rag_pipeline.build_forest import ForestArgs, ForestBuilder

    require_dashscope_api_key()
    LOGGER.info("[Stage3] Build forest: start")
    forest_json = Path(args.forest_output_json).resolve()
    forest_args = ForestArgs(
        input_json=str(input_json),
        output_json=str(forest_json),
        alpha=float(args.forest_alpha),
        theta=float(args.forest_theta),
        cluster_threshold=float(args.forest_cluster_threshold),
        summary_model=args.forest_summary_model,
        max_retry=int(args.forest_max_retry),
        log_level=args.log_level.upper(),
    )
    builder = ForestBuilder(forest_args)
    builder.run()
    LOGGER.info("[Stage3] Build forest: done -> %s", forest_json)
    return forest_json


def _plot_topology(json_path: Path, output_png: Path, dpi: int) -> None:
    try:
        from .plot_topology_map import extract_positions_and_episodes, load_graph
    except ImportError:
        from rag_pipeline.plot_topology_map import extract_positions_and_episodes, load_graph
    import matplotlib.pyplot as plt
    import networkx as nx

    graph = load_graph(json_path)
    pos, episode_map = extract_positions_and_episodes(graph)
    nodes_to_draw = [n for n in graph.nodes if n in pos]

    intra_edges = [
        (u, v)
        for u, v, d in graph.edges(data=True)
        if u in pos and v in pos and d.get("edge_type") in {"intra_episode", "intra_episode_spatial"}
    ]
    cross_edges = [
        (u, v)
        for u, v, d in graph.edges(data=True)
        if u in pos and v in pos and d.get("edge_type") == "cross_episode"
    ]

    episodes = sorted({episode_map[n] for n in nodes_to_draw})
    cmap = plt.get_cmap("tab20", max(len(episodes), 1))
    color_by_episode = {ep: cmap(i) for i, ep in enumerate(episodes)}
    node_colors = [color_by_episode[episode_map[n]] for n in nodes_to_draw]

    plt.figure(figsize=(12.0, 10.0))
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes_to_draw, node_size=20.0, node_color=node_colors, alpha=0.85)
    nx.draw_networkx_edges(graph, pos, edgelist=intra_edges, edge_color="black", width=1.2, alpha=0.45)
    nx.draw_networkx_edges(graph, pos, edgelist=cross_edges, edge_color="red", width=1.2, alpha=0.75, style="dashed")
    plt.axis("equal")
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=dpi)
    plt.close()


def _plot_forest(json_path: Path, output_png: Path, dpi: int) -> None:
    try:
        from .plot_forest_topology_map import extract_positions, is_cluster_parent, load_graph
    except ImportError:
        from rag_pipeline.plot_forest_topology_map import extract_positions, is_cluster_parent, load_graph
    import matplotlib.pyplot as plt
    import networkx as nx

    graph = load_graph(json_path)
    pos = extract_positions(graph)

    leaf_nodes = []
    cluster_nodes = []
    episode_map: Dict[Any, int] = {}
    for node_id in graph.nodes:
        if node_id not in pos:
            continue
        attr = graph.nodes[node_id]
        if is_cluster_parent(node_id, attr):
            cluster_nodes.append(node_id)
        else:
            leaf_nodes.append(node_id)
            episode_map[node_id] = int(attr.get("episode_id", -1))

    hierarchy_edges = [
        (u, v)
        for u, v, d in graph.edges(data=True)
        if u in pos and v in pos and d.get("edge_type") == "is_child_of"
    ]
    topology_edges = [
        (u, v)
        for u, v, d in graph.edges(data=True)
        if u in pos and v in pos and d.get("edge_type") != "is_child_of"
    ]

    episodes = sorted({episode_map[n] for n in leaf_nodes}) if leaf_nodes else []
    cmap = plt.get_cmap("tab20", max(len(episodes), 1))
    color_by_episode = {ep: cmap(i) for i, ep in enumerate(episodes)}
    leaf_colors = [color_by_episode[episode_map[n]] for n in leaf_nodes] if leaf_nodes else []

    plt.figure(figsize=(13.0, 10.0))
    nx.draw_networkx_edges(graph, pos, edgelist=topology_edges, edge_color="gray", width=1.0, alpha=0.35)
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=hierarchy_edges,
        edge_color="darkorange",
        width=1.4,
        alpha=0.7,
        style="dashed",
    )
    if leaf_nodes:
        nx.draw_networkx_nodes(graph, pos, nodelist=leaf_nodes, node_size=18.0, node_color=leaf_colors, alpha=0.85)
    if cluster_nodes:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=cluster_nodes,
            node_size=120.0,
            node_shape="s",
            node_color="gold",
            edgecolors="black",
            linewidths=0.8,
            alpha=0.95,
        )
    plt.axis("equal")
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=dpi)
    plt.close()


def run_build(args: argparse.Namespace) -> Dict[str, str]:
    topological_json = _build_stage1(args)
    semantics_json = _build_stage2_text_only(args, topological_json)
    forest_json = _build_stage3_forest(args, semantics_json)

    topology_png = Path(args.topology_png).resolve()
    forest_png = Path(args.forest_png).resolve()
    if not args.skip_plot:
        LOGGER.info("[Plot] Generating topology plot -> %s", topology_png)
        _plot_topology(topological_json, topology_png, dpi=int(args.plot_dpi))
        LOGGER.info("[Plot] Generating forest plot -> %s", forest_png)
        _plot_forest(forest_json, forest_png, dpi=int(args.plot_dpi))

    return {
        "topological_json": str(topological_json),
        "semantic_json": str(semantics_json),
        "forest_json": str(forest_json),
        "topology_png": str(topology_png),
        "forest_png": str(forest_png),
    }


def _retrieve_target_node(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from .rag_navigator import NavigatorArgs, RAGNavigator
    except ImportError:
        from rag_pipeline.rag_navigator import NavigatorArgs, RAGNavigator

    require_dashscope_api_key()
    navigator_args = NavigatorArgs(
        query=args.query,
        input_json=str(Path(args.forest_json).resolve()),
        macro_model=args.macro_model,
        embedding_model=args.embedding_model,
        max_retry=int(args.max_retry),
        retry_wait=float(args.retry_wait),
        log_level=args.log_level.upper(),
    )
    navigator = RAGNavigator(navigator_args)
    result = navigator.retrieve_top_leaf(args.query)
    LOGGER.info(
        "[Stage4] target node resolved: %s (cluster=%s score=%.6f)",
        result["target_node"],
        result["cluster_id"],
        result["score"],
    )
    return result


def _plan_dense_waypoints(
    args: argparse.Namespace,
    target_node: str,
    start_pose: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    try:
        from .rag_executor import TrajectoryExecutor
    except ImportError:
        from rag_pipeline.rag_executor import TrajectoryExecutor

    executor = TrajectoryExecutor(args.topology_json)
    if start_pose is None:
        start_pose = parse_start_pose_text(args.start_pose)
    result = executor.plan_dense_waypoints(start_pose=start_pose, target_node=target_node)
    output_path = executor.save_result(result, args.dense_output_json)
    LOGGER.info(
        "[Stage5] dense waypoints exported: %s (count=%d)",
        output_path,
        result["dense_waypoints_count"],
    )
    result["dense_output_json"] = str(output_path)
    return result


def run_query(args: argparse.Namespace) -> Dict[str, Any]:
    target_node = args.target_node
    retrieval_result: Optional[Dict[str, Any]] = None
    if not target_node:
        if not args.query:
            raise ValueError("query 模式下，若未提供 --target_node，则必须提供 --query。")
        retrieval_result = _retrieve_target_node(args)
        target_node = retrieval_result["target_node"]

    planning_result = _plan_dense_waypoints(args=args, target_node=str(target_node))
    return {
        "retrieval_result": retrieval_result,
        "planning_result": planning_result,
    }


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    build_outputs = run_build(args)
    if not args.query:
        raise ValueError("run_all 模式必须提供 --query。")
    args.forest_json = build_outputs["forest_json"]
    args.topology_json = build_outputs["topological_json"]
    query_outputs = run_query(args)
    return {"build": build_outputs, "query": query_outputs}


def _add_build_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_dir", type=str, required=True, help="LeRobot 数据集根目录")
    parser.add_argument("--output_dir", type=str, default="topology_output", help="Stage1 输出目录")
    parser.add_argument("--repo_name", type=str, default=None, help="可选 LeRobotDataset repo_name")
    parser.add_argument("--enable_clip", action="store_true", help="Stage1 启用 CLIP 过滤")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip_device", type=str, default="cuda")
    parser.add_argument("--clip_similarity_thresh", type=float, default=0.95)
    parser.add_argument("--turn_diff_thresh", type=float, default=4.0)
    parser.add_argument("--min_distance", type=float, default=0.3)
    parser.add_argument("--min_angle_deg", type=float, default=30.0)
    parser.add_argument("--intra_stitch_dist_thresh", type=float, default=0.1)
    parser.add_argument("--stitch_dist_thresh", type=float, default=0.1)
    parser.add_argument("--max_episodes", type=int, default=None)

    parser.add_argument(
        "--semantic_output_json",
        type=str,
        default="topology_output/semantic_topological_map_text_only.json",
        help="Stage2 输出 JSON",
    )
    parser.add_argument(
        "--semantic_caption_output_json",
        type=str,
        default="topology_output/semantic_topological_map_text_only_captions_only.json",
        help="Stage2 captions 输出 JSON",
    )
    parser.add_argument("--semantic_model", type=str, default="qwen3.5-plus-2026-02-15")
    parser.add_argument("--semantic_save_every", type=int, default=10)

    parser.add_argument(
        "--forest_output_json",
        type=str,
        default="topology_output/forest_topological_map.json",
        help="Stage3 输出 JSON",
    )
    parser.add_argument("--forest_alpha", type=float, default=0.5)
    parser.add_argument("--forest_theta", type=float, default=1.0)
    parser.add_argument("--forest_cluster_threshold", type=float, default=0.6)
    parser.add_argument("--forest_summary_model", type=str, default="qwen3-max-2026-01-23")
    parser.add_argument("--forest_max_retry", type=int, default=3)

    parser.add_argument("--skip_plot", action="store_true", help="跳过拓扑图与森林图绘制")
    parser.add_argument("--plot_dpi", type=int, default=200)
    parser.add_argument("--topology_png", type=str, default="topology_output/topological_map.png")
    parser.add_argument("--forest_png", type=str, default="topology_output/forest_topological_map.png")


def _add_query_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--forest_json",
        type=str,
        default="topology_output/forest_topological_map.json",
        help="Stage4 输入森林图",
    )
    parser.add_argument("--query", type=str, default="", help="自然语言检索指令")
    parser.add_argument("--target_node", type=str, default="", help="直接指定目标节点（可跳过 Stage4）")
    parser.add_argument("--macro_model", type=str, default="qwen-max")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-v4")
    parser.add_argument("--max_retry", type=int, default=3)
    parser.add_argument("--retry_wait", type=float, default=1.5)

    parser.add_argument(
        "--topology_json",
        type=str,
        default="topology_output/topological_map.json",
        help="Stage5 输入拓扑图",
    )
    parser.add_argument("--start_pose", type=str, required=True, help='当前位姿 "x,y,yaw"')
    parser.add_argument("--dense_output_json", type=str, default="dense_waypoints_output.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embodied-RAG unified entry")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别，如 INFO/DEBUG")
    sub = parser.add_subparsers(dest="command", required=True)

    parser_build = sub.add_parser("build", help="离线建图链路 (Stage1/2/3 + plot)")
    _add_build_flags(parser_build)

    parser_query = sub.add_parser("query", help="在线检索规划链路 (Stage4/5)")
    _add_query_flags(parser_query)

    parser_run_all = sub.add_parser("run_all", help="端到端一键执行 (build + query)")
    _add_build_flags(parser_run_all)
    _add_query_flags(parser_run_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    print_banner()
    LOGGER.info("运行命令: %s", args.command)
    LOGGER.info("HF endpoint: %s", os.environ.get("HUGGINGFACE_HUB_ENDPOINT"))

    if args.command == "build":
        result = run_build(args)
    elif args.command == "query":
        result = run_query(args)
    elif args.command == "run_all":
        result = run_all(args)
    else:
        raise ValueError(f"未知 command: {args.command}")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
