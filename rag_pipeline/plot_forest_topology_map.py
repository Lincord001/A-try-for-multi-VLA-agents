#!/usr/bin/env python3
"""
plot_forest_topology_map.py
---------------------------
可视化 build_forest.py 导出的 forest_topological_map.json：
1) 按物理坐标 (x, y) 绘制节点
2) 原始拓扑边与聚类层级边(is_child_of)分开绘制
3) 叶子节点按 episode_id 着色，簇父节点单独样式显示
4) 可选给簇父节点添加摘要标签
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from networkx.readwrite import json_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize forest_topological_map.json")
    parser.add_argument(
        "--json_path",
        type=str,
        default="topology_output/forest_topological_map.json",
        help="Path to forest_topological_map.json",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="topology_output/forest_topological_map.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        default=(13.0, 10.0),
        help="Figure size in inches, e.g. --figsize 13 10",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for saved figure",
    )
    parser.add_argument(
        "--leaf_node_size",
        type=float,
        default=18.0,
        help="Leaf node marker size",
    )
    parser.add_argument(
        "--cluster_node_size",
        type=float,
        default=120.0,
        help="Cluster parent marker size",
    )
    parser.add_argument(
        "--annotate_clusters",
        action="store_true",
        help="Annotate cluster parent nodes with semantic_caption",
    )
    parser.add_argument(
        "--cluster_only",
        action="store_true",
        help="Only draw cluster hierarchy skeleton (hide non-hierarchy topology edges)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive window in addition to saving PNG",
    )
    return parser.parse_args()


def load_graph(json_path: Path) -> nx.Graph:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return json_graph.node_link_graph(data)


def extract_positions(G: nx.Graph) -> Dict[Any, Tuple[float, float]]:
    pos: Dict[Any, Tuple[float, float]] = {}
    for node_id, attr in G.nodes(data=True):
        pose = attr.get("pose")
        if not isinstance(pose, (list, tuple)) or len(pose) < 2:
            continue
        pos[node_id] = (float(pose[0]), float(pose[1]))
    if not pos:
        raise ValueError("No valid node positions found. Expected node attr 'pose' like [x, y, theta].")
    return pos


def is_cluster_parent(node_id: Any, attr: Dict[str, Any]) -> bool:
    if attr.get("node_type") in {"cluster_parent", "cluster"}:
        return True
    return isinstance(node_id, str) and node_id.startswith("Cluster_")


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_path).resolve()
    output_png = Path(args.output_png).resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    G = load_graph(json_path)
    pos = extract_positions(G)

    nodes_with_pos = [n for n in G.nodes if n in pos]
    leaf_nodes = []
    cluster_nodes = []
    episode_map: Dict[Any, int] = {}

    for node_id in nodes_with_pos:
        attr = G.nodes[node_id]
        if is_cluster_parent(node_id, attr):
            cluster_nodes.append(node_id)
        else:
            leaf_nodes.append(node_id)
            episode_map[node_id] = int(attr.get("episode_id", -1))

    hierarchy_edges = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if u in pos and v in pos and d.get("edge_type") == "is_child_of"
    ]
    topology_edges = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if u in pos and v in pos and d.get("edge_type") != "is_child_of"
    ]

    episodes = sorted({episode_map[n] for n in leaf_nodes})
    cmap = plt.get_cmap("tab20", max(len(episodes), 1))
    color_by_episode = {ep: cmap(i) for i, ep in enumerate(episodes)}
    leaf_colors = [color_by_episode[episode_map[n]] for n in leaf_nodes]

    plt.figure(figsize=tuple(args.figsize))

    if not args.cluster_only:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=topology_edges,
            edge_color="gray",
            width=1.0,
            alpha=0.35,
        )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=hierarchy_edges,
        edge_color="darkorange",
        width=1.4,
        alpha=0.7,
        style="dashed",
    )

    if leaf_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=leaf_nodes,
            node_size=args.leaf_node_size,
            node_color=leaf_colors,
            alpha=0.9,
            linewidths=0.0,
        )

    if cluster_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=cluster_nodes,
            node_size=args.cluster_node_size,
            node_color="crimson",
            alpha=0.95,
            node_shape="^",
            edgecolors="white",
            linewidths=0.8,
        )

    if args.annotate_clusters:
        for node_id in cluster_nodes:
            text = G.nodes[node_id].get("semantic_caption")
            if isinstance(text, str) and text.strip():
                x, y = pos[node_id]
                plt.text(x, y, text.strip(), fontsize=8, color="crimson", ha="left", va="bottom")

    legend_items = []
    if not args.cluster_only:
        legend_items.append(Line2D([0], [0], color="gray", lw=1.2, alpha=0.5, label="Topology edges"))
    legend_items.extend(
        [
            Line2D([0], [0], color="darkorange", lw=1.4, ls="--", alpha=0.8, label="Cluster hierarchy edges"),
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor="crimson",
                markeredgecolor="white",
                markeredgewidth=0.8,
                markersize=9,
                label="Cluster parent",
            ),
        ]
    )
    for ep in episodes:
        legend_items.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                label=f"Episode {ep}",
                markerfacecolor=color_by_episode[ep],
                markersize=6,
            )
        )

    plt.legend(handles=legend_items, loc="best", fontsize=8, frameon=True)
    plt.title("Embodied-RAG: Forest Topological Map", fontsize=15)
    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(output_png, dpi=args.dpi)
    print(f"[OK] Saved forest topology figure to: {output_png}")
    print(f"[INFO] Leaf nodes: {len(leaf_nodes)} | Cluster parents: {len(cluster_nodes)}")
    if args.cluster_only:
        print(f"[INFO] Cluster-only mode: ON | Hierarchy edges: {len(hierarchy_edges)}")
    else:
        print(f"[INFO] Topology edges: {len(topology_edges)} | Hierarchy edges: {len(hierarchy_edges)}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
