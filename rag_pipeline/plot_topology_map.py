#!/usr/bin/env python3
"""
plot_topology_map.py
--------------------
可视化 build_topology.py 导出的 topological_map.json：
1) 按物理坐标 (x, y) 绘制节点
2) Episode 内边(intra_episode)与跨 Episode 边(cross_episode)分开绘制
3) 节点按 episode_id 着色
4) 保存 PNG，并可选显示窗口
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from networkx.readwrite import json_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize topological_map.json")
    parser.add_argument(
        "--json_path",
        type=str,
        default="topology_output/topological_map.json",
        help="Path to topological_map.json",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="topology_output/topological_map.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        default=(12.0, 10.0),
        help="Figure size in inches, e.g. --figsize 12 10",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="DPI for saved figure",
    )
    parser.add_argument(
        "--node_size",
        type=float,
        default=20.0,
        help="Node marker size",
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


def extract_positions_and_episodes(G: nx.Graph) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, int]]:
    pos: Dict[str, Tuple[float, float]] = {}
    episode_map: Dict[str, int] = {}

    for node, attr in G.nodes(data=True):
        pose = attr.get("pose")
        if not isinstance(pose, (list, tuple)) or len(pose) < 2:
            continue
        x = float(pose[0])
        y = float(pose[1])
        pos[node] = (x, y)
        episode_map[node] = int(attr.get("episode_id", -1))

    if not pos:
        raise ValueError("No valid node positions found. Expected node attr 'pose' like [x, y, theta].")

    return pos, episode_map


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_path).resolve()
    output_png = Path(args.output_png).resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    G = load_graph(json_path)
    pos, episode_map = extract_positions_and_episodes(G)

    # 仅绘制有坐标的节点，避免 draw 函数因缺失 pos 报错。
    nodes_to_draw = [n for n in G.nodes if n in pos]

    intra_edges = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if u in pos
        and v in pos
        and d.get("edge_type") in {"intra_episode", "intra_episode_spatial"}
    ]
    cross_edges = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if u in pos and v in pos and d.get("edge_type") == "cross_episode"
    ]

    episodes = sorted({episode_map[n] for n in nodes_to_draw})
    cmap = plt.get_cmap("tab20", max(len(episodes), 1))
    color_by_episode = {ep: cmap(i) for i, ep in enumerate(episodes)}
    node_colors = [color_by_episode[episode_map[n]] for n in nodes_to_draw]

    plt.figure(figsize=tuple(args.figsize))

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes_to_draw,
        node_size=args.node_size,
        node_color=node_colors,
        alpha=0.85,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=intra_edges,
        edge_color="black",
        width=1.2,
        alpha=0.45,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=cross_edges,
        edge_color="darkgreen",
        width=1.2,
        alpha=0.45,
    )

    legend_items = []
    for ep in episodes:
        legend_items.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                label=f"Episode {ep}",
                markerfacecolor=color_by_episode[ep],
                markersize=7,
            )
        )
    if legend_items:
        plt.legend(handles=legend_items, loc="best", fontsize=8, frameon=True)

    plt.title("Embodied-RAG: Semantic Topological Map", fontsize=15)
    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(output_png, dpi=args.dpi)
    print(f"[OK] Saved topology figure to: {output_png}")
    print(f"[INFO] Nodes drawn: {len(nodes_to_draw)}")
    print(f"[INFO] Intra edges: {len(intra_edges)} | Cross edges: {len(cross_edges)}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
