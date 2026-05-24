"""Small graph metrics used by the current and future G-Net layers."""

from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np


def graph_hausdorff_distance(graph: nx.Graph, node_set_a: Iterable[str], node_set_b: Iterable[str]) -> float:
    """Return Hausdorff distance between two node sets using stored 2D positions.

    This function is prepared for the future L7 arbitrator. For example, it can
    compare the current topology with an ideal or repaired topology.
    """
    points_a = _node_positions(graph, node_set_a)
    points_b = _node_positions(graph, node_set_b)

    if len(points_a) == 0 or len(points_b) == 0:
        return 0.0

    return float(max(_directed_hausdorff(points_a, points_b), _directed_hausdorff(points_b, points_a)))


def vertex_proximity_index(graph: nx.Graph, nodes: Iterable[str]) -> dict[str, float]:
    """Return closeness centrality for selected nodes.

    In plain language: the closer a node is to all other selected nodes, the
    higher its value. In the project this is used as a readable centrality metric
    for core routers.
    """
    selected_nodes = list(nodes)
    subgraph = graph.subgraph(selected_nodes)
    return {node: float(value) for node, value in nx.closeness_centrality(subgraph).items()}


def _node_positions(graph: nx.Graph, nodes: Iterable[str]) -> np.ndarray:
    return np.array([graph.nodes[node]["pos"] for node in nodes], dtype=float)


def _directed_hausdorff(points_from: np.ndarray, points_to: np.ndarray) -> float:
    """Distance from one point set to another in the Hausdorff sense."""
    pairwise_distances = np.linalg.norm(points_from[:, None, :] - points_to[None, :, :], axis=2)
    return float(np.max(np.min(pairwise_distances, axis=1)))
