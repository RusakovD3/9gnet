"""Graph metrics used by the current GNet9 topology."""

from __future__ import annotations

from typing import Iterable

import networkx as nx
def vertex_proximity_index(graph: nx.Graph, nodes: Iterable[str]) -> dict[str, float]:
    """Return closeness centrality for selected nodes.

    In plain language: the closer a node is to all other selected nodes, the
    higher its value. In the project this is used as a readable centrality metric
    for core routers.
    """
    selected_nodes = list(nodes)
    subgraph = graph.subgraph(selected_nodes)
    return {node: float(value) for node, value in nx.closeness_centrality(subgraph).items()}
