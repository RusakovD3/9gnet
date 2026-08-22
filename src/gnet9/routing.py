"""Единые правила поиска маршрутов в плоскости передачи данных GNet9.

Топология хранится как неориентированный граф и содержит не только сетевое
оборудование, но и двухподключённые серверы, многоподключённых Gold-абонентов, логические сервисы
и служебные уровни. Обычный ``networkx.shortest_path`` мог поэтому ошибочно
использовать конечное устройство как транзитный мост. Этот модуль оставляет
промежуточными вершинами только сетевое оборудование: маршрутизаторы ядра,
агрегирующие коммутаторы и явные узлы доступа RAN/OLT.
"""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from .models import NetworkModel


DATA_PLANE_TRANSIT_ROLES = frozenset(
    {
        "core-router",
        "aggregation-switch",
        "radio-access-node",
        "optical-line-terminal",
    }
)
LOGICAL_SERVICE_MEDIUM = "logical-service-binding"


def data_plane_routing_view(
    model: NetworkModel,
    *,
    endpoints: Iterable[str] = (),
    excluded_nodes: Iterable[str] = (),
) -> nx.Graph:
    """Вернуть read-only view, где конечные устройства не бывают транзитом.

    ``endpoints`` добавляются к разрешённым сетевым устройствам только для
    текущего поиска. Если конечная точка одновременно указана в
    ``excluded_nodes``, статус endpoint имеет приоритет: это сохраняет
    возможность найти путь *до* атакованной цели или защищаемого клиента.
    Логическая связь ``SVC -> SRV`` не является физическим каналом и никогда не
    включается в плоскость передачи данных.
    """
    endpoint_set = {str(node) for node in endpoints}
    excluded_set = {str(node) for node in excluded_nodes} - endpoint_set

    def node_allowed(node: str) -> bool:
        if node in excluded_set:
            return False
        role = str(model.graph.nodes[node].get("role", ""))
        return node in endpoint_set or role in DATA_PLANE_TRANSIT_ROLES

    def edge_allowed(source: str, target: str) -> bool:
        return model.graph.edges[source, target].get("medium") != LOGICAL_SERVICE_MEDIUM

    return nx.subgraph_view(
        model.graph,
        filter_node=node_allowed,
        filter_edge=edge_allowed,
    )


def shortest_data_path(
    model: NetworkModel,
    source: str,
    target: str,
    *,
    excluded_nodes: Iterable[str] = (),
) -> list[str]:
    """Найти минимальный по задержке путь без endpoint-транзита."""
    view = data_plane_routing_view(
        model,
        endpoints=(source, target),
        excluded_nodes=excluded_nodes,
    )
    return list(nx.shortest_path(view, source, target, weight="latency_ms"))


def shortest_data_path_length(
    model: NetworkModel,
    source: str,
    target: str,
    *,
    excluded_nodes: Iterable[str] = (),
) -> float:
    """Вернуть задержку кратчайшего допустимого data-plane-пути."""
    view = data_plane_routing_view(
        model,
        endpoints=(source, target),
        excluded_nodes=excluded_nodes,
    )
    return float(nx.shortest_path_length(view, source, target, weight="latency_ms"))


def has_data_path(
    model: NetworkModel,
    source: str,
    target: str,
    *,
    excluded_nodes: Iterable[str] = (),
) -> bool:
    """Проверить достижимость по тем же правилам, что и реальный поиск пути."""
    try:
        shortest_data_path(
            model,
            source,
            target,
            excluded_nodes=excluded_nodes,
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return False
    return True


def path_uses_only_data_plane_transit(model: NetworkModel, path: Iterable[str]) -> bool:
    """Проверить физические рёбра и транзит только через C/A/RAN/OLT."""
    nodes = list(path)
    if not nodes or any(node not in model.graph for node in nodes):
        return False
    if any(
        not model.graph.has_edge(source, target)
        or model.graph.edges[source, target].get("medium") == LOGICAL_SERVICE_MEDIUM
        for source, target in zip(nodes, nodes[1:])
    ):
        return False
    return all(
        node in model.graph
        and model.graph.nodes[node].get("role") in DATA_PLANE_TRANSIT_ROLES
        for node in nodes[1:-1]
    )
