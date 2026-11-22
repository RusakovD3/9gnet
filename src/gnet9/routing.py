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
from functools import cached_property

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


class FixedTopologyRoutes:
    """Пути и задержки для одного прогона с фиксированным графом."""

    def __init__(self, model: NetworkModel):
        self._model = model
        self._paths: dict[tuple[str, str], tuple[str, ...]] = {}
        self._transit_distances: dict[str, dict[str, float]] = {}

    def path(self, source: str, target: str) -> list[str]:
        key = (source, target)
        if key not in self._paths:
            self._paths[key] = tuple(shortest_data_path(self._model, source, target))
        return list(self._paths[key])

    def distance(self, source: str, target: str) -> float | None:
        graph = self._model.graph
        if source not in graph or target not in graph:
            return None
        if source not in self._transit_distances:
            view = data_plane_routing_view(self._model, endpoints=(source,))
            self._transit_distances[source] = nx.single_source_dijkstra_path_length(
                view, source, weight="latency_ms"
            )
        distances = self._transit_distances[source]
        if target in distances:
            return distances[target]
        # Конечное устройство можно добавить только последним ребром.
        # Один обход обслуживает все цели и не превращает сервер в транзит.
        neighbors = graph.pred[target] if graph.is_directed() else graph[target]
        return min((
            distances[peer] + attrs.get("latency_ms", 1)
            for peer, attrs in neighbors.items()
            if peer in distances and attrs.get("medium") != LOGICAL_SERVICE_MEDIUM
        ), default=None)

    @cached_property
    def latency_scale_ms(self) -> float:
        """Диаметр транзитных узлов и два максимальных плеча доступа, мс."""
        transit = data_plane_routing_view(self._model)
        diameter = 1.0
        for _, lengths in nx.all_pairs_dijkstra_path_length(transit, weight="latency_ms"):
            diameter = max(diameter, max(lengths.values(), default=0.0))

        graph = self._model.graph
        attachment = max((
            float(attrs.get("latency_ms", 0.0))
            for source, target, attrs in graph.edges(data=True)
            if attrs.get("medium") != LOGICAL_SERVICE_MEDIUM
            and (source in transit) != (target in transit)
        ), default=0.0)
        return diameter + 2.0 * attachment


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
