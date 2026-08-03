"""Graph and geometry metrics used by the current GNet9 topology."""

from __future__ import annotations

import math
from typing import Iterable, Any

import networkx as nx

from .routing import (
    DATA_PLANE_TRANSIT_ROLES,
    LOGICAL_SERVICE_MEDIUM,
    data_plane_routing_view,
    shortest_data_path_length,
)


def vertex_proximity_index(graph: nx.Graph, nodes: Iterable[str]) -> dict[str, float]:
    """Return closeness centrality for selected nodes.

    In plain language: the closer a node is to all other selected nodes, the
    higher its value. In the project this is used as a readable centrality metric
    for core routers.
    """
    selected_nodes = list(nodes)
    subgraph = graph.subgraph(selected_nodes)
    return {node: float(value) for node, value in nx.closeness_centrality(subgraph).items()}


def hausdorff_distance(points_a: Iterable[tuple[float, float]], points_b: Iterable[tuple[float, float]]) -> float:
    """Return extended symmetric Hausdorff distance for two 2D point sets.

    In this project it is used as an interpretable remapping signal: how far the
    affected/changed topology area is from the protected reference area.
    Two empty sets have distance zero; exactly one empty set has infinite
    distance because no finite matching point exists.
    """
    a = list(points_a)
    b = list(points_b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return math.inf
    return max(_directed_hausdorff(a, b), _directed_hausdorff(b, a))


def directed_hausdorff_distance(points_a: Iterable[tuple[float, float]], points_b: Iterable[tuple[float, float]]) -> float:
    """Return extended directed Hausdorff distance from set A to set B.

    The empty source has no unmatched points and therefore distance zero;
    a non-empty source compared with an empty target has infinite distance.
    """
    a = list(points_a)
    b = list(points_b)
    if not a:
        return 0.0
    if not b:
        return math.inf
    return _directed_hausdorff(a, b)


def gold_threat_proximity_view(model, attack_state: dict) -> dict[str, Any]:
    """Оценить экспозицию Gold-маршрутов в метрике задержки графа.

    Координаты рисунка не являются физической метрикой сети. Поэтому риск
    считается по кратчайшей задержке от наблюдаемой области воздействия до
    критичных узлов. Максимальная экспозиция сохраняет монотонность: добавление
    ещё одной близкой цели не может сделать атаку визуально «безопаснее».
    """
    events = list(attack_state.get("events", [])) if isinstance(attack_state, dict) else []
    if isinstance(attack_state, dict):
        events.extend(
            {
                "target_id": item.get("sensor_node_id") or item.get("suspected_target_id"),
                "routes": [],
            }
            for item in attack_state.get("precursors", [])
            if item.get("sensor_node_id") or item.get("suspected_target_id")
        )
    affected_nodes = sorted(set(_affected_targets_from_events(events)))
    critical_nodes = [
        node_id
        for node_id, attrs in model.graph.nodes(data=True)
        if attrs.get("critical_protection", {}).get("is_critical")
        and attrs.get("role") != "service"
    ]
    distances: list[tuple[float, float]] = []
    for critical in critical_nodes:
        candidates: list[float] = []
        for source in affected_nodes:
            try:
                candidates.append(shortest_data_path_length(model, source, critical))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        if not candidates:
            continue
        protection = model.graph.nodes[critical].get("critical_protection", {})
        weight = 1.0 + 4.0 * float(protection.get("critical_involvement_coefficient", 0.0))
        distances.append((min(candidates), weight))

    tau_ms = 12.0
    exposures = [(math.exp(-distance / tau_ms), weight) for distance, weight in distances]
    weighted_mean = (
        sum(exposure * weight for exposure, weight in exposures)
        / max(sum(weight for _, weight in exposures), 1e-9)
        if exposures else 0.0
    )
    maximum_exposure = max((value for value, _ in exposures), default=0.0)
    proximity_risk = 0.0 if not affected_nodes else 0.65 * maximum_exposure + 0.35 * weighted_mean
    minimum_distance = min((distance for distance, _ in distances), default=0.0)
    weighted_distance = (
        sum(distance * weight for distance, weight in distances)
        / max(sum(weight for _, weight in distances), 1e-9)
        if distances else 0.0
    )
    network_scale = max(_weighted_network_scale(model), 1e-9)
    normalized = min(1.0, weighted_distance / network_scale)
    return {
        "metric": "gold_threat_proximity_by_shortest_path_latency",
        "is_hausdorff_metric": False,
        "affected_node_count": len(affected_nodes),
        "critical_node_count": len(distances),
        "minimum_threat_to_gold_latency_ms": round(minimum_distance, 6),
        "weighted_mean_threat_to_gold_latency_ms": round(weighted_distance, 6),
        "normalized_weighted_proximity_distance": round(normalized, 6),
        "normalization_scale_ms": round(network_scale, 6),
        "normalization_scale_semantics": (
            "transit_core_latency_diameter_plus_two_maximum_physical_endpoint_attachments"
        ),
        "distance": round(minimum_distance, 6),
        "symmetric_distance": round(weighted_distance, 6),
        "normalized_distance": round(normalized, 6),
        "proximity_risk": round(proximity_risk, 6),
        "maximum_exposure": round(maximum_exposure, 6),
        "weighted_mean_exposure": round(weighted_mean, 6),
        "decay_tau_ms": tau_ms,
        "decay_tau_origin": "gnet9_calibrated_scenario_assumption",
        "legacy_aliases": {
            "distance": "minimum_threat_to_gold_latency_ms",
            "symmetric_distance": "weighted_mean_threat_to_gold_latency_ms",
            "normalized_distance": "normalized_weighted_proximity_distance",
        },
        "affected_nodes": affected_nodes,
    }


def attack_hausdorff_view(model, attack_state: dict) -> dict[str, Any]:
    """Совместимый alias; каноническая функция не является Hausdorff-метрикой."""
    return gold_threat_proximity_view(model, attack_state)


def gold_route_hausdorff_view(model, flows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Хаусдорфово расстояние между исходными и активными Gold-маршрутами.

    Это отдельная метрика стоимости ремаппинга: угроза измеряется функцией
    выше, а здесь оценивается, насколько далеко контроллер увёл маршруты от t0.
    """
    gold = [
        flow for flow in flows
        if flow.get("sla_grade") == "gold" and not flow.get("is_attack_traffic")
    ]
    scale = max(_weighted_network_scale(model), 1e-9)
    distance_cache: dict[tuple[str, str], float | None] = {}
    original_union: set[str] = set()
    active_union: set[str] = set()
    route_distances: list[float] = []
    original_to_active: list[float] = []
    active_to_original: list[float] = []
    edge_jaccard_distances: list[float] = []
    latency_stretches: list[float] = []
    changed = 0
    isolated = 0
    for flow in gold:
        original = [str(node) for node in flow.get("original_route", flow.get("route", []))]
        active = [str(node) for node in flow.get("route", [])]
        original_union.update(original)
        active_union.update(active)
        # Топология NetworkX неориентированная: одно физическое ребро не
        # должно считаться заменённым лишь из-за обратного порядка маршрута.
        original_edges = {
            tuple(sorted((source, target)))
            for source, target in zip(original, original[1:])
        }
        active_edges = {
            tuple(sorted((source, target)))
            for source, target in zip(active, active[1:])
        }
        edge_union = original_edges | active_edges
        edge_jaccard_distances.append(
            0.0
            if not edge_union
            else 1.0 - len(original_edges & active_edges) / len(edge_union)
        )
        if original and active:
            original_latency = _route_latency_ms(model.graph, original)
            active_latency = _route_latency_ms(model.graph, active)
            latency_stretches.append(active_latency / max(original_latency, 1e-9))
        if original != active:
            changed += 1
        if original and not active:
            # Пустое активное множество означает потерю маршрута, а не нулевое
            # расстояние. Берём масштаб сети как максимальную стоимость.
            directed_ab = directed_ba = scale
            isolated += 1
        else:
            directed_ab = _graph_directed_hausdorff(
                model,
                original,
                active,
                distance_cache,
            )
            directed_ba = _graph_directed_hausdorff(
                model,
                active,
                original,
                distance_cache,
            )
        original_to_active.append(directed_ab)
        active_to_original.append(directed_ba)
        route_distances.append(max(directed_ab, directed_ba))

    distance = max(route_distances, default=0.0)
    mean_distance = sum(route_distances) / max(len(route_distances), 1)
    directed_ab = max(original_to_active, default=0.0)
    directed_ba = max(active_to_original, default=0.0)
    return {
        "metric": "maximum_per_flow_gold_route_hausdorff_latency",
        "distance_ms": round(distance, 6),
        "mean_distance_ms": round(mean_distance, 6),
        "normalized_distance": round(min(1.0, distance / scale), 6),
        "normalization_scale_ms": round(scale, 6),
        "normalization_scale_semantics": (
            "transit_core_latency_diameter_plus_two_maximum_physical_endpoint_attachments"
        ),
        "directed_original_to_active_ms": round(directed_ab, 6),
        "directed_active_to_original_ms": round(directed_ba, 6),
        "changed_gold_flow_count": changed,
        "isolated_gold_flow_count": isolated,
        "maximum_edge_jaccard_distance": round(max(edge_jaccard_distances, default=0.0), 6),
        "mean_edge_jaccard_distance": round(
            sum(edge_jaccard_distances) / max(len(edge_jaccard_distances), 1),
            6,
        ),
        "maximum_latency_stretch_ratio": round(max(latency_stretches, default=1.0), 6),
        "mean_latency_stretch_ratio": round(
            sum(latency_stretches) / max(len(latency_stretches), 1),
            6,
        ),
        "interpretation": (
            "Hausdorff compares route node sets; edge Jaccard preserves edge changes and latency stretch preserves cost"
        ),
        "gold_flow_count": len(gold),
        "original_node_count": len(original_union),
        "active_node_count": len(active_union),
    }


def _route_latency_ms(graph: nx.Graph, route: list[str]) -> float:
    return sum(
        float(graph.edges[source, target].get("latency_ms", 0.0))
        for source, target in zip(route, route[1:])
    )


def _directed_hausdorff(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    return max(min(_euclidean(pa, pb) for pb in b) for pa in a)


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _affected_targets_from_events(events: Iterable[dict]) -> list[str]:
    return [str(event["target_id"]) for event in events if event.get("target_id")]


def _graph_directed_hausdorff(
    model,
    nodes_a: Iterable[str],
    nodes_b: Iterable[str],
    cache: dict[tuple[str, str], float | None],
) -> float:
    """Направленная маршрутная метрика без endpoint-транзита.

    Кеш ограничен одним снимком и устраняет повторные Dijkstra-вызовы для
    одинаковых пар узлов во множестве Gold-потоков.
    """
    a = [node for node in nodes_a if node in model.graph]
    b = [node for node in nodes_b if node in model.graph]
    if not a or not b:
        return 0.0
    directed: list[float] = []
    for source in a:
        reachable: list[float] = []
        for target in b:
            key = tuple(sorted((source, target)))
            if key not in cache:
                try:
                    cache[key] = shortest_data_path_length(model, source, target)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    cache[key] = None
            value = cache[key]
            if value is not None:
                reachable.append(value)
        if reachable:
            directed.append(min(reachable))
    return max(directed, default=0.0)


def _weighted_network_scale(model) -> float:
    """Оценить максимальный data-plane путь без all-pairs по всем абонентам.

    Верхняя граница складывается из диаметра транзитного ядра и двух самых
    длинных физических плеч доступа. Так нормализация учитывает radio/fixed
    endpoint-задержки, но никогда не превращает endpoint в промежуточный узел.
    """
    transit_graph = data_plane_routing_view(model)
    transit_nodes = list(transit_graph.nodes)
    transit_diameter = 1.0
    for source in transit_nodes:
        lengths = nx.single_source_dijkstra_path_length(
            transit_graph,
            source,
            weight="latency_ms",
        )
        transit_diameter = max(
            transit_diameter,
            max(
                (float(lengths.get(target, 0.0)) for target in transit_nodes),
                default=0.0,
            ),
        )

    maximum_attachment = 0.0
    for source, target, attrs in model.graph.edges(data=True):
        if attrs.get("medium") == LOGICAL_SERVICE_MEDIUM:
            continue
        source_role = model.graph.nodes[source].get("role")
        target_role = model.graph.nodes[target].get("role")
        exactly_one_transit = (
            (source_role in DATA_PLANE_TRANSIT_ROLES)
            != (target_role in DATA_PLANE_TRANSIT_ROLES)
        )
        if exactly_one_transit:
            maximum_attachment = max(
                maximum_attachment,
                float(attrs.get("latency_ms", 0.0)),
            )
    return transit_diameter + 2.0 * maximum_attachment
