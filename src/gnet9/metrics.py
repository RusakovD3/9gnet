"""Graph and geometry metrics used by the current GNet9 topology."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Any, Mapping

import networkx as nx
import numpy as np

from .routing import (
    DATA_PLANE_TRANSIT_ROLES,
    LOGICAL_SERVICE_MEDIUM,
    data_plane_routing_view,
    shortest_data_path_length,
)


# The value is deliberately much larger than the normalised dynamic state
# thresholds.  It makes an entity identifier part of the metric space: a
# changed C1 tensor may not silently be matched to C2 just because their
# current numerical values look similar.
STATE_HAUSDORFF_LABEL_MISMATCH_COST = 10.0
STATE_HAUSDORFF_ALERT_DISTANCE = 1.0
_COORDINATE_METRICS = {"x", "y", "x_mid", "y_mid", "coordinate_norm"}


@dataclass(frozen=True)
class StateTensorHausdorffReference:
    """Precomputed invariant part of a state-Hausdorff comparison with ``t0``."""

    records: Mapping[str, Mapping[str, Any]]
    record_ids: tuple[str, ...]
    basis: tuple[tuple[str, str], ...]
    basis_index: Mapping[tuple[str, str], int]
    scales: Mapping[tuple[str, str], float]
    vectors: np.ndarray
    coordinate_indices: tuple[int, ...]


def prepare_state_hausdorff_reference(
    reference_tensor_state: Mapping[str, Any],
) -> StateTensorHausdorffReference:
    """Compile the invariant ``t0`` half of the sparse state metric once."""
    records = _tensor_records(reference_tensor_state)
    record_ids = tuple(sorted(records))
    basis = tuple(_state_metric_basis(records))
    basis_index = {item: index for index, item in enumerate(basis)}
    scales = _state_metric_scales(records, records, list(basis))
    vectors = _state_delta_vectors(
        list(record_ids), records, records, basis_index, scales
    )
    vectors.setflags(write=False)
    coordinate_indices = tuple(
        index for index, (_, metric_name) in enumerate(basis)
        if metric_name in _COORDINATE_METRICS
    )
    return StateTensorHausdorffReference(
        records=records,
        record_ids=record_ids,
        basis=basis,
        basis_index=basis_index,
        scales=scales,
        vectors=vectors,
        coordinate_indices=coordinate_indices,
    )


def state_tensor_hausdorff_view(
    reference_tensor_state: Mapping[str, Any],
    current_tensor_state: Mapping[str, Any],
    *,
    time_index: int,
    prepared_reference: StateTensorHausdorffReference | None = None,
) -> dict[str, Any]:
    """Measure divergence of the current sparse G-Net state from ``t0``.

    Each tensor-bearing graph object is a labelled point in an augmented
    metric space.  Its coordinate is the vector of metric changes normalised
    by the ``t0`` scale.  The label has a finite discrete distance, so the
    usual symmetric Hausdorff calculation preserves object identity while
    still reporting added/removed tensor-bearing objects as structural drift.

    The returned distance has the requested monotonic interpretation: zero is
    the reference state and a larger value is worse.  It is not a distance to
    a failure set, where the opposite interpretation would apply.
    """
    current = _tensor_records(current_tensor_state)
    compiled_reference = prepared_reference or prepare_state_hausdorff_reference(
        reference_tensor_state
    )
    reference = compiled_reference.records
    reference_ids = list(compiled_reference.record_ids)
    current_ids = sorted(current)
    if not reference_ids and not current_ids:
        return _empty_state_hausdorff(time_index)

    current_basis = _state_metric_basis(current)
    if set(current_basis).issubset(compiled_reference.basis_index):
        basis = list(compiled_reference.basis)
        basis_index = compiled_reference.basis_index
        scales = compiled_reference.scales
        reference_vectors = compiled_reference.vectors
        coordinate_indices = list(compiled_reference.coordinate_indices)
    else:
        # Preserve the full metric if a later snapshot introduces a new
        # coordinate that was absent at t0.
        basis = sorted(set(compiled_reference.basis) | set(current_basis))
        basis_index = {item: index for index, item in enumerate(basis)}
        scales = _state_metric_scales(reference, current, basis)
        reference_vectors = _state_delta_vectors(
            reference_ids, reference, reference, basis_index, scales
        )
        coordinate_indices = [
            index for index, (_, metric_name) in enumerate(basis)
            if metric_name in _COORDINATE_METRICS
        ]
    current_vectors = _state_delta_vectors(
        current_ids, current, reference, basis_index, scales
    )
    directed_reference, directed_current, reference_index, current_index = (
        _labelled_state_hausdorff(
            reference_ids,
            reference_vectors,
            current_ids,
            current_vectors,
        )
    )
    distance = max(directed_reference, directed_current)
    coordinate_distance = _projected_hausdorff_distance(
        reference_ids,
        reference_vectors,
        current_ids,
        current_vectors,
        coordinate_indices,
    )
    missing = sorted(set(reference_ids) - set(current_ids))
    added = sorted(set(current_ids) - set(reference_ids))
    contributors = _state_hausdorff_contributors(
        current_ids,
        current_vectors,
        basis,
        limit=8,
    )
    return {
        "metric": "label_preserving_sparse_state_tensor_hausdorff",
        "higher_is_worse": True,
        "distance": _finite_or_none(distance),
        "normalized_distance": _normalise_state_hausdorff(distance),
        "alert_distance": STATE_HAUSDORFF_ALERT_DISTANCE,
        "directed_t0_to_current": _finite_or_none(directed_reference),
        "directed_current_to_t0": _finite_or_none(directed_current),
        "coordinate_hausdorff_distance": _finite_or_none(coordinate_distance),
        "time_index": int(time_index),
        "tensor_representation": {
            "storage": "sparse_coordinate_view_over_snapshot_history",
            "rank": 5,
            "axes": ["entity", "layer", "relation", "metric", "time"],
            "reference_time_index": 0,
            "current_time_index": int(time_index),
            "dense_allocation": "not_used",
            "point_count_t0": len(reference_ids),
            "point_count_current": len(current_ids),
            "metric_coordinate_count": len(basis),
        },
        "base_metric": {
            "formula": "d((id,z),(id',z')) = ||z-z'||_2 + 10*1[id!=id']",
            "state_coordinate": "z_k=(x_k(t)-x_k(t0))/s_k",
            "scale_semantics": "t0_magnitude_or_natural_unit_scale",
            "label_mismatch_cost": STATE_HAUSDORFF_LABEL_MISMATCH_COST,
            "hausdorff_formula": "max{sup_a inf_b d(a,b), sup_b inf_a d(a,b)}",
        },
        "structural_difference": {
            "missing_reference_tensor_ids": missing,
            "added_current_tensor_ids": added,
            "has_structural_difference": bool(missing or added),
        },
        "witnesses": {
            "t0_to_current_tensor_id": (
                reference_ids[reference_index] if reference_index is not None else None
            ),
            "current_to_t0_tensor_id": (
                current_ids[current_index] if current_index is not None else None
            ),
        },
        "top_current_contributors": contributors,
    }


def _tensor_records(tensor_state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for level, items in tensor_state.get("by_level", {}).items():
        for item in items:
            scope = str(item.get("scope", "node"))
            if scope == "edge":
                endpoints = "--".join(sorted((str(item.get("source")), str(item.get("target")))))
                object_id = f"edge:{endpoints}"
            else:
                object_id = f"node:{item.get('node_id')}"
            tensor_name = str(item.get("tensor_name", "tensor"))
            record_id = f"{object_id}:{tensor_name}"
            records[record_id] = {
                "level": str(item.get("level", level)),
                "metrics": {
                    str(name): float(value)
                    for name, value in item.get("metrics", {}).items()
                    if math.isfinite(float(value))
                },
                "units": {str(name): str(unit) for name, unit in item.get("units", {}).items()},
            }
    return records


def _state_metric_basis(
    records: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, str]]:
    return sorted(
        {
            (str(record["level"]), metric_name)
            for record in records.values()
            for metric_name in record["metrics"]
        }
    )


def _state_metric_scales(
    reference: Mapping[str, Mapping[str, Any]],
    current: Mapping[str, Mapping[str, Any]],
    basis: list[tuple[str, str]],
) -> dict[tuple[str, str], float]:
    coordinate_magnitudes = [
        abs(float(record["metrics"].get(metric_name, 0.0)))
        for record in reference.values()
        for metric_name in _COORDINATE_METRICS
        if metric_name in record["metrics"]
    ]
    coordinate_scale = max(max(coordinate_magnitudes, default=0.0), 1.0)
    scales: dict[tuple[str, str], float] = {}
    for level, metric_name in basis:
        values = [
            abs(float(record["metrics"].get(metric_name, 0.0)))
            for record in reference.values()
            if record["level"] == level and metric_name in record["metrics"]
        ]
        unit = next(
            (
                record["units"].get(metric_name, "")
                for record in [*reference.values(), *current.values()]
                if record["level"] == level and metric_name in record["units"]
            ),
            "",
        )
        if metric_name in _COORDINATE_METRICS or unit == "model-coordinate":
            scales[(level, metric_name)] = coordinate_scale
        elif unit in {"ratio", "boolean", "normalized"}:
            scales[(level, metric_name)] = 1.0
        elif unit == "%":
            scales[(level, metric_name)] = 100.0
        else:
            scales[(level, metric_name)] = max(max(values, default=0.0), 1.0)
    return scales


def _state_delta_vectors(
    record_ids: list[str],
    records: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, Any]],
    basis_index: Mapping[tuple[str, str], int],
    scales: Mapping[tuple[str, str], float],
) -> np.ndarray:
    vectors = np.zeros((len(record_ids), len(basis_index)), dtype=float)
    for row, record_id in enumerate(record_ids):
        record = records[record_id]
        baseline = reference.get(record_id, {})
        baseline_metrics = baseline.get("metrics", {})
        for metric_name, value in record["metrics"].items():
            key = (str(record["level"]), metric_name)
            column = basis_index[key]
            vectors[row, column] = (
                float(value) - float(baseline_metrics.get(metric_name, 0.0))
            ) / max(float(scales[key]), 1e-12)
    return vectors


def _labelled_state_hausdorff(
    ids_a: list[str],
    vectors_a: np.ndarray,
    ids_b: list[str],
    vectors_b: np.ndarray,
) -> tuple[float, float, int | None, int | None]:
    """Exact labelled Hausdorff distance without an O(N²) allocation.

    The dense pairwise matrix is mathematically convenient but needlessly
    allocates hundreds of megabytes for every dynamics tick.  Computing rows
    in bounded chunks gives the same minima and witnesses while retaining the
    sparse-state model's intended memory behaviour.
    """
    if not ids_a and not ids_b:
        return 0.0, 0.0, None, None
    if not ids_a or not ids_b:
        return math.inf, math.inf, None, None
    # In the normal dynamics path every tensor keeps its identity.  When the
    # corresponding labelled distance is below the fixed mismatch cost, that
    # same-label point is provably closer than any other label (whose distance
    # is at least the mismatch cost).  This exact short path is O(N·K); the
    # chunked all-pairs fallback below remains for a true structural change or
    # an exceptionally large state excursion.
    if len(ids_a) == len(ids_b) and set(ids_a) == set(ids_b):
        b_index_by_id = {item_id: index for index, item_id in enumerate(ids_b)}
        matching_b_indices = np.asarray(
            [b_index_by_id[item_id] for item_id in ids_a], dtype=int
        )
        same_label_distances = np.linalg.norm(
            vectors_a - vectors_b[matching_b_indices], axis=1
        )
        if float(np.max(same_label_distances)) < STATE_HAUSDORFF_LABEL_MISMATCH_COST:
            a_index = int(np.argmax(same_label_distances))
            return (
                float(same_label_distances[a_index]),
                float(same_label_distances[a_index]),
                a_index,
                int(matching_b_indices[a_index]),
            )
    b_min = np.full(len(ids_b), math.inf, dtype=float)
    a_min = np.empty(len(ids_a), dtype=float)
    b_ids = np.asarray(ids_b, dtype=object)
    chunk_size = 32
    for start in range(0, len(ids_a), chunk_size):
        stop = min(len(ids_a), start + chunk_size)
        euclidean = np.linalg.norm(
            vectors_a[start:stop, None, :] - vectors_b[None, :, :], axis=2
        )
        penalty = np.not_equal(
            np.asarray(ids_a[start:stop], dtype=object)[:, None], b_ids[None, :]
        ).astype(float) * STATE_HAUSDORFF_LABEL_MISMATCH_COST
        distances = euclidean + penalty
        a_min[start:stop] = distances.min(axis=1)
        b_min = np.minimum(b_min, distances.min(axis=0))
    a_to_b = a_min
    b_to_a = b_min
    a_index = int(np.argmax(a_to_b))
    b_index = int(np.argmax(b_to_a))
    return float(a_to_b[a_index]), float(b_to_a[b_index]), a_index, b_index


def _projected_hausdorff_distance(
    reference_ids: list[str],
    reference_vectors: np.ndarray,
    current_ids: list[str],
    current_vectors: np.ndarray,
    indices: list[int],
) -> float:
    if not indices:
        return 0.0
    first, second, _, _ = _labelled_state_hausdorff(
        reference_ids,
        reference_vectors[:, indices],
        current_ids,
        current_vectors[:, indices],
    )
    return max(first, second)


def _state_hausdorff_contributors(
    current_ids: list[str],
    current_vectors: np.ndarray,
    basis: list[tuple[str, str]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if not current_ids:
        return []
    scores = np.linalg.norm(current_vectors, axis=1)
    result: list[dict[str, Any]] = []
    for index in np.argsort(scores)[::-1][:limit]:
        score = float(scores[int(index)])
        if score <= 1e-12:
            continue
        components = np.argsort(np.abs(current_vectors[int(index)]))[::-1][:3]
        result.append(
            {
                "tensor_id": current_ids[int(index)],
                "normalized_state_distance": round(score, 8),
                "largest_metric_deltas": [
                    {
                        "level": basis[int(component)][0],
                        "metric": basis[int(component)][1],
                        "normalized_delta": round(
                            float(current_vectors[int(index), int(component)]), 8
                        ),
                    }
                    for component in components
                    if abs(float(current_vectors[int(index), int(component)])) > 1e-12
                ],
            }
        )
    return result


def _normalise_state_hausdorff(distance: float) -> float:
    if not math.isfinite(distance):
        return 1.0
    return round(min(1.0, max(0.0, distance / STATE_HAUSDORFF_ALERT_DISTANCE)), 8)


def _finite_or_none(value: float) -> float | None:
    return None if not math.isfinite(value) else round(float(value), 8)


def _empty_state_hausdorff(time_index: int) -> dict[str, Any]:
    return {
        "metric": "label_preserving_sparse_state_tensor_hausdorff",
        "higher_is_worse": True,
        "distance": 0.0,
        "normalized_distance": 0.0,
        "coordinate_hausdorff_distance": 0.0,
        "time_index": int(time_index),
        "tensor_representation": {
            "storage": "sparse_coordinate_view_over_snapshot_history",
            "rank": 5,
            "axes": ["entity", "layer", "relation", "metric", "time"],
            "dense_allocation": "not_used",
        },
        "top_current_contributors": [],
    }


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
        "proximity_risk": round(proximity_risk, 6),
        "maximum_exposure": round(maximum_exposure, 6),
        "weighted_mean_exposure": round(weighted_mean, 6),
        "decay_tau_ms": tau_ms,
        "decay_tau_origin": "gnet9_calibrated_scenario_assumption",
        "affected_nodes": affected_nodes,
    }


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
