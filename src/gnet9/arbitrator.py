"""L7 arbitrator analysis over G-Net tensor state.

The arbitrator does not own topology construction or time stepping. It receives
tensor snapshots from `dynamics.py`, reduces them to stable metrics and prepares
the fields later needed for remapping, Koopman/DMD and Lyapunov analysis.
"""

from __future__ import annotations

from typing import Any

from .models import NetworkModel, StateTensor


# Keep the observed set intentionally small: these metrics are enough for a
# readable first remapping signal and a compact state vector for time-series math.
ARBITRATOR_OBSERVED_METRICS = {
    "L1": ("sla_margin", "traffic_intensity_rho"),
    "L2": ("cpu_load_percent", "ram_load_percent", "stability_margin"),
    "EDGE": ("loss_probability", "stability_margin", "utilization"),
    "L8": ("terrain_risk",),
    "L7": ("hausdorff_distance", "lyapunov_value", "lyapunov_delta", "koopman_residual", "remap_pressure"),
}

STATE_VECTOR_METRICS = (
    ("L1", "sla_margin", "min"),
    ("L1", "traffic_intensity_rho", "mean"),
    ("L2", "cpu_load_percent", "max"),
    ("L2", "ram_load_percent", "max"),
    ("L2", "stability_margin", "min"),
    ("EDGE", "loss_probability", "max"),
    ("EDGE", "stability_margin", "min"),
    ("EDGE", "utilization", "mean"),
    ("L8", "terrain_risk", "max"),
    ("L7", "hausdorff_distance", "mean"),
    ("L7", "koopman_residual", "mean"),
    ("L7", "remap_pressure", "mean"),
)


def build_arbitrator_view(model: NetworkModel, tensor_state: dict[str, Any]) -> dict[str, Any]:
    """Return L7 analysis and a no-remap decision for the current snapshot.

    Healthy baseline dynamics should keep `remap.needed` false. Future degraded
    modes can feed the same tensor-state structure and let these thresholds start
    proposing remap actions.
    """
    aggregates = level_metric_aggregates(tensor_state, observed_only=True)
    full_aggregates = level_metric_aggregates(tensor_state, observed_only=False)
    l7_tensor = model.graph.nodes["ARB"].get("tensor") if "ARB" in model.graph.nodes else None
    l7_metrics = tensor_metrics(l7_tensor) if isinstance(l7_tensor, StateTensor) else {}
    remap_pressure = remap_pressure_from_tensors(aggregates, l7_metrics)
    lyapunov_value = lyapunov_value_from_tensors(aggregates, l7_metrics, remap_pressure)
    koopman_residual = koopman_residual_from_tensors(aggregates, l7_metrics, remap_pressure)
    state_vector = build_state_vector(full_aggregates)

    return {
        "node_id": "ARB",
        "input_tensor_counts": tensor_state["counts"],
        "level_metric_aggregates": aggregates,
        "state_vector": state_vector,
        "analysis": {
            "hausdorff_distance": l7_metrics.get("hausdorff_distance", 0.0),
            "lyapunov_value": lyapunov_value,
            "lyapunov_delta": l7_metrics.get("lyapunov_delta", 0.0),
            "koopman_residual": koopman_residual,
            "remap_pressure": remap_pressure,
            "decision_confidence": decision_confidence(remap_pressure, l7_metrics),
        },
        "remap": {
            "needed": remap_pressure > 0.20,
            "action": "NO_REMAP" if remap_pressure <= 0.20 else "PLAN_REMAP",
            "reason": "healthy_stationary_baseline" if remap_pressure <= 0.20 else "tensor_threshold_pressure",
            "candidate_actions": [] if remap_pressure <= 0.20 else ["reroute_high_pressure_flows", "increase_slice_reserve"],
        },
    }


def level_metric_aggregates(tensor_state: dict[str, Any], *, observed_only: bool) -> dict[str, dict[str, Any]]:
    """Aggregate tensor metrics by level using min, max and mean."""
    result: dict[str, dict[str, Any]] = {}
    for level, tensors in tensor_state["by_level"].items():
        allowed_metrics = set(ARBITRATOR_OBSERVED_METRICS.get(level, ())) if observed_only else None
        metric_values: dict[str, list[float]] = {}
        for item in tensors:
            for metric_name, value in item["metrics"].items():
                if allowed_metrics is not None and metric_name not in allowed_metrics:
                    continue
                metric_values.setdefault(metric_name, []).append(float(value))

        result[level] = {
            "tensor_count": len(tensors),
            "metrics": {
                metric_name: {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                }
                for metric_name, values in metric_values.items()
            },
        }
    return result


def build_state_vector(aggregates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build a compact numeric vector for Koopman/DMD and Lyapunov pipelines."""
    metric_names = [
        f"{level}.{metric_name}.{statistic}"
        for level, metric_name, statistic in STATE_VECTOR_METRICS
    ]
    vector = [
        aggregate_metric(aggregates, level, metric_name, statistic, 0.0)
        for level, metric_name, statistic in STATE_VECTOR_METRICS
    ]
    return {
        "metric_names": metric_names,
        "vector": vector,
    }


def remap_pressure_from_tensors(aggregates: dict[str, dict[str, Any]], l7_metrics: dict[str, float]) -> float:
    l1_min_sla = aggregate_metric(aggregates, "L1", "sla_margin", "min", 1.0)
    l2_max_cpu = aggregate_metric(aggregates, "L2", "cpu_load_percent", "max", 0.0)
    edge_min_stability = aggregate_metric(aggregates, "EDGE", "stability_margin", "min", 1.0)
    edge_max_loss = aggregate_metric(aggregates, "EDGE", "loss_probability", "max", 0.0)
    terrain_max_risk = aggregate_metric(aggregates, "L8", "terrain_risk", "max", 0.0)

    pressures = [
        (0.60 - l1_min_sla) / 0.60,
        (l2_max_cpu - 70.0) / 30.0,
        (0.20 - edge_min_stability) / 0.20,
        (edge_max_loss - 0.003) / 0.002,
        (terrain_max_risk - 0.50) / 0.50,
        l7_metrics.get("remap_pressure", 0.0),
    ]
    return round(max(0.0, min(1.0, max(pressures))), 6)


def lyapunov_value_from_tensors(
    aggregates: dict[str, dict[str, Any]],
    l7_metrics: dict[str, float],
    remap_pressure: float,
) -> float:
    edge_mean_stability = aggregate_metric(aggregates, "EDGE", "stability_margin", "mean", 1.0)
    l1_mean_sla = aggregate_metric(aggregates, "L1", "sla_margin", "mean", 1.0)
    baseline_value = l7_metrics.get("lyapunov_value", 0.0)
    stress = (1.0 - edge_mean_stability) * 0.10 + (1.0 - l1_mean_sla) * 0.10 + remap_pressure * 0.25
    return round(baseline_value + stress, 6)


def koopman_residual_from_tensors(
    aggregates: dict[str, dict[str, Any]],
    l7_metrics: dict[str, float],
    remap_pressure: float,
) -> float:
    edge_mean_loss = aggregate_metric(aggregates, "EDGE", "loss_probability", "mean", 0.0)
    l2_mean_cpu = aggregate_metric(aggregates, "L2", "cpu_load_percent", "mean", 0.0)
    baseline_residual = l7_metrics.get("koopman_residual", 0.0)
    residual = baseline_residual + edge_mean_loss * 10.0 + max(0.0, l2_mean_cpu - 50.0) / 1000.0 + remap_pressure * 0.20
    return round(residual, 6)


def decision_confidence(remap_pressure: float, l7_metrics: dict[str, float]) -> float:
    baseline_confidence = l7_metrics.get("decision_confidence", 0.90)
    return round(max(0.0, min(1.0, baseline_confidence - remap_pressure * 0.45)), 6)


def aggregate_metric(
    aggregates: dict[str, dict[str, Any]],
    level: str,
    metric_name: str,
    statistic: str,
    default: float,
) -> float:
    metric = aggregates.get(level, {}).get("metrics", {}).get(metric_name)
    if not metric:
        return default
    return float(metric.get(statistic, default))


def tensor_metrics(tensor: StateTensor) -> dict[str, float]:
    return {
        metric_name: float(tensor.data[index[0]])
        for metric_name, index in tensor.metric_index.items()
    }
