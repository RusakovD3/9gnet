"""Discrete baseline dynamics for G-Net.

The first dynamic mode is intentionally stationary: it repeats the same healthy
network state at fixed time steps. This gives downstream Koopman/DMD,
Hausdorff and Lyapunov code a stable snapshot format before failures and
remapping are introduced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .models import NetworkModel, StateTensor


@dataclass(frozen=True)
class DynamicsConfig:
    """Discrete simulation clock."""

    step_seconds: int = 5
    duration_seconds: int = 30
    include_t0: bool = True


def simulate_stationary_dynamics(model: NetworkModel, config: DynamicsConfig | None = None) -> dict[str, Any]:
    """Return full graph snapshots for a healthy stationary baseline."""
    config = config or DynamicsConfig()
    step_count = config.duration_seconds // config.step_seconds
    start_step = 0 if config.include_t0 else 1
    snapshots = [
        _snapshot(model, step_index, step_index * config.step_seconds)
        for step_index in range(start_step, step_count + 1)
    ]

    return {
        "mode": "stationary_healthy_baseline",
        "config": asdict(config),
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
    }


def _snapshot(model: NetworkModel, step_index: int, time_seconds: int) -> dict[str, Any]:
    return {
        "step_index": step_index,
        "time_seconds": time_seconds,
        "level_summary": model.level_summary,
        "nodes": [_node_snapshot(node_id, attrs) for node_id, attrs in model.graph.nodes(data=True)],
        "edges": [_edge_snapshot(source, target, attrs) for source, target, attrs in model.graph.edges(data=True)],
    }


def _node_snapshot(node_id: str, attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": node_id,
        "level": attrs.get("level"),
        "role": attrs.get("role"),
        "pos": _json_value(attrs.get("pos")),
        "tensors": _tensor_attrs(attrs),
    }


def _edge_snapshot(source: str, target: str, attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "medium": attrs.get("medium"),
        "logical_level": attrs.get("logical_level"),
        "physical_level": attrs.get("physical_level"),
        "capacity_mbps": attrs.get("capacity_mbps"),
        "latency_ms": attrs.get("latency_ms"),
        "redundancy": attrs.get("redundancy"),
        "tensors": _tensor_attrs(attrs),
    }


def _tensor_attrs(attrs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: _tensor_snapshot(value)
        for name, value in attrs.items()
        if isinstance(value, StateTensor)
    }


def _tensor_snapshot(tensor: StateTensor) -> dict[str, Any]:
    values = {
        metric_name: float(tensor.data[index[0]])
        for metric_name, index in tensor.metric_index.items()
    }
    return {
        "level": tensor.level,
        "metrics": values,
        "units": tensor.units,
        "vector": [float(value) for value in tensor.data.tolist()],
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value
