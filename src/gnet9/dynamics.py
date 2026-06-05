"""Discrete baseline dynamics for G-Net.

The first dynamic mode is intentionally stationary: it repeats the same healthy
network state at fixed time steps. This gives downstream Koopman/DMD,
Hausdorff and Lyapunov code a stable snapshot format before failures and
remapping are introduced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from .arbitrator import build_arbitrator_view, tensor_metrics
from .constants import DYNAMICS_STEP_SECONDS, DYNAMICS_STEPS
from .models import NetworkModel, StateTensor
from .packet_simulator import simulate_packet_snapshot


TENSOR_LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "EDGE")
SnapshotDetail = Literal["full", "tensor", "summary"]
PacketDetail = Literal["summary", "flows", "sample"]


@dataclass(frozen=True)
class DynamicsConfig:
    """Discrete simulation clock and export-detail settings.

    `snapshot_detail` controls how much graph/tensor data is written per step:
    full = graph + all tensors, tensor = all tensors without graph lists,
    summary = only compact tensor counts, state vector and arbitrator aggregates.
    """

    step_seconds: int = DYNAMICS_STEP_SECONDS
    step_count: int = DYNAMICS_STEPS
    include_t0: bool = True
    include_packet_simulation: bool = True
    snapshot_detail: SnapshotDetail = "full"
    packet_detail: PacketDetail = "sample"
    packet_sample_limit: int = 48

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration_seconds"] = self.step_seconds * self.step_count
        return data


def simulate_stationary_dynamics(model: NetworkModel, config: DynamicsConfig | None = None) -> dict[str, Any]:
    """Return full graph snapshots for a healthy stationary baseline."""
    config = config or DynamicsConfig()
    _validate_config(config)
    health = validate_healthy_baseline(model)
    if not health["ok"]:
        raise ValueError(f"Cannot simulate dynamics from unhealthy baseline: {health['violation_count']} violations")

    start_step = 0 if config.include_t0 else 1
    snapshots = [
        _snapshot(model, config, step_index, step_index * config.step_seconds)
        for step_index in range(start_step, config.step_count + 1)
    ]

    return {
        "mode": "stationary_healthy_baseline",
        "config": config.to_dict(),
        "health": health,
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
    }


def validate_healthy_baseline(model: NetworkModel) -> dict[str, Any]:
    """Check that t0 is inside the intended healthy SLA/SLO envelope."""
    violations: list[dict[str, Any]] = []
    checked_l1_points = 0
    checked_l2_nodes = 0
    checked_edges = 0

    for node_id, attrs in model.graph.nodes(data=True):
        level = attrs.get("level")
        if level == "L1":
            for point in attrs.get("monitoring", []):
                checked_l1_points += 1
                failed_flags = [
                    flag
                    for flag in ("bitrate_slo_ok", "latency_slo_ok", "loss_slo_ok", "jitter_slo_ok")
                    if not point.get(flag)
                ]
                if failed_flags or point.get("bitrate_drop_alarm"):
                    violations.append(
                        {
                            "scope": "L1",
                            "node": node_id,
                            "second": point.get("second"),
                            "failed_flags": failed_flags,
                            "bitrate_drop_alarm": bool(point.get("bitrate_drop_alarm")),
                        }
                    )

        if level == "L2" and isinstance(attrs.get("tensor"), StateTensor):
            checked_l2_nodes += 1
            metrics = tensor_metrics(attrs["tensor"])
            if metrics.get("ram_load_percent", 0.0) > 80.0 or metrics.get("cpu_load_percent", 0.0) > 80.0:
                violations.append(
                    {
                        "scope": "L2",
                        "node": node_id,
                        "ram_load_percent": metrics.get("ram_load_percent"),
                        "cpu_load_percent": metrics.get("cpu_load_percent"),
                    }
                )

    for source, target, attrs in model.graph.edges(data=True):
        tensor = attrs.get("tensor")
        if not isinstance(tensor, StateTensor):
            continue

        checked_edges += 1
        metrics = tensor_metrics(tensor)
        if metrics.get("loss_probability", 0.0) > 0.005 or metrics.get("stability_margin", 0.0) <= 0.0:
            violations.append(
                {
                    "scope": "EDGE",
                    "edge": [source, target],
                    "loss_probability": metrics.get("loss_probability"),
                    "stability_margin": metrics.get("stability_margin"),
                }
            )

    return {
        "ok": not violations,
        "checked_l1_points": checked_l1_points,
        "checked_l2_nodes": checked_l2_nodes,
        "checked_edges": checked_edges,
        "violation_count": len(violations),
        "violations": violations[:50],
    }


def _snapshot(model: NetworkModel, config: DynamicsConfig, step_index: int, time_seconds: int) -> dict[str, Any]:
    tensor_state = _tensor_state_snapshot(model)
    arbitrator = build_arbitrator_view(model, tensor_state)
    snapshot = {
        "step_index": step_index,
        "time_seconds": time_seconds,
        "level_summary": model.level_summary,
        "tensor_state": _format_tensor_state(tensor_state, config.snapshot_detail),
        "state_vector": arbitrator["state_vector"],
        "arbitrator": arbitrator,
    }

    if config.snapshot_detail == "full":
        snapshot["nodes"] = [_node_snapshot(node_id, attrs) for node_id, attrs in model.graph.nodes(data=True)]
        snapshot["edges"] = [_edge_snapshot(source, target, attrs) for source, target, attrs in model.graph.edges(data=True)]

    if config.include_packet_simulation:
        snapshot["traffic"] = simulate_packet_snapshot(
            model,
            step_index=step_index,
            time_seconds=time_seconds,
            step_seconds=config.step_seconds,
            detail=config.packet_detail,
            packet_sample_limit=config.packet_sample_limit,
        )

    return snapshot


def _tensor_state_snapshot(model: NetworkModel) -> dict[str, Any]:
    """Collect every StateTensor into a per-level structure for analysis."""
    by_level: dict[str, list[dict[str, Any]]] = {level: [] for level in TENSOR_LEVELS}

    for node_id, attrs in model.graph.nodes(data=True):
        for tensor_name, tensor in _raw_tensor_attrs(attrs).items():
            by_level[tensor.level].append(
                {
                    "scope": "node",
                    "node_id": node_id,
                    "role": attrs.get("role"),
                    "tensor_name": tensor_name,
                    **_tensor_snapshot(tensor),
                }
            )

    for source, target, attrs in model.graph.edges(data=True):
        for tensor_name, tensor in _raw_tensor_attrs(attrs).items():
            by_level[tensor.level].append(
                {
                    "scope": "edge",
                    "source": source,
                    "target": target,
                    "medium": attrs.get("medium"),
                    "tensor_name": tensor_name,
                    **_tensor_snapshot(tensor),
                }
            )

    counts = {level: len(items) for level, items in by_level.items()}
    return {
        "levels": list(TENSOR_LEVELS),
        "counts": counts,
        "by_level": by_level,
    }


def _format_tensor_state(tensor_state: dict[str, Any], detail: SnapshotDetail) -> dict[str, Any]:
    """Trim tensor output for long dynamics runs when full detail is not needed."""
    if detail in {"full", "tensor"}:
        return tensor_state
    return {
        "levels": tensor_state["levels"],
        "counts": tensor_state["counts"],
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
        for name, value in _raw_tensor_attrs(attrs).items()
    }


def _raw_tensor_attrs(attrs: dict[str, Any]) -> dict[str, StateTensor]:
    return {
        name: value
        for name, value in attrs.items()
        if isinstance(value, StateTensor)
    }


def _tensor_snapshot(tensor: StateTensor) -> dict[str, Any]:
    values = tensor_metrics(tensor)
    return {
        "level": tensor.level,
        "metrics": values,
        "units": tensor.units,
        "vector": [float(value) for value in tensor.data.tolist()],
    }


def _validate_config(config: DynamicsConfig) -> None:
    if config.step_seconds <= 0:
        raise ValueError("DynamicsConfig.step_seconds must be positive")
    if config.step_count < 0:
        raise ValueError("DynamicsConfig.step_count must be non-negative")
    if config.packet_sample_limit < 0:
        raise ValueError("DynamicsConfig.packet_sample_limit must be non-negative")
    if config.snapshot_detail not in {"full", "tensor", "summary"}:
        raise ValueError("DynamicsConfig.snapshot_detail must be one of: full, tensor, summary")
    if config.packet_detail not in {"summary", "flows", "sample"}:
        raise ValueError("DynamicsConfig.packet_detail must be one of: summary, flows, sample")


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value
