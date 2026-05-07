"""Tensor templates for G-Net nodes and edges.

Every tensor has the same compact shape: 2x2x2x2x2. The project does not try to
store every possible metric in every cell. Instead, each important metric is
placed into a named semantic slot. This keeps the model small and explainable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import TENSOR_SHAPE
from .models import Tensor5


@dataclass(frozen=True)
class TensorTemplate:
    """Description of tensor axes and metric coordinates for one layer."""

    level: str
    axis_names: tuple[str, str, str, str, str]
    metric_slots: dict[str, tuple[int, int, int, int, int]]


TEMPLATES = {
    "L0": TensorTemplate(
        "L0",
        ("service", "quality", "business", "demand", "time"),
        {
            "sla_attainment": (0, 0, 0, 0, 0),
            "slo_margin": (0, 0, 0, 0, 1),
            "demand_pressure": (0, 0, 0, 1, 0),
            "business_value": (0, 1, 0, 0, 0),
            "service_health": (1, 0, 0, 0, 0),
        },
    ),
    "L1": TensorTemplate(
        "L1",
        ("sla", "slo", "sli", "queue", "time"),
        {
            "sla_grade": (0, 0, 0, 0, 0),
            "slo_bitrate_ratio": (0, 0, 0, 0, 1),
            "sli_latency_ratio": (0, 0, 0, 1, 0),
            "sli_loss_ratio": (0, 1, 0, 0, 0),
            "queue_stability": (1, 0, 0, 0, 0),
            "sli_jitter_ratio": (1, 1, 0, 0, 0),
            "traffic_class": (1, 0, 1, 0, 0),
            "monitoring_period": (1, 0, 0, 1, 0),
            "bitrate_drop_alarm": (1, 0, 0, 0, 1),
        },
    ),
    "L2": TensorTemplate(
        "L2",
        ("device", "performance", "security", "resilience", "time"),
        {
            "vitality_V": (0, 0, 0, 0, 0),
            "immunity_I": (0, 0, 0, 0, 1),
            "damage_D": (0, 0, 0, 1, 0),
            "regeneration_R": (0, 1, 0, 0, 0),
            "attack_surface": (1, 0, 0, 0, 0),
        },
    ),
    "L3": TensorTemplate(
        "L3",
        ("channel", "signal", "errors", "forecast", "time"),
        {
            "snr": (0, 0, 0, 0, 0),
            "ber_inverse": (0, 0, 0, 0, 1),
            "interference_margin": (0, 0, 0, 1, 0),
            "koopman_residual_inverse": (0, 1, 0, 0, 0),
            "path_stability": (1, 0, 0, 0, 0),
        },
    ),
    "L4": TensorTemplate(
        "L4",
        ("physical", "redundancy", "repair", "risk", "time"),
        {
            "redundancy": (0, 0, 0, 0, 0),
            "repairability": (0, 0, 0, 0, 1),
            "link_health": (0, 0, 0, 1, 0),
            "geo_exposure_inverse": (0, 1, 0, 0, 0),
            "physical_margin": (1, 0, 0, 0, 0),
        },
    ),
    "L5": TensorTemplate(
        "L5",
        ("slice", "centrality", "capacity", "remap", "time"),
        {
            "slice_isolation": (0, 0, 0, 0, 0),
            "centrality": (0, 0, 0, 0, 1),
            "congestion_headroom": (0, 0, 0, 1, 0),
            "remap_readiness": (0, 1, 0, 0, 0),
            "grade": (1, 0, 0, 0, 0),
        },
    ),
    "L6": TensorTemplate(
        "L6",
        ("energy", "backup", "health", "tau", "time"),
        {
            "energy_remaining": (0, 0, 0, 0, 0),
            "backup_margin": (0, 0, 0, 0, 1),
            "power_health": (0, 0, 0, 1, 0),
            "tau_normalized": (0, 1, 0, 0, 0),
            "infra_risk_inverse": (1, 0, 0, 0, 0),
        },
    ),
    "L7": TensorTemplate(
        "L7",
        ("decision", "hausdorff", "game", "cost", "time"),
        {
            "decision_confidence": (0, 0, 0, 0, 0),
            "hausdorff_margin_inverse": (0, 0, 0, 0, 1),
            "game_value": (0, 0, 0, 1, 0),
            "action_cost_inverse": (0, 1, 0, 0, 0),
            "policy_stability": (1, 0, 0, 0, 0),
        },
    ),
    "L8": TensorTemplate(
        "L8",
        ("terrain", "geometry", "los", "placement", "time"),
        {
            "terrain_quality": (0, 0, 0, 0, 0),
            "distance_efficiency": (0, 0, 0, 0, 1),
            "los_margin": (0, 0, 0, 1, 0),
            "placement_fitness": (0, 1, 0, 0, 0),
            "topo_risk_inverse": (1, 0, 0, 0, 0),
        },
    ),
}

EDGE_TEMPLATE = TensorTemplate(
    "EDGE",
    ("transport", "performance", "security", "resilience", "time"),
    {
        "capacity_headroom": (0, 0, 0, 0, 0),
        "latency_inverse": (0, 0, 0, 0, 1),
        "loss_inverse": (0, 0, 0, 1, 0),
        "redundancy": (0, 1, 0, 0, 0),
        "attack_exposure_inverse": (1, 0, 0, 0, 0),
    },
)


def build_tensor(level: str, metrics: dict[str, float]) -> Tensor5:
    """Create a node tensor for a given G-Net level.

    Missing metrics are written as zero. This is intentional: a zero means that
    the metric was not provided for this particular object.
    """
    return _build_from_template(TEMPLATES[level], metrics)


def build_edge_tensor(metrics: dict[str, float]) -> Tensor5:
    """Create a tensor for an edge/communication channel."""
    return _build_from_template(EDGE_TEMPLATE, metrics)


def _build_from_template(template: TensorTemplate, metrics: dict[str, float]) -> Tensor5:
    data = np.zeros(TENSOR_SHAPE, dtype=float)
    for metric_name, slot in template.metric_slots.items():
        data[slot] = float(metrics.get(metric_name, 0.0))

    return Tensor5(
        level=template.level,
        axis_names=template.axis_names,
        metric_names=tuple(template.metric_slots.keys()),
        data=data,
        metric_slots=template.metric_slots,
    )
