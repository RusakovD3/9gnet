"""Layer tensor specifications for the G-Net baseline.

The current model uses simple numeric state vectors. This is deliberate:
Koopman/DMD, Lyapunov checks and distance calculations need stable numeric
features more than sparse high-order arrays with many empty cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import StateTensor


@dataclass(frozen=True)
class StateTensorSpec:
    """Named metrics, units and purpose for one layer tensor."""

    level: str
    metric_names: tuple[str, ...]
    units: dict[str, str]
    description: str


STATE_TENSOR_SPECS: dict[str, StateTensorSpec] = {
    "L0": StateTensorSpec(
        "L0",
        (
            "service_code",
            "bitrate_mbps",
            "latency_budget_ms",
            "jitter_budget_ms",
            "availability_target",
            "priority_code",
            "demand_pressure",
            "service_health",
        ),
        {
            "bitrate_mbps": "Mbps",
            "latency_budget_ms": "ms",
            "jitter_budget_ms": "ms",
            "availability_target": "ratio",
            "demand_pressure": "ratio",
            "service_health": "ratio",
        },
        "L0 service demand and quality target vector.",
    ),
    "L1": StateTensorSpec(
        "L1",
        (
            "access_type_code",
            "service_code",
            "request_rate_pps",
            "response_rate_pps",
            "traffic_intensity_rho",
            "traffic_distribution_cv",
            "processing_speed_mbps",
            "processing_delay_ms",
            "capex_opex_cost",
            "sla_margin",
        ),
        {
            "request_rate_pps": "packets/s",
            "response_rate_pps": "packets/s",
            "traffic_intensity_rho": "ratio",
            "traffic_distribution_cv": "ratio",
            "processing_speed_mbps": "Mbps",
            "processing_delay_ms": "ms",
            "capex_opex_cost": "normalized",
            "sla_margin": "ratio",
        },
        "L1 subscriber request, service, processing, queue and cost state.",
    ),
    "L2": StateTensorSpec(
        "L2",
        (
            "ram_used_gb",
            "ram_load_percent",
            "cpu_load_percent",
            "packet_processing_time_ms",
            "traffic_distribution_code",
            "port_delay_ms",
            "port_speed_mbps",
            "capex_opex_cost",
            "stability_margin",
        ),
        {
            "ram_used_gb": "GB",
            "ram_load_percent": "%",
            "cpu_load_percent": "%",
            "packet_processing_time_ms": "ms",
            "port_delay_ms": "ms",
            "port_speed_mbps": "Mbps",
            "capex_opex_cost": "normalized",
            "stability_margin": "ratio",
        },
        "L2 equipment load, packet processing, port and stability state.",
    ),
    "L3": StateTensorSpec(
        "L3",
        (
            "medium_code",
            "line_rate_mbps",
            "distance_m",
            "frequency_mhz",
            "attenuation_db",
            "noise_interference_db",
            "snr_db",
        ),
        {
            "line_rate_mbps": "Mbps",
            "distance_m": "m",
            "frequency_mhz": "MHz",
            "attenuation_db": "dB",
            "noise_interference_db": "dB",
            "snr_db": "dB",
        },
        "L3 transmission medium, attenuation and noise state.",
    ),
    "L4": StateTensorSpec(
        "L4",
        (
            "x_mid",
            "y_mid",
            "length_m",
            "cross_connect_present",
            "duct_capacity_used_ratio",
            "repair_time_hours",
        ),
        {
            "x_mid": "model-coordinate",
            "y_mid": "model-coordinate",
            "length_m": "m",
            "cross_connect_present": "boolean",
            "duct_capacity_used_ratio": "ratio",
            "repair_time_hours": "hours",
        },
        "L4 cable duct, spatial placement and repairability state.",
    ),
    "L5": StateTensorSpec(
        "L5",
        (
            "protocol_code",
            "socket_binding_present",
            "routing_mode_code",
            "remap_algorithm_code",
            "percolation_threshold",
            "reconfiguration_time_s",
        ),
        {
            "socket_binding_present": "boolean",
            "percolation_threshold": "ratio",
            "reconfiguration_time_s": "s",
        },
        "L5 protocol, socket identity, routing and remapping state.",
    ),
    "L6": StateTensorSpec(
        "L6",
        (
            "power_supply_code",
            "nominal_power_kw",
            "backup_autonomy_hours",
            "energy_reserve_ratio",
            "capex_opex_cost",
        ),
        {
            "nominal_power_kw": "kW",
            "backup_autonomy_hours": "hours",
            "energy_reserve_ratio": "ratio",
            "capex_opex_cost": "normalized",
        },
        "L6 power supply and energy cost state.",
    ),
    "L7": StateTensorSpec(
        "L7",
        (
            "hausdorff_distance",
            "lyapunov_value",
            "lyapunov_delta",
            "koopman_residual",
            "remap_pressure",
            "decision_confidence",
            "action_cost",
        ),
        {
            "hausdorff_distance": "model-coordinate",
            "lyapunov_value": "normalized",
            "lyapunov_delta": "normalized",
            "koopman_residual": "normalized",
            "remap_pressure": "ratio",
            "decision_confidence": "ratio",
            "action_cost": "normalized",
        },
        "L7 arbitrator vector for Koopman, Lyapunov and remapping decisions.",
    ),
    "L8": StateTensorSpec(
        "L8",
        (
            "x",
            "y",
            "coordinate_norm",
            "placement_role_code",
            "terrain_risk",
        ),
        {
            "x": "model-coordinate",
            "y": "model-coordinate",
            "coordinate_norm": "model-coordinate",
            "terrain_risk": "ratio",
        },
        "L8 placement coordinates for Hausdorff/topology distance calculations.",
    ),
    "EDGE": StateTensorSpec(
        "EDGE",
        (
            "capacity_mbps",
            "latency_ms",
            "loss_probability",
            "redundancy",
            "utilization",
            "stability_margin",
            "attack_exposure",
        ),
        {
            "capacity_mbps": "Mbps",
            "latency_ms": "ms",
            "loss_probability": "ratio",
            "redundancy": "ratio",
            "utilization": "ratio",
            "stability_margin": "ratio",
            "attack_exposure": "ratio",
        },
        "Transport edge state for network dynamics and stability calculations.",
    ),
}


def build_layer_tensor(level: str, metrics: dict[str, float]) -> StateTensor:
    """Build a layer state vector from named metrics."""
    return _build_from_spec(STATE_TENSOR_SPECS[level], metrics)


def build_transport_tensor(metrics: dict[str, float]) -> StateTensor:
    """Build the common transport edge state vector."""
    return _build_from_spec(STATE_TENSOR_SPECS["EDGE"], metrics)


def _build_from_spec(spec: StateTensorSpec, metrics: dict[str, float]) -> StateTensor:
    unknown = sorted(set(metrics) - set(spec.metric_names))
    if unknown:
        raise ValueError(f"Unknown {spec.level} tensor metrics: {', '.join(unknown)}")

    data = np.array([float(metrics.get(name, 0.0)) for name in spec.metric_names], dtype=float)
    return StateTensor(
        level=spec.level,
        axes=("metric",),
        metric_names=spec.metric_names,
        metric_index={name: (index,) for index, name in enumerate(spec.metric_names)},
        units=spec.units,
        description=spec.description,
        data=data,
    )
