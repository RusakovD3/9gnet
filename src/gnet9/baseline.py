"""Human-readable ideal baseline state for G-Net tensors.

This module is the single place where the t0 "healthy network" assumptions are
initialized. The topology builder should describe graph shape; this file should
describe tensor values.
"""

from __future__ import annotations

import numpy as np

from .constants import IDEAL_LOAD_FACTOR
from .l1_d0sl import D0SLSubscriberPolicy, SlaGrade, TrafficKind
from .models import ServiceProfile


# Stable numeric category codes used by tensor vectors.
SERVICE_CODES = {"Voice": 1.0, "Video": 2.0, "FTP": 3.0, "Telemetry": 4.0}
TRAFFIC_TO_SERVICE_CODE = {
    TrafficKind.BROADCAST_MP3: SERVICE_CODES["Video"],
    TrafficKind.FTP: SERVICE_CODES["FTP"],
    TrafficKind.DNS: SERVICE_CODES["Telemetry"],
}
ACCESS_TYPE_CODES = {"fixed": 0.0, "mobile": 1.0}
PLACEMENT_ROLE_CODES = {
    "terrain-anchor": 0.0,
    "mobile-subscriber": 1.0,
    "fixed-subscriber": 2.0,
    "aggregation-router": 3.0,
    "core-router": 4.0,
    "arbitrator": 7.0,
}


# L0 tensors are service intent: requested traffic and quality target.
SERVICE_HEALTH = 0.97
SERVICE_PRIORITY_CODES = {"gold": 1.0, "silver": 0.6, "bronze": 0.3}


# L1 tensors are deliberately not derived from random monitoring. They represent
# ideal, realistic subscriber-side operating points for each access/SLA class.
L1_ACCESS_GRADE_BASELINE = {
    ("mobile", SlaGrade.GOLD): {
        "traffic_intensity_rho": 0.31,
        "traffic_distribution_cv": 0.045,
        "processing_speed_mbps": 300.0,
        "capex_opex_cost": 0.42,
        "sla_margin": 0.88,
    },
    ("mobile", SlaGrade.SILVER): {
        "traffic_intensity_rho": 0.42,
        "traffic_distribution_cv": 0.060,
        "processing_speed_mbps": 150.0,
        "capex_opex_cost": 0.30,
        "sla_margin": 0.78,
    },
    ("mobile", SlaGrade.BRONZE): {
        "traffic_intensity_rho": 0.53,
        "traffic_distribution_cv": 0.085,
        "processing_speed_mbps": 75.0,
        "capex_opex_cost": 0.18,
        "sla_margin": 0.68,
    },
    ("fixed", SlaGrade.GOLD): {
        "traffic_intensity_rho": 0.24,
        "traffic_distribution_cv": 0.030,
        "processing_speed_mbps": 1000.0,
        "capex_opex_cost": 0.34,
        "sla_margin": 0.92,
    },
    ("fixed", SlaGrade.SILVER): {
        "traffic_intensity_rho": 0.32,
        "traffic_distribution_cv": 0.040,
        "processing_speed_mbps": 500.0,
        "capex_opex_cost": 0.22,
        "sla_margin": 0.84,
    },
    ("fixed", SlaGrade.BRONZE): {
        "traffic_intensity_rho": 0.43,
        "traffic_distribution_cv": 0.060,
        "processing_speed_mbps": 100.0,
        "capex_opex_cost": 0.12,
        "sla_margin": 0.74,
    },
}

L1_PROCESSING_DELAY_MS = {
    ("mobile", TrafficKind.BROADCAST_MP3): 24.0,
    ("mobile", TrafficKind.FTP): 55.0,
    ("mobile", TrafficKind.DNS): 12.0,
    ("fixed", TrafficKind.BROADCAST_MP3): 10.0,
    ("fixed", TrafficKind.FTP): 22.0,
    ("fixed", TrafficKind.DNS): 3.0,
}


# L2 tensors are explicit healthy operating points for active equipment.
L2_EQUIPMENT_BASELINE = {
    "core-router": {
        "ram_used_gb": 12.8,
        "ram_load_percent": 40.0,
        "cpu_load_percent": 24.0,
        "traffic_distribution_code": 0.72,
        "capex_opex_cost": 0.90,
        "stability_margin": 0.70,
    },
    "aggregation-router": {
        "ram_used_gb": 5.44,
        "ram_load_percent": 34.0,
        "cpu_load_percent": 18.0,
        "traffic_distribution_code": 0.58,
        "capex_opex_cost": 0.64,
        "stability_margin": 0.78,
    },
}

L2_GOLD_CPU_LOAD_PERCENT = 26.0


# L3/L4/EDGE values combine explicit medium assumptions with link geometry.
MEDIUM_BASELINE = {
    "logical-service-binding": {
        "medium_code": 0.0,
        "frequency_mhz": 0.0,
        "noise_interference_db": 0.0,
        "duct_capacity_used_ratio": 0.0,
        "repair_time_hours": 0.0,
        "loss_probability": 0.0,
        "attack_exposure": 0.08,
        "cross_connect_present": 0.0,
    },
    "fiber": {
        "medium_code": 1.0,
        "frequency_mhz": 0.0,
        "noise_interference_db": 2.0,
        "duct_capacity_used_ratio": 0.55,
        "repair_time_hours": 6.0,
        "loss_probability": 0.00001,
        "attack_exposure": 0.12,
        "cross_connect_present": 1.0,
    },
    "ethernet": {
        "medium_code": 2.0,
        "frequency_mhz": 0.0,
        "noise_interference_db": 8.0,
        "duct_capacity_used_ratio": 0.35,
        "repair_time_hours": 2.0,
        "loss_probability": 0.00010,
        "attack_exposure": 0.20,
        "cross_connect_present": 1.0,
    },
    "radio": {
        "medium_code": 4.0,
        "frequency_mhz": 2400.0,
        "noise_interference_db": 22.0,
        "duct_capacity_used_ratio": 0.0,
        "repair_time_hours": 0.5,
        "loss_probability": 0.00150,
        "attack_exposure": 0.30,
        "cross_connect_present": 0.0,
    },
    "radio-backhaul": {
        "medium_code": 4.5,
        "frequency_mhz": 2400.0,
        "noise_interference_db": 18.0,
        "duct_capacity_used_ratio": 0.0,
        "repair_time_hours": 0.5,
        "loss_probability": 0.00100,
        "attack_exposure": 0.26,
        "cross_connect_present": 0.0,
    },
}


# L5/L6 tensors are role templates. They are intentionally plain assignments.
L5_BY_ROLE = {
    "core-router": {
        "protocol_code": 3.0,
        "socket_binding_present": 1.0,
        "routing_mode_code": 1.0,
        "remap_algorithm_code": 2.0,
        "percolation_threshold": 0.72,
        "reconfiguration_time_s": 12.0,
    },
    "aggregation-router": {
        "protocol_code": 2.0,
        "socket_binding_present": 1.0,
        "routing_mode_code": 1.0,
        "remap_algorithm_code": 1.0,
        "percolation_threshold": 0.64,
        "reconfiguration_time_s": 20.0,
    },
}

L6_BY_ROLE = {
    "core-router": {
        "power_supply_code": 3.0,
        "nominal_power_kw": 0.85,
        "backup_autonomy_hours": 4.0,
        "energy_reserve_ratio": 0.85,
        "capex_opex_cost": 0.92,
    },
    "aggregation-router": {
        "power_supply_code": 2.0,
        "nominal_power_kw": 0.45,
        "backup_autonomy_hours": 2.0,
        "energy_reserve_ratio": 0.82,
        "capex_opex_cost": 0.66,
    },
    "mobile-subscriber": {
        "power_supply_code": 1.0,
        "nominal_power_kw": 0.005,
        "backup_autonomy_hours": 8.0,
        "energy_reserve_ratio": 0.70,
        "capex_opex_cost": 0.18,
    },
    "fixed-subscriber": {
        "power_supply_code": 2.0,
        "nominal_power_kw": 0.03,
        "backup_autonomy_hours": 0.5,
        "energy_reserve_ratio": 0.60,
        "capex_opex_cost": 0.12,
    },
}


# L7 is a healthy no-remap decision vector. Future experiments can mutate these
# values when attacks, overload or physical relocation are introduced.
L7_ARBITRATOR_BASELINE = {
    "hausdorff_distance": 0.0,
    "lyapunov_value": 0.12,
    "lyapunov_delta": -0.04,
    "koopman_residual": 0.02,
    "remap_pressure": 0.0,
    "decision_confidence": 0.91,
    "action_cost": 0.18,
}


def l0_service_tensor(profile: ServiceProfile) -> dict[str, float]:
    return {
        "service_code": SERVICE_CODES[profile.name],
        "bitrate_mbps": profile.bitrate_mbps,
        "latency_budget_ms": profile.latency_ms_max,
        "jitter_budget_ms": profile.jitter_ms_max,
        "availability_target": profile.availability_target,
        "priority_code": SERVICE_PRIORITY_CODES[profile.priority],
        "demand_pressure": IDEAL_LOAD_FACTOR,
        "service_health": SERVICE_HEALTH,
    }


def l1_subscriber_tensor(policy: D0SLSubscriberPolicy, access_kind: str) -> dict[str, float]:
    baseline = L1_ACCESS_GRADE_BASELINE[(access_kind, policy.grade)]
    request_rate_pps = _bitrate_to_pps(policy.target_bitrate_kbps)
    packet_loss_ratio = policy.packet_loss_budget_percent / 100.0 * 0.10
    return {
        "access_type_code": ACCESS_TYPE_CODES[access_kind],
        "service_code": TRAFFIC_TO_SERVICE_CODE[policy.traffic],
        "request_rate_pps": request_rate_pps,
        "response_rate_pps": request_rate_pps * (1.0 - packet_loss_ratio),
        "traffic_intensity_rho": baseline["traffic_intensity_rho"],
        "traffic_distribution_cv": baseline["traffic_distribution_cv"],
        "processing_speed_mbps": baseline["processing_speed_mbps"],
        "processing_delay_ms": L1_PROCESSING_DELAY_MS[(access_kind, policy.traffic)],
        "capex_opex_cost": baseline["capex_opex_cost"],
        "sla_margin": baseline["sla_margin"],
    }


def l2_equipment_tensor(role: str, criticality: str, port_speed_mbps: float, port_delay_ms: float) -> dict[str, float]:
    baseline = dict(L2_EQUIPMENT_BASELINE[role])
    if role == "core-router" and criticality == "gold":
        baseline["cpu_load_percent"] = L2_GOLD_CPU_LOAD_PERCENT

    return {
        **baseline,
        "packet_processing_time_ms": _packet_processing_time_ms(port_speed_mbps),
        "port_delay_ms": port_delay_ms,
        "port_speed_mbps": port_speed_mbps,
    }


def l3_medium_tensor(
    medium: str,
    capacity_mbps: float,
    source_pos: tuple[float, float],
    target_pos: tuple[float, float],
) -> dict[str, float]:
    baseline = MEDIUM_BASELINE[medium]
    link_distance_m = distance_m(source_pos, target_pos)
    link_attenuation_db = attenuation_db(medium, link_distance_m, baseline["frequency_mhz"])
    snr_db = max(0.0, 100.0 - link_attenuation_db - baseline["noise_interference_db"])
    return {
        "medium_code": baseline["medium_code"],
        "line_rate_mbps": capacity_mbps,
        "distance_m": link_distance_m,
        "frequency_mhz": baseline["frequency_mhz"],
        "attenuation_db": link_attenuation_db,
        "noise_interference_db": baseline["noise_interference_db"],
        "snr_db": snr_db,
    }


def l4_infrastructure_tensor(
    medium: str,
    source_pos: tuple[float, float],
    target_pos: tuple[float, float],
) -> dict[str, float]:
    baseline = MEDIUM_BASELINE[medium]
    source = np.array(source_pos, dtype=float)
    target = np.array(target_pos, dtype=float)
    midpoint = (source + target) / 2.0
    return {
        "x_mid": float(midpoint[0]),
        "y_mid": float(midpoint[1]),
        "length_m": distance_m(source_pos, target_pos),
        "cross_connect_present": baseline["cross_connect_present"],
        "duct_capacity_used_ratio": baseline["duct_capacity_used_ratio"],
        "repair_time_hours": baseline["repair_time_hours"],
    }


def l5_role_tensor(role: str) -> dict[str, float]:
    return L5_BY_ROLE["core-router" if role == "core-router" else "aggregation-router"]


def l6_power_tensor(role: str) -> dict[str, float]:
    return L6_BY_ROLE[role]


def l7_arbitrator_tensor() -> dict[str, float]:
    return L7_ARBITRATOR_BASELINE


def l8_placement_tensor(pos: tuple[float, float], role: str) -> dict[str, float]:
    x, y = pos
    return {
        "x": float(x),
        "y": float(y),
        "coordinate_norm": float(np.linalg.norm([x, y])),
        "placement_role_code": PLACEMENT_ROLE_CODES.get(role, 9.0),
        "terrain_risk": 0.15 if role == "terrain-anchor" else 0.08,
    }


def edge_transport_tensor(medium: str, capacity_mbps: float, latency_ms: float, redundancy: float) -> dict[str, float]:
    baseline = MEDIUM_BASELINE[medium]
    return {
        "capacity_mbps": capacity_mbps,
        "latency_ms": latency_ms,
        "loss_probability": baseline["loss_probability"],
        "redundancy": redundancy,
        "utilization": IDEAL_LOAD_FACTOR,
        "stability_margin": 1.0 - IDEAL_LOAD_FACTOR,
        "attack_exposure": baseline["attack_exposure"],
    }


def distance_m(source_pos: tuple[float, float], target_pos: tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(source_pos, dtype=float) - np.array(target_pos, dtype=float)) * 100.0)


def attenuation_db(medium: str, distance_meters: float, frequency_mhz: float) -> float:
    if medium in {"radio", "radio-backhaul"}:
        distance_km = max(distance_meters / 1000.0, 0.001)
        return float(32.44 + 20.0 * np.log10(distance_km) + 20.0 * np.log10(max(frequency_mhz, 1.0)))
    if medium == "fiber":
        return float(0.35 * distance_meters / 1000.0)
    if medium == "ethernet":
        return float(6.0 * distance_meters / 100.0)
    return 0.0


def _bitrate_to_pps(bitrate_kbps: float, packet_size_bytes: int = 1200) -> float:
    return max(0.1, bitrate_kbps * 1000.0 / (packet_size_bytes * 8.0))


def _packet_processing_time_ms(port_speed_mbps: float, mtu_bytes: int = 1500) -> float:
    return mtu_bytes * 8.0 / max(port_speed_mbps * 1_000_000.0, 1e-9) * 1000.0
