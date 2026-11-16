"""Числовые гипотезы по показаниям датчиков текущего шага."""

from __future__ import annotations

import math

import numpy as np


# При нескольких совпадениях более специфичные признаки имеют приоритет.
# Числа заданы для стенда в единицах соответствующего показателя.
SIGNATURE_LIMITS = {
    "bgp_hijack": {"invalid_prefix_origin_count": 1},
    "privilege_escalation": {"unexpected_privilege_transition_count": 1},
    "network_discovery": {"distinct_probed_services_count": 4},
    "user_execution": {"user_launched_process_count": 1, "endpoint_cpu_rise_ratio": 0.15},
    "exfiltration": {"unusual_egress_bytes_per_second": 20_000, "established_channel_transfer_ratio": 0.65},
    "mobile_network_dos": {"mobile_endpoint_flag": 1, "mobile_link_failure_ratio": 0.20},
    "dhcp_flood": {"dhcp_service_pressure_ratio": 0.30},
    "dns_flood": {"dns_query_pressure_ratio": 0.30},
    "application_flood": {"application_work_pressure_ratio": 0.30},
    "brute_force": {"authentication_failure_rate_per_second": 0.025, "authentication_failure_ratio": 0.80},
    "power_attack": {"power_observation_ratio": 1.0},
    "syn_flood": {"syn_backlog_growth_ratio": 0.18},
    "ddos": {"suspicious_source_count": 2, "source_entropy_ratio": 0.05,
             "traffic_acceleration_ratio": 0.15, "observed_reflection_ratio": 0.20},
    "dos": {"traffic_acceleration_ratio": 0.11, "target_concentration_ratio": 0.22},
}
SIGNATURE_COLUMNS = tuple(dict.fromkeys(name for row in SIGNATURE_LIMITS.values() for name in row))
SIGNATURE_MATRIX = np.array([[row.get(name, 0.0) for name in SIGNATURE_COLUMNS]
                             for row in SIGNATURE_LIMITS.values()])
SIGNATURE_MASK = SIGNATURE_MATRIX > 0.0
SIGNATURE_SCALE = np.where(SIGNATURE_MASK, SIGNATURE_MATRIX, 1.0)

# Альтернативы внутри группы объединяются максимумом, совместные признаки — минимумом.
FEATURE_GROUPS = {
    "traffic_burst": ({"traffic_acceleration_ratio": 0.70, "target_concentration_ratio": 0.90},),
    "distributed_sources": ({"observed_ingress_count": 4, "source_entropy_ratio": 0.35},),
    "reflection": ({"observed_reflection_ratio": 1.0},),
    "syn_backlog": ({"syn_backlog_growth_ratio": 0.90},),
    "service_pressure": ({"dns_query_pressure_ratio": 0.85}, {"dhcp_service_pressure_ratio": 0.85}),
    "application_pressure": ({"application_work_pressure_ratio": 0.85},),
    "service_discovery": ({"distinct_probed_services_count": 12},),
    "user_execution": ({"user_launched_process_count": 2, "endpoint_cpu_rise_ratio": 0.45},),
    "privilege_transition": ({"unexpected_privilege_transition_count": 2},),
    "egress": ({"unusual_egress_bytes_per_second": 75_000, "established_channel_transfer_ratio": 0.95},),
    "mobile_connectivity": ({"mobile_endpoint_flag": 1, "mobile_link_failure_ratio": 0.60},),
}
FEATURE_WEIGHTS = {
    "T1498.001": {"traffic_burst": 1.0},
    "T1498.002": {"reflection": 1.0, "distributed_sources": 0.65, "traffic_burst": 0.45},
    "T1499.001": {"syn_backlog": 1.0, "traffic_burst": 0.25},
    "T1499.002": {"service_pressure": 1.0},
    "T1499.003": {"application_pressure": 1.0, "traffic_burst": 0.20},
    "T1046": {"service_discovery": 1.0},
    "T1204": {"user_execution": 1.0, "application_pressure": 0.20},
    "T1068": {"privilege_transition": 1.0, "user_execution": 0.20},
    "T1041": {"egress": 1.0, "traffic_burst": 0.20},
    "T1464": {"mobile_connectivity": 1.0},
}
TECHNIQUE_BY_KIND = {
    "dos": "T1498.001", "ddos": "T1498.002", "syn_flood": "T1499.001",
    "dns_flood": "T1499.002", "dhcp_flood": "T1499.002", "application_flood": "T1499.003",
    "network_discovery": "T1046", "user_execution": "T1204", "privilege_escalation": "T1068",
    "exfiltration": "T1041", "mobile_network_dos": "T1464",
}
LAG_WEIGHTS = (1.0, 0.85, 0.70, 0.55, 0.40, 0.30, 0.20, 0.10)
WEIGHTED_MASKS = {technique: np.outer([features.get(name, 0.0) for name in FEATURE_GROUPS], LAG_WEIGHTS)
                  for technique, features in FEATURE_WEIGHTS.items()}
FLOOD_MECHANISMS = {
    "dos": ("udp_flood", "UDP flood"), "ddos": ("reflection_amplification", "отражённое усиление"),
    "syn_flood": ("syn_flood", "SYN flood"), "dhcp_flood": ("dhcp_flood", "DHCP flood"),
    "dns_flood": ("dns_flood", "DNS flood"), "application_flood": ("application_flood", "перегрузка приложения"),
    "mobile_network_dos": ("mobile_connectivity", "нарушение мобильной связи"),
}


def flood_fields(kind: str, ingress_count: float) -> dict:
    mechanism = FLOOD_MECHANISMS.get(kind)
    if mechanism is None:
        return {}
    family = "ddos" if ingress_count >= 2 else "dos" if ingress_count == 1 else None
    label = {"dos": "DoS", "ddos": "DDoS"}.get(family, "Отказ в обслуживании")
    return {"attack_family": family, "attack_mechanism": mechanism[0],
            "attack_label_ru": f"{label} ({mechanism[1]})"}


def _signal(signals: dict, name: str) -> float:
    try:
        value = float(signals.get(name, 0.0))
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _feature_vector(signals: dict) -> list[float]:
    return [max(min(max(0.0, min(1.0, _signal(signals, name) / scale)) for name, scale in group.items())
                for group in alternatives) for alternatives in FEATURE_GROUPS.values()]


def _weighted_scores(techniques: set[str], signals: dict, history: list[dict | None]) -> dict[str, float]:
    samples = [signals, *history[:len(LAG_WEIGHTS) - 1]]
    available = [index for index, sample in enumerate(samples) if sample is not None]
    observation = np.array([_feature_vector(samples[index]) for index in available]).T
    scores = {}
    for technique in sorted(techniques):
        mask = WEIGHTED_MASKS[technique][:, available]
        scores[technique] = float(np.sum(mask * observation) / np.sum(mask))
    return scores


def signature_catalog() -> dict:
    return {
        "value_origin": "gnet9_scenario_assumption",
        "row_kinds": list(SIGNATURE_LIMITS),
        "columns": list(SIGNATURE_COLUMNS),
        "column_units": [
            {"unusual_egress_bytes_per_second": "B/s", "authentication_failure_rate_per_second": "1/s",
             "mobile_endpoint_flag": "0/1"}.get(name, "count" if name.endswith("_count") else "ratio")
            for name in SIGNATURE_COLUMNS
        ],
        "minimum_values": SIGNATURE_MATRIX.tolist(),
        "hadamard_mask": SIGNATURE_MASK.astype(int).tolist(),
        "match_semantics": "minimum_observed_to_threshold_ratio_over_required_columns_at_least_one",
        "weighted_masks": {
            "rows": list(FEATURE_GROUPS), "step_lags": list(range(len(LAG_WEIGHTS))),
            "normalization_by_row": FEATURE_GROUPS, "lag_weights": LAG_WEIGHTS,
            "techniques": {technique: mask.tolist() for technique, mask in WEIGHTED_MASKS.items()},
            "score_semantics": "weighted_mean_of_normalized_sensor_evidence_over_observed_steps",
            "value_origin": "gnet9_engineering_hypothesis_without_external_validation",
        },
    }


def match_attack_signature(signals: dict, *, history: list[dict | None] | None = None) -> dict | None:
    """Сопоставить измерения с рядами матрицы без метки сценария.

    Строка совпадает, когда все её активные координаты достигают минимума.
    Например, один запуск при росте CPU на 0,30 даёт min(1, 2) = 1.
    Отсутствующие или неконечные значения не удовлетворяют порогу.
    """
    values = np.array([_signal(signals, name) for name in SIGNATURE_COLUMNS])
    power_control = _signal(signals, "power_control_anomaly_ratio")
    voltage_sag = _signal(signals, "power_voltage_sag_ratio")
    values[SIGNATURE_COLUMNS.index("power_observation_ratio")] = max(
        power_control / 0.18, voltage_sag / 0.02,
    )
    masked = (values / SIGNATURE_SCALE) * SIGNATURE_MASK
    margins = np.min(np.where(SIGNATURE_MASK, masked, np.inf), axis=1)
    matches = np.flatnonzero(margins >= 1.0)
    if matches.size == 0:
        return None
    kinds = tuple(SIGNATURE_LIMITS)
    candidates = {kinds[index]: float(margins[index]) for index in matches}
    techniques = {TECHNIQUE_BY_KIND[kind] for kind in candidates if kind in TECHNIQUE_BY_KIND}
    scores = _weighted_scores(techniques, signals, history or []) if techniques else {}
    ordered = sorted(candidates, key=lambda kind: (
        2 if kind not in {"dos", "ddos"} else 1 if kind == "ddos" else 0,
        scores.get(TECHNIQUE_BY_KIND.get(kind), 1.0), candidates[kind],
    ), reverse=True)
    kind = ordered[0]
    technique = TECHNIQUE_BY_KIND.get(kind)
    return {
        "kind": kind, "margin_ratio": round(candidates[kind], 6), "matching_kinds": ordered,
        "weighted_technique_id": technique,
        "weighted_score": round(scores[technique], 6) if technique else None,
        "weighted_technique_scores": {technique: round(score, 6) for technique, score in scores.items()},
        **flood_fields(kind, _signal(signals, "observed_ingress_count")),
        "observed_values": {name: float(value) for name, value in zip(SIGNATURE_COLUMNS, values)
                            if name in SIGNATURE_LIMITS[kind]},
    }
