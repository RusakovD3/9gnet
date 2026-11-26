"""Числовые гипотезы по показаниям датчиков текущего шага."""

from __future__ import annotations

import math

import numpy as np

from .constants import DYNAMICS_STEP_SECONDS


# Ориентиры для конкретных наблюдений, с единицами и источниками в каталоге.
MITRE_T1110_001_APT28_PROCEDURE_URL = "https://attack.mitre.org/techniques/T1110/001/"
MITRE_T1110_001_APT28_MIN_ATTEMPTS_PER_HOUR = 300
MITRE_T1110_001_ATTEMPTS_PER_SECOND = MITRE_T1110_001_APT28_MIN_ATTEMPTS_PER_HOUR / 3600
MITRE_AN0969_PACKET_RATE_PPS = 10_000
MITRE_AN0988_VOLUME_BYTES = 1_000_000
MITRE_AN0988_WINDOW_SECONDS = 300
# Длительность окна в секундах, стратегия и аналитика с примером настройки.
MITRE_OBSERVATION_WINDOWS = {
    "T1046": (60, "DET0376", "AN1057"),
    "T1499.001": (30, "DET0356", "AN1012"),
    "T1110.001": (600, "DET0551", "AN1521"),
}

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
    "authentication": ({"inferred_attempt_rate_per_account_per_second": MITRE_T1110_001_ATTEMPTS_PER_SECOND},),
    "packet_rate": ({"observed_packet_rate_pps": MITRE_AN0969_PACKET_RATE_PPS},),
    "egress_volume": ({"channel_volume_threshold_exceeded": 1},),
}
FEATURE_WEIGHTS = {
    "T1498.001": {"traffic_burst": 1.0, "packet_rate": 1.0},
    "T1498.002": {"reflection": 1.0, "distributed_sources": 0.65, "traffic_burst": 0.45},
    "T1499.001": {"syn_backlog": 1.0, "traffic_burst": 0.25},
    "T1499.002": {"service_pressure": 1.0},
    "T1499.003": {"application_pressure": 1.0, "traffic_burst": 0.20},
    "T1046": {"service_discovery": 1.0},
    "T1204": {"user_execution": 1.0, "application_pressure": 0.20},
    "T1068": {"privilege_transition": 1.0, "user_execution": 0.20},
    "T1041": {"egress": 1.0, "traffic_burst": 0.20, "egress_volume": 1.0},
    "T1464": {"mobile_connectivity": 1.0},
    "T1110.001": {"authentication": 1.0},
}
TECHNIQUE_BY_KIND = {
    "dos": "T1498.001", "ddos": "T1498.002", "syn_flood": "T1499.001",
    "dns_flood": "T1499.002", "dhcp_flood": "T1499.002", "application_flood": "T1499.003",
    "network_discovery": "T1046", "user_execution": "T1204", "privilege_escalation": "T1068",
    "exfiltration": "T1041", "mobile_network_dos": "T1464",
    "brute_force": "T1110.001",
}
LAG_WEIGHTS = (1.0, 0.85, 0.70, 0.55, 0.40, 0.30, 0.20, 0.10)
WEIGHTED_MASKS = {technique: np.outer([features.get(name, 0.0) for name in FEATURE_GROUPS], LAG_WEIGHTS)
                  for technique, features in FEATURE_WEIGHTS.items()}
# Сумма байтов уже включает историю датчика и входит в оценку один раз.
EGRESS_VOLUME_ROW = tuple(FEATURE_GROUPS).index("egress_volume")
WEIGHTED_MASKS["T1041"][EGRESS_VOLUME_ROW, 1:] = 0
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
    if name in ("authentication_failure_ratio", "established_channel_transfer_ratio") and not 0 <= value <= 1:
        return 0.0
    return value if math.isfinite(value) else 0.0


def _volume_window_steps(step_seconds: int) -> int:
    """Число полных интервалов, вместе занимающих строго меньше пяти минут."""
    if step_seconds <= 0:
        raise ValueError("Длительность шага должна быть положительной")
    return math.ceil(MITRE_AN0988_WINDOW_SECONDS / step_seconds) - 1


def signature_history_steps(step_seconds: int) -> int:
    """Число отсчётов вместе с текущим для масок и накопления объёма T1041."""
    return max(len(LAG_WEIGHTS), _volume_window_steps(step_seconds))


def _channel_rate(sample: dict) -> float:
    fraction = _signal(sample, "established_channel_transfer_ratio")
    return max(0.0, _signal(sample, "unusual_egress_bytes_per_second")) * fraction


def _observed_channel_bytes(signals: dict, history: list[dict | None], step_seconds: int) -> float | None:
    """Сумма (байт/с × доля × с) по известным интервалам одного датчика."""
    window_steps = _volume_window_steps(step_seconds)
    if not window_steps:
        return None
    current = _channel_rate(signals)
    if not current:
        return 0.0
    rates = [current]
    rates.extend(_channel_rate(sample) for sample in history[:window_steps - 1] if sample is not None)
    return math.fsum(rate * step_seconds for rate in rates)


def _feature_vector(signals: dict) -> list[float]:
    failure_fraction = _signal(signals, "authentication_failure_ratio")
    failures = _signal(signals, "authentication_failure_rate_per_second")
    attempt_rate = failures / failure_fraction if failure_fraction > 0 else 0.0
    signals = {**signals, "inferred_attempt_rate_per_account_per_second": attempt_rate}
    return [max(min(max(0.0, min(1.0, _signal(signals, name) / scale)) for name, scale in group.items())
                for group in alternatives) for alternatives in FEATURE_GROUPS.values()]


def _technique_mask(technique: str, step_seconds: int) -> np.ndarray:
    if step_seconds <= 0:
        raise ValueError("Длительность шага должна быть положительной")
    mask = WEIGHTED_MASKS[technique]
    if technique == "T1041" and _volume_window_steps(step_seconds) == 0:
        mask = mask.copy()
        mask[EGRESS_VOLUME_ROW] = 0
    if technique in MITRE_OBSERVATION_WINDOWS:
        window_seconds, _, _ = MITRE_OBSERVATION_WINDOWS[technique]
        ages_seconds = np.arange(len(LAG_WEIGHTS)) * step_seconds
        return mask * (ages_seconds <= window_seconds)
    return mask


def _weighted_scores(techniques: set[str], signals: dict, history: list[dict | None],
                     step_seconds: int) -> dict[str, float]:
    samples = [signals, *history[:len(LAG_WEIGHTS) - 1]]
    available = [index for index, sample in enumerate(samples) if sample is not None]
    observation = np.array([_feature_vector(samples[index]) for index in available]).T
    scores = {}
    for technique in sorted(techniques):
        mask = _technique_mask(technique, step_seconds)[:, available]
        scores[technique] = float(np.sum(mask * observation) / np.sum(mask))
    return scores


def signature_catalog(*, step_seconds: int = DYNAMICS_STEP_SECONDS) -> dict:
    references = {
        technique: {
            "maximum_lag_seconds": seconds,
            "window_origin": f"{analytic}_TimeWindow_example",
            "window_source_url": f"https://attack.mitre.org/detectionstrategies/{strategy}/",
        }
        for technique, (seconds, strategy, analytic) in MITRE_OBSERVATION_WINDOWS.items()
    }
    references["T1110.001"].update({
        "source_url": MITRE_T1110_001_APT28_PROCEDURE_URL,
        "origin": "apt28_procedure_example_lower_bound",
        "attempts_per_hour_per_account": MITRE_T1110_001_APT28_MIN_ATTEMPTS_PER_HOUR,
        "observed_metric": "inferred_attempt_rate_per_account_per_second",
        "derivation": "authentication_failure_rate_per_second / authentication_failure_ratio",
    })
    references["T1498.001"] = {
        "source_url": "https://attack.mitre.org/detectionstrategies/DET0343/",
        "origin": "AN0969_PacketRateThreshold_example",
        "observed_metric": "observed_packet_rate_pps",
        "packets_per_second": MITRE_AN0969_PACKET_RATE_PPS,
    }
    references["T1041"] = {
        "source_url": "https://attack.mitre.org/detectionstrategies/DET0348/#:~:text=DataVolumeThreshold",
        "analytic_id": "AN0988", "origin": "DataVolumeThreshold_example",
        "volume_strictly_greater_than_bytes": MITRE_AN0988_VOLUME_BYTES,
        "duration_strictly_less_than_seconds": MITRE_AN0988_WINDOW_SECONDS,
        "observed_metric": "observed_channel_volume_bytes",
        "derivation": "sum(unusual_egress_bytes_per_second * established_channel_transfer_ratio * step_seconds)",
    }
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
        "alternative_gate": {"exfiltration": {
            "replaces_required_column": "unusual_egress_bytes_per_second",
            "condition": "channel_volume_threshold_exceeded",
            "numeric_reference": "T1041",
        }},
        "weighted_masks": {
            "rows": list(FEATURE_GROUPS), "step_lags": list(range(len(LAG_WEIGHTS))),
            "normalization_by_row": FEATURE_GROUPS, "lag_weights": LAG_WEIGHTS,
            "step_seconds": step_seconds,
            "techniques": {technique: _technique_mask(technique, step_seconds).tolist()
                           for technique in WEIGHTED_MASKS},
            "numeric_references": references,
            "score_semantics": "weighted_mean_of_normalized_sensor_evidence_over_observed_steps",
            "value_origin": "gnet9_engineering_assumptions_except_listed_numeric_references",
        },
    }


def match_attack_signature(signals: dict, *, history: list[dict | None] | None = None,
                           step_seconds: int = DYNAMICS_STEP_SECONDS) -> dict | None:
    """Сопоставить измерения с рядами матрицы без метки сценария.

    Строка совпадает, когда все её активные координаты достигают минимума.
    Например, один запуск при росте CPU на 0,30 даёт min(1, 2) = 1.
    Отсутствующие или неконечные значения не удовлетворяют порогу.
    """
    history = history or []
    channel_bytes = _observed_channel_bytes(signals, history, step_seconds)
    volume_exceeded = channel_bytes is not None and channel_bytes > MITRE_AN0988_VOLUME_BYTES
    signals = {**signals, "channel_volume_threshold_exceeded": int(volume_exceeded)}
    values = np.array([_signal(signals, name) for name in SIGNATURE_COLUMNS])
    power_control = _signal(signals, "power_control_anomaly_ratio")
    voltage_sag = _signal(signals, "power_voltage_sag_ratio")
    values[SIGNATURE_COLUMNS.index("power_observation_ratio")] = max(
        power_control / 0.18, voltage_sag / 0.02,
    )
    masked = (values / SIGNATURE_SCALE) * SIGNATURE_MASK
    if volume_exceeded:
        row = tuple(SIGNATURE_LIMITS).index("exfiltration")
        column = SIGNATURE_COLUMNS.index("unusual_egress_bytes_per_second")
        masked[row, column] = max(masked[row, column], channel_bytes / MITRE_AN0988_VOLUME_BYTES)
    margins = np.min(np.where(SIGNATURE_MASK, masked, np.inf), axis=1)
    matches = np.flatnonzero(margins >= 1.0)
    if matches.size == 0:
        return None
    kinds = tuple(SIGNATURE_LIMITS)
    candidates = {kinds[index]: float(margins[index]) for index in matches}
    techniques = {TECHNIQUE_BY_KIND[kind] for kind in candidates if kind in TECHNIQUE_BY_KIND}
    scores = _weighted_scores(techniques, signals, history, step_seconds) if techniques else {}
    ordered = sorted(candidates, key=lambda kind: (
        2 if kind not in {"dos", "ddos"} else 1 if kind == "ddos" else 0,
        scores.get(TECHNIQUE_BY_KIND.get(kind), 1.0), candidates[kind],
    ), reverse=True)
    kind = ordered[0]
    technique = TECHNIQUE_BY_KIND.get(kind)
    observed = {name: float(value) for name, value in zip(SIGNATURE_COLUMNS, values)
                if name in SIGNATURE_LIMITS[kind]}
    if kind == "exfiltration":
        observed["observed_channel_volume_bytes"] = channel_bytes
    return {
        "kind": kind, "margin_ratio": round(candidates[kind], 6), "matching_kinds": ordered,
        "weighted_technique_id": technique,
        "weighted_score": round(scores[technique], 6) if technique else None,
        "weighted_technique_scores": {technique: round(score, 6) for technique, score in scores.items()},
        **flood_fields(kind, _signal(signals, "observed_ingress_count")),
        "observed_values": observed,
    }
