"""Воспроизводимые сценарии отказа в обслуживании по MITRE ATT&CK.

MITRE ATT&CK задаёт таксономию и поведение атак, а не универсальные нормативные
скорости. Большинство численных интенсивностей ниже являются явными параметрами
лабораторного сценария. Исключение отмечено отдельно: для T1110.001 используется
нижняя граница из официального примера процедуры MITRE. Это всё равно не стандарт
для любой сети и не повод выполнять реальный подбор паролей.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
from functools import lru_cache
import math
import random
from typing import Any

from .models import NetworkModel
from .routing import has_data_path, shortest_data_path
from .transport_queue import apply_transport_queues


MITRE_ATTACK_SOURCE = "https://attack.mitre.org/techniques/"
MITRE_T1110_001_APT28_PROCEDURE_URL = "https://attack.mitre.org/techniques/T1110/001/"
# MITRE's APT28 procedure example says "over 300 authentication attempts per
# hour per targeted account".  The simulator uses 300/h only as the documented
# lower-bound reference (one expected failed login every 12 seconds), not as a
# universal attack rate or as a real traffic generator.
MITRE_T1110_001_APT28_MIN_ATTEMPTS_PER_HOUR = 300
MITRE_T1110_001_ATTEMPTS_PER_SECOND = (
    MITRE_T1110_001_APT28_MIN_ATTEMPTS_PER_HOUR / 3_600
)


class AttackKind(str, Enum):
    DOS = "dos"
    DDOS = "ddos"
    SYN_FLOOD = "syn_flood"
    BRUTE_FORCE = "brute_force"
    POWER_ATTACK = "power_attack"


@dataclass(frozen=True)
class AttackTemporalCharacteristics:
    """ВВХ: временные характеристики одного воздействия."""

    start_step: int
    duration_steps: int
    rise_steps: int
    pulse_shape: str
    precursor_steps: int = 1


@dataclass(frozen=True)
class AttackTechnicalCharacteristics:
    """ТТХ: технические параметры генератора вредоносной нагрузки."""

    protocol: str
    packet_size_bytes: int
    offered_rate_mbps: float
    packet_rate_pps: int
    source_count: int
    amplification_factor: float
    incomplete_handshake_ratio: float
    value_origin: str = "scenario_assumption"
    packet_size_semantics: str = "application_payload_bytes_for_udp_or_ipv4_packet_bytes_for_tcp_syn"
    on_wire_size_bytes: int | None = None
    resource_axis: str = "network_bandwidth_or_target_resource"
    traffic_origin_semantics: str = "direct_from_listed_ingress_nodes"
    initiator_count: int | None = None
    external_reflector_count: int = 0
    authentication_attempt_rate_per_second: float | None = None
    authentication_failure_ratio: float = 0.0
    authentication_attempt_rate_origin: str = "not_applicable"
    authentication_attempt_rate_reference_url: str | None = None


@dataclass(frozen=True)
class WeibullArrivalSchedule:
    """Реализация renewal-процесса поступления атак для конечного стенда.

    Вейбулл задаёт интервалы между событиями, а не вероятность типа атаки.
    Непрерывные интервалы независимо генерируются обратным преобразованием и
    не масштабируются после выборки. Процесс останавливается на первой из двух
    границ: следующая атака за пределами окна правосторонне цензурируется, а
    защитный предел оставляет визуализацию и расчётный контур ограниченными.
    Поэтому число событий может быть меньше заданного верхнего предела.
    Параметры всё ещё являются воспроизводимым допущением стенда, а не
    статистической оценкой оператора связи.
    """

    shape: float
    scale_seconds: float
    expected_interarrival_seconds: float
    nominal_intensity_events_per_second: float
    raw_interarrival_seconds: tuple[float, ...]
    realized_interarrival_seconds: tuple[int, ...]
    realized_start_steps: tuple[int, ...]
    realized_start_seconds: tuple[int, ...]
    realized_hazard_per_second: tuple[float, ...]
    realized_survival_probability: tuple[float, ...]
    window_origin_step: int
    window_end_step: int
    seed: int
    rng_stream_seed: int
    maximum_event_count: int
    conditioning_scale_factor: float
    right_censored_interarrival_seconds: float | None
    termination_reason: str
    distribution: str = "weibull"
    conditioning: str = "finite_window_right_censoring_or_safety_cap_without_rescaling"
    generation_method: str = "inverse_cdf_iid_renewal_until_window_end_or_safety_cap"
    value_origin: str = "seeded_weibull_renewal_scenario_assumption"
    raw_intervals_are_iid_weibull: bool = True
    realized_intervals_are_iid_weibull: bool = False
    hazard_semantics: str = "unconditioned_weibull_hazard_evaluated_at_raw_sampled_interval"
    parameter_estimation_status: str = "not_fitted_to_operator_incident_data"
    mean_formula: str = "scale * Gamma(1 + 1/shape)"
    survival_formula: str = "exp(-(t/scale)**shape)"
    hazard_formula: str = "shape/scale * (t/scale)**(shape-1)"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttackProfile:
    attack_id: str
    name_ru: str
    kind: AttackKind
    mitre_technique_id: str
    mitre_technique_name: str
    related_technique_ids: tuple[str, ...]
    target_id: str
    target_type: str
    ingress_nodes: tuple[str, ...]
    temporal: AttackTemporalCharacteristics
    technical: AttackTechnicalCharacteristics
    latency_penalty_ms: float
    legitimate_loss_ratio: float
    target_resource_pressure: float
    description_ru: str
    power_failure_mode: str | None = None

    @property
    def mitre_url(self) -> str:
        return f"{MITRE_ATTACK_SOURCE}{self.mitre_technique_id.replace('.', '/')}/"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["mitre_url"] = self.mitre_url
        data["related_mitre_urls"] = [
            f"{MITRE_ATTACK_SOURCE}{technique_id.replace('.', '/')}/"
            for technique_id in self.related_technique_ids
        ]
        data["mitre_semantics"] = (
            "behavior_taxonomy_not_a_universal_numeric_rate_standard; "
            "a separately-labelled procedure-example value may be used where available"
        )
        data["impact_semantics"] = (
            "computed_from_path_bottleneck_and_target_resource_budget; "
            "numeric budgets are labelled GNet9 scenario assumptions"
        )
        return data


@dataclass(frozen=True)
class MitreTensorMapping:
    """Explicit link between an ATT&CK technique and observable G-Net state."""

    technique_id: str
    technique_name: str
    attack_kind: AttackKind
    gnet_levels: tuple[str, ...]
    tensor_metrics: tuple[str, ...]
    observation_metrics: tuple[str, ...]
    mitigation_ids: tuple[str, ...]
    mitigation_actions: tuple[str, ...]
    scope: str = "safe_in_memory_simulation"

    @property
    def mitre_url(self) -> str:
        return f"{MITRE_ATTACK_SOURCE}{self.technique_id.replace('.', '/')}/"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["attack_kind"] = self.attack_kind.value
        data["mitre_url"] = self.mitre_url
        data["mapping_semantics"] = (
            "expert_declared_observable_mapping; not a claim that all ATT&CK techniques are covered"
        )
        return data


MITRE_TENSOR_MAPPINGS = (
    MitreTensorMapping(
        "T1498.001", "Direct Network Flood", AttackKind.DOS,
        ("L1", "L2", "EDGE", "L7"),
        ("L1.sla_margin", "L2.cpu_load_percent", "EDGE.utilization", "EDGE.loss_probability"),
        ("attack_rate_mbps", "traffic_acceleration_ratio", "target_concentration_ratio"),
        (), ("ingress_acl_or_policer", "source_quarantine"),
    ),
    MitreTensorMapping(
        "T1498.002", "Reflection Amplification", AttackKind.DDOS,
        ("L1", "L2", "EDGE", "L7"),
        ("L1.sla_margin", "L2.cpu_load_percent", "EDGE.utilization", "EDGE.loss_probability"),
        ("attack_rate_mbps", "suspicious_source_count", "source_entropy_ratio"),
        (), ("upstream_flowspec", "scrubbing_center", "ingress_acl_or_policer"),
    ),
    MitreTensorMapping(
        "T1499.001", "OS Exhaustion Flood", AttackKind.SYN_FLOOD,
        ("L1", "L2", "EDGE", "L7"),
        ("L1.sla_margin", "L2.stability_margin", "EDGE.loss_probability"),
        ("syn_backlog_growth_ratio", "maximum_target_utilization_percent"),
        (), ("syn_proxy", "syn_cookies", "connection_rate_limit"),
    ),
    MitreTensorMapping(
        "T1110.001", "Password Guessing", AttackKind.BRUTE_FORCE,
        ("L1", "L7"),
        ("L1.authentication_failure_rate_per_second", "L1.sla_margin"),
        ("authentication_failure_rate_per_second", "authentication_failure_ratio", "account_lockout_pressure"),
        ("M1036", "M1032", "M1027", "M1018"),
        ("account_lockout_policy", "multi_factor_authentication", "conditional_access"),
    ),
    MitreTensorMapping(
        "T0831", "Manipulation of Control", AttackKind.POWER_ATTACK,
        ("L6", "L7"),
        ("L6.energy_reserve_ratio",),
        ("power_control_anomaly_ratio", "power_voltage_sag_ratio", "battery_discharge_rate_ratio"),
        (), ("revoke_management_session", "transfer_to_independent_power_domain"),
    ),
    MitreTensorMapping(
        "T1529", "System Shutdown/Reboot", AttackKind.POWER_ATTACK,
        ("L0", "L6", "L7"),
        ("L0.service_health", "L6.energy_reserve_ratio"),
        ("power_control_anomaly_ratio", "power_service_unavailability_ratio"),
        (), ("revoke_management_session", "failover_service_to_standby"),
    ),
)


def mitre_tensor_mapping_catalog() -> list[dict[str, Any]]:
    """Return the explicitly supported, observable ATT&CK-to-tensor mappings."""
    return [item.to_dict() for item in MITRE_TENSOR_MAPPINGS]


# Расписание рассчитано на стандартные 10 переходов по 5 секунд.
MITRE_DEMO_ATTACKS = (
    AttackProfile(
        "DOS_CORE_C1", "DoS на маршрутизатор ядра C1", AttackKind.DOS,
        "T1498.001", "Direct Network Flood", (), "C1", "core-router", ("F3_40",),
        AttackTemporalCharacteristics(2, 2, 1, "ramp-then-peak"),
        AttackTechnicalCharacteristics(
            "UDP", 512, 88.0, 19_031, 1, 1.0, 0.0,
            on_wire_size_bytes=578,
            resource_axis="router_control_plane_processing",
        ),
        38.0, 0.055, 0.62,
        "Один источник создаёт прямой UDP-флуд и повышает загрузку плоскости обработки C1.",
    ),
    AttackProfile(
        "DDOS_MEDIA_SERVER", "DDoS с отражённым усилением на медиасервер", AttackKind.DDOS,
        "T1498.002", "Reflection Amplification", (), "SRV_MEDIA", "service-server",
        ("M3_31", "M3_32", "F3_31", "F3_32"),
        AttackTemporalCharacteristics(5, 2, 1, "distributed-ramp-then-peak"),
        AttackTechnicalCharacteristics(
            "UDP reflection/amplification", 1_200, 24_000.0, 2_369_668,
            1_600, 80.0, 0.0,
            on_wire_size_bytes=1_266,
            resource_axis="converged_network_bandwidth",
            traffic_origin_semantics=(
                "listed_L1_nodes_initiate_spoofed_requests; external_reflectors_return_amplified_traffic"
            ),
            initiator_count=4,
            external_reflector_count=1_600,
        ),
        190.0, 0.22, 0.94,
        "Распределённые отражённые ответы превышают суммарную ёмкость двух 10-гигабитных подключений SRV_MEDIA.",
    ),
    AttackProfile(
        "SYN_GOLD_ENDPOINT", "SYN-флуд на оконечное устройство Gold", AttackKind.SYN_FLOOD,
        "T1499.001", "OS Exhaustion Flood", ("T1498.001",), "M1_01", "mobile-subscriber",
        ("M3_39", "F3_39"),
        AttackTemporalCharacteristics(8, 2, 1, "distributed-ramp-then-peak"),
        AttackTechnicalCharacteristics(
            "TCP SYN", 40, 142.0, 211_309, 320, 1.0, 0.998,
            on_wire_size_bytes=84,
            resource_axis="tcp_syn_backlog_and_connection_state",
            traffic_origin_semantics="listed_L1_nodes_represent_distributed_spoofed_sources",
            initiator_count=2,
        ),
        420.0, 0.38, 0.88,
        "Поток SYN заполняет очередь полуоткрытых соединений и радиоканал Gold-абонента M1_01.",
    ),
    AttackProfile(
        "BRUTE_FORCE_GOLD_ACCOUNT", "Медленный подбор пароля к учётной записи Gold", AttackKind.BRUTE_FORCE,
        "T1110.001", "Password Guessing", ("T1110",), "M1_01", "mobile-subscriber", ("F3_40",),
        AttackTemporalCharacteristics(4, 4, 3, "slow-constant-authentication-attempts"),
        AttackTechnicalCharacteristics(
            "HTTPS/TLS authentication", 0, 0.0, 0, 1, 1.0, 0.0,
            value_origin="mitre_t1110_001_apt28_procedure_example_lower_bound",
            resource_axis="online_authentication_failure_counter_and_account_lockout_policy",
            traffic_origin_semantics="single_external_identity_attempting_known_account_without_packet_emission",
            initiator_count=1,
            authentication_attempt_rate_per_second=MITRE_T1110_001_ATTEMPTS_PER_SECOND,
            authentication_failure_ratio=1.0,
            authentication_attempt_rate_origin=(
                "mitre_t1110_001_apt28_procedure_example_lower_bound_300_attempts_per_hour_per_account"
            ),
            authentication_attempt_rate_reference_url=MITRE_T1110_001_APT28_PROCEDURE_URL,
        ),
        0.0, 0.0, 0.06,
        "Нижняя граница из примера процедуры MITRE: 300 неуспешных попыток в час на учётную "
        "запись, то есть одна ожидаемая попытка за 12 секунд. Это не норматив MITRE, а безопасная "
        "модель журнальных событий T1110.001: она не генерирует пакеты и не выполняет подбор паролей.",
    ),
    AttackProfile(
        "POWER_AGG_A3", "Воздействие на питание агрегирующего коммутатора A3", AttackKind.POWER_ATTACK,
        "T0831", "Manipulation of Control", ("T1078", "T0826"),
        "A3", "aggregation-switch", (),
        AttackTemporalCharacteristics(10, 1, 1, "power-drop"),
        AttackTechnicalCharacteristics(
            "UPS/PDU input control and electrical telemetry", 0, 0.0, 0, 1, 1.0, 0.0,
            resource_axis="local_dual_feed_input_brownout_with_ups_ride_through",
            traffic_origin_semantics="facility_control_plane_or_physical_feed_disturbance_not_subscriber_traffic",
            initiator_count=1,
        ),
        0.0, 0.0, 0.82,
        "Краткий локальный провал обоих входов до ИБП: батарея удерживает выход, "
        "а сервис не прерывается, пока длительность меньше автономности.",
        power_failure_mode="brownout_feed_loss",
    ),
    AttackProfile(
        "POWER_RTC_SERVER", "Воздействие на питание сервера видеоконференций", AttackKind.POWER_ATTACK,
        "T1529", "System Shutdown/Reboot", ("T1078", "T0826"),
        "SRV_RTC", "service-server", (),
        AttackTemporalCharacteristics(10, 1, 1, "power-drop"),
        AttackTechnicalCharacteristics(
            "HTTPS/Redfish out-of-band management shutdown", 0, 0.0, 0, 1, 1.0, 0.0,
            resource_axis="server_shutdown_and_boot_health_recovery",
            traffic_origin_semantics="out_of_band_management_plane_after_valid_account_compromise",
            initiator_count=1,
        ),
        140.0, 0.18, 0.86,
        "Команда выключения через изолированную плоскость управления оставляет "
        "входное напряжение нормальным, затем требует загрузки и health-check.",
        power_failure_mode="cyber_shutdown",
    ),
)


# Новый демонстрационный профиль рассчитан на 60 переходов по 2 секунды.
# Восемь — это верхняя граница числа атак в одном прогоне, а не гарантия:
# renewal-процесс Вейбулла прекращается на правой границе окна. Квоты ниже
# задают состав полной демонстрационной серии и вероятностные веса типов.
PREDICTIVE_DEMO_DEFAULT_SEED = 42
PREDICTIVE_DEMO_STEP_SECONDS = 2
PREDICTIVE_DEMO_STEP_COUNT = 100
PREDICTIVE_DEMO_PRECURSOR_SECONDS = 20


def predictive_demo_minimum_steps(step_seconds: int) -> int:
    """Return the shortest runnable predictive-demo window.

    The scenario needs its 20-second causal precursor window (at least two
    samples) and enough trailing frames to begin and finish a scheduled event.
    Keeping this calculation next to the schedule avoids a different
    constraint in the CLI, simulator and documentation.
    """
    if step_seconds <= 0:
        raise ValueError("Длительность шага predictive-demo должна быть положительной")
    precursor_steps = max(
        2,
        math.ceil(PREDICTIVE_DEMO_PRECURSOR_SECONDS / step_seconds),
    )
    return precursor_steps + 5


# k < 1 задаёт убывающую интенсивность отказов (hazard) и тяжёлый хвост.
# Это не самовозбуждающийся процесс и не доказательство координации ботнета;
# число является только воспроизводимым допущением лабораторного сценария.
PREDICTIVE_DEMO_WEIBULL_SHAPE = 0.85
PREDICTIVE_DEMO_KIND_COUNTS = {
    AttackKind.DOS: 3,
    AttackKind.DDOS: 2,
    AttackKind.SYN_FLOOD: 2,
    AttackKind.POWER_ATTACK: 1,
}
PREDICTIVE_DEMO_KIND_WEIGHTS = {
    kind.value: count / sum(PREDICTIVE_DEMO_KIND_COUNTS.values())
    for kind, count in PREDICTIVE_DEMO_KIND_COUNTS.items()
}


TRAFFIC_SERVICE_NODES = {
    "voice": "SVC_VOICE",
    "vlc_av": "SVC_VLC",
    "ftp": "SVC_FTP",
    "dns": "SVC_DNS",
    "video_conference": "SVC_TELEMOST",
    "live_streaming": "SVC_LIVE",
}


# Эффективность — явное допущение симулятора, а не норматив MITRE ATT&CK.
# Значение показывает долю воздействия, которую подготовленная за предыдущий
# шаг защита способна отсечь. Для легитимных потоков коэффициент дополнительно
# умножается на приоритет SLA: Gold восстанавливается раньше Silver и Bronze.
PREVENTIVE_DEFENSE_EFFECTIVENESS = {
    AttackKind.DOS.value: 0.70,
    AttackKind.DDOS.value: 0.78,
    AttackKind.SYN_FLOOD.value: 0.84,
    AttackKind.BRUTE_FORCE.value: 0.95,
    AttackKind.POWER_ATTACK.value: 0.68,
}
SLA_PROTECTION_FACTOR = {"gold": 1.0, "silver": 0.72, "bronze": 0.45}
SLA_RESTORATION_PRIORITY = {"gold": 1, "silver": 2, "bronze": 3}
# Независимая от силы атаки ёмкость контура восстановления легитимных пакетов.
# Это лабораторный профиль PPS для ACL/policer/SYN-proxy/резервного питания,
# а не паспортная производительность конкретного ASIC.
DEFENSE_RECOVERY_CAPACITY_PPS = {
    "core-router": 1_200,
    "aggregation-switch": 1_000,
    "radio-access-node": 850,
    "optical-line-terminal": 900,
    "service-server": 800,
    "mobile-subscriber": 250,
    "fixed-subscriber": 250,
}

# Не паспортные значения Cisco/Dell, а калибруемые пороги стенда для трафика,
# адресованного самой цели (control plane, приложение, SYN backlog). Линейная
# ёмкость при этом всегда считается отдельно по фактическим рёбрам графа.
TARGET_PROCESSING_BUDGET_PPS = {
    "core-router": 250_000,
    "aggregation-switch": 150_000,
    "radio-access-node": 120_000,
    "optical-line-terminal": 130_000,
    "service-server": 500_000,
    "mobile-subscriber": 50_000,
    "fixed-subscriber": 75_000,
}
# Одноисточниковый DoS направляется на control plane или дорогой обработчик
# приложения, а не выдаётся за насыщение всей fabric. Пределы — калибруемые
# PPS-бюджеты стенда и намеренно ниже паспортной data-plane пересылки шасси.
TARGET_SINGLE_SOURCE_DOS_BUDGET_PPS = {
    "core-router": 25_000,
    "aggregation-switch": 20_000,
    "radio-access-node": 18_000,
    "optical-line-terminal": 18_000,
    "service-server": 40_000,
    "mobile-subscriber": 8_000,
    "fixed-subscriber": 15_000,
}
TARGET_SYN_STATE_BUDGET_PPS = {
    "core-router": 150_000,
    "aggregation-switch": 100_000,
    "radio-access-node": 80_000,
    "optical-line-terminal": 90_000,
    "service-server": 75_000,
    "mobile-subscriber": 10_000,
    "fixed-subscriber": 20_000,
}

# Stateful recovery is intentionally slower than the accelerated attack pulse.
# Values are laboratory boot/readiness assumptions, not vendor boot-time SLAs.
POWER_CYBER_RECOVERY_SECONDS = {
    "core-router": 60,
    "aggregation-switch": 45,
    "radio-access-node": 40,
    "optical-line-terminal": 40,
    "service-server": 35,
}
POWER_BROWNOUT_RECHARGE_SECONDS = 60

# В учебной топологии нет отдельного узла Internet/IXP. Эти маршрутизаторы
# поэтому играют роль трёх независимых внешних границ оператора для отражённых
# ответов. Это явное допущение сценария, а не утверждение о конкретной сети.
REFLECTION_TRANSIT_INGRESS_CANDIDATES = ("C1", "C5", "C9")


def attack_catalog(
    model: NetworkModel | str | None = None,
    scenario: str = "mitre-demo",
    *,
    seed: int | None = None,
    step_seconds: int | None = None,
    step_count: int | None = None,
    config: Any | None = None,
) -> list[dict[str, Any]]:
    """Вернуть каталог выбранного сценария без изменения старого API.

    Вызов ``attack_catalog()`` по-прежнему возвращает старый ``mitre-demo``.
    Для ``predictive-demo`` модель желательна: тогда источники и цели выбираются
    только среди реально присутствующих в графе узлов. Без модели используются
    идентификаторы штатной топологии GNet9.
    """
    if isinstance(model, str):
        scenario, model = model, None
    if scenario == "none":
        return []
    resolved_seed, resolved_step_seconds, resolved_step_count = _scenario_options(
        seed=seed,
        step_seconds=step_seconds,
        step_count=step_count,
        config=config,
    )
    profiles = _profiles_for_scenario(
        model,
        scenario,
        seed=resolved_seed,
        step_seconds=resolved_step_seconds,
        step_count=resolved_step_count,
        config=config,
    )
    catalog = [profile.to_dict() for profile in profiles]
    if model is not None:
        for item in catalog:
            item["target_device_context"] = target_device_context(
                model,
                str(item.get("target_id", "")),
            )
    if scenario == "predictive-demo" and not _custom_schedule(config):
        arrival_schedule = predictive_demo_weibull_schedule(
            seed=resolved_seed,
            step_seconds=resolved_step_seconds,
            step_count=resolved_step_count,
        )
        schedule_data = arrival_schedule.to_dict()
        for realization_index, item in enumerate(catalog):
            item["scenario_weight"] = 1.0 / max(len(catalog), 1)
            item["scenario_weight_semantics"] = "realized_event_mass"
            item["kind_weight"] = PREDICTIVE_DEMO_KIND_WEIGHTS[str(item["kind"])]
            item["scenario_weight_origin"] = "scenario_assumption"
            item["scenario_seed"] = resolved_seed
            item["time_scale_semantics"] = "accelerated_demo_not_real_incident_duration"
            # Полное скрытое расписание допустимо только в экспортируемом
            # каталоге испытания. В телеметрию Купмана оно не попадает.
            item["arrival_schedule"] = dict(schedule_data)
            item["arrival_realization_index"] = realization_index
            item["arrival_interval_seconds"] = (
                arrival_schedule.realized_interarrival_seconds[realization_index]
            )
            item["arrival_sample_interval_seconds"] = (
                arrival_schedule.raw_interarrival_seconds[realization_index]
            )
    if _custom_schedule(config):
        if config.attack_interval_steps is not None:
            distribution = "fixed_interval"
        elif scenario == "predictive-demo":
            distribution = "shifted_weibull"
        else:
            distribution = "shifted_catalog"
        start_steps = [profile.temporal.start_step for profile in profiles]
        precursor_start = min(
            (profile.temporal.start_step - profile.temporal.precursor_steps for profile in profiles),
            default=None,
        )
        for item in catalog:
            item["arrival_schedule"] = {
                "distribution": distribution,
                "realized_start_steps": list(start_steps),
                "first_attack_step": start_steps[0] if start_steps else None,
                "interval_steps": config.attack_interval_steps,
                "precursor_start_step": precursor_start,
            }
    return catalog


def target_device_context(model: NetworkModel, target_id: str) -> dict[str, Any]:
    """Bind an attack target's ВВХ/ТТХ interpretation to its model profile.

    A subscriber target inherits its explicit RAN/OLT access profile.  This
    makes the numerical scenario traceable to a concrete model envelope while
    preserving the distinction between vendor-published fields and assumptions.
    """
    if target_id not in model.graph:
        return {"status": "target_not_present_in_model"}
    attrs = model.graph.nodes[target_id]
    device_id = target_id
    device_attrs = attrs
    binding = "direct_target_profile"
    if attrs.get("level") == "L1":
        candidate = str(attrs.get("home_access", ""))
        if candidate in model.graph:
            device_id = candidate
            device_attrs = model.graph.nodes[candidate]
            binding = "subscriber_target_inherits_home_access_profile"
    l2_profile = device_attrs.get("l2_profile", {})
    server_profile = device_attrs.get("server_profile", {})
    profile = l2_profile if isinstance(l2_profile, Mapping) and l2_profile else server_profile
    profile = profile if isinstance(profile, Mapping) else {}
    vendor = str(profile.get("vendor", "Dell" if device_attrs.get("role") == "service-server" else ""))
    return {
        "binding": binding,
        "target_node_id": target_id,
        "capacity_device_id": device_id,
        "capacity_device_role": device_attrs.get("role"),
        "profile_name": profile.get("name", profile.get("model", device_attrs.get("platform_profile"))),
        "vendor": vendor or "GNet9 scenario",
        "source_url": profile.get("source_url", profile.get("source_url")),
        "verified_fields": list(profile.get("verified_fields", [])),
        "assumed_fields": list(profile.get("assumed_fields", [])),
        "power_semantics": device_attrs.get("power_architecture", {}).get("value_origin"),
        "semantics_ru": (
            "Параметры ВВХ/ТТХ атаки сопоставлены с профилем целевой ёмкости; "
            "сценарные поля не выдаются за паспортные характеристики."
        ),
    }


@lru_cache(maxsize=128)
def predictive_demo_weibull_schedule(
    *,
    seed: int = PREDICTIVE_DEMO_DEFAULT_SEED,
    step_seconds: int = PREDICTIVE_DEMO_STEP_SECONDS,
    step_count: int = PREDICTIVE_DEMO_STEP_COUNT,
    event_count: int = 8,
    shape: float = PREDICTIVE_DEMO_WEIBULL_SHAPE,
    precursor_steps: int | None = None,
) -> WeibullArrivalSchedule:
    """Сформировать воспроизводимый конечный renewal-процесс Вейбулла.

    ``event_count`` — предохранительный верхний предел для короткой учебной
    демонстрации, не фиксированное число событий. Каждый следующий интервал
    берётся из одной и той же iid Weibull-выборки. Интервал, выводящий начало
    следующей атаки за границу окна, сохраняется как правосторонне
    цензурированный и не превращается в событие; иначе процесс может закончиться
    на защитном пределе. Дискретизация нужна только для кадров визуализации:
    непрерывные значения не подгоняются под окно.
    """
    if step_seconds <= 0:
        raise ValueError("Длительность шага должна быть положительной")
    if event_count <= 0:
        raise ValueError("Число событий Вейбулла должно быть положительным")
    if shape <= 0.0 or not math.isfinite(shape):
        raise ValueError("Параметр формы Вейбулла должен быть конечным и положительным")
    resolved_precursor_steps = (
        predictive_demo_minimum_steps(step_seconds) - 5
        if precursor_steps is None
        else int(precursor_steps)
    )
    first_start_step = resolved_precursor_steps + 2
    last_start_step = step_count - 3
    available_slots = last_start_step - first_start_step + 1
    if resolved_precursor_steps < 2 or available_slots < 1:
        minimum_steps = (
            predictive_demo_minimum_steps(step_seconds)
            if precursor_steps is None
            else resolved_precursor_steps + 5
        )
        raise ValueError(
            f"Для окна предвестника {resolved_precursor_steps} шагов требуется не менее {minimum_steps} шагов "
            f"при окне предвестника {resolved_precursor_steps} шагов"
        )

    # Начало процесса лежит за один шаг до первой допустимой атаки. Времени
    # достаточно и для прогрева t0, и для минимум двух причинных измерений.
    window_origin_step = first_start_step - 1
    available_seconds = (last_start_step - window_origin_step) * step_seconds
    expected_interval = available_seconds / event_count
    scale_seconds = expected_interval / math.gamma(1.0 + 1.0 / shape)

    schedule_seed = (int(seed) ^ 0x5EED_B011) & ((1 << 64) - 1)
    rng = random.Random(schedule_seed)
    raw_intervals: list[float] = []
    realized_steps: list[int] = []
    previous_step = window_origin_step
    right_censored_interval: float | None = None
    for _ in range(event_count):
        raw_interval = (
            scale_seconds
            * (-math.log1p(-max(rng.random(), 1e-12))) ** (1.0 / shape)
        )
        # ceil сохраняет причинность: событие из непрерывного интервала
        # отображается в первый полностью наступивший дискретный кадр.
        interval_steps = max(1, math.ceil(raw_interval / step_seconds))
        realized_step = previous_step + interval_steps
        if realized_step > last_start_step:
            right_censored_interval = raw_interval
            break
        raw_intervals.append(raw_interval)
        realized_steps.append(realized_step)
        previous_step = realized_step
    termination_reason = (
        "right_censored_at_window_end"
        if right_censored_interval is not None
        else "safety_event_limit_reached"
    )

    prior_steps = [window_origin_step, *realized_steps[:-1]]
    realized_intervals = tuple(
        (current - previous) * step_seconds
        for previous, current in zip(prior_steps, realized_steps, strict=True)
    )
    hazards = tuple(
        shape / scale_seconds * (interval / scale_seconds) ** (shape - 1.0)
        for interval in raw_intervals
    )
    survival = tuple(
        math.exp(-((interval / scale_seconds) ** shape))
        for interval in raw_intervals
    )
    return WeibullArrivalSchedule(
        shape=round(shape, 6),
        scale_seconds=round(scale_seconds, 6),
        expected_interarrival_seconds=round(expected_interval, 6),
        nominal_intensity_events_per_second=round(1.0 / expected_interval, 9),
        raw_interarrival_seconds=tuple(round(value, 6) for value in raw_intervals),
        realized_interarrival_seconds=realized_intervals,
        realized_start_steps=tuple(realized_steps),
        realized_start_seconds=tuple(step * step_seconds for step in realized_steps),
        realized_hazard_per_second=tuple(round(value, 9) for value in hazards),
        realized_survival_probability=tuple(round(value, 9) for value in survival),
        window_origin_step=window_origin_step,
        window_end_step=last_start_step,
        seed=int(seed),
        rng_stream_seed=schedule_seed,
        maximum_event_count=event_count,
        conditioning_scale_factor=1.0,
        right_censored_interarrival_seconds=(
            round(right_censored_interval, 6)
            if right_censored_interval is not None
            else None
        ),
        termination_reason=termination_reason,
    )


def predictive_demo_attack_profiles(
    model: NetworkModel | None,
    *,
    seed: int = PREDICTIVE_DEMO_DEFAULT_SEED,
    step_seconds: int = PREDICTIVE_DEMO_STEP_SECONDS,
    step_count: int = PREDICTIVE_DEMO_STEP_COUNT,
) -> tuple[AttackProfile, ...]:
    """Построить воспроизводимое расписание не более чем из восьми атак.

    Генератор знает расписание, но наблюдателю оно не передаётся: Купман получает
    только плавно нарастающие показания датчиков. Это отделяет сценарий испытаний
    от входных признаков прогнозатора и исключает чтение будущего шага напрямую.
    """
    if step_seconds <= 0:
        raise ValueError("Длительность шага predictive-demo должна быть положительной")
    cache_key = (int(seed), int(step_seconds), int(step_count))
    if model is not None:
        cache = getattr(model, "_predictive_attack_profile_cache", {})
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
    # Нужны как минимум два причинных отсчёта: один наблюдает аномалию, второй
    # подтверждает её. ceil сохраняет окно не короче 20 с при любом такте.
    precursor_steps = max(
        2,
        math.ceil(PREDICTIVE_DEMO_PRECURSOR_SECONDS / step_seconds),
    )
    arrival_schedule = predictive_demo_weibull_schedule(
        seed=seed,
        step_seconds=step_seconds,
        step_count=step_count,
        event_count=sum(PREDICTIVE_DEMO_KIND_COUNTS.values()),
        precursor_steps=precursor_steps,
    )

    rng = random.Random(int(seed))
    all_kinds = [
        kind
        for kind, count in PREDICTIVE_DEMO_KIND_COUNTS.items()
        for _ in range(count)
    ]
    # В коротком окне renewal-процесс может успеть показать лишь часть серии.
    # Первые четыре позиции поэтому образуют одну перемешанную «витрину» всех
    # поддерживаемых воздействий; оставшиеся сохраняют заданные веса 3:2:2:1.
    # Это относится только к составу учебного сценария, не к Weibull-времени.
    kinds = list(PREDICTIVE_DEMO_KIND_COUNTS)
    rng.shuffle(kinds)
    remaining_kinds = list(all_kinds)
    for kind in kinds:
        remaining_kinds.remove(kind)
    rng.shuffle(remaining_kinds)
    kinds.extend(remaining_kinds)
    start_steps = arrival_schedule.realized_start_steps
    pools = _predictive_node_pools(model)
    used_targets: set[str] = set()
    occurrence = {kind: 0 for kind in AttackKind}
    profiles: list[AttackProfile] = []

    # Если окно заканчивается раньше, часть полного набора типов остаётся за
    # правой границей вместе со своим ещё не наступившим интервалом Вейбулла.
    for index, (kind, start_step) in enumerate(zip(kinds, start_steps), start=1):
        occurrence_index = occurrence[kind]
        occurrence[kind] += 1
        target_pool = _target_pool_for_kind(
            kind,
            occurrence_index,
            pools,
            scenario_seed=seed,
        )
        target_id = _choose_prefer_unused(rng, target_pool, used_targets)
        used_targets.add(target_id)
        target_type = _attack_target_type(model, target_id)
        power_failure_mode = (
            "cyber_shutdown"
            if kind == AttackKind.POWER_ATTACK and target_type == "service-server"
            else "brownout_feed_loss"
            if kind == AttackKind.POWER_ATTACK
            else None
        )

        ingress_count = {
            AttackKind.DOS: 1,
            AttackKind.DDOS: 8,
            AttackKind.SYN_FLOOD: 4,
            AttackKind.POWER_ATTACK: 1,
        }[kind]
        ingress_nodes = (
            ()
            if kind == AttackKind.POWER_ATTACK
            else _choose_ingress_nodes(
                rng,
                pools["bronze_subscribers"],
                count=ingress_count,
                excluded={target_id},
            )
        )
        technical = _predictive_technical_profile(
            model,
            kind,
            ingress_nodes,
            power_failure_mode=power_failure_mode,
        )
        duration_steps = {
            AttackKind.DOS: 2,
            AttackKind.DDOS: 3,
            AttackKind.SYN_FLOOD: 2,
            AttackKind.POWER_ATTACK: 2,
        }[kind]
        rise_steps = 1 if duration_steps <= 2 else 2
        latency_penalty, loss_ratio, pressure = {
            AttackKind.DOS: (46.0, 0.075, 0.70),
            AttackKind.DDOS: (210.0, 0.25, 0.95),
            AttackKind.SYN_FLOOD: (360.0, 0.34, 0.90),
            AttackKind.POWER_ATTACK: (
                (125.0, 0.20, 1.0)
                if power_failure_mode == "cyber_shutdown"
                else (0.0, 0.0, 0.88)
            ),
        }[kind]
        name_ru, description_ru = _predictive_attack_text(
            kind,
            target_id,
            power_failure_mode=power_failure_mode,
        )
        profiles.append(
            AttackProfile(
                attack_id=f"PRED-{index:02d}-{kind.value.upper()}-{target_id}",
                name_ru=name_ru,
                kind=kind,
                mitre_technique_id={
                    AttackKind.DOS: "T1498.001",
                    AttackKind.DDOS: "T1498.002",
                    AttackKind.SYN_FLOOD: "T1499.001",
                    AttackKind.POWER_ATTACK: (
                        "T1529" if power_failure_mode == "cyber_shutdown" else "T0831"
                    ),
                }[kind],
                mitre_technique_name={
                    AttackKind.DOS: "Direct Network Flood",
                    AttackKind.DDOS: "Reflection Amplification",
                    AttackKind.SYN_FLOOD: "OS Exhaustion Flood",
                    AttackKind.POWER_ATTACK: (
                        "System Shutdown/Reboot"
                        if power_failure_mode == "cyber_shutdown"
                        else "Manipulation of Control"
                    ),
                }[kind],
                related_technique_ids=(
                    ("T1498.001",)
                    if kind == AttackKind.SYN_FLOOD
                    else ("T1078", "T0826")
                    if kind == AttackKind.POWER_ATTACK
                    else ()
                ),
                target_id=target_id,
                target_type=target_type,
                ingress_nodes=ingress_nodes,
                temporal=AttackTemporalCharacteristics(
                    start_step=start_step,
                    duration_steps=duration_steps,
                    rise_steps=rise_steps,
                    pulse_shape=("power-drop" if kind == AttackKind.POWER_ATTACK else "smooth-ramp-then-peak"),
                    precursor_steps=precursor_steps,
                ),
                technical=technical,
                latency_penalty_ms=latency_penalty,
                legitimate_loss_ratio=loss_ratio,
                target_resource_pressure=pressure,
                description_ru=description_ru,
                power_failure_mode=power_failure_mode,
            )
        )
    result = tuple(profiles)
    if model is not None:
        cache = dict(getattr(model, "_predictive_attack_profile_cache", {}))
        cache[cache_key] = result
        setattr(model, "_predictive_attack_profile_cache", cache)
    return result


def _scenario_value(explicit: int | None, configured: int | None, default: int) -> int:
    """Приоритет: явный аргумент, настройка, стандартное значение."""
    if explicit is not None:
        return int(explicit)
    if configured is not None:
        return int(configured)
    return default


def _scenario_options(
    *,
    seed: int | None,
    step_seconds: int | None,
    step_count: int | None,
    config: Any | None,
) -> tuple[int, int, int]:
    config_seed = getattr(config, "attack_seed", getattr(config, "seed", None))
    return (
        _scenario_value(seed, config_seed, PREDICTIVE_DEMO_DEFAULT_SEED),
        _scenario_value(
            step_seconds, getattr(config, "step_seconds", None), PREDICTIVE_DEMO_STEP_SECONDS
        ),
        _scenario_value(
            step_count, getattr(config, "step_count", None), PREDICTIVE_DEMO_STEP_COUNT
        ),
    )


def _profiles_for_scenario(
    model: NetworkModel | None,
    scenario: str,
    *,
    seed: int,
    step_seconds: int,
    step_count: int,
    config: Any | None = None,
) -> tuple[AttackProfile, ...]:
    if scenario == "mitre-demo":
        profiles = MITRE_DEMO_ATTACKS
    elif scenario == "predictive-demo":
        profiles = predictive_demo_attack_profiles(
            model,
            seed=seed,
            step_seconds=step_seconds,
            step_count=step_count,
        )
    else:
        raise ValueError(f"Неизвестный сценарий атак: {scenario}")
    if not _custom_schedule(config) or not profiles:
        return profiles
    original_first = min(profile.temporal.start_step for profile in profiles)
    first = config.attack_start_step if config.attack_start_step is not None else original_first
    interval = config.attack_interval_steps
    adjusted = []
    for index, profile in enumerate(profiles):
        if interval is None:
            offset = profile.temporal.start_step - original_first
        else:
            offset = index * interval
        start = first + offset
        if start <= step_count:
            timing = replace(profile.temporal, start_step=start)
            adjusted.append(replace(profile, temporal=timing))
    return tuple(adjusted)


def _custom_schedule(config: Any | None) -> bool:
    return config is not None and (
        getattr(config, "attack_start_step", None) is not None
        or getattr(config, "attack_interval_steps", None) is not None
    )


def _predictive_node_pools(model: NetworkModel | None) -> dict[str, tuple[str, ...]]:
    if model is None:
        return {
            "bronze_subscribers": tuple(
                [f"M{group}_{index:02d}" for group in range(1, 4) for index in range(17, 81)]
                + [f"F{group}_{index:02d}" for group in range(1, 4) for index in range(21, 81)]
            ),
            "subscribers": tuple(
                [f"M{group}_{index:02d}" for group in range(1, 4) for index in range(1, 81)]
                + [f"F{group}_{index:02d}" for group in range(1, 4) for index in range(1, 81)]
            ),
            "core": tuple(f"C{index}" for index in range(1, 13)),
            "aggregation": tuple(f"A{index}" for index in range(1, 7)),
            "servers": ("SRV_MEDIA", "SRV_DATA", "SRV_RTC", "SRV_LIVE"),
        }

    graph = model.graph
    subscribers = sorted(node for node, attrs in graph.nodes(data=True) if attrs.get("level") == "L1")
    bronze = sorted(node for node in subscribers if graph.nodes[node].get("sla_grade") == "bronze")
    if not bronze:
        raise ValueError("В модели нет Bronze-абонентов — невозможно выбрать реальные источники атак")
    return {
        "bronze_subscribers": tuple(bronze),
        "subscribers": tuple(subscribers),
        "core": tuple(sorted(node for node, attrs in graph.nodes(data=True) if attrs.get("role") == "core-router")),
        "aggregation": tuple(
            sorted(node for node, attrs in graph.nodes(data=True) if attrs.get("role") == "aggregation-switch")
        ),
        "servers": tuple(sorted(node for node, attrs in graph.nodes(data=True) if attrs.get("role") == "service-server")),
    }


def _target_pool_for_kind(
    kind: AttackKind,
    occurrence_index: int,
    pools: dict[str, tuple[str, ...]],
    *,
    scenario_seed: int = PREDICTIVE_DEMO_DEFAULT_SEED,
) -> tuple[str, ...]:
    # Разные экземпляры одного типа атаки намеренно покрывают разные уровни сети.
    if kind == AttackKind.DOS:
        categories = (
            "core",
            "aggregation",
            "subscribers" if int(scenario_seed) % 2 == 0 else "servers",
        )
        category = categories[occurrence_index % 3]
    elif kind == AttackKind.DDOS:
        category = ("servers", "core")[occurrence_index % 2]
    elif kind == AttackKind.SYN_FLOOD:
        category = ("subscribers", "servers")[occurrence_index % 2]
    else:
        # При чётном seed демонстрируется питание сетевого узла, при нечётном
        # — сервера. Так обе допустимые области достижимы без изменения весов.
        category = ("aggregation", "servers")[(int(scenario_seed) + occurrence_index) % 2]
    candidates = pools[category]
    if not candidates:
        # Упрощённая пользовательская топология может не содержать выбранный класс.
        fallback = pools["servers"] + pools["core"] + pools["aggregation"]
        if kind == AttackKind.SYN_FLOOD:
            fallback += pools["subscribers"]
        if not fallback:
            raise ValueError(f"В модели нет допустимых целей для {kind.value}")
        return fallback
    return candidates


def _choose_prefer_unused(
    rng: random.Random,
    candidates: tuple[str, ...],
    used: set[str],
) -> str:
    available = [node for node in candidates if node not in used]
    return rng.choice(available or list(candidates))


def _choose_ingress_nodes(
    rng: random.Random,
    candidates: tuple[str, ...],
    *,
    count: int,
    excluded: set[str],
) -> tuple[str, ...]:
    available = [node for node in candidates if node not in excluded]
    if not available:
        raise ValueError("В модели нет L1-абонента, пригодного как источник атаки")
    if count <= len(available):
        return tuple(sorted(rng.sample(available, count)))
    # Для очень малой учебной топологии допускается повторный выбор, но в
    # штатной сети GNet9 все ingress_nodes уникальны и физически существуют.
    return tuple(sorted(rng.choice(available) for _ in range(count)))


def _attack_target_type(model: NetworkModel | None, target_id: str) -> str:
    if model is None:
        if target_id.startswith("C"):
            return "core-router"
        if target_id.startswith("A"):
            return "aggregation-switch"
        if target_id.startswith("RAN"):
            return "radio-access-node"
        if target_id.startswith("OLT"):
            return "optical-line-terminal"
        if target_id.startswith("SRV_"):
            return "service-server"
        return "mobile-subscriber" if target_id.startswith("M") else "fixed-subscriber"
    attrs = model.graph.nodes[target_id]
    role = attrs.get("role")
    if role in {"core-router", "aggregation-switch", "radio-access-node", "optical-line-terminal", "service-server"}:
        return str(role)
    access = str(attrs.get("access_type", "")).lower()
    return "mobile-subscriber" if access == "mobile" or target_id.startswith("M") else "fixed-subscriber"


def _subscriber_access_capacity_mbps(model: NetworkModel | None, subscriber: str) -> float:
    if model is None or subscriber not in model.graph:
        return 100.0 if subscriber.startswith("F") else 20.0
    capacities = [
        float(model.graph.edges[subscriber, neighbor].get("capacity_mbps", 0.0))
        for neighbor in model.graph.neighbors(subscriber)
    ]
    return max(capacities, default=20.0)


def _predictive_technical_profile(
    model: NetworkModel | None,
    kind: AttackKind,
    ingress_nodes: tuple[str, ...],
    *,
    power_failure_mode: str | None = None,
) -> AttackTechnicalCharacteristics:
    aggregate_access = sum(_subscriber_access_capacity_mbps(model, node) for node in ingress_nodes)
    if kind == AttackKind.POWER_ATTACK:
        cyber_shutdown = power_failure_mode == "cyber_shutdown"
        return AttackTechnicalCharacteristics(
            (
                "HTTPS/Redfish out-of-band management shutdown"
                if cyber_shutdown
                else "UPS/PDU input control and electrical telemetry"
            ),
            0,
            0.0,
            0,
            1,
            1.0,
            0.0,
            "capacity_bounded_lab_scenario_assumption",
            resource_axis=(
                "server_shutdown_and_boot_health_recovery"
                if cyber_shutdown
                else "local_dual_feed_input_brownout_with_ups_ride_through"
            ),
            traffic_origin_semantics=(
                "out_of_band_management_plane_after_valid_account_compromise"
                if cyber_shutdown
                else "facility_control_plane_or_physical_feed_disturbance_not_subscriber_traffic"
            ),
            initiator_count=1,
        )
    if kind == AttackKind.DOS:
        packet_size = 512
        rate = min(850.0, aggregate_access * 0.88)
        protocol = "UDP"
        amplification = 1.0
        incomplete = 0.0
    elif kind == AttackKind.DDOS:
        packet_size = 768
        amplification = 40.0
        rate = min(40_000.0, aggregate_access * 0.82 * amplification)
        protocol = "UDP reflection/amplification"
        incomplete = 0.0
    else:
        packet_size = 40
        rate = min(600.0, aggregate_access * 0.72)
        protocol = "TCP SYN"
        amplification = 1.0
        incomplete = 0.998
    on_wire_size = 84 if kind == AttackKind.SYN_FLOOD else packet_size + 66
    # Для UDP к полезной нагрузке добавляются IPv4/UDP, Ethernet header/FCS,
    # preamble/SFD и IFG (66 байт line-time). Для SYN минимальный Ethernet
    # кадр с padding занимает 84 байта времени линии, а не только 40 байт L3.
    packet_rate = int(rate * 1_000_000.0 / max(on_wire_size * 8, 1))
    victim_source_count = max(320, len(ingress_nodes) * 80) if kind == AttackKind.DDOS else len(ingress_nodes)
    return AttackTechnicalCharacteristics(
        protocol,
        packet_size,
        round(rate, 4),
        packet_rate,
        victim_source_count,
        amplification,
        incomplete,
        "capacity_bounded_lab_scenario_assumption",
        on_wire_size_bytes=on_wire_size,
        resource_axis=(
            "tcp_syn_backlog_and_connection_state"
            if kind == AttackKind.SYN_FLOOD
            else "target_control_plane_or_application_processing"
            if kind == AttackKind.DOS
            else "converged_network_bandwidth_and_target_processing"
        ),
        traffic_origin_semantics=(
            "listed_L1_nodes_initiate_spoofed_requests; external_reflectors_return_amplified_traffic"
            if kind == AttackKind.DDOS
            else "direct_from_listed_ingress_nodes"
        ),
        initiator_count=len(ingress_nodes),
        external_reflector_count=(victim_source_count if kind == AttackKind.DDOS else 0),
    )


def _predictive_attack_text(
    kind: AttackKind,
    target_id: str,
    *,
    power_failure_mode: str | None = None,
) -> tuple[str, str]:
    if kind == AttackKind.DOS:
        return (
            f"DoS от скомпрометированного абонента на {target_id}",
            "Один реальный Bronze-абонент исчерпывает доступную ему полосу прямым UDP-флудом.",
        )
    if kind == AttackKind.DDOS:
        return (
            f"DDoS от группы скомпрометированных абонентов на {target_id}",
            "Несколько скомпрометированных L1-абонентов инициируют подменённые запросы; "
            "внешние отражатели формируют усиленный UDP-поток к цели (MITRE T1498.002).",
        )
    if kind == AttackKind.SYN_FLOOD:
        return (
            f"SYN-флуд на {target_id}",
            "Реальные L1-источники создают незавершённые TCP-соединения и давление на очередь SYN.",
        )
    if power_failure_mode == "cyber_shutdown":
        return (
            f"Удалённое выключение {target_id}",
            "Команда через изолированную плоскость управления не вызывает просадку "
            "входа ИБП, но после неё требуется загрузка и проверка готовности.",
        )
    return (
        f"Краткий провал питания перед ИБП у {target_id}",
        "Локальный провал обоих входов короче автономности ИБП: выход удерживается "
        "батареей, после чего заряд постепенно восстанавливается.",
    )


def mark_critical_nodes(model: NetworkModel) -> dict[str, Any]:
    """Построить иерархию КВУ по фактически обслуживаемым Gold-абонентам.

    Для каждого Gold-абонента берётся его безопасный путь данных до сервиса.
    L2-узел получает один счётчик за абонента, если обслуживает его как точка
    доступа или находится на его маршруте.  Поэтому ``gold_subscriber_count``
    означает не общую степень графа, а число Gold-подписок, для которых этот
    маршрутизатор/коммутатор находится в рабочем пути.  Максимальный счётчик
    образует КВУ первого ранга; остальные ранги делают приоритет прозрачным.
    Это приоритет защиты для арбитра, а не самостоятельная команда ремаппинга.
    """
    gold_subscribers = [
        node for node, attrs in model.graph.nodes(data=True)
        if attrs.get("level") == "L1" and attrs.get("sla_grade") == "gold"
    ]
    transit_counts = {node: 0 for node in model.graph.nodes}
    gold_subscriber_counts = {node: 0 for node in model.graph.nodes}
    direct_gold_subscriber_counts = {node: 0 for node in model.graph.nodes}
    service_counts = {node: 0 for node in model.graph.nodes}
    recovery_reserve_counts = {node: 0 for node in model.graph.nodes}
    routes: list[list[str]] = []
    for subscriber in gold_subscribers:
        attrs = model.graph.nodes[subscriber]
        home_access = str(attrs.get("home_access", ""))
        if home_access in direct_gold_subscriber_counts:
            direct_gold_subscriber_counts[home_access] += 1
        service = TRAFFIC_SERVICE_NODES.get(attrs.get("traffic_kind"))
        if service not in model.graph:
            continue
        server = model.graph.nodes[service].get("hosted_on", service)
        route = shortest_data_path(model, subscriber, server)
        routes.append(route)
        for node in route:
            transit_counts[node] += 1
            if model.graph.nodes[node].get("level") == "L2":
                gold_subscriber_counts[node] += 1
        service_counts[service] += 1
        service_counts[server] += 1
        for standby in model.graph.nodes[service].get("standby_hosts", []):
            if standby in recovery_reserve_counts:
                recovery_reserve_counts[standby] += 1

    denominator = max(len(routes), 1)
    l2_ranked = sorted(
        (
            node
            for node, attrs in model.graph.nodes(data=True)
            if attrs.get("level") == "L2" and gold_subscriber_counts[node] > 0
        ),
        key=lambda node: (-gold_subscriber_counts[node], -transit_counts[node], str(node)),
    )
    rank_by_node = {node: index + 1 for index, node in enumerate(l2_ranked)}
    highest_gold_subscriber_count = max(
        (gold_subscriber_counts[node] for node in l2_ranked), default=0
    )

    def kvu_tier(node: str) -> str | None:
        rank = rank_by_node.get(node)
        if rank is None:
            return None
        if gold_subscriber_counts[node] == highest_gold_subscriber_count:
            return "K1"
        if rank <= max(2, math.ceil(len(l2_ranked) * 0.20)):
            return "K2"
        if rank <= max(4, math.ceil(len(l2_ranked) * 0.50)):
            return "K3"
        return "K4"

    critical_nodes: list[str] = []
    for node, attrs in model.graph.nodes(data=True):
        transit = transit_counts[node]
        hosted = service_counts[node]
        reserve = recovery_reserve_counts[node]
        is_gold_endpoint = attrs.get("level") == "L1" and attrs.get("sla_grade") == "gold"
        is_critical = transit > 0 or hosted > 0 or reserve > 0 or is_gold_endpoint
        involvement = min(
            1.0,
            transit / denominator
            + (0.20 if hosted else 0.0)
            + 0.15 * reserve / denominator,
        )
        network_gold_count = gold_subscriber_counts[node]
        rank = rank_by_node.get(node)
        attrs["critical_protection"] = {
            "is_critical": is_critical,
            "protection_priority": 1 if is_critical else 3,
            "gold_transit_flow_count": transit,
            "gold_subscriber_count": network_gold_count,
            "direct_gold_subscriber_count": direct_gold_subscriber_counts[node],
            "kvu_rank": rank,
            "kvu_tier": kvu_tier(node),
            "is_kvu": bool(
                attrs.get("level") == "L2"
                and network_gold_count == highest_gold_subscriber_count
                and network_gold_count > 0
            ),
            "kvu_definition": (
                "L2 node ranked by number of Gold subscribers whose access or data path it serves"
            ),
            "gold_service_flow_count": hosted,
            "gold_recovery_reserve_flow_count": reserve,
            "critical_involvement_coefficient": round(involvement, 6),
            "maximum_safe_utilization_percent": 80.0 if is_critical else 90.0,
            "policy": "do_not_accept_silver_or_bronze_remap_above_80_percent" if is_critical else "standard",
        }
        if is_critical:
            critical_nodes.append(node)
    return {
        "gold_subscriber_count": len(gold_subscribers),
        "gold_route_count": len(routes),
        "kvu_count": sum(
            1
            for node in l2_ranked
            if gold_subscriber_counts[node] == highest_gold_subscriber_count
        ),
        "kvu_hierarchy": [
            {
                "node_id": node,
                "role": model.graph.nodes[node].get("role"),
                "gold_subscriber_count": gold_subscriber_counts[node],
                "direct_gold_subscriber_count": direct_gold_subscriber_counts[node],
                "gold_transit_flow_count": transit_counts[node],
                "kvu_rank": rank_by_node[node],
                "kvu_tier": kvu_tier(node),
                "is_kvu": gold_subscriber_counts[node] == highest_gold_subscriber_count,
            }
            for node in l2_ranked
        ],
        "critical_node_count": len(critical_nodes),
        "critical_nodes": sorted(critical_nodes),
    }


def _attack_capacity_context(
    model: NetworkModel,
    profile: AttackProfile,
    routes: list[list[str]],
) -> dict[str, Any]:
    """Relate offered attack load to physical ingress and target resources.

    A bandwidth flood and a state-exhaustion attack are deliberately not
    interchangeable.  The former is normalized by unique final-hop capacity;
    the latter by an explicitly labelled target PPS/state budget.  This keeps
    a small subscriber DoS from pretending to saturate an 800-Gbit/s chassis.
    """
    route_bottlenecks: list[float] = []
    final_hops: set[tuple[str, str]] = set()
    for route in routes:
        capacities: list[float] = []
        for source, target in zip(route, route[1:]):
            attrs = model.graph.edges[source, target]
            if attrs.get("medium") == "logical-service-binding":
                continue
            capacities.append(float(attrs.get("capacity_mbps", 0.0)))
        if capacities:
            route_bottlenecks.append(min(capacities))
        if len(route) >= 2:
            edge = tuple(sorted((str(route[-2]), str(route[-1]))))
            if model.graph.edges[route[-2], route[-1]].get("medium") != "logical-service-binding":
                final_hops.add(edge)

    convergence_capacity = sum(
        float(model.graph.edges[source, target].get("capacity_mbps", 0.0))
        for source, target in final_hops
    )
    if convergence_capacity <= 0.0:
        convergence_capacity = sum(route_bottlenecks) or 1.0

    offered_rate = float(profile.technical.offered_rate_mbps)
    packet_rate = float(profile.technical.packet_rate_pps)
    bandwidth_ratio = offered_rate / convergence_capacity
    target_type = profile.target_type
    if profile.kind == AttackKind.POWER_ATTACK:
        processing_ratio = 0.0
        peak_pressure = float(profile.target_resource_pressure)
        pressure_basis = f"stateful_{profile.power_failure_mode or 'power_event'}_scenario"
    elif profile.kind == AttackKind.BRUTE_FORCE:
        processing_ratio = 0.0
        peak_pressure = float(profile.target_resource_pressure)
        pressure_basis = "slow_authentication_attempts_below_data_plane_saturation"
    elif profile.kind == AttackKind.SYN_FLOOD:
        budget = float(TARGET_SYN_STATE_BUDGET_PPS[target_type])
        processing_ratio = packet_rate / budget
        peak_pressure = min(1.0, max(bandwidth_ratio, processing_ratio))
        pressure_basis = "max(path_bandwidth_ratio,tcp_syn_state_rate_ratio)"
    elif profile.kind == AttackKind.DOS:
        budget = float(TARGET_SINGLE_SOURCE_DOS_BUDGET_PPS[target_type])
        processing_ratio = packet_rate / budget
        peak_pressure = min(1.0, max(bandwidth_ratio, processing_ratio))
        pressure_basis = "max(path_bandwidth_ratio,single_source_target_request_rate_ratio)"
    else:
        budget = float(TARGET_PROCESSING_BUDGET_PPS[target_type])
        processing_ratio = packet_rate / budget
        peak_pressure = min(1.0, max(bandwidth_ratio, processing_ratio))
        pressure_basis = "max(path_bandwidth_ratio,target_processing_rate_ratio)"

    return {
        "convergence_capacity_mbps": round(convergence_capacity, 4),
        "route_bottleneck_mbps": [round(value, 4) for value in route_bottlenecks],
        "peak_bandwidth_load_ratio": round(bandwidth_ratio, 6),
        "peak_target_processing_load_ratio": round(processing_ratio, 6),
        "target_resource_pressure": round(peak_pressure, 6),
        "target_resource_pressure_basis": pressure_basis,
        "target_resource_budget_origin": "gnet9_calibrated_scenario_assumption",
    }


def _reflection_route_context(
    model: NetworkModel,
    profile: AttackProfile,
) -> dict[str, Any]:
    """Разделить исходящие spoofed-запросы и входящие отражённые ответы.

    Внешние reflectors не добавляются как фиктивные вершины внутреннего графа:
    response route начинается в реально наблюдаемой точке ingress оператора.
    Внеоператорский сегмент сохраняется как семантика события.
    """
    core_nodes = sorted(
        node
        for node, attrs in model.graph.nodes(data=True)
        if attrs.get("role") == "core-router" and node != profile.target_id
    )
    preferred = [
        node
        for node in REFLECTION_TRANSIT_INGRESS_CANDIDATES
        if node in core_nodes and has_data_path(model, node, profile.target_id)
    ]
    fallback = [
        node
        for node in core_nodes
        if node not in preferred and has_data_path(model, node, profile.target_id)
    ]
    boundary_nodes = (preferred + fallback)[: min(3, len(preferred + fallback))]
    if not boundary_nodes:
        raise ValueError(
            f"Для отражённого DDoS на {profile.target_id} нет внешней точки ingress"
        )

    request_routes = [
        shortest_data_path(
            model,
            initiator,
            boundary_nodes[index % len(boundary_nodes)],
        )
        for index, initiator in enumerate(profile.ingress_nodes)
    ]
    response_routes = [
        shortest_data_path(
            model,
            ingress,
            profile.target_id,
        )
        for ingress in boundary_nodes
    ]
    amplification = max(float(profile.technical.amplification_factor), 1.0)
    return {
        "routes": response_routes,
        "request_routes": request_routes,
        "response_routes": response_routes,
        "external_response_ingress_nodes": boundary_nodes,
        "reflection_route_semantics": (
            "request_routes_are_spoofed_queries_from_L1_to_operator_egress; "
            "response_routes_are_amplified_external_responses_from_operator_ingress_to_victim"
        ),
        "external_segment_modeled": False,
        "external_segment_limitation": (
            "reflector_to_operator_path_is_outside_the_internal_gnet9_graph"
        ),
        "request_offered_rate_mbps": round(
            float(profile.technical.offered_rate_mbps) / amplification,
            6,
        ),
        "response_offered_rate_mbps": round(
            float(profile.technical.offered_rate_mbps),
            6,
        ),
    }


def active_attack_events(
    model: NetworkModel,
    scenario: str,
    *,
    step_index: int,
    time_seconds: int,
    step_seconds: int,
    step_count: int | None = None,
    seed: int | None = None,
    config: Any | None = None,
) -> list[dict[str, Any]]:
    if scenario == "none":
        return []
    resolved_seed, resolved_step_seconds, resolved_step_count = _scenario_options(
        seed=seed,
        step_seconds=step_seconds,
        step_count=step_count,
        config=config,
    )
    profiles = _profiles_for_scenario(
        model,
        scenario,
        seed=resolved_seed,
        step_seconds=resolved_step_seconds,
        step_count=resolved_step_count,
        config=config,
    )
    arrival_schedule = (
        predictive_demo_weibull_schedule(
            seed=resolved_seed,
            step_seconds=resolved_step_seconds,
            step_count=resolved_step_count,
        )
        if scenario == "predictive-demo" and not _custom_schedule(config)
        else None
    )

    events: list[dict[str, Any]] = []
    for realization_index, profile in enumerate(profiles):
        relative_step = step_index - profile.temporal.start_step
        if relative_step < 0 or relative_step >= profile.temporal.duration_steps:
            continue
        intensity = min(1.0, (relative_step + 1) / max(profile.temporal.rise_steps + 1, 1) + 0.25)
        intensity = round(intensity, 4)
        if profile.kind == AttackKind.DDOS:
            route_context = _reflection_route_context(model, profile)
            routes = route_context["response_routes"]
        else:
            routes = [
                shortest_data_path(model, ingress, profile.target_id)
                for ingress in profile.ingress_nodes
            ]
            route_context = {
                "routes": routes,
                "request_routes": [],
                "response_routes": routes,
                "reflection_route_semantics": "not_a_reflection_attack",
                "external_segment_modeled": True,
            }
        capacity_context = _attack_capacity_context(model, profile, routes)
        target_criticality = model.graph.nodes[profile.target_id].get("critical_protection", {})
        event = {
            **profile.to_dict(),
            "step_index": step_index,
            "time_seconds": time_seconds,
            "interval_seconds": resolved_step_seconds,
            "relative_step": relative_step,
            "intensity_ratio": intensity,
            **route_context,
            "target_criticality": target_criticality,
            "target_device_context": target_device_context(model, profile.target_id),
            "scenario_peak_pressure_prior": profile.target_resource_pressure,
            **capacity_context,
            "effective_offered_rate_mbps": round(profile.technical.offered_rate_mbps * intensity, 4),
            "effective_packet_rate_pps": int(profile.technical.packet_rate_pps * intensity),
        }
        if profile.kind == AttackKind.BRUTE_FORCE:
            event["effective_authentication_attempt_rate_per_second"] = round(
                float(profile.technical.authentication_attempt_rate_per_second or 0.0)
                * intensity,
                6,
            )
            event["effective_failed_authentication_attempt_count"] = round(
                float(event["effective_authentication_attempt_rate_per_second"])
                * resolved_step_seconds
                * float(profile.technical.authentication_failure_ratio),
                6,
            )
        if profile.kind == AttackKind.DDOS:
            event["effective_request_rate_mbps"] = round(
                float(route_context["request_offered_rate_mbps"]) * intensity,
                6,
            )
            event["effective_response_rate_mbps"] = event["effective_offered_rate_mbps"]
        if scenario == "predictive-demo":
            event["scenario_weight"] = 1.0 / max(len(profiles), 1)
            event["scenario_weight_semantics"] = "realized_event_mass"
            event["kind_weight"] = PREDICTIVE_DEMO_KIND_WEIGHTS[profile.kind.value]
            event["scenario_weight_origin"] = "scenario_assumption"
            event["scenario_seed"] = resolved_seed
            event["time_scale_semantics"] = "accelerated_demo_not_real_incident_duration"
            if arrival_schedule is not None:
                # В рабочем состоянии публикуется только уже состоявшийся
                # интервал. Массив будущих start_steps остаётся в каталоге
                # испытания и не может стать признаком прогнозатора.
                event["arrival_process"] = {
                    "distribution": arrival_schedule.distribution,
                    "shape": arrival_schedule.shape,
                    "scale_seconds": arrival_schedule.scale_seconds,
                    "expected_interarrival_seconds": (
                        arrival_schedule.expected_interarrival_seconds
                    ),
                    "nominal_intensity_events_per_second": (
                        arrival_schedule.nominal_intensity_events_per_second
                    ),
                    "conditioning": arrival_schedule.conditioning,
                    "value_origin": arrival_schedule.value_origin,
                }
                event["arrival_realization"] = {
                    "index": realization_index,
                    "elapsed_interarrival_seconds": (
                        arrival_schedule.realized_interarrival_seconds[realization_index]
                    ),
                    "sampled_interarrival_seconds": (
                        arrival_schedule.raw_interarrival_seconds[realization_index]
                    ),
                    "hazard_per_second": (
                        arrival_schedule.realized_hazard_per_second[realization_index]
                    ),
                    "survival_probability": (
                        arrival_schedule.realized_survival_probability[realization_index]
                    ),
                    "observability": "available_only_after_attack_onset",
                }
        events.append(event)
    return events


def advance_power_runtime_state(
    model: NetworkModel,
    events: list[dict[str, Any]],
    runtime_state: dict[str, dict[str, Any]] | None,
    *,
    step_index: int,
    time_seconds: int,
    step_seconds: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Apply causal UPS ride-through and stateful shutdown recovery.

    ``events`` contains only the scenario pulse for the current step.  The
    mutable ``runtime_state`` belongs to one dynamics run and carries reboot or
    battery recharge into later snapshots without mutating the ideal t0 graph.
    """
    state = runtime_state if runtime_state is not None else {}
    effective_events = [
        dict(event)
        for event in events
        if event.get("kind") != AttackKind.POWER_ATTACK.value
    ]
    active_power_targets: set[str] = set()
    public_state: dict[str, dict[str, Any]] = {}

    for source_event in events:
        if source_event.get("kind") != AttackKind.POWER_ATTACK.value:
            continue
        event = dict(source_event)
        target_id = str(event["target_id"])
        active_power_targets.add(target_id)
        attrs = model.graph.nodes[target_id]
        architecture = attrs.get("power_architecture", {})
        failure_mode = str(event.get("power_failure_mode") or "cyber_shutdown")
        baseline_reserve = _power_reserve_ratio(attrs)
        autonomy_seconds = max(
            1.0,
            float(architecture.get("ups_backup_autonomy_hours", 0.0)) * 3_600.0,
        )
        attack_id = str(event["attack_id"])
        entry = state.get(target_id)
        if entry is None or entry.get("attack_id") != attack_id:
            entry = {
                "attack_id": attack_id,
                "target_id": target_id,
                "target_type": event.get("target_type"),
                "failure_mode": failure_mode,
                "baseline_energy_reserve_ratio": baseline_reserve,
                "minimum_energy_reserve_ratio": baseline_reserve,
                "autonomy_seconds": autonomy_seconds,
                "active_duration_steps": int(event.get("temporal", {}).get("duration_steps", 1)),
                "recovery_started_step": None,
                "template_event": dict(event),
            }
            state[target_id] = entry
        entry["last_active_step"] = step_index
        entry["template_event"] = dict(event)

        if failure_mode == "brownout_feed_loss":
            active_elapsed_seconds = min(
                autonomy_seconds,
                (int(event.get("relative_step", 0)) + 1) * max(step_seconds, 1),
            )
            reserve = max(0.0, baseline_reserve - active_elapsed_seconds / autonomy_seconds)
            entry["minimum_energy_reserve_ratio"] = min(
                float(entry.get("minimum_energy_reserve_ratio", baseline_reserve)),
                reserve,
            )
            input_voltage_ratio = max(
                0.75,
                1.0 - 0.14 * float(event.get("intensity_ratio", 1.0)),
            )
            runtime = _power_runtime_view(
                architecture,
                entry,
                phase="ups_ride_through",
                input_voltage_ratio=input_voltage_ratio,
                service_availability_ratio=1.0,
                power_output_availability_ratio=1.0,
                energy_reserve_ratio=reserve,
                ups_on_battery=True,
                recovery_progress_ratio=0.0,
                recovery_duration_seconds=POWER_BROWNOUT_RECHARGE_SECONDS,
                device_state="online_on_ups",
            )
            event["power_service_impact_ratio"] = 0.0
            event["power_failure_semantics"] = "input_brownout_absorbed_by_ups_ride_through"
        else:
            runtime = _power_runtime_view(
                architecture,
                entry,
                phase="cyber_shutdown",
                input_voltage_ratio=1.0,
                service_availability_ratio=0.0,
                power_output_availability_ratio=1.0,
                energy_reserve_ratio=baseline_reserve,
                ups_on_battery=False,
                recovery_progress_ratio=0.0,
                recovery_duration_seconds=POWER_CYBER_RECOVERY_SECONDS.get(
                    str(event.get("target_type")), 45
                ),
                device_state="shutdown_command_executed",
            )
            # A successful shutdown is a discrete state change, not a gradual
            # voltage sag.  The scenario pulse controls when it happens; the
            # following boot/health-check remains in runtime_state.
            event["intensity_ratio"] = 1.0
            event["target_resource_pressure"] = 1.0
            event["power_service_impact_ratio"] = 1.0
            event["power_failure_semantics"] = "authorized_cyber_shutdown_with_normal_ups_input"

        entry["runtime"] = runtime
        event["event_phase"] = "active_impact"
        event["is_recovery_effect"] = False
        event["power_runtime"] = runtime
        effective_events.append(event)
        public_state[target_id] = runtime

    completed_targets: list[str] = []
    for target_id, entry in list(state.items()):
        if target_id in active_power_targets:
            continue
        architecture = model.graph.nodes[target_id].get("power_architecture", {})
        recovery_started_step = entry.get("recovery_started_step")
        if recovery_started_step is None:
            recovery_started_step = step_index
            entry["recovery_started_step"] = step_index
        elapsed_seconds = (step_index - int(recovery_started_step) + 1) * max(step_seconds, 1)
        failure_mode = str(entry.get("failure_mode"))

        if failure_mode == "brownout_feed_loss":
            recovery_seconds = POWER_BROWNOUT_RECHARGE_SECONDS
            progress = min(1.0, elapsed_seconds / max(recovery_seconds, 1))
            depleted = float(entry.get("minimum_energy_reserve_ratio", 0.0))
            baseline = float(entry.get("baseline_energy_reserve_ratio", depleted))
            reserve = depleted + (baseline - depleted) * progress
            runtime = _power_runtime_view(
                architecture,
                entry,
                phase="battery_recharge" if progress < 1.0 else "recovered",
                input_voltage_ratio=1.0,
                service_availability_ratio=1.0,
                power_output_availability_ratio=1.0,
                energy_reserve_ratio=reserve,
                ups_on_battery=False,
                recovery_progress_ratio=progress,
                recovery_duration_seconds=recovery_seconds,
                device_state="online_recharging" if progress < 1.0 else "online",
            )
            entry["runtime"] = runtime
            public_state[target_id] = runtime
            if progress >= 1.0:
                completed_targets.append(target_id)
            continue

        recovery_seconds = POWER_CYBER_RECOVERY_SECONDS.get(
            str(entry.get("target_type")), 45
        )
        progress = min(1.0, elapsed_seconds / max(recovery_seconds, 1))
        # Boot is unavailable for most of the interval.  The final 40% models
        # readiness/health checks rather than pretending that a server or router
        # instantly returns when the malicious command ends.
        service_availability = min(1.0, max(0.0, (progress - 0.60) / 0.40))
        phase = (
            "rebooting"
            if progress < 0.60
            else "health_check"
            if progress < 1.0
            else "recovered"
        )
        runtime = _power_runtime_view(
            architecture,
            entry,
            phase=phase,
            input_voltage_ratio=1.0,
            service_availability_ratio=service_availability,
            power_output_availability_ratio=1.0,
            energy_reserve_ratio=float(entry.get("baseline_energy_reserve_ratio", 1.0)),
            ups_on_battery=False,
            recovery_progress_ratio=progress,
            recovery_duration_seconds=recovery_seconds,
            device_state=phase,
        )
        entry["runtime"] = runtime
        public_state[target_id] = runtime
        if progress >= 1.0:
            completed_targets.append(target_id)
            continue

        recovery_event = dict(entry["template_event"])
        recovery_step = step_index - int(recovery_started_step) + 1
        recovery_event.update(
            name_ru=f"Восстановление {target_id} после удалённого выключения",
            description_ru="Загрузка и проверка готовности после T1529; вход ИБП остаётся штатным.",
            relative_step=int(entry.get("active_duration_steps", 1)) + recovery_step - 1,
            intensity_ratio=round(1.0 - service_availability, 6),
            target_resource_pressure=1.0,
            effective_offered_rate_mbps=0.0,
            effective_packet_rate_pps=0,
            event_phase="recovery_after_effect",
            is_recovery_effect=True,
            power_service_impact_ratio=1.0,
            power_failure_semantics="stateful_boot_and_health_check_after_cyber_shutdown",
            power_runtime=runtime,
            step_index=step_index,
            time_seconds=time_seconds,
            interval_seconds=step_seconds,
            routes=[],
        )
        effective_events.append(recovery_event)

    # Publish the final recovered sample once, then forget it before the next
    # transition.  The ideal graph itself was never mutated.
    for target_id in completed_targets:
        state.pop(target_id, None)

    return effective_events, public_state


def _power_runtime_view(
    architecture: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    phase: str,
    input_voltage_ratio: float,
    service_availability_ratio: float,
    power_output_availability_ratio: float,
    energy_reserve_ratio: float,
    ups_on_battery: bool,
    recovery_progress_ratio: float,
    recovery_duration_seconds: int,
    device_state: str,
) -> dict[str, Any]:
    baseline = float(entry.get("baseline_energy_reserve_ratio", energy_reserve_ratio))
    return {
        "attack_id": entry.get("attack_id"),
        "target_id": entry.get("target_id"),
        "target_type": entry.get("target_type"),
        "failure_mode": entry.get("failure_mode"),
        "phase": phase,
        "device_state": device_state,
        "input_voltage_ratio": round(max(0.0, min(1.0, input_voltage_ratio)), 6),
        "power_output_availability_ratio": round(
            max(0.0, min(1.0, power_output_availability_ratio)), 6
        ),
        "service_availability_ratio": round(
            max(0.0, min(1.0, service_availability_ratio)), 6
        ),
        "energy_reserve_ratio": round(max(0.0, min(1.0, energy_reserve_ratio)), 6),
        "energy_depletion_ratio": round(max(0.0, baseline - energy_reserve_ratio), 6),
        "ups_on_battery": bool(ups_on_battery),
        "ride_through_active": bool(ups_on_battery and service_availability_ratio >= 1.0),
        "recovery_progress_ratio": round(max(0.0, min(1.0, recovery_progress_ratio)), 6),
        "recovery_duration_seconds": int(recovery_duration_seconds),
        "site_power_domain_id": architecture.get("site_power_domain_id"),
        "local_fault_domain_id": architecture.get("local_fault_domain_id"),
        "feed_domains": list(architecture.get("feed_domains", [])),
        "ups_domains": list(architecture.get("ups_domains", [])),
        "state_origin": "gnet9_stateful_power_scenario",
    }


def observe_attack_precursors(
    model: NetworkModel,
    scenario: str,
    *,
    step_index: int,
    time_seconds: int,
    step_seconds: int,
    step_count: int | None = None,
    seed: int | None = None,
    config: Any | None = None,
) -> list[dict[str, Any]]:
    """Вернуть только наблюдаемую телеметрию до атаки, без будущего трафика.

    В демонстрационном сценарии в течение причинного окна постепенно появляются
    признаки, которые в реальной сети пришли бы от NetFlow/IDS/очередей/UPS:
    рост частоты обращений, числа источников, SYN-backlog или аномалия
    управляющего канала питания. Скрытое расписание атаки в вектор Купмана не
    передаётся.
    """
    if scenario == "none":
        return []
    resolved_seed, resolved_step_seconds, resolved_step_count = _scenario_options(
        seed=seed,
        step_seconds=step_seconds,
        step_count=step_count,
        config=config,
    )
    profiles = _profiles_for_scenario(
        model,
        scenario,
        seed=resolved_seed,
        step_seconds=resolved_step_seconds,
        step_count=resolved_step_count,
        config=config,
    )

    observations: list[dict[str, Any]] = []
    for profile_index, profile in enumerate(profiles, start=1):
        lead_steps = profile.temporal.start_step - step_index
        if lead_steps < 1 or lead_steps > profile.temporal.precursor_steps:
            continue
        maturity = (profile.temporal.precursor_steps - lead_steps + 1) / max(
            profile.temporal.precursor_steps,
            1,
        )
        signals = _precursor_signals(profile, maturity=maturity)
        common = {
            "observation_type": "pre_attack_telemetry",
            "observed_at_step": step_index,
            "observed_at_seconds": time_seconds,
            "signal_confidence": signals.pop("signal_confidence"),
            "signals": signals,
            "value_origin": "scenario_assumption",
        }
        if scenario == "predictive-demo":
            # Только сырые показания известного датчика. Здесь намеренно нет
            # attack_id, типа/начала атаки, горизонта и предполагаемой цели.
            # Даже correlation_id не кодирует цель или профиль испытания.
            observations.append(
                {
                    **common,
                    "correlation_id": f"TEL-{resolved_seed}-{step_index}-{profile_index}",
                    # Идентификатор означает место фактического измерения
                    # аномальной очереди/NetFlow/UPS, а не скрытую «будущую цель».
                    # Именно поэтому контроллер может локализовать область риска,
                    # не получая start_step, тип атаки или attack_id.
                    "sensor_node_id": profile.target_id,
                    # На слабой ранней ступени NetFlow видит лишь часть
                    # источников. Полный будущий набор ingress не раскрывается.
                    "observed_source_node_ids": list(profile.ingress_nodes)[
                        : min(
                            len(profile.ingress_nodes),
                            max(1, int(signals.get("suspicious_source_count", 1))),
                        )
                    ],
                    "description_ru": (
                        "Наблюдаемая локальная аномалия датчика: частота, энтропия источников, "
                        "рост очереди и состояние управления питанием. Интерпретацию и прогноз "
                        "самостоятельно выполняет анализатор Купмана."
                    ),
                }
            )
        else:
            observations.append(
                {
                    **common,
                    "correlation_id": f"TEL-{step_index}-{profile.target_id}",
                    "sensor_node_id": profile.target_id,
                    "suspected_target_id": profile.target_id,
                    "observed_source_node_ids": list(profile.ingress_nodes),
                    "target_type": profile.target_type,
                    "description_ru": (
                        "Наблюдаемые предвестники; активная атака и вредоносный "
                        "поток на этом шаге ещё не сформированы. Тип воздействия "
                        "определяется анализатором по сигналам, а не читается из расписания."
                    ),
                }
            )
    return observations


def apply_attack_effects(
    model: NetworkModel,
    legitimate_flows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    precursor_events: list[dict[str, Any]] | None = None,
    defense_plan: dict[str, Any] | None = None,
    identities: Mapping[str, Any] | None = None,
    scenario: str | None = None,
    power_runtime_state: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ухудшить затронутые легитимные потоки и добавить агрегированные flood-потоки."""
    precursor_events = precursor_events or []
    flows = [dict(flow) for flow in legitimate_flows]
    for flow in flows:
        flow["is_attack_traffic"] = False
        flow["pre_attack_dropped_packets"] = int(
            flow.get("observed_dropped_packets", 0)
        )
        flow["raw_attack_dropped_packets"] = 0
        flow["prevented_dropped_packets"] = 0
        flow["preventive_mitigation_ratio"] = 0.0
        flow["attack_latency_penalty_ms"] = 0.0

    event_mitigation = {
        event["attack_id"]: _event_defense_mitigation(defense_plan, event)
        for event in events
    }
    flow_mitigation, allocation_reports = _allocate_sla_defense_budget(
        flows,
        events,
        event_mitigation,
    )

    for flow in flows:
        for event in events:
            if not _flow_is_affected(flow, event):
                continue
            if (
                event.get("kind") == AttackKind.POWER_ATTACK.value
                and float(event.get("power_service_impact_ratio", 1.0)) <= 0.0
            ):
                # A short input sag held by the UPS is observable telemetry but
                # must not manufacture packet loss or an outage.
                continue
            intensity = float(event["intensity_ratio"])
            resource_pressure = float(event["target_resource_pressure"])
            raw_loss_ratio = min(
                0.95,
                float(event["legitimate_loss_ratio"]) * intensity * resource_pressure,
            )
            mitigation = flow_mitigation.get((event["attack_id"], str(flow["flow_id"])), 0.0)
            loss_ratio = raw_loss_ratio * (1.0 - mitigation)
            raw_dropped = min(int(flow["packet_count"]), math.ceil(int(flow["packet_count"]) * raw_loss_ratio))
            raw_already_dropped = int(flow.get("raw_attack_dropped_packets", 0))
            raw_increment = min(
                max(0, int(flow["packet_count"]) - raw_already_dropped),
                raw_dropped,
            )
            already_dropped = int(flow.get("observed_dropped_packets", 0))
            dropped = min(
                max(0, int(flow["packet_count"]) - already_dropped),
                math.ceil(int(flow["packet_count"]) * loss_ratio),
            )
            flow["observed_dropped_packets"] = already_dropped + dropped
            flow["raw_attack_dropped_packets"] = raw_already_dropped + raw_increment
            flow["prevented_dropped_packets"] += max(0, raw_increment - dropped)
            flow["preventive_mitigation_ratio"] = max(float(flow["preventive_mitigation_ratio"]), mitigation)
            if flow.get("transport") == "TCP":
                flow["observed_retransmissions"] = int(flow.get("observed_retransmissions", 0)) + dropped
            latency_penalty_ms = (
                float(event["latency_penalty_ms"])
                * intensity
                * resource_pressure
                * (1.0 - mitigation * 0.85)
            )
            flow["attack_latency_penalty_ms"] = round(
                float(flow.get("attack_latency_penalty_ms", 0.0))
                + latency_penalty_ms,
                4,
            )
            flow["one_way_latency_ms"] = round(
                float(flow.get("one_way_latency_ms", 0.0)) + latency_penalty_ms,
                4,
            )
            if flow.get("rtt_ms") is not None:
                flow["rtt_ms"] = round(
                    float(flow["rtt_ms"])
                    + 2.0 * latency_penalty_ms,
                    4,
                )
            flow.setdefault("attack_ids", []).append(event["attack_id"])
            flow["attack_impacted"] = True

    attack_flows = [
        flow
        for event in events
        for flow in _build_attack_flows(
            event,
            mitigation_ratio=event_mitigation[event["attack_id"]],
            quarantined_sources=_active_quarantined_sources(
                defense_plan,
                event,
                mitigation_ratio=event_mitigation[event["attack_id"]],
            ),
            identities=identities,
        )
    ]
    apply_transport_queues(model, flows)
    all_flows = flows + attack_flows
    for flow in flows:
        flow["slo_evaluation"] = _evaluate_flow_slo(model, flow)
    # Нормальный прикладной поток с уже атрибутированного атакующего узла
    # физически блокируется и остаётся видимым в flows, но не должен ухудшать
    # SLA добросовестных абонентов. Иначе успешный карантин ошибочно выглядел
    # бы как отказ сети. Исключение применяется только к явному действию
    # quarantine_attack_source, а не к обычной потере маршрута.
    sla_accounted_flows = [
        flow
        for flow in flows
        if not flow.get("security_excluded_from_sla_accounting")
    ]
    security_excluded_flows = [
        flow
        for flow in flows
        if flow.get("security_excluded_from_sla_accounting")
    ]
    legitimate_packets = sum(int(flow["packet_count"]) for flow in sla_accounted_flows)
    legitimate_dropped = sum(
        int(flow["observed_dropped_packets"]) for flow in sla_accounted_flows
    )
    non_attack_dropped = sum(
        int(flow.get("pre_attack_dropped_packets", 0))
        for flow in sla_accounted_flows
    )
    observed_attack_induced_dropped = sum(
        max(
            0,
            int(flow.get("observed_dropped_packets", 0))
            - int(flow.get("pre_attack_dropped_packets", 0)),
        )
        for flow in sla_accounted_flows
    )
    prevented_attack_induced_dropped = sum(
        int(flow.get("prevented_dropped_packets", 0))
        for flow in sla_accounted_flows
    )
    raw_attack_induced_dropped = (
        observed_attack_induced_dropped + prevented_attack_induced_dropped
    )
    # Counterfactual total under the same routing/isolation state, but without
    # preventive filtering of attack-induced loss.  This definition preserves
    # raw = observed + prevented even when a route is isolated independently.
    raw_legitimate_dropped = non_attack_dropped + raw_attack_induced_dropped
    impacted = [flow for flow in sla_accounted_flows if flow.get("attack_impacted")]
    impacted_gold = [flow for flow in impacted if flow.get("sla_grade") == "gold"]
    gold_flows = [
        flow for flow in sla_accounted_flows if flow.get("sla_grade") == "gold"
    ]
    gold_healthy = [flow for flow in gold_flows if _flow_meets_sla(model, flow)]
    sla_coverage_values = [
        float(flow["slo_evaluation"]["assessment_coverage_ratio"])
        for flow in sla_accounted_flows
    ]
    attack_packets = sum(int(flow["packet_count"]) for flow in attack_flows)
    attack_wire = sum(int(flow["wire_bytes"]) for flow in attack_flows)
    raw_attack_packets = sum(int(flow.get("raw_packet_count", flow["packet_count"])) for flow in attack_flows)
    raw_attack_wire = sum(int(flow.get("raw_wire_bytes", flow["wire_bytes"])) for flow in attack_flows)
    interval = int(flows[0].get("interval_seconds", 1)) if flows else 1
    raw_max_pressure = max(
        (float(event["target_resource_pressure"]) * float(event["intensity_ratio"]) for event in events),
        default=0.0,
    )
    max_pressure = max(
        (
            float(event["target_resource_pressure"])
            * float(event["intensity_ratio"])
            * (1.0 - event_mitigation[event["attack_id"]])
            for event in events
        ),
        default=0.0,
    )
    target_observations = [
        _target_observation(model, event, mitigation_ratio=event_mitigation[event["attack_id"]])
        for event in events
    ]
    authentication_observations = [
        observation
        for event, observation in zip(events, target_observations, strict=True)
        if event.get("kind") == AttackKind.BRUTE_FORCE.value
    ]
    precursor_summary = _precursor_summary(precursor_events)
    active_power = [
        (event, observation)
        for event, observation in zip(events, target_observations, strict=True)
        if event.get("kind") == AttackKind.POWER_ATTACK.value
    ]
    if active_power:
        active_power_pressure = max(
            float(item.get("raw_resource_pressure", 0.0))
            * (1.0 - float(item.get("preventive_mitigation_ratio", 0.0)))
            for _, item in active_power
        )
        cyber_observations = [
            item
            for event, item in active_power
            if event.get("power_failure_mode") == "cyber_shutdown"
        ]
        brownout_observations = [
            item
            for event, item in active_power
            if event.get("power_failure_mode") == "brownout_feed_loss"
        ]
        if cyber_observations:
            precursor_summary["power_control_anomaly_ratio"] = max(
                float(precursor_summary["power_control_anomaly_ratio"]),
                max(
                    1.0 - float(item.get("estimated_availability_ratio", 1.0))
                    for item in cyber_observations
                ),
            )
        if brownout_observations:
            precursor_summary["power_voltage_sag_ratio"] = max(
                float(precursor_summary["power_voltage_sag_ratio"]),
                max(
                    1.0 - float(item.get("ups_input_voltage_ratio", 1.0))
                    for item in brownout_observations
                ),
            )
            if any(bool(item.get("ups_on_battery")) for item in brownout_observations):
                precursor_summary["battery_discharge_rate_ratio"] = max(
                    float(precursor_summary["battery_discharge_rate_ratio"]),
                    min(1.0, active_power_pressure * 0.70),
                )
    power_records = [
        dict(value) for _, value in sorted((power_runtime_state or {}).items())
    ]
    intrusion_events = [event for event in events if not event.get("is_recovery_effect")]
    recovery_events = [event for event in events if event.get("is_recovery_effect")]
    defense_summary = _preventive_defense_summary(
        events,
        defense_plan,
        event_mitigation,
        allocation_reports,
    )
    return all_flows, {
        "scenario": scenario or ("mitre-demo" if events or precursor_events else "none"),
        "active": bool(events),
        "active_count": len(events),
        "intrusion_active": bool(intrusion_events),
        "intrusion_active_count": len(intrusion_events),
        "recovery_after_effect_active": bool(recovery_events),
        "recovery_after_effect_count": len(recovery_events),
        "events": events,
        "precursor_active": bool(precursor_events),
        "precursor_count": len(precursor_events),
        "precursors": precursor_events,
        **precursor_summary,
        "preventive_defense": defense_summary,
        "sla_restoration": _sla_restoration_summary(model, flows),
        "raw_attack_packet_count": raw_attack_packets,
        "attack_packet_count": attack_packets,
        "blocked_attack_packet_count": max(0, raw_attack_packets - attack_packets),
        "raw_attack_wire_bytes": raw_attack_wire,
        "attack_wire_bytes": attack_wire,
        "blocked_attack_wire_bytes": max(0, raw_attack_wire - attack_wire),
        "raw_attack_rate_mbps": round(raw_attack_wire * 8.0 / max(interval, 1) / 1_000_000.0, 4),
        "attack_rate_mbps": round(attack_wire * 8.0 / max(interval, 1) / 1_000_000.0, 4),
        "blocked_attack_rate_mbps": round((raw_attack_wire - attack_wire) * 8.0 / max(interval, 1) / 1_000_000.0, 4),
        "maximum_intensity_ratio": max((float(event["intensity_ratio"]) for event in events), default=0.0),
        "raw_target_resource_pressure": round(raw_max_pressure, 6),
        "target_resource_pressure": round(max_pressure, 6),
        "target_observations": target_observations,
        "authentication_failure_rate_per_second": round(
            sum(
                float(item.get("authentication_attempt_rate_per_second", 0.0))
                * float(item.get("authentication_failure_ratio", 0.0))
                for item in authentication_observations
            ),
            6,
        ),
        "authentication_failure_ratio": round(
            max(
                (float(item.get("authentication_failure_ratio", 0.0)) for item in authentication_observations),
                default=0.0,
            ),
            6,
        ),
        "failed_authentication_attempt_count": round(
            sum(float(item.get("failed_authentication_attempt_count", 0.0)) for item in authentication_observations),
            6,
        ),
        "account_lockout_pressure": round(
            max(
                (float(item.get("account_lockout_pressure", 0.0)) for item in authentication_observations),
                default=0.0,
            ),
            6,
        ),
        "power_runtime_state": power_records,
        "power_ride_through_active_count": sum(
            bool(item.get("ride_through_active")) for item in power_records
        ),
        "power_recovery_active_count": sum(
            item.get("phase") in {"battery_recharge", "rebooting", "health_check"}
            for item in power_records
        ),
        "power_service_unavailability_ratio": max(
            (1.0 - float(item.get("service_availability_ratio", 1.0)) for item in power_records),
            default=0.0,
        ),
        "power_energy_depletion_ratio": max(
            (float(item.get("energy_depletion_ratio", 0.0)) for item in power_records),
            default=0.0,
        ),
        "maximum_target_utilization_percent": max(
            (item["estimated_resource_utilization_percent"] for item in target_observations), default=0.0
        ),
        "impacted_legitimate_flow_count": len(impacted),
        "impacted_gold_flow_count": len(impacted_gold),
        "subscriber_application_flow_count": len(flows),
        "sla_accounted_flow_count": len(sla_accounted_flows),
        "security_excluded_flow_count": len(security_excluded_flows),
        "security_excluded_client_ids": sorted(
            str(flow.get("client_node")) for flow in security_excluded_flows
        ),
        "sla_accounting_semantics": (
            "explicitly_quarantined_attack_sources_are_visible_but_excluded_from_customer_sla"
        ),
        "slo_evaluation_semantics": (
            "service_aware_all_assessed_required_metrics_must_pass_with_modeled_assumed_not_modeled_evidence"
        ),
        "slo_temporal_aggregation": (
            "current_simulation_step_aggregate_not_rolling_policy_window_percentile"
        ),
        "slo_limited_coverage_flow_count": sum(
            flow["slo_evaluation"]["assessment_state"]
            == "compliant_with_modeling_limits"
            for flow in sla_accounted_flows
        ),
        "mean_slo_assessment_coverage_ratio": round(
            sum(sla_coverage_values) / max(len(sla_coverage_values), 1),
            6,
        ),
        "legitimate_packet_count": legitimate_packets,
        "raw_legitimate_dropped_packets": raw_legitimate_dropped,
        "legitimate_dropped_packets": legitimate_dropped,
        "prevented_legitimate_dropped_packets": (
            prevented_attack_induced_dropped
        ),
        "raw_attack_induced_legitimate_dropped_packets": (
            raw_attack_induced_dropped
        ),
        "observed_attack_induced_legitimate_dropped_packets": (
            observed_attack_induced_dropped
        ),
        "non_attack_legitimate_dropped_packets": non_attack_dropped,
        "raw_legitimate_drop_semantics": (
            "counterfactual_total_equals_observed_total_plus_prevented_attack_induced_drops"
        ),
        "raw_legitimate_loss_ratio": raw_legitimate_dropped / max(legitimate_packets, 1),
        "legitimate_loss_ratio": legitimate_dropped / max(legitimate_packets, 1),
        "legitimate_delivery_ratio": (legitimate_packets - legitimate_dropped) / max(legitimate_packets, 1),
        "gold_sla_compliance_ratio": len(gold_healthy) / max(len(gold_flows), 1),
    }


def _flow_is_affected(flow: dict[str, Any], event: dict[str, Any]) -> bool:
    if event.get("kind") == AttackKind.BRUTE_FORCE.value:
        # T1110.001 remains an authentication-log scenario.  It is not a
        # traffic generator and must never manufacture a data-plane loss.
        return False
    target = event["target_id"]
    return target in flow.get("route", []) or target in {
        flow.get("client_node"), flow.get("server_node"), flow.get("service_node")
    }


def _flow_meets_sla(model: NetworkModel, flow: dict[str, Any]) -> bool:
    """Вернуть service-aware результат, а не только общий latency/loss guardrail."""
    evaluation = flow.get("slo_evaluation")
    if not isinstance(evaluation, Mapping):
        evaluation = _evaluate_flow_slo(model, flow)
    return bool(evaluation.get("compliant", False))


def _evaluate_flow_slo(model: NetworkModel, flow: dict[str, Any]) -> dict[str, Any]:
    """Оценить прикладные SLO одного легитимного потока.

    ``modeled`` означает, что значение непосредственно следует из состояния
    пакетной модели. ``assumed`` — прозрачная производная оценка (например,
    timeout DNS по агрегированным потерям без сопоставления transaction ID).
    ``not_modeled`` никогда не участвует в логическом результате: это сохраняет
    честную границу модели и не подменяет отсутствие измерения нулём.
    """
    client = flow.get("client_node")
    if client not in model.graph:
        return {
            "traffic_kind": flow.get("traffic_kind"),
            "sla_grade": flow.get("sla_grade"),
            "compliant": False,
            "assessment_state": "not_assessable",
            "assessment_coverage_ratio": 0.0,
            "required_metric_count": 0,
            "assessed_metric_count": 0,
            "failed_metric_ids": ["subscriber_policy"],
            "not_modeled_metric_ids": [],
            "metrics": {},
            "semantics": "service_aware_slo_with_explicit_evidence_levels",
            "temporal_aggregation": (
                "current_simulation_step_aggregate_not_rolling_policy_window_percentile"
            ),
            "reason_ru": "для клиента не найдена исполняемая d0sl-политика",
        }

    attrs = model.graph.nodes[client]
    policy = attrs.get("d0sl_policy", {})
    traffic_kind = str(flow.get("traffic_kind") or attrs.get("traffic_kind") or "unknown")
    packet_count = max(0, int(flow.get("packet_count", 0)))
    dropped_packets = min(
        packet_count,
        max(0, int(flow.get("observed_dropped_packets", 0))),
    )
    delivery_ratio = (
        max(0.0, 1.0 - dropped_packets / packet_count)
        if packet_count > 0
        else 0.0
    )
    interval_seconds = max(float(flow.get("interval_seconds", 1.0) or 1.0), 1e-9)
    offered_payload_kbps = (
        max(0.0, float(flow.get("payload_bytes", 0.0)))
        * 8.0
        / interval_seconds
        / 1000.0
    )
    delivered_payload_kbps = offered_payload_kbps * delivery_ratio
    if flow.get("transport") == "TCP" and "transport_queue" in flow:
        delivered_payload_kbps = float(flow["transport_queue"]["goodput_mbps"]) * 1000.0
    loss_percent = (dropped_packets / packet_count * 100.0) if packet_count else 100.0
    available = bool(flow.get("route")) and not bool(flow.get("isolated"))
    available = available and flow.get("route_available", True) is not False
    available = available and flow.get("service_available", True) is not False
    metrics: dict[str, dict[str, Any]] = {}

    def add_metric(
        metric_id: str,
        *,
        label_ru: str,
        observed: float | bool | None,
        budget: float | bool | None,
        unit: str,
        comparison: str,
        evidence: str,
        value_origin: str,
        required: bool = True,
        note_ru: str | None = None,
    ) -> None:
        if evidence not in {"modeled", "assumed", "not_modeled"}:
            raise ValueError(f"Unknown SLO evidence level: {evidence}")
        compliant: bool | None
        if evidence == "not_modeled" or observed is None or budget is None:
            compliant = None
        elif comparison == "lte":
            compliant = float(observed) <= float(budget) + 1e-12
        elif comparison == "gte":
            compliant = float(observed) + 1e-12 >= float(budget)
        elif comparison == "eq":
            compliant = bool(observed) is bool(budget)
        else:
            raise ValueError(f"Unknown SLO comparison: {comparison}")
        metrics[metric_id] = {
            "metric_id": metric_id,
            "label_ru": label_ru,
            "observed_value": (
                round(observed, 6) if isinstance(observed, float) and math.isfinite(observed)
                else observed
            ),
            "budget_value": budget,
            "unit": unit,
            "comparison": comparison,
            "evidence": evidence,
            "value_origin": value_origin,
            "required": required,
            "compliant": compliant,
            "note_ru": note_ru,
        }

    add_metric(
        "service_availability",
        label_ru="доступность маршрута и экземпляра сервиса",
        observed=available,
        budget=True,
        unit="bool",
        comparison="eq",
        evidence="modeled",
        value_origin="routing_and_service_failover_state",
    )

    latency_ms = max(0.0, float(flow.get("one_way_latency_ms", 0.0)))
    latency_budget_ms = float(
        policy.get("latency_budget_ms", attrs.get("latency_budget_ms", math.inf))
    )
    loss_budget_percent = float(policy.get("packet_loss_budget_percent", math.inf))
    min_bitrate_kbps = float(policy.get("min_bitrate_kbps", 0.0))
    jitter_budget_ms = policy.get("jitter_budget_ms")

    if traffic_kind in {"voice", "vlc_av"}:
        add_metric(
            "media_bitrate_kbps",
            label_ru="доставленный битрейт медиапотока",
            observed=delivered_payload_kbps,
            budget=min_bitrate_kbps,
            unit="kbit/s",
            comparison="gte",
            evidence="modeled",
            value_origin="payload_bytes_scaled_by_aggregate_packet_delivery",
        )
        add_metric(
            "one_way_media_latency_ms",
            label_ru="односторонняя задержка медиапотока",
            observed=latency_ms,
            budget=latency_budget_ms,
            unit="ms",
            comparison="lte",
            evidence="modeled",
            value_origin="current_routed_path_plus_endpoint_packetization_when_applicable",
        )
        add_metric(
            "network_packet_loss_percent",
            label_ru="потери пакетов медиапотока",
            observed=loss_percent,
            budget=loss_budget_percent,
            unit="%",
            comparison="lte",
            evidence="modeled",
            value_origin="aggregate_flow_packet_counters",
        )
        add_metric(
            "jitter_ms",
            label_ru="джиттер медиапотока",
            observed=None,
            budget=jitter_budget_ms,
            unit="ms",
            comparison="lte",
            evidence="not_modeled",
            value_origin="no_per_packet_arrival_time_series",
            note_ru="без временного ряда прибытия пакетов джиттер не подменяется нулём",
        )

    elif traffic_kind == "ftp":
        data_segments = max(1, int(flow.get("data_segments", packet_count or 1)))
        retransmissions = max(0, int(flow.get("observed_retransmissions", 0)))
        retransmission_percent = retransmissions / data_segments * 100.0
        retransmission_budget = float(
            policy.get("tcp_retransmission_budget_percent", math.inf)
        )
        file_size_mib = float(policy.get("file_size_mib", 0.0))
        completion_budget = float(
            policy.get("completion_time_budget_seconds", math.inf)
        )
        # Здесь нет полноценного TCP congestion-control. Поэтому goodput и
        # completion являются консервативной производной по доставленным пакетам.
        completion_seconds = (
            file_size_mib * 1024.0 * 1024.0 * 8.0
            / max(delivered_payload_kbps * 1000.0, 1e-9)
        )
        add_metric(
            "ftp_goodput_kbps",
            label_ru="полезная скорость FTP",
            observed=delivered_payload_kbps,
            budget=min_bitrate_kbps,
            unit="kbit/s",
            comparison="gte",
            evidence="assumed",
            value_origin="payload_rate_scaled_by_aggregate_delivery_without_tcp_cwnd_model",
            note_ru="TCP congestion window и повторная доставка внутри шага не моделируются",
        )
        add_metric(
            "ftp_completion_time_seconds",
            label_ru="расчётное время передачи файла",
            observed=completion_seconds,
            budget=completion_budget,
            unit="s",
            comparison="lte",
            evidence="assumed",
            value_origin="configured_file_size_divided_by_derived_goodput",
        )
        add_metric(
            "tcp_retransmission_ratio_percent",
            label_ru="доля повторных передач TCP",
            observed=retransmission_percent,
            budget=retransmission_budget,
            unit="%",
            comparison="lte",
            evidence="assumed",
            value_origin="aggregate_retransmission_counter_over_data_segment_count",
            note_ru="модель не различает потерю ACK и сегмента данных при назначении retransmission",
        )
        add_metric(
            "ftp_payload_checksum_success",
            label_ru="совпадение контрольной суммы переданного файла",
            observed=None,
            budget=True,
            unit="bool",
            comparison="eq",
            evidence="not_modeled",
            value_origin="packet_sample_checksums_are_headers_not_end_to_end_file_validation",
            note_ru="синтетические checksum пакетов не являются проверкой целостности файла",
        )

    elif traffic_kind == "dns":
        queries = max(1, int(flow.get("dns_queries", 1)))
        # Пакетная модель хранит суммарные потери, но не transaction ID. Один
        # потерянный request/response консервативно считается одним timeout.
        timeout_count = max(
            int(flow.get("dns_timeout_count", 0)),
            min(queries, dropped_packets),
        )
        timeout_percent = timeout_count / queries * 100.0
        flow["dns_timeout_count"] = timeout_count
        if flow.get("rtt_ms") is not None:
            flow["dns_response_time_ms"] = round(float(flow["rtt_ms"]), 4)
        rtt_ms = flow.get("rtt_ms")
        add_metric(
            "dns_response_rtt_ms",
            label_ru="время ответа DNS",
            observed=None if rtt_ms is None else max(0.0, float(rtt_ms)),
            budget=latency_budget_ms,
            unit="ms",
            comparison="lte",
            evidence="modeled" if rtt_ms is not None else "not_modeled",
            value_origin="query_and_response_path_latency",
        )
        add_metric(
            "dns_timeout_ratio_percent",
            label_ru="доля DNS-запросов с таймаутом",
            observed=timeout_percent,
            budget=float(policy.get("timeout_budget_percent", math.inf)),
            unit="%",
            comparison="lte",
            evidence="assumed",
            value_origin="conservative_transaction_loss_from_aggregate_packet_drops",
        )
        add_metric(
            "dns_servfail_ratio_percent",
            label_ru="доля ответов SERVFAIL",
            observed=None,
            budget=None,
            unit="%",
            comparison="lte",
            evidence="not_modeled",
            value_origin="dns_rcode_not_simulated",
            note_ru="генератор не эмулирует внутреннюю рекурсию и коды ответа DNS",
        )
        add_metric(
            "dns_tcp_fallback_success",
            label_ru="успешность fallback DNS на TCP",
            observed=None,
            budget=True,
            unit="bool",
            comparison="eq",
            evidence="not_modeled",
            value_origin="udp_query_model_has_no_runtime_tcp_fallback",
        )

    elif traffic_kind == "video_conference":
        # Один multiplexed WebRTC flow даёт общий сетевой proxy; отдельные
        # media pipelines пока не симулируются, но бюджеты проверяются раздельно.
        add_metric(
            "conference_audio_latency_ms",
            label_ru="односторонняя задержка звука конференции",
            observed=latency_ms,
            budget=float(policy.get("audio_latency_budget_ms", latency_budget_ms)),
            unit="ms",
            comparison="lte",
            evidence="assumed",
            value_origin="shared_webrtc_transport_path_proxy",
        )
        add_metric(
            "conference_video_latency_ms",
            label_ru="односторонняя задержка видео конференции",
            observed=latency_ms,
            budget=float(policy.get("video_latency_budget_ms", latency_budget_ms)),
            unit="ms",
            comparison="lte",
            evidence="assumed",
            value_origin="shared_webrtc_transport_path_proxy",
        )
        add_metric(
            "conference_media_loss_percent",
            label_ru="потери медиапакетов конференции",
            observed=loss_percent,
            budget=loss_budget_percent,
            unit="%",
            comparison="lte",
            evidence="modeled",
            value_origin="aggregate_flow_packet_counters",
        )
        add_metric(
            "conference_jitter_ms",
            label_ru="джиттер конференции",
            observed=None,
            budget=jitter_budget_ms,
            unit="ms",
            comparison="lte",
            evidence="not_modeled",
            value_origin="no_per_packet_arrival_time_series",
        )

    elif traffic_kind == "live_streaming":
        live_edge_base_ms = float(flow.get("glass_to_glass_latency_ms", latency_ms))
        live_edge_latency_ms = (
            live_edge_base_ms
            + max(0.0, float(flow.get("additional_latency_ms", 0.0)))
            + max(0.0, float(flow.get("attack_latency_penalty_ms", 0.0)))
        )
        add_metric(
            "ll_hls_live_edge_latency_ms",
            label_ru="задержка относительно live edge",
            observed=live_edge_latency_ms,
            budget=latency_budget_ms,
            unit="ms",
            comparison="lte",
            evidence="assumed",
            value_origin="engineering_pipeline_baseline_plus_modeled_network_penalties",
            note_ru="encoder/player pipeline задан инженерной baseline-оценкой, а не медиаплеером",
        )
        add_metric(
            "ll_hls_delivered_bitrate_kbps",
            label_ru="доставленный битрейт LL-HLS",
            observed=delivered_payload_kbps,
            budget=min_bitrate_kbps,
            unit="kbit/s",
            comparison="gte",
            evidence="modeled",
            value_origin="http_payload_scaled_by_aggregate_packet_delivery",
        )
        add_metric(
            "ll_hls_transport_loss_percent",
            label_ru="потери транспортных пакетов LL-HLS",
            observed=loss_percent,
            budget=loss_budget_percent,
            unit="%",
            comparison="lte",
            evidence="modeled",
            value_origin="aggregate_flow_packet_counters",
        )
        add_metric(
            "ll_hls_startup_time_ms",
            label_ru="время запуска воспроизведения",
            observed=None,
            budget=policy.get("startup_time_budget_ms"),
            unit="ms",
            comparison="lte",
            evidence="not_modeled",
            value_origin="no_player_state_machine",
        )
        add_metric(
            "ll_hls_rebuffer_ratio_percent",
            label_ru="доля времени ребуферизации",
            observed=None,
            budget=policy.get("rebuffer_ratio_budget_percent"),
            unit="%",
            comparison="lte",
            evidence="not_modeled",
            value_origin="no_playback_buffer_state",
        )
        add_metric(
            "ll_hls_part_deadline_miss_ratio_percent",
            label_ru="доля пропущенных сроков доставки частей CMAF",
            observed=None,
            budget=policy.get("segment_deadline_miss_budget_percent"),
            unit="%",
            comparison="lte",
            evidence="not_modeled",
            value_origin="no_segment_or_part_deadline_events",
        )

    else:
        add_metric(
            "generic_one_way_latency_ms",
            label_ru="односторонняя задержка",
            observed=latency_ms,
            budget=latency_budget_ms,
            unit="ms",
            comparison="lte",
            evidence="modeled",
            value_origin="current_routed_path",
        )
        add_metric(
            "generic_packet_loss_percent",
            label_ru="потери пакетов",
            observed=loss_percent,
            budget=loss_budget_percent,
            unit="%",
            comparison="lte",
            evidence="modeled",
            value_origin="aggregate_flow_packet_counters",
        )

    required = [metric for metric in metrics.values() if metric["required"]]
    assessed = [metric for metric in required if metric["compliant"] is not None]
    failed = [metric for metric in assessed if metric["compliant"] is False]
    not_modeled = [
        metric for metric in required if metric["evidence"] == "not_modeled"
    ]
    compliant = bool(assessed) and not failed
    coverage = len(assessed) / max(len(required), 1)
    assessment_state = (
        "non_compliant"
        if failed
        else "compliant"
        if len(assessed) == len(required)
        else "compliant_with_modeling_limits"
        if compliant
        else "not_assessable"
    )
    evidence_counts = {
        evidence: sum(metric["evidence"] == evidence for metric in metrics.values())
        for evidence in ("modeled", "assumed", "not_modeled")
    }
    return {
        "traffic_kind": traffic_kind,
        "sla_grade": flow.get("sla_grade"),
        "compliant": compliant,
        "assessment_state": assessment_state,
        "assessment_coverage_ratio": round(coverage, 6),
        "required_metric_count": len(required),
        "assessed_metric_count": len(assessed),
        "failed_metric_ids": [metric["metric_id"] for metric in failed],
        "not_modeled_metric_ids": [metric["metric_id"] for metric in not_modeled],
        "evidence_counts": evidence_counts,
        "metrics": metrics,
        "semantics": "service_aware_slo_with_explicit_evidence_levels",
        "compliance_semantics": (
            "all_assessed_required_metrics_must_pass; not_modeled_metrics_are_excluded_and_reported"
        ),
        "temporal_aggregation": (
            "current_simulation_step_aggregate_not_rolling_policy_window_percentile"
        ),
    }


def _target_observation(
    model: NetworkModel,
    event: dict[str, Any],
    *,
    mitigation_ratio: float = 0.0,
) -> dict[str, Any]:
    attrs = model.graph.nodes[event["target_id"]]
    raw_pressure = float(event["target_resource_pressure"]) * float(event["intensity_ratio"])
    pressure = raw_pressure * (1.0 - mitigation_ratio)
    if attrs.get("role") == "service-server":
        baseline_cpu = float(attrs.get("runtime", {}).get("cpu_util_percent", 0.0))
    elif attrs.get("level") == "L2":
        baseline_cpu = float(attrs.get("l2_raw_baseline", {}).get("cpu_util", 0.0))
    else:
        baseline_cpu = float(attrs.get("kendall_queue", {}).get("utilization_rho", 0.0)) * 100.0
    ingress_capacity = max(float(event.get("convergence_capacity_mbps", 0.0)), 1.0)
    raw_network_utilization = min(
        100.0,
        float(event["effective_offered_rate_mbps"]) / ingress_capacity * 100.0,
    )
    network_utilization = raw_network_utilization * (1.0 - mitigation_ratio)
    power_reserve = _power_reserve_ratio(attrs)
    power_attack = event["kind"] == AttackKind.POWER_ATTACK.value
    brute_force = event["kind"] == AttackKind.BRUTE_FORCE.value
    runtime = event.get("power_runtime")
    if power_attack and isinstance(runtime, Mapping):
        # Электрический вход, выход ИБП и доступность сервиса — разные
        # физические величины. Краткий brownout может просадить вход, но не
        # нагрузку; команда T1529, наоборот, выключает сервис при штатном
        # напряжении. Поэтому здесь используется состояние автомата питания,
        # а не старая эвристика ``1 - resource_pressure``.
        service_availability = max(
            0.0, min(1.0, float(runtime.get("service_availability_ratio", 1.0)))
        )
        output_availability = max(
            0.0,
            min(1.0, float(runtime.get("power_output_availability_ratio", 1.0))),
        )
        input_voltage_ratio = max(
            0.0, min(1.0, float(runtime.get("input_voltage_ratio", 1.0)))
        )
        energy_reserve = max(
            0.0, min(1.0, float(runtime.get("energy_reserve_ratio", power_reserve)))
        )
        estimated_resource_utilization = baseline_cpu * service_availability
        estimated_queue_utilization = 0.0
    elif brute_force:
        # T1110.001 is represented as low-rate authentication telemetry, not
        # as a data-plane flood.  A failed login must not manufacture packet
        # loss or an outage in an unrelated subscriber service.
        service_availability = 1.0
        output_availability = 1.0
        input_voltage_ratio = 1.0
        energy_reserve = power_reserve
        estimated_resource_utilization = baseline_cpu
        estimated_queue_utilization = 0.0
    else:
        service_availability = max(0.0, 1.0 - pressure * 0.92)
        output_availability = 1.0
        input_voltage_ratio = 1.0
        # Сетевой flood при штатном вводе не должен искусственно разряжать
        # батарею ИБП. Энергетический резерв меняет только power-runtime.
        energy_reserve = power_reserve
        estimated_resource_utilization = min(100.0, baseline_cpu + pressure * 88.0)
        estimated_queue_utilization = min(100.0, pressure * 100.0)
    authentication_attempt_rate = (
        float(event.get("effective_authentication_attempt_rate_per_second", 0.0))
        * (1.0 - mitigation_ratio)
        if brute_force
        else 0.0
    )
    authentication_failure_ratio = (
        float(event["technical"].get("authentication_failure_ratio", 0.0))
        if brute_force
        else 0.0
    )
    failed_authentication_attempts = (
        authentication_attempt_rate
        * float(event.get("interval_seconds", 1))
        * authentication_failure_ratio
    )
    return {
        "target_id": event["target_id"],
        "target_type": event["target_type"],
        "raw_resource_pressure": round(raw_pressure, 6),
        "preventive_mitigation_ratio": round(mitigation_ratio, 6),
        "estimated_resource_utilization_percent": round(estimated_resource_utilization, 4),
        "raw_network_utilization_percent": round(raw_network_utilization, 4),
        "estimated_network_utilization_percent": round(network_utilization, 4),
        "convergence_capacity_mbps": round(ingress_capacity, 4),
        "target_resource_pressure_basis": event.get("target_resource_pressure_basis"),
        "target_resource_budget_origin": event.get("target_resource_budget_origin"),
        "estimated_queue_utilization_percent": round(estimated_queue_utilization, 4),
        "estimated_availability_ratio": round(service_availability, 6),
        "service_availability_ratio": round(service_availability, 6),
        "estimated_power_availability_ratio": round(output_availability, 6),
        "power_output_availability_ratio": round(output_availability, 6),
        "ups_input_voltage_ratio": round(input_voltage_ratio, 6),
        "estimated_energy_reserve_ratio": round(energy_reserve, 6),
        "energy_depletion_ratio": round(
            max(0.0, float(runtime.get("energy_depletion_ratio", 0.0)))
            if power_attack and isinstance(runtime, Mapping)
            else 0.0,
            6,
        ),
        "ups_on_battery": bool(runtime.get("ups_on_battery", False))
        if power_attack and isinstance(runtime, Mapping)
        else False,
        "ride_through_active": bool(runtime.get("ride_through_active", False))
        if power_attack and isinstance(runtime, Mapping)
        else False,
        "power_failure_mode": event.get("power_failure_mode") if power_attack else None,
        "power_runtime_phase": runtime.get("phase")
        if power_attack and isinstance(runtime, Mapping)
        else None,
        "power_architecture": attrs.get("power_architecture", {}),
        "power_failure_semantics": (
            event.get("power_failure_semantics", "explicit_power_runtime_state")
            if power_attack
            else "not_a_power_event"
        ),
        "syn_backlog_exhaustion_ratio": round(
            pressure * float(event["technical"].get("incomplete_handshake_ratio", 0.0)), 6
        ),
        "authentication_attempt_rate_per_second": round(authentication_attempt_rate, 6),
        "authentication_failure_ratio": round(authentication_failure_ratio, 6),
        "failed_authentication_attempt_count": round(failed_authentication_attempts, 6),
        "account_lockout_pressure": round(
            min(1.0, authentication_attempt_rate), 6
        ),
        "authentication_parameter_origin": (
            str(event["technical"].get("authentication_attempt_rate_origin", "not_applicable"))
            if brute_force
            else "not_applicable"
        ),
        "authentication_parameter_reference_url": (
            event["technical"].get("authentication_attempt_rate_reference_url")
            if brute_force
            else None
        ),
        "critical_protection": attrs.get("critical_protection", {}),
    }


def _build_attack_flows(
    event: dict[str, Any],
    *,
    mitigation_ratio: float = 0.0,
    quarantined_sources: set[str] | None = None,
    identities: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if event["kind"] in {AttackKind.POWER_ATTACK.value, AttackKind.BRUTE_FORCE.value}:
        return []
    is_reflection = event["kind"] == AttackKind.DDOS.value
    routes = event.get("response_routes" if is_reflection else "routes", [])
    if not routes:
        return []
    app = {
        "dos": "ATTACK_DOS",
        "ddos": "ATTACK_DDOS",
        "syn_flood": "ATTACK_SYN",
        "power_attack": "ATTACK_POWER",
    }[event["kind"]]
    transport = "TCP" if event["kind"] == "syn_flood" else "UDP"
    raw_total_packets = int(event["effective_packet_rate_pps"] * event["interval_seconds"])
    raw_total_wire = int(event["effective_offered_rate_mbps"] * 1_000_000.0 / 8.0 * event["interval_seconds"])
    total_packets = int(round(raw_total_packets * (1.0 - mitigation_ratio)))
    total_wire = int(round(raw_total_wire * (1.0 - mitigation_ratio)))
    quarantined_sources = quarantined_sources or set()
    quarantined_initiators = sorted(
        quarantined_sources & {str(node) for node in event.get("ingress_nodes", [])}
    )
    flows: list[dict[str, Any]] = []
    for index, route in enumerate(routes):
        source_node = str(route[0]) if route else "unknown"
        source_identity = identities.get(source_node) if identities is not None else None
        target_identity = (
            identities.get(str(event["target_id"]))
            if identities is not None
            else None
        )
        source_ip = (
            source_identity.get("ip")
            if isinstance(source_identity, Mapping)
            else getattr(source_identity, "ip", None)
        )
        target_ip = (
            target_identity.get("ip")
            if isinstance(target_identity, Mapping)
            else getattr(target_identity, "ip", None)
        )
        # У reflected DDoS источник внутри наблюдаемого графа — border ingress,
        # а не L1-инициатор. Его карантин останавливает новые spoofed-запросы,
        # но не обнуляет уже входящие ответы внешних reflectors.
        quarantined = source_node in quarantined_sources and not is_reflection
        packet_count = total_packets // len(routes) + (1 if index < total_packets % len(routes) else 0)
        wire_bytes = total_wire // len(routes) + (1 if index < total_wire % len(routes) else 0)
        raw_packet_count = raw_total_packets // len(routes) + (1 if index < raw_total_packets % len(routes) else 0)
        raw_wire_bytes = raw_total_wire // len(routes) + (1 if index < raw_total_wire % len(routes) else 0)
        if quarantined:
            packet_count = 0
            wire_bytes = 0
        pressure = float(event["target_resource_pressure"]) * float(event["intensity_ratio"])
        dropped = math.floor(packet_count * min(0.92, 0.20 + pressure * 0.65))
        technical = event["technical"]
        is_syn = event["kind"] == AttackKind.SYN_FLOOD.value
        payload_per_packet = 0 if is_syn else int(technical["packet_size_bytes"])
        l3_packet_bytes = 40 if is_syn else payload_per_packet + 28
        flows.append({
            "flow_id": f"{event['attack_id']}-{event['step_index']}-{index + 1}",
            "step_index": event["step_index"], "time_seconds": event["time_seconds"],
            "interval_seconds": event["interval_seconds"], "application": app,
            "transport": transport, "client_node": source_node, "service_node": event["target_id"],
            "server_node": event["target_id"],
            "client_ip": (
                f"198.51.100.{10 + index}"
                if is_reflection
                else source_ip or f"узел:{source_node}"
            ),
            "server_ip": target_ip or f"узел:{event['target_id']}",
            "client_port": 0, "server_port": 0,
            "packet_count": packet_count,
            "payload_bytes": packet_count * payload_per_packet,
            "wire_bytes": wire_bytes,
            "raw_packet_count": raw_packet_count, "raw_wire_bytes": raw_wire_bytes,
            "blocked_packet_count": max(0, raw_packet_count - packet_count),
            "blocked_wire_bytes": max(0, raw_wire_bytes - wire_bytes),
            "preventive_mitigation_ratio": round(mitigation_ratio, 6),
            "mtu_bytes": 1_500,
            "attack_payload_bytes_per_packet": payload_per_packet,
            "attack_l3_packet_bytes": l3_packet_bytes,
            "on_wire_size_bytes": int(
                technical.get("on_wire_size_bytes") or l3_packet_bytes
            ),
            "route": [] if quarantined else route,
            "original_route": route,
            "reverse_route": [] if quarantined else list(reversed(route)),
            "hop_count": 0 if quarantined else max(0, len(route) - 1),
            "one_way_latency_ms": round(float(event["latency_penalty_ms"]) * 0.08, 4),
            "rtt_ms": None, "expected_loss_ratio": 0.0,
            "observed_dropped_packets": dropped, "observed_retransmissions": 0,
            "sla_grade": "attack", "traffic_kind": event["kind"], "sequence_base": 0,
            "is_attack_traffic": True, "attack_id": event["attack_id"],
            "quarantined": quarantined,
            "routing_action": (
                "filter_amplified_response_at_transit_ingress"
                if is_reflection
                else "quarantine_attack_source"
                if quarantined
                else "attack_admitted_after_filtering"
            ),
            "attack_traffic_leg": (
                "amplified_external_reflector_response"
                if is_reflection
                else "direct_attack_source_to_target"
            ),
            "external_reflector_traffic": is_reflection,
            "operator_ingress_node": source_node if is_reflection else None,
            "spoofed_request_initiator_ids": (
                list(event.get("ingress_nodes", [])) if is_reflection else []
            ),
            "quarantined_request_initiator_ids": quarantined_initiators,
            "quarantine_semantics": (
                "blocks_new_trigger_requests_but_not_already_returning_reflector_responses"
                if is_reflection
                else "blocks_direct_source_route"
            ),
            "external_segment_modeled": bool(event.get("external_segment_modeled", True)),
            "external_source_address_semantics": (
                "RFC5737_TEST-NET-2_documentation_address_not_real_reflector"
                if is_reflection
                else "internal_modeled_identity"
            ),
            "mitre_technique_id": event["mitre_technique_id"], "source_count": event["technical"]["source_count"],
        })
    return flows


def _precursor_signals(
    profile: AttackProfile,
    *,
    maturity: float = 1.0,
) -> dict[str, float | int]:
    """Сформировать только сырые признаки с плавным нарастанием 0 → 1.

    Кубическая функция smoothstep не создаёт скачка на границах окна. Поэтому
    временной анализатор видит тренд датчиков, а не бинарный флаг из расписания.
    """
    source_count = max(1, int(profile.technical.source_count))
    source_entropy = min(1.0, math.log1p(source_count) / math.log1p(1_600))
    scan_rate = min(150_000.0, float(profile.technical.packet_rate_pps) * 0.02)
    if profile.kind == AttackKind.DOS:
        confidence, acceleration, syn_growth, power_anomaly, concentration = (0.68, 0.58, 0.0, 0.0, 0.94)
    elif profile.kind == AttackKind.DDOS:
        confidence, acceleration, syn_growth, power_anomaly, concentration = (0.82, 0.86, 0.0, 0.0, 0.89)
    elif profile.kind == AttackKind.SYN_FLOOD:
        confidence, acceleration, syn_growth, power_anomaly, concentration = (0.88, 0.76, 0.93, 0.0, 0.97)
    elif profile.kind == AttackKind.BRUTE_FORCE:
        confidence, acceleration, syn_growth, power_anomaly, concentration = (0.76, 0.0, 0.0, 0.0, 1.0)
        scan_rate = 0.0
    else:
        power_anomaly = 0.94 if profile.power_failure_mode == "cyber_shutdown" else 0.0
        confidence, acceleration, syn_growth, concentration = (0.90, 0.0, 0.0, 1.0)
        scan_rate = 0.0

    maturity = min(1.0, max(0.0, float(maturity)))
    ramp = maturity * maturity * (3.0 - 2.0 * maturity)
    # При 20-секундном окне первая ступень остаётся ниже порогов классификации,
    # а вторая уже даёт различимый профиль. Поэтому два причинных отсчёта по 2 с
    # дают ожидаемое упреждение 18 с без чтения скрытого start_step.
    visible_fraction = 0.29 + 0.71 * ramp
    brownout = (
        profile.kind == AttackKind.POWER_ATTACK
        and profile.power_failure_mode == "brownout_feed_loss"
    )
    power_sag = 0.12 * visible_fraction if brownout else 0.0
    battery_discharge = 0.75 * visible_fraction if brownout else 0.0
    authentication_rate = (
        float(profile.technical.authentication_attempt_rate_per_second or 0.0)
        * visible_fraction
        if profile.kind == AttackKind.BRUTE_FORCE
        else 0.0
    )
    authentication_failure_ratio = (
        float(profile.technical.authentication_failure_ratio)
        if profile.kind == AttackKind.BRUTE_FORCE
        else 0.0
    )
    return {
        "signal_confidence": round(confidence * (0.50 + 0.50 * ramp), 6),
        "suspicious_source_count": max(1, round(source_count * visible_fraction)),
        "source_entropy_ratio": round(source_entropy * visible_fraction, 6),
        "scan_rate_pps": round(scan_rate * visible_fraction, 3),
        "traffic_acceleration_ratio": round(acceleration * visible_fraction, 6),
        "syn_backlog_growth_ratio": round(syn_growth * visible_fraction, 6),
        "power_control_anomaly_ratio": round(power_anomaly * visible_fraction, 6),
        "ups_input_voltage_ratio": round(1.0 - power_sag, 6),
        "power_voltage_sag_ratio": round(power_sag, 6),
        "battery_discharge_rate_ratio": round(battery_discharge, 6),
        "authentication_failure_rate_per_second": round(authentication_rate, 6),
        "authentication_failure_ratio": round(authentication_failure_ratio, 6),
        "account_lockout_pressure": round(min(1.0, authentication_rate), 6),
        "target_concentration_ratio": round(0.30 + (concentration - 0.30) * ramp, 6),
    }


def _precursor_summary(precursors: list[dict[str, Any]]) -> dict[str, Any]:
    signals = [item.get("signals", {}) for item in precursors]
    return {
        "precursor_confidence": max((float(item.get("signal_confidence", 0.0)) for item in precursors), default=0.0),
        "suspicious_source_count": sum(int(item.get("suspicious_source_count", 0)) for item in signals),
        "source_entropy_ratio": max((float(item.get("source_entropy_ratio", 0.0)) for item in signals), default=0.0),
        "scan_rate_pps": sum(float(item.get("scan_rate_pps", 0.0)) for item in signals),
        "traffic_acceleration_ratio": max((float(item.get("traffic_acceleration_ratio", 0.0)) for item in signals), default=0.0),
        "syn_backlog_growth_ratio": max((float(item.get("syn_backlog_growth_ratio", 0.0)) for item in signals), default=0.0),
        "power_control_anomaly_ratio": max((float(item.get("power_control_anomaly_ratio", 0.0)) for item in signals), default=0.0),
        "power_voltage_sag_ratio": max((float(item.get("power_voltage_sag_ratio", 0.0)) for item in signals), default=0.0),
        "battery_discharge_rate_ratio": max((float(item.get("battery_discharge_rate_ratio", 0.0)) for item in signals), default=0.0),
        "authentication_failure_rate_per_second": sum(
            float(item.get("authentication_failure_rate_per_second", 0.0)) for item in signals
        ),
        "authentication_failure_ratio": max(
            (float(item.get("authentication_failure_ratio", 0.0)) for item in signals), default=0.0
        ),
        "account_lockout_pressure": max(
            (float(item.get("account_lockout_pressure", 0.0)) for item in signals), default=0.0
        ),
        "target_concentration_ratio": max((float(item.get("target_concentration_ratio", 0.0)) for item in signals), default=0.0),
    }


def _event_defense_mitigation(defense_plan: dict[str, Any] | None, event: dict[str, Any]) -> float:
    if not defense_plan:
        return 0.0
    event_step = int(event.get("step_index", -2))
    legacy_step = defense_plan.get("valid_for_step")
    valid_from = defense_plan.get("valid_from_step", legacy_step)
    valid_until = defense_plan.get("valid_until_step", legacy_step)
    if valid_from is None or valid_until is None:
        return 0.0
    if not int(valid_from) <= event_step <= int(valid_until):
        return 0.0
    attack_ids = set(defense_plan.get("attack_ids", []))
    targets = set(defense_plan.get("target_ids", []))
    kinds = set(defense_plan.get("attack_kinds", []))
    entity_records = defense_plan.get("entity_records", [])
    paired_match = any(
        isinstance(item, dict)
        and item.get("target_id") == event.get("target_id")
        and item.get("kind") == event.get("kind")
        for item in entity_records
    )
    matched = event.get("attack_id") in attack_ids or paired_match
    if not entity_records:
        matched = matched or (event.get("target_id") in targets and event.get("kind") in kinds)
    if not matched:
        return 0.0
    strength = min(1.0, max(0.0, float(defense_plan.get("mitigation_strength", 1.0))))
    return float(PREVENTIVE_DEFENSE_EFFECTIVENESS.get(str(event.get("kind")), 0.55)) * strength


def _active_quarantined_sources(
    defense_plan: dict[str, Any] | None,
    event: dict[str, Any],
    *,
    mitigation_ratio: float,
) -> set[str]:
    """Вернуть только реально наблюдённые L1-источники действующего плана."""
    if not defense_plan or mitigation_ratio <= 0.0:
        return set()
    control = defense_plan.get("routing_control", {})
    values = control.get("quarantine_sources", defense_plan.get("source_ids", []))
    planned = {values} if isinstance(values, str) else {str(value) for value in values}
    # План предиктивной подготовки не изолирует предположительный источник.
    # Но когда то же воздействие уже наблюдается, заранее подготовленный
    # ingress ACL/policer может немедленно применить карантин именно к его
    # фактическим L1 ingress-источникам, не дожидаясь следующего такта.
    if (
        control.get("source_quarantine_policy")
        == "active_event_or_explicit_confirmed_plan_only"
        and set(control.get("mitigation_controls", []))
        & {"ingress_acl_or_policer", "connection_rate_limit", "syn_proxy"}
    ):
        planned.update(str(value) for value in event.get("ingress_nodes", []))
    return planned & {str(value) for value in event.get("ingress_nodes", [])}


def _allocate_sla_defense_budget(
    flows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    event_mitigation: dict[str, float],
) -> tuple[dict[tuple[str, str], float], list[dict[str, Any]]]:
    """Распределить конечный ресурс защиты строго Gold → Silver → Bronze.

    Бюджет измеряется числом потерь, которые защитный контур способен
    предотвратить за интервал. Он вычисляется из фиксированной PPS-ёмкости
    защищаемого оборудования, не растёт вместе с атакой и является общим для
    всех одновременных воздействий на шаге.
    """
    allocations: dict[tuple[str, str], float] = {}
    if not events:
        return allocations, []
    candidates: list[dict[str, Any]] = []
    for event in events:
        attack_id = str(event["attack_id"])
        base_effectiveness = float(event_mitigation.get(attack_id, 0.0))
        for flow in flows:
            if not _flow_is_affected(flow, event):
                continue
            packet_count = int(flow.get("packet_count", 0))
            raw_loss_ratio = min(
                0.95,
                float(event["legitimate_loss_ratio"])
                * float(event["intensity_ratio"])
                * float(event["target_resource_pressure"]),
            )
            raw_dropped = min(packet_count, math.ceil(packet_count * raw_loss_ratio))
            grade = str(flow.get("sla_grade", "bronze"))
            cap = min(
                0.95,
                base_effectiveness * SLA_PROTECTION_FACTOR.get(grade, 0.45),
            )
            candidates.append(
                {
                    "attack_id": attack_id,
                    "flow_id": str(flow["flow_id"]),
                    "sla_grade": grade,
                    "priority": SLA_RESTORATION_PRIORITY.get(grade, 3),
                    "raw_dropped": raw_dropped,
                    "desired_prevented": raw_dropped * cap,
                    "mitigation_cap": cap,
                }
            )

    capacity_sources: dict[str, dict[str, Any]] = {}
    for event in events:
        attack_id = str(event["attack_id"])
        if float(event_mitigation.get(attack_id, 0.0)) <= 0.0:
            continue
        target_id = str(event["target_id"])
        if target_id in capacity_sources:
            continue
        target_type = str(event.get("target_type", "unknown"))
        capacity_pps = int(DEFENSE_RECOVERY_CAPACITY_PPS.get(target_type, 500))
        interval_seconds = int(event.get("interval_seconds", 1))
        capacity_sources[target_id] = {
            "target_id": target_id,
            "target_type": target_type,
            "capacity_pps": capacity_pps,
            "interval_seconds": interval_seconds,
            "capacity_packets": capacity_pps * interval_seconds,
        }

    desired_total = sum(float(item["desired_prevented"]) for item in candidates)
    capacity_total = sum(int(item["capacity_packets"]) for item in capacity_sources.values())
    protection_budget = min(desired_total, float(capacity_total))
    remaining = protection_budget
    by_sla = {
        grade: {"desired_prevented_packets": 0.0, "allocated_prevented_packets": 0.0}
        for grade in ("gold", "silver", "bronze")
    }
    for item in sorted(
        candidates,
        key=lambda value: (int(value["priority"]), str(value["attack_id"]), str(value["flow_id"])),
    ):
        desired = float(item["desired_prevented"])
        allocated = min(desired, max(0.0, remaining))
        raw_dropped = int(item["raw_dropped"])
        ratio = min(
            float(item["mitigation_cap"]),
            allocated / raw_dropped if raw_dropped > 0 else 0.0,
        )
        allocations[(str(item["attack_id"]), str(item["flow_id"]))] = ratio
        remaining -= allocated
        grade = str(item["sla_grade"])
        by_sla.setdefault(
            grade,
            {"desired_prevented_packets": 0.0, "allocated_prevented_packets": 0.0},
        )
        by_sla[grade]["desired_prevented_packets"] += desired
        by_sla[grade]["allocated_prevented_packets"] += allocated

    report = {
        "policy": "finite_strict_priority_budget",
        "order": ["gold", "silver", "bronze"],
        "attack_ids": sorted(str(event["attack_id"]) for event in events),
        "capacity_model": "fixed_target_recovery_pps",
        "capacity_sources": list(capacity_sources.values()),
        "desired_prevented_packets": round(desired_total, 3),
        "protection_budget_packets": round(protection_budget, 3),
        "allocated_prevented_packets": round(protection_budget - max(0.0, remaining), 3),
        "unused_budget_packets": round(max(0.0, remaining), 3),
        "by_sla": {
            grade: {
                key: round(float(value), 3)
                for key, value in values.items()
            }
            for grade, values in by_sla.items()
        },
        "value_origin": "scenario_assumption",
    }
    return allocations, [report]


def _preventive_defense_summary(
    events: list[dict[str, Any]],
    defense_plan: dict[str, Any] | None,
    event_mitigation: dict[str, float],
    allocation_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    protected = [event["attack_id"] for event in events if event_mitigation.get(event["attack_id"], 0.0) > 0.0]
    return {
        "prepared": bool(defense_plan),
        "effective": bool(protected),
        "plan_id": None if not defense_plan else defense_plan.get("plan_id"),
        "valid_for_step": None if not defense_plan else defense_plan.get("valid_for_step"),
        "valid_from_step": None if not defense_plan else defense_plan.get("valid_from_step"),
        "valid_until_step": None if not defense_plan else defense_plan.get("valid_until_step"),
        "protected_attack_ids": protected,
        "mitigation_controls": (
            []
            if not defense_plan
            else list(defense_plan.get("routing_control", {}).get("mitigation_controls", []))
        ),
        "maximum_mitigation_ratio": max(event_mitigation.values(), default=0.0),
        "sla_budget_allocations": allocation_reports,
        "value_origin": "scenario_assumption_not_mitre_effectiveness_percentage",
    }


def _sla_restoration_summary(model: NetworkModel, flows: list[dict[str, Any]]) -> dict[str, Any]:
    tiers: list[dict[str, Any]] = []
    for grade in ("gold", "silver", "bronze"):
        all_grade_flows = [flow for flow in flows if flow.get("sla_grade") == grade]
        excluded = [
            flow
            for flow in all_grade_flows
            if flow.get("security_excluded_from_sla_accounting")
        ]
        grade_flows = [
            flow
            for flow in all_grade_flows
            if not flow.get("security_excluded_from_sla_accounting")
        ]
        affected = [
            flow for flow in grade_flows
            if flow.get("attack_impacted")
            or flow.get("isolated")
            or flow.get("routing_action") not in {None, "keep_baseline_route"}
        ]
        compliant = [flow for flow in grade_flows if _flow_meets_sla(model, flow)]
        coverage_values = [
            float(flow.get("slo_evaluation", {}).get("assessment_coverage_ratio", 0.0))
            for flow in grade_flows
        ]
        limited = [
            flow for flow in grade_flows
            if flow.get("slo_evaluation", {}).get("assessment_state")
            == "compliant_with_modeling_limits"
        ]
        tiers.append(
            {
                "sla_grade": grade,
                "priority": SLA_RESTORATION_PRIORITY[grade],
                "subscriber_flow_count": len(all_grade_flows),
                "flow_count": len(grade_flows),
                "security_excluded_flow_count": len(excluded),
                "affected_flow_count": len(affected),
                "sla_compliant_flow_count": len(compliant),
                "slo_limited_coverage_flow_count": len(limited),
                "mean_slo_assessment_coverage_ratio": round(
                    sum(coverage_values) / max(len(coverage_values), 1),
                    6,
                ),
                "dropped_packets": sum(int(flow.get("observed_dropped_packets", 0)) for flow in grade_flows),
                "prevented_dropped_packets": sum(int(flow.get("prevented_dropped_packets", 0)) for flow in grade_flows),
                "status": "restore_first" if grade == "gold" and affected else "queued" if affected else "healthy",
            }
        )
    return {
        "policy": "strict_priority",
        "order": ["gold", "silver", "bronze"],
        "accounting_semantics": (
            "quarantined_attributed_attack_sources_are_reported_separately_from_customer_sla"
        ),
        "slo_semantics": (
            "service_aware_all_assessed_required_metrics_must_pass; explicitly_not_modeled_metrics_do_not_fake_success"
        ),
        "slo_temporal_aggregation": (
            "current_simulation_step_aggregate_not_rolling_policy_window_percentile"
        ),
        "tiers": tiers,
    }


def _power_reserve_ratio(attrs: dict[str, Any]) -> float:
    tensor = attrs.get("l6_tensor")
    if tensor is None or "energy_reserve_ratio" not in getattr(tensor, "metric_index", {}):
        return 1.0
    return float(tensor.data[tensor.metric_index["energy_reserve_ratio"][0]])
