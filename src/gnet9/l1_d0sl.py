"""Parser and executable model for L1 d0sl subscriber policies.

The project uses a small practical subset of d0sl syntax. The file
`policies/l1_policies.d0sl` describes SLA/SLO parameters, and this module turns
that text into typed Python objects used by the topology builder.
"""

from __future__ import annotations

import re
import math
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .constants import L1_MONITORING_SECONDS


class SlaGrade(str, Enum):
    """Supported SLA grades for L1 subscribers."""

    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"


class TrafficKind(str, Enum):
    """Supported traffic classes in the current L1 model."""

    FTP = "ftp"
    DNS = "dns"
    VOICE = "voice"
    VLC_AV = "vlc_av"
    VIDEO_CONFERENCE = "video_conference"
    LIVE_STREAMING = "live_streaming"


TRAFFIC_REFERENCE_URLS: dict[str, tuple[str, ...]] = {
    "voice": (
        "https://www.rfc-editor.org/rfc/rfc7587.html",
        "https://www.itu.int/rec/T-REC-G.114",
    ),
    "vlc_av": (
        "https://www.rfc-editor.org/rfc/rfc7587.html",
        "https://www.rfc-editor.org/rfc/rfc6184.html",
    ),
    "ftp": ("https://www.rfc-editor.org/rfc/rfc959.html",),
    "dns": (
        "https://www.rfc-editor.org/rfc/rfc1035.html",
        "https://www.rfc-editor.org/rfc/rfc7766.html",
    ),
    "video_conference": (
        "https://www.w3.org/TR/webrtc/",
        "https://www.rfc-editor.org/rfc/rfc7587.html",
        "https://www.rfc-editor.org/rfc/rfc7742.html",
        "https://www.rfc-editor.org/rfc/rfc8834.html",
        "https://www.itu.int/rec/T-REC-G.114",
    ),
    "live_streaming": (
        "https://developer.apple.com/documentation/http-live-streaming/"
        "hls-authoring-specification-for-apple-devices/",
    ),
}


@dataclass(frozen=True)
class D0SLSlo:
    """One SLO condition parsed from a d0sl policy block.

    Example: p95 latency must be <= 80 ms over a 10-second window.
    """

    name: str
    metric: str
    statistic: str
    operator: str
    value: float
    unit: str
    window_seconds: int


@dataclass(frozen=True)
class D0SLSubscriberPolicy:
    """Executable L1 subscriber policy.

    This object is the bridge between the text policy and the generated network.
    The builder uses it to create L1 nodes, queue models, synthetic monitoring
    samples and L1 state tensors.
    """

    name: str
    grade: SlaGrade
    traffic: TrafficKind
    codec: str
    target_bitrate_kbps: float
    min_bitrate_kbps: float
    latency_budget_ms: float
    packet_loss_budget_percent: float
    jitter_budget_ms: float
    monitoring_interval_seconds: int
    bitrate_drop_window_seconds: int
    slo: tuple[D0SLSlo, ...]
    packetization_ms: int | None = None
    rtp_clock_rate_hz: int | None = None
    rtp_payload_type: int | None = None
    channels: int | None = None
    inband_fec: str | None = None
    dtx: str | None = None
    audio_bitrate_kbps: float | None = None
    video_bitrate_kbps: float | None = None
    audio_latency_budget_ms: float | None = None
    video_latency_budget_ms: float | None = None
    failure_latency_ms: float | None = None
    tcp_retransmission_budget_percent: float | None = None
    request_rate_qps: float | None = None
    timeout_budget_percent: float | None = None
    file_size_mib: float | None = None
    completion_time_budget_seconds: float | None = None
    startup_time_budget_ms: float | None = None
    rebuffer_ratio_budget_percent: float | None = None
    segment_deadline_miss_budget_percent: float | None = None
    primary_slo_semantics: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["grade"] = self.grade.value
        result["traffic"] = self.traffic.value
        result["reference_urls"] = list(
            TRAFFIC_REFERENCE_URLS.get(self.traffic.value, ())
        )
        result["slo_value_origin"] = (
            "explicit_gnet9_engineering_policy_not_protocol_standard"
        )
        return result


@dataclass(frozen=True)
class L1QueueModel:
    """Simple Kendall queue model for one subscriber flow.

    Current model: M/M/1/128/∞/FIFO. ``128`` is the finite system capacity;
    the subscriber request population is unbounded. It is not a full network
    simulator, but the stationary M/M/1/K probabilities are calculated
    consistently instead of using the infinite-buffer approximation.
    """

    kendall: str
    arrival_rate_pps: float
    service_rate_pps: float
    servers: int
    capacity_packets: int
    queue_discipline: str
    utilization_rho: float
    blocking_probability: float
    effective_arrival_rate_pps: float
    mean_queue_depth_packets: float
    mean_system_time_ms: float
    stability_margin: float
    arrival_rate_origin: str
    service_rate_origin: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L1MonitoringPoint:
    """One synthetic monitoring sample for one L1 subscriber."""

    second: int
    bitrate_kbps: float
    latency_ms: float
    jitter_ms: float
    packet_loss_percent: float
    queue_depth_packets: int
    utilization_rho: float
    bitrate_slo_ok: bool
    latency_slo_ok: bool
    loss_slo_ok: bool
    jitter_slo_ok: bool
    bitrate_drop_alarm: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class D0SLParseError(ValueError):
    """Raised when the L1 d0sl policy file cannot be parsed."""


class D0SLPolicyCatalog:
    """Fast policy lookup table: (grade, traffic) -> D0SLSubscriberPolicy."""

    def __init__(self, policies: list[D0SLSubscriberPolicy]) -> None:
        self.policies = policies
        self._by_grade_traffic = {(policy.grade.value, policy.traffic.value): policy for policy in policies}

    def get(self, grade: str, traffic: str) -> D0SLSubscriberPolicy:
        key = (grade, traffic)
        if key in self._by_grade_traffic:
            return self._by_grade_traffic[key]

        available = ", ".join(f"{g}/{t}" for g, t in sorted(self._by_grade_traffic))
        raise KeyError(f"No d0sl L1 policy for {grade}/{traffic}. Available: {available}")

    def to_dict(self) -> list[dict[str, Any]]:
        return [policy.to_dict() for policy in self.policies]


def load_l1_d0sl_catalog(path: Path) -> D0SLPolicyCatalog:
    """Parse the L1 d0sl file and return a lookup catalog."""
    text = _strip_d0sl_comments(path.read_text(encoding="utf-8"))
    blocks = _extract_named_blocks(text, "SLA")
    if not blocks:
        raise D0SLParseError(f"No SLA blocks found in {path}")

    policies = [_parse_sla_block(name, body) for name, body in blocks]
    return D0SLPolicyCatalog(policies)


def _strip_d0sl_comments(text: str) -> str:
    """Remove // comments. Block comments are intentionally not supported."""
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def _extract_named_blocks(text: str, keyword: str) -> list[tuple[str, str]]:
    """Extract blocks like: SLA "NAME" { ... } or SLO "NAME" { ... }."""
    blocks: list[tuple[str, str]] = []
    search_from = 0
    marker = f'{keyword} "'

    while True:
        start = text.find(marker, search_from)
        if start == -1:
            return blocks

        name_start = start + len(marker)
        name_end = text.find('"', name_start)
        if name_end == -1:
            raise D0SLParseError(f"Unclosed {keyword} name")

        name = text[name_start:name_end]
        brace_start = text.find("{", name_end)
        if brace_start == -1:
            raise D0SLParseError(f"No opening brace for {keyword} {name}")

        brace_end = _find_matching_brace(text, brace_start)
        blocks.append((name, text[brace_start + 1 : brace_end]))
        search_from = brace_end + 1


def _find_matching_brace(text: str, opening_index: int) -> int:
    """Return the index of the closing brace matching `opening_index`."""
    depth = 0
    for index in range(opening_index, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise D0SLParseError("Unclosed block")


def _parse_sla_block(name: str, body: str) -> D0SLSubscriberPolicy:
    slo_blocks = _extract_named_blocks(body, "SLO")
    if not slo_blocks:
        raise D0SLParseError(f"SLA {name} has no SLO blocks")

    scalar_body = _remove_nested_slo_blocks(body, slo_blocks)

    return D0SLSubscriberPolicy(
        name=name,
        grade=SlaGrade(_read_string_field(scalar_body, "grade")),
        traffic=TrafficKind(_read_string_field(scalar_body, "traffic")),
        codec=_read_string_field(scalar_body, "codec"),
        target_bitrate_kbps=_read_float_field(scalar_body, "target_bitrate_kbps"),
        min_bitrate_kbps=_read_float_field(scalar_body, "min_bitrate_kbps"),
        latency_budget_ms=_read_float_field(scalar_body, "latency_budget_ms"),
        packet_loss_budget_percent=_read_float_field(scalar_body, "packet_loss_budget_percent"),
        jitter_budget_ms=_read_float_field(scalar_body, "jitter_budget_ms"),
        monitoring_interval_seconds=_read_int_field(scalar_body, "monitoring_interval_seconds"),
        bitrate_drop_window_seconds=_read_int_field(scalar_body, "bitrate_drop_window_seconds"),
        slo=tuple(_parse_slo_block(slo_name, slo_body) for slo_name, slo_body in slo_blocks),
        packetization_ms=_read_optional_int_field(scalar_body, "packetization_ms"),
        rtp_clock_rate_hz=_read_optional_int_field(scalar_body, "rtp_clock_rate_hz"),
        rtp_payload_type=_read_optional_int_field(scalar_body, "rtp_payload_type"),
        channels=_read_optional_int_field(scalar_body, "channels"),
        inband_fec=_read_optional_string_field(scalar_body, "inband_fec"),
        dtx=_read_optional_string_field(scalar_body, "dtx"),
        audio_bitrate_kbps=_read_optional_float_field(scalar_body, "audio_bitrate_kbps"),
        video_bitrate_kbps=_read_optional_float_field(scalar_body, "video_bitrate_kbps"),
        audio_latency_budget_ms=_read_optional_float_field(scalar_body, "audio_latency_budget_ms"),
        video_latency_budget_ms=_read_optional_float_field(scalar_body, "video_latency_budget_ms"),
        failure_latency_ms=_read_optional_float_field(scalar_body, "failure_latency_ms"),
        tcp_retransmission_budget_percent=_read_optional_float_field(scalar_body, "tcp_retransmission_budget_percent"),
        request_rate_qps=_read_optional_float_field(scalar_body, "request_rate_qps"),
        timeout_budget_percent=_read_optional_float_field(scalar_body, "timeout_budget_percent"),
        file_size_mib=_read_optional_float_field(scalar_body, "file_size_mib"),
        completion_time_budget_seconds=_read_optional_float_field(
            scalar_body,
            "completion_time_budget_seconds",
        ),
        startup_time_budget_ms=_read_optional_float_field(scalar_body, "startup_time_budget_ms"),
        rebuffer_ratio_budget_percent=_read_optional_float_field(
            scalar_body,
            "rebuffer_ratio_budget_percent",
        ),
        segment_deadline_miss_budget_percent=_read_optional_float_field(
            scalar_body,
            "segment_deadline_miss_budget_percent",
        ),
        primary_slo_semantics=_read_optional_string_field(scalar_body, "primary_slo_semantics"),
    )


def _remove_nested_slo_blocks(body: str, slo_blocks: list[tuple[str, str]]) -> str:
    """Leave only scalar SLA fields by removing nested SLO blocks."""
    result = body
    for slo_name, slo_body in slo_blocks:
        result = result.replace(f'SLO "{slo_name}" {{' + slo_body + "}", "")
    return result


def _parse_slo_block(name: str, body: str) -> D0SLSlo:
    return D0SLSlo(
        name=name,
        metric=_read_string_field(body, "metric"),
        statistic=_read_string_field(body, "statistic"),
        operator=_read_word_or_string_field(body, "operator"),
        value=_read_float_field(body, "value"),
        unit=_read_string_field(body, "unit"),
        window_seconds=_read_int_field(body, "window_seconds"),
    )


def _read_field(body: str, field: str, field_type: str = "string") -> str | float | int:
    """Read a field from d0sl body with automatic type conversion.
    
    field_type: 'string' (quoted), 'word' (unquoted identifier), 'float', or 'int'
    """
    patterns = {
        "string": rf'\b{field}\s*:\s*"([^"]+)"\s*;',
        "word": rf'\b{field}\s*:\s*"?([A-Za-z_][A-Za-z0-9_]*)"?\s*;',
        "float": rf'\b{field}\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*;',
        "int": rf'\b{field}\s*:\s*([0-9]+)\s*;',
    }
    pattern = patterns.get(field_type, patterns["string"])
    match = re.search(pattern, body)
    if not match:
        raise D0SLParseError(f"Missing or invalid field: {field}")
    value = match.group(1)
    if field_type == "float":
        return float(value)
    if field_type == "int":
        return int(value)
    return value


def _read_string_field(body: str, field: str) -> str:
    return _read_field(body, field, "string")


def _read_word_or_string_field(body: str, field: str) -> str:
    return _read_field(body, field, "word")


def _read_float_field(body: str, field: str) -> float:
    return _read_field(body, field, "float")


def _read_int_field(body: str, field: str) -> int:
    return _read_field(body, field, "int")


def _read_optional_int_field(body: str, field: str) -> int | None:
    match = re.search(rf'\b{field}\s*:\s*([0-9]+)\s*;', body)
    return int(match.group(1)) if match else None


def _read_optional_float_field(body: str, field: str) -> float | None:
    match = re.search(rf'\b{field}\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*;', body)
    return float(match.group(1)) if match else None


def _read_optional_string_field(body: str, field: str) -> str | None:
    match = re.search(rf'\b{field}\s*:\s*"([^"]+)"\s*;', body)
    return match.group(1) if match else None


def build_l1_queue_model(policy: D0SLSubscriberPolicy, *, packet_size_bytes: int | None = None) -> L1QueueModel:
    """Build internally consistent stationary M/M/1/K/FIFO parameters."""
    if packet_size_bytes is None:
        # Opus uses one codec payload per 20 ms RTP packet (50 packets/s).
        packet_size_bytes = (
            max(1, round(policy.target_bitrate_kbps * float(policy.packetization_ms or 20) / 8.0))
            if policy.traffic is TrafficKind.VOICE
            else 1200
        )
    bits_per_packet = packet_size_bytes * 8
    if policy.traffic is TrafficKind.DNS and policy.request_rate_qps is not None:
        # Одна DNS-транзакция даёт как минимум запрос и ответ на access-линии.
        # TCP fallback моделируется отдельно в пакетной модели и не раздувает t0.
        arrival_rate = max(0.1, 2.0 * policy.request_rate_qps)
        arrival_rate_origin = "dns_query_plus_response_from_explicit_request_rate_qps"
    elif policy.traffic is TrafficKind.VOICE and policy.packetization_ms:
        arrival_rate = 1000.0 / float(policy.packetization_ms)
        arrival_rate_origin = "opus_packetization_interval"
    else:
        arrival_rate = max(0.1, policy.target_bitrate_kbps * 1000.0 / bits_per_packet)
        arrival_rate_origin = "target_payload_bitrate_divided_by_packet_size"

    # Higher SLA gets more service reserve. For low-rate control traffic the
    # latency reserve, not the bitrate multiplier, determines the scheduler
    # service rate; otherwise a 64-kbit/s DNS profile would paradoxically have
    # a queueing time above its own 30-ms budget.
    service_multiplier = {SlaGrade.GOLD: 3.2, SlaGrade.SILVER: 2.4, SlaGrade.BRONZE: 1.9}[policy.grade]
    queue_delay_target_ms = min(
        policy.latency_budget_ms * 0.25,
        {SlaGrade.GOLD: 8.0, SlaGrade.SILVER: 15.0, SlaGrade.BRONZE: 25.0}[policy.grade],
    )
    latency_driven_service_rate = arrival_rate + 1000.0 / max(queue_delay_target_ms, 0.1)
    service_rate = max(
        arrival_rate * service_multiplier,
        latency_driven_service_rate,
    )
    rho = arrival_rate / service_rate
    capacity_packets = 128
    (
        blocking_probability,
        effective_arrival_rate,
        _mean_system_packets,
        mean_queue_packets,
        mean_system_time_ms,
    ) = _mm1k_stationary_metrics(
        arrival_rate,
        service_rate,
        capacity_packets,
    )

    return L1QueueModel(
        kendall="M/M/1/128/∞/FIFO",
        arrival_rate_pps=float(arrival_rate),
        service_rate_pps=float(service_rate),
        servers=1,
        capacity_packets=capacity_packets,
        queue_discipline="FIFO",
        utilization_rho=float(rho),
        blocking_probability=float(blocking_probability),
        effective_arrival_rate_pps=float(effective_arrival_rate),
        mean_queue_depth_packets=float(mean_queue_packets),
        mean_system_time_ms=float(mean_system_time_ms),
        stability_margin=float(1.0 - rho),
        arrival_rate_origin=arrival_rate_origin,
        service_rate_origin="sla_reserve_and_latency_budget_engineering_policy",
    )


def _mm1k_stationary_metrics(
    arrival_rate: float,
    service_rate: float,
    capacity: int,
) -> tuple[float, float, float, float, float]:
    """Return ``p_K, λ_eff, L, Lq, W_ms`` for a stationary M/M/1/K queue."""
    if not math.isfinite(arrival_rate) or not math.isfinite(service_rate) or arrival_rate < 0.0 or service_rate <= 0.0 or capacity < 1:
        raise ValueError("Некорректные параметры очереди M/M/1/K")
    rho = arrival_rate / service_rate
    if arrival_rate == 0.0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    if abs(rho - 1.0) <= 1e-8:
        p0 = 1.0 / (capacity + 1.0)
        blocking = p0
        mean_system = capacity / 2.0
    else:
        # Считаем распределение с конца при перегрузке: rho**K может
        # переполнить число даже у очереди на 128 пакетов.
        indices = np.arange(capacity + 1, dtype=float)
        powers = indices if rho < 1.0 else indices - capacity
        log_rho = math.log(arrival_rate) - math.log(service_rate)
        weights = np.exp(powers * log_rho)
        probabilities = weights / weights.sum()
        p0, blocking = float(probabilities[0]), float(probabilities[-1])
        mean_system = float(probabilities @ indices)
    busy_probability = 1.0 - p0
    # При сильной перегрузке p_K округляется до 1. Считаем обслуженный
    # поток через занятость канала, иначе вычитание даёт ложный ноль.
    effective_arrival = service_rate * busy_probability if rho >= 1.0 else arrival_rate * (1.0 - blocking)
    mean_queue = max(0.0, mean_system - busy_probability)
    mean_system_time_ms = (
        0.0
        if effective_arrival <= 1e-15
        else 1000.0 * mean_system / effective_arrival
    )
    return (
        blocking,
        effective_arrival,
        mean_system,
        mean_queue,
        mean_system_time_ms,
    )


def simulate_l1_monitoring(
    policy: D0SLSubscriberPolicy,
    queue_model: L1QueueModel,
    *,
    seconds: int = L1_MONITORING_SECONDS,
    seed: int = 42,
    degraded: bool = False,
) -> list[L1MonitoringPoint]:
    """Generate reproducible one-second monitoring samples for one subscriber.

    `degraded=True` is reserved for future attack/degradation scenarios. The
    current baseline normally uses `degraded=False`.
    """
    rng = np.random.default_rng(seed)
    points: list[L1MonitoringPoint] = []
    below_bitrate_counter = 0

    for second in range(seconds):
        bitrate = _sample_bitrate(policy, rng, second, degraded)
        latency = _sample_latency(policy, queue_model, rng)
        jitter = _sample_jitter(policy, rng)
        packet_loss = _sample_packet_loss(policy, rng)
        queue_depth = int(
            rng.poisson(max(0.05, queue_model.mean_queue_depth_packets))
        )

        below_bitrate_counter = below_bitrate_counter + 1 if bitrate < policy.min_bitrate_kbps else 0
        bitrate_drop_alarm = below_bitrate_counter >= policy.bitrate_drop_window_seconds

        points.append(
            L1MonitoringPoint(
                second=second,
                bitrate_kbps=float(bitrate),
                latency_ms=float(latency),
                jitter_ms=float(jitter),
                packet_loss_percent=float(packet_loss),
                queue_depth_packets=queue_depth,
                utilization_rho=float(queue_model.utilization_rho),
                bitrate_slo_ok=bool(bitrate >= policy.min_bitrate_kbps),
                latency_slo_ok=bool(latency <= policy.latency_budget_ms),
                loss_slo_ok=bool(packet_loss <= policy.packet_loss_budget_percent),
                jitter_slo_ok=bool(jitter <= policy.jitter_budget_ms),
                bitrate_drop_alarm=bool(bitrate_drop_alarm),
            )
        )

    return points


def _sample_bitrate(policy: D0SLSubscriberPolicy, rng: np.random.Generator, second: int, degraded: bool) -> float:
    if not degraded:
        baseline = policy.target_bitrate_kbps * (1.06 + rng.normal(0.0, 0.008))
        return float(np.clip(baseline, policy.min_bitrate_kbps * 1.03, policy.target_bitrate_kbps * 1.10))

    noise = rng.normal(0.0, 0.025)
    trend = -0.055 * max(0, second - 8) if degraded else 0.0
    return float(policy.target_bitrate_kbps * max(0.25, 1.0 + noise + trend))


def _sample_latency(policy: D0SLSubscriberPolicy, queue_model: L1QueueModel, rng: np.random.Generator) -> float:
    latency_base = min(policy.latency_budget_ms * 0.35, queue_model.mean_system_time_ms + 1.5)
    sample = latency_base + rng.gamma(shape=1.4, scale=max(policy.latency_budget_ms * 0.012, 0.25))
    return float(np.clip(sample, 0.1, policy.latency_budget_ms * 0.65))


def _sample_jitter(policy: D0SLSubscriberPolicy, rng: np.random.Generator) -> float:
    sample = rng.gamma(shape=1.4, scale=max(policy.jitter_budget_ms / 28.0, 0.12))
    return float(np.clip(sample, 0.03, policy.jitter_budget_ms * 0.45))


def _sample_packet_loss(policy: D0SLSubscriberPolicy, rng: np.random.Generator) -> float:
    return float(policy.packet_loss_budget_percent * rng.uniform(0.02, 0.12))
