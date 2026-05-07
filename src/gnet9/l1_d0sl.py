"""Parser and executable model for L1 d0sl subscriber policies.

The project uses a small practical subset of d0sl syntax. The file
`policies/l1_policies.d0sl` describes SLA/SLO parameters, and this module turns
that text into typed Python objects used by the topology builder.
"""

from __future__ import annotations

import re
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

    BROADCAST_MP3 = "broadcast_mp3"
    FTP = "ftp"
    DNS = "dns"


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
    samples and L1 tensors.
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

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["grade"] = self.grade.value
        result["traffic"] = self.traffic.value
        return result


@dataclass(frozen=True)
class L1QueueModel:
    """Simple Kendall queue model for one subscriber flow.

    Current model: M/M/1/128/finite/FIFO.
    It is not a full network simulator, but it gives a useful baseline load and
    stability estimate for every L1 subscriber.
    """

    kendall: str
    arrival_rate_pps: float
    service_rate_pps: float
    servers: int
    capacity_packets: int
    queue_discipline: str
    utilization_rho: float
    mean_system_time_ms: float
    stability_margin: float

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


def build_l1_policy(grade: str, traffic: str) -> D0SLSubscriberPolicy:
    """Build a fallback policy without reading d0sl.

    Main project flow uses `load_l1_d0sl_catalog()`. This function is left for
    tests, notebooks and quick experiments.
    """
    grade_enum = SlaGrade(grade)
    traffic_enum = TrafficKind(traffic)

    policy_params = {
        "bitrate": {SlaGrade.GOLD: 320.0, SlaGrade.SILVER: 128.0, SlaGrade.BRONZE: 64.0}[grade_enum],
        "latency": {TrafficKind.BROADCAST_MP3: 80.0, TrafficKind.FTP: 300.0, TrafficKind.DNS: 30.0}[traffic_enum],
        "jitter": {TrafficKind.BROADCAST_MP3: 30.0, TrafficKind.FTP: 100.0, TrafficKind.DNS: 10.0}[traffic_enum],
        "loss": {SlaGrade.GOLD: 0.5, SlaGrade.SILVER: 1.0, SlaGrade.BRONZE: 2.0}[grade_enum],
    }

    policy_name = f"L1_{grade_enum.value.upper()}_{traffic_enum.value.upper()}"
    slo = (
        D0SLSlo("Bitrate", "bitrate_kbps", "p95", "GTE", policy_params["bitrate"], "kbps", 10),
        D0SLSlo("Latency", "latency_ms", "p95", "LTE", policy_params["latency"], "ms", 10),
        D0SLSlo("PacketLoss", "packet_loss_percent", "avg", "LTE", policy_params["loss"], "%", 10),
        D0SLSlo("Jitter", "jitter_ms", "p95", "LTE", policy_params["jitter"], "ms", 10),
    )

    return D0SLSubscriberPolicy(
        name=policy_name,
        grade=grade_enum,
        traffic=traffic_enum,
        codec="MP3/VLC" if traffic_enum == TrafficKind.BROADCAST_MP3 else "TCP/IP",
        target_bitrate_kbps=policy_params["bitrate"],
        min_bitrate_kbps=policy_params["bitrate"],
        latency_budget_ms=policy_params["latency"],
        packet_loss_budget_percent=policy_params["loss"],
        jitter_budget_ms=policy_params["jitter"],
        monitoring_interval_seconds=1,
        bitrate_drop_window_seconds=10,
        slo=slo,
    )


def build_l1_queue_model(policy: D0SLSubscriberPolicy, *, packet_size_bytes: int = 1200) -> L1QueueModel:
    """Build M/M/1/K/FIFO queue parameters for one subscriber flow."""
    bits_per_packet = packet_size_bytes * 8
    arrival_rate = max(0.1, policy.target_bitrate_kbps * 1000.0 / bits_per_packet)

    # Higher SLA gets more service reserve, therefore lower utilization rho.
    service_multiplier = {SlaGrade.GOLD: 3.2, SlaGrade.SILVER: 2.4, SlaGrade.BRONZE: 1.9}[policy.grade]
    service_rate = arrival_rate * service_multiplier
    rho = arrival_rate / service_rate

    return L1QueueModel(
        kendall="M/M/1/128/finite/FIFO",
        arrival_rate_pps=float(arrival_rate),
        service_rate_pps=float(service_rate),
        servers=1,
        capacity_packets=128,
        queue_discipline="FIFO",
        utilization_rho=float(rho),
        mean_system_time_ms=float(1000.0 / max(service_rate - arrival_rate, 1e-9)),
        stability_margin=float(1.0 - rho),
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
        queue_depth = int(rng.poisson(max(1.0, queue_model.utilization_rho * 12.0)))

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
    noise = rng.normal(0.0, 0.025)
    trend = -0.055 * max(0, second - 8) if degraded else 0.0
    return float(policy.target_bitrate_kbps * max(0.25, 1.0 + noise + trend))


def _sample_latency(policy: D0SLSubscriberPolicy, queue_model: L1QueueModel, rng: np.random.Generator) -> float:
    latency_base = min(policy.latency_budget_ms * 0.45, queue_model.mean_system_time_ms + 2.0)
    return float(max(0.1, latency_base + rng.gamma(shape=1.6, scale=0.9)))


def _sample_jitter(policy: D0SLSubscriberPolicy, rng: np.random.Generator) -> float:
    return float(max(0.05, rng.gamma(shape=1.5, scale=max(policy.jitter_budget_ms / 18.0, 0.2))))


def _sample_packet_loss(policy: D0SLSubscriberPolicy, rng: np.random.Generator) -> float:
    return float(max(0.0, rng.normal(policy.packet_loss_budget_percent * 0.25, 0.05)))


def l1_tensor_metrics_from_monitoring(
    policy: D0SLSubscriberPolicy,
    queue_model: L1QueueModel,
    points: list[L1MonitoringPoint],
) -> dict[str, float]:
    """Convert raw monitoring samples into normalized L1 tensor metrics.

    The tensor stores compact indicators, not the whole monitoring history.
    Full history is exported separately to `l1_monitoring.csv`.
    """
    bitrates = np.array([point.bitrate_kbps for point in points], dtype=float)
    latencies = np.array([point.latency_ms for point in points], dtype=float)
    losses = np.array([point.packet_loss_percent for point in points], dtype=float)
    jitters = np.array([point.jitter_ms for point in points], dtype=float)

    return {
        "sla_grade": _grade_to_value(policy.grade),
        "slo_bitrate_ratio": _bounded_ratio(np.percentile(bitrates, 5), policy.min_bitrate_kbps),
        "sli_latency_ratio": _bounded_ratio(policy.latency_budget_ms, np.percentile(latencies, 95)),
        "sli_loss_ratio": _bounded_ratio(policy.packet_loss_budget_percent, float(np.mean(losses))),
        "queue_stability": float(np.clip(queue_model.stability_margin, 0.0, 1.0)),
        "sli_jitter_ratio": _bounded_ratio(policy.jitter_budget_ms, np.percentile(jitters, 95)),
        "traffic_class": _traffic_to_value(policy.traffic),
        "monitoring_period": min(1.0, len(points) / L1_MONITORING_SECONDS),
        "bitrate_drop_alarm": 1.0 if any(point.bitrate_drop_alarm for point in points) else 0.0,
    }


def _bounded_ratio(numerator: float, denominator: float) -> float:
    """Normalize a positive ratio to 0..1 with soft clipping at 1.5."""
    return float(np.clip(numerator / max(denominator, 1e-9), 0.0, 1.5) / 1.5)


def _grade_to_value(grade: SlaGrade) -> float:
    return {SlaGrade.GOLD: 1.0, SlaGrade.SILVER: 0.72, SlaGrade.BRONZE: 0.45}[grade]


def _traffic_to_value(traffic: TrafficKind) -> float:
    return {TrafficKind.BROADCAST_MP3: 1.0, TrafficKind.FTP: 0.70, TrafficKind.DNS: 0.85}[traffic]
