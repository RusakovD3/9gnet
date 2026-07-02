"""Воспроизводимые сценарии отказа в обслуживании по MITRE ATT&CK.

MITRE ATT&CK задаёт таксономию и поведение атак, но не нормативные скорости.
Численные интенсивности ниже являются явными параметрами лабораторного сценария:
они подобраны относительно ёмкости связей GNet9 и сохраняются вместе с источником
значения, чтобы будущие расчёты не смешивали стандарт и допущение модели.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any

import networkx as nx

from .models import NetworkModel


MITRE_ATTACK_SOURCE = "https://attack.mitre.org/techniques/"


class AttackKind(str, Enum):
    DOS = "dos"
    DDOS = "ddos"
    SYN_FLOOD = "syn_flood"


@dataclass(frozen=True)
class AttackTemporalCharacteristics:
    """ВВХ: временные характеристики одного воздействия."""

    start_step: int
    duration_steps: int
    rise_steps: int
    pulse_shape: str


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

    @property
    def mitre_url(self) -> str:
        return f"{MITRE_ATTACK_SOURCE}{self.mitre_technique_id.replace('.', '/')}/"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["mitre_url"] = self.mitre_url
        return data


# Расписание рассчитано на стандартные 10 переходов по 5 секунд.
MITRE_DEMO_ATTACKS = (
    AttackProfile(
        "DOS_CORE_C1", "DoS на маршрутизатор ядра C1", AttackKind.DOS,
        "T1498.001", "Direct Network Flood", (), "C1", "core-router", ("C7",),
        AttackTemporalCharacteristics(2, 2, 1, "ramp-then-peak"),
        AttackTechnicalCharacteristics("UDP", 512, 3_200.0, 780_000, 1, 1.0, 0.0),
        38.0, 0.055, 0.62,
        "Один источник создаёт прямой UDP flood и повышает загрузку плоскости обработки C1.",
    ),
    AttackProfile(
        "DDOS_MEDIA_SERVER", "DDoS с отражённым усилением на медиасервер", AttackKind.DDOS,
        "T1498.002", "Reflection Amplification", (), "SRV_MEDIA", "service-server",
        ("C2", "C6", "C10", "C12"),
        AttackTemporalCharacteristics(5, 2, 1, "distributed-ramp-then-peak"),
        AttackTechnicalCharacteristics("UDP", 1_200, 32_000.0, 3_250_000, 1_600, 28.0, 0.0),
        190.0, 0.22, 0.94,
        "Распределённые отражённые ответы перегружают оба 25-Гбит/с подключения SRV_MEDIA.",
    ),
    AttackProfile(
        "SYN_GOLD_ENDPOINT", "SYN flood на оконечное устройство Gold", AttackKind.SYN_FLOOD,
        "T1499.001", "OS Exhaustion Flood", ("T1498.001",), "M1_01", "mobile-subscriber",
        ("C11", "C8"),
        AttackTemporalCharacteristics(8, 2, 1, "distributed-ramp-then-peak"),
        AttackTechnicalCharacteristics("TCP SYN", 74, 142.0, 240_000, 320, 1.0, 0.998),
        420.0, 0.38, 0.88,
        "Поток SYN заполняет очередь полуоткрытых соединений и радиоканал Gold-абонента M1_01.",
    ),
)


TRAFFIC_SERVICE_NODES = {
    "voice": "SVC_VOICE",
    "broadcast_mp3": "SVC_VLC",
    "vlc_av": "SVC_VLC",
    "ftp": "SVC_FTP",
    "dns": "SVC_DNS",
    "video_conference": "SVC_TELEMOST",
    "live_streaming": "SVC_LIVE",
}


def attack_catalog() -> list[dict[str, Any]]:
    return [profile.to_dict() for profile in MITRE_DEMO_ATTACKS]


def mark_critical_nodes(model: NetworkModel) -> dict[str, Any]:
    """Пометить узлы, которые обслуживают хотя бы один Gold-поток.

    КВУ здесь — коэффициент критической вовлечённости: доля Gold-маршрутов,
    проходящих через узел, с дополнительным весом для сервиса и его сервера.
    Это подготовительный приоритет защиты, а не решение о переназначении.
    """
    gold_subscribers = [
        node for node, attrs in model.graph.nodes(data=True)
        if attrs.get("level") == "L1" and attrs.get("sla_grade") == "gold"
    ]
    transit_counts = {node: 0 for node in model.graph.nodes}
    service_counts = {node: 0 for node in model.graph.nodes}
    routes: list[list[str]] = []
    for subscriber in gold_subscribers:
        attrs = model.graph.nodes[subscriber]
        service = TRAFFIC_SERVICE_NODES.get(attrs.get("traffic_kind"))
        if service not in model.graph:
            continue
        server = model.graph.nodes[service].get("hosted_on", service)
        route = nx.shortest_path(model.graph, subscriber, server, weight="latency_ms")
        routes.append(route)
        for node in route:
            transit_counts[node] += 1
        service_counts[service] += 1
        service_counts[server] += 1

    denominator = max(len(routes), 1)
    critical_nodes: list[str] = []
    for node, attrs in model.graph.nodes(data=True):
        transit = transit_counts[node]
        hosted = service_counts[node]
        is_gold_endpoint = attrs.get("level") == "L1" and attrs.get("sla_grade") == "gold"
        is_critical = transit > 0 or hosted > 0 or is_gold_endpoint
        involvement = min(1.0, transit / denominator + (0.20 if hosted else 0.0))
        attrs["critical_protection"] = {
            "is_critical": is_critical,
            "protection_priority": 1 if is_critical else 3,
            "gold_transit_flow_count": transit,
            "gold_service_flow_count": hosted,
            "critical_involvement_coefficient": round(involvement, 6),
            "maximum_safe_utilization_percent": 80.0 if is_critical else 90.0,
            "policy": "do_not_accept_silver_or_bronze_remap_above_80_percent" if is_critical else "standard",
        }
        if is_critical:
            critical_nodes.append(node)
    return {
        "gold_subscriber_count": len(gold_subscribers),
        "gold_route_count": len(routes),
        "critical_node_count": len(critical_nodes),
        "critical_nodes": sorted(critical_nodes),
    }


def active_attack_events(
    model: NetworkModel,
    scenario: str,
    *,
    step_index: int,
    time_seconds: int,
    step_seconds: int,
) -> list[dict[str, Any]]:
    if scenario == "none":
        return []
    if scenario != "mitre-demo":
        raise ValueError(f"Неизвестный сценарий атак: {scenario}")

    events: list[dict[str, Any]] = []
    for profile in MITRE_DEMO_ATTACKS:
        relative_step = step_index - profile.temporal.start_step
        if relative_step < 0 or relative_step >= profile.temporal.duration_steps:
            continue
        intensity = min(1.0, (relative_step + 1) / max(profile.temporal.rise_steps + 1, 1) + 0.25)
        intensity = round(intensity, 4)
        routes = [
            nx.shortest_path(model.graph, ingress, profile.target_id, weight="latency_ms")
            for ingress in profile.ingress_nodes
        ]
        target_criticality = model.graph.nodes[profile.target_id].get("critical_protection", {})
        events.append({
            **profile.to_dict(),
            "step_index": step_index,
            "time_seconds": time_seconds,
            "interval_seconds": step_seconds,
            "relative_step": relative_step,
            "intensity_ratio": intensity,
            "routes": routes,
            "target_criticality": target_criticality,
            "effective_offered_rate_mbps": round(profile.technical.offered_rate_mbps * intensity, 4),
            "effective_packet_rate_pps": int(profile.technical.packet_rate_pps * intensity),
        })
    return events


def apply_attack_effects(
    model: NetworkModel,
    legitimate_flows: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ухудшить затронутые легитимные потоки и добавить агрегированные flood-потоки."""
    flows = [dict(flow) for flow in legitimate_flows]
    for flow in flows:
        flow["is_attack_traffic"] = False
        for event in events:
            if not _flow_is_affected(flow, event):
                continue
            intensity = float(event["intensity_ratio"])
            loss_ratio = min(0.95, float(event["legitimate_loss_ratio"]) * intensity)
            dropped = min(int(flow["packet_count"]), math.ceil(int(flow["packet_count"]) * loss_ratio))
            flow["observed_dropped_packets"] = int(flow.get("observed_dropped_packets", 0)) + dropped
            if flow.get("transport") == "TCP":
                flow["observed_retransmissions"] = int(flow.get("observed_retransmissions", 0)) + dropped
            flow["one_way_latency_ms"] = round(
                float(flow.get("one_way_latency_ms", 0.0)) + float(event["latency_penalty_ms"]) * intensity, 4
            )
            if flow.get("rtt_ms") is not None:
                flow["rtt_ms"] = round(float(flow["rtt_ms"]) + 2.0 * float(event["latency_penalty_ms"]) * intensity, 4)
            flow.setdefault("attack_ids", []).append(event["attack_id"])
            flow["attack_impacted"] = True

    attack_flows = [flow for event in events for flow in _build_attack_flows(event)]
    all_flows = flows + attack_flows
    legitimate_packets = sum(int(flow["packet_count"]) for flow in flows)
    legitimate_dropped = sum(int(flow["observed_dropped_packets"]) for flow in flows)
    impacted = [flow for flow in flows if flow.get("attack_impacted")]
    impacted_gold = [flow for flow in impacted if flow.get("sla_grade") == "gold"]
    gold_flows = [flow for flow in flows if flow.get("sla_grade") == "gold"]
    gold_healthy = [flow for flow in gold_flows if _flow_meets_sla(model, flow)]
    attack_packets = sum(int(flow["packet_count"]) for flow in attack_flows)
    attack_wire = sum(int(flow["wire_bytes"]) for flow in attack_flows)
    interval = int(flows[0].get("interval_seconds", 1)) if flows else 1
    max_pressure = max((float(event["target_resource_pressure"]) * float(event["intensity_ratio"]) for event in events), default=0.0)
    target_observations = [_target_observation(model, event) for event in events]
    return all_flows, {
        "scenario": "mitre-demo" if events else "none",
        "active": bool(events),
        "active_count": len(events),
        "events": events,
        "attack_packet_count": attack_packets,
        "attack_wire_bytes": attack_wire,
        "attack_rate_mbps": round(attack_wire * 8.0 / max(interval, 1) / 1_000_000.0, 4),
        "maximum_intensity_ratio": max((float(event["intensity_ratio"]) for event in events), default=0.0),
        "target_resource_pressure": round(max_pressure, 6),
        "target_observations": target_observations,
        "maximum_target_utilization_percent": max(
            (item["estimated_resource_utilization_percent"] for item in target_observations), default=0.0
        ),
        "impacted_legitimate_flow_count": len(impacted),
        "impacted_gold_flow_count": len(impacted_gold),
        "legitimate_packet_count": legitimate_packets,
        "legitimate_dropped_packets": legitimate_dropped,
        "legitimate_loss_ratio": legitimate_dropped / max(legitimate_packets, 1),
        "legitimate_delivery_ratio": (legitimate_packets - legitimate_dropped) / max(legitimate_packets, 1),
        "gold_sla_compliance_ratio": len(gold_healthy) / max(len(gold_flows), 1),
    }


def _flow_is_affected(flow: dict[str, Any], event: dict[str, Any]) -> bool:
    target = event["target_id"]
    return target in flow.get("route", []) or target in {
        flow.get("client_node"), flow.get("server_node"), flow.get("service_node")
    }


def _flow_meets_sla(model: NetworkModel, flow: dict[str, Any]) -> bool:
    client = flow.get("client_node")
    if client not in model.graph:
        return False
    attrs = model.graph.nodes[client]
    policy = attrs.get("d0sl_policy", {})
    latency_budget = float(policy.get("latency_budget_ms", attrs.get("latency_budget_ms", math.inf)))
    loss_budget_percent = float(policy.get("packet_loss_budget_percent", math.inf))
    observed_loss_percent = int(flow.get("observed_dropped_packets", 0)) / max(int(flow.get("packet_count", 0)), 1) * 100.0
    return float(flow.get("one_way_latency_ms", 0.0)) <= latency_budget and observed_loss_percent <= loss_budget_percent


def _target_observation(model: NetworkModel, event: dict[str, Any]) -> dict[str, Any]:
    attrs = model.graph.nodes[event["target_id"]]
    pressure = float(event["target_resource_pressure"]) * float(event["intensity_ratio"])
    if attrs.get("role") == "service-server":
        baseline_cpu = float(attrs.get("runtime", {}).get("cpu_util_percent", 0.0))
    elif attrs.get("level") == "L2":
        baseline_cpu = float(attrs.get("l2_raw_baseline", {}).get("cpu_util", 0.0))
    else:
        baseline_cpu = float(attrs.get("kendall_queue", {}).get("utilization_rho", 0.0)) * 100.0
    capacities = [
        float(model.graph.edges[event["target_id"], neighbor].get("capacity_mbps", 0.0))
        for neighbor in model.graph.neighbors(event["target_id"])
        if model.graph.edges[event["target_id"], neighbor].get("medium") != "logical-service-binding"
    ]
    ingress_capacity = max(sum(capacities), 1.0)
    network_utilization = min(100.0, float(event["effective_offered_rate_mbps"]) / ingress_capacity * 100.0)
    return {
        "target_id": event["target_id"],
        "target_type": event["target_type"],
        "estimated_resource_utilization_percent": round(min(100.0, baseline_cpu + pressure * 88.0), 4),
        "estimated_network_utilization_percent": round(network_utilization, 4),
        "estimated_queue_utilization_percent": round(min(100.0, pressure * 100.0), 4),
        "estimated_availability_ratio": round(max(0.0, 1.0 - pressure * 0.92), 6),
        "syn_backlog_exhaustion_ratio": round(
            pressure * float(event["technical"].get("incomplete_handshake_ratio", 0.0)), 6
        ),
        "critical_protection": attrs.get("critical_protection", {}),
    }


def _build_attack_flows(event: dict[str, Any]) -> list[dict[str, Any]]:
    routes = event.get("routes", [])
    if not routes:
        return []
    app = {
        "dos": "ATTACK_DOS",
        "ddos": "ATTACK_DDOS",
        "syn_flood": "ATTACK_SYN",
    }[event["kind"]]
    transport = "TCP" if event["kind"] == "syn_flood" else "UDP"
    total_packets = int(event["effective_packet_rate_pps"] * event["interval_seconds"])
    total_wire = int(event["effective_offered_rate_mbps"] * 1_000_000.0 / 8.0 * event["interval_seconds"])
    flows: list[dict[str, Any]] = []
    for index, route in enumerate(routes):
        packet_count = total_packets // len(routes) + (1 if index < total_packets % len(routes) else 0)
        wire_bytes = total_wire // len(routes) + (1 if index < total_wire % len(routes) else 0)
        pressure = float(event["target_resource_pressure"]) * float(event["intensity_ratio"])
        dropped = math.floor(packet_count * min(0.92, 0.20 + pressure * 0.65))
        flows.append({
            "flow_id": f"{event['attack_id']}-{event['step_index']}-{index + 1}",
            "step_index": event["step_index"], "time_seconds": event["time_seconds"],
            "interval_seconds": event["interval_seconds"], "application": app,
            "transport": transport, "client_node": route[0], "service_node": event["target_id"],
            "server_node": event["target_id"], "client_ip": "external-spoofed",
            "server_ip": "target", "client_port": 0, "server_port": 0,
            "packet_count": packet_count, "payload_bytes": 0, "wire_bytes": wire_bytes,
            "mtu_bytes": event["technical"]["packet_size_bytes"], "route": route,
            "reverse_route": list(reversed(route)), "hop_count": max(0, len(route) - 1),
            "one_way_latency_ms": round(float(event["latency_penalty_ms"]) * 0.08, 4),
            "rtt_ms": None, "expected_loss_ratio": 0.0,
            "observed_dropped_packets": dropped, "observed_retransmissions": 0,
            "sla_grade": "attack", "traffic_kind": event["kind"], "sequence_base": 0,
            "is_attack_traffic": True, "attack_id": event["attack_id"],
            "mitre_technique_id": event["mitre_technique_id"], "source_count": event["technical"]["source_count"],
        })
    return flows
