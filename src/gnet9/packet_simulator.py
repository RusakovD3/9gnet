"""In-memory TCP/IP packet model for G-Net dynamics.

The simulator does not open sockets and never sends packets through the host OS.
It builds realistic packet/header events inside Python so the dynamic snapshots
can expose packet-level traffic while staying deterministic and safe.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import networkx as nx

from .models import NetworkModel, StateTensor


ETHERNET_HEADER_BYTES = 14
ETHERNET_FCS_BYTES = 4
IPV4_HEADER_BYTES = 20
TCP_HEADER_BYTES = 20
UDP_HEADER_BYTES = 8
ETHERNET_MTU_BYTES = 1500
TCP_MSS_BYTES = ETHERNET_MTU_BYTES - IPV4_HEADER_BYTES - TCP_HEADER_BYTES
UDP_PAYLOAD_BYTES = 1180
DEFAULT_TTL = 64
PacketDetail = Literal["summary", "flows", "sample"]


TRAFFIC_APPS = {
    "broadcast_mp3": {
        "application": "RTP_MP3",
        "transport": "UDP",
        "service_node": "SVC_VIDEO",
        "server_port": 5004,
        "payload_unit_bytes": UDP_PAYLOAD_BYTES,
    },
    "ftp": {
        "application": "FTP_DATA",
        "transport": "TCP",
        "service_node": "SVC_FTP",
        "server_port": 21,
        "payload_unit_bytes": TCP_MSS_BYTES,
    },
    "dns": {
        "application": "DNS",
        "transport": "UDP",
        "service_node": "SVC_TELEM",
        "server_port": 53,
        "query_payload_bytes": 52,
        "response_payload_bytes": 180,
    },
}


@dataclass(frozen=True)
class NetworkIdentity:
    """Stable simulated L2/L3 endpoint identity for one graph node."""

    node_id: str
    ip: str
    mac: str
    mtu_bytes: int = ETHERNET_MTU_BYTES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_network_identities(model: NetworkModel) -> dict[str, NetworkIdentity]:
    """Assign deterministic private IPv4 and locally-administered MAC addresses."""
    identities: dict[str, NetworkIdentity] = {}
    used_ips: set[str] = set()

    for index, (node_id, attrs) in enumerate(sorted(model.graph.nodes(data=True)), start=1):
        ip = _ip_for_node(node_id, attrs, index)
        if ip in used_ips:
            ip = f"10.254.{index // 255}.{index % 255 or 1}"
        used_ips.add(ip)
        identities[node_id] = NetworkIdentity(node_id=node_id, ip=ip, mac=_mac_for_node(node_id))

    return identities


def simulate_packet_snapshot(
    model: NetworkModel,
    *,
    step_index: int,
    time_seconds: int,
    step_seconds: int,
    detail: PacketDetail = "sample",
    packet_sample_limit: int = 48,
) -> dict[str, Any]:
    """Build simulated traffic for one step.

    detail=summary stores only aggregate counters, flows adds one flow record per
    subscriber, and sample also stores representative TCP/UDP packet headers.
    """
    if detail not in {"summary", "flows", "sample"}:
        raise ValueError("Packet detail must be one of: summary, flows, sample")

    identities = build_network_identities(model)
    flows: list[dict[str, Any]] = []
    packet_sample: list[dict[str, Any]] = []

    subscribers = [
        (node_id, attrs)
        for node_id, attrs in sorted(model.graph.nodes(data=True))
        if attrs.get("level") == "L1"
    ]

    for flow_index, (subscriber_id, attrs) in enumerate(subscribers, start=1):
        flow = _build_flow(model, identities, subscriber_id, attrs, flow_index, step_index, time_seconds, step_seconds)
        flows.append(flow)
        if detail == "sample" and len(packet_sample) < packet_sample_limit:
            remaining = packet_sample_limit - len(packet_sample)
            packet_sample.extend(_sample_packets_for_flow(flow, identities, model, remaining))

    result: dict[str, Any] = {"summary": _traffic_summary(flows)}
    if detail in {"flows", "sample"}:
        result["flows"] = flows
    if detail == "sample":
        result["packet_sample"] = packet_sample
    return result


def _build_flow(
    model: NetworkModel,
    identities: dict[str, NetworkIdentity],
    subscriber_id: str,
    attrs: dict[str, Any],
    flow_index: int,
    step_index: int,
    time_seconds: int,
    step_seconds: int,
) -> dict[str, Any]:
    traffic_kind = attrs.get("traffic_kind")
    app = TRAFFIC_APPS[traffic_kind]
    service_node = app["service_node"]
    client_port = _client_port(subscriber_id, flow_index)
    payload_bps = _payload_bps(attrs, time_seconds)

    if app["transport"] == "TCP":
        return _build_tcp_flow(
            model,
            identities,
            subscriber_id,
            service_node,
            attrs,
            app,
            flow_index,
            step_index,
            time_seconds,
            step_seconds,
            client_port,
            payload_bps,
        )

    if traffic_kind == "dns":
        return _build_dns_flow(
            model,
            identities,
            subscriber_id,
            service_node,
            attrs,
            app,
            flow_index,
            step_index,
            time_seconds,
            step_seconds,
            client_port,
        )

    return _build_udp_media_flow(
        model,
        identities,
        subscriber_id,
        service_node,
        attrs,
        app,
        flow_index,
        step_index,
        time_seconds,
        step_seconds,
        client_port,
        payload_bps,
    )


def _build_tcp_flow(
    model: NetworkModel,
    identities: dict[str, NetworkIdentity],
    subscriber_id: str,
    service_node: str,
    attrs: dict[str, Any],
    app: dict[str, Any],
    flow_index: int,
    step_index: int,
    time_seconds: int,
    step_seconds: int,
    client_port: int,
    payload_bps: float,
) -> dict[str, Any]:
    data_route = _route(model, service_node, subscriber_id)
    ack_route = list(reversed(data_route))
    payload_bytes = max(TCP_MSS_BYTES, int(payload_bps * step_seconds / 8.0))
    data_segments = math.ceil(payload_bytes / TCP_MSS_BYTES)
    ack_segments = math.ceil(data_segments / 2)
    handshake_packets = 3 if step_index == 0 else 0
    tcp_packets = handshake_packets + data_segments + ack_segments
    wire_bytes = (
        payload_bytes
        + tcp_packets * (ETHERNET_HEADER_BYTES + ETHERNET_FCS_BYTES + IPV4_HEADER_BYTES + TCP_HEADER_BYTES)
    )
    one_way_latency_ms = _path_latency_ms(model, data_route, TCP_MSS_BYTES + IPV4_HEADER_BYTES + TCP_HEADER_BYTES)
    rtt_ms = one_way_latency_ms + _path_latency_ms(model, ack_route, IPV4_HEADER_BYTES + TCP_HEADER_BYTES)
    expected_loss = _path_expected_loss(model, data_route)

    flow_id = f"{subscriber_id}-{app['application']}-{step_index}"
    return {
        "flow_id": flow_id,
        "step_index": step_index,
        "time_seconds": time_seconds,
        "application": app["application"],
        "transport": "TCP",
        "client_node": subscriber_id,
        "server_node": service_node,
        "client_ip": identities[subscriber_id].ip,
        "server_ip": identities[service_node].ip,
        "client_port": client_port,
        "server_port": app["server_port"],
        "tcp_state": "ESTABLISHED",
        "handshake_packets": handshake_packets,
        "data_segments": data_segments,
        "ack_segments": ack_segments,
        "packet_count": tcp_packets,
        "payload_bytes": payload_bytes,
        "wire_bytes": wire_bytes,
        "mss_bytes": TCP_MSS_BYTES,
        "mtu_bytes": ETHERNET_MTU_BYTES,
        "route": data_route,
        "reverse_route": ack_route,
        "hop_count": max(0, len(data_route) - 1),
        "one_way_latency_ms": round(one_way_latency_ms, 4),
        "rtt_ms": round(rtt_ms, 4),
        "expected_loss_ratio": round(expected_loss, 8),
        "observed_dropped_packets": 0,
        "observed_retransmissions": 0,
        "sla_grade": attrs.get("sla_grade"),
        "traffic_kind": attrs.get("traffic_kind"),
        "sequence_base": _sequence_base(flow_id),
    }


def _build_udp_media_flow(
    model: NetworkModel,
    identities: dict[str, NetworkIdentity],
    subscriber_id: str,
    service_node: str,
    attrs: dict[str, Any],
    app: dict[str, Any],
    flow_index: int,
    step_index: int,
    time_seconds: int,
    step_seconds: int,
    client_port: int,
    payload_bps: float,
) -> dict[str, Any]:
    route = _route(model, service_node, subscriber_id)
    payload_unit = app["payload_unit_bytes"]
    payload_bytes = max(payload_unit, int(payload_bps * step_seconds / 8.0))
    datagrams = math.ceil(payload_bytes / payload_unit)
    wire_bytes = (
        payload_bytes
        + datagrams * (ETHERNET_HEADER_BYTES + ETHERNET_FCS_BYTES + IPV4_HEADER_BYTES + UDP_HEADER_BYTES)
    )
    one_way_latency_ms = _path_latency_ms(model, route, payload_unit + IPV4_HEADER_BYTES + UDP_HEADER_BYTES)
    expected_loss = _path_expected_loss(model, route)

    flow_id = f"{subscriber_id}-{app['application']}-{step_index}"
    return {
        "flow_id": flow_id,
        "step_index": step_index,
        "time_seconds": time_seconds,
        "application": app["application"],
        "transport": "UDP",
        "client_node": subscriber_id,
        "server_node": service_node,
        "client_ip": identities[subscriber_id].ip,
        "server_ip": identities[service_node].ip,
        "client_port": client_port,
        "server_port": app["server_port"],
        "udp_datagrams": datagrams,
        "packet_count": datagrams,
        "payload_bytes": payload_bytes,
        "wire_bytes": wire_bytes,
        "mtu_bytes": ETHERNET_MTU_BYTES,
        "route": route,
        "reverse_route": list(reversed(route)),
        "hop_count": max(0, len(route) - 1),
        "one_way_latency_ms": round(one_way_latency_ms, 4),
        "rtt_ms": None,
        "expected_loss_ratio": round(expected_loss, 8),
        "observed_dropped_packets": 0,
        "observed_retransmissions": 0,
        "sla_grade": attrs.get("sla_grade"),
        "traffic_kind": attrs.get("traffic_kind"),
        "sequence_base": _sequence_base(flow_id),
    }


def _build_dns_flow(
    model: NetworkModel,
    identities: dict[str, NetworkIdentity],
    subscriber_id: str,
    service_node: str,
    attrs: dict[str, Any],
    app: dict[str, Any],
    flow_index: int,
    step_index: int,
    time_seconds: int,
    step_seconds: int,
    client_port: int,
) -> dict[str, Any]:
    query_route = _route(model, subscriber_id, service_node)
    response_route = list(reversed(query_route))
    request_rate = _tensor_metric(attrs.get("tensor"), "request_rate_pps", default=1.0)
    query_count = max(1, int(round(request_rate * step_seconds)))
    query_bytes = int(app["query_payload_bytes"])
    response_bytes = int(app["response_payload_bytes"])
    payload_bytes = query_count * (query_bytes + response_bytes)
    packet_count = query_count * 2
    wire_bytes = (
        payload_bytes
        + packet_count * (ETHERNET_HEADER_BYTES + ETHERNET_FCS_BYTES + IPV4_HEADER_BYTES + UDP_HEADER_BYTES)
    )
    query_latency_ms = _path_latency_ms(model, query_route, query_bytes + IPV4_HEADER_BYTES + UDP_HEADER_BYTES)
    response_latency_ms = _path_latency_ms(model, response_route, response_bytes + IPV4_HEADER_BYTES + UDP_HEADER_BYTES)
    expected_loss = 1.0 - (1.0 - _path_expected_loss(model, query_route)) * (1.0 - _path_expected_loss(model, response_route))

    flow_id = f"{subscriber_id}-{app['application']}-{step_index}"
    return {
        "flow_id": flow_id,
        "step_index": step_index,
        "time_seconds": time_seconds,
        "application": app["application"],
        "transport": "UDP",
        "client_node": subscriber_id,
        "server_node": service_node,
        "client_ip": identities[subscriber_id].ip,
        "server_ip": identities[service_node].ip,
        "client_port": client_port,
        "server_port": app["server_port"],
        "dns_queries": query_count,
        "dns_responses": query_count,
        "packet_count": packet_count,
        "payload_bytes": payload_bytes,
        "wire_bytes": wire_bytes,
        "mtu_bytes": ETHERNET_MTU_BYTES,
        "route": query_route,
        "reverse_route": response_route,
        "hop_count": max(0, len(query_route) - 1),
        "one_way_latency_ms": round(query_latency_ms, 4),
        "rtt_ms": round(query_latency_ms + response_latency_ms, 4),
        "expected_loss_ratio": round(expected_loss, 8),
        "observed_dropped_packets": 0,
        "observed_retransmissions": 0,
        "sla_grade": attrs.get("sla_grade"),
        "traffic_kind": attrs.get("traffic_kind"),
        "sequence_base": _sequence_base(flow_id),
    }


def _sample_packets_for_flow(
    flow: dict[str, Any],
    identities: dict[str, NetworkIdentity],
    model: NetworkModel,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    if flow["transport"] == "TCP":
        return _tcp_packet_samples(flow, identities, model, limit)

    if flow["application"] == "DNS":
        return _dns_packet_samples(flow, identities, model, limit)

    return [
        _packet_event(
            model,
            identities,
            flow=flow,
            packet_role="udp_media",
            route=flow["route"],
            src_node=flow["server_node"],
            dst_node=flow["client_node"],
            protocol="UDP",
            src_port=flow["server_port"],
            dst_port=flow["client_port"],
            payload_bytes=min(UDP_PAYLOAD_BYTES, flow["payload_bytes"]),
            udp_length_bytes=UDP_HEADER_BYTES + min(UDP_PAYLOAD_BYTES, flow["payload_bytes"]),
            sequence_number=flow["sequence_base"],
        )
    ][:limit]


def _tcp_packet_samples(
    flow: dict[str, Any],
    identities: dict[str, NetworkIdentity],
    model: NetworkModel,
    limit: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    client = flow["client_node"]
    server = flow["server_node"]
    client_seq = flow["sequence_base"]
    server_seq = flow["sequence_base"] + 500_000

    if flow["handshake_packets"] and len(samples) < limit:
        samples.append(
            _packet_event(
                model,
                identities,
                flow=flow,
                packet_role="tcp_syn",
                route=flow["reverse_route"],
                src_node=client,
                dst_node=server,
                protocol="TCP",
                src_port=flow["client_port"],
                dst_port=flow["server_port"],
                payload_bytes=0,
                tcp_flags="S",
                sequence_number=client_seq,
                acknowledgment_number=0,
            )
        )

    if flow["handshake_packets"] and len(samples) < limit:
        samples.append(
            _packet_event(
                model,
                identities,
                flow=flow,
                packet_role="tcp_syn_ack",
                route=flow["route"],
                src_node=server,
                dst_node=client,
                protocol="TCP",
                src_port=flow["server_port"],
                dst_port=flow["client_port"],
                payload_bytes=0,
                tcp_flags="SA",
                sequence_number=server_seq,
                acknowledgment_number=client_seq + 1,
            )
        )

    if flow["handshake_packets"] and len(samples) < limit:
        samples.append(
            _packet_event(
                model,
                identities,
                flow=flow,
                packet_role="tcp_ack",
                route=flow["reverse_route"],
                src_node=client,
                dst_node=server,
                protocol="TCP",
                src_port=flow["client_port"],
                dst_port=flow["server_port"],
                payload_bytes=0,
                tcp_flags="A",
                sequence_number=client_seq + 1,
                acknowledgment_number=server_seq + 1,
            )
        )

    if len(samples) < limit:
        payload_bytes = min(TCP_MSS_BYTES, flow["payload_bytes"])
        samples.append(
            _packet_event(
                model,
                identities,
                flow=flow,
                packet_role="tcp_data",
                route=flow["route"],
                src_node=server,
                dst_node=client,
                protocol="TCP",
                src_port=flow["server_port"],
                dst_port=flow["client_port"],
                payload_bytes=payload_bytes,
                tcp_flags="PA",
                sequence_number=server_seq + 1,
                acknowledgment_number=client_seq + 1,
            )
        )

    if len(samples) < limit:
        samples.append(
            _packet_event(
                model,
                identities,
                flow=flow,
                packet_role="tcp_delayed_ack",
                route=flow["reverse_route"],
                src_node=client,
                dst_node=server,
                protocol="TCP",
                src_port=flow["client_port"],
                dst_port=flow["server_port"],
                payload_bytes=0,
                tcp_flags="A",
                sequence_number=client_seq + 1,
                acknowledgment_number=server_seq + 1 + min(TCP_MSS_BYTES, flow["payload_bytes"]),
            )
        )

    return samples[:limit]


def _dns_packet_samples(
    flow: dict[str, Any],
    identities: dict[str, NetworkIdentity],
    model: NetworkModel,
    limit: int,
) -> list[dict[str, Any]]:
    samples = [
        _packet_event(
            model,
            identities,
            flow=flow,
            packet_role="dns_query",
            route=flow["route"],
            src_node=flow["client_node"],
            dst_node=flow["server_node"],
            protocol="UDP",
            src_port=flow["client_port"],
            dst_port=flow["server_port"],
            payload_bytes=52,
            udp_length_bytes=UDP_HEADER_BYTES + 52,
            sequence_number=flow["sequence_base"],
        ),
        _packet_event(
            model,
            identities,
            flow=flow,
            packet_role="dns_response",
            route=flow["reverse_route"],
            src_node=flow["server_node"],
            dst_node=flow["client_node"],
            protocol="UDP",
            src_port=flow["server_port"],
            dst_port=flow["client_port"],
            payload_bytes=180,
            udp_length_bytes=UDP_HEADER_BYTES + 180,
            sequence_number=flow["sequence_base"] + 1,
        ),
    ]
    return samples[:limit]


def _packet_event(
    model: NetworkModel,
    identities: dict[str, NetworkIdentity],
    *,
    flow: dict[str, Any],
    packet_role: str,
    route: list[str],
    src_node: str,
    dst_node: str,
    protocol: str,
    src_port: int,
    dst_port: int,
    payload_bytes: int,
    sequence_number: int,
    tcp_flags: str | None = None,
    acknowledgment_number: int | None = None,
    udp_length_bytes: int | None = None,
) -> dict[str, Any]:
    transport_header_bytes = TCP_HEADER_BYTES if protocol == "TCP" else UDP_HEADER_BYTES
    total_length = IPV4_HEADER_BYTES + transport_header_bytes + payload_bytes
    protocol_number = 6 if protocol == "TCP" else 17
    ttl_at_destination = max(1, DEFAULT_TTL - max(0, len(route) - 1))
    packet_id = _packet_id(flow["flow_id"], packet_role, sequence_number)

    event: dict[str, Any] = {
        "packet_id": packet_id,
        "flow_id": flow["flow_id"],
        "packet_role": packet_role,
        "application": flow["application"],
        "route": route,
        "ethernet": {
            "ethertype": "0x0800",
            "mtu_bytes": ETHERNET_MTU_BYTES,
            "frame_header_bytes": ETHERNET_HEADER_BYTES,
            "frame_fcs_bytes": ETHERNET_FCS_BYTES,
            "hop_frames": _hop_frames(model, identities, route),
        },
        "ipv4": {
            "version": 4,
            "ihl_bytes": IPV4_HEADER_BYTES,
            "dscp": 0,
            "ecn": 0,
            "total_length_bytes": total_length,
            "identification": packet_id,
            "flags": ["DF"],
            "fragment_offset": 0,
            "ttl_start": DEFAULT_TTL,
            "ttl_at_destination": ttl_at_destination,
            "protocol": protocol,
            "protocol_number": protocol_number,
            "src_ip": identities[src_node].ip,
            "dst_ip": identities[dst_node].ip,
            "header_checksum": _checksum16(f"ip:{src_node}:{dst_node}:{protocol}:{total_length}:{packet_id}"),
        },
        "payload": {
            "bytes": payload_bytes,
            "hash": _checksum16(f"payload:{flow['flow_id']}:{packet_role}:{payload_bytes}:{sequence_number}"),
        },
    }

    if protocol == "TCP":
        event["tcp"] = {
            "src_port": src_port,
            "dst_port": dst_port,
            "sequence_number": sequence_number,
            "acknowledgment_number": acknowledgment_number or 0,
            "flags": tcp_flags or "A",
            "window_size": 64240,
            "header_length_bytes": TCP_HEADER_BYTES,
            "checksum": _checksum16(f"tcp:{src_port}:{dst_port}:{sequence_number}:{acknowledgment_number}:{payload_bytes}"),
        }
    else:
        event["udp"] = {
            "src_port": src_port,
            "dst_port": dst_port,
            "length_bytes": udp_length_bytes or (UDP_HEADER_BYTES + payload_bytes),
            "checksum": _checksum16(f"udp:{src_port}:{dst_port}:{payload_bytes}:{sequence_number}"),
        }

    return event


def _hop_frames(
    model: NetworkModel,
    identities: dict[str, NetworkIdentity],
    route: list[str],
) -> list[dict[str, Any]]:
    frames = []
    for index, (source, target) in enumerate(zip(route, route[1:])):
        edge = model.graph.edges[source, target]
        frames.append(
            {
                "hop_index": index,
                "source_node": source,
                "target_node": target,
                "src_mac": identities[source].mac,
                "dst_mac": identities[target].mac,
                "ttl_before_forward": DEFAULT_TTL - index,
                "edge_latency_ms": edge.get("latency_ms", 0.0),
                "edge_capacity_mbps": edge.get("capacity_mbps", 0.0),
                "medium": edge.get("medium"),
            }
        )
    return frames


def _traffic_summary(flows: list[dict[str, Any]]) -> dict[str, Any]:
    packet_count = sum(int(flow["packet_count"]) for flow in flows)
    payload_bytes = sum(int(flow["payload_bytes"]) for flow in flows)
    wire_bytes = sum(int(flow["wire_bytes"]) for flow in flows)
    dropped = sum(int(flow["observed_dropped_packets"]) for flow in flows)
    retransmissions = sum(int(flow["observed_retransmissions"]) for flow in flows)
    tcp_flows = sum(1 for flow in flows if flow["transport"] == "TCP")
    udp_flows = sum(1 for flow in flows if flow["transport"] == "UDP")
    latencies = [float(flow["one_way_latency_ms"]) for flow in flows]

    return {
        "flow_count": len(flows),
        "tcp_flow_count": tcp_flows,
        "udp_flow_count": udp_flows,
        "packet_count": packet_count,
        "payload_bytes": payload_bytes,
        "wire_bytes": wire_bytes,
        "observed_dropped_packets": dropped,
        "observed_retransmissions": retransmissions,
        "observed_loss_ratio": 0.0 if packet_count == 0 else dropped / packet_count,
        "mean_one_way_latency_ms": round(sum(latencies) / max(len(latencies), 1), 4),
        "max_one_way_latency_ms": round(max(latencies, default=0.0), 4),
    }


def _route(model: NetworkModel, source: str, target: str) -> list[str]:
    return nx.shortest_path(model.graph, source, target, weight="latency_ms")


def _path_latency_ms(model: NetworkModel, route: list[str], packet_bytes: int) -> float:
    latency = 0.0
    for source, target in zip(route, route[1:]):
        edge = model.graph.edges[source, target]
        capacity_mbps = max(float(edge.get("capacity_mbps", 1.0)), 1e-9)
        serialization_ms = packet_bytes * 8.0 / (capacity_mbps * 1_000_000.0) * 1000.0
        latency += float(edge.get("latency_ms", 0.0)) + serialization_ms

    for node_id in route[1:-1]:
        attrs = model.graph.nodes[node_id]
        if attrs.get("level") == "L2":
            latency += _tensor_metric(attrs.get("tensor"), "packet_processing_time_ms", default=0.0)

    return latency


def _path_expected_loss(model: NetworkModel, route: list[str]) -> float:
    survival = 1.0
    for source, target in zip(route, route[1:]):
        edge = model.graph.edges[source, target]
        loss_probability = _tensor_metric(edge.get("tensor"), "loss_probability", default=0.0)
        survival *= 1.0 - loss_probability
    return 1.0 - survival


def _payload_bps(attrs: dict[str, Any], time_seconds: int) -> float:
    monitoring = attrs.get("monitoring", [])
    if monitoring:
        point = monitoring[time_seconds % len(monitoring)]
        return float(point.get("bitrate_kbps", attrs.get("target_bitrate_kbps", 1.0)) * 1000.0)
    return float(attrs.get("target_bitrate_kbps", 1.0) * 1000.0)


def _tensor_metric(tensor: Any, metric_name: str, *, default: float) -> float:
    if not isinstance(tensor, StateTensor) or metric_name not in tensor.metric_index:
        return default
    return float(tensor.data[tensor.metric_index[metric_name][0]])


def _ip_for_node(node_id: str, attrs: dict[str, Any], fallback_index: int) -> str:
    level = attrs.get("level")
    role = attrs.get("role")

    if level == "L0":
        service_octet = {"SVC_VOICE": 10, "SVC_VIDEO": 20, "SVC_FTP": 30, "SVC_TELEM": 40}.get(node_id, fallback_index)
        return f"10.0.0.{service_octet}"

    if level == "L2":
        return f"10.10.0.{fallback_index % 250 + 1}"

    if role == "mobile-subscriber":
        group, subscriber = _subscriber_numbers(node_id)
        return f"10.1.{group}.{subscriber}"

    if role == "fixed-subscriber":
        group, subscriber = _subscriber_numbers(node_id)
        return f"10.2.{group}.{subscriber}"

    if level == "L7":
        return "10.70.0.1"

    if level == "L8":
        return f"10.80.0.{fallback_index % 250 + 1}"

    return f"10.254.{fallback_index // 250}.{fallback_index % 250 + 1}"


def _subscriber_numbers(node_id: str) -> tuple[int, int]:
    try:
        group_text, subscriber_text = node_id[1:].split("_", 1)
        return int(group_text), int(subscriber_text)
    except ValueError:
        return 0, 1


def _mac_for_node(node_id: str) -> str:
    digest = hashlib.blake2s(node_id.encode("utf-8"), digest_size=4).digest()
    return "02:09:%02x:%02x:%02x:%02x" % tuple(digest)


def _client_port(node_id: str, flow_index: int) -> int:
    digest = hashlib.blake2s(node_id.encode("utf-8"), digest_size=2).digest()
    offset = int.from_bytes(digest, "big") % 20_000
    return 40_000 + ((offset + flow_index) % 20_000)


def _sequence_base(flow_id: str) -> int:
    digest = hashlib.blake2s(flow_id.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % 2_000_000_000


def _packet_id(flow_id: str, packet_role: str, sequence_number: int) -> int:
    digest = hashlib.blake2s(f"{flow_id}:{packet_role}:{sequence_number}".encode("utf-8"), digest_size=2).digest()
    return int.from_bytes(digest, "big")


def _checksum16(text: str) -> str:
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=2).digest()
    return "0x%04x" % int.from_bytes(digest, "big")
