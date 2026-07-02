"""In-memory TCP/IP packet model for G-Net dynamics.

The simulator does not open sockets and never sends packets through the host OS.
It builds realistic packet/header events inside Python so the dynamic snapshots
can expose packet-level traffic while staying deterministic and safe.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal

import networkx as nx

from .addressing import NetworkIdentity, build_network_identities
from .attacks import apply_attack_effects
from .models import NetworkModel, StateTensor


ETHERNET_HEADER_BYTES = 14
ETHERNET_FCS_BYTES = 4
ETHERNET_PREAMBLE_SFD_BYTES = 8
ETHERNET_INTER_PACKET_GAP_BYTES = 12
IPV4_HEADER_BYTES = 20
TCP_HEADER_BYTES = 20
UDP_HEADER_BYTES = 8
RTP_HEADER_BYTES = 12
ETHERNET_OVERHEAD_BYTES = ETHERNET_HEADER_BYTES + ETHERNET_FCS_BYTES
ETHERNET_WIRE_OVERHEAD_BYTES = (
    ETHERNET_OVERHEAD_BYTES + ETHERNET_PREAMBLE_SFD_BYTES + ETHERNET_INTER_PACKET_GAP_BYTES
)
ETHERNET_MTU_BYTES = 1500
TCP_MSS_BYTES = ETHERNET_MTU_BYTES - IPV4_HEADER_BYTES - TCP_HEADER_BYTES
UDP_PAYLOAD_BYTES = 1180
DEFAULT_TTL = 64
PacketDetail = Literal["summary", "flows", "sample"]


TRAFFIC_APPS = {
    "voice": {
        "application": "RTP_OPUS",
        "transport": "UDP",
        "service_node": "SVC_VOICE",
        "server_port": 5002,
        "codec_profile_id": "voice_opus_rtp",
        "media_codec": "Opus",
        "rtp_clock_rate_hz": 48_000,
    },
    "broadcast_mp3": {
        "application": "RTP_MP3",
        "transport": "UDP",
        "service_node": "SVC_VLC",
        "server_port": 5004,
        "payload_unit_bytes": UDP_PAYLOAD_BYTES,
        "codec_profile_id": "vlc_opus_h264_rtp",
        "media_codec": "Opus + H.264/AVC",
        "rtp_clock_rate_hz": 90_000,
        "rtp_payload_type": 96,
    },
    "vlc_av": {
        "application": "RTP_VLC_AV",
        "transport": "UDP",
        "service_node": "SVC_VLC",
        "server_port": 5004,
        "payload_unit_bytes": UDP_PAYLOAD_BYTES,
        "codec_profile_id": "vlc_opus_h264_rtp",
        "media_codec": "Opus + H.264/AVC",
        "rtp_clock_rate_hz": 90_000,
        "rtp_payload_type": 96,
    },
    "ftp": {
        "application": "FTP_DATA",
        "transport": "TCP",
        "service_node": "SVC_FTP",
        "server_port": 21,
        "payload_unit_bytes": TCP_MSS_BYTES,
        "codec_profile_id": "ftp_tcp_binary",
        "media_codec": "binary/octet-stream",
    },
    "dns": {
        "application": "DNS",
        "transport": "UDP",
        "service_node": "SVC_DNS",
        "server_port": 53,
        "query_payload_bytes": 52,
        "response_payload_bytes": 180,
        "codec_profile_id": "dns_udp_messages",
        "media_codec": "DNS wire format",
    },
    "video_conference": {
        "application": "RTP_TELEMOST",
        "transport": "UDP",
        "service_node": "SVC_TELEMOST",
        "server_port": 5006,
        "payload_unit_bytes": UDP_PAYLOAD_BYTES,
        "codec_profile_id": "webrtc_opus_h264",
        "media_codec": "Opus + H.264/AVC",
        "rtp_clock_rate_hz": 90_000,
        "rtp_payload_type": 98,
    },
    "live_streaming": {
        "application": "LIVE_HLS",
        "transport": "TCP",
        "service_node": "SVC_LIVE",
        "server_port": 443,
        "payload_unit_bytes": TCP_MSS_BYTES,
        "codec_profile_id": "ll_hls_aac_h264",
        "media_codec": "AAC-LC + H.264/AVC",
    },
}


@dataclass(frozen=True)
class FlowContext:
    """All stable inputs needed to build one subscriber traffic flow."""

    model: NetworkModel
    identities: dict[str, NetworkIdentity]
    subscriber_id: str
    attrs: dict[str, Any]
    app: dict[str, Any]
    service_node: str
    server_node: str
    client_port: int
    step_index: int
    time_seconds: int
    step_seconds: int
    payload_bps: float
    flow_id: str


def simulate_packet_snapshot(
    model: NetworkModel,
    *,
    step_index: int,
    time_seconds: int,
    step_seconds: int,
    detail: PacketDetail = "sample",
    packet_sample_limit: int = 48,
    attack_events: list[dict[str, Any]] | None = None,
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

    flows, attack_state = apply_attack_effects(model, flows, attack_events or [])
    result: dict[str, Any] = {"summary": _traffic_summary(flows), "attack_state": attack_state}
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
    server_node = model.graph.nodes[service_node].get("hosted_on", service_node)
    client_port = _client_port(subscriber_id, flow_index)
    payload_bps = _payload_bps(attrs, time_seconds)
    flow_id = f"{subscriber_id}-{app['application']}-{step_index}"
    context = FlowContext(
        model=model,
        identities=identities,
        subscriber_id=subscriber_id,
        attrs=attrs,
        app=app,
        service_node=service_node,
        server_node=server_node,
        client_port=client_port,
        step_index=step_index,
        time_seconds=time_seconds,
        step_seconds=step_seconds,
        payload_bps=payload_bps,
        flow_id=flow_id,
    )

    if app["transport"] == "TCP":
        return _build_tcp_flow(context)

    if traffic_kind == "voice":
        return _build_voice_flow(context)

    if traffic_kind == "dns":
        return _build_dns_flow(context)

    return _build_udp_media_flow(context)


def _build_voice_flow(context: FlowContext) -> dict[str, Any]:
    """Build a mono Opus/RTP stream with one packet every 20 milliseconds."""
    route = _route(context.model, context.server_node, context.subscriber_id)
    policy = context.attrs.get("d0sl_policy", {})
    packetization_ms = int(policy.get("packetization_ms") or 20)
    rtp_clock_rate_hz = int(policy.get("rtp_clock_rate_hz") or 48_000)
    rtp_payload_type = int(policy.get("rtp_payload_type") or 111)
    channels = int(policy.get("channels") or 1)
    packets_per_second = 1000 // packetization_ms
    datagrams = packets_per_second * context.step_seconds
    codec_payload_bytes = max(1, int(round(context.payload_bps * context.step_seconds / 8.0)))
    codec_bytes_per_packet = max(1, int(round(codec_payload_bytes / datagrams)))
    wire_bytes = codec_payload_bytes + datagrams * (
        RTP_HEADER_BYTES + UDP_HEADER_BYTES + IPV4_HEADER_BYTES + ETHERNET_WIRE_OVERHEAD_BYTES
    )
    ip_packet_bytes = IPV4_HEADER_BYTES + UDP_HEADER_BYTES + RTP_HEADER_BYTES + codec_bytes_per_packet
    network_latency_ms = _path_latency_ms(context.model, route, ip_packet_bytes)
    # 20 ms framing plus the Opus look-ahead approximates the endpoint contribution.
    one_way_latency_ms = network_latency_ms + packetization_ms + 6.5
    expected_loss = _path_expected_loss(context.model, route)

    return _flow_record(
        context,
        transport="UDP",
        route=route,
        reverse_route=list(reversed(route)),
        packet_count=datagrams,
        payload_bytes=codec_payload_bytes,
        wire_bytes=wire_bytes,
        one_way_latency_ms=one_way_latency_ms,
        rtt_ms=None,
        expected_loss=expected_loss,
        extra={
            "rtp_packets": datagrams,
            "packetization_ms": packetization_ms,
            "packets_per_second": packets_per_second,
            "codec_payload_bytes_per_packet": codec_bytes_per_packet,
            "codec_bitrate_kbps": round(context.payload_bps / 1000.0, 3),
            "line_bitrate_kbps": round(wire_bytes * 8.0 / context.step_seconds / 1000.0, 3),
            "rtp_header_bytes": RTP_HEADER_BYTES,
            "rtp_clock_rate_hz": rtp_clock_rate_hz,
            "rtp_payload_type": rtp_payload_type,
            "channels": channels,
            "dtx_enabled": policy.get("dtx") == "enabled",
            "inband_fec_enabled": policy.get("inband_fec") == "enabled",
            "media_codec": context.app.get("media_codec", "Opus"),
        },
    )


def _build_tcp_flow(context: FlowContext) -> dict[str, Any]:
    data_route = _route(context.model, context.server_node, context.subscriber_id)
    ack_route = list(reversed(data_route))
    payload_bytes = _payload_bytes(context.payload_bps, context.step_seconds, TCP_MSS_BYTES)
    data_segments = math.ceil(payload_bytes / TCP_MSS_BYTES)
    ack_segments = math.ceil(data_segments / 2)
    handshake_packets = 3 if context.step_index == 0 else 0
    tcp_packets = handshake_packets + data_segments + ack_segments
    wire_bytes = _wire_bytes(payload_bytes, tcp_packets, TCP_HEADER_BYTES)
    one_way_latency_ms = _path_latency_ms(context.model, data_route, TCP_MSS_BYTES + IPV4_HEADER_BYTES + TCP_HEADER_BYTES)
    rtt_ms = one_way_latency_ms + _path_latency_ms(context.model, ack_route, IPV4_HEADER_BYTES + TCP_HEADER_BYTES)
    expected_loss = _path_expected_loss(context.model, data_route)

    return _flow_record(
        context,
        transport="TCP",
        route=data_route,
        reverse_route=ack_route,
        packet_count=tcp_packets,
        payload_bytes=payload_bytes,
        wire_bytes=wire_bytes,
        one_way_latency_ms=one_way_latency_ms,
        rtt_ms=rtt_ms,
        expected_loss=expected_loss,
        extra={
            "tcp_state": "ESTABLISHED",
            "handshake_packets": handshake_packets,
            "data_segments": data_segments,
            "ack_segments": ack_segments,
            "mss_bytes": TCP_MSS_BYTES,
            "tcp_retransmission_budget_percent": context.attrs.get("d0sl_policy", {}).get("tcp_retransmission_budget_percent"),
            "tcp_quality_grade": context.attrs.get("sla_grade") if context.attrs.get("traffic_kind") == "ftp" else None,
            "media_codec": context.app.get("media_codec"),
            "glass_to_glass_latency_ms": (
                round(float(context.attrs.get("latency_budget_ms", 0.0)) * 0.65, 3)
                if context.attrs.get("traffic_kind") == "live_streaming" else None
            ),
        },
    )


def _build_udp_media_flow(context: FlowContext) -> dict[str, Any]:
    route = _route(context.model, context.server_node, context.subscriber_id)
    payload_unit = context.app["payload_unit_bytes"]
    payload_bytes = _payload_bytes(context.payload_bps, context.step_seconds, payload_unit)
    datagrams = math.ceil(payload_bytes / payload_unit)
    has_rtp = context.app["application"].startswith("RTP_")
    media_header_bytes = RTP_HEADER_BYTES if has_rtp else 0
    wire_bytes = payload_bytes + datagrams * (
        ETHERNET_WIRE_OVERHEAD_BYTES + IPV4_HEADER_BYTES + UDP_HEADER_BYTES + media_header_bytes
    )
    one_way_latency_ms = _path_latency_ms(
        context.model,
        route,
        payload_unit + media_header_bytes + IPV4_HEADER_BYTES + UDP_HEADER_BYTES,
    )
    expected_loss = _path_expected_loss(context.model, route)
    extra = {
        "udp_datagrams": datagrams,
        "audio_bitrate_kbps": context.attrs.get("d0sl_policy", {}).get("audio_bitrate_kbps"),
        "video_bitrate_kbps": context.attrs.get("d0sl_policy", {}).get("video_bitrate_kbps"),
        "audio_latency_budget_ms": context.attrs.get("d0sl_policy", {}).get("audio_latency_budget_ms"),
        "video_latency_budget_ms": context.attrs.get("d0sl_policy", {}).get("video_latency_budget_ms"),
        "failure_latency_ms": context.attrs.get("d0sl_policy", {}).get("failure_latency_ms"),
        "media_codec": context.app.get("media_codec"),
    }
    if has_rtp:
        extra.update(
            {
                "rtp_packets": datagrams,
                "rtp_header_bytes": RTP_HEADER_BYTES,
                "rtp_clock_rate_hz": int(context.app.get("rtp_clock_rate_hz", 90_000)),
                "rtp_payload_type": int(context.app.get("rtp_payload_type", 96)),
                "rtp_media_codec": context.app.get("media_codec"),
            }
        )

    return _flow_record(
        context,
        transport="UDP",
        route=route,
        reverse_route=list(reversed(route)),
        packet_count=datagrams,
        payload_bytes=payload_bytes,
        wire_bytes=wire_bytes,
        one_way_latency_ms=one_way_latency_ms,
        rtt_ms=None,
        expected_loss=expected_loss,
        extra=extra,
    )


def _build_dns_flow(context: FlowContext) -> dict[str, Any]:
    query_route = _route(context.model, context.subscriber_id, context.server_node)
    response_route = list(reversed(query_route))
    request_rate = _tensor_metric(context.attrs.get("tensor"), "request_rate_pps", default=1.0)
    query_count = max(1, int(round(request_rate * context.step_seconds)))
    query_bytes = int(context.app["query_payload_bytes"])
    response_bytes = int(context.app["response_payload_bytes"])
    payload_bytes = query_count * (query_bytes + response_bytes)
    packet_count = query_count * 2
    wire_bytes = _wire_bytes(payload_bytes, packet_count, UDP_HEADER_BYTES)
    query_latency_ms = _path_latency_ms(context.model, query_route, query_bytes + IPV4_HEADER_BYTES + UDP_HEADER_BYTES)
    response_latency_ms = _path_latency_ms(context.model, response_route, response_bytes + IPV4_HEADER_BYTES + UDP_HEADER_BYTES)
    expected_loss = _round_trip_expected_loss(context.model, query_route, response_route)

    return _flow_record(
        context,
        transport="UDP",
        route=query_route,
        reverse_route=response_route,
        packet_count=packet_count,
        payload_bytes=payload_bytes,
        wire_bytes=wire_bytes,
        one_way_latency_ms=query_latency_ms,
        rtt_ms=query_latency_ms + response_latency_ms,
        expected_loss=expected_loss,
        extra={
            "dns_queries": query_count,
            "dns_responses": query_count,
            "media_codec": context.app.get("media_codec"),
        },
    )


def _flow_record(
    context: FlowContext,
    *,
    transport: str,
    route: list[str],
    reverse_route: list[str],
    packet_count: int,
    payload_bytes: int,
    wire_bytes: int,
    one_way_latency_ms: float,
    rtt_ms: float | None,
    expected_loss: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    service_attrs = context.model.graph.nodes[context.service_node]
    record = {
        "flow_id": context.flow_id,
        "step_index": context.step_index,
        "time_seconds": context.time_seconds,
        "interval_seconds": context.step_seconds,
        "application": context.app["application"],
        "transport": transport,
        "client_node": context.subscriber_id,
        "service_node": context.service_node,
        "server_node": context.server_node,
        "client_ip": context.identities[context.subscriber_id].ip,
        "server_ip": context.identities[context.server_node].ip,
        "client_port": context.client_port,
        "server_port": context.app["server_port"],
        "packet_count": packet_count,
        "payload_bytes": payload_bytes,
        "wire_bytes": wire_bytes,
        "mtu_bytes": ETHERNET_MTU_BYTES,
        "route": route,
        "reverse_route": reverse_route,
        "hop_count": max(0, len(route) - 1),
        "one_way_latency_ms": round(one_way_latency_ms, 4),
        "rtt_ms": None if rtt_ms is None else round(rtt_ms, 4),
        "expected_loss_ratio": round(expected_loss, 8),
        "observed_dropped_packets": 0,
        "observed_retransmissions": 0,
        "sla_grade": context.attrs.get("sla_grade"),
        "traffic_kind": context.attrs.get("traffic_kind"),
        "codec": context.attrs.get("codec"),
        "codec_profile_id": context.app.get("codec_profile_id") or service_attrs.get("codec_profile_id"),
        "codec_profile_name": service_attrs.get("codec_profile_name"),
        "codec_summary": service_attrs.get("codec_summary"),
        "sequence_base": _sequence_base(context.flow_id),
    }
    record.update(extra)
    return record


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

    if flow["application"] == "RTP_OPUS":
        return _voice_packet_samples(flow, identities, model, limit)

    media_payload = min(UDP_PAYLOAD_BYTES, flow["payload_bytes"])
    rtp_payload = RTP_HEADER_BYTES + media_payload if flow.get("rtp_header_bytes") else media_payload
    event = _packet_event(
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
        payload_bytes=rtp_payload,
        udp_length_bytes=UDP_HEADER_BYTES + rtp_payload,
        sequence_number=flow["sequence_base"],
    )
    if flow.get("rtp_header_bytes"):
        event["rtp"] = {
            "version": 2,
            "payload_type": flow["rtp_payload_type"],
            "marker": False,
            "sequence_number": flow["sequence_base"] & 0xFFFF,
            "timestamp": (flow["time_seconds"] * flow["rtp_clock_rate_hz"]) & 0xFFFFFFFF,
            "ssrc": flow["sequence_base"],
            "clock_rate_hz": flow["rtp_clock_rate_hz"],
            "codec": flow.get("rtp_media_codec") or flow.get("media_codec") or flow.get("codec"),
            "header_bytes": RTP_HEADER_BYTES,
            "media_payload_bytes": media_payload,
        }
    return [event][:limit]


def _voice_packet_samples(
    flow: dict[str, Any],
    identities: dict[str, NetworkIdentity],
    model: NetworkModel,
    limit: int,
) -> list[dict[str, Any]]:
    codec_bytes = int(flow["codec_payload_bytes_per_packet"])
    udp_payload_bytes = RTP_HEADER_BYTES + codec_bytes
    event = _packet_event(
        model,
        identities,
        flow=flow,
        packet_role="rtp_voice",
        route=flow["route"],
        src_node=flow["server_node"],
        dst_node=flow["client_node"],
        protocol="UDP",
        src_port=flow["server_port"],
        dst_port=flow["client_port"],
        payload_bytes=udp_payload_bytes,
        udp_length_bytes=UDP_HEADER_BYTES + udp_payload_bytes,
        sequence_number=flow["sequence_base"],
    )
    event["rtp"] = {
        "version": 2,
        "payload_type": flow["rtp_payload_type"],
        "marker": False,
        "sequence_number": flow["sequence_base"] & 0xFFFF,
        "timestamp": (flow["time_seconds"] * flow["rtp_clock_rate_hz"]) & 0xFFFFFFFF,
        "ssrc": flow["sequence_base"],
        "clock_rate_hz": flow["rtp_clock_rate_hz"],
        "packetization_ms": flow["packetization_ms"],
        "codec": "Opus",
        "channels": flow["channels"],
        "header_bytes": RTP_HEADER_BYTES,
        "codec_payload_bytes": codec_bytes,
    }
    return [event][:limit]


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
        "codec_profile_id": flow.get("codec_profile_id"),
        "codec_profile_name": flow.get("codec_profile_name"),
        "route": route,
        "ethernet": {
            "ethertype": "0x0800",
            "mtu_bytes": ETHERNET_MTU_BYTES,
            "frame_header_bytes": ETHERNET_HEADER_BYTES,
            "frame_fcs_bytes": ETHERNET_FCS_BYTES,
            "preamble_sfd_bytes": ETHERNET_PREAMBLE_SFD_BYTES,
            "inter_packet_gap_bytes": ETHERNET_INTER_PACKET_GAP_BYTES,
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
    interval_seconds = int(flows[0].get("interval_seconds", 0)) if flows else 0
    applications: dict[str, dict[str, Any]] = {}
    for application in sorted({str(flow["application"]) for flow in flows}):
        app_flows = [flow for flow in flows if flow["application"] == application]
        app_payload = sum(int(flow["payload_bytes"]) for flow in app_flows)
        app_wire = sum(int(flow["wire_bytes"]) for flow in app_flows)
        app_latencies = [float(flow["one_way_latency_ms"]) for flow in app_flows]
        applications[application] = {
            "flow_count": len(app_flows),
            "codec_profile_id": next((flow.get("codec_profile_id") for flow in app_flows if flow.get("codec_profile_id")), None),
            "codec_profile_name": next((flow.get("codec_profile_name") for flow in app_flows if flow.get("codec_profile_name")), None),
            "packet_count": sum(int(flow["packet_count"]) for flow in app_flows),
            "payload_bytes": app_payload,
            "wire_bytes": app_wire,
            "offered_rate_mbps": round(app_wire * 8.0 / max(interval_seconds, 1) / 1_000_000.0, 4),
            "protocol_efficiency_ratio": round(app_payload / max(app_wire, 1), 6),
            "mean_one_way_latency_ms": round(sum(app_latencies) / max(len(app_latencies), 1), 4),
            "max_one_way_latency_ms": round(max(app_latencies, default=0.0), 4),
        }

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
        "interval_seconds": interval_seconds,
        "offered_rate_mbps": round(wire_bytes * 8.0 / max(interval_seconds, 1) / 1_000_000.0, 4),
        "protocol_efficiency_ratio": round(payload_bytes / max(wire_bytes, 1), 6),
        "applications": applications,
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


def _round_trip_expected_loss(model: NetworkModel, outbound_route: list[str], return_route: list[str]) -> float:
    outbound_success = 1.0 - _path_expected_loss(model, outbound_route)
    return_success = 1.0 - _path_expected_loss(model, return_route)
    return 1.0 - outbound_success * return_success


def _payload_bytes(payload_bps: float, step_seconds: int, minimum_payload_bytes: int) -> int:
    return max(minimum_payload_bytes, int(payload_bps * step_seconds / 8.0))


def _wire_bytes(payload_bytes: int, packet_count: int, transport_header_bytes: int) -> int:
    return payload_bytes + packet_count * (ETHERNET_WIRE_OVERHEAD_BYTES + IPV4_HEADER_BYTES + transport_header_bytes)


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
