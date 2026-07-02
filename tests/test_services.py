from collections import Counter
from pathlib import Path

from src.gnet9.addressing import build_ip_address_plan
from src.gnet9.l1_d0sl import TrafficKind, load_l1_d0sl_catalog
from src.gnet9.packet_simulator import simulate_packet_snapshot
from src.gnet9.service_catalog import CODEC_CATALOG, QualityGrade, classify_latency, classify_tcp_retransmission
from src.gnet9.topology_builder import GNetBaselineBuilder


MODEL = GNetBaselineBuilder().build()


def test_services_are_hosted_on_redundantly_connected_servers() -> None:
    services = [(node, attrs) for node, attrs in MODEL.graph.nodes(data=True) if attrs.get("role") == "service"]
    servers = [(node, attrs) for node, attrs in MODEL.graph.nodes(data=True) if attrs.get("role") == "service-server"]
    assert len(services) == 6
    assert len(servers) == 4
    for service_id, attrs in services:
        assert list(MODEL.graph.neighbors(service_id)) == [attrs["hosted_on"]]
    for _, attrs in servers:
        assert attrs["server_profile"]["model"] == "Dell PowerEdge R660"
        assert attrs["server_profile"]["total_cores"] == 64
        assert len(attrs["hosted_services"]) >= 1
    for _, attrs in services:
        assert attrs["codec_profile_id"] in CODEC_CATALOG
        assert attrs["codec_profile"]["components"]
        assert attrs["ip_address"].startswith("10.0.0.")


def test_codec_profiles_are_attached_to_subscribers_and_flows() -> None:
    services = [attrs for _, attrs in MODEL.graph.nodes(data=True) if attrs.get("role") == "service"]
    subscribers = [attrs for _, attrs in MODEL.graph.nodes(data=True) if attrs.get("level") == "L1"]
    assert all(attrs["codec_profile_name"] for attrs in services)
    assert all(attrs["codec_profile_name"] for attrs in subscribers)

    traffic = simulate_packet_snapshot(MODEL, step_index=0, time_seconds=0, step_seconds=5, detail="sample", packet_sample_limit=1000)
    assert all(flow["codec_profile_id"] for flow in traffic["flows"])
    vlc = next(flow for flow in traffic["flows"] if flow["application"] == "RTP_VLC_AV")
    conference = next(flow for flow in traffic["flows"] if flow["application"] == "RTP_TELEMOST")
    assert vlc["rtp_media_codec"] == "Opus + H.264/AVC"
    assert conference["codec_profile_id"] == "webrtc_opus_h264"
    media_packet = next(packet for packet in traffic["packet_sample"] if packet["application"] == "RTP_VLC_AV")
    assert media_packet["rtp"]["codec"] == "Opus + H.264/AVC"


def test_ip_address_plan_maps_devices_to_private_addresses() -> None:
    plan = build_ip_address_plan(MODEL)
    by_id = {device["node_id"]: device for device in plan["devices"]}
    assert plan["device_count"] == MODEL.graph.number_of_nodes()
    assert by_id["SRV_MEDIA"]["ip"] == "10.0.0.10"
    assert by_id["SVC_VOICE"]["ip"] == "10.0.0.110"
    assert by_id["C1"]["ip"] == "10.10.0.1"
    assert by_id["A1"]["ip"] == "10.10.0.101"
    assert by_id["M1_01"]["ip"] == "10.1.1.1"
    assert by_id["F3_40"]["ip"] == "10.2.3.40"


def test_new_d0sl_profiles_are_complete() -> None:
    catalog = load_l1_d0sl_catalog(Path(__file__).resolve().parents[1] / "policies" / "l1_policies.d0sl")
    expected_vlc = {"gold": (320.0, 4000.0), "silver": (128.0, 2000.0), "bronze": (64.0, 1000.0)}
    for grade, (audio, video) in expected_vlc.items():
        policy = catalog.get(grade, TrafficKind.VLC_AV.value)
        assert policy.audio_bitrate_kbps == audio
        assert policy.video_bitrate_kbps == video
    assert catalog.get("gold", TrafficKind.VIDEO_CONFERENCE.value).video_latency_budget_ms == 70.0
    assert catalog.get("bronze", TrafficKind.VIDEO_CONFERENCE.value).audio_latency_budget_ms == 400.0
    assert catalog.get("gold", TrafficKind.LIVE_STREAMING.value).latency_budget_ms == 2000.0
    assert catalog.get("bronze", TrafficKind.LIVE_STREAMING.value).failure_latency_ms == 10000.0


def test_quality_classifiers_have_monotonic_non_overlapping_ranges() -> None:
    assert classify_tcp_retransmission(0.1) is QualityGrade.GOLD
    assert classify_tcp_retransmission(0.5) is QualityGrade.SILVER
    assert classify_tcp_retransmission(1.0) is QualityGrade.BRONZE
    assert classify_tcp_retransmission(40.0) is QualityGrade.FAILED
    assert classify_latency(70, gold_ms=70, silver_ms=150, bronze_ms=250, failed_ms=600) is QualityGrade.GOLD
    assert classify_latency(200, gold_ms=70, silver_ms=150, bronze_ms=250, failed_ms=600) is QualityGrade.BRONZE
    try:
        classify_latency(10, gold_ms=150, silver_ms=70, bronze_ms=250, failed_ms=600)
    except ValueError:
        pass
    else:
        raise AssertionError("Некорректный порядок границ должен отклоняться")


def test_each_application_has_40_flows_and_targets_a_physical_server() -> None:
    traffic = simulate_packet_snapshot(MODEL, step_index=0, time_seconds=0, step_seconds=5, detail="flows")
    counts = Counter(flow["application"] for flow in traffic["flows"])
    assert set(counts.values()) == {40}
    assert set(counts) == {"DNS", "FTP_DATA", "LIVE_HLS", "RTP_OPUS", "RTP_TELEMOST", "RTP_VLC_AV"}
    assert all(MODEL.graph.nodes[flow["server_node"]]["role"] == "service-server" for flow in traffic["flows"])
    ftp = next(flow for flow in traffic["flows"] if flow["application"] == "FTP_DATA" and flow["sla_grade"] == "gold")
    live = next(flow for flow in traffic["flows"] if flow["application"] == "LIVE_HLS" and flow["sla_grade"] == "gold")
    assert ftp["tcp_retransmission_budget_percent"] == 0.1
    assert live["glass_to_glass_latency_ms"] == 1300.0
