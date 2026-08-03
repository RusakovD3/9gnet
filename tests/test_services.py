from collections import Counter
from copy import deepcopy
from pathlib import Path

from src.gnet9.addressing import build_ip_address_plan
from src.gnet9.attacks import apply_attack_effects
from src.gnet9.constants import SUBSCRIBER_COUNT
from src.gnet9.l1_d0sl import TrafficKind, load_l1_d0sl_catalog
from src.gnet9.packet_simulator import TRAFFIC_APPS, simulate_packet_snapshot
from src.gnet9.routing import path_uses_only_data_plane_transit, shortest_data_path
from src.gnet9.service_catalog import CODEC_CATALOG, QualityGrade, classify_latency, classify_tcp_retransmission
from src.gnet9.topology_builder import GNetBaselineBuilder


MODEL = GNetBaselineBuilder().build()


def test_services_are_hosted_on_redundantly_connected_servers() -> None:
    services = [(node, attrs) for node, attrs in MODEL.graph.nodes(data=True) if attrs.get("role") == "service"]
    servers = [(node, attrs) for node, attrs in MODEL.graph.nodes(data=True) if attrs.get("role") == "service-server"]
    assert len(services) == 6
    assert len(servers) == 4
    server_ids = {node for node, _ in servers}
    for service_id, attrs in services:
        assert list(MODEL.graph.neighbors(service_id)) == [attrs["hosted_on"]]
        assert set(attrs["standby_hosts"]) == server_ids - {attrs["hosted_on"]}
        primary_domain = MODEL.graph.nodes[attrs["hosted_on"]]["power_architecture"][
            "local_fault_domain_id"
        ]
        assert all(
            MODEL.graph.nodes[standby]["power_architecture"]["local_fault_domain_id"]
            != primary_domain
            for standby in attrs["standby_hosts"]
        )
    for _, attrs in servers:
        assert attrs["server_profile"]["model"] == "Dell PowerEdge R660"
        assert attrs["server_profile"]["total_cores"] == 64
        assert attrs["server_profile"]["threads_total"] == 128
        assert attrs["server_profile"]["core_count_semantics"].startswith("total_physical_cores")
        assert len(attrs["hosted_services"]) >= 1
        assert attrs["server_profile"]["network_ports_gbps"] == (10, 10)
        power = attrs["power_architecture"]
        assert power["single_psu_failure_service_interruption"] is False
        assert power["psu_count"] == 2
        assert power["psu_rating_w_each"] == 800
        assert power["psu_efficiency_class"] == "Platinum"
        assert power["modeled_t0_power_w"] == 450.0
        assert power["ups_backup_autonomy_hours"] == 0.25
        assert "not_measured_server_draw" in power["psu_rating_semantics"]
        assert power["local_fault_domain_id"]
        assert len(set(power["ups_domains"])) == 2
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
    assert conference["media_security_profile"].startswith("DTLS-SRTP")
    assert conference["security_encapsulation_byte_overhead_modeled"] is False
    media_packet = next(packet for packet in traffic["packet_sample"] if packet["application"] == "RTP_VLC_AV")
    assert media_packet["rtp"]["codec"] == "Opus + H.264/AVC"


def test_protocol_profiles_export_sources_and_media_components_balance() -> None:
    services = [attrs for _, attrs in MODEL.graph.nodes(data=True) if attrs.get("role") == "service"]
    subscribers = [attrs for _, attrs in MODEL.graph.nodes(data=True) if attrs.get("level") == "L1"]
    assert all(attrs["codec_profile"]["reference_urls"] for attrs in services)
    assert all(attrs["d0sl_policy"]["reference_urls"] for attrs in subscribers)
    for attrs in subscribers:
        policy = attrs["d0sl_policy"]
        audio = policy.get("audio_bitrate_kbps")
        video = policy.get("video_bitrate_kbps")
        if audio is not None and video is not None:
            assert audio + video == policy["target_bitrate_kbps"]

    # RFC 6184 describes H.264 over RTP and must not be cited as the transport
    # specification for HTTP/fMP4 Low-Latency HLS.
    live_refs = CODEC_CATALOG["ll_hls_aac_h264"].reference_urls
    assert any("developer.apple.com" in url for url in live_refs)
    assert all("rfc6184" not in url for url in live_refs)


def test_l1_queue_uses_consistent_finite_capacity_kendall_model() -> None:
    subscribers = [attrs for _, attrs in MODEL.graph.nodes(data=True) if attrs.get("level") == "L1"]
    for attrs in subscribers:
        queue = attrs["kendall_queue"]
        assert queue["kendall"] == "M/M/1/128/∞/FIFO"
        assert 0.0 <= queue["blocking_probability"] < 1.0
        assert queue["arrival_rate_origin"]
        assert queue["effective_arrival_rate_pps"] <= queue["arrival_rate_pps"]
        assert queue["mean_queue_depth_packets"] < queue["capacity_packets"]
        assert queue["mean_system_time_ms"] <= attrs["latency_budget_ms"] * 0.25 + 1e-9


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
    assert {policy.traffic.value for policy in catalog.policies} == {
        "voice", "vlc_av", "ftp", "dns", "video_conference", "live_streaming",
    }
    expected_vlc = {"gold": (320.0, 4000.0), "silver": (128.0, 2000.0), "bronze": (64.0, 1000.0)}
    for grade, (audio, video) in expected_vlc.items():
        policy = catalog.get(grade, TrafficKind.VLC_AV.value)
        assert policy.audio_bitrate_kbps == audio
        assert policy.video_bitrate_kbps == video
    assert catalog.get("gold", TrafficKind.VIDEO_CONFERENCE.value).video_latency_budget_ms == 70.0
    assert catalog.get("bronze", TrafficKind.VIDEO_CONFERENCE.value).audio_latency_budget_ms == 400.0
    assert catalog.get("gold", TrafficKind.LIVE_STREAMING.value).latency_budget_ms == 2000.0
    assert catalog.get("bronze", TrafficKind.LIVE_STREAMING.value).failure_latency_ms == 10000.0
    assert catalog.get("gold", TrafficKind.LIVE_STREAMING.value).rebuffer_ratio_budget_percent == 0.1
    assert catalog.get("gold", TrafficKind.DNS.value).request_rate_qps == 2.0
    assert catalog.get("gold", TrafficKind.DNS.value).latency_budget_ms == 50.0
    assert catalog.get("silver", TrafficKind.DNS.value).latency_budget_ms == 80.0
    assert catalog.get("bronze", TrafficKind.DNS.value).latency_budget_ms == 120.0
    assert next(
        slo
        for slo in catalog.get("gold", TrafficKind.DNS.value).slo
        if slo.name == "Latency"
    ).value == 50.0
    assert catalog.get("bronze", TrafficKind.FTP.value).completion_time_budget_seconds == 1100.0
    assert catalog.get("gold", TrafficKind.VOICE.value).packet_loss_budget_percent == 0.1


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


def test_each_application_has_balanced_flows_and_targets_a_physical_server() -> None:
    traffic = simulate_packet_snapshot(MODEL, step_index=0, time_seconds=0, step_seconds=5, detail="flows")
    counts = Counter(flow["application"] for flow in traffic["flows"])
    assert set(counts.values()) == {SUBSCRIBER_COUNT // len(TRAFFIC_APPS)}
    assert set(counts) == {"DNS", "FTP_DATA", "LIVE_HLS", "RTP_OPUS", "RTP_TELEMOST", "RTP_VLC_AV"}
    assert all(MODEL.graph.nodes[flow["server_node"]]["role"] == "service-server" for flow in traffic["flows"])
    ftp = next(flow for flow in traffic["flows"] if flow["application"] == "FTP_DATA" and flow["sla_grade"] == "gold")
    live = next(flow for flow in traffic["flows"] if flow["application"] == "LIVE_HLS" and flow["sla_grade"] == "gold")
    dns = next(flow for flow in traffic["flows"] if flow["application"] == "DNS" and flow["sla_grade"] == "gold")
    assert ftp["tcp_retransmission_budget_percent"] == 0.1
    assert live["glass_to_glass_latency_ms"] == 1300.0
    assert live["startup_time_budget_ms"] == 1500.0
    assert dns["dns_request_rate_qps"] == 2.0


def test_service_aware_slo_evaluation_exports_honest_metric_coverage() -> None:
    traffic = simulate_packet_snapshot(
        MODEL,
        step_index=0,
        time_seconds=0,
        step_seconds=5,
        detail="flows",
    )
    flows = traffic["flows"]
    non_compliant = [
        (
            flow["flow_id"],
            flow["slo_evaluation"]["failed_metric_ids"],
            {
                metric_id: (
                    metric["observed_value"],
                    metric["budget_value"],
                )
                for metric_id, metric in flow["slo_evaluation"]["metrics"].items()
                if metric["compliant"] is False
            },
        )
        for flow in flows
        if not flow["slo_evaluation"]["compliant"]
    ]
    assert not non_compliant, non_compliant
    assert traffic["attack_state"]["gold_sla_compliance_ratio"] == 1.0
    assert traffic["attack_state"]["slo_evaluation_semantics"].startswith(
        "service_aware"
    )

    by_application = {
        application: next(flow for flow in flows if flow["application"] == application)
        for application in {
            "RTP_OPUS",
            "RTP_VLC_AV",
            "FTP_DATA",
            "DNS",
            "RTP_TELEMOST",
            "LIVE_HLS",
        }
    }

    voice_metrics = by_application["RTP_OPUS"]["slo_evaluation"]["metrics"]
    assert voice_metrics["media_bitrate_kbps"]["evidence"] == "modeled"
    assert voice_metrics["one_way_media_latency_ms"]["compliant"] is True
    assert voice_metrics["jitter_ms"]["evidence"] == "not_modeled"
    assert voice_metrics["jitter_ms"]["observed_value"] is None

    vlc_metrics = by_application["RTP_VLC_AV"]["slo_evaluation"]["metrics"]
    assert vlc_metrics["media_bitrate_kbps"]["compliant"] is True
    assert vlc_metrics["network_packet_loss_percent"]["evidence"] == "modeled"

    ftp = by_application["FTP_DATA"]["slo_evaluation"]
    assert ftp["metrics"]["ftp_goodput_kbps"]["evidence"] == "assumed"
    assert ftp["metrics"]["ftp_completion_time_seconds"]["compliant"] is True
    assert ftp["metrics"]["tcp_retransmission_ratio_percent"]["compliant"] is True
    assert ftp["metrics"]["ftp_payload_checksum_success"]["evidence"] == "not_modeled"
    assert ftp["assessment_state"] == "compliant_with_modeling_limits"

    dns = by_application["DNS"]["slo_evaluation"]
    assert dns["metrics"]["dns_response_rtt_ms"]["evidence"] == "modeled"
    assert dns["metrics"]["dns_timeout_ratio_percent"]["evidence"] == "assumed"
    assert dns["metrics"]["dns_servfail_ratio_percent"]["evidence"] == "not_modeled"
    assert dns["metrics"]["dns_tcp_fallback_success"]["evidence"] == "not_modeled"

    conference = by_application["RTP_TELEMOST"]["slo_evaluation"]["metrics"]
    assert conference["conference_audio_latency_ms"]["budget_value"] > conference[
        "conference_video_latency_ms"
    ]["budget_value"]
    assert conference["conference_audio_latency_ms"]["evidence"] == "assumed"
    assert conference["conference_video_latency_ms"]["compliant"] is True

    live = by_application["LIVE_HLS"]["slo_evaluation"]
    assert live["metrics"]["ll_hls_live_edge_latency_ms"]["evidence"] == "assumed"
    assert live["metrics"]["ll_hls_startup_time_ms"]["evidence"] == "not_modeled"
    assert live["metrics"]["ll_hls_rebuffer_ratio_percent"]["evidence"] == "not_modeled"
    assert live["metrics"]["ll_hls_part_deadline_miss_ratio_percent"]["evidence"] == "not_modeled"
    assert live["assessment_coverage_ratio"] < 1.0


def test_sla_summary_uses_service_aware_flow_evaluation() -> None:
    baseline = simulate_packet_snapshot(
        MODEL,
        step_index=0,
        time_seconds=0,
        step_seconds=5,
        detail="flows",
    )
    ftp = deepcopy(
        next(
            flow
            for flow in baseline["flows"]
            if flow["application"] == "FTP_DATA" and flow["sla_grade"] == "gold"
        )
    )
    ftp["observed_dropped_packets"] = ftp["packet_count"]
    ftp["observed_retransmissions"] = ftp["data_segments"]

    evaluated, attack_state = apply_attack_effects(MODEL, [ftp], [])
    slo = evaluated[0]["slo_evaluation"]
    assert slo["compliant"] is False
    assert slo["assessment_state"] == "non_compliant"
    assert "ftp_goodput_kbps" in slo["failed_metric_ids"]
    assert "ftp_completion_time_seconds" in slo["failed_metric_ids"]
    assert "tcp_retransmission_ratio_percent" in slo["failed_metric_ids"]
    assert attack_state["gold_sla_compliance_ratio"] == 0.0
    gold_tier = next(
        tier
        for tier in attack_state["sla_restoration"]["tiers"]
        if tier["sla_grade"] == "gold"
    )
    assert gold_tier["sla_compliant_flow_count"] == 0
    assert gold_tier["mean_slo_assessment_coverage_ratio"] < 1.0


def test_baseline_routes_never_use_an_endpoint_as_transit() -> None:
    traffic = simulate_packet_snapshot(
        MODEL,
        step_index=0,
        time_seconds=0,
        step_seconds=5,
        detail="flows",
    )
    for flow in traffic["flows"]:
        for field in ("route", "reverse_route"):
            route = flow[field]
            assert route
            assert path_uses_only_data_plane_transit(MODEL, route)
            assert all(
                MODEL.graph.edges[source, target].get("medium")
                != "logical-service-binding"
                for source, target in zip(route, route[1:])
            )


def test_dual_homed_server_cannot_shortcut_the_core_network() -> None:
    # В полном неориентированном графе C1—SRV_MEDIA—C4 имеет малую сумму
    # задержек, но физический сервер не является маршрутизатором между C1/C4.
    route = shortest_data_path(MODEL, "C1", "C4")
    assert route[0] == "C1" and route[-1] == "C4"
    assert "SRV_MEDIA" not in route
    assert path_uses_only_data_plane_transit(MODEL, route)
    assert not path_uses_only_data_plane_transit(MODEL, [])
    assert not path_uses_only_data_plane_transit(
        MODEL,
        ["SVC_VOICE", "SRV_MEDIA"],
    )
