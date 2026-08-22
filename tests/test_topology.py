from dataclasses import replace
from pathlib import Path

from src.gnet9.baseline import l0_service_tensor
from src.gnet9.constants import L2_NODE_COUNT, SUBSCRIBER_COUNT
from src.gnet9.dynamics import DynamicsConfig, simulate_stationary_dynamics, validate_healthy_baseline
from src.gnet9 import dynamics_charts
from src.gnet9.dynamics_charts import extract_application_series, extract_dynamics_chart_series
from src.gnet9.topology_builder import GNetBaselineBuilder


MODEL = GNetBaselineBuilder().build()


def test_l2_count() -> None:
    l2_nodes = [node for node, attrs in MODEL.graph.nodes(data=True) if attrs["level"] == "L2"]
    assert len(l2_nodes) == L2_NODE_COUNT


def test_l2_has_cisco_profiles_with_provenance() -> None:
    l2_nodes = [(node, attrs) for node, attrs in MODEL.graph.nodes(data=True) if attrs["level"] == "L2"]
    assert all(
        attrs["platform_family"]
        in {
            "NCS 5501",
            "Catalyst C9500-24Y4C",
            "Integrated gNodeB/UPF access edge",
            "XGS-PON OLT access shelf",
        }
        for _, attrs in l2_nodes
    )
    assert all("l2_raw_baseline" in attrs for _, attrs in l2_nodes)
    assert all("l2_health_index" in attrs for _, attrs in l2_nodes)
    assert all(attrs["l2_profile"]["verified_fields"] for _, attrs in l2_nodes)
    assert all(attrs["l2_profile"]["assumed_fields"] for _, attrs in l2_nodes)


def test_ncs5501_published_forwarding_memory_and_buffer_limits_are_preserved() -> None:
    """Do not silently replace published chassis limits with scenario guesses."""
    profile = MODEL.graph.nodes["C1"]["l2_profile"]

    assert profile["forwarding_mpps"] == 720.0
    assert profile["dram_gb"] == 32.0
    assert profile["buffer_mb"] == 4_112.0  # 16 MB on-chip + 4 GiB off-chip
    assert {"forwarding_mpps", "dram_gb", "buffer_mb"} <= set(profile["verified_fields"])
    assert {"forwarding_mpps", "dram_gb", "buffer_mb"}.isdisjoint(profile["assumed_fields"])
    assert any("platform-ar-wp" in url for url in profile["reference_urls"])


def test_c9500_published_default_core_sdm_route_scale_is_preserved() -> None:
    profile = MODEL.graph.nodes["A1"]["l2_profile"]

    assert profile["fib_routes"] == 212_000
    assert "fib_routes" in profile["verified_fields"]
    assert "fib_routes" not in profile["assumed_fields"]


def test_l2_power_values_preserve_vendor_semantics_and_model_assumptions() -> None:
    core = MODEL.graph.nodes["C1"]
    aggregation = MODEL.graph.nodes["A1"]

    assert core["l2_profile"]["typical_power_w"] == 240.0
    assert core["l2_profile"]["max_power_w"] == 370.0
    assert core["power_architecture"]["modeled_t0_power_w"] == 240.0
    assert "output_power" in core["power_architecture"]["vendor_power_value_semantics"]

    assert aggregation["l2_profile"]["typical_power_w"] is None
    assert aggregation["l2_profile"]["max_power_w"] is None
    assert aggregation["l2_profile"]["thermal_output_btu_per_hour"] == 1_454.0
    assert aggregation["l2_profile"]["thermal_output_equivalent_w"] == 426.0
    assert aggregation["power_architecture"]["modeled_t0_power_w"] == 250.0
    assert "thermal_equivalent" in aggregation["power_architecture"]["vendor_power_value_semantics"]
    assert aggregation["power_architecture"]["backup_autonomy_origin"].endswith("not_vendor_spec")


def test_selected_equipment_links_fit_published_port_speeds() -> None:
    """The topology must not invent port rates absent from selected platforms."""
    for source, target, attrs in MODEL.graph.edges(data=True):
        source_role = MODEL.graph.nodes[source].get("role")
        target_role = MODEL.graph.nodes[target].get("role")
        capacity = float(attrs.get("capacity_mbps", 0.0))
        if source_role == "core-router" and target_role == "core-router":
            assert capacity <= 100_000.0
        if {source_role, target_role} == {"core-router", "service-server"}:
            assert capacity == 10_000.0
            assert attrs["physical_profile"].startswith("Dell R660 10GbE")

    for node, attrs in MODEL.graph.nodes(data=True):
        if attrs.get("role") == "core-router":
            assert attrs["physical_high_speed_ports_used"] <= 6
        elif attrs.get("role") == "aggregation-switch":
            assert attrs["physical_high_speed_ports_used"] <= 4
        elif attrs.get("role") in {"radio-access-node", "optical-line-terminal"}:
            aggregation_links = [
                neighbor
                for neighbor in MODEL.graph.neighbors(node)
                if MODEL.graph.nodes[neighbor].get("role") == "aggregation-switch"
            ]
            assert len(aggregation_links) >= 2


def test_subscriber_access_is_independent_of_application_and_logically_aggregated() -> None:
    grouped_capacities: dict[tuple[str, str], set[float]] = {}
    grouped_latencies: dict[str, set[float]] = {}
    for node, attrs in MODEL.graph.nodes(data=True):
        if attrs.get("level") != "L1":
            continue
        key = (attrs["role"], attrs["sla_grade"])
        grouped_capacities.setdefault(key, set()).add(float(attrs["access_capacity_mbps"]))
        grouped_latencies.setdefault(attrs["role"], set()).add(float(attrs["access_latency_ms"]))
        primary = attrs["home_access"]
        edge = MODEL.graph.edges[node, primary]
        assert edge["consumes_c9500_physical_port"] is False
        assert attrs["access_architecture_reference_url"].startswith("https://www.itu.int/")
        assert edge["attachment_semantics"].startswith("logical_")

    assert all(len(capacities) == 1 for capacities in grouped_capacities.values())
    assert all(len(latencies) == 1 for latencies in grouped_latencies.values())
    assert grouped_capacities[("mobile-subscriber", "gold")] == {100.0}
    assert grouped_capacities[("fixed-subscriber", "silver")] == {500.0}


def test_tensor_is_state_vector() -> None:
    tensor = MODEL.graph.nodes["C1"]["tensor"]
    assert tensor.axes == ("metric",)
    assert tensor.data.shape == (len(tensor.metric_names),)
    assert tensor.to_dict()["shape"] == [len(tensor.metric_names)]


def test_l0_service_tensor_metrics() -> None:
    tensor = MODEL.graph.nodes["SVC_VLC"]["tensor"]
    assert tensor.metric_names == (
        "service_code",
        "bitrate_mbps",
        "latency_budget_ms",
        "jitter_budget_ms",
        "availability_target",
        "priority_code",
        "demand_pressure",
        "service_health",
    )


def test_l0_service_tensor_is_independent_of_localized_display_name() -> None:
    profile = next(service for service in MODEL.services if service.service_id == "SVC_VOICE")
    localized = replace(profile, name="Любое локализованное имя")
    assert l0_service_tensor(localized)["service_code"] == 1.0


def test_l1_tensor_has_access_service_processing_and_cost() -> None:
    tensor = MODEL.graph.nodes["M1_01"]["tensor"]
    assert "access_type_code" in tensor.metric_index
    assert "service_code" in tensor.metric_index
    assert "request_rate_pps" in tensor.metric_index
    assert "processing_speed_mbps" in tensor.metric_index
    assert "capex_opex_cost" in tensor.metric_index


def test_l2_tensor_has_load_port_and_stability_metrics() -> None:
    tensor = MODEL.graph.nodes["C1"]["tensor"]
    assert tensor.metric_names == (
        "ram_used_gb",
        "ram_load_percent",
        "cpu_load_percent",
        "packet_processing_time_ms",
        "traffic_distribution_code",
        "port_delay_ms",
        "port_speed_mbps",
        "capex_opex_cost",
        "stability_margin",
    )


def test_transport_edges_have_l3_l4_and_edge_tensors() -> None:
    edge_attrs = MODEL.graph.edges[MODEL.graph.nodes["M1_01"]["home_access"], "M1_01"]
    assert edge_attrs["l3_tensor"].metric_names == (
        "medium_code",
        "line_rate_mbps",
        "distance_m",
        "frequency_mhz",
        "attenuation_db",
        "noise_interference_db",
        "snr_db",
    )
    assert edge_attrs["l4_tensor"].metric_names == (
        "x_mid",
        "y_mid",
        "length_m",
        "cross_connect_present",
        "duct_capacity_used_ratio",
        "repair_time_hours",
    )
    assert "stability_margin" in edge_attrs["tensor"].metric_index


def test_l5_l6_l7_l8_tensors_are_present() -> None:
    node_attrs = MODEL.graph.nodes["C1"]
    assert "remap_algorithm_code" in node_attrs["l5_tensor"].metric_index
    assert "power_supply_code" in node_attrs["l6_tensor"].metric_index
    assert "koopman_residual" in MODEL.graph.nodes["ARB"]["tensor"].metric_index
    assert "x" in node_attrs["l8_tensor"].metric_index
    assert "y" in node_attrs["l8_tensor"].metric_index


def test_service_count() -> None:
    services = [node for node, attrs in MODEL.graph.nodes(data=True) if attrs.get("role") == "service"]
    servers = [node for node, attrs in MODEL.graph.nodes(data=True) if attrs.get("role") == "service-server"]
    assert len(services) == 6
    assert len(servers) == 4


def test_subscriber_count() -> None:
    l1_nodes = [node for node, attrs in MODEL.graph.nodes(data=True) if attrs["level"] == "L1"]
    assert len(l1_nodes) == SUBSCRIBER_COUNT


def test_l1_baseline_has_no_sla_violations() -> None:
    for _, attrs in MODEL.graph.nodes(data=True):
        if attrs.get("level") != "L1":
            continue

        for point in attrs["monitoring"]:
            assert point["bitrate_slo_ok"]
            assert point["latency_slo_ok"]
            assert point["loss_slo_ok"]
            assert point["jitter_slo_ok"]
            assert not point["bitrate_drop_alarm"]


def test_healthy_baseline_validation_passes() -> None:
    health = validate_healthy_baseline(MODEL)

    assert health["ok"]
    assert health["checked_l1_points"] == SUBSCRIBER_COUNT * 30
    assert health["checked_l2_nodes"] == L2_NODE_COUNT
    assert health["checked_edges"] == MODEL.graph.number_of_edges()
    assert health["violation_count"] == 0


def test_stationary_dynamics_snapshots_every_five_seconds() -> None:
    dynamics = simulate_stationary_dynamics(MODEL)
    snapshots = dynamics["snapshots"]

    assert dynamics["mode"] == "stationary_healthy_baseline"
    assert dynamics["config"]["step_count"] == 10
    assert dynamics["config"]["duration_seconds"] == 50
    assert dynamics["health"]["ok"]
    assert dynamics["ideal_t0"]["ok"]
    assert dynamics["ideal_t0"]["status"] == "IDEAL_REALISTIC_BASELINE"
    assert dynamics["ideal_t0"]["metrics"]["subscriber_count"] == SUBSCRIBER_COUNT
    assert dynamics["ideal_t0"]["metrics"]["aggregation_switch_count"] == 6
    assert dynamics["ideal_t0"]["metrics"]["core_router_count"] == 12
    assert dynamics["ideal_t0"]["metrics"]["radio_access_node_count"] == 6
    assert dynamics["ideal_t0"]["metrics"]["optical_line_terminal_count"] == 6
    assert dynamics["ideal_t0"]["metrics"]["service_count"] == 6
    assert dynamics["ideal_t0"]["metrics"]["service_server_count"] == 4
    assert dynamics["ideal_t0"]["metrics"]["maximum_planned_link_utilization"] <= 0.12
    assert dynamics["ideal_t0"]["metrics"]["maximum_observed_link_utilization_percent"] <= 25.0
    assert dynamics["ideal_t0"]["metrics"]["maximum_l2_high_speed_ports_used"] <= 6
    assert dynamics["ideal_t0"]["metrics"]["network_device_dual_feed_count"] == L2_NODE_COUNT
    assert dynamics["ideal_t0"]["metrics"]["network_device_power_provenance_count"] == L2_NODE_COUNT
    assert dynamics["ideal_t0"]["metrics"]["maximum_modeled_t0_network_device_power_w"] == 250.0
    assert dynamics["ideal_t0"]["metrics"]["service_server_l6_count"] == 4
    assert dynamics["ideal_t0"]["metrics"]["service_server_dual_feed_count"] == 4
    assert dynamics["ideal_t0"]["metrics"]["service_server_power_provenance_count"] == 4
    assert dynamics["ideal_t0"]["metrics"]["service_standby_cross_power_domain_count"] == 18
    assert dynamics["ideal_t0"]["metrics"]["local_power_fault_domain_count"] == L2_NODE_COUNT + 4
    assert dynamics["ideal_t0"]["metrics"]["minimum_link_stability_margin"] >= 0.88
    assert dynamics["ideal_t0"]["metrics"]["l7_decision"] == "NO_REMAP"
    assert [snapshot["time_seconds"] for snapshot in snapshots] == list(range(0, 51, 5))
    assert len(snapshots[0]["nodes"]) == MODEL.graph.number_of_nodes()
    assert len(snapshots[0]["edges"]) == MODEL.graph.number_of_edges()
    assert "tensor" in snapshots[0]["nodes"][0]["tensors"]
    assert "tensor" in snapshots[0]["edges"][0]["tensors"]
    assert set(snapshots[0]["tensor_state"]["levels"]) == {"L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "EDGE"}
    assert all(snapshots[0]["tensor_state"]["counts"][level] > 0 for level in snapshots[0]["tensor_state"]["levels"])
    assert snapshots[0]["state_vector"]["metric_names"]
    assert len(snapshots[0]["state_vector"]["metric_names"]) == len(snapshots[0]["state_vector"]["vector"])
    assert snapshots[0]["traffic"]["summary"]["flow_count"] == SUBSCRIBER_COUNT
    assert snapshots[0]["traffic"]["summary"]["observed_dropped_packets"] == 0
    assert snapshots[0]["traffic"]["summary"]["observed_retransmissions"] == 0


def test_stationary_dynamics_uses_configurable_step_count() -> None:
    dynamics = simulate_stationary_dynamics(MODEL, DynamicsConfig(step_seconds=5, step_count=3))

    assert dynamics["config"]["step_count"] == 3
    assert dynamics["config"]["duration_seconds"] == 15
    assert [snapshot["time_seconds"] for snapshot in dynamics["snapshots"]] == [0, 5, 10, 15]


def test_stationary_dynamics_can_disable_packet_simulation() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(step_seconds=5, step_count=1, include_packet_simulation=False),
    )

    assert "traffic" not in dynamics["snapshots"][0]


def test_stationary_dynamics_summary_detail_is_compact() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(
            step_seconds=5,
            step_count=1,
            snapshot_detail="summary",
            packet_detail="summary",
        ),
    )
    snapshot = dynamics["snapshots"][0]

    assert "nodes" not in snapshot
    assert "edges" not in snapshot
    assert "by_level" not in snapshot["tensor_state"]
    assert "state_vector" in snapshot
    assert "flows" not in snapshot["traffic"]
    assert "packet_sample" not in snapshot["traffic"]
    assert snapshot["traffic"]["summary"]["flow_count"] == SUBSCRIBER_COUNT


def test_stationary_dynamics_tensor_detail_keeps_tensor_values_without_graph_lists() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(step_seconds=5, step_count=1, snapshot_detail="tensor", include_packet_simulation=False),
    )
    snapshot = dynamics["snapshots"][0]

    assert "nodes" not in snapshot
    assert "edges" not in snapshot
    assert "by_level" in snapshot["tensor_state"]
    assert snapshot["tensor_state"]["by_level"]["L1"]


def test_arbitrator_observes_tensor_state_and_keeps_no_remap_baseline() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(step_seconds=5, step_count=1, include_packet_simulation=False),
    )
    snapshot = dynamics["snapshots"][0]
    arbitrator = snapshot["arbitrator"]

    assert arbitrator["node_id"] == "ARB"
    assert arbitrator["input_tensor_counts"] == snapshot["tensor_state"]["counts"]
    assert arbitrator["input_tensor_counts"]["L1"] == SUBSCRIBER_COUNT
    assert arbitrator["input_tensor_counts"]["EDGE"] == MODEL.graph.number_of_edges()
    assert "sla_margin" in arbitrator["level_metric_aggregates"]["L1"]["metrics"]
    assert "stability_margin" in arbitrator["level_metric_aggregates"]["EDGE"]["metrics"]
    assert arbitrator["state_vector"] == snapshot["state_vector"]
    assert arbitrator["analysis"]["remap_pressure"] == 0.0
    assert arbitrator["remap"]["needed"] is False
    assert arbitrator["remap"]["action"] == "NO_REMAP"


def test_packet_sample_limit_is_configurable() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(step_seconds=5, step_count=1, packet_sample_limit=5),
    )

    assert len(dynamics["snapshots"][0]["traffic"]["packet_sample"]) == 5


def test_packet_flow_detail_omits_representative_packet_samples() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(step_seconds=5, step_count=1, packet_detail="flows"),
    )
    traffic = dynamics["snapshots"][0]["traffic"]

    assert "flows" in traffic
    assert "packet_sample" not in traffic
    assert len(traffic["flows"]) == SUBSCRIBER_COUNT


def test_packet_simulation_builds_realistic_in_memory_headers() -> None:
    dynamics = simulate_stationary_dynamics(MODEL, DynamicsConfig(step_seconds=5, step_count=1))
    traffic = dynamics["snapshots"][0]["traffic"]
    summary = traffic["summary"]
    packets = traffic["packet_sample"]

    assert summary["flow_count"] == SUBSCRIBER_COUNT
    assert summary["tcp_flow_count"] > 0
    assert summary["udp_flow_count"] > 0
    assert summary["observed_loss_ratio"] == 0.0
    assert set(summary["applications"]) == {"DNS", "FTP_DATA", "LIVE_HLS", "RTP_OPUS", "RTP_TELEMOST", "RTP_VLC_AV"}
    assert sum(item["flow_count"] for item in summary["applications"].values()) == SUBSCRIBER_COUNT
    assert summary["offered_rate_mbps"] > 0.0
    assert 0.0 < summary["protocol_efficiency_ratio"] < 1.0
    assert packets

    protocols = {packet["ipv4"]["protocol"] for packet in packets}
    roles = {packet["packet_role"] for packet in packets}
    assert {"TCP", "UDP"}.issubset(protocols)
    assert {"tcp_syn", "tcp_data", "dns_query", "udp_media"}.issubset(roles)

    routed_packet = next(packet for packet in packets if len(packet["route"]) > 2)
    hop_frames = routed_packet["ethernet"]["hop_frames"]
    assert len(hop_frames) == len(routed_packet["route"]) - 1
    assert routed_packet["ipv4"]["ttl_at_destination"] < routed_packet["ipv4"]["ttl_start"]
    assert hop_frames[0]["dst_mac"] != hop_frames[-1]["dst_mac"]


def test_dynamics_chart_series_exposes_sla_traffic_and_arbitrator_metrics() -> None:
    dynamics = simulate_stationary_dynamics(MODEL, DynamicsConfig(step_seconds=5, step_count=1))
    series = extract_dynamics_chart_series(dynamics)

    assert series["time_seconds"] == [0.0, 5.0]
    assert len(series["sla_margin_min"]) == 2
    assert min(series["sla_margin_min"]) > 0.0
    assert len(series["packet_count"]) == 2
    assert min(series["packet_count"]) > 0.0
    assert min(series["traffic_rate_mbps"]) > 0.0
    # Небольшой диагностический шум допустим, но здоровый шаг не должен
    # пересекать управляющий порог планирования переназначения 0.20.
    assert max(series["remap_pressure"]) < 0.20
    assert series["state_hausdorff_normalized"][0] == 0.0
    assert len(series["state_hausdorff_normalized"]) == 2
    applications = extract_application_series(dynamics)
    assert set(applications) == {"DNS", "FTP_DATA", "LIVE_HLS", "RTP_OPUS", "RTP_TELEMOST", "RTP_VLC_AV"}
    assert all(len(values["offered_rate_mbps"]) == 2 for values in applications.values())


def test_chart_export_keeps_only_the_compact_current_chart_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    stale_names = (
        "dynamics_health.png",
        "traffic_composition.png",
        "link_capacity.png",
        "dynamics_arbitrator.png",
        "dynamics_attacks.png",
        "dynamics_remapping.png",
        "attack_arrival_weibull.png",
    )
    for stale_name in stale_names:
        (tmp_path / stale_name).write_bytes(b"obsolete chart")
    for name in ("_plot_dynamics_overview", "_plot_capacity_bottlenecks"):
        monkeypatch.setattr(dynamics_charts, name, lambda *args, **kwargs: None)

    paths = dynamics_charts.export_dynamics_charts(
        {"snapshots": []},
        tmp_path,
    )

    assert set(paths) == {"overview", "capacity"}
    assert all(not (tmp_path / stale_name).exists() for stale_name in stale_names)
