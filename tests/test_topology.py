from src.gnet9.dynamics import DynamicsConfig, simulate_stationary_dynamics, validate_healthy_baseline
from src.gnet9.topology_builder import GNetBaselineBuilder


MODEL = GNetBaselineBuilder().build()


def test_l2_count() -> None:
    l2_nodes = [node for node, attrs in MODEL.graph.nodes(data=True) if attrs["level"] == "L2"]
    assert len(l2_nodes) == 18  # 12 core + 6 aggregation routers


def test_l2_has_cisco_like_profiles() -> None:
    l2_nodes = [(node, attrs) for node, attrs in MODEL.graph.nodes(data=True) if attrs["level"] == "L2"]
    assert all(attrs["platform_family"] in {"NCS 5501", "Catalyst 9500", "ASR 1001-X"} for _, attrs in l2_nodes)
    assert all("l2_raw_baseline" in attrs for _, attrs in l2_nodes)
    assert all("l2_health_index" in attrs for _, attrs in l2_nodes)


def test_tensor_is_state_vector() -> None:
    tensor = MODEL.graph.nodes["C1"]["tensor"]
    assert tensor.axes == ("metric",)
    assert tensor.data.shape == (len(tensor.metric_names),)
    assert tensor.to_dict()["shape"] == [len(tensor.metric_names)]


def test_l0_service_tensor_metrics() -> None:
    tensor = MODEL.graph.nodes["SVC_VIDEO"]["tensor"]
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
    edge_attrs = MODEL.graph.edges["A1", "M1_01"]
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
    l0_nodes = [node for node, attrs in MODEL.graph.nodes(data=True) if attrs["level"] == "L0"]
    assert len(l0_nodes) == 4


def test_subscriber_count() -> None:
    l1_nodes = [node for node, attrs in MODEL.graph.nodes(data=True) if attrs["level"] == "L1"]
    assert len(l1_nodes) == 240


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
    assert health["checked_l1_points"] == 240 * 30
    assert health["checked_l2_nodes"] == 18
    assert health["checked_edges"] == MODEL.graph.number_of_edges()
    assert health["violation_count"] == 0


def test_stationary_dynamics_snapshots_every_five_seconds() -> None:
    dynamics = simulate_stationary_dynamics(MODEL)
    snapshots = dynamics["snapshots"]

    assert dynamics["mode"] == "stationary_healthy_baseline"
    assert dynamics["config"]["step_count"] == 6
    assert dynamics["config"]["duration_seconds"] == 30
    assert dynamics["health"]["ok"]
    assert [snapshot["time_seconds"] for snapshot in snapshots] == [0, 5, 10, 15, 20, 25, 30]
    assert len(snapshots[0]["nodes"]) == MODEL.graph.number_of_nodes()
    assert len(snapshots[0]["edges"]) == MODEL.graph.number_of_edges()
    assert "tensor" in snapshots[0]["nodes"][0]["tensors"]
    assert "tensor" in snapshots[0]["edges"][0]["tensors"]
    assert set(snapshots[0]["tensor_state"]["levels"]) == {"L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "EDGE"}
    assert all(snapshots[0]["tensor_state"]["counts"][level] > 0 for level in snapshots[0]["tensor_state"]["levels"])
    assert snapshots[0]["state_vector"]["metric_names"]
    assert len(snapshots[0]["state_vector"]["metric_names"]) == len(snapshots[0]["state_vector"]["vector"])
    assert snapshots[0]["traffic"]["summary"]["flow_count"] == 240
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
    assert snapshot["traffic"]["summary"]["flow_count"] == 240


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
    assert arbitrator["input_tensor_counts"]["L1"] == 240
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
    assert len(traffic["flows"]) == 240


def test_packet_simulation_builds_realistic_in_memory_headers() -> None:
    dynamics = simulate_stationary_dynamics(MODEL, DynamicsConfig(step_seconds=5, step_count=1))
    traffic = dynamics["snapshots"][0]["traffic"]
    summary = traffic["summary"]
    packets = traffic["packet_sample"]

    assert summary["flow_count"] == 240
    assert summary["tcp_flow_count"] > 0
    assert summary["udp_flow_count"] > 0
    assert summary["observed_loss_ratio"] == 0.0
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
