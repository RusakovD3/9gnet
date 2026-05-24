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
