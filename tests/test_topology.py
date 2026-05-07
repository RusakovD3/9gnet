from src.gnet9.l2_equipment import L2_RESOURCES, L2_SECURITY_STATES, L2_TELEMETRY_STEPS, L2_TRAFFIC_CLASSES, L2_TENSOR_METRICS
from src.gnet9.topology_builder import GNetBaselineBuilder


def test_l2_count() -> None:
    model = GNetBaselineBuilder().build()
    l2_nodes = [node for node, attrs in model.graph.nodes(data=True) if attrs["level"] == "L2"]
    assert len(l2_nodes) == 18  # 12 core + 6 aggregation routers


def test_l2_has_cisco_like_profiles() -> None:
    model = GNetBaselineBuilder().build()
    l2_nodes = [(node, attrs) for node, attrs in model.graph.nodes(data=True) if attrs["level"] == "L2"]
    assert all(attrs["platform_family"] in {"NCS 5501", "Catalyst 9500", "ASR 1001-X"} for _, attrs in l2_nodes)
    assert all("l2_raw_baseline" in attrs for _, attrs in l2_nodes)
    assert all("l2_health_index" in attrs for _, attrs in l2_nodes)


def test_l2_tensor_is_real_5d_equipment_tensor() -> None:
    model = GNetBaselineBuilder().build()
    node_attrs = model.graph.nodes["C1"]
    tensor = node_attrs["tensor"]
    assert tensor.axis_names == ("resource", "time", "security_state", "traffic_class", "metric")
    assert tensor.data.shape == (
        len(L2_RESOURCES),
        L2_TELEMETRY_STEPS,
        len(L2_SECURITY_STATES),
        len(L2_TRAFFIC_CLASSES),
        len(L2_TENSOR_METRICS),
    )


def test_service_count() -> None:
    model = GNetBaselineBuilder().build()
    l0_nodes = [node for node, attrs in model.graph.nodes(data=True) if attrs["level"] == "L0"]
    assert len(l0_nodes) == 4


def test_subscriber_count() -> None:
    model = GNetBaselineBuilder().build()
    l1_nodes = [node for node, attrs in model.graph.nodes(data=True) if attrs["level"] == "L1"]
    assert len(l1_nodes) == 240
