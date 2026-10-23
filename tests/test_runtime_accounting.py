"""Баланс очередей и однократный учёт сквозных потерь на малом графе."""

from copy import deepcopy
from types import SimpleNamespace

import networkx as nx
import pytest

from src.gnet9.gelenbe import gelenbe_queue
from src.gnet9.runtime_metrics import EDGE_METRICS, VERTEX_METRICS, network_runtime_metrics
from src.gnet9.transport_queue import apply_transport_queues


def _model():
    graph = nx.Graph()
    graph.add_node("server", level="L0", role="service-server", runtime={"cpu_util_percent": 18.0})
    graph.add_node("router", level="L2", cpu_load_percent=10.0)
    graph.add_node("client", level="L1", kendall_queue={
        "service_rate_pps": 100.0, "utilization_rho": 0.5, "capacity_packets": 128,
    })
    graph.add_edge("server", "router", capacity_mbps=10.0, latency_ms=1.0)
    graph.add_edge("router", "client", capacity_mbps=10.0, latency_ms=2.0)
    return SimpleNamespace(graph=graph)


def _flow(**changes):
    result = dict(client_node="client", service_node="server",
                  route=["server", "router", "client"], packet_count=100,
                  payload_bytes=80000, wire_bytes=100000, interval_seconds=2,
                  observed_dropped_packets=10, traffic_kind="ftp", transport="TCP",
                  one_way_latency_ms=10.0, rtt_ms=20.0,
                  normal_service_reference_packet_rate_pps=50.0)
    result.update(changes)
    return result


def test_gelenbe_two_place_known_delay_variance_and_flow_balance():
    # λ=μ+ν и K=2 дают p0=p1=p2=1/3. Допущенный пакет проходит
    # один или два экспоненциальных этапа по 0,5 с: E[T]=0,75 с,
    # Var(T)=(1,5+0,25)/4=0,4375 с².
    result = gelenbe_queue(2, 1, 1, 2)
    assert result["blocking_probability"] == pytest.approx(1 / 3)
    assert result["mean_system_packets"] == pytest.approx(1)
    assert result["mean_queue_packets"] == pytest.approx(1 / 3)
    assert result["mean_system_time_ms"] == pytest.approx(750)
    assert result["delay_variance_ms2"] == pytest.approx(437500)
    assert result["jitter_ms"] ** 2 == pytest.approx(437500)
    assert result["served_rate_pps"] + result["destroyed_rate_pps"] + result["overflow_rate_pps"] == pytest.approx(2)
    # Закон Литтла: L = λ_доп × E[T].
    assert result["effective_arrival_rate_pps"] * result["mean_system_time_ms"] / 1000 == pytest.approx(1)


def test_gelenbe_delay_survives_underflow_of_nonfull_state_probability():
    # Почти всегда заняты все 128 мест. Редкий допущенный пакет всё равно
    # должен пройти 128 этапов со средней длительностью 1/μ каждый.
    result = gelenbe_queue(1e300, 1e-30, 0, 128)
    assert result["mean_system_time_ms"] == pytest.approx(128e33)
    assert result["delay_variance_ms2"] == pytest.approx(128e66)


@pytest.mark.parametrize("reverse_route", [False, True])
def test_end_to_end_packet_losses_have_one_explicit_accounting_owner(reverse_route):
    model = _model()
    reference = deepcopy(nx.node_link_data(model.graph))
    flow = _flow(transport_queue={"service_rate_pps": 100.0, "negative_arrival_rate_pps": 25.0})
    if reverse_route:
        flow["route"] = list(reversed(flow["route"]))
    state = network_runtime_metrics(model, [flow], detail="flows")
    nodes = {node["id"]: node for node in state["nodes"]}
    edges = {frozenset((edge["source"], edge["target"])): edge for edge in state["edges"]}
    assert sum(node["observed_dropped_packets"] for node in nodes.values()) == 10
    assert sum(edge["observed_flow_drops"] for edge in edges.values()) == 10
    assert nodes["client"]["metrics"]["negative_arrival_rate_pps"] == 25
    assert nodes["router"]["metrics"]["negative_arrival_rate_pps"] == 0
    assert nodes["server"]["metrics"]["negative_arrival_rate_pps"] == 0
    assert nodes["router"]["exposed_end_to_end_flow_drops"] == 10
    assert edges[frozenset(("router", "client"))]["metrics"]["file_drop_fraction"] == 0.1
    assert edges[frozenset(("server", "router"))]["metrics"]["file_drop_fraction"] == 0
    assert "not_physical_loss" in state["loss_accounting_semantics"]
    assert tuple(nodes["client"]["metrics"]) == VERTEX_METRICS
    assert all(tuple(edge["metrics"]) == EDGE_METRICS for edge in edges.values())
    assert nx.node_link_data(model.graph) == reference


def test_attack_arrivals_remain_positive_competing_load_in_both_export_formats():
    model = _model()
    attack = _flow(client_node="server", route=["server", "router"],
                   service_node="router", is_attack_traffic=True, traffic_kind="dos",
                   packet_count=10, wire_bytes=10000, observed_dropped_packets=4)
    detailed = network_runtime_metrics(model, [attack], detail="flows")
    nodes = {node["id"]: node for node in detailed["nodes"]}
    assert nodes["router"]["attack_arrival_rate_pps"] == 5
    assert nodes["router"]["metrics"]["normal_arrival_rate_pps"] == 0
    assert nodes["router"]["metrics"]["negative_arrival_rate_pps"] == 0
    assert nodes["router"]["gelenbe_queue"]["arrival_rate_pps"] == 5
    compact = network_runtime_metrics(model, [attack])
    index = compact["node_ids"].index("router")
    assert compact["node_attack_arrival_rates_pps"][index] == 5
    assert sum(compact["edge_observed_flow_drops"]) == 4


def test_transport_attack_delay_is_credited_once_across_queue_and_recovery():
    model = _model()
    flow = _flow(observed_dropped_packets=20, attack_ids=["dos"],
                 attack_latency_penalty_ms=20.0, one_way_latency_ms=30.0, rtt_ms=60.0)
    attack = dict(attack_id="dos", packet_count=1000, interval_seconds=2)
    apply_transport_queues(model, [flow], [attack])
    queue = flow["transport_queue"]
    assert queue["queue_delay_increment_ms"] > 20
    # В исходных 30 мс уже 20 мс воздействия. Общая задержка равна
    # 10 мс пути + максимуму(20 мс, очередь + восстановление).
    expected = 10 + max(20, queue["queue_delay_increment_ms"] + 1000 * queue["tcp_recovery_seconds"])
    assert flow["one_way_latency_ms"] == pytest.approx(expected, abs=2e-4)
    assert queue["tcp_additional_delivery_delay_ms"] == pytest.approx(1000 * queue["tcp_recovery_seconds"])
    assert flow["observed_dropped_packets"] == queue["observed_dropped_packets"] == 20
    assert queue["attack_arrival_rate_pps"] == 500
    assert queue["normal_arrival_rate_pps"] == 50


def test_unrouted_flows_do_not_absorb_a_share_of_competing_attack_load():
    model = _model()
    active = _flow(attack_ids=["dos"])
    isolated = _flow(attack_ids=["dos"], isolated=True)
    attack = dict(attack_id="dos", packet_count=1000, interval_seconds=2)
    apply_transport_queues(model, [active, isolated], [attack])
    assert active["transport_queue"]["attack_arrival_rate_pps"] == 500
    assert "transport_queue" not in isolated


def test_cpu_and_buffer_metrics_are_queue_estimates_with_explicit_units():
    model = _model()
    state = network_runtime_metrics(model, [_flow()], detail="flows")
    for node in state["nodes"]:
        metrics = node["metrics"]
        assert 0 <= metrics["cpu_load_percent"] <= 100
        # Один пакет 1000 байт; память — средняя ожидающая очередь,
        # не вся оперативная память и не паспортная ёмкость буфера.
        assert metrics["buffer_memory_mib"] == pytest.approx(metrics["mean_queue_packets"] * 1000 / 1024 ** 2)
        assert node["gelenbe_queue"]["jitter_ms"] ** 2 == pytest.approx(metrics["delay_variance_ms2"])
    compact = network_runtime_metrics(model, [_flow()])
    assert "not_measured_ram" in compact["metric_origins"]["buffer_memory_mib"]
    assert compact["vertex_units"][7] == "ms2"
    assert compact["edge_units"][2] == "ms"
