"""Срок целого файла нельзя выводить из одного случайного малого окна."""

from types import SimpleNamespace

import networkx as nx
import pytest

from src.gnet9.attacks import _evaluate_flow_slo


def test_file_duration_uses_expected_rate_but_keeps_observed_damage():
    graph = nx.Graph()
    graph.add_node("client", traffic_kind="ftp", d0sl_policy={
        "file_size_mib": 1., "completion_time_budget_seconds": 12.,
        "tcp_retransmission_budget_percent": 1.,
    })
    model = SimpleNamespace(graph=graph)
    flow = dict(client_node="client", traffic_kind="ftp", sla_grade="bronze",
        transport="TCP", packet_count=100, data_segments=100, observed_retransmissions=0,
        payload_bytes=65536, observed_dropped_packets=0, interval_seconds=2,
        normal_expected_payload_rate_bps=1048576, route=["server", "client"],
        one_way_latency_ms=1.)
    # 1 МиБ / (1 Мибит/с) = 8 с независимо от реализации числа пакетов.
    for payload in (32768, 65536, 131072):
        metrics = _evaluate_flow_slo(model, dict(flow, payload_bytes=payload))["metrics"]
        assert metrics["ftp_completion_time_seconds"]["observed_value"] == pytest.approx(8.)
        assert metrics["ftp_completion_time_seconds"]["compliant"]
    damaged = _evaluate_flow_slo(model, dict(flow, observed_dropped_packets=50))["metrics"]
    assert damaged["ftp_completion_time_seconds"]["observed_value"] == pytest.approx(16.)
    assert not damaged["ftp_completion_time_seconds"]["compliant"]
    assert damaged["ftp_completion_time_seconds"]["evidence"] == "assumed"
