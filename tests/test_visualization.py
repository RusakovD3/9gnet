"""Показанные значения должны совпадать с сохранёнными наблюдениями."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from main import parse_args, _simulation_timing
from src.gnet9.address_visualizer import subscriber_address_groups
from src.gnet9.dynamics_charts import _link_load_rows, extract_dynamics_chart_series
from src.gnet9.network_layout import network_positions
from src.gnet9.topology_builder import GNetBaselineBuilder


def test_peak_load_uses_rate_at_each_step_not_largest_byte_count():
    graph = nx.Graph()
    graph.add_edge("a", "b", capacity_mbps=10)
    model = SimpleNamespace(graph=graph)
    def snapshot(step, byte_count, seconds):
        return {"step_index": step, "traffic": {"flows": [
            {"route": ["a", "b"], "wire_bytes": byte_count, "interval_seconds": seconds},
            {"route": ["a", "b"], "wire_bytes": 9000000, "interval_seconds": 1, "quarantined": True},
        ]}}
    rows = _link_load_rows(model, [snapshot(1, 1000000, 1), snapshot(2, 2000000, 10)])
    assert rows == [{"label": "a ↔ b", "utilization_percent": 80, "step_index": 1}]


def test_capacity_chart_reads_compact_saved_state_without_live_model():
    names = ["capacity_mbps", "speed_mbps"]
    compact = {"step_index": 4, "traffic": {"network_state": {
        "edge_metric_names": names, "edge_ids": [["a", "b"]], "edge_vectors": [[10, 8]]}}}
    full = {"step_index": 4, "traffic": {"network_state": {"edges": [
        {"source": "a", "target": "b", "metrics": dict(zip(names, [10, 8]))}]}}}
    assert _link_load_rows(None, [compact]) == _link_load_rows(None, [full])
    assert _link_load_rows(None, [compact])[0]["utilization_percent"] == 80


def test_sla_chart_shows_compliance_and_marks_missing_data():
    d = {"snapshots": [{"attacks": {"routing": {"by_sla": {"gold": {"availability_ratio": 1}}},
                         "sla_restoration": {"tiers": [
                             {"sla_grade": "gold", "flow_count": 10, "sla_compliant_flow_count": 9}]}},
                         "koopman": {"kalman_filter": {"applied_to_nominal_observation": False}}}]}
    series = extract_dynamics_chart_series(d)
    assert series["gold_availability_percent"] == [100]
    assert series["gold_sla_percent"] == [90]
    assert np.isnan(series["silver_sla_percent"][0])
    assert series["kalman_applied"] == [False]


def test_address_groups_cover_all_subscribers_without_invented_attachments():
    from src.gnet9.addressing import build_ip_address_plan
    model = GNetBaselineBuilder().build()
    groups = subscriber_address_groups(build_ip_address_plan(model)["devices"])
    assert len(groups) == 6
    assert sum(len(rows) for rows in groups.values()) == 480
    assert all(len({d["home_access"] for d in rows}) == 2 for rows in groups.values())
    rows = groups["10.1.1.0/24"]
    assert rows[0]["ip"] == "10.1.1.1" and rows[-1]["ip"] == "10.1.1.80"
    original = {n: a["pos"] for n, a in model.graph.nodes(data=True) if "pos" in a}
    positions = network_positions(model)
    assert len(positions) == len(model.graph)
    assert original == {n: a["pos"] for n, a in model.graph.nodes(data=True) if "pos" in a}


def test_default_demo_has_longer_training_and_explicit_clock():
    args = parse_args(["--attack-scenario", "predictive-demo"])
    assert _simulation_timing(args) == (160, 2)
    assert args.attack_start_step == 70
    # Пользовательское расписание остаётся главным.
    explicit = parse_args(["--attack-scenario", "predictive-demo", "--dynamics-steps", "100", "--attack-start-step", "50"])
    assert _simulation_timing(explicit) == (100, 2)
    assert explicit.attack_start_step == 50


def test_active_attack_and_nominal_queues_are_labelled_truthfully():
    from src.gnet9.flow_details import _interactive_snapshot_summary
    model = SimpleNamespace(graph=nx.Graph())
    state = {"step_index": 1, "attacks": {"active": True, "events": [{"name_ru": "Перегрузка"}],
             "sla_restoration": {"tiers": [{"sla_grade": "gold", "flow_count": 10, "sla_compliant_flow_count": 9}]}},
             "koopman": {"forecast_is_early_warning": False}}
    text = _interactive_snapshot_summary(state, [], {}, True, model)
    assert "НАБЛЮДАЕМАЯ АТАКА" in text
    assert "90.00%" in text
    t0 = _interactive_snapshot_summary({"step_index": 0}, [], {}, True, model)
    assert "без потерь, очередей" not in t0
    assert "SLA золотого класса (качество услуг): нет данных" in t0
