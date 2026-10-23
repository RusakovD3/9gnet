"""Карантин требует наблюдённого источника, а не списка предположений."""

from copy import deepcopy

import networkx as nx
import pytest

from src.gnet9.models import NetworkModel
from src.gnet9.packet_simulator import simulate_packet_snapshot
from src.gnet9.remapping import apply_gold_first_remap
from src.gnet9.topology_builder import GNetBaselineBuilder


@pytest.fixture
def source_flows():
    model = GNetBaselineBuilder().build()
    packet = simulate_packet_snapshot(
        model, step_index=0, time_seconds=0, step_seconds=2, detail="flows",
    )
    return model, packet["flows"][:2]


@pytest.mark.parametrize("stage", ["predictive_prestage", "confirmed_active_mitigation"])
def test_quarantine_list_and_stage_do_not_confirm_a_source(source_flows, stage):
    model, flows = source_flows
    source = flows[0]["client_node"]
    plan = {
        "valid_for_step": 1,
        "routing_control": {"quarantine_sources": [source], "protection_stage": stage},
    }
    remapped, summary = apply_gold_first_remap(model, flows, [], plan, step_index=1)
    assert summary["quarantine_sources"] == []
    assert not any(flow["isolated"] for flow in remapped)


def test_current_event_confirms_only_its_actual_source(source_flows):
    model, flows = source_flows
    source, innocent = (flow["client_node"] for flow in flows)
    plan = {
        "valid_for_step": 1,
        "routing_control": {
            "quarantine_sources": [source, innocent], "protection_stage": "predictive_prestage",
        },
    }
    events = [{"attack_id": "observed", "kind": "dos", "ingress_nodes": [source]}]
    remapped, summary = apply_gold_first_remap(model, flows, events, plan, step_index=1)
    assert summary["quarantine_sources"] == [source]
    assert remapped[0]["routing_action"] == "quarantine_attack_source"
    assert remapped[1]["isolated"] is False


@pytest.mark.parametrize("observation_type,stage,attack_id,expected", [
    ("active_event", "confirmed_active_mitigation", "observed", True),
    ("precursor", "confirmed_active_mitigation", "observed", False),
    ("active_event", "predictive_prestage", "observed", False),
    ("active_event", "confirmed_active_mitigation", "unrelated", False),
])
def test_retained_plan_requires_paired_active_evidence(
    source_flows, observation_type, stage, attack_id, expected,
):
    model, flows = source_flows
    source = flows[0]["client_node"]
    before = deepcopy(flows)
    model_before = deepcopy(model.to_serializable())
    plan = {
        "valid_from_step": 1, "valid_until_step": 2, "attack_ids": ["observed"],
        "routing_control": {
            "quarantine_sources": [source], "protection_stage": stage,
            "entity_records": [{
                "observation_type": observation_type, "attack_id": attack_id,
                "kind": "dos", "source_ids": [source],
            }],
        },
    }
    remapped, summary = apply_gold_first_remap(model, flows, [], plan, step_index=2)
    assert remapped[0]["isolated"] is expected
    assert summary["quarantine_sources"] == ([source] if expected else [])
    assert flows == before
    assert model.to_serializable() == model_before
    expired, _ = apply_gold_first_remap(model, flows, [], plan, step_index=3)
    assert not any(flow["isolated"] for flow in expired)


def test_inactive_plan_reports_observed_sources_without_applied_quarantine(source_flows):
    model, flows = source_flows
    source = flows[0]["client_node"]
    events = [{"attack_id": "observed", "kind": "dos", "ingress_nodes": [source]}]
    remapped, summary = apply_gold_first_remap(model, flows, events, None, step_index=1)
    assert summary["observed_source_ids"] == [source]
    assert summary["quarantine_sources"] == []
    assert not any(flow["isolated"] for flow in remapped)


@pytest.fixture
def shared_line():
    graph = nx.Graph()
    for node, role in (
        ("server", "service-server"), ("old", "core-router"), ("new", "core-router"),
        ("client", "fixed-subscriber"), ("attacker", "fixed-subscriber"),
    ):
        graph.add_node(node, role=role, level="L1" if "subscriber" in role else "L2")
    for source, target in (
        ("server", "old"), ("old", "client"), ("server", "new"),
        ("new", "client"), ("attacker", "new"),
    ):
        graph.add_edge(source, target, capacity_mbps=100.0, latency_ms=1.0)
    model = NetworkModel(graph, [], [], {})
    legitimate = {
        "client_node": "client", "server_node": "server", "sla_grade": "gold",
        "route": ["server", "old", "client"], "reverse_route": ["client", "old", "server"],
        "offered_rate_mbps": 1.1,
    }
    attack = {
        "client_node": "attacker", "server_node": "server", "sla_grade": "attack",
        "route": ["attacker", "new", "server"], "reverse_route": ["server", "new", "attacker"],
        "offered_rate_mbps": 79.0, "is_attack_traffic": True,
    }
    plan = {"valid_for_step": 1, "routing_control": {"avoid_nodes": ["old"]}}
    return model, legitimate, attack, plan


def test_current_flood_occupancy_precedes_legitimate_admission(shared_line):
    model, legitimate, attack, plan = shared_line
    remapped, summary = apply_gold_first_remap(
        model, [legitimate], [], plan, step_index=1, background_flows=[attack],
    )
    assert remapped[0]["isolated"]
    assert summary["network_capacity_rejected_candidate_count"] == 1
    assert summary["maximum_projected_edge_utilization_percent"] == pytest.approx(79.0)
    assert summary["maximum_projected_node_utilization_percent"] == pytest.approx(79.0)
    assert summary["maximum_admitted_route_utilization_percent"] == 0.0


@pytest.mark.parametrize("inactive_field", ["quarantined", "isolated", "empty_route", "zero_rate"])
def test_stopped_flood_does_not_occupy_the_reserve(shared_line, inactive_field):
    model, legitimate, attack, plan = shared_line
    if inactive_field == "empty_route":
        attack["route"] = []
    elif inactive_field == "zero_rate":
        attack["offered_rate_mbps"] = 0.0
    else:
        attack[inactive_field] = True
    remapped, summary = apply_gold_first_remap(
        model, [legitimate], [], plan, step_index=1, background_flows=[attack],
    )
    assert remapped[0]["route"] == ["server", "new", "client"]
    assert remapped[0]["isolated"] is False
    assert summary["maximum_projected_edge_utilization_percent"] == pytest.approx(1.1)
    assert summary["maximum_admitted_route_utilization_percent"] == pytest.approx(1.1)


@pytest.mark.parametrize("explicit_background", [False, True])
def test_background_and_reverse_path_are_reserved_once(shared_line, explicit_background):
    model, legitimate, attack, plan = shared_line
    legitimate["offered_rate_mbps"] = 1.0
    remapped, summary = apply_gold_first_remap(
        model, [legitimate, attack], [], plan, step_index=1,
        background_flows=[attack] if explicit_background else None,
    )
    assert remapped[0]["isolated"] is False
    assert summary["maximum_projected_edge_utilization_percent"] == pytest.approx(80.0)
    assert summary["maximum_projected_node_utilization_percent"] == pytest.approx(80.0)
    assert summary["maximum_admitted_route_utilization_percent"] == pytest.approx(80.0)


def test_inactive_plan_still_reports_total_current_occupancy(shared_line):
    model, legitimate, attack, _ = shared_line
    legitimate["route"] = ["server", "new", "client"]
    legitimate["reverse_route"] = ["client", "new", "server"]
    remapped, summary = apply_gold_first_remap(
        model, [legitimate], [], None, step_index=1, background_flows=[attack],
    )
    assert remapped[0]["isolated"] is False
    assert summary["plan_active"] is False
    assert summary["maximum_projected_edge_utilization_percent"] == pytest.approx(80.1)
    assert summary["maximum_admitted_route_utilization_percent"] == 0.0


def test_overloaded_background_does_not_become_a_safe_admitted_route(shared_line):
    model, legitimate, attack, plan = shared_line
    attack["offered_rate_mbps"] = 90.0
    remapped, summary = apply_gold_first_remap(
        model, [legitimate], [], plan, step_index=1, background_flows=[attack],
    )
    assert remapped[0]["isolated"]
    assert summary["maximum_projected_edge_utilization_percent"] == pytest.approx(90.0)
    assert summary["maximum_admitted_route_utilization_percent"] == 0.0
