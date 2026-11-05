"""Нагрузка служб: единицы, область ущерба, причинность и пакетные записи."""

from copy import deepcopy
from dataclasses import replace

import pytest

from main import _simulation_timing, parse_args
from src.gnet9.attacks import (
    SERVICE_REQUEST_ATTACKS, _attack_capacity_context, active_attack_events, attack_catalog,
    observe_attack_precursors,
)
from src.gnet9.koopman import THREAT_METRICS, _classify_precursor_signals, _predicted_precursor_kind
from src.gnet9.flow_details import FLOW_SHORT_LABELS
from src.gnet9.flow_visualizer import FLOW_COLORS
from src.gnet9.packet_simulator import simulate_packet_snapshot
from src.gnet9.routing import shortest_data_path, path_uses_only_data_plane_transit
from src.gnet9.topology_builder import GNetBaselineBuilder


@pytest.fixture(scope="module")
def model():
    return GNetBaselineBuilder().build()


@pytest.mark.parametrize("profile,ratio,budget", [
    (SERVICE_REQUEST_ATTACKS[0], 1.6, 5000.0),
    (SERVICE_REQUEST_ATTACKS[1], 1.5, 4000.0),
])
def test_request_work_uses_cpu_time_instead_of_bandwidth(model, profile, ratio, budget):
    routes = [shortest_data_path(model, source, profile.target_id) for source in profile.ingress_nodes]
    context = _attack_capacity_context(model, profile, routes)
    assert context["peak_target_processing_load_ratio"] == pytest.approx(ratio)
    assert context["request_cpu_budget_ms_per_second"] == budget
    assert context["peak_bandwidth_load_ratio"] < 0.01
    assert context["target_resource_pressure"] == 1.0
    # Нулевая стоимость обработки не создаёт прикладного ущерба даже при
    # ненулевом потоке. Четверть запросов даёт четверть процессорной работы.
    idle = replace(profile, technical=replace(profile.technical, request_cpu_cost_ms=0.0))
    assert _attack_capacity_context(model, idle, routes)["target_resource_pressure"] == 0.0
    small = replace(profile, technical=replace(profile.technical,
                    request_rate_per_second=profile.technical.request_rate_per_second / 4))
    assert _attack_capacity_context(model, small, routes)["target_resource_pressure"] == pytest.approx(ratio / 4)


@pytest.mark.parametrize("kind,step,app,port,size", [
    ("dns_flood", 12, "ATTACK_DNS", 53, 194),
    ("application_flood", 15, "ATTACK_APPLICATION", 443, 398),
])
def test_service_attack_packets_and_scope_preserve_reference(model, kind, step, app, port, size):
    reference = deepcopy(model.to_serializable())
    events = [event for event in active_attack_events(model, "mitre-demo", step_index=step,
              time_seconds=step * 5, step_seconds=5, step_count=16) if event["kind"] == kind]
    assert len(events) == 1
    packet = simulate_packet_snapshot(model, step_index=step, time_seconds=step * 5,
             step_seconds=5, detail="sample", packet_sample_limit=8, attack_events=events)
    control = simulate_packet_snapshot(model, step_index=step, time_seconds=step * 5,
              step_seconds=5, detail="flows")
    attacks = [flow for flow in packet["flows"] if flow.get("is_attack_traffic")]
    assert {flow["application"] for flow in attacks} == {app}
    assert app in FLOW_COLORS
    assert FLOW_SHORT_LABELS[app].startswith("АТАКА")
    for flow in attacks:
        assert flow["wire_bytes"] == flow["packet_count"] * size
        assert flow["server_port"] == port
        assert path_uses_only_data_plane_transit(model, flow["route"])
        assert 0 <= flow["observed_dropped_packets"] <= flow["packet_count"]
        assert flow["blocked_packet_count"] == 0
    control_by_client = {flow["client_node"]: flow for flow in control["flows"]}
    affected = [flow for flow in packet["flows"] if flow.get("attack_impacted")]
    assert affected
    assert any(flow["observed_dropped_packets"] > 0 for flow in affected)
    for flow in packet["flows"]:
        if flow.get("is_attack_traffic"):
            continue
        original = control_by_client[flow["client_node"]]
        assert flow["route"] == original["route"]
        assert flow["server_node"] == original["server_node"]
        assert flow["packet_count"] == original["packet_count"]
        assert flow["wire_bytes"] == original["wire_bytes"]
        assert 0 <= flow["observed_dropped_packets"] <= flow["packet_count"]
        if kind == "dns_flood" and flow["traffic_kind"] != "dns":
            assert not flow.get("attack_impacted", False)
            assert flow["observed_dropped_packets"] == original["observed_dropped_packets"]
        if kind == "application_flood" and flow["server_node"] != "SRV_MEDIA":
            assert not flow.get("attack_impacted", False)
    sample = next(item for item in packet["packet_sample"] if item["application"] == app)
    if kind == "application_flood":
        assert sample["tcp"]["flags"] == "PA"
        assert sample["attack"]["application_payload_bytes_per_packet"] == 320
        assert sample["attack"]["ipv4_packet_bytes"] == 360
    else:
        assert sample["udp"]["dst_port"] == 53
    target = packet["attack_state"]["target_observations"][0]
    assert target["syn_backlog_exhaustion_ratio"] == 0
    power = model.graph.nodes[events[0]["target_id"]]["l6_tensor"]
    assert target["estimated_energy_reserve_ratio"] == float(power.data[power.metric_index["energy_reserve_ratio"][0]])
    assert model.to_serializable() == reference


@pytest.mark.parametrize("kind", ["dns_flood", "application_flood"])
def test_service_precursor_classification_has_no_schedule_dependency(model, kind):
    catalog = attack_catalog(model, "predictive-demo", seed=42, step_seconds=2, step_count=60)
    profile = next(item for item in catalog if item["kind"] == kind)
    step = profile["temporal"]["start_step"] - 5
    records = observe_attack_precursors(model, "predictive-demo", step_index=step,
              time_seconds=step * 2, step_seconds=2, step_count=60, seed=42)
    precursor = next(item for item in records if item["sensor_node_id"] == profile["target_id"]
                     and _classify_precursor_signals(item["signals"]) == kind)
    assert not {"kind", "attack_id", "start_step", "arrival_schedule"} & precursor.keys()
    signals = precursor["signals"]
    assert _classify_precursor_signals(deepcopy(signals)) == kind
    assert _classify_precursor_signals({}) == "unknown_degradation_or_attack"
    # Подмена скрытых метаданных не меняет вывод из тех же измерений.
    poisoned = dict(precursor, kind="power_attack", attack_id="FUTURE", start_step=9999)
    assert _predicted_precursor_kind({"precursors": [poisoned]}) == kind
    assert {"OBS.dns_query_pressure_ratio", "OBS.application_work_pressure_ratio"} <= THREAT_METRICS


def test_default_mitre_demo_includes_both_new_events():
    assert _simulation_timing(parse_args(["--attack-scenario", "mitre-demo"])) == (16, 5)
    assert _simulation_timing(parse_args([])) == (10, 5)
    assert _simulation_timing(parse_args(["--attack-scenario", "mitre-demo", "--dynamics-steps", "0"])) == (0, 5)
