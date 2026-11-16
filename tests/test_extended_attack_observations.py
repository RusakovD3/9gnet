from copy import deepcopy

import pytest

from src.gnet9.attack_signatures import match_attack_signature, signature_catalog
from src.gnet9.attacks import active_attack_events, attack_catalog
from src.gnet9.dynamics import DynamicsConfig, simulate_stationary_dynamics
from src.gnet9.topology_builder import GNetBaselineBuilder


def test_ten_weighted_technique_masks_keep_strength_and_observed_lags():
    catalog = signature_catalog()["weighted_masks"]
    assert len(catalog["techniques"]) == 10
    for weights in catalog["techniques"].values():
        assert len(weights) == 11 and all(len(row) == 8 for row in weights)
        assert all(0 <= weight <= 1 for row in weights for weight in row)
    current = {"syn_backlog_growth_ratio": 0.36, "observed_ingress_count": 1}
    weaker = match_attack_signature(current)
    stronger = match_attack_signature({**current, "syn_backlog_growth_ratio": 0.72})
    sustained = match_attack_signature(current, history=[{"syn_backlog_growth_ratio": 0.72}])
    assert weaker["kind"] == stronger["kind"] == sustained["kind"] == "syn_flood"
    assert weaker["weighted_score"] == pytest.approx(0.4 / 1.25)
    assert weaker["weighted_score"] < stronger["weighted_score"]
    assert weaker["weighted_score"] < sustained["weighted_score"] < stronger["weighted_score"]
    assert match_attack_signature({}, history=[current]) is None
    ambiguous = {"unexpected_privilege_transition_count": 1, "distinct_probed_services_count": 6}
    assert match_attack_signature(ambiguous)["kind"] == "network_discovery"
    assert match_attack_signature(ambiguous, history=[{"unexpected_privilege_transition_count": 2}])["kind"] == "privilege_escalation"


def test_flood_family_depends_on_observed_ingress_and_keeps_the_mechanism():
    for mechanism, signal in (("syn_flood", "syn_backlog_growth_ratio"),
                              ("dhcp_flood", "dhcp_service_pressure_ratio")):
        single = match_attack_signature({signal: 0.8, "observed_ingress_count": 1, "suspicious_source_count": 100})
        multiple = match_attack_signature({signal: 0.8, "observed_ingress_count": 3})
        unknown = match_attack_signature({signal: 0.8, "suspicious_source_count": 100})
        assert single["attack_family"] == "dos"
        assert multiple["attack_family"] == "ddos"
        assert unknown["attack_family"] is None
        assert single["attack_mechanism"] == multiple["attack_mechanism"] == mechanism


def test_signature_needs_joint_evidence_and_ignores_labels():
    metrics = {"endpoint_cpu_rise_ratio": 0.8, "user_launched_process_count": 1}
    assert match_attack_signature({"endpoint_cpu_rise_ratio": 0.8}) is None
    first = match_attack_signature(metrics)
    assert first["kind"] == "user_execution"
    assert match_attack_signature(metrics | {"kind": "dos", "start_step": 100}) == first
    assert first["margin_ratio"] == pytest.approx(1.0)


def test_extended_catalog_and_active_signatures():
    model = GNetBaselineBuilder().build()
    catalog = {row["kind"]: row for row in attack_catalog(model) if 17 <= row["temporal"]["start_step"] <= 36}
    assert set(catalog) == {
        "dos", "bgp_hijack", "network_discovery", "user_execution",
        "privilege_escalation", "exfiltration", "mobile_network_dos",
    }
    assert catalog["bgp_hijack"]["mitre_technique_id"] is None
    assert catalog["mobile_network_dos"]["mitre_tactic_ids"] == ["TA0034"]
    assert model.graph.nodes[catalog["mobile_network_dos"]["target_id"]]["role"] == "mobile-subscriber"
    for kind, row in catalog.items():
        step = row["temporal"]["start_step"]
        event, = active_attack_events(model, "mitre-demo", step_index=step, time_seconds=5 * step, step_seconds=5)
        assert match_attack_signature(event["sensor_signals"])["kind"] == kind


def test_bgp_observation_and_execution_preserve_baseline_and_routes():
    model = GNetBaselineBuilder().build()
    original = deepcopy(model.graph)
    dynamics = simulate_stationary_dynamics(model, DynamicsConfig(
        step_count=45, step_seconds=5, attack_scenario="mitre-demo",
    ))
    snapshots = dynamics["snapshots"]
    bgp = snapshots[20]
    assert bgp["attacks"]["events"][0]["effective_packet_rate_pps"] == 0
    assert bgp["attacks"]["impacted_legitimate_flow_count"] == 0
    records = snapshots[26]["koopman"]["signature_entity_records"]
    assert any(record["observation_type"] == "observed_signature"
               and record["mitre_technique_candidates"] == ["T1204"] for record in records)
    assert snapshots[26]["attacks"]["signature_observations"][0]["signals"]["endpoint_cpu_rise_ratio"] > 0
    exfiltration = snapshots[32]["attacks"]
    assert exfiltration["attack_packet_count"] == exfiltration["impacted_legitimate_flow_count"] == 0
    mobile = snapshots[35]["attacks"]
    assert mobile["events"][0]["target_id"] == "M1_01" and mobile["impacted_legitimate_flow_count"] > 0
    for step, family, mechanism in ((8, "ddos", "syn_flood"), (38, "dos", "syn_flood"),
                                    (41, "dos", "dhcp_flood"), (44, "ddos", "dhcp_flood")):
        snapshot = snapshots[step]
        event, = snapshot["attacks"]["events"]
        record, = snapshot["koopman"]["signature_entity_records"]
        assert (event["attack_family"], event["attack_mechanism"]) == (family, mechanism)
        assert (record["attack_family"], record["attack_mechanism"]) == (family, mechanism)
        assert record["feature_match"]["weighted_score"] > 0.0
        if mechanism == "dhcp_flood":
            assert snapshot["attacks"]["attack_packet_count"] == 0
            assert snapshot["attacks"]["impacted_legitimate_flow_count"] == 0
    assert dynamics["koopman_evaluation"]["attack_onset_count"] == 18
    assert dynamics["koopman_evaluation"]["false_positive_warning_count"] == 0
    assert not snapshots[0]["attacks"]["active"]
    assert set(model.graph.edges) == set(original.edges)
