from copy import deepcopy

import pytest

from src.gnet9.attack_signatures import match_attack_signature, signature_catalog
from src.gnet9.attacks import active_attack_events, attack_catalog
from src.gnet9.dynamics import DynamicsConfig, simulate_stationary_dynamics
from src.gnet9.koopman import _forecast_entities
from src.gnet9.topology_builder import GNetBaselineBuilder


def test_weighted_technique_masks_keep_strength_and_observed_lags():
    catalog = signature_catalog()["weighted_masks"]
    assert set(catalog["techniques"]) == {
        "T1498.001", "T1498.002", "T1499.001", "T1499.002", "T1499.003",
        "T1046", "T1204", "T1068", "T1041", "T1464", "T1110.001",
    }
    for weights in catalog["techniques"].values():
        assert len(weights) == 14 and all(len(row) == 8 for row in weights)
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


def test_authentication_mask_uses_attempts_per_account_with_observed_failure_fraction():
    # 120 ошибок/ч при доле ошибок 0,8 означают 150 попыток/ч на запись.
    signals = {"authentication_failure_rate_per_second": 120 / 3600,
               "authentication_failure_ratio": 0.8}
    match = match_attack_signature(signals)
    assert match["weighted_technique_id"] == "T1110.001"
    assert match["weighted_score"] == pytest.approx(0.5)
    assert match_attack_signature({**signals, "authentication_failure_ratio": 0.5}) is None
    assert match_attack_signature({**signals, "authentication_failure_ratio": 1.1}) is None
    full = {**signals, "authentication_failure_rate_per_second": 300 / 3600,
            "authentication_failure_ratio": 1.0}
    assert match_attack_signature(full)["weighted_score"] == 1
    with_history = match_attack_signature(signals, history=[None, full])
    assert with_history["weighted_score"] == pytest.approx((0.5 + 0.70) / 1.70)


def test_discovery_mask_uses_real_elapsed_seconds_and_keeps_window_boundary():
    current = {"distinct_probed_services_count": 6}
    peak = {"distinct_probed_services_count": 12}
    history = [None, peak, peak]
    match = match_attack_signature(current, history=history, step_seconds=30)
    assert match["weighted_score"] == pytest.approx((0.5 + 0.70) / 1.70)
    assert match_attack_signature(current, history=history, step_seconds=31)["weighted_score"] == 0.5
    catalog = signature_catalog(step_seconds=30)["weighted_masks"]
    index = catalog["rows"].index("service_discovery")
    assert catalog["techniques"]["T1046"][index] == [1, 0.85, 0.70, 0, 0, 0, 0, 0]


def test_direct_flood_score_uses_packet_rate_observation_and_an0969_reference():
    # Полная выраженность всплеска, половина ориентира 10 000 пакетов/с.
    current = {"traffic_acceleration_ratio": 0.70, "target_concentration_ratio": 0.90,
               "observed_packet_rate_pps": 5_000}
    assert match_attack_signature(current)["weighted_score"] == pytest.approx(0.75)
    assert match_attack_signature({**current, "observed_packet_rate_pps": 20_000})["weighted_score"] == 1.0
    model = GNetBaselineBuilder().build()
    event, = active_attack_events(model, "mitre-demo", step_index=2, time_seconds=10, step_seconds=5)
    assert event["sensor_signals"]["observed_packet_rate_pps"] == event["effective_packet_rate_pps"]


@pytest.mark.parametrize(("signals", "window"), [
    ({"syn_backlog_growth_ratio": 0.36}, 30),
    ({"authentication_failure_rate_per_second": 150 / 3600, "authentication_failure_ratio": 1.0}, 600),
])
def test_technique_time_window_limits_history_in_seconds(signals, window):
    score = match_attack_signature(signals)["weighted_score"]
    assert match_attack_signature(signals, history=[{}], step_seconds=window)["weighted_score"] < score
    assert match_attack_signature(signals, history=[{}], step_seconds=window + 1)["weighted_score"] == score


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


def test_exfiltration_volume_uses_observed_channel_bytes_and_strict_bounds():
    # 12 500 байт/с × 0,8 × 10 с = 100 000 байт за один интервал.
    sample = {"unusual_egress_bytes_per_second": 12_500, "established_channel_transfer_ratio": 0.8}
    assert match_attack_signature(sample, history=[sample] * 9, step_seconds=10) is None
    match = match_attack_signature(sample, history=[sample] * 10, step_seconds=10)
    assert match["weighted_technique_id"] == "T1041"
    assert match["observed_values"]["observed_channel_volume_bytes"] == pytest.approx(1_100_000)
    assert match_attack_signature(sample, history=[sample] * 9 + [None], step_seconds=10) is None
    assert match_attack_signature({}, history=[sample] * 11, step_seconds=10) is None
    assert match_attack_signature({"observed_channel_volume_bytes": 2_000_000,
                                   "established_channel_transfer_ratio": 1}) is None
    slower = {**sample, "unusual_egress_bytes_per_second": 5_000}
    assert match_attack_signature(slower, history=[slower, slower], step_seconds=99)["kind"] == "exfiltration"
    assert match_attack_signature(slower, history=[slower, slower], step_seconds=100) is None
    invalid = {"unusual_egress_bytes_per_second": 25_000, "established_channel_transfer_ratio": 1.01}
    assert match_attack_signature(invalid) is None


def test_exfiltration_mask_scores_accumulated_volume_only_once():
    sample = {"unusual_egress_bytes_per_second": 50_000, "established_channel_transfer_ratio": 1}
    before = match_attack_signature(sample, history=[sample] * 3)
    after = match_attack_signature(sample, history=[sample] * 4)
    assert after["weighted_score"] > before["weighted_score"]
    masks = signature_catalog()["weighted_masks"]
    row = masks["rows"].index("egress_volume")
    assert masks["techniques"]["T1041"][row] == [1, 0, 0, 0, 0, 0, 0, 0]
    assert masks["numeric_references"]["T1041"]["analytic_id"] == "AN0988"
    coarse = match_attack_signature(sample, step_seconds=300)
    assert coarse["observed_values"]["observed_channel_volume_bytes"] is None
    assert coarse["weighted_score"] == pytest.approx((50_000 / 75_000) / 1.2)
    assert signature_catalog(step_seconds=300)["weighted_masks"]["techniques"]["T1041"][row] == [0] * 8


def test_volume_history_is_scoped_to_sensor_and_observation_channel():
    history = {}
    sample = {"unusual_egress_bytes_per_second": 10_000, "established_channel_transfer_ratio": 1}
    def reading(node):
        return {"sensor_node_id": node, "signals": sample}
    for step in range(1, 12):
        observations = [reading("a")]
        if step == 11:
            observations.append(reading("b"))
        state = {"signature_observations": observations, "precursors": [reading("a")] if step == 11 else []}
        result = _forecast_entities(state, history, step, 10)
        expected_targets = ["a"] if step == 11 else []
        assert [r["target_id"] for r in result["signature_entity_records"]] == expected_targets
        assert result["next_entity_records"] == []


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
