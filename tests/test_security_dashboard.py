"""Панель связывает наблюдения, не создавая решений или знания будущего."""

from copy import deepcopy

import pytest

from src.gnet9.security_dashboard import build_security_dashboard


def snapshot(time, *, events=(), precursors=(), records=(), stage="observe", observations=()):
    return {
        "time_seconds": time, "step_index": time // 2,
        "attacks": {"events": list(events), "precursors": list(precursors),
                    "target_observations": list(observations)},
        "koopman": {"forecast_entity_records": list(records)},
        "arbitrator": {"recommendation": {
            "response_stage": stage, "execution_mode": "human_only",
            "requires_human_action": stage in {"prepare", "protect", "emergency"},
            "steps_ru": ["Проверьте журнал до решения об ограничении доступа."]}},
    }


def event(kind="dos", target="C1", event_id="A1", **extra):
    return {"attack_id": event_id, "kind": kind, "target_id": target,
            "ingress_nodes": ["RAN_A1"], "routes": [["RAN_A1", "A1", target]],
            "effective_offered_rate_mbps": 100, "effective_packet_rate_pps": 1000, **extra}


def precursor(target="C1", **extra):
    return {"sensor_node_id": target, "observed_source_node_ids": ["RAN_A1"],
            "signals": {"scan_rate_pps": 12}, **extra}


def record(kind="dos", target="C1"):
    return {"observation_type": "precursor", "kind": kind, "target_id": target,
            "source_ids": ["RAN_A1"]}


def test_nominal_has_no_alarms_and_preserves_saved_recommendation():
    snapshots = [snapshot(0)]
    before = deepcopy(snapshots)
    panel = build_security_dashboard(snapshots, 0)
    assert panel["alarms"] == []
    assert sum(panel["severity_counts"].values()) == 0
    assert panel["severity"] == "info"
    assert not panel["recommendation"]["requires_human_action"]
    panel["recommendation"]["steps_ru"].append("Изменение текста в панели")
    assert snapshots == before


@pytest.mark.parametrize(("stage", "level"), [
    ("observe", "info"), ("confirm_observation", "watch"),
    ("prepare", "warning"), ("protect", "high"), ("emergency", "critical"),
])
def test_severity_reads_saved_stage_not_risk_threshold(stage, level):
    current = snapshot(4, events=[event()], stage=stage)
    current["koopman"]["forecast_risk_score"] = 0.99 if stage == "observe" else 0.0
    panel = build_security_dashboard([current], 0)
    assert panel["severity"] == level
    assert panel["severity_counts"][level] == 1
    assert panel["alarms"][0]["severity"] == level
    assert panel["alarms"][0]["severity_scope"] == "network"
    current["arbitrator"]["trigger_card"] = {"response_stage": stage}
    del current["arbitrator"]["recommendation"]["response_stage"]
    assert build_security_dashboard([current], 0)["severity"] == level


def test_precursor_merges_with_observed_event_without_future_knowledge():
    snapshots = [snapshot(0), snapshot(2, precursors=[precursor()], records=[record()], stage="prepare"),
                 snapshot(4, events=[event()], stage="protect",
                          observations=[{"target_id": "C1", "service_availability_ratio": 0.75}])]
    before = deepcopy(snapshots)
    warning = build_security_dashboard(snapshots, 1)
    alarm = warning["alarms"][0]
    assert alarm["phase"] == "precursor" and alarm["routes"] == []
    assert alarm["kind"] == "dos" and alarm["source_ids"] == ["RAN_A1"]
    assert alarm["chain"][1]["status"] == "unknown"
    active = build_security_dashboard(snapshots, 2)
    assert len(active["alarms"]) == 1
    alarm = active["alarms"][0]
    assert alarm["phase"] == "active"
    assert alarm["first_seen_seconds"] == 2 and alarm["last_seen_seconds"] == 4
    assert [item["status"] for item in alarm["chain"]] == ["observed", "current", "current", "unknown"]
    assert snapshots == before


def test_overlapping_types_have_distinct_colors_and_only_global_urgency():
    current = snapshot(4, events=[event(), event("dns_flood", "SRV_DATA", "A2"),
                                  event("brute_force", "C2", "A3")], stage="emergency")
    panel = build_security_dashboard([current], 0)
    assert len(panel["alarms"]) == 3
    assert len({alarm["color"] for alarm in panel["alarms"]}) == 3
    assert panel["severity_counts"]["critical"] == 3
    assert {alarm["severity_scope"] for alarm in panel["alarms"]} == {"network"}


def test_observation_gap_does_not_attach_old_precursor_to_new_attack():
    snapshots = [snapshot(2, precursors=[precursor()], records=[record()]),
                 snapshot(4), snapshot(6, events=[event()])]
    active = next(a for a in build_security_dashboard(snapshots, 2)["alarms"] if a["phase"] == "active")
    assert active["first_seen_seconds"] == 6
    assert active["chain"][0]["status"] == "unknown"


def test_reflected_ddos_distinguishes_query_initiators_from_response_ingress():
    current = snapshot(2, events=[event("ddos", ingress_nodes=["M1_01"],
        routes=[["C5", "C6", "C1"]], external_response_ingress_nodes=["C5"],
        external_segment_modeled=False)])
    alarm = build_security_dashboard([current], 0)["alarms"][0]
    assert alarm["source_ids"] == ["C5"]
    assert alarm["routes"][0][0] == "C5"
    assert any("M1_01" in line for line in alarm["evidence_lines"])
    assert any("Внешний участок" in line for line in alarm["evidence_lines"])


def test_past_impact_is_not_presented_as_current_after_availability_returns():
    snapshots = [snapshot(2, events=[event()], observations=[
        {"target_id": "C1", "service_availability_ratio": .5}]),
        snapshot(4, events=[event()], observations=[
        {"target_id": "C1", "service_availability_ratio": 1.0}])]
    assert build_security_dashboard(snapshots, 1)["alarms"][0]["chain"][2]["status"] == "observed"


def test_classic_chain_does_not_invent_exploitation_or_successful_login():
    current = snapshot(2, events=[event("brute_force", effective_failed_authentication_attempt_count=10)],
                       observations=[{"target_id": "C1", "service_availability_ratio": 1.0}])
    alarm = build_security_dashboard([current], 0)["alarms"][0]
    assert all(stage["status"] == "unknown" for stage in alarm["classic_chain"])
    assert any("Успешный вход" in line for line in alarm["evidence_lines"])
    current = snapshot(2, events=[event()], observations=[
        {"target_id": "C1", "service_availability_ratio": 0.8}])
    chain = build_security_dashboard([current], 0)["alarms"][0]["classic_chain"]
    assert {stage["key"] for stage in chain if stage["status"] == "observed"} == {"delivery", "objective"}


def test_exfiltration_retains_observation_time_and_candidate_without_claiming_success():
    current = snapshot(4, events=[event(
        "exfiltration", target="F1_01", mitre_technique_id="T1041",
        sensor_signals={"unusual_egress_bytes_per_second": 50_000},
        routes=[], effective_packet_rate_pps=0, effective_offered_rate_mbps=0,
    )])
    current["koopman"]["signature_entity_records"] = [{
        "target_id": "F1_01", "kind": "exfiltration", "mitre_technique_candidates": ["T1041"],
        "feature_match": {"margin_ratio": 1.1},
    }]
    alarm = build_security_dashboard([snapshot(2), current, snapshot(6)], 2)["alarms"][0]
    assert alarm["mitre_technique_ids"] == alarm["mitre_technique_candidates"] == ["T1041"]
    assert alarm["mitre_tactic_ids"] == ["TA0010"]
    assert alarm["observation_history"] == [{"phase": "active", "time_seconds": 4}]
    assert alarm["classic_chain"][-1]["status"] == "unknown"
    assert any("50000 байт/с" in line for line in alarm["evidence_lines"])


@pytest.mark.parametrize("kind", ["power_attack", "brute_force"])
def test_non_packet_events_do_not_create_flood_paths(kind):
    # Общий конструктор событий сохраняет routes и для управляющих сигналов,
    # но пакетная модель не выпускает по ним атакующие пакеты этих двух типов.
    current = snapshot(2, events=[event(kind)])
    alarm = build_security_dashboard([current], 0)["alarms"][0]
    assert alarm["source_ids"] == ["RAN_A1"]
    assert alarm["routes"] == []
    assert alarm["classic_chain"][2]["status"] == "unknown"
    assert not any("Мбит/с" in line or "пак./с" in line for line in alarm["evidence_lines"])


def test_multiple_classified_precursors_at_one_sensor_stay_separate():
    current = snapshot(2, precursors=[precursor()], records=[record(), record("dns_flood")])
    alarms = build_security_dashboard([current], 0)["alarms"]
    assert {alarm["kind"] for alarm in alarms} == {"dos", "dns_flood"}


def test_power_recovery_and_absent_signal_do_not_mean_resolved():
    attack = event("power_attack", event_id="P1", routes=[], effective_offered_rate_mbps=0,
                   effective_packet_rate_pps=0)
    snapshots = [snapshot(2, events=[attack]),
                 snapshot(4, events=[dict(attack, is_recovery_effect=True, power_runtime={
                     "service_availability_ratio": 0.25, "input_voltage_ratio": 1.0})]),
                 snapshot(6)]
    recovery = build_security_dashboard(snapshots, 1)["alarms"][0]
    assert recovery["phase"] == "recovery" and recovery["routes"] == []
    assert recovery["chain"][1]["status"] == "observed"
    assert recovery["chain"][3]["status"] == "unknown"
    history = build_security_dashboard(snapshots, 2)
    assert history["history_count"] == 1 and sum(history["severity_counts"].values()) == 0
    alarm = history["alarms"][0]
    assert alarm["phase"] == "history" and alarm["last_seen_seconds"] == 4
    assert any("устранение причины не подтверждено" in line for line in alarm["evidence_lines"])


def test_voltage_sag_is_not_outage_and_shared_sla_is_not_individual_damage():
    current = snapshot(2, events=[event("power_attack", routes=[])], observations=[{
        "target_id": "C1", "ups_input_voltage_ratio": 0.5, "service_availability_ratio": 1.0}])
    current["attacks"]["sla_restoration"] = {"tiers": [
        {"sla_grade": "gold", "flow_count": 10, "sla_compliant_flow_count": 0}]}
    alarm = build_security_dashboard([current], 0)["alarms"][0]
    assert alarm["chain"][2]["status"] == "unknown"
    assert alarm["classic_chain"][-1]["status"] == "unknown"
    assert not any("SLA" in line for line in alarm["evidence_lines"])


def test_future_snapshots_and_hidden_precursor_schedule_cannot_change_panel():
    snapshots = [snapshot(0), snapshot(2, precursors=[precursor()], records=[record()]),
                 snapshot(4, events=[event()])]
    expected = build_security_dashboard(snapshots, 1)
    altered = deepcopy(snapshots)
    altered[2] = {"time_seconds": 999, "attacks": {"events": [event("power_attack", "SECRET")]}}
    altered[1]["future_attack_schedule"] = [{"start_step": 1000, "target": "SECRET"}]
    altered[1]["attacks"]["precursors"][0].update(
        attack_id="SECRET", kind="power_attack", target_id="SECRET", temporal={"start_step": 1000})
    assert build_security_dashboard(altered, 1) == expected


def test_unclassified_precursor_is_not_classified_by_schedule():
    current = snapshot(2, precursors=[precursor(kind="power_attack", attack_id="future")])
    alarm = build_security_dashboard([current], 0)["alarms"][0]
    assert alarm["kind"] == "unknown_degradation_or_attack"
    assert alarm["routes"] == []


@pytest.mark.parametrize("index", [-1, 1])
def test_invalid_snapshot_index_is_rejected(index):
    with pytest.raises(ValueError):
        build_security_dashboard([snapshot(0)], index)
