from src.gnet9.attacks import MITRE_DEMO_ATTACKS, attack_catalog
from src.gnet9.constants import SUBSCRIBER_COUNT
from src.gnet9.dynamics import DynamicsConfig, simulate_stationary_dynamics
from src.gnet9.routing import path_uses_only_data_plane_transit
from src.gnet9.topology_builder import GNetBaselineBuilder


MODEL = GNetBaselineBuilder().build()


def test_attack_catalog_uses_official_mitre_techniques_and_explicit_assumptions() -> None:
    catalog = attack_catalog()
    assert {item["mitre_technique_id"] for item in catalog} == {
        "T1498.001", "T1498.002", "T1499.001", "T0831", "T1529"
    }
    power = [item for item in catalog if item["kind"] == "power_attack"]
    assert {item["power_failure_mode"] for item in power} == {
        "brownout_feed_loss", "cyber_shutdown"
    }
    assert all({"T1078", "T0826"}.issubset(item["related_technique_ids"]) for item in power)
    assert next(item for item in power if item["power_failure_mode"] == "brownout_feed_loss")[
        "mitre_technique_id"
    ] == "T0831"
    assert next(item for item in power if item["power_failure_mode"] == "cyber_shutdown")[
        "mitre_technique_id"
    ] == "T1529"
    assert {item["kind"] for item in catalog} == {"dos", "ddos", "syn_flood", "power_attack"}
    assert all(item["mitre_url"].startswith("https://attack.mitre.org/techniques/") for item in catalog)
    assert all(item["technical"]["value_origin"] == "scenario_assumption" for item in catalog)
    assert all(item["temporal"]["duration_steps"] >= 1 for item in catalog)
    dos = next(item for item in catalog if item["kind"] == "dos")
    ddos = next(item for item in catalog if item["kind"] == "ddos")
    syn = next(item for item in catalog if item["kind"] == "syn_flood")
    assert dos["technical"]["on_wire_size_bytes"] == 512 + 66
    assert ddos["technical"]["on_wire_size_bytes"] == 1_200 + 66
    assert syn["technical"]["packet_size_bytes"] == 40
    assert syn["technical"]["on_wire_size_bytes"] == 84


def test_gold_routes_mark_critical_nodes_and_safe_remap_limit() -> None:
    for node_id in ("C1", "SRV_MEDIA", "M1_01"):
        protection = MODEL.graph.nodes[node_id]["critical_protection"]
        assert protection["is_critical"] is True
        assert protection["protection_priority"] == 1
        assert protection["maximum_safe_utilization_percent"] == 80.0
    assert MODEL.graph.nodes["M1_01"]["sla_grade"] == "gold"


def test_mitre_demo_schedule_and_network_reaction() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(step_count=10, snapshot_detail="summary", packet_detail="flows", attack_scenario="mitre-demo"),
    )
    snapshots = dynamics["snapshots"]
    active_steps = [snapshot["step_index"] for snapshot in snapshots if snapshot["attacks"]["active"]]
    assert active_steps == [2, 3, 5, 6, 8, 9, 10]
    assert snapshots[0]["attacks"]["attack_rate_mbps"] == 0.0
    assert snapshots[0]["attacks"]["legitimate_loss_ratio"] == 0.0
    assert snapshots[0]["koopman"]["forecast_horizon_seconds"] == 5
    assert snapshots[0]["koopman"]["alert_level"] == "NORMAL"
    assert snapshots[6]["attacks"]["raw_attack_rate_mbps"] == 24_000.0
    assert snapshots[6]["attacks"]["attack_rate_mbps"] < snapshots[6]["attacks"]["raw_attack_rate_mbps"]
    assert snapshots[6]["attacks"]["blocked_attack_rate_mbps"] > 0.0
    routing = snapshots[6]["attacks"]["routing"]
    assert routing["failover_flow_count"] > 0
    assert routing["maximum_projected_utilization_percent"] <= 80.0
    assert routing["by_sla"]["gold"]["availability_ratio"] == 1.0
    assert all(
        "SRV_MEDIA" not in flow.get("route", [])
        for flow in snapshots[6]["traffic"]["flows"]
        if not flow.get("is_attack_traffic") and flow.get("original_server_node") == "SRV_MEDIA"
    )
    assert snapshots[6]["attacks"]["raw_target_resource_pressure"] >= 0.90
    assert snapshots[6]["arbitrator"]["analysis"]["attack_pressure"] < snapshots[6]["attacks"]["raw_target_resource_pressure"]
    assert snapshots[6]["arbitrator"]["analysis"]["koopman_forecast_risk"] > 0.50
    assert snapshots[6]["arbitrator"]["remap"]["reason"] == "mitre_attack_observed_with_koopman_confirmation"
    assert snapshots[7]["attacks"]["active"] is False
    assert snapshots[10]["attacks"]["active_count"] == 2
    assert {event["kind"] for event in snapshots[10]["attacks"]["events"]} == {"power_attack"}
    assert snapshots[10]["attacks"]["attack_rate_mbps"] == 0.0
    assert snapshots[10]["arbitrator"]["analysis"]["lyapunov_value"] > snapshots[0]["arbitrator"]["analysis"]["lyapunov_value"]
    assert snapshots[10]["arbitrator"]["remap"]["koopman_forecast"]["attack_kind"] == "power_attack"


def test_attack_flows_are_visible_and_separate_from_legitimate_flows() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(step_count=3, snapshot_detail="summary", packet_detail="flows", attack_scenario="mitre-demo"),
    )
    flows = dynamics["snapshots"][3]["traffic"]["flows"]
    attack_flows = [flow for flow in flows if flow.get("is_attack_traffic")]
    legitimate_flows = [flow for flow in flows if not flow.get("is_attack_traffic")]
    assert len(legitimate_flows) == SUBSCRIBER_COUNT
    assert {flow["application"] for flow in attack_flows} == {"ATTACK_DOS"}
    assert all(flow["mitre_technique_id"] == "T1498.001" for flow in attack_flows)
    assert len(MITRE_DEMO_ATTACKS) == 5


def test_koopman_predictor_exports_causal_forecast_and_residual() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(step_count=6, snapshot_detail="summary", packet_detail="flows", attack_scenario="mitre-demo"),
    )
    t0 = dynamics["snapshots"][0]["koopman"]
    attack = dynamics["snapshots"][6]["koopman"]
    assert t0["training_reference"]["trained_on"] == "ideal_t0_neighbourhood"
    assert t0["training_reference"]["training_sample_count"] == 64
    assert t0["training_reference"]["training_rank"] == t0["training_reference"]["nominal_metric_count"]
    assert t0["training_reference"]["training_data_origin"] == "deterministic_synthetic_metric_neighbourhood"
    assert t0["training_reference"]["ideal_operator_spectral_radius"] < 1.0
    assert len(t0["t0_signature"]) == 64
    assert t0["forecast_horizon_seconds"] == 5
    assert t0["forecast_attack_probability"] < 0.25
    assert t0["reference_distance"] == 0.0
    assert t0["predicted_reference_distance"] == 0.0
    assert t0["one_step_residual"] == 0.0
    assert t0["lyapunov_value"] == 0.0
    assert attack["forecast_attack_probability"] > t0["forecast_attack_probability"]
    assert attack["predicted_reference_distance"] > t0["predicted_reference_distance"]
    assert attack["predicted_attack_score"] > 0.0
    assert attack["baseline_frozen_due_to_threat"] is True
    # Один переход меняет невязку, но не подменяет полноценную оценку оператора.
    assert attack["operator_drift_norm"] == t0["operator_drift_norm"]
    assert attack["one_step_residual"] > 0.0
    assert attack["threat_proximity"]["affected_node_count"] > 0
    assert attack["threat_proximity"]["is_hausdorff_metric"] is False


def test_koopman_warns_five_seconds_before_every_attack_onset() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(step_count=10, snapshot_detail="summary", packet_detail="flows", attack_scenario="mitre-demo"),
    )
    evaluation = dynamics["koopman_evaluation"]
    assert evaluation["attack_onset_count"] == 5
    assert evaluation["predicted_before_onset_count"] == 5
    assert evaluation["minimum_observed_lead_seconds"] >= 5
    assert evaluation["false_positive_attack_id_count"] == 0
    assert evaluation["all_attacks_predicted_at_least_one_step_ahead"] is True
    assert evaluation["preventive_defense_applied_count"] == 5

    for record in evaluation["records"]:
        warning = dynamics["snapshots"][record["warning_step"]]
        assert warning["koopman"]["forecast_target_step"] == record["onset_step"]
        assert warning["koopman"]["lead_time_seconds"] == 5
        assert record["target_id"] in warning["koopman"]["forecast_target_ids"]
        assert record["kind"] in warning["koopman"]["forecast_next_attack_kind"].split("+")
        assert warning["koopman"]["forecast_correlation_ids"]
        assert warning["koopman"]["forecast_attack_ids"] == []
        assert warning["koopman"]["probability_is_calibrated"] is False
        assert warning["arbitrator"]["remap"]["koopman_forecast"]["probability_is_calibrated"] is False
        assert all(
            "attack_id" not in item
            and "suspected_kind" not in item
            and "forecast_horizon_steps" not in item
            and "forecast_horizon_seconds" not in item
            for item in warning["attacks"]["precursors"]
        )
        assert warning["arbitrator"]["prevention"]["status"] == "ARMED_FOR_NEXT_STEP"


def test_healthy_run_has_no_false_positive_and_keeps_t0_operator() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(step_count=10, snapshot_detail="summary", packet_detail="summary", attack_scenario="none"),
    )
    fingerprints = set()
    for revision, snapshot in enumerate(dynamics["snapshots"], start=1):
        koopman = snapshot["koopman"]
        fingerprints.add(koopman["ideal_operator_fingerprint"])
        assert koopman["analysis_revision"] == revision
        assert koopman["forecast_attack_probability"] < 0.25
        assert koopman["forecast_attack_kind"] == "none"
        assert koopman["alert_level"] == "NORMAL"
        assert koopman["online_baseline_update_applied"] is False
        assert snapshot["arbitrator"]["remap"]["action"] == "NO_REMAP"
        assert snapshot["arbitrator"]["prevention"]["status"] == "STANDBY"
    assert len(fingerprints) == 1


def test_preventive_defense_reduces_damage_with_gold_first_priority() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(step_count=6, snapshot_detail="summary", packet_detail="flows", attack_scenario="mitre-demo"),
    )
    ddos = dynamics["snapshots"][6]
    attacks = ddos["attacks"]
    assert attacks["preventive_defense"]["effective"] is True
    assert attacks["attack_rate_mbps"] < attacks["raw_attack_rate_mbps"]
    routing = attacks["routing"]
    assert routing["policy"] == "gold_then_silver_then_bronze"
    assert routing["failover_flow_count"] > 0
    assert routing["maximum_projected_utilization_percent"] <= 80.0
    assert [tier["priority"] for tier in routing["tiers"]] == [1, 2, 3]
    assert routing["by_sla"]["gold"]["availability_ratio"] >= routing["by_sla"]["silver"]["availability_ratio"]
    reflected = [
        flow for flow in ddos["traffic"]["flows"]
        if flow.get("application") == "ATTACK_DDOS"
    ]
    assert reflected
    assert all(not flow.get("quarantined") for flow in reflected)
    assert all(
        flow.get("attack_traffic_leg") == "amplified_external_reflector_response"
        for flow in reflected
    )
    assert all(flow.get("quarantined_request_initiator_ids") for flow in reflected)
    plan = ddos["arbitrator"]["remap"]["sla_restoration_plan"]
    assert plan["order"] == ["gold", "silver", "bronze"]
    assert [tier["priority"] for tier in plan["tiers"]] == [1, 2, 3]


def test_reflection_ddos_separates_trigger_and_amplified_response_routes() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(
            step_count=6,
            snapshot_detail="summary",
            packet_detail="sample",
            packet_sample_limit=64,
            attack_scenario="mitre-demo",
        ),
    )
    snapshot = dynamics["snapshots"][6]
    event = next(item for item in snapshot["attacks"]["events"] if item["kind"] == "ddos")
    assert event["request_routes"]
    assert event["response_routes"]
    assert all(route[0] in event["ingress_nodes"] for route in event["request_routes"])
    assert all(route[-1] in event["external_response_ingress_nodes"] for route in event["request_routes"])
    assert all(route[0] in event["external_response_ingress_nodes"] for route in event["response_routes"])
    assert all(route[-1] == "SRV_MEDIA" for route in event["response_routes"])
    assert all(
        path_uses_only_data_plane_transit(MODEL, route)
        for route in event["request_routes"] + event["response_routes"]
    )
    assert event["effective_request_rate_mbps"] == event["effective_response_rate_mbps"] / 80.0
    assert event["external_segment_modeled"] is False
    reflected_packet = next(
        packet
        for packet in snapshot["traffic"]["packet_sample"]
        if packet.get("application") == "ATTACK_DDOS"
    )
    assert reflected_packet["ipv4"]["src_ip"].startswith("198.51.100.")
    assert reflected_packet["attack"]["operator_ingress_node"] in event[
        "external_response_ingress_nodes"
    ]


def test_t0_training_is_independent_of_snapshot_export_mode() -> None:
    common = dict(step_count=2, snapshot_detail="summary", packet_detail="summary", attack_scenario="mitre-demo")
    with_t0 = simulate_stationary_dynamics(MODEL, DynamicsConfig(include_t0=True, **common))
    without_t0 = simulate_stationary_dynamics(MODEL, DynamicsConfig(include_t0=False, **common))
    assert with_t0["snapshots"][0]["koopman"]["t0_signature"] == without_t0["snapshots"][0]["koopman"]["t0_signature"]
    assert without_t0["snapshots"][0]["step_index"] == 1
    assert without_t0["snapshots"][0]["koopman"]["forecast_is_early_warning"] is True


def test_subscribers_have_explicit_recovery_priority() -> None:
    expected = {"gold": 1, "silver": 2, "bronze": 3}
    for _, attrs in MODEL.graph.nodes(data=True):
        if attrs.get("level") == "L1":
            assert attrs["recovery_priority"] == expected[attrs["sla_grade"]]


def test_packet_sample_reflects_quarantine_and_post_defense_flow_state() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(
            step_count=3,
            snapshot_detail="summary",
            packet_detail="sample",
            packet_sample_limit=64,
            attack_scenario="mitre-demo",
        ),
    )
    traffic = dynamics["snapshots"][3]["traffic"]
    packets = traffic["packet_sample"]
    attack_flows = [flow for flow in traffic["flows"] if flow.get("is_attack_traffic")]
    assert attack_flows
    assert all(flow.get("quarantined") and not flow.get("route") for flow in attack_flows)
    assert all(packet["application"] != "ATTACK_DOS" for packet in packets)
    assert dynamics["snapshots"][3]["attacks"]["raw_attack_rate_mbps"] > 0.0
    assert dynamics["snapshots"][3]["attacks"]["attack_rate_mbps"] == 0.0
    assert dynamics["snapshots"][3]["attacks"]["routing"]["rerouted_flow_count"] > 0


def test_syn_is_quarantined_and_power_attack_has_no_fake_packet() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(
            step_count=10,
            snapshot_detail="summary",
            packet_detail="sample",
            packet_sample_limit=32,
            attack_scenario="mitre-demo",
        ),
    )
    syn_packets = dynamics["snapshots"][8]["traffic"]["packet_sample"]
    assert all(packet["packet_role"] != "attack_syn" for packet in syn_packets)
    syn_flows = [
        flow for flow in dynamics["snapshots"][8]["traffic"]["flows"]
        if flow.get("application") == "ATTACK_SYN"
    ]
    assert syn_flows and all(flow.get("quarantined") for flow in syn_flows)
    assert dynamics["snapshots"][8]["attacks"]["routing"]["protected_endpoint_flow_count"] > 0

    power_snapshot = dynamics["snapshots"][10]
    assert power_snapshot["attacks"]["active"] is True
    assert all(
        not packet["application"].startswith("ATTACK_")
        for packet in power_snapshot["traffic"]["packet_sample"]
    )


def test_power_runtime_separates_ups_ride_through_from_cyber_shutdown() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(
            step_count=12,
            snapshot_detail="summary",
            packet_detail="flows",
            attack_scenario="mitre-demo",
        ),
    )
    onset = dynamics["snapshots"][10]
    runtime = {
        item["target_id"]: item for item in onset["attacks"]["power_runtime_state"]
    }
    a3 = runtime["A3"]
    rtc = runtime["SRV_RTC"]
    assert a3["failure_mode"] == "brownout_feed_loss"
    assert a3["ups_on_battery"] is True
    assert a3["ride_through_active"] is True
    assert a3["input_voltage_ratio"] < 1.0
    assert a3["power_output_availability_ratio"] == 1.0
    assert a3["service_availability_ratio"] == 1.0
    assert rtc["failure_mode"] == "cyber_shutdown"
    assert rtc["input_voltage_ratio"] == 1.0
    assert rtc["power_output_availability_ratio"] == 1.0
    assert rtc["service_availability_ratio"] == 0.0

    observations = {
        item["target_id"]: item for item in onset["attacks"]["target_observations"]
    }
    assert observations["A3"]["estimated_availability_ratio"] == 1.0
    assert observations["A3"]["ups_on_battery"] is True
    assert observations["SRV_RTC"]["estimated_availability_ratio"] == 0.0
    assert observations["SRV_RTC"]["estimated_power_availability_ratio"] == 1.0
    routing = onset["attacks"]["routing"]
    assert "A3" in routing["service_transparent_power_targets"]
    assert "A3" not in routing["avoid_nodes"]
    assert "SRV_RTC" in routing["avoid_nodes"]

    recovery = dynamics["snapshots"][11]
    recovery_runtime = {
        item["target_id"]: item
        for item in recovery["attacks"]["power_runtime_state"]
    }
    assert recovery["attacks"]["intrusion_active"] is False
    assert recovery_runtime["A3"]["phase"] == "battery_recharge"
    assert recovery_runtime["A3"]["service_availability_ratio"] == 1.0
    assert recovery_runtime["SRV_RTC"]["phase"] == "rebooting"
    assert recovery_runtime["SRV_RTC"]["service_availability_ratio"] == 0.0

    for snapshot in dynamics["snapshots"]:
        attack = snapshot["attacks"]
        assert attack["raw_legitimate_dropped_packets"] == (
            attack["legitimate_dropped_packets"]
            + attack["prevented_legitimate_dropped_packets"]
        )
        assert attack["legitimate_dropped_packets"] == (
            attack["observed_attack_induced_legitimate_dropped_packets"]
            + attack["non_attack_legitimate_dropped_packets"]
        )
