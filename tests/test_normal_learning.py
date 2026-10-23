"""Проверки обычных поступлений, фильтра, уничтожения и связей уровней."""

import copy
import math

import numpy as np
import pytest

from src.gnet9.attacks import predictive_demo_mixed_schedule, attack_catalog, active_attack_events
from src.gnet9.dynamics import DynamicsConfig, simulate_stationary_dynamics
from src.gnet9.gelenbe import gelenbe_queue
from src.gnet9.hypernetwork import hypernetwork_embedding
from src.gnet9.kalman import DiagonalKalmanFilter
from src.gnet9.koopman import KoopmanOnlineAnalyzer
from src.gnet9.normal_traffic import poisson_arrivals
from src.gnet9.packet_simulator import simulate_packet_snapshot
from src.gnet9.runtime_metrics import VERTEX_METRICS, EDGE_METRICS
from src.gnet9.topology_builder import GNetBaselineBuilder


def test_poisson_moments_and_independent_reproducible_streams():
    values = np.array([poisson_arrivals(20, node_id="test", step=step) for step in range(1, 6001)])
    assert values.mean() == pytest.approx(20, abs=0.3)
    assert values.var() == pytest.approx(20, abs=1.2)
    assert poisson_arrivals(0, node_id="test", step=1) == 0
    assert poisson_arrivals(0, node_id="test", step=0) == 1
    assert poisson_arrivals(20, node_id="test", step=2) == poisson_arrivals(20, node_id="test", step=2)
    with pytest.raises(ValueError):
        poisson_arrivals(float("inf"), node_id="test", step=1)


def test_kalman_known_gain_preserves_unfiltered_threat():
    filter_ = DiagonalKalmanFilter(2, process_variance=1, measurement_variance=1)
    assert filter_.update(np.array([2., 7.]), np.array([True, False])) == pytest.approx([1, 7])
    assert filter_.gain == pytest.approx([0.5, 1])
    assert filter_.variance == pytest.approx([0.5, 0])
    assert filter_.update(np.array([2., 9.]), np.array([True, False])) == pytest.approx([1.6, 9])


def test_kalman_reduces_noise_without_future_samples():
    random = np.random.default_rng(1729)
    measurements = random.normal(0, .05, 300)
    filter_ = DiagonalKalmanFilter(1)
    filtered = [filter_.update(np.array([x]), np.array([True]))[0] for x in measurements]
    assert np.mean(np.square(filtered)) < .4 * np.mean(np.square(measurements))
    other = DiagonalKalmanFilter(1)
    prefix = [other.update(np.array([x]), np.array([True]))[0] for x in measurements[:30]]
    assert prefix == filtered[:30]
    other.update(np.array([10000]), np.array([True]))
    assert prefix == filtered[:30]


def test_observed_threat_does_not_pollute_nominal_filter_memory():
    model = GNetBaselineBuilder().build()
    reference = {"metric_names": ["L2.cpu_load_percent", "OBS.normal_arrival_rate_pps"],
                 "vector": [10., 100.]}
    analyzer = KoopmanOnlineAnalyzer.from_t0(model, reference, step_seconds=2)
    clean_filter = DiagonalKalmanFilter(2)
    mask = np.ones(2, dtype=bool)
    for step, vector, observed_threat in [(0, [10., 100.], False), (1, [11., 102.], False),
                                         (2, [95., 800.], True), (3, [10., 100.], False)]:
        state = {**reference, "vector": vector}
        before = analyzer.kalman_filter.state.copy() if analyzer.kalman_filter else None
        result = analyzer.analyze_step(model, state, attack_state={"events": [{"kind": "dos"}]} if observed_threat else {},
                                      step_index=step, time_seconds=2 * step)["kalman_filter"]
        if observed_threat:
            assert not result["applied_to_nominal_observation"]
            assert result["raw_normalized_vector"] == result["filtered_normalized_vector"]
            assert analyzer.kalman_filter.state == pytest.approx(before)
        else:
            expected = clean_filter.update(np.array(result["raw_normalized_vector"]), mask)
            assert result["filtered_normalized_vector"] == pytest.approx(expected)


def test_gelenbe_known_solution_and_balance():
    q = gelenbe_queue(2, 1, 1, 1)
    assert q["blocking_probability"] == .5
    assert q["served_rate_pps"] == q["destroyed_rate_pps"] == .5
    assert q["overflow_rate_pps"] == 1
    assert q["mean_system_time_ms"] == 500
    assert q["delay_variance_ms2"] == 250000
    for arrival, service, negative in [(0, 1, 2), (2, 1, 0), (500, 200, 50), (1e20, 1, 1)]:
        q = gelenbe_queue(arrival, service, negative, 128)
        assert all(math.isfinite(v) for v in q.values())
        assert q["served_rate_pps"] + q["destroyed_rate_pps"] + q["overflow_rate_pps"] == pytest.approx(arrival)
        assert 0 <= q["mean_system_packets"] <= 128
        assert q["jitter_ms"] ** 2 == pytest.approx(q["delay_variance_ms2"])
    with pytest.raises(ValueError):
        gelenbe_queue(1, 0, 0, 1)


def test_type_specific_weibull_matches_hazard_and_slow_authentication():
    model = GNetBaselineBuilder().build()
    catalog = attack_catalog(model, "predictive-demo", seed=42, step_seconds=2, step_count=60)
    schedule = predictive_demo_mixed_schedule(seed=42, step_seconds=2, step_count=60)
    assert schedule.raw_intervals_are_iid_weibull is False
    assert {item["kind"] for item in catalog} == {"dos", "ddos", "syn_flood", "power_attack", "brute_force"}
    for item, duration, k, scale, hazard in zip(catalog, schedule.raw_interarrival_seconds,
            schedule.event_shapes, schedule.event_scales_seconds, schedule.realized_hazard_per_second):
        assert 0 < k < 1 if item["kind"] == "brute_force" else k > 1
        assert hazard == pytest.approx(k / scale * (duration / scale) ** (k - 1), rel=2e-5)
    brute = next(item for item in catalog if item["kind"] == "brute_force")
    assert brute["technical"]["offered_rate_mbps"] == 0
    assert brute["technical"]["authentication_attempt_rate_per_second"] < 1


def test_runtime_metrics_and_hypernetwork_do_not_mutate_reference():
    model = GNetBaselineBuilder().build()
    reference = copy.deepcopy(model.to_serializable())
    packet = simulate_packet_snapshot(model, step_index=1, time_seconds=2, step_seconds=2, detail="flows")
    state = packet["network_state"]
    assert len(state["vertex_metric_names"]) == 11
    assert len(state["edge_metric_names"]) == 9
    assert len(state["nodes"]) == model.graph.number_of_nodes()
    assert len(state["edges"]) == model.graph.number_of_edges()
    assert all(tuple(n["metrics"]) == VERTEX_METRICS for n in state["nodes"])
    assert all(tuple(e["metrics"]) == EDGE_METRICS for e in state["edges"])
    for flow in packet["flows"]:
        if flow["transport"] == "TCP":
            assert flow["packet_count"] == flow["data_segments"] + flow["ack_segments"]
        elif flow["traffic_kind"] == "dns":
            assert flow["packet_count"] == 2 * flow["dns_queries"]
        assert flow["arrival_process"] == "poisson"
    hypernetwork = hypernetwork_embedding(model)
    assert any(len(r["members"]) > 2 for r in hypernetwork["relations"])
    assert sum(r["kind"] == "service_dependency" for r in hypernetwork["relations"]) == 480
    assert model.to_serializable() == reference


def test_normal_learning_uses_filtered_observations_and_keeps_sla():
    model = GNetBaselineBuilder().build()
    result = simulate_stationary_dynamics(model, DynamicsConfig(step_count=4, step_seconds=2))
    assert result["ideal_t0"]["ok"]
    assert result["sla_resilience"]["time_mean"] == 1
    assert result["training_summary"]["nominal_transitions"] == 4
    for step in result["snapshots"][1:]:
        kalman = step["koopman"]["kalman_filter"]
        names = step["state_vector"]["metric_names"]
        assert "OBS.normal_arrival_rate_pps" in kalman["filtered_metric_names"]
        raw, filtered = kalman["raw_normalized_vector"], kalman["filtered_normalized_vector"]
        for name in ("OBS.legitimate_loss_ratio", "OBS.gold_sla_compliance_ratio", "OBS.attack_rate_mbps"):
            index = names.index(name)
            assert raw[index] == filtered[index]


def test_confirmed_direct_sources_stop_packet_damage():
    model = GNetBaselineBuilder().build()
    profile = next(item for item in attack_catalog(model) if item["kind"] == "dos")
    step = profile["temporal"]["start_step"]
    event = next(item for item in active_attack_events(model, "mitre-demo", step_index=step,
        time_seconds=step * 2, step_seconds=2) if item["attack_id"] == profile["attack_id"])
    plan = {"valid_from_step": step, "valid_until_step": step, "attack_ids": [event["attack_id"]],
            "mitigation_strength": 1.0, "source_ids": event["ingress_nodes"],
            "routing_control": {"quarantine_sources": event["ingress_nodes"]}}
    result = simulate_packet_snapshot(model, step_index=step, time_seconds=step * 2,
        step_seconds=2, attack_events=[event], defense_plan=plan, detail="flows")
    legitimate = [f for f in result["flows"] if not f.get("is_attack_traffic") and not f.get("isolated")]
    assert sum(f["observed_dropped_packets"] for f in legitimate) == 0
    assert all(f["attack_latency_penalty_ms"] == 0 for f in legitimate)
    assert sum(f["packet_count"] for f in result["flows"] if f.get("is_attack_traffic")) == 0
    assert result["attack_state"]["raw_attack_rate_mbps"] > 0


def test_low_demand_does_not_hide_loss_and_empty_flow_is_available():
    from src.gnet9.attacks import _evaluate_flow_slo
    model = GNetBaselineBuilder().build()
    result = simulate_packet_snapshot(model, step_index=4, time_seconds=8, step_seconds=2, detail="flows")
    voice = next(f for f in result["flows"] if f["flow_id"] == "M2_23-RTP_OPUS-4")
    evaluation = voice["slo_evaluation"]
    bitrate = evaluation["metrics"]["media_bitrate_kbps"]
    assert bitrate["observed_value"] < bitrate["budget_value"]
    assert bitrate["effective_budget_value"] == bitrate["observed_value"]
    assert evaluation["compliant"]
    impaired = dict(voice, observed_dropped_packets=voice["packet_count"] // 2)
    assert not _evaluate_flow_slo(model, impaired)["compliant"]
    idle = dict(voice, packet_count=0, payload_bytes=0, observed_dropped_packets=0)
    assert _evaluate_flow_slo(model, idle)["compliant"]
    idle["route_available"] = False
    assert not _evaluate_flow_slo(model, idle)["compliant"]
