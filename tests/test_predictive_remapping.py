from collections import Counter
from copy import deepcopy
from functools import lru_cache
import math

import networkx as nx
import numpy as np
import pytest

from src.gnet9.attacks import (
    active_attack_events,
    attack_catalog,
    observe_attack_precursors,
    predictive_demo_minimum_steps,
    predictive_demo_weibull_schedule,
)
from src.gnet9.constants import SUBSCRIBER_COUNT
from src.gnet9.dynamics import DynamicsConfig, simulate_stationary_dynamics
from src.gnet9.koopman import (
    KoopmanOnlineAnalyzer,
    _discrete_lyapunov_matrix,
    _operator_from_covariances,
    forecast_horizon_steps_for_sample_period,
)
from src.gnet9.metrics import (
    directed_hausdorff_distance,
    hausdorff_distance,
    prepare_state_hausdorff_reference,
    state_tensor_hausdorff_view,
)
from src.gnet9.packet_simulator import simulate_packet_snapshot
from src.gnet9.remapping import apply_gold_first_remap
from src.gnet9.routing import path_uses_only_data_plane_transit
from src.gnet9.topology_builder import GNetBaselineBuilder


MODEL = GNetBaselineBuilder().build()


@lru_cache(maxsize=1)
def _predictive_dynamics():
    return simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(
            step_seconds=2,
            step_count=60,
            snapshot_detail="summary",
            packet_detail="flows",
            attack_scenario="predictive-demo",
            attack_seed=42,
        ),
    )


@lru_cache(maxsize=4)
def _seeded_predictive_dynamics(seed: int):
    return simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(
            step_seconds=2,
            step_count=60,
            snapshot_detail="summary",
            packet_detail="summary",
            attack_scenario="predictive-demo",
            attack_seed=seed,
        ),
    )


def test_predictive_catalog_is_weighted_reproducible_and_uses_l1_sources() -> None:
    kwargs = dict(seed=42, step_seconds=2, step_count=60)
    first = attack_catalog(MODEL, "predictive-demo", **kwargs)
    second = attack_catalog(MODEL, "predictive-demo", **kwargs)
    assert first == second
    # Вейбулловский renewal-процесс имеет переменное число приходов в конечном
    # окне. Для seed=42 первые четыре позиции специально покрывают все виды.
    assert 1 <= len(first) <= 8
    assert Counter(item["kind"] for item in first) == {
        "dos": 1,
        "ddos": 1,
        "syn_flood": 1,
        "power_attack": 1,
    }
    assert {item["scenario_weight_origin"] for item in first} == {"scenario_assumption"}
    assert sum(item["scenario_weight"] for item in first) == 1.0
    assert {item["scenario_weight_semantics"] for item in first} == {"realized_event_mass"}
    assert {item["kind"]: item["kind_weight"] for item in first} == {
        "dos": 3 / 8,
        "ddos": 2 / 8,
        "syn_flood": 2 / 8,
        "power_attack": 1 / 8,
    }
    assert all(
        MODEL.graph.nodes[source]["level"] == "L1"
        and MODEL.graph.nodes[source]["sla_grade"] == "bronze"
        for item in first
        for source in item["ingress_nodes"]
    )
    assert all(len(item["ingress_nodes"]) == 1 for item in first if item["kind"] == "dos")
    assert all(not item["ingress_nodes"] for item in first if item["kind"] == "power_attack")
    assert all(
        MODEL.graph.nodes[item["target_id"]]["role"] in {"core-router", "aggregation-switch", "service-server"}
        for item in first if item["kind"] == "power_attack"
    )
    assert {item["mitre_semantics"] for item in first} == {
        "behavior_taxonomy_not_a_universal_numeric_rate_standard; "
        "a separately-labelled procedure-example value may be used where available"
    }
    assert all(item["mitre_url"].startswith("https://attack.mitre.org/techniques/") for item in first)
    expected_mitre = {
        "dos": "T1498.001",
        "ddos": "T1498.002",
        "syn_flood": "T1499.001",
        "power_attack": "T0831",
    }
    assert all(item["mitre_technique_id"] == expected_mitre[item["kind"]] for item in first)
    power = next(item for item in first if item["kind"] == "power_attack")
    assert "T1078" in power["related_technique_ids"]
    assert "T0826" in power["related_technique_ids"]
    assert power["power_failure_mode"] == "brownout_feed_loss"


def test_attack_effect_is_normalized_by_path_and_target_resource() -> None:
    catalog = attack_catalog(MODEL, "predictive-demo", seed=42, step_seconds=2, step_count=60)
    onset_by_id = {item["attack_id"]: item["temporal"]["start_step"] for item in catalog}
    events = [
        event
        for item in catalog
        for event in active_attack_events(
            MODEL,
            "predictive-demo",
            step_index=onset_by_id[item["attack_id"]],
            time_seconds=onset_by_id[item["attack_id"]] * 2,
            step_seconds=2,
            step_count=60,
            seed=42,
        )
        if event["attack_id"] == item["attack_id"]
    ]
    assert events
    assert all(event["convergence_capacity_mbps"] > 0.0 for event in events)
    assert all(0.0 <= event["target_resource_pressure"] <= 1.0 for event in events)
    assert all(event["target_resource_budget_origin"] == "gnet9_calibrated_scenario_assumption" for event in events)
    assert all(event["technical"]["on_wire_size_bytes"] == 84 for event in events if event["kind"] == "syn_flood")


def test_weibull_renewal_schedule_is_reproducible_bounded_and_exported() -> None:
    first = predictive_demo_weibull_schedule(seed=42, step_seconds=2, step_count=60)
    second = predictive_demo_weibull_schedule(seed=42, step_seconds=2, step_count=60)
    other_seed = predictive_demo_weibull_schedule(seed=43, step_seconds=2, step_count=60)
    assert first == second
    assert first.raw_interarrival_seconds != other_seed.raw_interarrival_seconds
    assert first.distribution == "weibull"
    assert first.value_origin == "seeded_weibull_renewal_scenario_assumption"
    assert first.conditioning == "finite_window_right_censoring_or_safety_cap_without_rescaling"
    assert first.generation_method == "inverse_cdf_iid_renewal_until_window_end_or_safety_cap"
    assert first.raw_intervals_are_iid_weibull is True
    assert first.realized_intervals_are_iid_weibull is False
    assert first.parameter_estimation_status == "not_fitted_to_operator_incident_data"
    assert 1 <= len(first.realized_start_steps) <= first.maximum_event_count
    assert tuple(sorted(first.realized_start_steps)) == first.realized_start_steps
    assert len(set(first.realized_start_steps)) == len(first.realized_start_steps)
    assert first.realized_start_steps[-1] < 60
    assert all(value > 0 and value % 2 == 0 for value in first.realized_interarrival_seconds)
    assert sum(first.realized_interarrival_seconds) <= (
        first.window_end_step - first.window_origin_step
    ) * 2
    assert first.conditioning_scale_factor == 1.0
    assert first.right_censored_interarrival_seconds is not None
    assert first.termination_reason == "right_censored_at_window_end"
    assert len(first.raw_interarrival_seconds) == len(first.realized_interarrival_seconds)
    assert all(value > 0.0 for value in first.realized_hazard_per_second)
    assert all(0.0 < value < 1.0 for value in first.realized_survival_probability)

    catalog = attack_catalog(MODEL, "predictive-demo", seed=42, step_seconds=2, step_count=60)
    assert tuple(item["temporal"]["start_step"] for item in catalog) == first.realized_start_steps
    assert all(item["arrival_schedule"]["shape"] == first.shape for item in catalog)
    assert tuple(item["arrival_interval_seconds"] for item in catalog) == first.realized_interarrival_seconds


def test_weibull_schedule_respects_arbitrary_positive_step_duration() -> None:
    for step_seconds, step_count in ((1, 100), (3, 48), (5, 30)):
        schedule = predictive_demo_weibull_schedule(
            seed=17,
            step_seconds=step_seconds,
            step_count=step_count,
        )
        assert 1 <= len(schedule.realized_start_steps) <= schedule.maximum_event_count
        assert schedule.realized_start_steps[-1] <= step_count - 3
        assert all(
            interval > 0 and interval % step_seconds == 0
            for interval in schedule.realized_interarrival_seconds
        )


def test_predictive_precursor_contains_only_current_sensor_telemetry() -> None:
    catalog = attack_catalog(MODEL, "predictive-demo", seed=42, step_seconds=2, step_count=60)
    first_onset = min(item["temporal"]["start_step"] for item in catalog)
    precursor_steps = max(item["temporal"]["precursor_steps"] for item in catalog)
    observation_step = first_onset - precursor_steps
    observations = observe_attack_precursors(
        MODEL,
        "predictive-demo",
        step_index=observation_step,
        time_seconds=observation_step * 2,
        step_seconds=2,
        step_count=60,
        seed=42,
    )
    assert observations
    forbidden = {
        "attack_id", "kind", "start_step", "forecast_horizon_steps",
        "forecast_horizon_seconds", "suspected_target_id", "target_id",
        "arrival_schedule", "arrival_process", "arrival_realization",
    }
    assert all(not (forbidden & set(item)) for item in observations)
    assert all(item["sensor_node_id"] in MODEL.graph for item in observations)
    assert all(item["signals"] for item in observations)

    first_event = active_attack_events(
        MODEL,
        "predictive-demo",
        step_index=first_onset,
        time_seconds=first_onset * 2,
        step_seconds=2,
        step_count=60,
        seed=42,
    )[0]
    assert "arrival_schedule" not in first_event
    assert first_event["arrival_process"]["distribution"] == "weibull"
    assert first_event["arrival_realization"]["observability"] == "available_only_after_attack_onset"


def test_predictive_demo_warns_before_attack_without_false_warning() -> None:
    dynamics = _predictive_dynamics()
    evaluation = dynamics["koopman_evaluation"]
    expected_attack_count = len(
        attack_catalog(MODEL, "predictive-demo", seed=42, step_seconds=2, step_count=60)
    )
    assert evaluation["attack_onset_count"] == expected_attack_count
    assert evaluation["predicted_before_onset_count"] == expected_attack_count
    assert evaluation["minimum_observed_lead_seconds"] >= 18
    assert evaluation["maximum_observed_lead_seconds"] <= 20
    assert evaluation["false_positive_warning_count"] == 0
    assert evaluation["preventive_defense_applied_count"] == expected_attack_count
    assert evaluation["forecast_horizon_seconds"] == 20
    assert evaluation["prediction_precision"] == 1.0
    assert evaluation["prediction_recall"] == 1.0
    assert evaluation["prediction_f1"] == 1.0
    assert evaluation["prediction_slo_seconds"] == 10
    assert evaluation["prediction_slo_predicted_count"] == expected_attack_count
    assert evaluation["prediction_slo_recall"] == 1.0
    assert evaluation["prediction_slo_misses"] == []
    assert evaluation["prediction_slo_status"] == "PASSED_SYNTHETIC_SCENARIO"
    assert evaluation["independent_holdout_validation_performed"] is False
    assert evaluation["generalization_claimed"] is False
    assert all(item["recall"] == 1.0 for item in evaluation["per_attack_kind"].values())
    assert dynamics["ideal_t0"]["ok"] is True
    assert dynamics["snapshots"][0]["koopman"]["warning_confirmation_samples"] == 2
    assert dynamics["snapshots"][0]["koopman"]["uses_augmented_hankel_operator"] is False
    assert dynamics["snapshots"][0]["koopman"]["prediction_slo_seconds"] == 10
    assert dynamics["snapshots"][0]["koopman"]["prediction_slo_forecast_seconds"] == 10
    resilience = dynamics["resilience_evaluation"]
    assert resilience["status"] == "RESILIENT_GOLD_CONTINUITY_AND_RECOVERY"
    assert resilience["final_state_recovered"] is True
    assert resilience["minimum_sla_compliance_ratio"]["gold"] == 1.0
    assert resilience["episodes"][0]["recovered_step"] is not None
    assert resilience["episodes"][0]["recovery_status"] == (
        "recovered_before_next_attack_or_end"
    )
    assert resilience["episodes"][1]["recovery_status"] == (
        "recovered_before_next_attack_or_end"
    )
    final_koopman = dynamics["snapshots"][-1]["koopman"]
    final_attacks = dynamics["snapshots"][-1]["attacks"]
    assert final_attacks["legitimate_delivery_ratio"] == 1.0
    assert final_attacks["routing"]["plan_active"] is False
    assert final_attacks["routing"]["isolated_flow_count"] == 0
    assert final_koopman["forecast_is_early_warning"] is False
    # При меньшем числе приходов в окне у контура остаётся время завершить
    # cooldown; это более сильная проверка полного восстановления.
    assert final_koopman["post_incident_cooldown_active"] is False
    assert final_koopman["risk_state"] == "nominal"
    assert final_koopman["alert_level"] == "NORMAL"
    assert final_koopman["forecast_attack_kind"] == "none"


def test_predictive_ten_second_slo_is_stable_across_seeded_attack_series() -> None:
    """Проверить целевой запас на независимых воспроизводимых сценариях стенда.

    Это регрессия для модели, не независимая операторская валидация: все
    сценарии используют только причинную телеметрию, а onset сверяется уже
    после прогона в ``_koopman_evaluation``.
    """
    for seed in (17, 41, 42, 43, 48):
        dynamics = _predictive_dynamics() if seed == 42 else _seeded_predictive_dynamics(seed)
        evaluation = dynamics["koopman_evaluation"]
        assert evaluation["prediction_slo_seconds"] == 10
        assert evaluation["attack_onset_count"] > 0
        assert evaluation["prediction_slo_predicted_count"] == evaluation["attack_onset_count"]
        assert evaluation["prediction_slo_recall"] == 1.0
        assert evaluation["false_positive_warning_count"] == 0
        assert evaluation["prediction_slo_status"] == "PASSED_SYNTHETIC_SCENARIO"
        assert evaluation["minimum_observed_lead_seconds"] >= 10


def test_approved_external_threshold_is_explicitly_propagated_to_koopman() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(
            step_count=0,
            snapshot_detail="summary",
            packet_detail="flows",
            warning_risk_threshold=0.77,
            warning_threshold_origin="chronological_external_telemetry_calibration",
        ),
    )

    koopman = dynamics["snapshots"][0]["koopman"]
    assert koopman["warning_risk_threshold"] == 0.77
    assert koopman["warning_threshold_origin"] == "chronological_external_telemetry_calibration"
    assert koopman["probability_is_calibrated"] is False


def test_predictive_simulation_rejects_a_window_shorter_than_its_precursor() -> None:
    required = predictive_demo_minimum_steps(2)
    with pytest.raises(ValueError, match=f"at least {required} steps"):
        simulate_stationary_dynamics(
            MODEL,
            DynamicsConfig(
                step_seconds=2,
                step_count=required - 1,
                snapshot_detail="summary",
                packet_detail="summary",
                attack_scenario="predictive-demo",
            ),
        )


def test_representative_overlapping_failures_preserve_gold_and_release_overlay() -> None:
    # seed=41 stresses overlapping server failures; seed=48 simultaneously
    # affects the access segment of one mobile Gold group.
    for seed in (41, 48):
        dynamics = _seeded_predictive_dynamics(seed)
        resilience = dynamics["resilience_evaluation"]
        assert resilience["minimum_sla_compliance_ratio"]["gold"] >= 0.95
        assert resilience["final_state_recovered"] is True
        final_attacks = dynamics["snapshots"][-1]["attacks"]
        assert final_attacks["legitimate_delivery_ratio"] == 1.0
        assert final_attacks["routing"]["plan_active"] is False


def test_observed_tensor_overlay_reacts_without_mutating_t0_graph() -> None:
    dynamics = _predictive_dynamics()
    snapshots = dynamics["snapshots"]
    t0_overlay = snapshots[0]["tensor_state"]["overlay"]
    assert t0_overlay["mode"] == "non_mutating_observed_telemetry_overlay"
    assert {"L0", "L1", "EDGE"}.issubset(t0_overlay["updated_levels"])
    assert t0_overlay["structurally_invariant_levels"] == ["L3", "L4", "L5", "L8"]

    l2_attack_snapshots = [
        snapshot
        for snapshot in snapshots
        if any(
            event.get("target_id") in MODEL.graph
            and MODEL.graph.nodes[event["target_id"]].get("level") == "L2"
            for event in snapshot["attacks"].get("events", [])
        )
    ]
    assert l2_attack_snapshots
    assert any(
        "L2" in snapshot["tensor_state"]["overlay"]["updated_levels"]
        for snapshot in l2_attack_snapshots
    )
    assert any(
        snapshot["arbitrator"]["level_metric_aggregates"]["L2"]["metrics"]["cpu_load_percent"]["max"]
        > snapshots[0]["arbitrator"]["level_metric_aggregates"]["L2"]["metrics"]["cpu_load_percent"]["max"]
        for snapshot in l2_attack_snapshots
    )

    power_snapshots = [
        snapshot for snapshot in snapshots
        if snapshot["attacks"].get("power_runtime_state")
    ]
    assert power_snapshots
    assert any(
        "L6" in snapshot["tensor_state"]["overlay"]["updated_levels"]
        for snapshot in power_snapshots
    )

    # Overlay хранится в снимке; исходный StateTensor графа остаётся эталонным.
    c1_cpu = MODEL.graph.nodes["C1"]["tensor"].data[
        MODEL.graph.nodes["C1"]["tensor"].metric_index["cpu_load_percent"]
    ]
    assert float(c1_cpu) in {16.0, 16.8}


def test_koopman_exports_component_ablation_and_honest_energy_metric() -> None:
    dynamics = _predictive_dynamics()
    for snapshot in dynamics["snapshots"]:
        koopman = snapshot["koopman"]
        ablation = koopman["forecast_component_ablation"]
        assert set(ablation) >= {
            "koopman_dmd_only_max_risk",
            "koopman_plus_delay_trend_max_risk",
            "hybrid_with_precursor_classifier_max_risk",
        }
        assert koopman["finite_time_metric_semantics"].endswith(
            "not_maximal_nonlinear_ftle"
        )
        assert koopman["finite_time_energy_growth_rate_per_second"] == koopman[
            "finite_time_lyapunov_exponent"
        ]
        assert isinstance(
            koopman["lyapunov_basis_rebased_after_operator_update"],
            bool,
        )
        assert koopman["observed_operator_fingerprint"] == koopman[
            "operator_fingerprint_used"
        ]
        assert koopman["lyapunov_matrix_fingerprint"] == koopman[
            "lyapunov_matrix_fingerprint_used"
        ]
        assert koopman["lyapunov_solver"]["operator_revision"] == koopman[
            "operator_revision_used"
        ]
        assert koopman["lyapunov_solver"]["method"] == "smith_doubling_iteration"
        assert koopman["operator_revision_after_update"] == (
            koopman["operator_revision_used"]
            + int(koopman["update_applied_for_next_step"])
        )
        analysis = snapshot["arbitrator"]["analysis"]
        assert analysis["koopman_operator_revision_used"] == koopman[
            "operator_revision_used"
        ]
        assert analysis["koopman_operator_revision_after_update"] == koopman[
            "operator_revision_after_update"
        ]
        assert analysis["koopman_update_applied_for_next_step"] == koopman[
            "update_applied_for_next_step"
        ]


def test_koopman_horizon_discretization_never_rounds_above_twenty_seconds() -> None:
    expected = {
        1: (20, 20),
        2: (10, 20),
        3: (6, 18),
        6: (3, 18),
        7: (2, 14),
        20: (1, 20),
    }
    for step_seconds, (step_count, horizon_seconds) in expected.items():
        actual_steps = forecast_horizon_steps_for_sample_period(step_seconds)
        assert actual_steps == step_count
        assert actual_steps * step_seconds == horizon_seconds <= 20

    # При более редкой телеметрии подшага <=20 с не существует: явно остаётся
    # один coarse-step, а экспорт помечает превышение конфигурационного лимита.
    assert forecast_horizon_steps_for_sample_period(21) == 1
    assert forecast_horizon_steps_for_sample_period(24) == 1


def test_koopman_snapshot_uses_one_frozen_operator_revision() -> None:
    """Online update текущего перехода должен влиять лишь на следующий шаг."""
    baseline = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(
            step_seconds=2,
            step_count=0,
            snapshot_detail="summary",
            packet_detail="summary",
            attack_scenario="none",
        ),
    )
    state = baseline["snapshots"][0]["state_vector"]
    analyzer = KoopmanOnlineAnalyzer.from_t0(MODEL, state, step_seconds=2)
    no_attack = {"scenario": "none", "events": [], "precursors": [], "routing": {}}

    previous_state = {
        "metric_names": list(state["metric_names"]),
        "vector": list(state["vector"]),
    }
    # Однокоординатная вариация с RMS 0,007 остаётся здоровой, но является
    # информативным ненулевым входом для ковариаций online DMD.
    normalized_offset = 0.007 * len(state["metric_names"]) ** 0.5
    previous_state["vector"][0] += float(analyzer.scale_vector[0]) * normalized_offset
    first = analyzer.analyze_step(
        MODEL,
        previous_state,
        attack_state=no_attack,
        step_index=0,
        time_seconds=0,
    )
    perturbed = {
        "metric_names": list(state["metric_names"]),
        "vector": list(state["vector"]),
    }
    # Следующее малое номинальное отклонение обновляет DMD, но не затрагивает
    # OBS.* признаки угрозы и не должно создавать предупреждение.
    perturbed["vector"][0] += (
        float(analyzer.scale_vector[0]) * normalized_offset * 1.15
    )
    updated = analyzer.analyze_step(
        MODEL,
        perturbed,
        attack_state=no_attack,
        step_index=1,
        time_seconds=2,
    )

    assert updated["update_applied_for_next_step"] is True
    assert updated["operator_revision_used"] == first["operator_revision_after_update"] == 0
    assert updated["operator_revision_after_update"] == 1
    assert updated["operator_update_effective_from_step"] == 2
    assert updated["online_baseline_update_semantics"] == "committed_for_next_step_only"
    assert updated["observed_operator_fingerprint"] == updated["operator_fingerprint_used"]
    assert updated["lyapunov_matrix_fingerprint"] == updated["lyapunov_matrix_fingerprint_used"]
    assert updated["lyapunov_solver"]["operator_revision"] == updated["operator_revision_used"]
    assert updated["lyapunov_solver_after_update"]["operator_revision"] == updated[
        "operator_revision_after_update"
    ]
    assert updated["lyapunov_solver_after_update"]["effective_from_step"] == 2

    next_step = analyzer.analyze_step(
        MODEL,
        perturbed,
        attack_state=no_attack,
        step_index=2,
        time_seconds=4,
    )
    assert next_step["operator_revision_used"] == updated["operator_revision_after_update"]
    assert next_step["operator_fingerprint_used"] == updated["operator_fingerprint_after_update"]
    assert next_step["lyapunov_matrix_fingerprint_used"] == updated[
        "lyapunov_matrix_fingerprint_after_update"
    ]
    assert next_step["lyapunov_basis_rebased_after_operator_update"] is True


def test_gold_first_overlay_reroutes_fails_over_and_respects_capacity() -> None:
    dynamics = _predictive_dynamics()
    active = [snapshot for snapshot in dynamics["snapshots"] if snapshot["attacks"]["active"]]
    assert active
    assert all(
        snapshot["attacks"]["routing"]["maximum_projected_utilization_percent"] <= 80.0
        for snapshot in active
    )
    assert any(snapshot["attacks"]["routing"]["rerouted_flow_count"] > 0 for snapshot in active)
    assert any(snapshot["attacks"]["routing"]["failover_flow_count"] > 0 for snapshot in active)
    assert any(
        snapshot["attacks"]["routing"]["gold_route_hausdorff"]["normalized_distance"] > 0.0
        for snapshot in active
    )

    for snapshot in active:
        routing = snapshot["attacks"]["routing"]
        for flow in snapshot["traffic"]["flows"]:
            if flow.get("is_attack_traffic") or flow.get("isolated"):
                continue
            assert path_uses_only_data_plane_transit(MODEL, flow.get("route", []))
            assert path_uses_only_data_plane_transit(MODEL, flow.get("reverse_route", []))
        for event in snapshot["attacks"]["events"]:
            assert all(
                path_uses_only_data_plane_transit(MODEL, route)
                for route in event.get("request_routes", [])
                + event.get("response_routes", [])
            )
            target = event["target_id"]
            role = MODEL.graph.nodes[target].get("role")
            if role not in {"core-router", "aggregation-switch"}:
                continue
            if event.get("power_runtime", {}).get("ride_through_active"):
                # Просадка, удержанная ИБП, наблюдается, но узел не считается
                # отказавшим и не обязан исчезать из активного маршрута.
                continue
            rerouted = [
                flow for flow in snapshot["traffic"]["flows"]
                if not flow.get("is_attack_traffic")
                and target in flow.get("original_route", [])
                and flow.get("rerouted")
            ]
            assert all(target not in flow["route"] for flow in rerouted)


def test_predictive_prestage_never_quarantines_unconfirmed_subscriber_source() -> None:
    dynamics = _predictive_dynamics()
    prestaged = [
        snapshot
        for snapshot in dynamics["snapshots"]
        if snapshot["attacks"]["routing"].get("plan_active")
        and snapshot["attacks"]["routing"].get("protection_stage") == "predictive_prestage"
        and snapshot["koopman"].get("forecast_is_early_warning")
        and not snapshot["attacks"].get("active")
    ]
    assert prestaged
    assert all(
        not snapshot["attacks"]["routing"]["quarantine_sources"]
        for snapshot in prestaged
    )
    assert all(
        snapshot["attacks"]["routing"]["protection_stage"] == "predictive_prestage"
        for snapshot in prestaged
    )
    assert all(
        not any(
            flow.get("routing_action") == "quarantine_attack_source"
            for flow in snapshot["traffic"]["flows"]
        )
        for snapshot in prestaged
    )


def test_observed_attack_marks_active_mitigation_even_for_prepared_plan() -> None:
    dynamics = _predictive_dynamics()
    active = [snapshot for snapshot in dynamics["snapshots"] if snapshot["attacks"]["active"]]
    assert active
    assert all(
        snapshot["attacks"]["routing"]["protection_stage"]
        == "confirmed_active_mitigation"
        for snapshot in active
    )


def test_power_brownout_ride_through_does_not_manufacture_an_outage() -> None:
    dynamics = _predictive_dynamics()
    snapshot = next(
        item for item in dynamics["snapshots"]
        if any(
            event["kind"] == "power_attack"
            and MODEL.graph.nodes[event["target_id"]].get("role") == "aggregation-switch"
            for event in item["attacks"]["events"]
        )
    )
    event = next(
        event for event in snapshot["attacks"]["events"]
        if event["kind"] == "power_attack"
        and MODEL.graph.nodes[event["target_id"]].get("role") == "aggregation-switch"
    )
    target = event["target_id"]
    routing = snapshot["attacks"]["routing"]
    assert routing["by_sla"]["gold"]["availability_ratio"] == 1.0
    assert target in routing["service_transparent_power_targets"]
    assert target not in routing["avoid_nodes"]
    runtime = next(
        item for item in snapshot["attacks"]["power_runtime_state"]
        if item["target_id"] == target
    )
    assert runtime["ride_through_active"] is True
    assert runtime["service_availability_ratio"] == 1.0
    observation = next(
        item for item in snapshot["attacks"]["target_observations"]
        if item["target_id"] == target
    )
    assert observation["estimated_availability_ratio"] == 1.0
    assert observation["estimated_power_availability_ratio"] == 1.0


def test_endpoint_syn_victim_stays_online_while_attackers_are_quarantined() -> None:
    dynamics = _predictive_dynamics()
    snapshot = next(
        item for item in dynamics["snapshots"]
        if any(
            event["kind"] == "syn_flood"
            and MODEL.graph.nodes[event["target_id"]].get("level") == "L1"
            for event in item["attacks"]["events"]
        )
    )
    event = next(
        event for event in snapshot["attacks"]["events"]
        if event["kind"] == "syn_flood" and MODEL.graph.nodes[event["target_id"]].get("level") == "L1"
    )
    victim = next(
        flow for flow in snapshot["traffic"]["flows"]
        if flow.get("client_node") == event["target_id"] and not flow.get("is_attack_traffic")
    )
    assert victim["endpoint_protected"] is True
    assert victim["isolated"] is False
    assert set(event["ingress_nodes"]).issubset(snapshot["attacks"]["routing"]["quarantine_sources"])
    quarantined_normal_flows = [
        flow
        for flow in snapshot["traffic"]["flows"]
        if not flow.get("is_attack_traffic")
        and flow.get("client_node") in event["ingress_nodes"]
    ]
    assert quarantined_normal_flows
    assert all(flow["isolated"] for flow in quarantined_normal_flows)
    assert all(
        flow["security_excluded_from_sla_accounting"]
        for flow in quarantined_normal_flows
    )
    assert snapshot["attacks"]["security_excluded_flow_count"] >= len(
        quarantined_normal_flows
    )


def test_transport_connectivity_and_single_subscriber_access_are_explicit() -> None:
    l2_nodes = [node for node, attrs in MODEL.graph.nodes(data=True) if attrs.get("level") == "L2"]
    backbone = MODEL.graph.subgraph(
        [
            node
            for node in l2_nodes
            if MODEL.graph.nodes[node].get("role") in {"core-router", "aggregation-switch"}
        ]
    )
    assert nx.node_connectivity(backbone) == 3
    assert nx.edge_connectivity(backbone) == 3
    assert max(dict(backbone.degree()).values()) <= 6
    access_nodes = [
        node
        for node in l2_nodes
        if MODEL.graph.nodes[node].get("role") in {"radio-access-node", "optical-line-terminal"}
    ]
    assert access_nodes
    assert all(
        sum(
            1
            for neighbor in MODEL.graph.neighbors(node)
            if MODEL.graph.nodes[neighbor].get("role") == "aggregation-switch"
        )
        >= 2
        for node in access_nodes
    )
    subscribers = [
        (node, attrs) for node, attrs in MODEL.graph.nodes(data=True)
        if attrs.get("level") == "L1"
    ]
    assert len(subscribers) == SUBSCRIBER_COUNT
    assert all(MODEL.graph.degree(node) == 1 for node, _ in subscribers)
    assert all(
        set(MODEL.graph.neighbors(node)) == {attrs["home_access"]}
        for node, attrs in subscribers
    )
    assert all(
        not {"secondary_access", "tertiary_access", "resilient_access_mode"} & set(attrs)
        for _, attrs in subscribers
    )
    for node, attrs in subscribers:
        edge = MODEL.graph.edges[node, attrs["home_access"]]
        assert edge["access_role"] == "primary"
        assert edge["standby"] is False
        assert edge["consumes_c9500_physical_port"] is False
    ideal = _predictive_dynamics()["ideal_t0"]["metrics"]
    assert ideal["single_access_subscriber_count"] == SUBSCRIBER_COUNT
    assert ideal["service_standby_replica_count"] == 18
    assert ideal["l2_node_connectivity"] == 3
    assert ideal["maximum_l2_backbone_degree"] <= 6
    assert ideal["minimum_access_uplink_count"] >= 2


def test_t0_uses_home_access_and_access_failure_isolated_without_subscriber_reserve() -> None:
    baseline_flows = [
        flow for flow in _predictive_dynamics()["snapshots"][0]["traffic"]["flows"]
        if flow.get("sla_grade") == "gold" and not flow.get("is_attack_traffic")
    ]
    assert baseline_flows
    for flow in baseline_flows:
        client = flow["client_node"]
        route = flow["route"]
        access = route[1] if route[0] == client else route[-2]
        assert access == MODEL.graph.nodes[client]["home_access"]

    original = next(flow for flow in baseline_flows if flow["client_node"] == "M3_01")
    remapped, summary = apply_gold_first_remap(
        MODEL,
        [original],
        [],
        {
            "valid_for_step": 1,
            "routing_control": {
                "valid_for_step": 1,
                    "avoid_nodes": ["RAN5"],
                "maximum_projected_utilization_percent": 80.0,
            },
        },
        step_index=1,
    )
    recovered = remapped[0]
    assert summary["isolated_flow_count"] == 1
    assert recovered["active_access_node"] == "RAN5"
    assert recovered["active_access_role"] == "primary"
    assert recovered["routing_action"] == "isolate_no_safe_route"


def test_server_failover_admission_checks_cpu_ram_sessions_without_mutating_t0() -> None:
    original = next(
        flow
        for flow in _predictive_dynamics()["snapshots"][0]["traffic"]["flows"]
        if flow.get("sla_grade") == "gold"
        and not flow.get("is_attack_traffic")
        and MODEL.graph.nodes[flow["service_node"]].get("standby_hosts")
    )
    model = deepcopy(MODEL)
    service = original["service_node"]
    primary = original["server_node"]
    first_standby, second_standby, *_ = model.graph.nodes[service]["standby_hosts"]

    # Первый кандидат имеет меньше одного процента safe-headroom. Перенос даже
    # одной сессии должен пересечь все три 80-процентные границы, после чего
    # remapper обязан проверить следующий standby.
    overloaded_runtime = model.graph.nodes[first_standby]["runtime"]
    overloaded_runtime.update({
        "cpu_util_percent": 79.95,
        "ram_util_percent": 79.95,
        "network_util_percent": 79.95,
    })
    runtime_before = deepcopy(overloaded_runtime)
    remapped, summary = apply_gold_first_remap(
        model,
        [original],
        [],
        {
            "target_ids": [primary],
            "attack_kinds": ["power_attack"],
            "valid_for_step": 1,
            "routing_control": {
                "target_ids": [primary],
                "attack_kinds": ["power_attack"],
                "valid_for_step": 1,
            },
        },
        step_index=1,
    )

    assert remapped[0]["server_node"] == second_standby
    assert remapped[0]["failover_active"] is True
    assert summary["server_resource_rejected_candidate_count"] == 1
    assert summary["server_resource_rejection_reasons"] == {
        "cpu_above_80_percent": 1,
        "ram_above_80_percent": 1,
        "sessions_above_80_percent": 1,
    }
    assert summary["maximum_projected_server_resource_utilization_percent"] <= 80.0
    resources = {
        item["server_id"]: item for item in summary["projected_server_resources"]
    }
    assert resources[first_standby]["transferred_flow_count"] == 0
    assert resources[second_standby]["transferred_flow_count"] == 1
    assert summary["server_resource_admission_semantics"]["mutation"].startswith(
        "local_overlay_only"
    )
    assert model.graph.nodes[first_standby]["runtime"] == runtime_before


def test_server_resource_summary_exports_baseline_headroom_when_plan_is_inactive() -> None:
    _, summary = apply_gold_first_remap(
        MODEL,
        [],
        [],
        None,
        step_index=0,
    )
    assert len(summary["projected_server_resources"]) == 4
    assert summary["server_resource_rejected_candidate_count"] == 0
    assert summary["network_capacity_rejected_candidate_count"] == 0
    assert summary["maximum_projected_server_cpu_utilization_percent"] == 22.0
    assert summary["maximum_projected_server_ram_utilization_percent"] == 38.0
    assert 0.0 < summary["maximum_projected_server_session_utilization_percent"] < 80.0
    assert summary["maximum_projected_utilization_percent"] == 38.0

    traffic = simulate_packet_snapshot(
        MODEL,
        step_index=0,
        time_seconds=0,
        step_seconds=2,
        detail="summary",
    )
    exported = traffic["attack_state"]["routing"]
    assert exported["maximum_projected_server_resource_utilization_percent"] == 38.0
    assert exported["maximum_projected_utilization_percent"] == 38.0
    assert exported["maximum_projected_network_utilization_percent"] == max(
        exported["maximum_projected_edge_utilization_percent"],
        exported["maximum_projected_node_utilization_percent"],
    )


def test_lyapunov_and_koopman_react_to_controlled_configuration() -> None:
    dynamics = _predictive_dynamics()
    snapshots = dynamics["snapshots"]
    assert snapshots[0]["koopman"]["lyapunov_value"] == 0.0
    assert any(snapshot["koopman"]["lyapunov_value"] > 0.0 for snapshot in snapshots[1:])
    assert any(snapshot["koopman"]["lyapunov_derivative_per_second"] < 0.0 for snapshot in snapshots[1:])
    assert any(snapshot["koopman"]["lyapunov_remap_pressure"] > 0.0 for snapshot in snapshots[1:])
    assert all(
        snapshot["arbitrator"]["analysis"]["remap_pressure"]
        >= snapshot["koopman"]["lyapunov_remap_pressure"]
        for snapshot in snapshots
    )
    assert any(
        snapshot["attacks"]["routing"]["route_change_ratio"] > 0.0
        and snapshot["koopman"]["reference_distance"] > 0.0
        for snapshot in snapshots[1:]
    )
    assert snapshots[0]["koopman"]["lyapunov_solver"]["residual_frobenius"] < 1e-6
    assert snapshots[0]["koopman"]["lyapunov_solver"]["iterations"] <= 32
    assert snapshots[0]["attacks"]["routing"]["gold_route_hausdorff"]["normalized_distance"] == 0.0
    route_metric = snapshots[0]["attacks"]["routing"]["gold_route_hausdorff"]
    assert route_metric["maximum_edge_jaccard_distance"] == 0.0
    assert route_metric["maximum_latency_stretch_ratio"] == 1.0
    assert route_metric["normalization_scale_ms"] > 0.0
    assert "endpoint_attachments" in route_metric["normalization_scale_semantics"]
    assert snapshots[0]["koopman"]["threat_proximity"]["is_hausdorff_metric"] is False
    assert snapshots[0]["koopman"]["threat_proximity"]["normalization_scale_ms"] > 0.0


def test_seed_changes_reachable_server_power_and_dos_targets() -> None:
    odd = attack_catalog(MODEL, "predictive-demo", seed=43, step_seconds=2, step_count=60)
    power = next(item for item in odd if item["kind"] == "power_attack")
    assert MODEL.graph.nodes[power["target_id"]]["role"] == "service-server"
    assert power["power_failure_mode"] == "cyber_shutdown"
    assert power["mitre_technique_id"] == "T1529"
    assert {"T1078", "T0826"}.issubset(power["related_technique_ids"])
    dos_roles = {MODEL.graph.nodes[item["target_id"]]["role"] for item in odd if item["kind"] == "dos"}
    assert {"core-router", "aggregation-switch", "service-server"}.issubset(dos_roles)


def test_power_precursor_is_observable_and_paired_without_future_schedule() -> None:
    dynamics = _predictive_dynamics()
    power_onset = next(
        snapshot for snapshot in dynamics["snapshots"]
        if any(event["kind"] == "power_attack" and event.get("relative_step") == 0 for event in snapshot["attacks"]["events"])
    )
    target = next(event["target_id"] for event in power_onset["attacks"]["events"] if event["kind"] == "power_attack")
    warning = next(
        snapshot for snapshot in dynamics["snapshots"]
        if snapshot["time_seconds"] < power_onset["time_seconds"]
        and snapshot["koopman"]["forecast_is_early_warning"]
        and any(
            record.get("target_id") == target and record.get("kind") == "power_attack"
            for record in snapshot["koopman"]["forecast_entity_records"]
        )
    )
    assert warning["attacks"]["power_voltage_sag_ratio"] > 0.0
    assert warning["attacks"]["battery_discharge_rate_ratio"] > 0.0
    assert warning["koopman"]["maximum_warning_lead_seconds"] >= 5


def test_dmd_operator_matches_a_known_linear_transition() -> None:
    """Ковариационная DMD-формула должна восстанавливать K на полном ранге."""
    expected = np.asarray([[0.82, 0.08], [-0.03, 0.91]], dtype=float)
    observations = np.asarray(
        [[1.0, 0.0, 1.0, -0.5], [0.0, 1.0, 1.0, 0.75]],
        dtype=float,
    )
    gram = observations @ observations.T
    cross = (expected @ observations) @ observations.T

    actual = _operator_from_covariances(gram, cross, regularization=0.0)

    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_discrete_lyapunov_solver_matches_diagonal_closed_form() -> None:
    """Для диагональной K решение P известно: P_ii = 1 / (1 - K_ii**2)."""
    operator = np.diag([0.50, 0.80, 0.95])
    expected = np.diag(1.0 / (1.0 - np.diag(operator) ** 2))

    matrix, iterations, residual = _discrete_lyapunov_matrix(operator)

    assert iterations <= 32
    assert residual < 1e-9
    assert np.allclose(matrix, expected, rtol=1e-10, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(matrix) > 0.0)


def test_hausdorff_functions_match_an_elementary_geometry() -> None:
    left = [(0.0, 0.0), (2.0, 0.0)]
    right = [(0.0, 0.0), (5.0, 0.0)]

    assert directed_hausdorff_distance(left, right) == 2.0
    assert directed_hausdorff_distance(right, left) == 3.0
    assert hausdorff_distance(left, right) == 3.0
    assert hausdorff_distance([], []) == 0.0
    assert math.isinf(hausdorff_distance(left, []))
    assert directed_hausdorff_distance([], right) == 0.0
    assert math.isinf(directed_hausdorff_distance(left, []))


def test_sparse_state_hausdorff_is_zero_at_t0_and_grows_with_drift() -> None:
    reference = {
        "by_level": {
            "L1": [
                {
                    "scope": "node",
                    "node_id": "M1_01",
                    "tensor_name": "state",
                    "level": "L1",
                    "metrics": {"sla_margin": 0.50},
                    "units": {"sla_margin": "ratio"},
                }
            ]
        }
    }
    current = {
        "by_level": {
            "L1": [
                {
                    "scope": "node",
                    "node_id": "M1_01",
                    "tensor_name": "state",
                    "level": "L1",
                    "metrics": {"sla_margin": 0.20},
                    "units": {"sla_margin": "ratio"},
                }
            ]
        }
    }
    baseline = state_tensor_hausdorff_view(reference, reference, time_index=0)
    drift = state_tensor_hausdorff_view(reference, current, time_index=1)
    cached_drift = state_tensor_hausdorff_view(
        reference,
        current,
        time_index=1,
        prepared_reference=prepare_state_hausdorff_reference(reference),
    )
    assert baseline["distance"] == 0.0
    assert baseline["higher_is_worse"] is True
    assert drift["distance"] > baseline["distance"]
    assert cached_drift == drift
    assert drift["directed_t0_to_current"] == drift["directed_current_to_t0"]
    assert drift["witnesses"]["t0_to_current_tensor_id"] == "node:M1_01:state"


def test_exported_weibull_hazard_and_survival_match_declared_formulas() -> None:
    schedule = predictive_demo_weibull_schedule(
        seed=42,
        step_seconds=2,
        step_count=60,
    )
    shape = schedule.shape
    scale = schedule.scale_seconds

    for interval, hazard, survival in zip(
        schedule.raw_interarrival_seconds,
        schedule.realized_hazard_per_second,
        schedule.realized_survival_probability,
        strict=True,
    ):
        expected_hazard = shape / scale * (interval / scale) ** (shape - 1.0)
        expected_survival = math.exp(-((interval / scale) ** shape))
        assert math.isclose(hazard, expected_hazard, rel_tol=2e-6, abs_tol=1e-9)
        assert math.isclose(survival, expected_survival, rel_tol=2e-6, abs_tol=1e-9)
