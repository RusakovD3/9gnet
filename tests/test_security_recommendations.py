"""Прогноз помогает человеку и не изменяет условия наблюдаемой сети."""

from copy import deepcopy
import inspect

import pytest

from src.gnet9 import koopman
from src.gnet9.attacks import apply_attack_effects, build_current_attack_flows
from src.gnet9.dynamics import DynamicsConfig, simulate_stationary_dynamics
from src.gnet9.packet_simulator import simulate_packet_snapshot
from src.gnet9.topology_builder import GNetBaselineBuilder


@pytest.fixture(scope="module")
def observed_run():
    model = GNetBaselineBuilder().build()
    reference = deepcopy(model.to_serializable())
    config = DynamicsConfig(step_count=24, step_seconds=2, attack_scenario="predictive-demo",
                            attack_start_step=16, attack_seed=42,
                            snapshot_detail="summary", packet_detail="flows")
    result = simulate_stationary_dynamics(model, config)
    assert model.to_serializable() == reference
    return model, config, result


def test_attacks_and_recommendations_preserve_routes_servers_and_t0(observed_run):
    _, _, result = observed_run
    snapshots = result["snapshots"]
    initial = {flow["client_node"]: (flow["route"], flow["reverse_route"], flow["server_node"])
               for flow in snapshots[0]["traffic"]["flows"]}
    assert result["ideal_t0"]["ok"]
    assert any(snapshot["attacks"]["active"] for snapshot in snapshots)
    assert len({snapshot["koopman"]["ideal_operator_fingerprint"] for snapshot in snapshots}) == 1
    for snapshot in snapshots:
        assert "remap" not in snapshot["arbitrator"]
        assert "prevention" not in snapshot["arbitrator"]
        assert "preventive_defense" not in snapshot["attacks"]
        assert "routing" not in snapshot["attacks"]
        assert snapshot["attacks"]["blocked_attack_packet_count"] == 0
        assert snapshot["attacks"]["attack_packet_count"] == snapshot["attacks"]["raw_attack_packet_count"]
        assert snapshot["attacks"]["route_observation"]["distance_ms"] == 0
        for flow in snapshot["traffic"]["flows"]:
            assert not flow.get("isolated") and not flow.get("quarantined")
            assert not flow.get("rerouted") and not flow.get("failover_active")
            if not flow.get("is_attack_traffic"):
                assert (flow["route"], flow["reverse_route"], flow["server_node"]) == initial[flow["client_node"]]


def test_recommendation_arrives_with_early_warning_and_keeps_observed_window(observed_run):
    _, _, result = observed_run
    warning = next(snapshot for snapshot in result["snapshots"]
                   if snapshot["koopman"]["forecast_is_early_warning"] and not snapshot["attacks"]["active"])
    recommendation = warning["arbitrator"]["recommendation"]
    assert recommendation["requires_human_action"] is True
    assert recommendation["execution_mode"] == "human_only"
    assert recommendation["target_ids"] == warning["koopman"]["forecast_target_ids"]
    assert recommendation["forecast_window_start_seconds"] > warning["time_seconds"]
    assert recommendation["forecast_window_end_seconds"] == warning["koopman"]["forecast_window_end_seconds"]
    assert recommendation["steps_ru"]
    assert any("человек" in step for step in recommendation["steps_ru"])
    assert recommendation["sla_priority_order"] == ["gold", "silver", "bronze"]
    assert "defense_plan" not in recommendation
    assert "sdn_intent" not in recommendation


def test_recommendation_text_cannot_change_future_observations(observed_run, monkeypatch):
    model, config, reference = observed_run
    original = koopman._recommendation

    def altered(*args, **kwargs):
        return original(*args, **kwargs) | {"steps_ru": ["Проверка независимости текста"]}

    monkeypatch.setattr(koopman, "_recommendation", altered)
    actual = simulate_stationary_dynamics(model, config)
    for before, after in zip(reference["snapshots"], actual["snapshots"]):
        assert before["traffic"] == after["traffic"]
        assert before["attacks"] == after["attacks"]
        assert before["state_vector"] == after["state_vector"]
        assert before["koopman"]["forecast_risk_score"] == after["koopman"]["forecast_risk_score"]


def test_packet_model_has_no_control_plan_input():
    for function in (simulate_packet_snapshot, apply_attack_effects, build_current_attack_flows):
        assert "defense_plan" not in inspect.signature(function).parameters


def test_primary_action_identifies_the_power_check_from_the_operator_plan():
    from src.gnet9.koopman import _recommendation

    advice = _recommendation(0.8, "power_attack", attack_state={}, warning_confirmed=True,
                             lyapunov_remap_pressure=0.)
    assert "напряжение" in advice.get("primary_action_ru", "")
    assert advice["primary_action_ru"] in advice["steps_ru"]


def test_healthy_step_recommends_routine_monitoring_without_a_target():
    advice = koopman._recommendation(0., "none", attack_state={"events": [], "precursors": []},
                                     warning_confirmed=False, lyapunov_remap_pressure=0.)
    assert advice["requires_human_action"] is False
    assert "штатное наблюдение" in advice["primary_action_ru"]
    assert advice["primary_action_ru"] in advice["steps_ru"]
