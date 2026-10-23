"""Защита наблюдений, настройка шума и запрет обучения на слабом ущербе."""

import numpy as np
import pytest

from main import parse_args
from src.gnet9.dynamics import DynamicsConfig, simulate_stationary_dynamics
from src.gnet9.kalman import DiagonalKalmanFilter
from src.gnet9.koopman import KoopmanOnlineAnalyzer
from src.gnet9.local_observation import is_smoothable_metric
from src.gnet9.topology_builder import GNetBaselineBuilder


@pytest.fixture(scope="module")
def model():
    return GNetBaselineBuilder().build()


@pytest.mark.parametrize("name", ["L1.authentication_failure_rate_per_second.mean",
    "L6.energy_reserve_ratio.min", "EDGE.loss_probability.max", "L1.syn_ack_fraction.mean",
    "OBS.legitimate_loss_ratio", "L2.negative_arrival_rate_pps"])
def test_threats_and_losses_are_not_noise_even_with_statistic_suffix(name):
    assert not is_smoothable_metric(name)
    assert is_smoothable_metric("L2.cpu_load_percent.mean")


def test_unfiltered_measurement_is_exact_even_after_large_previous_value():
    kalman = DiagonalKalmanFilter(1)
    kalman.update(np.array([1e16]), np.array([False]))
    assert kalman.update(np.array([1.]), np.array([False]))[0] == 1.


@pytest.mark.parametrize("name,reference,damaged", [
    ("OBS.gold_sla_compliance_ratio", 1., .9999),
    ("OBS.silver_delivery_ratio", 1., .999),
    ("OBS.legitimate_loss_ratio", 0., .00001),
])
def test_weak_quality_damage_freezes_filter_and_two_sided_training(model, name, reference, damaged):
    state = {"metric_names": [name], "vector": [reference]}
    analyzer = KoopmanOnlineAnalyzer.from_t0(model, state, step_seconds=2)
    def observe(step, value):
        return analyzer.analyze_step(model, {**state, "vector": [value]},
            attack_state={}, step_index=step, time_seconds=step * 2)
    observe(0, reference)
    revision = analyzer.operator_revision
    bad = observe(1, damaged)
    assert not bad["kalman_filter"]["healthy_quality_observation"]
    assert not bad["kalman_filter"]["applied_to_nominal_observation"]
    assert not bad["nominal_training"]["transition_accepted"]
    assert analyzer.operator_revision == revision
    recovered = observe(2, reference)
    assert not recovered["nominal_training"]["transition_accepted"]
    # Для следующего спокойного отсчёта снимаем затухающий прогноз угрозы:
    # даже если он ещё активен, переход по-прежнему не может быть обучен ошибочно.
    assert analyzer.nominal_observation_count == 0


def test_noise_configuration_reaches_all_filter_channels(model):
    result = simulate_stationary_dynamics(model, DynamicsConfig(step_count=1, step_seconds=2,
        kalman_process_variance=.01, kalman_measurement_variance=.04))
    config = result["config"]
    assert config["kalman_process_variance"] == .01
    for snapshot in result["snapshots"]:
        kalman = snapshot["koopman"]["kalman_filter"]
        assert kalman["process_variance"] == kalman["local_observation"]["process_variance"] == .01
        assert kalman["measurement_variance"] == kalman["local_observation"]["measurement_variance"] == .04
    args = parse_args(["--kalman-process-variance", ".01", "--kalman-measurement-variance", ".04"])
    assert args.kalman_process_variance == .01 and args.kalman_measurement_variance == .04


@pytest.mark.parametrize("option,value", [("--kalman-process-variance", "-1"),
    ("--kalman-process-variance", "nan"), ("--kalman-measurement-variance", "0"),
    ("--kalman-measurement-variance", "inf")])
def test_invalid_noise_rejected_before_start(option, value):
    with pytest.raises(SystemExit):
        parse_args([option, value])


def test_future_attack_schedule_does_not_change_clean_prefix(model):
    results = [simulate_stationary_dynamics(model, DynamicsConfig(step_count=30, step_seconds=2,
        attack_scenario="predictive-demo", attack_seed=42, attack_start_step=start))
        for start in (25, 27)]
    for left, right in zip(results[0]["snapshots"][:15], results[1]["snapshots"][:15]):
        assert left["traffic"]["summary"]["normal_arrival_rate_pps"] == right["traffic"]["summary"]["normal_arrival_rate_pps"]
        for field in ("kalman_filter", "forecast_risk_score", "operator_fingerprint_used", "lyapunov_value"):
            assert left["koopman"][field] == right["koopman"][field]
