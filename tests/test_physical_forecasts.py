import numpy as np
import pytest

from src.gnet9.koopman import KoopmanOnlineAnalyzer
from src.gnet9.topology_builder import GNetBaselineBuilder


MODEL = GNetBaselineBuilder().build()


def test_novel_loss_does_not_drive_cpu_or_line_outside_observations():
    names = ["L2.cpu_load_percent.max", "EDGE.utilization.mean", "EDGE.loss_probability.max",
             "OBS.gold_sla_compliance_ratio"]
    baseline = {"metric_names": names, "vector": [18., .015, .0015, 1.]}
    analyzer = KoopmanOnlineAnalyzer.from_t0(MODEL, baseline, step_seconds=2)
    analyzer.analyze_step(MODEL, baseline, attack_state={}, step_index=0, time_seconds=0)
    state = {"metric_names": names, "vector": [18., 1.25, .579, .6875]}
    analysis = analyzer.analyze_step(MODEL, state, attack_state={}, step_index=1, time_seconds=2)
    for forecast in analysis["horizon_forecasts"]:
        assert forecast["predicted_vector"] == pytest.approx(state["vector"])
        assert forecast["physical_projection"]["corrected_metric_count"] == 0
    assert analysis["predicted_next_vector"]["vector"] == analysis["horizon_forecasts"][0]["predicted_vector"]
    domains = analysis["training_reference"]["physical_domains"]
    assert domains[names[1]] == {"lower": 0., "upper": None}
    assert domains[names[0]]["upper"] == 100.
    assert domains[names[-1]]["upper"] == 1.


def test_identifiability_reports_observed_inputs_separately_from_prior():
    names = ["L2.cpu_load_percent.max", "OBS.precursor_confidence"]
    state = {"metric_names": names, "vector": [18., 0.]}
    analyzer = KoopmanOnlineAnalyzer.from_t0(MODEL, state, step_seconds=2)
    spectrum = None
    for step in range(6):
        analysis = analyzer.analyze_step(MODEL, {"metric_names": names, "vector": [18.1, 0.]},
                                         attack_state={}, step_index=step, time_seconds=2 * step)
        spectrum = analysis["nominal_training"]["observed_input_spectrum"]
        assert spectrum["sample_count"] == max(0, step - 1)
    assert spectrum["rank"] == 1
    assert spectrum["singular_values"][1] == 0.
    assert analysis["training_reference"]["synthetic_input_spectrum"]["sample_count"] == 63
    assert np.all(np.isfinite(analysis["nominal_training"]["diagonal_coefficients_used"]))


def test_future_schedule_and_scenario_label_cannot_change_any_issued_horizon():
    baseline = {"metric_names": ["OBS.precursor_confidence", "L2.cpu_load_percent.max"],
                "vector": [0., 18.]}
    measurement = {**baseline, "vector": [.9, 18.]}
    traces = []
    for scenario, onset in (("mitre-demo", 4), ("predictive-demo", 44)):
        model = KoopmanOnlineAnalyzer.from_t0(MODEL, baseline, step_seconds=2)
        observation = {
            "scenario": scenario,
            "future_attack_start_step": onset,
            "precursors": [{"sensor_node_id": "C2", "signals": {"syn_backlog_growth_ratio": .9},
                            "attack_id": str(onset), "start_step": onset}],
        }
        traces.append([model.analyze_step(MODEL, measurement, attack_state=observation,
                                          step_index=step, time_seconds=2 * step) for step in range(2)])
    for first, second in zip(*traces):
        assert first["horizon_forecasts"] == second["horizon_forecasts"]
        assert first["forecast_entity_records"] == second["forecast_entity_records"]
        assert first["numeric_forecast_origin"] == second["numeric_forecast_origin"]
    assert [row["forecast_is_early_warning"] for row in traces[0]] == [False, True]


def test_measured_lyapunov_delta_includes_residual_on_same_operator():
    baseline = {"metric_names": ["OBS.precursor_confidence"], "vector": [0.]}
    model = KoopmanOnlineAnalyzer.from_t0(MODEL, baseline, step_seconds=2)
    for step, value in enumerate((.4, .2)):
        analysis = model.analyze_step(MODEL, {**baseline, "vector": [value]},
                                      attack_state={}, step_index=step, time_seconds=2 * step)
    terms = analysis["lyapunov_transition"]
    p = 1. / (1. - .85**2)
    assert terms["observed_delta_raw"] == pytest.approx(-.12 * p)
    assert terms["unforced_delta_raw"] == pytest.approx(-.16)
    assert terms["cross_term_raw"] == pytest.approx(2 * .34 * p * (-.14))
    assert terms["residual_energy_raw"] == pytest.approx(.14**2 * p)


def test_affine_ordinary_arrival_model_on_known_stationary_mean_and_slope():
    metric = "OBS.normal_arrival_rate_pps"
    state = {"metric_names": [metric], "vector": [100.]}
    analyzer = KoopmanOnlineAnalyzer.from_t0(MODEL, state, step_seconds=2)
    # y = 100 + 0.5 (x − 100). Обычные поступления измеряются и при нагрузке атаки.
    analyzer.nominal_arrivals.add_pair(80., 90.)
    analyzer.nominal_arrivals.add_pair(120., 110.)
    observed = {"events": [{"target_id": "C2", "kind": "dos"}]}
    result = analyzer.analyze_step(MODEL, {**state, "vector": [110.]},
                                   attack_state=observed, step_index=1, time_seconds=2)
    assert result["numeric_forecast_origin"][metric] == "observed_affine_arrival_dmd"
    forecasts = result["horizon_forecasts"]
    assert [forecasts[h - 1]["predicted_vector"][0] for h in (1, 5, 10)] == pytest.approx(
        [105., 100.3125, 100.009765625])
    outside = analyzer.analyze_step(MODEL, {**state, "vector": [150.]},
                                    attack_state=observed, step_index=2, time_seconds=4)
    assert outside["numeric_forecast_origin"][metric] == "nominal_arrival_mean_outside_observed_range"
    assert outside["predicted_next_vector"]["vector"] == [100.]
    assert outside["nominal_training"]["ordinary_arrival_fit_used"]["transition_count"] == 2


def test_arrival_fit_uses_only_past_completed_raw_healthy_pairs():
    state = {"metric_names": ["OBS.normal_arrival_rate_pps"], "vector": [100.]}
    analyzer = KoopmanOnlineAnalyzer.from_t0(MODEL, state, step_seconds=2)
    fits = []
    for step, value in enumerate((102., 101., 99.)):
        report = analyzer.analyze_step(MODEL, {**state, "vector": [value]},
                                       attack_state={}, step_index=step, time_seconds=step * 2)
        fits.append(report["nominal_training"]["ordinary_arrival_fit_used"])
    assert [fit["transition_count"] for fit in fits] == [0, 0, 1]
    assert fits[-1]["mean_of_pair_endpoints_pps"] == 101.5
    assert fits[-1]["observed_input_range_pps"] == [102., 102.]


@pytest.mark.parametrize("changed_names,values", [
    (["EDGE.utilization.mean", "L2.cpu_load_percent.max"], [.02, 18.]),
    (["L2.cpu_load_percent.max", "EDGE.utilization.mean"], [float("nan"), .02]),
])
def test_damaged_measurement_is_rejected_before_mutating_forecast_history(changed_names, values):
    state = {"metric_names": ["L2.cpu_load_percent.max", "EDGE.utilization.mean"], "vector": [18., .02]}
    analyzer = KoopmanOnlineAnalyzer.from_t0(MODEL, state, step_seconds=2)
    observed = {"events": [{"kind": "dos", "target_id": "C1"}]}
    with pytest.raises(ValueError):
        analyzer.analyze_step(MODEL, {"metric_names": changed_names, "vector": values},
                              attack_state=observed, step_index=1, time_seconds=2)
    assert analyzer.analysis_revision == 0 and analyzer.previous_vector is None
