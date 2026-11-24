"""Проверки прогноза нарушений SLA по измерениям конкретного потока."""

from copy import deepcopy

import pytest

from src.gnet9.service_forecast import ServiceSLAForecaster, evaluate_service_forecasts
from src.gnet9.service_forecast import operator_warnings


def _flow(value, *, client="A", budget=10., sign="lte", tick=0):
    metric = {"comparison": sign, "label_ru": "задержка", "unit": "ms",
              "observed_value": value, "budget_value": budget, "effective_budget_value": budget,
              "compliant": value <= budget if sign == "lte" else value >= budget,
              "evidence": "modeled", "required": True}
    return {"client_node": client, "flow_id": f"{client}-application-{tick}",
            "traffic_kind": "voice", "sla_grade": "gold",
            "slo_evaluation": {"metrics": {"latency": metric}}}


def test_linear_headroom_uses_prior_healthy_pairs_and_measured_flow_identity():
    forecaster = ServiceSLAForecaster(2, retain_trace=True)
    views = []
    for i, value in enumerate((2, 4, 6, 8, 10, 12, 12)):
        views.append(forecaster.observe([_flow(value, tick=i)], time_seconds=i*2))
    assert not any(w["method"] == "koopman" for w in views[2]["warnings"])
    w = next(w for w in views[3]["warnings"] if w["method"] == "koopman")
    assert (w["flow_id"], w["training_pairs_used"], w["coefficient_used"]) == ("A", 1, 1)
    assert (w["horizon_seconds"], w["target_time_seconds"], w["predicted_value"]) == (4, 10, 12)
    assert not views[4]["onsets"] and len(views[5]["onsets"]) == 1 and not views[6]["onsets"]
    result = evaluate_service_forecasts(views)
    assert set(result["by_horizon_seconds"]) == {"2", "10", "20"}
    measured = result["methods"]["koopman"]
    assert (measured["hits"], measured["misses"], measured["onsets"]) == (1, 0, 1)
    assert measured["timing_error_seconds"] == [0] and measured["lead_seconds"] == [4]
    assert result["methods"]["persistence"]["misses"] == 1
    forecaster.case_traces(result)
    assert [v["value"] for v in result["cases"]["hit"]["measurements"]] == [2, 4, 6, 8, 10, 12, 12]
    assert result["worst_service"]["time_seconds"] == 10


def test_gap_is_censored_and_first_failed_measurement_has_unknown_onset():
    model = ServiceSLAForecaster(2)
    views = []
    for i, value in enumerate((2, 4, 6, 8, None, 15)):
        views.append(model.observe([] if value is None else [_flow(value)], time_seconds=2*i))
    scores = evaluate_service_forecasts(views)["methods"]["koopman"]
    assert scores["censored"] == 1 and scores["left_censored_impairments"] == 1
    assert scores["onsets"] == scores["false_alarms"] == 0 and scores["recall"] is None
    assert scores["observed_seconds"] == 6 and scores["warning_seconds"] == 0


def test_occupancy_counts_overlapping_warnings_once_and_false_episodes_once_each():
    model = ServiceSLAForecaster(2)
    views = []
    for i, value in enumerate((2, 4, 6, 8, *([8]*10))):
        observations = [_flow(value, client=c) for c in ("A", "B")]
        views.append(model.observe(observations, time_seconds=i*2))
    row = evaluate_service_forecasts(views)["methods"]["koopman"]
    assert row["false_alarms"] == 2 and row["warning_seconds"] == 20
    assert row["observed_seconds"] == 26 and row["warning_occupancy_ratio"] == pytest.approx(20/26)
    assert row["false_alarms_per_observed_hour"] == pytest.approx(2*3600/26)


def test_effective_demand_lower_limit_and_no_autonomous_change_from_later_measurements():
    forecaster = ServiceSLAForecaster(2)
    views = []
    for i, value in enumerate((12, 10, 8, 6, 4, 2)):
        f = _flow(value, sign="gte", budget=4.)
        f["slo_evaluation"]["metrics"]["latency"]["budget_value"] = 100.
        f["attack_id"] = f"future-label-{i}"
        views.append(forecaster.observe([f], time_seconds=2*i))
    stable = deepcopy(views)
    w = next(w for w in views[3]["warnings"] if w["method"] == "koopman")
    assert (w["budget"], w["policy_budget"], w["predicted_value"]) == (4, 100, 2)
    forecaster.observe([_flow(500.)], time_seconds=12)
    assert views == stable


def test_numeric_forecast_survives_changes_of_hidden_labels_and_rejects_unhealthy_training():
    models = [ServiceSLAForecaster(2), ServiceSLAForecaster(2)]
    history = []
    for i, value in enumerate((2, 3, 4, 7, 8, 9, 11, 1, 2, 3, 4)):
        observations = [_flow(value) for _ in models]
        observations[1].update(attack_id="unrelated", attack_start_step=70, attacker="hidden")
        results = [m.observe([f], time_seconds=i*2, nominal=i != 3) for m, f in zip(models, observations)]
        assert results[0] == results[1]
        history.append(models[0].fit.get(("voice", "gold", "latency"), (0, 0, 0))[2])
    # Только тройки, заканчивающиеся на шагах 2, 9 и 10, полностью исправны.
    assert history == [0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 3]


def test_integration_summary_and_flows_use_same_480_observations():
    from src.gnet9.dynamics import DynamicsConfig, simulate_stationary_dynamics
    from src.gnet9.topology_builder import GNetBaselineBuilder

    model = GNetBaselineBuilder().build()
    config = {"step_count": 1, "step_seconds": 2, "snapshot_detail": "summary"}
    runs = [simulate_stationary_dynamics(model, DynamicsConfig(**config, packet_detail=d))
            for d in ("summary", "flows")]
    assert runs[0]["service_sla_evaluation"] == runs[1]["service_sla_evaluation"]
    for s, full in zip(runs[0]["snapshots"], runs[1]["snapshots"]):
        assert "flows" not in s["traffic"]
        assert s["service_sla_forecast"] == full["service_sla_forecast"]
        services = s["service_sla_forecast"]["services"]
        assert len(services) == 6 and sum(row["assessed_flows"] for row in services.values()) == 480


def test_rate_projected_demand_and_delivered_value_share_the_same_future_budget():
    model = ServiceSLAForecaster(2)

    def rate(offer, delivered):
        flow = _flow(delivered, budget=min(100., offer), sign="gte")
        metric = flow["slo_evaluation"]["metrics"]["latency"]
        metric.update(unit="kbit/s", budget_value=100.)
        flow.update(payload_bytes=offer * 250., interval_seconds=2,
                    arrival_process="poisson", observed_dropped_packets=int(offer-delivered),
                    packet_count=200)
        return flow

    for i, offer in enumerate((60, 60, 100)):
        model.observe([rate(offer, offer)], time_seconds=2*i)
    # Обучение на двух завершённых переходах даёт (60+100)/2=80.
    candidates = model.observe([rate(160, 120)], time_seconds=6, nominal=False)["warnings"]
    w = next(w for w in candidates if w["method"] == "koopman")
    assert w["forecast_source"] == "poisson_demand_delivery_koopman"
    assert (w["measured_value"], w["budget"], w["predicted_value"], w["predicted_budget"]) == (120, 100, 60, 80)
    assert (w["training_pairs_used"], w["delivery_efficiency"], w["target_time_seconds"]) == (2, .75, 8)
    assert model.nominal_offers[("voice", "A")] == (80, 2)
    # Без потерь одно возвращение спроса к 80 не создаёт нарушения.
    assert not any(w["method"] == "koopman" for w in
                   model.observe([rate(160, 160)], time_seconds=8, nominal=False)["warnings"])


def test_common_route_delay_crossing_uses_observed_peers_and_resets_at_gap():
    model = ServiceSLAForecaster(2)

    def snapshot(t, readings, other_route=False):
        flows = []
        for client, v in zip(("A", "B", "C"), readings):
            flow = _flow(v, client=client)
            flow.update(route=["server", "other" if client == "C" and other_route else "switch", client],
                        observed_dropped_packets=int(t > 0), packet_count=100)
            flows.append(flow)
        return model.observe(flows, time_seconds=t, nominal=False)

    snapshot(0, (7, 4, 4))
    warnings = snapshot(2, (7, 7, 7))["warnings"]
    w = next(w for w in warnings if w["method"] == "koopman" and w["flow_id"] == "A")
    assert (w["forecast_source"], w["route_drift_ms_per_step"], w["observed_route_flows"]) == (
        "shared_route_delay_translation", 3, 3)
    assert (w["margin_delta"], w["predicted_value"], w["horizon_seconds"], w["training_pairs_used"]) == (0, 13, 4, None)
    assert not any(w["method"] == "trend" and w["flow_id"] == "A" for w in warnings)
    snapshot(4, ())
    assert not snapshot(6, (7, 7, 7))["warnings"]
    assert not any(w["method"] == "koopman" for w in snapshot(8, (8, 8, 8), other_route=True)["warnings"])


def test_missing_service_measurements_cannot_form_a_healthy_offer_pair():
    model = ServiceSLAForecaster(2)
    for i in range(2):
        missing = _flow(2., client="unmeasured")
        missing.update(payload_bytes=1000, interval_seconds=2, arrival_process="poisson")
        missing["slo_evaluation"]["metrics"]["latency"].update(observed_value=None, compliant=None)
        model.observe([_flow(2.), missing], time_seconds=i*2)
    assert ("voice", "unmeasured") not in model.nominal_offers


def test_operator_issues_only_calibrated_crossing_at_supported_period():
    warning = dict(method="koopman", observed_loss_fraction=.01, horizon_seconds=2,
                   criterion="one_way_media_latency_ms")
    view = {"step_seconds": 2, "warnings": [warning, dict(warning, observed_loss_fraction=0.),
            dict(warning, horizon_seconds=4), dict(warning, criterion="jitter_ms"), dict(warning, method="trend")]}
    assert operator_warnings(view) == [warning]
    assert not operator_warnings({**view, "step_seconds": 1})
