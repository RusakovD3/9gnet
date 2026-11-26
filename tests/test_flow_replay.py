"""По сохранённой телеметрии должен получиться тот же прогноз, что в симуляторе."""

import gzip
import json
from functools import partial
from unittest.mock import patch

from src.gnet9.route_stress import OPERATOR_VERSION, RouteStressMonitor
from tools.flow_replay import (
    _original_misses, _score_events, _with_lagged_sensors, archive_info, compare_lagged,
    compare_operator, observations, record, replay, verify,
)
from tools.flow_model_comparison import _choose, _criterion
from tools.forecast_experiment import compare_at_alarm_budget


def test_two_second_rule_is_selected_only_for_a_timely_crossing():
    identity = {"service": "voice", "flow_id": "A", "criterion": "latency"}
    warning = {**identity, "method": "koopman", "issue_time_seconds": 0,
               "target_time_seconds": 2, "horizon_seconds": 2, "observed_loss_fraction": .01}
    def series(onset):
        return [{"step_seconds": 2, "time_seconds": t, "measurement_gaps": [],
                 "services": {"voice": {"assessed_flows": 1}},
                 "onsets": [{**identity, "measured_value": 12}] if t == onset else [],
                 "warnings": [warning] if t == 0 else []} for t in range(0, 21, 2)]

    late = series(18)
    assert _choose(late, "koopman", ["latency"], _criterion)["latency"] is None
    timely = _choose(series(2), "koopman", ["latency"], _criterion)["latency"]
    assert timely["issue_horizon_seconds"] == 2 and timely["confirmations"] == 1
    repeat = compare_at_alarm_budget(late, {"confirmation_samples": {"koopman": 1}},
                                     horizon_seconds=2)
    assert repeat["methods"]["koopman"]["false_alarms"] == 1


def test_capture_round_trip_preserves_metric_flags_physical_values_and_forecasts(tmp_path):
    path = tmp_path / "observations.jsonl.gz"
    digest = record(path, scenario="none", seed=11, step_count=2, step_seconds=2, start_step=None)
    assert archive_info(path)["step_seconds"] == 2
    assert archive_info(path)["operator_version"] == OPERATOR_VERSION
    assert archive_info(path)["recorded_sensor_lag_steps"] == 0
    frames = list(observations(path))
    assert [(t, len(flows)) for t, _, flows in frames] == [(0, 480), (2, 480), (4, 480)]
    assert frames[0][2][0]["arrival_process"] == "expected_reference"
    for flow in frames[1][2]:
        assert flow["arrival_process"] == "poisson"
        assert flow["interval_seconds"] == 2
        assert flow["packet_count"] >= flow["observed_dropped_packets"] >= 0
        assert flow["payload_bytes"] >= 0
        assert not {"attack_id", "attack_ids", "attack_kind", "is_attack_traffic"} & flow.keys()
        assert flow["route"] and flow["home_access_node"]
    metrics = frames[1][2][0]["slo_evaluation"]["metrics"]
    assert metrics["service_availability"]["observed_value"] is True
    assert metrics["service_availability"]["budget_value"] is True
    assert metrics["service_availability"]["compliant"] is True
    assert verify(path, replay(path)) == digest
    comparison = compare_operator(path)
    assert comparison["verified_raw_sha256"] == digest
    assert comparison["baseline"]["onsets"] == comparison["updated"]["onsets"] == 0
    assert comparison["updated"]["false_alarms"] == 0
    # В исправном примере ни одна версия не выпускает предупреждений. Контрольная
    # сумма подходит также к прежнему формату заголовка без номера версии.
    with gzip.open(path, "rt", encoding="utf8") as source:
        header, frames = json.loads(source.readline()), source.read()
    del header["operator_version"], header["recorded_sensor_lag_steps"]
    legacy = tmp_path / "legacy.jsonl.gz"
    with gzip.open(legacy, "wt", encoding="utf8") as stream:
        stream.write(json.dumps(header) + "\n" + frames)
    historical = compare_operator(legacy)
    assert historical["recorded_operator_version"] == 2
    assert historical["verified_operator_sha256"] == comparison["verified_operator_sha256"]


def test_operator_comparison_replays_actual_warnings_from_recorded_sensor_data(tmp_path):
    path = tmp_path / "operator.jsonl.gz"
    # Подпись прежней выдачи должна проходить проверку и после смены рабочего прогноза.
    with (patch("src.gnet9.dynamics.RouteStressMonitor", partial(RouteStressMonitor, operator_version=3)),
          patch("tools.flow_replay.OPERATOR_VERSION", 3)):
        record(path, scenario="mitre-demo", seed=7, step_count=6, start_step=3, vary_demand=True, demand_seed=17)
    rows = [sample for _, _, _, readings in observations(path, with_sensors=True) for sample in readings]
    assert rows and all(set(sample) == {"sensor_node_id", "signals"} for sample in rows)
    paired = compare_operator(path)
    assert paired["recorded_operator_version"] == 3
    assert paired["archive"]["demand_seed"] == 17
    assert paired["baseline"]["onsets"] == paired["updated"]["onsets"] > 0
    assert paired["updated"]["by_forecast_source"]["route_acceleration_one_drop"]["hits"] > 0
    assert set(paired["updated"]["lead_seconds"]) == {2}
    for version in (3, 4):
        lagged = compare_lagged(path, candidate_version=version)
        assert lagged["recorded_operator_version"] == 3
        assert lagged["candidate_version"] == version
        assert lagged["verified_operator_sha256"] == paired["verified_operator_sha256"]
        assert set(lagged["scores"]) == {"v2_timely", "v2_lagged", f"v{version}_lagged"}
        assert lagged["scores"]["v2_lagged"]["onsets"] == lagged["scores"][f"v{version}_lagged"]["onsets"]
        assert lagged["original_misses"]["count"] == lagged["scores"]["v2_timely"]["misses"]
        assert f"v{version}_only" in lagged["lagged_changes"]["hits"]
        if version == OPERATOR_VERSION:
            assert compare_lagged(path) == lagged


def test_lag_keeps_flow_at_issue_time_and_does_not_carry_sensors_across_gaps():
    frames = [(t, True, [t], readings) for t, readings in
              ((0, ["A"]), (2, None), (4, []), (6, ["B"]), (10, ["C"]), (12, []))]
    shifted = list(_with_lagged_sensors(frames, 2))
    assert [row[:4] for row in shifted] == frames
    assert [row[4] for row in shifted] == [None, ["A"], None, [], None, ["C"]]


def test_missed_cohort_matches_exact_step_flow_and_criterion_and_preserves_censoring():
    def frame(t, failed=(), warned=(), gaps=()):
        def stream(client):
            return {"flow_id": client, "criterion": "delay", "service": "voice"}
        return {"time_seconds": t, "step_seconds": 2, "services": {"voice": {"assessed_flows": 5}},
                "measurement_gaps": [("voice", client, "delay") for client in gaps],
                "onsets": [{**stream(client), "time_seconds": t, "measured_value": 11, "budget": 10}
                           for client in failed],
                "warnings": [{**stream(client), "method": "operator", "forecast_source": "route",
                              "issue_time_seconds": t, "target_time_seconds": t + 2, "horizon_seconds": 2}
                             for client in warned]}

    # A нарушает один и тот же критерий дважды. В исходные пропуски попадает только первое начало.
    def series(warnings_at_zero):
        return [frame(0, warned=warnings_at_zero), frame(2, failed=("A", "B", "C", "D"), gaps=("G",)),
                frame(4, warned=("A",)), frame(6, failed=("A",), warned=("Z",))]

    timely, original = _score_events(series(("D",)))
    old_score, old = _score_events(series(("A", "B", "D", "G", "X")))
    new_score, new = _score_events(series(("B", "C", "G", "X")))
    assert timely["hits"] == 2 and timely["misses"] == 3
    for score in (old_score, new_score):
        assert score["by_forecast_source"]["route"]["false_alarms"] == 1
        assert score["by_criterion"]["delay"]["censored"] == 2  # пропуск G и конец наблюдения Z
    for version in (3, 4):
        cohort = _original_misses(original["misses"], old["hits"], new["hits"], version)
        assert cohort["count"] == 3
        assert cohort["by_criterion"]["delay"] == {
            "count": 3, "v2_lagged_hits": 2, f"v{version}_lagged_hits": 2, "v2_only": 1, "shared": 1,
            f"v{version}_only": 1, "neither": 0,
        }
        assert [(row["time_seconds"], row["flow_id"], row["v2_lagged_source"], row[f"v{version}_lagged_source"])
                for row in cohort["events"]] == [(2, "A", "route", None), (2, "B", "route", "route"),
                                                 (2, "C", None, "route")]
