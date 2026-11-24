"""По сохранённой телеметрии должен получиться тот же прогноз, что в симуляторе."""

from tools.flow_replay import archive_info, observations, record, replay, verify


def test_capture_round_trip_preserves_metric_flags_physical_values_and_forecasts(tmp_path):
    path = tmp_path / "observations.jsonl.gz"
    digest = record(path, scenario="none", seed=11, step_count=2, step_seconds=2, start_step=None)
    assert archive_info(path)["step_seconds"] == 2
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
