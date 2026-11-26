"""Проверки независимой хронологической валидации внешней телеметрии."""

from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.gnet9.telemetry_validation import (
    TelemetryPoint,
    TelemetryValidationConfig,
    TelemetryValidationError,
    load_pilot_calibration,
    read_telemetry_csv,
    validate_external_telemetry,
    write_validation_report,
    _evaluate_threshold,
)


def _two_part_trace(warnings, onsets, *, period=50):
    return [TelemetryPoint(t, 0.9 if t % period in warnings else 0.1,
                           t % period in onsets, f"event-{t}", "")
            for t in range(2 * period)]


def test_overlapping_lead_windows_can_match_two_distinct_episodes():
    points = _two_part_trace({1, 7}, {20, 24})
    config = TelemetryValidationConfig(training_fraction=0.5, minimum_events_per_partition=2)
    report = validate_external_telemetry(points, config)
    metrics = report["independent_holdout"]["metrics"]
    assert metrics["predicted_onset_count"] == 2
    assert metrics["false_warning_episode_count"] == 0
    assert metrics["minimum_observed_lead_seconds"] == 17
    assert report["deployment_gate"]["passed"]


def test_late_update_is_separate_from_an_initial_warning_outside_lead_window():
    points = [TelemetryPoint(t, float(2 <= t <= 12), t == 23, "", "") for t in range(40)]
    metrics = _evaluate_threshold(points, 0.5, TelemetryValidationConfig(), max_gap=1.5)
    # Фронт в 2 с опередил событие на 21 с, ближайшее обновление в окне — в 3 с.
    assert metrics["predicted_onset_count"] == 0
    assert metrics["false_warning_episode_count"] == 1
    assert metrics["within_episode_updates"]["predicted_onset_count"] == 1
    assert metrics["within_episode_updates"]["minimum_observed_lead_seconds"] == 20


def test_calibration_finds_the_only_feasible_threshold_among_513_scores():
    scores = list(range(513))
    for timestamp, score in ((5, 512), (35, 511), (512, 510)):
        j = scores.index(score)
        scores[timestamp], scores[j] = scores[j], scores[timestamp]
    points = [TelemetryPoint(start + t, v / 512, t in (20, 50), "", "")
              for start in (0, 513) for t, v in enumerate(scores)]
    config = TelemetryValidationConfig(training_fraction=0.5, minimum_events_per_partition=2)
    report = validate_external_telemetry(points, config)
    assert report["deployment_gate"]["passed"]
    assert report["threshold_selection"]["selected_warning_risk_threshold"] == 511 / 512


def test_long_episodes_cannot_pass_on_updates_and_have_an_occupancy_budget():
    def points(onset):
        return [TelemetryPoint(t, float(t % 100 != 0), t % 100 == onset, "", "") for t in range(1000)]
    config = TelemetryValidationConfig(training_fraction=0.5)
    late = points(50)
    retrospective = validate_external_telemetry(late, config)
    assert not retrospective["deployment_gate"]["passed"]
    metrics = _evaluate_threshold(late[500:], 1.0, config, max_gap=1.5)
    assert metrics["warning_occupancy_ratio"] == pytest.approx(494 / 499)
    assert metrics["predicted_onset_count"] == 0
    assert metrics["within_episode_updates"]["predicted_onset_count"] == 5
    timely_points = points(15)
    timely = validate_external_telemetry(timely_points, config)
    assert timely["independent_holdout"]["metrics"]["predicted_onset_count"] == 5
    assert not timely["deployment_gate"]["passed"]
    relaxed = replace(config, maximum_warning_occupancy_ratio=1.0)
    assert validate_external_telemetry(timely_points, relaxed)["deployment_gate"]["passed"]


def test_earlier_schema_requires_revalidation_from_source_csv(tmp_path):
    path = tmp_path / "old.json"
    write_validation_report({"report_version": 2}, path)
    with pytest.raises(TelemetryValidationError, match="CSV"):
        load_pilot_calibration(path)


def test_pilot_loader_checks_actual_occupancy_in_each_partition(tmp_path):
    config = TelemetryValidationConfig(training_fraction=0.5, minimum_events_per_partition=1)
    report = validate_external_telemetry(_two_part_trace({5}, {20}), config)
    path = tmp_path / "inconsistent.json"
    for metrics in (report["threshold_selection"]["training_metrics"], report["independent_holdout"]["metrics"]):
        measured = metrics["warning_occupancy_ratio"]
        metrics["warning_occupancy_ratio"] = 1.0
        write_validation_report(report, path)
        with pytest.raises(TelemetryValidationError):
            load_pilot_calibration(path)
        metrics["warning_occupancy_ratio"] = measured


@pytest.mark.parametrize("gap, expected_matches, left_censored", [(0, 1, 1), (100, 2, 0)])
def test_continuing_warning_across_holdout_boundary_counts_only_exposure(gap, expected_matches, left_censored):
    config = TelemetryValidationConfig(training_fraction=0.5, minimum_events_per_partition=2)
    points = [TelemetryPoint(t + (gap if t >= 50 else 0), float(t in {5, 75} or 25 <= t <= 55),
                             t in {20, 45, 65, 90}, "", "") for t in range(100)]
    report = validate_external_telemetry(points, config)
    assert report["threshold_selection"]["selected_warning_risk_threshold"] == 1
    assert report["threshold_selection"]["training_metrics"]["predicted_onset_count"] == 2
    metrics = report["independent_holdout"]["metrics"]
    assert metrics["predicted_onset_count"] == expected_matches
    assert metrics["left_censored_warning_count"] == left_censored
    assert metrics["false_warning_episode_count"] == 0
    assert metrics["warning_duration_seconds"] == 7
    assert metrics["observed_duration_seconds"] == 49


def test_recall_limit_controls_allowed_misses_in_both_partitions():
    points = _two_part_trace({5, 35, 65, 95}, {20, 50, 80, 110, 140}, period=160)
    config = TelemetryValidationConfig(training_fraction=0.5, minimum_recall=0.8)
    report = validate_external_telemetry(points, config)
    assert report["independent_holdout"]["metrics"]["missed_attack_count"] == 1
    assert report["deployment_gate"]["passed"]
    assert not validate_external_telemetry(points, replace(config, minimum_recall=0.81))["deployment_gate"]["passed"]


def test_healthy_sample_between_warnings_always_ends_episode():
    times = [0, 10, 10.1, 10.2, 20, 20.2, 30, 40, 50, 60, 70, 80]
    points = [TelemetryPoint(t, float(t in (10, 10.2)), t in (20, 20.2), "", "") for t in times]
    report = validate_external_telemetry(points, TelemetryValidationConfig(training_fraction=0.7))
    metrics = report["threshold_selection"]["training_metrics"]
    assert metrics["warning_risk_threshold"] == 1.0
    assert metrics["warning_episode_count"] == 2


def test_programmatic_telemetry_requires_chronological_samples():
    points = _two_part_trace({1}, {20})
    points[0], points[1] = points[1], points[0]
    with pytest.raises(TelemetryValidationError, match="возрастать"):
        validate_external_telemetry(points)


def test_subsecond_exposure_and_single_sample_partition():
    points = [TelemetryPoint(t, 0.5, False, "", "") for t in (0, 0.25, 0.5)]
    report = validate_external_telemetry(points)
    training = report["threshold_selection"]["training_metrics"]
    assert training["observed_duration_seconds"] == 0.25
    assert training["right_censored_warning_count"] == 1
    assert training["false_warning_episode_count"] == 0
    assert report["independent_holdout"]["metrics"]["false_warnings_per_hour"] is None
    assert not report["deployment_gate"]["passed"]


def test_ten_hour_gap_does_not_dilute_false_warnings_or_alarm_occupancy():
    config = TelemetryValidationConfig(minimum_lead_seconds=1, maximum_lead_seconds=2)
    def evaluate(times):
        points = [TelemetryPoint(t, float(t == 1), False, "", "") for t in times]
        return _evaluate_threshold(points, 0.5, config, max_gap=1.5)
    regular = evaluate(range(7))
    interrupted = evaluate([0, 1, 2, 3, 36003, 36004, 36005, 36006])
    for metrics in (regular, interrupted):
        assert metrics["false_warnings_per_hour"] == 600
        assert metrics["observed_duration_seconds"] == 6
        assert metrics["warning_duration_seconds"] == 1
        assert metrics["warning_occupancy_ratio"] == 1 / 6
    assert interrupted["gap_duration_seconds"] == 36000


def test_warning_cannot_match_onset_across_gap_and_is_censored_at_trace_end():
    config = TelemetryValidationConfig(minimum_lead_seconds=1, maximum_lead_seconds=3)
    points = [TelemetryPoint(t, float(t in (1, 5)), t == 4, "", "") for t in [0, 1, 4, 5, 6]]
    metrics = _evaluate_threshold(points, 0.5, config, max_gap=1.5)
    assert metrics["predicted_onset_count"] == 0
    assert metrics["right_censored_warning_count"] == 2
    assert metrics["false_warning_episode_count"] == 0


@pytest.mark.parametrize("rows", [
    "timestamp_seconds,risk_score,risk_score,attack_onset\n0,0.1,0.9,0",
    "timestamp_seconds,risk_score,attack_onset\n0,0.1,0,1",
])
def test_ambiguous_csv_columns_are_rejected(tmp_path, rows):
    trace = tmp_path / "ambiguous.csv"
    trace.write_text(rows + "\n1,0.1,0,0\n2,0.1,0,0\n", encoding="utf-8")
    with pytest.raises(TelemetryValidationError, match="столбц"):
        read_telemetry_csv(trace)


@pytest.mark.parametrize(("section", "field", "value"), [
    ("deployment_gate", "passed", "false"),
    ("deployment_profile", "eligible_for_pilot", 1),
    ("deployment_profile", "warning_risk_threshold", True),
    ("deployment_profile", "warning_risk_threshold", 0.42),
    ("independent_holdout", "quality_gate_passed", False),
])
def test_inconsistent_calibration_report_is_rejected(tmp_path, section, field, value):
    report = validate_external_telemetry(_two_part_trace({5}, {20}),
        TelemetryValidationConfig(training_fraction=0.5, minimum_events_per_partition=1))
    assert report["deployment_gate"]["passed"]
    report[section][field] = value
    path = tmp_path / "bad.json"
    write_validation_report(report, path)
    with pytest.raises(TelemetryValidationError):
        load_pilot_calibration(path)


@pytest.mark.parametrize("payload", [[], None, {"report_version": 1, "deployment_profile": []}])
def test_calibration_report_requires_json_objects(tmp_path, payload):
    path = tmp_path / "wrong_shape.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TelemetryValidationError):
        load_pilot_calibration(path)


def _write_known_trace(path: Path, *, add_false_warning: bool = False) -> None:
    """Создать обезличенный ряд: каждый инцидент имеет предвестник ровно за 10 с."""

    onsets = {80, 130, 180, 230, 280, 380, 430, 480, 530, 580}
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("timestamp_seconds", "risk_score", "attack_onset", "attack_id", "attack_kind"),
        )
        writer.writeheader()
        for timestamp in range(0, 605, 5):
            is_onset = timestamp in onsets
            is_precursor = timestamp + 10 in onsets
            writer.writerow(
                {
                    "timestamp_seconds": timestamp,
                    "risk_score": 0.92 if is_precursor or (add_false_warning and timestamp == 335) else 0.04,
                    "attack_onset": int(is_onset),
                    "attack_id": f"attack-{timestamp}" if is_onset else "",
                    "attack_kind": "dos" if is_onset else "",
                }
            )


def test_external_telemetry_uses_chronological_holdout_without_leakage(tmp_path: Path) -> None:
    trace = tmp_path / "telemetry.csv"
    _write_known_trace(trace)

    report = validate_external_telemetry(
        read_telemetry_csv(trace),
        TelemetryValidationConfig(minimum_events_per_partition=3),
        source_path=trace,
    )

    assert report["threshold_selection"]["selection_uses_holdout"] is False
    assert report["validation_scope"] == "threshold_only_on_supplied_risk_scores"
    assert report["independent_holdout"]["performed"] is True
    assert report["deployment_gate"]["passed"] is True
    assert report["independent_holdout"]["metrics"]["recall"] == 1.0
    assert report["independent_holdout"]["metrics"]["minimum_observed_lead_seconds"] == 10.0
    assert "event_records" not in report["independent_holdout"]["metrics"]
    assert "missed_attack_ids" not in report["independent_holdout"]["metrics"]
    assert report["data_contract"]["source_sha256"]


def test_failed_external_holdout_report_cannot_be_applied_as_pilot_profile(tmp_path: Path) -> None:
    trace = tmp_path / "telemetry.csv"
    _write_known_trace(trace, add_false_warning=True)
    report = validate_external_telemetry(
        read_telemetry_csv(trace),
        TelemetryValidationConfig(
            minimum_events_per_partition=3,
            maximum_false_warnings_per_hour=0.0,
        ),
    )
    report_path = tmp_path / "failed_report.json"
    write_validation_report(report, report_path)

    assert report["deployment_gate"]["passed"] is False
    with pytest.raises(TelemetryValidationError, match="не прошёл"):
        load_pilot_calibration(report_path)


def test_external_csv_requires_strict_monotonic_time_and_required_columns(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.csv"
    invalid.write_text(
        "timestamp_seconds,risk_score,attack_onset\n10,0.1,0\n10,0.2,1\n15,0.1,0\n",
        encoding="utf-8",
    )

    with pytest.raises(TelemetryValidationError, match="строго возрастать"):
        read_telemetry_csv(invalid)


def test_calibration_preserves_threshold_precision_when_scores_are_close(tmp_path: Path) -> None:
    trace = tmp_path / "telemetry.csv"
    _write_known_trace(trace)
    points = [replace(point, risk_score=0.9200004 if point.risk_score > 0.5 else 0.9200002)
              for point in read_telemetry_csv(trace)]
    report = validate_external_telemetry(
        points, TelemetryValidationConfig(minimum_events_per_partition=3),
    )
    assert report["deployment_gate"]["passed"] is True
    assert report["threshold_selection"]["selected_warning_risk_threshold"] == 0.9200004
    assert report["independent_holdout"]["metrics"]["false_warning_episode_count"] == 0
    path = tmp_path / "report.json"
    write_validation_report(report, path)
    assert load_pilot_calibration(path)["warning_risk_threshold"] == 0.9200004


def test_external_csv_does_not_treat_missing_attack_label_as_healthy(tmp_path: Path) -> None:
    path = tmp_path / "missing_label.csv"
    path.write_text("timestamp_seconds,risk_score,attack_onset\n0,0.1,0\n5,0.9,\n10,0.1,1\n",
                    encoding="utf-8")
    with pytest.raises(TelemetryValidationError, match="attack_onset"):
        read_telemetry_csv(path)


@pytest.mark.parametrize("field", ["minimum_lead_seconds", "maximum_lead_seconds",
    "maximum_false_warnings_per_hour", "minimum_events_per_partition", "maximum_warning_occupancy_ratio"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_external_validation_rejects_nonfinite_configuration(field: str, value: float) -> None:
    with pytest.raises(TelemetryValidationError, match="конечными"):
        replace(TelemetryValidationConfig(), **{field: value}).validate()
