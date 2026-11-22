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
    assert report["threshold_selection"]["training_metrics"]["false_warnings_per_hour"] == 14400
    assert report["independent_holdout"]["metrics"]["false_warnings_per_hour"] is None
    assert not report["deployment_gate"]["passed"]


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
    "maximum_false_warnings_per_hour", "minimum_events_per_partition"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_external_validation_rejects_nonfinite_configuration(field: str, value: float) -> None:
    with pytest.raises(TelemetryValidationError, match="конечными"):
        replace(TelemetryValidationConfig(), **{field: value}).validate()
