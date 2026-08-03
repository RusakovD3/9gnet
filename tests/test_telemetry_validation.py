"""Проверки независимой хронологической валидации внешней телеметрии."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.gnet9.telemetry_validation import (
    TelemetryValidationConfig,
    TelemetryValidationError,
    load_pilot_calibration,
    read_telemetry_csv,
    validate_external_telemetry,
    write_validation_report,
)


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
