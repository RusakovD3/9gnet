import csv
import json
from pathlib import Path

import pytest

from main import export_l1_monitoring, export_stationary_dynamics, parse_args
from src.gnet9.attacks import predictive_demo_minimum_steps
from src.gnet9.constants import SUBSCRIBER_COUNT
from src.gnet9.dynamics import DynamicsConfig
from src.gnet9.topology_builder import GNetBaselineBuilder


def test_l1_monitoring_csv_exports_single_access_field(tmp_path: Path) -> None:
    model = GNetBaselineBuilder().build()
    destination = tmp_path / "l1_monitoring.csv"

    export_l1_monitoring(model, destination)

    with destination.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == SUBSCRIBER_COUNT * 30
    assert "home_access" in rows[0]
    assert not {"secondary_access", "tertiary_access", "resilient_access_mode"} & set(rows[0])
    gold_mobile = next(
        row
        for row in rows
        if row["role"] == "mobile-subscriber" and row["sla_grade"] == "gold"
    )
    assert gold_mobile["home_access"]
    assert any("policies/l1_policies.d0sl" in note for note in model.notes)
    assert all("D:\\programming\\9gnet" not in note for note in model.notes)


def test_compact_dynamics_export_keeps_runtime_flows_without_serializing_them(
    tmp_path: Path,
) -> None:
    model = GNetBaselineBuilder().build()
    destination = tmp_path / "network_dynamics.json"

    runtime = export_stationary_dynamics(
        model,
        destination,
        DynamicsConfig(
            step_count=1,
            snapshot_detail="summary",
            packet_detail="flows",
        ),
        packet_detail="summary",
    )
    with (tmp_path / "step_log.csv").open(encoding="utf-8-sig", newline="") as stream:
        journal = list(csv.DictReader(stream))
    assert [row["шаг"] for row in journal] == ["0", "1"]
    assert journal[1]["фаза"] == "обучение на исправной сети"
    assert journal[1]["решение"] == "наблюдать"
    exported = json.loads(destination.read_text(encoding="utf-8"))

    assert runtime["snapshots"][0]["traffic"]["flows"]
    assert "flows" not in exported["snapshots"][0]["traffic"]
    assert "packet_sample" not in exported["snapshots"][0]["traffic"]
    assert exported["config"]["packet_detail"] == "summary"
    assert exported["config"]["runtime_packet_detail"] == "flows"


@pytest.mark.parametrize(
    "option",
    ("--dynamics-steps", "--step-seconds", "--packet-sample-limit"),
)
def test_cli_rejects_negative_simulation_sizes(option: str) -> None:
    with pytest.raises(SystemExit):
        parse_args([option, "-1"])


def test_cli_allows_t0_only_and_empty_packet_sample() -> None:
    args = parse_args(["--dynamics-steps", "0", "--packet-sample-limit", "0"])
    assert args.dynamics_steps == 0
    assert args.packet_sample_limit == 0


def test_cli_rejects_zero_step_duration() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--step-seconds", "0"])


def test_cli_rejects_attack_scenario_without_packet_model() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--no-packet-simulation", "--attack-scenario", "predictive-demo"])


def test_cli_rejects_predictive_window_shorter_than_its_precursor() -> None:
    required = predictive_demo_minimum_steps(2)
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--attack-scenario",
                "predictive-demo",
                "--dynamics-steps",
                str(required - 1),
            ]
        )


def test_cli_rejects_simultaneous_validation_and_calibration_profile() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--validate-telemetry-csv",
                "telemetry.csv",
                "--calibration-report",
                "profile.json",
            ]
        )
