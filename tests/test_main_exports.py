import csv
import json
from pathlib import Path
import re
import sys

import pytest

from main import _write_dynamics_json, export_attack_catalog, export_l1_monitoring, export_stationary_dynamics, parse_args
from src.gnet9.attacks import attack_catalog, predictive_demo_minimum_steps
from src.gnet9.constants import SUBSCRIBER_COUNT
from src.gnet9.detection_rules import export_detection_rules
from src.gnet9.dynamics import DynamicsConfig
from src.gnet9.topology_builder import GNetBaselineBuilder


@pytest.mark.parametrize("detailed", [False, True])
def test_reports_only_writes_linked_results_in_separate_directory(tmp_path, monkeypatch, detailed):
    import main

    root = Path(main.__file__).parent
    policy = tmp_path / "policies" / "l1_policies.d0sl"
    policy.parent.mkdir()
    policy.write_bytes((root / "policies" / policy.name).read_bytes())
    monkeypatch.setattr(main, "__file__", str(tmp_path / "main.py"))
    monkeypatch.setattr(sys, "argv", ["main.py", "--reports-only", "--dynamics-steps", "1"] + (["--detailed-output"] if detailed else []))
    main.main()

    directory = tmp_path / "output" / "reports"
    assert set(directory.parent.iterdir()) == {directory}
    if detailed:
        directory = directory / "details"
    assert not list(directory.rglob("*.png"))
    assert not (directory / "baseline_topology.json").exists()
    report = json.loads((directory / "run_summary.json").read_bytes())
    if detailed:
        assert report == json.loads((directory / "network_dynamics.json").read_bytes())["result_summary"]
        assert report["files"]["detection_rules"] == "detection_rules/manifest.json"
    else:
        assert {p.name for p in directory.iterdir()} == {"run_summary.md", "run_summary.json", "step_log.csv"}
    assert report["observation"]["snapshot_count"] == 2
    links = re.findall(r"\]\(([^)]+)\)", (directory / "run_summary.md").read_text(encoding="utf-8"))
    for name in [*report["files"].values(), *links]:
        assert (directory / name).is_file(), name


def test_dynamics_json_replaces_only_a_complete_export(tmp_path: Path) -> None:
    path = tmp_path / "network_dynamics.json"
    payload = {"export_metadata": {"schema_version": 3}, "snapshots": [
        {"step_index": 0, "label": "эталон"}, {"step_index": 1, "missing": None},
    ]}
    _write_dynamics_json(path, payload)
    original = path.read_bytes()
    assert json.loads(original) == payload
    assert len([line for line in original.splitlines() if b'"step_index"' in line]) == 2
    corrupt = {**payload, "snapshots": [*payload["snapshots"], {"step_index": 2, "risk": float("nan")}]}
    with pytest.raises(ValueError):
        _write_dynamics_json(path, corrupt)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


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


def test_l1_streamed_csv_keeps_last_complete_export_on_error(tmp_path, monkeypatch):
    destination = tmp_path / "monitoring.csv"
    destination.write_bytes(b"previous complete export")
    def subscribers(model):
        yield "C1", {"monitoring": [{"second": 0}]}
        raise ValueError("incomplete monitoring")

    monkeypatch.setattr("main.iter_l1_subscribers", subscribers)
    with pytest.raises(ValueError, match="incomplete monitoring"):
        export_l1_monitoring(None, destination)
    assert destination.read_bytes() == b"previous complete export"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("scenario", ["predictive-demo", "mitre-demo"])
def test_exported_attack_catalog_matches_custom_simulation_schedule(tmp_path: Path, scenario: str) -> None:
    model = GNetBaselineBuilder().build()
    config = DynamicsConfig(step_count=100, step_seconds=2, attack_scenario=scenario,
                            attack_start_step=50, attack_interval_steps=10)
    path = tmp_path / "catalog.json"
    export_attack_catalog(model, path, config)
    exported = json.loads(path.read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(attack_catalog(model, scenario, config=config)))
    assert exported["scenarios"][scenario] == expected
    masks = exported["observation_signature_matrix"]["weighted_masks"]
    assert masks["step_seconds"] == config.step_seconds
    refs = masks["numeric_references"]
    assert refs["T1046"]["maximum_lag_seconds"] == 60
    assert refs["T1110.001"]["attempts_per_hour_per_account"] == 300


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
    assert journal[1]["фаза"] == "начальный участок наблюдения"
    assert journal[1]["переход_принят_в_обучение"] == "да"
    assert journal[1]["SLA_gold"] == "1.0"
    assert journal[1]["фильтр_применён"] == "да"
    assert journal[1]["решение"] == "наблюдать"
    exported = json.loads(destination.read_text(encoding="utf-8"))

    assert runtime["snapshots"][0]["traffic"]["flows"]
    assert "flows" not in exported["snapshots"][0]["traffic"]
    assert "packet_sample" not in exported["snapshots"][0]["traffic"]
    original_state = runtime["snapshots"][0]["traffic"]["network_state"]
    compact_state = exported["snapshots"][0]["traffic"]["network_state"]
    assert "nodes" not in compact_state and "edges" not in compact_state
    assert compact_state["node_ids"] == [n["id"] for n in original_state["nodes"]]
    assert compact_state["vertex_vectors"] == [list(n["metrics"].values()) for n in original_state["nodes"]]
    assert compact_state["edge_vectors"] == [list(e["metrics"].values()) for e in original_state["edges"]]
    assert exported["config"]["packet_detail"] == "summary"
    assert exported["config"]["runtime_packet_detail"] == "flows"
    assert exported["export_metadata"]["schema_version"] == 3
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["training"]["accepted_transitions_total"] == 1
    assert summary["prediction"]["evaluated"] is False
    assert (tmp_path / "run_summary.md").is_file()
    rules = tmp_path / "detection_rules"
    report = json.loads((rules / "manifest.json").read_text(encoding="utf-8"))
    assert report["sigma"]["rule_count"] == report["suricata"]["rule_count"] == 0
    assert not (rules / "gnet9.rules").exists()


def test_attack_run_exports_local_rules_and_keeps_source_unconfirmed(tmp_path: Path) -> None:
    model = GNetBaselineBuilder().build()
    destination = tmp_path / "network_dynamics.json"
    runtime = export_stationary_dynamics(
        model, destination, DynamicsConfig(step_count=8, step_seconds=5, attack_scenario="mitre-demo"),
    )
    rules = tmp_path / "detection_rules"
    report = json.loads((rules / "manifest.json").read_text(encoding="utf-8"))
    assert report["yara"]["rule_count"] == 0
    sigma = (rules / "gnet9.yml").read_text(encoding="utf-8")
    assert "attack.t1499.001" in sigma
    logs = [json.loads(line) for line in (rules / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["technique_id"] == "T1499.001" for row in logs)
    signatures = (rules / "gnet9.rules").read_text(encoding="utf-8").splitlines()
    assert len(signatures) == 1
    assert signatures[0].startswith(f"alert tcp any any -> {model.graph.nodes['M1_01']['ip_address']} any")
    assert "tcp.flags:S,CE;" in signatures[0]
    active = [snapshot for snapshot in runtime["snapshots"] if snapshot["attacks"]["events"]]
    assert active
    for snapshot in active:
        trigger = snapshot["arbitrator"]["trigger_card"]
        assert trigger["source_confirmed"] is False
        assert "review_confirmed_attack_sources" not in trigger["controls"]

    export_stationary_dynamics(model, destination, DynamicsConfig(step_count=0))
    assert not (rules / "gnet9.rules").exists()
    assert not (rules / "gnet9.yml").exists()
    assert not (rules / "observations.jsonl").exists()


def test_sigma_matches_observed_hypotheses_and_preserves_simulation_truth(tmp_path: Path) -> None:
    snapshot = {
        "step_index": 2, "time_seconds": 10,
        "koopman": {"forecast_entity_records": [
            {"observation_type": "precursor", "target_id": "C1", "kind": "dos",
             "mitre_technique_candidates": ["T1498.001"]},
        ], "signature_entity_records": [
            {"observation_type": "observed_signature", "target_id": "C2", "kind": "dos",
             "mitre_technique_candidates": ["T1498.001"], "feature_match": {"margin_ratio": 1.0}},
            {"observation_type": "observed_signature", "target_id": "C1", "kind": "bgp_hijack",
             "mitre_technique_candidates": []},
        ]},
        "attacks": {"events": [
            {"target_id": "M1_01", "kind": "brute_force", "mitre_technique_id": "T1110.001"},
        ]},
    }
    # Прогноз продолжающегося события и журнал сигнатур ссылаются на одно измерение.
    snapshot["koopman"]["forecast_entity_records"].append(snapshot["koopman"]["signature_entity_records"][0])
    export_detection_rules(None, [snapshot], tmp_path)

    logs = [
        json.loads(line)
        for line in (tmp_path / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["classification_source"] for row in logs] == [
        "precursor_heuristic", "observed_numeric_hypothesis", "observed_numeric_hypothesis",
        "simulated_active_event",
    ]
    assert logs[1]["feature_match"] == {"margin_ratio": 1.0}
    assert logs[2]["technique_id"] is None and logs[2]["attack_kind"] == "bgp_hijack"
    assert logs[3]["technique_id"] == "T1110.001"
    sigma = (tmp_path / "gnet9.yml").read_text(encoding="utf-8")
    assert "technique_id: 'T1498.001'" in sigma
    assert "attack_kind: 'bgp_hijack'" in sigma and "attack.bgp_hijack" not in sigma
    assert "classification_source:\n            - 'precursor_heuristic'" in sigma
    assert "'observed_numeric_hypothesis'" in sigma
    assert "T1110.001" not in sigma and "simulated_active_event" not in sigma


def test_yara_rule_matches_only_the_benign_fixture_and_clears_on_rerun(tmp_path: Path) -> None:
    snapshot = {
        "step_index": 2, "time_seconds": 10, "koopman": {"forecast_entity_records": []},
        "attacks": {"events": [
            {"target_id": "M1_01", "kind": "user_execution", "mitre_technique_id": "T1204"},
        ]},
    }
    export_detection_rules(None, [snapshot], tmp_path)

    report = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    yara = report["yara"]
    assert yara["rule_count"] == 1
    assert yara["semantics"] == "benign_simulator_fixture_only"
    rule = (tmp_path / yara["file"]).read_text(encoding="utf-8")
    fixture = (tmp_path / yara["sample"]).read_bytes()
    literal, size = re.search(
        r'\$fixture = (".+")\n    condition:\n        \$fixture at 0 and filesize == (\d+)', rule,
    ).groups()
    marker = json.loads(literal).encode("ascii")
    assert len(fixture) == int(size) and fixture == marker
    assert b"GNET9_SYNTHETIC_USER_EXECUTION" in fixture and b"T1204" in fixture
    assert b"T1204" != marker

    export_detection_rules(None, [], tmp_path)
    assert not (tmp_path / yara["file"]).exists()
    assert not (tmp_path / yara["sample"]).exists()


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


def test_cli_allows_external_warning_occupancy_budget():
    args = parse_args(["--validate-telemetry-csv", "telemetry.csv",
                       "--telemetry-max-warning-occupancy-ratio", "0.15"])
    assert args.telemetry_max_warning_occupancy_ratio == 0.15
