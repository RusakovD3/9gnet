"""Краткий результат не подменяет отсутствие данных успехом."""

from copy import deepcopy
import csv
import json

import pytest

from main import _dynamics_export_view, export_step_log, export_stationary_dynamics
from src.gnet9.decision_dialogue import build_algorithm_dialogue
from src.gnet9.run_report import build_run_summary, export_run_summary


def _snapshot(step, accepted, compliant=9, total=10):
    return {"step_index": step, "time_seconds": step * 2,
            "training": {"phase": "nominal_training", "nominal_transitions": int(accepted), "operator_updated": accepted},
            "koopman": {"nominal_training": {"transition_accepted": accepted}},
            "attacks": {"sla_restoration": {"tiers": [
                {"sla_grade": "gold", "flow_count": total, "sla_compliant_flow_count": compliant}]}}}


def test_planned_training_is_distinct_from_accepted_and_sla_uses_denominator(tmp_path):
    dynamics = {"config": {"attack_scenario": "none"},
        "training_summary": {"clean_training_steps": 10, "nominal_transitions": 1, "first_precursor_step": 2},
        "snapshots": [_snapshot(0, False, 10), _snapshot(1, True), _snapshot(2, False, 8)],
        "koopman_evaluation": {"attack_onset_count": 0, "prediction_precision": 0., "prediction_recall": 0.}}
    original = deepcopy(dynamics)
    report = export_run_summary(dynamics, tmp_path)
    assert report["training"]["planned_initial_clean_steps"] == 10
    assert report["training"]["accepted_transitions_before_first_precursor"] == 1
    assert report["prediction"]["prediction_precision"] is None
    assert report["prediction"]["prediction_slo_status"] == "NOT_APPLICABLE_NO_OBSERVED_ONSETS"
    row = report["sla"]["by_class"]["gold"]
    assert row["minimum_compliance_ratio"] == .8 and row["worst_step"] == 2
    assert row["accounted_flows_at_worst_step"] == 10
    assert report["sla"]["by_class"]["silver"]["minimum_compliance_ratio"] is None
    text = (tmp_path / "run_summary.md").read_text(encoding="utf-8")
    assert "80.00%" in text and "нет данных" in text
    assert json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8")) == report
    assert dynamics == original


def test_empty_sla_class_and_missing_observations_are_unavailable():
    report = build_run_summary({"snapshots": [_snapshot(1, False, 0, 0)]})
    assert report["sla"]["by_class"]["gold"]["minimum_compliance_ratio"] is None
    assert report["protection"]["maximum_total_projected_utilization_percent"] is None
    assert report["prediction"]["attack_onset_count"] is None
    assert report["sla"]["weighted_time_mean_ratio"] is None
    assert build_run_summary({})["training"]["accepted_transitions_before_first_precursor"] is None


def test_total_overload_is_not_relabelled_as_admitted_route_load():
    snapshot = _snapshot(1, False)
    snapshot["attacks"]["routing"] = {"maximum_projected_utilization_percent": 140.,
        "maximum_admitted_route_utilization_percent": 79., "maximum_safe_utilization_percent": 80.}
    protection = build_run_summary({"snapshots": [snapshot]})["protection"]
    assert protection["maximum_total_projected_utilization_percent"] == 140.
    assert protection["maximum_admitted_route_utilization_percent"] == 79.
    assert protection["admission_limit_percent"] == 80.


def test_export_metadata_keeps_existing_keys_and_input_immutable():
    dynamics = {"config": {"packet_detail": "flows"}, "snapshots": [
        {"traffic": {"flows": [{"payload_bytes": 1}], "summary": {"packet_count": 1}}}]}
    original = deepcopy(dynamics)
    exported = _dynamics_export_view(dynamics, "summary")
    assert exported["export_metadata"]["time_unit"] == "s"
    assert "flows" not in exported["snapshots"][0]["traffic"]
    assert exported["snapshots"][0]["traffic"]["summary"]["packet_count"] == 1
    assert dynamics == original


def test_csv_missing_packet_observations_are_blank_and_actions_are_executed(tmp_path):
    snapshot = _snapshot(1, False)
    snapshot["koopman"].update(lyapunov_derivative_per_second=0., forecast_risk_score=0.)
    snapshot["arbitrator"] = {"trigger_card": {"signals": {"hausdorff": {"value": 0.}},
        "attack_kind": "none", "response_stage": "observe", "reason_ru": "наблюдать"}}
    # Имя этапа берём из существующего публичного словаря представления.
    from src.gnet9.constants import RESPONSE_STAGE_DISPLAY_NAMES
    snapshot["arbitrator"]["trigger_card"]["response_stage"] = next(iter(RESPONSE_STAGE_DISPLAY_NAMES))
    snapshot["attacks"]["routing"] = {"actions": [{"action": "quarantine_attack_source"}]}
    target = tmp_path / "step_log.csv"
    export_step_log({"snapshots": [snapshot]}, target)
    with target.open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["обычный_трафик_пакетов_с"] == row["атаки_пакетов_с"] == ""
    assert row["отклонение_до_фильтра"] == ""
    assert row["SLA_gold"] == "0.9" and row["SLA_silver"] == ""
    assert row["исполненные_действия"] == "quarantine_attack_source"
    assert row["переход_принят_в_обучение"] == "нет"


def test_nonfinite_json_is_rejected_without_overwriting_previous_result(tmp_path, monkeypatch):
    import main
    target = tmp_path / "network_dynamics.json"
    target.write_text("previous result", encoding="utf-8")
    monkeypatch.setattr(main, "simulate_stationary_dynamics", lambda *args: {
        "config": {"packet_detail": "summary"}, "snapshots": [], "invalid_value": float("nan")})
    with pytest.raises(ValueError):
        export_stationary_dynamics(None, target)
    assert target.read_text(encoding="utf-8") == "previous result"
    assert not (tmp_path / "run_summary.json").exists()


def test_dialogue_distinguishes_proposed_plan_from_executed_actions():
    snapshot = _snapshot(1, False)
    snapshot["arbitrator"] = {"remap": {"action": "NO_REMAP"}}
    snapshot["attacks"]["routing"] = {"actions": [{"action": "quarantine_attack_source"}]}
    snapshot["koopman"]["kalman_filter"] = {"applied_to_nominal_observation": False}
    report = build_algorithm_dialogue({"snapshots": [snapshot]})
    turns = {turn["algorithm"]: turn for turn in report["records"][0]["turns"]}
    assert "Предложение арбитра" in turns["Выбор защитного действия"]["message_ru"]
    assert turns["Фактическое исполнение и качество услуг"]["evidence"]["actions"] == [{"action": "quarantine_attack_source"}]
    assert "Сглаживание обойдено" in turns["Нормировка и фильтр Калмана"]["message_ru"]
