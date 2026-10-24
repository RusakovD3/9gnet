"""Краткий итог уже рассчитанного прогона, без повторных решений защиты."""

from __future__ import annotations

import json
import math
import platform
from importlib import metadata
from pathlib import Path

from .constants import ATTACK_KIND_DISPLAY_NAMES


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return value if math.isfinite(value) else None


def _maximum(values):
    available = [number for value in values if (number := _number(value)) is not None]
    return max(available) if available else None


def _sla_by_class(snapshots):
    result = {grade: {"observed_steps": 0, "minimum_compliance_ratio": None,
                      "worst_step": None, "worst_time_seconds": None,
                      "compliant_flows_at_worst_step": None, "accounted_flows_at_worst_step": None,
                      "assessment_coverage_ratio_at_worst_step": None}
              for grade in ("gold", "silver", "bronze")}
    for snapshot in snapshots:
        for tier in snapshot.get("attacks", {}).get("sla_restoration", {}).get("tiers", []):
            grade = tier.get("sla_grade")
            count = _number(tier.get("flow_count"))
            compliant = _number(tier.get("sla_compliant_flow_count"))
            if grade not in result or count is None or count <= 0 or compliant is None:
                continue
            item = result[grade]
            item["observed_steps"] += 1
            ratio = compliant / count
            if item["minimum_compliance_ratio"] is None or ratio < item["minimum_compliance_ratio"]:
                item.update(minimum_compliance_ratio=ratio, worst_step=snapshot.get("step_index"),
                    worst_time_seconds=snapshot.get("time_seconds"),
                    compliant_flows_at_worst_step=compliant, accounted_flows_at_worst_step=count,
                    assessment_coverage_ratio_at_worst_step=_number(tier.get("mean_slo_assessment_coverage_ratio")))
    return result


def _prediction_summary(evaluation):
    result = {key: evaluation.get(key) for key in (
        "attack_onset_count", "predicted_before_onset_count", "prediction_recall",
        "prediction_precision", "false_positive_warning_count", "minimum_observed_lead_seconds",
        "maximum_observed_lead_seconds", "prediction_slo_seconds", "prediction_slo_status")}
    result["evaluated"] = bool(evaluation.get("attack_onset_count"))
    if not result["evaluated"]:
        for key in ("prediction_recall", "prediction_precision", "minimum_observed_lead_seconds",
                    "maximum_observed_lead_seconds"):
            result[key] = None
        result["prediction_slo_status"] = ("NOT_APPLICABLE_NO_OBSERVED_ONSETS"
            if result["attack_onset_count"] == 0 else None)
    return result


def build_run_summary(dynamics):
    """Сохранять отсутствие наблюдений как null, а не как успех или нулевую нагрузку."""
    snapshots = dynamics.get("snapshots", [])
    evaluation = dynamics.get("koopman_evaluation", {})
    training = dynamics.get("training_summary", {})
    routing = [snapshot.get("attacks", {}).get("routing", {}) for snapshot in snapshots]
    precursor_step = training.get("first_precursor_step")
    accepted = [snapshot for snapshot in snapshots
                if snapshot.get("koopman", {}).get("nominal_training", {}).get("transition_accepted") is True]
    acceptance_available = any("nominal_training" in snapshot.get("koopman", {}) for snapshot in snapshots)
    finite_times = [time for snapshot in snapshots
                    if (time := _number(snapshot.get("time_seconds"))) is not None]
    observed_attacks = [snapshot for snapshot in snapshots if snapshot.get("attacks", {}).get("events")]
    return {
        "schema_version": 1,
        "scope": "offline_simulation_not_independent_external_validation",
        "runtime": {"python": platform.python_version(),
                    "packages": {name: metadata.version(name) for name in ("numpy", "networkx", "matplotlib")}},
        "config": dict(dynamics.get("config", {})),
        "observation": {
            "snapshot_count": len(snapshots),
            "first_time_seconds": min(finite_times) if finite_times else None,
            "last_time_seconds": max(finite_times) if finite_times else None,
            "ideal_t0_ok": dynamics.get("ideal_t0", {}).get("ok"),
            "first_observed_attack_step": observed_attacks[0].get("step_index") if observed_attacks else None,
        },
        "training": {
            "planned_initial_clean_steps": training.get("clean_training_steps"),
            "accepted_transitions_before_first_precursor": sum(
                1 for snapshot in accepted
                if precursor_step is None or snapshot.get("step_index", 0) < precursor_step)
                if acceptance_available else None,
            "accepted_transitions_total": training.get("nominal_transitions"),
            "acceptance_flags_available": acceptance_available,
            "updated_rule_applies_from_next_step": True,
        },
        "prediction": _prediction_summary(evaluation),
        "sla": {
            "weighted_time_mean_ratio": _number(dynamics.get("sla_resilience", {}).get("time_mean")),
            "weighted_minimum_ratio": _number(dynamics.get("sla_resilience", {}).get("minimum")),
            "by_class": _sla_by_class(snapshots),
            "accounting": "confirmed_attack_sources_excluded;all_other_isolated_flows_count_as_failures",
        },
        "protection": {
            "maximum_rerouted_flows_in_one_step": _maximum(row.get("rerouted_flow_count") for row in routing),
            "maximum_failover_flows_in_one_step": _maximum(row.get("failover_flow_count") for row in routing),
            "maximum_isolated_flows_in_one_step": _maximum(row.get("isolated_flow_count") for row in routing),
            "maximum_admitted_route_utilization_percent": _maximum(
                row.get("maximum_admitted_route_utilization_percent") for row in routing),
            "maximum_total_projected_utilization_percent": _maximum(
                row.get("maximum_projected_utilization_percent") for row in routing),
            "admission_limit_percent": next((row["maximum_safe_utilization_percent"] for row in routing
                                                if "maximum_safe_utilization_percent" in row), None),
            "scope": "maxima_of_step_counts_not_counts_of_distinct_lifetime_actions",
        },
        "events": [{key: record.get(key) for key in (
            "attack_id", "kind", "target_id", "first_step", "last_step", "actions", "response_stages")}
            for record in dynamics.get("attack_response_evaluation", [])],
        "interpretation_ru": [
            "Риск — расчётная оценка, не калиброванная вероятность атаки.",
            "Отсутствующее измерение записано как null; это не подтверждение исправности.",
            "Без наблюдённых начал атак точность предупреждения не оценивается.",
            "Купман обучается на исправных парах; тензорный канал — отдельная проверка согласованности.",
            "Ляпунов относится к выбранной линейной модели; внешний перенос точности не подтверждён.",
            "SLA относится к проверяемым требованиям; полнота их моделирования указана отдельно в assessment_coverage_ratio_at_worst_step.",
        ],
        "files": {"step_log": "step_log.csv", "complete_result": "network_dynamics.json",
                  "decisions": "algorithm_dialogue.md", "overview": "charts/dynamics_overview.png",
                  "capacity": "charts/capacity_bottlenecks.png"},
    }


def _display(value, *, percent=False):
    number = _number(value)
    if number is None:
        return "нет данных"
    return f"{number * 100:.2f}%" if percent else f"{number:g}"


def _percent_points(value):
    number = _number(value)
    return "нет данных" if number is None else f"{number:g}%"


def export_run_summary(dynamics, directory: Path):
    summary = build_run_summary(dynamics)
    directory.mkdir(parents=True, exist_ok=True)
    json_path, text_path = directory / "run_summary.json", directory / "run_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    prediction, training = summary["prediction"], summary["training"]
    lines = ["# Итог прогона GNet9", "",
        f"Сценарий: `{summary['config'].get('attack_scenario', 'не указан')}`. "
        f"Период наблюдений: {_display(summary['observation']['first_time_seconds'])}–"
        f"{_display(summary['observation']['last_time_seconds'])} с.", "",
        f"В обучение принято переходов: {_display(training['accepted_transitions_total'])}; "
        f"из них до первых предвестников — {_display(training['accepted_transitions_before_first_precursor'])}. "
        f"Начальный участок по плану: {_display(training['planned_initial_clean_steps'])} шагов.", ""]
    if prediction.get("attack_onset_count"):
        lines.append(f"Предупреждены {_display(prediction['predicted_before_onset_count'])} "
            f"из {_display(prediction['attack_onset_count'])} начал атак; "
            f"ложных эпизодов — {_display(prediction['false_positive_warning_count'])}; "
            f"минимальное упреждение — {_display(prediction['minimum_observed_lead_seconds'])} с.")
    else:
        lines.append("Наблюдённых начал атак нет: точность предупреждений по этому прогону не оценивается.")
    lines += ["", f"Средняя взвешенная доля выполнения SLA: {_display(summary['sla']['weighted_time_mean_ratio'], percent=True)}.",
              "", "| Класс | Худшее выполнение SLA | Шаг | Выполнили / учтены |", "|---|---:|---:|---:|"]
    for grade, label in (("gold", "Золотой"), ("silver", "Серебряный"), ("bronze", "Бронзовый")):
        row = summary["sla"]["by_class"][grade]
        lines.append(f"| {label} | {_display(row['minimum_compliance_ratio'], percent=True)} "
            f"| {_display(row['worst_step'])} | {_display(row['compliant_flows_at_worst_step'])} / "
            f"{_display(row['accounted_flows_at_worst_step'])} |")
    protection = summary["protection"]
    lines += ["", f"Максимальная загрузка ресурсов принятых новых маршрутов: "
        f"{_percent_points(protection['maximum_admitted_route_utilization_percent'])} "
        f"при пределе допуска {_percent_points(protection['admission_limit_percent'])}.",
        "Общая перегрузка сети и допуск новых маршрутов — разные показатели.", "", "## Наблюдённые события", ""]
    for event in summary["events"]:
        lines.append(f"- `{event['attack_id']}`: {ATTACK_KIND_DISPLAY_NAMES.get(event['kind'], event['kind'])}; "
            f"цель `{event['target_id']}`, шаги {_display(event['first_step'])}–{_display(event['last_step'])}.")
    if not summary["events"]:
        lines.append("События не наблюдались.")
    lines += ["", "## Как читать результат", ""]
    lines += [f"- {text}" for text in summary["interpretation_ru"]]
    lines += ["", "Детали: [журнал шагов](step_log.csv), [расчёты JSON](network_dynamics.json), "
        "[пояснения решений](algorithm_dialogue.md), [обзор](charts/dynamics_overview.png), "
        "[ресурсы](charts/capacity_bottlenecks.png).", ""]
    text_path.write_text("\n".join(lines), encoding="utf-8")
    return summary
