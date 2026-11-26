"""Краткий итог уже рассчитанного прогона, без повторных решений защиты."""

from __future__ import annotations

import json
import math
import platform
from importlib import metadata
from pathlib import Path

from .constants import ATTACK_KIND_DISPLAY_NAMES, DYNAMICS_CHART_FILES, RESPONSE_STAGE_DISPLAY_NAMES
from .flow_details import TRAFFIC_LABELS
from .security_dashboard import peak_security_step


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return value if math.isfinite(value) else None


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
        "prediction_precision", "false_positive_warning_count", "right_censored_warning_count", "minimum_observed_lead_seconds",
        "maximum_observed_lead_seconds", "prediction_slo_seconds", "prediction_slo_status",
        "warning_episode_count", "observed_duration_seconds", "gap_duration_seconds",
        "warning_duration_seconds", "warning_occupancy_ratio", "false_warnings_per_simulated_hour")}
    result["evaluated"] = bool(evaluation.get("attack_onset_count"))
    if not result["evaluated"]:
        for key in ("prediction_recall", "prediction_precision", "minimum_observed_lead_seconds",
                    "maximum_observed_lead_seconds"):
            result[key] = None
        result["prediction_slo_status"] = ("NOT_APPLICABLE_NO_OBSERVED_ONSETS"
            if result["attack_onset_count"] == 0 else None)
    return result


def build_run_summary(dynamics, *, include_charts=True):
    """Сохранять отсутствие наблюдений как null, а не как успех или нулевую нагрузку."""
    snapshots = dynamics.get("snapshots", [])
    evaluation = dynamics.get("koopman_evaluation", {})
    training = dynamics.get("training_summary", {})
    precursor_step = training.get("first_precursor_step")
    accepted = [snapshot for snapshot in snapshots
                if snapshot.get("koopman", {}).get("nominal_training", {}).get("transition_accepted") is True]
    acceptance_available = any("nominal_training" in snapshot.get("koopman", {}) for snapshot in snapshots)
    finite_times = [time for snapshot in snapshots
                    if (time := _number(snapshot.get("time_seconds"))) is not None]
    observed_attacks = [snapshot for snapshot in snapshots if snapshot.get("attacks", {}).get("events")]
    action_snapshots = [s for s in snapshots
                        if s.get("arbitrator", {}).get("recommendation", {}).get("requires_human_action")]
    return {
        "schema_version": 2,
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
            "first_gold_disadvantage": dynamics.get("resilience_evaluation", {}).get("first_gold_disadvantage"),
            "worst_service": dynamics.get("service_sla_evaluation", {}).get("worst_service"),
            "accounting": "all_legitimate_flows_on_fixed_routes",
        },
        "recommendations": [
            {"step_index": snapshot.get("step_index"), "time_seconds": snapshot.get("time_seconds"),
             **snapshot["arbitrator"]["recommendation"]}
            for snapshot in action_snapshots
        ],
        "highlighted_recommendation_index": peak_security_step(action_snapshots) if action_snapshots else None,
        "service_sla_forecast": {
            k: v for k, v in dynamics.get("service_sla_evaluation", {}).items()
            if k in {"maximum_horizon_seconds", "methods", "by_horizon_seconds", "by_service", "cases",
                     "attribution", "route_stress", "operator_alert_evaluation", "operator_rule_version"}
        },
        "service_sla_runtime": dynamics.get("service_sla_runtime", {}),
        "operation_mode": "forecast_and_human_recommendations_only",
        "events": [{key: record.get(key) for key in (
            "attack_id", "kind", "attack_family", "attack_mechanism", "attack_label_ru",
            "target_id", "first_step", "last_step", "actions", "response_stages")}
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
                  "decisions": "algorithm_dialogue.md",
                  "detection_rules": "detection_rules/manifest.json",
                  **({key: f"charts/{name}" for key, name in DYNAMICS_CHART_FILES.items()} if include_charts else {}),
                  **({"security_dashboard": "charts/security_dashboard.png"}
                     if include_charts and any("traffic" in s for s in snapshots) else {})},
    }


def _display(value, *, percent=False):
    number = _number(value)
    if number is None:
        return "нет данных"
    return f"{number * 100:.2f}%" if percent else f"{number:g}"


def _service_forecast_text(summary):
    forecast = summary["service_sla_forecast"]
    lines = ["", "## Нарушения SLA по потокам", ""]
    worst = summary["sla"]["worst_service"]
    if worst is None:
        return lines + ["Измерения критериев потоков отсутствуют."]
    service = TRAFFIC_LABELS.get(worst["service"], worst["service"])
    if worst["failed_flows"]:
        criterion = worst["criterion_at_worst"]
        lines.append(f"Худшая доля выполнения SLA у услуги «{service}» в {worst['time_seconds']} с "
                     f"составила {_display(worst['compliance_ratio'], percent=True)}. "
                     f"Нарушения у {worst['failed_flows']} из {worst['assessed_flows']} потоков. "
                     f"У потока `{criterion['flow_id']}` показатель «{criterion['label_ru']}» "
                     f"составил {_display(criterion['measured_value'])} {criterion['unit']} "
                     f"при пороге {_display(criterion['budget'])} {criterion['unit']}.")
    else:
        lines.append("Все потоки выполнили проверенные требования на каждом наблюдавшемся шаге.")
    lines += ["", "| Услуга | Минимум выполнения SLA | Время минимума, с | Первый прогноз Купмана (поток и критерий) | Выдан / ожидался, с |",
              "|---|---:|---:|---|---:|"]
    for service, row in forecast.get("by_service", {}).items():
        warning = row["first_warning"]
        description = f"`{warning['flow_id']}` · {warning['label_ru']}" if warning else "Предупреждений нет"
        timing = f"{warning['issue_time_seconds']} / {warning['target_time_seconds']}" if warning else "—"
        lines.append(f"| {TRAFFIC_LABELS.get(service, service)} "
                     f"| {_display(row['minimum_compliance_ratio'], percent=True)} "
                     f"| {row['worst_time_seconds']} | {description} | {timing} |")
    lines += ["", "Измерения получены из пакетной модели. Для каждого потока Купман прогнозирует "
              "запас до действующего порога. Сигнатуры атак в этот прогноз не входят. "
              "Текущее нарушение учитывается только при первом наблюдаемом переходе через порог.", "",
              "| Горизонт, с | Метод | Предупреждены / начала | Ложные | Ложные / ч | Время под тревогой | Окно не проверено | Средняя ошибка момента, с¹ |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    method_names = {"koopman": "Купман", "persistence": "Последнее значение", "trend": "Линейный тренд"}
    for h, methods in forecast.get("by_horizon_seconds", {}).items():
        for method, row in methods.items():
            lines.append(f"| {h} | {method_names[method]} | {row['hits']} / {row['onsets']} "
                         f"| {row['false_alarms']} | {_display(row['false_alarms_per_observed_hour'])} "
                         f"| {_display(row['warning_occupancy_ratio'], percent=True)} | {row['censored']} "
                         f"| {_display(row['mean_absolute_timing_error_seconds'])} |")
    lines += ["", "¹ Среднее абсолютное расхождение прогноза и факта среди предупреждённых нарушений."]
    sources = forecast.get("methods", {}).get("koopman", {}).get("by_forecast_source", {})
    source_names = {"healthy_margin_increment": "Изменение запаса, обучение на исправных наблюдениях",
                    "shared_route_delay_translation": "Общий наклон задержки на пути с потерями",
                    "poisson_demand_delivery_koopman": "Спрос и эффективность доставки"}
    if sources:
        lines += ["", "| Источник первого предупреждения в окне | Попадания | Ложные | Исход неизвестен |",
                  "|---|---:|---:|---:|"]
        for source, counts in sources.items():
            lines.append(f"| {source_names.get(source, source)} | {counts['hits']} | "
                         f"{counts['false_alarms']} | {counts['censored']} |")
    lines += [
              "", f"Примеры ниже соответствуют окну {forecast.get('maximum_horizon_seconds', 0)} с. "
              "Нарушение того же критерия должно начаться до конца окна. "
              "Момент первого ожидаемого пересечения сравнивается с фактическим отдельно.",
              "", "| Исход прогноза Купмана | Услуга и поток | Критерий | Выдан / ожидался / начался, с |",
              "|---|---|---|---|"]
    for key, label in (("hit", "Предупреждён"), ("false_alarm", "Ложная тревога"), ("miss", "Пропущен")):
        case = forecast.get("cases", {}).get(key)
        if case is None:
            lines.append(f"| {label} | Такого случая в прогоне нет | — | — |")
            continue
        times = [case.get("issue_time_seconds"), case.get("target_time_seconds"),
                 case.get("actual_time_seconds", case.get("time_seconds"))]
        lines.append(f"| {label} | {TRAFFIC_LABELS.get(case['service'], case['service'])}, "
                     f"`{case['flow_id']}` | {case['label_ru']} | "
                     + " / ".join("—" if t is None else str(t) for t in times) + " |")
    if "service_forecast" in summary["files"]:
        lines += ["", "[График измерений, порогов и прогнозов по этим случаям](charts/service_sla_forecast.png)."]
    issued = forecast.get("operator_alert_evaluation", {})
    if issued:
        description = {
            2: "Правило учитывает изменение задержки потока, скачок показания датчика на пути и цену одного пакета.",
            3: "Прогноз учитывает очередь, спрос и потери медиапотоков. Для DNS проверяются нагрузка "
               "на сервер, число запросов и наблюдаемые тайм-ауты.",
            4: "Для прогноза битрейта проверяются средний спрос и его разброс у абонентов. "
               "Прогноз задержки учитывает очередь, прогноз тайм-аутов DNS — число запросов и тайм-аутов сервера.",
        }.get(forecast.get("operator_rule_version"), "Прежнее правило предупреждало о задержке медиапотока при потере.")
        lines += ["", description,
                  "", "| Окно зачёта, с | Предупреждены / все начала | Ложные / ч |",
                  "|---:|---:|---:|"]
        for horizon, row in issued.items():
            lines.append(f"| {horizon} | {row['hits']} / {row['onsets']} | "
                         f"{_display(row['false_alarms_per_observed_hour'])} |")
    conditional = forecast.get("route_stress", {}).get("methods", {}).get("route_stress_one_drop")
    if conditional:
        lines += ["", "По датчику на пути и счётчику потока отдельно проверяется, "
                  "нарушит ли порог ещё одна потеря при текущем спросе. "
                  f"Предупреждены {conditional['hits']} из {conditional['onsets']} начал по проверяемым "
                  f"критериям потерь, ложных тревог — {conditional['false_alarms']}, "
                  f"окон с неизвестным исходом — {conditional['censored']}."]
    runtime = summary["service_sla_runtime"]
    if runtime:
        lines += ["", f"Время вычисления прогноза на этом компьютере в среднем "
                  f"{_display(runtime['offline_compute_mean_ms'])} мс на шаг. "
                  "Доставка предупреждений оператору в опыте не измерялась."]
    return lines


def export_run_summary(dynamics, directory: Path):
    return write_run_summary(build_run_summary(dynamics), directory)


def write_run_summary(summary, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    json_path, text_path = directory / "run_summary.json", directory / "run_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    prediction, training = summary["prediction"], summary["training"]
    lines = ["# Итог прогона GNet9", "",
        f"Сценарий: `{summary['config'].get('attack_scenario', 'не указан')}`. "
        f"Период наблюдений: {_display(summary['observation']['first_time_seconds'])}–"
        f"{_display(summary['observation']['last_time_seconds'])} с.", ""]
    if "security_dashboard" in summary["files"]:
        lines += ["[Открыть обзор ключевого шага](charts/security_dashboard.png)", ""]
    if prediction.get("attack_onset_count"):
        lines.append(f"Предупреждены {_display(prediction['predicted_before_onset_count'])} "
            f"из {_display(prediction['attack_onset_count'])} начал атак; "
            f"ложных эпизодов — {_display(prediction['false_positive_warning_count'])}; "
            f"минимальное упреждение — {_display(prediction['minimum_observed_lead_seconds'])} с.")
    else:
        lines.append("Наблюдённых начал атак нет: точность предупреждений по этому прогону не оценивается.")
    if prediction.get("observed_duration_seconds") is not None:
        lines.append(
            f"Время под тревогой — {_display(prediction['warning_duration_seconds'])} с "
            f"из {_display(prediction['observed_duration_seconds'])} с наблюдения "
            f"({_display(prediction['warning_occupancy_ratio'], percent=True)}). "
            f"Ложных эпизодов за час наблюдения — {_display(prediction['false_warnings_per_simulated_hour'])}."
        )
    if prediction.get("right_censored_warning_count"):
        lines.append(f"Непроверенные предупреждения с окном за концом прогона — "
                     f"{prediction['right_censored_warning_count']}.")
    at_issue_horizon = summary["service_sla_forecast"].get("operator_alert_evaluation", {}).get("2")
    if at_issue_horizon:
        lines += ["", "**Предупреждения о нарушении SLA (требований к качеству услуг) за 2 с.** "
                  f"Подтверждено {at_issue_horizon['hits']} из {at_issue_horizon['onsets']} начал нарушений "
                  f"по потокам. Ложных предупреждений — {at_issue_horizon['false_alarms']}, "
                  f"с незавершённым окном проверки — {at_issue_horizon['censored']}."]
    lines += ["", f"Средняя взвешенная доля выполнения SLA: {_display(summary['sla']['weighted_time_mean_ratio'], percent=True)}.",
              "", "| Класс | Худшее выполнение SLA | Шаг | Выполнили / учтены | Охват требований¹ |",
              "|---|---:|---:|---:|---:|"]
    for grade, label in (("gold", "Золотой"), ("silver", "Серебряный"), ("bronze", "Бронзовый")):
        row = summary["sla"]["by_class"][grade]
        lines.append(f"| {label} | {_display(row['minimum_compliance_ratio'], percent=True)} "
            f"| {_display(row['worst_step'])} | {_display(row['compliant_flows_at_worst_step'])} / "
            f"{_display(row['accounted_flows_at_worst_step'])} | "
            f"{_display(row['assessment_coverage_ratio_at_worst_step'], percent=True)} |")
    lines += ["", "¹ Средняя по потокам доля обязательных требований, которые удалось проверить на указанном шаге."]
    exception = summary["sla"].get("first_gold_disadvantage")
    if exception:
        lines += ["", f"Впервые золотой класс уступил другому классу во время воздействия "
                  f"на шаге {exception['step_index']} ({exception['time_seconds']} с)."]
    lines += ["", "## Решение для оператора", ""]
    if summary["recommendations"]:
        item = summary["recommendations"][summary.get("highlighted_recommendation_index", 0)]
        stage = RESPONSE_STAGE_DISPLAY_NAMES.get(item.get("response_stage"), "проверка")
        targets = ", ".join(item.get("target_ids", ())) or "не определены"
        lines += [f"**Шаг {item['step_index']} · {item['time_seconds']} с · {stage}.** "
                  f"Наблюдаемые цели — {targets}.", ""]
        primary_action = item.get("primary_action_ru") or next(iter(item.get("steps_ru", ())), None)
        if primary_action:
            lines.append(primary_action)
        lines += ["", "Полный план действий для каждого шага доступен в [журнале](step_log.csv)."]
    else:
        lines.append("Сигналов, требующих защитных действий человека, не было.")
    lines += ["", "<details>", "<summary>Проверка прогноза, обучение и наблюдённые события</summary>", "",
              f"В обучение принято переходов: {_display(training['accepted_transitions_total'])}; "
              f"до первых предвестников — {_display(training['accepted_transitions_before_first_precursor'])}. "
              f"Начальный участок по плану: {_display(training['planned_initial_clean_steps'])} шагов."]
    lines += _service_forecast_text(summary)
    lines += ["", "## Наблюдённые события", ""]
    for event in summary["events"]:
        label = event["attack_label_ru"] or ATTACK_KIND_DISPLAY_NAMES.get(event["kind"], event["kind"])
        lines.append(f"- `{event['attack_id']}` · {label}. "
            f"цель `{event['target_id']}`, шаги {_display(event['first_step'])}–{_display(event['last_step'])}.")
    if not summary["events"]:
        lines.append("События не наблюдались.")
    lines += ["", "</details>"]
    lines += ["", "## Как читать результат", ""]
    lines += [f"- {text}" for text in summary["interpretation_ru"]]
    lines += ["", "Подробности в [журнале шагов](step_log.csv), [расчётах JSON](network_dynamics.json) "
              "и [пояснениях решений](algorithm_dialogue.md).",
              "Состав правил наблюдения приведён в [перечне Sigma, Suricata и YARA](detection_rules/manifest.json).", ""]
    if "koopman" in summary["files"]:
        lines += ["[Прогноз, доставка и качество услуг на одной шкале времени](charts/incident_overview.png)", "",
                  "[Купман](charts/koopman_risk.png) · [Ляпунов](charts/lyapunov_growth.png) · "
                  "[Хронология](charts/event_timeline.png) · [Арбитр](charts/arbitrator_signals.png)", "",
                  "[Обычный трафик](charts/native_traffic.png) · [Трафик атак](charts/attack_traffic.png) · "
                  "[SLA сети](charts/sla_by_class.png) · [SLA атакованных узлов](charts/attacked_node_sla.png)", "",
                  "[Линии с наибольшей предложенной нагрузкой](charts/capacity_bottlenecks.png) · "
                  "[Предложенная нагрузка по шагам](charts/link_utilization.png) · "
                  "[Ресурсы](charts/resource_utilization.png).", ""]
    text_path.write_text("\n".join(lines), encoding="utf-8")
    return summary
