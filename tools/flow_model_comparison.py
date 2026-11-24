"""Сравнить потоковые прогнозы и замороженные правила выдачи на одной записи."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gnet9.service_forecast import (
    ServiceSLAForecaster, _delivery_projection, _identity, _offered_rate, _score_window, operator_warnings,
    warning_meets_rule,
)
from tools.flow_replay import archive_info, observations
from tools.forecast_experiment import compare_at_alarm_budget


def _family(w):
    if w["unit"] == "kbit/s":
        return "rate"
    return "delay" if w["unit"] == "ms" else "other"


def _criterion(w):
    return w["criterion"]


def _filtered(views, method, rules, group_by=_family):
    previous = {}
    result = []
    for view in views:
        current, warnings = {}, []
        for w in view["warnings"]:
            if w["method"] != method:
                continue
            rule = rules.get(group_by(w))
            if not warning_meets_rule(w, rule):
                continue
            key = _identity(w)
            current[key] = previous.get(key, 0) + 1
            if current[key] >= rule["confirmations"]:
                warnings.append(w)
        result.append({**view, "warnings": warnings})
        previous = current
    return result


def _choose(views, method, groups, group_by):
    # При нулевом бюджете каждое непересекающееся семейство тоже должно
    # уложиться в ноль. При отсутствии подтверждённых попаданий воздерживаемся.
    selected = {}
    for group in groups:
        best = (0, 0)
        selected[group] = None
        for loss, issue_horizon in ((False, 20), (True, 20), (True, 2)):
            for n in (1, 2, 3, 5, 10):
                rule = {"confirmations": n, "require_loss": loss, "issue_horizon_seconds": issue_horizon}
                candidate = _filtered(views, method, {group: rule}, group_by)
                result, _ = _score_window(candidate, method, 20)
                quality = result["hits"], sum(result["lead_seconds"])
                if not result["false_alarms"] and quality > best:
                    selected[group], best = rule, quality
    return selected


def _score(views, method, horizon=20):
    row, examples = _score_window(views, method, horizon)
    return {key: row[key] for key in ("hits", "onsets", "false_alarms", "censored", "recall",
                                      "observed_seconds", "false_alarms_per_observed_hour",
                                      "warning_occupancy_ratio", "mean_absolute_timing_error_seconds")} | {
        "matching_window_seconds": horizon, "by_forecast_source": row["by_forecast_source"],
        "median_lead_seconds": median(row["lead_seconds"]) if row["hits"] else None,
        "first_hit": examples["hit"],
    }


def _rate_errors(history, totals, identified_totals, flows, model, step):
    # Ошибка точечного прогноза доставленного битрейта в исходных кбит/с.
    # Предыдущая запись хранит числа, выпущенные до поступления текущей.
    current = {}
    for f in flows:
        stream = f["traffic_kind"], f["client_node"]
        offer = _offered_rate(f)
        if f.get("arrival_process") != "poisson" or not offer:
            continue
        for name, m in f["slo_evaluation"]["metrics"].items():
            if (m["unit"] != "kbit/s" or not m["required"] or m["compliant"] is None
                    or m["comparison"] != "gte"):
                continue
            key = (*stream, name)
            value, budget = m["observed_value"], m["effective_budget_value"]
            estimates = {}
            prior = history[-1].get(key) if history else None
            if prior and stream in model.nominal_offers:
                avg = model.nominal_offers[stream][0]
                delivered, _, _ = _delivery_projection(offer, value, avg, m["budget_value"])
                # fit один и тот же у обоих вариантов и ещё не обновлён текущим кадром.
                num, den, _ = model.fit.get((stream[0], f["sla_grade"], name), (0., 0., 0))
                coefficient = max(-1., min(1., num / den)) if den > 1e-18 else None
                d_margin = (value-budget) - (prior[0]-prior[1])
                for h in (2, 10, 20):
                    # Штатный тренд продолжает изменение запаса при текущем пороге.
                    before = None
                    if coefficient is not None:
                        advance = sum(coefficient ** i for i in range(1, h//step + 1))
                        before = max(0., value + advance * d_margin)
                    estimates[h] = (delivered, max(0., value + h/step*d_margin), value, before)
            current[key] = value, budget, estimates, bool(f.get("observed_dropped_packets"))
            for h in (2, 10, 20):
                n = h // step
                if len(history) < n or key not in history[-n]:
                    continue
                old = history[-n][key]
                if h not in old[2] or any(key not in history[-j] for j in range(1, n)):
                    continue
                for subset in (["all", "loss_at_issue"] if old[3] else ["all"]):
                    total = totals[subset, stream[0], h]
                    total[0] += 1
                    predictions = old[2][h]
                    for i, predicted in enumerate(predictions[:3], 1):
                        total[i] += abs(predicted - value)
                    if predictions[3] is not None:
                        paired = identified_totals[subset, stream[0], h]
                        paired[0] += 1
                        for i, predicted in enumerate(predictions, 1):
                            paired[i] += abs(predicted - value)
    history.append(current)


def compare(path, calibration=None, *, per_criterion=False):
    info = archive_info(path)
    grouping = "criterion" if per_criterion else "family"
    if calibration is not None and calibration.get("policy_grouping", "family") != grouping:
        raise ValueError("Группировка предоставленной калибровки отличается от запрошенной")
    if info["step_seconds"] != 2:
        raise ValueError("Опыт задан для горизонтов 2/10/20 с при периоде 2 с")
    if calibration is None and (info["scenario"], info["seed"], info["vary_demand"], info["step_count"]) != (
            "predictive-demo", 11, False, 160):
        raise ValueError("Для подбора выделен только полный predictive-demo/11")
    models = {"before": ServiceSLAForecaster(2, delivery_model=False, route_delay=False),
              "after": ServiceSLAForecaster(2)}
    views = {name: [] for name in models}
    history, totals = deque(maxlen=10), defaultdict(lambda: [0, 0., 0., 0.])
    identified_totals = defaultdict(lambda: [0, 0., 0., 0., 0.])
    criteria = set()
    pending_delay = []
    delay_errors = {"after": 0., "trend": 0., "persistence": 0.}
    delay_pairs = 0
    for t, nominal, flows in observations(path):
        if calibration is None and per_criterion:
            criteria.update(name for f in flows for name, metric in f["slo_evaluation"]["metrics"].items()
                            if metric["required"] and metric["compliant"] is not None)
        _rate_errors(history, totals, identified_totals, flows, models["after"], 2)
        for variant, model in models.items():
            views[variant].append(model.observe(flows, time_seconds=t, nominal=nominal))
        # Заявленная точка рабочего правила — ровно следующий отсчёт (2 с).
        for w in pending_delay:
            actual = models["after"].previous_delay.get(_identity(w))
            if actual is None or t != w["target_time_seconds"]:
                continue
            delay_pairs += 1
            value = w["measured_value"]
            delay_errors["after"] += abs(w["predicted_value"] - actual)
            delay_errors["trend"] += abs(max(0., value - w["margin_delta"]) - actual)
            delay_errors["persistence"] += abs(value - actual)
        pending_delay = operator_warnings(views["after"][-1])
    table, policies, repeats = {}, {}, {}
    group_by = _criterion if per_criterion else _family
    groups = sorted(criteria) if per_criterion else ("rate", "delay", "other")
    for label, variant, method in (("before", "before", "koopman"), ("after", "after", "koopman"),
                                   ("trend", "after", "trend")):
        series = views[variant]
        policies[label] = (_choose(series, method, groups, group_by) if calibration is None
                           else calibration["stratified_policy"][label])
        original = (None if calibration is None else {"confirmation_samples":
                    {method: calibration["repeat_policy"][label]}})
        comparison = compare_at_alarm_budget(series, original)
        repeats[label] = comparison["confirmation_samples"][method]
        repeat_rule = {"confirmations": repeats[label], "require_loss": False} if repeats[label] else None
        selected_series = _filtered(series, method, policies[label], group_by)
        table[label] = {"raw": _score(series, method),
                        "repeat": _score(_filtered(series, method,
                            dict.fromkeys(("rate", "delay", "other"), repeat_rule)), method),
                        "stratified": _score(selected_series, method)}
        if per_criterion:
            table[label]["stratified_by_matching_window"] = {
                str(h): _score(selected_series, method, h) for h in (2, 10, 20)}
    return {"archive": path.name, "seed": info["seed"], "scenario": info["scenario"],
            "vary_demand": info["vary_demand"], "calibration_run": "predictive-demo/11",
            "validation_budget_false_per_observed_hour": 0, "repeat_policy": repeats,
            "policy_grouping": grouping, "stratified_policy": policies, "warnings": table,
            "issued_media_latency_mae_ms": {"n": delay_pairs, "horizon_seconds": 2,
                **{m: v / delay_pairs if delay_pairs else None for m, v in delay_errors.items()}},
            "rate_mae_kbps": [
                {"subset": subset, "service": service, "horizon_seconds": h, "n": row[0],
                 **{m: v/row[0] for m, v in zip(("after", "trend", "persistence"), row[1:])}}
                for (subset, service, h), row in sorted(totals.items())],
            "rate_mae_kbps_identified": [
                {"subset": subset, "service": service, "horizon_seconds": h, "n": row[0],
                 **{m: v/row[0] for m, v in zip(("after", "trend", "persistence", "before"), row[1:])}}
                for (subset, service, h), row in sorted(identified_totals.items())]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--per-criterion", action="store_true",
                        help="калибровать и оценивать правила отдельно для каждого критерия SLA")
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf8")) if args.calibration else None
    result = compare(args.archive, calibration, per_criterion=args.per_criterion)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
