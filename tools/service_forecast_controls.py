"""Контроли по рядам измерений задержки Gold-телефонии (мс)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gnet9.service_forecast import ServiceSLAForecaster, evaluate_service_forecasts


def flow(value):
    budget = 150.0
    return {"flow_id": "voice-1", "traffic_kind": "voice", "sla_grade": "gold",
            "slo_evaluation": {"metrics": {"latency_ms": {
                "unit": "ms", "budget_value": budget, "observed_value": value,
                "label_ru": "задержка", "value_origin": "synthetic_control_series",
                "compliant": value <= budget, "comparison": "lte", "required": True}}}}


def replay(values):
    model = ServiceSLAForecaster(2, retain_trace=True)
    # Отдельная исправная калибровочная траектория с плавно затухающими приращениями.
    for i in range(25):
        model.observe([flow(85 - 40 * .94 ** i)], time_seconds=2*i)
    model.observe([], time_seconds=50)
    views = []
    for i, v in enumerate(values):
        views.append(model.observe([] if v is None else [flow(v)], time_seconds=52 + 2*i, nominal=False))
    return views, model


def score(views, limits):
    selected = []
    for view in views:
        warnings = [w for w in view["warnings"] if w["horizon_seconds"] <= limits[w["method"]]]
        selected.append({**view, "warnings": warnings})
    return evaluate_service_forecasts(selected)


def run():
    # Единственный параметр отбора — предельный горизонт выдачи. Бюджет на
    # валидации 0 ложных эпизодов, при равной полноте берём более длинное окно.
    validation = [[110]*6 + [110+3*i for i in range(17)] + [158]*12,
                  [108]*6 + [114, 118, 122, 125, 126] + [123]*18]
    methods = ("koopman", "trend", "persistence")
    limits = {}
    for method in methods:
        candidates = []
        for h in (0, 2, 10, 20):
            results = [score(replay(v)[0], dict.fromkeys(methods, h))["methods"][method] for v in validation]
            if sum(r["false_alarms"] for r in results) == 0:
                candidates.append((sum(r["hits"] for r in results), h))
        limits[method] = max(candidates)[1]

    base = [110]*10
    gradual = [115, 120, 125, 130, 135, 140]
    cases = {
        "длинный предвестник": (base + gradual + [145,150,155,160], 6),
        "такой же сигнал без нарушения": (base + gradual + [138,136,133,130], None),
        "короткий сильный предвестник": (base + [120,140,165,170], 1),
        "легитимный всплеск": (base + [120,135,149,125,115], None),
        "физический отказ без предвестника": (base + [180,178,175], 0),
        "разрыв телеметрии": (base + gradual + [None,None,None,170,175,165], 6),
        "отложенный колебательный режим": (base + [127,117,137,129,148,135,153,143,161], 3),
    }
    rows = {}
    for name, (signal, offset) in cases.items():
        series = signal + [110]*15
        views, model = replay(series)
        result = evaluate_service_forecasts(views)
        model.case_traces(result)
        result["equal_validation_budget"] = score(views, limits)["methods"]
        result["source_values_ms"] = series
        result["time_origin_seconds"] = 52
        result["cause_hypothesis"] = "unknown"
        # Метки присоединяются после выпуска всех прогнозов.
        result["impact_time_seconds"] = None if offset is None else 52 + 2*(len(base)+offset)
        result["first_observed_breach_seconds"] = next(
            (v["time_seconds"] for v, x in zip(views,series) if x is not None and x > 150), None)
        impact = result["impact_time_seconds"]
        measured = result["first_observed_breach_seconds"]
        result["impact_to_first_impaired_measurement_seconds"] = (
            measured - impact if measured is not None and impact is not None else None)
        rows[name] = result
    return {"validation_budget_false_alarms_per_hour": 0, "selected_horizon_seconds": limits,
            "test_cases": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    from src.gnet9.dynamics_charts import _plot_service_forecast
    cases = {}
    for outcome, scenario in (("hit", "длинный предвестник"), ("false_alarm", "такой же сигнал без нарушения"),
                              ("miss", "физический отказ без предвестника")):
        case = result["test_cases"][scenario]
        cases[outcome] = {**case["cases"][outcome], "impact_time_seconds": case["impact_time_seconds"]}
    _plot_service_forecast({"cases": cases, "step_seconds": 2}, args.output.with_suffix(".png"))
