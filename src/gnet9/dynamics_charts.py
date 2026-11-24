"""Отдельные графики по наблюдениям GNet9."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
from .arbitrator import TRIGGER_BANDS
from .constants import DYNAMICS_CHART_FILES
from .flow_details import TRAFFIC_LABELS
from .plot_style import (PANEL, TEXT, MUTED, BLUE, GREEN,
                         AMBER, RED, PURPLE, save_figure, style_axis)
from .security_dashboard import ATTACK_COLORS


AggregateSeriesSpec = tuple[str, str, str, str, float]
NestedSeriesSpec = tuple[str, tuple[str, ...], float]

ATTACK_KIND_COLORS = ATTACK_COLORS
SLA_GRADES = {"gold": "золотой", "silver": "серебряный", "bronze": "бронзовый"}

AGGREGATE_SERIES: tuple[AggregateSeriesSpec, ...] = (
    ("sla_margin_min", "L1", "sla_margin", "min", 1.0),
    ("sla_margin_mean", "L1", "sla_margin", "mean", 1.0),
    ("l2_cpu_max", "L2", "cpu_load_percent", "max", 1.0),
)

TRAFFIC_SERIES: tuple[NestedSeriesSpec, ...] = (
    ("normal_arrival_rate_pps", ("traffic", "summary", "normal_arrival_rate_pps"), 1.0),
    ("attack_arrival_rate_pps", ("traffic", "summary", "attack_arrival_rate_pps"), 1.0),
    ("filtered_normal_arrival_rate_pps", ("koopman", "kalman_filter", "filtered_normal_arrival_rate_pps"), 1.0),
    ("raw_attack_rate_mbps", ("attacks", "raw_attack_rate_mbps"), 1.0),
    ("packet_count", ("traffic", "summary", "packet_count"), 1.0),
    ("traffic_rate_mbps", ("traffic", "summary", "offered_rate_mbps"), 1.0),
    ("traffic_delivered_rate_mbps", ("traffic", "summary", "delivered_rate_mbps"), 1.0),
    ("traffic_efficiency_percent", ("traffic", "summary", "protocol_efficiency_ratio"), 100.0),
)

ARBITRATOR_SERIES: tuple[NestedSeriesSpec, ...] = (
    ("remap_pressure", ("arbitrator", "analysis", "remap_pressure"), 1.0),
    ("lyapunov_remap_pressure", ("arbitrator", "analysis", "lyapunov_remap_pressure"), 1.0),
    ("koopman_forecast_risk", ("arbitrator", "analysis", "koopman_forecast_risk"), 1.0),
    (
        "state_hausdorff_normalized",
        ("arbitrator", "analysis", "state_hausdorff", "normalized_distance"),
        1.0,
    ),
)

ATTACK_SERIES: tuple[NestedSeriesSpec, ...] = (
    ("attack_intensity_percent", ("attacks", "maximum_intensity_ratio"), 100.0),
    ("precursor_confidence_percent", ("attacks", "precursor_confidence"), 100.0),
    ("legitimate_delivery_percent", ("attacks", "legitimate_delivery_ratio"), 100.0),
    ("koopman_warning_percent", ("koopman", "forecast_is_early_warning"), 100.0),
)


def export_dynamics_charts(
    dynamics: dict[str, Any],
    output_dir: Path,
    *,
    model=None,
    flow_snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Сохранить каждый график в отдельном PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    series = extract_dynamics_chart_series(dynamics)
    snapshots = dynamics.get("snapshots", [])
    detail = flow_snapshots if flow_snapshots is not None else snapshots
    charts = {key: output_dir / name for key, name in DYNAMICS_CHART_FILES.items()}
    _plot_koopman(series, dynamics, charts["koopman"])
    _plot_lyapunov(series, dynamics, charts["lyapunov"])
    _plot_native_traffic(series, detail, charts["native_traffic"])
    _plot_attack_traffic(snapshots, charts["attack_traffic"])
    _plot_sla_by_class(series, charts["sla"])
    _plot_target_sla(detail or snapshots, charts["target_sla"])
    _plot_service_forecast(dynamics.get("service_sla_evaluation", {}), charts["service_forecast"])
    _plot_timeline(dynamics, charts["timeline"])
    _plot_arbitrator(series, charts["arbitrator"])
    _plot_capacity_bottlenecks(model, detail, charts["capacity"])
    _plot_link_utilization(model, detail, charts["links"])
    _plot_resource_utilization(detail, charts["resources"])

    # Не оставлять в каталоге старые перегруженные диаграммы: так результат
    # текущего запуска не смешивается с историческими файлами.
    for legacy_name in (
        "dynamics_overview.png",
        "dynamics_health.png",
        "traffic_composition.png",
        "link_capacity.png",
        "dynamics_arbitrator.png",
        "dynamics_attacks.png",
        "dynamics_remapping.png",
        "attack_arrival_weibull.png",
        "dynamics_sla.png",
        "dynamics_load_stability.png",
        "dynamics_traffic.png",
    ):
        (output_dir / legacy_name).unlink(missing_ok=True)
    return charts


def extract_dynamics_chart_series(dynamics: dict[str, Any]) -> dict[str, list[float]]:
    """Извлечь сохранённые ряды; отсутствующий результат не заменять успехом."""
    snapshots = dynamics.get("snapshots", [])
    names = (
        "time_seconds", "steps",
        *(name for name, *_ in AGGREGATE_SERIES),
        *(name for name, *_ in TRAFFIC_SERIES),
        *(name for name, *_ in ARBITRATOR_SERIES),
        *(name for name, *_ in ATTACK_SERIES),
    )
    series = {name: [] for name in names}
    for step, snapshot in zip(_steps(snapshots), snapshots):
        series["steps"].append(step)
        series["time_seconds"].append(float(snapshot.get("time_seconds", 0.0)))
        for name, level, metric, statistic, multiplier in AGGREGATE_SERIES:
            series[name].append(_aggregate(snapshot, level, metric, statistic) * multiplier)
        for name, path, multiplier in (*TRAFFIC_SERIES, *ARBITRATOR_SERIES, *ATTACK_SERIES):
            series[name].append(_nested_float(snapshot, path, default=float("nan")) * multiplier)
    for grade in SLA_GRADES:
        values = []
        for snapshot in snapshots:
            tiers = snapshot.get("attacks", {}).get("sla_restoration", {}).get("tiers", [])
            tier = next((t for t in tiers if t.get("sla_grade") == grade), {})
            count = tier.get("flow_count", 0)
            compliant = tier.get("sla_compliant_flow_count")
            values.append(100 * compliant / count if count and compliant is not None else float("nan"))
        series[f"{grade}_sla_percent"] = values
    series["kalman_applied"] = [bool(s.get("koopman", {}).get("kalman_filter", {}).get("applied_to_nominal_observation")) for s in snapshots]
    return series


def extract_application_series(dynamics: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    """Оставить компактный программный доступ к сводке сервисного трафика."""
    snapshots = dynamics.get("snapshots", [])
    names = sorted({
        name
        for snapshot in snapshots
        for name in snapshot.get("traffic", {}).get("summary", {}).get("applications", {})
    })
    metrics = (
        "flow_count", "packet_count", "offered_rate_mbps", "carried_rate_mbps",
        "delivered_rate_mbps", "protocol_efficiency_ratio",
        "mean_one_way_latency_ms", "max_one_way_latency_ms",
    )
    result = {name: {metric: [] for metric in metrics} for name in names}
    for snapshot in snapshots:
        applications = snapshot.get("traffic", {}).get("summary", {}).get("applications", {})
        for name in names:
            values = applications.get(name, {})
            for metric in metrics:
                result[name][metric].append(_nested_float(values, (metric,), default=float("nan")))
    return result


def _steps(snapshots):
    return [int(snapshot.get("step_index", index)) for index, snapshot in enumerate(snapshots)]


def _step_axis(ax, steps):
    ax.set_xlabel("шаг модели", color=MUTED, fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if steps:
        ax.set_xlim(min(steps) - .5, max(steps) + .5)


def _plot_koopman(series, dynamics, path):
    fig, ax = plt.subplots(figsize=(11.5, 4.7))
    steps = series["steps"]
    snapshots = dynamics.get("snapshots", [])
    confidence = [value / 100 for value in series["precursor_confidence_percent"]]
    risk = [_nested_float(s, ("koopman", "forecast_risk_score"), default=fallback)
            for s, fallback in zip(snapshots, series["koopman_forecast_risk"])]
    resolved = _nested_float(dynamics, ("config", "resolved_warning_risk_threshold"), default=np.nan)
    threshold = [_nested_float(s, ("koopman", "warning_risk_threshold"), default=resolved) for s in snapshots]
    _plot_line(ax, steps, confidence, "наблюдаемый предвестник", BLUE, linestyle="--")
    _plot_line(ax, steps, risk, "оценка риска Купмана", AMBER)
    _plot_line(ax, steps, threshold, "действующий порог предупреждения", GREEN, linestyle="--")
    if any(np.isfinite(risk)):
        for index, boundary in enumerate(TRIGGER_BANDS["koopman"]):
            ax.axhline(boundary, color=MUTED, alpha=.66, linestyle=":", linewidth=1.1,
                       label="границы срочности арбитра" if index == 0 else None)
    _step_axis(ax, steps)
    _style_axes(ax, "Риск по прогнозу Купмана", "оценка 0–1", (0, 1), legend_columns=2)
    fig.text(.095, .035, "Оценка без единиц. Границы арбитра задают срочность проверки.",
             color=MUTED, fontsize=8)
    _save(fig, path, bottom=.32, left=.095)


def _plot_lyapunov(series, dynamics, path):
    fig, ax = plt.subplots(figsize=(11.5, 4.7))
    growth = []
    for snapshot in dynamics.get("snapshots", []):
        derivative = _nested_float(snapshot, ("koopman", "lyapunov_derivative_per_second"), default=np.nan)
        scale = _nested_float(snapshot, ("koopman", "lyapunov_remap_response_scale_per_second"), default=np.nan)
        growth.append(derivative / (abs(derivative) + scale) if scale > 0 and np.isfinite(derivative) else np.nan)
    _plot_line(ax, series["steps"], growth, "нормированная скорость изменения V", PURPLE)
    ax.axhline(0, color=MUTED, linestyle=":", linewidth=1.1)
    _step_axis(ax, series["steps"])
    _style_axes(ax, "Изменение функции Ляпунова", "скорость на шкале −1…+1", (-1, 1))
    fig.text(.095, .035, "Преобразование r / (|r| + s), где r — сохранённая скорость V за секунду, s — масштаб из результата.",
             color=MUTED, fontsize=8)
    _save(fig, path, bottom=.30, left=.095)


def _expected_native_rates(snapshots):
    rates = []
    for snapshot in snapshots:
        values = [_nested_float(flow, ("normal_expected_packet_rate_pps",), default=np.nan)
                  for flow in snapshot.get("traffic", {}).get("flows", [])
                  if not flow.get("is_attack_traffic")]
        rates.append(sum(values) if values else np.nan)
    return rates


def _plot_native_traffic(series, snapshots, path):
    fig, ax = plt.subplots(figsize=(11.5, 4.7))
    steps = series["steps"]
    actual = series["normal_arrival_rate_pps"]
    filtered = [value if enabled else np.nan for value, enabled in
                zip(series["filtered_normal_arrival_rate_pps"], series["kalman_applied"])]
    expected = dict(zip(_steps(snapshots), _expected_native_rates(snapshots)))
    expected_values = [expected.get(step, np.nan) for step in steps]
    _plot_line(ax, steps, actual, "обычные поступления", BLUE)
    _plot_line(ax, steps, expected_values, "расчётная средняя интенсивность", MUTED, linestyle=":")
    _plot_line(ax, steps, filtered, "Калман на исправных наблюдениях", GREEN)
    _step_axis(ax, steps)
    _style_axes(ax, "Обычный трафик и поступления по Пуассону", "пакетов/с", legend_columns=3)
    finite = [value for values in (actual, filtered, expected_values) for value in values if np.isfinite(value)]
    if finite:
        padding = max((max(finite) - min(finite)) * .15, max(finite) * .0001, 1e-9)
        ax.set_ylim(max(0, min(finite) - padding), max(finite) + padding)
    fig.text(.095, .035, "Шаг 0 — нагруженный эталон. Вертикальная шкала увеличена для сравнения.",
             color=MUTED, fontsize=8)
    _save(fig, path, bottom=.30, left=.095)


def _observed_event_id(event):
    return str(event.get("attack_id") or f"{event.get('kind')}:{event.get('target_id', '')}")


def _event_display_label(event):
    name = event.get("attack_label_ru") or _observed_event_id(event)
    target = event.get("target_id")
    return f"{name} · {target}" if target else name


def _event_packet_rates(snapshots):
    result = {}
    for step, snapshot in zip(_steps(snapshots), snapshots):
        for event in snapshot.get("attacks", {}).get("events", []):
            if event.get("is_recovery_effect"):
                continue
            key = _observed_event_id(event)
            series = result.setdefault(key, {"label": _event_display_label(event), "rates": {}})
            series["rates"][step] = _nested_float(event, ("effective_packet_rate_pps",), default=np.nan)
    return result


def _plot_attack_traffic(snapshots, path):
    events = _event_packet_rates(snapshots)
    rows = [(event["label"], event["rates"]) for event in events.values()]
    maximum = max((rate for _, rates in rows for rate in rates.values() if np.isfinite(rate)), default=1)
    _plot_observed_grid(
        _steps(snapshots), rows, path,
        title="Пакетная интенсивность каждого наблюдённого события",
        colorbar_label="предложено пакетов/с",
        caption="0 — воздействие без пакетов. Пустая клетка — событие или интенсивность не наблюдались.",
        norm=SymLogNorm(linthresh=10, vmin=0, vmax=max(1, maximum)),
        cmap="plasma",
        label_zero=True,
    )


def _plot_sla_by_class(series, path):
    fig, ax = plt.subplots(figsize=(11.5, 4.7))
    for grade, color, pattern in zip(SLA_GRADES, (AMBER, BLUE, PURPLE), ("-", "--", ":")):
        _plot_line(ax, series["steps"], series[f"{grade}_sla_percent"],
                   SLA_GRADES[grade], color, linestyle=pattern)
    _step_axis(ax, series["steps"])
    _style_axes(ax, "Выполнение SLA по всей сети", "% потоков своего класса", (0, 100), legend_columns=3)
    _save(fig, path, bottom=.30, left=.095)


def _target_sla_rows(snapshots):
    """Учитывать каждый оценённый легитимный поток один раз для каждой цели."""
    targets = {str(event["target_id"]) for snapshot in snapshots
               for event in snapshot.get("attacks", {}).get("events", []) if event.get("target_id")}
    rows = {(target, grade): {} for target in sorted(targets) for grade in SLA_GRADES}
    for step, snapshot in zip(_steps(snapshots), snapshots):
        events_by_id = {_observed_event_id(event): str(event["target_id"])
                        for event in snapshot.get("attacks", {}).get("events", []) if event.get("target_id")}
        for flow in snapshot.get("traffic", {}).get("flows", []):
            if flow.get("is_attack_traffic") or flow.get("security_excluded_from_sla_accounting"):
                continue
            grade = flow.get("sla_grade")
            compliant = flow.get("slo_evaluation", {}).get("compliant")
            if grade not in SLA_GRADES or compliant is None:
                continue
            associated = set(flow.get("route", []))
            associated.update(flow.get(key) for key in ("client_node", "server_node", "service_node"))
            associated.update(events_by_id.get(attack_id) for attack_id in flow.get("attack_ids", []))
            for target in associated & targets:
                counts = rows[target, grade]
                passed, total = counts.get(step, (0, 0))
                counts[step] = (passed + int(compliant), total + 1)
    return rows


def _plot_service_forecast(evaluation, path):
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9))
    cases = evaluation.get("cases", {})
    outcomes = (("hit", "Предупреждённое нарушение"), ("false_alarm", "Ложная тревога"),
                ("miss", "Пропущенное нарушение"))
    for ax, (key, label) in zip(axes, outcomes):
        case = cases.get(key)
        if case is None:
            _style_axes(ax, label, "", legend=False)
            ax.text(.5, .5, "Такого случая в прогоне нет", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        samples = case.get("measurements", [])
        times = [s["time_seconds"] for s in samples]
        values = [s["value"] for s in samples]
        budgets = [s["budget"] for s in samples]
        period = evaluation.get("step_seconds")
        # Пропуски должны разрывать линию наблюдений.
        for i in reversed(range(1, len(times))):
            if period and times[i] > times[i-1] + period:
                times.insert(i, np.nan)
                values.insert(i, np.nan)
                budgets.insert(i, np.nan)
        ax.plot(times, values, marker="o", markersize=3, color=BLUE, label="Измерение")
        ax.step(times, budgets, where="post", color=AMBER, linestyle=":", label="Порог SLA")
        if "issue_time_seconds" in case:
            ax.axvline(case["issue_time_seconds"], color=MUTED, linewidth=1, label="Выдан прогноз")
            ax.plot([case["issue_time_seconds"], case["target_time_seconds"]],
                    [case["measured_value"]] * 2, color=GREEN, linestyle="-.",
                    label="Последнее значение")
            if "trend_at_target" in case:
                ax.plot([case["issue_time_seconds"], case["target_time_seconds"]],
                        [case["measured_value"], case["trend_at_target"]], linestyle=":", color=TEXT,
                        label="Тренд потока")
            source = {"shared_route_delay_translation": "Наклон на общем пути",
                      "poisson_demand_delivery_koopman": "Спрос и доставка",
                      "healthy_margin_increment": "Изменение запаса"}.get(case.get("forecast_source"), "Прогноз Купмана")
            ax.plot([case["issue_time_seconds"], case["target_time_seconds"]],
                    [case["measured_value"], case["predicted_value"]],
                    linestyle="--", marker="D", markersize=4, color=PURPLE, label=source)
            if "predicted_budget" in case and case["predicted_budget"] != case.get("budget"):
                ax.scatter([case["target_time_seconds"]], [case["predicted_budget"]], marker="x",
                           color=AMBER, label="Порог при прогнозируемом спросе")
        impact_time = case.get("impact_time_seconds", evaluation.get("impact_time_seconds"))
        if impact_time is not None:
            ax.axvline(impact_time, color=AMBER, linestyle="-.", label="Начало воздействия")
        for i, t in enumerate(case.get("flow_impact_starts_seconds", ())):
            ax.axvline(t, color=AMBER, linestyle="-.", label=None if i else "Начало воздействия на поток")
        if key != "false_alarm":
            actual = case["time_seconds"] if key == "miss" else case["actual_time_seconds"]
            ax.axvline(actual, color=RED, linestyle="--", label="Начало нарушения")
        service = TRAFFIC_LABELS.get(case["service"], case["service"])
        _style_axes(ax, f"{label} · {service} · {case['flow_id']} · {case['label_ru']}",
                    case["unit"], legend=False)
        ax.legend(loc="best", fontsize=7, ncol=2, facecolor=PANEL, labelcolor=TEXT)
        ax.set_xlabel("Время модели, с", color=MUTED, fontsize=9)
    fig.suptitle("Прогноз Купмана по критериям SLA отдельных потоков", color=TEXT, fontsize=15)
    fig.subplots_adjust(left=.09, right=.97, hspace=.65, bottom=.07, top=.91)
    save_figure(fig, path)


def _plot_target_sla(snapshots, path):
    counts = _target_sla_rows(snapshots)
    active_steps = {}
    steps = _steps(snapshots)
    for step, snapshot in zip(steps, snapshots):
        for event in snapshot.get("attacks", {}).get("events", []):
            target = event.get("target_id")
            if target and not event.get("is_recovery_effect"):
                active_steps.setdefault(target, set()).add(step)
    targets_with_data = {target for (target, _), values in counts.items() if values}
    rows, highlights = [], []
    for (target, grade), values in counts.items():
        if values:
            rows.append((f"{target} · {SLA_GRADES[grade]}",
                         {step: 100 * passed / total for step, (passed, total) in values.items()}))
        elif target not in targets_with_data and grade == "gold":
            rows.append((f"{target} · нет данных", {}))
        else:
            continue
        highlights.extend((len(rows) - 1, step) for step in sorted(active_steps.get(target, ())))
    _plot_observed_grid(
        steps, rows, path,
        title="Выполнение SLA у атакованных узлов",
        colorbar_label="% оценённых потоков своего класса",
        caption="Учтены связанные с узлом потоки. Рамка — воздействие на узел. Пустая клетка — нет оценённых потоков.",
        norm=Normalize(0, 100),
        cmap="RdYlGn",
        highlights=highlights,
    )


def _plot_observed_grid(
    steps, rows, path, *, title, colorbar_label, caption, norm, cmap, highlights=(), label_zero=False,
):
    fig, ax = plt.subplots(figsize=(13.5, max(3.7, 2.25 + .31 * len(rows))))
    boxes, values = [], []
    for row, (_, series) in enumerate(rows):
        for step, value in series.items():
            if not np.isfinite(value):
                continue
            boxes.append(Rectangle((step - .46, row - .42), .92, .84))
            values.append(value)
    if values:
        heat = PatchCollection(boxes, cmap=cmap, norm=norm, linewidths=0)
        heat.set_array(np.asarray(values))
        ax.add_collection(heat)
        legend = fig.colorbar(heat, ax=ax, pad=.025, fraction=.035)
        legend.set_label(colorbar_label, color=MUTED, fontsize=9)
        legend.ax.tick_params(colors=MUTED, labelsize=8)
        for box, value in zip(boxes, values):
            if len(steps) <= 30 or (label_zero and value == 0):
                red, green, blue, _ = heat.to_rgba(value)
                color = PANEL if .2126 * red + .7152 * green + .0722 * blue > .48 else TEXT
                ax.text(box.get_x() + .46, box.get_y() + .42, f"{value:.4g}",
                        ha="center", va="center", color=color, fontsize=6.5)
    else:
        ax.text(.5, .5, "Нет наблюдений для графика", transform=ax.transAxes,
                color=MUTED, ha="center", fontsize=9)
    if highlights:
        outlines = [Rectangle((step - .46, row - .42), .92, .84) for row, step in highlights]
        ax.add_collection(PatchCollection(outlines, facecolor="none", edgecolor=TEXT,
                                          linewidth=.65, zorder=2))
    _step_axis(ax, steps)
    style_axis(ax, title)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.set_ylim(max(len(rows) - .5, .5), -.5)
    ax.grid(False)
    fig.text(.075, .035, caption, color=MUTED, fontsize=8)
    _save(fig, path, bottom=.18, left=.28)


def _plot_arbitrator(series, path):
    fig, ax = plt.subplots(figsize=(11.5, 4.7))
    for key, label, color in (("state_hausdorff_normalized", "расстояние Хаусдорфа", BLUE),
                              ("lyapunov_remap_pressure", "вклад Ляпунова в решение", PURPLE),
                              ("remap_pressure", "необходимость защиты", RED)):
        _plot_line(ax, series["steps"], series[key], label, color)
    _step_axis(ax, series["steps"])
    _style_axes(ax, "Основания решения арбитра", "оценка 0–1", (0, 1), legend_columns=3)
    _save(fig, path, bottom=.30, left=.095)


def _plot_capacity_bottlenecks(model, snapshots, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.7))
    top = sorted(_link_load_rows(model, snapshots),
                 key=lambda row: row["utilization_percent"], reverse=True)[:8]
    bars = ax.barh([row["label"] for row in reversed(top)], [row["utilization_percent"] for row in reversed(top)],
                   color=BLUE, height=.6)
    for bar, row in zip(bars, reversed(top)):
        ax.text(bar.get_width() + .8, bar.get_y() + bar.get_height() / 2,
                f"{row['utilization_percent']:.2f}% · шаг {row['step_index']}", va="center", color=TEXT, fontsize=9)
    if not top:
        ax.text(.5, .5, "Нет данных о нагрузке линий", transform=ax.transAxes, color=MUTED, ha="center")
    _style_axes(ax, "Восемь линий с наибольшей нагрузкой", "", legend=False)
    ax.set_xlabel("% пропускной способности линии", color=MUTED, fontsize=9)
    ax.set_xlim(0, max([100, *(row["utilization_percent"] * 1.3 for row in top)]))
    ax.grid(False, axis="y")
    ax.grid(axis="x", color=MUTED, linewidth=.5, alpha=.18)
    _save(fig, path, bottom=.17, left=.17)


def _plot_link_utilization(model, snapshots, path):
    fig, ax = plt.subplots(figsize=(11.5, 4.7))
    steps = _steps(snapshots)
    peak_series = [_snapshot_peak_load(model, snapshot) for snapshot in snapshots]
    _plot_line(ax, steps, peak_series, "максимальная нагрузка линии", BLUE)
    _step_axis(ax, steps)
    _style_axes(ax, "Максимальная нагрузка линий на каждом шаге", "% пропускной способности",
                _positive_limits(peak_series, minimum=100))
    _save(fig, path, bottom=.30, left=.095)


def _plot_resource_utilization(snapshots, path):
    fig, ax = plt.subplots(figsize=(11.5, 4.7))
    steps = _steps(snapshots)
    cpu = []
    for snapshot in snapshots:
        state = snapshot.get("traffic", {}).get("network_state", {})
        value = state.get("node_metric_maxima", {}).get("cpu_load_percent")
        if value is None:
            value = max((node.get("metrics", {}).get("cpu_load_percent", np.nan)
                         for node in state.get("nodes", [])), default=np.nan)
        cpu.append(float(value))
    target_resource = [_nested_float(snapshot, ("attacks", "maximum_target_utilization_percent"), default=np.nan)
                       for snapshot in snapshots]
    _plot_line(ax, steps, cpu, "CPU (процессор)", BLUE)
    _plot_line(ax, steps, target_resource, "ресурс атакованной цели", AMBER)
    _step_axis(ax, steps)
    _style_axes(ax, "Оценки загрузки ресурсов", "% расчётной ёмкости",
                _positive_limits(cpu, target_resource, minimum=100), legend_columns=2)
    fig.text(.095, .035, "CPU — оценка по очереди. Ресурс цели — результат расчёта воздействия.",
             color=MUTED, fontsize=8)
    _save(fig, path, bottom=.30, left=.095)


def _accepted_training(snapshot: dict[str, Any]) -> bool | None:
    """Принятие перехода читается из результата, а не из плана сценария."""
    koopman = snapshot.get("koopman", {})
    training = koopman.get("nominal_training", {})
    if "transition_accepted" in training:
        return bool(training["transition_accepted"])
    # Совместимость с прежним результатом: обновление применялось после
    # принятия исправной пары. Отсутствие этого поля не означает отказ.
    for source, key in ((snapshot.get("training", {}), "operator_updated"),
                        (koopman, "update_applied_for_next_step")):
        if key in source:
            return bool(source[key])
    return None


def _accepted_transition_count(dynamics: dict[str, Any]) -> int | None:
    flags = [_accepted_training(snapshot) for snapshot in dynamics.get("snapshots", [])]
    if any(flag is not None for flag in flags):
        return sum(flag is True for flag in flags)
    actual = dynamics.get("training_summary", {}).get("nominal_transitions")
    return int(actual) if actual is not None else None


def _training_windows(dynamics: dict[str, Any]) -> list[tuple[float, float]]:
    snapshots = dynamics.get("snapshots", [])
    windows = []
    for previous, current in zip(snapshots, snapshots[1:]):
        if _accepted_training(current) is True:
            windows.append((float(previous.get("time_seconds", 0)),
                            float(current.get("time_seconds", 0))))
    return windows


def _observation_period(dynamics: dict[str, Any]) -> float:
    configured = dynamics.get("config", {}).get("step_seconds")
    if configured is not None and float(configured) > 0:
        return float(configured)
    time = [float(s.get("time_seconds", 0)) for s in dynamics.get("snapshots", [])]
    return next((end - start for start, end in zip(time, time[1:]) if end > start), 1.0)


def _set_time_limits(axis, dynamics: dict[str, Any]) -> None:
    time = [float(s.get("time_seconds", 0)) for s in dynamics.get("snapshots", [])]
    if time:
        axis.set_xlim(min(time), max(time) + _observation_period(dynamics))


def _plot_timeline(dynamics: dict[str, Any], path: Path) -> None:
    """Полосы строятся по наблюдениям и принятым переходам."""
    windows = _attack_windows(dynamics)
    snapshots = dynamics.get("snapshots", [])
    resets = [snapshot for snapshot in snapshots if _kalman_window_reset(snapshot)]
    labels = ["Обучение принято", "Предупреждение"]
    if resets:
        labels.append("Сброс Калмана")
    event_labels = {window["attack_id"]: window["label"] for window in windows}
    event_rows = {key: row for row, key in enumerate(event_labels, start=len(labels))}
    labels.extend(event_labels.values())
    fig, axis = plt.subplots(figsize=(13.5, max(3.7, 2.2 + .32 * len(labels))))
    period = _observation_period(dynamics)
    for start, end in _training_windows(dynamics):
        axis.broken_barh([(start, end - start)], (-.3, .6), color=GREEN, alpha=.8)
    for snapshot in snapshots:
        if snapshot.get("koopman", {}).get("forecast_is_early_warning"):
            axis.broken_barh([(float(snapshot.get("time_seconds", 0)), period)], (.7, .6),
                             color=AMBER, alpha=.85)
    for snapshot in resets:
        axis.broken_barh([(float(snapshot.get("time_seconds", 0)), period)], (1.7, .6),
                         color=PURPLE, alpha=.85)
    for window in windows:
        lane = event_rows[window["attack_id"]]
        color = ATTACK_KIND_COLORS.get(window["kind"], RED)
        start, width = window["start"], window["end"] - window["start"]
        recovery = window["recovery"]
        axis.broken_barh([(start, width)], (lane - .3, .6),
                         facecolors="none" if recovery else color,
                         edgecolors=color if recovery else "none",
                         hatch="//" if recovery else None, alpha=.85)
    transitions = _accepted_transition_count(dynamics)
    count = "нет данных" if transitions is None else str(transitions)
    style_axis(axis, "Хронология наблюдений")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_ylim(len(labels) - .5, -.5)
    axis.grid(False, axis="y")
    axis.set_xlabel("время от t0, с", color=MUTED, fontsize=8)
    axis.tick_params(axis="y", length=0, labelsize=8)
    _set_time_limits(axis, dynamics)
    fig.text(.075, .070, f"Принято переходов в обучение: {count}. Наблюдалось событий: {len(event_rows)}.",
             color=TEXT, fontsize=8)
    fig.text(.075, .035, "Предупреждение отмечает момент выдачи прогноза. Штриховка обозначает восстановление.",
             color=MUTED, fontsize=8)
    _save(fig, path, bottom=.21, left=.27)


def _kalman_window_reset(snapshot):
    kalman = snapshot.get("koopman", {}).get("kalman_filter", {})
    local = kalman.get("local_observation") or {}
    return bool(local.get("window", {}).get("global_reset_reasons")) or any(
        local.get(kind, {}).get("covariance_reset_entity_count", 0) for kind in ("vertices", "edges")
    )


def _positive_limits(*series, minimum=1.0) -> tuple[float, float]:
    finite = [float(value) for values in series for value in values if np.isfinite(value)]
    return 0.0, max(minimum, max(finite, default=0) * 1.12)


def _attack_windows(dynamics: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    step_seconds = _observation_period(dynamics)
    for snapshot in dynamics.get("snapshots", []):
        for event in snapshot.get("attacks", {}).get("events", []):
            key = _observed_event_id(event)
            recovery = bool(event.get("is_recovery_effect"))
            time = float(snapshot.get("time_seconds", 0.0))
            windows = grouped.setdefault((key, recovery), [])
            if windows and time <= windows[-1]["end"] + 1e-9:
                windows[-1]["end"] = max(windows[-1]["end"], time + step_seconds)
            else:
                windows.append({"attack_id": key, "start": time, "end": time + step_seconds,
                                "label": _event_display_label(event),
                                "kind": str(event.get("kind", "")), "recovery": recovery})
    return [window for windows in grouped.values() for window in windows]


def _snapshot_link_loads(model, snapshot):
    """Скорости линий в Мбит/с из сохранённых показателей или полных потоков."""
    state = snapshot.get("traffic", {}).get("network_state", {})
    if "edge_vectors" in state:
        names = state.get("edge_metric_names", [])
        if "speed_mbps" in names and "capacity_mbps" in names:
            speed, capacity = names.index("speed_mbps"), names.index("capacity_mbps")
            return [(tuple(edge), values[speed], values[capacity])
                    for edge, values in zip(state.get("edge_ids", []), state["edge_vectors"])
                    if len(values) > max(speed, capacity)]
    if "edges" in state:
        return [((e["source"], e["target"]), e["metrics"]["speed_mbps"], e["metrics"]["capacity_mbps"])
                for e in state["edges"] if "speed_mbps" in e.get("metrics", {}) and "capacity_mbps" in e.get("metrics", {})]
    rates = {}
    for flow in snapshot.get("traffic", {}).get("flows", []):
        if flow.get("isolated") or flow.get("quarantined"):
            continue
        rate = int(flow.get("wire_bytes", 0)) * 8 / max(float(flow.get("interval_seconds", 1)), 1e-9) / 1e6
        for source, target in zip(flow.get("route", []), flow.get("route", [])[1:]):
            edge = tuple(sorted((source, target)))
            rates[edge] = rates.get(edge, 0) + rate
    return [(edge, rate, float(model.graph.edges[edge].get("capacity_mbps", 0)))
            for edge, rate in rates.items() if model is not None and model.graph.has_edge(*edge)]


def _snapshot_peak_load(model, snapshot):
    return max((rate / capacity * 100 for _, rate, capacity in _snapshot_link_loads(model, snapshot)
                if np.isfinite(rate) and np.isfinite(capacity) and capacity > 0), default=float("nan"))


def _link_load_rows(model, snapshots):
    """Для каждой линии сохранить её пик и шаг, на котором он достигнут."""
    peaks = {}
    for snapshot in snapshots:
        for edge, rate, capacity in _snapshot_link_loads(model, snapshot):
            if not np.isfinite(rate) or not np.isfinite(capacity) or capacity <= 0:
                continue
            edge = tuple(sorted(edge))
            utilization = rate / capacity * 100
            if edge not in peaks or utilization > peaks[edge]["utilization_percent"]:
                peaks[edge] = {"label": " ↔ ".join(edge), "utilization_percent": utilization,
                               "step_index": snapshot.get("step_index", 0)}
    return list(peaks.values())


def _plot_line(ax, x, y, label: str, color: str, *, linestyle: str = "-") -> None:
    finite = np.isfinite(y)
    if not finite.any():
        return
    # На длинном ряду один отсчёт по обе стороны разрыва виден как точка.
    isolated = finite & ~np.r_[False, finite[:-1]] & ~np.r_[finite[1:], False]
    markers = None if len(x) <= 30 else np.flatnonzero(isolated).tolist()
    ax.plot(x, y, linewidth=1.8,
            linestyle=linestyle, color=color, label=label,
            marker="o", markevery=markers, markersize=2.8, clip_on=False)


def _style_axes(
    ax,
    title: str,
    ylabel: str,
    ylim=None,
    *,
    legend: bool = True,
    legend_columns: int = 1,
) -> None:
    style_axis(ax, title, ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    handles, labels = ax.get_legend_handles_labels()
    if not handles and legend:
        ax.text(.5, .5, "Нет данных для этого показателя", transform=ax.transAxes,
                color=MUTED, ha="center", va="center", fontsize=9)
    if legend and handles:
        # Легенда находится под данными, а не поверх линий и столбцов.
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=legend_columns,
            facecolor=PANEL,
            edgecolor="none",
            labelcolor=TEXT,
            fontsize=8,
            frameon=False,
        )


def _save(
    fig,
    path: Path,
    *,
    bottom: float = 0.05,
    left: float = 0.075,
) -> None:
    fig.subplots_adjust(left=left, right=0.975, bottom=bottom, top=.89)
    save_figure(fig, path)


def _aggregate(snapshot: dict[str, Any], level: str, metric: str, statistic: str) -> float:
    values = snapshot.get("arbitrator", {}).get("level_metric_aggregates", {}).get(level, {}).get("metrics", {}).get(metric, {})
    return _nested_float(values, (statistic,), default=float("nan"))


def _nested_float(source: dict[str, Any], path: tuple[str, ...], *, default: float = 0.0) -> float:
    value: Any = source
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default
