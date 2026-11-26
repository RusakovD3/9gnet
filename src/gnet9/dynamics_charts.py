"""Отдельные графики по наблюдениям GNet9."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory
import numpy as np
from .arbitrator import TRIGGER_BANDS
from .constants import DYNAMICS_CHART_FILES
from .flow_details import TRAFFIC_LABELS
from .chart_style import BACKGROUND, TEXT, MUTED, BLUE, GREEN, AMBER, RED, PURPLE, style_axis
from .plot_style import save_figure


AggregateSeriesSpec = tuple[str, str, str, str, float]
NestedSeriesSpec = tuple[str, tuple[str, ...], float]

SLA_GRADES = {"gold": "золотой", "silver": "серебряный", "bronze": "бронзовый"}
SLA_SCOPE = "Доля учтённых абонентских потоков своего класса, выполнивших проверенные требования SLA на этом шаге."

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
    run = RunTimeAxis.from_result(dynamics)
    line_peaks, peak_series = _link_load_summary(model, detail)
    risk, threshold = _risk_series(series, dynamics)
    _plot_overview(series, risk, threshold, run, charts["overview"])
    _plot_koopman(series, risk, threshold, run, charts["koopman"])
    _plot_lyapunov(series, dynamics, run, charts["lyapunov"])
    _plot_native_traffic(series, detail, run, charts["native_traffic"])
    _plot_attack_traffic(snapshots, run, charts["attack_traffic"])
    _plot_sla_by_class(series, run, charts["sla"])
    _plot_target_sla(detail or snapshots, run, charts["target_sla"])
    _plot_service_forecast(dynamics.get("service_sla_evaluation", {}), charts["service_forecast"], run)
    _plot_timeline(dynamics, run, charts["timeline"])
    _plot_arbitrator(series, run, charts["arbitrator"])
    _plot_capacity_bottlenecks(line_peaks, charts["capacity"])
    _plot_link_utilization(_steps(detail), peak_series, run, charts["links"])
    _plot_resource_utilization(detail, run, charts["resources"])

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
        counts = []
        for snapshot in snapshots:
            tiers = snapshot.get("attacks", {}).get("sla_restoration", {}).get("tiers", [])
            tier = next((t for t in tiers if t.get("sla_grade") == grade), {})
            count = tier.get("flow_count")
            counts.append(float(count) if count is not None else np.nan)
            compliant = tier.get("sla_compliant_flow_count")
            values.append(100 * compliant / count if count and compliant is not None else float("nan"))
        series[f"{grade}_sla_percent"] = values
        series[f"{grade}_flow_count"] = counts
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


@dataclass
class RunTimeAxis:
    """Общее время наблюдений, в секундах от t0, для всех диаграмм прогона."""
    by_step: dict[int, float]
    period: float
    windows: list[dict]
    starts: tuple[float, ...]
    ends: tuple[float, ...]
    recoveries: tuple[float, ...]

    @classmethod
    def from_result(cls, dynamics):
        snapshots = dynamics.get("snapshots", [])
        period = _observation_period(dynamics)
        by_step = {step: float(s.get("time_seconds", step * period))
                   for step, s in zip(_steps(snapshots), snapshots)}
        windows = _attack_windows(dynamics, period, by_step)
        active = [w for w in windows if not w["recovery"]]
        observed = {by_step[step] for step, s in zip(by_step, snapshots)
                    if "events" in s.get("attacks", {})}
        recovered = {by_step[e["recovered_step"]]
                     for e in dynamics.get("resilience_evaluation", {}).get("episodes", [])
                     if e.get("recovered_step") in by_step}
        return cls(by_step, period, windows,
                   tuple(sorted({w["start"] for w in active})),
                   tuple(sorted({w["end"] for w in active if w["end"] in observed})),
                   tuple(sorted(recovered)))

    def seconds(self, steps):
        return [self.by_step.get(step, np.nan) for step in steps]


def _save_time(fig, axes, run, path, *, bottom=.28, top=.83, hspace=None, show_recovery=True):
    """Одинаковый прямоугольник проекции по X во всех PNG, включая карты."""
    fig.set_figwidth(14)
    fig.subplots_adjust(left=.24, right=.92, bottom=bottom, top=top, hspace=hspace)
    limits = (min(run.by_step.values()), max(run.by_step.values()) + run.period) if run.by_step else (0, run.period)
    guides = [(run.starts, RED, "-", "Начало атаки"),
              (run.ends, AMBER, "--", "Конец атаки")]
    if show_recovery:
        guides.append((run.recoveries, GREEN, ":", "Доставка восстановлена"))
    # У совпавших границ одна координата X. Делим высоту, чтобы видеть каждый тип.
    at_time = {}
    for index, (times, _, _, _) in enumerate(guides):
        for t in times:
            at_time.setdefault(t, []).append(index)
    handles = []
    for index, (times, color, style, label) in enumerate(guides):
        if not times:
            continue
        low = np.array([at_time[t].index(index) / len(at_time[t]) for t in times])
        high = low + np.array([1 / len(at_time[t]) for t in times])
        for ax in axes:
            ax.vlines(times, low, high, transform=ax.get_xaxis_transform(), colors=color,
                      linestyles=style, linewidth=.85, alpha=.52, zorder=1.8)
        handles.append(Line2D([], [], color=color, linestyle=style, label=label))
    for ax in axes:
        ax.set_xlim(*limits)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=run.period.is_integer()))
    axes[-1].set_xlabel("Время от t0, с", color=MUTED, fontsize=9)
    if handles:
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.925, .998),
                   ncol=len(handles), frameon=False, labelcolor=TEXT, fontsize=9)
    save_figure(fig, path, tight=False)


def _risk_series(series, dynamics):
    snapshots = dynamics.get("snapshots", [])
    risk = [_nested_float(s, ("koopman", "forecast_risk_score"), default=fallback)
            for s, fallback in zip(snapshots, series["koopman_forecast_risk"])]
    resolved = _nested_float(dynamics, ("config", "resolved_warning_risk_threshold"), default=np.nan)
    threshold = [_nested_float(s, ("koopman", "warning_risk_threshold"), default=resolved) for s in snapshots]
    return risk, threshold


def _plot_overview(series, risk, threshold, run, path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    times = run.seconds(series["steps"])
    _plot_line(axes[0], times, risk, "Риск Купмана", BLUE)
    _plot_line(axes[0], times, threshold, "Порог предупреждения", MUTED, linestyle="--")
    delivery = series["legitimate_delivery_percent"]
    _plot_line(axes[1], times, delivery, _minimum_caption(times, delivery), GREEN)
    warned = np.asarray(series["koopman_warning_percent"]) == 100
    issued = warned & np.isfinite(risk)
    if issued.any():
        axes[0].scatter(np.asarray(times)[issued], np.asarray(risk)[issued], marker="v",
                        color=AMBER, s=24, label="Выдано предупреждение", clip_on=False, zorder=3)
    _sla_lines(axes[2], series, times)
    panels = [
        ("Прогноз угрозы", "Оценка 0–1", (0, 1)),
        ("Доставка обычных пакетов", "% пакетов", None),
        ("Выполнение SLA по всей сети", "% потоков класса", (0, 100)),
    ]
    for ax, (title, unit, limits) in zip(axes, panels):
        _style_axes(ax, title, unit, limits, legend=False)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            _side_legend(ax, handles, labels)
        else:
            ax.text(.5, .5, "Нет измерений", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
    finite = np.asarray(delivery)[np.isfinite(delivery)]
    if finite.size:
        # Один процентный пункт даёт читаемую шкалу и при постоянной доставке.
        axes[1].set_ylim(max(0, min(99, finite.min()) - .1), 100.1)
    fig.text(.24, .025, SLA_SCOPE, color=MUTED, fontsize=8)
    _save_time(fig, axes, run, path, bottom=.12, top=.90, hspace=.63)


def _plot_koopman(series, risk, threshold, run, path):
    fig, ax = plt.subplots(figsize=(14, 4.7))
    steps = run.seconds(series["steps"])
    confidence = [value / 100 for value in series["precursor_confidence_percent"]]
    _plot_line(ax, steps, confidence, "наблюдаемый предвестник", BLUE, linestyle="--")
    _plot_line(ax, steps, risk, "оценка риска Купмана", AMBER)
    _plot_line(ax, steps, threshold, "действующий порог предупреждения", GREEN, linestyle="--")
    if any(np.isfinite(risk)):
        for index, boundary in enumerate(TRIGGER_BANDS["koopman"]):
            ax.axhline(boundary, color=MUTED, alpha=.66, linestyle=":", linewidth=1.1,
                       label="границы срочности арбитра" if index == 0 else None)
    _style_axes(ax, "Риск по прогнозу Купмана", "оценка 0–1", (0, 1))
    fig.text(.24, .035, "Оценка без единиц. Границы арбитра задают срочность проверки.",
             color=MUTED, fontsize=8)
    _save_time(fig, [ax], run, path, bottom=.21)


def _plot_lyapunov(series, dynamics, run, path):
    fig, ax = plt.subplots(figsize=(11.5, 4.7))
    growth = []
    for snapshot in dynamics.get("snapshots", []):
        derivative = _nested_float(snapshot, ("koopman", "lyapunov_derivative_per_second"), default=np.nan)
        scale = _nested_float(snapshot, ("koopman", "lyapunov_remap_response_scale_per_second"), default=np.nan)
        growth.append(derivative / (abs(derivative) + scale) if scale > 0 and np.isfinite(derivative) else np.nan)
    _plot_line(ax, run.seconds(series["steps"]), growth, "нормированная скорость изменения V", PURPLE)
    ax.axhline(0, color=MUTED, linestyle=":", linewidth=1.1)
    _style_axes(ax, "Изменение функции Ляпунова", "скорость на шкале −1…+1", (-1, 1))
    fig.text(.24, .035, "Преобразование r / (|r| + s), где r — сохранённая скорость V за секунду, s — масштаб из результата.",
             color=MUTED, fontsize=8)
    _save_time(fig, [ax], run, path, bottom=.21)


def _expected_native_rates(snapshots):
    rates = []
    for snapshot in snapshots:
        values = [_nested_float(flow, ("normal_expected_packet_rate_pps",), default=np.nan)
                  for flow in snapshot.get("traffic", {}).get("flows", [])
                  if not flow.get("is_attack_traffic")]
        rates.append(sum(values) if values else np.nan)
    return rates


def _plot_native_traffic(series, snapshots, run, path):
    fig, ax = plt.subplots(figsize=(11.5, 4.7))
    steps = series["steps"]
    actual = series["normal_arrival_rate_pps"]
    filtered = [value if enabled else np.nan for value, enabled in
                zip(series["filtered_normal_arrival_rate_pps"], series["kalman_applied"])]
    expected = dict(zip(_steps(snapshots), _expected_native_rates(snapshots)))
    expected_values = [expected.get(step, np.nan) for step in steps]
    steps = run.seconds(steps)
    _plot_line(ax, steps, actual, "обычные поступления", BLUE)
    _plot_line(ax, steps, expected_values, "расчётная средняя интенсивность", MUTED, linestyle=":")
    _plot_line(ax, steps, filtered, "Калман на исправных наблюдениях", GREEN)
    _style_axes(ax, "Обычный трафик и поступления по Пуассону", "пакетов/с")
    finite = [value for values in (actual, filtered, expected_values) for value in values if np.isfinite(value)]
    if finite:
        padding = max((max(finite) - min(finite)) * .15, max(finite) * .0001, 1e-9)
        ax.set_ylim(max(0, min(finite) - padding), max(finite) + padding)
    fig.text(.24, .035, "Шаг 0 — нагруженный эталон. Вертикальная шкала увеличена для сравнения.",
             color=MUTED, fontsize=8)
    _save_time(fig, [ax], run, path, bottom=.21)


def _observed_event_id(event):
    return str(event.get("attack_id") or f"{event.get('kind')}:{event.get('target_id', '')}")


def _event_display_label(event):
    name = event.get("attack_label_ru") or event.get("name_ru") or _observed_event_id(event)
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


def _plot_attack_traffic(snapshots, run, path):
    events = _event_packet_rates(snapshots)
    rows = [(event["label"], event["rates"]) for event in events.values()]
    maximum = max((rate for _, rates in rows for rate in rates.values() if np.isfinite(rate)), default=1)
    _plot_observed_grid(
        run, rows, path,
        title="Пакетная интенсивность каждого наблюдённого события",
        colorbar_label="предложено пакетов/с",
        caption="0 — воздействие без пакетов. Пустая клетка — событие или интенсивность не наблюдались.",
        norm=SymLogNorm(linthresh=10, vmin=0, vmax=max(1, maximum)),
        cmap="plasma",
        label_zero=True,
    )


def _count_range(counts):
    """Число потоков на шаге либо диапазон, если состав менялся."""
    observed = [int(count) for count in counts if np.isfinite(count)]
    if not observed:
        return "?"
    low, high = min(observed), max(observed)
    return str(low) if low == high else f"{low}–{high}"


def _minimum_caption(times, percentages):
    """Минимальная измеренная доля в процентах и время первого минимума в секундах."""
    observed = [(value, t) for value, t in zip(percentages, times) if np.isfinite(value) and np.isfinite(t)]
    if not observed:
        return ""
    minimum, first_time = min(observed)
    percent = np.format_float_positional(minimum, precision=2, trim="-").replace(".", ",")
    if minimum < 100 and percent == "100":
        percent = "<100"
    if all(value == minimum for value, _ in observed):
        return f"{percent} % по измерениям"
    return f"мин. {percent} % в {first_time:g} с"


def _sla_lines(ax, series, times):
    for grade, color, pattern in zip(SLA_GRADES, (AMBER, BLUE, PURPLE), ("-", "--", ":")):
        count = _count_range(series[f"{grade}_flow_count"])
        values = series[f"{grade}_sla_percent"]
        label = f"{SLA_GRADES[grade]} · потоков: {count}\n{_minimum_caption(times, values)}"
        _plot_line(ax, times, values, label, color, linestyle=pattern)


def _plot_sla_by_class(series, run, path):
    fig, ax = plt.subplots(figsize=(14, 4.7))
    _sla_lines(ax, series, run.seconds(series["steps"]))
    _style_axes(ax, "Выполнение SLA по всей сети", "% потоков своего класса", (0, 100))
    fig.text(.24, .035, SLA_SCOPE, color=MUTED, fontsize=8)
    _save_time(fig, [ax], run, path, bottom=.21)


def _target_sla_rows(snapshots):
    """Учитывать каждый оценённый легитимный поток один раз для каждой цели."""
    targets = {str(event["target_id"]) for snapshot in snapshots
               for event in snapshot.get("attacks", {}).get("events", []) if event.get("target_id")}
    if not targets:
        return {}
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


def _plot_service_forecast(evaluation, path, run=None):
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
            if run is None:  # В отдельном контрольном эксперименте сравниваются методы.
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
        _style_axes(ax, f"{label} · поток {case['flow_id']}\n{service} · {case['label_ru']}",
                    case["unit"], legend=False)
        if run is None:
            ax.legend(loc="best", fontsize=8, ncol=2, facecolor=BACKGROUND, labelcolor=TEXT)
            ax.set_xlabel("Время модели, с", color=MUTED, fontsize=9)
        else:
            _side_legend(ax, *ax.get_legend_handles_labels())
    if run is not None:
        _save_time(fig, list(axes), run, path, bottom=.08, top=.91, hspace=.8)
        return
    fig.suptitle("Прогноз Купмана по критериям SLA отдельных потоков", color=TEXT, fontsize=15)
    fig.subplots_adjust(left=.09, right=.97, hspace=.65, bottom=.07, top=.91)
    save_figure(fig, path)


def _plot_target_sla(snapshots, run, path):
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
            count = _count_range(total for _, total in values.values())
            rows.append((f"{target} · {SLA_GRADES[grade]} · потоков: {count}",
                         {step: 100 * passed / total for step, (passed, total) in values.items()}))
        elif target not in targets_with_data and grade == "gold":
            rows.append((f"{target} · нет данных", {}))
        else:
            continue
        highlights.extend((len(rows) - 1, step) for step in sorted(active_steps.get(target, ())))
    _plot_observed_grid(
        run, rows, path,
        title="Выполнение SLA у атакованных узлов",
        colorbar_label="% оценённых потоков своего класса",
        caption=("Учтены оценённые абонентские потоки класса, связанные с узлом через маршрут, конечную точку или событие.\n"
                 "Рамка — воздействие на узел. Пустая клетка — нет оценённых потоков."),
        norm=Normalize(0, 100),
        cmap=LinearSegmentedColormap.from_list("sla", ["#982e29", "#e39360", "#e4edf2"]),
        highlights=highlights,
    )


def _plot_observed_grid(
    run, rows, path, *, title, colorbar_label, caption, norm, cmap, highlights=(), label_zero=False,
):
    fig, ax = plt.subplots(figsize=(14, max(4.3, 2.25 + .44 * len(rows))))
    vertices, values, centers = [], [], []
    for row, (_, series) in enumerate(rows):
        for step, value in series.items():
            if step not in run.by_step or not np.isfinite(value):
                continue
            start = run.by_step[step]
            end = start + run.period
            vertices.append([(start, row - .42), (end, row - .42),
                             (end, row + .42), (start, row + .42)])
            if len(run.by_step) <= 30 or (label_zero and value == 0):
                centers.append(((start + end) / 2, row, value))
            values.append(value)
    if values:
        heat = PolyCollection(vertices, cmap=cmap, norm=norm, linewidths=0)
        heat.set_array(np.asarray(values))
        ax.add_collection(heat)
        legend = fig.colorbar(heat, cax=ax.inset_axes([1.015, 0, .013, 1]))
        legend.set_label(colorbar_label, color=MUTED, fontsize=9)
        legend.ax.tick_params(colors=MUTED, labelsize=8)
        for x, y, value in centers:
            red, green, blue, _ = heat.to_rgba(value)
            color = TEXT if .2126 * red + .7152 * green + .0722 * blue > .55 else BACKGROUND
            ax.text(x, y, f"{value:.4g}", ha="center", va="center", color=color, fontsize=6.5)
    else:
        ax.text(.5, .5, "Нет наблюдений для графика", transform=ax.transAxes,
                color=MUTED, ha="center", fontsize=9)
    if highlights:
        outlines = [Rectangle((run.by_step[step], row - .42), run.period, .84) for row, step in highlights if step in run.by_step]
        ax.add_collection(PatchCollection(outlines, facecolor="none", edgecolor=TEXT,
                                          linewidth=.65, zorder=2))
    style_axis(ax, title)
    ax.set_yticks(range(len(rows)), [fill(label, 42) for label, _ in rows])
    ax.set_ylim(max(len(rows) - .5, .5), -.5)
    ax.grid(False)
    fig.text(.24, .16 / fig.get_figheight(), caption, color=MUTED, fontsize=8)
    _save_time(fig, [ax], run, path, bottom=.85 / fig.get_figheight(), top=1 - .85 / fig.get_figheight())


def _plot_arbitrator(series, run, path):
    fig, ax = plt.subplots(figsize=(11.5, 4.7))
    for key, label, color in (("state_hausdorff_normalized", "расстояние Хаусдорфа", BLUE),
                              ("lyapunov_remap_pressure", "вклад Ляпунова в решение", PURPLE),
                              ("remap_pressure", "необходимость защиты", RED)):
        _plot_line(ax, run.seconds(series["steps"]), series[key], label, color)
    _style_axes(ax, "Основания решения арбитра", "оценка 0–1", (0, 1))
    _save_time(fig, [ax], run, path, bottom=.21)


def _plot_capacity_bottlenecks(line_peaks, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.7))
    top = sorted(line_peaks,
                 key=lambda row: row["utilization_percent"], reverse=True)[:8]
    loads = [row["utilization_percent"] for row in reversed(top)]
    bars = ax.barh([row["label"] for row in reversed(top)], loads,
                   color=[RED if load > 100 else BLUE for load in loads], height=.55)
    for bar, row in zip(bars, reversed(top)):
        ax.text(bar.get_width() + .8, bar.get_y() + bar.get_height() / 2,
                f"{row['utilization_percent']:.2f}% · шаг {row['step_index']}", va="center", color=TEXT, fontsize=9)
    if not top:
        ax.text(.5, .5, "Нет данных о нагрузке линий", transform=ax.transAxes, color=MUTED, ha="center")
    _style_axes(ax, "Восемь линий с наибольшей нагрузкой", "", legend=False)
    ax.set_xlabel("Предложенная нагрузка, % пропускной способности линии", color=MUTED, fontsize=9)
    ax.set_xlim(0, max([110, *(load * 1.3 for load in loads)]))
    if top:
        ax.axvline(100, color=MUTED, linestyle="--", linewidth=1, zorder=0)
        ax.set_title("Пунктир — ёмкость линии (100 %)", loc="right", color=MUTED, fontsize=10)
    ax.grid(False, axis="y")
    ax.grid(axis="x", color=MUTED, linewidth=.5, alpha=.18)
    _save(fig, path, bottom=.17, left=.17)


def _plot_link_utilization(steps, peak_series, run, path):
    fig, ax = plt.subplots(figsize=(11.5, 4.7))
    steps = run.seconds(steps)
    _plot_line(ax, steps, peak_series, "максимальная предложенная нагрузка", BLUE)
    ax.axhline(100, color=MUTED, linestyle="--", linewidth=1.1, label="пропускная способность линии")
    _style_axes(ax, "Максимальная предложенная нагрузка линий на каждом шаге", "% пропускной способности",
                _positive_limits(peak_series, minimum=100))
    _save_time(fig, [ax], run, path, bottom=.21)


def _cpu_peak(state):
    """Максимум CPU (%) и все узлы с таким измерением в этом снимке."""
    metric = "cpu_load_percent"
    if "nodes" in state:
        samples = [(node.get("id"), node.get("metrics", {}).get(metric)) for node in state["nodes"]]
    elif metric in state.get("vertex_metric_names", ()):
        index = state["vertex_metric_names"].index(metric)
        samples = [(node, row[index]) for node, row in zip(state.get("node_ids", ()), state.get("vertex_vectors", ()))
                   if index < len(row)]
    else:
        samples = []
    finite = [(node, value) for node, value in samples if value is not None and np.isfinite(value)]
    peak = state.get("node_metric_maxima", {}).get(metric)
    if peak is None:
        peak = max((value for _, value in finite), default=np.nan)
    return float(peak), tuple(sorted(str(node) for node, value in finite if node is not None and value == peak))


def _plot_resource_utilization(snapshots, run, path):
    times = run.seconds(_steps(snapshots))
    peaks = [_cpu_peak(s.get("traffic", {}).get("network_state", {})) for s in snapshots]
    cpu = [value for value, _ in peaks]
    leaders = {}
    for t, (value, names) in zip(times, peaks):
        if np.isfinite(value):
            for name in names or ("Узел не указан",):
                leaders.setdefault(name, []).append((t, run.period))
    if len(leaders) > 1:
        lane_height = .35 + .27 * len(leaders)
        fig, axes = plt.subplots(2, 1, figsize=(14, 4.9 + lane_height), sharex=True,
                                 gridspec_kw={"height_ratios": (2.5, lane_height)})
        ax, lane = axes
        style_axis(lane, "Узлы с максимальной загрузкой CPU на каждом шаге")
        for row, intervals in enumerate(leaders.values()):
            lane.broken_barh(intervals, (row - .32, .64), color=BLUE)
        lane.set_yticks(range(len(leaders)), list(leaders))
        lane.set_ylim(len(leaders) - .5, -.5)
        lane.grid(False)
    else:
        fig, ax = plt.subplots(figsize=(14, 4.7))
        axes = [ax]
    target_resource = [_nested_float(snapshot, ("attacks", "maximum_target_utilization_percent"), default=np.nan)
                       for snapshot in snapshots]
    _plot_line(ax, times, cpu, "Максимум CPU по всем узлам", BLUE)
    _plot_line(ax, times, target_resource, "ресурс атакованной цели", AMBER)
    title = "Максимальная загрузка процессора и ресурс атакованной цели"
    if len(leaders) == 1:
        title += f"  ·  CPU: {next(iter(leaders))}"
    _style_axes(ax, title, "% расчётной ёмкости",
                _positive_limits(cpu, target_resource, minimum=100))
    _save_time(fig, axes, run, path, bottom=.21, hspace=.65)


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


def _observation_period(dynamics: dict[str, Any]) -> float:
    configured = dynamics.get("config", {}).get("step_seconds")
    if configured is not None and float(configured) > 0:
        return float(configured)
    snapshots = dynamics.get("snapshots", [])
    stamps = [(step, float(s["time_seconds"])) for step, s in zip(_steps(snapshots), snapshots)
              if s.get("time_seconds") is not None]
    for (previous, start), (current, end) in zip(stamps, stamps[1:]):
        if end > start and current > previous:
            return (end - start) / (current - previous)
    return 1.0


def _step_bars(starts, period):
    """Объединить соседние отсчёты в непрерывные полосы (начало и ширина в секундах)."""
    bars = []
    for start in starts:
        if bars and start == bars[-1][0] + bars[-1][1]:
            first, width = bars[-1]
            bars[-1] = (first, width + period)
        else:
            bars.append((start, period))
    return bars


def _plot_timeline(dynamics, run, path):
    """Показать принятые переходы обучения, обычные пакеты и активные фазы атак."""
    snapshots = dynamics.get("snapshots", [])
    times = list(run.by_step.values())
    period = run.period
    training = [previous for previous, current, snapshot in zip(times, times[1:], snapshots[1:])
                if current - previous == period and _accepted_training(snapshot) is True]
    traffic = []
    traffic_measured = False
    for t, snapshot in zip(times, snapshots):
        summary = snapshot.get("traffic", {}).get("summary", {})
        rate = _nested_float(summary, ("normal_arrival_rate_pps",), default=np.nan)
        if not np.isfinite(rate):
            rate = _nested_float(summary, ("legitimate_packet_count",), default=np.nan) / period
        if np.isfinite(rate):
            traffic_measured = True
            if rate > 0:
                traffic.append(t)

    attacks = [window for window in run.windows if not window["recovery"]]
    event_labels = {w["attack_id"]: w["label"] for w in attacks}
    labels = ["Обучение Купмана", "Обычный трафик"]
    event_rows = {key: row for row, key in enumerate(event_labels, start=len(labels))}
    labels.extend(f"Атака · {name}" for name in event_labels.values())
    if not attacks:
        labels.append("Атаки")
    fig, ax = plt.subplots(figsize=(14, max(4.1, 2.5 + .39 * len(labels))))
    for starts, row, color in ((training, 0, GREEN), (traffic, 1, BLUE)):
        if starts:
            ax.broken_barh(_step_bars(starts, period), (row - .26, .52), color=color)
    for window in attacks:
        row = event_rows[window["attack_id"]]
        ax.broken_barh([(window["start"], window["end"] - window["start"])],
                       (row - .26, .52), color=RED)

    if not traffic:
        ax.text(.01, 1, "Обычных поступлений не зарегистрировано" if traffic_measured else "Нет данных о трафике",
                transform=ax.get_yaxis_transform(), va="center", color=MUTED, fontsize=9)
    if not attacks:
        ax.text(.01, 2, "Активные атаки не наблюдались", transform=ax.get_yaxis_transform(),
                va="center", color=MUTED, fontsize=9)

    style_axis(ax, "Обучение, обычный трафик и атаки")
    ax.set_yticks(range(len(labels)), [fill(label, 42) for label in labels])
    ax.set_ylim(len(labels) - .5, -.5)
    ax.grid(True, axis="y", color=MUTED, alpha=.15, linewidth=.6)
    ax.tick_params(axis="y", length=0, labelsize=9)
    transitions = _accepted_transition_count(dynamics)
    count = "нет данных" if transitions is None else str(transitions)
    fig.text(.24, .18 / fig.get_figheight(),
             f"Принято переходов в обучение: {count}. Обычный трафик отмечен на шагах с поступлением абонентских пакетов.",
             color=MUTED, fontsize=8)
    _save_time(fig, [ax], run, path, bottom=.85 / fig.get_figheight(), top=1 - .85 / fig.get_figheight(),
               show_recovery=False)


def _positive_limits(*series, minimum=1.0) -> tuple[float, float]:
    finite = [float(value) for values in series for value in values if np.isfinite(value)]
    return 0.0, max(minimum, max(finite, default=0) * 1.12)


def _attack_windows(dynamics, step_seconds, by_step) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for time, snapshot in zip(by_step.values(), dynamics.get("snapshots", [])):
        for event in snapshot.get("attacks", {}).get("events", []):
            key = _observed_event_id(event)
            recovery = bool(event.get("is_recovery_effect"))
            windows = grouped.setdefault((key, recovery), [])
            if windows and time <= windows[-1]["end"] + 1e-9:
                windows[-1]["end"] = max(windows[-1]["end"], time + step_seconds)
            else:
                windows.append({"attack_id": key, "start": time, "end": time + step_seconds,
                                "label": _event_display_label(event),
                                "kind": str(event.get("kind", "")), "recovery": recovery})
    return [window for windows in grouped.values() for window in windows]


def _snapshot_link_loads(model, snapshot):
    """Предложенная нагрузка линий в Мбит/с из снимка или потоков."""
    state = snapshot.get("traffic", {}).get("network_state", {})
    if "edge_vectors" in state:
        names = state.get("edge_metric_names", [])
        if "speed_mbps" in names and "capacity_mbps" in names:
            offered, capacity = names.index("speed_mbps"), names.index("capacity_mbps")
            return [(tuple(edge), values[offered], values[capacity])
                    for edge, values in zip(state.get("edge_ids", []), state["edge_vectors"])
                    if len(values) > max(offered, capacity)]
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


def _link_load_summary(model, snapshots):
    """Пики каждой линии и каждого снимка за один обход предложенной нагрузки."""
    per_step = []
    peaks = {}
    for snapshot in snapshots:
        maximum = None
        for edge, rate, capacity in _snapshot_link_loads(model, snapshot):
            if not np.isfinite(rate) or not np.isfinite(capacity) or capacity <= 0:
                continue
            edge = tuple(sorted(edge))
            utilization = rate / capacity * 100
            maximum = utilization if maximum is None else max(maximum, utilization)
            if edge not in peaks or utilization > peaks[edge]["utilization_percent"]:
                peaks[edge] = {"label": " ↔ ".join(edge), "utilization_percent": utilization,
                               "step_index": snapshot.get("step_index", 0)}
        per_step.append(np.nan if maximum is None else maximum)
    return list(peaks.values()), per_step


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
) -> None:
    style_axis(ax, title, ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    handles, labels = ax.get_legend_handles_labels()
    if not handles and legend:
        ax.text(.5, .5, "Нет данных для этого показателя", transform=ax.transAxes,
                color=MUTED, ha="center", va="center", fontsize=9)
    if legend and handles:
        _side_legend(ax, handles, labels)


def _side_legend(ax, handles, labels):
    wrapped = ["\n".join(fill(line, 26) for line in label.splitlines()) for label in labels]
    anchor = blended_transform_factory(ax.figure.transFigure, ax.transAxes)
    ax.legend(handles, wrapped, loc="center left", bbox_to_anchor=(.015, .5), bbox_transform=anchor,
              frameon=False, labelcolor=TEXT, fontsize=9)


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
