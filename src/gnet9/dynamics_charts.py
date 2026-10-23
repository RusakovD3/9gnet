"""Небольшой набор читаемых графиков динамики GNet9.

Данные для каждого графика берутся только из ``network_dynamics.json``:
временные ряды — из ``snapshots``, а загрузка каналов — из сохранённых
модельных потоков. Модуль намеренно создаёт два графика вместо набора похожих
панелей: оператору важнее увидеть ход события и реальные узкие места, чем
повторять одну и ту же метрику в нескольких дашбордах.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from .arbitrator import TRIGGER_BANDS
from .remapping import MAX_REMAP_UTILIZATION_PERCENT
from .plot_style import (BACKGROUND, PANEL, TEXT, MUTED, BLUE, GREEN,
                         AMBER, RED, PURPLE, save_figure, style_axis)


AggregateSeriesSpec = tuple[str, str, str, str, float]
NestedSeriesSpec = tuple[str, tuple[str, ...], float]

ATTACK_KIND_COLORS = {
    "dos": "#ef4444",
    "ddos": "#b91c1c",
    "syn_flood": "#f43f5e",
    "brute_force": "#f59e0b",
    "power_attack": "#facc15",
}

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
    ("admitted_attack_rate_mbps", ("attacks", "attack_rate_mbps"), 1.0),
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
    ("gold_availability_percent", ("attacks", "routing", "by_sla", "gold", "availability_ratio"), 100.0),
    ("koopman_warning_percent", ("koopman", "forecast_is_early_warning"), 100.0),
    ("rerouted_flows", ("attacks", "routing", "rerouted_flow_count"), 1.0),
    ("failover_flows", ("attacks", "routing", "failover_flow_count"), 1.0),
    ("isolated_flows", ("attacks", "routing", "isolated_flow_count"), 1.0),
    ("post_remap_utilization_percent", ("attacks", "routing", "maximum_projected_utilization_percent"), 1.0),
)


def export_dynamics_charts(
    dynamics: dict[str, Any],
    output_dir: Path,
    *,
    model=None,
    flow_snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Экспортировать две диаграммы без дублирования показателей."""
    output_dir.mkdir(parents=True, exist_ok=True)
    series = extract_dynamics_chart_series(dynamics)
    charts = {
        "overview": output_dir / "dynamics_overview.png",
        "capacity": output_dir / "capacity_bottlenecks.png",
    }
    _plot_dynamics_overview(series, dynamics, charts["overview"])
    _plot_capacity_bottlenecks(
        model,
        flow_snapshots or dynamics.get("snapshots", []),
        charts["capacity"],
    )

    # Не оставлять в каталоге старые перегруженные диаграммы: так результат
    # текущего запуска не смешивается с историческими файлами.
    for legacy_name in (
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
        "time_seconds",
        *(name for name, *_ in AGGREGATE_SERIES),
        *(name for name, *_ in TRAFFIC_SERIES),
        *(name for name, *_ in ARBITRATOR_SERIES),
        *(name for name, *_ in ATTACK_SERIES),
    )
    series = {name: [] for name in names}
    for snapshot in snapshots:
        series["time_seconds"].append(float(snapshot.get("time_seconds", 0.0)))
        for name, level, metric, statistic, multiplier in AGGREGATE_SERIES:
            series[name].append(_aggregate(snapshot, level, metric, statistic) * multiplier)
        for name, path, multiplier in (*TRAFFIC_SERIES, *ARBITRATOR_SERIES, *ATTACK_SERIES):
            default = 1.0 if name in {"gold_availability_percent", "legitimate_delivery_percent"} else 0.0
            series[name].append(_nested_float(snapshot, path, default=default) * multiplier)
    for grade in ("gold", "silver", "bronze"):
        values = []
        for snapshot in snapshots:
            tiers = snapshot.get("attacks", {}).get("sla_restoration", {}).get("tiers", [])
            tier = next((t for t in tiers if t["sla_grade"] == grade), {})
            count = tier.get("flow_count", 0)
            values.append(100 * tier["sla_compliant_flow_count"] / count if count else float("nan"))
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
                result[name][metric].append(float(values.get(metric, 0.0)))
    return result


def _plot_dynamics_overview(series, dynamics, path: Path) -> None:
    """Сигнал → оценка → качество услуг → меры → обычный трафик → атаки."""
    fig, axes = _dashboard("GNet9 · состояние сети и работа защиты", 3, 2, (16, 13.8))
    time = series["time_seconds"]
    ax = axes[0, 0]
    _plot_line(ax, time, [v / 100 for v in series["precursor_confidence_percent"]], "наблюдаемый предвестник", BLUE)
    _plot_line(ax, time, series["koopman_forecast_risk"], "оценка риска Купмана", AMBER)
    _plot_step(ax, time, [v / 100 for v in series["koopman_warning_percent"]], "подтверждённое предупреждение", PURPLE)
    _style_axes(ax, "01  Признаки угрозы и предупреждение", "условная шкала 0…1", (0, 1.06))

    ax = axes[0, 1]
    for key, label, color in (("state_hausdorff_normalized", "изменение состояния", BLUE),
                              ("lyapunov_remap_pressure", "вклад Ляпунова", PURPLE),
                              ("remap_pressure", "необходимость защиты", RED)):
        _plot_line(ax, time, series[key], label, color)
    ax.axhline(TRIGGER_BANDS["decision"][0], color=MUTED, linewidth=.8, linestyle=":", label="граница наблюдения")
    _style_axes(ax, "02  Основания решения арбитра", "нормированная оценка", (0, 1.06))

    ax = axes[1, 0]
    for grade, label, color, pattern in (("gold", "золотой", AMBER, "-"),
                                        ("silver", "серебряный", BLUE, "--"),
                                        ("bronze", "бронзовый", PURPLE, ":")):
        _plot_line(ax, time, series[f"{grade}_sla_percent"], label, color, linestyle=pattern)
    finite = [v for grade in ("gold", "silver", "bronze") for v in series[f"{grade}_sla_percent"] if np.isfinite(v)]
    minimum = min(finite, default=0)
    lower = max(0, minimum - max(1, .2 * (100 - minimum)))
    _style_axes(ax, "03  Выполнение требований к качеству услуг", "% учтённых потоков", (lower, 100.3))
    if not finite:
        ax.text(.5, .5, "Нет данных о качестве услуг", transform=ax.transAxes, color=MUTED, ha="center")

    ax = axes[1, 1]
    _plot_step(ax, time, series["rerouted_flows"], "сменили маршрут", BLUE)
    _plot_step(ax, time, series["failover_flows"], "перешли на резервный сервер", PURPLE)
    _plot_step(ax, time, series["isolated_flows"], "изолированы", RED)
    _style_axes(ax, "04  Применённые меры защиты", "число потоков", (0, max(1, max(series["rerouted_flows"] + series["failover_flows"] + series["isolated_flows"], default=0)) * 1.15))

    ax = axes[2, 0]
    _plot_line(ax, time, series["normal_arrival_rate_pps"], "обычные поступления", BLUE)
    filtered = [v if enabled else float("nan") for v, enabled in zip(series["filtered_normal_arrival_rate_pps"], series["kalman_applied"])]
    _plot_line(ax, time, filtered, "Калман на исправных наблюдениях", GREEN)
    _style_axes(ax, "05  Обычный трафик и сглаживание шума", "пакетов/с")

    ax = axes[2, 1]
    _plot_line(ax, time, series["raw_attack_rate_mbps"], "предложенный поток атак", RED)
    _plot_line(ax, time, series["admitted_attack_rate_mbps"], "допущен после защиты", AMBER)
    _style_axes(ax, "06  Пакетный поток атак", "Мбит/с")
    ax.set_ylim(bottom=0)

    for axis in axes.flat:
        axis.set_xlabel("время от t0, с", color=MUTED, fontsize=9)
        if time:
            axis.set_xlim(time[0], max(time[-1], time[0] + 1))
    _shade_attack_windows(axes, dynamics)
    training = dynamics.get("training_summary", {})
    clean_steps = training.get("clean_training_steps", 0)
    training_end = clean_steps * dynamics.get("config", {}).get("step_seconds", 1)
    for axis in axes.flat:
        if training_end > 0:
            axis.axvspan(0, training_end, color=GREEN, alpha=.045, zorder=0)
    first = training.get("first_attack_step")
    fig.text(.07, .936, f"Шагов: {dynamics.get('config', {}).get('step_count', 0)}     ·     Чистых переходов обучения: {clean_steps}     ·     Первая атака: {'нет' if first is None else 'шаг ' + str(first)}", color=TEXT, fontsize=11)
    fig.text(.07, .908, "Зелёный фон — начальное обучение; красный и жёлтый — наблюдаемые атаки. Риск не является вероятностью.", color=MUTED, fontsize=9)
    fig.text(.07, .015, "Источник: network_dynamics.json. Попытки входа и сбои питания учитываются по событиям, а не как пакетная перегрузка.", color=MUTED, fontsize=8)
    _save(fig, path, bottom=.085, top=.85)


def _plot_capacity_bottlenecks(model, snapshots, path: Path) -> None:
    fig, axes = _dashboard("GNet9 · нагрузка и запас ресурсов", 2, 1, (13, 10))
    ax, trend = axes[:, 0]
    rows = _link_load_rows(model, snapshots)
    top = sorted(rows, key=lambda row: row["utilization_percent"], reverse=True)[:8]
    limit = MAX_REMAP_UTILIZATION_PERCENT
    bars = ax.barh([r["label"] for r in reversed(top)], [r["utilization_percent"] for r in reversed(top)],
                   color=[RED if r["utilization_percent"] > limit else BLUE for r in reversed(top)], height=.6)
    for bar, row in zip(bars, reversed(top)):
        ax.text(bar.get_width() + .8, bar.get_y() + bar.get_height() / 2,
                f"{row['utilization_percent']:.2f}% · шаг {row['step_index']}", va="center", color=TEXT, fontsize=9)
    if not top:
        ax.text(.5, .5, "Нет данных о нагрузке линий", transform=ax.transAxes, color=MUTED, ha="center")
    _style_axes(ax, "01  Пиковая нагрузка линий за весь расчёт", "% пропускной способности", legend=False)
    ax.axvline(limit, color=RED, linestyle="--", linewidth=1)
    ax.set_xlim(0, max(limit, max((r["utilization_percent"] for r in top), default=0) * 1.25) + 8)
    ax.text(limit, 1.01, f"допуск новых маршрутов: {limit:g}%", transform=ax.get_xaxis_transform(), color=RED, ha="right", fontsize=8)
    time = [s.get("time_seconds", 0) for s in snapshots]
    # Ряд по каждому шагу, а не максимум из ограниченного списка top-8.
    peak_series = [_snapshot_peak_load(model, s) for s in snapshots]
    _plot_line(trend, time, peak_series, "максимальная нагрузка линии", BLUE)
    projected = [_nested_float(s, ("attacks", "routing", "maximum_projected_utilization_percent"), default=float("nan")) for s in snapshots]
    _plot_line(trend, time, projected, "оценка ресурсов после смены маршрутов", PURPLE)
    trend.axhline(limit, color=RED, linestyle="--", linewidth=1, label=f"допуск новых маршрутов: {limit:g}%")
    _style_axes(trend, "02  Нагрузка во времени", "%", (0, max(105, max(peak_series, default=0) * 1.1)))
    trend.set_xlabel("время от t0, с", color=MUTED)
    fig.text(.15, .93, "Пик считается по скорости на каждом шаге. Пропускная способность и правила допуска взяты из модели.", color=MUTED, fontsize=9)
    _save(fig, path, bottom=.12, top=.86, left=.15)


def _attack_windows(dynamics: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    step_seconds = float(dynamics.get("config", {}).get("step_seconds", 1.0))
    for snapshot in dynamics.get("snapshots", []):
        for event in snapshot.get("attacks", {}).get("events", []):
            temporal = event.get("temporal", {})
            key = str(event.get("attack_id") or f"{event.get('kind')}:{temporal.get('start_step', 0)}")
            window = grouped.setdefault(key, {
                "start": float(snapshot.get("time_seconds", 0.0)),
                "end": float(snapshot.get("time_seconds", 0.0)) + step_seconds,
                "kind": str(event.get("kind", "")),
            })
            window["end"] = max(window["end"], float(snapshot.get("time_seconds", 0.0)) + step_seconds)
    return list(grouped.values())


def _shade_attack_windows(axes, dynamics: dict[str, Any]) -> None:
    for window in _attack_windows(dynamics):
        color = ATTACK_KIND_COLORS.get(window["kind"], "#fb7185")
        for axis in np.asarray(axes).flat:
            axis.axvspan(window["start"], window["end"], color=color, alpha=0.075, zorder=0)


def _snapshot_link_loads(model, snapshot):
    """Скорости линий в Мбит/с из сохранённых показателей или полных потоков."""
    state = snapshot.get("traffic", {}).get("network_state", {})
    if "edge_vectors" in state:
        names = state["edge_metric_names"]
        speed, capacity = names.index("speed_mbps"), names.index("capacity_mbps")
        return [(tuple(edge), values[speed], values[capacity]) for edge, values in zip(state["edge_ids"], state["edge_vectors"])]
    if "edges" in state:
        return [((e["source"], e["target"]), e["metrics"]["speed_mbps"], e["metrics"]["capacity_mbps"]) for e in state["edges"]]
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
    return max((rate / capacity * 100 for _, rate, capacity in _snapshot_link_loads(model, snapshot) if capacity > 0), default=0)


def _link_load_rows(model, snapshots):
    """Для каждой линии сохранить её пик и шаг, на котором он достигнут."""
    peaks = {}
    for snapshot in snapshots:
        for edge, rate, capacity in _snapshot_link_loads(model, snapshot):
            if capacity <= 0:
                continue
            edge = tuple(sorted(edge))
            utilization = rate / capacity * 100
            if edge not in peaks or utilization > peaks[edge]["utilization_percent"]:
                peaks[edge] = {"label": " ↔ ".join(edge), "utilization_percent": utilization,
                               "step_index": snapshot.get("step_index", 0)}
    return list(peaks.values())


def _dashboard(title: str, rows: int, columns: int, size: tuple[float, float]):
    fig, axes = plt.subplots(rows, columns, figsize=size, squeeze=False)
    fig.suptitle(title, color="#f8fafc", fontsize=17, fontweight="bold", y=0.975)
    fig.set_facecolor(BACKGROUND)
    for axis in axes.flat:
        axis.set_facecolor(PANEL)
    return fig, axes


def _plot_line(ax, x, y, label: str, color: str, *, linestyle: str = "-") -> None:
    ax.plot(x, y, linewidth=1.8,
            linestyle=linestyle, color=color, label=label)


def _plot_step(ax, x, y, label: str, color: str) -> None:
    ax.step(x, y, where="post", linewidth=1.9, color=color, label=label)


def _style_axes(
    ax,
    title: str,
    ylabel: str,
    ylim=None,
    *,
    legend: bool = True,
    legend_columns: int = 2,
) -> None:
    style_axis(ax, title, ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend:
        # Легенда находится под данными, а не поверх линий и столбцов.
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.23),
            ncol=legend_columns,
            facecolor=PANEL,
            edgecolor="#475569",
            labelcolor="#e2e8f0",
            fontsize=8,
        )


def _save(
    fig,
    path: Path,
    *,
    bottom: float = 0.05,
    top: float = 0.89,
    left: float = 0.075,
) -> None:
    fig.subplots_adjust(left=left, right=0.985, bottom=bottom, top=top, hspace=0.65, wspace=0.26)
    save_figure(fig, path)


def _aggregate(snapshot: dict[str, Any], level: str, metric: str, statistic: str) -> float:
    values = snapshot.get("arbitrator", {}).get("level_metric_aggregates", {}).get(level, {}).get("metrics", {}).get(metric, {})
    return float(values.get(statistic, 0.0))


def _nested_float(source: dict[str, Any], path: tuple[str, ...], *, default: float = 0.0) -> float:
    value: Any = source
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return float(value if value is not None else default)
