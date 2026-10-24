"""Небольшой набор читаемых графиков динамики GNet9.

Данные для каждого графика берутся только из ``network_dynamics.json``:
временные ряды и загрузка каналов — из сохранённых наблюдений ``snapshots``.
Модуль намеренно создаёт два графика вместо набора похожих
панелей: оператору важнее увидеть ход события и реальные узкие места, чем
повторять одну и ту же метрику в нескольких дашбордах.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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
            series[name].append(_nested_float(snapshot, path, default=float("nan")) * multiplier)
    for grade in ("gold", "silver", "bronze"):
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


def _plot_dynamics_overview(series, dynamics, path: Path) -> None:
    """Сигнал → оценка → качество услуг → меры → обычный трафик → атаки."""
    fig = plt.figure(figsize=(16, 15.7), facecolor=BACKGROUND)
    grid = fig.add_gridspec(2, 1, height_ratios=(.9, 6), hspace=.20)
    timeline = fig.add_subplot(grid[0, 0])
    panels = grid[1, 0].subgridspec(3, 2, hspace=.87, wspace=.25)
    axes = np.array([[fig.add_subplot(panels[row, column]) for column in range(2)]
                     for row in range(3)])
    fig.suptitle("GNet9 · состояние сети и работа защиты", color=TEXT,
                 fontsize=20, fontweight="bold", x=.075, ha="left", y=.975)
    time = series["time_seconds"]
    ax = axes[0, 0]
    _plot_line(ax, time, [v / 100 for v in series["precursor_confidence_percent"]], "наблюдаемый предвестник", BLUE, linestyle="--")
    _plot_line(ax, time, series["koopman_forecast_risk"], "риск Купмана (не вероятность)", AMBER)
    threshold = [_nested_float(s, ("koopman", "warning_risk_threshold"), default=float("nan"))
                 for s in dynamics.get("snapshots", [])]
    _plot_line(ax, time, threshold, "сохранённая граница предупреждения", MUTED, linestyle=":")
    _style_axes(ax, "01  Сигнал и прогноз Купмана", "оценка, без единиц", _positive_limits(series["koopman_forecast_risk"], threshold, minimum=1.06))

    ax = axes[0, 1]
    for key, label, color in (("state_hausdorff_normalized", "изменение состояния", BLUE),
                              ("lyapunov_remap_pressure", "вклад Ляпунова", PURPLE),
                              ("remap_pressure", "необходимость защиты", RED)):
        _plot_line(ax, time, series[key], label, color)
    _style_axes(ax, "02  Основания решения арбитра", "оценка, без единиц",
                _positive_limits(series["state_hausdorff_normalized"], series["lyapunov_remap_pressure"],
                                 series["remap_pressure"], minimum=1.06))

    ax = axes[1, 0]
    for grade, label, color, pattern in (("gold", "золотой", AMBER, "-"),
                                        ("silver", "серебряный", BLUE, "--"),
                                        ("bronze", "бронзовый", PURPLE, ":")):
        values = series[f"{grade}_sla_percent"]
        present = [value for value in values if np.isfinite(value)]
        if present:
            label += f" · минимум {min(present):.2f}%"
        _plot_line(ax, time, values, label, color, linestyle=pattern)
    finite = [v for grade in ("gold", "silver", "bronze") for v in series[f"{grade}_sla_percent"] if np.isfinite(v)]
    minimum = min(finite, default=0)
    lower = max(0, minimum - max(1, .2 * (100 - minimum)))
    _style_axes(ax, "03  Потоки, выполнившие SLA (требования качества)", "% потоков своего класса", (lower, 100.3))
    ax.set_yticks([tick for tick in ax.get_yticks() if lower <= tick <= 100])
    if finite and lower > 0:
        ax.text(.01, .04, f"Шкала увеличена: {lower:.1f}–100%", transform=ax.transAxes,
                color=MUTED, fontsize=8, va="bottom")

    ax = axes[1, 1]
    _plot_step(ax, time, series["rerouted_flows"], "сменили маршрут", BLUE)
    _plot_step(ax, time, series["failover_flows"], "перешли на резервный сервер", PURPLE)
    _plot_step(ax, time, series["isolated_flows"], "изолированы", RED)
    _style_axes(ax, "04  Действующая защита на каждом шаге", "число потоков",
                _positive_limits(series["rerouted_flows"], series["failover_flows"], series["isolated_flows"], minimum=1))

    ax = axes[2, 0]
    _plot_line(ax, time, series["normal_arrival_rate_pps"], "обычные поступления", BLUE)
    filtered = [v if enabled else float("nan") for v, enabled in zip(series["filtered_normal_arrival_rate_pps"], series["kalman_applied"])]
    _plot_line(ax, time, filtered, "Калман на исправных наблюдениях", GREEN)
    _style_axes(ax, "05  Обычный трафик · поступления Пуассона", "пакетов/с")
    native = [value for values in (series["normal_arrival_rate_pps"], filtered)
              for value in values if np.isfinite(value)]
    if native:
        padding = max((max(native) - min(native)) * .15, max(native) * .0001, 1e-9)
        ax.set_ylim(max(0, min(native) - padding), max(native) + padding)
        if ax.get_ylim()[0] > 0:
            ax.text(.01, .04, "Шкала увеличена для сравнения шума", transform=ax.transAxes,
                    color=MUTED, fontsize=8, va="bottom")

    ax = axes[2, 1]
    _plot_line(ax, time, series["raw_attack_rate_mbps"], "предложено до защиты", RED)
    _plot_line(ax, time, series["admitted_attack_rate_mbps"], "допущено после защиты", AMBER, linestyle="--")
    _style_axes(ax, "06  Пакетный поток атак", "Мбит/с")
    ax.set_ylim(bottom=0)

    for axis in axes.flat:
        axis.set_xlabel("время от t0, с", color=MUTED, fontsize=9)
        _set_time_limits(axis, dynamics)
    _shade_attack_windows(axes, dynamics)
    _plot_event_timeline(timeline, dynamics)
    transitions = _accepted_transition_count(dynamics)
    accepted_text = "нет данных" if transitions is None else str(transitions)
    observed_attacks = {window["attack_id"] for window in _attack_windows(dynamics)}
    steps = max((s.get("step_index", 0) for s in dynamics.get("snapshots", [])), default=0)
    fig.text(.075, .944, f"Шагов наблюдения: {steps}     ·     Принято переходов в обучение: {accepted_text}     ·     Наблюдалось атак: {len(observed_attacks)}",
             color=TEXT, fontsize=11)
    fig.text(.075, .923, "Шкала сверху показывает фактические записи. Предупреждение — время выдачи прогноза, а не длительность будущей атаки.",
             color=MUTED, fontsize=9)
    fig.text(.075, .015, "Источник: network_dynamics.json · Разрыв линии — нет данных. Тонированный фон — наблюдаемая атака. Меры защиты могут пересекаться по потокам.",
             color=MUTED, fontsize=8)
    _save(fig, path, bottom=.115, top=.88)


def _plot_capacity_bottlenecks(model, snapshots, path: Path) -> None:
    fig = plt.figure(figsize=(16, 10.8), facecolor=BACKGROUND)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.15, 1))
    ax = fig.add_subplot(grid[0, :])
    trend = fig.add_subplot(grid[1, 0])
    resources = fig.add_subplot(grid[1, 1])
    fig.suptitle("GNet9 · линии и ресурсы резервирования", color=TEXT,
                 fontsize=20, fontweight="bold", x=.075, ha="left", y=.975)
    rows = _link_load_rows(model, snapshots)
    top = sorted(rows, key=lambda row: row["utilization_percent"], reverse=True)[:8]
    limits = [_nested_float(s, ("attacks", "routing", "maximum_safe_utilization_percent"), default=float("nan"))
              for s in snapshots]
    finite_limits = {value for value in limits if np.isfinite(value)}
    limit = next(iter(finite_limits)) if len(finite_limits) == 1 else None
    bars = ax.barh([r["label"] for r in reversed(top)], [r["utilization_percent"] for r in reversed(top)],
                   color=BLUE, height=.6)
    for bar, row in zip(bars, reversed(top)):
        ax.text(bar.get_width() + .8, bar.get_y() + bar.get_height() / 2,
                f"{row['utilization_percent']:.2f}% · шаг {row['step_index']}", va="center", color=TEXT, fontsize=9)
    if not top:
        ax.text(.5, .5, "Нет данных о нагрузке линий", transform=ax.transAxes, color=MUTED, ha="center")
    _style_axes(ax, "01  Пиковая нагрузка линий за весь расчёт", "", legend=False)
    ax.set_xlabel("% пропускной способности линии", color=MUTED, fontsize=9)
    ax.set_xlim(0, max([100, *(r["utilization_percent"] * 1.3 for r in top), *finite_limits]))
    ax.grid(False, axis="y")
    ax.grid(axis="x", color=MUTED, linewidth=.5, alpha=.18)
    if limit is not None:
        ax.axvline(limit, color=RED, linestyle="--", linewidth=1)
        ax.text(limit, 1.01, f"допуск нового маршрута: {limit:g}%", transform=ax.get_xaxis_transform(), color=RED, ha="right", fontsize=8)
    time = [s.get("time_seconds", 0) for s in snapshots]
    # Ряд по каждому шагу, а не максимум из ограниченного списка top-8.
    peak_series = [_snapshot_peak_load(model, s) for s in snapshots]
    _plot_line(trend, time, peak_series, "максимальная нагрузка линии", BLUE)
    projected = [_nested_float(s, ("attacks", "routing", "maximum_projected_edge_utilization_percent"), default=float("nan")) for s in snapshots]
    projected_nodes = [_nested_float(s, ("attacks", "routing", "maximum_projected_node_utilization_percent"), default=float("nan")) for s in snapshots]
    _plot_line(trend, time, projected, "линии: оценка резервирования", PURPLE, linestyle="--")
    _plot_line(trend, time, projected_nodes, "узлы: оценка пропускной способности", GREEN, linestyle=":")
    _plot_line(trend, time, limits, "сохранённый допуск нового маршрута", RED, linestyle="--")
    _style_axes(trend, "02  Пропускная способность сети", "% своей пропускной способности",
                _positive_limits(peak_series, projected, projected_nodes, limits, minimum=100))
    resource_series = []
    for metric, label, color, pattern in (("cpu", "CPU (процессор)", BLUE, "-"),
                                         ("ram", "RAM (оперативная память)", PURPLE, "--"),
                                         ("session", "сессии (расчётная ёмкость)", AMBER, ":")):
        values = [_nested_float(s, ("attacks", "routing", f"maximum_projected_server_{metric}_utilization_percent"), default=float("nan")) for s in snapshots]
        resource_series.append(values)
        _plot_line(resources, time, values, label, color, linestyle=pattern)
    _plot_line(resources, time, limits, "сохранённый допуск нового маршрута", RED, linestyle="--")
    _style_axes(resources, "03  Ресурсы серверов · оценка резервирования", "% ёмкости соответствующего ресурса",
                _positive_limits(*resource_series, limits, minimum=100))
    if any(np.isfinite(value) for value in limits) and not any(
        np.isfinite(value) for values in resource_series for value in values
    ):
        resources.text(.5, .5, "Нет раздельных оценок ресурсов", transform=resources.transAxes,
                       color=MUTED, ha="center", fontsize=9)
    for axis in (trend, resources):
        axis.set_xlabel("время от t0, с", color=MUTED)
        _set_time_limits(axis, {"snapshots": snapshots})
    fig.text(.075, .932, "Вверху — восемь линий с наибольшим наблюдаемым пиком. Внизу — максимумы каждого вида ресурса на каждом шаге.", color=MUTED, fontsize=10)
    fig.text(.075, .908, "Оценки учитывают известную нагрузку и резервирование; это не измерение будущего. Допуск новых маршрутов читается из результата расчёта.", color=MUTED, fontsize=9)
    fig.text(.075, .022, "Разрыв линии — нет данных. CPU, память и сессии имеют разные ёмкости; общий максимум не подменяет нагрузку линии. Ёмкость сессий — допущение стенда.", color=MUTED, fontsize=8)
    _save(fig, path, bottom=.20, top=.84, left=.12, hspace=.45)


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


def _plot_event_timeline(axis, dynamics: dict[str, Any]) -> None:
    """Полосы — шаги наблюдения; события не берутся из будущего расписания."""
    period = _observation_period(dynamics)
    for start, end in _training_windows(dynamics):
        axis.broken_barh([(start, end - start)], (4.7, .6), color=GREEN, alpha=.8)
    for snapshot in dynamics.get("snapshots", []):
        if snapshot.get("koopman", {}).get("forecast_is_early_warning"):
            axis.broken_barh([(float(snapshot.get("time_seconds", 0)), period)], (3.7, .6), color=AMBER, alpha=.85)
    lanes = {"dos": 3, "ddos": 3, "syn_flood": 3, "brute_force": 2, "power_attack": 1}
    labels = {"dos": "DoS", "ddos": "DDoS", "syn_flood": "SYN"}
    extent = max((float(s.get("time_seconds", 0)) for s in dynamics.get("snapshots", [])), default=0) + period
    for window in _attack_windows(dynamics):
        lane = lanes.get(window["kind"], 3)
        color = ATTACK_KIND_COLORS.get(window["kind"], RED)
        start, width = window["start"], window["end"] - window["start"]
        axis.broken_barh([(start, width)], (lane - .3, .6), color=color, alpha=.85)
        if window["kind"] in labels and width / max(extent, 1) > .045:
            axis.text(start + width / 2, lane, labels[window["kind"]], ha="center", va="center",
                      fontsize=7, color=BACKGROUND, fontweight="bold")
    style_axis(axis, "Наблюдения по времени · цветная полоса соответствует записанному шагу")
    axis.set_yticks([5, 4, 3, 2, 1], ["Обучение принято", "Предупреждение", "Пакетные атаки", "Попытки входа", "Сбой питания"])
    axis.set_ylim(.4, 5.6)
    axis.grid(False, axis="y")
    axis.set_xlabel("время от t0, с", color=MUTED, fontsize=8)
    axis.tick_params(axis="y", length=0, labelsize=8)
    _set_time_limits(axis, dynamics)


def _positive_limits(*series, minimum=1.0) -> tuple[float, float]:
    finite = [float(value) for values in series for value in values if np.isfinite(value)]
    return 0.0, max(minimum, max(finite, default=0) * 1.12)


def _attack_windows(dynamics: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    step_seconds = _observation_period(dynamics)
    for snapshot in dynamics.get("snapshots", []):
        for event in snapshot.get("attacks", {}).get("events", []):
            key = str(event.get("attack_id") or f"{event.get('kind')}:{event.get('target_id', '')}")
            time = float(snapshot.get("time_seconds", 0.0))
            windows = grouped.setdefault(key, [])
            if windows and time <= windows[-1]["end"] + 1e-9:
                windows[-1]["end"] = max(windows[-1]["end"], time + step_seconds)
            else:
                windows.append({"attack_id": key, "start": time, "end": time + step_seconds,
                                "kind": str(event.get("kind", ""))})
    return [window for windows in grouped.values() for window in windows]


def _shade_attack_windows(axes, dynamics: dict[str, Any]) -> None:
    for window in _attack_windows(dynamics):
        color = ATTACK_KIND_COLORS.get(window["kind"], "#fb7185")
        for axis in np.asarray(axes).flat:
            axis.axvspan(window["start"], window["end"], color=color, alpha=0.035, zorder=0)


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


def _dashboard(title: str, rows: int, columns: int, size: tuple[float, float]):
    fig, axes = plt.subplots(rows, columns, figsize=size, squeeze=False)
    fig.suptitle(title, color="#f8fafc", fontsize=17, fontweight="bold", y=0.975)
    fig.set_facecolor(BACKGROUND)
    for axis in axes.flat:
        axis.set_facecolor(PANEL)
    return fig, axes


def _plot_line(ax, x, y, label: str, color: str, *, linestyle: str = "-") -> None:
    if not any(np.isfinite(value) for value in y):
        return
    ax.plot(x, y, linewidth=1.8,
            linestyle=linestyle, color=color, label=label,
            marker="o" if len(x) == 1 else None, markersize=3)


def _plot_step(ax, x, y, label: str, color: str) -> None:
    if not any(np.isfinite(value) for value in y):
        return
    ax.step(x, y, where="post", linewidth=1.9, color=color, label=label,
            marker="o" if len(x) == 1 else None, markersize=3)


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
    top: float = 0.89,
    left: float = 0.075,
    hspace: float = 0.82,
) -> None:
    fig.subplots_adjust(left=left, right=0.975, bottom=bottom, top=top, hspace=hspace, wspace=0.25)
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
