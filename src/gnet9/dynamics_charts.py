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
    """Извлечь ровно те ряды, которые показаны на итоговых диаграммах."""
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


def _plot_dynamics_overview(
    series: dict[str, list[float]],
    dynamics: dict[str, Any],
    path: Path,
) -> None:
    """Показать одну цепочку: сигнал, решение, действие и результат."""
    fig, axes = _dashboard("GNet9 · ход сценария и результат защиты", 2, 2, (16, 11.5))
    time = series["time_seconds"]

    ax = axes[0, 0]
    _plot_line(ax, time, series["precursor_confidence_percent"], "предвестник", "#38bdf8")
    _plot_line(ax, time, [value * 100.0 for value in series["koopman_forecast_risk"]], "риск Купмана", "#fbbf24")
    _plot_line(ax, time, series["attack_intensity_percent"], "факт атаки", "#fb7185")
    _plot_step(ax, time, series["koopman_warning_percent"], "подтверждённое предупреждение", "#a78bfa")
    _style_axes(
        ax,
        "1. Наблюдаемый сигнал → предупреждение → факт",
        "% условной шкалы",
        (0, 105),
        legend_columns=2,
    )

    ax = axes[0, 1]
    _plot_line(ax, time, series["state_hausdorff_normalized"], "отличие от эталона", "#f97316")
    _plot_line(ax, time, series["koopman_forecast_risk"], "риск Купмана", "#fbbf24")
    _plot_line(ax, time, series["lyapunov_remap_pressure"], "вклад Ляпунова", "#a78bfa")
    _plot_line(ax, time, series["remap_pressure"], "необходимость защиты", "#f87171")
    for bound in TRIGGER_BANDS["decision"]:
        ax.axhline(bound, color="#cbd5e1", linewidth=0.7, linestyle="--", alpha=0.45)
    _style_axes(
        ax,
        "2. Основания для защиты",
        "нормированная оценка 0…1",
        (0, 1.02),
        legend_columns=2,
    )

    ax = axes[1, 0]
    _plot_step(ax, time, series["rerouted_flows"], "обойдены по другому маршруту", "#22d3ee")
    _plot_step(ax, time, series["failover_flows"], "переведены на резервный сервер", "#a78bfa")
    _plot_step(ax, time, series["isolated_flows"], "временно изолированы", "#fb7185")
    upper = max(1.0, max(
        series["rerouted_flows"] + series["failover_flows"] + series["isolated_flows"],
        default=0.0,
    ) * 1.18)
    _style_axes(ax, "3. Какие меры защита реально применила", "число потоков", (0, upper), legend_columns=1)

    ax = axes[1, 1]
    _plot_line(ax, time, series["gold_availability_percent"], "доступность золотого класса", "#fbbf24")
    _plot_line(ax, time, series["legitimate_delivery_percent"], "доставка полезных данных", "#34d399")
    _plot_line(ax, time, series["post_remap_utilization_percent"], "максимальная загрузка после защиты", "#38bdf8")
    ax.axhline(80.0, color="#f87171", linewidth=1.0, linestyle="--", label="предел ресурса 80 %")
    _style_axes(
        ax,
        "4. Качество обслуживания после защиты",
        "%",
        (0, 105),
        legend_columns=1,
    )

    for axis in axes.ravel():
        axis.set_xlabel("время от t0, с", color="#cbd5e1")
    _shade_attack_windows(axes, dynamics)
    clean_steps = dynamics.get("training_summary", {}).get("clean_training_steps", 0)
    training_end = clean_steps * dynamics.get("config", {}).get("step_seconds", 1)
    if training_end > 0:
        for axis in axes.ravel():
            axis.axvspan(0, training_end, color="#22c55e", alpha=0.055, zorder=0)
        axes[0, 0].text(0.02, 0.95, f"Исправная сеть: {clean_steps} шагов обучения", transform=axes[0, 0].transAxes, color="#86efac", fontsize=8, va="top")
    fig.text(
        0.5,
        0.925,
        "Источник всех рядов: снимки snapshots в network_dynamics.json. Риск Купмана — оценка, а не вероятность.",
        ha="center",
        va="center",
        color="#94a3b8",
        fontsize=8.1,
    )
    _save(fig, path, bottom=0.09, top=0.85)


def _plot_capacity_bottlenecks(model, snapshots: list[dict[str, Any]], path: Path) -> None:
    """Показать только наиболее важный риск ёмкости, без второго дубля графика."""
    fig, ax = _single_chart("GNet9 · наиболее загруженные связи", (13.5, 7.4))
    if model is None:
        ax.text(0.5, 0.5, "Для расчёта ёмкости нужна модель топологии", transform=ax.transAxes,
                ha="center", va="center", color="#cbd5e1", fontsize=12)
        _style_axes(ax, "Нет данных о каналах", "% доступной ёмкости", (0, 100), legend=False)
        _save(fig, path)
        return

    top = sorted(_link_load_rows(model, snapshots), key=lambda row: row["utilization_percent"], reverse=True)[:8]
    if not top:
        ax.text(0.5, 0.5, "В снимках нет модельных потоков", transform=ax.transAxes,
                ha="center", va="center", color="#cbd5e1", fontsize=12)
        _style_axes(ax, "Нет данных о каналах", "% доступной ёмкости", (0, 100), legend=False)
        _save(fig, path)
        return

    labels = [row["label"] for row in reversed(top)]
    values = [row["utilization_percent"] for row in reversed(top)]
    colors = ["#f87171" if value >= 80.0 else "#60a5fa" for value in values]
    bars = ax.barh(labels, values, color=colors, height=0.62)
    ax.axvline(80.0, color="#f87171", linewidth=1.2, linestyle="--")
    for bar, value in zip(bars, values):
        ax.text(value + 0.65, bar.get_y() + bar.get_height() / 2, f"{value:.2f} %",
                va="center", color="#e2e8f0", fontsize=9)
    limit = max(80.0, max(values, default=0.0) * 1.20) + 4.0
    _style_axes(
        ax,
        "Пиковая расчётная загрузка за весь прогон",
        "% доступной ёмкости",
        legend=False,
    )
    ax.set_xlim(0, limit)
    ax.text(
        80.0,
        0.98,
        "предел допуска 80 %",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        color="#fca5a5",
        fontsize=8.6,
        bbox={"boxstyle": "round,pad=0.18", "fc": "#111827", "ec": "#f87171", "alpha": 0.95},
    )
    ax.set_xlabel("максимум по всем шагам сценария", color="#cbd5e1")
    fig.text(
        0.5,
        0.925,
        "Источник: сохранённые модельные потоки из снимков network_dynamics.json.",
        ha="center", va="center", color="#94a3b8", fontsize=8.1,
    )
    _save(fig, path, bottom=0.12, top=0.84, left=0.15)


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


def _link_load_rows(model, snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    peak_bytes: dict[frozenset[str], int] = {}
    interval_by_edge: dict[frozenset[str], int] = {}
    for snapshot in snapshots:
        current: dict[frozenset[str], int] = {}
        for flow in snapshot.get("traffic", {}).get("flows", []):
            interval = int(flow.get("interval_seconds", 1))
            route = flow.get("route", [])
            for source, target in zip(route, route[1:]):
                edge = frozenset((source, target))
                current[edge] = current.get(edge, 0) + int(flow.get("wire_bytes", 0))
                interval_by_edge[edge] = interval
        for edge, wire_bytes in current.items():
            peak_bytes[edge] = max(peak_bytes.get(edge, 0), wire_bytes)

    rows = []
    for edge, wire_bytes in peak_bytes.items():
        source, target = sorted(edge)
        if not model.graph.has_edge(source, target):
            continue
        capacity = float(model.graph.edges[source, target].get("capacity_mbps", 0.0))
        rate = wire_bytes * 8.0 / max(interval_by_edge.get(edge, 1), 1) / 1_000_000.0
        rows.append({
            "label": f"{source} ↔ {target}",
            "utilization_percent": rate / max(capacity, 1e-9) * 100.0,
        })
    return rows


def _dashboard(title: str, rows: int, columns: int, size: tuple[float, float]):
    fig, axes = plt.subplots(rows, columns, figsize=size, squeeze=False)
    fig.suptitle(title, color="#f8fafc", fontsize=17, fontweight="bold", y=0.975)
    _dark_figure(fig, axes)
    return fig, axes


def _single_chart(title: str, size: tuple[float, float]):
    fig, ax = plt.subplots(figsize=size)
    fig.suptitle(title, color="#f8fafc", fontsize=17, fontweight="bold", y=0.975)
    _dark_figure(fig, (ax,))
    return fig, ax


def _dark_figure(fig, axes) -> None:
    fig.patch.set_facecolor("#0f172a")
    for axis in np.asarray(axes).flat:
        axis.set_facecolor("#111827")


def _plot_line(ax, x, y, label: str, color: str, *, linestyle: str = "-") -> None:
    mark_every = max(1, len(x) // 16) if x else 1
    ax.plot(x, y, marker="o", markevery=mark_every, markersize=3.2, linewidth=1.9,
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
    legend_columns: int = 1,
) -> None:
    ax.set_title(title, color="#f8fafc", fontsize=11.8, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, color="#cbd5e1")
    ax.tick_params(colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, axis="y", color="#334155", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend:
        # Легенда находится под данными, а не поверх линий и столбцов.
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=legend_columns,
            facecolor="#172033",
            edgecolor="#475569",
            labelcolor="#e2e8f0",
            fontsize=7.7,
        )


def _save(
    fig,
    path: Path,
    *,
    bottom: float = 0.05,
    top: float = 0.89,
    left: float = 0.075,
) -> None:
    fig.subplots_adjust(left=left, right=0.985, bottom=bottom, top=top, hspace=0.78, wspace=0.28)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.12, facecolor=fig.get_facecolor())
    plt.close(fig)


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
