"""Компактные диагностические графики динамики GNet9."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


AggregateSeriesSpec = tuple[str, str, str, str, float]
NestedSeriesSpec = tuple[str, tuple[str, ...], float]

APP_COLORS = {
    "RTP_OPUS": "#ef4444",
    "RTP_VLC_AV": "#f97316",
    "FTP_DATA": "#3b82f6",
    "DNS": "#a855f7",
    "RTP_TELEMOST": "#14b8a6",
    "LIVE_HLS": "#eab308",
    "ATTACK_DOS": "#ff1744",
    "ATTACK_DDOS": "#b91c1c",
    "ATTACK_SYN": "#f43f5e",
    "ATTACK_POWER": "#facc15",
}
APP_LABELS = {
    "RTP_OPUS": "Голос / Opus",
    "RTP_VLC_AV": "VLC / голос и видео",
    "FTP_DATA": "Файлы / FTP",
    "DNS": "Имена / DNS",
    "RTP_TELEMOST": "Видеоконференция",
    "LIVE_HLS": "Прямая трансляция",
    "ATTACK_DOS": "Атака DoS",
    "ATTACK_DDOS": "Атака DDoS",
    "ATTACK_SYN": "Атака SYN-флуд",
    "ATTACK_POWER": "Атака на питание",
}

SLA_COLORS = {
    "gold": "#fbbf24",
    "silver": "#cbd5e1",
    "bronze": "#c084fc",
}
ATTACK_KIND_LABELS = {
    "dos": "DoS",
    "ddos": "DDoS",
    "syn_flood": "SYN-флуд",
    "power_attack": "Питание",
}
ATTACK_KIND_COLORS = {
    "dos": "#ef4444",
    "ddos": "#b91c1c",
    "syn_flood": "#f43f5e",
    "power_attack": "#facc15",
}

AGGREGATE_SERIES: tuple[AggregateSeriesSpec, ...] = (
    ("sla_margin_min", "L1", "sla_margin", "min", 1.0),
    ("sla_margin_mean", "L1", "sla_margin", "mean", 1.0),
    ("l2_cpu_mean", "L2", "cpu_load_percent", "mean", 1.0),
    ("l2_cpu_max", "L2", "cpu_load_percent", "max", 1.0),
    ("l2_ram_max", "L2", "ram_load_percent", "max", 1.0),
    ("edge_stability_min", "EDGE", "stability_margin", "min", 1.0),
    ("edge_stability_mean", "EDGE", "stability_margin", "mean", 1.0),
    ("edge_loss_max_percent", "EDGE", "loss_probability", "max", 100.0),
)

TRAFFIC_SERIES: tuple[NestedSeriesSpec, ...] = (
    ("packet_count", ("traffic", "summary", "packet_count"), 1.0),
    ("traffic_rate_mbps", ("traffic", "summary", "offered_rate_mbps"), 1.0),
    ("traffic_carried_rate_mbps", ("traffic", "summary", "carried_rate_mbps"), 1.0),
    ("traffic_delivered_rate_mbps", ("traffic", "summary", "delivered_rate_mbps"), 1.0),
    ("traffic_efficiency_percent", ("traffic", "summary", "protocol_efficiency_ratio"), 100.0),
    ("traffic_mean_latency_ms", ("traffic", "summary", "mean_one_way_latency_ms"), 1.0),
    ("traffic_max_latency_ms", ("traffic", "summary", "max_one_way_latency_ms"), 1.0),
    ("traffic_observed_loss_percent", ("traffic", "summary", "observed_loss_ratio"), 100.0),
)

ARBITRATOR_SERIES: tuple[NestedSeriesSpec, ...] = (
    ("remap_pressure", ("arbitrator", "analysis", "remap_pressure"), 1.0),
    ("lyapunov_value", ("arbitrator", "analysis", "lyapunov_value"), 1.0),
    ("lyapunov_delta", ("arbitrator", "analysis", "lyapunov_delta"), 1.0),
    ("lyapunov_derivative", ("arbitrator", "analysis", "lyapunov_derivative_per_second"), 1.0),
    ("lyapunov_remap_pressure", ("arbitrator", "analysis", "lyapunov_remap_pressure"), 1.0),
    (
        "finite_time_energy_growth",
        ("arbitrator", "analysis", "finite_time_energy_growth_rate_per_second"),
        1.0,
    ),
    ("koopman_residual", ("arbitrator", "analysis", "koopman_residual"), 1.0),
    ("koopman_reference_distance", ("koopman", "reference_distance"), 1.0),
    ("koopman_forecast_risk", ("arbitrator", "analysis", "koopman_forecast_risk"), 1.0),
    (
        "koopman_dmd_only_risk",
        ("koopman", "forecast_component_ablation", "koopman_dmd_only_max_risk"),
        1.0,
    ),
    (
        "koopman_delay_trend_risk",
        ("koopman", "forecast_component_ablation", "koopman_plus_delay_trend_max_risk"),
        1.0,
    ),
    ("koopman_predicted_distance", ("arbitrator", "analysis", "koopman_predicted_reference_distance"), 1.0),
    ("koopman_operator_drift", ("arbitrator", "analysis", "koopman_operator_drift"), 1.0),
    ("gold_threat_proximity_risk", ("arbitrator", "analysis", "gold_threat_proximity_risk"), 1.0),
    ("route_hausdorff_normalized", ("arbitrator", "analysis", "route_hausdorff_normalized"), 1.0),
    ("route_edge_jaccard", ("arbitrator", "analysis", "route_edge_jaccard_distance"), 1.0),
    ("route_latency_stretch", ("arbitrator", "analysis", "route_latency_stretch_ratio"), 1.0),
    ("decision_confidence", ("arbitrator", "analysis", "decision_confidence"), 1.0),
)

ATTACK_SERIES: tuple[NestedSeriesSpec, ...] = (
    ("attack_active_count", ("attacks", "active_count"), 1.0),
    ("attack_intensity_percent", ("attacks", "maximum_intensity_ratio"), 100.0),
    ("attack_rate_mbps", ("attacks", "attack_rate_mbps"), 1.0),
    ("raw_attack_rate_mbps", ("attacks", "raw_attack_rate_mbps"), 1.0),
    ("blocked_attack_rate_mbps", ("attacks", "blocked_attack_rate_mbps"), 1.0),
    ("precursor_confidence_percent", ("attacks", "precursor_confidence"), 100.0),
    ("legitimate_loss_percent", ("attacks", "legitimate_loss_ratio"), 100.0),
    ("legitimate_delivery_percent", ("attacks", "legitimate_delivery_ratio"), 100.0),
    ("impacted_gold_flows", ("attacks", "impacted_gold_flow_count"), 1.0),
    ("gold_sla_compliance_percent", ("attacks", "gold_sla_compliance_ratio"), 100.0),
    ("attack_pressure", ("arbitrator", "analysis", "attack_pressure"), 1.0),
    ("target_utilization_percent", ("attacks", "maximum_target_utilization_percent"), 1.0),
    ("koopman_warning_percent", ("koopman", "forecast_is_early_warning"), 100.0),
    ("rerouted_flows", ("attacks", "routing", "rerouted_flow_count"), 1.0),
    ("failover_flows", ("attacks", "routing", "failover_flow_count"), 1.0),
    ("isolated_flows", ("attacks", "routing", "isolated_flow_count"), 1.0),
    ("gold_availability_percent", ("attacks", "routing", "by_sla", "gold", "availability_ratio"), 100.0),
    ("silver_availability_percent", ("attacks", "routing", "by_sla", "silver", "availability_ratio"), 100.0),
    ("bronze_availability_percent", ("attacks", "routing", "by_sla", "bronze", "availability_ratio"), 100.0),
    ("gold_restoration_percent", ("attacks", "routing", "by_sla", "gold", "restoration_ratio"), 100.0),
    ("silver_restoration_percent", ("attacks", "routing", "by_sla", "silver", "restoration_ratio"), 100.0),
    ("bronze_restoration_percent", ("attacks", "routing", "by_sla", "bronze", "restoration_ratio"), 100.0),
    ("post_remap_utilization_percent", ("attacks", "routing", "maximum_projected_utilization_percent"), 1.0),
    ("post_remap_edge_utilization_percent", ("attacks", "routing", "maximum_projected_edge_utilization_percent"), 1.0),
    ("post_remap_node_utilization_percent", ("attacks", "routing", "maximum_projected_node_utilization_percent"), 1.0),
    ("post_remap_server_cpu_percent", ("attacks", "routing", "maximum_projected_server_cpu_utilization_percent"), 1.0),
    ("post_remap_server_ram_percent", ("attacks", "routing", "maximum_projected_server_ram_utilization_percent"), 1.0),
    ("post_remap_server_sessions_percent", ("attacks", "routing", "maximum_projected_server_session_utilization_percent"), 1.0),
    ("power_voltage_sag_percent", ("attacks", "power_voltage_sag_ratio"), 100.0),
    (
        "power_service_unavailability_percent",
        ("attacks", "power_service_unavailability_ratio"),
        100.0,
    ),
    (
        "power_energy_depletion_percent",
        ("attacks", "power_energy_depletion_ratio"),
        100.0,
    ),
)


def export_dynamics_charts(
    dynamics: dict[str, Any],
    output_dir: Path,
    *,
    model=None,
    flow_snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Экспортировать компактный набор неперекрывающихся диагностик."""
    output_dir.mkdir(parents=True, exist_ok=True)
    series = extract_dynamics_chart_series(dynamics)
    application_series = extract_application_series(dynamics)

    charts = {
        "health": output_dir / "dynamics_health.png",
        "traffic": output_dir / "traffic_composition.png",
        "capacity": output_dir / "link_capacity.png",
        "arbitrator": output_dir / "dynamics_arbitrator.png",
        "attacks": output_dir / "dynamics_attacks.png",
        "remapping": output_dir / "dynamics_remapping.png",
    }

    _plot_health_dashboard(series, charts["health"])
    _plot_traffic_composition(series, application_series, charts["traffic"])
    _plot_link_capacity(model, flow_snapshots or dynamics.get("snapshots", []), charts["capacity"])
    _plot_arbitrator(series, dynamics, charts["arbitrator"])
    _plot_attacks(series, dynamics, charts["attacks"])
    _plot_remapping(series, dynamics, charts["remapping"])

    # График процесса поступления создаётся только тогда, когда сценарий
    # действительно экспортировал параметры Вейбулла. Обычный прогон не
    # получает пустой или декоративный файл.
    arrival_records = _arrival_records(dynamics)
    if arrival_records:
        charts["arrival"] = output_dir / "attack_arrival_weibull.png"
        _plot_attack_arrivals(arrival_records, charts["arrival"])
    else:
        # Не оставлять результат предыдущего predictive-demo рядом с новым
        # прогоном без процесса поступления атак: такой файл выглядел бы как
        # актуальный, хотя к текущей динамике уже не относится.
        (output_dir / "attack_arrival_weibull.png").unlink(missing_ok=True)

    # Generated by older releases and superseded by dynamics_health.png.
    for legacy_name in ("dynamics_sla.png", "dynamics_load_stability.png", "dynamics_traffic.png"):
        (output_dir / legacy_name).unlink(missing_ok=True)
    return charts


def extract_dynamics_chart_series(dynamics: dict[str, Any]) -> dict[str, list[float]]:
    """Извлечь временные ряды из сводок арбитратора и трафика."""
    snapshots = dynamics.get("snapshots", [])
    series_keys = (
        "time_seconds",
        *(name for name, *_ in AGGREGATE_SERIES),
        *(name for name, *_ in TRAFFIC_SERIES),
        *(name for name, *_ in ARBITRATOR_SERIES),
        *(name for name, *_ in ATTACK_SERIES),
        *(f"{grade}_slo_compliance_percent" for grade in ("gold", "silver", "bronze")),
        *(f"{grade}_slo_coverage_percent" for grade in ("gold", "silver", "bronze")),
    )
    series: dict[str, list[float]] = {key: [] for key in series_keys}
    for snapshot in snapshots:
        series["time_seconds"].append(float(snapshot.get("time_seconds", 0.0)))
        for name, level, metric, statistic, multiplier in AGGREGATE_SERIES:
            series[name].append(_aggregate(snapshot, level, metric, statistic) * multiplier)
        for name, path, multiplier in (*TRAFFIC_SERIES, *ARBITRATOR_SERIES, *ATTACK_SERIES):
            default = 1.0 if name in {
                "gold_availability_percent", "silver_availability_percent", "bronze_availability_percent",
                "gold_restoration_percent", "silver_restoration_percent", "bronze_restoration_percent",
            } else 0.0
            series[name].append(_nested_float(snapshot, path, default=default) * multiplier)
        tiers = {
            str(tier.get("sla_grade", "")): tier
            for tier in snapshot.get("attacks", {}).get("sla_restoration", {}).get("tiers", [])
        }
        for grade in ("gold", "silver", "bronze"):
            tier = tiers.get(grade, {})
            flow_count = int(tier.get("flow_count", 0))
            compliance = (
                1.0
                if flow_count <= 0
                else float(tier.get("sla_compliant_flow_count", 0)) / flow_count
            )
            coverage = float(tier.get("mean_slo_assessment_coverage_ratio", 1.0))
            series[f"{grade}_slo_compliance_percent"].append(compliance * 100.0)
            series[f"{grade}_slo_coverage_percent"].append(coverage * 100.0)
    return series


def extract_application_series(dynamics: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    """Вернуть ряды приложений, доступные даже в режиме краткой сводки."""
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


def _plot_health_dashboard(series: dict[str, list[float]], path: Path) -> None:
    fig, axes = _dashboard("GNet9 · состояние сети", 2, 2, (14, 9))
    time = series["time_seconds"]

    ax = axes[0, 0]
    _plot_line(ax, time, series["sla_margin_min"], "минимальный", "#34d399")
    _plot_line(ax, time, series["sla_margin_mean"], "средний", "#60a5fa")
    ax.axhline(0.60, color="#f87171", linestyle="--", linewidth=1.2, label="порог наблюдения 0,60")
    _style_axes(ax, "Запас SLA абонентов L1", "доля", (0, 1.02))

    ax = axes[0, 1]
    _plot_line(ax, time, series["l2_cpu_mean"], "ЦП: средняя", "#22d3ee")
    _plot_line(ax, time, series["l2_cpu_max"], "ЦП: максимальная", "#60a5fa")
    _plot_line(ax, time, series["l2_ram_max"], "ОЗУ: максимальная", "#c084fc")
    ax.axhline(80.0, color="#f87171", linestyle="--", linewidth=1.2, label="предельная загрузка")
    _style_axes(ax, "Оборудование L2", "%", (0, 100))

    ax = axes[1, 0]
    _plot_line(ax, time, series["edge_stability_min"], "минимальная", "#34d399")
    _plot_line(ax, time, series["edge_stability_mean"], "средняя", "#60a5fa")
    _style_axes(ax, "Устойчивость транспортных связей", "доля", (0, 1.02))

    ax = axes[1, 1]
    _plot_line(ax, time, series["traffic_rate_mbps"], "предложенная канальная нагрузка", "#fbbf24")
    _plot_line(ax, time, series["traffic_carried_rate_mbps"], "передано по маршрутам", "#38bdf8")
    _plot_line(ax, time, series["traffic_delivered_rate_mbps"], "доставлено", "#34d399")
    efficiency = series["traffic_efficiency_percent"]
    ax2 = ax.twinx()
    _plot_line(ax2, time, efficiency, "протокольная эффективность", "#a78bfa")
    ax2.set_ylabel("эффективность, %", color="#cbd5e1")
    ax2.tick_params(colors="#cbd5e1")
    ax2.set_ylim(0, 100)
    _style_axes(ax, "Предложенная нагрузка и протокольная эффективность", "Мбит/с")
    _combined_legend(ax, ax2)
    _save(fig, path)


def _plot_traffic_composition(
    series: dict[str, list[float]],
    applications: dict[str, dict[str, list[float]]],
    path: Path,
) -> None:
    fig, axes = _dashboard("GNet9 · профиль сервисного трафика", 2, 2, (14, 9))
    time = series["time_seconds"]
    names = list(applications)
    service_names = [name for name in names if not name.startswith("ATTACK_")]

    for name in service_names:
        color = APP_COLORS.get(name, "#94a3b8")
        label = APP_LABELS.get(name, name)
        _plot_line(axes[0, 0], time, applications[name]["delivered_rate_mbps"], label, color)
        _plot_line(axes[0, 1], time, applications[name]["packet_count"], label, color)
    _style_axes(axes[0, 0], "Доставленная скорость по приложениям", "Мбит/с")
    _style_axes(axes[0, 1], "Пакеты за один шаг", "пакеты")

    labels = [APP_LABELS.get(name, name) for name in service_names]
    colors = [APP_COLORS.get(name, "#94a3b8") for name in service_names]
    efficiency = [applications[name]["protocol_efficiency_ratio"][-1] * 100 if applications[name]["protocol_efficiency_ratio"] else 0 for name in service_names]
    axes[1, 0].bar(labels, efficiency, color=colors, width=0.62)
    _style_axes(axes[1, 0], "Доля полезной нагрузки в канальном трафике", "%", (0, 100), legend=False)
    axes[1, 0].tick_params(axis="x", rotation=12)
    _annotate_bars(axes[1, 0], efficiency, "{:.1f}%")

    x = np.arange(len(service_names))
    means = [applications[name]["mean_one_way_latency_ms"][-1] if applications[name]["mean_one_way_latency_ms"] else 0 for name in service_names]
    maxima = [applications[name]["max_one_way_latency_ms"][-1] if applications[name]["max_one_way_latency_ms"] else 0 for name in service_names]
    axes[1, 1].bar(x - 0.18, means, 0.36, color="#38bdf8", label="средняя")
    axes[1, 1].bar(x + 0.18, maxima, 0.36, color="#fb7185", label="максимальная")
    axes[1, 1].set_xticks(x, labels, rotation=12)
    _style_axes(axes[1, 1], "Односторонняя задержка", "мс")
    _save(fig, path)


def _plot_link_capacity(model, snapshots: list[dict[str, Any]], path: Path) -> None:
    fig, axes = _dashboard("GNet9 · фактическая загрузка каналов", 1, 2, (15, 7))
    axes = axes.ravel()
    if model is None:
        for ax in axes:
            ax.text(0.5, 0.5, "модель недоступна", ha="center", va="center", color="#cbd5e1")
        _save(fig, path)
        return

    links = _link_load_rows(model, snapshots)
    top = sorted(links, key=lambda item: item["rate_mbps"], reverse=True)[:15]
    labels = [item["label"] for item in reversed(top)]
    rates = [item["rate_mbps"] for item in reversed(top)]
    utilizations = [item["utilization_percent"] for item in reversed(top)]
    colors = [item["color"] for item in reversed(top)]

    axes[0].barh(labels, rates, color=colors)
    _style_axes(axes[0], "15 наиболее нагруженных связей", "Мбит/с", legend=False)
    axes[1].barh(labels, utilizations, color=colors)
    _style_axes(axes[1], "Использование доступной ёмкости", "% ёмкости", legend=False)
    for index, value in enumerate(utilizations):
        axes[1].text(value, index, f" {value:.3f}%", va="center", color="#e2e8f0", fontsize=8)
    _save(fig, path)


def _plot_arbitrator(
    series: dict[str, list[float]],
    dynamics: dict[str, Any],
    path: Path,
) -> None:
    """Показать вклад сигналов в решение и качество модели Купмана.

    Ляпунов и Хаусдорф подробно вынесены на панель восстановления, поэтому
    здесь они остаются только как входы управляющего решения и не дублируют
    диагностические кривые.
    """
    fig, axes = _dashboard("GNet9 · анализатор Купмана и решение L7", 1, 3, (19, 6.5))
    remap_ax, signals_ax, ablation_ax = axes.ravel()
    time = series["time_seconds"]
    _plot_line(remap_ax, time, series["remap_pressure"], "давление переназначения", "#f87171")
    _plot_line(remap_ax, time, series["koopman_forecast_risk"], "прогноз риска Купмана", "#fbbf24")
    _plot_line(remap_ax, time, series["lyapunov_remap_pressure"], "обратная связь Ляпунова", "#c084fc")
    _plot_line(remap_ax, time, series["gold_threat_proximity_risk"], "близость к критичным маршрутам золотого класса", "#38bdf8")
    remap_ax.axhline(0.20, color="#fbbf24", linewidth=1.3, linestyle="--", label="порог планирования переназначения")
    remap_values = (
        series["remap_pressure"]
        + series["koopman_forecast_risk"]
        + series["lyapunov_remap_pressure"]
        + series["gold_threat_proximity_risk"]
    )
    remap_top = min(1.02, max(0.30, max(remap_values, default=0.0) * 1.10))
    _style_axes(
        remap_ax,
        "Из чего складывается решение о переназначении",
        "нормированная оценка 0…1",
        (0, remap_top),
        legend_columns=2,
    )
    remap_ax.set_xlabel("время, с", color="#cbd5e1")

    _plot_line(signals_ax, time, series["koopman_reference_distance"], "наблюдаемое отклонение от t0", "#38bdf8")
    _plot_line(signals_ax, time, series["koopman_residual"], "невязка Купмана", "#c084fc")
    _plot_line(
        signals_ax,
        time,
        series["koopman_predicted_distance"],
        "прогнозное отклонение от t0",
        "#fb7185",
    )
    _plot_line(signals_ax, time, series["koopman_operator_drift"], "дрейф оператора", "#fbbf24")
    signal_values = (
        series["koopman_reference_distance"] + series["koopman_residual"]
        + series["koopman_predicted_distance"] + series["koopman_operator_drift"]
    )
    signal_min = min(-0.05, min(signal_values, default=0.0) * 1.10)
    signal_max = max(0.20, max(signal_values, default=0.0) * 1.10)
    signals_ax.axhline(0.0, color="#94a3b8", linewidth=0.9, alpha=0.7)
    _style_axes(
        signals_ax,
        "Наблюдение и прогноз состояния сети",
        "нормированное отклонение",
        (signal_min, signal_max),
        legend_columns=2,
    )
    signals_ax.set_xlabel("время, с", color="#cbd5e1")
    evaluation = dynamics.get("koopman_evaluation", {})
    horizon = int(evaluation.get("forecast_horizon_seconds", 0))
    prediction_slo = int(evaluation.get("prediction_slo_seconds", 0))
    signals_ax.text(
        0.99,
        0.02,
        f"Аналитический горизонт: {horizon} с · SLO решения: ≥{prediction_slo} с\n"
        "Риск — оценка, не калиброванная вероятность",
        transform=signals_ax.transAxes,
        ha="right",
        va="bottom",
        color="#94a3b8",
        fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.35", "fc": "#0f172a", "ec": "#334155", "alpha": 0.88},
    )

    _plot_line(
        ablation_ax,
        time,
        series["koopman_dmd_only_risk"],
        "только DMD Купмана",
        "#38bdf8",
    )
    _plot_line(
        ablation_ax,
        time,
        series["koopman_delay_trend_risk"],
        "DMD + причинный тренд",
        "#a78bfa",
    )
    _plot_line(
        ablation_ax,
        time,
        series["koopman_forecast_risk"],
        "гибрид с классификатором",
        "#fbbf24",
    )
    ablation_ax.axhline(0.30, color="#f87171", linewidth=1.1, linestyle="--", label="порог предупреждения")
    _style_axes(
        ablation_ax,
        "Вклад компонентов прогноза",
        "риск 0…1 (не вероятность)",
        (0, 1.02),
    )
    ablation_ax.set_xlabel("время, с", color="#cbd5e1")
    for window in _attack_windows(dynamics):
        ablation_ax.axvspan(
            window["start"],
            window["end"],
            color=ATTACK_KIND_COLORS.get(str(window.get("kind")), "#fb7185"),
            alpha=0.06,
            zorder=0,
        )
    _save(fig, path)


def _plot_attacks(series: dict[str, list[float]], dynamics: dict[str, Any], path: Path) -> None:
    fig, axes = _dashboard("GNet9 · атаки, прогноз и результат защиты", 2, 2, (15, 9))
    # Для этого графика нужен дополнительный ряд с расшифровкой событий.
    # Отдельный figure-текст сохраняет разные размеры шрифта заголовка и ленты.
    fig.suptitle("")
    fig.text(
        0.5, 0.975, "GNet9 · атаки, прогноз и результат защиты",
        ha="center", va="top", color="#f8fafc", fontsize=18, fontweight="bold",
    )
    time = series["time_seconds"]

    ax = axes[0, 0]
    _plot_line(ax, time, series["attack_intensity_percent"], "фактическая интенсивность атаки", "#fb7185")
    horizon = int(dynamics.get("koopman_evaluation", {}).get("forecast_horizon_seconds", 0))
    _plot_line(ax, time, series["precursor_confidence_percent"], "достоверность предвестников", "#38bdf8")
    _plot_line(ax, time, [value * 100.0 for value in series["koopman_forecast_risk"]], "оценка риска Купмана", "#fbbf24")
    _plot_step(
        ax,
        time,
        series["koopman_warning_percent"],
        f"подтверждённое предупреждение (анализ до {horizon} с)",
        "#a78bfa",
    )
    _plot_nonzero_line(ax, time, series["power_voltage_sag_percent"], "просадка напряжения", "#facc15")
    _plot_nonzero_line(
        ax,
        time,
        series["power_service_unavailability_percent"],
        "недоступность сервиса из-за питания",
        "#fb7185",
    )
    _plot_nonzero_line(
        ax,
        time,
        series["power_energy_depletion_percent"],
        "расход энергорезерва ИБП",
        "#f59e0b",
    )
    _style_axes(ax, "Предвестник → предупреждение → атака", "% шкалы", (0, 105), legend_columns=2)

    ax = axes[0, 1]
    _plot_line(ax, time, series["raw_attack_rate_mbps"], "до превентивной защиты", "#ef4444")
    _plot_line(ax, time, series["attack_rate_mbps"], "прошло после защиты", "#f97316")
    _plot_line(ax, time, series["blocked_attack_rate_mbps"], "заблокировано", "#34d399")
    _style_axes(ax, "Вредоносный трафик: до и после защиты", "Мбит/с")
    ax.set_yscale("symlog", linthresh=1.0)

    ax = axes[1, 0]
    _plot_line(ax, time, series["legitimate_delivery_percent"], "доставлено", "#34d399")
    _plot_line(ax, time, series["legitimate_loss_percent"], "потеряно", "#f87171")
    _style_axes(ax, "Качество легитимного трафика", "% пакетов", (0, 105))

    _plot_forecast_quality(axes[1, 1], dynamics.get("koopman_evaluation", {}))
    time_axes = (axes[0, 0], axes[0, 1], axes[1, 0])
    for axis in time_axes:
        axis.set_xlabel("время, с", color="#cbd5e1")
    attack_windows = _attack_windows(dynamics)
    for index, window in enumerate(attack_windows, start=1):
        color = ATTACK_KIND_COLORS.get(str(window.get("kind")), "#fb7185")
        for axis in time_axes:
            axis.axvspan(window["start"], window["end"], color=color, alpha=0.08, zorder=0)
        # Внутри ряда остаётся только номер: полная расшифровка вынесена в
        # отдельную ленту, иначе близкие события неизбежно перекрываются.
        axes[0, 0].text(
            (window["start"] + window["end"]) / 2.0,
            101.5 if index % 2 else 96.0,
            str(index),
            ha="center", va="bottom",
            color="#fde68a" if window.get("kind") == "power_attack" else "#fecaca",
            fontsize=7.5, fontweight="bold",
        )
    if attack_windows:
        entries = [
            f"{index} — {window['label']}"
            for index, window in enumerate(attack_windows, start=1)
        ]
        split_at = (len(entries) + 1) // 2
        fig.text(
            0.5,
            0.915,
            "События: " + "  ·  ".join(entries[:split_at])
            + "\n" + "  ·  ".join(entries[split_at:]),
            ha="center",
            va="center",
            color="#cbd5e1",
            fontsize=7.0,
            linespacing=1.35,
        )
    _save(fig, path, top=0.87 if attack_windows else 0.95)


def _plot_remapping(
    series: dict[str, list[float]],
    dynamics: dict[str, Any],
    path: Path,
) -> None:
    """Показать фактические управляющие действия и устойчивость после них."""
    fig, axes = _dashboard("GNet9 · предиктивное переназначение", 2, 3, (20, 9.5))
    time = series["time_seconds"]

    ax = axes[0, 0]
    _plot_line(ax, time, series["rerouted_flows"], "переназначено", "#22d3ee")
    _plot_line(ax, time, series["failover_flows"], "переключено на резервный сервер", "#a78bfa")
    _plot_line(ax, time, series["isolated_flows"], "временно изолировано", "#fb7185")
    _style_axes(ax, "Действия над легитимными потоками", "потоки")

    ax = axes[0, 1]
    for grade, label in (("gold", "золотой"), ("silver", "серебряный"), ("bronze", "бронзовый")):
        values = series[f"{grade}_availability_percent"]
        color = SLA_COLORS[grade]
        _plot_line(ax, time, values, f"доступно: {label}", color)
        if time and values:
            ax.fill_between(time, values, 100.0, color=color, alpha=0.07)
        restoration = series[f"{grade}_restoration_percent"]
        if any(abs(left - right) > 1e-6 for left, right in zip(values, restoration)):
            _plot_line(
                ax,
                time,
                restoration,
                f"восстановлено среди затронутых: {label}",
                color,
                linestyle="--",
            )
    _style_axes(
        ax,
        "Доступность и восстановление по классу SLA",
        "% потоков",
        (0, 105),
        legend_columns=2,
    )

    ax = axes[0, 2]
    for grade, label in (("gold", "золотой"), ("silver", "серебряный"), ("bronze", "бронзовый")):
        color = SLA_COLORS[grade]
        _plot_line(
            ax,
            time,
            series[f"{grade}_slo_compliance_percent"],
            f"требования соблюдены: {label}",
            color,
        )
        _plot_line(
            ax,
            time,
            series[f"{grade}_slo_coverage_percent"],
            f"охват оценки: {label}",
            color,
            linestyle=":",
        )
    _style_axes(
        ax,
        "Прикладные SLO и охват измеряемых метрик",
        "% потоков / метрик",
        (0, 105),
        legend_columns=2,
    )
    ax = axes[1, 0]
    _plot_line(ax, time, series["lyapunov_value"], "энергия отклонения V(t)", "#60a5fa")
    ax2 = ax.twinx()
    _plot_line(ax2, time, series["lyapunov_derivative"], "скорость dV/dt", "#22d3ee")
    _plot_line(
        ax2,
        time,
        series["finite_time_energy_growth"],
        "логарифмическая скорость роста энергии",
        "#f97316",
    )
    ax2.axhline(0.0, color="#94a3b8", linewidth=1.0, linestyle="--", label="0: граница роста/восстановления")
    ax2.set_ylabel("dV/dt и скорость роста, 1/с", color="#cbd5e1")
    ax2.tick_params(colors="#cbd5e1")
    derivative_values = series["lyapunov_derivative"] + series["finite_time_energy_growth"]
    derivative_limit = max(0.02, max((abs(value) for value in derivative_values), default=0.0) * 1.15)
    ax2.set_ylim(-derivative_limit, derivative_limit)
    _style_axes(ax, "Устойчивость по Ляпунову", "V(t), нормированная энергия", (0, None))
    _combined_legend(ax, ax2, columns=2)

    ax = axes[1, 1]
    _plot_line(ax, time, series["gold_threat_proximity_risk"], "экспозиция золотого класса к атаке", "#fb7185")
    _plot_line(ax, time, series["route_hausdorff_normalized"], "изменение маршрутов золотого класса", "#38bdf8")
    _plot_line(ax, time, series["route_edge_jaccard"], "замена рёбер маршрутов золотого класса", "#a78bfa")
    ax2 = ax.twinx()
    _plot_line(
        ax2,
        time,
        series["route_latency_stretch"],
        "удлинение задержки маршрута",
        "#fbbf24",
        linestyle="--",
    )
    ax2.axhline(1.0, color="#94a3b8", linewidth=0.9, linestyle=":", label="исходная задержка ×1")
    stretch_max = max(series["route_latency_stretch"], default=1.0)
    ax2.set_ylim(0.9, max(1.1, stretch_max * 1.08))
    ax2.set_ylabel("отношение задержки, ×", color="#cbd5e1")
    ax2.tick_params(colors="#cbd5e1")
    _style_axes(ax, "Изменение маршрутов золотого класса", "нормированное значение", (0, 1.02))
    _combined_legend(ax, ax2, columns=2)

    ax = axes[1, 2]
    resource_series = (
        ("post_remap_edge_utilization_percent", "каналы сети", "#38bdf8"),
        ("post_remap_node_utilization_percent", "сетевые узлы", "#22d3ee"),
        ("post_remap_server_cpu_percent", "серверы: ЦП", "#fbbf24"),
        ("post_remap_server_ram_percent", "серверы: ОЗУ", "#a78bfa"),
        ("post_remap_server_sessions_percent", "серверы: сеансы", "#34d399"),
    )
    for key, label, color in resource_series:
        _plot_line(ax, time, series[key], label, color)
    ax.axhline(80.0, color="#f87171", linewidth=1.0, linestyle="--", label="предел допуска 80 %")
    _style_axes(
        ax,
        "Проверка ресурсов перед переназначением",
        "максимальная прогнозная загрузка, %",
        (0, 100),
        legend_columns=2,
    )

    for axis in axes.ravel():
        axis.set_xlabel("время, с", color="#cbd5e1")
    _shade_attack_windows(axes, dynamics)
    _save(fig, path)


def _attack_windows(dynamics: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    step_seconds = float(dynamics.get("config", {}).get("step_seconds", 1.0))
    for snapshot in dynamics.get("snapshots", []):
        for event in snapshot.get("attacks", {}).get("events", []):
            temporal = event.get("temporal", {})
            group_key = str(
                event.get("attack_id")
                or f"{event.get('kind')}:{temporal.get('start_step', event.get('step_index'))}"
            )
            kind_label = ATTACK_KIND_LABELS.get(str(event.get("kind")), "Атака")
            target_label = str(event.get("target_id", ""))
            item = grouped.setdefault(group_key, {
                "start": float(snapshot.get("time_seconds", 0.0)),
                "end": float(snapshot.get("time_seconds", 0.0)) + step_seconds,
                "kind": event.get("kind"),
                "label": f"{kind_label} → {target_label}" if target_label else kind_label,
            })
            item["end"] = max(item["end"], float(snapshot.get("time_seconds", 0.0)) + step_seconds)
    return list(grouped.values())


def _shade_attack_windows(axes, dynamics: dict[str, Any]) -> None:
    """Ненавязчиво связать реакцию контроллера с фактическими атаками."""
    for window in _attack_windows(dynamics):
        color = ATTACK_KIND_COLORS.get(str(window.get("kind")), "#fb7185")
        for axis in np.asarray(axes).flat:
            axis.axvspan(window["start"], window["end"], color=color, alpha=0.055, zorder=0)


def _plot_forecast_quality(ax, evaluation: dict[str, Any]) -> None:
    """Сводная проверка прогноза без выдачи risk score за вероятность."""
    metrics = [
        float(evaluation.get("prediction_precision", 0.0)) * 100.0,
        float(evaluation.get("prediction_recall", 0.0)) * 100.0,
        float(evaluation.get("prediction_f1", 0.0)) * 100.0,
    ]
    labels = ["точность\nпредупреждений", "полнота\nатак", "F1-мера"]
    colors = ["#38bdf8", "#34d399", "#a78bfa"]
    if not evaluation or int(evaluation.get("attack_onset_count", 0)) == 0:
        ax.text(
            0.5, 0.52, "Нет завершённых атак\nдля проверки прогноза",
            transform=ax.transAxes, ha="center", va="center", color="#cbd5e1", fontsize=11,
        )
        _style_axes(ax, "Проверка предсказательной способности", "%", (0, 105), legend=False)
        return

    bars = ax.bar(labels, metrics, color=colors, width=0.58, alpha=0.9)
    for bar, value in zip(bars, metrics):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(102.0, value + 2.0),
            f"{value:.1f}%",
            ha="center", va="bottom", color="#f8fafc", fontsize=9, fontweight="bold",
        )
    horizon = int(evaluation.get("forecast_horizon_seconds", 0))
    predicted = int(evaluation.get("predicted_before_onset_count", 0))
    total = int(evaluation.get("attack_onset_count", 0))
    false_count = int(evaluation.get("false_positive_warning_count", 0))
    median_lead = float(evaluation.get("median_observed_lead_seconds", 0.0))
    prediction_slo = int(evaluation.get("prediction_slo_seconds", 0))
    slo_predicted = int(evaluation.get("prediction_slo_predicted_count", 0))
    slo_status = str(
        evaluation.get("prediction_slo_status_ru", evaluation.get("prediction_slo_status", "—"))
    )
    ax.text(
        0.5,
        0.04,
        f"Синтетический сценарий: {predicted} из {total} атак заранее · "
        f"медианное упреждение {median_lead:g} с\n"
        f"SLO ≥{prediction_slo} с: {slo_predicted}/{total} · {slo_status} · "
        f"горизонт {horizon} с · ложных предупреждений {false_count}",
        transform=ax.transAxes,
        ha="center", va="bottom", color="#cbd5e1", fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.35", "fc": "#0f172a", "ec": "#334155", "alpha": 0.90},
    )
    _style_axes(ax, "Проверка прогноза после прогона", "%", (0, 110), legend=False)


def _arrival_records(dynamics: dict[str, Any]) -> list[dict[str, Any]]:
    """Извлечь только уже состоявшиеся реализации процесса поступления атак.

    Имена ключей читаются адаптивно: старые сценарии продолжают работать, а
    отсутствие метаданных Вейбулла просто отключает дополнительный график.
    Будущее расписание или скрытые onset-метки здесь не используются.
    """
    records: dict[str, dict[str, Any]] = {}
    for snapshot in dynamics.get("snapshots", []):
        for event in snapshot.get("attacks", {}).get("events", []):
            attack_id = str(event.get("attack_id", ""))
            if not attack_id or attack_id in records:
                continue
            process = event.get("arrival_process") or event.get("weibull_process") or {}
            realization = event.get("arrival_realization") or event.get("weibull_realization") or {}
            distribution = str(
                process.get("distribution")
                or event.get("arrival_distribution")
                or ""
            ).lower()
            if "weibull" not in distribution:
                continue
            shape = _first_number(process, ("shape", "shape_k", "weibull_shape"))
            scale = _first_number(process, ("scale_seconds", "scale", "weibull_scale_seconds"))
            interval = _first_number(
                realization,
                (
                    "interval_seconds",
                    "elapsed_interval_seconds",
                    "elapsed_interarrival_seconds",
                    "realized_interarrival_seconds",
                ),
                fallback=_first_number(event, ("arrival_interval_seconds",)),
            )
            sampled_interval = _first_number(
                realization,
                ("sampled_interarrival_seconds", "raw_interarrival_seconds"),
                fallback=_first_number(event, ("arrival_sample_interval_seconds",)),
            )
            expected = _first_number(process, ("expected_interarrival_seconds", "mean_interval_seconds"))
            hazard = _first_number(realization, ("hazard_per_second", "hazard"))
            survival = _first_number(realization, ("survival_probability", "survival"))
            records[attack_id] = {
                "attack_id": attack_id,
                "kind": str(event.get("kind", "attack")),
                "target_id": str(event.get("target_id", "?")),
                "onset_seconds": float(event.get("time_seconds", snapshot.get("time_seconds", 0.0))),
                "shape": shape,
                "scale_seconds": scale,
                "expected_interval_seconds": expected,
                "interval_seconds": interval,
                "sampled_interval_seconds": sampled_interval or interval,
                "hazard_per_second": hazard,
                "survival_probability": survival,
            }
    return sorted(records.values(), key=lambda item: item["onset_seconds"])


def _plot_attack_arrivals(records: list[dict[str, Any]], path: Path) -> None:
    """Показать дискретные кадры и независимую выборку Вейбулла."""
    fig, axes = _dashboard("GNet9 · процесс поступления атак по Вейбуллу", 1, 2, (15, 6.5))
    intervals_ax, distribution_ax = axes.ravel()
    labels = [
        f"{ATTACK_KIND_LABELS.get(item['kind'], item['kind'])}\n{item['target_id']}"
        for item in records
    ]
    intervals = [float(item.get("interval_seconds", 0.0)) for item in records]
    expected = [float(item.get("expected_interval_seconds", 0.0)) for item in records]
    colors = [ATTACK_KIND_COLORS.get(item["kind"], "#94a3b8") for item in records]
    positions = np.arange(len(records))
    intervals_ax.bar(positions, intervals, color=colors, width=0.62, label="интервал в кадрах")
    intervals_ax.plot(positions, expected, color="#f8fafc", marker="D", markersize=4, linewidth=1.4, label="математическое ожидание")
    intervals_ax.set_xticks(positions, labels, rotation=20, ha="right")
    _style_axes(
        intervals_ax,
        "Интервалы между событиями в дискретной модели",
        "секунды",
        legend_columns=2,
    )

    shape = next((float(item["shape"]) for item in records if float(item.get("shape", 0.0)) > 0.0), 0.0)
    scale = next((float(item["scale_seconds"]) for item in records if float(item.get("scale_seconds", 0.0)) > 0.0), 0.0)
    if shape > 0.0 and scale > 0.0:
        max_interval = max(intervals + [scale * 2.0, 1.0])
        elapsed = np.linspace(0.0, max_interval, 180)
        normalized = np.maximum(elapsed / scale, 1e-12)
        survival = np.exp(-(elapsed / scale) ** shape)
        hazard = (shape / scale) * normalized ** (shape - 1.0)
        hazard[0] = hazard[1] if len(hazard) > 1 else 0.0
        _plot_line(distribution_ax, elapsed, survival * 100.0, "вероятность отсутствия атаки S(t)", "#38bdf8")
        hazard_ax = distribution_ax.twinx()
        _plot_line(hazard_ax, elapsed, hazard, "интенсивность риска h(t)", "#fb7185")
        sampled_intervals = [float(item.get("sampled_interval_seconds", 0.0)) for item in records]
        realized_survival = [float(item.get("survival_probability", 0.0)) * 100.0 for item in records]
        realized_hazard = [float(item.get("hazard_per_second", 0.0)) for item in records]
        distribution_ax.scatter(
            sampled_intervals,
            realized_survival,
            s=28,
            c=colors,
            edgecolors="#e2e8f0",
            linewidths=0.55,
            zorder=5,
            label="S(t) независимой выборки",
        )
        hazard_ax.scatter(
            sampled_intervals,
            realized_hazard,
            s=24,
            marker="x",
            c=colors,
            linewidths=1.2,
            zorder=5,
            label="h(t) независимой выборки",
        )
        hazard_ax.set_ylabel("h(t), 1/с", color="#cbd5e1")
        hazard_ax.tick_params(colors="#cbd5e1")
        _style_axes(
            distribution_ax,
            f"Процесс Вейбулла: k={shape:.2f}, масштаб λ={scale:.1f} с",
            "S(t), %",
            (0, 105),
        )
        _combined_legend(distribution_ax, hazard_ax)
        distribution_ax.text(
            0.5,
            0.03,
            "Независимые интервалы; события за границей окна не рисуются",
            transform=distribution_ax.transAxes,
            ha="center", va="bottom", color="#94a3b8", fontsize=7.4,
        )
    else:
        distribution_ax.text(
            0.5, 0.5, "Параметры формы и масштаба\nв сценарии не экспортированы",
            transform=distribution_ax.transAxes, ha="center", va="center", color="#cbd5e1",
        )
        _style_axes(distribution_ax, "Распределение Вейбулла", "", legend=False)
    intervals_ax.set_xlabel("события в порядке появления", color="#cbd5e1")
    distribution_ax.set_xlabel("время с предыдущего события, с", color="#cbd5e1")
    _save(fig, path)


def _first_number(source: Any, keys: tuple[str, ...], *, fallback: float = 0.0) -> float:
    if not isinstance(source, dict):
        return float(fallback)
    for key in keys:
        value = source.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return float(fallback)


def _link_load_rows(model, snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    peak_bytes: dict[frozenset[str], int] = {}
    interval_by_edge: dict[frozenset[str], int] = {}
    for snapshot in snapshots:
        flows = snapshot.get("traffic", {}).get("flows", [])
        current: dict[frozenset[str], int] = {}
        for flow in flows:
            interval = int(flow.get("interval_seconds", 1))
            for source, target in zip(flow.get("route", []), flow.get("route", [])[1:]):
                edge = frozenset((source, target))
                current[edge] = current.get(edge, 0) + int(flow.get("wire_bytes", 0))
                interval_by_edge[edge] = interval
        for edge, wire_bytes in current.items():
            peak_bytes[edge] = max(peak_bytes.get(edge, 0), wire_bytes)

    rows = []
    for edge, wire_bytes in peak_bytes.items():
        source, target = sorted(edge)
        capacity = float(model.graph.edges[source, target].get("capacity_mbps", 0.0))
        rate = wire_bytes * 8.0 / max(interval_by_edge.get(edge, 1), 1) / 1_000_000.0
        roles = {model.graph.nodes[node].get("role") for node in edge}
        color = "#22d3ee" if "mobile-subscriber" in roles else "#fbbf24" if "fixed-subscriber" in roles else "#60a5fa"
        rows.append({
            "label": f"{source} ↔ {target}",
            "rate_mbps": rate,
            "utilization_percent": rate / max(capacity, 1e-9) * 100.0,
            "color": color,
        })
    return rows


def _dashboard(title: str, rows: int, columns: int, size: tuple[float, float]):
    fig, axes = plt.subplots(rows, columns, figsize=size, squeeze=False)
    fig.suptitle(title, color="#f8fafc", fontsize=18, fontweight="bold", y=0.98)
    _dark_figure(fig, axes)
    return fig, axes


def _dark_figure(fig, axes) -> None:
    fig.patch.set_facecolor("#0f172a")
    for ax in np.asarray(axes).flat:
        ax.set_facecolor("#111827")


def _plot_line(
    ax,
    x,
    y,
    label: str,
    color: str,
    *,
    linestyle: str = "-",
) -> None:
    """Нарисовать читаемый ряд без маркера на каждом из десятков шагов."""
    mark_every = max(1, len(x) // 18) if len(x) else 1
    ax.plot(
        x,
        y,
        marker="o",
        markevery=mark_every,
        markersize=3.3,
        linewidth=1.9,
        linestyle=linestyle,
        color=color,
        label=label,
    )


def _plot_step(ax, x, y, label: str, color: str) -> None:
    ax.step(x, y, where="post", linewidth=1.9, color=color, label=label)


def _plot_nonzero_line(ax, x, y, label: str, color: str) -> None:
    """Не добавлять в легенду ряд, который не относится к этому сценарию."""
    if any(abs(float(value)) > 1e-12 for value in y):
        _plot_line(ax, x, y, label, color)


def _style_axes(
    ax,
    title: str,
    ylabel: str,
    ylim=None,
    *,
    legend: bool = True,
    legend_columns: int = 1,
) -> None:
    ax.set_title(title, color="#f8fafc", fontsize=12.5, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, color="#cbd5e1")
    ax.tick_params(colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, axis="y", color="#334155", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend(
            loc="best",
            ncol=legend_columns,
            facecolor="#172033",
            edgecolor="#475569",
            labelcolor="#e2e8f0",
            fontsize=7.7,
        )


def _combined_legend(ax, ax2, *, columns: int = 1) -> None:
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    current_legend = ax.get_legend()
    if current_legend is not None:
        current_legend.remove()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="best",
        ncol=columns,
        facecolor="#172033",
        edgecolor="#475569",
        labelcolor="#e2e8f0",
        fontsize=7.7,
    )


def _annotate_bars(ax, values: list[float], pattern: str) -> None:
    for index, value in enumerate(values):
        ax.text(index, value + max(values or [1]) * 0.025, pattern.format(value), ha="center", color="#e2e8f0", fontsize=9)


def _save(fig, path: Path, *, top: float = 0.95) -> None:
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _aggregate(snapshot: dict[str, Any], level: str, metric: str, statistic: str) -> float:
    values = snapshot.get("arbitrator", {}).get("level_metric_aggregates", {}).get(level, {}).get("metrics", {}).get(metric, {})
    return float(values.get(statistic, 0.0))


def _nested_float(source: dict[str, Any], path: tuple[str, ...], *, default: float = 0.0) -> float:
    value: Any = source
    for key in path:
        if not isinstance(value, dict):
            return default
        if key not in value:
            return default
        value = value[key]
    return float(value if value is not None else default)
