"""L7 arbitrator analysis over G-Net tensor state.

The arbitrator does not own topology construction or time stepping. It receives
tensor snapshots from `dynamics.py`, reduces them to stable metrics and prepares
the fields later needed for remapping, Koopman/DMD and Lyapunov analysis.
"""

from __future__ import annotations

from typing import Any
import math

from .models import NetworkModel, StateTensor


# Keep the observed set intentionally small: these metrics are enough for a
# readable first remapping signal and a compact state vector for time-series math.
ARBITRATOR_OBSERVED_METRICS = {
    "L1": ("sla_margin", "traffic_intensity_rho"),
    "L2": ("cpu_load_percent", "ram_load_percent", "stability_margin"),
    "EDGE": ("loss_probability", "stability_margin", "utilization"),
    "L8": ("terrain_risk",),
    "L7": (
        "gold_threat_minimum_distance_ms",
        "lyapunov_value",
        "lyapunov_delta",
        "koopman_residual",
        "remap_pressure",
    ),
}

STATE_VECTOR_METRICS = (
    ("L1", "sla_margin", "min"),
    ("L1", "traffic_intensity_rho", "mean"),
    ("L2", "cpu_load_percent", "max"),
    ("L2", "ram_load_percent", "max"),
    ("L2", "stability_margin", "min"),
    ("EDGE", "loss_probability", "max"),
    ("EDGE", "stability_margin", "min"),
    ("EDGE", "utilization", "mean"),
    ("L8", "terrain_risk", "max"),
    ("L7", "gold_threat_minimum_distance_ms", "mean"),
    ("L7", "koopman_residual", "mean"),
    ("L7", "remap_pressure", "mean"),
)


TRIGGER_BANDS = {
    "hausdorff": (0.25, 0.50, 0.75),
    "koopman": (0.25, 0.35, 0.70),
    "lyapunov": (0.20, 0.60, 0.85),
    "decision": (0.20, 0.50, 0.75),
}
STATE_NAMES = ("normal", "watch", "alert", "critical")
STATE_LABELS = ("норма", "наблюдение", "защита", "срочная защита")
THREAT_CONTROLS = {
    "dos": ["rate_limit_attack_traffic", "protect_gold_paths"],
    "ddos": ["rate_limit_attack_traffic", "protect_gold_paths", "distribute_service_load"],
    "syn_flood": ["enable_syn_protection", "protect_endpoint_queue"],
    "brute_force": ["enforce_account_lockout_policy", "review_failed_authentication_events"],
    "power_attack": ["check_power_domain", "prefer_nodes_with_energy_reserve"],
}


def trigger_state(value: float, signal: str) -> dict[str, Any]:
    """Назвать диапазон, сохранив исходное число и точные границы."""
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Оценка состояния должна быть конечным неотрицательным числом")
    bounds = TRIGGER_BANDS[signal]
    index = sum(value >= bound for bound in bounds)
    return {
        "value": value, "state": STATE_NAMES[index], "label_ru": STATE_LABELS[index],
        "lower_inclusive": 0.0 if index == 0 else bounds[index - 1],
        "upper_exclusive": bounds[index] if index < 3 else None,
    }


def build_trigger_card(
    *, hausdorff: float, risk: float, lyapunov_pressure: float,
    decision_pressure: float, confirmed: bool, attack_kind: str,
    source_confirmed: bool,
) -> dict[str, Any]:
    """Карта состояния связывает наблюдения с допустимой защитой."""
    signals = {
        "hausdorff": trigger_state(hausdorff, "hausdorff"),
        "koopman": trigger_state(risk, "koopman"),
        "lyapunov": trigger_state(lyapunov_pressure, "lyapunov"),
        "decision": trigger_state(decision_pressure, "decision"),
    }
    controls = sorted({control for kind in attack_kind.split("+") for control in THREAT_CONTROLS.get(kind, [])})
    # Порог исполнения сохраняет строгую границу > 0,20. Одиночное
    # отклонение любого расчёта не подтверждает источник или класс угрозы.
    observe_limit, protection_limit, emergency_limit = TRIGGER_BANDS["decision"]
    if decision_pressure <= observe_limit:
        action, stage = "NO_REMAP", "observe"
    elif not confirmed:
        action, stage = "OBSERVE_PRECURSOR", "confirm_observation"
    else:
        action = "PLAN_REMAP"
        stage = "prepare" if decision_pressure < protection_limit else "protect" if decision_pressure < emergency_limit else "emergency"
    if source_confirmed and confirmed and stage == "emergency" and set(attack_kind.split("+")) & {"dos", "ddos", "syn_flood"}:
        controls.append("isolate_confirmed_attack_sources")
    return {
        "signals": signals, "attack_kind": attack_kind, "confirmed": confirmed,
        "action": action, "response_stage": stage, "controls": controls if confirmed else [],
        "source_quarantine_allowed": source_confirmed and confirmed and bool(set(attack_kind.split("+")) & {"dos", "ddos", "syn_flood"}),
        "reason_ru": (
            "Оснований для изменения маршрутов нет." if action == "NO_REMAP" else
            "Сигнал требует повторного наблюдения; источник не изолируется." if not confirmed else
            f"Угроза подтверждена; диапазон реакции: {signals['decision']['label_ru']}."
        ),
    }


def build_arbitrator_view(
    model: NetworkModel,
    tensor_state: dict[str, Any],
    *,
    observation: dict[str, Any] | None = None,
    state_hausdorff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return L7 analysis and a no-remap decision for the current snapshot.

    Healthy baseline dynamics should keep `remap.needed` false. Future degraded
    modes can feed the same tensor-state structure and let these thresholds start
    proposing remap actions.
    """
    aggregates = level_metric_aggregates(tensor_state, observed_only=True)
    full_aggregates = level_metric_aggregates(tensor_state, observed_only=False)
    l7_tensor = model.graph.nodes["ARB"].get("tensor") if "ARB" in model.graph.nodes else None
    l7_metrics = tensor_metrics(l7_tensor) if isinstance(l7_tensor, StateTensor) else {}
    observation = observation or {}
    state_hausdorff = state_hausdorff or {}
    attack_signals = _attack_signals(observation)
    remap_pressure = remap_pressure_from_tensors(
        aggregates,
        l7_metrics,
        attack_pressure=attack_signals["attack_pressure"],
        state_hausdorff_drift=float(state_hausdorff.get("normalized_distance", 0.0)),
    )
    lyapunov_value = lyapunov_value_from_tensors(aggregates, l7_metrics, remap_pressure)
    koopman_residual = koopman_residual_from_tensors(aggregates, l7_metrics, remap_pressure)
    state_vector = build_state_vector(
        full_aggregates,
        observation=observation,
        state_hausdorff=state_hausdorff,
    )
    attack_active = bool(observation.get("attacks", {}).get("active", False))
    needs_remap = remap_pressure > TRIGGER_BANDS["decision"][0]

    return {
        "node_id": "ARB",
        "input_tensor_counts": tensor_state["counts"],
        "level_metric_aggregates": aggregates,
        "state_vector": state_vector,
        "analysis": {
            "gold_threat_minimum_distance_ms": l7_metrics.get(
                "gold_threat_minimum_distance_ms",
                0.0,
            ),
            "lyapunov_value": lyapunov_value,
            "lyapunov_delta": l7_metrics.get("lyapunov_delta", 0.0),
            "koopman_residual": koopman_residual,
            "remap_pressure": remap_pressure,
            "decision_confidence": decision_confidence(remap_pressure, l7_metrics),
            "state_hausdorff": state_hausdorff,
            **attack_signals,
        },
        "critical_node_hierarchy": critical_node_hierarchy(model),
        "observations": observation,
        "remap": {
            "needed": needs_remap,
            "action": "PLAN_REMAP" if needs_remap else "NO_REMAP",
            "reason": (
                "healthy_stationary_baseline" if not needs_remap
                else "mitre_attack_observed" if attack_active else "tensor_threshold_pressure"
            ),
            "candidate_actions": [] if not needs_remap else [
                "rate_limit_attack_traffic", "protect_gold_paths", "reroute_high_pressure_flows"
            ],
        },
    }


def critical_node_hierarchy(model: NetworkModel, *, limit: int = 12) -> list[dict[str, Any]]:
    """Return the ranked L2 КВУ view used by the arbitrator and UI.

    It intentionally reads the hierarchy prepared by ``mark_critical_nodes``
    instead of recalculating routes in the control loop.
    """
    rows = []
    for node_id, attrs in model.graph.nodes(data=True):
        protection = attrs.get("critical_protection", {})
        if attrs.get("level") != "L2" or protection.get("kvu_rank") is None:
            continue
        rows.append(
            {
                "node_id": node_id,
                "role": attrs.get("role"),
                "kvu_rank": int(protection["kvu_rank"]),
                "kvu_tier": protection.get("kvu_tier"),
                "is_kvu": bool(protection.get("is_kvu")),
                "gold_subscriber_count": int(protection.get("gold_subscriber_count", 0)),
                "direct_gold_subscriber_count": int(protection.get("direct_gold_subscriber_count", 0)),
                "gold_transit_flow_count": int(protection.get("gold_transit_flow_count", 0)),
            }
        )
    return sorted(rows, key=lambda row: (row["kvu_rank"], row["node_id"]))[:limit]


def evaluate_defense_game(
    model: NetworkModel,
    *,
    target_ids: list[str],
    threat_pressure: float,
    matrix_koopman_pressure: float,
) -> dict[str, Any]:
    """Evaluate a small defender--attacker normal-form game for L7.

    The result is a transparent finite game, not a claim that an adversary
    actually follows the model.  A pure Nash equilibrium is reported only when
    mutual best responses exist.  Otherwise the output names its conservative
    maximin fallback rather than mislabelling it as Nash equilibrium.
    """
    hierarchy = critical_node_hierarchy(model, limit=100)
    max_gold = max((row["gold_subscriber_count"] for row in hierarchy), default=0)
    target_rows = []
    for target_id in target_ids:
        attrs = model.graph.nodes[target_id] if target_id in model.graph else {}
        protection = attrs.get("critical_protection", {})
        target_rows.append(
            {
                "target_id": target_id,
                "gold_subscriber_count": int(protection.get("gold_subscriber_count", 0)),
                "gold_transit_flow_count": int(protection.get("gold_transit_flow_count", 0)),
                "critical_involvement_coefficient": float(
                    protection.get("critical_involvement_coefficient", 0.0)
                ),
                "kvu_rank": protection.get("kvu_rank"),
            }
        )
    target_criticality = max(
        (
            max(
                row["gold_subscriber_count"] / max(max_gold, 1),
                row["critical_involvement_coefficient"],
            )
            for row in target_rows
        ),
        default=(0.25 if hierarchy else 0.0),
    )
    target_criticality = min(1.0, max(0.0, target_criticality))
    effective_threat = min(
        1.0,
        max(float(threat_pressure), 0.35 * float(matrix_koopman_pressure)),
    )
    defender_actions = (
        ("NO_REMAP", 0.00, 0.00),
        ("OBSERVE_PRECURSOR", 0.12, 0.025),
        ("PLAN_REMAP", 0.58, 0.12),
        ("ISOLATE_CONFIRMED_SOURCES_AND_REMAP", 0.76, 0.23),
    )
    attacker_actions = (("MAINTAIN", 0.72), ("ESCALATE", 1.00), ("DISTRIBUTE", 0.88))
    matrix: list[dict[str, Any]] = []
    for defense_name, effectiveness, action_cost in defender_actions:
        for attack_name, attack_multiplier in attacker_actions:
            damage = min(
                1.0,
                effective_threat * attack_multiplier * (0.25 + 0.75 * target_criticality) * (1.0 - effectiveness),
            )
            matrix.append(
                {
                    "defender_action": defense_name,
                    "attacker_action": attack_name,
                    "defender_payoff": round(1.0 - damage - action_cost, 6),
                    "attacker_payoff": round(damage, 6),
                }
            )

    pure_equilibria: list[dict[str, Any]] = []
    for cell in matrix:
        defender_best = max(
            item["defender_payoff"]
            for item in matrix
            if item["attacker_action"] == cell["attacker_action"]
        )
        attacker_best = max(
            item["attacker_payoff"]
            for item in matrix
            if item["defender_action"] == cell["defender_action"]
        )
        if cell["defender_payoff"] >= defender_best - 1e-12 and cell["attacker_payoff"] >= attacker_best - 1e-12:
            pure_equilibria.append(cell)
    if pure_equilibria:
        selected = max(pure_equilibria, key=lambda cell: cell["defender_payoff"])
        selection_mode = "pure_strategy_nash_equilibrium"
    else:
        worst_case = {
            action: min(
                item["defender_payoff"] for item in matrix if item["defender_action"] == action
            )
            for action, _, _ in defender_actions
        }
        selected_action = max(worst_case, key=worst_case.get)
        selected = max(
            (item for item in matrix if item["defender_action"] == selected_action),
            key=lambda item: item["attacker_payoff"],
        )
        selection_mode = "maximin_fallback_no_pure_nash_equilibrium"
    return {
        "model": "finite_defender_attacker_normal_form_game",
        "semantics_ru": (
            "Игровая оценка помогает арбитру сравнить цену защиты и ожидаемый ущерб; "
            "она не моделирует волю реального нарушителя и не заменяет подтверждение телеметрией."
        ),
        "target_rows": target_rows,
        "target_criticality": round(target_criticality, 6),
        "threat_pressure": round(effective_threat, 6),
        "matrix_koopman_pressure": round(float(matrix_koopman_pressure), 6),
        "payoff_matrix": matrix,
        "pure_nash_equilibria": pure_equilibria,
        "equilibrium_found": bool(pure_equilibria),
        "selection_mode": selection_mode,
        "selected": selected,
        "recommended_action": selected["defender_action"],
    }


def level_metric_aggregates(tensor_state: dict[str, Any], *, observed_only: bool) -> dict[str, dict[str, Any]]:
    """Aggregate tensor metrics by level using min, max and mean."""
    result: dict[str, dict[str, Any]] = {}
    for level, tensors in tensor_state["by_level"].items():
        allowed_metrics = set(ARBITRATOR_OBSERVED_METRICS.get(level, ())) if observed_only else None
        metric_values: dict[str, list[float]] = {}
        for item in tensors:
            for metric_name, value in item["metrics"].items():
                if allowed_metrics is not None and metric_name not in allowed_metrics:
                    continue
                metric_values.setdefault(metric_name, []).append(float(value))

        result[level] = {
            "tensor_count": len(tensors),
            "metrics": {
                metric_name: {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                }
                for metric_name, values in metric_values.items()
            },
        }
    return result


def build_state_vector(
    aggregates: dict[str, dict[str, Any]],
    *,
    observation: dict[str, Any] | None = None,
    state_hausdorff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact numeric vector for Koopman/DMD and Lyapunov pipelines."""
    metric_names = [
        f"{level}.{metric_name}.{statistic}"
        for level, metric_name, statistic in STATE_VECTOR_METRICS
    ]
    vector = [
        aggregate_metric(aggregates, level, metric_name, statistic, 0.0)
        for level, metric_name, statistic in STATE_VECTOR_METRICS
    ]
    observation = observation or {}
    attacks = observation.get("attacks", {})
    traffic = observation.get("traffic", {})
    routing = attacks.get("routing", observation.get("routing", {}))
    sla_tiers = {
        str(item.get("sla_grade")): item
        for item in attacks.get("sla_restoration", {}).get("tiers", [])
    }
    observed_names = [
        "OBS.attack_active_count", "OBS.attack_intensity_ratio", "OBS.attack_rate_mbps",
        "OBS.legitimate_loss_ratio", "OBS.legitimate_delivery_ratio",
        "OBS.impacted_gold_flow_count", "OBS.gold_sla_compliance_ratio",
        "OBS.maximum_target_utilization_percent",
        "OBS.precursor_count", "OBS.precursor_confidence",
        "OBS.suspicious_source_count", "OBS.source_entropy_ratio",
        "OBS.scan_rate_pps", "OBS.traffic_acceleration_ratio",
        "OBS.syn_backlog_growth_ratio", "OBS.power_control_anomaly_ratio",
        "OBS.power_voltage_sag_ratio", "OBS.battery_discharge_rate_ratio",
        "OBS.target_concentration_ratio",
        "OBS.authentication_failure_rate_per_second",
        "OBS.authentication_failure_ratio", "OBS.account_lockout_pressure",
        "OBS.state_hausdorff_distance", "OBS.state_hausdorff_normalized",
        "OBS.coordinate_hausdorff_distance",
        "OBS.rerouted_gold_flow_count", "OBS.rerouted_silver_flow_count",
        "OBS.rerouted_bronze_flow_count", "OBS.isolated_flow_count",
        "OBS.service_failover_count", "OBS.route_change_ratio",
        "OBS.maximum_post_remap_utilization_percent",
        "OBS.gold_delivery_ratio", "OBS.silver_delivery_ratio", "OBS.bronze_delivery_ratio",
    ]
    observed_vector = [
        float(attacks.get("active_count", 0.0)),
        float(attacks.get("maximum_intensity_ratio", 0.0)),
        float(attacks.get("attack_rate_mbps", 0.0)),
        float(attacks.get("legitimate_loss_ratio", traffic.get("observed_loss_ratio", 0.0))),
        float(attacks.get("legitimate_delivery_ratio", 1.0)),
        float(attacks.get("impacted_gold_flow_count", 0.0)),
        float(attacks.get("gold_sla_compliance_ratio", 1.0)),
        float(attacks.get("maximum_target_utilization_percent", 0.0)),
        float(attacks.get("precursor_count", 0.0)),
        float(attacks.get("precursor_confidence", 0.0)),
        float(attacks.get("suspicious_source_count", 0.0)),
        float(attacks.get("source_entropy_ratio", 0.0)),
        float(attacks.get("scan_rate_pps", 0.0)),
        float(attacks.get("traffic_acceleration_ratio", 0.0)),
        float(attacks.get("syn_backlog_growth_ratio", 0.0)),
        float(attacks.get("power_control_anomaly_ratio", 0.0)),
        float(attacks.get("power_voltage_sag_ratio", 0.0)),
        float(attacks.get("battery_discharge_rate_ratio", 0.0)),
        float(attacks.get("target_concentration_ratio", 0.0)),
        float(attacks.get("authentication_failure_rate_per_second", 0.0)),
        float(attacks.get("authentication_failure_ratio", 0.0)),
        float(attacks.get("account_lockout_pressure", 0.0)),
        float((state_hausdorff or {}).get("distance") or 0.0),
        float((state_hausdorff or {}).get("normalized_distance", 0.0)),
        float((state_hausdorff or {}).get("coordinate_hausdorff_distance") or 0.0),
        float(routing.get("by_sla", {}).get("gold", {}).get("rerouted_flow_count", 0.0)),
        float(routing.get("by_sla", {}).get("silver", {}).get("rerouted_flow_count", 0.0)),
        float(routing.get("by_sla", {}).get("bronze", {}).get("rerouted_flow_count", 0.0)),
        float(routing.get("isolated_flow_count", 0.0)),
        float(routing.get("service_failover_flow_count", 0.0)),
        float(routing.get("route_change_ratio", 0.0)),
        float(routing.get("maximum_projected_utilization_percent", 0.0)),
        _tier_compliance_ratio(sla_tiers, "gold"),
        _tier_compliance_ratio(sla_tiers, "silver"),
        _tier_compliance_ratio(sla_tiers, "bronze"),
    ]
    return {
        "metric_names": metric_names + observed_names,
        "vector": vector + observed_vector,
    }


def _tier_compliance_ratio(tiers: dict[str, dict[str, Any]], grade: str) -> float:
    """Вернуть долю SLA-совместимых потоков уровня, сохранив идеальный t0=1."""
    tier = tiers.get(grade)
    if not tier:
        return 1.0
    return float(tier.get("sla_compliant_flow_count", 0.0)) / max(float(tier.get("flow_count", 0.0)), 1.0)


def remap_pressure_from_tensors(
    aggregates: dict[str, dict[str, Any]],
    l7_metrics: dict[str, float],
    *,
    attack_pressure: float = 0.0,
    state_hausdorff_drift: float = 0.0,
) -> float:
    l1_min_sla = aggregate_metric(aggregates, "L1", "sla_margin", "min", 1.0)
    l2_max_cpu = aggregate_metric(aggregates, "L2", "cpu_load_percent", "max", 0.0)
    edge_min_stability = aggregate_metric(aggregates, "EDGE", "stability_margin", "min", 1.0)
    edge_max_loss = aggregate_metric(aggregates, "EDGE", "loss_probability", "max", 0.0)
    terrain_max_risk = aggregate_metric(aggregates, "L8", "terrain_risk", "max", 0.0)
    # A non-zero state Hausdorff value records routine observed-versus-t0
    # variation as well as damage. It remains visible to the analyst and the
    # predictor, but only a material normalised divergence can independently
    # request remapping; otherwise a healthy packet snapshot would create a
    # false L7 pressure.
    state_hausdorff_remap_pressure = max(
        0.0, (state_hausdorff_drift - 0.25) / 0.75
    )

    pressures = [
        (0.60 - l1_min_sla) / 0.60,
        (l2_max_cpu - 70.0) / 30.0,
        (0.20 - edge_min_stability) / 0.20,
        (edge_max_loss - 0.003) / 0.002,
        (terrain_max_risk - 0.50) / 0.50,
        l7_metrics.get("remap_pressure", 0.0),
        attack_pressure,
        state_hausdorff_remap_pressure,
    ]
    pressure = max(0.0, min(1.0, max(pressures)))
    # Keep every positive mathematical component in the diagnostic value.  The
    # remapping decision has its own 0.20 control threshold, so small values
    # remain observable without creating a false mitigation action.
    return round(pressure, 6)


def _attack_signals(observation: dict[str, Any]) -> dict[str, float]:
    attacks = observation.get("attacks", {})
    active_count = float(attacks.get("active_count", 0.0))
    intensity = float(attacks.get("maximum_intensity_ratio", 0.0))
    resource = float(attacks.get("target_resource_pressure", 0.0))
    loss = float(attacks.get("legitimate_loss_ratio", 0.0))
    gold_compliance = float(attacks.get("gold_sla_compliance_ratio", 1.0))
    precursor_confidence = float(attacks.get("precursor_confidence", 0.0))
    attack_pressure = max(resource, intensity * 0.65, min(1.0, loss * 4.0), 1.0 - gold_compliance)
    return {
        "attack_active_count": active_count,
        "attack_intensity_ratio": intensity,
        "attack_pressure": round(attack_pressure, 6),
        "gold_sla_compliance_ratio": gold_compliance,
        "precursor_active_count": float(attacks.get("precursor_count", 0.0)),
        "precursor_confidence": precursor_confidence,
        "early_warning_pressure": round(precursor_confidence, 6),
    }


def lyapunov_value_from_tensors(
    aggregates: dict[str, dict[str, Any]],
    l7_metrics: dict[str, float],
    remap_pressure: float,
) -> float:
    edge_mean_stability = aggregate_metric(aggregates, "EDGE", "stability_margin", "mean", 1.0)
    l1_mean_sla = aggregate_metric(aggregates, "L1", "sla_margin", "mean", 1.0)
    baseline_value = l7_metrics.get("lyapunov_value", 0.0)
    stress = (1.0 - edge_mean_stability) * 0.10 + (1.0 - l1_mean_sla) * 0.10 + remap_pressure * 0.25
    return round(baseline_value + stress, 6)


def koopman_residual_from_tensors(
    aggregates: dict[str, dict[str, Any]],
    l7_metrics: dict[str, float],
    remap_pressure: float,
) -> float:
    edge_mean_loss = aggregate_metric(aggregates, "EDGE", "loss_probability", "mean", 0.0)
    l2_mean_cpu = aggregate_metric(aggregates, "L2", "cpu_load_percent", "mean", 0.0)
    baseline_residual = l7_metrics.get("koopman_residual", 0.0)
    residual = baseline_residual + edge_mean_loss * 10.0 + max(0.0, l2_mean_cpu - 50.0) / 1000.0 + remap_pressure * 0.20
    return round(residual, 6)


def decision_confidence(remap_pressure: float, l7_metrics: dict[str, float]) -> float:
    baseline_confidence = l7_metrics.get("decision_confidence", 0.90)
    return round(max(0.0, min(1.0, baseline_confidence - remap_pressure * 0.45)), 6)


def aggregate_metric(
    aggregates: dict[str, dict[str, Any]],
    level: str,
    metric_name: str,
    statistic: str,
    default: float,
) -> float:
    metric = aggregates.get(level, {}).get("metrics", {}).get(metric_name)
    if not metric:
        return default
    return float(metric.get(statistic, default))


def tensor_metrics(tensor: StateTensor) -> dict[str, float]:
    return {
        metric_name: float(tensor.data[index[0]])
        for metric_name, index in tensor.metric_index.items()
    }
