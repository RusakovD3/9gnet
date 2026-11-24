"""Discrete GNet9 observations, attacks, forecasts and human recommendations.

The compatibility name :func:`simulate_stationary_dynamics` is retained from
the original healthy-baseline implementation.  It now also runs the MITRE and
predictive scenarios, causal Koopman/DMD analysis, Lyapunov analysis and
recommendations for a human operator while preserving the ideal ``t0`` reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Literal

import networkx as nx

from .attacks import (
    active_attack_events,
    advance_power_runtime_state,
    attack_catalog,
    observe_attack_precursors,
    predictive_demo_minimum_steps,
)
from .arbitrator import build_arbitrator_view, tensor_metrics
from .constants import (
    ACCESS_DEVICE_ROLES,
    DYNAMICS_STEP_SECONDS,
    DYNAMICS_STEPS,
    FIXED_SUBSCRIBERS_PER_AGG,
    MOBILE_SUBSCRIBERS_PER_AGG,
)
from .koopman import (
    KOOPMAN_WARNING_RISK_THRESHOLD,
    KoopmanOnlineAnalyzer,
    apply_koopman_to_arbitrator,
)
from .metrics import (
    StateTensorHausdorffReference,
    prepare_state_hausdorff_reference,
    state_tensor_hausdorff_view,
)
from .models import NetworkModel, StateTensor
from .kalman import DEFAULT_PROCESS_VARIANCE, DEFAULT_MEASUREMENT_VARIANCE, validate_noise_variances
from .hypernetwork import hypernetwork_embedding
from .l1_d0sl import SLA_GRADE_WEIGHTS, weighted_sla_compliance
from .packet_simulator import TRAFFIC_APPS, simulate_packet_snapshot
from .routing import FixedTopologyRoutes
from .runtime_metrics import LINK_LOAD_RATIO_ORIGIN, compact_runtime_state
from .service_forecast import ServiceSLAForecaster, _score_window, evaluate_service_forecasts, operator_warnings
from .route_stress import RouteStressMonitor, evaluate_route_stress
from .tensor_matrix import (
    build_tensor_matrix_view,
    tensor_matrix_export,
)


TENSOR_LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "EDGE")
SnapshotDetail = Literal["full", "tensor", "summary"]
PacketDetail = Literal["summary", "flows", "sample"]
AttackScenario = Literal["none", "mitre-demo", "predictive-demo"]


@dataclass(frozen=True)
class DynamicsConfig:
    """Discrete simulation clock and export-detail settings.

    `snapshot_detail` controls how much graph/tensor data is written per step:
    full = graph + all tensors, tensor = all tensors without graph lists,
    summary = only compact tensor counts, state vector and arbitrator aggregates.
    """

    step_seconds: int = DYNAMICS_STEP_SECONDS
    step_count: int = DYNAMICS_STEPS
    include_t0: bool = True
    include_packet_simulation: bool = True
    snapshot_detail: SnapshotDetail = "full"
    packet_detail: PacketDetail = "sample"
    packet_sample_limit: int = 48
    attack_scenario: AttackScenario = "none"
    attack_seed: int = 42
    attack_start_step: int | None = None
    attack_interval_steps: int | None = None
    prediction_slo_seconds: int | None = None
    warning_risk_threshold: float | None = None
    warning_threshold_origin: str = "model_default_not_externally_validated"
    kalman_process_variance: float = DEFAULT_PROCESS_VARIANCE
    kalman_measurement_variance: float = DEFAULT_MEASUREMENT_VARIANCE

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration_seconds"] = self.step_seconds * self.step_count
        data["resolved_prediction_slo_seconds"] = _resolved_prediction_slo_seconds(self)
        data["resolved_warning_risk_threshold"] = _resolved_warning_risk_threshold(self)
        return data


def simulate_stationary_dynamics(
    model: NetworkModel, config: DynamicsConfig | None = None, *, compact_traffic_history: bool = False,
) -> dict[str, Any]:
    """Return baseline or attacked snapshots using the compatibility API name."""
    config = config or DynamicsConfig()
    _validate_config(config)
    health = validate_healthy_baseline(model)
    if not health["ok"]:
        raise ValueError(f"Cannot simulate dynamics from unhealthy baseline: {health['violation_count']} violations")

    snapshots: list[dict[str, Any]] = []
    sla_forecaster = ServiceSLAForecaster(config.step_seconds, retain_trace=True)
    route_monitor = RouteStressMonitor(config.step_seconds)
    route_views = []
    impacts_by_flow = {}
    previous_impacts = set()
    sla_compute_ms = []
    power_runtime_state: dict[str, dict[str, Any]] = {}
    routing = FixedTopologyRoutes(model)
    catalog = attack_catalog(model, config.attack_scenario, config=config)
    first_precursor_step = min(
        (item["temporal"]["start_step"] - item["temporal"]["precursor_steps"] for item in catalog),
        default=config.step_count + 1,
    )
    reference_snapshot = _snapshot(
        model,
        config,
        0,
        0,
        power_runtime_state=power_runtime_state,
        routing=routing,
    )
    reference_tensor_state = reference_snapshot.pop("_analysis_tensor_state")
    reference_tensor_matrices = build_tensor_matrix_view(reference_tensor_state)
    reference_snapshot["tensor_matrices"] = tensor_matrix_export(
        reference_tensor_matrices,
        include_values=config.snapshot_detail in {"full", "tensor"},
    )
    state_hausdorff_reference = prepare_state_hausdorff_reference(
        reference_tensor_state
    )
    reference_hausdorff = state_tensor_hausdorff_view(
        reference_tensor_state,
        reference_tensor_state,
        time_index=0,
        prepared_reference=state_hausdorff_reference,
    )
    reference_snapshot["arbitrator"] = build_arbitrator_view(
        model,
        reference_tensor_state,
        observation={
            "traffic": reference_snapshot.get("traffic", {}).get("summary", {}),
            "attacks": reference_snapshot.get("attacks", {}),
        },
        state_hausdorff=reference_hausdorff,
    )
    reference_snapshot["state_vector"] = reference_snapshot["arbitrator"]["state_vector"]
    koopman_analyzer = KoopmanOnlineAnalyzer.from_t0(
        model,
        reference_snapshot["state_vector"],
        tensor_matrices=reference_tensor_matrices,
        step_seconds=config.step_seconds,
        required_prediction_lead_seconds=_resolved_prediction_slo_seconds(config),
        warning_risk_threshold=_resolved_warning_risk_threshold(config),
        warning_threshold_origin=config.warning_threshold_origin,
        runtime_state=reference_snapshot.get("traffic", {}).get("network_state"),
        kalman_process_variance=config.kalman_process_variance,
        kalman_measurement_variance=config.kalman_measurement_variance,
    )
    # Фильтру и обучению нужен предшественник первого перехода даже тогда,
    # когда вызывающая сторона просит убрать t0 из списка снимков.
    for step_index in range(config.step_count + 1):
        snapshot = (
            reference_snapshot
            if step_index == 0
            else _snapshot(
                model,
                config,
                step_index,
                step_index * config.step_seconds,
                power_runtime_state=power_runtime_state,
                reference_tensor_state=reference_tensor_state,
                state_hausdorff_reference=state_hausdorff_reference,
                routing=routing,
            )
        )
        if step_index == 0:
            analysis_tensor_state = reference_tensor_state
            tensor_matrices = reference_tensor_matrices
        else:
            analysis_tensor_state = snapshot.pop("_analysis_tensor_state")
            tensor_matrices = build_tensor_matrix_view(analysis_tensor_state)
            snapshot["tensor_matrices"] = tensor_matrix_export(
                tensor_matrices,
                include_values=config.snapshot_detail in {"full", "tensor"},
            )
        koopman_view = koopman_analyzer.analyze_step(
            model,
            snapshot["state_vector"],
            tensor_matrices=tensor_matrices,
            attack_state=snapshot.get("attacks", {}),
            step_index=step_index,
            time_seconds=step_index * config.step_seconds,
            runtime_state=snapshot.get("traffic", {}).get("network_state"),
            routing=routing,
        )
        apply_koopman_to_arbitrator(
            snapshot,
            koopman_view,
            model=model,
        )
        traffic = snapshot.get("traffic", {})
        sla_started = perf_counter()
        snapshot["service_sla_forecast"] = sla_forecaster.observe(
            traffic.get("flows", ()), time_seconds=snapshot["time_seconds"],
            nominal=koopman_view["nominal_training"]["observation_eligible"],
        )
        view = snapshot["service_sla_forecast"]
        view["operator_warnings"] = operator_warnings(view)
        route_view = route_monitor.observe(traffic.get("flows", ()), snapshot.get("attacks", {}).get("precursors", ()),
                                           time_seconds=snapshot["time_seconds"])
        snapshot["service_sla_forecast"]["route_stress_warnings"] = route_view["warnings"]
        if step_index or config.include_t0:
            route_views.append(route_view)
        sla_compute_ms.append((perf_counter() - sla_started) * 1000)
        # Только разметка для ретроспективного графика, после выпуска прогнозов.
        impacts = {(f["traffic_kind"], f["client_node"], event)
                   for f in traffic.get("flows", ()) if f.get("attack_impacted")
                   for event in f.get("attack_ids", ())}
        for service, client, _ in impacts - previous_impacts:
            impacts_by_flow.setdefault((service, client), set()).add(snapshot["time_seconds"])
        previous_impacts = impacts
        if config.packet_detail == "summary":
            traffic.pop("flows", None)
        if step_index == 0:
            ideal_t0 = validate_ideal_t0(model, snapshot)
            if not ideal_t0["ok"]:
                raise ValueError(f"t0 is not ideal: {ideal_t0['violation_count']} violations")
            phase = "reference"
        elif step_index < first_precursor_step:
            phase = "nominal_training"
        else:
            phase = "observation_and_recommendations"
        snapshot["training"] = {
            "phase": phase,
            "clean_training_steps": min(config.step_count, max(0, first_precursor_step - 1)),
            "nominal_transitions": koopman_view["observed_nominal_transition_count_after_update"],
            "operator_updated": koopman_view["update_applied_for_next_step"],
        }
        if step_index or config.include_t0:
            if compact_traffic_history and traffic:
                # После прогноза и проверки t0 отчёту достаточно агрегатов и векторов.
                traffic.pop("flows", None)
                traffic.pop("packet_sample", None)
                traffic["network_state"] = compact_runtime_state(traffic["network_state"])
            snapshots.append(snapshot)

    service_views = [s["service_sla_forecast"] for s in snapshots]
    service_evaluation = evaluate_service_forecasts(service_views)
    if config.step_seconds == 2:
        issued = [{**v, "warnings": v["operator_warnings"]} for v in service_views]
        service_evaluation["operator_alert_evaluation"] = {
            str(h): _score_window(issued, "koopman", h)[0] for h in (2, 10, 20)}
    service_evaluation["route_stress"] = evaluate_route_stress(route_views, service_views)
    sla_forecaster.case_traces(service_evaluation)
    for case in service_evaluation["cases"].values():
        if case and case.get("measurements"):
            measured = case["measurements"]
            times = impacts_by_flow.get((case["service"], case["flow_id"]), ())
            case["flow_impact_starts_seconds"] = sorted(
                t for t in times if measured[0]["time_seconds"] <= t <= measured[-1]["time_seconds"])
    return {
        "mode": (
            "predictive_weighted_attack_demonstration"
            if config.attack_scenario == "predictive-demo"
            else "mitre_attack_demonstration"
            if config.attack_scenario == "mitre-demo"
            else "stationary_healthy_baseline"
        ),
        "config": config.to_dict(),
        "operation_mode": "forecast_and_human_recommendations_only",
        "attack_catalog": catalog,
        "training_summary": {
            "clean_training_steps": min(config.step_count, max(0, first_precursor_step - 1)),
            "first_precursor_step": first_precursor_step if catalog else None,
            "first_attack_step": min((item["temporal"]["start_step"] for item in catalog), default=None),
            "nominal_transitions": koopman_analyzer.nominal_observation_count,
        },
        "health": health,
        "ideal_t0": ideal_t0,
        "snapshot_count": len(snapshots),
        "service_sla_evaluation": service_evaluation,
        "service_sla_runtime": {
            "offline_compute_mean_ms": sum(sla_compute_ms) / len(sla_compute_ms),
            "offline_compute_max_ms": max(sla_compute_ms),
            "external_delivery_ms": None,
        },
        "koopman_evaluation": _koopman_evaluation(
            snapshots,
            step_seconds=config.step_seconds,
            required_lead_seconds=_resolved_prediction_slo_seconds(config),
        ),
        "resilience_evaluation": _resilience_evaluation(
            snapshots,
            step_seconds=config.step_seconds,
        ),
        "attack_response_evaluation": _attack_response_evaluation(snapshots),
        "hypernetwork": hypernetwork_embedding(model),
        "sla_resilience": _sla_resilience(snapshots, config.step_seconds),
        "tensor_matrix_contract": {
            "representation": "one_dense_entity_by_named_metric_matrix_per_level",
            "levels": list(TENSOR_LEVELS),
            "koopman_input": "factorised_kronecker_observables_derived_from_complete_matrices",
            "summary_export_semantics": (
                "summary stores matrix axes, hashes and norms; full and tensor store every value"
            ),
        },
        "snapshots": snapshots,
    }


def _attack_response_evaluation(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Свести наблюдения каждого события после расчёта, сохранив его тип."""
    records: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        analysis = snapshot["arbitrator"]["analysis"]
        recommendation = snapshot["arbitrator"]["recommendation"]
        for event in snapshot.get("attacks", {}).get("events", []):
            record = records.setdefault(event["attack_id"], {
                "attack_id": event["attack_id"], "kind": event["kind"], "target_id": event["target_id"],
                **{key: event[key] for key in ("attack_family", "attack_mechanism", "attack_label_ru") if key in event},
                "first_step": snapshot["step_index"], "last_step": snapshot["step_index"],
                "maximum_hausdorff": 0.0, "maximum_koopman_risk": 0.0,
                "maximum_lyapunov_pressure": 0.0,
                "response_stages": [], "actions": [],
                "scope": "whole_network_snapshots_during_event_not_isolated_causal_effect",
            })
            record["last_step"] = snapshot["step_index"]
            record["maximum_hausdorff"] = max(record["maximum_hausdorff"], float(analysis["state_hausdorff"].get("normalized_distance", 0.0)))
            record["maximum_koopman_risk"] = max(record["maximum_koopman_risk"], float(snapshot["koopman"]["forecast_risk_score"]))
            record["maximum_lyapunov_pressure"] = max(record["maximum_lyapunov_pressure"], float(snapshot["koopman"]["lyapunov_remap_pressure"]))
            for key, value in (("actions", recommendation["action"]), ("response_stages", recommendation["response_stage"])):
                if value not in record[key]:
                    record[key].append(value)
    return list(records.values())


def validate_ideal_t0(model: NetworkModel, snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """Проверить, что t0 структурно идеален и имеет реалистичный запас по SLA."""
    violations: list[dict[str, Any]] = []
    operational_nodes = [
        node_id for node_id, attrs in model.graph.nodes(data=True) if attrs.get("level") in {"L0", "L1", "L2"}
    ]
    operational_graph_connected = nx.is_connected(model.graph.subgraph(operational_nodes))
    metrics: dict[str, Any] = {
        "operational_graph_connected": operational_graph_connected,
        "subscriber_count": 0,
        "aggregation_switch_count": 0,
        "core_router_count": 0,
        "radio_access_node_count": 0,
        "optical_line_terminal_count": 0,
        "service_count": 0,
        "service_server_count": 0,
        "single_access_subscriber_count": 0,
        "service_standby_replica_count": 0,
        "l2_node_connectivity": 0,
        "maximum_l2_infrastructure_degree": 0,
        "maximum_l2_backbone_degree": 0,
        "minimum_access_uplink_count": 0,
        "maximum_l2_high_speed_ports_used": 0,
        "network_device_dual_feed_count": 0,
        "network_device_power_provenance_count": 0,
        "maximum_modeled_t0_network_device_power_w": 0.0,
        "service_server_l6_count": 0,
        "service_server_dual_feed_count": 0,
        "service_server_power_provenance_count": 0,
        "service_standby_cross_power_domain_count": 0,
        "local_power_fault_domain_count": 0,
        "maximum_observed_link_utilization_percent": 0.0,
        "minimum_bitrate_headroom_ratio": float("inf"),
        "maximum_latency_budget_ratio": 0.0,
        "maximum_jitter_budget_ratio": 0.0,
        "maximum_loss_budget_ratio": 0.0,
        "maximum_l1_queue_utilization": 0.0,
        "maximum_l1_queue_blocking_probability": 0.0,
        "maximum_l1_queue_delay_budget_ratio": 0.0,
        "maximum_l2_cpu_percent": 0.0,
        "maximum_l2_ram_percent": 0.0,
        "maximum_planned_link_utilization": 0.0,
        "minimum_link_stability_margin": 1.0,
        "koopman_t0_reference_distance": 0.0,
        "koopman_t0_one_step_residual": 0.0,
        "lyapunov_t0_value": 0.0,
    }
    local_power_fault_domains: dict[str, str] = {}
    if not operational_graph_connected:
        violations.append({"scope": "TOPOLOGY", "reason": "operational_graph_disconnected"})

    for node_id, attrs in model.graph.nodes(data=True):
        level, role = attrs.get("level"), attrs.get("role")
        if level == "L1":
            metrics["subscriber_count"] += 1
            neighbors = set(model.graph.neighbors(node_id))
            home_access = str(attrs.get("home_access"))
            expected_access = {home_access}
            if neighbors != expected_access:
                violations.append({"scope": "L1", "node": node_id, "reason": "invalid_single_access_attachment"})
            else:
                metrics["single_access_subscriber_count"] += 1
            primary_edge = model.graph.get_edge_data(node_id, home_access, default={})
            if (
                primary_edge.get("access_role") != "primary"
                or primary_edge.get("standby") is not False
                or primary_edge.get("consumes_c9500_physical_port") is not False
                or not str(primary_edge.get("attachment_semantics", "")).startswith("logical_")
                or float(attrs.get("access_capacity_mbps", 0.0)) <= 0.0
            ):
                violations.append(
                    {"scope": "L1", "node": node_id, "reason": "unrealistic_access_attachment_semantics"}
                )
            policy = attrs.get("d0sl_policy", {})
            min_bitrate = max(float(attrs.get("min_bitrate_kbps", 0.0)), 1e-9)
            latency_budget = max(float(policy.get("latency_budget_ms", 0.0)), 1e-9)
            jitter_budget = max(float(policy.get("jitter_budget_ms", 0.0)), 1e-9)
            loss_budget = max(float(policy.get("packet_loss_budget_percent", 0.0)), 1e-9)
            queue = attrs.get("kendall_queue", {})
            queue_utilization = float(queue.get("utilization_rho", 1.0))
            queue_blocking = float(queue.get("blocking_probability", 1.0))
            queue_delay_ratio = float(queue.get("mean_system_time_ms", latency_budget)) / latency_budget
            metrics["maximum_l1_queue_utilization"] = max(
                metrics["maximum_l1_queue_utilization"],
                queue_utilization,
            )
            metrics["maximum_l1_queue_blocking_probability"] = max(
                metrics["maximum_l1_queue_blocking_probability"],
                queue_blocking,
            )
            metrics["maximum_l1_queue_delay_budget_ratio"] = max(
                metrics["maximum_l1_queue_delay_budget_ratio"],
                queue_delay_ratio,
            )
            if (
                queue.get("kendall") != "M/M/1/128/∞/FIFO"
                or queue_utilization >= 1.0
                or queue_blocking > 1e-6
                or queue_delay_ratio > 0.25 + 1e-9
            ):
                violations.append(
                    {
                        "scope": "L1_QUEUE",
                        "node": node_id,
                        "reason": "queue_not_inside_ideal_reserve",
                    }
                )
            for point in attrs.get("monitoring", []):
                bitrate_ratio = float(point.get("bitrate_kbps", 0.0)) / min_bitrate
                latency_ratio = float(point.get("latency_ms", 0.0)) / latency_budget
                jitter_ratio = float(point.get("jitter_ms", 0.0)) / jitter_budget
                loss_ratio = float(point.get("packet_loss_percent", 0.0)) / loss_budget
                metrics["minimum_bitrate_headroom_ratio"] = min(metrics["minimum_bitrate_headroom_ratio"], bitrate_ratio)
                metrics["maximum_latency_budget_ratio"] = max(metrics["maximum_latency_budget_ratio"], latency_ratio)
                metrics["maximum_jitter_budget_ratio"] = max(metrics["maximum_jitter_budget_ratio"], jitter_ratio)
                metrics["maximum_loss_budget_ratio"] = max(metrics["maximum_loss_budget_ratio"], loss_ratio)
                if bitrate_ratio < 1.0 or latency_ratio > 0.80 or jitter_ratio > 0.80 or loss_ratio > 0.50:
                    violations.append({"scope": "L1", "node": node_id, "second": point.get("second"), "reason": "insufficient_sla_headroom"})

        elif role == "aggregation-switch":
            metrics["aggregation_switch_count"] += 1
            neighbors = list(model.graph.neighbors(node_id))
            access_nodes = [
                node
                for node in neighbors
                if model.graph.nodes[node].get("role") in ACCESS_DEVICE_ROLES
                and model.graph.nodes[node].get("home_aggregation") == node_id
            ]
            downstream_clients = [
                node
                for node, client_attrs in model.graph.nodes(data=True)
                if client_attrs.get("level") == "L1"
                and client_attrs.get("home_access") in access_nodes
            ]
            core_links = [node for node in neighbors if model.graph.nodes[node].get("role") == "core-router"]
            expected_clients = (
                MOBILE_SUBSCRIBERS_PER_AGG
                if node_id in {"A1", "A3", "A5"}
                else FIXED_SUBSCRIBERS_PER_AGG
            )
            if len(access_nodes) < 2 or len(downstream_clients) != expected_clients or len(core_links) < 3:
                violations.append({"scope": "L2", "node": node_id, "reason": "aggregation_redundancy_or_client_count"})
        elif role == "core-router":
            metrics["core_router_count"] += 1
            core_neighbors = [node for node in model.graph.neighbors(node_id) if model.graph.nodes[node].get("role") == "core-router"]
            if len(core_neighbors) < 2:
                violations.append({"scope": "L2", "node": node_id, "reason": "core_ring_not_redundant"})
        elif role in ACCESS_DEVICE_ROLES:
            if role == "radio-access-node":
                metrics["radio_access_node_count"] += 1
            elif role == "optical-line-terminal":
                metrics["optical_line_terminal_count"] += 1
            aggregation_links = [
                node
                for node in model.graph.neighbors(node_id)
                if model.graph.nodes[node].get("role") == "aggregation-switch"
            ]
            clients = [
                node
                for node in model.graph.neighbors(node_id)
                if model.graph.nodes[node].get("level") == "L1"
                and model.graph.nodes[node].get("home_access") == node_id
            ]
            if len(aggregation_links) < 2 or not clients:
                violations.append({"scope": "L2_ACCESS", "node": node_id, "reason": "insufficient_access_uplinks_or_clients"})
        elif role == "service":
            metrics["service_count"] += 1
            if model.graph.degree(node_id) != 1:
                violations.append({"scope": "L0", "node": node_id, "reason": "invalid_service_attachment"})
            host = attrs.get("hosted_on")
            if host not in model.graph or model.graph.nodes[host].get("role") != "service-server":
                violations.append({"scope": "L0", "node": node_id, "reason": "invalid_service_host"})
            standby_hosts = list(attrs.get("standby_hosts", []))
            valid_standby = [
                standby for standby in standby_hosts
                if standby in model.graph
                and standby != host
                and model.graph.nodes[standby].get("role") == "service-server"
            ]
            metrics["service_standby_replica_count"] += len(valid_standby)
            if not valid_standby:
                violations.append({"scope": "L0", "node": node_id, "reason": "missing_service_standby"})
            elif host in model.graph:
                host_domain = str(
                    model.graph.nodes[host].get("power_architecture", {}).get("local_fault_domain_id", "")
                )
                for standby in valid_standby:
                    standby_domain = str(
                        model.graph.nodes[standby].get("power_architecture", {}).get("local_fault_domain_id", "")
                    )
                    if host_domain and standby_domain and host_domain != standby_domain:
                        metrics["service_standby_cross_power_domain_count"] += 1
                    else:
                        violations.append(
                            {
                                "scope": "L0_POWER",
                                "node": node_id,
                                "reason": "standby_not_in_independent_local_power_domain",
                            }
                        )
        elif role == "service-server":
            metrics["service_server_count"] += 1
            runtime = attrs.get("runtime", {})
            power = attrs.get("power_architecture", {})
            l6_tensor = attrs.get("l6_tensor")
            l6_values = tensor_metrics(l6_tensor) if isinstance(l6_tensor, StateTensor) else {}
            modeled_power_w = float(power.get("modeled_t0_power_w", 0.0))
            tensor_power_w = float(l6_values.get("nominal_power_kw", 0.0)) * 1_000.0
            psu_rating_w = float(power.get("psu_rating_w_each", 0.0))
            local_domain = str(power.get("local_fault_domain_id", ""))
            ups_domains = [str(value) for value in power.get("ups_domains", [])]
            if isinstance(l6_tensor, StateTensor):
                metrics["service_server_l6_count"] += 1
            if int(power.get("feed_count", 0)) >= 2:
                metrics["service_server_dual_feed_count"] += 1
            if power.get("modeled_t0_power_origin") and power.get("psu_rating_semantics"):
                metrics["service_server_power_provenance_count"] += 1
            core_links = [neighbor for neighbor in model.graph.neighbors(node_id) if model.graph.nodes[neighbor].get("role") == "core-router"]
            incompatible_links = [
                neighbor
                for neighbor in core_links
                if float(model.graph.edges[node_id, neighbor].get("capacity_mbps", 0.0)) != 10_000.0
            ]
            if (
                len(core_links) < 2
                or incompatible_links
                or int(attrs.get("power_architecture", {}).get("feed_count", 0)) < 2
                or runtime.get("cpu_util_percent", 100.0) > 50.0
                or runtime.get("ram_util_percent", 100.0) > 60.0
                or runtime.get("storage_util_percent", 100.0) > 70.0
                or runtime.get("temperature_c", 100.0) > 60.0
                or not isinstance(l6_tensor, StateTensor)
                or modeled_power_w <= 0.0
                or abs(modeled_power_w - tensor_power_w) > 1e-6
                or int(power.get("psu_count", 0)) != 2
                or psu_rating_w != 800.0
                or power.get("psu_efficiency_class") != "Platinum"
                or power.get("psu_hot_swappable") is not True
                or modeled_power_w >= psu_rating_w
                or "not_measured_server_draw" not in str(power.get("psu_rating_semantics", ""))
                or not str(power.get("psu_source_url", "")).startswith("https://www.dell.com/")
                or len(set(ups_domains)) != 2
                or not local_domain
                or not all(value.startswith(f"{local_domain}/") for value in ups_domains)
                or not power.get("backup_autonomy_origin")
            ):
                violations.append({"scope": "L0_SERVER", "node": node_id, "reason": "insufficient_server_redundancy_or_headroom"})

        if role in {"core-router", "aggregation-switch", "radio-access-node", "optical-line-terminal", "service-server"}:
            local_domain = str(attrs.get("power_architecture", {}).get("local_fault_domain_id", ""))
            if not local_domain:
                violations.append({"scope": "L6", "node": node_id, "reason": "missing_local_power_fault_domain"})
            elif local_domain in local_power_fault_domains:
                violations.append(
                    {
                        "scope": "L6",
                        "node": node_id,
                        "reason": "local_power_fault_domain_reused",
                        "other_node": local_power_fault_domains[local_domain],
                    }
                )
            else:
                local_power_fault_domains[local_domain] = node_id

        if level == "L2" and isinstance(attrs.get("tensor"), StateTensor):
            values = tensor_metrics(attrs["tensor"])
            metrics["maximum_l2_cpu_percent"] = max(metrics["maximum_l2_cpu_percent"], values.get("cpu_load_percent", 0.0))
            metrics["maximum_l2_ram_percent"] = max(metrics["maximum_l2_ram_percent"], values.get("ram_load_percent", 0.0))
            metrics["maximum_l2_high_speed_ports_used"] = max(
                metrics["maximum_l2_high_speed_ports_used"],
                int(attrs.get("physical_high_speed_ports_used", 0)),
            )
            power = attrs.get("power_architecture", {})
            modeled_power_w = float(power.get("modeled_t0_power_w", 0.0))
            l6_tensor = attrs.get("l6_tensor")
            l6_values = tensor_metrics(l6_tensor) if isinstance(l6_tensor, StateTensor) else {}
            tensor_power_w = float(l6_values.get("nominal_power_kw", 0.0)) * 1_000.0
            metrics["maximum_modeled_t0_network_device_power_w"] = max(
                metrics["maximum_modeled_t0_network_device_power_w"],
                modeled_power_w,
            )
            if int(power.get("feed_count", 0)) >= 2:
                metrics["network_device_dual_feed_count"] += 1
            if power.get("modeled_t0_power_origin") and power.get("vendor_power_value_semantics"):
                metrics["network_device_power_provenance_count"] += 1

            power_invalid = (
                int(power.get("feed_count", 0)) < 2
                or modeled_power_w <= 0.0
                or abs(modeled_power_w - tensor_power_w) > 1e-6
                or not power.get("backup_autonomy_origin")
                or len(set(power.get("ups_domains", []))) != 2
                or not power.get("local_fault_domain_id")
            )
            if role == "core-router":
                power_invalid = power_invalid or abs(
                    modeled_power_w - float(power.get("vendor_typical_output_power_w", 0.0))
                ) > 1e-6
            elif role == "aggregation-switch":
                thermal_bound = float(power.get("vendor_thermal_output_equivalent_w", 0.0))
                power_invalid = power_invalid or thermal_bound <= 0.0 or modeled_power_w >= thermal_bound
            if power_invalid:
                violations.append(
                    {"scope": "L6", "node": node_id, "reason": "invalid_power_provenance_or_t0_value"}
                )

    metrics["local_power_fault_domain_count"] = len(local_power_fault_domains)

    for source, target, attrs in model.graph.edges(data=True):
        tensor = attrs.get("tensor")
        if not isinstance(tensor, StateTensor):
            continue
        values = tensor_metrics(tensor)
        utilization = values.get("utilization", 0.0)
        stability = values.get("stability_margin", 0.0)
        metrics["maximum_planned_link_utilization"] = max(metrics["maximum_planned_link_utilization"], utilization)
        metrics["minimum_link_stability_margin"] = min(metrics["minimum_link_stability_margin"], stability)
        if utilization > 0.25 or stability < 0.70:
            violations.append({"scope": "EDGE", "edge": [source, target], "reason": "insufficient_capacity_reserve"})

    if metrics["maximum_l2_cpu_percent"] > 50.0 or metrics["maximum_l2_ram_percent"] > 60.0:
        violations.append({"scope": "L2", "reason": "insufficient_equipment_headroom"})

    l2_nodes = [
        node for node, attrs in model.graph.nodes(data=True) if attrs.get("level") == "L2"
    ]
    l2_graph = model.graph.subgraph(l2_nodes)
    backbone_nodes = [
        node
        for node in l2_nodes
        if model.graph.nodes[node].get("role") in {"core-router", "aggregation-switch"}
    ]
    backbone_graph = model.graph.subgraph(backbone_nodes)
    metrics["l2_node_connectivity"] = nx.node_connectivity(backbone_graph)
    metrics["maximum_l2_infrastructure_degree"] = max(dict(l2_graph.degree()).values(), default=0)
    metrics["maximum_l2_backbone_degree"] = max(dict(backbone_graph.degree()).values(), default=0)
    access_uplink_counts = [
        sum(
            1
            for neighbor in model.graph.neighbors(node)
            if model.graph.nodes[neighbor].get("role") == "aggregation-switch"
        )
        for node in l2_nodes
        if model.graph.nodes[node].get("role") in ACCESS_DEVICE_ROLES
    ]
    metrics["minimum_access_uplink_count"] = min(access_uplink_counts, default=0)
    if (
        metrics["l2_node_connectivity"] < 3
        or metrics["maximum_l2_backbone_degree"] > 6
        or metrics["minimum_access_uplink_count"] < 2
    ):
        violations.append({"scope": "L2", "reason": "insufficient_transport_connectivity"})

    if snapshot is not None:
        recommendation = snapshot.get("arbitrator", {}).get("recommendation", {})
        analysis = snapshot.get("arbitrator", {}).get("analysis", {})
        if recommendation.get("action") != "OBSERVE" or analysis.get("remap_pressure", 1.0) > 0.05 or analysis.get("decision_confidence", 0.0) < 0.80:
            violations.append({"scope": "L7", "reason": "baseline_requires_security_review"})
        metrics["l7_decision"] = recommendation.get("action")
        koopman = snapshot.get("koopman", {})
        metrics["koopman_t0_reference_distance"] = float(koopman.get("reference_distance", 1.0))
        metrics["koopman_t0_one_step_residual"] = float(koopman.get("one_step_residual", 1.0))
        metrics["lyapunov_t0_value"] = float(koopman.get("lyapunov_value", 1.0))
        if (
            metrics["koopman_t0_reference_distance"] > 1e-9
            or metrics["koopman_t0_one_step_residual"] > 1e-9
            or metrics["lyapunov_t0_value"] > 1e-9
            or bool(koopman.get("forecast_is_early_warning", False))
        ):
            violations.append({"scope": "L7", "reason": "non_ideal_koopman_t0"})
        if "traffic" in snapshot:
            traffic = snapshot["traffic"].get("summary", {})
            if traffic.get("observed_dropped_packets", 0) or traffic.get("observed_retransmissions", 0) or traffic.get("observed_loss_ratio", 0.0):
                violations.append({"scope": "TRAFFIC", "reason": "observed_loss_or_retransmission"})
            metrics["observed_loss_ratio"] = float(traffic.get("observed_loss_ratio", 0.0))
            metrics["observed_retransmissions"] = int(traffic.get("observed_retransmissions", 0))
            metrics["maximum_observed_link_utilization_percent"] = float(
                traffic.get("maximum_observed_link_utilization_percent", 0.0)
            )
            if metrics["maximum_observed_link_utilization_percent"] > 25.0:
                violations.append({"scope": "TRAFFIC", "reason": "insufficient_observed_link_headroom"})

    if metrics["minimum_bitrate_headroom_ratio"] == float("inf"):
        metrics["minimum_bitrate_headroom_ratio"] = 0.0
    return {
        "ok": not violations,
        "status": "IDEAL_REALISTIC_BASELINE" if not violations else "INVALID_BASELINE",
        "violation_count": len(violations),
        "metrics": metrics,
        "violations": violations[:100],
    }


def validate_healthy_baseline(model: NetworkModel) -> dict[str, Any]:
    """Check that t0 is inside the intended healthy SLA/SLO envelope."""
    violations: list[dict[str, Any]] = []
    checked_l1_points = 0
    checked_l2_nodes = 0
    checked_edges = 0

    for node_id, attrs in model.graph.nodes(data=True):
        level = attrs.get("level")
        if level == "L1":
            for point in attrs.get("monitoring", []):
                checked_l1_points += 1
                failed_flags = [
                    flag
                    for flag in ("bitrate_slo_ok", "latency_slo_ok", "loss_slo_ok", "jitter_slo_ok")
                    if not point.get(flag)
                ]
                if failed_flags or point.get("bitrate_drop_alarm"):
                    violations.append(
                        {
                            "scope": "L1",
                            "node": node_id,
                            "second": point.get("second"),
                            "failed_flags": failed_flags,
                            "bitrate_drop_alarm": bool(point.get("bitrate_drop_alarm")),
                        }
                    )

        if level == "L2" and isinstance(attrs.get("tensor"), StateTensor):
            checked_l2_nodes += 1
            metrics = tensor_metrics(attrs["tensor"])
            if metrics.get("ram_load_percent", 0.0) > 80.0 or metrics.get("cpu_load_percent", 0.0) > 80.0:
                violations.append(
                    {
                        "scope": "L2",
                        "node": node_id,
                        "ram_load_percent": metrics.get("ram_load_percent"),
                        "cpu_load_percent": metrics.get("cpu_load_percent"),
                    }
                )

    for source, target, attrs in model.graph.edges(data=True):
        tensor = attrs.get("tensor")
        if not isinstance(tensor, StateTensor):
            continue

        checked_edges += 1
        metrics = tensor_metrics(tensor)
        if metrics.get("loss_probability", 0.0) > 0.005 or metrics.get("stability_margin", 0.0) <= 0.0:
            violations.append(
                {
                    "scope": "EDGE",
                    "edge": [source, target],
                    "loss_probability": metrics.get("loss_probability"),
                    "stability_margin": metrics.get("stability_margin"),
                }
            )

    return {
        "ok": not violations,
        "checked_l1_points": checked_l1_points,
        "checked_l2_nodes": checked_l2_nodes,
        "checked_edges": checked_edges,
        "violation_count": len(violations),
        "violations": violations[:50],
    }


def _snapshot(
    model: NetworkModel,
    config: DynamicsConfig,
    step_index: int,
    time_seconds: int,
    *,
    power_runtime_state: dict[str, dict[str, Any]] | None = None,
    reference_tensor_state: dict[str, Any] | None = None,
    state_hausdorff_reference: StateTensorHausdorffReference | None = None,
    routing: FixedTopologyRoutes | None = None,
) -> dict[str, Any]:
    scheduled_attack_events = active_attack_events(
        model,
        config.attack_scenario,
        step_index=step_index,
        time_seconds=time_seconds,
        step_seconds=config.step_seconds,
        step_count=config.step_count,
        seed=config.attack_seed,
        config=config,
    )
    attack_events, power_runtime = advance_power_runtime_state(
        model,
        scheduled_attack_events,
        power_runtime_state,
        step_index=step_index,
        time_seconds=time_seconds,
        step_seconds=config.step_seconds,
    )
    attack_precursors = observe_attack_precursors(
        model,
        config.attack_scenario,
        step_index=step_index,
        time_seconds=time_seconds,
        step_seconds=config.step_seconds,
        step_count=config.step_count,
        seed=config.attack_seed,
        config=config,
    )
    traffic = None
    if config.include_packet_simulation:
        traffic = simulate_packet_snapshot(
            model,
            step_index=step_index,
            time_seconds=time_seconds,
            step_seconds=config.step_seconds,
            detail=config.packet_detail,
            retain_flows=True,
            packet_sample_limit=config.packet_sample_limit,
            attack_events=attack_events,
            attack_precursors=attack_precursors,
            attack_scenario=config.attack_scenario,
            power_runtime_state=power_runtime,
            routing=routing,
        )

    tensor_state = _tensor_state_snapshot(model)
    if traffic is not None:
        tensor_state = _apply_observed_tensor_overlay(model, tensor_state, traffic)
    observation = None if traffic is None else {
        "traffic": traffic.get("summary", {}),
        "attacks": traffic.get("attack_state", {}),
    }
    state_hausdorff = (
        state_tensor_hausdorff_view(
            reference_tensor_state,
            tensor_state,
            time_index=step_index,
            prepared_reference=state_hausdorff_reference,
        )
        if reference_tensor_state is not None
        else None
    )
    arbitrator = build_arbitrator_view(
        model,
        tensor_state,
        observation=observation,
        state_hausdorff=state_hausdorff,
    )
    snapshot = {
        "step_index": step_index,
        "time_seconds": time_seconds,
        "level_summary": model.level_summary,
        "tensor_state": _format_tensor_state(tensor_state, config.snapshot_detail),
        "state_vector": arbitrator["state_vector"],
        "arbitrator": arbitrator,
        "attacks": (
            traffic.get("attack_state", {}) if traffic is not None
            else {
                "scenario": config.attack_scenario,
                "active": bool(attack_events),
                "events": attack_events,
                "precursor_active": bool(attack_precursors),
                "precursors": attack_precursors,
                "power_runtime_state": power_runtime,
            }
        ),
        "_analysis_tensor_state": tensor_state,
    }

    if config.snapshot_detail == "full":
        snapshot["nodes"] = [_node_snapshot(node_id, attrs) for node_id, attrs in model.graph.nodes(data=True)]
        snapshot["edges"] = [_edge_snapshot(source, target, attrs) for source, target, attrs in model.graph.edges(data=True)]

    if traffic is not None:
        traffic.pop("attack_state", None)
        snapshot["traffic"] = traffic

    return snapshot


def _tensor_state_snapshot(model: NetworkModel) -> dict[str, Any]:
    """Collect every StateTensor into a per-level structure for analysis."""
    by_level: dict[str, list[dict[str, Any]]] = {level: [] for level in TENSOR_LEVELS}

    for node_id, attrs in model.graph.nodes(data=True):
        for tensor_name, tensor in _raw_tensor_attrs(attrs).items():
            by_level[tensor.level].append(
                {
                    "scope": "node",
                    "node_id": node_id,
                    "role": attrs.get("role"),
                    "sla_grade": attrs.get("sla_grade"),
                    "traffic_kind": attrs.get("traffic_kind"),
                    "tensor_name": tensor_name,
                    **_tensor_snapshot(tensor),
                }
            )

    for source, target, attrs in model.graph.edges(data=True):
        for tensor_name, tensor in _raw_tensor_attrs(attrs).items():
            by_level[tensor.level].append(
                {
                    "scope": "edge",
                    "source": source,
                    "target": target,
                    "medium": attrs.get("medium"),
                    "tensor_name": tensor_name,
                    **_tensor_snapshot(tensor),
                }
            )

    counts = {level: len(items) for level, items in by_level.items()}
    return {
        "levels": list(TENSOR_LEVELS),
        "counts": counts,
        "by_level": by_level,
        "overlay": {
            "mode": "baseline_t0_without_observed_overlay",
            "updated_tensor_count": 0,
            "updated_metric_count": 0,
        },
    }


def _apply_observed_tensor_overlay(
    model: NetworkModel,
    tensor_state: dict[str, Any],
    traffic: dict[str, Any],
) -> dict[str, Any]:
    """Наложить телеметрию шага на копию тензоров, не изменяя эталонный граф.

    L3/L4/L5/L8 остаются структурными, пока сценарий не моделирует изменение
    среды, кабельной инфраструктуры, протокольной конфигурации или географии.
    Loss на ребре — экспозиция к end-to-end потерям проходящих потоков, а не
    заявление о физическом порте, на котором пакет был отброшен.
    """
    summary = traffic.get("summary", {})
    attacks = traffic.get("attack_state", {})
    updated_tensors: set[str] = set()
    updated_metric_count = 0

    def set_metric(item: dict[str, Any], name: str, value: float, origin: str) -> None:
        nonlocal updated_metric_count
        metrics = item.get("metrics", {})
        if name not in metrics:
            return
        value = float(value)
        metric_names = list(metrics)
        metrics[name] = value
        item["vector"][metric_names.index(name)] = value
        item.setdefault("observed_metric_origins", {})[name] = origin
        identity = item.get("node_id") or f"{item.get('source')}--{item.get('target')}"
        updated_tensors.add(f"{item.get('level')}:{identity}:{item.get('tensor_name')}")
        updated_metric_count += 1

    # L0: фактически доставленная доля приложения снижает здоровье сервиса.
    service_by_application = {
        str(app["application"]): str(app["service_node"])
        for app in TRAFFIC_APPS.values()
    }
    service_health: dict[str, float] = {}
    for application, app_summary in summary.get("applications", {}).items():
        service_node = service_by_application.get(str(application))
        if not service_node:
            continue
        offered = float(app_summary.get("offered_rate_mbps", 0.0))
        delivered = float(app_summary.get("delivered_rate_mbps", 0.0))
        service_health[service_node] = 1.0 if offered <= 1e-12 else max(
            0.0, min(1.0, delivered / offered)
        )
    for item in tensor_state["by_level"].get("L0", []):
        node_id = str(item.get("node_id", ""))
        if node_id in service_health:
            baseline_health = float(item["metrics"].get("service_health", 1.0))
            set_metric(
                item,
                "service_health",
                min(baseline_health, service_health[node_id]),
                "delivered_application_rate_divided_by_offered_rate",
            )

    # L1: итоговая service-aware совместимость класса уменьшает SLA margin.
    tier_ratios: dict[str, float] = {}
    for tier in attacks.get("sla_restoration", {}).get("tiers", []):
        grade = str(tier.get("sla_grade", ""))
        count = float(tier.get("flow_count", 0.0))
        tier_ratios[grade] = (
            1.0 if count <= 0.0
            else float(tier.get("sla_compliant_flow_count", 0.0)) / count
        )
    target_observations = {
        str(item.get("target_id")): item
        for item in attacks.get("target_observations", [])
        if item.get("target_id")
    }
    for item in tensor_state["by_level"].get("L1", []):
        node_id = str(item.get("node_id", ""))
        grade = str(item.get("sla_grade", ""))
        if grade in tier_ratios:
            baseline_margin = float(item["metrics"].get("sla_margin", 0.0))
            set_metric(
                item,
                "sla_margin",
                baseline_margin * max(0.0, min(1.0, tier_ratios[grade])),
                "service_aware_sla_compliance_ratio_by_grade",
            )
        authentication = target_observations.get(node_id)
        if authentication and float(authentication.get("authentication_failure_ratio", 0.0)) > 0.0:
            set_metric(
                item,
                "authentication_failure_rate_per_second",
                float(authentication.get("authentication_attempt_rate_per_second", 0.0))
                * float(authentication.get("authentication_failure_ratio", 0.0)),
                "t1110_failed_authentication_telemetry",
            )

    # L2: target observation меняет CPU и запас устойчивости только у цели.
    for item in tensor_state["by_level"].get("L2", []):
        observation = target_observations.get(str(item.get("node_id", "")))
        if not observation:
            continue
        utilization = max(
            float(observation.get("estimated_resource_utilization_percent", 0.0)),
            float(observation.get("estimated_network_utilization_percent", 0.0)),
        )
        set_metric(
            item,
            "cpu_load_percent",
            float(observation.get("estimated_resource_utilization_percent", 0.0)),
            "attack_target_resource_observation",
        )
        set_metric(
            item,
            "stability_margin",
            min(float(item["metrics"].get("stability_margin", 1.0)), max(0.0, 1.0 - utilization / 100.0)),
            "one_minus_maximum_observed_target_utilization",
        )

    # L6: только узел, участвующий в power-runtime, получает новый energy reserve.
    power_observations = {
        str(item.get("target_id")): item
        for item in attacks.get("power_runtime_state", [])
        if item.get("target_id")
    }
    for item in tensor_state["by_level"].get("L6", []):
        observation = power_observations.get(str(item.get("node_id", "")))
        if observation:
            set_metric(
                item,
                "energy_reserve_ratio",
                float(observation.get("energy_reserve_ratio", 1.0)),
                "ups_runtime_observation",
            )

    # EDGE: наблюдаемая полоса и flow-weighted loss exposure каждого ребра.
    links = {
        tuple(sorted((str(row.get("source")), str(row.get("target"))))): row
        for row in summary.get("observed_links", [])
    }
    for item in tensor_state["by_level"].get("EDGE", []):
        key = tuple(sorted((str(item.get("source")), str(item.get("target")))))
        row = links.get(key)
        if not row:
            continue
        utilization = max(0.0, float(row.get("utilization_percent", 0.0)) / 100.0)
        loss_exposure = max(
            0.0,
            float(row.get("end_to_end_flow_loss_exposure_ratio", 0.0)),
        )
        set_metric(item, "utilization", utilization, LINK_LOAD_RATIO_ORIGIN)
        set_metric(
            item,
            "stability_margin",
            max(0.0, 1.0 - utilization),
            "one_minus_observed_link_utilization",
        )
        set_metric(
            item,
            "loss_probability",
            max(float(item["metrics"].get("loss_probability", 0.0)), loss_exposure),
            "maximum_of_physical_baseline_and_end_to_end_flow_loss_exposure",
        )

    tensor_state["overlay"] = {
        "mode": "non_mutating_observed_telemetry_overlay",
        "updated_tensor_count": len(updated_tensors),
        "updated_metric_count": updated_metric_count,
        "updated_levels": sorted({item.split(":", 1)[0] for item in updated_tensors}),
        "structurally_invariant_levels": ["L3", "L4", "L5", "L8"],
        "source_semantics": {
            "EDGE.loss_probability": summary.get("observed_link_loss_semantics"),
            "L2": "target_resource_observation_only",
            "L6": "power_runtime_targets_only",
        },
    }
    return tensor_state


def _format_tensor_state(tensor_state: dict[str, Any], detail: SnapshotDetail) -> dict[str, Any]:
    """Trim tensor output for long dynamics runs when full detail is not needed."""
    if detail in {"full", "tensor"}:
        return tensor_state
    return {
        "levels": tensor_state["levels"],
        "counts": tensor_state["counts"],
        "overlay": tensor_state.get("overlay", {}),
    }


def _node_snapshot(node_id: str, attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": node_id,
        "level": attrs.get("level"),
        "role": attrs.get("role"),
        "pos": _json_value(attrs.get("pos")),
        "tensors": _tensor_attrs(attrs),
    }


def _edge_snapshot(source: str, target: str, attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "medium": attrs.get("medium"),
        "logical_level": attrs.get("logical_level"),
        "physical_level": attrs.get("physical_level"),
        "capacity_mbps": attrs.get("capacity_mbps"),
        "latency_ms": attrs.get("latency_ms"),
        "redundancy": attrs.get("redundancy"),
        "tensors": _tensor_attrs(attrs),
    }


def _tensor_attrs(attrs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: _tensor_snapshot(value)
        for name, value in _raw_tensor_attrs(attrs).items()
    }


def _raw_tensor_attrs(attrs: dict[str, Any]) -> dict[str, StateTensor]:
    return {
        name: value
        for name, value in attrs.items()
        if isinstance(value, StateTensor)
    }


def _tensor_snapshot(tensor: StateTensor) -> dict[str, Any]:
    values = tensor_metrics(tensor)
    return {
        "level": tensor.level,
        "metrics": values,
        "units": tensor.units,
        "vector": [float(value) for value in tensor.data.tolist()],
    }


@dataclass
class _WarningEpisode:
    target_id: str
    kind: str
    # Шаг выдачи, время выдачи (с), начало и конец окна в шагах.
    samples: list[tuple[int, int, int, int]]
    segment_end_step: int


def _warning_episodes_and_exposure(snapshots, step_seconds):
    """Соседние отсчёты одной цели и типа составляют один эпизод.

    Статус левого отсчёта действует до следующего отсчёта внутри непрерывного
    участка. У последнего отсчёта нет длительности. При шаге 2 с одна тревога
    перед следующим отсчётом даёт 2 с под тревогой.
    """
    episodes: list[_WarningEpisode] = []
    observed_seconds = warning_seconds = 0
    start = 0
    while start < len(snapshots):
        end = start
        while end + 1 < len(snapshots):
            left, right = snapshots[end:end + 2]
            dt = right["time_seconds"] - left["time_seconds"]
            if right["step_index"] != left["step_index"] + 1 or not 0 < dt <= step_seconds:
                break
            end += 1
        active = {}
        for index in range(start, end + 1):
            snapshot = snapshots[index]
            step = snapshot["step_index"]
            time = snapshot["time_seconds"]
            koopman = snapshot.get("koopman", {})
            warning = koopman.get("forecast_is_early_warning", False)
            if index < end:
                dt = snapshots[index + 1]["time_seconds"] - time
                observed_seconds += dt
                if warning:
                    warning_seconds += dt
            pairs = _forecast_entity_pairs(koopman) if warning else set()
            current = {}
            for target, kind in sorted(pairs):
                episode = active.get((target, kind))
                if episode is None:
                    episode = _WarningEpisode(target, kind, [], snapshots[end]["step_index"])
                    episodes.append(episode)
                episode.samples.append((step, time, *_forecast_step_window(koopman)))
                current[target, kind] = episode
            active = current
        start = end + 1
    span = snapshots[-1]["time_seconds"] - snapshots[0]["time_seconds"] if snapshots else 0
    return episodes, observed_seconds, warning_seconds, span - observed_seconds


def _match_warning_episodes(episodes, onsets, step_seconds):
    """Максимальное число пар. При равенстве предпочитаем ранние начала и сигналы."""
    graph = nx.Graph()
    first_signals = {}
    latest_step = max((ep.samples[-1][0] for ep in episodes), default=0)
    onset_priority = (latest_step + 1) * (len(onsets) + 1)
    for onset_index, (snapshot, event) in enumerate(onsets):
        step, time = snapshot["step_index"], snapshot["time_seconds"]
        for episode_index, episode in enumerate(episodes):
            if (step > episode.segment_end_step or episode.target_id != event.get("target_id")
                    or (episode.kind and episode.kind != event.get("kind"))):
                continue
            for issue_step, issue_time, window_start, window_end in episode.samples:
                if time - issue_time >= step_seconds and window_start <= step <= window_end:
                    first_signals[onset_index, episode_index] = (issue_step, issue_time)
                    graph.add_edge(("onset", onset_index), ("warning", episode_index),
                                   weight=-(onset_index * onset_priority + issue_step))
                    break
    matched = {}
    for left, right in nx.max_weight_matching(graph, maxcardinality=True):
        onset, episode = (left[1], right[1]) if left[0] == "onset" else (right[1], left[1])
        matched[onset] = (episode, *first_signals[onset, episode])
    return matched


def _koopman_evaluation(
    snapshots: list[dict[str, Any]],
    *,
    step_seconds: int,
    required_lead_seconds: int,
) -> dict[str, Any]:
    """Оценить причинный прогноз: truth используется только после симуляции.

    Предиктору не передаётся эта функция или таблица onset. Здесь уже после
    прогона проверяется, попало ли реальное начало воздействия в выданное им
    окно и совпали ли наблюдаемая цель и класс телеметрии.
    """
    onset_events = [(snap, event) for snap in snapshots
                    for event in snap.get("attacks", {}).get("events", [])
                    if int(event.get("relative_step", -1)) == 0]
    episodes, observed_seconds, warning_seconds, gap_seconds = _warning_episodes_and_exposure(
        snapshots, step_seconds)
    matched = _match_warning_episodes(episodes, onset_events, step_seconds)
    records = []
    for index, (snapshot, event) in enumerate(onset_events):
        match = matched.get(index)
        onset_time = snapshot["time_seconds"]
        records.append({
            "attack_id": str(event["attack_id"]), "kind": event.get("kind"),
            "target_id": event.get("target_id"), "onset_step": snapshot["step_index"],
            "onset_time_seconds": onset_time, "warning_episode_index": match[0] if match else None,
            "warning_step": match[1] if match else None, "warning_time_seconds": match[2] if match else None,
            "lead_time_seconds": onset_time - match[2] if match else 0,
            "predicted_before_onset": match is not None,
        })
    false_warnings = []
    censored_warnings = []
    used = {match[0] for match in matched.values()}
    for index, episode in enumerate(episodes):
        if index in used:
            continue
        end = max(sample[3] for sample in episode.samples)
        destination = censored_warnings if end > episode.segment_end_step else false_warnings
        destination.append({
            "warning_episode_index": index, "warning_step": episode.samples[0][0],
            "last_warning_step": episode.samples[-1][0],
            "forecast_window_start_step": min(sample[2] for sample in episode.samples),
            "forecast_window_end_step": end, "target_id": episode.target_id,
            "predicted_kind": episode.kind or None,
            "counting_semantics": "deduplicated_contiguous_warning_episode",
        })
    predicted_count = sum(1 for item in records if item["predicted_before_onset"])
    leads = [int(item["lead_time_seconds"]) for item in records if item["predicted_before_onset"]]
    slo_predicted_count = sum(
        bool(item["predicted_before_onset"])
        and int(item["lead_time_seconds"]) >= required_lead_seconds
        for item in records
    )
    slo_misses = [
        {
            "attack_id": item["attack_id"],
            "kind": item["kind"],
            "target_id": item["target_id"],
            "lead_time_seconds": item["lead_time_seconds"],
        }
        for item in records
        if not (
            bool(item["predicted_before_onset"])
            and int(item["lead_time_seconds"]) >= required_lead_seconds
        )
    ]
    recall = predicted_count / max(len(records), 1)
    precision = predicted_count / max(predicted_count + len(false_warnings), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    ordered_leads = sorted(leads)
    median_lead = (
        0.0
        if not ordered_leads
        else float(ordered_leads[len(ordered_leads) // 2])
        if len(ordered_leads) % 2
        else (ordered_leads[len(ordered_leads) // 2 - 1] + ordered_leads[len(ordered_leads) // 2]) / 2.0
    )
    per_kind = {}
    for kind in sorted({str(item["kind"]) for item in records}):
        kind_records = [item for item in records if str(item["kind"]) == kind]
        kind_predicted = sum(bool(item["predicted_before_onset"]) for item in kind_records)
        per_kind[kind] = {
            "onset_count": len(kind_records),
            "predicted_count": kind_predicted,
            "recall": kind_predicted / max(len(kind_records), 1),
        }
    for kind in sorted(set(per_kind) | {str(item["predicted_kind"]) for item in false_warnings}):
        false_count = sum(str(item["predicted_kind"]) == kind for item in false_warnings)
        predicted_for_kind = int(per_kind.get(kind, {}).get("predicted_count", 0))
        kind_precision = predicted_for_kind / max(predicted_for_kind + false_count, 1)
        kind_recall = float(per_kind.get(kind, {}).get("recall", 0.0))
        per_kind.setdefault(kind, {"onset_count": 0, "predicted_count": 0, "recall": 0.0})
        per_kind[kind].update(
            false_warning_episode_count=false_count,
            precision=kind_precision,
            f1=(
                2.0 * kind_precision * kind_recall
                / max(kind_precision + kind_recall, 1e-12)
            ),
        )
    lead_time_coverage = {
        f"at_least_{threshold}_seconds": {
            "attack_count": sum(lead >= threshold for lead in leads),
            "ratio": sum(lead >= threshold for lead in leads) / max(len(records), 1),
        }
        for threshold in (5, 10, 15, 20)
    }
    slo_status = (
        "PASSED_SYNTHETIC_SCENARIO"
        if records and not slo_misses and not false_warnings and not censored_warnings
        else "FAILED_OR_NOT_APPLICABLE"
    )
    slo_status_ru = (
        "пройдено в синтетическом сценарии"
        if slo_status == "PASSED_SYNTHETIC_SCENARIO"
        else "не пройдено или нет атак для проверки"
    )
    return {
        "forecast_horizon_seconds": max(
            (
                int(snapshot.get("koopman", {}).get(
                    "maximum_forecast_horizon_seconds",
                    snapshot.get("koopman", {}).get("forecast_horizon_seconds", step_seconds),
                ))
                for snapshot in snapshots
            ),
            default=step_seconds,
        ),
        "attack_onset_count": len(records),
        "predicted_before_onset_count": predicted_count,
        "prediction_recall": recall,
        "prediction_precision": precision,
        "prediction_f1": f1,
        "validation_scope": "single_seed_fixed_topology_showcase_post_run_scoring",
        "independent_holdout_validation_performed": False,
        "generalization_claimed": False,
        "causal_observation_contract": (
            "predictor_reads_only_current_and_past_sensor_telemetry; onset_truth_is_used_only_here_after_run"
        ),
        "accuracy_interpretation_ru": (
            "Метрики относятся к воспроизводимому демонстрационному сценарию с синтетическими "
            "предвестниками и не являются оценкой точности на операторских данных."
        ),
        "probability_calibration_status": "not_calibrated_risk_score",
        "false_positive_attack_id_count": len(false_warnings),
        "false_positive_warning_count": len(false_warnings),
        "false_positive_warnings": false_warnings,
        "right_censored_warning_count": len(censored_warnings),
        "right_censored_warnings": censored_warnings,
        "minimum_observed_lead_seconds": min(leads, default=0),
        "maximum_observed_lead_seconds": max(leads, default=0),
        "mean_observed_lead_seconds": sum(leads) / max(len(leads), 1),
        "median_observed_lead_seconds": median_lead,
        "lead_time_coverage": lead_time_coverage,
        "prediction_slo_seconds": required_lead_seconds,
        "prediction_slo_predicted_count": slo_predicted_count,
        "prediction_slo_recall": slo_predicted_count / max(len(records), 1),
        "prediction_slo_misses": slo_misses,
        "prediction_slo_status": slo_status,
        "prediction_slo_status_ru": slo_status_ru,
        "prediction_slo_semantics_ru": (
            "Строгая постпроверка: каждое фактическое начало должно быть "
            f"предупреждено минимум за {required_lead_seconds} с; ложные "
            "предупреждения считаются отдельными эпизодами. Если окно выходит за пределы "
            "прогона и совпадения нет, предупреждение остаётся непроверенным."
        ),
        "warning_episode_count": len(episodes),
        "observed_duration_seconds": observed_seconds,
        "gap_duration_seconds": gap_seconds,
        "warning_duration_seconds": warning_seconds,
        "warning_occupancy_ratio": warning_seconds / observed_seconds if observed_seconds else None,
        "false_warnings_per_simulated_hour": (
            len(false_warnings) * 3600.0 / observed_seconds if observed_seconds else None
        ),
        "per_attack_kind": per_kind,
        "all_attacks_predicted_at_least_one_step_ahead": bool(records) and predicted_count == len(records),
        "records": records,
    }


def _resilience_evaluation(
    snapshots: list[dict[str, Any]],
    *,
    step_seconds: int,
) -> dict[str, Any]:
    """Свести непрерывность сервиса и восстановление без скрытия деградации.

    Это постобработка уже рассчитанных снимков. Она не участвует в решениях
    Купмана/арбитра и поэтому не создаёт утечку будущего состояния в защиту.
    """
    attack_snapshots = [
        snapshot
        for snapshot in snapshots
        if snapshot.get("attacks", {}).get("active")
    ]
    grades = ("gold", "silver", "bronze")
    minimum_sla_compliance: dict[str, float] = {}
    first_gold_disadvantage = None
    complete_steps = 0
    gold_steps = 0
    for snapshot in attack_snapshots:
        tiers = snapshot.get("attacks", {}).get("sla_restoration", {}).get("tiers", [])
        ratios = {tier["sla_grade"]: tier["sla_compliant_flow_count"] / tier["flow_count"]
                  for tier in tiers if tier.get("sla_grade") in grades
                  and tier.get("flow_count", 0) > 0 and tier.get("sla_compliant_flow_count") is not None}
        for grade, ratio in ratios.items():
            minimum_sla_compliance[grade] = min(minimum_sla_compliance.get(grade, ratio), ratio)
        complete_steps += len(ratios) == len(grades)
        if "gold" in ratios:
            gold_steps += 1
            if first_gold_disadvantage is None and ratios["gold"] < max(ratios.values()):
                first_gold_disadvantage = {"step_index": snapshot["step_index"],
                                           "time_seconds": snapshot["time_seconds"],
                                           "compliance_ratio_by_grade": ratios}

    gold_never_worse = None
    if first_gold_disadvantage is not None:
        gold_never_worse = False
    elif attack_snapshots and complete_steps == len(attack_snapshots):
        gold_never_worse = True

    episodes: list[list[dict[str, Any]]] = []
    for snapshot in attack_snapshots:
        if (
            not episodes
            or int(snapshot.get("step_index", 0))
            > int(episodes[-1][-1].get("step_index", 0)) + 1
        ):
            episodes.append([snapshot])
        else:
            episodes[-1].append(snapshot)

    recovery_records: list[dict[str, Any]] = []
    for episode_index, episode in enumerate(episodes):
        first, last = episode[0], episode[-1]
        last_time = int(last.get("time_seconds", 0))
        next_episode = (
            episodes[episode_index + 1]
            if episode_index + 1 < len(episodes)
            else None
        )
        next_episode_start_time = (
            None
            if next_episode is None
            else int(next_episode[0].get("time_seconds", 0))
        )
        recovered = next(
            (
                snapshot
                for snapshot in snapshots
                if int(snapshot.get("time_seconds", 0)) > last_time
                and (
                    next_episode_start_time is None
                    or int(snapshot.get("time_seconds", 0))
                    < next_episode_start_time
                )
                and not snapshot.get("attacks", {}).get("active")
                and float(
                    snapshot.get("attacks", {}).get(
                        "legitimate_delivery_ratio",
                        1.0,
                    )
                ) >= 0.999
            ),
            None,
        )
        recovery_records.append(
            {
                "attack_episode_start_step": int(first.get("step_index", 0)),
                "attack_episode_end_step": int(last.get("step_index", 0)),
                "next_attack_episode_start_step": (
                    None
                    if next_episode is None
                    else int(next_episode[0].get("step_index", 0))
                ),
                "recovered_step": (
                    None if recovered is None else int(recovered.get("step_index", 0))
                ),
                "recovery_seconds_after_last_active_sample": (
                    None
                    if recovered is None
                    else int(recovered.get("time_seconds", 0)) - last_time
                ),
                "recovery_status": (
                    "recovered_before_next_attack_or_end"
                    if recovered is not None
                    else "interrupted_by_next_attack"
                    if next_episode is not None
                    else "not_observed_before_simulation_end"
                ),
            }
        )

    final_snapshot = snapshots[-1] if snapshots else {}
    final_attacks = final_snapshot.get("attacks", {})
    final_recovered = bool(snapshots) and (
        not final_attacks.get("active")
        and float(final_attacks.get("legitimate_delivery_ratio", 1.0)) >= 0.999
    )
    minimum_delivery = min(
        (
            float(snapshot.get("attacks", {}).get("legitimate_delivery_ratio", 1.0))
            for snapshot in attack_snapshots
        ),
        default=1.0,
    )
    if not attack_snapshots:
        status = "NOT_APPLICABLE_NO_ATTACKS"
    elif final_recovered and gold_steps < len(attack_snapshots):
        status = "RECOVERED_GOLD_CONTINUITY_UNASSESSED"
    elif final_recovered and minimum_sla_compliance["gold"] >= 0.95:
        status = "RESILIENT_GOLD_CONTINUITY_AND_RECOVERY"
    elif final_recovered:
        status = "DEGRADED_BUT_RECOVERED"
    else:
        status = "RECOVERY_INCOMPLETE_IN_SIMULATION_WINDOW"
    return {
        "status": status,
        "assessment_scope": "post_run_observation_only",
        "attack_snapshot_count": len(attack_snapshots),
        "attack_episode_count": len(episodes),
        "minimum_legitimate_delivery_ratio": round(minimum_delivery, 6),
        "minimum_sla_compliance_ratio": {
            grade: round(minimum_sla_compliance[grade], 6) if grade in minimum_sla_compliance else None
            for grade in grades
        },
        "gold_continuity_target_ratio": 0.95,
        "gold_continuity_target_origin": "explicit_showcase_engineering_assumption",
        "gold_never_worse_than_silver_or_bronze": gold_never_worse,
        "first_gold_disadvantage": first_gold_disadvantage,
        "final_state_recovered": final_recovered,
        "sampling_interval_seconds": step_seconds,
        "episodes": recovery_records,
    }


def _forecast_step_window(koopman: dict[str, Any]) -> tuple[int, int]:
    """Привести старый одношаговый и новый многогоризонтный API к одному окну."""
    target = int(koopman.get("forecast_target_step", -1))
    start = int(koopman.get("forecast_window_start_step", target))
    end = int(koopman.get("forecast_window_end_step", target))
    return min(start, end), max(start, end)


def _forecast_entity_pairs(koopman: dict[str, Any]) -> set[tuple[str, str]]:
    """Сохранить соответствие «наблюдаемый узел → классифицированный тип»."""
    records = koopman.get("forecast_entity_records", [])
    pairs = {
        (str(item["target_id"]), str(item.get("kind") or ""))
        for item in records
        if isinstance(item, dict) and item.get("target_id")
    }
    if pairs:
        return pairs
    # Совместимость со старыми односписочными снимками.
    targets = [str(item) for item in koopman.get("forecast_target_ids", [])]
    kinds = [
        item for item in str(koopman.get("forecast_next_attack_kind", "")).split("+")
        if item and item != "none"
    ]
    return {(target, kind) for target in targets for kind in (kinds or [""])}


def _validate_config(config: DynamicsConfig) -> None:
    validate_noise_variances(config.kalman_process_variance, config.kalman_measurement_variance)
    if config.attack_interval_steps is not None and config.attack_interval_steps <= 0:
        raise ValueError("Интервал атак должен быть положительным")
    if config.attack_start_step is not None:
        if not 1 <= config.attack_start_step <= config.step_count:
            raise ValueError("Первый шаг атаки должен попадать в интервал моделирования")
        precursor_steps = predictive_demo_minimum_steps(config.step_seconds) - 5 if config.attack_scenario == "predictive-demo" else 1
        if config.attack_start_step <= precursor_steps:
            raise ValueError("До первой атаки нужно оставить место для наблюдения предвестников после t0")
    if config.attack_scenario == "none" and (config.attack_start_step is not None or config.attack_interval_steps is not None):
        raise ValueError("Для расписания атак выберите сценарий атак")
    if config.step_seconds <= 0:
        raise ValueError("DynamicsConfig.step_seconds must be positive")
    if config.step_count < 0:
        raise ValueError("DynamicsConfig.step_count must be non-negative")
    if config.packet_sample_limit < 0:
        raise ValueError("DynamicsConfig.packet_sample_limit must be non-negative")
    if config.snapshot_detail not in {"full", "tensor", "summary"}:
        raise ValueError("DynamicsConfig.snapshot_detail must be one of: full, tensor, summary")
    if config.packet_detail not in {"summary", "flows", "sample"}:
        raise ValueError("DynamicsConfig.packet_detail must be one of: summary, flows, sample")
    if config.attack_scenario not in {"none", "mitre-demo", "predictive-demo"}:
        raise ValueError("DynamicsConfig.attack_scenario must be one of: none, mitre-demo, predictive-demo")
    if config.attack_scenario == "predictive-demo":
        minimum_steps = predictive_demo_minimum_steps(config.step_seconds)
        if config.step_count < minimum_steps:
            raise ValueError(
                "predictive-demo requires at least "
                f"{minimum_steps} steps for a {config.step_seconds}-second step "
                "so its causal precursor window and an attack fit in the run"
            )
    if config.attack_scenario != "none" and not config.include_packet_simulation:
        raise ValueError("Сценарий атак требует включённой пакетной симуляции")
    if (
        config.prediction_slo_seconds is not None
        and config.prediction_slo_seconds <= 0
    ):
        raise ValueError("DynamicsConfig.prediction_slo_seconds must be positive or None")
    if (
        config.warning_risk_threshold is not None
        and not 0.0 <= config.warning_risk_threshold <= 1.0
    ):
        raise ValueError("DynamicsConfig.warning_risk_threshold must be in [0, 1] or None")


def _resolved_prediction_slo_seconds(config: DynamicsConfig) -> int:
    """Вернуть целевой минимальный запас времени для контроля прогноза.

    Для predictive-demo по умолчанию действует требование «не менее 10 секунд».
    Длинный 20-секундный rollout при этом сохраняется как дополнительная
    аналитическая информация: его не нужно искусственно ухудшать до SLO.
    """
    if config.prediction_slo_seconds is not None:
        return int(config.prediction_slo_seconds)
    return 10 if config.attack_scenario == "predictive-demo" else int(config.step_seconds)


def _resolved_warning_risk_threshold(config: DynamicsConfig) -> float:
    """Вернуть активный порог, чтобы экспорт конфигурации не скрывал фактическое значение."""

    return (
        float(KOOPMAN_WARNING_RISK_THRESHOLD)
        if config.warning_risk_threshold is None
        else float(config.warning_risk_threshold)
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _sla_resilience(snapshots, step_seconds):
    """Среднее соблюдение SLA за фактически пройденный промежуток."""
    points = []
    for snapshot in snapshots:
        if snapshot["step_index"] == 0:
            continue
        tiers = snapshot.get("attacks", {}).get("sla_restoration", {}).get("tiers", [])
        score = weighted_sla_compliance(tiers)
        if score is not None:
            points.append({"step_index": snapshot["step_index"], "score": score})
    return {"grade_weights": dict(SLA_GRADE_WEIGHTS), "time_mean": sum(p["score"] for p in points) / len(points) if points else None,
            "minimum": min((p["score"] for p in points), default=None),
            "observed_duration_seconds": len(points) * step_seconds, "points": points,
            "definition": "weighted_compliant_flows_divided_by_weighted_accounted_flows",
            "scope": "observed_steps_not_infinite_horizon_optimum"}
