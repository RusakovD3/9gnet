"""Entry point for generating the G-Net baseline artifacts."""

import argparse
import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.gnet9.address_visualizer import draw_ip_address_map, export_ip_address_plan
from src.gnet9.attacks import (
    PREDICTIVE_DEMO_DEFAULT_SEED,
    PREDICTIVE_DEMO_STEP_COUNT,
    PREDICTIVE_DEMO_STEP_SECONDS,
    attack_catalog,
    mitre_tensor_mapping_catalog,
    predictive_demo_minimum_steps,
)
from src.gnet9.debug_visualizer import DebugFlowWindow, RuntimeCallTracer, export_debug_artifacts
from src.gnet9.decision_dialogue import export_algorithm_dialogue
from src.gnet9.dynamics import DynamicsConfig, simulate_stationary_dynamics
from src.gnet9.dynamics_charts import export_dynamics_charts
from src.gnet9.flow_visualizer import (
    create_service_flow_window,
    draw_service_flow_map,
    ensure_interactive_backend,
    show_visualization_windows,
)
from src.gnet9.service_catalog import CODEC_CATALOG
from src.gnet9.standards_audit import build_standards_and_equipment_audit
from src.gnet9.telemetry_validation import (
    TelemetryValidationConfig,
    TelemetryValidationError,
    load_pilot_calibration,
    read_telemetry_csv,
    validate_external_telemetry,
    write_validation_report,
)
from src.gnet9.topology_builder import GNetBaselineBuilder
from src.gnet9.visualizer import GNetVisualizer


L1_MONITORING_FIELDS = [
    "subscriber_id",
    "role",
    "home_access",
    "sla_grade",
    "traffic_kind",
    "codec",
    "access_profile",
    "access_capacity_mbps",
    "access_latency_ms",
    "access_architecture_reference_url",
    "second",
    "bitrate_kbps",
    "latency_ms",
    "jitter_ms",
    "packet_loss_percent",
    "queue_depth_packets",
    "utilization_rho",
    "bitrate_slo_ok",
    "latency_slo_ok",
    "loss_slo_ok",
    "jitter_slo_ok",
    "bitrate_drop_alarm",
]


def iter_nodes_by_level(model, level: str):
    """Yield graph nodes for one G-Net level."""
    return (
        (node_id, attrs)
        for node_id, attrs in model.graph.nodes(data=True)
        if attrs.get("level") == level
    )


def iter_l1_subscribers(model):
    return iter_nodes_by_level(model, "L1")


def iter_l2_equipment(model):
    return iter_nodes_by_level(model, "L2")


def export_l2_equipment_profiles(model, path: Path) -> None:
    """Export Cisco L2 profiles, provenance and synthetic baseline telemetry."""
    rows = []
    for node_id, attrs in iter_l2_equipment(model):
        rows.append(
            {
                "node_id": node_id,
                "role": attrs.get("role"),
                "platform_profile": attrs.get("platform_profile"),
                "platform_family": attrs.get("platform_family"),
                "l2_profile": attrs.get("l2_profile"),
                "l2_raw_baseline": attrs.get("l2_raw_baseline"),
                "l2_load_index": attrs.get("l2_load_index"),
                "l2_scale_pressure": attrs.get("l2_scale_pressure"),
                "l2_mgmt_pressure": attrs.get("l2_mgmt_pressure"),
                "l2_thermal_pressure": attrs.get("l2_thermal_pressure"),
                "l2_health_index": attrs.get("l2_health_index"),
                "physical_high_speed_ports_used": attrs.get("physical_high_speed_ports_used"),
                "power_architecture": attrs.get("power_architecture"),
            }
        )
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def export_l0_server_service_catalog(model, path: Path) -> None:
    """Экспортировать серверы, размещённые сервисы и базовую телеметрию L0."""
    server_runtime = {
        node_id: {
            "runtime": attrs.get("runtime", {}),
            "power_architecture": attrs.get("power_architecture", {}),
            "physical_links": [
                {
                    "peer": neighbor,
                    "capacity_mbps": model.graph.edges[node_id, neighbor].get("capacity_mbps"),
                    "physical_profile": model.graph.edges[node_id, neighbor].get("physical_profile"),
                }
                for neighbor in model.graph.neighbors(node_id)
                if model.graph.edges[node_id, neighbor].get("consumes_server_nic")
            ],
        }
        for node_id, attrs in model.graph.nodes(data=True)
        if attrs.get("role") == "service-server"
    }
    payload = {
        "servers": [
            {
                **asdict(server),
                **server_runtime.get(server.server_id, {}),
                "standby_services": model.graph.nodes[server.server_id].get("standby_services", []),
            }
            for server in model.servers
        ],
        "services": [
            {
                **asdict(service),
                "primary_server_id": model.graph.nodes[service.service_id].get("primary_server_id"),
                "standby_hosts": model.graph.nodes[service.service_id].get("standby_hosts", []),
                "failover_policy": model.graph.nodes[service.service_id].get("failover_policy"),
            }
            for service in model.services
        ],
        "codec_profiles": [asdict(profile) for profile in CODEC_CATALOG.values()],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_attack_catalog(
    model,
    path: Path,
    predictive_config: DynamicsConfig | None = None,
) -> None:
    """Экспортировать MITRE-профили и рассчитанные приоритеты защиты узлов."""
    critical_nodes = [
        {"node_id": node_id, "role": attrs.get("role"), **attrs.get("critical_protection", {})}
        for node_id, attrs in model.graph.nodes(data=True)
        if attrs.get("critical_protection", {}).get("is_critical")
    ]
    predictive_config = predictive_config or DynamicsConfig(
        step_seconds=PREDICTIVE_DEMO_STEP_SECONDS,
        step_count=PREDICTIVE_DEMO_STEP_COUNT,
        attack_scenario="predictive-demo",
        attack_seed=PREDICTIVE_DEMO_DEFAULT_SEED,
    )
    payload = {
        # `attacks` оставлен совместимым alias для прежнего mitre-demo.
        "attacks": attack_catalog(model, "mitre-demo"),
        "tensor_mappings": mitre_tensor_mapping_catalog(),
        "scenarios": {
            "mitre-demo": attack_catalog(model, "mitre-demo"),
            "predictive-demo": attack_catalog(
                model,
                "predictive-demo",
                seed=predictive_config.attack_seed,
                step_seconds=predictive_config.step_seconds,
                step_count=predictive_config.step_count,
            ),
        },
        "predictive_scenario_configuration": {
            "seed": predictive_config.attack_seed,
            "step_seconds": predictive_config.step_seconds,
            "step_count": predictive_config.step_count,
            "prediction_slo_seconds": predictive_config.to_dict().get(
                "resolved_prediction_slo_seconds"
            ),
        },
        "critical_nodes": critical_nodes,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_l1_monitoring(model, path: Path) -> None:
    """Export full second-by-second L1 monitoring history to CSV."""
    rows: list[dict[str, Any]] = []
    for node_id, attrs in iter_l1_subscribers(model):
        base = _l1_export_base(node_id, attrs)
        for point in attrs.get("monitoring", []):
            rows.append({**base, **point})

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=L1_MONITORING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def export_l1_profiles(model, path: Path) -> None:
    """Export compact L1 subscriber profiles to JSON."""
    profile_keys = [
        "target_bitrate_kbps",
        "min_bitrate_kbps",
        "latency_budget_ms",
        "d0sl_policy",
        "kendall_queue",
    ]
    profiles = []
    for node_id, attrs in iter_l1_subscribers(model):
        profile = _l1_export_base(node_id, attrs)
        profile.update({key: attrs.get(key) for key in profile_keys})
        profiles.append(profile)
    path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")


def export_parsed_d0sl_catalog(model, path: Path) -> None:
    """Export unique d0sl policies that were actually used by L1 nodes."""
    unique_policies = {}
    for _, attrs in iter_l1_subscribers(model):
        policy = attrs.get("d0sl_policy")
        if policy:
            unique_policies[policy["name"]] = policy

    path.write_text(json.dumps(list(unique_policies.values()), ensure_ascii=False, indent=2), encoding="utf-8")


def _dynamics_export_view(dynamics: dict[str, Any], packet_detail: str) -> dict[str, Any]:
    """Build a shallow, non-mutating view with the requested packet detail.

    The visualizers need flow records in memory, but writing all detailed
    flow dictionaries for every step makes a normal 60-step export unnecessarily
    large.  Only snapshot/traffic containers are copied; the immutable aggregate
    values are shared until ``json.dumps`` serializes the view.
    """
    if packet_detail not in {"summary", "flows", "sample"}:
        raise ValueError("packet_detail must be one of: summary, flows, sample")

    runtime_detail = str(dynamics.get("config", {}).get("packet_detail", packet_detail))
    exported = dict(dynamics)
    exported["config"] = {
        **dict(dynamics.get("config", {})),
        "packet_detail": packet_detail,
        "runtime_packet_detail": runtime_detail,
    }
    exported_snapshots: list[dict[str, Any]] = []
    for source_snapshot in dynamics.get("snapshots", []):
        snapshot = dict(source_snapshot)
        source_traffic = source_snapshot.get("traffic")
        if isinstance(source_traffic, dict):
            traffic = dict(source_traffic)
            if packet_detail == "summary":
                traffic.pop("flows", None)
                traffic.pop("packet_sample", None)
            elif packet_detail == "flows":
                traffic.pop("packet_sample", None)
            snapshot["traffic"] = traffic
        exported_snapshots.append(snapshot)
    exported["snapshots"] = exported_snapshots
    return exported


def export_stationary_dynamics(
    model,
    path: Path,
    config: DynamicsConfig | None = None,
    *,
    packet_detail: str | None = None,
) -> dict[str, Any]:
    """Simulate once, export the selected detail and retain runtime flows."""
    dynamics = simulate_stationary_dynamics(model, config)
    export_detail = packet_detail or str(dynamics.get("config", {}).get("packet_detail", "sample"))
    export_view = _dynamics_export_view(dynamics, export_detail)
    path.write_text(json.dumps(export_view, ensure_ascii=False, indent=2), encoding="utf-8")
    return dynamics


def _l1_export_base(node_id: str, attrs: dict[str, Any]) -> dict[str, Any]:
    """Common L1 fields used by both profile and monitoring exports."""
    return {
        "subscriber_id": node_id,
        "role": attrs.get("role"),
        "home_access": attrs.get("home_access"),
        "sla_grade": attrs.get("sla_grade"),
        "traffic_kind": attrs.get("traffic_kind"),
        "codec": attrs.get("codec"),
        "access_profile": attrs.get("access_profile"),
        "access_capacity_mbps": attrs.get("access_capacity_mbps"),
        "access_latency_ms": attrs.get("access_latency_ms"),
        "access_architecture_reference_url": attrs.get(
            "access_architecture_reference_url"
        ),
    }


def _positive_int(value: str) -> int:
    """Argparse converter that rejects zero and negative simulation sizes."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("значение должно быть положительным целым числом")
    return parsed


def _non_negative_int(value: str) -> int:
    """Argparse converter for counts where zero is a meaningful empty run."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("значение не может быть отрицательным")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Создать артефакты эталонного состояния G-Net 9.")
    parser.add_argument(
        "--dynamics-steps",
        type=_non_negative_int,
        default=None,
        help=(
            "Число переходов после t0. По умолчанию "
            f"{PREDICTIVE_DEMO_STEP_COUNT} для predictive-demo и "
            f"{DynamicsConfig().step_count} для остальных режимов."
        ),
    )
    parser.add_argument(
        "--step-seconds",
        type=_positive_int,
        default=None,
        help=(
            "Длительность шага динамики в секундах. По умолчанию "
            f"{PREDICTIVE_DEMO_STEP_SECONDS} с для predictive-demo и "
            f"{DynamicsConfig().step_seconds} с для остальных режимов."
        ),
    )
    parser.add_argument(
        "--prediction-slo-seconds",
        type=_positive_int,
        default=None,
        help=(
            "Минимальное упреждение, по которому оценивается прогноз. "
            "По умолчанию predictive-demo проверяется по SLO 10 с; "
            "длинный аналитический прогноз Купмана сохраняется отдельно."
        ),
    )
    parser.add_argument(
        "--validate-telemetry-csv",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "Проверить обезличенную внешнюю телеметрию: хронологически подобрать "
            "порог по ранней части ряда и проверить его на независимой проверочной "
            "части (holdout). "
            "Создаёт output/telemetry_validation_report.json и не запускает сеть."
        ),
    )
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=None,
        metavar="JSON",
        help=(
            "Применить только прошедший независимую проверку (holdout) отчёт внешней "
            "телеметрии как порог "
            "предупреждения для симуляции. Автоматических изменений реальной сети не выполняет."
        ),
    )
    parser.add_argument(
        "--telemetry-training-fraction",
        type=float,
        default=0.70,
        help="Доля ранней хронологической телеметрии для выбора порога (по умолчанию 0.70).",
    )
    parser.add_argument(
        "--telemetry-min-events-per-partition",
        type=_positive_int,
        default=5,
        help="Минимум подтверждённых начал атак в каждой части проверки (по умолчанию 5).",
    )
    parser.add_argument(
        "--telemetry-min-precision",
        type=float,
        default=0.90,
        help="Минимальная точность допуска внешнего порога (по умолчанию 0.90).",
    )
    parser.add_argument(
        "--telemetry-min-recall",
        type=float,
        default=0.90,
        help="Минимальная полнота допуска внешнего порога (по умолчанию 0.90).",
    )
    parser.add_argument(
        "--telemetry-max-false-warnings-per-hour",
        type=float,
        default=0.10,
        help="Максимум ложных эпизодов в час для допуска внешнего порога (по умолчанию 0.10).",
    )
    parser.add_argument(
        "--attack-seed",
        type=int,
        default=PREDICTIVE_DEMO_DEFAULT_SEED,
        help="Начальное число воспроизводимого взвешенного распределения атак.",
    )
    parser.add_argument(
        "--packet-sample-limit",
        type=_non_negative_int,
        default=48,
        help="Число показательных пакетных событий в каждом снимке динамики.",
    )
    parser.add_argument(
        "--snapshot-detail",
        choices=("full", "tensor", "summary"),
        default="summary",
        help=(
            "Детализация снимка: полный граф, только тензоры или краткая сводка. "
            "По умолчанию summary: полный граф уже хранится отдельно в baseline_topology.json."
        ),
    )
    parser.add_argument(
        "--packet-detail",
        choices=("summary", "flows", "sample"),
        default="summary",
        help=(
            "Детализация трафика в network_dynamics.json. По умолчанию summary; "
            "потоки для карт и окна рассчитываются один раз и остаются только в памяти."
        ),
    )
    parser.add_argument(
        "--no-packet-simulation",
        action="store_true",
        help="Экспортировать снимки динамики без пакетных событий TCP/IP.",
    )
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="После экспорта открыть интерактивное окно с анимацией сервисных потоков.",
    )
    parser.add_argument(
        "--debug-diagram",
        action="store_true",
        help="Записать блок-схему, граф вызовов функций и полную трассу запуска в output/debug.",
    )
    parser.add_argument(
        "--show-debug",
        action="store_true",
        help="Открыть окно графической диагностики и создать артефакты output/debug.",
    )
    parser.add_argument(
        "--attack-scenario",
        choices=("none", "mitre-demo", "predictive-demo"),
        default="none",
        help=(
            "Сценарий: none, фиксированный mitre-demo или predictive-demo "
            "с причинным прогнозом, взвешенными атаками и приоритетным переназначением Gold."
        ),
    )
    args = parser.parse_args(argv)
    if args.no_packet_simulation and args.show_window:
        parser.error("интерактивная карта потоков требует пакетную модель; уберите --no-packet-simulation")
    if args.no_packet_simulation and args.attack_scenario != "none":
        parser.error("сценарий атак требует пакетную модель; уберите --no-packet-simulation")
    if args.attack_scenario == "predictive-demo":
        resolved_step_seconds = args.step_seconds or PREDICTIVE_DEMO_STEP_SECONDS
        resolved_step_count = (
            args.dynamics_steps
            if args.dynamics_steps is not None
            else PREDICTIVE_DEMO_STEP_COUNT
        )
        minimum_steps = predictive_demo_minimum_steps(resolved_step_seconds)
        if resolved_step_count < minimum_steps:
            parser.error(
                "predictive-demo требует не менее "
                f"{minimum_steps} шагов при длительности шага "
                f"{resolved_step_seconds} с: нужны причинные предвестники и место для атаки"
            )
    if args.validate_telemetry_csv and args.calibration_report:
        parser.error("используйте либо --validate-telemetry-csv, либо --calibration-report, но не оба аргумента")
    return args


def _routing_action_score(snapshot: dict[str, Any]) -> int:
    routing = snapshot.get("attacks", {}).get("routing", {})
    return (
        int(routing.get("rerouted_flow_count", 0))
        + int(routing.get("failover_flow_count", 0))
        + int(routing.get("isolated_flow_count", 0))
        + int(routing.get("protected_endpoint_flow_count", 0))
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.validate_telemetry_csv:
        telemetry_config = TelemetryValidationConfig(
            training_fraction=args.telemetry_training_fraction,
            minimum_events_per_partition=args.telemetry_min_events_per_partition,
            minimum_precision=args.telemetry_min_precision,
            minimum_recall=args.telemetry_min_recall,
            maximum_false_warnings_per_hour=args.telemetry_max_false_warnings_per_hour,
        )
        try:
            telemetry = read_telemetry_csv(args.validate_telemetry_csv)
            report = validate_external_telemetry(
                telemetry,
                telemetry_config,
                source_path=args.validate_telemetry_csv,
            )
        except TelemetryValidationError as error:
            raise SystemExit(f"Ошибка проверки внешней телеметрии: {error}") from error
        report_path = output_dir / "telemetry_validation_report.json"
        write_validation_report(report, report_path)
        gate = report["deployment_gate"]
        metrics = report["independent_holdout"]["metrics"]
        print(f"Отчёт валидации телеметрии: {report_path}")
        print(
            "Независимый holdout: "
            f"precision={metrics['precision'] * 100:.1f}%, "
            f"recall={metrics['recall'] * 100:.1f}%, "
            f"ложных эпизодов/ч={metrics['false_warnings_per_hour']:.3f}; "
            f"допуск={gate['status']}."
        )
        return
    calibration_profile: dict[str, Any] | None = None
    if args.calibration_report:
        try:
            calibration_profile = load_pilot_calibration(args.calibration_report)
        except TelemetryValidationError as error:
            raise SystemExit(f"Ошибка применения отчёта калибровки: {error}") from error
    # Артефакты ранних ручных прототипов больше не входят в контракт output/.
    # Удаляем только известные имена, не затрагивая пользовательские файлы.
    for legacy_preview in (
        "interactive_flow_preview.png",
        "koopman_warning_window_test.png",
    ):
        (output_dir / legacy_preview).unlink(missing_ok=True)
    debug_enabled = args.debug_diagram or args.show_debug
    runtime_tracer = RuntimeCallTracer(project_root)
    if debug_enabled:
        runtime_tracer.start()

    d0sl_policy_path = project_root / "policies" / "l1_policies.d0sl"
    model = GNetBaselineBuilder(d0sl_policy_path=d0sl_policy_path).build()
    predictive_demo = args.attack_scenario == "predictive-demo"
    export_packet_detail = args.packet_detail
    runtime_packet_detail = (
        "flows"
        if not args.no_packet_simulation and export_packet_detail == "summary"
        else export_packet_detail
    )
    dynamics_config = DynamicsConfig(
        step_count=(
            args.dynamics_steps
            if args.dynamics_steps is not None
            else PREDICTIVE_DEMO_STEP_COUNT
            if predictive_demo
            else DynamicsConfig().step_count
        ),
        step_seconds=(
            args.step_seconds
            if args.step_seconds is not None
            else PREDICTIVE_DEMO_STEP_SECONDS
            if predictive_demo
            else DynamicsConfig().step_seconds
        ),
        packet_sample_limit=args.packet_sample_limit,
        include_packet_simulation=not args.no_packet_simulation,
        snapshot_detail=args.snapshot_detail,
        packet_detail=runtime_packet_detail,
        attack_scenario=args.attack_scenario,
        attack_seed=args.attack_seed,
        prediction_slo_seconds=args.prediction_slo_seconds,
        warning_risk_threshold=(
            calibration_profile["warning_risk_threshold"]
            if calibration_profile else None
        ),
        warning_threshold_origin=(
            calibration_profile["warning_threshold_origin"]
            if calibration_profile
            else "model_default_not_externally_validated"
        ),
    )
    artifacts = {
        "json": output_dir / "baseline_topology.json",
        "graphml": output_dir / "baseline_topology.graphml",
        "summary": output_dir / "baseline_summary.txt",
        "l1_profiles": output_dir / "l1_d0sl_profiles.json",
        "l1_monitoring": output_dir / "l1_monitoring.csv",
        "l2_profiles": output_dir / "l2_equipment_profiles.json",
        "l0_servers": output_dir / "l0_server_service_catalog.json",
        "ip_plan": output_dir / "ip_address_plan.json",
        "attacks": output_dir / "mitre_attack_catalog.json",
        "d0sl_parsed": output_dir / "l1_d0sl_parsed.json",
        "d0sl_source": output_dir / "l1_policies.d0sl",
        "dynamics": output_dir / "network_dynamics.json",
        "algorithm_dialogue_json": output_dir / "algorithm_dialogue.json",
        "algorithm_dialogue_md": output_dir / "algorithm_dialogue.md",
        "standards_audit": output_dir / "standards_and_equipment_audit.json",
        "charts_dir": output_dir / "charts",
        "network_png": output_dir / "network_logic.png",
        "layers_png": output_dir / "layer_scheme.png",
        "flows_png": output_dir / "service_flows.png",
        "remapped_flows_png": output_dir / "service_flows_remapped.png",
        "ip_map_png": output_dir / "ip_address_map.png",
        "debug_dir": output_dir / "debug",
    }

    visualizer = GNetVisualizer(model)
    visualizer.draw_network_logic(artifacts["network_png"])
    visualizer.draw_layer_scheme(artifacts["layers_png"])
    draw_ip_address_map(model, artifacts["ip_map_png"])

    model.export_json(artifacts["json"])
    model.export_graphml(artifacts["graphml"])
    model.export_summary(artifacts["summary"])
    export_l1_profiles(model, artifacts["l1_profiles"])
    export_l1_monitoring(model, artifacts["l1_monitoring"])
    export_l2_equipment_profiles(model, artifacts["l2_profiles"])
    export_l0_server_service_catalog(model, artifacts["l0_servers"])
    artifacts["standards_audit"].write_text(
        json.dumps(build_standards_and_equipment_audit(model), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    export_ip_address_plan(model, artifacts["ip_plan"])
    export_attack_catalog(
        model,
        artifacts["attacks"],
        dynamics_config if predictive_demo else None,
    )
    export_parsed_d0sl_catalog(model, artifacts["d0sl_parsed"])
    dynamics = export_stationary_dynamics(
        model,
        artifacts["dynamics"],
        dynamics_config,
        packet_detail=export_packet_detail,
    )
    export_algorithm_dialogue(
        dynamics,
        artifacts["algorithm_dialogue_json"],
        artifacts["algorithm_dialogue_md"],
    )
    flow_snapshots = dynamics["snapshots"] if dynamics_config.include_packet_simulation else []
    if flow_snapshots:
        draw_service_flow_map(model, flow_snapshots[0]["traffic"]["flows"], artifacts["flows_png"])
        most_active_snapshot = max(flow_snapshots, key=_routing_action_score)
        if _routing_action_score(most_active_snapshot) > 0:
            attack_names = ", ".join(
                str(event.get("name_ru", event.get("kind", "")))
                for event in most_active_snapshot.get("attacks", {}).get("events", [])
            ) or "предиктивная защита до начала воздействия"
            draw_service_flow_map(
                model,
                most_active_snapshot["traffic"]["flows"],
                artifacts["remapped_flows_png"],
                title_suffix=(
                    f"максимум управляющих действий · шаг {most_active_snapshot.get('step_index', 0)} · "
                    f"t = {most_active_snapshot.get('time_seconds', 0)} с · {attack_names}"
                ),
            )
        else:
            artifacts["remapped_flows_png"].unlink(missing_ok=True)
    else:
        draw_service_flow_map(
            model,
            [],
            artifacts["flows_png"],
            title_suffix="пакетная модель отключена: показана только топология",
        )
        artifacts["remapped_flows_png"].unlink(missing_ok=True)
    chart_paths = export_dynamics_charts(
        dynamics,
        artifacts["charts_dir"],
        model=model,
        flow_snapshots=flow_snapshots,
    )
    shutil.copyfile(d0sl_policy_path, artifacts["d0sl_source"])
    debug_paths: dict[str, Path] = {}
    if debug_enabled:
        runtime_tracer.stop()
        debug_paths = export_debug_artifacts(runtime_tracer, artifacts["debug_dir"])

    print("Готово.")
    print(f"Артефакты сохранены в: {output_dir}")
    print(
        "Динамика: "
        f"{dynamics_config.step_count} шагов по {dynamics_config.step_seconds} с, "
        f"детализация снимка={dynamics_config.snapshot_detail}, "
        f"пакетная симуляция={dynamics_config.include_packet_simulation}, "
        f"детализация экспорта пакетов={export_packet_detail}"
        + (
            f", внутренний расчёт={runtime_packet_detail}"
            if runtime_packet_detail != export_packet_detail
            else ""
        )
        + f", сценарий атак={dynamics_config.attack_scenario}"
    )
    koopman_evaluation = dynamics.get("koopman_evaluation", {})
    if koopman_evaluation.get("attack_onset_count"):
        print(
            "Купман: заранее распознано "
            f"{koopman_evaluation.get('predicted_before_onset_count', 0)} из "
            f"{koopman_evaluation.get('attack_onset_count', 0)} воздействий; "
            f"минимальное упреждение={koopman_evaluation.get('minimum_observed_lead_seconds', 0)} с; "
            f"precision текущего стенда={koopman_evaluation.get('prediction_precision', 0.0) * 100:.1f}%; "
            f"recall={koopman_evaluation.get('prediction_recall', 0.0) * 100:.1f}%; "
            f"максимальный горизонт={koopman_evaluation.get('forecast_horizon_seconds', 0)} с; "
            f"защита применена к {koopman_evaluation.get('preventive_defense_applied_count', 0)} воздействиям."
        )
        print(
            "SLO прогноза: не менее "
            f"{koopman_evaluation.get('prediction_slo_seconds', 0)} с — "
            f"{koopman_evaluation.get('prediction_slo_predicted_count', 0)} из "
            f"{koopman_evaluation.get('attack_onset_count', 0)}; "
            f"статус={koopman_evaluation.get('prediction_slo_status_ru', '—')}."
        )
        if not koopman_evaluation.get("independent_holdout_validation_performed", False):
            print(
                "  Важно: это постоценка синтетического single-seed сценария, "
                "а не подтверждённая точность на независимых операторских данных."
            )
    print("Графики динамики:")
    for path in chart_paths.values():
        print(f"  - {path}")
    print(f"Карта сервисных потоков: {artifacts['flows_png']}")
    if artifacts["remapped_flows_png"].exists():
        print(f"Карта наиболее активного переназначения: {artifacts['remapped_flows_png']}")
    print(f"Карта IP-адресации: {artifacts['ip_map_png']}")
    print(f"Диалог алгоритмов: {artifacts['algorithm_dialogue_md']}")
    print(f"Аудит RFC и характеристик: {artifacts['standards_audit']}")
    if debug_paths:
        print("Графическая диагностика:")
        for path in debug_paths.values():
            print(f"  - {path}")

    windows = []
    if args.show_window or args.show_debug:
        # Backend выбирается до создания первой GUI-фигуры. Это важно и для
        # одиночного --show-debug: FigureCanvasAgg не умеет показывать окно.
        ensure_interactive_backend()
    if args.show_window:
        windows.append(create_service_flow_window(model, flow_snapshots))
    if args.show_debug:
        windows.append(DebugFlowWindow(runtime_tracer))
    if windows:
        show_visualization_windows(*windows)


if __name__ == "__main__":
    main()
