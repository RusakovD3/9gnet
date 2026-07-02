"""Entry point for generating the G-Net baseline artifacts."""

import argparse
import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.gnet9.address_visualizer import draw_ip_address_map, export_ip_address_plan
from src.gnet9.attacks import attack_catalog
from src.gnet9.debug_visualizer import DebugFlowWindow, RuntimeCallTracer, export_debug_artifacts
from src.gnet9.dynamics import DynamicsConfig, simulate_stationary_dynamics
from src.gnet9.dynamics_charts import export_dynamics_charts
from src.gnet9.flow_visualizer import create_service_flow_window, draw_service_flow_map, show_visualization_windows
from src.gnet9.service_catalog import CODEC_CATALOG
from src.gnet9.topology_builder import GNetBaselineBuilder
from src.gnet9.visualizer import GNetVisualizer


L1_MONITORING_FIELDS = [
    "subscriber_id",
    "role",
    "home_access",
    "sla_grade",
    "traffic_kind",
    "codec",
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
            }
        )
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def export_l0_server_service_catalog(model, path: Path) -> None:
    """Экспортировать серверы, размещённые сервисы и базовую телеметрию L0."""
    server_runtime = {
        node_id: attrs.get("runtime", {})
        for node_id, attrs in model.graph.nodes(data=True)
        if attrs.get("role") == "service-server"
    }
    payload = {
        "servers": [{**asdict(server), "runtime": server_runtime.get(server.server_id, {})} for server in model.servers],
        "services": [asdict(service) for service in model.services],
        "codec_profiles": [asdict(profile) for profile in CODEC_CATALOG.values()],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_attack_catalog(model, path: Path) -> None:
    """Экспортировать MITRE-профили и рассчитанные приоритеты защиты узлов."""
    critical_nodes = [
        {"node_id": node_id, "role": attrs.get("role"), **attrs.get("critical_protection", {})}
        for node_id, attrs in model.graph.nodes(data=True)
        if attrs.get("critical_protection", {}).get("is_critical")
    ]
    payload = {"attacks": attack_catalog(), "critical_nodes": critical_nodes}
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


def export_stationary_dynamics(model, path: Path, config: DynamicsConfig | None = None) -> dict[str, Any]:
    """Export stationary snapshots for the healthy baseline."""
    dynamics = simulate_stationary_dynamics(model, config)
    path.write_text(json.dumps(dynamics, ensure_ascii=False, indent=2), encoding="utf-8")
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
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Создать артефакты эталонного состояния G-Net 9.")
    parser.add_argument(
        "--dynamics-steps",
        type=int,
        default=None,
        help="Число пятисекундных переходов после t0. По умолчанию берётся из constants.py.",
    )
    parser.add_argument(
        "--packet-sample-limit",
        type=int,
        default=48,
        help="Число показательных пакетных событий в каждом снимке динамики.",
    )
    parser.add_argument(
        "--snapshot-detail",
        choices=("full", "tensor", "summary"),
        default="full",
        help="Детализация снимка: полный граф, только тензоры или краткая сводка.",
    )
    parser.add_argument(
        "--packet-detail",
        choices=("summary", "flows", "sample"),
        default="sample",
        help="Детализация трафика внутри каждого снимка динамики.",
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
        choices=("none", "mitre-demo"),
        default="none",
        help="Сценарий воздействий: none или демонстрация DoS/DDoS/SYN flood по MITRE ATT&CK.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_enabled = args.debug_diagram or args.show_debug
    runtime_tracer = RuntimeCallTracer(project_root)
    if debug_enabled:
        runtime_tracer.start()

    d0sl_policy_path = project_root / "policies" / "l1_policies.d0sl"
    model = GNetBaselineBuilder(d0sl_policy_path=d0sl_policy_path).build()
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
        "charts_dir": output_dir / "charts",
        "network_png": output_dir / "network_logic.png",
        "layers_png": output_dir / "layer_scheme.png",
        "flows_png": output_dir / "service_flows.png",
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
    export_ip_address_plan(model, artifacts["ip_plan"])
    export_attack_catalog(model, artifacts["attacks"])
    export_parsed_d0sl_catalog(model, artifacts["d0sl_parsed"])
    dynamics_config = DynamicsConfig(
        step_count=args.dynamics_steps if args.dynamics_steps is not None else DynamicsConfig().step_count,
        packet_sample_limit=args.packet_sample_limit,
        include_packet_simulation=not args.no_packet_simulation,
        snapshot_detail=args.snapshot_detail,
        packet_detail=args.packet_detail,
        attack_scenario=args.attack_scenario,
    )
    dynamics = export_stationary_dynamics(model, artifacts["dynamics"], dynamics_config)
    flow_snapshots = dynamics["snapshots"]
    if not flow_snapshots or not flow_snapshots[0].get("traffic", {}).get("flows"):
        flow_dynamics = simulate_stationary_dynamics(
            model,
            DynamicsConfig(
                step_count=dynamics_config.step_count,
                step_seconds=dynamics_config.step_seconds,
                snapshot_detail="summary",
                packet_detail="flows",
                attack_scenario=dynamics_config.attack_scenario,
            ),
        )
        flow_snapshots = flow_dynamics["snapshots"]
    draw_service_flow_map(model, flow_snapshots[0]["traffic"]["flows"], artifacts["flows_png"])
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
        f"детализация пакетов={dynamics_config.packet_detail}"
        f", сценарий атак={dynamics_config.attack_scenario}"
    )
    print("Графики динамики:")
    for path in chart_paths.values():
        print(f"  - {path}")
    print(f"Карта сервисных потоков: {artifacts['flows_png']}")
    print(f"Карта IP-адресации: {artifacts['ip_map_png']}")
    if debug_paths:
        print("Графическая диагностика:")
        for path in debug_paths.values():
            print(f"  - {path}")

    windows = []
    if args.show_window:
        windows.append(create_service_flow_window(model, flow_snapshots))
    if args.show_debug:
        windows.append(DebugFlowWindow(runtime_tracer))
    if windows:
        show_visualization_windows(*windows)


if __name__ == "__main__":
    main()
