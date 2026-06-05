"""Entry point for generating the G-Net baseline artifacts."""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from src.gnet9.dynamics import DynamicsConfig, simulate_stationary_dynamics
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
    """Export Cisco-like L2 equipment profiles and baseline raw telemetry."""
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


def export_stationary_dynamics(model, path: Path, config: DynamicsConfig | None = None) -> None:
    """Export stationary snapshots for the healthy baseline."""
    dynamics = simulate_stationary_dynamics(model, config)
    path.write_text(json.dumps(dynamics, ensure_ascii=False, indent=2), encoding="utf-8")


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
    parser = argparse.ArgumentParser(description="Generate G-Net 9 baseline artifacts.")
    parser.add_argument(
        "--dynamics-steps",
        type=int,
        default=None,
        help="Number of 5-second dynamics transitions after t0. Default comes from constants.py.",
    )
    parser.add_argument(
        "--packet-sample-limit",
        type=int,
        default=48,
        help="Representative simulated packet events to store per dynamics snapshot.",
    )
    parser.add_argument(
        "--snapshot-detail",
        choices=("full", "tensor", "summary"),
        default="full",
        help="Dynamics snapshot detail: full graph, tensor-only, or compact summary.",
    )
    parser.add_argument(
        "--packet-detail",
        choices=("summary", "flows", "sample"),
        default="sample",
        help="Traffic export detail inside each dynamics snapshot.",
    )
    parser.add_argument(
        "--no-packet-simulation",
        action="store_true",
        help="Export dynamics snapshots without in-memory TCP/IP packet events.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    d0sl_policy_path = project_root / "policies" / "l1_policies.d0sl"
    model = GNetBaselineBuilder(d0sl_policy_path=d0sl_policy_path).build()
    artifacts = {
        "json": output_dir / "baseline_topology.json",
        "graphml": output_dir / "baseline_topology.graphml",
        "summary": output_dir / "baseline_summary.txt",
        "l1_profiles": output_dir / "l1_d0sl_profiles.json",
        "l1_monitoring": output_dir / "l1_monitoring.csv",
        "l2_profiles": output_dir / "l2_equipment_profiles.json",
        "d0sl_parsed": output_dir / "l1_d0sl_parsed.json",
        "d0sl_source": output_dir / "l1_policies.d0sl",
        "dynamics": output_dir / "network_dynamics.json",
        "network_png": output_dir / "network_logic.png",
        "layers_png": output_dir / "layer_scheme.png",
    }

    visualizer = GNetVisualizer(model)
    visualizer.draw_network_logic(artifacts["network_png"])
    visualizer.draw_layer_scheme(artifacts["layers_png"])

    model.export_json(artifacts["json"])
    model.export_graphml(artifacts["graphml"])
    model.export_summary(artifacts["summary"])
    export_l1_profiles(model, artifacts["l1_profiles"])
    export_l1_monitoring(model, artifacts["l1_monitoring"])
    export_l2_equipment_profiles(model, artifacts["l2_profiles"])
    export_parsed_d0sl_catalog(model, artifacts["d0sl_parsed"])
    dynamics_config = DynamicsConfig(
        step_count=args.dynamics_steps if args.dynamics_steps is not None else DynamicsConfig().step_count,
        packet_sample_limit=args.packet_sample_limit,
        include_packet_simulation=not args.no_packet_simulation,
        snapshot_detail=args.snapshot_detail,
        packet_detail=args.packet_detail,
    )
    export_stationary_dynamics(model, artifacts["dynamics"], dynamics_config)
    shutil.copyfile(d0sl_policy_path, artifacts["d0sl_source"])

    print("Done.")
    print(f"Artifacts saved to: {output_dir}")
    print(
        "Dynamics: "
        f"{dynamics_config.step_count} steps x {dynamics_config.step_seconds}s, "
        f"snapshot_detail={dynamics_config.snapshot_detail}, "
        f"packet_simulation={dynamics_config.include_packet_simulation}, "
        f"packet_detail={dynamics_config.packet_detail}"
    )


if __name__ == "__main__":
    main()
