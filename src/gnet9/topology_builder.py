"""Builder for the 9-level baseline G-Net topology.

The builder creates a reproducible t0 state: no attacks, no overload, stable
queues and normal SLA/SLO values. This baseline is the point of comparison for
future experiments with failures, attacks, remapping and forecasting.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np

from .constants import (
    AGGREGATION_FIXED,
    AGGREGATION_MOBILE,
    CRITICALITY_COLORS,
    DEFAULT_SERVICES,
    FIXED_SUBSCRIBERS_PER_AGG,
    L2_NODE_COUNT,
    L1_MONITORING_SECONDS,
    MOBILE_SUBSCRIBERS_PER_AGG,
)
from .l1_d0sl import (
    TrafficKind,
    build_l1_queue_model,
    load_l1_d0sl_catalog,
    simulate_l1_monitoring,
)
from .baseline import (
    edge_transport_tensor,
    l0_service_tensor,
    l1_subscriber_tensor,
    l2_equipment_tensor,
    l3_medium_tensor,
    l4_infrastructure_tensor,
    l5_role_tensor,
    l6_power_tensor,
    l7_arbitrator_tensor,
    l8_placement_tensor,
)
from .metrics import vertex_proximity_index
from .models import NetworkModel, ServiceProfile, SliceProfile
from .tensors import build_layer_tensor, build_transport_tensor
from .l2_equipment import (
    build_l2_raw_baseline,
    build_l2_summary_metrics,
    l2_profile_for_role,
)


class GNetBaselineBuilder:
    """Build a readable and future-ready baseline topology.

    Important modeling decision: L2 contains only active network equipment
    from the 9-level model: core and aggregation routers. Subscribers are L1 and
    connect directly to aggregation routers. There are no fake access-layer nodes.
    """

    def __init__(self, d0sl_policy_path: Path | None = None) -> None:
        self.graph = nx.Graph()
        self.services = [ServiceProfile(**asdict(service)) for service in DEFAULT_SERVICES]
        self.slices: list[SliceProfile] = []

        project_root = Path(__file__).resolve().parents[2]
        self.d0sl_policy_path = d0sl_policy_path or project_root / "policies" / "l1_policies.d0sl"
        self.l1_policy_catalog = load_l1_d0sl_catalog(self.d0sl_policy_path)

    def build(self) -> NetworkModel:
        """Generate the full graph and return it as a NetworkModel."""
        self._add_terrain_anchors()
        core_nodes = self._add_core_routers()
        self._add_aggregation_routers()
        self._connect_core_to_aggregation()
        self._add_subscribers()
        self._add_services()
        self._annotate_core_metrics(core_nodes)
        self._add_arbitrator()
        self._attach_placement_tensors()
        self._validate()

        level_summary = Counter(attrs["level"] for _, attrs in self.graph.nodes(data=True))
        return NetworkModel(
            graph=self.graph,
            services=self.services,
            slices=self.slices,
            level_summary=dict(level_summary),
            notes=self._build_notes(),
        )

    def _build_notes(self) -> list[str]:
        return [
            "Topology is generated as t0 ideal baseline for future Koopman/Lyapunov experiments.",
            "L2 contains only core and aggregation nodes from the 9-level model.",
            "Layer tensors are numeric state vectors with explicit metric names and units.",
            "Subscribers are connected directly to aggregation nodes.",
            f"L1 subscribers use executable d0sl SLA/SLO/SLI policies from {self.d0sl_policy_path}.",
            "L1 traffic classes: broadcast MP3, FTP and DNS.",
            "L3/L4 tensors are attached to transport edges; L5/L6 tensors are attached to equipment and subscribers where relevant.",
            "L7 tensor stores baseline Koopman/Lyapunov/Hausdorff decision features.",
            "L8 tensors store coordinates for physical/user placement and Hausdorff distance calculations.",
        ]

    # ---------------------------------------------------------------------
    # L8: topo-base
    # ---------------------------------------------------------------------
    def _add_terrain_anchors(self) -> None:
        """Add terrain/topology anchor nodes used as the L8 background."""
        topo_nodes = {
            "TERRAIN_NW": (-6.0, 7.0),
            "TERRAIN_NE": (6.0, 7.0),
            "TERRAIN_W": (-8.0, 0.5),
            "TERRAIN_C": (0.0, 0.0),
            "TERRAIN_E": (8.0, 0.5),
            "TERRAIN_SW": (-6.0, -7.0),
            "TERRAIN_SE": (6.0, -7.0),
        }

        for node_id, pos in topo_nodes.items():
            self.graph.add_node(
                node_id,
                level="L8",
                role="terrain-anchor",
                label=node_id.replace("TERRAIN_", ""),
                pos=pos,
                color="#d9d9d9",
                visible_in_logic=False,
                tensor=build_layer_tensor("L8", l8_placement_tensor(pos, "terrain-anchor")),
            )

    # ---------------------------------------------------------------------
    # L5/L2: core rings and active equipment
    # ---------------------------------------------------------------------
    def _add_core_routers(self) -> list[str]:
        """Build 12 core routers grouped into 3 rings."""
        ring_specs = {
            "RING_A": {"center": (-4.5, 1.7), "radius": 1.65, "nodes": ["C1", "C2", "C3", "C4"]},
            "RING_B": {"center": (0.0, 1.7), "radius": 1.65, "nodes": ["C5", "C6", "C7", "C8"]},
            "RING_C": {"center": (4.5, 1.7), "radius": 1.65, "nodes": ["C9", "C10", "C11", "C12"]},
        }

        core_nodes: list[str] = []
        for ring_name, spec in ring_specs.items():
            ring_nodes = spec["nodes"]
            for index, node_id in enumerate(ring_nodes):
                pos = self._ring_position(spec["center"], spec["radius"], index, len(ring_nodes))
                criticality = "gold" if node_id in {"C1", "C5", "C9"} else "silver"
                self._add_core_router(node_id, pos, ring_name, criticality)
                core_nodes.append(node_id)

            self._connect_ring(ring_nodes)

        self._connect_inter_ring_links()
        self._add_slice_profiles(core_nodes)
        return core_nodes

    @staticmethod
    def _ring_position(center: tuple[float, float], radius: float, index: int, count: int) -> tuple[float, float]:
        center_x, center_y = center
        angle = math.pi / 2 + index * (2 * math.pi / count)
        return (center_x + radius * math.cos(angle), center_y + radius * math.sin(angle))

    def _add_core_router(self, node_id: str, pos: tuple[float, float], ring_name: str, criticality: str) -> None:
        self._add_l2_router(
            node_id,
            role="core-router",
            pos=pos,
            criticality=criticality,
            port_speed_mbps=400_000.0,
            port_delay_ms=0.08,
            ring=ring_name,
            power_zone=f"PWR_{ring_name[-1]}",
        )

    def _connect_ring(self, ring_nodes: list[str]) -> None:
        for source, target in zip(ring_nodes, ring_nodes[1:] + ring_nodes[:1]):
            self._add_transport_edge(
                source,
                target,
                medium="fiber",
                capacity_mbps=400_000.0,
                latency_ms=2.2,
                redundancy=0.96,
                logical_level="L5",
                physical_level="L4",
            )

    def _connect_inter_ring_links(self) -> None:
        for source, target in [("C2", "C5"), ("C4", "C7"), ("C6", "C9"), ("C8", "C11"), ("C10", "C1"), ("C12", "C3")]:
            self._add_transport_edge(
                source,
                target,
                medium="fiber",
                capacity_mbps=200_000.0,
                latency_ms=3.4,
                redundancy=0.91,
                logical_level="L5",
                physical_level="L4",
            )

    def _add_slice_profiles(self, core_nodes: list[str]) -> None:
        self.slices.extend(
            [
                SliceProfile("GoldBackbone", "gold", ["C1", "C2", "C5", "C6", "C9", "C10"], 0.30),
                SliceProfile("SilverEnterprise", "silver", ["C3", "C4", "C7", "C8", "C11", "C12"], 0.22),
                SliceProfile("BronzeBestEffort", "bronze", core_nodes, 0.15),
            ]
        )

    def _add_aggregation_routers(self) -> None:
        """Add 6 aggregation routers."""
        aggregation_positions = {
            "A1": (-5.8, -1.2),
            "A2": (-3.2, -1.2),
            "A3": (-1.0, -1.2),
            "A4": (1.0, -1.2),
            "A5": (3.2, -1.2),
            "A6": (5.8, -1.2),
        }

        for node_id, pos in aggregation_positions.items():
            self._add_l2_router(
                node_id,
                role="aggregation-router",
                pos=pos,
                criticality="silver",
                port_speed_mbps=100_000.0,
                port_delay_ms=0.18,
            )

    def _add_l2_router(
        self,
        node_id: str,
        *,
        role: str,
        pos: tuple[float, float],
        criticality: str,
        port_speed_mbps: float,
        port_delay_ms: float,
        **extra_attrs,
    ) -> None:
        profile = l2_profile_for_role(role, criticality=criticality)
        raw_l2 = build_l2_raw_baseline(profile, role=role, criticality=criticality)
        summary_l2 = build_l2_summary_metrics(raw_l2, profile)

        self.graph.add_node(
            node_id,
            level="L2",
            role=role,
            label=node_id,
            pos=pos,
            slice_grade=criticality,
            visible_in_logic=True,
            color=CRITICALITY_COLORS[criticality],
            platform_profile=profile.name,
            platform_family=profile.model_family,
            platform_source=profile.source_note,
            platform_source_url=profile.source_url,
            l2_profile=profile.to_dict(),
            l2_raw_baseline=raw_l2,
            **summary_l2,
            **extra_attrs,
            tensor=build_layer_tensor("L2", l2_equipment_tensor(role, criticality, port_speed_mbps, port_delay_ms)),
            l5_tensor=build_layer_tensor("L5", l5_role_tensor(role)),
            l6_tensor=build_layer_tensor("L6", l6_power_tensor(role)),
        )

    def _connect_core_to_aggregation(self) -> None:
        """Connect every aggregation router to primary and secondary core routers."""
        connections = [
            ("A1", "C1", 100_000.0, 0.90, "C6", 40_000.0, 0.82),
            ("A2", "C3", 100_000.0, 0.90, "C8", 40_000.0, 0.82),
            ("A3", "C5", 100_000.0, 0.90, "C10", 40_000.0, 0.82),
            ("A4", "C7", 100_000.0, 0.90, "C12", 40_000.0, 0.82),
            ("A5", "C9", 100_000.0, 0.90, "C2", 40_000.0, 0.82),
            ("A6", "C11", 100_000.0, 0.90, "C4", 40_000.0, 0.82),
        ]
        for agg, primary, prim_cap, prim_red, secondary, sec_cap, sec_red in connections:
            self._add_transport_edge(
                primary,
                agg,
                medium="fiber",
                capacity_mbps=prim_cap,
                latency_ms=4.4,
                redundancy=prim_red,
                logical_level="L4",
                physical_level="L4",
            )
            self._add_transport_edge(
                secondary,
                agg,
                medium="fiber",
                capacity_mbps=sec_cap,
                latency_ms=5.0,
                redundancy=sec_red,
                logical_level="L4",
                physical_level="L4",
            )

    # ---------------------------------------------------------------------
    # L1: subscribers
    # ---------------------------------------------------------------------
    def _add_subscribers(self) -> None:
        """Add mobile and fixed subscribers from d0sl policies."""
        mobile_spread = np.linspace(210, 330, MOBILE_SUBSCRIBERS_PER_AGG, endpoint=False)
        fixed_spread = np.linspace(200, 340, FIXED_SUBSCRIBERS_PER_AGG, endpoint=False)

        for group_index, aggregation_node in enumerate(AGGREGATION_MOBILE, start=1):
            self._add_subscriber_group(
                prefix="M",
                role="mobile-subscriber",
                label="M",
                aggregation_node=aggregation_node,
                group_index=group_index,
                angles_deg=mobile_spread,
                grade_fn=lambda index: "gold" if index <= 8 else "bronze",
                traffic_shift=group_index,
                medium="radio",
                color="#b7e4c7",
                visible_limit=14,
                seed_base=1000,
            )

        for group_index, aggregation_node in enumerate(AGGREGATION_FIXED, start=1):
            self._add_subscriber_group(
                prefix="F",
                role="fixed-subscriber",
                label="PC",
                aggregation_node=aggregation_node,
                group_index=group_index,
                angles_deg=fixed_spread,
                grade_fn=lambda index: "silver" if index <= 10 else "bronze",
                traffic_shift=group_index + 1,
                medium="ethernet",
                color="#95d5b2",
                visible_limit=10,
                seed_base=2000,
            )

    def _add_subscriber_group(
        self,
        *,
        prefix: str,
        role: str,
        label: str,
        aggregation_node: str,
        group_index: int,
        angles_deg: np.ndarray,
        grade_fn,
        traffic_shift: int,
        medium: str,
        color: str,
        visible_limit: int,
        seed_base: int,
    ) -> None:
        traffic_cycle = [TrafficKind.BROADCAST_MP3.value, TrafficKind.FTP.value, TrafficKind.DNS.value]
        x0, y0 = self.graph.nodes[aggregation_node]["pos"]

        for subscriber_index, angle_deg in enumerate(angles_deg, start=1):
            node_id = f"{prefix}{group_index}_{subscriber_index:02d}"
            grade = grade_fn(subscriber_index)
            traffic = traffic_cycle[(subscriber_index + traffic_shift) % len(traffic_cycle)]
            policy = self.l1_policy_catalog.get(grade, traffic)
            queue_model = build_l1_queue_model(policy)
            monitoring = simulate_l1_monitoring(
                policy,
                queue_model,
                seconds=L1_MONITORING_SECONDS,
                seed=seed_base + group_index * 100 + subscriber_index,
            )

            self.graph.add_node(
                node_id,
                level="L1",
                role=role,
                label=label,
                pos=self._subscriber_position(x0, y0, angle_deg, subscriber_index, prefix),
                home_access=aggregation_node,
                sla_grade=policy.grade.value,
                traffic_kind=policy.traffic.value,
                codec=policy.codec,
                target_bitrate_kbps=policy.target_bitrate_kbps,
                min_bitrate_kbps=policy.min_bitrate_kbps,
                latency_budget_ms=policy.latency_budget_ms,
                d0sl_policy=policy.to_dict(),
                kendall_queue=queue_model.to_dict(),
                monitoring=[point.to_dict() for point in monitoring],
                visible_in_logic=subscriber_index <= visible_limit,
                color=color,
                l6_tensor=build_layer_tensor("L6", l6_power_tensor(role)),
                tensor=build_layer_tensor("L1", l1_subscriber_tensor(policy, "mobile" if prefix == "M" else "fixed")),
            )

            self._connect_subscriber(aggregation_node, node_id, policy, medium)

    @staticmethod
    def _subscriber_position(x0: float, y0: float, angle_deg: float, subscriber_index: int, prefix: str) -> tuple[float, float]:
        angle = np.deg2rad(angle_deg)
        if prefix == "M":
            radius = 1.7 + 0.17 * (subscriber_index % 3)
            return (x0 + radius * np.cos(angle), y0 - 1.35 + radius * np.sin(angle) * 0.55)

        radius = 1.5 + 0.13 * (subscriber_index % 4)
        return (x0 + radius * np.cos(angle), y0 - 1.2 + radius * np.sin(angle) * 0.52)

    def _connect_subscriber(self, aggregation_node: str, node_id: str, policy, medium: str) -> None:
        if medium == "radio":
            capacity_mbps = max(10.0, policy.target_bitrate_kbps / 1000.0 * 50.0)
            latency_ms = 8.0 if policy.traffic.value == TrafficKind.DNS.value else 11.0
            redundancy = 0.42
        else:
            capacity_mbps = max(100.0, policy.target_bitrate_kbps / 1000.0 * 80.0)
            latency_ms = 1.2 if policy.traffic.value != TrafficKind.DNS.value else 0.8
            redundancy = 0.60

        self._add_transport_edge(
            aggregation_node,
            node_id,
            medium=medium,
            capacity_mbps=capacity_mbps,
            latency_ms=latency_ms,
            redundancy=redundancy,
            logical_level="L3",
            physical_level="L4",
        )

    # ---------------------------------------------------------------------
    # L0, L7 and common helpers
    # ---------------------------------------------------------------------
    def _add_services(self) -> None:
        service_positions = {
            "SVC_VOICE": (-6.4, 5.5),
            "SVC_VIDEO": (-2.2, 6.3),
            "SVC_FTP": (2.2, 6.3),
            "SVC_TELEM": (6.4, 5.5),
        }
        service_to_core = {"SVC_VOICE": "C1", "SVC_VIDEO": "C5", "SVC_FTP": "C8", "SVC_TELEM": "C9"}
        profiles = {service.name.lower(): service for service in self.services}
        profile_map = {"SVC_VOICE": profiles["voice"], "SVC_VIDEO": profiles["video"], "SVC_FTP": profiles["ftp"], "SVC_TELEM": profiles["telemetry"]}

        for service_id, pos in service_positions.items():
            profile = profile_map[service_id]
            self.graph.add_node(
                service_id,
                level="L0",
                role="service",
                label=profile.name,
                pos=pos,
                visible_in_logic=True,
                color="#d8f3dc",
                tensor=build_layer_tensor("L0", l0_service_tensor(profile)),
            )
            self._add_transport_edge(
                service_id,
                service_to_core[service_id],
                medium="logical-service-binding",
                capacity_mbps=max(2_000.0, profile.bitrate_mbps * 500),
                latency_ms=max(1.0, profile.latency_ms_max / 20.0),
                redundancy=0.90,
                logical_level="L0",
                physical_level="L5",
            )

    def _annotate_core_metrics(self, core_nodes: Iterable[str]) -> None:
        """Add centrality values to core nodes after the graph is connected."""
        for node_id, score in vertex_proximity_index(self.graph, core_nodes).items():
            self.graph.nodes[node_id]["centrality"] = score

    def _add_arbitrator(self) -> None:
        """Add the L7 arbitrator node.

        It is not drawn on the detailed logic map yet, but it exists in the model
        and on the layer scheme.
        """
        self.graph.add_node(
            "ARB",
            level="L7",
            role="arbitrator",
            label="Arbiter",
            pos=(0.0, 8.2),
            visible_in_logic=False,
            color="#f4a261",
            tensor=build_layer_tensor(
                "L7",
                l7_arbitrator_tensor(),
            ),
        )

    def _attach_placement_tensors(self) -> None:
        """Attach L8 placement coordinates to all placed non-service nodes."""
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("level") == "L0" or "pos" not in attrs:
                continue
            l8_tensor = build_layer_tensor("L8", l8_placement_tensor(attrs["pos"], attrs.get("role", "")))
            attrs["l8_tensor"] = l8_tensor
            if attrs.get("level") == "L8":
                attrs["tensor"] = l8_tensor

    def _add_transport_edge(
        self,
        source: str,
        target: str,
        *,
        medium: str,
        capacity_mbps: float,
        latency_ms: float,
        redundancy: float,
        logical_level: str,
        physical_level: str,
    ) -> None:
        """Add an edge with transport metadata and an edge tensor."""
        self.graph.add_edge(
            source,
            target,
            medium=medium,
            logical_level=logical_level,
            physical_level=physical_level,
            capacity_mbps=capacity_mbps,
            latency_ms=latency_ms,
            redundancy=redundancy,
            l3_tensor=build_layer_tensor(
                "L3",
                l3_medium_tensor(
                    medium,
                    capacity_mbps,
                    self.graph.nodes[source].get("pos", (0.0, 0.0)),
                    self.graph.nodes[target].get("pos", (0.0, 0.0)),
                ),
            ),
            l4_tensor=build_layer_tensor(
                "L4",
                l4_infrastructure_tensor(
                    medium,
                    self.graph.nodes[source].get("pos", (0.0, 0.0)),
                    self.graph.nodes[target].get("pos", (0.0, 0.0)),
                ),
            ),
            tensor=build_transport_tensor(edge_transport_tensor(medium, capacity_mbps, latency_ms, redundancy)),
        )

    def _validate(self) -> None:
        """Basic sanity checks for the baseline graph."""
        l2_nodes = [node for node, attrs in self.graph.nodes(data=True) if attrs["level"] == "L2"]
        if len(l2_nodes) != L2_NODE_COUNT:
            raise ValueError(f"L2 node count is not equal to {L2_NODE_COUNT}.")
        if not nx.is_connected(self.graph.subgraph(l2_nodes)):
            raise ValueError("L2 graph must be connected.")
