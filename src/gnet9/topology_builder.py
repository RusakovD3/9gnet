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

from .attacks import SLA_RESTORATION_PRIORITY, mark_critical_nodes
from .constants import (
    ACCESS_DEVICE_ROLES,
    AGGREGATION_FIXED,
    AGGREGATION_MOBILE,
    CRITICALITY_COLORS,
    FIXED_SUBSCRIBERS_PER_AGG,
    FIXED_ACCESS_BY_AGG,
    L2_NODE_COUNT,
    L1_MONITORING_SECONDS,
    MOBILE_ACCESS_BY_AGG,
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
from .addressing import attach_network_identities
from .models import NetworkModel, SliceProfile
from .service_catalog import (
    CODEC_CATALOG,
    SERVER_BASELINE_RUNTIME,
    SERVER_CATALOG,
    SERVICE_CATALOG,
    TRAFFIC_CODEC_PROFILE,
    codec_summary,
)
from .tensors import build_layer_tensor, build_transport_tensor
from .l2_equipment import (
    build_l2_raw_baseline,
    build_l2_summary_metrics,
    l2_profile_for_role,
)


IMT2020_ACCESS_REFERENCE = (
    "https://www.itu.int/en/ITU-R/study-groups/rsg5/rwp5d/imt-2020/"
    "Documents/S01-1_Requirements%20for%20IMT-2020_Rev.pdf"
)
XGS_PON_ACCESS_REFERENCE = "https://www.itu.int/rec/T-REC-G.9807.1/en"


# Fault domains are intentionally scoped.  ``UPS_A`` by itself looked global
# and made it impossible to tell whether a power event should affect one rack
# or the whole topology.  A site identifier is retained separately so a future
# common-site failure can still be propagated explicitly.
SERVER_LOCAL_POWER_DOMAINS = {
    "SRV_MEDIA": ("DC_MAIN", "RACK_MEDIA"),
    "SRV_DATA": ("DC_MAIN", "RACK_DATA"),
    "SRV_RTC": ("DC_MAIN", "RACK_RTC"),
    "SRV_LIVE": ("DC_MAIN", "RACK_LIVE"),
}


def _local_power_path(node_id: str, site_id: str, local_domain: str | None = None) -> dict[str, object]:
    """Return two node-local A/B paths with unambiguous fault-domain IDs."""
    fault_domain_id = f"{site_id}/{local_domain or node_id}"
    return {
        "site_power_domain_id": site_id,
        "local_fault_domain_id": fault_domain_id,
        "fault_domain_scope": "node_local_dual_power_path",
        "feed_domains": [f"{fault_domain_id}/FEED_A", f"{fault_domain_id}/FEED_B"],
        "ups_domains": [f"{fault_domain_id}/UPS_A", f"{fault_domain_id}/UPS_B"],
        "pdu_domains": [f"{fault_domain_id}/PDU_A", f"{fault_domain_id}/PDU_B"],
    }


class GNetBaselineBuilder:
    """Build a readable and future-ready baseline topology.

    Important modeling decision: L2 contains active network equipment from the
    9-level model: core routers, aggregation switches and explicit access
    devices.  L1 edges are still logical subscriber attachments, so individual
    subscribers do not consume C9500 front-panel ports one by one.
    """

    def __init__(self, d0sl_policy_path: Path | None = None) -> None:
        self.graph = nx.Graph()
        self.services = list(SERVICE_CATALOG)
        self.servers = list(SERVER_CATALOG)
        self.slices: list[SliceProfile] = []

        project_root = Path(__file__).resolve().parents[2]
        self.d0sl_policy_path = d0sl_policy_path or project_root / "policies" / "l1_policies.d0sl"
        try:
            self.d0sl_policy_display_path = self.d0sl_policy_path.resolve().relative_to(
                project_root.resolve()
            ).as_posix()
        except ValueError:
            # Для внешнего пользовательского каталога абсолютный путь остаётся
            # полезной provenance-записью; штатный проектный путь переносим.
            self.d0sl_policy_display_path = str(self.d0sl_policy_path)
        self.l1_policy_catalog = load_l1_d0sl_catalog(self.d0sl_policy_path)

    def build(self) -> NetworkModel:
        """Generate the full graph and return it as a NetworkModel."""
        self._add_terrain_anchors()
        core_nodes = self._add_core_routers()
        self._add_aggregation_routers()
        self._connect_core_to_aggregation()
        self._add_access_equipment()
        self._add_subscribers()
        self._add_services()
        self._annotate_core_metrics(core_nodes)
        self._add_arbitrator()
        self._attach_placement_tensors()
        self._validate()

        level_summary = Counter(attrs["level"] for _, attrs in self.graph.nodes(data=True))
        model = NetworkModel(
            graph=self.graph,
            services=self.services,
            slices=self.slices,
            level_summary=dict(level_summary),
            servers=self.servers,
            notes=self._build_notes(),
        )
        attach_network_identities(model)
        mark_critical_nodes(model)
        return model

    def _build_notes(self) -> list[str]:
        return [
            "Топология создана как идеальное эталонное состояние t0 для будущих экспериментов Koopman/Lyapunov.",
            "L2 содержит узлы ядра, агрегации и явный access-слой RAN*/OLT* девятиуровневой модели.",
            "Тензоры уровней — числовые векторы состояния с явными именами и единицами метрик.",
            "Рёбра L1 являются логическими subscriber-привязками к RAN*/OLT*: отдельные абоненты не расходуют физические порты C9500.",
            "Абоненты L1 используют исполняемые политики d0sl SLA/SLO/SLI из "
            f"{self.d0sl_policy_display_path}.",
            "Классы трафика L1: голос Opus, VLC-аудио/видео, FTP, DNS, видеоконференция и прямая трансляция.",
            "Кодеки и протокольные профили L0 описаны единым каталогом CodecProfile и привязаны к сервисам, абонентам и потокам.",
            "Каждый узел имеет детерминированный private IPv4 и локально-администрируемый MAC-адрес для безопасной пакетной модели.",
            "Логические сервисы L0 размещены на отдельных физических серверах Dell PowerEdge R660.",
            "Каждый сервис имеет три standby-реплики на остальных R660; active/standby переключается временным routing overlay.",
            "Каждый агрегирующий коммутатор имеет три uplink к ядру; связность backbone L2 равна 3, степень backbone не выше 6.",
            "У каждого абонента одна линия последней мили; резервирование предусмотрено в L2 и на сервисных серверах.",
            "Тензоры L3/L4 закреплены за транспортными связями; L5/L6 — за оборудованием и абонентами, где это применимо.",
            "Тензор L7 хранит эталонные признаки решения Koopman/Lyapunov/Hausdorff.",
            "Тензоры L8 хранят координаты размещения; Хаусдорф Gold-маршрутов считается отдельно в метрике задержки графа.",
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
            port_speed_mbps=100_000.0,
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
                capacity_mbps=100_000.0,
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
                capacity_mbps=100_000.0,
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
        """Add six aggregation switches."""
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
                role="aggregation-switch",
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
        profile = l2_profile_for_role(role)
        raw_l2 = build_l2_raw_baseline(profile, role=role, criticality=criticality)
        summary_l2 = build_l2_summary_metrics(raw_l2, profile)
        l6_values = l6_power_tensor(role)
        if profile.typical_power_w is not None:
            modeled_power_origin = "vendor_typical_output_power_at_25_celsius"
        elif profile.thermal_output_equivalent_w is not None:
            modeled_power_origin = "gnet9_t0_assumption_below_vendor_thermal_upper_bound"
        else:
            modeled_power_origin = "gnet9_t0_scenario_assumption"
        site_power_domain_id = str(extra_attrs.get("power_zone") or f"PWR_ACCESS_{node_id}")
        power_architecture = {
            "feed_count": 2,
            "psu_redundancy": "1+1 hot-swappable",
            "single_psu_failure_service_interruption": False,
            "modeled_power_attack_scope": "explicit_local_fault_domain_only",
            "vendor_typical_output_power_w": profile.typical_power_w,
            "vendor_max_output_power_w": profile.max_power_w,
            "vendor_thermal_output_btu_per_hour": profile.thermal_output_btu_per_hour,
            "vendor_thermal_output_equivalent_w": profile.thermal_output_equivalent_w,
            "vendor_power_value_semantics": profile.power_value_semantics,
            "modeled_t0_power_w": round(l6_values["nominal_power_kw"] * 1_000.0, 3),
            "modeled_t0_power_origin": modeled_power_origin,
            "ups_backup_autonomy_hours": l6_values["backup_autonomy_hours"],
            "backup_autonomy_origin": "gnet9_site_scenario_assumption_not_vendor_spec",
            "value_origin": "vendor_redundancy_plus_gnet9_site_assumption",
            **_local_power_path(node_id, site_power_domain_id),
        }

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
            power_architecture=power_architecture,
            **summary_l2,
            **extra_attrs,
            tensor=build_layer_tensor("L2", l2_equipment_tensor(role, criticality, port_speed_mbps, port_delay_ms)),
            l5_tensor=build_layer_tensor("L5", l5_role_tensor(role)),
            l6_tensor=build_layer_tensor("L6", l6_values),
        )

    def _connect_core_to_aggregation(self) -> None:
        """Подключить агрегацию к трём независимым узлам ядра.

        Третий uplink повышает связность инфраструктурного L2-графа до трёх,
        при этом степень каждого транзитного узла остаётся не выше шести.
        Абонентские access-порты в эту степень намеренно не включаются.
        """
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
            self.graph.edges[primary, agg]["uplink_role"] = "primary"
            self.graph.edges[secondary, agg]["uplink_role"] = "secondary"

        tertiary_core = {
            "A1": "C8",
            "A2": "C10",
            "A3": "C12",
            "A4": "C2",
            "A5": "C4",
            "A6": "C6",
        }
        for agg, core in tertiary_core.items():
            self._add_transport_edge(
                core,
                agg,
                medium="fiber",
                capacity_mbps=40_000.0,
                latency_ms=5.6,
                redundancy=0.78,
                logical_level="L4",
                physical_level="L4",
            )
            self.graph.edges[core, agg]["uplink_role"] = "tertiary"

    def _add_access_equipment(self) -> None:
        """Add explicit active access devices below aggregation."""
        mobile_offsets = (-0.55, 0.55)
        fixed_offsets = (-0.50, 0.50)
        for aggregation_node, access_nodes in MOBILE_ACCESS_BY_AGG.items():
            agg_x, agg_y = self.graph.nodes[aggregation_node]["pos"]
            secondary_aggregation = self._next_aggregation(aggregation_node, AGGREGATION_MOBILE)
            for index, node_id in enumerate(access_nodes):
                self._add_l2_router(
                    node_id,
                    role="radio-access-node",
                    pos=(agg_x + mobile_offsets[index], agg_y - 1.05),
                    criticality="silver",
                    port_speed_mbps=25_000.0,
                    port_delay_ms=0.32,
                    home_aggregation=aggregation_node,
                    secondary_aggregation=secondary_aggregation,
                    access_index=index + 1,
                    access_domain="mobile",
                    power_zone=f"PWR_RAN_{aggregation_node}",
                )
                self._connect_access_to_aggregation(
                    node_id,
                    primary_aggregation=aggregation_node,
                    secondary_aggregation=secondary_aggregation,
                    medium="radio-backhaul",
                    primary_capacity_mbps=25_000.0,
                    secondary_capacity_mbps=10_000.0,
                    primary_latency_ms=1.6,
                    secondary_latency_ms=2.4,
                )

        for aggregation_node, access_nodes in FIXED_ACCESS_BY_AGG.items():
            agg_x, agg_y = self.graph.nodes[aggregation_node]["pos"]
            secondary_aggregation = self._next_aggregation(aggregation_node, AGGREGATION_FIXED)
            for index, node_id in enumerate(access_nodes):
                self._add_l2_router(
                    node_id,
                    role="optical-line-terminal",
                    pos=(agg_x + fixed_offsets[index], agg_y - 1.05),
                    criticality="silver",
                    port_speed_mbps=25_000.0,
                    port_delay_ms=0.26,
                    home_aggregation=aggregation_node,
                    secondary_aggregation=secondary_aggregation,
                    access_index=index + 1,
                    access_domain="fixed",
                    power_zone=f"PWR_OLT_{aggregation_node}",
                )
                self._connect_access_to_aggregation(
                    node_id,
                    primary_aggregation=aggregation_node,
                    secondary_aggregation=secondary_aggregation,
                    medium="fiber",
                    primary_capacity_mbps=25_000.0,
                    secondary_capacity_mbps=10_000.0,
                    primary_latency_ms=0.7,
                    secondary_latency_ms=1.2,
                )

    def _connect_access_to_aggregation(
        self,
        access_node: str,
        *,
        primary_aggregation: str,
        secondary_aggregation: str,
        medium: str,
        primary_capacity_mbps: float,
        secondary_capacity_mbps: float,
        primary_latency_ms: float,
        secondary_latency_ms: float,
    ) -> None:
        for aggregation_node, access_role, capacity_mbps, latency_ms, redundancy in (
            (primary_aggregation, "primary", primary_capacity_mbps, primary_latency_ms, 0.88),
            (secondary_aggregation, "secondary", secondary_capacity_mbps, secondary_latency_ms, 0.76),
        ):
            self._add_transport_edge(
                access_node,
                aggregation_node,
                medium=medium,
                capacity_mbps=capacity_mbps,
                latency_ms=latency_ms,
                redundancy=redundancy,
                logical_level="L3",
                physical_level="L4",
            )
            self.graph.edges[access_node, aggregation_node].update(
                uplink_role=access_role,
                access_backhaul=True,
                access_node=access_node,
                aggregation_node=aggregation_node,
                consumes_c9500_physical_port=True,
                value_origin="gnet9_engineering_access_backhaul_profile",
            )

    @staticmethod
    def _next_aggregation(aggregation_node: str, ordered_nodes: tuple[str, ...]) -> str:
        index = ordered_nodes.index(aggregation_node)
        return ordered_nodes[(index + 1) % len(ordered_nodes)]

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
                access_nodes=MOBILE_ACCESS_BY_AGG[aggregation_node],
                group_index=group_index,
                angles_deg=mobile_spread,
                grade_fn=lambda index: "gold" if index <= 16 else "bronze",
                traffic_shift=group_index - 1,
                medium="radio",
                color="#b7e4c7",
                visible_limit=18,
                seed_base=1000,
            )

        for group_index, aggregation_node in enumerate(AGGREGATION_FIXED, start=1):
            self._add_subscriber_group(
                prefix="F",
                role="fixed-subscriber",
                label="PC",
                access_nodes=FIXED_ACCESS_BY_AGG[aggregation_node],
                group_index=group_index,
                angles_deg=fixed_spread,
                grade_fn=lambda index: "silver" if index <= 20 else "bronze",
                traffic_shift=group_index + 2,
                medium="ethernet",
                color="#95d5b2",
                visible_limit=14,
                seed_base=2000,
            )

    def _add_subscriber_group(
        self,
        *,
        prefix: str,
        role: str,
        label: str,
        access_nodes: tuple[str, ...],
        group_index: int,
        angles_deg: np.ndarray,
        grade_fn,
        traffic_shift: int,
        medium: str,
        color: str,
        visible_limit: int,
        seed_base: int,
    ) -> None:
        traffic_cycle = [
            TrafficKind.VOICE.value,
            TrafficKind.VLC_AV.value,
            TrafficKind.FTP.value,
            TrafficKind.DNS.value,
            TrafficKind.VIDEO_CONFERENCE.value,
            TrafficKind.LIVE_STREAMING.value,
        ]
        subscribers_per_access = max(1, math.ceil(len(angles_deg) / max(len(access_nodes), 1)))

        for subscriber_index, angle_deg in enumerate(angles_deg, start=1):
            node_id = f"{prefix}{group_index}_{subscriber_index:02d}"
            access_index = min((subscriber_index - 1) // subscribers_per_access, len(access_nodes) - 1)
            access_node = access_nodes[access_index]
            x0, y0 = self.graph.nodes[access_node]["pos"]
            traffic = traffic_cycle[(subscriber_index + traffic_shift) % len(traffic_cycle)]
            policy = self.l1_policy_catalog.get(grade_fn(subscriber_index), traffic)
            codec_profile_id = TRAFFIC_CODEC_PROFILE[policy.traffic.value]
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
                home_access=access_node,
                parent_aggregation=self.graph.nodes[access_node].get("home_aggregation"),
                sla_grade=policy.grade.value,
                recovery_priority=SLA_RESTORATION_PRIORITY[policy.grade.value],
                recovery_policy="gold_then_silver_then_bronze",
                traffic_kind=policy.traffic.value,
                codec=policy.codec,
                codec_profile_id=codec_profile_id,
                codec_profile_name=CODEC_CATALOG[codec_profile_id].display_name,
                codec_summary=codec_summary(codec_profile_id),
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

            self._connect_subscriber(access_node, node_id, policy, medium)

    @staticmethod
    def _subscriber_position(x0: float, y0: float, angle_deg: float, subscriber_index: int, prefix: str) -> tuple[float, float]:
        angle = np.deg2rad(angle_deg)
        if prefix == "M":
            radius = 1.7 + 0.17 * (subscriber_index % 3)
            return (x0 + radius * np.cos(angle), y0 - 1.35 + radius * np.sin(angle) * 0.55)

        radius = 1.5 + 0.13 * (subscriber_index % 4)
        return (x0 + radius * np.cos(angle), y0 - 1.2 + radius * np.sin(angle) * 0.52)

    @staticmethod
    def _subscriber_access_capacity_mbps(role: str, grade: str) -> float:
        """Return an access tariff independent of the selected application.

        These are reproducible GNet9 scenario profiles, not claims about an
        operator subscriber distribution.  Fixed access represents a logical
        service over an abstracted shared XGS-PON domain; mobile access is a
        provisioned bearer behind an abstracted gNodeB/UPF path.
        """
        if role == "mobile-subscriber":
            return {"gold": 100.0, "silver": 75.0, "bronze": 50.0}[grade]
        if role == "fixed-subscriber":
            return {"gold": 1_000.0, "silver": 500.0, "bronze": 100.0}[grade]
        raise ValueError(f"Неизвестная роль доступа: {role}")

    def _connect_subscriber(self, access_node: str, node_id: str, policy, medium: str) -> None:
        role = str(self.graph.nodes[node_id]["role"])
        grade = str(policy.grade.value)
        capacity_mbps = self._subscriber_access_capacity_mbps(role, grade)
        if medium == "radio":
            latency_ms = 8.0
            redundancy = 0.42
            access_profile = "5G provisioned bearer via explicit RAN/UPF access node"
            attachment = "logical_5g_bearer_via_ran_upf_access_node"
            architecture_reference = IMT2020_ACCESS_REFERENCE
        else:
            latency_ms = 1.2
            redundancy = 0.60
            access_profile = "XGS-PON service profile via explicit OLT access node"
            attachment = "logical_fixed_access_via_olt_access_node"
            architecture_reference = XGS_PON_ACCESS_REFERENCE

        self.graph.nodes[node_id].update(
            access_profile=access_profile,
            access_capacity_mbps=capacity_mbps,
            access_latency_ms=latency_ms,
            access_value_origin="gnet9_engineering_access_profile",
            access_architecture_reference_url=architecture_reference,
        )

        self._add_transport_edge(
            access_node,
            node_id,
            medium=medium,
            capacity_mbps=capacity_mbps,
            latency_ms=latency_ms,
            redundancy=redundancy,
            logical_level="L3",
            physical_level="L4",
        )
        self.graph.edges[access_node, node_id].update(
            access_role="primary",
            standby=False,
            visible_in_logic=True,
            attachment_semantics=attachment,
            consumes_c9500_physical_port=False,
            consumes_access_physical_port=False,
            value_origin="gnet9_engineering_access_profile",
            access_architecture_reference_url=architecture_reference,
        )

    # ---------------------------------------------------------------------
    # L0, L7 and common helpers
    # ---------------------------------------------------------------------
    def _add_services(self) -> None:
        server_positions = {
            "SRV_MEDIA": (-6.0, 5.0),
            "SRV_DATA": (-2.0, 5.0),
            "SRV_RTC": (2.0, 5.0),
            "SRV_LIVE": (6.0, 5.0),
        }
        server_to_core = {
            "SRV_MEDIA": ("C1", "C4"),
            "SRV_DATA": ("C5", "C8"),
            "SRV_RTC": ("C9", "C12"),
            "SRV_LIVE": ("C10", "C11"),
        }
        service_positions = {
            "SVC_VOICE": (-7.0, 7.7),
            "SVC_VLC": (-5.0, 7.7),
            "SVC_FTP": (-3.0, 7.7),
            "SVC_DNS": (-1.0, 7.7),
            "SVC_TELEMOST": (2.0, 7.7),
            "SVC_LIVE": (6.0, 7.7),
        }
        # Контейнерные standby-реплики каждого сервиса размещаются на трёх
        # остальных R660. Два сетевых порта защищают только линию, тогда как
        # N+3 placement сохраняет Gold при перекрывающихся отказах primary и
        # одного/двух standby в ускоренном сценарии. Серверы находятся в
        # независимых локальных power domains, а capacity проверяется remapper.
        standby_hosts = {
            "SVC_VOICE": ("SRV_RTC", "SRV_LIVE", "SRV_DATA"),
            "SVC_VLC": ("SRV_LIVE", "SRV_RTC", "SRV_DATA"),
            "SVC_FTP": ("SRV_LIVE", "SRV_RTC", "SRV_MEDIA"),
            "SVC_DNS": ("SRV_LIVE", "SRV_RTC", "SRV_MEDIA"),
            "SVC_TELEMOST": ("SRV_MEDIA", "SRV_LIVE", "SRV_DATA"),
            "SVC_LIVE": ("SRV_RTC", "SRV_MEDIA", "SRV_DATA"),
        }

        for server in self.servers:
            l6_values = l6_power_tensor("service-server")
            server_site, server_rack = SERVER_LOCAL_POWER_DOMAINS[server.server_id]
            self.graph.add_node(
                server.server_id,
                level="L0",
                role="service-server",
                label=server.model,
                pos=server_positions[server.server_id],
                visible_in_logic=True,
                color="#a78bfa",
                server_profile=asdict(server),
                hosted_services=list(server.hosted_service_ids),
                runtime=asdict(SERVER_BASELINE_RUNTIME[server.server_id]),
                power_architecture={
                    "feed_count": 2,
                    "psu_count": server.psu_count,
                    "psu_redundancy": "1+1 hot-swappable",
                    "psu_rating_w_each": server.psu_rating_w_each,
                    "psu_efficiency_class": server.psu_efficiency_class,
                    "psu_hot_swappable": server.psu_hot_swappable,
                    "psu_source_url": server.psu_source_url,
                    "psu_rating_semantics": server.psu_rating_semantics,
                    "redundant_usable_capacity_w": server.psu_rating_w_each,
                    "redundant_capacity_semantics": "one_psu_must_carry_selected_configuration_in_1_plus_1_mode",
                    "single_psu_failure_service_interruption": False,
                    "modeled_power_attack_scope": "explicit_local_fault_domain_only",
                    "modeled_t0_power_w": round(l6_values["nominal_power_kw"] * 1_000.0, 3),
                    "modeled_t0_power_origin": "gnet9_selected_configuration_input_draw_assumption",
                    "ups_backup_autonomy_hours": l6_values["backup_autonomy_hours"],
                    "backup_autonomy_origin": "gnet9_datacenter_ups_scenario_assumption_not_dell_spec",
                    "value_origin": "Dell_supported_PSU_configuration_plus_gnet9_draw_and_UPS_assumptions",
                    **_local_power_path(server.server_id, server_site, server_rack),
                },
                l6_tensor=build_layer_tensor("L6", l6_values),
            )
            for core_id in server_to_core[server.server_id]:
                self._add_transport_edge(
                    server.server_id,
                    core_id,
                    medium="fiber",
                    capacity_mbps=10_000.0,
                    latency_ms=0.35,
                    redundancy=0.95,
                    logical_level="L0",
                    physical_level="L4",
                )
                self.graph.edges[server.server_id, core_id].update(
                    physical_profile="Dell R660 10GbE OCP to Cisco NCS-5501 10GbE port",
                    consumes_server_nic=True,
                    consumes_ncs_10g_port=True,
                    value_origin="vendor_compatible_selected_configuration",
                )

        for profile in self.services:
            service_id = profile.service_id
            codec_profile = CODEC_CATALOG[profile.codec_profile_id]
            self.graph.add_node(
                service_id,
                level="L0",
                role="service",
                label=profile.name,
                pos=service_positions[service_id],
                visible_in_logic=True,
                color="#d8f3dc",
                tensor=build_layer_tensor("L0", l0_service_tensor(profile)),
                hosted_on=profile.server_id,
                primary_server_id=profile.server_id,
                active_server_id=profile.server_id,
                standby_hosts=list(standby_hosts[service_id]),
                failover_policy="gold_first_active_standby",
                category=profile.category,
                platform=profile.platform,
                audio_codec=profile.audio_codec,
                video_codec=profile.video_codec,
                critical_latency_ms=profile.critical_latency_ms,
                codec_profile_id=profile.codec_profile_id,
                codec_profile_name=codec_profile.display_name,
                codec_summary=codec_summary(profile.codec_profile_id),
                codec_profile=asdict(codec_profile),
                service_profile=asdict(profile),
            )
            self._add_transport_edge(
                service_id,
                profile.server_id,
                medium="logical-service-binding",
                capacity_mbps=100_000.0,
                latency_ms=0.05,
                redundancy=0.99,
                logical_level="L0",
                physical_level="L5",
            )

        for service_id, backups in standby_hosts.items():
            for backup in backups:
                self.graph.nodes[backup].setdefault("standby_services", []).append(service_id)
                self.graph.nodes[backup].setdefault("cluster_roles", {})[service_id] = "standby"

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
        infrastructure = self.graph.subgraph(l2_nodes)
        backbone_nodes = [
            node
            for node in l2_nodes
            if self.graph.nodes[node].get("role") in {"core-router", "aggregation-switch"}
        ]
        backbone = self.graph.subgraph(backbone_nodes)
        if max(dict(backbone.degree()).values(), default=0) > 6:
            raise ValueError("Backbone L2 degree must remain <= 6.")
        infrastructure_connectivity = nx.node_connectivity(backbone)
        if infrastructure_connectivity < 3:
            raise ValueError("L2 backbone must tolerate any two independent node cuts.")
        for node_id in l2_nodes:
            role = self.graph.nodes[node_id].get("role")
            physical_high_speed_edges = [
                (node_id, neighbor)
                for neighbor in infrastructure.neighbors(node_id)
                if float(infrastructure.edges[node_id, neighbor].get("capacity_mbps", 0.0)) >= 40_000.0
            ]
            port_budget = 6 if role == "core-router" else 4
            if len(physical_high_speed_edges) > port_budget:
                raise ValueError(
                    f"{node_id} exceeds its physical >=40G port budget: "
                    f"{len(physical_high_speed_edges)} > {port_budget}."
                )
            if role in ACCESS_DEVICE_ROLES:
                aggregation_links = [
                    neighbor
                    for neighbor in infrastructure.neighbors(node_id)
                    if self.graph.nodes[neighbor].get("role") == "aggregation-switch"
                ]
                if len(aggregation_links) < 2:
                    raise ValueError(f"{node_id} must have at least two aggregation uplinks.")
        for node_id in l2_nodes:
            self.graph.nodes[node_id]["infrastructure_degree"] = infrastructure.degree(node_id)
            self.graph.nodes[node_id]["backbone_node_connectivity"] = infrastructure_connectivity
            self.graph.nodes[node_id]["infrastructure_node_connectivity"] = infrastructure_connectivity
            self.graph.nodes[node_id]["physical_high_speed_ports_used"] = sum(
                1
                for neighbor in infrastructure.neighbors(node_id)
                if float(infrastructure.edges[node_id, neighbor].get("capacity_mbps", 0.0)) >= 40_000.0
            )
