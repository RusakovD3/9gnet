"""Shared data models used by the G-Net builder, exporters and visualizer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .constants import SERVICE_DISPLAY_NAMES


@dataclass
class StateTensor:
    """Numeric state vector with explicit metric semantics."""

    level: str
    metric_names: tuple[str, ...]
    data: np.ndarray
    axes: tuple[str, ...]
    metric_index: dict[str, tuple[int, ...]]
    units: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "axes": list(self.axes),
            "metric_names": list(self.metric_names),
            "shape": list(self.data.shape),
            "data": self.data.tolist(),
            "metric_index": {name: list(index) for name, index in self.metric_index.items()},
            "units": self.units,
            "description": self.description,
        }


@dataclass
class CodecComponent:
    """One media or application payload component used by an L0 service."""

    media: str
    codec: str
    profile: str
    payload_or_container: str
    bitrate_note: str
    clock_rate_hz: int | None = None
    channels: int | None = None
    packetization_ms: int | None = None
    frame_rate_fps: float | None = None


@dataclass
class CodecProfile:
    """Human-readable codec/protocol profile attached to L0 and L1."""

    profile_id: str
    display_name: str
    transport_stack: str
    application_protocol: str
    components: tuple[CodecComponent, ...]
    realism_note: str


@dataclass
class ServiceProfile:
    """Runtime L0 service description."""

    name: str
    bitrate_mbps: float
    latency_ms_max: float
    jitter_ms_max: float
    availability_target: float
    priority: str
    service_id: str = ""
    server_id: str = ""
    category: str = ""
    platform: str = ""
    audio_codec: str | None = None
    video_codec: str | None = None
    critical_latency_ms: float | None = None
    codec_profile_id: str = ""


@dataclass(frozen=True)
class ServerProfile:
    """Паспорт и выбранная конфигурация физического сервера L0."""

    server_id: str
    model: str
    cpu: str
    cpu_sockets: int
    total_cores: int
    ram_gb: int
    storage_tb: float
    network_ports_gbps: tuple[int, ...]
    hosted_service_ids: tuple[str, ...]
    source_url: str


@dataclass(frozen=True)
class ServerRuntimeMetrics:
    """Измеряемое состояние сервера, отделённое от паспортной конфигурации."""

    cpu_util_percent: float
    ram_util_percent: float
    network_util_percent: float
    storage_util_percent: float
    temperature_c: float
    active_sessions: int
    health: float


@dataclass
class SliceProfile:
    """Logical L5 slice: a group of core nodes with a priority and capacity reserve."""

    name: str
    priority: str
    node_ids: list[str]
    capacity_reserve_ratio: float


@dataclass
class NetworkModel:
    """Full generated G-Net model.

    `graph` is the main object. All nodes and edges are stored there together
    with tensors and metadata. The remaining fields are export-friendly summaries.
    """

    graph: nx.Graph
    services: list[ServiceProfile]
    slices: list[SliceProfile]
    level_summary: dict[str, int]
    servers: list[ServerProfile] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def _to_json_value(value: Any) -> Any:
        """Convert numpy/tensor/python objects to JSON-safe values."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, StateTensor):
            return value.to_dict()
        if isinstance(value, tuple):
            return list(value)
        return value

    def _serializable_nodes(self) -> list[dict[str, Any]]:
        nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            item = {key: self._to_json_value(value) for key, value in attrs.items()}
            item["id"] = node_id
            nodes.append(item)
        return nodes

    def _serializable_edges(self) -> list[dict[str, Any]]:
        edges = []
        for source, target, attrs in self.graph.edges(data=True):
            item = {key: self._to_json_value(value) for key, value in attrs.items()}
            item["source"] = source
            item["target"] = target
            edges.append(item)
        return edges

    def to_serializable(self) -> dict[str, Any]:
        return {
            "services": [asdict(service) for service in self.services],
            "servers": [asdict(server) for server in self.servers],
            "slices": [asdict(slice_profile) for slice_profile in self.slices],
            "level_summary": self.level_summary,
            "notes": self.notes,
            "nodes": self._serializable_nodes(),
            "edges": self._serializable_edges(),
        }

    def export_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_serializable(), ensure_ascii=False, indent=2), encoding="utf-8")

    def export_graphml(self, path: Path) -> None:
        """Export graph to GraphML.

        GraphML supports only scalar attributes. Lists, dictionaries and tensors
        are therefore serialized to JSON strings before export.
        """
        graph_copy = nx.Graph()

        for node_id, attrs in self.graph.nodes(data=True):
            graph_copy.add_node(node_id, **self._format_attrs_for_export(attrs))

        for source, target, attrs in self.graph.edges(data=True):
            graph_copy.add_edge(source, target, **self._format_attrs_for_export(attrs))

        nx.write_graphml(graph_copy, path)

    def _format_attrs_for_export(self, attrs: dict[str, Any]) -> dict[str, Any]:
        """Format attributes for GraphML or other exports, converting complex types to JSON strings."""
        result: dict[str, Any] = {}
        for key, value in attrs.items():
            value = self._to_json_value(value)
            if value is None:
                result[key] = ""
            elif isinstance(value, (list, dict)):
                result[key] = json.dumps(value, ensure_ascii=False)
            else:
                result[key] = value
        return result

    def export_summary(self, path: Path) -> None:
        lines = [
            "Сводка эталонной девятиуровневой топологии G-Net",
            "=" * 40,
            "",
            "Уровни:",
        ]
        for level, count in sorted(self.level_summary.items()):
            lines.append(f"  {level}: {count}")

        lines.extend(["", "Сервисы:"])
        for service in self.services:
            display_name = SERVICE_DISPLAY_NAMES.get(service.name, service.name)
            codecs = " + ".join(item for item in (service.audio_codec, service.video_codec) if item) or "протокольный профиль без медиакодека"
            lines.append(
                f"  - {display_name}: {service.bitrate_mbps} Мбит/с, "
                f"задержка <= {service.latency_ms_max} мс, "
                f"доступность {service.availability_target:.4f}, "
                f"кодеки/профиль: {codecs}"
            )

        lines.extend(["", "Физические серверы сервисов:"])
        for server in self.servers:
            ports = "+".join(f"{speed}G" for speed in server.network_ports_gbps)
            lines.append(
                f"  - {server.server_id}: {server.model}, {server.total_cores} ядер, "
                f"ОЗУ {server.ram_gb} ГБ, хранилище {server.storage_tb:.2f} ТБ, сеть {ports}; "
                f"сервисы: {', '.join(server.hosted_service_ids)}"
            )

        slice_names = {
            "GoldBackbone": "Золотой магистральный",
            "SilverEnterprise": "Серебряный корпоративный",
            "BronzeBestEffort": "Бронзовый без гарантий",
        }
        priority_names = {"gold": "золотой", "silver": "серебряный", "bronze": "бронзовый"}
        lines.extend(["", "Сетевые срезы:"])
        for slice_profile in self.slices:
            slice_name = slice_names.get(slice_profile.name, slice_profile.name)
            priority = priority_names.get(slice_profile.priority, slice_profile.priority)
            lines.append(
                f"  - {slice_name}: приоритет={priority}, "
                f"узлов={len(slice_profile.node_ids)}, резерв={slice_profile.capacity_reserve_ratio:.2f}"
            )

        lines.extend(
            [
                "",
                f"Всего узлов графа: {self.graph.number_of_nodes()}",
                f"Всего связей графа: {self.graph.number_of_edges()}",
                "",
                "Примечания:",
            ]
        )
        lines.extend(f"  - {note}" for note in self.notes)
        path.write_text("\n".join(lines), encoding="utf-8")
