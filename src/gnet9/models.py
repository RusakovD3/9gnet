"""Shared data models used by the G-Net builder, exporters and visualizer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


@dataclass
class Tensor5:
    """Compact 5th-order tensor with semantic axes and named metric slots.

    The tensor itself is a numeric numpy array. The `metric_slots` dictionary
    explains where every named metric is stored inside that array.
    """

    level: str
    axis_names: tuple[str, str, str, str, str]
    metric_names: tuple[str, ...]
    data: np.ndarray
    metric_slots: dict[str, tuple[int, int, int, int, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "axis_names": list(self.axis_names),
            "metric_names": list(self.metric_names),
            "data": self.data.tolist(),
            "metric_slots": {name: list(slot) for name, slot in self.metric_slots.items()},
        }


@dataclass
class ServiceProfile:
    """Runtime L0 service description."""

    name: str
    bitrate_mbps: float
    latency_ms_max: float
    jitter_ms_max: float
    availability_target: float
    priority: str


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
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert numpy/tensor/python objects to JSON-safe values."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Tensor5):
            return value.to_dict()
        if isinstance(value, tuple):
            return list(value)
        return value

    def _serializable_nodes(self) -> list[dict[str, Any]]:
        nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            item = {key: self._json_safe(value) for key, value in attrs.items()}
            item["id"] = node_id
            nodes.append(item)
        return nodes

    def _serializable_edges(self) -> list[dict[str, Any]]:
        edges = []
        for source, target, attrs in self.graph.edges(data=True):
            item = {key: self._json_safe(value) for key, value in attrs.items()}
            item["source"] = source
            item["target"] = target
            edges.append(item)
        return edges

    def to_serializable(self) -> dict[str, Any]:
        return {
            "services": [asdict(service) for service in self.services],
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
            value = self._json_safe(value)
            if isinstance(value, (list, dict)):
                result[key] = json.dumps(value, ensure_ascii=False)
            else:
                result[key] = value
        return result

    def export_summary(self, path: Path) -> None:
        lines = [
            "Baseline G-Net 9-level topology summary",
            "=" * 40,
            "",
            "Levels:",
        ]
        for level, count in sorted(self.level_summary.items()):
            lines.append(f"  {level}: {count}")

        lines.extend(["", "Services:"])
        for service in self.services:
            lines.append(
                f"  - {service.name}: {service.bitrate_mbps} Mbps, "
                f"latency <= {service.latency_ms_max} ms, "
                f"availability {service.availability_target:.4f}"
            )

        lines.extend(["", "Slices:"])
        for slice_profile in self.slices:
            lines.append(
                f"  - {slice_profile.name}: priority={slice_profile.priority}, "
                f"nodes={len(slice_profile.node_ids)}, reserve={slice_profile.capacity_reserve_ratio:.2f}"
            )

        lines.extend(
            [
                "",
                f"Total graph nodes: {self.graph.number_of_nodes()}",
                f"Total graph edges: {self.graph.number_of_edges()}",
                "",
                "Notes:",
            ]
        )
        lines.extend(f"  - {note}" for note in self.notes)
        path.write_text("\n".join(lines), encoding="utf-8")
