"""Visualization helpers for the generated G-Net model.

The visualizer does not change the model. It only reads graph nodes/edges and
exports two PNG images:
- network_logic.png: detailed topology view;
- layer_scheme.png: abstract 9-level G-Net layer scheme.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

from .constants import LEVEL_COLORS

# Visualization constants
MEDIUM_COLORS = {
    "fiber": "#355070",
    "radio": "#2a9d8f",
    "ethernet": "#3a86ff",
    "logical-service-binding": "#588157",
}

EDGE_VISUAL_STYLES = {
    "fiber": {"line_color": "#cfcfcf", "line_width": 4.8, "alpha": 0.35},
    "radio": {"line_color": "#d8efe6", "line_width": 2.8, "alpha": 0.22},
}

NODE_SIZES = {"L0": 2600, "L2": 1200, "L1": 220, "default": 900}

FONTSIZE_CONFIGS = {
    "label_l1": 7.6,
    "label_default": 9.0,
    "badge": 9.4,
    "panel": 9.7,
    "legend": 10.5,
}


class GNetVisualizer:
    def __init__(self, model) -> None:
        self.model = model
        self.graph = model.graph

    def draw_network_logic(self, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(24, 14))
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 15)
        ax.axis("off")

        pos = self._logic_canvas_positions()

        self._draw_logic_topobase(ax, pos)
        self._draw_logic_l6_zones(ax)
        self._draw_logic_l5_rings(ax)
        self._draw_logic_edges(ax, pos)
        self._draw_logic_nodes(ax, pos)
        self._draw_logic_labels(ax, pos)
        self._draw_logic_minimal_panels(ax)

        ax.set_title("Сеть G-Net", fontsize=19, fontweight="bold", pad=12)

        fig.tight_layout()
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    def draw_layer_scheme(self, path: Path) -> None:
        """
        Figure 2.
        Circular scheme of all 9 G-Net layers.

        Important visual rules:
        - arrows stop at the border of level circles and do not go through them;
        - small inner circles are shown only for levels that contain explicit graph nodes;
        - L3/L4/L5/L6/L7 are semantic/control levels here, so no fake inner nodes are drawn.
        """
        fig, ax = plt.subplots(figsize=(20, 13))
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 13)
        ax.axis("off")

        positions = {
            "L7": (10.0, 11.1),
            "L8": (3.6, 9.4),
            "L6": (16.2, 9.2),
            "L4": (6.4, 7.0),
            "L5": (13.7, 7.0),
            "L2": (5.7, 4.0),
            "L3": (10.0, 5.1),
            "L1": (14.2, 4.0),
            "L0": (10.0, 1.5),
        }

        radius = {
            "L7": 0.88,
            "L8": 0.88,
            "L6": 0.88,
            "L5": 0.88,
            "L4": 0.88,
            "L3": 0.88,
            "L2": 1.18,
            "L1": 1.02,
            "L0": 0.90,
        }

        labels = {
            "L8": "L8\nТопооснова",
            "L7": "L7\nАрбитр",
            "L6": "L6\nЭлектропитание",
            "L5": "L5\nЯдро",
            "L4": "L4\nЛинии",
            "L3": "L3\nСреда",
            "L2": "L2\nАктивное\nоборудование",
            "L1": "L1\nАбоненты",
            "L0": "L0\nСервисы",
        }

        inclusion_edges = [
            ("L0", "L1", 0.18),
            ("L1", "L2", -0.10),
            ("L2", "L3", 0.15),
            ("L2", "L4", -0.18),
            ("L2", "L6", 0.35),
            ("L2", "L8", 0.42),
            ("L3", "L4", -0.16),
            ("L3", "L8", 0.30),
            ("L4", "L8", 0.08),
            ("L5", "L2", 0.20),
            ("L5", "L3", -0.16),
            ("L6", "L4", -0.10),
        ]

        for source, target, rad in inclusion_edges:
            self._scheme_arrow(
                ax,
                positions[source],
                positions[target],
                "#2f3e46",
                start_radius=radius[source],
                end_radius=radius[target],
                rad=rad,
                linestyle="-",
                lw=1.9,
                mutation_scale=15,
            )

        monitor_edges = [
            ("L2", "L7", 0.10),
            ("L3", "L7", 0.00),
            ("L4", "L7", -0.10),
            ("L5", "L7", 0.10),
            ("L6", "L7", -0.18),
            ("L8", "L7", 0.18),
        ]

        for source, target, rad in monitor_edges:
            self._scheme_arrow(
                ax,
                positions[source],
                positions[target],
                "#5f789f",
                start_radius=radius[source],
                end_radius=radius[target],
                rad=rad,
                linestyle="--",
                lw=1.35,
                mutation_scale=13,
            )

        for level, (x, y) in positions.items():
            circle = Circle(
                (x, y),
                radius[level],
                facecolor=LEVEL_COLORS[level],
                edgecolor="#2f2f2f",
                linewidth=1.55,
                alpha=0.96,
                zorder=5,
            )
            ax.add_patch(circle)
            ax.text(
                x,
                y + 0.12,
                labels[level],
                ha="center",
                va="center",
                fontsize=11.5,
                fontweight="bold" if level in {"L2", "L7"} else "normal",
                zorder=7,
            )

        self._scheme_internal_icons(ax, positions)
        self._draw_scheme_legend(ax)
        self._draw_scheme_relations(ax)

        ax.set_title(
            "Схематичная сеть взаимодействия всех 9 уровней G-Net",
            fontsize=18,
            fontweight="bold",
            pad=18,
        )

        fig.tight_layout()
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    def _logic_canvas_positions(self) -> dict[str, tuple[float, float]]:
        transformed: dict[str, tuple[float, float]] = {}
        for node_id, attrs in self.graph.nodes(data=True):
            x, y = attrs["pos"]
            tx = 4.0 + (x + 10.0) * 0.78
            ty = 1.3 + (y + 8.7) * 0.64
            transformed[node_id] = (tx, ty)
        return transformed

    def _draw_logic_topobase(self, ax, pos: dict[str, tuple[float, float]]) -> None:
        x = np.linspace(0, 24, 500)
        y = np.linspace(0, 15, 320)
        xx, yy = np.meshgrid(x, y)
        zz = (
            0.80 * np.exp(-((xx - 6.5) ** 2 + (yy - 8.2) ** 2) / 20)
            + 0.55 * np.exp(-((xx - 16.0) ** 2 + (yy - 7.0) ** 2) / 24)
            + 0.42 * np.exp(-((xx - 11.7) ** 2 + (yy - 11.2) ** 2) / 13)
            - 0.30 * np.exp(-((xx - 10.5) ** 2 + (yy - 5.2) ** 2) / 10)
        )
        ax.contourf(xx, yy, zz, levels=12, cmap="Greys", alpha=0.08)
        ax.contour(xx, yy, zz, levels=8, colors="#9a9a9a", linewidths=0.38, alpha=0.22)

        boundary = FancyBboxPatch(
            (3.3, 1.0),
            17.7,
            11.9,
            boxstyle="round,pad=0.08,rounding_size=0.24",
            facecolor=(1.0, 1.0, 1.0, 0.0),
            edgecolor="#b0b0b0",
            linewidth=1.0,
            linestyle=(0, (3, 3)),
        )
        ax.add_patch(boundary)
        ax.text(3.55, 12.58, "L8 топооснова", fontsize=10.4, fontweight="bold", color="#606060")

        topo_labels = {
            "TERRAIN_NW": "NW",
            "TERRAIN_NE": "NE",
            "TERRAIN_W": "W",
            "TERRAIN_C": "C",
            "TERRAIN_E": "E",
            "TERRAIN_SW": "SW",
            "TERRAIN_SE": "SE",
        }
        for node_id, text in topo_labels.items():
            x0, y0 = pos[node_id]
            ax.scatter([x0], [y0], s=42, color="#dadada", edgecolors="#777777", linewidths=0.6, zorder=1)
            ax.text(x0 + 0.15, y0 + 0.12, text, fontsize=7.8, color="#707070")

    def _draw_logic_l6_zones(self, ax) -> None:
        zones = [
            (5.0, 4.6, 4.3, 4.9, "L6 зона A", "#ffe8a1"),
            (9.65, 4.6, 4.3, 4.9, "L6 зона B", "#ffe8a1"),
            (14.3, 4.6, 4.3, 4.9, "L6 зона C", "#ffe8a1"),
        ]
        for x, y, w, h, text, color in zones:
            rect = FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=0.18",
                facecolor=color,
                edgecolor="#c9a227",
                linewidth=0.9,
                alpha=0.12,
                zorder=0,
            )
            ax.add_patch(rect)
            ax.text(x + 0.14, y + h - 0.20, text, fontsize=8.7, color="#8a6f00", va="top")

    def _draw_logic_l5_rings(self, ax) -> None:
        rings = [
            (7.3, 8.1, 2.55, "RING A / L5", "#f4a261"),
            (11.0, 8.1, 2.55, "RING B / L5", "#e9c46a"),
            (14.7, 8.1, 2.55, "RING C / L5", "#8ecae6"),
        ]
        for x, y, radius, text, color in rings:
            circle = Circle(
                (x, y),
                radius=radius,
                fill=False,
                linestyle=(0, (4, 2)),
                linewidth=1.8,
                color=color,
                alpha=0.95,
                zorder=2,
            )
            ax.add_patch(circle)
            ax.text(x, y + radius + 0.28, text, ha="center", fontsize=9.8, fontweight="bold")

    def _draw_logic_edges(self, ax, pos: dict[str, tuple[float, float]]) -> None:
        for source, target, attrs in self.graph.edges(data=True):
            source_visible = self.graph.nodes[source].get("visible_in_logic", False)
            target_visible = self.graph.nodes[target].get("visible_in_logic", False)
            if not (source_visible and target_visible):
                continue

            x1, y1 = pos[source]
            x2, y2 = pos[target]
            medium = attrs["medium"]
            color = MEDIUM_COLORS.get(medium, "#666666")
            style = ":" if medium == "logical-service-binding" else "-"
            
            # Draw background line for specific mediums
            if medium in EDGE_VISUAL_STYLES:
                style_config = EDGE_VISUAL_STYLES[medium]
                ax.plot([x1, x2], [y1, y2], color=style_config["line_color"], linewidth=style_config["line_width"], alpha=style_config["alpha"], zorder=2)
            
            ax.plot(
                [x1, x2],
                [y1, y2],
                linestyle=style,
                linewidth=2.6 if medium == "fiber" else 1.8,
                color=color,
                alpha=0.86 if medium != "logical-service-binding" else 0.78,
                zorder=3,
            )

    def _draw_logic_nodes(self, ax, pos: dict[str, tuple[float, float]]) -> None:
        node_groups = {"L0": [], "L1": [], "L2": []}
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("visible_in_logic", False) and attrs["level"] in node_groups:
                node_groups[attrs["level"]].append(node_id)

        for level, node_ids in node_groups.items():
            xs = [pos[node_id][0] for node_id in node_ids]
            ys = [pos[node_id][1] for node_id in node_ids]
            colors = [self.graph.nodes[node_id]["color"] for node_id in node_ids]
            size = NODE_SIZES.get(level, NODE_SIZES["default"])
            ax.scatter(xs, ys, s=size, c=colors, edgecolors="#303030", linewidths=0.9, zorder=5)

    def _draw_logic_labels(self, ax, pos: dict[str, tuple[float, float]]) -> None:
        for node_id, attrs in self.graph.nodes(data=True):
            if not attrs.get("visible_in_logic", False):
                continue
            x, y = pos[node_id]
            level = attrs["level"]
            if level == "L1":
                label = "M" if attrs["role"] == "mobile-subscriber" else "PC"
                fontsize = FONTSIZE_CONFIGS["label_l1"]
            else:
                label = attrs["label"]
                fontsize = FONTSIZE_CONFIGS["label_default"]
            ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, fontweight="bold", zorder=6)

        area_badges = [
            (6.95, 12.05, "L0 сервисы", "#d8f3dc"),
            (10.8, 7.05, "L2 активка", "#a9def9"),
            (5.10, 3.05, "L1 mobile → A1/A3/A5", "#b7e4c7"),
            (13.40, 3.05, "L1 fixed → A2/A4/A6", "#b7e4c7"),
        ]
        for x, y, text, color in area_badges:
            ax.text(
                x,
                y,
                text,
                fontsize=FONTSIZE_CONFIGS["badge"],
                bbox=dict(boxstyle="round,pad=0.18", fc=color, ec="#777777", alpha=0.90),
                zorder=7,
            )

    def _draw_logic_minimal_panels(self, ax) -> None:
        inclusion_panel = FancyBboxPatch(
            (0.7, 9.05),
            2.45,
            3.25,
            boxstyle="round,pad=0.06",
            facecolor="white",
            edgecolor="#6d6d6d",
            linewidth=0.9,
            alpha=0.95,
        )
        ax.add_patch(inclusion_panel)
        inclusion_text = (
            "Вложения\n"
            "L0⊂L1⊂L2\n"
            "L2⊂L3,L4,L6,L8\n"
            "L3⊂L4,L8\n"
            "L4⊂L8\n"
            "L5⊂L2,L3\n"
            "L6⊂L4\n"
            "L7=арбитр"
        )
        ax.text(1.92, 10.67, inclusion_text, ha="center", va="center", fontsize=9.7)

        small_legend = FancyBboxPatch(
            (19.2, 9.25),
            3.0,
            2.9,
            boxstyle="round,pad=0.06",
            facecolor="white",
            edgecolor="#6d6d6d",
            linewidth=0.9,
            alpha=0.95,
        )
        ax.add_patch(small_legend)
        ax.text(20.7, 11.72, "Легенда", ha="center", va="center", fontsize=10.5, fontweight="bold")
        ax.text(19.45, 11.2, "узлы: L0 / L1 / L2", fontsize=9.2)
        ax.text(19.45, 10.78, "кольца: L5", fontsize=9.2)
        ax.text(19.45, 10.36, "зоны питания: L6", fontsize=9.2)
        ax.text(19.45, 9.94, "среда/линейка: L3/L4", fontsize=9.2)
        ax.text(19.45, 9.52, "фон: L8", fontsize=9.2)

        arb_box = FancyBboxPatch(
            (8.55, 13.12),
            6.9,
            0.8,
            boxstyle="round,pad=0.05",
            facecolor=LEVEL_COLORS["L7"],
            edgecolor="#6b4d2f",
            linewidth=1.0,
            alpha=0.93,
        )
        ax.add_patch(arb_box)
        ax.text(12.0, 13.52, "L7 арбитр", ha="center", va="center", fontsize=10.0, fontweight="bold")

    def _scheme_internal_icons(self, ax, positions: dict[str, tuple[float, float]]) -> None:
        """
        Draw small inner nodes only where the level has explicit graph nodes:
        - L0: service nodes;
        - L1: subscriber nodes;
        - L2: active equipment nodes;
        - L8: topo-base anchor nodes.

        L3/L4/L5/L6/L7 are shown as semantic/control layers in this scheme,
        so fake inner nodes are not drawn there.
        """
        self._draw_small_nodes(
            ax,
            center=positions["L8"],
            offsets=[(-0.35, -0.28), (0.0, -0.40), (0.35, -0.28), (-0.18, -0.62), (0.18, -0.62), (-0.48, -0.52), (0.48, -0.52)],
            color="#d1d1d1",
            edge_color="#666666",
            size=0.065,
        )

        l2_offsets = []
        start_x = -0.54
        start_y = -0.48
        step = 0.22
        for row in range(3):
            for col in range(6):
                l2_offsets.append((start_x + col * step, start_y - row * step))
        self._draw_small_nodes(
            ax,
            center=positions["L2"],
            offsets=l2_offsets,
            color="#5aa9e6",
            edge_color="#2d2d2d",
            size=0.065,
        )

    def _draw_small_nodes(
        self,
        ax,
        *,
        center: tuple[float, float],
        offsets: list[tuple[float, float]],
        color: str,
        edge_color: str,
        size: float,
    ) -> None:
        cx, cy = center
        for dx, dy in offsets:
            ax.add_patch(
                Circle(
                    (cx + dx, cy + dy),
                    size,
                    color=color,
                    ec=edge_color,
                    lw=0.5,
                    zorder=8,
                )
            )

    def _draw_scheme_legend(self, ax) -> None:
        legend_box = FancyBboxPatch(
            (0.55, 3.9),
            3.3,
            2.2,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor="white",
            edgecolor="#707070",
            linewidth=1.0,
            alpha=0.96,
            zorder=10,
        )
        ax.add_patch(legend_box)
        ax.text(2.2, 5.55, "Типы связей", ha="center", va="center", fontsize=12, fontweight="bold", zorder=11)

        self._legend_arrow(ax, (0.9, 5.05), (1.55, 5.05), "#2f3e46", linestyle="-")
        ax.text(1.8, 5.05, "вложение / включение", fontsize=10.2, va="center", zorder=11)

        self._legend_arrow(ax, (0.9, 4.55), (1.55, 4.55), "#5f789f", linestyle="--")
        ax.text(1.8, 4.55, "мониторинг в L7", fontsize=10.2, va="center", zorder=11)

    def _legend_arrow(
        self,
        ax,
        start: tuple[float, float],
        end: tuple[float, float],
        color: str,
        *,
        linestyle: str | tuple = "-",
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.5,
                color=color,
                linestyle=linestyle,
                zorder=11,
            )
        )

    def _draw_scheme_relations(self, ax) -> None:
        relation_box = FancyBboxPatch(
            (16.2, 2.05),
            3.1,
            4.5,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor="white",
            edgecolor="#707070",
            linewidth=1.0,
            alpha=0.96,
            zorder=10,
        )
        ax.add_patch(relation_box)
        relation_text = (
            "Вложение\n\n"
            "L0 ⊂ L1\n"
            "L1 ⊂ L2\n"
            "L2 ⊂ L3, L4, L6, L8\n"
            "L3 ⊂ L4, L8\n"
            "L4 ⊂ L8\n"
            "L5 ⊂ L2, L3\n"
            "L6 ⊂ L4\n\n"
            "L7 = арбитр"
        )
        ax.text(17.75, 4.25, relation_text, ha="center", va="center", fontsize=10.8, zorder=11)

    def _scheme_arrow(
        self,
        ax,
        start: tuple[float, float],
        end: tuple[float, float],
        color: str,
        *,
        start_radius: float,
        end_radius: float,
        rad: float,
        linestyle: str | tuple = "-",
        lw: float = 1.8,
        mutation_scale: float = 15,
    ) -> None:
        safe_start, safe_end = self._trim_arrow_to_circles(
            start,
            end,
            start_radius=start_radius,
            end_radius=end_radius,
            padding=0.05,
        )
        arrow = FancyArrowPatch(
            safe_start,
            safe_end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
            linestyle=linestyle,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
        ax.add_patch(arrow)

    def _trim_arrow_to_circles(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        start_radius: float,
        end_radius: float,
        padding: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        start_arr = np.array(start, dtype=float)
        end_arr = np.array(end, dtype=float)
        vector = end_arr - start_arr
        distance = float(np.linalg.norm(vector))
        if distance == 0:
            return start, end

        direction = vector / distance
        safe_start = start_arr + direction * (start_radius + padding)
        safe_end = end_arr - direction * (end_radius + padding)
        return (float(safe_start[0]), float(safe_start[1])), (float(safe_end[0]), float(safe_end[1]))
