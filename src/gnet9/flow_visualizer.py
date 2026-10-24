"""Читаемая статическая и интерактивная визуализация сервисного трафика."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import textwrap
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Button, Slider
import numpy as np

from .constants import SERVICE_DISPLAY_NAMES
from .flow_details import (FLOW_SHORT_LABELS, _flow_summary, _flow_is_visible,
                           _interactive_snapshot_summary, _interactive_status_sections,
                           _node_hover_text)
from .network_layout import network_positions as _positions
from .plot_style import BACKGROUND, PANEL, TEXT, BORDER, save_figure


FLOW_COLORS = {
    "RTP_OPUS": "#ef4444",
    "RTP_VLC_AV": "#f97316",
    "FTP_DATA": "#3b82f6",
    "DNS": "#a855f7",
    "RTP_TELEMOST": "#14b8a6",
    "LIVE_HLS": "#eab308",
    "ATTACK_DOS": "#ff1744",
    "ATTACK_DDOS": "#b91c1c",
    "ATTACK_SYN": "#f43f5e",
}
LEVEL_COLORS = {"L0": "#10b981", "L2": "#60a5fa", "L7": "#fb923c", "L8": "#94a3b8"}
NODE_SIZES = {"L0": 175, "L2": 64, "L7": 105, "L8": 28}
SUBSCRIBER_STYLES = {
    "mobile-subscriber": {"color": "#22d3ee", "marker": "o", "label": "Мобильные абоненты (М)"},
    "fixed-subscriber": {"color": "#fbbf24", "marker": "s", "label": "Фиксированные абоненты (Ф)"},
}
ATTACK_TARGET_COLORS = {
    "dos": "#ef4444",
    "ddos": "#b91c1c",
    "syn_flood": "#f43f5e",
    "power_attack": "#facc15",
}


def aggregate_flow_edges(flows: list[dict[str, Any]]) -> dict[frozenset[str], dict[str, Any]]:
    """Суммировать потоки, байты и приложения на каждой пройденной связи."""
    result: dict[frozenset[str], dict[str, Any]] = {}
    for flow in flows:
        if not _flow_is_visible(flow):
            continue
        route = flow.get("route", [])
        for source, target in zip(route, route[1:]):
            key = frozenset((source, target))
            item = result.setdefault(
                key,
                {
                    "flow_count": 0,
                    "wire_bytes": 0,
                    "applications": Counter(),
                    "rerouted_flow_count": 0,
                    "failover_flow_count": 0,
                },
            )
            item["flow_count"] += 1
            item["wire_bytes"] += int(flow.get("wire_bytes", 0))
            item["applications"][flow.get("application", "unknown")] += 1
            item["rerouted_flow_count"] += int(bool(flow.get("rerouted")))
            item["failover_flow_count"] += int(bool(flow.get("failover_active")))
    return result


def draw_service_flow_map(
    model,
    flows: list[dict[str, Any]],
    path: Path,
    *,
    title_suffix: str = "",
) -> None:
    """Сохранить сеть с наблюдаемыми потоками текущего шага."""
    draw_network_map(model, path, flows=flows, title_suffix=title_suffix)


def draw_network_map(model, path: Path, *, flows=None, title_suffix="") -> None:
    """Общий способ рисования сети; None означает схему без нагрузки."""
    topology_only = flows is None
    flows = flows or []
    pos = _positions(model)
    loads = None if topology_only else aggregate_flow_edges(flows)
    max_bytes = max((item["wire_bytes"] for item in (loads or {}).values()), default=1)
    fig, ax = plt.subplots(figsize=(24, 15))
    fig.subplots_adjust(bottom=0.055, left=0.014, right=0.755, top=0.93)
    title = "GNet9 · схема сети" if topology_only else "GNet9 · потоки и нагрузка текущего шага"
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    _style_axes(ax, title)
    _draw_access_zones(ax, model, pos)
    _draw_topology(ax, model, pos, loads=loads, max_bytes=max_bytes)
    _draw_labels(ax, model, pos)
    isolated_nodes = sorted({
        str(flow.get("client_node")) for flow in flows
        if flow.get("isolated") and flow.get("client_node") in pos
    })
    if isolated_nodes:
        ax.scatter(
            [pos[node][0] for node in isolated_nodes],
            [pos[node][1] for node in isolated_nodes],
            s=190, marker="x", c="#fb7185", linewidths=2.5, zorder=16,
        )
    failover_servers = sorted({
        str(flow.get("server_node")) for flow in flows
        if flow.get("failover_active") and flow.get("server_node") in pos
    })
    if failover_servers:
        ax.scatter(
            [pos[node][0] for node in failover_servers],
            [pos[node][1] for node in failover_servers],
            s=310, facecolors="none", edgecolors="#34d399",
            linewidths=2.5, zorder=16,
        )
    _set_topology_view(ax, pos)
    _draw_legend(ax, static=True, topology_only=topology_only, applications=set(f.get("application") for f in flows),
                 show_isolation=bool(isolated_nodes), show_failover=bool(failover_servers))
    summary = (f"СОСТАВ СЕТИ\n\n{sum(a.get('level') == 'L1' for _, a in model.graph.nodes(data=True))} абонентов\n"
               f"{sum(a.get('level') == 'L2' for _, a in model.graph.nodes(data=True))} сетевых устройств\n"
               f"{len(model.servers)} сервера · {len(model.services)} услуг\n\n"
               "Каждая точка снизу — абонент.\n"
               "Цвет различает мобильный\nи проводной доступ.\n\n"
               "Координаты схематичны.\nЛинии взяты из графа сети.") if topology_only else _flow_summary(flows)
    ax.text(
        1.018, 0.98, summary, transform=ax.transAxes, va="top", fontsize=9.2, color=TEXT,
        bbox={"boxstyle": "round,pad=0.5", "fc": PANEL, "ec": BORDER, "alpha": 0.92},
    )
    save_figure(fig, path)


class ServiceFlowWindow:
    """Интерактивное окно трафика с выбором и просмотром точного шага."""

    def __init__(self, model, snapshots: list[dict[str, Any]], *, particles_per_frame: int = 52) -> None:
        if not any(isinstance(snapshot.get("traffic", {}).get("flows"), list) for snapshot in snapshots):
            raise ValueError("Для интерактивного окна требуется packet_detail='flows' или 'sample'")
        usable = list(snapshots)
        self.model = model
        self.snapshots = usable
        self.pos = _positions(model)
        # Число частиц ограничено: визуальная плотность сохраняется, а окно
        # остаётся отзывчивым и на длинном 20-секундном горизонте прогноза.
        self.particles_per_frame = min(52, max(1, particles_per_frame))
        # Окно открывается на точном снимке t0; анимация начинается только
        # после явного нажатия пользователем кнопки «Продолжить».
        self.paused = True
        self.frame_in_step = 0
        self.frames_per_step = 32
        self.step_index = 0
        self._syncing_slider = False
        self.edge_loads = [aggregate_flow_edges(snapshot.get("traffic", {}).get("flows", [])) for snapshot in usable]
        self.max_wire_bytes = max(1, max(
            (item["wire_bytes"] for loads in self.edge_loads for item in loads.values()), default=1
        ))

        # The main network area is intentionally wider than the side panels:
        # 480 endpoints are readable without zooming into a single access group.
        self.fig, self.ax = plt.subplots(figsize=(28, 16))
        self.fig.subplots_adjust(bottom=0.125, right=0.770, left=0.018, top=0.94)
        self.status_ax = self.fig.add_axes([0.790, 0.295, 0.200, 0.650])
        self.legend_ax = self.fig.add_axes([0.790, 0.140, 0.200, 0.140])
        self.help_ax = self.fig.add_axes([0.790, 0.025, 0.200, 0.100])
        for panel_ax in (self.status_ax, self.legend_ax, self.help_ax):
            panel_ax.set_axis_off()
            panel_ax.set_facecolor("none")
        _style_axes(self.ax, "GNet9 · потоки по шагам")
        _draw_access_zones(self.ax, model, self.pos)
        _draw_topology(self.ax, model, self.pos)
        _draw_labels(self.ax, model, self.pos)
        _set_topology_view(self.ax, self.pos)
        _draw_legend(self.legend_ax, compact=True)

        self.edge_artists: dict[frozenset[str], Any] = {}
        for source, target in model.graph.edges():
            x1, y1 = self.pos[source]
            x2, y2 = self.pos[target]
            line, = self.ax.plot([x1, x2], [y1, y2], color="#64748b", linewidth=0.1, alpha=0.0, zorder=3)
            self.edge_artists[frozenset((source, target))] = line

        self.particle_glow = self.ax.scatter([], [], s=[], c=[], alpha=0.18, linewidths=0, zorder=7)
        self.particles = self.ax.scatter([], [], s=[], c=[], edgecolors="white", linewidths=0.45, zorder=8)
        self.attack_targets = self.ax.scatter(
            [], [], s=[], facecolors="none", edgecolors="#ff1744", linewidths=2.8, alpha=0.95, zorder=15
        )
        self.warning_targets = self.ax.scatter(
            [], [], s=[], facecolors="none", edgecolors="#c084fc", linewidths=2.2,
            linestyles="--", alpha=0.95, zorder=14,
        )
        self.isolated_nodes = self.ax.scatter(
            [], [], s=[], marker="x", c="#fb7185", linewidths=2.8, alpha=0.96, zorder=16,
        )
        self.failover_servers = self.ax.scatter(
            [], [], s=[], facecolors="none", edgecolors="#34d399",
            linewidths=2.6, alpha=0.96, zorder=16,
        )
        panel_style = {"boxstyle": "round,pad=0.55", "fc": "#172033", "ec": "#475569", "alpha": 0.96}
        self.status_cards = []
        top = 1.0
        for height in (0.205, 0.205, 0.235, 0.205, 0.150):
            bottom = top - height
            panel = FancyBboxPatch((0, bottom + 0.01), 1.0, height - 0.018,
                                  transform=self.status_ax.transAxes, boxstyle="round,pad=0.008",
                                  facecolor=PANEL, edgecolor=BORDER, linewidth=1)
            self.status_ax.add_patch(panel)
            heading = self.status_ax.text(0.025, top - 0.018, "", va="top", fontsize=11.8,
                                          fontweight="bold", transform=self.status_ax.transAxes)
            body = self.status_ax.text(0.025, top - 0.052, "", va="top", fontsize=11.0,
                                       color=TEXT, linespacing=1.22, transform=self.status_ax.transAxes)
            self.status_cards.append((heading, body, panel))
            top = bottom
        self.status = self.status_cards[0][1]
        self.help_text = self.help_ax.text(
            0.0, 1.0,
            "КАК ЧИТАТЬ И УПРАВЛЯТЬ\n"
            "Точки — условная выборка потоков, не отдельные пакеты.\n"
            "Толщина линии — объём за шаг; цвет — услуга.\n"
            "← / → — шаг; пробел — воспроизведение / пауза.\n"
            "Наведение — показатели выбранного шага.",
            transform=self.help_ax.transAxes, va="top", fontsize=9.2, color="#dbeafe",
            linespacing=1.12, bbox=panel_style,
        )
        self.hover_note = self.ax.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
            color="#f8fafc", fontsize=8.5, zorder=30,
            bbox={"boxstyle": "round,pad=0.45", "fc": "#020617", "ec": "#38bdf8", "alpha": 0.97},
            arrowprops={"arrowstyle": "->", "color": "#38bdf8"},
        )
        self.hover_note.set_visible(False)
        # Карточка доступна для каждого отображаемого узла, включая опорные
        # географические точки L8.  Это сохраняет правило интерфейса:
        # если объект виден, его смысл можно узнать наведением.
        self.hover_nodes = list(model.graph.nodes)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        slider_ax = self.fig.add_axes([0.18, 0.045, 0.41, 0.027])
        self.slider = Slider(
            slider_ax, "Шаг", 0, max(1, len(usable) - 1), valinit=0, valstep=1,
            valfmt="%d", color="#38bdf8", initcolor="none"
        )
        self.slider.label.set_color("#f8fafc")
        self.slider.valtext.set_color("#f8fafc")
        tick_count = min(len(usable), 11)
        step_ticks = sorted({int(round(value)) for value in np.linspace(0, len(usable) - 1, tick_count)})
        slider_ax.set_xticks(step_ticks)
        slider_ax.set_xticklabels(
            [str(usable[value].get("step_index", value)) for value in step_ticks],
            color="#cbd5e1",
            fontsize=7,
        )
        slider_ax.tick_params(axis="x", length=2, pad=3, colors="#cbd5e1")
        self.slider.on_changed(self._select_step)
        self.slider.track.set_facecolor("#334155")
        self.playback_status = self.fig.text(
            0.16, 0.090, "", ha="left", va="center",
            color="#e2e8f0", fontsize=10.5, fontweight="bold",
        )
        self.previous_button = Button(self.fig.add_axes([0.022, 0.027, 0.06, 0.055]), "← Назад", color=PANEL, hovercolor=BORDER)
        self.next_button = Button(self.fig.add_axes([0.086, 0.027, 0.06, 0.055]), "Вперёд →", color=PANEL, hovercolor=BORDER)
        for step_button, direction in ((self.previous_button, -1), (self.next_button, 1)):
            step_button.label.set_color(TEXT)
            step_button.on_clicked(lambda event, shift=direction: self._move_step(shift))
        button_ax = self.fig.add_axes([0.665, 0.027, 0.105, 0.055])
        self.button = Button(button_ax, "▶  Продолжить", color="#2563eb", hovercolor="#3b82f6")
        self.button.label.set_color("#f8fafc")
        self.button.label.set_fontweight("bold")
        self.button.on_clicked(self._toggle_pause)
        self._set_playback_style()
        self._render_snapshot(update_edges=True)
        self.animation = FuncAnimation(self.fig, self._update, interval=48, blit=False, cache_frame_data=False)

    def show(self) -> None:
        manager = plt.get_current_fig_manager()
        try:
            manager.window.state("zoomed")
        except (AttributeError, RuntimeError):
            pass
        plt.show()

    def _on_hover(self, event) -> None:
        if event.inaxes is not self.ax or event.x is None or event.y is None:
            if self.hover_note.get_visible():
                self.hover_note.set_visible(False)
                self.fig.canvas.draw_idle()
            return
        display_points = self.ax.transData.transform([self.pos[node] for node in self.hover_nodes])
        distances = np.hypot(display_points[:, 0] - event.x, display_points[:, 1] - event.y)
        nearest_index = int(np.argmin(distances))
        if distances[nearest_index] > 13.0:
            if self.hover_note.get_visible():
                self.hover_note.set_visible(False)
                self.fig.canvas.draw_idle()
            return
        node_id = self.hover_nodes[nearest_index]
        self.hover_note.xy = self.pos[node_id]
        right_half = event.x > (self.ax.bbox.x0 + self.ax.bbox.x1) / 2.0
        upper_half = event.y > (self.ax.bbox.y0 + self.ax.bbox.y1) / 2.0
        self.hover_note.set_position((-12 if right_half else 12, -12 if upper_half else 12))
        self.hover_note.set_ha("right" if right_half else "left")
        self.hover_note.set_va("top" if upper_half else "bottom")
        self.hover_note.set_text(_node_hover_text(self.model, node_id, self.snapshots[self.step_index]))
        self.hover_note.set_visible(True)
        self.fig.canvas.draw_idle()

    def _select_step(self, value: float) -> None:
        if self._syncing_slider:
            return
        self.step_index = max(0, min(len(self.snapshots) - 1, int(value)))
        if int(value) != self.step_index:
            self._sync_slider()
        self.frame_in_step = (
            0
            if self.step_index == 0
            else int(round(((self.step_index * 0.173) % 1.0) * self.frames_per_step))
        )
        self.paused = True
        self.hover_note.set_visible(False)
        self._set_playback_style()
        self._render_snapshot(update_edges=True)
        self.fig.canvas.draw_idle()

    def _move_step(self, direction: int) -> None:
        """Выбрать соседний сохранённый снимок и остановить воспроизведение."""
        self._select_step(self.step_index + direction)
        self._sync_slider()

    def _on_key(self, event) -> None:
        if event.key in {"right", "left"}:
            self._move_step(1 if event.key == "right" else -1)
        elif event.key in {" ", "space"}:
            self._toggle_pause(event)
        elif event.key in {"home", "end"}:
            self._select_step(0 if event.key == "home" else len(self.snapshots) - 1)
            self._sync_slider()

    def _toggle_pause(self, _event) -> None:
        restarted = False
        if self.paused and self.step_index == len(self.snapshots) - 1 and self.frame_in_step >= self.frames_per_step:
            self.step_index = 0
            self.frame_in_step = 0
            self.hover_note.set_visible(False)
            restarted = True
            self._sync_slider()
        self.paused = not self.paused
        self._set_playback_style()
        self._render_snapshot(update_edges=restarted)
        self.fig.canvas.draw_idle()

    def _update(self, _frame_number):
        update_edges = False
        if not self.paused:
            previous_step = self.step_index
            self.step_index, self.frame_in_step, finished = _advance_playback(
                step_index=self.step_index,
                frame_in_step=self.frame_in_step,
                final_step=len(self.snapshots) - 1,
                frames_per_step=self.frames_per_step,
            )
            if self.step_index != previous_step:
                self.hover_note.set_visible(False)
                self._sync_slider()
                self._set_playback_style()
                update_edges = True
            if finished:
                self.paused = True
                self._set_playback_style(finished=True)
        return self._render_snapshot(update_edges=update_edges)

    def _sync_slider(self) -> None:
        self._syncing_slider = True
        try:
            self.slider.set_val(self.step_index)
            self.slider.valtext.set_text(str(self.snapshots[self.step_index].get("step_index", self.step_index)))
        finally:
            self._syncing_slider = False

    def _set_playback_style(self, *, finished: bool = False) -> None:
        snapshot = self.snapshots[self.step_index]
        actual_step = int(snapshot.get("step_index", self.step_index))
        self.slider.valtext.set_text(str(actual_step))
        final_actual_step = int(
            self.snapshots[-1].get("step_index", len(self.snapshots) - 1)
        )
        prefix = (
            f"Шаг {actual_step} из {final_actual_step} · "
            f"t = {snapshot.get('time_seconds', 0)} с"
        )
        if finished:
            self.button.label.set_text("↻  Повторить")
            self.playback_status.set_text(f"{prefix} · ЗАВЕРШЕНО")
            self.playback_status.set_color("#86efac")
            self.slider.poly.set_facecolor("#22c55e")
        elif self.paused:
            self.button.label.set_text("▶  Продолжить")
            self.playback_status.set_text(f"{prefix} · ПАУЗА")
            self.playback_status.set_color("#bae6fd")
            self.slider.poly.set_facecolor("#38bdf8")
        else:
            self.button.label.set_text("Ⅱ  Пауза")
            self.playback_status.set_text(f"{prefix} · ВОСПРОИЗВЕДЕНИЕ")
            self.playback_status.set_color("#86efac")
            self.slider.poly.set_facecolor("#22c55e")

    def _render_snapshot(self, *, update_edges: bool):
        snapshot = self.snapshots[self.step_index]
        flows = snapshot.get("traffic", {}).get("flows", [])
        loads = self.edge_loads[self.step_index]
        if update_edges:
            for key, line in self.edge_artists.items():
                load = loads.get(key)
                if not load:
                    line.set_alpha(0.0)
                    continue
                dominant = load["applications"].most_common(1)[0][0]
                line.set_color(FLOW_COLORS.get(dominant, "#ffbe0b"))
                line.set_linewidth(0.8 + 5.2 * np.sqrt(load["wire_bytes"] / self.max_wire_bytes))
                line.set_alpha(0.88)
                line.set_linestyle("--" if load.get("rerouted_flow_count") else "-")

        selected = _spread_sample(flows, self.particles_per_frame, self.step_index + self.frame_in_step)
        points: list[tuple[float, float]] = []
        colors: list[str] = []
        sizes: list[float] = []
        for offset, flow in enumerate(selected):
            progress = _particle_progress(
                step_index=int(snapshot.get("step_index", 0)),
                frame=self.frame_in_step,
                offset=offset,
                particle_count=len(selected),
                paused=self.paused,
                frames_per_step=self.frames_per_step,
            )
            point = _point_on_route(flow.get("route", []), self.pos, progress)
            if point is None:
                continue
            points.append(point)
            colors.append(FLOW_COLORS.get(flow.get("application"), "#ffbe0b"))
            sizes.append(46.0 if flow.get("is_attack_traffic") else 23.0 if flow.get("transport") == "TCP" else 17.0)
        self.particles.set_offsets(np.asarray(points) if points else np.empty((0, 2)))
        self.particles.set_color(colors)
        self.particles.set_sizes(sizes)
        self.particle_glow.set_offsets(np.asarray(points) if points else np.empty((0, 2)))
        self.particle_glow.set_color(colors)
        self.particle_glow.set_sizes([size * 3.0 for size in sizes])
        target_events = [
            event for event in snapshot.get("attacks", {}).get("events", [])
            if event.get("target_id") in self.pos
        ]
        target_nodes = [event["target_id"] for event in target_events]
        self.attack_targets.set_offsets(
            np.asarray([self.pos[node] for node in target_nodes]) if target_nodes else np.empty((0, 2))
        )
        self.attack_targets.set_sizes([360.0 + 35.0 * np.sin(self.frame_in_step / 4.0) for _ in target_nodes])
        self.attack_targets.set_edgecolors([
            ATTACK_TARGET_COLORS.get(str(event.get("kind")), "#ff1744")
            for event in target_events
        ])
        koopman = snapshot.get("koopman", {})
        warning_nodes = (
            [
                node for node in koopman.get("forecast_target_ids", [])
                if node in self.pos
            ]
            if koopman.get("forecast_is_early_warning")
            else []
        )
        self.warning_targets.set_offsets(
            np.asarray([self.pos[node] for node in warning_nodes]) if warning_nodes else np.empty((0, 2))
        )
        self.warning_targets.set_sizes(
            [430.0 + 45.0 * np.sin(self.frame_in_step / 5.0) for _ in warning_nodes]
        )
        isolated_nodes = sorted({
            str(flow.get("client_node"))
            for flow in flows
            if flow.get("isolated") and flow.get("client_node") in self.pos
        })
        self.isolated_nodes.set_offsets(
            np.asarray([self.pos[node] for node in isolated_nodes]) if isolated_nodes else np.empty((0, 2))
        )
        self.isolated_nodes.set_sizes([190.0 for _ in isolated_nodes])
        failover_servers = sorted({
            str(flow.get("server_node"))
            for flow in flows
            if flow.get("failover_active") and flow.get("server_node") in self.pos
        })
        self.failover_servers.set_offsets(
            np.asarray([self.pos[node] for node in failover_servers])
            if failover_servers else np.empty((0, 2))
        )
        self.failover_servers.set_sizes([310.0 for _ in failover_servers])
        for section, (heading, body, panel) in zip(
                _interactive_status_sections(snapshot, flows, loads, self.paused, self.model), self.status_cards):
            heading.set_text(section["title"])
            heading.set_color(section["color"])
            body.set_text("\n".join(textwrap.fill(line, width=57) for line in section["lines"]))
            panel.set_edgecolor(section["color"])
        return (
            self.particle_glow,
            self.particles,
            self.attack_targets,
            self.warning_targets,
            self.isolated_nodes,
            self.failover_servers,
            *(text for heading, body, _ in self.status_cards for text in (heading, body)),
            *self.edge_artists.values(),
        )


def show_service_flow_window(model, snapshots: list[dict[str, Any]]) -> None:
    create_service_flow_window(model, snapshots).show()


def create_service_flow_window(model, snapshots: list[dict[str, Any]]) -> ServiceFlowWindow:
    """Создать окно без запуска блокирующего цикла Matplotlib.

    Это позволяет сначала создать окно сети и окно диагностики, а затем показать
    их одновременно одним общим циклом обработки событий.
    """
    ensure_interactive_backend()
    return ServiceFlowWindow(model, snapshots)


def show_visualization_windows(*windows: Any) -> None:
    """Показать одно или несколько заранее созданных окон одновременно."""
    for window in windows:
        manager = getattr(getattr(window, "fig", None), "canvas", None)
        manager = getattr(manager, "manager", None)
        try:
            manager.window.state("zoomed")
        except (AttributeError, RuntimeError):
            pass
    plt.show()


def ensure_interactive_backend() -> None:
    if plt.get_backend().lower() != "agg":
        return
    try:
        plt.switch_backend("TkAgg")
    except ImportError as exc:
        raise RuntimeError(
            "Не удалось открыть окно: Matplotlib использует неинтерактивный режим Agg, а Tkinter недоступен. "
            "Установите Python с компонентом Tcl/Tk или выберите интерактивный режим Matplotlib."
        ) from exc


def _draw_topology(ax, model, pos, *, loads=None, max_bytes: int = 1) -> None:
    for source, target in model.graph.edges():
        x1, y1 = pos[source]
        x2, y2 = pos[target]
        load = None if loads is None else loads.get(frozenset((source, target)))
        edge_attrs = model.graph.edges[source, target]
        standby = bool(edge_attrs.get("standby"))
        endpoint_attrs = (model.graph.nodes[source], model.graph.nodes[target])
        subscriber_role = next(
            (
                str(attrs.get("role"))
                for attrs in endpoint_attrs
                if attrs.get("level") == "L1"
            ),
            None,
        )
        if subscriber_role in SUBSCRIBER_STYLES:
            # На сводной карте линия последней мили кодируется типом абонента,
            # а не приложением. Иначе сотни линий создают цветной «ковёр» и
            # скрывают связь конкретного абонента со своей точкой доступа.
            color = SUBSCRIBER_STYLES[subscriber_role]["color"]
            width = 0.60 if load else 0.36
            alpha, zorder = (0.62, 2) if load else (0.30, 1)
        elif load:
            dominant = load["applications"].most_common(1)[0][0]
            color = FLOW_COLORS.get(dominant, "#ffbe0b")
            width = 0.8 + 5.2 * np.sqrt(load["wire_bytes"] / max_bytes)
            alpha, zorder = 0.88, 3
        elif loads is None:
            color, width, alpha, zorder = "#94a3b8", 1.05, .65, 1
        else:
            color = "#22d3ee" if standby else "#64748b"
            width, alpha, zorder = (0.65, 0.32, 2) if standby else (0.48, 0.36, 1)
        # Two visually different dashed styles prevent an inactive L2 reserve
        # from being confused with a route already remapped by the arbitrator.
        linestyle = "--" if load and load.get("rerouted_flow_count") else ":" if standby else "-"
        ax.plot(
            [x1, x2], [y1, y2], color=color, linewidth=width,
            alpha=alpha, zorder=zorder, linestyle=linestyle,
        )

    levels: dict[str, list[str]] = {}
    for node_id, attrs in model.graph.nodes(data=True):
        level = attrs.get("level", "?")
        if level not in {"L1", "L2"} and attrs.get("role") != "service-server":
            levels.setdefault(level, []).append(node_id)
    for level, node_ids in levels.items():
        ax.scatter(
            [pos[node][0] for node in node_ids], [pos[node][1] for node in node_ids],
            s=NODE_SIZES.get(level, 22), c=LEVEL_COLORS.get(level, "#cbd5e1"),
            edgecolors="#f8fafc", linewidths=0.35, alpha=0.94, zorder=5,
        )
    for role, marker, size in (
        ("core-router", "o", 58),
        ("aggregation-switch", "s", 62),
        ("radio-access-node", "^", 48),
        ("optical-line-terminal", "v", 48),
    ):
        node_ids = [node for node, attrs in model.graph.nodes(data=True) if attrs.get("role") == role]
        ax.scatter(
            [pos[node][0] for node in node_ids], [pos[node][1] for node in node_ids],
            s=size * 1.34, c=LEVEL_COLORS["L2"], marker=marker, edgecolors="#f8fafc",
            linewidths=0.55, alpha=0.98, zorder=6,
        )
    server_nodes = [node for node, attrs in model.graph.nodes(data=True) if attrs.get("role") == "service-server"]
    ax.scatter(
        [pos[node][0] for node in server_nodes], [pos[node][1] for node in server_nodes],
        s=150, c="#a78bfa", marker="D", edgecolors="#f8fafc",
        linewidths=0.8, alpha=0.98, zorder=6,
    )
    for role, style in SUBSCRIBER_STYLES.items():
        node_ids = [node for node, attrs in model.graph.nodes(data=True) if attrs.get("role") == role]
        ax.scatter(
            [pos[node][0] for node in node_ids], [pos[node][1] for node in node_ids],
            s=30, c=style["color"], marker=style["marker"], edgecolors="#0f172a",
            linewidths=0.35, alpha=0.95, zorder=6,
        )
    protected_nodes = [
        node for node, attrs in model.graph.nodes(data=True)
        if attrs.get("critical_protection", {}).get("is_critical")
        and attrs.get("role") in {
            "core-router",
            "aggregation-switch",
            "radio-access-node",
            "optical-line-terminal",
            "service-server",
        }
    ]
    ax.scatter(
        [pos[node][0] for node in protected_nodes], [pos[node][1] for node in protected_nodes],
        s=175, facecolors="none", edgecolors="#fbbf24", linewidths=1.15, alpha=0.9, zorder=7,
    )


def _draw_access_zones(ax, model, pos) -> None:
    """Group subscribers by their aggregation switch without hiding links."""
    for access_node in sorted(
        node for node, attrs in model.graph.nodes(data=True)
        if attrs.get("role") in {"radio-access-node", "optical-line-terminal"}
    ):
        nodes = [
            node for node, attrs in model.graph.nodes(data=True)
            if attrs.get("level") == "L1" and attrs.get("home_access") == access_node
        ]
        if not nodes:
            continue
        xs, ys = [pos[node][0] for node in nodes], [pos[node][1] for node in nodes]
        role = model.graph.nodes[nodes[0]].get("role")
        style = SUBSCRIBER_STYLES[role]
        x0, y0 = min(xs) - 0.18, min(ys) - 0.22
        width, height = max(xs) - min(xs) + 0.36, max(ys) - min(ys) + 0.44
        ax.add_patch(FancyBboxPatch(
            (x0, y0), width, height,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            facecolor=style["color"], edgecolor=style["color"],
            linewidth=0.8, linestyle=(0, (3, 3)), alpha=0.07, zorder=0,
        ))
        access_kind = "М" if role == "mobile-subscriber" else "Ф"
        ax.text(
            x0 + width / 2, y0 - 0.14, f"{access_node} · {access_kind}\n{len(nodes)} абон.",
            ha="center", va="top", fontsize=6.5, color=style["color"], alpha=0.92,
            linespacing=1.15, zorder=8,
        )


def _draw_labels(ax, model, pos) -> None:
    for node_id, attrs in model.graph.nodes(data=True):
        level = attrs.get("level")
        if level not in {"L0", "L2", "L7"}:
            continue
        x, y = pos[node_id]
        label = node_id
        if attrs.get("role") == "service-server":
            label = node_id.replace("SRV_", "СЕРВЕР ")
        elif level == "L0":
            label = SERVICE_DISPLAY_NAMES.get(attrs.get("label"), attrs.get("label", node_id)).upper()
        elif level == "L7":
            label = "Арбитр"
        ax.text(
            x, y + (0.34 if level == "L0" else 0.20), label, ha="center",
            fontsize=8.5, color=TEXT, zorder=7,
        )


def _style_axes(ax, title: str) -> None:
    ax.figure.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_title(title, color="#f8fafc", fontsize=16, pad=12)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def _set_topology_view(ax, pos: dict[str, tuple[float, float]]) -> None:
    """Zoom to real topology bounds so legends and panels do not steal the graph area."""
    if not pos:
        return
    xs = [point[0] for point in pos.values()]
    ys = [point[1] for point in pos.values()]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    ax.set_xlim(min(xs) - max(width * 0.035, 0.55), max(xs) + max(width * 0.035, 0.55))
    ax.set_ylim(min(ys) - max(height * 0.10, 0.80), max(ys) + max(height * 0.08, 0.65))


def _draw_legend(ax, *, static=False, compact=False, applications=None,
                 show_isolation=True, show_failover=True, topology_only=False) -> None:
    flow_labels = FLOW_SHORT_LABELS
    flow_handles = [Line2D([0], [0], color=color, lw=3, label=flow_labels[name]) for name, color in FLOW_COLORS.items() if applications is None or name in applications]
    subscriber_handles = []
    for style in SUBSCRIBER_STYLES.values():
        subscriber_handles.append(Line2D(
            [0], [0], marker=style["marker"], color="none", markerfacecolor=style["color"],
            markeredgecolor="#0f172a", markersize=7, label=style["label"],
        ))
    semantic_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LEVEL_COLORS["L0"], markersize=8, label="Сервис L0"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#a78bfa", markersize=7, label="Физический сервер L0"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LEVEL_COLORS["L2"], markersize=7, label="C — маршрутизатор ядра"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=LEVEL_COLORS["L2"], markersize=7, label="A — агрегирующий коммутатор"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LEVEL_COLORS["L7"], markersize=7, label="АРБ — арбитратор L7"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LEVEL_COLORS["L8"], markersize=5, label="Опорная точка местности L8"),
        Line2D([0], [0], marker="o", color="#fbbf24", markerfacecolor="none", markersize=9, label="Критически важный узел золотого класса"),
        Line2D([0], [0], marker="o", color="#ef4444", markerfacecolor="none", markersize=10, label="Цель сетевой атаки"),
        Line2D([0], [0], marker="o", color="#facc15", markerfacecolor="none", markersize=10, label="Цель атаки на питание"),
        Line2D([0], [0], marker="o", color="#c084fc", markerfacecolor="none", markersize=10, label="Подтверждённая область прогноза Купмана"),
        Line2D([0], [0], marker="x", color="#fb7185", markersize=9, linestyle="none", label="Временно изолированный абонент"),
        Line2D([0], [0], marker="o", color="#34d399", markerfacecolor="none", markersize=10, label="Активный резервный сервер"),
        Line2D([0], [0], color="#e2e8f0", lw=2.0, linestyle="--", label="Штрихи — переназначенный маршрут"),
        Line2D([0], [0], color="#22d3ee", lw=1.5, linestyle=":", label="Точки — резервная связь L2 без трафика"),
        Line2D([0], [0], color="#cbd5e1", lw=0.7, label="Линия абонента: цвет M/Ф"),
        Line2D([0], [0], color="#94a3b8", lw=0.7, alpha=0.7, label="Связь без трафика"),
    ]
    semantic_handles.extend([
        Line2D([0], [0], marker="^", color="none", markerfacecolor=LEVEL_COLORS["L2"], markersize=7, label="RAN — узел радиодоступа"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor=LEVEL_COLORS["L2"], markersize=7, label="OLT — узел оптического доступа"),
    ])
    if static:
        # Только обозначения, которые действительно видны на этом рисунке.
        handles = [*flow_handles, *subscriber_handles, *semantic_handles[:7],
                   *([semantic_handles[10]] if show_isolation else []),
                   *([semantic_handles[11]] if show_failover else []),
                   *(semantic_handles[12:] if not topology_only else semantic_handles[-2:]) ]
        legend_options = {"loc": "lower left", "bbox_to_anchor": (1.018, .015), "ncol": 1}
    elif compact:
        handles = [
            *flow_handles[:6],
            *subscriber_handles,
            semantic_handles[7], semantic_handles[9], semantic_handles[10],
            semantic_handles[11], semantic_handles[12], semantic_handles[13],
        ]
        for handle, label in zip(handles[-6:], ("Цель атаки", "Область прогноза", "Изолированный абонент",
                                               "Резервный сервер", "Маршрут изменён", "Резерв без потока")):
            handle.set_label(label)
        legend_options = {"loc": "upper left", "bbox_to_anchor": (0.0, 1.0), "ncol": 2,
                          "columnspacing": 0.9, "handlelength": 1.7, "borderaxespad": 0.0}
    else:
        handles = [*flow_handles, *subscriber_handles, *semantic_handles]
        legend_options = (
        {"loc": "lower left", "bbox_to_anchor": (1.018, 0.015), "ncol": 2, "columnspacing": 1.0}
        if static else {"loc": "lower center", "bbox_to_anchor": (0.5, -0.105), "ncol": 5, "columnspacing": 1.1}
        )
    legend = ax.legend(
        handles=handles, framealpha=0.96, facecolor="#111827", edgecolor="#475569",
        labelcolor="#e2e8f0", fontsize=8.2 if static else 8.6 if compact else 7.05,
        title="Узлы и связи" if topology_only else "Условные обозначения" if compact else "Узлы и потоки",
        title_fontsize=10.0 if compact else 8.3, **legend_options,
    )
    legend.get_title().set_color("#f8fafc")
    legend.set_zorder(20)


def _spread_sample(flows: list[dict[str, Any]], limit: int, frame: int) -> list[dict[str, Any]]:
    flows = [flow for flow in flows if _flow_is_visible(flow)]
    if len(flows) <= limit:
        return flows
    attack_flows = [flow for flow in flows if flow.get("is_attack_traffic")]
    legitimate = [flow for flow in flows if not flow.get("is_attack_traffic")]
    regular_limit = max(0, limit - len(attack_flows))
    stride = max(1, len(legitimate) // max(regular_limit, 1))
    start = frame % stride
    return attack_flows[:limit] + legitimate[start::stride][:regular_limit]


def _advance_playback(
    *, step_index: int, frame_in_step: int, final_step: int, frames_per_step: int
) -> tuple[int, int, bool]:
    """Advance one animation frame and report whether playback has finished."""
    next_frame = frame_in_step + 1
    if next_frame < frames_per_step:
        return step_index, next_frame, False
    if step_index < final_step:
        return step_index + 1, 0, False
    return final_step, frames_per_step, True


def _particle_progress(
    *,
    step_index: int,
    frame: int,
    offset: int,
    particle_count: int,
    paused: bool,
    frames_per_step: int = 32,
) -> float:
    """Return route progress while preserving the semantics of the t0 snapshot."""
    if step_index == 0 and paused:
        return 0.0
    spacing = offset / max(1, particle_count)
    phase = min(1.0, max(0.0, frame / max(frames_per_step, 1)))
    return (phase + spacing) % 1.0


def _point_on_route(route: list[str], pos: dict[str, tuple[float, float]], progress: float) -> tuple[float, float] | None:
    if any(node not in pos for node in route):
        return None
    points = [pos[node] for node in route if node in pos]
    if len(points) < 2:
        return None
    lengths = [float(np.hypot(x2 - x1, y2 - y1)) for (x1, y1), (x2, y2) in zip(points, points[1:])]
    total = sum(lengths)
    if total <= 0:
        return points[0]
    distance = progress * total
    for (x1, y1), (x2, y2), length in zip(points, points[1:], lengths):
        if distance <= length:
            ratio = distance / max(length, 1e-12)
            return (x1 + ratio * (x2 - x1), y1 + ratio * (y2 - y1))
        distance -= length
    return points[-1]
