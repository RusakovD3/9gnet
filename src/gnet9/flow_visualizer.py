"""Readable static and interactive views of simulated service traffic."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Button, Slider
import numpy as np

from .constants import SERVICE_DISPLAY_NAMES


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
FLOW_LABELS = {
    "RTP_OPUS": "Голос · Opus по RTP/UDP",
    "RTP_VLC_AV": "VLC · голос и видео по RTP/UDP",
    "FTP_DATA": "Файлы · FTP по TCP",
    "DNS": "DNS · запросы имён по UDP",
    "RTP_TELEMOST": "Видеоконференция · RTP/UDP",
    "LIVE_HLS": "Прямая трансляция · LL-HLS/TCP",
    "ATTACK_DOS": "АТАКА · DoS / прямой flood",
    "ATTACK_DDOS": "АТАКА · DDoS / отражённое усиление",
    "ATTACK_SYN": "АТАКА · SYN flood",
}
FLOW_SHORT_LABELS = {
    "RTP_OPUS": "Голос · Opus/RTP",
    "RTP_VLC_AV": "VLC · аудио/видео",
    "FTP_DATA": "Файлы · FTP/TCP",
    "DNS": "DNS · запросы/UDP",
    "RTP_TELEMOST": "Видеоконференция",
    "LIVE_HLS": "Прямая трансляция",
    "ATTACK_DOS": "АТАКА · DoS",
    "ATTACK_DDOS": "АТАКА · DDoS",
    "ATTACK_SYN": "АТАКА · SYN flood",
}
TRAFFIC_LABELS = {
    "voice": "голос", "broadcast_mp3": "аудиовещание MP3", "vlc_av": "VLC: голос и видео",
    "ftp": "передача файлов FTP", "dns": "запросы DNS",
    "video_conference": "видеоконференция", "live_streaming": "прямая трансляция",
}
SLA_LABELS = {"gold": "золотой", "silver": "серебряный", "bronze": "бронзовый"}
LEVEL_COLORS = {"L0": "#10b981", "L2": "#60a5fa", "L7": "#fb923c", "L8": "#94a3b8"}
NODE_SIZES = {"L0": 175, "L2": 64, "L7": 105, "L8": 28}
SUBSCRIBER_STYLES = {
    "mobile-subscriber": {"color": "#22d3ee", "marker": "o", "label": "Мобильные абоненты (М)"},
    "fixed-subscriber": {"color": "#fbbf24", "marker": "s", "label": "Фиксированные абоненты (Ф)"},
}


def _positions(model) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    missing: list[str] = []
    for node_id, attrs in model.graph.nodes(data=True):
        if "pos" in attrs:
            x, y = attrs["pos"]
            positions[node_id] = (float(x) * 1.55, float(y) * 1.04)
        else:
            missing.append(node_id)
    for index, node_id in enumerate(missing):
        positions[node_id] = (-10.5 + index * 0.4, 8.0)

    # Render each 40-user access group as a spacious 8 x 5 endpoint rack.
    # This changes only the display layout; graph coordinates and metrics stay intact.
    access_groups: dict[str, list[str]] = {}
    for node_id, attrs in model.graph.nodes(data=True):
        if attrs.get("level") == "L1":
            access_groups.setdefault(attrs["home_access"], []).append(node_id)
    for access_node, node_ids in access_groups.items():
        center_x, access_y = positions[access_node]
        for index, node_id in enumerate(sorted(node_ids)):
            row, column = divmod(index, 8)
            positions[node_id] = (center_x + (column - 3.5) * 0.42, access_y - 1.42 - row * 0.62)
    return positions


def aggregate_flow_edges(flows: list[dict[str, Any]]) -> dict[frozenset[str], dict[str, Any]]:
    """Aggregate flow count, bytes and applications for every traversed edge."""
    result: dict[frozenset[str], dict[str, Any]] = {}
    for flow in flows:
        route = flow.get("route", [])
        for source, target in zip(route, route[1:]):
            key = frozenset((source, target))
            item = result.setdefault(key, {"flow_count": 0, "wire_bytes": 0, "applications": Counter()})
            item["flow_count"] += 1
            item["wire_bytes"] += int(flow.get("wire_bytes", 0))
            item["applications"][flow.get("application", "unknown")] += 1
    return result


def draw_service_flow_map(model, flows: list[dict[str, Any]], path: Path) -> None:
    """Save a full topology map with traffic load overlaid on graph edges."""
    pos = _positions(model)
    loads = aggregate_flow_edges(flows)
    max_bytes = max((item["wire_bytes"] for item in loads.values()), default=1)
    fig, ax = plt.subplots(figsize=(32, 15))
    fig.subplots_adjust(bottom=0.055, left=0.014, right=0.755, top=0.94)
    _style_axes(ax, "G-Net: сервисные потоки и загрузка связей")
    _draw_access_zones(ax, model, pos)
    _draw_topology(ax, model, pos, loads=loads, max_bytes=max_bytes)
    _draw_labels(ax, model, pos)
    _set_topology_view(ax, pos)
    _draw_legend(ax, static=True)
    ax.text(
        1.018, 0.98, _flow_summary(flows), transform=ax.transAxes, va="top", fontsize=9.8, color="#e9ecef",
        bbox={"boxstyle": "round,pad=0.5", "fc": "#212529", "ec": "#6c757d", "alpha": 0.92},
    )
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


class ServiceFlowWindow:
    """Interactive traffic view with a selectable, inspectable dynamics step."""

    def __init__(self, model, snapshots: list[dict[str, Any]], *, particles_per_frame: int = 52) -> None:
        usable = [snapshot for snapshot in snapshots if snapshot.get("traffic", {}).get("flows")]
        if not usable:
            raise ValueError("Для интерактивного окна требуется packet_detail='flows' или 'sample'")
        self.model = model
        self.snapshots = usable
        self.pos = _positions(model)
        self.particles_per_frame = max(1, particles_per_frame)
        # Open on an exact, inspectable t0 snapshot.  Animation starts only
        # after the user explicitly presses Continue.
        self.paused = True
        self.frame_in_step = 0
        self.frames_per_step = 48
        self.step_index = 0
        self._syncing_slider = False
        self.edge_loads = [aggregate_flow_edges(snapshot["traffic"]["flows"]) for snapshot in usable]
        self.max_wire_bytes = max(
            (item["wire_bytes"] for loads in self.edge_loads for item in loads.values()), default=1
        )

        self.fig, self.ax = plt.subplots(figsize=(22, 13))
        self.fig.subplots_adjust(bottom=0.20, right=0.765, left=0.018, top=0.94)
        _style_axes(self.ax, "G-Net: сервисные потоки по шагам динамики")
        _draw_access_zones(self.ax, model, self.pos)
        _draw_topology(self.ax, model, self.pos)
        _draw_labels(self.ax, model, self.pos)
        _set_topology_view(self.ax, self.pos)
        _draw_legend(self.ax)

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
        panel_style = {"boxstyle": "round,pad=0.55", "fc": "#172033", "ec": "#475569", "alpha": 0.96}
        self.status = self.ax.text(
            1.018, 0.98, "", transform=self.ax.transAxes, va="top", fontsize=9.0,
            color="#f8fafc", linespacing=1.25, bbox=panel_style,
        )
        self.help_text = self.ax.text(
            1.018, 0.39,
            "КАК ЧИТАТЬ СХЕМУ\n\n"
            "C — маршрутизатор ядра\n"
            "A — агрегирующий коммутатор\n"
            "Зелёный узел — сервис L0\n"
            "Фиолетовый ромб — физический сервер\n"
            "АРБ — арбитратор уровня L7\n\n"
            "Opus/RTP — голос по UDP\n"
            "MP3/RTP — аудиопоток по UDP\n\n"
            "Цвет линии — приложение\n"
            "Толщина — объём за интервал\n"
            "Точка — пакет на маршруте\n\n"
            "Наведите курсор на любой\n"
            "узел для подробной карточки.\n\n"
            "Выбор шага автоматически\n"
            "останавливает анимацию и\n"
            "фиксирует состояние снимка.",
            transform=self.ax.transAxes, va="top", fontsize=8.2, color="#dbeafe",
            linespacing=1.25, bbox=panel_style,
        )
        self.hover_note = self.ax.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
            color="#f8fafc", fontsize=8.5, zorder=30,
            bbox={"boxstyle": "round,pad=0.45", "fc": "#020617", "ec": "#38bdf8", "alpha": 0.97},
            arrowprops={"arrowstyle": "->", "color": "#38bdf8"},
        )
        self.hover_note.set_visible(False)
        self.hover_nodes = [
            node for node, attrs in model.graph.nodes(data=True)
            if attrs.get("level") in {"L0", "L1", "L2", "L7"}
        ]
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)

        slider_ax = self.fig.add_axes([0.14, 0.048, 0.49, 0.027])
        self.slider = Slider(
            slider_ax, "Шаг динамики · ПАУЗА", 0, len(usable) - 1, valinit=0, valstep=1,
            valfmt="%d", color="#38bdf8", initcolor="none"
        )
        self.slider.label.set_color("#f8fafc")
        self.slider.valtext.set_color("#f8fafc")
        tick_count = min(len(usable), 11)
        step_ticks = sorted({int(round(value)) for value in np.linspace(0, len(usable) - 1, tick_count)})
        slider_ax.set_xticks(step_ticks)
        slider_ax.set_xticklabels([str(value) for value in step_ticks], color="#cbd5e1", fontsize=7)
        slider_ax.tick_params(axis="x", length=2, pad=3, colors="#cbd5e1")
        self.slider.on_changed(self._select_step)
        self.slider.track.set_facecolor("#334155")
        button_ax = self.fig.add_axes([0.675, 0.030, 0.115, 0.055])
        self.button = Button(button_ax, "▶  Продолжить", color="#2563eb", hovercolor="#3b82f6")
        self.button.label.set_color("#f8fafc")
        self.button.label.set_fontweight("bold")
        self.button.on_clicked(self._toggle_pause)
        self._render_snapshot(update_edges=True)
        self.animation = FuncAnimation(self.fig, self._update, interval=55, blit=False, cache_frame_data=False)

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
        self.hover_note.set_text(_node_hover_text(self.model, node_id))
        self.hover_note.set_visible(True)
        self.fig.canvas.draw_idle()

    def _select_step(self, value: float) -> None:
        if self._syncing_slider:
            return
        self.step_index = int(value)
        self.frame_in_step = 0
        self.paused = True
        self._set_playback_style()
        self._render_snapshot(update_edges=True)
        self.fig.canvas.draw_idle()

    def _toggle_pause(self, _event) -> None:
        if self.paused and self.step_index == len(self.snapshots) - 1 and self.frame_in_step >= self.frames_per_step:
            self.step_index = 0
            self.frame_in_step = 0
            self._sync_slider()
        self.paused = not self.paused
        self._set_playback_style()
        self._render_snapshot(update_edges=False)
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
                self._sync_slider()
                update_edges = True
            if finished:
                self.paused = True
                self._set_playback_style(finished=True)
        return self._render_snapshot(update_edges=update_edges)

    def _sync_slider(self) -> None:
        self._syncing_slider = True
        try:
            self.slider.set_val(self.step_index)
        finally:
            self._syncing_slider = False

    def _set_playback_style(self, *, finished: bool = False) -> None:
        if finished:
            self.button.label.set_text("↻  Повторить")
            self.slider.label.set_text("Шаг динамики · ЗАВЕРШЕНО")
            self.slider.poly.set_facecolor("#22c55e")
        elif self.paused:
            self.button.label.set_text("▶  Продолжить")
            self.slider.label.set_text("Шаг динамики · ПАУЗА")
            self.slider.poly.set_facecolor("#38bdf8")
        else:
            self.button.label.set_text("Ⅱ  Пауза")
            self.slider.label.set_text("Шаг динамики · ВОСПРОИЗВЕДЕНИЕ")
            self.slider.poly.set_facecolor("#22c55e")

    def _render_snapshot(self, *, update_edges: bool):
        snapshot = self.snapshots[self.step_index]
        flows = snapshot["traffic"]["flows"]
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
        target_nodes = [
            event.get("target_id") for event in snapshot.get("attacks", {}).get("events", [])
            if event.get("target_id") in self.pos
        ]
        self.attack_targets.set_offsets(
            np.asarray([self.pos[node] for node in target_nodes]) if target_nodes else np.empty((0, 2))
        )
        self.attack_targets.set_sizes([360.0 + 35.0 * np.sin(self.frame_in_step / 4.0) for _ in target_nodes])
        self.status.set_text(_snapshot_summary(snapshot, flows, loads, self.paused, self.model))
        return (self.particle_glow, self.particles, self.attack_targets, self.status, *self.edge_artists.values())


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


# Обратная совместимость для внутренних вызовов предыдущих версий проекта.
_ensure_interactive_backend = ensure_interactive_backend


def _draw_topology(ax, model, pos, *, loads=None, max_bytes: int = 1) -> None:
    for source, target in model.graph.edges():
        x1, y1 = pos[source]
        x2, y2 = pos[target]
        load = None if loads is None else loads.get(frozenset((source, target)))
        if load:
            dominant = load["applications"].most_common(1)[0][0]
            color = FLOW_COLORS.get(dominant, "#ffbe0b")
            width = 0.8 + 5.2 * np.sqrt(load["wire_bytes"] / max_bytes)
            alpha, zorder = 0.88, 3
        else:
            color, width, alpha, zorder = "#64748b", 0.48, 0.36, 1
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=alpha, zorder=zorder)

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
    for role, marker, size in (("core-router", "o", 58), ("aggregation-switch", "s", 62)):
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
        and attrs.get("role") in {"core-router", "aggregation-switch", "service-server"}
    ]
    ax.scatter(
        [pos[node][0] for node in protected_nodes], [pos[node][1] for node in protected_nodes],
        s=175, facecolors="none", edgecolors="#fbbf24", linewidths=1.15, alpha=0.9, zorder=7,
    )


def _draw_access_zones(ax, model, pos) -> None:
    """Group subscribers by their aggregation switch without hiding links."""
    for access_node in sorted(
        node for node, attrs in model.graph.nodes(data=True) if attrs.get("role") == "aggregation-switch"
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
        access_kind = "МОБИЛЬНЫЕ" if role == "mobile-subscriber" else "ФИКСИРОВАННЫЕ"
        ax.text(
            x0 + width / 2, y0 - 0.14, f"{access_node} · {access_kind}\n{len(nodes)} АБОНЕНТОВ",
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
            label = "АРБ"
        ax.text(
            x, y + (0.34 if level == "L0" else 0.20), label, ha="center",
            fontsize=7.2, color="#f8fafc", zorder=7,
        )


def _style_axes(ax, title: str) -> None:
    ax.figure.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
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


def _draw_legend(ax, *, static: bool = False) -> None:
    flow_labels = FLOW_LABELS if static else FLOW_SHORT_LABELS
    handles = [Line2D([0], [0], color=color, lw=3, label=flow_labels[name]) for name, color in FLOW_COLORS.items()]
    for style in SUBSCRIBER_STYLES.values():
        handles.append(Line2D(
            [0], [0], marker=style["marker"], color="none", markerfacecolor=style["color"],
            markeredgecolor="#0f172a", markersize=7, label=style["label"],
        ))
    handles.extend([
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LEVEL_COLORS["L0"], markersize=8, label="Сервис L0"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#a78bfa", markersize=7, label="Физический сервер L0"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LEVEL_COLORS["L2"], markersize=7, label="C — маршрутизатор ядра"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=LEVEL_COLORS["L2"], markersize=7, label="A — агрегирующий коммутатор"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LEVEL_COLORS["L7"], markersize=7, label="АРБ — арбитратор L7"),
        Line2D([0], [0], marker="o", color="#fbbf24", markerfacecolor="none", markersize=9, label="Критически важный Gold-узел"),
        Line2D([0], [0], marker="o", color="#ff1744", markerfacecolor="none", markersize=10, label="Текущая цель атаки"),
        Line2D([0], [0], color="#94a3b8", lw=0.7, alpha=0.7, label="Связь без трафика"),
    ])
    legend_options = (
        {"loc": "lower left", "bbox_to_anchor": (1.018, 0.02), "ncol": 1}
        if static else {"loc": "lower center", "bbox_to_anchor": (0.5, -0.095), "ncol": 4}
    )
    legend = ax.legend(
        handles=handles, framealpha=0.96, facecolor="#111827", edgecolor="#475569",
        labelcolor="#e2e8f0", fontsize=7.6 if static else 7.25,
        title="Узлы и сервисные потоки", title_fontsize=8.3, **legend_options,
    )
    legend.get_title().set_color("#f8fafc")
    legend.set_zorder(20)


def _flow_summary(flows: list[dict[str, Any]]) -> str:
    apps = Counter(flow.get("application", "unknown") for flow in flows)
    legitimate_count = sum(1 for flow in flows if not flow.get("is_attack_traffic"))
    attack_count = len(flows) - legitimate_count
    packets = sum(int(flow.get("packet_count", 0)) for flow in flows)
    wire_bytes = sum(int(flow.get("wire_bytes", 0)) for flow in flows)
    interval = int(flows[0].get("interval_seconds", 0)) if flows else 0
    megabytes = wire_bytes / 1_000_000
    rate_mbps = wire_bytes * 8.0 / max(interval, 1) / 1_000_000
    lines = [
        f"Легитимные потоки: {legitimate_count} = {legitimate_count} абонентов",
        "По 1 активному потоку на пользователя",
        f"Атакующие агрегированные потоки: {attack_count}",
        f"Пакеты за интервал {interval} с: {packets:,}",
        f"Сформировано: {megabytes:,.1f} МБ",
        f"Это {megabytes * 8:,.1f} Мбит за {interval} с",
        f"Суммарная канальная скорость: {rate_mbps:,.1f} Мбит/с",
        "Приложения:",
    ]
    lines.extend(f"  {FLOW_SHORT_LABELS.get(name, name)}: {count}" for name, count in sorted(apps.items()))
    return "\n".join(lines)


def _snapshot_summary(snapshot, flows, loads, paused: bool, model) -> str:
    latencies = [float(flow.get("one_way_latency_ms", 0.0)) for flow in flows]
    busiest = max(loads.items(), key=lambda item: item[1]["wire_bytes"], default=(frozenset(), {"wire_bytes": 0}))
    edge_name = " ↔ ".join(sorted(busiest[0])) if busiest[0] else "—"
    remap_code = snapshot.get("arbitrator", {}).get("remap", {}).get("action", "—")
    remap = {"NO_REMAP": "переназначение не требуется", "PLAN_REMAP": "требуется план переназначения"}.get(remap_code, remap_code)
    mode = "ПАУЗА — выбранный снимок" if paused else f"ВОСПРОИЗВЕДЕНИЕ → следующий шаг"
    t0_note = "\nt0: пакеты находятся в точках-источниках" if paused and snapshot.get("step_index", 0) == 0 else ""
    breakdown = _active_link_breakdown(model, loads)
    attacks = snapshot.get("attacks", {})
    attack_names = ", ".join(event.get("name_ru", event.get("attack_id", "")) for event in attacks.get("events", []))
    attack_note = (
        f"\n\nАТАКА АКТИВНА: {attack_names}\n"
        f"Вредоносная скорость: {attacks.get('attack_rate_mbps', 0.0):,.1f} Мбит/с\n"
        f"Потери легитимных пакетов: {attacks.get('legitimate_loss_ratio', 0.0) * 100:.2f}%\n"
        f"Затронуто Gold-потоков: {attacks.get('impacted_gold_flow_count', 0)}"
        if attacks.get("active") else "\n\nАтаки: нет"
    )
    return (
        f"ШАГ {snapshot.get('step_index', 0)} · t = {snapshot.get('time_seconds', 0)} с\n"
        f"{mode}{t0_note}\n\n{_flow_summary(flows)}\n"
        f"Используемых связей: {len(loads)} из {model.graph.number_of_edges()}\n"
        f"  абонент → коммутатор A: {breakdown['access']}\n"
        f"  коммутатор A → маршрутизатор C: {breakdown['uplink']}\n"
        f"  маршрутизатор C → маршрутизатор C: {breakdown['core']}\n"
        f"  маршрутизатор C → сервер: {breakdown['service']}\n"
        f"Средняя задержка: {np.mean(latencies) if latencies else 0.0:.2f} мс\n"
        f"Макс. задержка: {max(latencies, default=0.0):.2f} мс\n\n"
        f"Самая нагруженная связь:\n{edge_name}\n"
        f"{busiest[1]['wire_bytes'] / 1_000_000:.1f} МБ за шаг\n\n"
        f"Решение L7: {remap}{attack_note}"
    )


def _active_link_breakdown(model, loads) -> dict[str, int]:
    counts = {"access": 0, "uplink": 0, "core": 0, "service": 0, "other": 0}
    for edge in loads:
        nodes = tuple(edge)
        attrs = [model.graph.nodes[node] for node in nodes]
        levels = {item.get("level") for item in attrs}
        roles = {item.get("role") for item in attrs}
        if "L1" in levels:
            counts["access"] += 1
        elif roles == {"aggregation-switch", "core-router"}:
            counts["uplink"] += 1
        elif roles == {"core-router"}:
            counts["core"] += 1
        elif "L0" in levels:
            counts["service"] += 1
        else:
            counts["other"] += 1
    return counts


def _node_hover_text(model, node_id: str) -> str:
    attrs = model.graph.nodes[node_id]
    role = attrs.get("role")
    if role in {"core-router", "aggregation-switch"}:
        return _equipment_hover_text(model, node_id, attrs)
    if role == "service-server":
        profile = attrs.get("server_profile", {})
        runtime = attrs.get("runtime", {})
        return (
            f"{node_id} · Физический сервер L0\n"
            f"IP: {attrs.get('ip_address')} · MAC: {attrs.get('mac_address')}\n"
            f"Модель: {profile.get('model')}\n"
            f"ЦП: {profile.get('cpu')} · {profile.get('total_cores')} ядер\n"
            f"ОЗУ: {profile.get('ram_gb')} ГБ · хранилище: {profile.get('storage_tb')} ТБ\n"
            f"Сетевые порты: {profile.get('network_ports_gbps')} Гбит/с\n"
            f"Сервисы: {', '.join(attrs.get('hosted_services', []))}\n"
            f"Текущая нагрузка: ЦП {runtime.get('cpu_util_percent', 0):.1f}%, "
            f"ОЗУ {runtime.get('ram_util_percent', 0):.1f}%, сеть {runtime.get('network_util_percent', 0):.1f}%\n"
            f"Диски {runtime.get('storage_util_percent', 0):.1f}%, "
            f"температура {runtime.get('temperature_c', 0):.1f} °C, сеансов {runtime.get('active_sessions', 0)}"
            f"{_protection_hover(attrs)}"
        )
    if attrs.get("level") == "L1":
        kind = "Мобильный абонент" if role == "mobile-subscriber" else "Фиксированный абонент"
        return (
            f"{node_id} · {kind}\n"
            f"IP: {attrs.get('ip_address')} · MAC: {attrs.get('mac_address')}\n"
            f"Коммутатор доступа: {attrs.get('home_access')}\n"
            f"Класс SLA: {SLA_LABELS.get(attrs.get('sla_grade'), attrs.get('sla_grade'))}\n"
            f"Приложение: {TRAFFIC_LABELS.get(attrs.get('traffic_kind'), attrs.get('traffic_kind'))}\n"
            f"Кодек/профиль: {attrs.get('codec_profile_name') or attrs.get('codec')}\n"
            f"Целевая скорость: {attrs.get('target_bitrate_kbps', 0):.0f} кбит/с"
            f"{_protection_hover(attrs)}"
        )
    if attrs.get("level") == "L0":
        service_name = SERVICE_DISPLAY_NAMES.get(attrs.get("label"), attrs.get("label"))
        return (
            f"{node_id} · Логический сервис L0\nНазначение: {service_name}\n"
            f"VIP: {attrs.get('ip_address')} · MAC: {attrs.get('mac_address')}\n"
            f"Платформа: {attrs.get('platform')}\nСервер: {attrs.get('hosted_on')}\n"
            f"Профиль: {attrs.get('codec_profile_name')}\n"
            f"Аудиокодек: {attrs.get('audio_codec') or '—'} · Видеокодек: {attrs.get('video_codec') or '—'}"
        )
    if role == "arbitrator":
        return "АРБ · Арбитратор L7\nНаблюдает метрики всех уровней\nПринимает решение о переназначении"
    return f"{node_id}\nРоль: {role}\nУровень: {attrs.get('level')}"


def _equipment_hover_text(model, node_id: str, attrs: dict[str, Any]) -> str:
    role = attrs.get("role")
    device_name = "Маршрутизатор ядра" if role == "core-router" else "Агрегирующий коммутатор"
    raw = attrs.get("l2_raw_baseline", {})
    profile = attrs.get("l2_profile", {})
    neighbors = list(model.graph.neighbors(node_id))
    clients = [node for node in neighbors if model.graph.nodes[node].get("level") == "L1"]
    uplinks = [node for node in neighbors if model.graph.nodes[node].get("level") in {"L0", "L2"}]
    verified = set(profile.get("verified_fields", ()))
    dram_origin = "Cisco" if "dram_gb" in verified else "модель"
    forwarding_origin = "Cisco" if "forwarding_mpps" in verified else "модель"
    return (
        f"{node_id} · {device_name}\n"
        f"IP: {attrs.get('ip_address')} · MAC: {attrs.get('mac_address')}\n"
        f"Платформа: {attrs.get('platform_family')}\n"
        f"Профиль: {attrs.get('platform_profile')}\n"
        f"Базовая загрузка ЦП: {raw.get('cpu_util', 0):.1f}% [модель]\n"
        f"Базовая загрузка ОЗУ: {raw.get('ram_util', 0):.1f} / {profile.get('dram_gb', 0):.1f} ГБ [{dram_origin}]\n"
        f"Системная пропускная способность: {profile.get('throughput_gbps', 0):,.0f} Гбит/с [Cisco]\n"
        f"Скорость пересылки: {profile.get('forwarding_mpps', 0):,.0f} млн пакетов/с [{forwarding_origin}]\n"
        f"Порты: {profile.get('port_configuration', '—')} [Cisco]\n"
        f"Подключений: {len(neighbors)}"
        + (f" · клиентов: {len(clients)}" if clients else "")
        + f"\nСвязи: {', '.join(uplinks)}"
        + _protection_hover(attrs)
    )


def _protection_hover(attrs: dict[str, Any]) -> str:
    protection = attrs.get("critical_protection", {})
    if not protection.get("is_critical"):
        return ""
    return (
        "\nКРИТИЧЕСКИЙ GOLD-УЗЕЛ · приоритет защиты 1"
        f"\nGold-потоков через узел: {protection.get('gold_transit_flow_count', 0)}"
        f" · КВУ: {protection.get('critical_involvement_coefficient', 0.0):.3f}"
        f"\nПредел загрузки при переназначении: {protection.get('maximum_safe_utilization_percent', 80.0):.0f}%"
    )


def _spread_sample(flows: list[dict[str, Any]], limit: int, frame: int) -> list[dict[str, Any]]:
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
    frames_per_step: int = 48,
) -> float:
    """Return route progress while preserving the semantics of the t0 snapshot."""
    if step_index == 0 and paused:
        return 0.0
    spacing = offset / max(1, particle_count)
    if paused:
        return (step_index * 0.173 + spacing) % 1.0
    phase = min(1.0, max(0.0, frame / max(frames_per_step, 1)))
    return (phase + spacing) % 1.0


def _point_on_route(route: list[str], pos: dict[str, tuple[float, float]], progress: float) -> tuple[float, float] | None:
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
