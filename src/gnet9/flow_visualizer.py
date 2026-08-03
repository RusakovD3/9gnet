"""Читаемая статическая и интерактивная визуализация сервисного трафика."""

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
    "ATTACK_DOS": "АТАКА · DoS / прямой поток",
    "ATTACK_DDOS": "АТАКА · DDoS / отражённое усиление",
    "ATTACK_SYN": "АТАКА · SYN-флуд",
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
    "ATTACK_SYN": "АТАКА · SYN-флуд",
}
TRAFFIC_LABELS = {
    "voice": "голос", "vlc_av": "VLC: голос и видео",
    "ftp": "передача файлов FTP", "dns": "запросы DNS",
    "video_conference": "видеоконференция", "live_streaming": "прямая трансляция",
}
SLA_LABELS = {"gold": "золотой", "silver": "серебряный", "bronze": "бронзовый"}
SLO_METRIC_LABELS = {
    "one_way_mouth_to_ear_latency": "односторонняя задержка речи",
    "network_packet_loss": "сетевая потеря пакетов",
    "jitter": "джиттер",
    "one_way_media_latency": "односторонняя задержка медиапотока",
    "media_bitrate": "полезный битрейт медиапотока",
    "goodput": "полезная скорость передачи",
    "file_completion_time": "время передачи файла",
    "transfer_success": "успешность передачи",
    "tcp_retransmission_ratio": "доля повторных передач TCP",
    "response_time_p95_p99": "время ответа p95/p99",
    "timeout_ratio": "доля тайм-аутов",
    "servfail_ratio": "доля ответов SERVFAIL",
    "tcp_fallback_success": "успешность перехода DNS на TCP",
    "audio_one_way_latency": "односторонняя задержка звука",
    "video_one_way_latency": "односторонняя задержка видео",
    "media_loss": "потеря медиапакетов",
    "live_edge_latency": "отставание от прямого эфира",
    "startup_time": "время запуска воспроизведения",
    "rebuffer_ratio": "доля времени ребуферизации",
    "part_deadline_miss_ratio": "доля пропущенных сроков LL-HLS",
}
MODEL_LIMITATION_LABELS = {
    "latency_and_jitter_fields_are_compatibility_guardrails_not_primary_ftp_slo": (
        "задержка и джиттер для FTP — вспомогательные ограничения, а не основные SLO"
    ),
    "bitrate_is_equivalent_load_for_generator_not_a_dns_service_slo": (
        "битрейт DNS — эквивалент нагрузки генератора, а не SLO сервиса"
    ),
    "srtp_srtcp_and_dtls_byte_overhead_not_yet_counted": (
        "служебные байты SRTP/SRTCP и DTLS пока не входят в расчёт трафика"
    ),
    "jitter_field_is_a_transport_guardrail_not_primary_ll_hls_user_slo": (
        "джиттер — транспортное ограничение, а не основной пользовательский SLO LL-HLS"
    ),
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

    # Render each 40-user access group as a compact 10 x 4 endpoint rack.
    # The access devices are deliberately close in the engineering layout.  The
    # previous 8-column rack was wider than the gap between neighbouring RAN/OLT
    # nodes, so two subscriber groups visually crossed each other.  A four-column
    # rack preserves a generous pixel gap between every device group while the
    # ten rows remain easy to inspect in the enlarged figure.
    # This changes only the display layout; graph coordinates and metrics stay intact.
    access_groups: dict[str, list[str]] = {}
    for node_id, attrs in model.graph.nodes(data=True):
        if attrs.get("level") == "L1":
            access_groups.setdefault(attrs["home_access"], []).append(node_id)
    for access_node, node_ids in access_groups.items():
        center_x, access_y = positions[access_node]
        for index, node_id in enumerate(sorted(node_ids)):
            row, column = divmod(index, 4)
            positions[node_id] = (
                center_x + (column - 1.5) * 0.28,
                access_y - 1.28 - row * 0.42,
            )
    return positions


def aggregate_flow_edges(flows: list[dict[str, Any]]) -> dict[frozenset[str], dict[str, Any]]:
    """Суммировать потоки, байты и приложения на каждой пройденной связи."""
    result: dict[frozenset[str], dict[str, Any]] = {}
    for flow in flows:
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
    """Сохранить полную топологию с фактической нагрузкой поверх связей."""
    pos = _positions(model)
    loads = aggregate_flow_edges(flows)
    max_bytes = max((item["wire_bytes"] for item in loads.values()), default=1)
    fig, ax = plt.subplots(figsize=(36, 17))
    fig.subplots_adjust(bottom=0.055, left=0.014, right=0.770, top=0.94)
    title = "G-Net: сервисные потоки и загрузка связей"
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
    _draw_legend(ax, static=True)
    ax.text(
        1.018, 0.98, _flow_summary(flows), transform=ax.transAxes, va="top", fontsize=9.8, color="#e9ecef",
        bbox={"boxstyle": "round,pad=0.5", "fc": "#212529", "ec": "#6c757d", "alpha": 0.92},
    )
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


class ServiceFlowWindow:
    """Интерактивное окно трафика с выбором и просмотром точного шага."""

    def __init__(self, model, snapshots: list[dict[str, Any]], *, particles_per_frame: int = 52) -> None:
        usable = [snapshot for snapshot in snapshots if snapshot.get("traffic", {}).get("flows")]
        if not usable:
            raise ValueError("Для интерактивного окна требуется packet_detail='flows' или 'sample'")
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
        self.edge_loads = [aggregate_flow_edges(snapshot["traffic"]["flows"]) for snapshot in usable]
        self.max_wire_bytes = max(
            (item["wire_bytes"] for loads in self.edge_loads for item in loads.values()), default=1
        )

        # The main network area is intentionally wider than the side panels:
        # 480 endpoints are readable without zooming into a single access group.
        self.fig, self.ax = plt.subplots(figsize=(28, 16))
        self.fig.subplots_adjust(bottom=0.125, right=0.770, left=0.018, top=0.94)
        self.status_ax = self.fig.add_axes([0.790, 0.455, 0.200, 0.485])
        self.legend_ax = self.fig.add_axes([0.790, 0.245, 0.200, 0.185])
        self.help_ax = self.fig.add_axes([0.790, 0.065, 0.200, 0.150])
        for panel_ax in (self.status_ax, self.legend_ax, self.help_ax):
            panel_ax.set_axis_off()
            panel_ax.set_facecolor("none")
        _style_axes(self.ax, "G-Net: сервисные потоки по шагам динамики")
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
        self.status = self.status_ax.text(
            0.0, 1.0, "", transform=self.status_ax.transAxes, va="top", fontsize=7.15,
            color="#f8fafc", linespacing=1.06, bbox=panel_style, wrap=True,
        )
        self.help_text = self.help_ax.text(
            0.0, 1.0,
            "КАК ЧИТАТЬ\n\n"
            "Цвет магистрали — ведущий сервис;\n"
            "линия абонента окрашена как M или F.\n"
            "Толщина — объём за шаг, точка — пакет.\n"
            "Красное кольцо — атака; фиолетовое — прогноз.\n\n"
            "Наведите на узел для карточки.\n"
            "Шкала выбирает снимок и ставит паузу.",
            transform=self.help_ax.transAxes, va="top", fontsize=7.15, color="#dbeafe",
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

        slider_ax = self.fig.add_axes([0.12, 0.045, 0.52, 0.027])
        self.slider = Slider(
            slider_ax, "Шаг", 0, len(usable) - 1, valinit=0, valstep=1,
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
            0.12, 0.090, "", ha="left", va="center",
            color="#e2e8f0", fontsize=8.5, fontweight="bold",
        )
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
        self.hover_note.set_text(_node_hover_text(self.model, node_id))
        self.hover_note.set_visible(True)
        self.fig.canvas.draw_idle()

    def _select_step(self, value: float) -> None:
        if self._syncing_slider:
            return
        self.step_index = int(value)
        self.frame_in_step = (
            0
            if self.step_index == 0
            else int(round(((self.step_index * 0.173) % 1.0) * self.frames_per_step))
        )
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
        finally:
            self._syncing_slider = False

    def _set_playback_style(self, *, finished: bool = False) -> None:
        snapshot = self.snapshots[self.step_index]
        actual_step = int(snapshot.get("step_index", self.step_index))
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
        self.status.set_text(_interactive_snapshot_summary(snapshot, flows, loads, self.paused, self.model))
        return (
            self.particle_glow,
            self.particles,
            self.attack_targets,
            self.warning_targets,
            self.isolated_nodes,
            self.failover_servers,
            self.status,
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


# Обратная совместимость для внутренних вызовов предыдущих версий проекта.
_ensure_interactive_backend = ensure_interactive_backend


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


def _draw_legend(ax, *, static: bool = False, compact: bool = False) -> None:
    flow_labels = FLOW_LABELS if static else FLOW_SHORT_LABELS
    flow_handles = [Line2D([0], [0], color=color, lw=3, label=flow_labels[name]) for name, color in FLOW_COLORS.items()]
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
    if compact:
        handles = [
            *flow_handles[:6],
            *subscriber_handles,
            semantic_handles[1], semantic_handles[2], semantic_handles[3],
            semantic_handles[7], semantic_handles[8], semantic_handles[11],
            semantic_handles[12], semantic_handles[13],
        ]
        legend_options = {"loc": "upper left", "bbox_to_anchor": (0.0, 1.0), "ncol": 1, "borderaxespad": 0.0}
    else:
        handles = [*flow_handles, *subscriber_handles, *semantic_handles]
        legend_options = (
        {"loc": "lower left", "bbox_to_anchor": (1.018, 0.015), "ncol": 2, "columnspacing": 1.0}
        if static else {"loc": "lower center", "bbox_to_anchor": (0.5, -0.105), "ncol": 5, "columnspacing": 1.1}
        )
    legend = ax.legend(
        handles=handles, framealpha=0.96, facecolor="#111827", edgecolor="#475569",
        labelcolor="#e2e8f0", fontsize=7.15 if static else 6.7 if compact else 7.05,
        title="Условные обозначения" if compact else "Узлы и сервисные потоки",
        title_fontsize=7.6 if compact else 8.3, **legend_options,
    )
    legend.get_title().set_color("#f8fafc")
    legend.set_zorder(20)


def _flow_summary(flows: list[dict[str, Any]]) -> str:
    apps = Counter(flow.get("application", "unknown") for flow in flows)
    legitimate_count = sum(1 for flow in flows if not flow.get("is_attack_traffic"))
    attack_count = len(flows) - legitimate_count
    packets = sum(int(flow.get("packet_count", 0)) for flow in flows)
    wire_bytes = sum(int(flow.get("wire_bytes", 0)) for flow in flows)
    routed_flows = [
        flow for flow in flows
        if flow.get("route") and not flow.get("isolated") and int(flow.get("packet_count", 0)) > 0
    ]
    routed_wire_bytes = sum(int(flow.get("wire_bytes", 0)) for flow in routed_flows)
    delivered_packets = sum(
        max(0, int(flow.get("packet_count", 0)) - int(flow.get("observed_dropped_packets", 0)))
        for flow in routed_flows
    )
    interval = int(flows[0].get("interval_seconds", 0)) if flows else 0
    megabytes = wire_bytes / 1_000_000
    offered_rate_mbps = wire_bytes * 8.0 / max(interval, 1) / 1_000_000
    carried_rate_mbps = routed_wire_bytes * 8.0 / max(interval, 1) / 1_000_000
    rerouted = sum(bool(flow.get("rerouted")) for flow in flows if not flow.get("is_attack_traffic"))
    failover = sum(bool(flow.get("failover_active")) for flow in flows if not flow.get("is_attack_traffic"))
    isolated = sum(bool(flow.get("isolated")) for flow in flows if not flow.get("is_attack_traffic"))
    lines = [
        f"Легитимные потоки: {legitimate_count} = {legitimate_count} абонентов",
        "По 1 активному потоку на пользователя",
        f"Атакующие агрегированные потоки: {attack_count}",
        f"Сформировано пакетов за {interval} с: {packets:,}",
        f"Доставлено пакетов: {delivered_packets:,}",
        f"Предложено: {megabytes:,.1f} МБ · {offered_rate_mbps:,.1f} Мбит/с",
        f"Передано по маршрутам: {routed_wire_bytes / 1_000_000:,.1f} МБ · {carried_rate_mbps:,.1f} Мбит/с",
        f"Переназначено / резерв / изолировано: {rerouted} / {failover} / {isolated}",
        "Приложения:",
    ]
    lines.extend(f"  {FLOW_SHORT_LABELS.get(name, name)}: {count}" for name, count in sorted(apps.items()))
    return "\n".join(lines)


def _interactive_snapshot_summary(snapshot, flows, loads, paused: bool, model) -> str:
    """Краткая карточка для окна: она всегда помещается в выделенную панель."""
    legitimate = [flow for flow in flows if not flow.get("is_attack_traffic")]
    active_attacks = snapshot.get("attacks", {})
    koopman = snapshot.get("koopman", {})
    routing = active_attacks.get("routing", {})
    by_sla = routing.get("by_sla", {})
    latencies = [
        float(flow.get("one_way_latency_ms", 0.0))
        for flow in legitimate
        if not flow.get("isolated")
    ]
    mode = "ПАУЗА" if paused else "ВОСПРОИЗВЕДЕНИЕ"
    step = int(snapshot.get("step_index", 0))
    time_seconds = float(snapshot.get("time_seconds", 0.0))
    remap = snapshot.get("arbitrator", {}).get("remap", {}).get("action", "—")
    attack_names = ", ".join(
        str(event.get("name_ru") or event.get("attack_id") or "атака")
        for event in active_attacks.get("events", [])[:2]
    )
    if len(active_attacks.get("events", [])) > 2:
        attack_names += f" и ещё {len(active_attacks['events']) - 2}"
    warning = bool(koopman.get("forecast_is_early_warning"))
    targets = ", ".join(map(str, koopman.get("forecast_target_ids", [])[:3])) or "—"
    status = "ПРОГНОЗ ПОДТВЕРЖДЁН" if warning else "признаков атаки нет"
    gold = by_sla.get("gold", {})
    max_link = max(
        (item.get("wire_bytes", 0) for item in loads.values()),
        default=0,
    ) / 1_000_000
    lines = [
        f"ШАГ {step} · t = {time_seconds:.0f} с · {mode}",
        "─" * 30,
        f"Легитимных потоков: {len(legitimate)}",
        f"Связей с трафиком: {len(loads)} из {model.graph.number_of_edges()}",
        f"Средняя / макс. задержка: {np.mean(latencies) if latencies else 0.0:.2f} / {max(latencies, default=0.0):.2f} мс",
        f"Макс. объём на связи: {max_link:.1f} МБ за шаг",
        "",
        f"КУПМАН: {status}",
        f"Риск: {float(koopman.get('forecast_risk_score', 0.0)) * 100:.1f}% (не вероятность)",
        f"Цель: {targets}",
        f"Окно: {koopman.get('forecast_window_start_seconds', '—')}…{koopman.get('forecast_window_end_seconds', '—')} с",
        f"Запас: ≥ {koopman.get('prediction_slo_seconds', '—')} с",
        "",
        f"L7: {remap}",
        f"Переназначено / резерв сервера / изолировано: {routing.get('rerouted_flow_count', 0)} / {routing.get('failover_flow_count', 0)} / {routing.get('isolated_flow_count', 0)}",
        f"Доступность Gold: {float(gold.get('availability_ratio', 1.0)) * 100:.1f}%",
    ]
    if active_attacks.get("active"):
        lines.extend([
            "",
            f"АТАКА: {attack_names}",
            f"После защиты: {float(active_attacks.get('attack_rate_mbps', 0.0)):,.1f} Мбит/с",
            f"Этап: {routing.get('protection_stage', 'подтверждение/защита')}",
        ])
    elif step == 0:
        lines.append("t0: эталон без потерь, очередей и ремаппинга")
    return "\n".join(lines)


def _snapshot_summary(snapshot, flows, loads, paused: bool, model) -> str:
    latencies = [
        float(flow.get("one_way_latency_ms", 0.0)) for flow in flows
        if not flow.get("is_attack_traffic") and not flow.get("isolated")
    ]
    busiest = max(loads.items(), key=lambda item: item[1]["wire_bytes"], default=(frozenset(), {"wire_bytes": 0}))
    edge_name = " ↔ ".join(sorted(busiest[0])) if busiest[0] else "—"
    remap_code = snapshot.get("arbitrator", {}).get("remap", {}).get("action", "—")
    remap = {
        "NO_REMAP": "переназначение не требуется",
        "OBSERVE_PRECURSOR": "наблюдение: требуется второй отсчёт",
        "PLAN_REMAP": "выполняется план переназначения",
    }.get(remap_code, remap_code)
    mode = "ПАУЗА — выбранный снимок" if paused else f"ВОСПРОИЗВЕДЕНИЕ → следующий шаг"
    t0_note = "\nt0: пакеты находятся в точках-источниках" if paused and snapshot.get("step_index", 0) == 0 else ""
    breakdown = _active_link_breakdown(model, loads)
    attacks = snapshot.get("attacks", {})
    attack_names = ", ".join(event.get("name_ru", event.get("attack_id", "")) for event in attacks.get("events", []))
    koopman = snapshot.get("koopman", {})
    observed_targets = [
        str(item.get("sensor_node_id") or item.get("suspected_target_id") or "цель")
        for item in attacks.get("precursors", [])
    ]
    forecast_targets = [str(node) for node in koopman.get("forecast_target_ids", [])]
    precursor_targets = ", ".join(dict.fromkeys(observed_targets or forecast_targets)) or "не локализована"
    kind_labels = {
        "dos": "DoS",
        "ddos": "DDoS",
        "syn_flood": "SYN-флуд",
        "power_attack": "деградация питания",
    }
    raw_kinds = str(koopman.get("forecast_next_attack_kind", "")).split("+")
    predicted_kind = " + ".join(kind_labels.get(kind, "аномальное воздействие") for kind in raw_kinds)
    precursor_names = f"{predicted_kind} → область {precursor_targets}"
    forecast_window = (
        f"{koopman.get('forecast_window_start_seconds', '—')}…"
        f"{koopman.get('forecast_window_end_seconds', '—')} с"
    )
    warning_note = (
        "\n\nПОДТВЕРЖДЁННЫЙ РАННИЙ ПРОГНОЗ КУПМАНА\n"
        f"Окно прогноза: {forecast_window} · максимум "
        f"{koopman.get('maximum_forecast_horizon_seconds', 0)} с\n"
        f"Целевой запас на решение: не менее {koopman.get('prediction_slo_seconds', 0)} с\n"
        f"Оценка риска (не вероятность): {koopman.get('forecast_risk_score', 0.0) * 100:.1f}%\n"
        f"Ожидается: {precursor_names}\n"
        "Защита подготовлена; приоритет: золотой → серебряный → бронзовый"
        if koopman.get("forecast_is_early_warning")
        else ""
    )
    routing = attacks.get("routing", {})
    by_sla = routing.get("by_sla", {})
    slo_tiers = {
        str(tier.get("sla_grade", "")): tier
        for tier in attacks.get("sla_restoration", {}).get("tiers", [])
    }
    route_hausdorff = routing.get("gold_route_hausdorff") or koopman.get("route_hausdorff", {})
    restoration_values = [
        by_sla.get(grade, {}).get("restoration_ratio", 1.0) * 100.0
        for grade in ("gold", "silver", "bronze")
    ]
    slo_compliance_values: list[float] = []
    slo_coverage_values: list[float] = []
    for grade in ("gold", "silver", "bronze"):
        tier = slo_tiers.get(grade, {})
        flow_count = int(tier.get("flow_count", 0))
        slo_compliance_values.append(
            100.0
            if flow_count <= 0
            else float(tier.get("sla_compliant_flow_count", 0)) / flow_count * 100.0
        )
        slo_coverage_values.append(
            float(tier.get("mean_slo_assessment_coverage_ratio", 1.0)) * 100.0
        )
    slo_note = (
        "\n\nПРИКЛАДНЫЕ ТРЕБОВАНИЯ SLO"
        f"\nСоблюдены, золотой/серебряный/бронзовый: "
        f"{slo_compliance_values[0]:.1f}% / {slo_compliance_values[1]:.1f}% / "
        f"{slo_compliance_values[2]:.1f}%"
        f"\nОхват измеряемых метрик: {slo_coverage_values[0]:.1f}% / "
        f"{slo_coverage_values[1]:.1f}% / {slo_coverage_values[2]:.1f}%"
    )
    routing_note = (
        "\n\nУПРАВЛЯЕМАЯ КОНФИГУРАЦИЯ\n"
        f"Переназначено: {routing.get('rerouted_flow_count', 0)} · "
        f"резерв: {routing.get('failover_flow_count', 0)} · "
        f"изолировано: {routing.get('isolated_flow_count', 0)}\n"
        f"Доступность золотой/серебряный/бронзовый: "
        f"{by_sla.get('gold', {}).get('availability_ratio', 1.0) * 100:.1f}% / "
        f"{by_sla.get('silver', {}).get('availability_ratio', 1.0) * 100:.1f}% / "
        f"{by_sla.get('bronze', {}).get('availability_ratio', 1.0) * 100:.1f}%\n"
        f"Восстановлено среди затронутых: "
        f"{restoration_values[0]:.1f}% / {restoration_values[1]:.1f}% / {restoration_values[2]:.1f}%\n"
        f"Макс. ресурс после переназначения: "
        f"{routing.get('maximum_projected_utilization_percent', 0.0):.2f}% (предел 80%)\n"
        f"  сеть — канал {routing.get('maximum_projected_edge_utilization_percent', 0.0):.2f}%, "
        f"узел {routing.get('maximum_projected_node_utilization_percent', 0.0):.2f}%\n"
        f"  сервер — ЦП {routing.get('maximum_projected_server_cpu_utilization_percent', 0.0):.2f}%, "
        f"ОЗУ {routing.get('maximum_projected_server_ram_utilization_percent', 0.0):.2f}%, "
        f"сеансы {routing.get('maximum_projected_server_session_utilization_percent', 0.0):.2f}%\n"
        f"Отклонено кандидатов: сеть {routing.get('network_capacity_rejected_candidate_count', 0)}, "
        f"сервер {routing.get('server_resource_rejected_candidate_count', 0)}\n"
        f"Хаусдорф маршрутов золотого класса: {route_hausdorff.get('normalized_distance', 0.0):.3f}\n"
        f"Ляпунов: V={koopman.get('lyapunov_value', 0.0):.4f} · "
        f"dV/dt={koopman.get('lyapunov_derivative_per_second', 0.0):+.4f} 1/с"
        if routing.get("plan_active") or routing.get("rerouted_flow_count") or routing.get("isolated_flow_count")
        else ""
    )
    defense = attacks.get("preventive_defense", {})
    defense_note = (
        f"\nПредотвращено: {attacks.get('blocked_attack_rate_mbps', 0.0):,.1f} Мбит/с вредоносного трафика"
        f"; {attacks.get('prevented_legitimate_dropped_packets', 0):,} потерь пакетов"
        if defense.get("effective")
        else ""
    )
    packet_attack_active = any(
        str(event.get("kind")) != "power_attack"
        for event in attacks.get("events", [])
    )
    attack_rate_note = (
        f"Вредоносная скорость: {attacks.get('attack_rate_mbps', 0.0):,.1f} Мбит/с "
        f"из {attacks.get('raw_attack_rate_mbps', 0.0):,.1f} Мбит/с до защиты\n"
        if packet_attack_active
        else "Вредоносный канальный поток отсутствует: воздействие идёт через управление питанием\n"
    )
    attack_note = (
        f"\n\nАТАКА АКТИВНА: {attack_names}\n"
        f"{_active_attack_details(attacks.get('events', []))}"
        f"{attack_rate_note}"
        f"Потери легитимных пакетов: {attacks.get('legitimate_loss_ratio', 0.0) * 100:.2f}%\n"
        f"Затронуто потоков золотого класса: {attacks.get('impacted_gold_flow_count', 0)}{defense_note}"
        if attacks.get("active") else "\n\nАтаки: нет"
    )
    power_runtime_note = _power_runtime_summary(attacks.get("power_runtime_state", []))
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
        f"Решение L7: {remap}{slo_note}{warning_note}{routing_note}{attack_note}{power_runtime_note}"
    )


def _active_attack_details(events: list[dict[str, Any]]) -> str:
    """Кратко показать физический смысл активного воздействия."""
    lines: list[str] = []
    kind_labels = {
        "dos": "DoS",
        "ddos": "DDoS",
        "syn_flood": "SYN-флуд",
        "power_attack": "деградация питания",
    }
    for event in events[:3]:
        technical = event.get("technical", {})
        ingress = event.get("ingress_nodes", [])
        sources = ", ".join(str(node) for node in ingress[:3])
        if len(ingress) > 3:
            sources += f" и ещё {len(ingress) - 3}"
        protocol = str(technical.get("protocol", "—"))
        source_text = sources or "канал управления"
        lines.append(
            f"• {kind_labels.get(str(event.get('kind')), 'атака')}: "
            f"{source_text} → {event.get('target_id', '—')} · {protocol}"
        )
        offered_rate = float(event.get("effective_offered_rate_mbps", 0.0) or 0.0)
        packet_rate = int(event.get("effective_packet_rate_pps", 0) or 0)
        packet_size = int(
            technical.get("on_wire_size_bytes")
            or technical.get("packet_size_bytes", 0)
            or 0
        )
        if offered_rate > 0.0 or packet_rate > 0:
            lines.append(
                f"  {offered_rate:,.1f} Мбит/с · {packet_rate / 1000:,.1f} тыс. пакетов/с"
                + (f" · на линии {packet_size} байт/пакет" if packet_size else "")
            )
            lines.append(
                f"  входная ёмкость {float(event.get('convergence_capacity_mbps', 0.0)):,.1f} Мбит/с · "
                f"полоса {float(event.get('peak_bandwidth_load_ratio', 0.0)) * float(event.get('intensity_ratio', 1.0)) * 100:.1f}% · "
                f"ресурс цели {float(event.get('peak_target_processing_load_ratio', 0.0)) * float(event.get('intensity_ratio', 1.0)) * 100:.1f}%"
            )
        runtime = event.get("power_runtime", {})
        if str(event.get("kind")) == "power_attack" and runtime:
            mode_label = {
                "brownout_feed_loss": "просадка входа, нагрузку удерживает ИБП",
                "cyber_shutdown": "команда выключения через контур управления",
            }.get(str(event.get("power_failure_mode")), "событие электропитания")
            phase_label = {
                "ups_ride_through": "работа от батареи без перерыва сервиса",
                "cyber_shutdown": "сервис выключен",
                "battery_recharge": "заряд батареи восстанавливается",
                "rebooting": "загрузка устройства",
                "health_check": "проверка готовности",
                "recovered": "восстановлено",
            }.get(str(runtime.get("phase")), str(runtime.get("phase", "—")))
            lines.append(f"  режим: {mode_label}; состояние: {phase_label}")
            lines.append(
                f"  вход ИБП {float(runtime.get('input_voltage_ratio', 1.0)) * 100:.1f}% · "
                f"выход {float(runtime.get('power_output_availability_ratio', 1.0)) * 100:.1f}% · "
                f"сервис {float(runtime.get('service_availability_ratio', 1.0)) * 100:.1f}% · "
                f"энергорезерв {float(runtime.get('energy_reserve_ratio', 1.0)) * 100:.1f}%"
            )
        reflector_count = int(technical.get("external_reflector_count", 0) or 0)
        if reflector_count:
            lines.append(
                f"  инициаторов L1: {int(technical.get('initiator_count', len(ingress)) or len(ingress))} · "
                f"внешних отражателей в профиле: {reflector_count:,} · "
                f"усиление ×{float(technical.get('amplification_factor', 1.0)):.1f}"
            )
        process = event.get("arrival_process", {})
        realization = event.get("arrival_realization", {})
        if str(process.get("distribution", "")).lower() == "weibull":
            lines.append(
                f"  Вейбулл: k={float(process.get('shape') or 0.0):.2f}, "
                f"λ={float(process.get('scale_seconds') or 0.0):.1f} с, "
                f"независимая выборка {float(realization.get('sampled_interarrival_seconds') or 0.0):.1f} с, "
                f"кадрированный интервал {float(realization.get('elapsed_interarrival_seconds') or 0.0):.1f} с"
            )
    return "\n".join(lines) + ("\n" if lines else "")


def _power_runtime_summary(records: list[dict[str, Any]]) -> str:
    """Показать продолжающееся восстановление даже после окончания импульса."""
    if not records:
        return ""
    phase_labels = {
        "ups_ride_through": "ИБП удерживает нагрузку",
        "cyber_shutdown": "сервис выключен",
        "battery_recharge": "заряд ИБП восстанавливается",
        "rebooting": "устройство загружается",
        "health_check": "проверяется готовность",
        "recovered": "работа восстановлена",
    }
    lines = ["\n\nЭЛЕКТРОПИТАНИЕ И ВОССТАНОВЛЕНИЕ"]
    for runtime in records[:4]:
        phase = phase_labels.get(str(runtime.get("phase")), str(runtime.get("phase", "—")))
        lines.append(
            f"{runtime.get('target_id', '—')}: {phase} · сервис "
            f"{float(runtime.get('service_availability_ratio', 1.0)) * 100:.1f}% · "
            f"энергорезерв {float(runtime.get('energy_reserve_ratio', 1.0)) * 100:.1f}%"
        )
    if len(records) > 4:
        lines.append(f"… и ещё {len(records) - 4}")
    return "\n".join(lines)


def _active_link_breakdown(model, loads) -> dict[str, int]:
    counts = {"access": 0, "uplink": 0, "core": 0, "service": 0, "other": 0}
    for edge in loads:
        nodes = tuple(edge)
        attrs = [model.graph.nodes[node] for node in nodes]
        levels = {item.get("level") for item in attrs}
        roles = {item.get("role") for item in attrs}
        if "L1" in levels:
            counts["access"] += 1
        elif roles & {"radio-access-node", "optical-line-terminal"}:
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
    if role in {"core-router", "aggregation-switch", "radio-access-node", "optical-line-terminal"}:
        return _equipment_hover_text(model, node_id, attrs)
    if role == "service-server":
        profile = attrs.get("server_profile", {})
        runtime = attrs.get("runtime", {})
        power = attrs.get("power_architecture", {})
        return (
            f"{node_id} · Физический сервер L0\n"
            f"IP: {attrs.get('ip_address')} · MAC: {attrs.get('mac_address')}\n"
            f"Модель: {profile.get('model')}\n"
            f"ЦП: {profile.get('cpu')} · {profile.get('total_cores')} физических ядер · "
            f"{profile.get('threads_total', 0)} потоков\n"
            f"ОЗУ: {profile.get('ram_gb')} ГБ · хранилище: {profile.get('storage_tb')} ТБ\n"
            f"Выбранные сетевые порты: {profile.get('network_ports_gbps')} Гбит/с\n"
            f"Питание: {power.get('feed_count', 0)} ввода · "
            f"{_power_redundancy_label(power.get('psu_redundancy'))}\n"
            f"БП: {power.get('psu_count', 0)} × {power.get('psu_rating_w_each', 0)} Вт · "
            f"{power.get('psu_efficiency_class', '—')} [Dell; номинал БП, не потребление]\n"
            f"T0: {float(power.get('modeled_t0_power_w') or 0.0):.0f} Вт [сценарий] · "
            f"ИБП {float(power.get('ups_backup_autonomy_hours') or 0.0):.2f} ч [сценарий]\n"
            f"Локальный домен отказа: {power.get('local_fault_domain_id', '—')}\n"
            f"Сервисы: {', '.join(attrs.get('hosted_services', []))}\n"
            f"Резервные реплики: {', '.join(attrs.get('standby_services', [])) or 'нет'}\n"
            f"Базовая нагрузка T0: ЦП {runtime.get('cpu_util_percent', 0):.1f}%, "
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
            f"Логическая точка агрегации: {attrs.get('home_access')}\n"
            "Линия доступа: одна; резервной линии абонента нет\n"
            f"Профиль доступа: {attrs.get('access_profile', '—')}\n"
            f"Доступ: {attrs.get('access_capacity_mbps', 0):.0f} Мбит/с · "
            f"базовая задержка {attrs.get('access_latency_ms', 0):.1f} мс\n"
            f"Класс SLA: {SLA_LABELS.get(attrs.get('sla_grade'), attrs.get('sla_grade'))}\n"
            f"Приложение: {TRAFFIC_LABELS.get(attrs.get('traffic_kind'), attrs.get('traffic_kind'))}\n"
            f"Кодек/профиль: {attrs.get('codec_profile_name') or attrs.get('codec')}\n"
            f"Целевая скорость: {attrs.get('target_bitrate_kbps', 0):.0f} кбит/с"
            f"{_protection_hover(attrs)}"
        )
    if attrs.get("level") == "L0":
        service_name = SERVICE_DISPLAY_NAMES.get(attrs.get("label"), attrs.get("label"))
        service_profile = attrs.get("service_profile", {})
        primary_metrics = ", ".join(
            SLO_METRIC_LABELS.get(metric, metric)
            for metric in service_profile.get("primary_slo_metrics", ())
        ) or "—"
        limitations = ", ".join(
            MODEL_LIMITATION_LABELS.get(limitation, limitation)
            for limitation in service_profile.get("modeling_limitations", ())
        ) or "нет"
        return (
            f"{node_id} · Логический сервис L0\nНазначение: {service_name}\n"
            f"Виртуальный IP (VIP): {attrs.get('ip_address')} · MAC: {attrs.get('mac_address')}\n"
            f"Платформа: {attrs.get('platform')}\nОсновной сервер: {attrs.get('hosted_on')}\n"
            f"Резервный сервер: {', '.join(attrs.get('standby_hosts', [])) or 'нет'}\n"
            f"Профиль: {attrs.get('codec_profile_name')}\n"
            f"Аудиокодек: {attrs.get('audio_codec') or '—'} · Видеокодек: {attrs.get('video_codec') or '—'}\n"
            f"Основные SLO-метрики: {primary_metrics}\n"
            f"Ограничения модели: {limitations}"
        )
    if role == "arbitrator":
        return "АРБ · Арбитратор L7\nНаблюдает метрики всех уровней\nПринимает решение о переназначении"
    if attrs.get("level") == "L8":
        x, y = attrs.get("pos", ("—", "—"))
        return (
            f"{node_id} · Опорная точка местности L8\n"
            f"Координаты модели: x={x}, y={y}\n"
            "Используется как географическая основа размещения;\n"
            "не является транзитным сетевым устройством"
        )
    return f"{node_id}\nРоль: {role}\nУровень: {attrs.get('level')}"


def _equipment_hover_text(model, node_id: str, attrs: dict[str, Any]) -> str:
    role = attrs.get("role")
    device_name = {
        "core-router": "Маршрутизатор ядра",
        "aggregation-switch": "Агрегирующий коммутатор",
        "radio-access-node": "RAN/UPF access node",
        "optical-line-terminal": "OLT access node",
    }.get(role, "L2 equipment")
    raw = attrs.get("l2_raw_baseline", {})
    profile = attrs.get("l2_profile", {})
    neighbors = list(model.graph.neighbors(node_id))
    clients = [
        node for node in neighbors
        if model.graph.nodes[node].get("level") == "L1"
        and model.graph.nodes[node].get("home_access") == node_id
    ]
    if role == "aggregation-switch":
        access_neighbors = [
            node for node in neighbors
            if model.graph.nodes[node].get("role") in {"radio-access-node", "optical-line-terminal"}
        ]
        clients = [
            node for node, client_attrs in model.graph.nodes(data=True)
            if client_attrs.get("level") == "L1"
            and client_attrs.get("home_access") in access_neighbors
        ]
    uplinks = [node for node in neighbors if model.graph.nodes[node].get("level") in {"L0", "L2"}]
    verified = set(profile.get("verified_fields", ()))
    dram_origin = "Cisco" if "dram_gb" in verified else "модель"
    forwarding_origin = "Cisco" if "forwarding_mpps" in verified else "модель"
    throughput_origin = "Cisco" if "throughput_gbps" in verified else "модель"
    port_origin = "Cisco" if str(profile.get("vendor")) == "Cisco" else "модель"
    power = attrs.get("power_architecture", {})
    if power.get("vendor_typical_output_power_w") is not None:
        vendor_power_text = (
            f"{float(power['vendor_typical_output_power_w']):.0f} Вт типично / "
            f"{float(power.get('vendor_max_output_power_w') or 0.0):.0f} Вт максимум "
            "выходной мощности [Cisco]"
        )
    else:
        vendor_power_text = (
            "типичное потребление не опубликовано; "
            f"{float(power.get('vendor_thermal_output_btu_per_hour') or 0.0):.0f} BTU/ч "
            f"≈ {float(power.get('vendor_thermal_output_equivalent_w') or 0.0):.0f} Вт "
            "теплового верхнего предела [Cisco]"
        )
    if (
        power.get("vendor_typical_output_power_w") is None
        and power.get("vendor_thermal_output_equivalent_w") is None
    ):
        vendor_power_text = "GNet9 scenario power envelope [not vendor spec]"
    return (
        f"{node_id} · {device_name}\n"
        f"IP: {attrs.get('ip_address')} · MAC: {attrs.get('mac_address')}\n"
        f"Платформа: {attrs.get('platform_family')}\n"
        f"Профиль: {attrs.get('platform_profile')}\n"
        f"Базовая загрузка ЦП: {raw.get('cpu_util', 0):.1f}% [модель]\n"
        f"Базовая загрузка ОЗУ: {raw.get('ram_util', 0):.1f} / {profile.get('dram_gb', 0):.1f} ГБ [{dram_origin}]\n"
        f"Системная пропускная способность: {profile.get('throughput_gbps', 0):,.0f} Гбит/с [{throughput_origin}]\n"
        f"Скорость пересылки: {profile.get('forwarding_mpps', 0):,.0f} млн пакетов/с [{forwarding_origin}]\n"
        f"Порты: {profile.get('port_configuration', '—')} [{port_origin}]\n"
        f"Логических соседей графа: {len(neighbors)}"
        + (f" · клиентов: {len(clients)}" if clients else "")
        + f"\nТранзитная степень: {attrs.get('infrastructure_degree', '—')} (предел ≤ 6)"
        + f"\nФизических портов ≥40G занято: {attrs.get('physical_high_speed_ports_used', '—')}"
        + f"\nL1-привязки абстрагируют RAN/PON и не расходуют по одному порту C9500"
        + f"\nПитание: {power.get('feed_count', 0)} ввода · "
        + _power_redundancy_label(power.get("psu_redundancy"))
        + f"\nМощность: {vendor_power_text}"
        + f"\nT0: {float(power.get('modeled_t0_power_w') or 0.0):.0f} Вт · "
        + f"ИБП {float(power.get('ups_backup_autonomy_hours') or 0.0):.1f} ч [сценарий]"
        + f"\nСвязи: {', '.join(uplinks)}"
        + _protection_hover(attrs)
    )


def _protection_hover(attrs: dict[str, Any]) -> str:
    protection = attrs.get("critical_protection", {})
    if not protection.get("is_critical"):
        return ""
    return (
        "\nКРИТИЧЕСКИЙ УЗЕЛ ЗОЛОТОГО КЛАССА · приоритет защиты 1"
        f"\nПотоков золотого класса через узел: {protection.get('gold_transit_flow_count', 0)}"
        f" · резервных: {protection.get('gold_recovery_reserve_flow_count', 0)}"
        f" · КВУ: {protection.get('critical_involvement_coefficient', 0.0):.3f}"
        f"\nПредел загрузки при переназначении: {protection.get('maximum_safe_utilization_percent', 80.0):.0f}%"
    )


def _power_redundancy_label(value: Any) -> str:
    return {
        "1+1 hot-swappable": "1+1, горячая замена",
        "1+1 hot-plug": "1+1, горячее подключение",
    }.get(str(value), str(value or "—"))


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
    frames_per_step: int = 32,
) -> float:
    """Return route progress while preserving the semantics of the t0 snapshot."""
    if step_index == 0 and paused:
        return 0.0
    spacing = offset / max(1, particle_count)
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
