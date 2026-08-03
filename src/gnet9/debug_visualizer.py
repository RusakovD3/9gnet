"""Графическая диагностика архитектуры и фактического выполнения GNet9.

Модуль намеренно использует уже установленные NetworkX и Matplotlib. Статическая
схема объясняет предметные этапы, а ``RuntimeCallTracer`` показывает только те
Python-вызовы, которые действительно произошли внутри проекта при текущем запуске.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from time import perf_counter
import sys
import textwrap
from typing import Any, Callable

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.widgets import Button
import networkx as nx


@dataclass(frozen=True)
class ProcessBlock:
    """Один смысловой этап работы приложения."""

    block_id: str
    title: str
    stage: str
    description: str
    functions: tuple[str, ...]
    position: tuple[float, float]


PROCESS_BLOCKS = (
    ProcessBlock("cli", "Параметры запуска", "Управление", "Разбор режима, детализации и числа шагов.", ("parse_args",), (0.0, 3.0)),
    ProcessBlock("build", "Сборка модели", "Модель", "Создание воспроизводимой девятиуровневой структуры G-Net.", ("GNetBaselineBuilder.build",), (2.2, 3.0)),
    ProcessBlock("policy", "Политики d0sl", "Модель", "Разбор SLA/SLO и назначение профилей абонентам.", ("load_l1_d0sl_catalog", "build_l1_queue_model"), (4.4, 4.1)),
    ProcessBlock("topology", "Топология и тензоры", "Модель", "Узлы, связи, оборудование, серверы и тензорные метрики.", ("_build_l2", "build_tensor"), (4.4, 2.7)),
    ProcessBlock("validate", "Проверка эталона t0", "Контроль", "Контроль связности, SLA, оборудования и отсутствия фактических потерь.", ("validate_healthy_baseline", "validate_ideal_t0"), (6.7, 4.1)),
    ProcessBlock("clock", "Шаг динамики", "Динамика", "Формирование снимка состояния для каждого момента времени.", ("simulate_stationary_dynamics", "_snapshot"), (6.7, 2.7)),
    ProcessBlock("tensor", "Снимок тензоров", "Наблюдение", "Сбор метрик уровней L0–L8 и транспортных связей.", ("_tensor_state_snapshot",), (9.0, 4.1)),
    ProcessBlock("traffic", "Пакеты и потоки", "Наблюдение", "Маршруты без транзита через endpoint, заголовки, задержки, потери и загрузка каналов.", ("simulate_packet_snapshot", "shortest_data_path"), (9.0, 2.7)),
    ProcessBlock("attacks", "Сценарий MITRE ATT&CK", "Наблюдение", "Расписание DoS, DDoS, SYN-флуда, атак на питание, ВВХ/ТТХ и воздействие на легитимный трафик.", ("active_attack_events", "apply_attack_effects"), (9.0, 1.25)),
    ProcessBlock("arb", "Арбитратор L7", "Анализ", "Агрегация наблюдений, вектор состояния и базовое решение о переназначении.", ("build_arbitrator_view", "build_state_vector"), (11.3, 4.1)),
    ProcessBlock("koopman", "Купман / устойчивость", "Анализ", "DMD с задержанной историей сравнивает состояние с t0; близость угрозы и Хаусдорф Gold-маршрутов считаются раздельно, а функция Ляпунова показывает рост энергии отклонения.", ("KoopmanOnlineAnalyzer.analyze_step", "gold_threat_proximity_view", "gold_route_hausdorff_view"), (11.3, 1.25)),
    ProcessBlock("remap", "Защита и переназначение", "Анализ", "Подтверждённый прогноз запускает обход узла, резерв сервиса, карантин и строгое Gold → Silver → Bronze резервирование до 80%.", ("apply_koopman_to_arbitrator", "apply_gold_first_remap"), (13.6, 1.25)),
    ProcessBlock("charts", "Графики и отчёты", "Представление", "Экспорт диаграмм, JSON, CSV и карт сети.", ("export_dynamics_charts", "draw_service_flow_map"), (13.6, 2.7)),
    ProcessBlock("window", "Интерактивные окна", "Представление", "Независимый просмотр сети и графической диагностики.", ("ServiceFlowWindow", "DebugFlowWindow"), (15.9, 3.0)),
)

PROCESS_EDGES = (
    ("cli", "build"), ("build", "policy"), ("build", "topology"),
    ("policy", "validate"), ("topology", "clock"), ("validate", "clock"),
    ("clock", "tensor"), ("clock", "attacks"), ("attacks", "traffic"), ("tensor", "arb"),
    ("traffic", "arb"), ("arb", "koopman"), ("attacks", "koopman"), ("koopman", "remap"),
    ("remap", "traffic"), ("remap", "charts"), ("traffic", "charts"), ("arb", "charts"),
    ("charts", "window"),
)

STAGE_COLORS = {
    "Управление": "#64748b", "Модель": "#2563eb", "Контроль": "#16a34a",
    "Динамика": "#0891b2", "Наблюдение": "#7c3aed", "Анализ": "#ea580c",
    "Представление": "#db2777",
}


class RuntimeCallTracer:
    """Лёгкий агрегирующий профилировщик вызовов внутри каталога проекта."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.started_at = 0.0
        self.elapsed_seconds = 0.0
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[tuple[str, str], int] = defaultdict(int)
        self._starts: dict[int, tuple[str, float]] = {}
        self._frame_cache: dict[str, tuple[bool, str]] = {}
        self._previous_profiler: Callable[..., Any] | None = None
        self._root_key: str | None = None

    def start(self) -> "RuntimeCallTracer":
        if self.started_at:
            return self
        self.started_at = perf_counter()
        caller = sys._getframe(1)
        root = self._frame_identity(caller)
        if root is not None:
            key, relative_file, function_name = root
            self._root_key = key
            self.nodes.setdefault(key, {
                "id": key, "function": function_name, "file": relative_file,
                "module": _module_group(relative_file), "calls": 1, "total_time_ms": 0.0,
            })
        self._previous_profiler = sys.getprofile()
        sys.setprofile(self._profile)
        return self

    def stop(self) -> None:
        if not self.started_at:
            return
        sys.setprofile(self._previous_profiler)
        self.elapsed_seconds = perf_counter() - self.started_at
        if self._root_key is not None:
            self.nodes[self._root_key]["total_time_ms"] += self.elapsed_seconds * 1000.0
        self.started_at = 0.0
        self._starts.clear()

    def __enter__(self) -> "RuntimeCallTracer":
        return self.start()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.stop()

    def _profile(self, frame, event: str, _arg) -> None:
        if event == "call":
            current = self._frame_identity(frame)
            if current is None:
                return
            key, relative_file, function_name = current
            node = self.nodes.setdefault(key, {
                "id": key, "function": function_name, "file": relative_file,
                "module": _module_group(relative_file), "calls": 0, "total_time_ms": 0.0,
            })
            node["calls"] += 1
            self._starts[id(frame)] = (key, perf_counter())
            parent = self._frame_identity(frame.f_back) if frame.f_back is not None else None
            if parent is not None:
                self.edges[(parent[0], key)] += 1
        elif event == "return":
            started = self._starts.pop(id(frame), None)
            if started is not None:
                key, start_time = started
                self.nodes[key]["total_time_ms"] += (perf_counter() - start_time) * 1000.0

    def _frame_identity(self, frame) -> tuple[str, str, str] | None:
        filename = frame.f_code.co_filename
        cached = self._frame_cache.get(filename)
        if cached is None:
            try:
                path = Path(filename).resolve()
                relative = path.relative_to(self.project_root).as_posix()
                accepted = relative == "main.py" or relative.startswith("src/gnet9/")
            except (OSError, ValueError):
                accepted, relative = False, filename
            cached = (accepted, relative)
            self._frame_cache[filename] = cached
        accepted, relative = cached
        if not accepted or relative.endswith("debug_visualizer.py"):
            return None
        function_name = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
        if function_name.startswith("<"):
            return None
        return f"{relative}:{function_name}", relative, function_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "function_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": sorted(self.nodes.values(), key=lambda item: (-item["total_time_ms"], item["id"])),
            "edges": [
                {"source": source, "target": target, "calls": calls}
                for (source, target), calls in sorted(self.edges.items())
            ],
        }


def export_debug_artifacts(trace: RuntimeCallTracer | dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Сохранить блок-схему, граф вызовов и полную машинно-читаемую трассу."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_data = trace.to_dict() if isinstance(trace, RuntimeCallTracer) else trace
    paths = {
        "process": output_dir / "process_flow.png",
        "calls": output_dir / "runtime_call_graph.png",
        "trace": output_dir / "runtime_trace.json",
        "process_json": output_dir / "process_flow.json",
    }
    paths["trace"].write_text(json.dumps(trace_data, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["process_json"].write_text(json.dumps({
        "blocks": [{**asdict(block), "position": list(block.position)} for block in PROCESS_BLOCKS],
        "edges": [{"source": source, "target": target} for source, target in PROCESS_EDGES],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(25, 10))
    _draw_process_graph(ax)
    fig.savefig(paths["process"], dpi=210, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(24, 15))
    _draw_runtime_graph(ax, trace_data)
    fig.savefig(paths["calls"], dpi=210, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return paths


class DebugFlowWindow:
    """Интерактивное окно с переключением между процессами и вызовами."""

    def __init__(self, trace: RuntimeCallTracer | dict[str, Any]) -> None:
        self.trace_data = trace.to_dict() if isinstance(trace, RuntimeCallTracer) else trace
        self.mode = "process"
        self.node_positions: dict[str, tuple[float, float]] = {}
        self.node_details: dict[str, str] = {}
        self.fig, self.ax = plt.subplots(figsize=(20, 11))
        self.fig.subplots_adjust(left=0.035, right=0.985, bottom=0.10, top=0.93)
        self.process_button = Button(self.fig.add_axes([0.36, 0.025, 0.13, 0.045]), "Блоки процесса", color="#2563eb", hovercolor="#3b82f6")
        self.calls_button = Button(self.fig.add_axes([0.51, 0.025, 0.13, 0.045]), "Вызовы функций", color="#334155", hovercolor="#475569")
        for button in (self.process_button, self.calls_button):
            button.label.set_color("white")
            button.label.set_fontweight("bold")
        self.process_button.on_clicked(lambda _event: self._set_mode("process"))
        self.calls_button.on_clicked(lambda _event: self._set_mode("calls"))
        self.hover_note = self.ax.annotate(
            "", xy=(0, 0), xytext=(14, 14), textcoords="offset points", fontsize=9,
            color="#f8fafc", zorder=50,
            bbox={"boxstyle": "round,pad=0.55", "fc": "#020617", "ec": "#38bdf8", "alpha": 0.98},
            arrowprops={"arrowstyle": "->", "color": "#38bdf8"},
        )
        self.hover_note.set_visible(False)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._render()

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self.process_button.ax.set_facecolor("#2563eb" if mode == "process" else "#334155")
        self.calls_button.ax.set_facecolor("#2563eb" if mode == "calls" else "#334155")
        self._render()
        self.fig.canvas.draw_idle()

    def _render(self) -> None:
        self.ax.clear()
        self.hover_note = self.ax.annotate(
            "", xy=(0, 0), xytext=(14, 14), textcoords="offset points", fontsize=9,
            color="#f8fafc", zorder=50,
            bbox={"boxstyle": "round,pad=0.55", "fc": "#020617", "ec": "#38bdf8", "alpha": 0.98},
            arrowprops={"arrowstyle": "->", "color": "#38bdf8"},
        )
        self.hover_note.set_visible(False)
        if self.mode == "process":
            self.node_positions, self.node_details = _draw_process_graph(self.ax)
        else:
            self.node_positions, self.node_details = _draw_runtime_graph(self.ax, self.trace_data)

    def _on_hover(self, event) -> None:
        if event.inaxes is not self.ax or not self.node_positions or event.x is None or event.y is None:
            if self.hover_note.get_visible():
                self.hover_note.set_visible(False)
                self.fig.canvas.draw_idle()
            return
        nodes = list(self.node_positions)
        display = self.ax.transData.transform([self.node_positions[node] for node in nodes])
        distances = [math.hypot(point[0] - event.x, point[1] - event.y) for point in display]
        nearest = min(range(len(nodes)), key=distances.__getitem__)
        if distances[nearest] > 45.0:
            if self.hover_note.get_visible():
                self.hover_note.set_visible(False)
                self.fig.canvas.draw_idle()
            return
        node_id = nodes[nearest]
        self.hover_note.xy = self.node_positions[node_id]
        self.hover_note.set_text(self.node_details[node_id])
        self.hover_note.set_visible(True)
        self.fig.canvas.draw_idle()


def _draw_process_graph(ax) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    _style_axes(ax, "GNet9 · логика работы приложения по блокам")
    positions = {block.block_id: block.position for block in PROCESS_BLOCKS}
    details: dict[str, str] = {}
    by_id = {block.block_id: block for block in PROCESS_BLOCKS}
    for source, target in PROCESS_EDGES:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        ax.add_patch(FancyArrowPatch(
            (x1 + 0.72, y1), (x2 - 0.72, y2), arrowstyle="-|>", mutation_scale=15,
            linewidth=1.7, color="#64748b", alpha=0.85, connectionstyle="arc3,rad=0.02", zorder=1,
        ))
    for block in PROCESS_BLOCKS:
        x, y = block.position
        color = STAGE_COLORS[block.stage]
        ax.add_patch(FancyBboxPatch(
            (x - 0.73, y - 0.47), 1.46, 0.94,
            boxstyle="round,pad=0.04,rounding_size=0.08", facecolor=color,
            edgecolor="#e2e8f0", linewidth=1.3, alpha=0.94, zorder=3,
        ))
        function_hint = "\n".join(block.functions[:2])
        ax.text(x, y + 0.12, textwrap.fill(block.title, 20), ha="center", va="center", color="white", fontsize=9.3, fontweight="bold", zorder=4)
        ax.text(x, y - 0.20, function_hint, ha="center", va="center", color="#e2e8f0", fontsize=6.8, zorder=4)
        details[block.block_id] = (
            f"{block.title}\nЭтап: {block.stage}\n\n{block.description}\n\n"
            f"Ключевые функции:\n" + "\n".join(f"• {name}" for name in block.functions)
        )
    ax.set_xlim(-1.0, 14.6)
    ax.set_ylim(0.55, 5.0)
    legend = "   ".join(f"● {stage}" for stage in STAGE_COLORS)
    ax.text(0.5, 0.02, legend, transform=ax.transAxes, ha="center", color="#cbd5e1", fontsize=8.5)
    return positions, details


def _draw_runtime_graph(ax, trace_data: dict[str, Any], *, node_limit: int = 55) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    _style_axes(ax, "GNet9 · фактические вызовы функций текущего запуска")
    all_nodes = {item["id"]: item for item in trace_data.get("nodes", [])}
    ranked = sorted(
        all_nodes.values(),
        key=lambda item: (float(item.get("total_time_ms", 0.0)), math.log1p(int(item.get("calls", 0)))),
        reverse=True,
    )[:node_limit]
    selected = {item["id"] for item in ranked}
    graph = nx.DiGraph()
    graph.add_nodes_from(selected)
    for edge in trace_data.get("edges", []):
        if edge["source"] in selected and edge["target"] in selected and edge["source"] != edge["target"]:
            graph.add_edge(edge["source"], edge["target"], calls=edge["calls"])
    if not graph.nodes:
        ax.text(0.5, 0.5, "Вызовы не записаны. Запустите с --debug-diagram или --show-debug.", transform=ax.transAxes, ha="center", color="white", fontsize=12)
        return {}, {}
    positions = nx.spring_layout(graph, seed=42, k=max(0.45, 2.2 / math.sqrt(len(graph))), iterations=160)
    modules = sorted({all_nodes[node]["module"] for node in graph})
    palette = ["#2563eb", "#7c3aed", "#0891b2", "#16a34a", "#ea580c", "#db2777", "#64748b"]
    module_colors = {module: palette[index % len(palette)] for index, module in enumerate(modules)}
    max_calls = max(int(all_nodes[node]["calls"]) for node in graph)
    max_edge_calls = max((data["calls"] for _, _, data in graph.edges(data=True)), default=1)
    nx.draw_networkx_edges(
        graph, positions, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=12,
        width=[0.4 + 2.3 * data["calls"] / max_edge_calls for _, _, data in graph.edges(data=True)],
        edge_color="#64748b", alpha=0.42, connectionstyle="arc3,rad=0.06",
    )
    sizes = [260 + 1050 * math.sqrt(int(all_nodes[node]["calls"]) / max_calls) for node in graph]
    colors = [module_colors[all_nodes[node]["module"]] for node in graph]
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_size=sizes, node_color=colors, edgecolors="#e2e8f0", linewidths=0.8, alpha=0.94)
    labels = {node: _short_function_name(all_nodes[node]["function"]) for node in graph}
    nx.draw_networkx_labels(graph, positions, labels=labels, ax=ax, font_size=6.4, font_color="white")
    details = {
        node: (
            f"{all_nodes[node]['function']}\nФайл: {all_nodes[node]['file']}\n"
            f"Вызовов: {all_nodes[node]['calls']:,}\n"
            f"Суммарное время с вложенными вызовами: {all_nodes[node]['total_time_ms']:.3f} мс"
        )
        for node in graph
    }
    ax.text(
        0.01, 0.01,
        f"Показано {len(graph)} из {trace_data.get('function_count', len(all_nodes))} функций · "
        f"весь запуск {trace_data.get('elapsed_seconds', 0.0):.3f} с · размер узла = число вызовов",
        transform=ax.transAxes, color="#cbd5e1", fontsize=8.5,
    )
    return {node: tuple(point) for node, point in positions.items()}, details


def _style_axes(ax, title: str) -> None:
    ax.figure.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#111827")
    ax.set_title(title, color="#f8fafc", fontsize=16, fontweight="bold", pad=18)
    ax.set_axis_off()


def _module_group(relative_file: str) -> str:
    return Path(relative_file).stem


def _short_function_name(name: str) -> str:
    short = name.split(".<locals>.")[-1]
    return "\n".join(textwrap.wrap(short, width=18)[:2])
