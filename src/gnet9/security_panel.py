"""Правая панель наблюдений: список тревог, цепочка атаки и советы человеку."""

from __future__ import annotations

import textwrap

from matplotlib.patches import FancyBboxPatch

from .plot_style import BORDER, MUTED, PANEL, TEXT
from .security_dashboard import build_security_dashboard


class SecurityPanel:
    """Показывает сохранённые решения; выбор строки связывает панель с картой."""

    PAGE_SIZE = 4

    def __init__(self, fig, snapshots, on_selection):
        self.fig = fig
        self.snapshots = snapshots
        self.on_selection = on_selection
        self.selected_key = None
        self.page = 0
        self.recommendation_offset = 0
        self.evidence_offset = 0
        self.manual_selection = False
        self.data = {}
        self.visible_alarms = []
        self.axes = {
            "summary": fig.add_axes([.640, .825, .345, .075]),
            "alarms": fig.add_axes([.640, .585, .345, .230]),
            "chain": fig.add_axes([.640, .340, .345, .235]),
            "recommendations": fig.add_axes([.640, .140, .345, .190]),
        }
        self.visible = True
        fig.canvas.mpl_connect("button_press_event", self._click)
        fig.canvas.mpl_connect("scroll_event", self._scroll)
        fig.canvas.mpl_connect("resize_event", self._resize)

    @property
    def selected(self):
        return next((a for a in self.data.get("alarms", []) if a["key"] == self.selected_key), None)

    def set_visible(self, visible):
        self.visible = visible
        for ax in self.axes.values():
            ax.set_visible(visible)

    def _resize(self, _event):
        if self.data:
            self.draw()
            self.fig.canvas.draw_idle()

    def update(self, step_index):
        self.data = build_security_dashboard(self.snapshots, step_index)
        alarms = self.data["alarms"]
        selected = self.selected
        if selected is None:
            self.manual_selection = False
        if selected is None or (selected["phase"] == "history" and not self.manual_selection
                                and any(a["phase"] != "history" for a in alarms)):
            self.selected_key = alarms[0]["key"] if alarms else None
            self.page = 0
        self.page = min(self.page, max(0, (len(alarms) - 1) // self.PAGE_SIZE))
        self.recommendation_offset = 0
        self.evidence_offset = 0
        self.draw()

    def _base(self, name, title, color=TEXT):
        ax = self.axes[name]
        ax.clear()
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((.006, .012), .988, .976,
                     boxstyle="round,pad=0.005", facecolor=PANEL, edgecolor=BORDER))
        ax.text(.025, .95, title, color=color, fontsize=12, fontweight="bold", va="top")
        return ax

    def _wrap_width(self, name, fontsize):
        """Ограничить строку доступной шириной панели, включая Full HD."""
        pixels_per_char = fontsize * self.fig.dpi / 72 * .65
        return max(30, int(self.axes[name].bbox.width * .90 / pixels_per_char))

    def draw(self):
        self._draw_summary()
        self._draw_alarms()
        self._draw_chain()
        self._draw_recommendations()

    def _draw_summary(self):
        ax = self._base("summary", "ТРЕВОГИ · СРОЧНОСТЬ ОБЩЕЙ СИТУАЦИИ")
        labels = self.data["level_labels"]
        colors = self.data["level_colors"]
        for i, (key, label) in enumerate(labels.items()):
            count = self.data["severity_counts"].get(key, 0)
            x = .022 + i * .195
            ax.add_patch(FancyBboxPatch((x, .31), .183, .30, boxstyle="round,pad=.003",
                         facecolor="#1e293b", edgecolor=colors[key] if count else BORDER, linewidth=1))
            ax.text(x + .012, .53, f"{count}  {label}", fontsize=9.4,
                    color=colors[key] if count else MUTED, va="top")
        ax.text(.025, .18, "Уровень из расчёта · цвет строки и карты — тип атаки", fontsize=8.5, color=MUTED, va="top")

    def _draw_alarms(self):
        alarms = self.data["alarms"]
        pages = max(1, (len(alarms) + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        current = len(alarms) - self.data["history_count"]
        ax = self._base("alarms", f"ТРЕВОГИ: {current} · ИСТОРИЯ: {self.data['history_count']}")
        ax.text(.975, .95, f"‹  {self.page + 1}/{pages}  ›", color="#bae6fd", fontsize=11,
                ha="right", va="top")
        self.visible_alarms = alarms[self.page * self.PAGE_SIZE:(self.page + 1) * self.PAGE_SIZE]
        if not alarms:
            ax.text(.04, .70, "Наблюдаемых атак и предвестников нет", color="#86efac", fontsize=11)
            ax.text(.04, .54, "Продолжайте наблюдение за сетью.", color=MUTED, fontsize=10)
        for i, alarm in enumerate(self.visible_alarms):
            top = .82 - i * .18
            selected = alarm["key"] == self.selected_key
            ax.add_patch(FancyBboxPatch((.025, top - .155), .95, .151, boxstyle="round,pad=.002",
                         facecolor="#1e293b" if selected else PANEL,
                         edgecolor=alarm["color"] if selected else BORDER, linewidth=1.4 if selected else .6))
            ax.plot([.033, .033], [top - .14, top - .015], color=alarm["color"], linewidth=3)
            ax.text(.05, top - .018, f"{alarm['target_id']} · {alarm['phase_label']}", color=alarm["color"],
                    fontsize=10.2, va="top", fontweight="bold")
            name = textwrap.shorten(alarm["kind_label"], width=65, placeholder="…")
            ax.text(.05, top - .077, name, color=TEXT, fontsize=9.8, va="top")
            ax.text(.95, top - .102, alarm["severity_label"], color=alarm["severity_color"], fontsize=9,
                    va="top", ha="right")
        ax.text(.025, .045, "Выбор строки → место на карте · колесо / ‹ › → страницы",
                color=MUTED, fontsize=8.7, va="bottom")

    def _draw_chain(self):
        alarm = self.selected
        ax = self._base("chain", "KILL CHAIN · ЦЕПОЧКА РАЗВИТИЯ АТАКИ")
        if alarm is None:
            ax.text(.035, .70, "Выберите тревогу, когда появятся наблюдения.", color=MUTED, fontsize=10)
            return
        color = alarm["color"]
        source = ", ".join(alarm["source_ids"][:4]) or "источник не установлен"
        if len(alarm["source_ids"]) > 4:
            source += f" и ещё {len(alarm['source_ids']) - 4}"
        ax.text(.03, .82, f"{source} → {alarm['target_id']}", color=color, fontsize=10, va="top")
        techniques = alarm.get("mitre_technique_ids") or alarm.get("mitre_technique_candidates", [])
        identifiers = " / ".join([*techniques, *alarm.get("mitre_tactic_ids", [])])
        prefix = "гипотеза " if not alarm.get("mitre_technique_ids") else ""
        identifiers = f" · {prefix}{identifiers}" if identifiers else ""
        ax.text(.03, .73, f"Наблюдения: {alarm['first_seen_seconds']:g}…{alarm['last_seen_seconds']:g} с{identifiers}",
                color=MUTED, fontsize=9, va="top")
        for i, stage in enumerate(alarm["chain"]):
            x = .035 + i * .243
            known = stage["status"] in {"observed", "current"}
            stage_color = color if known else MUTED
            ax.add_patch(FancyBboxPatch((x, .52), .216, .102, boxstyle="round,pad=.003",
                         facecolor="#1e293b", edgecolor=stage_color if known else BORDER))
            ax.text(x + .008, .60, f"{'●' if known else '○'} {stage['label']}", color=stage_color, fontsize=10, va="top")
            if i < 3:
                ax.text(x + .225, .58, "›", color=MUTED, fontsize=11, va="top")
        # Полная классическая цепочка не выводится из одного сетевого признака.
        classic = alarm.get("classic_chain", [])
        labels = [f"{'●' if s['status'] in {'observed', 'current'} else '○'} {s['label']}" for s in classic]
        ax.text(.03, .46, "Классическая схема · ○ нет данных, ● наблюдалось", color=MUTED, fontsize=8.6, va="top")
        ax.text(.03, .37, " → ".join(labels[:4]) + "\n" + " → ".join(labels[4:]),
                color="#cbd5e1", fontsize=9, va="top", linespacing=1.35)
        evidence = self._evidence_lines()
        self.evidence_offset = min(self.evidence_offset, max(0, len(evidence) - 2))
        ax.text(.03, .225, "\n".join(evidence[self.evidence_offset:self.evidence_offset + 2]),
                color=TEXT, fontsize=10, va="top", linespacing=1.25)
        ax.text(.03, .035, f"Факты {self.evidence_offset + 1}–{min(len(evidence), self.evidence_offset + 2)} из {len(evidence)} · колесо / клик ‹ ›",
                color=MUTED, fontsize=8.5, va="bottom")
        ax.text(.97, .035, "‹    ›", color="#bae6fd", fontsize=12, ha="right", va="bottom")

    def _evidence_lines(self):
        evidence = self.selected["evidence_lines"] if self.selected else []
        return [line for fact in evidence for line in textwrap.wrap(fact, width=self._wrap_width("chain", 10))] or ["Нет отдельных измерений последствий этой атаки."]

    def _recommendation_lines(self):
        rec = self.data["recommendation"]
        lines = []
        for number, step in enumerate(rec.get("steps_ru", []), 1):
            lines.extend(textwrap.wrap(f"{number}. {step}", width=self._wrap_width("recommendations", 9.7), subsequent_indent="   "))
        return lines or ["Продолжайте наблюдение; сохранённой рекомендации для этого шага нет."]

    def _draw_recommendations(self):
        rec = self.data["recommendation"]
        requires = rec.get("requires_human_action")
        title = "НУЖНО РЕШЕНИЕ ЧЕЛОВЕКА" if requires else "ДЕЙСТВИЯ · НАБЛЮДЕНИЕ"
        ax = self._base("recommendations", title, "#fbbf24" if requires else "#86efac")
        ax.text(.03, .79, "Для текущего шага · приоритет: золотой → серебряный → бронзовый",
                color=MUTED, fontsize=8.8, va="top")
        lines = self._recommendation_lines()
        self.recommendation_offset = min(self.recommendation_offset, max(0, len(lines) - 6))
        visible = lines[self.recommendation_offset:self.recommendation_offset + 6]
        ax.text(.03, .67, "\n".join(visible), fontsize=9.7, color=TEXT, va="top", linespacing=1.35)
        suffix = f"Строки {self.recommendation_offset + 1}–{min(len(lines), self.recommendation_offset + 6)} из {len(lines)} · колесо / клик ‹ ›"
        ax.text(.03, .08, suffix, color=MUTED, fontsize=8.5, va="bottom")
        ax.text(.97, .08, "‹    ›", color="#bae6fd", fontsize=12, ha="right", va="bottom")

    def _click(self, event):
        if not self.visible or event.xdata is None or event.ydata is None:
            return
        if event.inaxes is self.axes["alarms"]:
            if event.ydata > .86 and event.xdata > .74:
                self._page(1 if event.xdata > .9 else -1)
                return
            index = int((.82 - event.ydata) / .18)
            if 0 <= index < len(self.visible_alarms) and .10 <= event.ydata <= .82:
                self.selected_key = self.visible_alarms[index]["key"]
                self.manual_selection = True
                self.evidence_offset = 0
                self.draw()
                self.on_selection()
        elif event.inaxes is self.axes["chain"] and event.ydata < .10:
            self.evidence_offset = max(0, self.evidence_offset + (2 if event.xdata > .95 else -2))
            self._draw_chain()
        elif event.inaxes is self.axes["recommendations"] and event.ydata < .20:
            self.recommendation_offset = max(0, self.recommendation_offset + (6 if event.xdata > .95 else -6))
            self._draw_recommendations()
        self.fig.canvas.draw_idle()

    def _page(self, direction):
        last = max(0, (len(self.data["alarms"]) - 1) // self.PAGE_SIZE)
        self.page = min(last, max(0, self.page + direction))
        self._draw_alarms()
        self.fig.canvas.draw_idle()

    def _scroll(self, event):
        if not self.visible:
            return
        direction = -1 if event.button == "up" else 1
        if event.inaxes is self.axes["alarms"]:
            self._page(direction)
        elif event.inaxes is self.axes["recommendations"]:
            self.recommendation_offset = max(0, self.recommendation_offset + 3 * direction)
            self._draw_recommendations()
            self.fig.canvas.draw_idle()
        elif event.inaxes is self.axes["chain"]:
            self.evidence_offset = max(0, self.evidence_offset + direction)
            self._draw_chain()
            self.fig.canvas.draw_idle()
