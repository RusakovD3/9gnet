"""Схема сети и девять уровней: только данные и связи текущей модели."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .constants import LEVEL_NAMES
from .flow_visualizer import draw_network_map
from .plot_style import BACKGROUND, PANEL, TEXT, MUTED, BLUE, BORDER, save_figure


class GNetVisualizer:
    def __init__(self, model) -> None:
        self.model = model
        self.graph = model.graph

    def draw_network_logic(self, path: Path) -> None:
        draw_network_map(self.model, path)

    def draw_layer_scheme(self, path: Path) -> None:
        """Карточки уровней показывают их роль, без условных физических объектов."""
        descriptions = {
            "L0": f"{len(self.model.services)} услуг · {len(self.model.servers)} сервера",
            "L1": f"{sum(a.get('level') == 'L1' for _, a in self.graph.nodes(data=True))} абонентов",
            "L2": f"{sum(a.get('level') == 'L2' for _, a in self.graph.nodes(data=True))} сетевых устройств",
            "L3": "Ёмкость · затухание · помехи",
            "L4": "Длина линий · восстановление",
            "L5": "Правила передачи · приоритеты",
            "L6": "Питание · запас энергии",
            "L7": "Состояние → прогноз → защита",
            "L8": "Координаты · местность",
        }
        fig, ax = plt.subplots(figsize=(13, 8))
        fig.set_facecolor(BACKGROUND)
        ax.set(xlim=(-.15, 12.3), ylim=(0, 7))
        ax.axis("off")
        ax.set_title("GNet9 · устройство модели", color=TEXT, fontsize=20,
                     fontweight="bold", loc="left", pad=24)
        ax.text(0, 6.75, "L — уровень. Карточки описывают данные, а не дополнительные устройства.",
                color=MUTED, fontsize=10)
        for index, (level, title) in enumerate(LEVEL_NAMES.items()):
            row, column = divmod(index, 3)
            x, y = column * 4.05, 4.7 - row * 1.8
            ax.add_patch(FancyBboxPatch((x, y), 3.75, 1.45,
                         boxstyle="round,pad=0.04,rounding_size=0.12",
                         facecolor=PANEL, edgecolor=BORDER, linewidth=1))
            ax.text(x + .2, y + 1.1, level, color=BLUE, fontsize=13, fontweight="bold")
            ax.text(x + .2, y + .7, title, color=TEXT, fontsize=10, fontweight="bold")
            ax.text(x + .2, y + .28, descriptions[level], color=MUTED, fontsize=9)
        ax.text(0, .5, "Свойства оборудования и линий → наблюдения по уровням → арбитр L7", color=TEXT, fontsize=11)
        ax.text(0, .1, "Защита меняет рабочие маршруты; исходное исправное состояние t0 сохраняется.", color=MUTED, fontsize=10)
        save_figure(fig, path)
