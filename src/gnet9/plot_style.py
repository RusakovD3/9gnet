"""Общее оформление карт и графиков; расчёты модели сюда не входят."""

from pathlib import Path

import matplotlib.pyplot as plt

BACKGROUND = "#0f172a"
PANEL = "#111827"
TEXT = "#f1f5f9"
MUTED = "#aabbd0"
BORDER = "#334155"
BLUE = "#38bdf8"
GREEN = "#34d399"
AMBER = "#fbbf24"
RED = "#fb7185"
PURPLE = "#a78bfa"


def save_figure(fig, path: Path, *, dpi: int = 180) -> None:
    """Сохранить рисунок и закрыть его даже при ошибке записи."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.18,
                    facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)


def style_axis(ax, title: str, ylabel: str = "") -> None:
    ax.figure.set_facecolor(BACKGROUND)
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", loc="left", pad=14)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=BORDER, linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
