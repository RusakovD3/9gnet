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


def save_figure(fig, path: Path, *, dpi: int = 180, tight: bool = True) -> None:
    """Сохранить рисунок и закрыть его даже при ошибке записи."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight" if tight else None, pad_inches=0.18,
                    facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
