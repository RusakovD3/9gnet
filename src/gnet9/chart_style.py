"""Светлое оформление печатных графиков и статической сводки."""

BACKGROUND = "#ffffff"
TEXT = "#202d39"
MUTED = "#475769"
BORDER = "#d0d7df"
BLUE = "#226398"
GREEN = "#18745e"
AMBER = "#a05b06"
RED = "#ae362e"
PURPLE = "#72558e"


def style_axis(ax, title: str, ylabel: str = "") -> None:
    ax.figure.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", loc="left", pad=12)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=.6, pad=5)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BORDER)
    ax.grid(axis="y", color=BORDER, linewidth=.6, alpha=.65)
    ax.set_axisbelow(True)
