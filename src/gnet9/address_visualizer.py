"""Полный план адресов в JSON и компактная читаемая карта подсетей."""

import ipaddress
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .addressing import build_ip_address_plan
from .plot_style import BACKGROUND, PANEL, TEXT, MUTED, BORDER, BLUE, GREEN, AMBER, save_figure


def export_ip_address_plan(model, path: Path) -> None:
    path.write_text(json.dumps(build_ip_address_plan(model), ensure_ascii=False, indent=2), encoding="utf-8")


def subscriber_address_groups(devices):
    """Группировать по заданной подсети, не по выдуманной связи с коммутатором."""
    groups = {}
    for device in devices:
        if device["level"] == "L1":
            groups.setdefault(device["subnet"], []).append(device)
    return {subnet: sorted(rows, key=lambda d: ipaddress.ip_address(d["ip"]))
            for subnet, rows in sorted(groups.items())}


def draw_ip_address_map(model, path: Path) -> None:
    devices = build_ip_address_plan(model)["devices"]
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_facecolor(BACKGROUND)
    ax.set(xlim=(-.15, 15.2), ylim=(0, 10))
    ax.axis("off")
    ax.set_title("GNet9 · устройства и подсети", color=TEXT, fontsize=21,
                 fontweight="bold", loc="left", pad=16)
    ax.text(0, 9.7, "IP — сетевой адрес. Полный список адресов каждого абонента: ip_address_plan.json", color=MUTED, fontsize=11)
    for x, width, title, levels, columns in (
        (0, 4.5, "L0 · серверы и адреса услуг", {"L0"}, 1),
        (4.8, 6, "L2 · сетевые устройства", {"L2"}, 2),
        (11.1, 3.9, "L7 / L8 · служебные узлы", {"L7", "L8"}, 1),
    ):
        rows = [d for d in devices if d["level"] in levels]
        _card(ax, x, 5.65, width, 3.7, title)
        count = max(1, (len(rows) + columns - 1) // columns)
        for i, d in enumerate(rows):
            column, row = divmod(i, count)
            ax.text(x + .18 + column * width / columns, 8.6 - row * .18,
                    f"{d['node_id']:<11} {d['ip']}", color=BLUE if d["level"] == "L2" else GREEN,
                    fontsize=8.1, family="DejaVu Sans Mono")
    for i, (subnet, rows) in enumerate(subscriber_address_groups(devices).items()):
        row, column = divmod(i, 3)
        x, y = column * 5.1, 3.1 - row * 2.15
        color = BLUE if rows[0]["graph_role"] == "mobile-subscriber" else AMBER
        kind = "Мобильные" if rows[0]["graph_role"] == "mobile-subscriber" else "Проводные"
        _card(ax, x, y, 4.8, 1.85, f"{kind} · {len(rows)} абонентов")
        access = ", ".join(sorted({d["home_access"] for d in rows}))
        ax.text(x + .18, y + 1.12, subnet, color=color, fontsize=13, fontweight="bold")
        ax.text(x + .18, y + .67, f"Адреса: {rows[0]['ip']} — {rows[-1]['ip']}", color=TEXT, fontsize=9)
        ax.text(x + .18, y + .25, f"Узлы доступа: {access}", color=MUTED, fontsize=9)
    ax.text(0, .32, "C — маршрутизатор · A — коммутатор · RAN — радиодоступ · OLT — оптический доступ", color=MUTED, fontsize=10)
    ax.text(0, 0, "SRV — сервер · SVC — адрес услуги. Группировка сокращает рисунок; полный план адресов сохраняется.", color=MUTED, fontsize=10)
    save_figure(fig, path)


def _card(ax, x, y, width, height, title):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                 boxstyle="round,pad=0.04,rounding_size=0.12",
                 facecolor=PANEL, edgecolor=BORDER))
    ax.text(x + .18, y + height - .35, title, color=TEXT, fontsize=10, fontweight="bold")
