"""IP address plan exports and a readable address map PNG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .addressing import build_ip_address_plan


ROLE_COLORS = {
    "service-server": "#a78bfa",
    "service": "#10b981",
    "core-router": "#60a5fa",
    "aggregation-switch": "#93c5fd",
    "radio-access-node": "#38bdf8",
    "optical-line-terminal": "#2dd4bf",
    "mobile-subscriber": "#22d3ee",
    "fixed-subscriber": "#fbbf24",
    "arbitrator": "#fb923c",
    "terrain-anchor": "#94a3b8",
}


def export_ip_address_plan(model, path: Path) -> None:
    """Write complete IP/MAC mapping for every graph node."""
    path.write_text(json.dumps(build_ip_address_plan(model), ensure_ascii=False, indent=2), encoding="utf-8")


def draw_ip_address_map(model, path: Path) -> None:
    """Save a separate visual map of device-to-IP relationships."""
    plan = build_ip_address_plan(model)
    devices = plan["devices"]

    fig, ax = plt.subplots(figsize=(28, 18))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 18)
    ax.axis("off")
    ax.set_title("G-Net · карта IP-адресации устройств", color="#f8fafc", fontsize=22, fontweight="bold", pad=16)

    _draw_top_panel(ax, devices)
    _draw_access_panels(ax, devices)
    _draw_address_legend(ax)

    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_top_panel(ax, devices: list[dict[str, Any]]) -> None:
    l0 = [device for device in devices if device["level"] == "L0"]
    l2 = [device for device in devices if device["level"] == "L2"]
    l7_l8 = [device for device in devices if device["level"] in {"L7", "L8"}]
    _draw_box(ax, 0.6, 13.35, 7.7, 3.45, "L0 · серверы и вирт. адреса сервисов · 10.0.0.0/24", l0, columns=1, font_size=7.0)
    _draw_box(ax, 8.7, 13.35, 9.0, 3.45, "L2 · петлевые адреса и управление · 10.10.0.0/24", l2, columns=2, font_size=6.8)
    _draw_box(ax, 18.1, 13.35, 5.1, 3.45, "L7/L8 · служебные адреса", l7_l8, columns=1, font_size=6.4, limit=10)

    text = (
        "Адреса образуют частный лабораторный план.\n"
        "Они используются в JSON, GraphML, выборке пакетов и визуализации.\n"
        "Абоненты сгруппированы по единственной линии доступа.\n"
        "Резервирование предусмотрено в L2 и на серверах,\n"
        "но не в последней миле абонента.\n"
        "Реальные сетевые пакеты проект не отправляет."
    )
    ax.text(
        23.6,
        16.75,
        text,
        va="top",
        fontsize=7.7,
        color="#dbeafe",
        linespacing=1.3,
        bbox={"boxstyle": "round,pad=0.55", "fc": "#172033", "ec": "#475569", "alpha": 0.96},
    )


def _draw_access_panels(ax, devices: list[dict[str, Any]]) -> None:
    panel_specs = [
        ("A1", "M1", "mobile-subscriber", "10.1.1.0/24", 0.6, 8.05),
        ("A3", "M2", "mobile-subscriber", "10.1.2.0/24", 9.6, 8.05),
        ("A5", "M3", "mobile-subscriber", "10.1.3.0/24", 18.6, 8.05),
        ("A2", "F1", "fixed-subscriber", "10.2.1.0/24", 0.6, 2.55),
        ("A4", "F2", "fixed-subscriber", "10.2.2.0/24", 9.6, 2.55),
        ("A6", "F3", "fixed-subscriber", "10.2.3.0/24", 18.6, 2.55),
    ]
    for access, prefix, role, subnet, x, y in panel_specs:
        group_devices = [
            device for device in devices
            if device["graph_role"] == role and device["node_id"].startswith(prefix + "_")
        ]
        title = f"{access} · {'мобильные' if role == 'mobile-subscriber' else 'фиксированные'} · {subnet}"
        _draw_box(ax, x, y, 8.2, 4.6, title, group_devices, columns=4, font_size=5.65)


def _draw_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    devices: list[dict[str, Any]],
    *,
    columns: int,
    font_size: float,
    limit: int | None = None,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.06,rounding_size=0.13",
            facecolor="#111827",
            edgecolor="#475569",
            linewidth=1.1,
            alpha=0.98,
        )
    )
    ax.text(x + 0.2, y + height - 0.34, title, color="#f8fafc", fontsize=8.8, fontweight="bold", va="top")
    rows = devices[:limit] if limit else devices
    if limit and len(devices) > limit:
        rows = rows + [{"node_id": f"… ещё {len(devices) - limit}", "ip": "", "role": ""}]
    column_width = width / columns
    row_count = max(1, (len(rows) + columns - 1) // columns)
    line_step = min(0.31, (height - 0.8) / max(row_count, 1))
    for index, device in enumerate(rows):
        column = index // row_count
        row = index % row_count
        tx = x + 0.22 + column * column_width
        ty = y + height - 0.82 - row * line_step
        color = ROLE_COLORS.get(device.get("graph_role"), "#e2e8f0")
        text = f"{device.get('node_id', ''):<8} {device.get('ip', '')}"
        ax.text(tx, ty, text, color=color, fontsize=font_size, family="DejaVu Sans Mono", va="top")


def _draw_address_legend(ax) -> None:
    lines = [
        ("SRV_*", "физический сервер", ROLE_COLORS["service-server"]),
        ("SVC_*", "виртуальный адрес сервиса", ROLE_COLORS["service"]),
        ("C*", "маршрутизатор ядра", ROLE_COLORS["core-router"]),
        ("A*", "агрегирующий коммутатор", ROLE_COLORS["aggregation-switch"]),
        ("RAN*", "мобильный access", ROLE_COLORS["radio-access-node"]),
        ("OLT*", "фиксированный access", ROLE_COLORS["optical-line-terminal"]),
        ("M*", "мобильный абонент", ROLE_COLORS["mobile-subscriber"]),
        ("F*", "фиксированный абонент", ROLE_COLORS["fixed-subscriber"]),
    ]
    ax.add_patch(
        FancyBboxPatch(
            (23.6, 12.82),
            3.8,
            2.0,
            boxstyle="round,pad=0.06,rounding_size=0.13",
            facecolor="#111827",
            edgecolor="#475569",
            linewidth=1.0,
            alpha=0.98,
        )
    )
    ax.text(23.82, 14.54, "Обозначения", color="#f8fafc", fontsize=8.8, fontweight="bold", va="top")
    for index, (prefix, label, color) in enumerate(lines):
        ax.text(23.82, 14.18 - index * 0.25, f"{prefix:<5} {label}", color=color, fontsize=7.2, family="DejaVu Sans Mono")
