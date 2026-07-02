"""Deterministic private IP/MAC addressing for the simulated G-Net topology."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from .models import NetworkModel


ETHERNET_MTU_BYTES = 1500


@dataclass(frozen=True)
class NetworkIdentity:
    """Stable simulated L2/L3 identity for one graph node.

    The addresses are private laboratory addresses. They are used only inside
    generated artifacts and packet snapshots; no packet is sent to the host OS.
    """

    node_id: str
    ip: str
    mac: str
    subnet: str
    scope: str
    role: str
    mtu_bytes: int = ETHERNET_MTU_BYTES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


NETWORKS = {
    "l0_services": {
        "subnet": "10.0.0.0/24",
        "title": "L0: серверы и виртуальные адреса сервисов",
        "description": "Физические серверы SRV_* и логические сервисные VIP SVC_*.",
    },
    "l2_management": {
        "subnet": "10.10.0.0/24",
        "title": "L2: управление маршрутизаторами и коммутаторами",
        "description": "Loopback/management-адреса C1–C12 и A1–A6.",
    },
    "mobile_access": {
        "subnet": "10.1.0.0/16",
        "title": "L1: мобильные абоненты",
        "description": "10.1.<группа>.<номер>, группы M1–M3.",
    },
    "fixed_access": {
        "subnet": "10.2.0.0/16",
        "title": "L1: фиксированные абоненты",
        "description": "10.2.<группа>.<номер>, группы F1–F3.",
    },
    "l7_control": {
        "subnet": "10.70.0.0/24",
        "title": "L7: арбитратор",
        "description": "Служебный адрес арбитратора состояния.",
    },
    "l8_topology": {
        "subnet": "10.80.0.0/24",
        "title": "L8: топооснова",
        "description": "Адреса географических опорных узлов только для экспорта и диагностики.",
    },
}


def build_network_identities(model: NetworkModel) -> dict[str, NetworkIdentity]:
    """Assign deterministic private IPv4 and locally-administered MAC addresses."""
    identities: dict[str, NetworkIdentity] = {}
    used_ips: set[str] = set()

    for fallback_index, (node_id, attrs) in enumerate(sorted(model.graph.nodes(data=True)), start=1):
        ip = _ip_for_node(node_id, attrs, fallback_index)
        if ip in used_ips:
            ip = f"10.254.{fallback_index // 250}.{fallback_index % 250 + 1}"
        used_ips.add(ip)
        identities[node_id] = NetworkIdentity(
            node_id=node_id,
            ip=ip,
            mac=_mac_for_node(node_id),
            subnet=_subnet_for_node(node_id, attrs),
            scope=_scope_for_node(node_id, attrs),
            role=_role_ru(attrs.get("role", "")),
        )

    return identities


def attach_network_identities(model: NetworkModel) -> dict[str, NetworkIdentity]:
    """Store address data on graph nodes and return the same identity mapping."""
    identities = build_network_identities(model)
    for node_id, identity in identities.items():
        attrs = model.graph.nodes[node_id]
        attrs["ip_address"] = identity.ip
        attrs["mac_address"] = identity.mac
        attrs["ip_subnet"] = identity.subnet
        attrs["ip_scope"] = identity.scope
        attrs["network_identity"] = identity.to_dict()
    return identities


def build_ip_address_plan(model: NetworkModel) -> dict[str, Any]:
    """Return complete export-friendly address plan for all graph nodes."""
    identities = build_network_identities(model)
    devices = []
    for node_id, identity in sorted(identities.items(), key=lambda item: _address_sort_key(item[1].ip, item[0])):
        attrs = model.graph.nodes[node_id]
        devices.append(
            {
                **identity.to_dict(),
                "level": attrs.get("level"),
                "graph_role": attrs.get("role"),
                "label": attrs.get("label"),
                "home_access": attrs.get("home_access"),
                "hosted_on": attrs.get("hosted_on"),
                "hosted_services": attrs.get("hosted_services", []),
                "traffic_kind": attrs.get("traffic_kind"),
                "sla_grade": attrs.get("sla_grade"),
                "codec_profile_id": attrs.get("codec_profile_id"),
                "codec_profile_name": attrs.get("codec_profile_name"),
            }
        )
    return {"networks": NETWORKS, "devices": devices, "device_count": len(devices)}


def _ip_for_node(node_id: str, attrs: dict[str, Any], fallback_index: int) -> str:
    level = attrs.get("level")
    role = attrs.get("role")

    service_octet = {
        "SRV_MEDIA": 10,
        "SRV_DATA": 20,
        "SRV_RTC": 30,
        "SRV_LIVE": 40,
        "SVC_VOICE": 110,
        "SVC_VLC": 120,
        "SVC_FTP": 130,
        "SVC_DNS": 140,
        "SVC_TELEMOST": 150,
        "SVC_LIVE": 160,
    }
    if node_id in service_octet:
        return f"10.0.0.{service_octet[node_id]}"

    if role == "core-router" and node_id.startswith("C"):
        return f"10.10.0.{_numeric_suffix(node_id, fallback_index)}"

    if role == "aggregation-switch" and node_id.startswith("A"):
        return f"10.10.0.{100 + _numeric_suffix(node_id, fallback_index)}"

    if role == "mobile-subscriber":
        group, subscriber = _subscriber_numbers(node_id)
        return f"10.1.{group}.{subscriber}"

    if role == "fixed-subscriber":
        group, subscriber = _subscriber_numbers(node_id)
        return f"10.2.{group}.{subscriber}"

    if level == "L7":
        return "10.70.0.1"

    if level == "L8":
        return f"10.80.0.{fallback_index % 250 + 1}"

    return f"10.254.{fallback_index // 250}.{fallback_index % 250 + 1}"


def _subnet_for_node(node_id: str, attrs: dict[str, Any]) -> str:
    role = attrs.get("role")
    level = attrs.get("level")
    if level == "L0":
        return NETWORKS["l0_services"]["subnet"]
    if role in {"core-router", "aggregation-switch"}:
        return NETWORKS["l2_management"]["subnet"]
    if role == "mobile-subscriber":
        group, _ = _subscriber_numbers(node_id)
        return f"10.1.{group}.0/24"
    if role == "fixed-subscriber":
        group, _ = _subscriber_numbers(node_id)
        return f"10.2.{group}.0/24"
    if level == "L7":
        return NETWORKS["l7_control"]["subnet"]
    if level == "L8":
        return NETWORKS["l8_topology"]["subnet"]
    return "10.254.0.0/16"


def _scope_for_node(node_id: str, attrs: dict[str, Any]) -> str:
    role = attrs.get("role")
    level = attrs.get("level")
    if level == "L0":
        return "l0_services"
    if role in {"core-router", "aggregation-switch"}:
        return "l2_management"
    if role == "mobile-subscriber":
        return "mobile_access"
    if role == "fixed-subscriber":
        return "fixed_access"
    if level == "L7":
        return "l7_control"
    if level == "L8":
        return "l8_topology"
    return "other"


def _role_ru(role: str) -> str:
    return {
        "service-server": "физический сервер L0",
        "service": "логический сервис L0",
        "core-router": "маршрутизатор ядра C",
        "aggregation-switch": "агрегирующий коммутатор A",
        "mobile-subscriber": "мобильный абонент",
        "fixed-subscriber": "фиксированный абонент",
        "arbitrator": "арбитратор L7",
        "terrain-anchor": "опорный узел L8",
    }.get(role, role or "неизвестно")


def _numeric_suffix(node_id: str, fallback: int) -> int:
    digits = "".join(ch for ch in node_id if ch.isdigit())
    return int(digits) if digits else fallback


def _subscriber_numbers(node_id: str) -> tuple[int, int]:
    try:
        group_text, subscriber_text = node_id[1:].split("_", 1)
        return int(group_text), int(subscriber_text)
    except ValueError:
        return 0, 1


def _mac_for_node(node_id: str) -> str:
    digest = hashlib.blake2s(node_id.encode("utf-8"), digest_size=4).digest()
    return "02:09:%02x:%02x:%02x:%02x" % tuple(digest)


def _address_sort_key(ip: str, node_id: str) -> tuple[int, int, int, int, str]:
    octets = tuple(int(part) for part in ip.split("."))
    return (*octets, node_id)
