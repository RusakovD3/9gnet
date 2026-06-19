"""L2 equipment capacities with explicit provenance.

Vendor-published figures and simulation assumptions are deliberately kept in
the same exported profile but are labelled separately.  This prevents an
assumed queue size or telemetry value from being presented as a Cisco spec.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np


class L2Resource(str, Enum):
    CPU = "cpu_util"
    RAM = "ram_util"
    FIB = "fib_usage"
    TCAM = "tcam_usage"
    PPS = "pps_util"
    THROUGHPUT = "throughput_util"
    QUEUE = "queue_util"
    CRYPTO = "crypto_util"
    CONTROL_PLANE = "control_plane_load"
    TEMPERATURE = "temperature"


@dataclass(frozen=True)
class L2EquipmentProfile:
    """Capacity envelope for one concrete equipment model."""

    name: str
    vendor: str
    model_family: str
    role: str
    source_note: str
    source_url: str
    verified_fields: tuple[str, ...]
    assumed_fields: tuple[str, ...]
    port_configuration: str
    throughput_gbps: float
    forwarding_mpps: float
    dram_gb: float
    fib_routes: int
    tcam_entries: int
    buffer_mb: float
    crypto_gbps: float
    control_plane_sessions: int
    operating_temp_c: float
    typical_power_w: float | None = None
    max_power_w: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Numeric fields absent from a public data sheet remain useful as model limits,
# but they are listed in assumed_fields and must be labelled as such in UI/export.
CISCO_LIKE_PROFILES: dict[str, L2EquipmentProfile] = {
    "C9500_AGG": L2EquipmentProfile(
        name="C9500_AGG",
        vendor="Cisco",
        model_family="Catalyst C9500-24Y4C",
        role="aggregation-switch",
        source_note="Cisco Catalyst 9500 data sheet, C9500-24Y4C row. Published chassis limits are not live telemetry.",
        source_url="https://www.cisco.com/c/en/us/products/collateral/switches/catalyst-9500-series-switches/nb-06-cat9500-ser-data-sheet-cte-en.html",
        verified_fields=("throughput_gbps", "forwarding_mpps", "dram_gb", "buffer_mb", "port_configuration"),
        assumed_fields=("fib_routes", "tcam_entries", "crypto_gbps", "control_plane_sessions", "operating_temp_c"),
        port_configuration="24 x 1/10/25G + 4 x 40/100G",
        throughput_gbps=2_000.0,
        forwarding_mpps=1_000.0,
        dram_gb=16.0,
        fib_routes=128_000,
        tcam_entries=256_000,
        buffer_mb=36.0,
        crypto_gbps=0.0,
        control_plane_sessions=2_000,
        operating_temp_c=40.0,
    ),
    "NCS5501_CORE": L2EquipmentProfile(
        name="NCS5501_CORE",
        vendor="Cisco",
        model_family="NCS 5501",
        role="core-router",
        source_note="Cisco NCS 5500 data sheet: NCS-5501 system throughput up to 800 Gbps and up to 1M FIB; other limits below are explicit simulation assumptions.",
        source_url="https://www.cisco.com/c/en/us/products/collateral/routers/network-convergence-system-5500-series/datasheet-c78-737935.pdf",
        verified_fields=("throughput_gbps", "fib_routes", "port_configuration", "operating_temp_c", "typical_power_w", "max_power_w"),
        assumed_fields=("forwarding_mpps", "dram_gb", "tcam_entries", "buffer_mb", "crypto_gbps", "control_plane_sessions"),
        port_configuration="48 x 1/10G + 6 x 40/100G",
        throughput_gbps=800.0,
        forwarding_mpps=500.0,
        dram_gb=32.0,
        fib_routes=1_000_000,
        tcam_entries=512_000,
        buffer_mb=256.0,
        crypto_gbps=0.0,
        control_plane_sessions=4_000,
        operating_temp_c=55.0,
        typical_power_w=240.0,
        max_power_w=370.0,
    ),
}


def l2_profile_for_role(role: str, *, criticality: str = "silver") -> L2EquipmentProfile:
    """Pick the concrete Cisco profile assigned to a G-Net L2 role."""
    if role == "core-router":
        return CISCO_LIKE_PROFILES["NCS5501_CORE"]
    return CISCO_LIKE_PROFILES["C9500_AGG"]


def build_l2_raw_baseline(profile: L2EquipmentProfile, *, role: str, criticality: str) -> dict[str, float]:
    """Return baseline raw telemetry values for one active device.

    Raw values use real units where possible: Gbps, Mpps, routes, entries, MB,
    sessions and Celsius. The tensor receives selected raw and normalized values.
    """
    # A healthy baseline carries tens of Mbit/s, whereas these chassis provide
    # hundreds of Gbit/s.  Values are deterministic synthetic telemetry, not
    # measurements read from a physical Cisco device.
    role_factor = 0.0008 if role == "core-router" else 0.0005
    grade_factor = 1.05 if criticality == "gold" else 1.0
    crypto_baseline = 0.0

    return {
        L2Resource.CPU.value: 16.0 * grade_factor if role == "core-router" else 12.0,
        L2Resource.RAM.value: profile.dram_gb * (0.35 if role == "core-router" else 0.30),
        L2Resource.FIB.value: profile.fib_routes * (0.10 if role == "core-router" else 0.08),
        L2Resource.TCAM.value: profile.tcam_entries * (0.12 if role == "core-router" else 0.10),
        L2Resource.PPS.value: profile.forwarding_mpps * role_factor,
        L2Resource.THROUGHPUT.value: profile.throughput_gbps * role_factor,
        L2Resource.QUEUE.value: profile.buffer_mb * (0.05 if role == "core-router" else 0.04),
        L2Resource.CRYPTO.value: crypto_baseline,
        L2Resource.CONTROL_PLANE.value: profile.control_plane_sessions * (0.05 if role == "core-router" else 0.04),
        L2Resource.TEMPERATURE.value: 32.0 if role == "core-router" else 30.0,
    }


def build_l2_summary_metrics(raw: dict[str, float], profile: L2EquipmentProfile) -> dict[str, float]:
    """Return compact normalized metrics for quick filtering and visualization."""
    ratios = _normalized_ratios(raw, profile)
    load = float(np.mean([ratios[L2Resource.CPU.value], ratios[L2Resource.PPS.value], ratios[L2Resource.THROUGHPUT.value], ratios[L2Resource.QUEUE.value]]))
    scale_pressure = float(np.mean([ratios[L2Resource.FIB.value], ratios[L2Resource.TCAM.value]]))
    mgmt_pressure = ratios[L2Resource.CONTROL_PLANE.value]
    thermal_pressure = ratios[L2Resource.TEMPERATURE.value]
    health = 1.0 - min(1.0, max(load, scale_pressure, thermal_pressure) * 0.72)
    return {
        "l2_load_index": round(load, 4),
        "l2_scale_pressure": round(scale_pressure, 4),
        "l2_mgmt_pressure": round(mgmt_pressure, 4),
        "l2_thermal_pressure": round(thermal_pressure, 4),
        "l2_health_index": round(health, 4),
    }


def _normalized_ratios(raw: dict[str, float], profile: L2EquipmentProfile) -> dict[str, float]:
    limits = {
        L2Resource.CPU.value: 100.0,
        L2Resource.RAM.value: max(profile.dram_gb, 1.0),
        L2Resource.FIB.value: max(profile.fib_routes, 1),
        L2Resource.TCAM.value: max(profile.tcam_entries, 1),
        L2Resource.PPS.value: max(profile.forwarding_mpps, 1e-9),
        L2Resource.THROUGHPUT.value: max(profile.throughput_gbps, 1e-9),
        L2Resource.QUEUE.value: max(profile.buffer_mb, 1e-9),
        L2Resource.CRYPTO.value: max(profile.crypto_gbps, 1.0),
        L2Resource.CONTROL_PLANE.value: max(profile.control_plane_sessions, 1),
        L2Resource.TEMPERATURE.value: max(profile.operating_temp_c, 1.0),
    }
    return {name: float(np.clip(raw.get(name, 0.0) / limit, 0.0, 1.0)) for name, limit in limits.items()}
