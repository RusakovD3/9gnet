"""Cisco-like L2 active equipment model for G-Net.

The numbers are intentionally vendor-realistic, not device-emulator-exact: they
come from Cisco public datasheets / architecture papers and are used as capacity
ceilings and exported raw telemetry for the reproducible baseline model. The
human-readable L2 tensor operating point lives in `baseline.py`.
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
    """Capacity envelope for a Cisco-like active network device."""

    name: str
    vendor: str
    model_family: str
    role: str
    source_note: str
    source_url: str
    throughput_gbps: float
    forwarding_mpps: float
    dram_gb: float
    fib_routes: int
    tcam_entries: int
    buffer_mb: float
    crypto_gbps: float
    control_plane_sessions: int
    operating_temp_c: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Public Cisco figures used as modeling anchors:
# - ASR1001-X: up to 20 Gbps forwarding throughput, 8 GB DRAM; Cisco HIG/data sheet.
# - ASR1001-X integrated ESP: up to 19 Mpps, up to 8 Gbps bandwidth depending license/features.
# - Catalyst 9500 high-performance fixed core: up to 12.8 Tbps full-duplex, up to 8 Bpps.
# - NCS-5501: up to 800 Gbps throughput, up to 1M FIB entries, 1+1 power redundancy.
# These values are not meant to certify a specific product configuration. They are
# normalized capacity ceilings for the G-Net simulation.
CISCO_LIKE_PROFILES: dict[str, L2EquipmentProfile] = {
    "ASR1001X_EDGE": L2EquipmentProfile(
        name="ASR1001X_EDGE",
        vendor="Cisco-like",
        model_family="ASR 1001-X",
        role="aggregation-router",
        source_note="Cisco ASR 1001-X public data sheet / hardware installation guide: 20 Gbps forwarding throughput class, 8 GB DRAM; integrated ESP table lists up to 19 Mpps.",
        source_url="https://www.cisco.com/c/en/us/products/collateral/routers/asr-1000-series-aggregation-services-routers/datasheet-c78-731632.html",
        throughput_gbps=20.0,
        forwarding_mpps=19.0,
        dram_gb=8.0,
        fib_routes=1_000_000,
        tcam_entries=250_000,
        buffer_mb=64.0,
        crypto_gbps=8.0,
        control_plane_sessions=1_000,
        operating_temp_c=40.0,
    ),
    "C9500_AGG": L2EquipmentProfile(
        name="C9500_AGG",
        vendor="Cisco-like",
        model_family="Catalyst 9500",
        role="aggregation-router",
        source_note="Cisco Catalyst 9500 data sheet / architecture paper: core/aggregation switch family, high-performance variants up to Tbps switching and Bpps forwarding classes.",
        source_url="https://www.cisco.com/c/en/us/products/collateral/switches/catalyst-9500-series-switches/nb-06-cat9500-ser-data-sheet-cte-en.html",
        throughput_gbps=2_000.0,
        forwarding_mpps=1_000.0,
        dram_gb=16.0,
        fib_routes=64_000,
        tcam_entries=256_000,
        buffer_mb=80.0,
        crypto_gbps=0.0,
        control_plane_sessions=2_000,
        operating_temp_c=40.0,
    ),
    "NCS5501_CORE": L2EquipmentProfile(
        name="NCS5501_CORE",
        vendor="Cisco-like",
        model_family="NCS 5501",
        role="core-router",
        source_note="Cisco NCS 5501 public data sheet: up to 800 Gbps system throughput and up to 1M FIB entries.",
        source_url="https://www.cisco.com/c/en/us/products/collateral/routers/network-convergence-system-5500-series/datasheet-c78-737935.html",
        throughput_gbps=800.0,
        forwarding_mpps=800.0,
        dram_gb=32.0,
        fib_routes=1_000_000,
        tcam_entries=512_000,
        buffer_mb=256.0,
        crypto_gbps=0.0,
        control_plane_sessions=4_000,
        operating_temp_c=40.0,
    ),
}


def l2_profile_for_role(role: str, *, criticality: str = "silver") -> L2EquipmentProfile:
    """Pick a Cisco-like profile for a G-Net L2 node role."""
    if role == "core-router":
        return CISCO_LIKE_PROFILES["NCS5501_CORE"]
    if role == "aggregation-router" and criticality == "gold":
        return CISCO_LIKE_PROFILES["ASR1001X_EDGE"]
    return CISCO_LIKE_PROFILES["C9500_AGG"]


def build_l2_raw_baseline(profile: L2EquipmentProfile, *, role: str, criticality: str) -> dict[str, float]:
    """Return baseline raw telemetry values for one active device.

    Raw values use real units where possible: Gbps, Mpps, routes, entries, MB,
    sessions and Celsius. The tensor receives selected raw and normalized values.
    """
    role_factor = 0.46 if role == "core-router" else 0.38
    grade_factor = 1.08 if criticality == "gold" else 1.0
    crypto_baseline = min(profile.crypto_gbps * 0.22, 1.6) if profile.crypto_gbps else 0.0

    return {
        L2Resource.CPU.value: 24.0 * grade_factor if role == "core-router" else 18.0,
        L2Resource.RAM.value: profile.dram_gb * (0.40 if role == "core-router" else 0.34),
        L2Resource.FIB.value: profile.fib_routes * (0.52 if role == "core-router" else 0.25),
        L2Resource.TCAM.value: profile.tcam_entries * (0.43 if role == "core-router" else 0.31),
        L2Resource.PPS.value: profile.forwarding_mpps * role_factor,
        L2Resource.THROUGHPUT.value: profile.throughput_gbps * role_factor,
        L2Resource.QUEUE.value: profile.buffer_mb * (0.33 if role == "core-router" else 0.28),
        L2Resource.CRYPTO.value: crypto_baseline,
        L2Resource.CONTROL_PLANE.value: profile.control_plane_sessions * (0.18 if role == "core-router" else 0.12),
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
