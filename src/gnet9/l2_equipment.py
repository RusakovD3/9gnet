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
    thermal_output_btu_per_hour: float | None = None
    thermal_output_equivalent_w: float | None = None
    power_value_semantics: str = "not_published_for_selected_profile"
    reference_urls: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Numeric fields absent from a public data sheet remain useful as model limits,
# but they are listed in assumed_fields and must be labelled as such in UI/export.
L2_EQUIPMENT_PROFILES: dict[str, L2EquipmentProfile] = {
    "C9500_AGG": L2EquipmentProfile(
        name="C9500_AGG",
        vendor="Cisco",
        model_family="Catalyst C9500-24Y4C",
        role="aggregation-switch",
        source_note=(
            "Cisco Catalyst 9500 data sheet, C9500-24Y4C row: 2.0 Tbps, "
            "1 Bpps, 16 GB DRAM, 36 MB UADP 3.0 buffer, and up to 45 C below "
            "6000 ft. Its standard SDM core template publishes 212,000 "
            "combined IPv4/IPv6 LPM/host entries. "
            "The same data sheet gives 1454 BTU/h total output with a 650 W AC "
            "PSU (about 426 W using Cisco's 1000 BTU/h = 293 W conversion); "
            "this is a thermal upper bound, not a published typical electrical draw."
        ),
        source_url="https://www.cisco.com/c/en/us/products/collateral/switches/catalyst-9500-series-switches/nb-06-cat9500-ser-data-sheet-cte-en.html",
        verified_fields=(
            "throughput_gbps", "forwarding_mpps", "dram_gb", "buffer_mb",
            "port_configuration", "operating_temp_c", "fib_routes",
            "thermal_output_btu_per_hour",
            "thermal_output_equivalent_w",
        ),
        assumed_fields=("tcam_entries", "crypto_gbps", "control_plane_sessions"),
        port_configuration="24 x 1/10/25G + 4 x 40/100G",
        throughput_gbps=2_000.0,
        forwarding_mpps=1_000.0,
        dram_gb=16.0,
        fib_routes=212_000,
        tcam_entries=256_000,
        buffer_mb=36.0,
        crypto_gbps=0.0,
        control_plane_sessions=2_000,
        operating_temp_c=45.0,
        thermal_output_btu_per_hour=1_454.0,
        thermal_output_equivalent_w=426.0,
        power_value_semantics=(
            "thermal_equivalent_upper_bound_from_1454_btu_per_hour; "
            "not_typical_electrical_input_power"
        ),
    ),
    "NCS5501_CORE": L2EquipmentProfile(
        name="NCS5501_CORE",
        vendor="Cisco",
        model_family="NCS 5501",
        role="core-router",
        source_note=(
            "Cisco NCS 5500 data sheet: NCS-5501 system throughput up to 800 Gbps, "
            "up to 1M FIB, and output power 240 W typical at 25 C / 370 W maximum "
            "at 55 C. The Cisco fixed-platform architecture white paper adds "
            "720 MPPS, 32 GB DRAM and separate 16 MB on-chip plus 4 GB off-chip "
            "buffers. Cisco says facility input power is output divided by 0.91; "
            "the remaining limits below are explicit simulation assumptions."
        ),
        source_url="https://www.cisco.com/c/en/us/products/collateral/routers/network-convergence-system-5500-series/datasheet-c78-737935.html",
        verified_fields=(
            "throughput_gbps", "forwarding_mpps", "dram_gb", "fib_routes",
            "buffer_mb", "port_configuration", "operating_temp_c",
            "typical_power_w", "max_power_w",
        ),
        assumed_fields=("tcam_entries", "crypto_gbps", "control_plane_sessions"),
        port_configuration="48 x 1/10G + 6 x 40/100G",
        throughput_gbps=800.0,
        forwarding_mpps=720.0,
        dram_gb=32.0,
        fib_routes=1_000_000,
        tcam_entries=512_000,
        # Единое поле модели хранит суммарную физическую ёмкость двух уровней:
        # 16 MB on-chip + 4 GiB off-chip. Это не означает единую FIFO-очередь.
        buffer_mb=4_112.0,
        crypto_gbps=0.0,
        control_plane_sessions=4_000,
        operating_temp_c=55.0,
        typical_power_w=240.0,
        max_power_w=370.0,
        power_value_semantics=(
            "vendor_output_power; typical_at_25_celsius; maximum_at_55_celsius; "
            "facility_input_equals_output_divided_by_0.91"
        ),
        reference_urls=(
            "https://www.cisco.com/c/en/us/products/collateral/routers/"
            "network-convergence-system-5500-series/ncs-5500-5700-platform-ar-wp.html",
        ),
    ),
    "RAN_ACCESS_EDGE": L2EquipmentProfile(
        name="RAN_ACCESS_EDGE",
        vendor="GNet9 scenario",
        model_family="Integrated gNodeB/UPF access edge",
        role="radio-access-node",
        source_note=(
            "Scenario capacity envelope for an active mobile access node. "
            "It represents a colocated gNodeB/UPF edge attachment used to make "
            "subscriber access explicit in the graph. Numeric limits are GNet9 "
            "engineering assumptions, not vendor-published chassis claims."
        ),
        source_url="gnet9://scenario/l2/radio-access-node",
        verified_fields=("role", "port_configuration"),
        assumed_fields=(
            "throughput_gbps", "forwarding_mpps", "dram_gb", "fib_routes",
            "tcam_entries", "buffer_mb", "crypto_gbps",
            "control_plane_sessions", "operating_temp_c",
        ),
        port_configuration="4 x 10/25G SFP28 backhaul + timing/control",
        throughput_gbps=100.0,
        forwarding_mpps=150.0,
        dram_gb=16.0,
        fib_routes=128_000,
        tcam_entries=64_000,
        buffer_mb=256.0,
        crypto_gbps=20.0,
        control_plane_sessions=5_000,
        operating_temp_c=50.0,
        power_value_semantics="gnet9_scenario_power_envelope_not_vendor_spec",
    ),
    "OLT_ACCESS": L2EquipmentProfile(
        name="OLT_ACCESS",
        vendor="GNet9 scenario",
        model_family="XGS-PON OLT access shelf",
        role="optical-line-terminal",
        source_note=(
            "Scenario capacity envelope for a fixed-access OLT shelf. "
            "It makes the ONU/OLT domain explicit while keeping individual ONUs "
            "abstracted at L1. Numeric limits are GNet9 engineering assumptions."
        ),
        source_url="gnet9://scenario/l2/optical-line-terminal",
        verified_fields=("role", "port_configuration"),
        assumed_fields=(
            "throughput_gbps", "forwarding_mpps", "dram_gb", "fib_routes",
            "tcam_entries", "buffer_mb", "crypto_gbps",
            "control_plane_sessions", "operating_temp_c",
        ),
        port_configuration="8 x XGS-PON service ports + 4 x 10/25G uplinks",
        throughput_gbps=160.0,
        forwarding_mpps=120.0,
        dram_gb=12.0,
        fib_routes=64_000,
        tcam_entries=32_000,
        buffer_mb=128.0,
        crypto_gbps=0.0,
        control_plane_sessions=3_000,
        operating_temp_c=45.0,
        power_value_semantics="gnet9_scenario_power_envelope_not_vendor_spec",
    ),
}


def l2_profile_for_role(role: str) -> L2EquipmentProfile:
    """Pick the equipment profile assigned to a G-Net L2 role."""
    if role == "core-router":
        return L2_EQUIPMENT_PROFILES["NCS5501_CORE"]
    if role == "aggregation-switch":
        return L2_EQUIPMENT_PROFILES["C9500_AGG"]
    if role == "radio-access-node":
        return L2_EQUIPMENT_PROFILES["RAN_ACCESS_EDGE"]
    if role == "optical-line-terminal":
        return L2_EQUIPMENT_PROFILES["OLT_ACCESS"]
    raise ValueError(f"Unsupported L2 equipment role: {role}")


def build_l2_raw_baseline(profile: L2EquipmentProfile, *, role: str, criticality: str) -> dict[str, float]:
    """Return baseline raw telemetry values for one active device.

    Raw values use real units where possible: Gbps, Mpps, routes, entries, MB,
    sessions and Celsius. The tensor receives selected raw and normalized values.
    """
    # Здоровый baseline несёт сотни Мбит/с, тогда как шасси дают сотни Гбит/с
    # или Тбит/с. Значения детерминированы и включают небольшой фоновый запас;
    # это не телеметрия, считанная с физического устройства Cisco.
    role_factor = {
        "core-router": 0.00025,
        "aggregation-switch": 0.00020,
        "radio-access-node": 0.00035,
        "optical-line-terminal": 0.00030,
    }[role]
    grade_factor = 1.05 if criticality == "gold" else 1.0
    crypto_baseline = 0.0
    cpu_base = {
        "core-router": 16.0 * grade_factor,
        "aggregation-switch": 12.0,
        "radio-access-node": 18.0,
        "optical-line-terminal": 10.0,
    }[role]
    ram_ratio = {
        "core-router": 0.35,
        "aggregation-switch": 0.30,
        "radio-access-node": 0.28,
        "optical-line-terminal": 0.25,
    }[role]
    scale_ratio = {
        "core-router": 0.10,
        "aggregation-switch": 0.08,
        "radio-access-node": 0.09,
        "optical-line-terminal": 0.07,
    }[role]
    tcam_ratio = {
        "core-router": 0.12,
        "aggregation-switch": 0.10,
        "radio-access-node": 0.11,
        "optical-line-terminal": 0.08,
    }[role]
    queue_ratio = {
        "core-router": 0.05,
        "aggregation-switch": 0.04,
        "radio-access-node": 0.06,
        "optical-line-terminal": 0.05,
    }[role]
    control_ratio = {
        "core-router": 0.05,
        "aggregation-switch": 0.04,
        "radio-access-node": 0.07,
        "optical-line-terminal": 0.05,
    }[role]
    temperature = {
        "core-router": 32.0,
        "aggregation-switch": 30.0,
        "radio-access-node": 33.0,
        "optical-line-terminal": 31.0,
    }[role]

    return {
        L2Resource.CPU.value: cpu_base,
        L2Resource.RAM.value: profile.dram_gb * ram_ratio,
        L2Resource.FIB.value: profile.fib_routes * scale_ratio,
        L2Resource.TCAM.value: profile.tcam_entries * tcam_ratio,
        L2Resource.PPS.value: profile.forwarding_mpps * role_factor,
        L2Resource.THROUGHPUT.value: profile.throughput_gbps * role_factor,
        L2Resource.QUEUE.value: profile.buffer_mb * queue_ratio,
        L2Resource.CRYPTO.value: crypto_baseline,
        L2Resource.CONTROL_PLANE.value: profile.control_plane_sessions * control_ratio,
        L2Resource.TEMPERATURE.value: temperature,
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
