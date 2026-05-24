"""Project-wide constants for the G-Net baseline model.

This file intentionally contains only static configuration: names, colors,
service templates and topology sizes. Keeping these values in one place makes
it easier to tune the experiment without digging through the builder logic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceTemplate:
    """Input template for an L0 service before it is added to the graph."""

    name: str
    bitrate_mbps: float
    latency_ms_max: float
    jitter_ms_max: float
    availability_target: float
    priority: str


# Human-readable names of the 9 G-Net layers.
LEVEL_NAMES = {
    "L0": "Services",
    "L1": "Subscribers",
    "L2": "Active equipment",
    "L3": "Medium",
    "L4": "Linear infrastructure",
    "L5": "Core / slicing",
    "L6": "Infrastructure / power",
    "L7": "Arbitrator",
    "L8": "Topo-base",
}

# Colors are used only by the visualizer. They do not affect calculations.
LEVEL_COLORS = {
    "L0": "#d8f3dc",
    "L1": "#b7e4c7",
    "L2": "#a9def9",
    "L3": "#cdb4db",
    "L4": "#f3c4fb",
    "L5": "#ffc8a2",
    "L6": "#ffd166",
    "L7": "#f4a261",
    "L8": "#d9d9d9",
}

CRITICALITY_COLORS = {
    "gold": "#f4a261",
    "silver": "#8ecae6",
    "bronze": "#90be6d",
}

# L0 service baseline: normal t0 state, no attack and no overload.
DEFAULT_SERVICES = (
    ServiceTemplate("Voice", 0.128, 50.0, 10.0, 0.9995, "gold"),
    ServiceTemplate("Video", 8.0, 80.0, 20.0, 0.9990, "gold"),
    ServiceTemplate("FTP", 25.0, 300.0, 100.0, 0.9950, "silver"),
    ServiceTemplate("Telemetry", 0.256, 30.0, 5.0, 0.9999, "gold"),
)

# L1 subscriber generation settings.
MOBILE_SUBSCRIBERS_PER_AGG = 40
FIXED_SUBSCRIBERS_PER_AGG = 40
AGGREGATION_MOBILE = ("A1", "A3", "A5")
AGGREGATION_FIXED = ("A2", "A4", "A6")

# L2 active equipment: 12 core routers + 6 aggregation routers.
AGGREGATION_COUNT = 6
CORE_COUNT = 12
L2_NODE_COUNT = CORE_COUNT + AGGREGATION_COUNT

# Baseline normalization constants.
IDEAL_LOAD_FACTOR = 0.55

# Monitoring length used for L1 synthetic observations.
L1_MONITORING_SECONDS = 30
