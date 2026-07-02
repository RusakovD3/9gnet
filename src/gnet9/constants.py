"""Project-wide constants for the G-Net baseline model.

This file intentionally contains only static configuration: names, colors,
service templates and topology sizes. Keeping these values in one place makes
it easier to tune the experiment without digging through the builder logic.
"""

# Human-readable names of the 9 G-Net layers.
LEVEL_NAMES = {
    "L0": "Сервисы",
    "L1": "Абоненты",
    "L2": "Активное оборудование",
    "L3": "Среда передачи",
    "L4": "Линейная инфраструктура",
    "L5": "Ядро и срезы",
    "L6": "Инфраструктура и питание",
    "L7": "Арбитратор",
    "L8": "Топооснова",
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

# Localized names are used only in human-facing reports and visualizations.
# Stable English service identifiers remain unchanged in JSON and code.
SERVICE_DISPLAY_NAMES = {
    "Voice": "Голос",
    "VLC AV": "VLC: голос и видео",
    "FTP": "FTP",
    "DNS": "DNS",
    "Telemost": "Видеоконференция «Телемост»",
    "Live Streaming": "Прямая трансляция",
}

# L1 subscriber generation settings.
MOBILE_SUBSCRIBERS_PER_AGG = 40
FIXED_SUBSCRIBERS_PER_AGG = 40
AGGREGATION_MOBILE = ("A1", "A3", "A5")
AGGREGATION_FIXED = ("A2", "A4", "A6")

# L2 active equipment: 12 core routers + 6 aggregation switches.
AGGREGATION_COUNT = 6
CORE_COUNT = 12
L2_NODE_COUNT = CORE_COUNT + AGGREGATION_COUNT

# Реалистичная плановая нагрузка идеального t0. Значения не подменяют
# фактическую пакетную загрузку, которая отдельно рассчитывается по потокам.
SERVICE_DEMAND_PRESSURE = 0.20
BASELINE_LINK_UTILIZATION = {
    "logical-service-binding": 0.08,
    "fiber": 0.06,
    "ethernet": 0.08,
    "radio": 0.12,
    "radio-backhaul": 0.08,
}

# Monitoring length used for L1 synthetic observations.
L1_MONITORING_SECONDS = 30

# Discrete dynamics defaults. DYNAMICS_STEPS is the number of transitions after
# t0; with include_t0=True the exported trajectory has DYNAMICS_STEPS + 1
# snapshots.
DYNAMICS_STEP_SECONDS = 5
DYNAMICS_STEPS = 10
