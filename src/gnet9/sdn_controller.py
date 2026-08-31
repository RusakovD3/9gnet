"""Safe SDN intent compiler for GNet9 remapping.

This is deliberately *not* an OpenFlow controller process.  It converts an L7
decision into a serialisable, reviewable intent that the in-memory remapping
engine consumes.  It opens no sockets, does not import a controller runtime and
cannot configure a device.  A real OS-Ken/OpenFlow adapter needs separately
approved controller credentials, a testbed and device-specific validation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def compile_sdn_intent(
    *,
    step_index: int,
    remap: dict[str, Any],
    defense_plan: dict[str, Any] | None,
    game: dict[str, Any],
) -> dict[str, Any]:
    """Compile a deterministic simulation-only SDN intent from L7 evidence."""
    defense_plan = defense_plan or {}
    control = defense_plan.get("routing_control", {})
    target_ids = sorted(
        str(value) for value in control.get("target_ids", []) if value
    )
    source_ids = sorted(
        str(value) for value in control.get("source_ids", []) if value
    )
    action = str(remap.get("action", "NO_REMAP"))
    requested_actions = list(remap.get("candidate_actions", []))
    selected = game.get("selected", {})
    payload = {
        "step_index": int(step_index),
        "action": action,
        "targets": target_ids,
        "sources": source_ids,
        "game_action": game.get("recommended_action"),
        "requested_actions": requested_actions,
    }
    intent_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    active = bool(defense_plan and remap.get("needed"))
    return {
        "intent_id": f"gnet9-sdn-{intent_id}",
        "controller": "GNet9 simulation SDN intent controller",
        "execution_mode": "simulation_only_no_network_io",
        "southbound_transport": "none",
        "external_controller_adapter": {
            "candidate": "OS-Ken / OpenFlow 1.3",
            "status": "not_connected_by_design",
            "reason": (
                "the model has no authorised controller endpoint, credentials or testbed; "
                "the intent is reviewable before any external deployment"
            ),
        },
        "active": active,
        "intent": {
            "action": action,
            "recommended_game_action": game.get("recommended_action"),
            "selection_mode": game.get("selection_mode"),
            "target_ids": target_ids,
            "source_ids": source_ids,
            "priority_policy": "gold_then_silver_then_bronze",
            "capacity_guard_percent": 80.0,
            "requested_actions": requested_actions,
            "selected_defender_payoff": selected.get("defender_payoff"),
            "selected_attacker_payoff": selected.get("attacker_payoff"),
        },
        "application_contract": {
            "consumer": "apply_gold_first_remap",
            "effect": "in_memory_route_overlay_only",
            "never": ["open_socket", "send_openflow", "configure_network_device"],
        },
    }
