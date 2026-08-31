"""Human-readable trace of the deterministic GNet9 decision pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_algorithm_dialogue(dynamics: dict[str, Any]) -> dict[str, Any]:
    """Build evidence-backed messages for every simulated time step.

    The dialogue is a rendered trace of values already calculated by the model,
    not communication between autonomous agents and not a new decision source.
    """
    records: list[dict[str, Any]] = []
    for snapshot in dynamics.get("snapshots", []):
        analysis = snapshot.get("arbitrator", {}).get("analysis", {})
        koopman = snapshot.get("koopman", {})
        remap = snapshot.get("arbitrator", {}).get("remap", {})
        game = snapshot.get("arbitrator", {}).get("game_theory", {})
        hierarchy = snapshot.get("arbitrator", {}).get("critical_node_hierarchy", [])
        state_hausdorff = analysis.get("state_hausdorff", {})
        normalized_hausdorff = float(state_hausdorff.get("normalized_distance", 0.0) or 0.0)
        risk = float(koopman.get("forecast_risk_score", 0.0) or 0.0)
        lyapunov_derivative = float(koopman.get("lyapunov_derivative_per_second", 0.0) or 0.0)
        selected = game.get("selected", {})
        top_kvu = hierarchy[0] if hierarchy else {}
        turns = [
            {
                "algorithm": "Hausdorff состояния",
                "evidence": {
                    "distance": state_hausdorff.get("distance"),
                    "normalized_distance": normalized_hausdorff,
                },
                "message_ru": (
                    "Отклонение от t0 низкое; само по себе не требует ремаппинга."
                    if normalized_hausdorff <= 0.25
                    else "Отклонение состояния от t0 выросло; арбитр учитывает его вместе с телеметрией."
                ),
            },
            {
                "algorithm": "Koopman/DMD",
                "evidence": {
                    "risk_score": risk,
                    "early_warning": bool(koopman.get("forecast_is_early_warning")),
                    "targets": koopman.get("forecast_target_ids", []),
                    "tensor_matrix_residual": koopman.get("tensor_matrix_koopman", {}).get("one_step_residual"),
                },
                "message_ru": (
                    "Причинный прогноз не видит подтверждённой угрозы."
                    if risk < 0.25
                    else "Прогноз обнаружил риск; для защитного действия всё ещё требуется фактическая атака или подтверждённый предвестник."
                ),
            },
            {
                "algorithm": "Ляпунов",
                "evidence": {
                    "value": koopman.get("lyapunov_value"),
                    "derivative_per_second": lyapunov_derivative,
                    "remap_pressure": koopman.get("lyapunov_remap_pressure"),
                },
                "message_ru": (
                    "Энергия состояния не растёт."
                    if lyapunov_derivative <= 0.0
                    else "Нормированная энергия растёт; её вклад усилит давление на защитный план."
                ),
            },
            {
                "algorithm": "КВУ",
                "evidence": top_kvu,
                "message_ru": (
                    "Иерархия КВУ пуста: нет L2-узла, обслуживающего Gold-путь."
                    if not top_kvu
                    else f"Наивысший приоритет у {top_kvu.get('node_id')}: "
                    f"{top_kvu.get('gold_subscriber_count', 0)} Gold-абонентов в его пути."
                ),
            },
            {
                "algorithm": "Арбитр и игра Нэша",
                "evidence": {
                    "selection_mode": game.get("selection_mode"),
                    "recommended_action": game.get("recommended_action"),
                    "defender_payoff": selected.get("defender_payoff"),
                    "attacker_payoff": selected.get("attacker_payoff"),
                },
                "message_ru": (
                    f"Арбитр выбрал {remap.get('action', 'NO_REMAP')}; "
                    f"игровая оценка рекомендует {game.get('recommended_action', 'NO_REMAP')}."
                ),
            },
            {
                "algorithm": "SDN intent controller",
                "evidence": remap.get("sdn_intent", {}),
                "message_ru": (
                    "Сформирован только план намерений для внутреннего overlay; внешнее оборудование не подключается."
                ),
            },
        ]
        records.append(
            {
                "step_index": snapshot.get("step_index"),
                "time_seconds": snapshot.get("time_seconds"),
                "turns": turns,
            }
        )
    return {
        "kind": "deterministic_explanatory_algorithm_trace",
        "semantics_ru": (
            "Текстовая расшифровка уже рассчитанных значений Hausdorff, Koopman, "
            "Ляпунова, КВУ, игры и SDN-плана; это не отдельный автономный алгоритм."
        ),
        "records": records,
    }


def export_algorithm_dialogue(dynamics: dict[str, Any], json_path: Path, markdown_path: Path) -> dict[str, Any]:
    """Write a machine-readable trace and a concise reviewer-friendly version."""
    dialogue = build_algorithm_dialogue(dynamics)
    json_path.write_text(json.dumps(dialogue, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Диалог алгоритмов GNet9", "", dialogue["semantics_ru"], ""]
    for record in dialogue["records"]:
        lines.extend(
            [
                f"## Шаг {record.get('step_index')} · t = {record.get('time_seconds')} с",
                "",
            ]
        )
        for turn in record["turns"]:
            lines.append(f"- **{turn['algorithm']}**: {turn['message_ru']}")
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return dialogue
