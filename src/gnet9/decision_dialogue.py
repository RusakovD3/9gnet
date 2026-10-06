"""Пошаговое объяснение уже рассчитанных решений GNet9."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import ACTION_DISPLAY_NAMES
from .arbitrator import TRIGGER_BANDS


def build_algorithm_dialogue(dynamics: dict[str, Any]) -> dict[str, Any]:
    """Объяснить результаты каждого шага, не выполняя новых расчётов защиты."""
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
        card = snapshot.get("arbitrator", {}).get("trigger_card", {})
        training = snapshot.get("training", {})
        turns = [
            {
                "algorithm": "Хаусдорф: отличие от эталона",
                "evidence": {
                    "distance": state_hausdorff.get("distance"),
                    "normalized_distance": normalized_hausdorff,
                },
                "message_ru": (
                    "Отклонение от исходного состояния небольшое; менять маршруты по этой причине не требуется."
                    if normalized_hausdorff < TRIGGER_BANDS["hausdorff"][0]
                    else "Отклонение от исходного состояния выросло; при выборе защиты учитываются и остальные наблюдения."
                ),
            },
            {
                "algorithm": "Прогноз Купмана",
                "evidence": {
                    "risk_score": risk,
                    "early_warning": bool(koopman.get("forecast_is_early_warning")),
                    "targets": koopman.get("forecast_target_ids", []),
                    "tensor_matrix_residual": koopman.get("tensor_matrix_koopman", {}).get("one_step_residual"),
                },
                "message_ru": (
                    "Оценка риска по текущим и прошлым наблюдениям невысока."
                    if risk < TRIGGER_BANDS["koopman"][0]
                    else "Оценка риска выросла; защита требует подтверждённого события или повторяющихся признаков угрозы."
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
                    "Мера отклонения состояния не растёт."
                    if lyapunov_derivative <= 0.0
                    else "Мера отклонения растёт; это усиливает основание для защиты."
                ),
            },
            {
                "algorithm": "Критически важные узлы",
                "evidence": top_kvu,
                "message_ru": (
                    "Не найдено сетевых устройств на путях абонентов золотого класса."
                    if not top_kvu
                    else f"Наивысший приоритет у {top_kvu.get('node_id')}: "
                    f"через него обслуживаются {top_kvu.get('gold_subscriber_count', 0)} абонентов золотого класса."
                ),
            },
            {
                "algorithm": "Выбор защитного действия",
                "evidence": {
                    "selection_mode": game.get("selection_mode"),
                    "recommended_action": game.get("recommended_action"),
                    "defender_payoff": selected.get("defender_payoff"),
                    "attacker_payoff": selected.get("attacker_payoff"),
                },
                "message_ru": (
                    f"Решение: {ACTION_DISPLAY_NAMES.get(remap.get('action', 'NO_REMAP'), 'неизвестное действие')}; "
                    f"сравнение вариантов рекомендует: {ACTION_DISPLAY_NAMES.get(game.get('recommended_action', 'NO_REMAP'), 'неизвестное действие')}."
                ),
            },
            {
                "algorithm": "План управления сетью",
                "evidence": remap.get("sdn_intent", {}),
                "message_ru": (
                    "План описывает временные изменения внутри модели. Команды реальному оборудованию не отправляются."
                ),
            },
        ]
        records.append(
            {
                "step_index": snapshot.get("step_index"),
                "time_seconds": snapshot.get("time_seconds"),
                "turns": turns,
                "trigger_card": card,
                "training": training,
            }
        )
    return {
        "kind": "deterministic_explanatory_algorithm_trace",
        "semantics_ru": (
            "Пояснения к сравнению с эталоном, прогнозу и выбору защиты. "
            "Текст составлен из результатов расчёта и не принимает самостоятельных решений."
        ),
        "records": records,
    }


def export_algorithm_dialogue(dynamics: dict[str, Any], json_path: Path, markdown_path: Path) -> dict[str, Any]:
    """Сохранить пояснения как данные и как текст для чтения."""
    dialogue = build_algorithm_dialogue(dynamics)
    json_path.write_text(json.dumps(dialogue, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Как GNet9 выбирает защиту", "", dialogue["semantics_ru"], ""]
    for record in dialogue["records"]:
        lines.extend(
            [
                f"## Шаг {record.get('step_index')} · t = {record.get('time_seconds')} с",
                "",
            ]
        )
        for turn in record["turns"]:
            lines.append(f"- **{turn['algorithm']}**: {turn['message_ru']}")
        training = record["training"]
        lines.append(f"- **Обучение**: использовано исправных переходов — {training.get('nominal_transitions', 0)}; обновление модели для следующего шага — {'да' if training.get('operator_updated') else 'нет'}.")
        card = record["trigger_card"]
        for name, label in (("hausdorff", "Хаусдорф"), ("koopman", "Купман"), ("lyapunov", "Ляпунов")):
            signal = card.get("signals", {}).get(name, {})
            if signal:
                lines.append(f"- **Диапазон {label}**: {signal['value']:.3f} — {signal['label_ru']}.")
        lines.append(f"- **Основание решения**: {card.get('reason_ru', 'нет данных')}")
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return dialogue
