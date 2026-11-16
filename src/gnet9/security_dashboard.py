"""Данные панели тревог: только сохранённые наблюдения, без новых порогов.

Срочность относится ко всей сети и читается из готовой рекомендации L7.
Цепочки показывают наличие наблюдения, а не доказательство проникновения.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .attacks import MITRE_TACTICS_BY_TECHNIQUE
from .constants import ATTACK_KIND_DISPLAY_NAMES


ATTACK_COLORS = {
    "dos": "#ef4444", "ddos": "#fb7185", "syn_flood": "#f472b6",
    "dns_flood": "#a78bfa", "application_flood": "#fb923c",
    "brute_force": "#22d3ee", "power_attack": "#facc15",
    "dhcp_flood": "#2dd4bf",
}
KIND_LABELS = {
    "dos": "DoS · перегрузка", "ddos": "DDoS · распределённая перегрузка",
    "syn_flood": "SYN · неполные соединения", "dns_flood": "DNS · частые запросы",
    "application_flood": "Приложение · тяжёлые запросы",
    "brute_force": "Вход · неудачные попытки", "power_attack": "Питание · нарушение",
}
LEVEL_LABELS = {
    "critical": "Критический", "high": "Высокий", "warning": "Подготовка",
    "watch": "Проверка", "info": "Наблюдение",
}
LEVEL_COLORS = {
    "critical": "#fb7185", "high": "#fb923c", "warning": "#fbbf24",
    "watch": "#93c5fd", "info": "#94a3b8",
}
STAGE_LEVELS = {
    "observe": "info", "confirm_observation": "watch", "prepare": "warning",
    "protect": "high", "emergency": "critical",
}
PHASE_LABELS = {
    "precursor": "Признаки угрозы", "active": "Наблюдаемое воздействие",
    "recovery": "Восстановление после воздействия", "history": "История наблюдений",
}
FLOOD_KINDS = {"dos", "ddos", "syn_flood", "dns_flood", "application_flood"}


def build_security_dashboard(snapshots: list[dict[str, Any]], step_index: int) -> dict[str, Any]:
    """Подготовить тревоги до выбранного индекса включительно.

    ``step_index`` — индекс сохранённого снимка, не номер шага модели.
    Время в полях first/last_seen_seconds измеряется в секундах модели.
    Например, признак в t=2 и воздействие в t=4 дают first_seen_seconds=2.
    Будущие снимки и расписание событий не читаются; входы не изменяются.
    """
    if not snapshots or not 0 <= step_index < len(snapshots):
        raise ValueError("Нужен существующий индекс снимка")
    current = snapshots[step_index]
    recommendation = deepcopy(current.get("arbitrator", {}).get("recommendation", {}))
    severity = _severity(current)
    records: dict[str, dict[str, Any]] = {}
    current_keys: set[str] = set()
    for index in range(step_index + 1):
        snapshot = snapshots[index]
        previous_keys = current_keys
        current_keys = set()
        now = float(snapshot.get("time_seconds", 0.0))
        for alarm in _snapshot_alarms(snapshot):
            key = alarm["key"]
            previous = records.get(key)
            if alarm["phase"] == "precursor" and key not in previous_keys:
                previous = None
            # Сопоставление с предвестником опирается на уже определённый тип
            # и место измерения. Неизвестный тип не приписывается будущей атаке.
            precursor_key = _sensor_key(alarm["target_id"], alarm["kind"])
            precursor = (records.get(precursor_key)
                         if alarm["phase"] != "precursor" and precursor_key in previous_keys else None)
            alarm["first_seen_seconds"] = (
                previous["first_seen_seconds"] if previous else
                precursor["first_seen_seconds"] if precursor else now
            )
            alarm["last_seen_seconds"] = now
            related = previous or precursor or {}
            history = list(related.get("observation_history", []))
            if not history or history[-1]["phase"] != alarm["phase"]:
                history.append({"phase": alarm["phase"], "time_seconds": now})
            alarm["observation_history"] = history
            alarm["mitre_technique_candidates"] = sorted(set(alarm["mitre_technique_candidates"])
                                                        | set(related.get("mitre_technique_candidates", [])))
            techniques = alarm["mitre_technique_ids"] or alarm["mitre_technique_candidates"]
            alarm["mitre_tactic_ids"] = sorted({MITRE_TACTICS_BY_TECHNIQUE[item]
                                               for item in techniques
                                               if item in MITRE_TACTICS_BY_TECHNIQUE})
            alarm["_precursor_seen"] = bool(
                alarm["phase"] == "precursor" or precursor or
                previous and previous.get("_precursor_seen")
            )
            alarm["_active_seen"] = bool(
                alarm["phase"] == "active" or previous and previous.get("_active_seen")
            )
            alarm["_impact_seen"] = bool(
                alarm["_impact_lines"] or previous and previous.get("_impact_seen")
            )
            alarm["_delivery_seen"] = bool(
                alarm["_delivery_seen"] or previous and previous.get("_delivery_seen")
            )
            records[key] = alarm
            current_keys.add(key)
            if precursor and precursor_key != key:
                records.pop(precursor_key)
                current_keys.discard(precursor_key)
    alarms = []
    for key, record in records.items():
        alarm = deepcopy(record)
        if key not in current_keys:
            alarm["phase"] = "history"
            alarm["severity"] = "info"
            alarm["evidence_lines"].append(
                "Сигнал больше не наблюдается; устранение причины не подтверждено."
            )
        alarm["phase_label"] = PHASE_LABELS[alarm["phase"]]
        alarm["severity_label"] = LEVEL_LABELS[alarm["severity"]]
        alarm["severity_color"] = LEVEL_COLORS[alarm["severity"]]
        alarm["severity_scope"] = "network"
        alarm["chain"] = _local_chain(alarm)
        alarm["classic_chain"] = _classic_chain(alarm)
        alarms.append({key: value for key, value in alarm.items() if not key.startswith("_")})
    level_order = list(LEVEL_LABELS)
    alarms.sort(key=lambda alarm: (
        alarm["phase"] == "history", level_order.index(alarm["severity"]),
        alarm["target_id"], alarm["kind"], alarm["key"],
    ))
    return {
        "alarms": alarms,
        "severity_counts": {
            level: sum(alarm["severity"] == level and alarm["phase"] != "history" for alarm in alarms)
            for level in LEVEL_LABELS
        },
        "history_count": sum(alarm["phase"] == "history" for alarm in alarms),
        "recommendation": recommendation,
        "severity": severity, "severity_label": LEVEL_LABELS[severity],
        "severity_color": LEVEL_COLORS[severity], "severity_scope": "network",
        "level_labels": dict(LEVEL_LABELS), "level_colors": dict(LEVEL_COLORS),
    }


def _severity(snapshot: dict[str, Any]) -> str:
    arbitrator = snapshot.get("arbitrator", {})
    stage = arbitrator.get("recommendation", {}).get("response_stage")
    if stage is None:
        stage = arbitrator.get("trigger_card", {}).get("response_stage")
    return STAGE_LEVELS.get(stage, "info")


def _sensor_key(target: str, kind: str) -> str:
    return f"sensor:{target}:{kind}"


def _base_alarm(key, target, kind, phase, severity, sources, routes, evidence, impact):
    return {
        "key": key, "target_id": target, "kind": kind,
        "kind_label": KIND_LABELS.get(kind, ATTACK_KIND_DISPLAY_NAMES.get(kind, "Тип не определён")),
        "color": ATTACK_COLORS.get(kind, "#cbd5e1"), "severity": severity,
        "phase": phase, "source_ids": sorted(set(map(str, sources))),
        "routes": [list(route) for route in routes if len(route) >= 2],
        "mitre_technique_ids": [],
        "mitre_technique_candidates": [],
        "evidence_lines": evidence, "_impact_lines": impact, "_delivery_seen": False,
    }


def _snapshot_alarms(snapshot):
    attacks = snapshot.get("attacks", {})
    koopman = snapshot.get("koopman", {})
    severity = _severity(snapshot)
    observations = attacks.get("target_observations", [])
    matches = {(item.get("target_id"), item.get("kind")): item
               for item in koopman.get("signature_entity_records", [])}
    result = []
    event_pairs = set()
    for index, event in enumerate(attacks.get("events", [])):
        target = str(event.get("target_id", ""))
        if not target:
            continue
        kind = str(event.get("kind", "unknown_degradation_or_attack"))
        event_pairs.add((target, kind))
        event_id = str(event.get("attack_id") or kind)
        phase = "recovery" if event.get("is_recovery_effect") else "active"
        observation = observations[index] if index < len(observations) else {}
        if observation.get("target_id") != target:
            observation = {}
        evidence, impact = _event_evidence(event, observation)
        numeric_record = matches.get((target, kind), {})
        match = numeric_record.get("feature_match")
        if match:
            evidence.append(f"Совпадение с числовой строкой стенда: {match['margin_ratio']:g} от порога.")
        sources = event.get("ingress_nodes", [])
        if kind == "ddos" and event.get("external_response_ingress_nodes"):
            # У отражения инициаторы запросов и входящие ответы — разные
            # участки. Цветной путь routes показывает именно ответы к цели.
            sources = event["external_response_ingress_nodes"]
        elif kind == "ddos" and event.get("external_segment_modeled") is False:
            sources = [route[0] for route in event.get("routes", []) if route]
        alarm = _base_alarm(
            f"event:{event_id}:{target}", target, kind, phase, severity,
            sources,
            event.get("routes", []) if kind in FLOOD_KINDS and phase == "active" else [],
            evidence, impact,
        )
        if event.get("attack_label_ru"):
            alarm["kind_label"] = event["attack_label_ru"]
        alarm["_delivery_seen"] = bool(
            phase == "active" and kind in FLOOD_KINDS and alarm["routes"] and
            (float(event.get("effective_packet_rate_pps", 0)) > 0 or
             float(event.get("effective_offered_rate_mbps", 0)) > 0)
        )
        if event.get("mitre_technique_id"):
            alarm["mitre_technique_ids"] = [event["mitre_technique_id"]]
        alarm["mitre_technique_candidates"] = numeric_record.get("mitre_technique_candidates", [])
        result.append(alarm)
    classified = {}
    for record in koopman.get("forecast_entity_records", []):
        if record.get("observation_type") == "precursor" and record.get("target_id"):
            classified.setdefault(str(record["target_id"]), []).append(record)
    sensors = {}
    for item in attacks.get("precursors", []):
        target = str(item.get("sensor_node_id", ""))
        if target:
            sensors.setdefault(target, []).append(item)
    if koopman.get("forecast_is_early_warning"):
        for target in koopman.get("forecast_target_ids", []):
            sensors.setdefault(str(target), [])
    for target, items in sensors.items():
        kinds = {str(record.get("kind", "unknown_degradation_or_attack")): record
                 for record in classified.get(target, [{}])}
        for kind, record in kinds.items():
            if (target, kind) in event_pairs:
                continue
            sources = [source for item in items for source in item.get("observed_source_node_ids", [])]
            if not sources:
                sources = record.get("source_ids", [])
            evidence = ["Место измерения: " + target, "Тип — оценка анализатора по текущим признакам."]
            if not items:
                evidence.append("Сохранено раннее предупреждение; исходные показания отсутствуют.")
            else:
                evidence.extend(_precursor_evidence(items))
            alarm = _base_alarm(
                _sensor_key(target, kind), target, kind, "precursor", severity,
                sources, [], evidence, [],
            )
            alarm["mitre_technique_candidates"] = record.get("mitre_technique_candidates", [])
            if record.get("attack_label_ru"):
                alarm["kind_label"] = record["attack_label_ru"]
            result.append(alarm)
    return result


def _precursor_evidence(items):
    # Сохранённые значения с исходными единицами; без повторной классификации.
    fields = {
        "scan_rate_pps": ("Проверочные обращения", "пак./с"),
        "authentication_failure_rate_per_second": ("Неудачные входы", "1/с"),
        "syn_backlog_growth_ratio": ("Рост очереди SYN", "доля"),
        "power_voltage_sag_ratio": ("Просадка входного напряжения", "доля"),
        "power_control_anomaly_ratio": ("Аномалия управления питанием", "доля"),
    }
    lines = []
    for field, (label, unit) in fields.items():
        values = [float(item.get("signals", {}).get(field, 0)) for item in items]
        value = max(values, default=0.0)
        if value > 0:
            lines.append(f"{label}: {value:g} {unit}")
    return lines or ["Датчик зарегистрировал локальную аномалию."]


def _event_evidence(event, observation):
    evidence, impact = [], []
    signals = event.get("sensor_signals", {})
    for field, label, units in (
        ("dhcp_service_pressure_ratio", "Нагрузка службы DHCP", "доля"),
        ("invalid_prefix_origin_count", "Несоответствия происхождения префикса", "событ."),
        ("unexpected_privilege_transition_count", "Неожиданные переходы привилегий", "событ."),
        ("distinct_probed_services_count", "Разные службы в журнале обращений", "служб"),
        ("user_launched_process_count", "Запуски по действию пользователя", "событ."),
        ("endpoint_cpu_rise_ratio", "Рост загрузки CPU", "доля"),
        ("unusual_egress_bytes_per_second", "Аномальная исходящая передача", "байт/с"),
        ("mobile_link_failure_ratio", "Сбой доступности мобильной линии", "доля"),
    ):
        if field in signals:
            evidence.append(f"{label}: {signals[field]:g} {units}")
    if event.get("kind") == "ddos" and event.get("external_response_ingress_nodes"):
        evidence.append("Вход отражённых ответов: " + ", ".join(event["external_response_ingress_nodes"]))
        evidence.append("Инициаторы исходящих запросов: " + (", ".join(event.get("ingress_nodes", [])) or "не сохранены"))
        if event.get("external_segment_modeled") is False:
            evidence.append("Внешний участок отражения не моделируется; первоисточник не подтверждён.")
    packet_attack = event.get("kind") in FLOOD_KINDS
    if packet_attack and "effective_offered_rate_mbps" in event:
        evidence.append(f"Входящий поток: {float(event['effective_offered_rate_mbps']):g} Мбит/с")
    if packet_attack and "effective_packet_rate_pps" in event:
        evidence.append(f"Атакующие поступления: {float(event['effective_packet_rate_pps']):g} пак./с")
    if "effective_failed_authentication_attempt_count" in event:
        evidence.append(f"Неудачные входы за шаг: {float(event['effective_failed_authentication_attempt_count']):g}")
        evidence.append("Успешный вход этими данными не подтверждён.")
    runtime = event.get("power_runtime", {})
    voltage = observation.get("ups_input_voltage_ratio", runtime.get("input_voltage_ratio"))
    if event.get("kind") == "power_attack" and voltage is not None:
        evidence.append(f"Входное напряжение ИБП: {100 * float(voltage):g}% от номинала")
        evidence.append("Просадка входа сама по себе не означает отключение услуги.")
    availability = observation.get("service_availability_ratio", runtime.get("service_availability_ratio"))
    if availability is not None:
        line = f"Расчётная доступность услуги на цели: {100 * float(availability):g}%"
        evidence.append(line)
        if float(availability) < 1.0:
            impact.append(line)
    if event.get("is_recovery_effect"):
        evidence.append("Сохранено остаточное воздействие во время загрузки и проверки готовности.")
    if not evidence:
        evidence.append("Событие наблюдается; отдельные показатели не сохранены.")
    return evidence, impact


def _local_chain(alarm):
    historical = alarm["phase"] == "history"
    precursor = alarm["_precursor_seen"]
    active = alarm["_active_seen"]
    impact = alarm["_impact_seen"]
    return [
        {"key": "precursor", "label": "Признаки", "status": (
            "current" if alarm["phase"] == "precursor" else "observed" if precursor else "unknown"
        ), "detail": "Зарегистрированы датчиком" if precursor else "Предвестник не наблюдался"},
        {"key": "active", "label": "Воздействие", "status": (
            "current" if alarm["phase"] == "active" else "observed" if active else "unknown"
        ), "detail": "Сохранено наблюдаемое событие" if active else "Не подтверждено"},
        {"key": "impact", "label": "Последствия", "status": (
            "current" if alarm["_impact_lines"] and not historical else "observed" if impact else "unknown"
        ), "detail": "Есть расчёт снижения доступности цели" if impact else "Отдельный ущерб не подтверждён"},
        {"key": "verification", "label": "Проверка", "status": "unknown",
         "detail": ("Восстановление наблюдается; устранение требует проверки"
                    if alarm["phase"] == "recovery" else "Устранение требует подтверждения человеком")},
    ]


def _classic_chain(alarm):
    delivery = alarm["_delivery_seen"]
    objective = alarm["_impact_seen"] and alarm["kind"] != "brute_force"
    stages = [
        ("reconnaissance", "Разведка", False, "Намерение проверочных обращений не установлено"),
        ("weaponization", "Подготовка", False, "Данных о подготовке средств атаки нет"),
        ("delivery", "Доставка", delivery, "Наблюдаются пакеты по сохранённому маршруту" if delivery else "Доставка средств атаки не установлена"),
        ("exploitation", "Эксплуатация", False, "Эксплуатация уязвимости не подтверждена"),
        ("installation", "Закрепление", False, "Установка вредоносного средства не наблюдалась"),
        ("command_control", "Управление", False, "Канал управления злоумышленника не установлен"),
        ("objective", "Цель", objective, "Зафиксирована расчётная потеря доступности цели" if objective else "Достижение цели злоумышленника не установлено"),
    ]
    return [{"key": key, "label": label, "status": "observed" if seen else "unknown", "detail": detail}
            for key, label, seen, detail in stages]
