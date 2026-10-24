"""Подписи и карточки сети; чтение наблюдений без рисования и новых решений."""

from collections import Counter
from typing import Any

from .constants import ACTION_DISPLAY_NAMES, ATTACK_KIND_DISPLAY_NAMES, SERVICE_DISPLAY_NAMES

FLOW_SHORT_LABELS = {
    "RTP_OPUS": "Голос · Opus/RTP",
    "RTP_VLC_AV": "VLC · аудио/видео",
    "FTP_DATA": "Файлы · FTP/TCP",
    "DNS": "DNS · запросы/UDP",
    "RTP_TELEMOST": "Видеоконференция",
    "LIVE_HLS": "Прямая трансляция",
    "ATTACK_DOS": "АТАКА · DoS",
    "ATTACK_DDOS": "АТАКА · DDoS",
    "ATTACK_SYN": "АТАКА · SYN-флуд",
}
TRAFFIC_LABELS = {
    "voice": "голос", "vlc_av": "VLC: голос и видео",
    "ftp": "передача файлов FTP", "dns": "запросы DNS",
    "video_conference": "видеоконференция", "live_streaming": "прямая трансляция",
}
SLA_LABELS = {"gold": "золотой", "silver": "серебряный", "bronze": "бронзовый"}
SLO_METRIC_LABELS = {
    "one_way_mouth_to_ear_latency": "односторонняя задержка речи",
    "network_packet_loss": "сетевая потеря пакетов",
    "jitter": "разброс задержки",
    "one_way_media_latency": "односторонняя задержка медиапотока",
    "media_bitrate": "полезная скорость медиапотока",
    "goodput": "полезная скорость передачи",
    "file_completion_time": "время передачи файла",
    "transfer_success": "успешность передачи",
    "tcp_retransmission_ratio": "доля повторных передач TCP",
    "response_time_p95_p99": "время ответа для 95% / 99% запросов",
    "timeout_ratio": "доля запросов без ответа вовремя",
    "servfail_ratio": "доля ответов с ошибкой сервера",
    "tcp_fallback_success": "успешность перехода DNS на TCP",
    "audio_one_way_latency": "односторонняя задержка звука",
    "video_one_way_latency": "односторонняя задержка видео",
    "media_loss": "потеря медиапакетов",
    "live_edge_latency": "отставание от прямого эфира",
    "startup_time": "время запуска воспроизведения",
    "rebuffer_ratio": "доля времени ожидания загрузки",
    "part_deadline_miss_ratio": "доля опоздавших частей трансляции",
}
MODEL_LIMITATION_LABELS = {
    "latency_and_jitter_fields_are_compatibility_guardrails_not_primary_ftp_slo": (
        "основные требования к передаче файла — скорость и время завершения"
    ),
    "bitrate_is_equivalent_load_for_generator_not_a_dns_service_slo": (
        "поиск сетевого адреса оценивается по времени ответа и его успешности"
    ),
    "srtp_srtcp_and_dtls_byte_overhead_not_yet_counted": (
        "защита медиапотока задана профилем услуги"
    ),
    "jitter_field_is_a_transport_guardrail_not_primary_ll_hls_user_slo": (
        "трансляция оценивается по отставанию от эфира и непрерывности просмотра"
    ),
}

def _flow_summary(flows: list[dict[str, Any]]) -> str:
    apps = Counter(flow.get("application", "unknown") for flow in flows)
    legitimate_count = sum(1 for flow in flows if not flow.get("is_attack_traffic"))
    attack_count = len(flows) - legitimate_count
    packets = sum(int(flow.get("packet_count", 0)) for flow in flows)
    wire_bytes = sum(int(flow.get("wire_bytes", 0)) for flow in flows)
    routed_flows = [
        flow for flow in flows
        if _flow_is_visible(flow)
    ]
    routed_wire_bytes = sum(int(flow.get("wire_bytes", 0)) for flow in routed_flows)
    delivered_packets = sum(
        max(0, int(flow.get("packet_count", 0)) - int(flow.get("observed_dropped_packets", 0)))
        for flow in routed_flows
    )
    interval = int(flows[0].get("interval_seconds", 0)) if flows else 0
    megabytes = wire_bytes / 1_000_000
    offered_rate_mbps = wire_bytes * 8.0 / max(interval, 1) / 1_000_000
    carried_rate_mbps = routed_wire_bytes * 8.0 / max(interval, 1) / 1_000_000
    rerouted = sum(bool(flow.get("rerouted")) for flow in flows if not flow.get("is_attack_traffic"))
    failover = sum(bool(flow.get("failover_active")) for flow in flows if not flow.get("is_attack_traffic"))
    isolated = sum(bool(flow.get("isolated")) for flow in flows if not flow.get("is_attack_traffic"))
    lines = [
        f"Обычные потоки: {legitimate_count}",
        "По одному потоку на абонента",
        f"Атакующие агрегированные потоки: {attack_count}",
        f"Сформировано пакетов за {interval} с: {packets:,}",
        f"Доставлено пакетов: {delivered_packets:,}",
        f"Предложено: {megabytes:,.1f} МБ · {offered_rate_mbps:,.1f} Мбит/с",
        f"Передано по маршрутам: {routed_wire_bytes / 1_000_000:,.1f} МБ · {carried_rate_mbps:,.1f} Мбит/с",
        f"Переназначено / резерв / изолировано: {rerouted} / {failover} / {isolated}",
        "Приложения:",
    ]
    lines.extend(f"  {FLOW_SHORT_LABELS.get(name, name)}: {count}" for name, count in sorted(apps.items()))
    lines.extend(["", "Цвет линии — основная услуга.", "Толщина — общий объём на линии."])
    return "\n".join(lines)


def _flow_is_visible(flow: dict[str, Any]) -> bool:
    """Анимация показывает выборку только реально маршрутизируемых потоков."""
    return (len(flow.get("route") or []) >= 2 and not flow.get("isolated")
            and not flow.get("quarantined") and int(flow.get("packet_count", 0)) > 0
            and flow.get("route_available", True) is not False)


def _interactive_status_sections(snapshot, flows, loads, paused: bool, model):
    """Готовые наблюдения и решения; интерфейс не пересчитывает пороги."""
    attacks = snapshot.get("attacks", {})
    koopman = snapshot.get("koopman", {})
    routing = attacks.get("routing", {})
    step = int(snapshot.get("step_index", 0))
    now = float(snapshot.get("time_seconds", 0.0))
    warning = bool(koopman.get("forecast_is_early_warning"))
    if attacks.get("active"):
        state, color = "НАБЛЮДАЕМАЯ АТАКА", "#fb7185"
    elif attacks.get("precursor_active"):
        state, color = "НАБЛЮДАЮТСЯ ПРИЗНАКИ УГРОЗЫ", "#fbbf24"
    elif koopman.get("risk_state", "nominal") != "nominal":
        state, color = "ПРОДОЛЖАЕТСЯ НАБЛЮДЕНИЕ", "#fbbf24"
    else:
        state, color = "ИСПРАВНАЯ СЕТЬ", "#86efac"
    ordinary = [f for f in flows if not f.get("is_attack_traffic")]
    carried = [f for f in flows if _flow_is_visible(f)]
    normal_rate = sum(float(f.get("wire_bytes", 0)) * 8 / max(float(f.get("interval_seconds", 1)), 1e-9) / 1e6
                      for f in carried if not f.get("is_attack_traffic"))
    attack_rate = sum(float(f.get("wire_bytes", 0)) * 8 / max(float(f.get("interval_seconds", 1)), 1e-9) / 1e6
                      for f in carried if f.get("is_attack_traffic"))
    observed = [state, f"Обычный трафик: {normal_rate:,.1f} Мбит/с",
                f"Атакующий трафик: {attack_rate:,.1f} Мбит/с"]
    if "flows" not in snapshot.get("traffic", {}):
        observed[1:3] = ["Обычный трафик: нет подробных потоков",
                         "Атакующий трафик: нет подробных потоков"]
    events = attacks.get("events", [])
    if events:
        kinds = list(dict.fromkeys(str(e.get("kind", "")) for e in events))
        names = {"dos": "DoS (перегрузка)", "ddos": "DDoS (распределённая перегрузка)",
                 "syn_flood": "SYN-флуд (неполные соединения)", "brute_force": "попытки входа",
                 "power_attack": "нарушение питания"}
        observed.append("Воздействие: " + ", ".join(names.get(kind, kind) for kind in kinds[:2]))
    elif step == 0:
        observed.append("t0: исправный эталон с рабочей нагрузкой")
    observed.append(f"Связей с потоком: {len(loads)} / {model.graph.number_of_edges()}")

    forecast = ["ПРЕДУПРЕЖДЕНИЕ ПОДТВЕРЖДЕНО" if warning else "Подтверждённого предупреждения нет"]
    if warning:
        start = koopman.get("forecast_window_start_seconds")
        end = koopman.get("forecast_window_end_seconds")
        if start is not None and end is not None:
            forecast.append(f"Окно: t = {start:g}…{end:g} с")
            forecast.append(f"От текущего шага: {float(start) - now:+g}…{float(end) - now:+g} с")
        forecast.append("Область: " + (", ".join(map(str, koopman.get("forecast_target_ids", [])[:3])) or "не локализована"))
    else:
        horizon = koopman.get("maximum_forecast_horizon_seconds")
        forecast.append(f"Горизонт расчёта: до {horizon:g} с" if horizon is not None else "Горизонт: нет текущей оценки")
    score = koopman.get("forecast_risk_score")
    forecast.append(f"Риск: {100 * float(score):.1f}% · не вероятность" if score is not None else "Риск: нет текущей оценки")

    tiers = {t.get("sla_grade"): t for t in attacks.get("sla_restoration", {}).get("tiers", [])}
    quality = []
    for grade, label in (("gold", "Золотой"), ("silver", "Серебряный"), ("bronze", "Бронзовый")):
        tier = tiers.get(grade, {})
        total = int(tier.get("flow_count", 0))
        compliant = tier.get("sla_compliant_flow_count")
        quality.append(f"{label}: {100 * int(compliant) / total:.2f}% · {compliant}/{total} потоков"
                       if total and compliant is not None else f"{label}: нет данных")
    accounted = attacks.get("sla_accounted_flow_count", sum(int(t.get("flow_count", 0)) for t in tiers.values()))
    excluded = attacks.get("security_excluded_flow_count", sum(int(t.get("security_excluded_flow_count", 0)) for t in tiers.values()))
    quality.append(f"Учтено в SLA: {accounted} · исключено: {excluded}")
    limited = sum(int(t.get("slo_limited_coverage_flow_count", 0)) for t in tiers.values())
    quality.append(f"Неполная оценка требований: {limited} потоков" if limited else "Показаны сохранённые оценки требований")

    rerouted = routing.get("rerouted_flow_count", sum(bool(f.get("rerouted")) for f in ordinary))
    failover = routing.get("failover_flow_count", sum(bool(f.get("failover_active")) for f in ordinary))
    isolated = routing.get("isolated_flow_count", sum(bool(f.get("isolated")) for f in ordinary))
    quarantined = {f.get("client_node") for f in flows if f.get("quarantined") or f.get("routing_action") == "quarantine_attack_source"}
    blocked = int(attacks.get("blocked_attack_packet_count", 0))
    actions = [f"Смена маршрута: {rerouted} · резерв сервера: {failover}",
               f"Изолировано потоков: {isolated} · карантин: {len(quarantined)}",
               f"Отфильтровано атакующих пакетов: {blocked:,}"]
    remap = snapshot.get("arbitrator", {}).get("remap", {}).get("action")
    if remap:
        actions.append("Решение на следующий шаг: " + {
            "NO_REMAP": "сохранить маршруты", "OBSERVE_PRECURSOR": "наблюдать",
            "PLAN_REMAP": "подготовить маршруты",
            "ISOLATE_CONFIRMED_SOURCES_AND_REMAP": "изоляция и смена маршрутов",
        }.get(remap, ACTION_DISPLAY_NAMES.get(remap, remap)))

    training = snapshot.get("training", {})
    nominal = koopman.get("nominal_training", {})
    accepted = nominal.get("transition_accepted", training.get("operator_updated"))
    if step == 0:
        learning = ["t0 зафиксирован; перехода для обучения ещё нет"]
    elif accepted is True:
        learning = ["Обучение: принят исправный переход", "Обновлённый оператор — со следующего шага"]
    elif accepted is False:
        reason = "угроза или нарушение качества" if nominal.get("observation_eligible") is False else "нужны два исправных наблюдения подряд"
        learning = ["Обучение приостановлено", reason.capitalize()]
    else:
        learning = ["Обучение: нет текущей оценки"]
    kalman = koopman.get("kalman_filter", {})
    if "applied_to_nominal_observation" in kalman:
        learning.append("Калман: обычный шум сглажен" if kalman["applied_to_nominal_observation"] else "Калман: признаки угрозы сохранены")
    return [
        {"title": f"СЕЙЧАС · ШАГ {step} · {now:g} с", "lines": observed, "color": color},
        {"title": "ПРОГНОЗ КУПМАНА", "lines": forecast, "color": "#c4b5fd"},
        {"title": "КАЧЕСТВО УСЛУГ · SLA", "lines": quality, "color": "#bae6fd"},
        {"title": "ВЫПОЛНЕНО НА ЭТОМ ШАГЕ", "lines": actions, "color": "#86efac"},
        {"title": "ОБУЧЕНИЕ И ФИЛЬТР", "lines": learning, "color": "#cbd5e1"},
    ]


def _interactive_snapshot_summary(snapshot, flows, loads, paused: bool, model) -> str:
    """Текстовый вариант тех же карточек, без нового решения в интерфейсе."""
    sections = _interactive_status_sections(snapshot, flows, loads, paused, model)
    text = "\n\n".join(section["title"] + "\n" + "\n".join(section["lines"]) for section in sections)
    # Сохраняем точную подпись текстового экспорта для прежних потребителей.
    if sections[2]["lines"][0] == "Золотой: нет данных":
        text += "\nSLA золотого класса (качество услуг): нет данных"
    return text


def _node_hover_text(model, node_id: str, snapshot=None) -> str:
    if snapshot is not None:
        return _current_node_hover_text(model, node_id, snapshot)
    attrs = model.graph.nodes[node_id]
    role = attrs.get("role")
    if role in {"core-router", "aggregation-switch", "radio-access-node", "optical-line-terminal"}:
        return _equipment_hover_text(model, node_id, attrs)
    if role == "service-server":
        profile = attrs.get("server_profile", {})
        runtime = attrs.get("runtime", {})
        power = attrs.get("power_architecture", {})
        return (
            f"{node_id} · Физический сервер L0\n"
            f"IP: {attrs.get('ip_address')} · MAC: {attrs.get('mac_address')}\n"
            f"Модель: {profile.get('model')}\n"
            f"ЦП: {profile.get('cpu')} · {profile.get('total_cores')} физических ядер · "
            f"{profile.get('threads_total', 0)} потоков\n"
            f"ОЗУ: {profile.get('ram_gb')} ГБ · хранилище: {profile.get('storage_tb')} ТБ\n"
            f"Выбранные сетевые порты: {profile.get('network_ports_gbps')} Гбит/с\n"
            f"Питание: {power.get('feed_count', 0)} ввода · "
            f"{_power_redundancy_label(power.get('psu_redundancy'))}\n"
            f"БП: {power.get('psu_count', 0)} × {power.get('psu_rating_w_each', 0)} Вт · "
            f"{power.get('psu_efficiency_class', '—')} [Dell; номинал БП, не потребление]\n"
            f"T0: {float(power.get('modeled_t0_power_w') or 0.0):.0f} Вт [сценарий] · "
            f"ИБП {float(power.get('ups_backup_autonomy_hours') or 0.0):.2f} ч [сценарий]\n"
            f"Локальный домен отказа: {power.get('local_fault_domain_id', '—')}\n"
            f"Сервисы: {', '.join(attrs.get('hosted_services', []))}\n"
            f"Резервные реплики: {', '.join(attrs.get('standby_services', [])) or 'нет'}\n"
            f"Базовая нагрузка T0: ЦП {runtime.get('cpu_util_percent', 0):.1f}%, "
            f"ОЗУ {runtime.get('ram_util_percent', 0):.1f}%, сеть {runtime.get('network_util_percent', 0):.1f}%\n"
            f"Диски {runtime.get('storage_util_percent', 0):.1f}%, "
            f"температура {runtime.get('temperature_c', 0):.1f} °C, сеансов {runtime.get('active_sessions', 0)}"
            f"{_protection_hover(attrs)}"
        )
    if attrs.get("level") == "L1":
        kind = "Мобильный абонент" if role == "mobile-subscriber" else "Фиксированный абонент"
        return (
            f"{node_id} · {kind}\n"
            f"IP: {attrs.get('ip_address')} · MAC: {attrs.get('mac_address')}\n"
            f"Логическая точка агрегации: {attrs.get('home_access')}\n"
            "Линия доступа: одна; резервной линии абонента нет\n"
            f"Профиль доступа: {attrs.get('access_profile', '—')}\n"
            f"Доступ: {attrs.get('access_capacity_mbps', 0):.0f} Мбит/с · "
            f"базовая задержка {attrs.get('access_latency_ms', 0):.1f} мс\n"
            f"Класс SLA: {SLA_LABELS.get(attrs.get('sla_grade'), attrs.get('sla_grade'))}\n"
            f"Приложение: {TRAFFIC_LABELS.get(attrs.get('traffic_kind'), attrs.get('traffic_kind'))}\n"
            f"Кодек/профиль: {attrs.get('codec_profile_name') or attrs.get('codec')}\n"
            f"Целевая скорость: {attrs.get('target_bitrate_kbps', 0):.0f} кбит/с"
            f"{_protection_hover(attrs)}"
        )
    if attrs.get("level") == "L0":
        service_name = SERVICE_DISPLAY_NAMES.get(attrs.get("label"), attrs.get("label"))
        service_profile = attrs.get("service_profile", {})
        primary_metrics = ", ".join(
            SLO_METRIC_LABELS.get(metric, metric)
            for metric in service_profile.get("primary_slo_metrics", ())
        ) or "—"
        limitations = ", ".join(
            MODEL_LIMITATION_LABELS.get(limitation, limitation)
            for limitation in service_profile.get("modeling_limitations", ())
        ) or "нет"
        return (
            f"{node_id} · Логический сервис L0\nНазначение: {service_name}\n"
            f"Виртуальный IP (VIP): {attrs.get('ip_address')} · MAC: {attrs.get('mac_address')}\n"
            f"Платформа: {attrs.get('platform')}\nОсновной сервер: {attrs.get('hosted_on')}\n"
            f"Резервный сервер: {', '.join(attrs.get('standby_hosts', [])) or 'нет'}\n"
            f"Профиль: {attrs.get('codec_profile_name')}\n"
            f"Аудиокодек: {attrs.get('audio_codec') or '—'} · Видеокодек: {attrs.get('video_codec') or '—'}\n"
            f"Основные показатели качества: {primary_metrics}\n"
            f"Ограничения модели: {limitations}"
        )
    if role == "arbitrator":
        return "АРБ · Арбитратор L7\nНаблюдает метрики всех уровней\nПринимает решение о переназначении"
    if attrs.get("level") == "L8":
        x, y = attrs.get("pos", ("—", "—"))
        return (
            f"{node_id} · Опорная точка местности L8\n"
            f"Координаты модели: x={x}, y={y}\n"
            "Используется как географическая основа размещения;\n"
            "не является транзитным сетевым устройством"
        )
    return f"{node_id}\nРоль: {role}\nУровень: {attrs.get('level')}"


def _current_node_hover_text(model, node_id: str, snapshot) -> str:
    """Текущие значения из снимка; конфигурация графа не выдаётся за измерение."""
    attrs = model.graph.nodes[node_id]
    state = snapshot.get("traffic", {}).get("network_state", {})
    row = next((n for n in state.get("nodes", []) if n.get("id") == node_id), None)
    metrics = row.get("metrics", {}) if row else {}
    attack_rate = row.get("attack_arrival_rate_pps") if row else None
    if not row and node_id in state.get("node_ids", []):
        index = state["node_ids"].index(node_id)
        vectors = state.get("vertex_vectors", [])
        metrics = dict(zip(state.get("vertex_metric_names", []), vectors[index])) if index < len(vectors) else {}
        rates = state.get("node_attack_arrival_rates_pps", [])
        attack_rate = rates[index] if index < len(rates) else None
    role = {
        "core-router": "маршрутизатор", "aggregation-switch": "коммутатор",
        "radio-access-node": "радиодоступ", "optical-line-terminal": "оптический доступ",
        "service-server": "сервер", "service": "услуга", "arbitrator": "арбитратор",
        "mobile-subscriber": "мобильный абонент", "fixed-subscriber": "проводной абонент",
    }.get(attrs.get("role"), attrs.get("level", "объект модели"))
    lines = [f"{node_id} · {role}",
             f"ШАГ {snapshot.get('step_index', '—')} · t = {snapshot.get('time_seconds', '—')} с",
             f"IP: {attrs.get('ip_address', '—')}"]
    if not metrics:
        lines.extend(["Текущие показатели отсутствуют в снимке.",
                      "Доступна только конфигурация эталона t0."])
        return "\n".join(lines)
    def value(name, digits=2, factor=1.0):
        raw = metrics.get(name)
        return f"{float(raw) * factor:,.{digits}f}" if raw is not None else "—"
    lines.extend([
        "ТЕКУЩИЕ ПОКАЗАТЕЛИ · расчёт модели",
        f"Обычное поступление: {value('normal_arrival_rate_pps', 1)} пак./с",
        f"Атакующий поток: {float(attack_rate):,.1f} пак./с" if attack_rate is not None else "Атакующий поток: нет отдельной оценки",
        f"Уничтожение (эквивалент): {value('negative_arrival_rate_pps', 1)} пак./с",
        f"Обслуживание: {value('service_rate_pps', 1)} пак./с · занятость: {value('utilization_ratio', 1, 100)}%",
        f"Учётные потери: {value('destroyed_packet_fraction', 2, 100)}% · очередь: {value('mean_queue_packets')} пак.",
        f"Задержка очереди: {value('mean_delay_ms')} мс",
        f"Дисперсия задержки: {value('delay_variance_ms2')} мс²",
        f"ЦП: {value('cpu_load_percent', 1)}% · буфер: {value('buffer_memory_mib', 4)} МиБ [оценки]",
    ])
    profiles = {code: name for name, code in state.get("attack_profiles", {}).items()}
    profile = profiles.get(metrics.get("attack_profile"))
    if profile:
        lines.append("Профиль: " + ATTACK_KIND_DISPLAY_NAMES.get(profile, {"normal": "обычный", "mixed": "смешанный"}.get(profile, profile)))
    own_flows = [f for f in snapshot.get("traffic", {}).get("flows", []) if f.get("client_node") == node_id and not f.get("is_attack_traffic")]
    if own_flows:
        flow = own_flows[0]
        if flow.get("security_excluded_from_sla_accounting"):
            lines.append("Карантин: поток исключён из клиентского SLA")
        elif flow.get("isolated"):
            lines.append("Поток изолирован; передача приостановлена")
        elif not flow.get("route"):
            lines.append("Маршрут отсутствует; передача приостановлена")
        else:
            lines.append("Текущий маршрут: " + " → ".join(map(str, flow["route"])))
        lines.append(f"Потери потока: {flow.get('observed_dropped_packets', 0)} / {flow.get('packet_count', 0)} пакетов")
    lines.append("Потери приписаны учётной точке; место сброса неизвестно.")
    return "\n".join(lines)


def _equipment_hover_text(model, node_id: str, attrs: dict[str, Any]) -> str:
    role = attrs.get("role")
    device_name = {
        "core-router": "Маршрутизатор ядра",
        "aggregation-switch": "Агрегирующий коммутатор",
        "radio-access-node": "узел радиодоступа",
        "optical-line-terminal": "узел оптического доступа",
    }.get(role, "сетевое оборудование")
    raw = attrs.get("l2_raw_baseline", {})
    profile = attrs.get("l2_profile", {})
    neighbors = list(model.graph.neighbors(node_id))
    clients = [
        node for node in neighbors
        if model.graph.nodes[node].get("level") == "L1"
        and model.graph.nodes[node].get("home_access") == node_id
    ]
    if role == "aggregation-switch":
        access_neighbors = [
            node for node in neighbors
            if model.graph.nodes[node].get("role") in {"radio-access-node", "optical-line-terminal"}
        ]
        clients = [
            node for node, client_attrs in model.graph.nodes(data=True)
            if client_attrs.get("level") == "L1"
            and client_attrs.get("home_access") in access_neighbors
        ]
    uplinks = [node for node in neighbors if model.graph.nodes[node].get("level") in {"L0", "L2"}]
    verified = set(profile.get("verified_fields", ()))
    dram_origin = "Cisco" if "dram_gb" in verified else "модель"
    forwarding_origin = "Cisco" if "forwarding_mpps" in verified else "модель"
    throughput_origin = "Cisco" if "throughput_gbps" in verified else "модель"
    port_origin = "Cisco" if str(profile.get("vendor")) == "Cisco" else "модель"
    power = attrs.get("power_architecture", {})
    if power.get("vendor_typical_output_power_w") is not None:
        vendor_power_text = (
            f"{float(power['vendor_typical_output_power_w']):.0f} Вт типично / "
            f"{float(power.get('vendor_max_output_power_w') or 0.0):.0f} Вт максимум "
            "выходной мощности [Cisco]"
        )
    else:
        vendor_power_text = (
            "типичное потребление не опубликовано; "
            f"{float(power.get('vendor_thermal_output_btu_per_hour') or 0.0):.0f} BTU/ч "
            f"≈ {float(power.get('vendor_thermal_output_equivalent_w') or 0.0):.0f} Вт "
            "теплового верхнего предела [Cisco]"
        )
    if (
        power.get("vendor_typical_output_power_w") is None
        and power.get("vendor_thermal_output_equivalent_w") is None
    ):
        vendor_power_text = "GNet9 scenario power envelope [not vendor spec]"
    return (
        f"{node_id} · {device_name}\n"
        f"IP: {attrs.get('ip_address')} · MAC: {attrs.get('mac_address')}\n"
        f"Платформа: {attrs.get('platform_family')}\n"
        f"Профиль: {attrs.get('platform_profile')}\n"
        f"Базовая загрузка ЦП: {raw.get('cpu_util', 0):.1f}% [модель]\n"
        f"Базовая загрузка ОЗУ: {raw.get('ram_util', 0):.1f} / {profile.get('dram_gb', 0):.1f} ГБ [{dram_origin}]\n"
        f"Системная пропускная способность: {profile.get('throughput_gbps', 0):,.0f} Гбит/с [{throughput_origin}]\n"
        f"Скорость пересылки: {profile.get('forwarding_mpps', 0):,.0f} млн пакетов/с [{forwarding_origin}]\n"
        f"Порты: {profile.get('port_configuration', '—')} [{port_origin}]\n"
        f"Логических соседей графа: {len(neighbors)}"
        + (f" · клиентов: {len(clients)}" if clients else "")
        + f"\nТранзитная степень: {attrs.get('infrastructure_degree', '—')} (предел ≤ 6)"
        + f"\nФизических портов ≥40G занято: {attrs.get('physical_high_speed_ports_used', '—')}"
        + f"\nL1-привязки абстрагируют RAN/PON и не расходуют по одному порту C9500"
        + f"\nПитание: {power.get('feed_count', 0)} ввода · "
        + _power_redundancy_label(power.get("psu_redundancy"))
        + f"\nМощность: {vendor_power_text}"
        + f"\nT0: {float(power.get('modeled_t0_power_w') or 0.0):.0f} Вт · "
        + f"ИБП {float(power.get('ups_backup_autonomy_hours') or 0.0):.1f} ч [сценарий]"
        + f"\nСвязи: {', '.join(uplinks)}"
        + _protection_hover(attrs)
    )


def _protection_hover(attrs: dict[str, Any]) -> str:
    protection = attrs.get("critical_protection", {})
    if not protection.get("is_critical"):
        return ""
    return (
        "\nКРИТИЧЕСКИЙ УЗЕЛ ЗОЛОТОГО КЛАССА · приоритет защиты 1"
        f"\nПотоков золотого класса через узел: {protection.get('gold_transit_flow_count', 0)}"
        f" · резервных: {protection.get('gold_recovery_reserve_flow_count', 0)}"
        f" · КВУ: {protection.get('critical_involvement_coefficient', 0.0):.3f}"
        + (
            f"\nИерархия КВУ: {protection.get('kvu_tier')} · место {protection.get('kvu_rank')}"
            f" · абонентов золотого класса в пути: {protection.get('gold_subscriber_count', 0)}"
            f" (на доступе: {protection.get('direct_gold_subscriber_count', 0)})"
            if protection.get("kvu_rank") is not None
            else ""
        )
        + f"\nПредел загрузки при переназначении: {protection.get('maximum_safe_utilization_percent', 80.0):.0f}%"
    )


def _power_redundancy_label(value: Any) -> str:
    return {
        "1+1 hot-swappable": "1+1, горячая замена",
        "1+1 hot-plug": "1+1, горячее подключение",
    }.get(str(value), str(value or "—"))
