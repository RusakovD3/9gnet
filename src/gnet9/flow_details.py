"""Подписи и карточки сети; чтение наблюдений без рисования и новых решений."""

from collections import Counter
from typing import Any

import numpy as np

from .constants import ACTION_DISPLAY_NAMES, SERVICE_DISPLAY_NAMES

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
        if flow.get("route") and not flow.get("isolated") and int(flow.get("packet_count", 0)) > 0
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


def _interactive_snapshot_summary(snapshot, flows, loads, paused: bool, model) -> str:
    """Краткая карточка для окна: она всегда помещается в выделенную панель."""
    legitimate = [flow for flow in flows if not flow.get("is_attack_traffic")]
    active_attacks = snapshot.get("attacks", {})
    koopman = snapshot.get("koopman", {})
    routing = active_attacks.get("routing", {})
    latencies = [
        float(flow.get("one_way_latency_ms", 0.0))
        for flow in legitimate
        if not flow.get("isolated")
    ]
    mode = "ПАУЗА" if paused else "ВОСПРОИЗВЕДЕНИЕ"
    step = int(snapshot.get("step_index", 0))
    time_seconds = float(snapshot.get("time_seconds", 0.0))
    remap = snapshot.get("arbitrator", {}).get("remap", {}).get("action", "—")
    attack_names = ", ".join(
        str(event.get("name_ru") or event.get("attack_id") or "атака")
        for event in active_attacks.get("events", [])[:2]
    )
    if len(active_attacks.get("events", [])) > 2:
        attack_names += f" и ещё {len(active_attacks['events']) - 2}"
    warning = bool(koopman.get("forecast_is_early_warning"))
    targets = ", ".join(map(str, koopman.get("forecast_target_ids", [])[:3])) or "—"
    if active_attacks.get("active"):
        status = "НАБЛЮДАЕМАЯ АТАКА"
    elif warning:
        status = "ПРЕДУПРЕЖДЕНИЕ ПОДТВЕРЖДЕНО"
    elif active_attacks.get("precursor_active"):
        status = "НАБЛЮДАЮТСЯ ПРИЗНАКИ УГРОЗЫ"
    elif koopman.get("risk_state", "nominal") != "nominal":
        status = "ПРОДОЛЖАЕТСЯ НАБЛЮДЕНИЕ"
    else:
        status = "ИСПРАВНАЯ СЕТЬ"
    tiers = active_attacks.get("sla_restoration", {}).get("tiers", [])
    gold = next((t for t in tiers if t["sla_grade"] == "gold"), {})
    count = gold.get("flow_count", 0)
    gold_sla = f"{100 * gold['sla_compliant_flow_count'] / count:.2f}%" if count else "нет данных"
    max_link = max(
        (item.get("wire_bytes", 0) for item in loads.values()),
        default=0,
    ) / 1_000_000
    lines = [
        f"ШАГ {step} · t = {time_seconds:.0f} с · {mode}",
        "─" * 30,
        f"Обычных потоков: {len(legitimate)}",
        f"Связей с трафиком: {len(loads)} из {model.graph.number_of_edges()}",
        f"Средняя / макс. задержка: {np.mean(latencies) if latencies else 0.0:.2f} / {max(latencies, default=0.0):.2f} мс",
        f"Макс. объём на связи: {max_link:.1f} МБ за шаг",
        "",
        f"КУПМАН: {status}",
        f"Риск: {float(koopman.get('forecast_risk_score', 0.0)) * 100:.1f}% (не вероятность)",
        f"Цель: {targets}",
        f"Окно: {koopman.get('forecast_window_start_seconds', '—')}…{koopman.get('forecast_window_end_seconds', '—')} с",
        f"Запас: ≥ {koopman.get('prediction_slo_seconds', '—')} с",
        "",
        f"Защита: {ACTION_DISPLAY_NAMES.get(remap, remap)}",
        f"Переназначено / резерв сервера / изолировано: {routing.get('rerouted_flow_count', 0)} / {routing.get('failover_flow_count', 0)} / {routing.get('isolated_flow_count', 0)}",
        f"SLA золотого класса (качество услуг): {gold_sla}",
    ]
    if active_attacks.get("active"):
        lines.extend([
            "",
            f"АТАКА: {attack_names}",
            f"После защиты: {float(active_attacks.get('attack_rate_mbps', 0.0)):,.1f} Мбит/с",
            f"Этап: {routing.get('protection_stage', 'подтверждение/защита')}",
        ])
    elif step == 0:
        lines.append("t0: исправный эталон с рабочей нагрузкой")
    return "\n".join(lines)


def _node_hover_text(model, node_id: str) -> str:
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
