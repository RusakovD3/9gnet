"""Очередь абонента и восстановление передачи после потерь."""

from __future__ import annotations

from typing import Any

from .gelenbe import gelenbe_queue
from .l1_d0sl import _mm1k_stationary_metrics
from .models import NetworkModel


def apply_transport_queues(model: NetworkModel, flows: list[dict[str, Any]], attack_flows: list[dict[str, Any]] | None = None) -> None:
    """Оценить ограниченную очередь по уже наблюдаемым потерям.

    Скорость обслуживания задаётся резервом SLA, потери оценивают уничтожение. Это
    приближение по данным потока, а не измеренная очередь физического порта.
    TCP тратит время на повторную передачу; UDP её не выполняет.
    """
    attack_rates: dict[str, float] = {}
    affected_counts: dict[str, int] = {}
    for attack in attack_flows or []:
        if not attack.get("quarantined"):
            key = str(attack["attack_id"])
            attack_rates[key] = attack_rates.get(key, 0.0) + int(attack["packet_count"]) / max(float(attack["interval_seconds"]), 1e-9)
    for flow in flows:
        client = flow.get("client_node")
        if client not in model.graph or flow.get("isolated") or not flow.get("route"):
            continue
        if not model.graph.nodes[client].get("kendall_queue"):
            continue
        for key in flow.get("attack_ids", []):
            affected_counts[key] = affected_counts.get(key, 0) + 1
    for flow in flows:
        client = flow.get("client_node")
        if client not in model.graph or flow.get("isolated") or not flow.get("route"):
            continue
        queue = model.graph.nodes[client].get("kendall_queue", {})
        if not queue:
            continue
        packet_count = int(flow["packet_count"])
        interval = max(float(flow["interval_seconds"]), 1e-9)
        lost = min(packet_count, max(0, int(flow.get("observed_dropped_packets", 0))))
        loss = lost / max(packet_count, 1)
        arrival = packet_count / interval
        nominal_rho = float(queue["utilization_rho"])
        expected_arrival = float(flow.get("normal_service_reference_packet_rate_pps", flow.get("normal_expected_packet_rate_pps", arrival)))
        nominal_service = max(float(queue["service_rate_pps"]), expected_arrival / max(nominal_rho, 1e-9))
        service = nominal_service
        attack_arrival = sum(attack_rates.get(key, 0.0) / max(affected_counts.get(key, 1), 1) for key in flow.get("attack_ids", []))
        # Наблюдаемая доля потерь оценивает долю уничтожающих завершений.
        # Она не создаёт новый счётчик потерь и не обозначает поток атак.
        negative = service * loss / max(1.0 - loss, 1e-3)
        capacity = int(queue["capacity_packets"])
        estimate = gelenbe_queue(arrival + attack_arrival, service, negative, capacity)
        system_ms = estimate["mean_system_time_ms"]
        _, _, _, _, nominal_ms = _mm1k_stationary_metrics(arrival, nominal_service, capacity)
        # Ущерб уже учтён событием. Очередь добавляет лишь задержку, не
        # отбрасывает те же пакеты повторно и не подменяет наблюдаемые потери.
        queue_delay_increment = max(0.0, system_ms - nominal_ms)
        attack_delay = max(0.0, float(flow.get("attack_latency_penalty_ms", 0.0)))
        extra_delay = max(0.0, queue_delay_increment - attack_delay)
        flow["one_way_latency_ms"] = round(float(flow["one_way_latency_ms"]) + extra_delay, 4)
        if flow.get("rtt_ms") is not None:
            flow["rtt_ms"] = round(float(flow["rtt_ms"]) + 2.0 * extra_delay, 4)
        recovery_seconds = 0.0
        if flow.get("transport") == "TCP" and lost:
            # Ожидаемое число повторных попыток на успешно переданный пакет.
            # Последовательные потери снижают полезную скорость, даже если
            # объём предложенных данных на этом шаге остался прежним.
            recovery_seconds = (float(flow.get("rtt_ms") or 0.0) / 1000.0) * loss / max(1.0 - loss, 1e-3)
        # Уже внесённая задержка атаки покрывает очередь и восстановление
        # один раз. Повторное вычитание всей задержки теряло часть ожидания.
        remaining_attack_delay = max(0.0, attack_delay - queue_delay_increment)
        retry_delay_ms = max(0.0, recovery_seconds * 1000.0 - remaining_attack_delay)
        flow["one_way_latency_ms"] = round(float(flow["one_way_latency_ms"]) + retry_delay_ms, 4)
        delivered_payload = float(flow["payload_bytes"]) * (1.0 - loss)
        goodput = delivered_payload * 8.0 / (interval + recovery_seconds) / 1_000_000.0
        flow["transport_queue"] = {
            **estimate,
            "model": "finite_gelenbe_observation_based_cancellation",
            "capacity_packets": capacity,
            "attack_arrival_rate_pps": attack_arrival,
            "normal_arrival_rate_pps": arrival,
            "observed_dropped_packets": lost,
            "negative_rate_origin": "cancellation_share_inferred_from_observed_flow_loss",
            "mean_system_time_ms": system_ms, "additional_delay_ms": extra_delay,
            "queue_delay_increment_ms": queue_delay_increment,
            "tcp_recovery_seconds": recovery_seconds,
            "tcp_additional_delivery_delay_ms": retry_delay_ms,
            "retransmission_policy": "retry_after_loss" if flow.get("transport") == "TCP" else "no_transport_retry",
            "goodput_mbps": goodput,
            "service_rate_origin": "fixed_sla_reserve_with_protocol_overhead",
        }
        flow["goodput_mbps"] = round(goodput, 6)
