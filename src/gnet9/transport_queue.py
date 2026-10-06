"""Очередь абонента и восстановление передачи после потерь."""

from __future__ import annotations

from typing import Any

from .l1_d0sl import _mm1k_stationary_metrics
from .models import NetworkModel


def apply_transport_queues(model: NetworkModel, flows: list[dict[str, Any]]) -> None:
    """Оценить ограниченную очередь по уже наблюдаемым потерям.

    Оценка обслуживания уменьшается на долю недоступного ресурса. Это
    приближение по данным потока, а не измеренная очередь физического порта.
    TCP тратит время на повторную передачу; UDP её не выполняет.
    """
    for flow in flows:
        client = flow.get("client_node")
        if client not in model.graph or flow.get("isolated") or not flow.get("route"):
            continue
        queue = model.graph.nodes[client].get("kendall_queue", {})
        if not queue:
            continue
        packet_count = int(flow["packet_count"])
        interval = max(float(flow["interval_seconds"]), 1e-9)
        lost = min(packet_count, int(flow.get("observed_dropped_packets", 0)))
        loss = lost / max(packet_count, 1)
        arrival = packet_count / interval
        nominal_rho = float(queue["utilization_rho"])
        nominal_service = arrival / max(nominal_rho, 1e-9)
        service = nominal_service * max(1e-3, 1.0 - loss)
        capacity = int(queue["capacity_packets"])
        blocking, admitted, mean_system, mean_waiting, system_ms = _mm1k_stationary_metrics(arrival, service, capacity)
        _, _, _, _, nominal_ms = _mm1k_stationary_metrics(arrival, nominal_service, capacity)
        # Ущерб уже учтён событием. Очередь добавляет лишь задержку, не
        # отбрасывает те же пакеты повторно и не подменяет наблюдаемые потери.
        extra_delay = max(0.0, system_ms - nominal_ms)
        flow["one_way_latency_ms"] = round(float(flow["one_way_latency_ms"]) + extra_delay, 4)
        if flow.get("rtt_ms") is not None:
            flow["rtt_ms"] = round(float(flow["rtt_ms"]) + 2.0 * extra_delay, 4)
        recovery_seconds = 0.0
        if flow.get("transport") == "TCP" and lost:
            # Ожидаемое число повторных попыток на успешно переданный пакет.
            # Последовательные потери снижают полезную скорость, даже если
            # объём предложенных данных на этом шаге остался прежним.
            recovery_seconds = (float(flow.get("rtt_ms") or 0.0) / 1000.0) * loss / max(1.0 - loss, 1e-3)
        delivered_payload = float(flow["payload_bytes"]) * (1.0 - loss)
        goodput = delivered_payload * 8.0 / (interval + recovery_seconds) / 1_000_000.0
        flow["transport_queue"] = {
            "model": "finite_mm1k_observation_based_service",
            "arrival_rate_pps": arrival, "service_rate_pps": service,
            "utilization_rho": arrival / max(service, 1e-9),
            "capacity_packets": capacity, "blocking_probability": blocking,
            "effective_arrival_rate_pps": admitted,
            "mean_system_packets": mean_system, "mean_queue_packets": mean_waiting,
            "mean_system_time_ms": system_ms, "additional_delay_ms": extra_delay,
            "tcp_recovery_seconds": recovery_seconds,
            "retransmission_policy": "retry_after_loss" if flow.get("transport") == "TCP" else "no_transport_retry",
            "goodput_mbps": goodput,
            "service_rate_origin": "nominal_queue_reserve_scaled_by_observed_delivery",
        }
        flow["goodput_mbps"] = round(goodput, 6)
