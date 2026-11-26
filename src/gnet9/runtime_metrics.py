"""Именованные показатели графа текущего шага; исходный граф не изменяется."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .gelenbe import gelenbe_queue
from .routing import LOGICAL_SERVICE_MEDIUM


VERTEX_METRICS = (
    "normal_arrival_rate_pps", "negative_arrival_rate_pps", "service_rate_pps",
    "utilization_ratio", "destroyed_packet_fraction", "mean_queue_packets",
    "mean_delay_ms", "delay_variance_ms2", "attack_profile", "cpu_load_percent",
    "buffer_memory_mib",
)
VERTEX_UNITS = ("packets/s", "packets/s", "packets/s", "ratio", "ratio",
                "packets", "ms", "ms2", "profile_code", "percent", "MiB")
EDGE_METRICS = (
    "capacity_mbps", "propagation_delay_ms", "jitter_ms", "loss_probability",
    "reliability_ratio", "utilization_ratio", "speed_mbps", "syn_ack_fraction",
    "file_drop_fraction",
)
EDGE_UNITS = ("Mbit/s", "ms", "ms", "ratio", "ratio", "ratio", "Mbit/s", "ratio", "ratio")
OFFERED_LINK_RATE_ORIGIN = "offered_wire_load_on_route_before_end_to_end_losses"
LINK_LOAD_RATIO_ORIGIN = "offered_wire_rate_divided_by_link_capacity"
ATTACK_PROFILES = {"normal": 0, "dos": 1, "ddos": 2, "syn_flood": 3,
                   "brute_force": 4, "power_attack": 5, "mixed": 6,
                   "dns_flood": 7, "application_flood": 8,
                   "bgp_hijack": 9, "privilege_escalation": 10, "network_discovery": 11,
                   "user_execution": 12, "exfiltration": 13, "mobile_network_dos": 14, "dhcp_flood": 15}


def _tensor_value(attrs: dict[str, Any], name: str, default: float = 0.0) -> float:
    tensor = attrs.get("tensor")
    if tensor is not None and name in tensor.metric_names:
        return float(tensor.data[tensor.metric_names.index(name)])
    return float(attrs.get(name, default))


def _packet_rate(flow):
    return int(flow["packet_count"]) / max(float(flow["interval_seconds"]), 1e-9)


def _loss_accounting_endpoint(flow, route):
    """Одна точка учёта сквозных потерь, а не установленное место сброса.

    Обычный поток учитывается у абонента, атака — в конце своего маршрута.
    Поток из 100 пакетов с 10 потерями на трёх линиях даёт 10, не 30
    учётных потерь; экспозиция маршрута сохраняется отдельным полем.
    """
    client = flow.get("client_node")
    if not flow.get("is_attack_traffic") and client in route:
        return client
    return route[-1] if route else None


def _loss_accounting_edge(flow, route):
    endpoint = _loss_accounting_endpoint(flow, route)
    if len(route) < 2:
        return None
    # DNS хранит маршрут от клиента; медиапотоки — к клиенту.
    pair = route[:2] if endpoint == route[0] else route[-2:]
    return tuple(sorted(pair))


def network_runtime_metrics(model, flows: list[dict[str, Any]], events=(), *, detail="summary") -> dict[str, Any]:
    """Суммировать поступления на маршрутах; потери линии не локализованы.

    Сквозные потери принадлежат одной учётной точке у абонента, у атак —
    получателю. Физическое место потери не наблюдается. Нагрузка, память
    и очереди — оценки модели, а не данные оборудования.
    speed_mbps хранит предложенную нагрузку линии в Мбит/с до учёта потерь.
    При нагрузке 15 Мбит/с и ёмкости 10 Мбит/с utilization_ratio равен 1,5.
    file_drop_fraction относится к пакетам передачи файлов, не числу файлов.
    """
    node_flows, edge_flows = defaultdict(list), defaultdict(list)
    node_losses, edge_losses = defaultdict(list), defaultdict(list)
    for flow in flows:
        if flow.get("isolated") or flow.get("quarantined"):
            continue
        route = list(flow.get("route", []))
        endpoint = _loss_accounting_endpoint(flow, route)
        if endpoint is not None:
            node_losses[endpoint].append(flow)
        edge = _loss_accounting_edge(flow, route)
        if edge is not None:
            edge_losses[edge].append(flow)
        for node in set(route + ([flow["service_node"]] if flow.get("service_node") else [])):
            node_flows[node].append(flow)
        for source, target in zip(route, route[1:]):
            edge_flows[tuple(sorted((source, target)))].append(flow)
    active_by_node = defaultdict(set)
    endpoint_cpu_rise = defaultdict(float)
    for event in events:
        target = event["target_id"]
        active_by_node[target].add(event["kind"])
        endpoint_cpu_rise[target] = max(endpoint_cpu_rise[target],
                                        float(event.get("sensor_signals", {}).get("endpoint_cpu_rise_ratio", 0.0)))
    nodes = []
    for node, attrs in sorted(model.graph.nodes(data=True)):
        carried = node_flows[node]
        normal = [f for f in carried if not f.get("is_attack_traffic")]
        attacking = [f for f in carried if f.get("is_attack_traffic")]
        arrival = sum(_packet_rate(flow) for flow in normal)
        attack_rate = sum(_packet_rate(flow) for flow in attacking)
        # Абонент имеет заданный резерв обслуживания; у общего устройства
        # используется пропускная способность и средний размер пакета.
        byte_count = sum(int(f["wire_bytes"]) for f in carried)
        packet_count = sum(int(f["packet_count"]) for f in carried)
        packet_bytes = byte_count / max(packet_count, 1) if packet_count else 1200.0
        declared = attrs.get("kendall_queue", {})
        if declared:
            service = sum(float(f.get("transport_queue", {}).get("service_rate_pps", 0.0)) for f in normal) or float(declared["service_rate_pps"])
        else:
            # Привязка услуги к хосту задаёт условную полосу виртуальной услуги.
            # Аппаратная пропускная способность сервера складывается из его портов.
            capacity = sum(
                float(edge.get("capacity_mbps", 0.0)) for edge in model.graph[node].values()
                if attrs.get("role") == "service" or edge.get("medium") != LOGICAL_SERVICE_MEDIUM
            )
            service = max(1.0, capacity * 1_000_000.0 / (8.0 * max(packet_bytes, 1.0)))
        # Эквивалент уничтожения не переносится вдоль всего маршрута.
        # Атакующие пакеты выше входят только в положительное поступление.
        accounted_normal = [f for f in node_losses[node] if not f.get("is_attack_traffic")]
        negative = sum(float(f.get("transport_queue", {}).get("negative_arrival_rate_pps", 0.0)) for f in accounted_normal)
        queue = gelenbe_queue(arrival + attack_rate, service, negative, int(declared.get("capacity_packets", 128)))
        kinds = active_by_node[node] | {str(f["traffic_kind"]) for f in attacking}
        profile = next(iter(kinds)) if len(kinds) == 1 else "mixed" if kinds else "normal"
        observed_drops = sum(int(f.get("observed_dropped_packets", 0)) for f in accounted_normal)
        normal_packets = sum(int(f["packet_count"]) for f in normal)
        baseline_cpu = _tensor_value(attrs, "cpu_load_percent", float(attrs.get("runtime", {}).get("cpu_util_percent", 0.0)))
        cpu = max(baseline_cpu, 100.0 * queue["busy_probability"]) if attrs.get("level") == "L2" or attrs.get("role") == "service-server" else baseline_cpu
        cpu = min(100.0, cpu + 100 * endpoint_cpu_rise[node])
        values = (arrival, negative, service, min(1.0, queue["busy_probability"]),
                  observed_drops / max(normal_packets, 1), queue["mean_queue_packets"],
                  queue["mean_system_time_ms"], queue["delay_variance_ms2"],
                  ATTACK_PROFILES[profile], cpu,
                  queue["mean_queue_packets"] * packet_bytes / (1024 ** 2))
        nodes.append({"id": node, "metrics": dict(zip(VERTEX_METRICS, values)),
                      "attack_profile_name": profile, "attack_arrival_rate_pps": attack_rate,
                      "gelenbe_queue": queue,
                      "observed_dropped_packets": observed_drops,
                      "exposed_end_to_end_flow_drops": sum(int(f.get("observed_dropped_packets", 0)) for f in normal),
                      "loss_origin": "end_to_end_accounting_at_endpoint_not_physical_loss_location",
                      "queue_capacity_packets": int(declared.get("capacity_packets", 128))})
    edges = []
    for source, target, attrs in sorted(model.graph.edges(data=True)):
        routed_flows = edge_flows[tuple(sorted((source, target)))]
        accounted = edge_losses[tuple(sorted((source, target)))]
        packets = sum(int(f["packet_count"]) for f in routed_flows)
        drops = sum(int(f.get("observed_dropped_packets", 0)) for f in accounted)
        offered_rate_mbps = sum(
            int(f["wire_bytes"]) * 8.0 / max(float(f["interval_seconds"]), 1e-9) / 1e6 for f in routed_flows
        )
        capacity = float(attrs.get("capacity_mbps", 0.0))
        files = [f for f in routed_flows if f.get("traffic_kind") == "ftp"]
        file_packets = sum(int(f["packet_count"]) for f in files)
        file_drops = sum(int(f.get("observed_dropped_packets", 0)) for f in accounted if f.get("traffic_kind") == "ftp")
        # В штатном рукопожатии один SYN-ACK из трёх пакетов.
        syn_ack = sum(int(f.get("handshake_packets", 0)) // 3 for f in routed_flows)
        loss = max(_tensor_value(attrs, "loss_probability"), drops / max(packets, 1))
        reliability = float(attrs.get("reliability_ratio", 1.0 - loss))
        values = (capacity, float(attrs.get("latency_ms", 0.0)), float(attrs.get("jitter_ms", 0.0)),
                  loss,
                  max(0.0, min(1.0, reliability)), offered_rate_mbps / max(capacity, 1e-9), offered_rate_mbps,
                  syn_ack / max(packets, 1), file_drops / max(file_packets, 1))
        edges.append({"source": source, "target": target, "metrics": dict(zip(EDGE_METRICS, values)),
                      "packet_count": packets, "observed_flow_drops": drops,
                      "exposed_end_to_end_flow_drops": sum(int(f.get("observed_dropped_packets", 0)) for f in routed_flows),
                      "jitter_origin": "configured" if "jitter_ms" in attrs else "not_configured",
                      "reliability_origin": "configured" if "reliability_ratio" in attrs else "observed_flow_survival_proxy",
                      "loss_origin": "configured_and_single_endpoint_accounting_proxy_not_physical_loss_location",
                      "file_drop_semantics": "file_transfer_packet_loss_not_complete_file_count"})
    result = {"representation": "attributed_multilayer_graph_observation",
            "loss_accounting_semantics": "single_endpoint_proxy_not_physical_loss_localization",
            "attack_arrival_semantics": "positive_competing_packet_load_not_negative_cancellation",
            "file_drop_semantics": "file_transfer_packet_loss_not_complete_file_count",
            "edge_metric_semantics": {"speed_mbps": OFFERED_LINK_RATE_ORIGIN, "utilization_ratio": LINK_LOAD_RATIO_ORIGIN},
            "vertex_metric_names": list(VERTEX_METRICS), "vertex_units": list(VERTEX_UNITS),
            "edge_metric_names": list(EDGE_METRICS), "edge_units": list(EDGE_UNITS),
            "attack_profiles": ATTACK_PROFILES, "nodes": nodes, "edges": edges}
    result["node_metric_maxima"] = {name: max((n["metrics"][name] for n in nodes), default=0.0) for name in VERTEX_METRICS if name != "attack_profile"}
    return compact_runtime_state(result) if detail == "summary" else result


def compact_runtime_state(state: dict[str, Any]) -> dict[str, Any]:
    """Сохранить все показатели без повторения имён в каждой строке."""
    if "nodes" not in state or "edges" not in state:
        return state
    result = dict(state)
    nodes, edges = result.pop("nodes"), result.pop("edges")
    result.update(
        node_ids=[n["id"] for n in nodes],
        vertex_vectors=[list(n["metrics"].values()) for n in nodes],
        node_attack_arrival_rates_pps=[n["attack_arrival_rate_pps"] for n in nodes],
        node_observed_dropped_packets=[n["observed_dropped_packets"] for n in nodes],
        edge_ids=[[e["source"], e["target"]] for e in edges],
        edge_vectors=[list(e["metrics"].values()) for e in edges],
        edge_observed_flow_drops=[e["observed_flow_drops"] for e in edges],
        metric_origins={"cpu_load_percent": "configured_or_queue_busy_estimate",
                        "negative_arrival_rate_pps": "observed_flow_loss_inference_at_single_accounting_endpoint",
                        "loss_probability": "configured_and_single_endpoint_accounting_proxy_not_physical_loss_location",
                        "reliability_ratio": "configured_or_packet_survival_proxy",
                        "jitter_ms": "configured_or_zero_when_unspecified",
                        "buffer_memory_mib": "stationary_waiting_packets_times_mean_wire_packet_size_not_measured_ram",
                        "file_drop_fraction": "file_transfer_packet_loss"},
    )
    return result
