"""Двухсекундные прогнозы SLA по фактическим счётчикам абонентских потоков."""

from collections import defaultdict
from math import expm1, isfinite, sqrt

from .service_forecast import _offered_rate


LOSS_GROWTH_WEIGHT = 0.25
DEMAND_SPREAD_WEIGHT = 1.5
DNS_PRESSURE_LIMIT = 0.5
DNS_TIMEOUT_RISK_LIMIT = 1 / 3


def flow_projection_inputs(flows, *, include_spread=False):
    """Один кадр: предложенные кбит/с и число запросов усредняются в сопоставимых группах."""
    offers = defaultdict(list)
    queries = defaultdict(list)
    server_counts = defaultdict(lambda: [0, 0])
    for flow in flows:
        if flow.get("security_excluded_from_sla_accounting"):
            continue
        metrics = flow.get("slo_evaluation", {}).get("metrics", {})
        grade = flow.get("sla_grade")
        rate = metrics.get("media_bitrate_kbps", {})
        if rate.get("required", True) and rate.get("budget_value") is not None:
            offered = _offered_rate(flow)
            if offered is not None:
                offers[flow["traffic_kind"], grade, rate["budget_value"]].append(offered)
        if flow.get("traffic_kind") == "dns" and flow.get("server_node") and "dns_queries" in flow:
            n, k = flow["dns_queries"], flow.get("dns_timeout_count", 0)
            if 0 <= k <= n:
                server = flow["server_node"]
                queries[server, grade].append(n)
                server_counts[server][0] += n
                server_counts[server][1] += k
    group_offers = {}
    for group, values in offers.items():
        if len(values) >= 3:
            avg = sum(values) / len(values)
            spread = sqrt(sum((v - avg) ** 2 for v in values) / len(values)) if include_spread else None
            group_offers[group] = avg, spread
    return group_offers, {g: sum(a) / len(a) for g, a in queries.items() if len(a) >= 2}, server_counts


def project_bitrate(flow, metric, previous_loss, loss, group_offers):
    """При спросе 100 кбит/с и росте потерь с 0 до 16% прогноз равен 80 кбит/с."""
    if (not metric.get("compliant") or not metric.get("required", True) or metric.get("unit") != "kbit/s"
            or metric.get("comparison") != "gte" or previous_loss is None or not loss):
        return None
    policy = metric.get("budget_value")
    group = group_offers.get((flow["traffic_kind"], flow.get("sla_grade"), policy))
    if group is None:
        return None
    demand, spread = group
    projected_loss = min(1., loss + LOSS_GROWTH_WEIGHT * max(0., loss - previous_loss))
    budget = min(demand, policy)
    delivered = demand * (1. - projected_loss)
    if delivered >= budget - 1e-6:
        return None
    inputs = {"group_offered_kbps": demand, "current_loss_fraction": loss,
              "previous_loss_fraction": previous_loss, "projected_loss_fraction": projected_loss}
    if spread is not None:
        # Допуск по разбросу разных абонентов за этот же отсчёт, кбит/с.
        upper_offer = demand + DEMAND_SPREAD_WEIGHT * spread
        upper_delivered = upper_offer * (1. - projected_loss)
        if upper_delivered >= min(policy, upper_offer) - 1e-6:
            return None
        inputs.update(group_spread_kbps=spread, upper_offered_kbps=upper_offer,
                      upper_delivered_kbps=upper_delivered, spread_weight=DEMAND_SPREAD_WEIGHT)
    return delivered, budget, inputs


def project_dns_timeout(flow, metric, readings, group_queries, server_counts):
    """Пуассоновская оценка: при четырёх запросах и доле тайм-аутов 1/8 риск ≈ 0,393."""
    if (not metric.get("compliant") or not metric.get("required", True) or metric.get("unit") != "%"
            or metric.get("comparison") != "lte" or not flow.get("dns_queries")):
        return None
    server = flow.get("server_node")
    pressure = readings.get((server, "dns_query_pressure_ratio"), 0.)
    if not DNS_PRESSURE_LIMIT < pressure <= 1.:
        return None
    n, lost = server_counts.get(server, (0, 0))
    demand = group_queries.get((server, flow.get("sla_grade")), 0.)
    limit = metric.get("effective_budget_value", metric.get("budget_value"))
    if not n or not lost or not demand or limit is None or not isfinite(limit) or 100 / demand <= limit:
        return None
    fraction = lost / n
    risk = -expm1(-demand * fraction)
    if risk >= DNS_TIMEOUT_RISK_LIMIT and 100 * fraction > limit:
        return 100 * fraction, {"server_node": server, "group_queries_per_interval": demand,
                                "server_queries": n, "server_timeouts": lost, "dns_pressure": pressure,
                                "estimated_probability_at_least_one_timeout": risk}
    return None
