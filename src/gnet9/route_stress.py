"""Условное предупреждение SLA по датчику и цене одной дополнительной потери."""

from collections import Counter, defaultdict
from math import isfinite
from statistics import median

from .koopman import KOOPMAN_MAXIMUM_FORECAST_SECONDS


LOSS_CRITERIA = {"network_packet_loss_percent", "conference_media_loss_percent",
                 "ll_hls_transport_loss_percent", "dns_timeout_ratio_percent"}
SIGNALS = ("dns_query_pressure_ratio", "syn_backlog_growth_ratio", "traffic_acceleration_ratio")
METHOD = "route_stress_one_drop"


class RouteStressMonitor:
    def __init__(self, step_seconds):
        if step_seconds <= 0:
            raise ValueError("Длительность шага должна быть положительной")
        self.period = step_seconds
        self.horizon = KOOPMAN_MAXIMUM_FORECAST_SECONDS // step_seconds * step_seconds
        self.previous = {}
        self.last_time = None

    def observe(self, flows, precursor_observations, *, time_seconds):
        # Читаются только место измерения и три числовых канала.
        readings = {}
        for sample in precursor_observations:
            sensor = sample.get("sensor_node_id")
            if not sensor:
                continue
            for channel in SIGNALS:
                value = sample.get("signals", {}).get(channel, 0.)
                if isinstance(value, (int, float)) and isfinite(value) and value > 0.:
                    key = sensor, channel
                    readings[key] = max(value, readings.get(key, 0.))
        rising = defaultdict(list)
        if self.last_time == time_seconds - self.period:
            for (sensor, channel), value in sorted(readings.items()):
                previous = self.previous.get((sensor, channel))
                if previous is not None and previous < value:
                    rising[sensor].append((sensor, channel, previous, value))
        self.previous, self.last_time = readings, time_seconds

        warnings, one_drop_only, sensor_only = [], [], []
        has_measurements = False
        for flow in flows:
            if flow.get("security_excluded_from_sla_accounting"):
                continue
            metrics = flow.get("slo_evaluation", {}).get("metrics", {})
            service = flow.get("traffic_kind")
            client = flow.get("client_node") or flow.get("flow_id")
            matching = [record for node in (flow.get("route") or ()) for record in rising.get(node, ())
                        if record[1] != "dns_query_pressure_ratio" or
                        (service == "dns" and node == flow.get("server_node"))]
            for criterion, metric in metrics.items():
                if (criterion not in LOSS_CRITERIA or not metric.get("required", True)
                        or metric.get("compliant") is None or metric.get("comparison") != "lte"
                        or metric.get("unit") != "%"):
                    continue
                has_measurements = True
                if not self.horizon or not metric["compliant"]:
                    continue
                is_dns = criterion == "dns_timeout_ratio_percent"
                count = flow.get("dns_queries" if is_dns else "packet_count", 0)
                losses = flow.get("dns_timeout_count" if is_dns else "observed_dropped_packets", 0)
                budget = metric.get("effective_budget_value", metric.get("budget_value"))
                if not count or losses >= count or budget is None:
                    continue
                key = (service, client, criterion)
                if matching:
                    sensor_only.append(key)
                # При тех же N пакетах/запросах ещё один недоставленный даёт 100(k+1)/N %.
                conditional = 100. * (losses + 1) / count
                if conditional <= budget + 1e-12:
                    continue
                one_drop_only.append(key)
                if matching:
                    warnings.append({
                        "method": METHOD, "service": service, "flow_id": client, "criterion": criterion,
                        "grade": flow.get("sla_grade"), "issue_time_seconds": time_seconds,
                        "window_end_seconds": time_seconds + self.horizon, "unit": "%",
                        "measured_value": metric["observed_value"], "budget": budget,
                        "observed_denominator": count, "observed_loss_count": losses,
                        "denominator_unit": "queries" if is_dns else "packets",
                        "conditional_value_if_one_more_loss": conditional,
                        "sensors": [{"sensor_id": node, "signal": name, "previous": old, "current": new}
                                    for node, name, old, new in matching],
                    })
        return {"time_seconds": time_seconds, "step_seconds": self.period,
                "has_measurements": has_measurements, "horizon_seconds": self.horizon,
                "warnings": warnings, "one_drop_only": one_drop_only, "sensor_only": sensor_only}


def evaluate_route_stress(route_views, service_views):
    """Каждое начало расходует одно окно той же услуги, клиента и критерия."""
    if len(route_views) != len(service_views):
        raise ValueError("Нужно одинаковое число шагов датчиков и SLA")
    methods = (METHOD, "one_drop_only", "sensor_only")
    pending = {method: {} for method in methods}
    counts = {method: Counter() for method in methods}
    leads = {method: [] for method in methods}
    by_criterion = {method: defaultdict(Counter) for method in methods}
    observed_seconds = 0
    previous_time, previously_measured = None, False

    def count(method, key, field):
        counts[method][field] += 1
        by_criterion[method][key[2]][field] += 1

    for route_view, sla in zip(route_views, service_views):
        t = route_view["time_seconds"]
        if sla["time_seconds"] != t:
            raise ValueError("Времена шагов датчиков и SLA расходятся")
        contiguous = previous_time is not None and t - previous_time == route_view["step_seconds"]
        if contiguous and previously_measured and route_view["has_measurements"]:
            observed_seconds += route_view["step_seconds"]
        previous_time, previously_measured = t, route_view["has_measurements"]
        for method in methods:
            opened = pending[method]
            if not contiguous:
                for key in opened:
                    count(method, key, "censored")
                opened.clear()
            for gap in sla["measurement_gaps"]:
                key = tuple(gap)
                if opened.pop(key, None) is not None:
                    count(method, key, "censored")
            for onset in sla["onsets"]:
                if onset["criterion"] not in LOSS_CRITERIA:
                    continue
                key = onset["service"], onset["flow_id"], onset["criterion"]
                count(method, key, "onsets")
                if key in opened and opened[key][0] < t <= opened[key][1]:
                    issue, _ = opened.pop(key)
                    count(method, key, "hits")
                    leads[method].append(t - issue)
                else:
                    count(method, key, "misses")
            for key, (_, end) in list(opened.items()):
                if end <= t:
                    count(method, key, "false_alarms")
                    del opened[key]
            if method == METHOD:
                issued = [(w["service"], w["flow_id"], w["criterion"]) for w in route_view["warnings"]]
            else:
                issued = route_view[method]
            for key in issued:
                key = tuple(key)
                count(method, key, "issued_samples")
                if key not in opened:
                    count(method, key, "issued_windows")
                    opened[key] = t, t + route_view["horizon_seconds"]
    for method, opened in pending.items():
        for key in opened:
            count(method, key, "censored")
    results = {}
    for method, totals in counts.items():
        verified = totals["hits"] + totals["false_alarms"]
        results[method] = {
            **{k: totals[k] for k in ("onsets", "hits", "misses", "false_alarms", "censored",
                                      "issued_samples", "issued_windows")},
            "precision": totals["hits"] / verified if verified else None,
            "recall": totals["hits"] / totals["onsets"] if totals["onsets"] else None,
            "observed_hours": observed_seconds / 3600.,
            "false_alarms_per_observed_hour": (
                totals["false_alarms"] * 3600. / observed_seconds if observed_seconds else None),
            "median_lead_seconds": median(leads[method]) if leads[method] else None,
            "lead_seconds": leads[method], "by_criterion": dict(by_criterion[method]),
        }
    return {"methods": results, "criteria": sorted(LOSS_CRITERIA),
            "horizon_seconds": route_views[0]["horizon_seconds"] if route_views else None,
            "condition": "one_additional_loss_at_current_demand_and_rising_observed_route_stress",
            "sensor_origin": "simulated_scenario_assumption",
            "attribution": "route_sensor_and_packet_counters"}
