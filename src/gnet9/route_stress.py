"""Прогноз SLA на следующий отсчёт и условная оценка потерь на маршруте."""

from collections import Counter, defaultdict
from math import isfinite
from statistics import median

from .koopman import KOOPMAN_MAXIMUM_FORECAST_SECONDS
from .sla_flow_projection import flow_projection_inputs, project_bitrate, project_dns_timeout


LOSS_CRITERIA = {"network_packet_loss_percent", "conference_media_loss_percent",
                 "ll_hls_transport_loss_percent", "dns_timeout_ratio_percent"}
SIGNALS = ("dns_query_pressure_ratio", "syn_backlog_growth_ratio", "traffic_acceleration_ratio")
METHOD = "route_stress_one_drop"
OPERATOR_PERIOD = 2
OPERATOR_VERSION = 4
DELAY_STEP_WEIGHT = 0.5
TRAFFIC_ACCELERATION_JUMP = 0.5


def _next_step_warning(flow, criterion, metric, predicted, time_seconds, source, *, forecast_budget=None, **inputs):
    budget = metric.get("effective_budget_value", metric.get("budget_value"))
    limit = budget if forecast_budget is None else forecast_budget
    measured = metric["observed_value"]
    lower_bound = metric["comparison"] == "gte"
    return {"method": "operator", "forecast_source": source, "service": flow["traffic_kind"],
            "flow_id": flow.get("client_node") or flow["flow_id"], "grade": flow.get("sla_grade"),
            "criterion": criterion, "label_ru": metric.get("label_ru", criterion),
            "unit": metric["unit"], "comparison": metric["comparison"], "measured_value": measured,
            "budget": budget, "measured_headroom": measured - budget if lower_bound else budget - measured,
            "predicted_value": predicted, "predicted_budget": limit,
            "predicted_headroom": predicted - limit if lower_bound else limit - predicted,
            "issue_time_seconds": time_seconds, "horizon_seconds": OPERATOR_PERIOD,
            "target_time_seconds": time_seconds + OPERATOR_PERIOD, **inputs}


class RouteStressMonitor:
    def __init__(self, step_seconds, *, operator_version=OPERATOR_VERSION):
        if step_seconds <= 0:
            raise ValueError("Длительность шага должна быть положительной")
        self.period = step_seconds
        if operator_version not in (2, 3, OPERATOR_VERSION):
            raise ValueError("Неизвестная версия рабочего прогноза SLA")
        self.operator_version = operator_version
        self.horizon = KOOPMAN_MAXIMUM_FORECAST_SECONDS // step_seconds * step_seconds
        self.previous = {}
        self.previous_delay = {}
        self.previous_loss = {}
        self.had_sensor_coverage = False
        self.last_time = None

    def _delay_warnings(self, flow, metrics, adjacent, delays, time_seconds):
        queue = flow.get("transport_queue", {})
        criteria = ("one_way_media_latency_ms",)
        if self.operator_version >= 3:
            criteria += ("conference_audio_latency_ms", "conference_video_latency_ms")
        for criterion in criteria:
            delay = metrics.get(criterion, {})
            value = delay.get("observed_value")
            limit = delay.get("effective_budget_value", delay.get("budget_value"))
            if (delay.get("compliant") is None or not delay.get("required", True) or delay.get("unit") != "ms"
                    or delay.get("comparison") != "lte" or value is None or not isfinite(value)
                    or limit is None or not isfinite(limit)):
                continue
            key = flow.get("traffic_kind"), flow.get("client_node") or flow.get("flow_id"), criterion
            extra = queue.get("additional_delay_ms")
            base = value - extra if extra is not None and isfinite(extra) and 0 <= extra <= value else None
            delays[key] = value, base
            if not (adjacent and key in self.previous_delay and delay["compliant"] and flow.get("observed_dropped_packets", 0) > 0):
                continue
            previous, old_base = self.previous_delay[key]
            queue_measured = base is not None and old_base is not None
            if criterion != "one_way_media_latency_ms" and not (queue_measured and queue.get("utilization_rho", 0.) >= 1):
                continue
            forecast = value + DELAY_STEP_WEIGHT * (value - previous)
            source = "damped_flow_delay"
            details = {"previous_value_ms": previous, "step_weight": DELAY_STEP_WEIGHT}
            if self.operator_version >= 3 and queue_measured:
                # При насыщении текущую добавку очереди не переносим в наклон базовой задержки.
                forecast = max(value, base + DELAY_STEP_WEIGHT * (base - old_base))
                source = "queue_adjusted_flow_delay"
                details.update(base_delay_ms=base, previous_base_delay_ms=old_base)
            if forecast > limit:
                yield _next_step_warning(flow, criterion, delay, forecast, time_seconds, source, **details)

    def observe(self, flows, precursor_observations, *, time_seconds):
        # Читаются только место измерения и три числовых канала.
        readings = {}
        for sample in precursor_observations or ():
            sensor = sample.get("sensor_node_id")
            if not sensor:
                continue
            for channel in SIGNALS:
                value = sample.get("signals", {}).get(channel, 0.)
                if isinstance(value, (int, float)) and isfinite(value) and value > 0.:
                    key = sensor, channel
                    readings[key] = max(value, readings.get(key, 0.))
        rising = defaultdict(list)
        jumps = {}
        adjacent = self.last_time == time_seconds - self.period
        if adjacent:
            for (sensor, channel), value in sorted(readings.items()):
                previous = self.previous.get((sensor, channel))
                if previous is not None and previous < value:
                    rising[sensor].append((sensor, channel, previous, value))
                # Полный снимок перечисляет аномалии, отсутствие узла в нём означает 0.
                # None означает отсутствие телеметрии за интервал.
                if (self.period == OPERATOR_PERIOD and self.had_sensor_coverage
                        and channel == "traffic_acceleration_ratio" and value <= 1.
                        and value - (previous or 0.) >= TRAFFIC_ACCELERATION_JUMP):
                    jumps[sensor] = {"sensor_id": sensor, "signal": channel,
                                     "previous": previous or 0., "current": value}
        self.previous, self.last_time = readings, time_seconds
        self.had_sensor_coverage = precursor_observations is not None

        warnings, one_drop_only, sensor_only = [], [], []
        issued, delays, losses_now = [], {}, {}
        has_measurements = False
        new_projection = self.period == OPERATOR_PERIOD and self.operator_version >= 3
        if new_projection:
            flows = tuple(flows)
            offers, queries, server_counts = flow_projection_inputs(flows, include_spread=self.operator_version >= 4)
        for flow in flows:
            if flow.get("security_excluded_from_sla_accounting"):
                continue
            metrics = flow.get("slo_evaluation", {}).get("metrics", {})
            service = flow.get("traffic_kind")
            client = flow.get("client_node") or flow.get("flow_id")
            if self.period == OPERATOR_PERIOD:
                issued.extend(self._delay_warnings(flow, metrics, adjacent, delays, time_seconds))
            if new_projection and "media_bitrate_kbps" in metrics:
                metric = metrics["media_bitrate_kbps"]
                packets = flow.get("packet_count", 0)
                lost = flow.get("observed_dropped_packets", 0)
                if packets > 0 and 0 <= lost <= packets and metric.get("compliant") is not None:
                    stream = service, client
                    losses_now[stream] = lost / packets
                    past = self.previous_loss.get(stream) if adjacent else None
                    rate = project_bitrate(flow, metric, past, losses_now[stream], offers)
                    if rate:
                        predicted, budget, inputs = rate
                        issued.append(_next_step_warning(flow, "media_bitrate_kbps", metric, predicted,
                            time_seconds, "group_demand_growing_loss", forecast_budget=budget, **inputs))
            if new_projection and adjacent and "dns_timeout_ratio_percent" in metrics:
                metric = metrics["dns_timeout_ratio_percent"]
                dns = project_dns_timeout(flow, metric, readings, queries, server_counts)
                if dns:
                    predicted, inputs = dns
                    issued.append(_next_step_warning(flow, "dns_timeout_ratio_percent", metric, predicted,
                        time_seconds, "server_dns_timeout_pressure", **inputs))
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
                if criterion == "network_packet_loss_percent" and (jumps or new_projection):
                    sensors = [jumps[node] for node in (flow.get("route") or ()) if node in jumps]
                    if sensors:
                        issued.append(_next_step_warning(flow, criterion, metric, conditional, time_seconds,
                            "route_acceleration_one_drop", observed_denominator=count,
                            observed_loss_count=losses, sensors=sensors))
                    elif (new_projection and adjacent and losses > 0
                          and flow.get("transport_queue", {}).get("utilization_rho", 0.) > 1.):
                        issued.append(_next_step_warning(flow, criterion, metric, conditional, time_seconds,
                            "overloaded_queue_one_drop", observed_denominator=count, observed_loss_count=losses))
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
        self.previous_delay = delays
        self.previous_loss = losses_now
        return {"time_seconds": time_seconds, "step_seconds": self.period, "operator_warnings": issued,
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
