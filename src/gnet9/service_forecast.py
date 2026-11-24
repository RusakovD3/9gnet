"""Причинный прогноз запаса до порога SLA для отдельного потока.

Для верхнего ограничения запас x = budget - value, для нижнего наоборот.
Оба наблюдения (x, d = x[t] - x[t-1]) измерены в единицах критерия.
Шаг оператора Купмана переводит их в (x + a*d, a*d). При a=1, x=3 мс,
d=-2 мс/шаг первое строгое нарушение наступит через два шага.
"""

from __future__ import annotations

from array import array
from collections import defaultdict
from math import isfinite
from statistics import mean, median

from .koopman import KOOPMAN_MAXIMUM_FORECAST_SECONDS


METHODS = ("koopman", "persistence", "trend")
CASE_TYPES = ("hit", "false_alarm", "miss")
# Выбор на целом predictive-demo/11 для периода 2 с. Соответствие бюджету
# проверяется заново на каждом независимом прогоне.
CALIBRATED_FLOW_RULES = {"one_way_media_latency_ms": {
    "confirmations": 1, "require_loss": True, "issue_horizon_seconds": 2}}


def warning_meets_rule(warning, rule):
    return bool(rule and (not rule["require_loss"] or warning["observed_loss_fraction"])
                and warning["horizon_seconds"] <= rule.get("issue_horizon_seconds", 20))


def operator_warnings(view):
    """Правила выбранного режима используют одно текущее измерение потерь."""
    if view["step_seconds"] != 2:
        return []
    return [w for w in view["warnings"] if w["method"] == "koopman"
            and warning_meets_rule(w, CALIBRATED_FLOW_RULES.get(w["criterion"]))]


def _crossing(margin, delta, coefficient, steps):
    """Первый дискретный шаг с отрицательным запасом и величина этого запаса."""
    if delta == 0 or coefficient == 0:
        return None
    for step in range(1, steps + 1):
        delta *= coefficient
        margin += delta
        if margin < -1e-9:
            return step, margin
    return None


def _identity(row):
    return row["service"], row["flow_id"], row["criterion"]


def _offered_rate(flow):
    """Предложенная полезная нагрузка, кбит/с, за текущий интервал."""
    size, period = flow.get("payload_bytes"), flow.get("interval_seconds")
    if size is None or period is None or period <= 0:
        return None
    value = 0.008 * size / period
    return value if isfinite(value) and value >= 0 else None


def _delivery_projection(offered, delivered, nominal_offer, policy_limit):
    """Условное среднее при независимом спросе и сохранении эффективности.

    На (1,q,e,q·e) оператор даёт (1,μ,e,μ·e). Например, при μ=80 кбит/с
    и e=0,75 следующий полезный битрейт равен 60 кбит/с. Все более дальние
    условные средние совпадают. Для порога используется тот же прогноз спроса.
    """
    efficiency = min(1., max(0., delivered / offered))
    return nominal_offer * efficiency, min(policy_limit, nominal_offer), efficiency


def _route_delay_drift(flows, previous):
    """Медиана изменений задержки на общей части пути у потоков с потерями.

    Каждая группа содержит минимум три различных потока. Одно изменение
    само по себе не устанавливает знак медианы. Единица — мс за отсчёт.
    """
    groups = defaultdict(dict)
    memberships = {}
    for flow in flows:
        client = flow.get("client_node")
        if (not client or not flow.get("route") or not flow.get("observed_dropped_packets")
                or flow.get("security_excluded_from_sla_accounting")):
            continue
        path = tuple(node for node in flow["route"] if node != client)
        if not path:
            continue
        service = flow["traffic_kind"]
        for criterion, metric in flow.get("slo_evaluation", {}).get("metrics", {}).items():
            if metric.get("unit") != "ms" or metric.get("comparison") != "lte":
                continue
            key = (service, client, criterion)
            old = previous.get(key)
            value = metric.get("observed_value")
            if (old is None or value is None or not isfinite(value)
                    or not metric.get("required", True) or metric.get("compliant") is None):
                continue
            group = path, service, criterion
            groups[group][client] = value - old
            memberships[key] = group
    estimates = {group: (median(deltas.values()), len(deltas))
                 for group, deltas in groups.items() if len(deltas) >= 3}
    return {key: estimates[group] for key, group in memberships.items() if group in estimates}


class ServiceSLAForecaster:
    """Один анализатор на равномерно измеренный ряд. Обучение действует со следующего шага."""

    def __init__(self, step_seconds, *, retain_trace=False, delivery_model=True, route_delay=True):
        if step_seconds <= 0:
            raise ValueError("Период наблюдений должен быть положительным")
        self.step_seconds = step_seconds
        self.steps = int(KOOPMAN_MAXIMUM_FORECAST_SECONDS // step_seconds)
        self.previous = {}
        self.last_time = None
        self.previous_nominal = False
        # Суммы d[t-1]*d[t] и d[t-1]² отдельно для критерия, услуги и класса.
        self.fit = {}
        self.delivery_model = delivery_model
        self.nominal_offers = {}
        self.previous_offers = {}
        self.route_delay = route_delay
        self.previous_delay = {}
        self.trace = defaultdict(lambda: array("d")) if retain_trace else None

    def observe(self, flows, *, time_seconds, nominal=True):
        if self.last_time is not None and time_seconds != self.last_time + self.step_seconds:
            raise ValueError("Для пропущенного шага передайте пустые наблюдения")
        services, warnings, onsets, left_censored = {}, [], [], []
        current, contributions = {}, defaultdict(lambda: [0., 0., 0])
        failed_any = False
        offers = {}
        delays = {}
        if self.route_delay:
            flows = tuple(flows)
        route_drift = _route_delay_drift(flows, self.previous_delay) if self.route_delay else {}
        for flow in flows:
            if flow.get("security_excluded_from_sla_accounting"):
                continue
            metrics = flow.get("slo_evaluation", {}).get("metrics", {})
            if not metrics:
                continue
            service = flow["traffic_kind"]
            flow_id = flow.get("client_node") or flow["flow_id"]
            grade = flow.get("sla_grade")
            stream = (service, flow_id)
            offered = _offered_rate(flow) if flow.get("arrival_process") == "poisson" else None
            summary = services.setdefault(service, {
                "assessed_flows": 0, "failed_flows": 0, "assessed_criteria": 0,
                "unassessed_required_criteria": 0, "worst_observed": None,
                "numeric_criteria": 0, "identified_numeric_criteria": 0, "earliest_koopman_warning": None,
            })
            assessed, failed = False, False
            for criterion, metric in metrics.items():
                if metric.get("required", True) is False:
                    continue
                value = metric.get("observed_value")
                budget = metric.get("effective_budget_value", metric.get("budget_value"))
                comparison = metric.get("comparison")
                passed = metric.get("compliant")
                if (passed is None or value is None or budget is None
                        or not isfinite(value) or not isfinite(budget)
                        or comparison not in {"eq", "lte", "gte"}):
                    summary["unassessed_required_criteria"] += 1
                    continue
                assessed = True
                failed |= not passed
                summary["assessed_criteria"] += 1
                key = (service, flow_id, criterion)
                unit = metric["unit"]
                if self.route_delay and unit == "ms" and comparison == "lte":
                    delays[key] = value
                sign = -1 if comparison == "lte" else 1
                margin = (float(value == budget) - 1 if comparison == "eq"
                          else sign * (float(value) - float(budget)))
                previous = self.previous.get(key)
                delta = margin - previous[0] if previous is not None else None
                current[key] = margin, delta, passed
                if self.trace is not None:
                    self.trace[key].extend((time_seconds, value, budget))

                # Значение в долях порога только для ранжирования. При нулевом
                # бюджете / булевом условии мерой нарушения служит его наличие.
                severity = (-margin / abs(budget) if budget and comparison != "eq" else float(not passed))
                info = {
                    "service": service, "flow_id": flow_id, "grade": grade, "criterion": criterion,
                    "label_ru": metric.get("label_ru", criterion), "unit": unit, "comparison": comparison,
                    "measured_value": value, "budget": budget, "policy_budget": metric.get("budget_value"),
                    "measured_headroom": margin,
                    "evidence": metric.get("evidence"), "value_origin": metric.get("value_origin"),
                }
                worst = summary["worst_observed"]
                if worst is None or (not passed, severity) > (not worst["compliant"], worst["relative_excess"]):
                    summary["worst_observed"] = {**info, "compliant": passed, "relative_excess": severity}
                if not passed and previous is not None and previous[2]:
                    onsets.append({**info, "time_seconds": time_seconds})
                if not passed and previous is None:
                    left_censored.append({**info, "time_seconds": time_seconds})
                # У дискретной доступности нет физической производной.
                if comparison == "eq":
                    continue
                summary["numeric_criteria"] += 1
                if previous is None:
                    continue
                group = (service, grade, criterion)
                num, den, count = self.fit.get(group, (0., 0., 0))
                rate_model = (self.delivery_model and comparison == "gte" and unit == "kbit/s"
                              and offered is not None and offered > 0
                              and metric.get("budget_value") is not None
                              and stream in self.nominal_offers)
                summary["identified_numeric_criteria"] += int(rate_model or (key not in route_drift and den > 1e-18))
                if key in route_drift:
                    summary["route_drift_criteria"] = summary.get("route_drift_criteria", 0) + 1
                if passed and self.steps:
                    # При отсутствии возбуждения оператор не идентифицирован.
                    methods = [("trend", 1.)]
                    if rate_model or key in route_drift:
                        methods.append(("koopman", None))
                    elif den > 1e-18:
                        methods.append(("koopman", min(1., max(-1., num / den))))
                    for method, coefficient in methods:
                        forecast_budget = budget
                        pairs = count if method == "koopman" else None
                        origin = "healthy_margin_increment" if method == "koopman" else "last_margin_increment"
                        inputs = {}
                        if coefficient is None and rate_model:
                            average, pairs = self.nominal_offers[stream]
                            prediction, forecast_budget, efficiency = _delivery_projection(
                                offered, value, average, metric["budget_value"])
                            # Показания SLA округлены до шести знаков экспортёром.
                            if prediction >= forecast_budget - 1e-6:
                                continue
                            n = 1
                            origin = "poisson_demand_delivery_koopman"
                            inputs = {"offered_kbps": offered, "nominal_offered_kbps": average,
                                      "delivery_efficiency": efficiency}
                        elif coefficient is None:
                            drift, peers = route_drift[key]
                            crossing = _crossing(margin, -drift, 1., self.steps)
                            if crossing is None:
                                continue
                            n, projected_margin = crossing
                            prediction = budget - projected_margin
                            origin = "shared_route_delay_translation"
                            inputs = {"route_drift_ms_per_step": drift, "observed_route_flows": peers}
                            pairs = None
                        else:
                            crossing = _crossing(margin, delta, coefficient, self.steps)
                            if crossing is None:
                                continue
                            n, projected_margin = crossing
                            prediction = max(0., budget + sign * projected_margin)
                        horizon = n * self.step_seconds
                        if unit == "%":
                            prediction = min(prediction, 100.)
                        if sign * (prediction - forecast_budget) >= -1e-9:
                            continue
                        warning = {
                            **info, "method": method, "issue_time_seconds": time_seconds,
                            "target_time_seconds": time_seconds + horizon, "horizon_seconds": horizon,
                            "predicted_value": prediction,
                            "predicted_headroom": sign * (prediction - forecast_budget),
                            "predicted_budget": forecast_budget, "forecast_source": origin, **inputs,
                            "observed_loss_fraction": (flow.get("observed_dropped_packets", 0)
                                                       / max(1, flow.get("packet_count", 0))),
                            "margin_delta": delta, "coefficient_used": coefficient,
                            "training_pairs_used": pairs,
                        }
                        warnings.append(warning)
                        earliest = summary["earliest_koopman_warning"]
                        if method == "koopman" and (earliest is None or horizon < earliest["horizon_seconds"]):
                            summary["earliest_koopman_warning"] = warning
                previous_delta = previous[1]
                if self.previous_nominal and previous_delta is not None and previous[2] and passed:
                    totals = contributions[group]
                    totals[0] += previous_delta * delta
                    totals[1] += previous_delta * previous_delta
                    totals[2] += 1
            summary["assessed_flows"] += int(assessed)
            if assessed and offered is not None:
                offers[stream] = offered
            summary["failed_flows"] += int(failed)
            failed_any |= failed
        healthy = bool(current) and nominal and not failed_any
        if healthy:
            if self.previous_nominal:
                for stream, offered in offers.items():
                    if stream in self.previous_offers:
                        average, count = self.nominal_offers.get(stream, (0., 0))
                        self.nominal_offers[stream] = average + (offered - average) / (count + 1), count + 1
            for group, (dn, dd, dc) in contributions.items():
                num, den, count = self.fit.get(group, (0., 0., 0))
                self.fit[group] = num + dn, den + dd, count + dc
        gaps = sorted(self.previous.keys() - current.keys())
        # Разность после пропуска или нештатного состояния не образует
        # исправную пару для обучения следующего шага.
        if not (healthy and self.previous_nominal):
            current = {k: (m, None, p) for k, (m, d, p) in current.items()}
        self.previous, self.previous_nominal = current, healthy
        self.previous_offers = offers
        self.previous_delay = delays
        self.last_time = time_seconds
        return {"time_seconds": time_seconds, "step_seconds": self.step_seconds, "services": services,
                "warnings": warnings, "onsets": onsets, "measurement_gaps": gaps,
                "left_censored_impairments": left_censored}

    def case_traces(self, evaluation):
        """Сохранить физические измерения только в окрестности выбранных примеров."""
        if self.trace is None:
            return
        for case in evaluation["cases"].values():
            if case is None:
                continue
            samples = iter(self.trace[_identity(case)])
            center = case.get("issue_time_seconds", case.get("time_seconds"))
            if "margin_delta" in case and "horizon_seconds" in case:
                sign = -1 if case["comparison"] == "lte" else 1
                trend = max(0., case["measured_value"] + sign * case["margin_delta"]
                            * case["horizon_seconds"] / self.step_seconds)
                case["trend_at_target"] = min(100., trend) if case["unit"] == "%" else trend
            case["measurements"] = [
                {"time_seconds": t, "value": v, "budget": b}
                for t, v, b in zip(samples, samples, samples)
                if center - KOOPMAN_MAXIMUM_FORECAST_SECONDS <= t
                <= center + KOOPMAN_MAXIMUM_FORECAST_SECONDS + self.step_seconds
            ]


def _score_window(views, method, horizon):
    """Одна открытая тревога на поток и критерий, исход измеряется один раз."""
    pending = {}
    cases = dict.fromkeys(CASE_TYPES)
    totals = dict.fromkeys(("onsets", "hits", "misses", "false_alarms", "censored",
                            "left_censored_impairments", "observed_seconds", "warning_seconds"), 0)
    by_service = {}
    leads, errors = [], []
    sources = {}

    def count(service, field):
        totals[field] += 1
        by_service[service][field] += 1

    def outcome(warning, field):
        count(warning["service"], field)
        source = warning.get("forecast_source", method)
        sources.setdefault(source, dict.fromkeys(("hits", "false_alarms", "censored"), 0))[field] += 1

    last_observed = set()
    for view in views:
        t = view["time_seconds"]
        for service in view["services"]:
            by_service.setdefault(service, dict.fromkeys(totals, 0))
        for key in view["measurement_gaps"]:
            warning = pending.pop(tuple(key), None)
            if warning is not None:
                outcome(warning, "censored")
        observed = {s for s, row in view["services"].items() if row["assessed_flows"]}
        common = last_observed & observed
        warned = {w["service"] for w in pending.values()} & common
        for field, covered in (("observed_seconds", common), ("warning_seconds", warned)):
            if covered:
                totals[field] += view["step_seconds"]
                for s in covered:
                    by_service[s][field] += view["step_seconds"]
        last_observed = observed
        for item in view.get("left_censored_impairments", ()):
            count(item["service"], "left_censored_impairments")
        for onset in view["onsets"]:
            service = onset["service"]
            count(service, "onsets")
            warning = pending.pop(_identity(onset), None)
            if warning is None:
                count(service, "misses")
                if cases["miss"] is None:
                    cases["miss"] = dict(onset)
            else:
                outcome(warning, "hits")
                leads.append(t - warning["issue_time_seconds"])
                errors.append(warning["target_time_seconds"] - t)
                if cases["hit"] is None:
                    cases["hit"] = {**warning, "actual_time_seconds": t, "actual_value": onset["measured_value"]}
        for key, warning in tuple(pending.items()):
            if warning["issue_time_seconds"] + horizon <= t:
                outcome(warning, "false_alarms")
                pending.pop(key)
                if cases["false_alarm"] is None:
                    cases["false_alarm"] = {**warning, "observed_through_seconds": t}
        for warning in view["warnings"]:
            if warning["method"] == method and warning["horizon_seconds"] <= horizon:
                pending.setdefault(_identity(warning), warning)
    for warning in pending.values():
        outcome(warning, "censored")
    for row in [totals, *by_service.values()]:
        verified = row["hits"] + row["false_alarms"]
        row["precision"] = row["hits"] / verified if verified else None
        row["recall"] = row["hits"] / row["onsets"] if row["onsets"] else None
        duration = row["observed_seconds"]
        row["false_alarms_per_observed_hour"] = 3600 * row["false_alarms"] / duration if duration else None
        row["warning_occupancy_ratio"] = row["warning_seconds"] / duration if duration else None
    return {**totals, "lead_seconds": leads, "timing_error_seconds": errors,
            "mean_absolute_timing_error_seconds": mean(map(abs, errors)) if errors else None,
            "by_service": by_service, "by_forecast_source": sources}, cases


def evaluate_service_forecasts(views):
    """Сверить прогнозы с началом нарушения того же условия того же потока."""
    period = views[0]["step_seconds"] if views else None
    max_horizon = (int(KOOPMAN_MAXIMUM_FORECAST_SECONDS // period) * period if period else 0)
    horizons = sorted({h for h in (period, 2, 10, max_horizon)
                       if h and period and h <= max_horizon and h % period == 0})
    by_horizon = {}
    cases = dict.fromkeys(CASE_TYPES)
    for h in horizons:
        methods = {}
        for method in METHODS:
            methods[method], examples = _score_window(views, method, h)
            if h == max_horizon and method == "koopman":
                cases = examples
        by_horizon[str(h)] = methods
    worst = None
    by_service = {}
    for view in views:
        for service, row in view["services"].items():
            if not row["assessed_flows"]:
                continue
            ratio = (row["assessed_flows"] - row["failed_flows"]) / row["assessed_flows"]
            summary = by_service.setdefault(service, {
                "minimum_compliance_ratio": ratio, "worst_time_seconds": view["time_seconds"],
                "first_warning": None, "identified_criterion_observations": 0, "numeric_criterion_observations": 0,
                "route_drift_criterion_observations": 0,
            })
            if ratio < summary["minimum_compliance_ratio"]:
                summary.update(minimum_compliance_ratio=ratio, worst_time_seconds=view["time_seconds"])
            summary["identified_criterion_observations"] += row.get("identified_numeric_criteria", 0)
            summary["route_drift_criterion_observations"] += row.get("route_drift_criteria", 0)
            summary["numeric_criterion_observations"] += row.get("numeric_criteria", 0)
            if worst is None or ratio < worst["compliance_ratio"]:
                worst = {"service": service, "time_seconds": view["time_seconds"], "compliance_ratio": ratio,
                         "assessed_flows": row["assessed_flows"], "failed_flows": row["failed_flows"],
                         "criterion_at_worst": row["worst_observed"]}
        for warning in view["warnings"]:
            if warning["method"] == "koopman":
                service = by_service[warning["service"]]
                earlier = service["first_warning"]
                if earlier is None or (earlier["issue_time_seconds"] == view["time_seconds"]
                                       and warning["horizon_seconds"] < earlier["horizon_seconds"]):
                    service["first_warning"] = warning
    return {"maximum_horizon_seconds": max_horizon, "step_seconds": period,
            "methods": by_horizon.get(str(max_horizon), {}),
            "by_horizon_seconds": by_horizon, "cases": cases, "worst_service": worst, "by_service": by_service,
            "attribution": {"measurement": "required per-flow simulated SLO metrics and their effective limits",
                            "signature": "none", "numeric_model": "Poisson demand and delivery efficiency for bitrate; "
                            "measured shared-route drift under packet loss; "
                            "healthy-pair delay-coordinate Koopman block for remaining numeric criteria",
                            "baselines": ["persistence", "trend"]},
            "matching": "first warning until horizon endpoint; same flow, service and criterion; "
                        "positive lead; gap or incomplete run end is censored",
            "timing_error_definition": "target_time_seconds minus measured_onset_seconds"}
