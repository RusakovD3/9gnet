"""Сохранить измерения симулятора и повторить прогноз по тем же потокам."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gnet9.route_stress import OPERATOR_VERSION, RouteStressMonitor, SIGNALS
from src.gnet9.service_forecast import (
    ServiceSLAForecaster, _score_window, baseline_operator_warnings, evaluate_service_forecasts,
)
from tools.forecast_experiment import compare_at_alarm_budget


# Архив содержит только доступные для наблюдения поля. Численное
# значение, факт выполнения и происхождение сохраняются без пересчёта.
FLOW_STATIC = ("client_node", "traffic_kind", "sla_grade", "application", "server_node",
               "service_node", "transport", "home_access_node")
FLOW_VALUES = ("flow_id", "interval_seconds", "packet_count", "observed_dropped_packets",
               "observed_retransmissions", "payload_bytes", "wire_bytes", "one_way_latency_ms",
               "rtt_ms", "goodput_mbps", "dns_queries", "dns_timeout_count", "data_segments", "arrival_process")
QUEUE_VALUES = ("normal_arrival_rate_pps", "service_rate_pps", "utilization_rho", "mean_queue_packets",
                "mean_system_time_ms", "jitter_ms", "additional_delay_ms", "tcp_recovery_seconds",
                "goodput_mbps", "observed_dropped_packets")
METRIC_STATIC = ("label_ru", "unit", "comparison", "evidence", "value_origin")
METRIC_VALUES = ("observed_value", "budget_value", "effective_budget_value", "compliant", "required")


def _json(item):
    return json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class _Writer:
    def __init__(self, stream):
        self.stream = stream
        self.schemas = {}

    def frame(self, time_seconds, nominal, flows, sensors):
        added, rows = [], []
        for flow in flows:
            if flow.get("security_excluded_from_sla_accounting"):
                continue
            measurements = flow.get("slo_evaluation", {}).get("metrics", {})
            if not measurements:
                continue
            description = tuple((criterion, *(metric.get(k) for k in METRIC_STATIC))
                                for criterion, metric in measurements.items())
            schema = (tuple(flow.get(k) for k in FLOW_STATIC), tuple(flow.get("route", ())), description)
            index = self.schemas.get(schema)
            if index is None:
                index = len(self.schemas)
                self.schemas[schema] = index
                added.append(schema)
            # Изменяемые флаги и действующий порог находятся в самой строке.
            values = []
            for metric in measurements.values():
                normalized = {"required": True, "effective_budget_value": metric.get("budget_value"), **metric}
                values.append([normalized.get(k) for k in METRIC_VALUES])
            queue = flow.get("transport_queue", {})
            rows.append([index, [flow.get(k) for k in FLOW_VALUES], [queue.get(k) for k in QUEUE_VALUES], values])
        self.stream.write(_json({"t": time_seconds, "nominal": nominal, "schemas": added, "rows": rows,
                                 "route_sensors": sensors}) + "\n")


def _header(stream):
    data = json.loads(stream.readline())
    if data.get("format") != "gnet9-flow-observations-v1":
        raise ValueError("Неизвестный формат измерений")
    return data


def archive_info(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        return _header(stream)


def observations(path, *, with_sensors=False):
    """По одному кадру (время, пригодность к обучению, потоки), в порядке симуляции."""
    with gzip.open(path, "rt", encoding="utf8") as stream:
        info = _header(stream)
        schemas = []
        count = 0
        for line in stream:
            frame = json.loads(line)
            if "views_sha256" in frame:
                if count != frame["frame_count"] or count != info["step_count"] + 1:
                    raise ValueError("Число измерений не совпало с заголовком")
                return
            schemas.extend(frame["schemas"])
            flows = []
            for index, values, queue_values, metric_values in frame["rows"]:
                static, route, description = schemas[index]
                flow = dict(zip(info["flow_static_fields"], static, strict=True))
                flow.update((k, v) for k, v in zip(info["flow_value_fields"], values, strict=True) if v is not None)
                flow["route"] = route.copy()
                flow["time_seconds"] = frame["t"]
                flow["step_index"] = count
                flow["transport_queue"] = {k: v for k, v in zip(info["queue_value_fields"], queue_values, strict=True)
                                           if v is not None}
                metrics = {}
                for (name, *labels), row in zip(description, metric_values, strict=True):
                    metric = dict(zip(info["metric_static_fields"], labels, strict=True))
                    metric.update(zip(info["metric_value_fields"], row, strict=True))
                    metric["metric_id"] = name
                    metrics[name] = metric
                flow["slo_evaluation"] = {"metrics": metrics}
                flows.append(flow)
            sample = frame["t"], frame["nominal"], flows
            yield (*sample, frame.get("route_sensors")) if with_sensors else sample
            count += 1
        raise ValueError("В архиве отсутствует завершающая запись")


def _add_digest(digest, view):
    # Сравниваются в том числе все выданные прогнозы, их сроки и пропуски.
    digest.update(_json(view).encode("utf8"))
    digest.update(b"\n")


def replay(path, forecaster=None):
    info = archive_info(path)
    if forecaster is None:
        forecaster = ServiceSLAForecaster(info["step_seconds"], retain_trace=True)
    return [forecaster.observe(flows, time_seconds=t, nominal=nominal)
            for t, nominal, flows in observations(path)]


def record(path, *, scenario, seed, vary_demand=False, step_count=160, step_seconds=2, start_step=70,
           demand_seed=391):
    import numpy as np
    from src.gnet9 import packet_simulator
    from src.gnet9.dynamics import DynamicsConfig, simulate_stationary_dynamics
    from src.gnet9.normal_traffic import poisson_arrivals
    from src.gnet9.topology_builder import GNetBaselineBuilder

    config = DynamicsConfig(step_seconds=step_seconds, step_count=step_count, attack_start_step=start_step,
                            attack_seed=seed, attack_scenario=scenario, snapshot_detail="summary",
                            packet_detail="summary")
    info = {"format": "gnet9-flow-observations-v1", "step_seconds": step_seconds, "step_count": step_count,
            "scenario": scenario, "seed": seed, "vary_demand": vary_demand, "start_step": start_step,
            "demand_seed": demand_seed if vary_demand else None, "records_route_sensors": True,
            "operator_version": OPERATOR_VERSION, "recorded_sensor_lag_steps": 0,
            "flow_static_fields": FLOW_STATIC, "flow_value_fields": FLOW_VALUES,
            "queue_value_fields": QUEUE_VALUES, "metric_static_fields": METRIC_STATIC,
            "metric_value_fields": METRIC_VALUES}
    original = ServiceSLAForecaster.observe
    digest = hashlib.sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    # При сбое не оставлять частичный архив по пути готового результата.
    with NamedTemporaryFile(prefix=path.name+".", dir=path.parent, delete=False) as temporary:
        temp = Path(temporary.name)
    try:
        with temp.open("wb") as binary, gzip.GzipFile(filename="", fileobj=binary, mode="wb",
                                                      compresslevel=2, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf8", newline="\n") as stream:
                stream.write(_json(info) + "\n")
                writer = _Writer(stream)
                nominal_at = {}

                def capture(model, flows, *, time_seconds, nominal=True):
                    nominal_at[time_seconds] = nominal
                    view = original(model, flows, time_seconds=time_seconds, nominal=nominal)
                    _add_digest(digest, view)
                    return view

                original_route = RouteStressMonitor.observe
                def capture_route(monitor, flows, readings, *, time_seconds):
                    sensors = None if readings is None else [
                        {"sensor_node_id": row["sensor_node_id"],
                         "signals": {k: row["signals"][k] for k in SIGNALS if k in row.get("signals", {})}}
                        for row in readings if row.get("sensor_node_id")]
                    writer.frame(time_seconds, nominal_at.pop(time_seconds), flows, sensors)
                    return original_route(monitor, flows, readings, time_seconds=time_seconds)

                def shifted(expected_count, *, node_id, step):
                    # Тот же отложенный профиль, что и forecast_experiment --vary-demand.
                    factor = 1 + .12 * np.sin(2 * np.pi * step / 32)
                    return poisson_arrivals(factor * expected_count, node_id=node_id, step=step, seed=demand_seed)

                demand = patch.object(packet_simulator, "poisson_arrivals", shifted) if vary_demand else nullcontext()
                with (patch.object(ServiceSLAForecaster, "observe", capture),
                      patch.object(RouteStressMonitor, "observe", capture_route), demand):
                    result = simulate_stationary_dynamics(GNetBaselineBuilder().build(), config)
                issued = hashlib.sha256()
                for snap in result["snapshots"]:
                    _add_digest(issued, snap["service_sla_forecast"]["operator_warnings"])
                stream.write(_json({"views_sha256": digest.hexdigest(), "operator_sha256": issued.hexdigest(),
                                    "frame_count": result["snapshot_count"]}) + "\n")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return digest.hexdigest()


def _footer(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        for last in stream:
            pass
    return json.loads(last)


def verify(path, views):
    digest = hashlib.sha256()
    for view in views:
        _add_digest(digest, view)
    expected = _footer(path)["views_sha256"]
    if digest.hexdigest() != expected:
        raise ValueError(f"Прогноз отличается от версии или параметров при записи: "
                         f"{digest.hexdigest()} вместо {expected}")
    return digest.hexdigest()


def _with_lagged_sensors(frames, period):
    previous_time, previous_sensors = None, None
    for t, nominal, flows, sensors in frames:
        delayed = previous_sensors if previous_time == t - period else None
        yield t, nominal, flows, sensors, delayed
        previous_time, previous_sensors = t, sensors


def _operator_views(path, variants, *, baseline=False):
    """Один раз прочитать измерения, сверить записанный выпуск и подготовить общий исход SLA."""
    info = archive_info(path)
    if info["step_seconds"] != 2 or not info.get("records_route_sensors"):
        raise ValueError("Для сравнения запишите двухсекундный ряд с датчиками командой record.")
    # В первых архивах с датчиками записывалась вторая версия без задержки.
    recorded = info.get("operator_version", 2), info.get("recorded_sensor_lag_steps", 0)
    if recorded[1] != 0:
        raise ValueError("Для эксперимента нужны исходные кадры датчиков без сдвига.")
    monitors = {key: RouteStressMonitor(2, operator_version=key[0])
                for key in dict.fromkeys([recorded, *variants.values()])}
    forecaster = ServiceSLAForecaster(2)
    raw_digest, issued_digest = hashlib.sha256(), hashlib.sha256()
    series = {name: [] for name in variants}
    if baseline:
        series["baseline"] = []
    for t, nominal, flows, sensors, delayed in _with_lagged_sensors(observations(path, with_sensors=True), 2):
        raw = forecaster.observe(flows, time_seconds=t, nominal=nominal)
        _add_digest(raw_digest, raw)
        issued = {key: monitor.observe(flows, delayed if key[1] else sensors, time_seconds=t)["operator_warnings"]
                  for key, monitor in monitors.items()}
        _add_digest(issued_digest, issued[recorded])
        common = {k: raw[k] for k in ("time_seconds", "step_seconds", "services", "onsets",
                                      "measurement_gaps", "left_censored_impairments")}
        for name, key in variants.items():
            series[name].append({**common, "warnings": issued[key]})
        if baseline:
            series["baseline"].append({**common, "warnings": baseline_operator_warnings(raw)})
    footer = _footer(path)
    if (raw_digest.hexdigest(), issued_digest.hexdigest()) != (footer["views_sha256"], footer["operator_sha256"]):
        raise ValueError("Прогнозы при чтении расходятся с выдачей во время симуляции.")
    verification = {"archive": info, "matching_window_seconds": 2, "recorded_operator_version": recorded[0],
                    "verified_raw_sha256": raw_digest.hexdigest(), "verified_operator_sha256": issued_digest.hexdigest()}
    return verification, series


def compare_operator(path):
    """Сравнить исходное правило Купмана с версией рабочего прогноза на момент записи."""
    version = archive_info(path).get("operator_version", 2)
    verification, series = _operator_views(path, {"updated": (version, 0)}, baseline=True)
    return {**verification, "baseline": _score_window(series["baseline"], "koopman", 2)[0],
            "updated": _score_window(series["updated"], "operator", 2)[0]}


def _score_events(views):
    events = {field: {} for field in ("hits", "misses", "false_alarms", "censored")}
    by_criterion = {}

    def collect(field, row):
        # При горизонте ровно в один отсчёт целевое время попадания совпадает с началом нарушения.
        t = row["time_seconds"] if field == "misses" else row["target_time_seconds"]
        key = (t, row["service"], row["flow_id"], row["criterion"])
        events[field][key] = row
        counts = by_criterion.setdefault(row["criterion"], dict.fromkeys(events, 0))
        counts[field] += 1

    score = _score_window(views, "operator", 2, on_outcome=collect)[0]
    score["by_criterion"] = by_criterion
    return score, events


def _intersection_counts(older, newer, candidate_version):
    return {"v2_only": len(older - newer), "shared": len(older & newer),
            f"v{candidate_version}_only": len(newer - older)}


def _original_misses(misses, older_hits, newer_hits, candidate_version):
    keys = misses.keys()
    older, newer = keys & older_hits.keys(), keys & newer_hits.keys()

    def summarize(group):
        return {"count": len(group), "v2_lagged_hits": len(group & older),
                f"v{candidate_version}_lagged_hits": len(group & newer),
                **_intersection_counts(group & older, group & newer, candidate_version),
                "neither": len(group - older - newer)}

    criteria = {}
    events = []
    for key, onset in sorted(misses.items()):
        t, service, flow_id, criterion = key
        criteria.setdefault(criterion, set()).add(key)
        events.append({"time_seconds": t, "service": service, "flow_id": flow_id, "criterion": criterion,
                       "observed_value": onset["measured_value"], "budget": onset["budget"],
                       "v2_lagged_source": older_hits.get(key, {}).get("forecast_source"),
                       f"v{candidate_version}_lagged_source": newer_hits.get(key, {}).get("forecast_source")})
    return {"count": len(keys), "overlap": summarize(keys),
            "by_criterion": {name: summarize(group) for name, group in sorted(criteria.items())}, "events": events}


def compare_lagged(path, *, candidate_version=OPERATOR_VERSION):
    """Сопоставить одни и те же начала по услуге, потоку, критерию и моменту времени."""
    if candidate_version not in (3, 4):
        raise ValueError("Для сравнения с версией 2 укажите кандидата 3 или 4.")
    candidate = f"v{candidate_version}_lagged"
    variants = {"v2_timely": (2, 0), "v2_lagged": (2, 1), candidate: (candidate_version, 1)}
    verification, series = _operator_views(path, variants)
    scores, events = {}, {}
    for name, views in series.items():
        scores[name], events[name] = _score_events(views)
    older, newer = events["v2_lagged"], events[candidate]
    return {**verification, "candidate_version": candidate_version, "sensor_lag_seconds": 2,
            "flow_counter_lag_seconds": 0, "scores": scores,
            "original_misses": _original_misses(events["v2_timely"]["misses"], older["hits"], newer["hits"],
                                                candidate_version),
            "lagged_changes": {field: _intersection_counts(older[field].keys(), newer[field].keys(), candidate_version)
                               for field in ("hits", "false_alarms", "censored")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("record", help="записать измерения штатной симуляции")
    capture.add_argument("path", type=Path)
    capture.add_argument("--scenario", choices=("predictive-demo", "mitre-demo"), default="predictive-demo")
    capture.add_argument("--seed", type=int, required=True)
    capture.add_argument("--vary-demand", action="store_true")
    capture.add_argument("--demand-seed", type=int, default=391, help="Начальное число ряда переменного спроса")
    capture.add_argument("--start-step", type=int, default=70, help="Первый шаг воздействий в симуляции")
    compare = commands.add_parser("compare-operator", help="сравнить версии рабочего прогноза за 2 с")
    compare.add_argument("path", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    lagged = commands.add_parser("compare-lagged", help="сравнить кандидата с версией 2 при задержке датчиков на 2 с")
    lagged.add_argument("path", type=Path)
    lagged.add_argument("--output", type=Path, required=True)
    lagged.add_argument("--candidate-version", type=int, choices=(3, 4), default=OPERATOR_VERSION,
                        help="версия рабочего прогноза для сравнения (по умолчанию %(default)s)")
    scoring = commands.add_parser("replay", help="пересчитать оценку по архиву")
    scoring.add_argument("path", type=Path)
    scoring.add_argument("--verify", action="store_true", help="сравнить прогнозы с версией при записи")
    scoring.add_argument("--output", type=Path)
    scoring.add_argument("--delivery-model", action=argparse.BooleanOptionalAction, default=None,
                         help="включить или выключить прогноз скорости по доставке")
    scoring.add_argument("--route-delay", action=argparse.BooleanOptionalAction, default=None,
                         help="включить или выключить прогноз задержки по потокам одного пути")
    group = scoring.add_mutually_exclusive_group()
    group.add_argument("--calibration", type=Path, help="JSON с ранее выбранным sla_alarm_budget")
    group.add_argument("--fit-alarm-budget", action="store_true")
    args = parser.parse_args()
    if args.command == "record":
        print(record(args.path, scenario=args.scenario, seed=args.seed, vary_demand=args.vary_demand,
                     demand_seed=args.demand_seed, start_step=args.start_step))
        return
    if args.command in {"compare-operator", "compare-lagged"}:
        if args.command == "compare-lagged":
            result = compare_lagged(args.path, candidate_version=args.candidate_version)
        else:
            result = compare_operator(args.path)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        return
    info = archive_info(args.path)
    if args.fit_alarm_budget and (info["scenario"], info["seed"], info["vary_demand"]) != ("predictive-demo", 11, False):
        parser.error("Для подбора используется отдельный калибровочный прогон predictive-demo/11.")
    options = {k: v for k, v in (("delivery_model", args.delivery_model), ("route_delay", args.route_delay))
               if v is not None}
    forecaster = ServiceSLAForecaster(info["step_seconds"], retain_trace=True, **options)
    views = replay(args.path, forecaster)
    if args.verify:
        print(verify(args.path, views))
    evaluation = evaluate_service_forecasts(views)
    forecaster.case_traces(evaluation)
    result = {"service_sla_evaluation": evaluation}
    if args.fit_alarm_budget or args.calibration:
        calibration = None
        if args.calibration:
            calibration = json.loads(args.calibration.read_text(encoding="utf8"))["sla_alarm_budget"]
        result["sla_alarm_budget"] = compare_at_alarm_budget(views, calibration)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(_json(result) + "\n", encoding="utf8")
    elif not args.verify:
        print(_json(result))


if __name__ == "__main__":
    main()
