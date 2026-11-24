"""Измерить сигналы перед нарушением в учебном и отложенном прогонах."""

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gnet9.attacks import active_attack_events, advance_power_runtime_state, observe_attack_precursors
from src.gnet9.dynamics import DynamicsConfig
from src.gnet9.packet_simulator import simulate_packet_snapshot
from src.gnet9.route_stress import SIGNALS, RouteStressMonitor, evaluate_route_stress
from src.gnet9.routing import FixedTopologyRoutes
from src.gnet9.topology_builder import GNetBaselineBuilder
from tools.forecast_experiment import varying_demand


def packets_for_run(scenario, seed):
    model = GNetBaselineBuilder().build()
    routing, power = FixedTopologyRoutes(model), {}
    config = DynamicsConfig(step_seconds=2, step_count=160, attack_start_step=70,
                            attack_seed=seed, attack_scenario=scenario)
    for step in range(config.step_count + 1):
        options = dict(step_index=step, time_seconds=step * 2, step_seconds=2)
        params = dict(**options, step_count=config.step_count, seed=config.attack_seed, config=config)
        events = active_attack_events(model, config.attack_scenario, **params)
        events, runtime = advance_power_runtime_state(model, events, power, **options)
        sensors = observe_attack_precursors(model, config.attack_scenario, **params)
        packet = simulate_packet_snapshot(model, **options, detail="summary", retain_flows=True,
            attack_events=events, attack_precursors=sensors, attack_scenario=config.attack_scenario,
            power_runtime_state=runtime, routing=routing)
        yield step * 2, packet["flows"], packet["summary"]["observed_links"], sensors


def tally(counter, onset, flags):
    label = "onset" if onset else "no_onset"
    counter[label + "_n"] += 1
    for flag in flags:
        counter[label + "_" + flag] += 1


def audit(scenario, seed):
    monitor = RouteStressMonitor(2)
    route_views, service_views, sensor_series, dns_qps = [], [], [], []
    previous, previous_losses, baseline_latency, previous_sensors = {}, {}, {}, {}
    counts, first_loss, by_criterion = Counter(), Counter(), defaultdict(Counter)
    baseline_util = {}
    for time_seconds, packets, links, samples in packets_for_run(scenario, seed):
        flows = [f for f in packets if f.get("slo_evaluation") and not f.get("security_excluded_from_sla_accounting")]
        route_views.append(monitor.observe(flows, samples, time_seconds=time_seconds))
        util = {tuple(sorted((r["source"], r["target"]))): r["utilization_percent"] for r in links}
        if not time_seconds:
            baseline_util = dict(util)
        sensor_nodes = {s["sensor_node_id"] for s in samples if s.get("signals")}
        sensors = {}
        for sample in samples:
            for name in SIGNALS:
                value = sample.get("signals", {}).get(name, 0.)
                if value:
                    key = sample["sensor_node_id"], name
                    sensors[key] = max(value, sensors.get(key, 0.))
        if sensors:
            sensor_series.append({"time_seconds": time_seconds, "signals": [
                [node, name, value] for (node, name), value in sorted(sensors.items())]})
        dns_qps.append([time_seconds, sum(f.get("dns_queries", 0) for f in flows) / 2])
        access = Counter(f["home_access_node"] for f in flows if f["observed_dropped_packets"])
        server = Counter(f["server_node"] for f in flows if f["observed_dropped_packets"])
        current, current_losses, onsets = {}, {}, []
        for f in flows:
            client, service = f["client_node"], f["traffic_kind"]
            route = f["route"]
            edges = [tuple(sorted(e)) for e in zip(route, route[1:])]
            latency = f["one_way_latency_ms"]
            delay = latency > baseline_latency.setdefault(client, latency) + .1
            loss = bool(f["observed_dropped_packets"])
            flags = {name for name, present in {
                "own_loss": loss, "delay_above_t0_0.1ms": delay, "own_loss_or_delay": loss or delay,
                "other_access_flow_loss": access[f["home_access_node"]] > int(loss),
                "other_server_flow_loss": server[f["server_node"]] > int(loss),
                "route_rise_10pp": any(util[e] - baseline_util.get(e, 0.) > 10 for e in edges),
                "sensor_on_route": bool(sensor_nodes.intersection(route)),
            }.items() if present}
            for (node, name), value in sensors.items():
                if node not in route or (name == "dns_query_pressure_ratio" and
                                         (service != "dns" or f["server_node"] != node)):
                    continue
                flags.add(name)
                if 0 < previous_sensors.get((node, name), 0) < value:
                    flags.add("rising_" + name)
            old_loss = previous_losses.get(client)
            if old_loss is not None and not old_loss[0] and f["packet_count"] > 0:
                tally(first_loss, loss, old_loss[1])
            if f["packet_count"] > 0:
                current_losses[client] = loss, flags
            for name, metric in f["slo_evaluation"]["metrics"].items():
                if not metric["required"] or metric["compliant"] is None:
                    continue
                key = service, client, name
                passed = metric["compliant"]
                old = previous.get(key)
                local_flags = flags.copy()
                margin = None
                if metric["comparison"] in ("lte", "gte"):
                    sign = 1 if metric["comparison"] == "gte" else -1
                    margin = sign * (metric["observed_value"] - metric["effective_budget_value"])
                    if passed and old is not None and old[2] is not None and 2 * margin - old[2] < -1e-9:
                        local_flags.add("direct_margin_trend")
                        if loss or delay:
                            local_flags.add("distressed_margin_trend")
                if old is not None and old[0]:
                    tally(counts, not passed, old[1])
                    tally(by_criterion[name], not passed, old[1])
                    if not passed:
                        onsets.append(dict(service=service, flow_id=client, criterion=name))
                current[key] = passed, local_flags, margin
        service_views.append(dict(time_seconds=time_seconds, onsets=onsets, measurement_gaps=previous.keys()-current.keys()))
        previous, previous_losses, previous_sensors = current, current_losses, sensors
    return {"scenario": scenario, "seed": seed, "step_seconds": 2, "attack_start_step": 70,
            "slo_transitions": counts, "by_criterion": by_criterion,
            "first_observed_packet_loss_transitions": first_loss,
            "ordinary_dns_queries_per_second": dns_qps, "observed_route_sensor_series": sensor_series,
            "route_stress_evaluation": evaluate_route_stress(route_views, service_views)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", action="store_true",
                        help="mitre-demo/83 с изменённым спросом; по умолчанию predictive-demo/31")
    args = parser.parse_args()
    scenario, seed = ("mitre-demo", 83) if args.holdout else ("predictive-demo", 31)
    destination = ROOT / "output" / "flow-review" / f"observability-seed{seed}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    demand = patch("src.gnet9.packet_simulator.poisson_arrivals", varying_demand) if args.holdout else nullcontext()
    with demand:
        result = audit(scenario, seed)
    result["vary_demand"] = args.holdout
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    print(destination)
