"""Повторить опыт 0/2928 в исходном режиме прогнозиста SLA и разобрать предвестники."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARCHIVE_HASH = "f90f139a4a8ec5a84e511358f1fc025919c9a5cdfc96f3374d43d91f15b4c270"
PERIOD = 2
HORIZON = 20
# Замороженные подтверждения из отдельной прежней серии predictive-demo/11.
REPEATS = {"koopman": 2, "trend": 10, "persistence": 1}
CHANNELS = ("latency_ms", "loss_percent", "arrival_pps")


def capture(seed, directory, legacy):
    from src.gnet9 import dynamics
    from src.gnet9.topology_builder import GNetBaselineBuilder

    peers = defaultdict(dict)
    coefficients = defaultdict(dict)
    eligible = []
    models = []

    class Recorder(legacy.ServiceSLAForecaster):
        def __init__(self, step_seconds, **kwargs):
            super().__init__(step_seconds, retain_trace=True, delivery_model=False, route_delay=False)
            models.append(self)

        def observe(self, flows, *, time_seconds, nominal=True):
            eligible.append(nominal)
            for group, (num, den, _) in self.fit.items():
                if den > 1e-18:
                    coefficients[group][time_seconds] = max(-1., min(1., num / den))
            for flow in flows:
                if flow.get("security_excluded_from_sla_accounting") or not flow.get("slo_evaluation"):
                    continue
                key = flow["traffic_kind"], flow.get("client_node") or flow["flow_id"]
                packets = flow["packet_count"]
                loss = 100 * flow["observed_dropped_packets"] / packets if packets else None
                peers[key][time_seconds] = [flow["one_way_latency_ms"], loss, packets / PERIOD]
            return super().observe(flows, time_seconds=time_seconds, nominal=nominal)

    config = dynamics.DynamicsConfig(step_seconds=PERIOD, step_count=160, attack_start_step=70,
        attack_seed=seed, attack_scenario="predictive-demo", snapshot_detail="summary", packet_detail="summary")
    with patch.object(dynamics, "ServiceSLAForecaster", Recorder), patch.object(
            dynamics, "evaluate_service_forecasts", legacy.evaluate_service_forecasts):
        run = dynamics.simulate_stationary_dynamics(GNetBaselineBuilder().build(), config)
    views = []
    for snap in run["snapshots"]:
        view = snap["service_sla_forecast"]
        views.append({**{k: view[k] for k in ("time_seconds", "step_seconds", "warnings", "onsets",
                                             "measurement_gaps", "left_censored_impairments")}, "services": {
            name: {"assessed_flows": row["assessed_flows"]} for name, row in view["services"].items()}})
    trace = {"config": config.to_dict(), "eligible": eligible, "views": views,
             "predictor": {"delivery_model": False, "route_delay": False,
                           "source_sha256": hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest()},
             "measurements": [[*key, list(data)] for key, data in models[0].trace.items()],
             "peers": [[*key, data] for key, data in peers.items()],
             "coefficients": [[*key, data] for key, data in coefficients.items()]}
    path = directory / f"legacy-measurements-{seed}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=3) as file:
        json.dump(trace, file, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    print(f"Сохранены измерения seed={seed}: {path}", flush=True)


def identity(item):
    return item["service"], item["flow_id"], item["criterion"]


def confirmed(views, repeats):
    result, previous = [], {}
    for view in views:
        current, warnings = {}, []
        for record in view["warnings"]:
            key = record["method"], *identity(record)
            current[key] = previous.get(key, 0) + 1
            if current[key] >= repeats[record["method"]]:
                warnings.append(record)
        result.append({**view, "warnings": warnings})
        previous = current
    return result


def score(views, legacy):
    """Воспроизвести назначения архивной проверки, сверить итоги с её реализацией."""
    methods, matched = {}, {}
    for method in legacy.METHODS:
        pending, hits = {}, set()
        issued = 0
        for view in views:
            t = view["time_seconds"]
            for key in view["measurement_gaps"]:
                pending.pop(tuple(key), None)
            for onset in view["onsets"]:
                key = identity(onset)
                if pending.pop(key, None) is not None:
                    hits.add((*key, t))
            pending = {k: issue for k, issue in pending.items() if issue + HORIZON > t}
            for warning in view["warnings"]:
                if warning["method"] == method and warning["horizon_seconds"] <= HORIZON:
                    pending.setdefault(identity(warning), t)
                    issued += 1
        totals, _ = legacy._score_window(views, method, HORIZON)
        if len(hits) != totals["hits"]:
            raise AssertionError(f"Число сопоставленных начал {method} расходится с архивной проверкой")
        methods[method] = {key: totals[key] for key in
                           ("hits", "onsets", "false_alarms", "censored", "observed_seconds", "warning_seconds")}
        methods[method]["issued_samples"] = issued
        matched[method] = hits
    return methods, matched


def prewindow(measured, end, *, continuous_pass=False):
    """Последовательный хвост наблюдений до end включительно (2–20 с до события)."""
    result = []
    for index in range(HORIZON // PERIOD):
        time = end - index * PERIOD
        if time not in measured:
            break
        value, budget, margin = measured[time]
        # Наблюдение в архиве округлено до 6 десятичных знаков.
        if continuous_pass and margin < -1e-6:
            break
        result.append((time, value, budget, margin))
    return result[::-1]


def excursion(values):
    if len(values) < 2:
        return None
    return max(values) - min(values)


def greatest_decline(values):
    """Худшее падение запаса между любыми двумя последовательными во времени отсчётами."""
    if len(values) < 2:
        return None
    peak = values[0]
    decline = 0.
    for value in values[1:]:
        decline = max(decline, peak - value)
        peak = max(peak, value)
    return decline


def onset_row(item, trace, peers, coefficients, calm_ends, raw_hits, confirmed_hits):
    service, flow, criterion = key = identity(item)
    onset = item["time_seconds"]
    own = trace[key]
    window = prewindow(own, onset - PERIOD, continuous_pass=True)
    if not window:
        raise AssertionError("Ожидалось исправное измерение непосредственно перед началом")
    margins = [r[3] for r in window]
    past = {r[0]: r for r in window}
    # Для фонового сравнения берём предыдущие десять исправных отсчётов
    # того же потока на непрерывном номинальном участке вне окна события.
    control = []
    for end in reversed(calm_ends):
        if end >= onset - HORIZON:
            continue
        possible = prewindow(own, end, continuous_pass=True)
        if len(possible) == HORIZON // PERIOD:
            control = possible
            break
    control_margins = [r[3] for r in control]
    group = service, item["grade"], criterion
    available = [coefficients.get(group, {}).get(r[0]) for r in window]
    change = greatest_decline(margins)
    calm_change = greatest_decline(control_margins)
    onset_margin = own[onset][2]
    row = {
        "key": [service, flow, criterion, onset], "unit": item["unit"],
        "comparison": item["comparison"], "before_samples": len(window),
        "calm_end_seconds": control[-1][0] if control else None,
        "changed_measurement": excursion([r[1] for r in window]) > 1e-6 if len(window) > 1 else None,
        "changed_budget": excursion([r[2] for r in window]) > 1e-6 if len(window) > 1 else None,
        "changed_margin": excursion(margins) > 1e-6 if len(window) > 1 else None,
        "declined_margin": change > 1e-6 if change is not None else None,
        "calm_declined_margin": (calm_change or 0.) > 1e-6 if control else None,
        "decline_above_calm": change > calm_change + 1e-6 if change is not None and control else None,
        "net_decline_t20_to_t2": margins[0] - margins[-1] if len(window) == 10 else None,
        "max_decline_before": change, "max_decline_calm": calm_change,
        "margin_at_t2": margins[-1], "margin_at_onset": onset_margin,
        "decline_at_onset": margins[-1] - onset_margin,
        "identified_before": any(a is not None for a in available),
        "positive_coefficient_before": any(a is not None and a > 0 for a in available),
        "coefficient_at_t2": available[-1],
        "hits": {m: [(*key, onset) in raw_hits[m], (*key, onset) in confirmed_hits[m]] for m in REPEATS},
        "earlier_margin_minus_t2": {lead: past[onset-lead][3] - margins[-1]
                                    for lead in range(2, 21, 2) if onset-lead in past},
        "peers": {},
    }
    observations = peers[(service, flow)]
    for i, channel in enumerate(CHANNELS):
        def variation(times):
            values = [observations[r[0]][i] for r in times if r[0] in observations]
            return excursion(values) if len(values) == len(times) and None not in values else None
        before, calm = variation(window), variation(control)
        row["peers"][channel] = {"range": before, "calm_range": calm,
                                 "above_calm": before > calm + 1e-6 if None not in (before, calm) else None}
    return row


def summarize(rows):
    def dist(values):
        values = [v for v in values if v is not None]
        return {"n": len(values), "median": float(np.median(values)), "max": max(values)} if values else {"n": 0}

    def flags(values):
        values = [v for v in values if v is not None]
        return {"yes": sum(values), "n": len(values)}

    return {
        "onsets": len(rows), "complete_10_sample_windows": sum(r["before_samples"] == 10 for r in rows),
        "evidence": {k: flags(r[k] for r in rows) for k in
                     ("changed_measurement", "changed_budget", "changed_margin", "declined_margin",
                      "calm_declined_margin", "decline_above_calm", "identified_before", "positive_coefficient_before")},
        "hits": {m: {"before_repeat": sum(r["hits"][m][0] for r in rows),
                     "after_repeat": sum(r["hits"][m][1] for r in rows),
                     "lost_to_repeat": sum(r["hits"][m] == [True, False] for r in rows)} for m in REPEATS},
        "peers": {c: {"range": dist(r["peers"][c]["range"] for r in rows),
                      "different": flags((v > 1e-6 if v is not None else None)
                                        for v in (r["peers"][c]["range"] for r in rows)),
                      "above_matched_calm": flags(r["peers"][c]["above_calm"] for r in rows)} for c in CHANNELS},
    }


def summarize_criteria(rows):
    by_criterion = {}
    for criterion in sorted({r["key"][2] for r in rows}):
        group = [r for r in rows if r["key"][2] == criterion]
        section = summarize(group)
        section["unit"] = group[0]["unit"]
        for field in ("margin_at_t2", "margin_at_onset", "max_decline_before", "max_decline_calm",
                      "decline_at_onset", "coefficient_at_t2"):
            values = [r[field] for r in group if r[field] is not None]
            section[field] = float(np.median(values)) if values else None
        section["lag_vs_t2"] = {}
        for lead in range(2, 21, 2):
            deltas = [r["earlier_margin_minus_t2"][lead] for r in group if lead in r["earlier_margin_minus_t2"]]
            section["lag_vs_t2"][lead] = {"n": len(deltas), "changed": sum(abs(d) > 1e-6 for d in deltas),
                                          "declined_by_t2": sum(d > 1e-6 for d in deltas)}
        by_criterion[criterion] = section
    return by_criterion


def analyze(seed, directory, legacy):
    path = directory / f"legacy-measurements-{seed}.json.gz"
    with gzip.open(path, encoding="utf-8", mode="rt") as file:
        replay = json.load(file)
    views = replay["views"]
    onsets = [row for v in views for row in v["onsets"]]
    criteria = {identity(row): row["comparison"] for row in onsets}
    measured = {}
    for service, flow, criterion, series in replay["measurements"]:
        key = service, flow, criterion
        if key not in criteria:
            continue
        comparison = criteria[key]
        samples = iter(series)
        measured[key] = {int(t): (v, b, b-v if comparison == "lte" else v-b if comparison == "gte"
                                  else float(v == b)-1) for t, v, b in zip(samples, samples, samples)}
    peers = {(s, f): {int(t): v for t, v in series.items()} for s, f, series in replay["peers"]}
    coefficients = {(s, g, c): {int(t): a for t, a in series.items()} for s, g, c, series in replay["coefficients"]}
    healthy = replay["eligible"]
    calm_ends = [i * PERIOD for i in range(9, len(healthy)) if all(healthy[i-9:i+1])]
    raw, raw_hits = score(views, legacy)
    strict, strict_hits = score(confirmed(views, REPEATS), legacy)
    rows = [onset_row(r, measured, peers, coefficients, calm_ends, raw_hits, strict_hits) for r in onsets]
    report = {
        "seed": seed, "reference_archive_sha256": ARCHIVE_HASH,
        "predictor": replay.get("predictor", {"source_sha256": ARCHIVE_HASH, "source": "archived_module"}),
        "config": replay["config"], "repeats": REPEATS,
        "protocol_ru": "Единица разбора — начало нарушения одного критерия одного потока. "
                       "Окно 2–20 с до начала ограничено непрерывными исправными измерениями критерия. "
                       "Изменение считается при разнице больше 0,000001 в единицах критерия. "
                       "Фоновое окно — ближайшие более ранние 10 номинальных отсчётов этого потока, "
                       "целиком до окна начала. Фоновые окна могут использоваться повторно. "
                       "Соседние признаки: диапазон односторонней задержки в мс, наблюдаемой доли "
                       "потерь в процентах, поступлений в пакетах/с. Разбор описательный.",
        "raw_score": raw, "confirmed_score": strict, "all": summarize(rows), "by_criterion": summarize_criteria(rows),
    }
    output = directory / f"legacy-audit-{seed}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with gzip.open(directory / f"legacy-onsets-{seed}.json.gz", "wt", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    print(f"seed={seed} Купман до/после подтверждения: {raw['koopman']['hits']}/"
          f"{strict['koopman']['hits']} из {len(rows)}", flush=True)


def summarize_seeds(directory):
    rows, runs = [], {}
    for seed in (42, 7, 1729):
        report = json.loads((directory / f"legacy-audit-{seed}.json").read_text(encoding="utf-8"))
        with gzip.open(directory / f"legacy-onsets-{seed}.json.gz", mode="rt", encoding="utf-8") as file:
            seed_rows = json.load(file)
        for row in seed_rows:
            row["earlier_margin_minus_t2"] = {int(k): v for k, v in row["earlier_margin_minus_t2"].items()}
        rows.extend(seed_rows)
        runs[seed] = {k: report[k] for k in ("raw_score", "confirmed_score", "all")}
    result = {"reference_archive_sha256": ARCHIVE_HASH, "seeds": runs, "all": summarize(rows),
              "by_criterion": summarize_criteria(rows), "protocol_ru": report["protocol_ru"],
              "repeats": REPEATS}
    (directory / "legacy-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 7, 1729))
    parser.add_argument("--directory", type=Path, default=ROOT / "output" / "flow-review")
    parser.add_argument("--stage", choices=("capture", "analyze", "both", "summarize"), default="both")
    args = parser.parse_args()
    from src.gnet9 import service_forecast as legacy
    if args.stage == "summarize":
        summarize_seeds(args.directory)
        return
    if args.seed is None:
        parser.error("--seed нужен для выбранного этапа")
    if args.stage != "analyze":
        args.directory.mkdir(parents=True, exist_ok=True)
        capture(args.seed, args.directory, legacy)
    if args.stage != "capture":
        analyze(args.seed, args.directory, legacy)


if __name__ == "__main__":
    main()
