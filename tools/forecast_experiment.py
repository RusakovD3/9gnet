"""Повторить численный опыт рецензии с выбранной версией исходного кода."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np


def varying_demand(expected_count, *, node_id, step):
    """Отложенный профиль обычного спроса, независимый от расписания воздействий."""
    from src.gnet9.normal_traffic import poisson_arrivals

    factor = 1 + .12 * np.sin(2 * np.pi * step / 32)
    return poisson_arrivals(factor * expected_count, node_id=node_id, step=step, seed=391)


METRICS = (
    "L2.cpu_load_percent.max", "EDGE.utilization.mean",
    "OBS.gold_sla_compliance_ratio", "OBS.normal_arrival_rate_pps",
)


def replay_signature(snapshots):
    from src.gnet9.attack_signatures import match_attack_signature

    previous = set()
    replay = []
    for snapshot in snapshots:
        current = set()
        pairs = []
        for row in snapshot["attacks"].get("precursors", []):
            sensor, signals = row["sensor_node_id"], row["signals"]
            if signals:
                current.add(sensor)
                match = match_attack_signature(signals)
                if sensor in previous and match:
                    pairs.append({"target_id": sensor, "kind": match["kind"]})
        step = snapshot["step_index"]
        replay.append({**snapshot, "koopman": {
            "forecast_is_early_warning": bool(pairs), "forecast_entity_records": pairs,
            "forecast_window_start_step": step + 1, "forecast_window_end_step": step + 10,
        }})
        previous = current
    return replay


def compare_at_alarm_budget(views, calibration):
    """Задержать выдачу до N последовательных прогнозов того же нарушения."""
    from src.gnet9.service_forecast import METHODS, _score_window

    if calibration is None:
        # Число подтверждений подбирается только в выделенном полном прогоне.
        candidates = {method: [1, 2, 3, 5, 10, None] for method in METHODS}
    else:
        candidates = {m: [n] for m, n in calibration["confirmation_samples"].items()}
    selected, scores = {}, {}
    for method, options in candidates.items():
        best = None
        for required in options:
            filtered, previous = [], {}
            for view in views:
                warnings, current = [], {}
                for w in view["warnings"]:
                    if w["method"] != method:
                        continue
                    key = w["service"], w["flow_id"], w["criterion"]
                    current[key] = previous.get(key, 0) + 1
                    if required is not None and current[key] >= required:
                        warnings.append(w)
                filtered.append({**view, "warnings": warnings})
                previous = current
            row, _ = _score_window(filtered, method, 20)
            if calibration is None and row["false_alarms"] > 0:
                continue
            quality = (row["hits"], sum(row["lead_seconds"]))
            if best is None or quality > best:
                best = quality
                selected[method] = required
                scores[method] = {k: v for k, v in row.items() if k != "by_service"}
    return {"validation_budget_false_alarms_per_hour": 0, "confirmation_samples": selected,
            "selection_run": "predictive-demo/11", "methods": scores}


def rerun(args):
    sys.path.insert(0, str(args.source.resolve()))
    from src.gnet9 import koopman
    from src.gnet9.dynamics import DynamicsConfig, simulate_stationary_dynamics, _koopman_evaluation
    from src.gnet9.topology_builder import GNetBaselineBuilder

    config = DynamicsConfig(step_seconds=2, step_count=160, attack_start_step=70,
                            attack_seed=args.seed, attack_scenario=args.scenario,
                            snapshot_detail="summary", packet_detail="summary")
    original = koopman._multi_horizon_forecasts

    def capture(**kwargs):
        # Исходная версия возвращает все векторы вызывающей функции, но пишет только первый.
        rows, vectors = original(**kwargs)
        for row, vector in zip(rows, vectors):
            row["predicted_vector"] = vector.tolist()
        return rows, vectors

    context = patch.object(koopman, "_multi_horizon_forecasts", capture) if args.legacy else nullcontext()
    if args.disable_injection:
        context = patch.object(koopman, "_inject_precursor_forecast",
                               lambda names, prediction, state, **kwargs: prediction)
    if args.disable_operator:
        def hold_state(**kwargs):
            # Абляция канала риска. Проекция равна текущему состоянию, ошибки
            # обученной матрицы выключены. Тренд, геометрия и сигнатура работают.
            kwargs.update(operator=np.eye(len(kwargs["metric_names"])), residual=0., operator_drift=0.)
            return original(**kwargs)
        context = patch.object(koopman, "_multi_horizon_forecasts", hold_state)
    from src.gnet9 import packet_simulator
    demand = (patch.object(packet_simulator, "poisson_arrivals", varying_demand)
              if args.vary_demand else nullcontext())
    with context, demand:
        run = simulate_stationary_dynamics(GNetBaselineBuilder().build(), config)
    snapshots = run["snapshots"]
    names = snapshots[0]["state_vector"]["metric_names"]
    actual = np.array([s["state_vector"]["vector"] for s in snapshots])
    history = []
    errors = {}
    nominal_mean = []
    accepted_values = []
    for s, measurement in zip(snapshots, actual):
        if s["koopman"]["nominal_training"]["observation_eligible"]:
            accepted_values.append(measurement)
        nominal_mean.append(np.mean(accepted_values, axis=0) if accepted_values else measurement)
    for horizon in (1, 5, 10):
        forecasts = np.array([s["koopman"]["horizon_forecasts"][horizon-1]["predicted_vector"]
                              for s in snapshots[:-horizon]])
        target = actual[horizon:]
        previous = actual[:-horizon]
        prior = np.vstack([previous[:1], previous[:-1]])
        methods = {"model": forecasts, "persistence": previous, "trend": previous + horizon * (previous-prior),
                   "nominal_mean": np.array(nominal_mean[:-horizon])}
        if not args.legacy:
            methods["diagonal_without_support_gate_or_projection"] = np.array([
                s["koopman"]["horizon_forecasts"][horizon-1]["dmd_raw_vector"] for s in snapshots[:-horizon]])
        errors[horizon * 2] = {metric: {method: float(np.abs(values[:, i] - target[:, i]).mean())
                                         for method, values in methods.items()}
                               for metric in METRICS for i in [names.index(metric)]}
        if horizon == 1:
            for i, metric in enumerate(METRICS):
                j = names.index(metric)
                history.append({"metric": metric, "min": float(forecasts[:, j].min()),
                                "max": float(forecasts[:, j].max()),
                                "below_zero": int((forecasts[:, j] < 0).sum()),
                                "above_upper": int((forecasts[:, j] > (100, np.inf, 1, np.inf)[i]).sum()),
                                "forecast_at_146s": float(forecasts[72, j]),
                                "actual_at_146s": float(target[72, j])})
    clean = run["training_summary"]["first_precursor_step"]
    accepted = [snapshots[i-1]["koopman"]["kalman_filter"]["filtered_normalized_vector"]
                for i in range(1, clean) if snapshots[i]["koopman"]["nominal_training"]["transition_accepted"]]
    x = np.array(accepted).reshape(-1, len(names))
    singular = np.linalg.svd(x, compute_uv=False)
    signature = _koopman_evaluation(replay_signature(snapshots), step_seconds=2, required_lead_seconds=10)
    output = {"config": config.to_dict(), "source": str(args.source), "numpy_version": np.__version__,
              "vary_demand": args.vary_demand,
              "measurements_sha256": hashlib.sha256(actual.astype("<f8").tobytes()).hexdigest(),
              "mae": errors, "one_step": history, "accepted_shape": x.shape,
              "svd_abs_tolerance": 1e-7, "singular_values": singular.tolist(),
              "effective_rank": int(np.count_nonzero(singular > 1e-7)),
              "varying_coordinates": [name for name, span in zip(names, np.ptp(x, axis=0)) if span > 1e-7] if len(x) else [],
              "synthetic_rank": snapshots[0]["koopman"]["training_reference"]["training_rank"],
              "attack_signature": signature, "attack_full": run["koopman_evaluation"]}
    if not args.legacy:
        from collections import Counter
        output["injection_enabled"] = not args.disable_injection
        output["risk_operator_enabled"] = not args.disable_operator
        output["numeric_origin_counts"] = {
            m: dict(Counter(s["koopman"]["numeric_forecast_origin"][m] for s in snapshots[:-1]))
            for m in names}
        output["projection_count"] = sum(row["physical_projection"]["corrected_metric_count"]
                                          for s in snapshots for row in s["koopman"]["horizon_forecasts"])
        output["lyapunov_changed_response_steps"] = [
            s["step_index"] for s in snapshots if s["arbitrator"]["trigger_card"]["response_stage"]
            != s["arbitrator"]["analysis"]["response_stage_without_lyapunov"]]
        channels = [s["koopman"]["tensor_matrix_koopman"] for s in snapshots[1:]]
        output["tensor_mean_step_rmse"] = {k: float(np.mean([v[k] for v in channels]))
            for k in ("one_step_residual", "residual_without_level_coupling")}
        output["healthy_innovations"] = {}
        for kind in ("vertices", "edges"):
            values = np.concatenate([s["koopman"]["kalman_filter"]["local_observation"][kind]
                                     ["innovation_chi_square_by_entity"] for s in snapshots[1:clean]])
            output["healthy_innovations"][kind] = {"count": len(values), "p99": float(np.quantile(values, .99)),
                                                  "fraction_over_24_7": float(np.mean(values > 24.7))}
    for key in ("service_sla_evaluation", "service_sla_runtime"):
        if key in run:
            output[key] = run[key]
    if args.fit_alarm_budget or args.calibration:
        calibration = None if args.fit_alarm_budget else json.loads(
            args.calibration.read_text(encoding="utf8"))["sla_alarm_budget"]
        output["sla_alarm_budget"] = compare_at_alarm_budget(
            [s["service_sla_forecast"] for s in snapshots], calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    print(args.output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--legacy", action="store_true", help="извлечь исходные векторы до экспорта")
    variant.add_argument("--disable-injection", action="store_true", help="выключить подстановку из предвестника")
    variant.add_argument("--disable-operator", action="store_true", help="убрать линейную динамику из оценки угрозы")
    parser.add_argument("--vary-demand", action="store_true", help="периодический обычный спрос и независимый пуассоновский ряд")
    parser.add_argument("--scenario", choices=("predictive-demo", "mitre-demo"), default="predictive-demo")
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument("--fit-alarm-budget", action="store_true")
    calibration.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    if args.fit_alarm_budget and (args.seed != 11 or args.scenario != "predictive-demo" or args.vary_demand):
        parser.error("Выбор числа подтверждений предусмотрен только на калибровочном прогоне predictive-demo/11")
    rerun(args)
