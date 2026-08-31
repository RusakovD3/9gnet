"""Гибридный прогноз GNet9: Koopman/DMD для нормы и анализ предвестников.

Оператор нормальной динамики обучается на малой детерминированной окрестности
идеального `t0`. Он измеряет отклонение и одношаговую невязку. Тип угрозы
определяется отдельно только по уже наблюдаемой телеметрии NetFlow/IDS/UPS —
идеальное состояние само по себе не содержит примеров будущих атак.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any

import numpy as np

from .attacks import MITRE_T1110_001_ATTEMPTS_PER_SECOND
from .arbitrator import evaluate_defense_game
from .metrics import gold_threat_proximity_view
from .sdn_controller import compile_sdn_intent
from .tensor_matrix import TensorKroneckerKoopman, TensorMatrixView


THREAT_METRICS = {
    "OBS.attack_active_count",
    "OBS.attack_intensity_ratio",
    "OBS.attack_rate_mbps",
    "OBS.legitimate_loss_ratio",
    "OBS.legitimate_delivery_ratio",
    "OBS.impacted_gold_flow_count",
    "OBS.gold_sla_compliance_ratio",
    "OBS.maximum_target_utilization_percent",
    "OBS.precursor_count",
    "OBS.precursor_confidence",
    "OBS.suspicious_source_count",
    "OBS.source_entropy_ratio",
    "OBS.scan_rate_pps",
    "OBS.traffic_acceleration_ratio",
    "OBS.syn_backlog_growth_ratio",
    "OBS.power_control_anomaly_ratio",
    "OBS.power_voltage_sag_ratio",
    "OBS.battery_discharge_rate_ratio",
    "OBS.authentication_failure_rate_per_second",
    "OBS.authentication_failure_ratio",
    "OBS.account_lockout_pressure",
    "OBS.state_hausdorff_distance",
    "OBS.state_hausdorff_normalized",
    "OBS.coordinate_hausdorff_distance",
    "OBS.target_concentration_ratio",
    "OBS.rerouted_gold_flow_count",
    "OBS.rerouted_silver_flow_count",
    "OBS.rerouted_bronze_flow_count",
    "OBS.isolated_flow_count",
    "OBS.service_failover_count",
    "OBS.route_change_ratio",
    "OBS.maximum_post_remap_utilization_percent",
    "OBS.gold_delivery_ratio",
    "OBS.silver_delivery_ratio",
    "OBS.bronze_delivery_ratio",
}

# Нормированная скорость роста V(t), при которой обратная связь Ляпунова
# считается полностью активной. Это явный инженерный масштаб контроллера,
# а не универсальная граница устойчивости для реального оператора связи.
LYAPUNOV_REMAP_RESPONSE_SCALE_PER_SECOND = 0.025

# Горизонт — верхняя граница причинного прогноза, а не обещание безусловно
# распознать любую атаку за 20 секунд. Для штатного такта 2 с анализатор
# строит десять последовательных rollout-состояний. Более дальняя экстраполяция
# для короткого лабораторного ряда была бы заметно менее надёжной.
KOOPMAN_MAXIMUM_FORECAST_SECONDS = 20
# Окно локального тренда короче горизонта: оно достаточно для подавления
# единичного выброса, но не смешивает несколько соседних эпизодов воздействия.
KOOPMAN_TREND_WINDOW_SECONDS = 12
# Порог относится к ненормированному risk-score стенда. Сам по себе он не
# создаёт тревогу: дополнительно нужны распознанный класс, локализованный
# сенсор и два последовательных причинных отсчёта.
KOOPMAN_WARNING_RISK_THRESHOLD = 0.30


def forecast_horizon_steps_for_sample_period(step_seconds: int) -> int:
    """Вернуть целое число шагов без скрытого превышения 20-секундного лимита.

    При периоде дискретизации не более лимита горизонт округляется вниз до
    полного числа отсчётов. Если один отсчёт уже длиннее лимита, единственная
    честная дискретная возможность — прогноз ровно на один укрупнённый шаг.
    """

    sample_period_seconds = max(1, int(step_seconds))
    if sample_period_seconds > KOOPMAN_MAXIMUM_FORECAST_SECONDS:
        return 1
    return max(1, KOOPMAN_MAXIMUM_FORECAST_SECONDS // sample_period_seconds)


@dataclass
class KoopmanOnlineAnalyzer:
    """Online Koopman/DMD state trained around ideal t0."""

    metric_names: list[str]
    reference_vector: np.ndarray
    scale_vector: np.ndarray
    ideal_operator: np.ndarray
    observed_operator: np.ndarray
    training_residual: float
    step_seconds: int
    reference_context: dict[str, Any]
    online_gram: np.ndarray
    online_cross: np.ndarray
    lyapunov_matrix: np.ndarray
    lyapunov_solver_iterations: int
    lyapunov_solver_residual: float
    state_history: list[np.ndarray] = field(default_factory=list)
    sensor_streaks: dict[str, int] = field(default_factory=dict)
    delay_embedding_depth: int = 4
    maximum_forecast_horizon_steps: int = 5
    required_prediction_lead_seconds: int = 10
    warning_risk_threshold: float = KOOPMAN_WARNING_RISK_THRESHOLD
    warning_threshold_origin: str = "model_default_not_externally_validated"
    previous_vector: np.ndarray | None = None
    previous_analysis_operator_revision: int | None = None
    analysis_revision: int = 0
    operator_revision: int = 0
    nominal_observation_count: int = 0
    last_observed_threat_step: int | None = None
    tensor_koopman: TensorKroneckerKoopman | None = None

    @classmethod
    def from_t0(
        cls,
        model,
        state_vector: dict[str, Any],
        *,
        tensor_matrices: TensorMatrixView | None = None,
        step_seconds: int,
        required_prediction_lead_seconds: int = 10,
        warning_risk_threshold: float | None = None,
        warning_threshold_origin: str = "model_default_not_externally_validated",
    ) -> "KoopmanOnlineAnalyzer":
        metric_names = list(state_vector["metric_names"])
        reference = np.asarray(state_vector["vector"], dtype=float)
        scale = _scale_vector(metric_names, reference)
        training = _ideal_training_vectors(metric_names, reference, scale)
        normalized_training = (training - reference) / scale
        x = normalized_training[:-1].T
        y = normalized_training[1:].T
        online_gram = x @ x.T
        online_cross = y @ x.T
        operator = _operator_from_covariances(online_gram, online_cross)
        operator = _stabilize_threat_dimensions(operator, metric_names)
        operator = _limit_spectral_radius(operator)
        residual = _training_residual(operator, normalized_training)
        lyapunov_matrix, lyapunov_iterations, lyapunov_residual = _discrete_lyapunov_matrix(operator)
        return cls(
            metric_names=metric_names,
            reference_vector=reference,
            scale_vector=scale,
            ideal_operator=operator,
            observed_operator=operator.copy(),
            training_residual=residual,
            step_seconds=step_seconds,
            reference_context=_reference_context(
                model,
                metric_names=metric_names,
                reference=reference,
                operator=operator,
                training_sample_count=int(training.shape[0]),
                training_rank=int(np.linalg.matrix_rank(normalized_training)),
            ),
            online_gram=online_gram,
            online_cross=online_cross,
            lyapunov_matrix=lyapunov_matrix,
            lyapunov_solver_iterations=lyapunov_iterations,
            lyapunov_solver_residual=lyapunov_residual,
            delay_embedding_depth=max(
                2,
                math.ceil(KOOPMAN_TREND_WINDOW_SECONDS / max(step_seconds, 1)),
            ),
            maximum_forecast_horizon_steps=max(
                1,
                forecast_horizon_steps_for_sample_period(step_seconds),
            ),
            required_prediction_lead_seconds=max(
                1,
                int(required_prediction_lead_seconds),
            ),
            warning_risk_threshold=(
                KOOPMAN_WARNING_RISK_THRESHOLD
                if warning_risk_threshold is None
                else min(1.0, max(0.0, float(warning_risk_threshold)))
            ),
            warning_threshold_origin=warning_threshold_origin,
            tensor_koopman=(
                TensorKroneckerKoopman.from_reference(tensor_matrices)
                if tensor_matrices is not None
                else None
            ),
        )

    def analyze_step(
        self,
        model,
        state_vector: dict[str, Any],
        *,
        tensor_matrices: TensorMatrixView | None = None,
        attack_state: dict[str, Any],
        step_index: int,
        time_seconds: int,
    ) -> dict[str, Any]:
        self.analysis_revision += 1
        # Текущий снимок является атомарным: residual, rollout, спектр и
        # Ляпунов используют одну и ту же пару K/P. Возможное online-обновление
        # выполняется лишь после анализа и становится рабочим со следующего шага.
        operator_revision_used = self.operator_revision
        operator_used = self.observed_operator.copy()
        lyapunov_matrix_used = self.lyapunov_matrix.copy()
        lyapunov_solver_iterations_used = self.lyapunov_solver_iterations
        lyapunov_solver_residual_used = self.lyapunov_solver_residual
        nominal_observation_count_used = self.nominal_observation_count
        operator_fingerprint_used = _matrix_fingerprint(operator_used)
        lyapunov_fingerprint_used = _matrix_fingerprint(lyapunov_matrix_used)
        current = np.asarray(state_vector["vector"], dtype=float)
        tensor_matrix_koopman = (
            self.tensor_koopman.analyze(tensor_matrices)
            if self.tensor_koopman is not None and tensor_matrices is not None
            else {
                "model": "factorised_kronecker_tensor_koopman",
                "available": False,
                "semantics_ru": "Матрицы тензоров не переданы в этот вызов анализатора.",
            }
        )
        z_current = self._normalize(current)
        residual = 0.0
        transition_residual_vector = np.zeros_like(z_current)
        if self.previous_vector is not None:
            z_previous = self._normalize(self.previous_vector)
            predicted_current = operator_used @ z_previous
            transition_residual_vector = z_current - predicted_current
            residual = _rms(transition_residual_vector)

        deviation = _rms(z_current)
        operator_drift = _operator_drift(operator_used, self.ideal_operator)
        active_attack_score = _attack_score(self.metric_names, current)
        precursor_score = _precursor_score(self.metric_names, current)
        threat_score = max(active_attack_score, precursor_score)
        observable_precursor_now = _observable_precursor_present(
            attack_state,
            precursor_score,
        )
        observable_attack_now = bool(attack_state.get("events", []))
        if observable_precursor_now or observable_attack_now:
            self.last_observed_threat_step = step_index
        threat_proximity = gold_threat_proximity_view(
            model,
            _sanitized_geometry_observation(attack_state),
        )
        route_hausdorff = attack_state.get("routing", {}).get(
            "gold_route_hausdorff",
            {
                "metric": "maximum_per_flow_gold_route_hausdorff_latency",
                "distance_ms": 0.0,
                "normalized_distance": 0.0,
                "changed_gold_flow_count": 0,
                "gold_flow_count": 0,
            },
        )

        # В прогноз попадают лишь текущий вектор и его прошлая телеметрия. Ни шаг
        # начала атаки, ни её скрытый идентификатор здесь не используются.
        delay_history = [*self.state_history, z_current.copy()][-self.delay_embedding_depth :]
        observed_trend = _delay_trend(delay_history, self.metric_names)
        horizon_forecasts, predicted_vectors = _multi_horizon_forecasts(
            metric_names=self.metric_names,
            reference_vector=self.reference_vector,
            scale_vector=self.scale_vector,
            operator=operator_used,
            z_current=z_current,
            observed_trend=observed_trend,
            attack_state=attack_state,
            precursor_score=precursor_score,
            active_attack_score=active_attack_score,
            residual=residual,
            operator_drift=operator_drift,
            threat_proximity_risk=float(threat_proximity["proximity_risk"]),
            step_index=step_index,
            time_seconds=time_seconds,
            step_seconds=self.step_seconds,
            maximum_horizon_steps=self.maximum_forecast_horizon_steps,
            warning_risk_threshold=self.warning_risk_threshold,
        )
        first_forecast = horizon_forecasts[0]
        predicted_next = predicted_vectors[0]
        predicted_reference_distance = float(first_forecast["predicted_reference_distance"])
        predicted_attack_score = float(first_forecast["predicted_attack_score"])
        forecast_risk = max(float(item["risk_score"]) for item in horizon_forecasts)
        slo_horizon_steps = min(
            self.maximum_forecast_horizon_steps,
            max(
                1,
                math.ceil(
                    self.required_prediction_lead_seconds / max(self.step_seconds, 1)
                ),
            ),
        )
        slo_forecast = horizon_forecasts[slo_horizon_steps - 1]
        component_ablation = {
            "koopman_dmd_only_max_risk": max(
                float(item["koopman_dmd_only_risk_score"])
                for item in horizon_forecasts
            ),
            "koopman_plus_delay_trend_max_risk": max(
                float(item["koopman_plus_delay_trend_risk_score"])
                for item in horizon_forecasts
            ),
            "hybrid_with_precursor_classifier_max_risk": forecast_risk,
            "semantics": (
                "diagnostic_ablation_on_same_observation_not_independent_validation"
            ),
        }

        # Для dV/dt оба соседних состояния всегда оцениваются одной формой
        # P_used. Если предыдущий шаг использовал иную ревизию, энергия его
        # состояния пересчитывается сейчас, а не смешивается с прежним базисом.
        comparable_previous_lyapunov = 0.0
        lyapunov_basis_rebased = bool(
            self.previous_vector is not None
            and self.previous_analysis_operator_revision is not None
            and self.previous_analysis_operator_revision != operator_revision_used
        )
        if self.previous_vector is not None:
            z_previous_for_energy = self._normalize(self.previous_vector)
            _, comparable_previous_lyapunov = _lyapunov_state_energy(
                z_previous_for_energy,
                lyapunov_matrix_used,
            )
        lyapunov_raw, lyapunov_value = _lyapunov_state_energy(
            z_current,
            lyapunov_matrix_used,
        )
        lyapunov_delta = lyapunov_value - comparable_previous_lyapunov
        lyapunov_derivative = lyapunov_delta / max(float(self.step_seconds), 1e-9)
        lyapunov_remap_pressure = min(
            1.0,
            max(0.0, lyapunov_derivative) / LYAPUNOV_REMAP_RESPONSE_SCALE_PER_SECOND,
        )
        finite_time_energy_growth = _finite_time_energy_growth_rate(
            comparable_previous_lyapunov,
            lyapunov_value,
            self.step_seconds,
        )
        predicted_attack = _predicted_attack_kind(attack_state, forecast_risk)
        predicted_next_attack = _predicted_precursor_kind(attack_state)
        current_attack = _current_attack_kind(attack_state)
        cooldown_age_steps = (
            None
            if self.last_observed_threat_step is None
            else max(0, step_index - self.last_observed_threat_step)
        )
        post_incident_cooldown = bool(
            not observable_attack_now
            and not observable_precursor_now
            and cooldown_age_steps is not None
            and 0 < cooldown_age_steps <= self.delay_embedding_depth
            and forecast_risk >= 0.25
        )
        if post_incident_cooldown:
            predicted_attack = "post_incident_residual"
        risk_state = (
            "active_attack"
            if observable_attack_now
            else "causal_precursor_forecast"
            if observable_precursor_now
            else "post_incident_cooldown"
            if post_incident_cooldown
            else "unattributed_anomaly"
            if forecast_risk >= 0.25
            else "nominal"
        )
        displayed_alert_level = (
            "RECOVERY_WATCH"
            if post_incident_cooldown
            else _alert_level(forecast_risk)
        )
        forecast_entities = _forecast_entities(attack_state)
        observed_sensors = {
            str(item.get("sensor_node_id"))
            for item in attack_state.get("precursors", [])
            if item.get("sensor_node_id") and isinstance(item.get("signals"), dict)
        }
        self.sensor_streaks = {
            sensor: self.sensor_streaks.get(sensor, 0) + 1
            for sensor in observed_sensors
        }
        required_confirmation_samples = (
            1 if attack_state.get("scenario") == "mitre-demo" else 2
        )
        confirmed_sensors = {
            sensor for sensor, count in self.sensor_streaks.items()
            if count >= required_confirmation_samples
        }
        if confirmed_sensors:
            forecast_entities["next_target_ids"] = sorted(
                set(forecast_entities["next_target_ids"]) & confirmed_sensors
            )
        warning_forecasts = [item for item in horizon_forecasts if bool(item["is_warning"])]
        early_warning = bool(
            _observable_precursor_present(attack_state, precursor_score)
            and bool(confirmed_sensors)
            and predicted_next_attack not in {"none", "unknown_degradation_or_attack"}
            and warning_forecasts
        )
        first_warning = warning_forecasts[0] if early_warning else horizon_forecasts[0]
        last_warning = warning_forecasts[-1] if early_warning else horizon_forecasts[-1]
        forecast_target_step = int(first_warning["target_step"])
        forecast_target_time_seconds = int(first_warning["target_time_seconds"])
        forecast_window_start_step = forecast_target_step
        forecast_window_end_step = int(last_warning["target_step"])
        forecast_window_start_seconds = forecast_target_time_seconds
        forecast_window_end_seconds = int(last_warning["target_time_seconds"])
        earliest_warning_lead_seconds = (
            forecast_window_start_seconds - time_seconds if early_warning else 0
        )
        maximum_warning_lead_seconds = (
            forecast_window_end_seconds - time_seconds if early_warning else 0
        )

        spectral_radius = float(max(abs(np.linalg.eigvals(operator_used)))) if operator_used.size else 0.0
        actual_forecast_horizon_seconds = (
            self.maximum_forecast_horizon_steps * self.step_seconds
        )
        coarse_forecast_step = self.step_seconds > KOOPMAN_MAXIMUM_FORECAST_SECONDS
        protective_pressure = max(0.0, (forecast_risk - 0.20) / 0.80)
        recommendation = _recommendation(
            forecast_risk,
            predicted_attack,
            attack_state=attack_state,
            step_index=step_index,
            step_seconds=self.step_seconds,
            valid_from_step=forecast_window_start_step,
            valid_until_step=forecast_window_end_step,
            defense_authorized=early_warning or bool(attack_state.get("events", [])),
            lyapunov_remap_pressure=lyapunov_remap_pressure,
        )

        # Обучение номинальной динамике коммитится после завершения всех
        # вычислений снимка. Новые K/P не меняют ни одно значение текущего шага.
        update_applied_for_next_step = False
        if self.previous_vector is not None and threat_score < 0.10 and forecast_risk < 0.25:
            z_previous_for_update = self._normalize(self.previous_vector)
            self.nominal_observation_count += 1
            update_applied_for_next_step = self._update_observed_operator(
                z_previous_for_update,
                z_current,
            )
        operator_revision_after_update = self.operator_revision
        operator_fingerprint_after_update = _matrix_fingerprint(self.observed_operator)
        lyapunov_fingerprint_after_update = _matrix_fingerprint(self.lyapunov_matrix)
        spectral_radius_after_update = (
            float(max(abs(np.linalg.eigvals(self.observed_operator))))
            if self.observed_operator.size
            else 0.0
        )

        # Состояние становится предыдущим только после завершения анализа и
        # потенциального коммита модели для следующего шага.
        self.previous_vector = current.copy()
        self.previous_analysis_operator_revision = operator_revision_used
        self.state_history.append(z_current.copy())
        history_limit = max(
            self.delay_embedding_depth + 1,
            self.maximum_forecast_horizon_steps + 1,
        )
        if len(self.state_history) > history_limit:
            del self.state_history[:-history_limit]
        return {
            "model": "causal_koopman_dmd_plus_lag_trend_and_precursor_classifier",
            "training_reference": self.reference_context,
            "training_residual": round(self.training_residual, 8),
            "t0_signature": self.reference_context["t0_signature"],
            "step_index": step_index,
            "analysis_revision": self.analysis_revision,
            "transition_count": max(0, self.analysis_revision - 1),
            "operator_revision_used": operator_revision_used,
            "operator_revision_after_update": operator_revision_after_update,
            "update_applied_for_next_step": update_applied_for_next_step,
            "operator_update_effective_from_step": (
                step_index + 1 if update_applied_for_next_step else None
            ),
            "operator_revision_semantics": (
                "all_current_snapshot_metrics_use_operator_revision_used;"
                "operator_revision_after_update_is_effective_from_next_step"
            ),
            "time_seconds": time_seconds,
            "delay_embedding_depth": self.delay_embedding_depth,
            "history_trend_depth": self.delay_embedding_depth,
            "history_trend_window_seconds": self.delay_embedding_depth * self.step_seconds,
            "delay_history_sample_count": len(delay_history),
            "delay_state_dimension": len(self.metric_names),
            "lagged_feature_count": len(self.metric_names) * self.delay_embedding_depth,
            "uses_augmented_hankel_operator": False,
            "warning_confirmation_samples": required_confirmation_samples,
            "confirmed_sensor_nodes": sorted(confirmed_sensors),
            "maximum_forecast_horizon_steps": self.maximum_forecast_horizon_steps,
            "maximum_forecast_horizon_seconds": actual_forecast_horizon_seconds,
            "prediction_slo_seconds": self.required_prediction_lead_seconds,
            "prediction_slo_forecast_steps": slo_horizon_steps,
            "prediction_slo_forecast_seconds": slo_horizon_steps * self.step_seconds,
            "prediction_slo_forecast_risk_score": float(slo_forecast["risk_score"]),
            "prediction_slo_forecast_target_step": int(slo_forecast["target_step"]),
            "prediction_slo_forecast_target_time_seconds": int(
                slo_forecast["target_time_seconds"]
            ),
            "prediction_slo_semantics_ru": (
                "Целевой минимум времени на защитное решение; аналитический "
                "rollout может быть длиннее и не ухудшается до этого порога."
            ),
            "configured_forecast_horizon_seconds": KOOPMAN_MAXIMUM_FORECAST_SECONDS,
            "forecast_horizon_semantics": "maximum_causal_rollout_not_guaranteed_attack_lead_time",
            "forecast_horizon_discretization": (
                "single_coarse_step_exceeds_configured_maximum"
                if coarse_forecast_step
                else "whole_steps_floored_not_to_exceed_configured_maximum"
            ),
            "forecast_horizon_exceeds_configured_maximum": coarse_forecast_step,
            "warning_risk_threshold": self.warning_risk_threshold,
            "warning_threshold_origin": self.warning_threshold_origin,
            "observation_cutoff_seconds": time_seconds,
            "future_schedule_used_by_predictor": False,
            "horizon_forecasts": horizon_forecasts,
            "forecast_component_ablation": component_ablation,
            "tensor_matrix_koopman": tensor_matrix_koopman,
            # Одношаговые поля сохранены как совместимый с прежним JSON интерфейс.
            "forecast_horizon_seconds": self.step_seconds,
            "forecast_horizon_steps": 1,
            "forecast_lead_time_seconds": earliest_warning_lead_seconds,
            "lead_time_seconds": earliest_warning_lead_seconds,
            "earliest_warning_lead_seconds": earliest_warning_lead_seconds,
            "maximum_warning_lead_seconds": maximum_warning_lead_seconds,
            "forecast_is_early_warning": early_warning,
            "forecast_target_step": forecast_target_step,
            "forecast_target_time_seconds": forecast_target_time_seconds,
            "forecast_window_start_step": forecast_window_start_step,
            "forecast_window_end_step": forecast_window_end_step,
            "forecast_window_start_seconds": forecast_window_start_seconds,
            "forecast_window_end_seconds": forecast_window_end_seconds,
            "reference_distance": round(deviation, 6),
            "predicted_reference_distance": round(predicted_reference_distance, 6),
            "one_step_residual": round(residual, 6),
            "operator_drift_norm": round(operator_drift, 6),
            "operator_spectral_radius": round(spectral_radius, 6),
            "operator_spectral_radius_after_update": round(
                spectral_radius_after_update,
                6,
            ),
            "ideal_operator_fingerprint": _matrix_fingerprint(self.ideal_operator),
            # Совместимое поле теперь однозначно относится к K_used.
            "observed_operator_fingerprint": operator_fingerprint_used,
            "operator_fingerprint_used": operator_fingerprint_used,
            "operator_fingerprint_after_update": operator_fingerprint_after_update,
            # Старый флаг сохранён, но его временная семантика теперь явная.
            "online_baseline_update_applied": update_applied_for_next_step,
            "online_baseline_update_semantics": "committed_for_next_step_only",
            "observed_nominal_transition_count": nominal_observation_count_used,
            "observed_nominal_transition_count_after_update": self.nominal_observation_count,
            "online_operator_update_count": operator_revision_used,
            "online_operator_update_count_after_update": operator_revision_after_update,
            "baseline_frozen_due_to_threat": bool(threat_score >= 0.10 or forecast_risk >= 0.25),
            "active_attack_score": round(active_attack_score, 6),
            "precursor_score": round(precursor_score, 6),
            "predicted_attack_score": round(predicted_attack_score, 6),
            "forecast_attack_probability": round(forecast_risk, 6),
            "forecast_risk_score": round(forecast_risk, 6),
            "probability_is_calibrated": False,
            "risk_score_semantics": "uncalibrated_detection_risk_rank",
            "forecast_attack_kind": predicted_attack,
            "forecast_next_attack_kind": predicted_next_attack,
            "current_attack_kind": current_attack,
            "forecast_attack_ids": [] if early_warning else forecast_entities["attack_ids"],
            "current_attack_ids": forecast_entities["attack_ids"],
            "forecast_correlation_ids": forecast_entities["correlation_ids"],
            "forecast_entity_records": (
                forecast_entities["next_entity_records"]
                if early_warning else forecast_entities["current_entity_records"]
            ),
            "current_attack_entity_records": forecast_entities["current_entity_records"],
            "forecast_target_ids": (
                forecast_entities["next_target_ids"] if early_warning else forecast_entities["target_ids"]
            ),
            "current_attack_target_ids": forecast_entities["current_target_ids"],
            "forecast_window": f"{forecast_window_start_seconds}-{forecast_window_end_seconds}s",
            "protective_pressure": round(min(1.0, protective_pressure), 6),
            "alert_level": displayed_alert_level,
            "raw_alert_level_from_risk": _alert_level(forecast_risk),
            "risk_state": risk_state,
            "post_incident_cooldown_active": post_incident_cooldown,
            "post_incident_cooldown_age_seconds": (
                None
                if cooldown_age_steps is None
                else cooldown_age_steps * self.step_seconds
            ),
            "post_incident_cooldown_window_seconds": (
                self.delay_embedding_depth * self.step_seconds
            ),
            "risk_state_semantics": (
                "recovery_watch_preserves_causal_lag_residual_without_claiming_a_new_attack"
            ),
            "lyapunov_value": round(lyapunov_value, 6),
            "lyapunov_raw_value": round(lyapunov_raw, 6),
            "lyapunov_delta": round(lyapunov_delta, 6),
            "lyapunov_derivative_per_second": round(lyapunov_derivative, 6),
            "lyapunov_basis_rebased_after_operator_update": lyapunov_basis_rebased,
            "lyapunov_previous_state_recomputed_on_matrix_used": bool(
                self.analysis_revision > 1
            ),
            "lyapunov_remap_pressure": round(lyapunov_remap_pressure, 6),
            "lyapunov_remap_response_scale_per_second": LYAPUNOV_REMAP_RESPONSE_SCALE_PER_SECOND,
            "finite_time_energy_growth_rate_per_second": round(finite_time_energy_growth, 6),
            # Совместимый alias для прежних JSON-потребителей. Это не оценка
            # наибольшего FTLE нелинейной сети; точная семантика дана выше.
            "finite_time_lyapunov_exponent": round(finite_time_energy_growth, 6),
            "finite_time_metric_semantics": (
                "logarithmic_growth_rate_of_quadratic_lyapunov_energy_not_maximal_nonlinear_ftle"
            ),
            # Несуффиксные поля P описывают ту же ревизию, что residual/forecast.
            "lyapunov_matrix_fingerprint": lyapunov_fingerprint_used,
            "lyapunov_matrix_fingerprint_used": lyapunov_fingerprint_used,
            "lyapunov_matrix_fingerprint_after_update": lyapunov_fingerprint_after_update,
            "lyapunov_solver": {
                "equation": "K.T @ P @ K - P = -I",
                "method": "smith_doubling_iteration",
                "operator_revision": operator_revision_used,
                "iterations": lyapunov_solver_iterations_used,
                "residual_frobenius": round(lyapunov_solver_residual_used, 8),
                "normalization": "z.T @ P @ z / trace(P)",
            },
            "lyapunov_solver_after_update": {
                "operator_revision": operator_revision_after_update,
                "iterations": self.lyapunov_solver_iterations,
                "residual_frobenius": round(self.lyapunov_solver_residual, 8),
                "update_applied_for_next_step": update_applied_for_next_step,
                "effective_from_step": (
                    step_index + 1 if update_applied_for_next_step else None
                ),
            },
            "lyapunov_trend": (
                "growing"
                if lyapunov_delta > 0.01
                else "recovering"
                if lyapunov_delta < -0.01
                else "stable"
            ),
            "threat_proximity": threat_proximity,
            "route_hausdorff": route_hausdorff,
            "predicted_next_vector": {
                "metric_names": self.metric_names,
                "vector": [round(float(value), 6) for value in predicted_next.tolist()],
            },
            "top_residual_metrics": _top_residuals(self.metric_names, transition_residual_vector),
            "arbitrator_recommendation": recommendation,
        }

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        return (vector - self.reference_vector) / self.scale_vector

    def _denormalize(self, normalized: np.ndarray) -> np.ndarray:
        return self.reference_vector + normalized * self.scale_vector

    def _update_observed_operator(self, previous: np.ndarray, current: np.ndarray) -> bool:
        # Детерминированная микровариация здоровой очереди не является новой
        # динамикой и не должна постепенно сдвигать эталонный оператор t0.
        if _rms(previous) < 0.005 and _rms(current) < 0.005:
            return False
        forgetting_factor = 0.985
        self.online_gram = forgetting_factor * self.online_gram + np.outer(previous, previous)
        self.online_cross = forgetting_factor * self.online_cross + np.outer(current, previous)
        candidate = _operator_from_covariances(self.online_gram, self.online_cross)
        candidate = _stabilize_threat_dimensions(candidate, self.metric_names)
        self.observed_operator = _limit_spectral_radius(candidate)
        (
            self.lyapunov_matrix,
            self.lyapunov_solver_iterations,
            self.lyapunov_solver_residual,
        ) = _discrete_lyapunov_matrix(self.observed_operator)
        self.operator_revision += 1
        return True


def apply_koopman_to_arbitrator(
    snapshot: dict[str, Any],
    koopman: dict[str, Any],
    *,
    model=None,
) -> dict[str, Any] | None:
    """Передать прогноз в L7 и вернуть защитный план для следующего шага."""
    snapshot["koopman"] = koopman
    arbitrator = snapshot.get("arbitrator", {})
    analysis = arbitrator.setdefault("analysis", {})
    analysis["koopman_residual"] = float(koopman["one_step_residual"])
    analysis["koopman_forecast_risk"] = float(koopman["forecast_attack_probability"])
    analysis["koopman_predicted_reference_distance"] = float(koopman["predicted_reference_distance"])
    analysis["koopman_operator_drift"] = float(koopman["operator_drift_norm"])
    analysis["koopman_spectral_radius"] = float(koopman["operator_spectral_radius"])
    analysis["koopman_operator_revision_used"] = int(koopman["operator_revision_used"])
    analysis["koopman_operator_revision_after_update"] = int(
        koopman["operator_revision_after_update"]
    )
    analysis["koopman_update_applied_for_next_step"] = bool(
        koopman["update_applied_for_next_step"]
    )
    analysis["prediction_horizon_seconds"] = int(
        koopman.get("maximum_forecast_horizon_seconds", koopman["forecast_horizon_seconds"])
    )
    analysis["prediction_horizon_steps"] = int(koopman.get("maximum_forecast_horizon_steps", 1))
    analysis["prediction_window_end_seconds"] = int(
        koopman.get("forecast_window_end_seconds", koopman["forecast_target_time_seconds"])
    )
    threat_proximity = koopman["threat_proximity"]
    analysis["gold_threat_minimum_distance_ms"] = float(
        threat_proximity.get("minimum_threat_to_gold_latency_ms", 0.0)
    )
    analysis["gold_threat_proximity_risk"] = float(
        threat_proximity["proximity_risk"]
    )
    analysis["route_hausdorff_distance_ms"] = float(
        koopman.get("route_hausdorff", {}).get("distance_ms", 0.0)
    )
    analysis["route_hausdorff_normalized"] = float(
        koopman.get("route_hausdorff", {}).get("normalized_distance", 0.0)
    )
    analysis["route_edge_jaccard_distance"] = float(
        koopman.get("route_hausdorff", {}).get("maximum_edge_jaccard_distance", 0.0)
    )
    analysis["route_latency_stretch_ratio"] = float(
        koopman.get("route_hausdorff", {}).get("maximum_latency_stretch_ratio", 1.0)
    )
    analysis["lyapunov_value"] = float(koopman["lyapunov_value"])
    analysis["lyapunov_delta"] = float(koopman["lyapunov_delta"])
    analysis["lyapunov_derivative_per_second"] = float(
        koopman.get("lyapunov_derivative_per_second", 0.0)
    )
    analysis["lyapunov_remap_pressure"] = float(
        koopman.get("lyapunov_remap_pressure", 0.0)
    )
    analysis["finite_time_energy_growth_rate_per_second"] = float(
        koopman.get("finite_time_energy_growth_rate_per_second", 0.0)
    )
    analysis["finite_time_lyapunov_exponent"] = analysis[
        "finite_time_energy_growth_rate_per_second"
    ]
    analysis["koopman_early_warning"] = bool(koopman["forecast_is_early_warning"])
    analysis["koopman_precursor_score"] = float(koopman["precursor_score"])
    tensor_matrix_koopman = koopman.get("tensor_matrix_koopman", {})
    matrix_residual = float(tensor_matrix_koopman.get("one_step_residual", 0.0))
    matrix_maximum_residual = float(tensor_matrix_koopman.get("maximum_residual", 0.0))
    matrix_pressure = min(1.0, max(matrix_residual, matrix_maximum_residual * 0.25))
    analysis["kronecker_tensor_koopman_residual"] = matrix_residual
    analysis["kronecker_tensor_koopman_pressure"] = round(matrix_pressure, 6)

    predictive_pressure = float(koopman["protective_pressure"])
    stability_pressure = float(koopman.get("lyapunov_remap_pressure", 0.0))
    current_pressure = float(analysis.get("remap_pressure", 0.0))
    new_pressure = max(current_pressure, predictive_pressure, stability_pressure)
    # ``remap_pressure`` is a continuous diagnostic score.  Actioning it is
    # separately gated below at 0.20, which keeps the Lyapunov contribution
    # visible and preserves the aggregation invariant exactly.
    new_pressure = round(new_pressure, 6)
    analysis["remap_pressure"] = new_pressure
    remap = arbitrator.setdefault("remap", {})
    attack_active = bool(snapshot.get("attacks", {}).get("active"))
    early_warning = bool(koopman.get("forecast_is_early_warning"))
    control_triggered = attack_active or early_warning
    if new_pressure > 0.20 and control_triggered:
        remap["needed"] = True
        remap["action"] = "PLAN_REMAP"
        if early_warning and attack_active:
            remap["reason"] = "active_attack_and_koopman_next_attack_warning"
        elif early_warning:
            remap["reason"] = "koopman_early_warning_before_attack"
        elif attack_active:
            remap["reason"] = "mitre_attack_observed_with_koopman_confirmation"
        else:
            remap["reason"] = "koopman_predictive_attack_warning"
        candidates = set(remap.get("candidate_actions", []))
        candidates.update(koopman["arbitrator_recommendation"]["candidate_actions"])
        remap["candidate_actions"] = sorted(candidates)
    elif new_pressure > 0.20:
        # Первый отсчёт predictive-demo остаётся режимом наблюдения. Это
        # защищает маршрутизацию от реакции на единичный сенсорный выброс.
        remap["needed"] = False
        remap["action"] = "OBSERVE_PRECURSOR"
        remap["reason"] = "unconfirmed_precursor_requires_second_sample"
        remap["candidate_actions"] = []
    else:
        remap["needed"] = False
        remap["action"] = "NO_REMAP"
        remap["reason"] = "healthy_stationary_baseline"
        remap["candidate_actions"] = []
    if model is not None:
        target_ids = list(koopman.get("forecast_target_ids", [])) or list(
            koopman.get("current_attack_target_ids", [])
        )
        game = evaluate_defense_game(
            model,
            target_ids=target_ids,
            threat_pressure=max(
                float(koopman.get("forecast_risk_score", 0.0)),
                new_pressure,
            ),
            matrix_koopman_pressure=matrix_pressure,
        )
        arbitrator["game_theory"] = game
        remap["game_recommended_action"] = game["recommended_action"]
        remap["game_selection_mode"] = game["selection_mode"]
        if remap.get("needed") and game["recommended_action"] == "ISOLATE_CONFIRMED_SOURCES_AND_REMAP":
            remap["candidate_actions"] = sorted(
                set(remap.get("candidate_actions", [])) | {"isolate_confirmed_attack_sources"}
            )
    else:
        game = {
            "model": "finite_defender_attacker_normal_form_game",
            "available": False,
            "semantics_ru": "Для игровой оценки нужен граф модели.",
        }
    remap["koopman_forecast"] = {
        "alert_level": koopman["alert_level"],
        "risk_score_next_step": koopman.get("horizon_forecasts", [{}])[0].get(
            "risk_score", koopman["forecast_risk_score"]
        ),
        "maximum_risk_score": koopman["forecast_risk_score"],
        "probability_is_calibrated": False,
        # Совместимый alias для старых JSON-потребителей; это не вероятность.
        "attack_probability_next_step": koopman["forecast_attack_probability"],
        "attack_kind": (
            koopman["forecast_next_attack_kind"]
            if koopman.get("forecast_is_early_warning")
            else koopman["forecast_attack_kind"]
        ),
        "horizon_seconds": koopman.get(
            "maximum_forecast_horizon_seconds", koopman["forecast_horizon_seconds"]
        ),
        "lead_time_seconds": koopman["forecast_lead_time_seconds"],
        "window_start_step": koopman.get("forecast_window_start_step"),
        "window_end_step": koopman.get("forecast_window_end_step"),
        "target_ids": koopman["forecast_target_ids"],
    }
    remap["lyapunov_feedback"] = {
        "pressure": stability_pressure,
        "derivative_per_second": float(koopman.get("lyapunov_derivative_per_second", 0.0)),
        "response_scale_per_second": float(
            koopman.get(
                "lyapunov_remap_response_scale_per_second",
                LYAPUNOV_REMAP_RESPONSE_SCALE_PER_SECOND,
            )
        ),
        "policy": "positive_dV_dt_strengthens_remap_pressure",
    }
    recommendation = koopman.get("arbitrator_recommendation", {})
    defense_plan = recommendation.get("defense_plan")
    sdn_intent = compile_sdn_intent(
        step_index=int(snapshot.get("step_index", 0)),
        remap=remap,
        defense_plan=defense_plan,
        game=game,
    )
    remap["sdn_intent"] = sdn_intent
    if defense_plan is not None:
        defense_plan["sdn_intent"] = sdn_intent
    attacks = snapshot.get("attacks", {})
    armed_status = (
        "ARMED_FOR_NEXT_STEP"
        if attacks.get("scenario") == "mitre-demo"
        else "ARMED_FOR_FORECAST_WINDOW"
    )
    arbitrator["prevention"] = {
        "status": armed_status if defense_plan else "STANDBY",
        "defense_plan": defense_plan,
        "applied_defense": attacks.get("preventive_defense", {}),
        "raw_attack_rate_mbps": float(attacks.get("raw_attack_rate_mbps", 0.0)),
        "admitted_attack_rate_mbps": float(attacks.get("attack_rate_mbps", 0.0)),
        "blocked_attack_rate_mbps": float(attacks.get("blocked_attack_rate_mbps", 0.0)),
        "prevented_legitimate_dropped_packets": int(attacks.get("prevented_legitimate_dropped_packets", 0)),
    }
    remap["sla_restoration_plan"] = recommendation.get(
        "sla_restoration_plan",
        {
            "policy": "strict_priority",
            "order": ["gold", "silver", "bronze"],
            "tiers": attacks.get("sla_restoration", {}).get("tiers", []),
        },
    )
    return defense_plan


def _operator_from_covariances(
    gram: np.ndarray,
    cross: np.ndarray,
    regularization: float = 1e-4,
) -> np.ndarray:
    """Оценить DMD-оператор по накопленным ковариациям H @ pinv(G)."""
    stabilized_gram = gram + regularization * np.eye(gram.shape[0])
    return cross @ np.linalg.pinv(stabilized_gram)


def _limit_spectral_radius(operator: np.ndarray, maximum: float = 0.995) -> np.ndarray:
    """Сохранить устойчивость номинального одношагового прогноза."""
    if not operator.size:
        return operator
    radius = float(max(abs(np.linalg.eigvals(operator))))
    if not np.isfinite(radius) or radius <= maximum:
        return operator
    return operator * (maximum / radius)


def _discrete_lyapunov_matrix(
    operator: np.ndarray,
    *,
    maximum_iterations: int = 32,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, int, float]:
    """Решить ``K.T P K - P = -I`` удваивающей итерацией Смита.

    Для устойчивого дискретного оператора решение равно
    ``P = sum((K**j).T @ (K**j), j=0..inf)``. На каждой итерации уже
    накопленная сумма удваивает покрытый диапазон степеней: 1, 2, 4, 8, … .
    Это математически эквивалентно последовательному суммированию, но требует
    логарифмического числа матричных умножений и не добавляет зависимость SciPy.
    """
    dimension = int(operator.shape[0])
    if dimension == 0:
        return np.zeros_like(operator), 0, 0.0
    identity = np.eye(dimension, dtype=float)
    power = operator.copy()
    matrix = identity.copy()
    iteration_count = 0
    convergence_scale = max(float(np.linalg.norm(identity, ord="fro")), 1.0)
    for iteration in range(1, maximum_iterations + 1):
        increment = power.T @ matrix @ power
        matrix += increment
        iteration_count = iteration
        if float(np.linalg.norm(increment, ord="fro")) <= tolerance * convergence_scale:
            break
        power = power @ power
    matrix = 0.5 * (matrix + matrix.T)
    residual_matrix = operator.T @ matrix @ operator - matrix + identity
    residual = float(np.linalg.norm(residual_matrix, ord="fro") / convergence_scale)
    return matrix, iteration_count, residual


def _lyapunov_state_energy(normalized_state: np.ndarray, matrix: np.ndarray) -> tuple[float, float]:
    """Вернуть сырое ``z.T P z`` и сопоставимое между размерностями значение."""
    if not normalized_state.size or not matrix.size or np.linalg.norm(normalized_state) <= 1e-15:
        return 0.0, 0.0
    raw = max(0.0, float(normalized_state.T @ matrix @ normalized_state))
    normalized = raw / max(float(np.trace(matrix)), 1e-12)
    return raw, normalized


def _finite_time_energy_growth_rate(previous: float, current: float, step_seconds: int) -> float:
    """Return logarithmic growth rate of quadratic Lyapunov energy.

    The factor 1/2 converts energy growth to an amplitude-like rate.  This is a
    useful closed-loop diagnostic, but deliberately is not called a maximal
    finite-time Lyapunov exponent of the nonlinear physical network.
    """
    if previous <= 1e-15 and current <= 1e-15:
        return 0.0
    epsilon = 1e-12
    interval = max(float(step_seconds), 1e-9)
    return float(np.log((max(current, 0.0) + epsilon) / (max(previous, 0.0) + epsilon)) / (2.0 * interval))


def _delay_trend(history: list[np.ndarray], metric_names: list[str]) -> np.ndarray:
    """Оценить робастный локальный тренд только наблюдаемых threat-метрик."""
    if not history:
        return np.zeros(len(metric_names), dtype=float)
    if len(history) < 2:
        return np.zeros_like(history[-1], dtype=float)
    samples = np.vstack(history)
    differences = np.diff(samples, axis=0)
    weights = np.arange(1, differences.shape[0] + 1, dtype=float)
    weighted_trend = np.average(differences, axis=0, weights=weights)
    median_trend = np.median(differences, axis=0)
    # Медиана гасит одиночный всплеск датчика, а взвешенная составляющая
    # сохраняет чувствительность к свежему ускорению атаки.
    trend = 0.65 * weighted_trend + 0.35 * median_trend
    allowed = np.asarray([name in THREAT_METRICS for name in metric_names], dtype=bool)
    trend[~allowed] = 0.0
    # Ограничение не даёт единичному выбросу разогнать экстраполяцию.
    return np.clip(trend, -0.75, 0.75)


def _multi_horizon_forecasts(
    *,
    metric_names: list[str],
    reference_vector: np.ndarray,
    scale_vector: np.ndarray,
    operator: np.ndarray,
    z_current: np.ndarray,
    observed_trend: np.ndarray,
    attack_state: dict[str, Any],
    precursor_score: float,
    active_attack_score: float,
    residual: float,
    operator_drift: float,
    threat_proximity_risk: float,
    step_index: int,
    time_seconds: int,
    step_seconds: int,
    maximum_horizon_steps: int,
    warning_risk_threshold: float,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Повторно применить Koopman и скорректировать threat-координаты трендом."""
    forecasts: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    z_iterated = z_current.copy()
    z_dmd_only = z_current.copy()
    threat_indices = [index for index, name in enumerate(metric_names) if name in THREAT_METRICS]
    precursor_kind = _predicted_precursor_kind(attack_state)
    current_kind = _current_attack_kind(attack_state)
    predicted_kind = precursor_kind if precursor_kind != "none" else current_kind
    entities = _forecast_entities(attack_state)
    target_ids = entities["next_target_ids"] or entities["current_target_ids"]
    precursor_observed = _observable_precursor_present(attack_state, precursor_score)

    for horizon_steps in range(1, max(1, maximum_horizon_steps) + 1):
        dmd_projection = operator @ z_dmd_only
        z_dmd_only = np.clip(dmd_projection, -8.0, 8.0)
        dmd_prediction = reference_vector + z_dmd_only * scale_vector
        dmd_distance = _rms(z_dmd_only)
        dmd_attack_score = _attack_score(metric_names, dmd_prediction)
        dmd_risk = _forecast_risk(
            deviation=dmd_distance,
            residual=residual / np.sqrt(float(horizon_steps)),
            operator_drift=operator_drift,
            attack_score=dmd_attack_score,
            precursor_only=precursor_score > 0.0 and active_attack_score == 0.0,
            threat_proximity_risk=0.0,
        )
        operator_projection = operator @ z_iterated
        trend_projection = z_current + observed_trend * horizon_steps
        trend_weight = min(0.78, 0.38 + 0.08 * horizon_steps)
        candidate = operator_projection.copy()
        for index in threat_indices:
            candidate[index] = (
                (1.0 - trend_weight) * operator_projection[index]
                + trend_weight * trend_projection[index]
            )
        candidate = np.clip(candidate, -8.0, 8.0)
        trend_prediction = reference_vector + candidate * scale_vector
        trend_distance = _rms(candidate)
        trend_attack_score = _attack_score(metric_names, trend_prediction)
        trend_risk = _forecast_risk(
            deviation=trend_distance,
            residual=residual / np.sqrt(float(horizon_steps)),
            operator_drift=operator_drift,
            attack_score=trend_attack_score,
            precursor_only=precursor_score > 0.0 and active_attack_score == 0.0,
            threat_proximity_risk=(
                threat_proximity_risk if precursor_observed else 0.0
            ),
        )
        predicted = trend_prediction
        predicted = _inject_precursor_forecast(
            metric_names,
            predicted,
            attack_state,
            precursor_score=precursor_score,
            horizon_steps=horizon_steps,
        )
        z_iterated = (predicted - reference_vector) / scale_vector
        predicted_distance = _rms(z_iterated)
        predicted_attack_score = _attack_score(metric_names, predicted)
        risk = _forecast_risk(
            deviation=predicted_distance,
            residual=residual / np.sqrt(float(horizon_steps)),
            operator_drift=operator_drift,
            attack_score=predicted_attack_score,
            precursor_only=precursor_score > 0.0 and active_attack_score == 0.0,
            threat_proximity_risk=(
                threat_proximity_risk
                if precursor_observed or active_attack_score > 0.0
                else 0.0
            ),
        )
        forecasts.append(
            {
                "horizon_steps": horizon_steps,
                "horizon_seconds": horizon_steps * step_seconds,
                "target_step": step_index + horizon_steps,
                "target_time_seconds": time_seconds + horizon_steps * step_seconds,
                "risk_score": round(risk, 6),
                "koopman_dmd_only_risk_score": round(dmd_risk, 6),
                "koopman_plus_delay_trend_risk_score": round(trend_risk, 6),
                "precursor_classifier_risk_increment": round(max(0.0, risk - trend_risk), 6),
                "predicted_attack_score": round(predicted_attack_score, 6),
                "predicted_reference_distance": round(predicted_distance, 6),
                "attack_kind": predicted_kind,
                "target_ids": target_ids,
                "alert_level": _alert_level(risk),
                "is_warning": bool(
                    precursor_observed
                    and risk >= warning_risk_threshold
                ),
                "prediction_origin": (
                    "repeated_koopman_projection_plus_observed_delay_trend_plus_causal_precursor_classifier"
                ),
            }
        )
        vectors.append(predicted)
    return forecasts, vectors


def _observable_precursor_present(attack_state: dict[str, Any], precursor_score: float) -> bool:
    """Проверить наличие измеренного сигнала, не обращаясь к расписанию атаки."""
    if precursor_score <= 0.0 or not isinstance(attack_state, dict):
        return False
    return any(
        isinstance(item.get("signals"), dict) and bool(item["signals"])
        for item in attack_state.get("precursors", [])
    )


def _sanitized_geometry_observation(attack_state: dict[str, Any]) -> dict[str, Any]:
    """Подготовить геометрию Хаусдорфа только из уже наблюдаемых узлов-сенсоров."""
    if not isinstance(attack_state, dict):
        return {"events": [], "precursors": []}
    current_events = [
        {"target_id": item.get("target_id"), "routes": []}
        for item in attack_state.get("events", [])
        if item.get("target_id")
    ]
    sensor_precursors = [
        {"suspected_target_id": item.get("sensor_node_id")}
        for item in attack_state.get("precursors", [])
        if item.get("sensor_node_id")
    ]
    return {"events": current_events, "precursors": sensor_precursors}


def _stabilize_threat_dimensions(operator: np.ndarray, metric_names: list[str]) -> np.ndarray:
    stabilized = operator.copy()
    for index, name in enumerate(metric_names):
        if name in THREAT_METRICS:
            stabilized[index, :] = 0.0
            stabilized[index, index] = 0.85
    return stabilized


def _training_residual(operator: np.ndarray, normalized_training: np.ndarray) -> float:
    residuals = []
    for previous, current in zip(normalized_training[:-1], normalized_training[1:]):
        residuals.append(_rms(current - operator @ previous))
    return float(np.mean(residuals)) if residuals else 0.0


def _ideal_training_vectors(metric_names: list[str], reference: np.ndarray, scale: np.ndarray) -> np.ndarray:
    rows = []
    sample_count = 64
    for step in range(sample_count):
        phase = step / (sample_count - 1) * 4.0 * np.pi
        perturb = np.zeros_like(reference)
        if step == 0:
            rows.append(reference.copy())
            continue
        for index, name in enumerate(metric_names):
            if name.startswith("OBS."):
                continue
            # Разные частоты дают полноранговую номинальную выборку вместо
            # нескольких одинаковых гармоник со сдвигом фазы.
            harmonic = np.sin((1.0 + index * 0.17) * phase + index * 0.13)
            harmonic += 0.28 * np.cos((2.3 + index * 0.11) * phase + index * 0.07)
            if "sla_margin" in name or "stability_margin" in name:
                perturb[index] = 0.0048 * harmonic
            elif "cpu_load_percent" in name or "ram_load_percent" in name or "utilization" in name:
                perturb[index] = 0.0080 * harmonic
            elif "loss_probability" in name:
                perturb[index] = 0.0016 * harmonic
            elif "terrain_risk" in name:
                perturb[index] = 0.0008 * harmonic
            else:
                perturb[index] = 0.0024 * harmonic
        rows.append(reference + perturb * scale)
    return np.vstack(rows)


def _scale_vector(metric_names: list[str], reference: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(reference), 1.0)
    overrides = {
        "OBS.attack_active_count": 3.0,
        "OBS.attack_intensity_ratio": 1.0,
        "OBS.attack_rate_mbps": 32_000.0,
        "OBS.legitimate_loss_ratio": 0.40,
        "OBS.legitimate_delivery_ratio": 0.40,
        "OBS.impacted_gold_flow_count": 80.0,
        "OBS.gold_sla_compliance_ratio": 0.40,
        "OBS.maximum_target_utilization_percent": 100.0,
        "OBS.precursor_count": 5.0,
        "OBS.precursor_confidence": 1.0,
        "OBS.suspicious_source_count": 1_600.0,
        "OBS.source_entropy_ratio": 1.0,
        "OBS.scan_rate_pps": 150_000.0,
        "OBS.traffic_acceleration_ratio": 1.0,
        "OBS.syn_backlog_growth_ratio": 1.0,
        "OBS.power_control_anomaly_ratio": 1.0,
        "OBS.power_voltage_sag_ratio": 0.25,
        "OBS.battery_discharge_rate_ratio": 1.0,
        "OBS.target_concentration_ratio": 1.0,
        "OBS.rerouted_gold_flow_count": 48.0,
        "OBS.rerouted_silver_flow_count": 60.0,
        "OBS.rerouted_bronze_flow_count": 372.0,
        "OBS.isolated_flow_count": 480.0,
        "OBS.service_failover_count": 480.0,
        "OBS.route_change_ratio": 1.0,
        "OBS.maximum_post_remap_utilization_percent": 100.0,
        "OBS.gold_delivery_ratio": 0.40,
        "OBS.silver_delivery_ratio": 0.40,
        "OBS.bronze_delivery_ratio": 0.40,
    }
    for index, name in enumerate(metric_names):
        if name in overrides:
            scale[index] = overrides[name]
        elif "loss_probability" in name:
            scale[index] = 0.005
        elif "cpu_load_percent" in name or "ram_load_percent" in name:
            scale[index] = 100.0
        elif "utilization" in name:
            scale[index] = 1.0
    return np.maximum(scale, 1e-9)


def _reference_context(
    model,
    *,
    metric_names: list[str],
    reference: np.ndarray,
    operator: np.ndarray,
    training_sample_count: int,
    training_rank: int,
) -> dict[str, Any]:
    sla_counts: dict[str, int] = {}
    traffic_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    critical_count = 0
    tensor_count = 0
    for _, attrs in model.graph.nodes(data=True):
        role = attrs.get("role")
        if role:
            role_counts[role] = role_counts.get(role, 0) + 1
        if attrs.get("level") == "L1":
            sla = attrs.get("sla_grade", "unknown")
            traffic = attrs.get("traffic_kind", "unknown")
            sla_counts[sla] = sla_counts.get(sla, 0) + 1
            traffic_counts[traffic] = traffic_counts.get(traffic, 0) + 1
        if attrs.get("critical_protection", {}).get("is_critical"):
            critical_count += 1
        tensor_count += sum(
            1
            for value in attrs.values()
            if hasattr(value, "metric_index") and hasattr(value, "data")
        )
    for _, _, attrs in model.graph.edges(data=True):
        tensor_count += sum(
            1
            for value in attrs.values()
            if hasattr(value, "metric_index") and hasattr(value, "data")
        )
    signature_payload = {
        "metric_names": metric_names,
        "reference": np.round(reference, 9).tolist(),
        "ideal_operator": np.round(operator, 9).tolist(),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    spectral_radius = float(max(abs(np.linalg.eigvals(operator)))) if operator.size else 0.0
    return {
        "trained_on": "ideal_t0_neighbourhood",
        "training_mode": "synthetic_healthy_neighbourhood_dmd",
        "training_data_origin": "deterministic_synthetic_metric_neighbourhood",
        "operator_estimator": "regularized_covariance_dmd",
        "training_sample_count": training_sample_count,
        "training_rank": training_rank,
        "nominal_metric_count": sum(name not in THREAT_METRICS for name in metric_names),
        "state_metric_count": len(metric_names),
        "source_tensor_count": tensor_count,
        "t0_signature": signature,
        "t0_reference_distance": 0.0,
        "ideal_operator_spectral_radius": round(spectral_radius, 8),
        "reference_state": {
            name: round(float(reference[index]), 8)
            for index, name in enumerate(metric_names)
        },
        "subscriber_sla_counts": sla_counts,
        "traffic_kind_counts": traffic_counts,
        "role_counts": role_counts,
        "critical_node_count": critical_count,
    }


def _attack_score(metric_names: list[str], vector: np.ndarray) -> float:
    values = {name: float(vector[index]) for index, name in enumerate(metric_names)}
    return max(
        min(1.0, values.get("OBS.attack_active_count", 0.0) / 3.0),
        min(1.0, values.get("OBS.attack_intensity_ratio", 0.0)),
        min(1.0, values.get("OBS.attack_rate_mbps", 0.0) / 32_000.0),
        min(1.0, values.get("OBS.legitimate_loss_ratio", 0.0) * 4.0),
        min(1.0, max(0.0, 1.0 - values.get("OBS.legitimate_delivery_ratio", 1.0)) * 3.0),
        min(1.0, values.get("OBS.impacted_gold_flow_count", 0.0) / 80.0),
        min(1.0, max(0.0, 1.0 - values.get("OBS.gold_sla_compliance_ratio", 1.0)) * 3.0),
        min(1.0, values.get("OBS.maximum_target_utilization_percent", 0.0) / 100.0),
        min(
            1.0,
            values.get("OBS.authentication_failure_rate_per_second", 0.0)
            / MITRE_T1110_001_ATTEMPTS_PER_SECOND,
        ),
        min(1.0, values.get("OBS.authentication_failure_ratio", 0.0)),
        min(1.0, values.get("OBS.account_lockout_pressure", 0.0)),
        min(1.0, values.get("OBS.state_hausdorff_normalized", 0.0)),
        min(1.0, values.get("OBS.isolated_flow_count", 0.0) / 80.0),
        min(1.0, max(0.0, 1.0 - values.get("OBS.gold_delivery_ratio", 1.0)) * 3.0),
    )


def _precursor_score(metric_names: list[str], vector: np.ndarray) -> float:
    values = {name: float(vector[index]) for index, name in enumerate(metric_names)}
    confidence = min(1.0, values.get("OBS.precursor_confidence", 0.0))
    return max(
        confidence,
        min(1.0, values.get("OBS.precursor_count", 0.0) / 5.0),
        min(1.0, values.get("OBS.suspicious_source_count", 0.0) / 1_600.0),
        min(1.0, values.get("OBS.scan_rate_pps", 0.0) / 150_000.0),
        min(1.0, values.get("OBS.traffic_acceleration_ratio", 0.0) * 0.90),
        min(1.0, values.get("OBS.syn_backlog_growth_ratio", 0.0) * 0.96),
        min(1.0, values.get("OBS.power_control_anomaly_ratio", 0.0) * 0.98),
        min(1.0, values.get("OBS.power_voltage_sag_ratio", 0.0) / 0.15),
        min(1.0, values.get("OBS.battery_discharge_rate_ratio", 0.0)),
        min(
            1.0,
            values.get("OBS.authentication_failure_rate_per_second", 0.0)
            / MITRE_T1110_001_ATTEMPTS_PER_SECOND,
        ),
        min(1.0, values.get("OBS.authentication_failure_ratio", 0.0)),
        min(1.0, values.get("OBS.account_lockout_pressure", 0.0)),
    )


def _forecast_risk(
    *,
    deviation: float,
    residual: float,
    operator_drift: float,
    attack_score: float,
    precursor_only: bool,
    threat_proximity_risk: float,
) -> float:
    active_deviation = deviation > 0.03 or attack_score > 0.0
    residual_signal = min(1.0, residual / 0.18) if active_deviation else min(0.20, residual / 0.18)
    drift_signal = min(1.0, operator_drift / 0.08) if active_deviation else 0.0
    model_signal = max(
        min(1.0, deviation / 0.25),
        residual_signal,
        drift_signal,
    )
    if precursor_only:
        # До onset это не калиброванная вероятность, поэтому одиночная большая
        # невязка не должна автоматически превращаться в «100 % атаки».
        model_signal = 0.45 * model_signal + 0.55 * attack_score
    geometry_signal = (
        0.35 * threat_proximity_risk if attack_score > 0.0 else 0.0
    )
    upper_bound = 0.95 if precursor_only else 1.0
    return round(min(upper_bound, max(attack_score, model_signal, geometry_signal)), 6)


def _classify_precursor_signals(signals: dict[str, Any]) -> str:
    """Классифицировать предвестник только по наблюдаемой телеметрии.

    Пороги заданы в нормированных координатах демонстрационного датчика. Они
    намеренно требуют второй ступени плавного 20-секундного тренда; отдельно
    анализатор проверяет два последовательных отсчёта, поэтому одиночный
    выброс не авторизует превентивную защиту.
    """
    if (
        float(signals.get("authentication_failure_rate_per_second", 0.0)) >= 0.05
        and float(signals.get("authentication_failure_ratio", 0.0)) >= 0.8
    ):
        return "brute_force"
    if (
        float(signals.get("power_control_anomaly_ratio", 0.0)) >= 0.18
        or float(signals.get("power_voltage_sag_ratio", 0.0)) >= 0.02
    ):
        return "power_attack"
    if float(signals.get("syn_backlog_growth_ratio", 0.0)) >= 0.18:
        return "syn_flood"
    if (
        # В модели источниками являются реальные L1-абоненты, поэтому даже
        # 3 независимых источника уже отличают распределённое воздействие от
        # одноисточникового DoS. Порог 100 относился бы к внешнему ботнету и
        # пропускал бы локальную распределённую атаку.
        int(signals.get("suspicious_source_count", 0)) >= 2
        and float(signals.get("source_entropy_ratio", 0.0)) >= 0.05
        and float(signals.get("traffic_acceleration_ratio", 0.0)) >= 0.15
    ):
        return "ddos"
    if (
        # Для одноисточникового DoS второй последовательный отсчёт пересекает
        # оба порога; первая слабая ступень smoothstep остаётся ниже них.
        float(signals.get("traffic_acceleration_ratio", 0.0)) >= 0.11
        and float(signals.get("target_concentration_ratio", 0.0)) >= 0.22
    ):
        return "dos"
    return "unknown_degradation_or_attack"


def _predicted_attack_kind(attack_state: dict[str, Any], forecast_risk: float) -> str:
    events = attack_state.get("events", []) if isinstance(attack_state, dict) else []
    precursors = attack_state.get("precursors", []) if isinstance(attack_state, dict) else []
    if events or precursors:
        kinds = {
            str(event.get("kind", "unknown"))
            for event in events
        }
        kinds.update(
            _classify_precursor_signals(item.get("signals", {}))
            for item in precursors
        )
        return "+".join(sorted(kinds))
    if forecast_risk >= 0.50:
        return "unknown_degradation_or_attack"
    if forecast_risk >= 0.25:
        return "watch"
    return "none"


def _predicted_precursor_kind(attack_state: dict[str, Any]) -> str:
    precursors = attack_state.get("precursors", []) if isinstance(attack_state, dict) else []
    if not precursors:
        return "none"
    return "+".join(sorted({_classify_precursor_signals(item.get("signals", {})) for item in precursors}))


def _current_attack_kind(attack_state: dict[str, Any]) -> str:
    events = attack_state.get("events", []) if isinstance(attack_state, dict) else []
    if not events:
        return "none"
    return "+".join(sorted({str(event.get("kind", "unknown")) for event in events}))


def _alert_level(risk: float) -> str:
    if risk >= 0.70:
        return "CRITICAL"
    if risk >= 0.35:
        return "WARNING"
    if risk >= 0.25:
        return "WATCH"
    return "NORMAL"


def _recommendation(
    risk: float,
    predicted_attack: str,
    *,
    attack_state: dict[str, Any],
    step_index: int,
    step_seconds: int,
    valid_from_step: int,
    valid_until_step: int,
    defense_authorized: bool,
    lyapunov_remap_pressure: float,
) -> dict[str, Any]:
    restoration = _sla_restoration_plan(attack_state)
    decision_pressure = max(risk, lyapunov_remap_pressure)
    if decision_pressure < 0.25:
        return {
            "decision": "observe",
            "decision_pressure": round(decision_pressure, 6),
            "candidate_actions": [],
            "defense_plan": None,
            "sla_restoration_plan": restoration,
        }
    actions = [
        "raise_l7_watch",
        "prepare_nearest_horizon_defense",
        "prepare_multi_horizon_defense",
        "protect_gold_paths",
    ]
    if lyapunov_remap_pressure > 0.0:
        actions.append("monitor_positive_lyapunov_drift_after_remap")
    if "power_attack" in predicted_attack:
        actions.extend(["check_power_domain", "prefer_nodes_with_energy_reserve"])
    if "ddos" in predicted_attack or "dos" in predicted_attack:
        actions.extend(["rate_limit_attack_traffic", "isolate_flood_ingress"])
    if "ddos" in predicted_attack:
        actions.extend(
            [
                "request_upstream_flowspec_or_scrubbing",
                "prepare_rtbh_only_as_last_resort",
            ]
        )
    if "syn_flood" in predicted_attack:
        actions.extend(["enable_syn_protection", "protect_endpoint_queue"])
    if "brute_force" in predicted_attack:
        actions.extend(
            [
                "enforce_account_lockout_policy",
                "require_multi_factor_authentication",
                "review_failed_authentication_events",
            ]
        )
    entities = _forecast_entities(attack_state)
    defense_plan = None
    # В predictive-demo одно измерение ещё не исполняет сетевое действие:
    # план разрешается только после подтверждения сенсора двумя отсчётами.
    # Уже активное воздействие, напротив, требует немедленной реакции.
    if defense_authorized and (entities["attack_ids"] or entities["target_ids"]):
        defense_plan = _defense_plan(
            step_index=step_index,
            step_seconds=step_seconds,
            entities=entities,
            actions=actions,
            risk=decision_pressure,
            valid_from_step=valid_from_step,
            valid_until_step=valid_until_step,
        )
    return {
        "decision": "prepare_or_remap",
        "decision_pressure": round(decision_pressure, 6),
        "candidate_actions": sorted(set(actions)),
        "defense_plan": defense_plan,
        "sla_restoration_plan": restoration,
    }


def _forecast_entities(attack_state: dict[str, Any]) -> dict[str, Any]:
    events = attack_state.get("events", []) if isinstance(attack_state, dict) else []
    precursors = attack_state.get("precursors", []) if isinstance(attack_state, dict) else []
    # Для раннего прогноза допустимы только идентификатор узла, на котором
    # измерен сигнал, и сами сигналы. attack_id/start_step намеренно не читаются.
    next_records = [
        {
            "observation_type": "precursor",
            "target_id": str(item.get("sensor_node_id")),
            "kind": _classify_precursor_signals(item.get("signals", {})),
            "source_ids": sorted(str(source) for source in item.get("observed_source_node_ids", []) if source),
        }
        for item in precursors
        if item.get("sensor_node_id")
        and _classify_precursor_signals(item.get("signals", {})) != "unknown_degradation_or_attack"
    ]
    current_records = [
        {
            "observation_type": "active_event",
            "attack_id": str(item.get("attack_id", "")),
            "target_id": str(item.get("target_id")),
            "kind": str(item.get("kind", "unknown")),
            "source_ids": sorted(str(source) for source in item.get("ingress_nodes", []) if source),
        }
        for item in events
        if item.get("target_id")
    ]
    sensor_nodes = sorted({str(item["target_id"]) for item in next_records})
    current_targets = sorted({str(item["target_id"]) for item in current_records})
    observed_sources = sorted(
        {
            str(source)
            for item in precursors
            for source in item.get("observed_source_node_ids", [])
            if source
        }
        | {
            str(source)
            for event in events
            for source in event.get("ingress_nodes", [])
            if source
        }
    )
    return {
        "attack_ids": sorted(
            str(item.get("attack_id")) for item in events if item.get("attack_id")
        ),
        "correlation_ids": sorted(
            {
                str(item.get("correlation_id"))
                for item in precursors
                if item.get("correlation_id")
            }
        ),
        "target_ids": sorted(set(current_targets) | set(sensor_nodes)),
        "next_target_ids": sensor_nodes,
        "current_target_ids": current_targets,
        "source_ids": observed_sources,
        "attack_kinds": sorted({str(item["kind"]) for item in [*current_records, *next_records]}),
        "entity_records": [*current_records, *next_records],
        "current_entity_records": current_records,
        "next_entity_records": next_records,
    }


def _defense_plan(
    *,
    step_index: int,
    step_seconds: int,
    entities: dict[str, Any],
    actions: list[str],
    risk: float,
    valid_from_step: int,
    valid_until_step: int,
) -> dict[str, Any]:
    identity = "|".join([
        *entities["attack_ids"], *entities["target_ids"],
        *entities["attack_kinds"], *entities.get("source_ids", []),
    ])
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]
    kinds = set(entities["attack_kinds"])
    mitigation_controls: list[str] = []
    attack_confirmed = bool(entities["attack_ids"])
    if kinds & {"dos", "ddos"}:
        mitigation_controls.append("ingress_acl_or_policer")
        mitigation_controls.append(
            "source_quarantine" if attack_confirmed else "observe_source_before_quarantine"
        )
    if "ddos" in kinds:
        mitigation_controls.extend(["upstream_flowspec", "scrubbing_center", "rtbh_last_resort"])
    if "syn_flood" in kinds:
        mitigation_controls.extend(["syn_proxy", "syn_cookies", "connection_rate_limit"])
    if "power_attack" in kinds:
        mitigation_controls.extend(["revoke_management_session", "transfer_to_independent_power_domain"])
    if "brute_force" in kinds:
        mitigation_controls.extend(
            [
                "account_lockout_policy",
                "multi_factor_authentication",
                "conditional_access",
            ]
        )
    routing_control = {
        "valid_from_step": valid_from_step,
        "valid_until_step": max(valid_from_step, valid_until_step),
        "target_ids": list(entities["target_ids"]),
        "attack_kinds": list(entities["attack_kinds"]),
        "source_ids": list(entities.get("source_ids", [])),
        "quarantine_sources": (
            list(entities.get("source_ids", []))
            if attack_confirmed and kinds & {"dos", "ddos", "syn_flood"}
            else []
        ),
        "protection_stage": (
            "confirmed_active_mitigation" if attack_confirmed else "predictive_prestage"
        ),
        "source_quarantine_policy": (
            "active_event_or_explicit_confirmed_plan_only"
        ),
        "entity_records": list(entities.get("entity_records", [])),
        "maximum_projected_utilization_percent": 80.0,
        "sla_order": ["gold", "silver", "bronze"],
        "actions": [
            "reroute_around_network_target",
            "failover_service_to_standby",
            "protect_endpoint_with_acl_or_syn_proxy",
            *(
                ["quarantine_observed_attack_sources"]
                if kinds & {"dos", "ddos", "syn_flood"}
                else []
            ),
        ],
        "mitigation_controls": mitigation_controls,
        "ddos_routing_semantics": (
            "local_reroute_handles_failed_node_or_server; upstream_controls_handle_external_link_saturation"
        ),
    }
    return {
        "plan_id": f"KOOPMAN-{valid_from_step}-{digest}",
        "created_at_step": step_index,
        # Старый ключ оставлен: потребитель, понимающий только один шаг,
        # применит защиту в начале нового прогнозного окна.
        "valid_for_step": valid_from_step,
        "valid_from_step": valid_from_step,
        "valid_until_step": max(valid_from_step, valid_until_step),
        "horizon_seconds": max(1, valid_until_step - step_index) * step_seconds,
        "forecast_risk_score": round(risk, 6),
        "probability_is_calibrated": False,
        # Совместимый alias, оставленный для старой схемы артефактов.
        "forecast_probability": round(risk, 6),
        **entities,
        "actions": sorted(set(actions)),
        "sla_priority_order": ["gold", "silver", "bronze"],
        "critical_node_max_utilization_percent": 80.0,
        "admission_policy": "protect_gold_then_restore_silver_then_bronze",
        "routing_control": routing_control,
        "effectiveness_origin": "scenario_assumption",
    }


def _sla_restoration_plan(attack_state: dict[str, Any]) -> dict[str, Any]:
    current_tiers = {
        item.get("sla_grade"): item
        for item in attack_state.get("sla_restoration", {}).get("tiers", [])
    }
    tiers = []
    actions = {
        "gold": "reserve_capacity_and_restore_first",
        "silver": "restore_after_gold_is_stable",
        "bronze": "best_effort_after_higher_priorities",
    }
    for priority, grade in enumerate(("gold", "silver", "bronze"), start=1):
        current = current_tiers.get(grade, {})
        tiers.append(
            {
                "priority": priority,
                "sla_grade": grade,
                "action": actions[grade],
                "affected_flow_count": int(current.get("affected_flow_count", 0)),
                "sla_compliant_flow_count": int(current.get("sla_compliant_flow_count", 0)),
            }
        )
    return {
        "policy": "strict_priority_non_preemptive_with_gold_reservation",
        "order": ["gold", "silver", "bronze"],
        "tiers": tiers,
    }


def _inject_precursor_forecast(
    metric_names: list[str],
    predicted_next: np.ndarray,
    attack_state: dict[str, Any],
    *,
    precursor_score: float,
    horizon_steps: int = 1,
) -> np.ndarray:
    precursors = attack_state.get("precursors", []) if isinstance(attack_state, dict) else []
    if not precursors:
        return predicted_next
    result = predicted_next.copy()
    index = {name: offset for offset, name in enumerate(metric_names)}

    def set_at_least(name: str, value: float) -> None:
        if name in index:
            result[index[name]] = max(float(result[index[name]]), value)

    # Это прогноз возможного состояния, а не подмена текущего наблюдения.
    # Сила экстраполяции растёт с горизонтом, но определяется только уже
    # измеренным precursor_score.
    activation = min(1.0, 0.72 + 0.07 * max(0, horizon_steps - 1))
    set_at_least("OBS.attack_active_count", float(len(precursors)) * activation)
    set_at_least("OBS.attack_intensity_ratio", min(1.0, precursor_score * activation))
    set_at_least(
        "OBS.maximum_target_utilization_percent",
        min(100.0, precursor_score * (80.0 + 4.0 * max(0, horizon_steps - 1))),
    )
    signals = [item.get("signals", {}) for item in precursors]
    authentication_rate = sum(
        float(item.get("authentication_failure_rate_per_second", 0.0))
        for item in signals
    )
    authentication_failure_ratio = max(
        (float(item.get("authentication_failure_ratio", 0.0)) for item in signals),
        default=0.0,
    )
    account_lockout_pressure = max(
        (float(item.get("account_lockout_pressure", 0.0)) for item in signals),
        default=0.0,
    )
    if authentication_rate > 0.0:
        set_at_least(
            "OBS.authentication_failure_rate_per_second",
            authentication_rate * activation,
        )
        set_at_least(
            "OBS.authentication_failure_ratio", authentication_failure_ratio,
        )
        set_at_least(
            "OBS.account_lockout_pressure", account_lockout_pressure * activation,
        )
    return result


def _top_residuals(metric_names: list[str], residual_vector: np.ndarray, limit: int = 5) -> list[dict[str, Any]]:
    if not residual_vector.size:
        return []
    indices = np.argsort(np.abs(residual_vector))[::-1][:limit]
    return [
        {"metric": metric_names[int(index)], "normalized_residual": round(float(residual_vector[int(index)]), 6)}
        for index in indices
        if abs(float(residual_vector[int(index)])) > 1e-9
    ]


def _operator_drift(observed: np.ndarray, ideal: np.ndarray) -> float:
    return float(
        np.linalg.norm(observed - ideal, ord="fro")
        / max(float(np.linalg.norm(ideal, ord="fro")), 1e-9)
    )


def _matrix_fingerprint(matrix: np.ndarray) -> str:
    return hashlib.sha256(np.round(matrix, 10).tobytes()).hexdigest()


def _rms(vector: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(vector)))) if vector.size else 0.0
