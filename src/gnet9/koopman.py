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

from .kalman import DiagonalKalmanFilter, DEFAULT_PROCESS_VARIANCE, DEFAULT_MEASUREMENT_VARIANCE
from .local_observation import LocalObservationFilter, is_smoothable_metric
from .attack_signatures import LAG_WEIGHTS, match_attack_signature
from .attacks import MITRE_T1110_001_ATTEMPTS_PER_SECOND, MITRE_TENSOR_MAPPINGS
from .arbitrator import build_trigger_card, evaluate_defense_game
from .metrics import gold_threat_proximity_view
from .routing import FixedTopologyRoutes
from .tensor_matrix import TensorKroneckerKoopman, TensorMatrixView, TensorLevelMatrix


THREAT_METRICS = {
    "OBS.dns_query_pressure_ratio",
    "OBS.application_work_pressure_ratio",
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
NOMINAL_THREAT_SCORE_LIMIT = 0.10
INPUT_SINGULAR_VALUE_TOLERANCE = 1e-7
DIAGONAL_RIDGE = 1e-4
MAXIMUM_NOMINAL_COEFFICIENT = 0.995
CONFIRMATION_SECONDS = 4
NORMAL_ARRIVAL_METRIC = "OBS.normal_arrival_rate_pps"


@dataclass
class NominalArrivalFit:
    """Аффинный AR(1) в пакетах/с по парам исходных исправных наблюдений."""

    count: int = 0
    mean_x: float = 0.0
    mean_y: float = 0.0
    centered_xx: float = 0.0
    centered_xy: float = 0.0
    low: float = math.inf
    high: float = -math.inf

    def add_pair(self, x: float, y: float) -> None:
        self.count += 1
        delta = x - self.mean_x
        self.mean_x += delta / self.count
        self.mean_y += (y - self.mean_y) / self.count
        self.centered_xx += delta * (x - self.mean_x)
        self.centered_xy += delta * (y - self.mean_y)
        self.low = min(self.low, x)
        self.high = max(self.high, x)

    def parameters(self) -> dict[str, Any]:
        mean = (self.mean_x + self.mean_y) / 2.0
        raw_slope = None
        if self.count >= 2 and self.high > self.low:
            # Переход от раздельных средних x,y к общему стационарному среднему.
            offset = self.mean_x - mean
            xx = self.centered_xx + self.count * offset**2
            xy = self.centered_xy + self.count * offset * (self.mean_y - mean)
            raw_slope = xy / xx
        return {
            "metric": NORMAL_ARRIVAL_METRIC,
            "input": "raw_normal_arrivals_from_accepted_healthy_pairs",
            "transition_count": self.count,
            "mean_of_pair_endpoints_pps": mean if self.count else None,
            "raw_lag_one_coefficient": raw_slope,
            "lag_one_coefficient": (
                float(np.clip(raw_slope, -MAXIMUM_NOMINAL_COEFFICIENT, MAXIMUM_NOMINAL_COEFFICIENT))
                if raw_slope is not None else None
            ),
            "observed_input_range_pps": [self.low, self.high] if self.count else None,
        }


def _physical_domains(names: list[str]) -> dict[str, dict[str, float | None]]:
    """Пределы компактного вектора в исходных единицах симулятора."""
    fractions = {
        "L1.sla_margin.min", "L2.stability_margin.min", "EDGE.stability_margin.min",
        "EDGE.loss_probability.max", "L8.terrain_risk.max", "L7.remap_pressure.mean",
        "OBS.state_hausdorff_normalized", "OBS.account_lockout_pressure", "OBS.precursor_confidence",
    }
    nonnegative = {
        "L1.traffic_intensity_rho.mean", "EDGE.utilization.mean",
        "L7.gold_threat_minimum_distance_ms.mean", "L7.koopman_residual.mean",
        "OBS.normal_arrival_rate_pps",
    }
    percentages = {"L2.cpu_load_percent.max", "L2.ram_load_percent.max",
                   "OBS.maximum_target_utilization_percent"}
    reserved = {"OBS.rerouted_gold_flow_count", "OBS.rerouted_silver_flow_count",
                "OBS.rerouted_bronze_flow_count", "OBS.isolated_flow_count",
                "OBS.service_failover_count", "OBS.route_change_ratio",
                "OBS.maximum_post_remap_utilization_percent"}
    domains = {}
    for name in names:
        lower = 0.0 if name in THREAT_METRICS | nonnegative | percentages | fractions else None
        upper = None
        if name in reserved:
            upper = 0.0
        elif name in percentages:
            upper = 100.0
        elif name in fractions or (name in THREAT_METRICS and name.endswith("_ratio")):
            upper = 1.0
        domains[name] = {"lower": lower, "upper": upper}
    return domains


def _input_spectrum(inputs: np.ndarray) -> dict[str, Any]:
    """SVD входов до регуляризации, в единицах нормированного состояния."""
    singular = np.linalg.svd(inputs, compute_uv=False)
    retained = singular[singular > INPUT_SINGULAR_VALUE_TOLERANCE]
    return {"singular_values": singular.tolist(), "rank": int(retained.size),
            "feature_count": inputs.shape[1], "absolute_tolerance": INPUT_SINGULAR_VALUE_TOLERANCE,
            "condition_number_on_retained_subspace": (
                float(retained[0] / retained[-1]) if retained.size else None)}


def _healthy_quality_observation(metric_names, vector, attack_state) -> bool:
    """Даже слабое нарушение SLA/потери исключают наблюдение из нормы.

    Проверяются исходные значения, до фильтра и прогноза. Отсутствующий
    показатель не заменяется выдуманным измерением. Допуск 1e-9 относится
    только к погрешности представления долей, а не к допустимому ущербу.
    """
    full_ratios = {"OBS.gold_sla_compliance_ratio", "OBS.legitimate_delivery_ratio",
                   "OBS.gold_delivery_ratio", "OBS.silver_delivery_ratio", "OBS.bronze_delivery_ratio"}
    zero_counts = {"OBS.legitimate_loss_ratio", "OBS.impacted_gold_flow_count",
                   "OBS.isolated_flow_count", "OBS.rerouted_gold_flow_count",
                   "OBS.rerouted_silver_flow_count", "OBS.rerouted_bronze_flow_count",
                   "OBS.service_failover_count", "OBS.route_change_ratio"}
    for name, value in zip(metric_names, vector):
        if name in full_ratios and value < 1.0 - 1e-9:
            return False
        if name in zero_counts and abs(value) > 1e-9:
            return False
    for tier in attack_state.get("sla_restoration", {}).get("tiers", []):
        if int(tier.get("sla_compliant_flow_count", 0)) < int(tier.get("flow_count", 0)):
            return False
    return True


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
    observed_input_qr: np.ndarray
    observed_input_low: np.ndarray
    observed_input_high: np.ndarray
    lyapunov_matrix: np.ndarray
    lyapunov_solver_iterations: int
    lyapunov_solver_residual: float
    state_history: list[np.ndarray] = field(default_factory=list)
    sensor_streaks: dict[str, int] = field(default_factory=dict)
    last_sensor_step: int | None = None
    signature_history: dict[tuple[str, str], dict[int, dict | None]] = field(default_factory=dict)
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
    nominal_arrivals: NominalArrivalFit = field(default_factory=NominalArrivalFit)
    previous_raw_arrival_pps: float | None = None
    last_observed_threat_step: int | None = None
    previous_transition_was_nominal: bool = True
    tensor_koopman: TensorKroneckerKoopman | None = None
    kalman_filter: DiagonalKalmanFilter | None = None
    local_filter: LocalObservationFilter | None = None
    tensor_filters: dict[str, DiagonalKalmanFilter] = field(default_factory=dict)

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
        runtime_state: dict[str, Any] | None = None,
        kalman_process_variance: float = DEFAULT_PROCESS_VARIANCE,
        kalman_measurement_variance: float = DEFAULT_MEASUREMENT_VARIANCE,
    ) -> "KoopmanOnlineAnalyzer":
        metric_names = list(state_vector["metric_names"])
        reference = np.asarray(state_vector["vector"], dtype=float)
        if (not metric_names or len(set(metric_names)) != len(metric_names)
                or reference.shape != (len(metric_names),) or not np.isfinite(reference).all()):
            raise ValueError("Эталону нужны уникальные имена и конечное значение каждой координаты")
        scale = _scale_vector(metric_names, reference)
        training = _ideal_training_vectors(metric_names, reference, scale)
        normalized_training = (training - reference) / scale
        x = normalized_training[:-1].T
        y = normalized_training[1:].T
        diagonal = np.sum(x * y, axis=1) / (np.sum(x * x, axis=1) + DIAGONAL_RIDGE)
        operator = np.diag(np.clip(diagonal, 0.0, MAXIMUM_NOMINAL_COEFFICIENT))
        operator = _stabilize_threat_dimensions(operator, metric_names)
        residual = _training_residual(operator, normalized_training)
        lyapunov_matrix, lyapunov_iterations, lyapunov_residual = _discrete_lyapunov_matrix(operator)
        return cls(
            metric_names=metric_names,
            kalman_filter=DiagonalKalmanFilter(len(metric_names),
                process_variance=kalman_process_variance,
                measurement_variance=kalman_measurement_variance),
            local_filter=(LocalObservationFilter(runtime_state,
                process_variance=kalman_process_variance,
                measurement_variance=kalman_measurement_variance)
                if runtime_state is not None else None),
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
                input_spectrum=_input_spectrum(x.T),
            ),
            online_gram=np.zeros_like(operator),
            online_cross=np.zeros_like(operator),
            observed_input_qr=np.empty((0, len(reference))),
            observed_input_low=np.full_like(reference, np.inf),
            observed_input_high=np.full_like(reference, -np.inf),
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
                TensorKroneckerKoopman.from_reference(tensor_matrices, model=model)
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
        runtime_state: dict[str, Any] | None = None,
        routing: FixedTopologyRoutes | None = None,
    ) -> dict[str, Any]:
        raw_current = np.asarray(state_vector["vector"], dtype=float)
        if (list(state_vector.get("metric_names", self.metric_names)) != self.metric_names
                or raw_current.shape != self.reference_vector.shape or not np.isfinite(raw_current).all()):
            raise ValueError("Ожидается конечный вектор в порядке координат эталона")
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
        observed_spectrum = _input_spectrum(self.observed_input_qr)
        observed_spectrum["sample_count"] = nominal_observation_count_used
        operator_fingerprint_used = _matrix_fingerprint(operator_used)
        lyapunov_fingerprint_used = _matrix_fingerprint(lyapunov_matrix_used)
        current = raw_current
        raw_normalized = self._normalize(current)
        active_attack_score = _attack_score(self.metric_names, current)
        precursor_score = _precursor_score(self.metric_names, current)
        observable_precursor_now = _observable_precursor_present(attack_state, precursor_score)
        observed_events = attack_state.get("events", [])
        observable_attack_now = bool(observed_events)
        healthy_quality = _healthy_quality_observation(self.metric_names, current, attack_state)
        if self.kalman_filter is None:
            self.kalman_filter = DiagonalKalmanFilter(len(self.metric_names))
        filter_mask = np.asarray([name not in THREAT_METRICS and is_smoothable_metric(name)
                                  for name in self.metric_names])
        # Наблюдаемая угроза не меняет сохранённую оценку исправной сети.
        # Её текущий ущерб проходит в прогноз без задержки от сглаживания.
        filter_applied = (healthy_quality and not observable_attack_now and not observable_precursor_now
                          and max(active_attack_score, precursor_score) < NOMINAL_THREAT_SCORE_LIMIT)
        local_observation = (
            self.local_filter.analyze(
                runtime_state, nominal=filter_applied, step_index=step_index,
                event_active=observable_attack_now, precursor_active=observable_precursor_now,
                observed_events=tuple((str(event.get("target_id", "")), str(event.get("kind", "")),
                                       bool(event.get("is_recovery_effect"))) for event in observed_events),
            )
            if self.local_filter is not None and runtime_state is not None else None
        )
        tensor_input = self._filter_tensor_observation(tensor_matrices) if filter_applied else tensor_matrices
        tensor_matrix_koopman = (
            self.tensor_koopman.analyze(tensor_input)
            if self.tensor_koopman is not None and tensor_input is not None
            else {
                "model": "factorised_kronecker_tensor_koopman", "available": False,
                "semantics_ru": "Матрицы тензоров не переданы в этот вызов анализатора.",
            }
        )
        tensor_matrix_koopman["input_kalman_filtered"] = (
            filter_applied and self.tensor_koopman is not None and tensor_input is not None
        )
        z_current = (self.kalman_filter.update(raw_normalized, filter_mask)
                     if filter_applied else raw_normalized.copy())
        current = np.where(filter_mask & filter_applied, self._denormalize(z_current), current)
        residual = 0.0
        transition_residual_vector = np.zeros_like(z_current)
        if self.previous_vector is not None:
            z_previous = self._normalize(self.previous_vector)
            predicted_current = operator_used @ z_previous
            transition_residual_vector = z_current - predicted_current
            residual = _rms(transition_residual_vector)

        deviation = _rms(z_current)
        operator_drift = _operator_drift(operator_used, self.ideal_operator)
        threat_score = max(active_attack_score, precursor_score)
        if observable_precursor_now or observable_attack_now:
            self.last_observed_threat_step = step_index
        threat_proximity = gold_threat_proximity_view(
            model,
            _sanitized_geometry_observation(attack_state),
            routing=routing,
        )
        route_hausdorff = attack_state.get(
            "route_observation",
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
        forecast_entities = _forecast_entities(attack_state, self.signature_history, step_index, self.step_seconds)
        observed_sensors = {
            str(item["sensor_node_id"]) for item in attack_state.get("precursors", [])
            if item.get("sensor_node_id") and isinstance(item.get("signals"), dict) and item["signals"]
        }
        previous_streaks = self.sensor_streaks if self.last_sensor_step == step_index - 1 else {}
        self.sensor_streaks = {sensor: previous_streaks.get(sensor, 0) + 1 for sensor in observed_sensors}
        self.last_sensor_step = step_index
        required_confirmation_samples = max(1, math.ceil(CONFIRMATION_SECONDS / max(1, self.step_seconds)))
        confirmed_records = [record for record in forecast_entities["next_entity_records"]
                             if self.sensor_streaks[record["target_id"]] >= required_confirmation_samples]
        confirmed_sensors = {record["target_id"] for record in confirmed_records}
        predicted_next_attack = _predicted_precursor_kind(attack_state, confirmed_records)
        numeric_origins = self._numeric_origins(z_current, nominal=filter_applied)
        arrival_fit = self.nominal_arrivals.parameters()
        arrival_index = None
        if NORMAL_ARRIVAL_METRIC in self.metric_names:
            arrival_index = self.metric_names.index(NORMAL_ARRIVAL_METRIC)
            if self.nominal_arrivals.count == 0:
                origin = "persistence_no_healthy_arrival_pair"
            elif arrival_fit["lag_one_coefficient"] is None:
                origin = "nominal_arrival_mean_insufficient_variation"
            elif self.nominal_arrivals.low <= raw_current[arrival_index] <= self.nominal_arrivals.high:
                origin = "observed_affine_arrival_dmd"
            else:
                origin = "nominal_arrival_mean_outside_observed_range"
            numeric_origins[arrival_index] = origin
        support_ranges = [
            [float(low), float(high)] if low <= high else None
            for low, high in zip(self.observed_input_low, self.observed_input_high)
        ]
        horizon_forecasts = _multi_horizon_forecasts(
            metric_names=self.metric_names,
            reference_vector=self.reference_vector,
            scale_vector=self.scale_vector,
            operator=operator_used,
            z_current=z_current,
            raw_current=raw_current,
            numeric_origins=numeric_origins,
            arrival_index=arrival_index,
            arrival_fit=arrival_fit,
            physical_domains=self.reference_context["physical_domains"],
            observed_trend=observed_trend,
            attack_state=attack_state,
            target_ids=sorted(confirmed_sensors) or forecast_entities["current_target_ids"],
            precursor_confirmed=bool(confirmed_records),
            precursor_kind=predicted_next_attack,
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
        predicted_next = first_forecast["predicted_vector"]
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
        lyapunov_transition = None
        lyapunov_basis_rebased = bool(
            self.previous_vector is not None
            and self.previous_analysis_operator_revision is not None
            and self.previous_analysis_operator_revision != operator_revision_used
        )
        if self.previous_vector is not None:
            previous_raw_energy, comparable_previous_lyapunov = _lyapunov_state_energy(
                z_previous,
                lyapunov_matrix_used,
            )
            error_energy = lyapunov_matrix_used @ transition_residual_vector
            lyapunov_transition = {
                "unforced_delta_raw": -float(z_previous @ z_previous),
                "cross_term_raw": 2.0 * float(predicted_current @ error_energy),
                "residual_energy_raw": float(transition_residual_vector @ error_energy),
                "observed_delta_raw": float(z_current @ lyapunov_matrix_used @ z_current) - previous_raw_energy,
            }
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
        predicted_attack = _predicted_attack_kind(attack_state, forecast_risk, predicted_next_attack)
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
        warning_forecasts = [item for item in horizon_forecasts if bool(item["is_warning"])]
        early_warning = bool(warning_forecasts)
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
            warning_confirmed=early_warning or bool(attack_state.get("events", [])),
            lyapunov_remap_pressure=lyapunov_remap_pressure,
        )

        # Обучение номинальной динамике коммитится после завершения всех
        # вычислений снимка. Новые K/P не меняют ни одно значение текущего шага.
        update_applied_for_next_step = False
        nominal_now = (healthy_quality and not observable_attack_now and not observable_precursor_now
                       and threat_score < NOMINAL_THREAT_SCORE_LIMIT and forecast_risk < 0.25)
        transition_accepted = (self.previous_vector is not None
                               and self.previous_transition_was_nominal and nominal_now)
        if transition_accepted:
            z_previous_for_update = self._normalize(self.previous_vector)
            self.nominal_observation_count += 1
            if arrival_index is not None and self.previous_raw_arrival_pps is not None:
                self.nominal_arrivals.add_pair(self.previous_raw_arrival_pps, float(raw_current[arrival_index]))
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
        if arrival_index is not None:
            self.previous_raw_arrival_pps = float(raw_current[arrival_index])
        self.previous_transition_was_nominal = nominal_now
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
            "kalman_filter": {
                "model": "diagonal_random_walk_on_normalized_observations",
                "process_variance": self.kalman_filter.process_variance,
                "measurement_variance": self.kalman_filter.measurement_variance,
                "applied_to_nominal_observation": filter_applied,
                "healthy_quality_observation": healthy_quality,
                "filtered_metric_names": [name for name, enabled in zip(self.metric_names, filter_mask) if enabled and filter_applied],
                "raw_normalized_vector": raw_normalized.tolist(),
                "filtered_normalized_vector": z_current.tolist(),
                "gain_vector": (self.kalman_filter.gain if filter_applied else np.ones_like(raw_normalized)).tolist(),
                "posterior_variance_vector": self.kalman_filter.variance.tolist(),
                "covariance_factorization": "diagonal_cholesky",
                "local_observation": local_observation,
                "koopman_input_uncertainty": {
                    "semantics": "one_step_propagated_input_noise_not_total_forecast_error",
                    "operator_revision_used": operator_revision_used,
                    "normalized_std_vector": np.sqrt(
                        (operator_used ** 2) @ np.where(
                            filter_mask & filter_applied, self.kalman_filter.variance,
                            self.kalman_filter.measurement_variance)
                    ).tolist(),
                },
                "filtered_normal_arrival_rate_pps": float(current[self.metric_names.index("OBS.normal_arrival_rate_pps")]) if "OBS.normal_arrival_rate_pps" in self.metric_names else None,
            },
            "training_reference": self.reference_context,
            "training_residual": round(self.training_residual, 8),
            "numeric_forecast_origin": dict(zip(self.metric_names, numeric_origins)),
            "nominal_training": {
                "observation_eligible": nominal_now,
                "transition_accepted": transition_accepted,
                "input": "filtered_normalized_observations",
                "requires_two_healthy_observations": True,
                "observed_input_spectrum": observed_spectrum,
                "diagonal_coefficients_used": np.diag(operator_used).tolist(),
                "observed_normalized_input_ranges_used": support_ranges,
                "ridge_penalty": DIAGONAL_RIDGE,
                "ordinary_arrival_fit_used": arrival_fit,
                "maximum_coefficient": MAXIMUM_NOMINAL_COEFFICIENT,
            },
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
            "warning_confirmation_seconds": CONFIRMATION_SECONDS,
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
            "baseline_frozen_due_to_threat": not nominal_now,
            "active_attack_score": round(active_attack_score, 6),
            "precursor_score": round(precursor_score, 6),
            "predicted_attack_score": round(predicted_attack_score, 6),
            "forecast_attack_probability": forecast_risk,
            "forecast_risk_score": forecast_risk,
            "probability_is_calibrated": False,
            "risk_score_semantics": "uncalibrated_detection_risk_rank",
            "forecast_attack_kind": predicted_attack,
            "forecast_next_attack_kind": predicted_next_attack,
            "current_attack_kind": current_attack,
            "forecast_attack_ids": [] if early_warning else forecast_entities["attack_ids"],
            "current_attack_ids": forecast_entities["attack_ids"],
            "forecast_correlation_ids": forecast_entities["correlation_ids"],
            "forecast_entity_records": (
                confirmed_records
                if early_warning else forecast_entities["current_entity_records"]
            ),
            "signature_entity_records": forecast_entities["signature_entity_records"],
            "current_attack_entity_records": forecast_entities["current_entity_records"],
            "forecast_target_ids": (
                sorted(confirmed_sensors) if early_warning else forecast_entities["target_ids"]
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
            "lyapunov_transition": lyapunov_transition,
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
                "applies_to": "unforced_linear_map_with_fixed_operator_revision",
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
                "vector": predicted_next,
            },
            "top_residual_metrics": _top_residuals(self.metric_names, transition_residual_vector),
            "arbitrator_recommendation": recommendation,
        }

    def _filter_tensor_observation(self, observation: TensorMatrixView | None) -> TensorMatrixView | None:
        """Очистить обычные показатели по локальным осям, сохранив исходный снимок."""
        if observation is None or self.tensor_koopman is None:
            return observation
        matrices = {}
        for level in observation.levels:
            observed = observation.matrices[level]
            reference = self.tensor_koopman.reference.matrices[level]
            if (observed.entity_ids != reference.entity_ids
                    or observed.metric_names != reference.metric_names):
                raise ValueError("Оси фильтра тензора должны совпадать с t0")
            if not observed.values.size:
                matrices[level] = observed
                continue
            scale = np.maximum(np.abs(reference.values), 1.0)
            normalized = (observed.values - reference.values) / scale
            mask = np.broadcast_to([is_smoothable_metric(name) for name in observed.metric_names],
                                   observed.values.shape)
            if level not in self.tensor_filters:
                self.tensor_filters[level] = DiagonalKalmanFilter(normalized.size,
                    process_variance=self.kalman_filter.process_variance,
                    measurement_variance=self.kalman_filter.measurement_variance)
            filtered = self.tensor_filters[level].update(normalized.ravel(), mask.ravel())
            values = np.where(mask, reference.values + filtered.reshape(normalized.shape) * scale,
                              observed.values)
            matrices[level] = TensorLevelMatrix(level, observed.entity_ids, observed.metric_names, values)
        return TensorMatrixView(observation.levels, matrices)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        return (vector - self.reference_vector) / self.scale_vector

    def _denormalize(self, normalized: np.ndarray) -> np.ndarray:
        return self.reference_vector + normalized * self.scale_vector

    def _numeric_origins(self, normalized: np.ndarray, *, nominal: bool) -> list[str]:
        """Определить источник численного прогноза до наблюдения его цели."""
        origins = []
        for i, name in enumerate(self.metric_names):
            if name in THREAT_METRICS:
                origin = "persistence_untrained_threat_coordinate"
            elif not nominal:
                origin = "persistence_non_nominal_observation"
            elif self.online_gram[i, i] <= INPUT_SINGULAR_VALUE_TOLERANCE ** 2:
                origin = "persistence_unexcited_coordinate"
            elif not self.observed_input_low[i] <= normalized[i] <= self.observed_input_high[i]:
                origin = "persistence_outside_observed_range"
            else:
                origin = "observed_diagonal_dmd"
            origins.append(origin)
        return origins

    def _update_observed_operator(self, previous: np.ndarray, current: np.ndarray) -> bool:
        # Рабочий оператор учится и на небольших изменениях исправного трафика.
        # reference_vector и ideal_operator остаются неизменным эталоном t0.
        self.online_gram += np.outer(previous, previous)
        self.online_cross += np.outer(current, previous)
        self.observed_input_low = np.minimum(self.observed_input_low, previous)
        self.observed_input_high = np.maximum(self.observed_input_high, previous)
        # R имеет те же сингулярные числа, что и все накопленные входные строки.
        self.observed_input_qr = np.linalg.qr(np.vstack((self.observed_input_qr, previous)), mode="r")
        prior = np.diag(self.ideal_operator)
        xx = self.online_gram.diagonal()
        coefficients = (self.online_cross.diagonal() + DIAGONAL_RIDGE * prior) / (xx + DIAGONAL_RIDGE)
        coefficients = np.where(xx > INPUT_SINGULAR_VALUE_TOLERANCE ** 2, coefficients, prior)
        candidate = np.diag(np.clip(coefficients, 0.0, MAXIMUM_NOMINAL_COEFFICIENT))
        self.observed_operator = _stabilize_threat_dimensions(candidate, self.metric_names)
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
) -> None:
    """Сохранить прогноз и рекомендации; не создавать управляющих планов."""
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
    # Историческое имя оси сохранено; значение описывает только диагностику.
    new_pressure = round(new_pressure, 6)
    analysis["remap_pressure"] = new_pressure
    attacks = snapshot.get("attacks", {})
    confirmed = bool(attacks.get("active") or koopman.get("forecast_is_early_warning"))
    card = build_trigger_card(
        hausdorff=float(analysis.get("state_hausdorff", {}).get("normalized_distance", 0.0)),
        risk=float(koopman["forecast_risk_score"]), lyapunov_pressure=stability_pressure,
        decision_pressure=new_pressure, confirmed=confirmed,
        attack_kind=str(koopman.get("forecast_attack_kind", "none")),
        # Точка входа в модель, особенно у отражённого трафика, не подтверждает источник.
        source_confirmed=False,
    )
    arbitrator["trigger_card"] = card
    without_energy = build_trigger_card(
        hausdorff=float(analysis.get("state_hausdorff", {}).get("normalized_distance", 0.0)),
        risk=float(koopman["forecast_risk_score"]), lyapunov_pressure=0.0,
        decision_pressure=max(current_pressure, predictive_pressure), confirmed=confirmed,
        attack_kind=card["attack_kind"], source_confirmed=False,
    )
    analysis["response_stage_without_lyapunov"] = without_energy["response_stage"]
    recommendation = dict(koopman["arbitrator_recommendation"])
    recommendation.update(
        action=card["action"], response_stage=card["response_stage"],
        attack_kind=card["attack_kind"],
        reason_ru=card["reason_ru"],
        target_ids=list(koopman.get("forecast_target_ids", [])),
        forecast_window_start_seconds=koopman.get("forecast_window_start_seconds"),
        forecast_window_end_seconds=koopman.get("forecast_window_end_seconds"),
        early_warning=bool(koopman.get("forecast_is_early_warning")),
        risk_score=float(koopman["forecast_risk_score"]),
        probability_is_calibrated=False,
    )
    arbitrator["recommendation"] = recommendation
    if model is not None:
        arbitrator["game_theory"] = evaluate_defense_game(
            model, target_ids=list(koopman.get("forecast_target_ids", [])) or list(koopman.get("current_attack_target_ids", [])),
            threat_pressure=max(float(koopman["forecast_risk_score"]), new_pressure),
            matrix_koopman_pressure=matrix_pressure,
            sla_tiers=attacks.get("sla_restoration", {}).get("tiers"),
        )


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
    raw_current: np.ndarray,
    numeric_origins: list[str],
    arrival_index: int | None,
    arrival_fit: dict[str, Any],
    physical_domains: dict[str, dict[str, float | None]],
    observed_trend: np.ndarray,
    attack_state: dict[str, Any],
    target_ids: list[str],
    precursor_confirmed: bool,
    precursor_kind: str,
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
) -> list[dict[str, Any]]:
    """Численная телеметрия и отдельный риск по наблюдаемым признакам."""
    forecasts: list[dict[str, Any]] = []
    z_dmd_only = z_current.copy()
    z_trend = z_current.copy()
    learned = np.asarray([origin == "observed_diagonal_dmd" for origin in numeric_origins])
    lower = np.array([physical_domains[name]["lower"] if physical_domains[name]["lower"] is not None
                      else -np.inf for name in metric_names])
    upper = np.array([physical_domains[name]["upper"] if physical_domains[name]["upper"] is not None
                      else np.inf for name in metric_names])
    threat_indices = [index for index, name in enumerate(metric_names) if name in THREAT_METRICS]
    current_kind = _current_attack_kind(attack_state)
    predicted_kind = precursor_kind if precursor_confirmed or current_kind == "none" else current_kind
    precursor_observed = _observable_precursor_present(attack_state, precursor_score)

    for horizon_steps in range(1, max(1, maximum_horizon_steps) + 1):
        z_dmd_only = operator @ z_dmd_only
        raw_dmd = reference_vector + z_dmd_only * scale_vector
        dmd_prediction = np.clip(raw_dmd, lower, upper)
        numeric_candidate = np.where(learned, raw_dmd, raw_current)
        if arrival_index is not None and arrival_fit["transition_count"]:
            mean_pps = arrival_fit["mean_of_pair_endpoints_pps"]
            numeric_candidate[arrival_index] = mean_pps
            if numeric_origins[arrival_index] == "observed_affine_arrival_dmd":
                slope = arrival_fit["lag_one_coefficient"]
                numeric_candidate[arrival_index] += slope**horizon_steps * (raw_current[arrival_index] - mean_pps)
        numeric = np.clip(numeric_candidate, lower, upper)
        changed = np.flatnonzero(numeric != numeric_candidate)
        dmd_distance = _rms((dmd_prediction - reference_vector) / scale_vector)
        dmd_attack_score = _attack_score(metric_names, dmd_prediction)
        dmd_risk = _forecast_risk(
            deviation=dmd_distance,
            residual=residual / np.sqrt(float(horizon_steps)),
            operator_drift=operator_drift,
            attack_score=dmd_attack_score,
            precursor_only=precursor_score > 0.0 and active_attack_score == 0.0,
            threat_proximity_risk=0.0,
        )
        operator_projection = operator @ z_trend
        trend_projection = z_current + observed_trend * horizon_steps
        trend_weight = min(0.78, 0.38 + 0.08 * horizon_steps)
        candidate = operator_projection.copy()
        for index in threat_indices:
            candidate[index] = (
                (1.0 - trend_weight) * operator_projection[index]
                + trend_weight * trend_projection[index]
            )
        trend_prediction = np.clip(reference_vector + candidate * scale_vector, lower, upper)
        z_trend = (trend_prediction - reference_vector) / scale_vector
        trend_distance = _rms(z_trend)
        trend_attack_score = _attack_score(metric_names, trend_prediction)
        trend_risk = _forecast_risk(
            deviation=trend_distance,
            residual=residual / np.sqrt(float(horizon_steps)),
            operator_drift=operator_drift,
            attack_score=trend_attack_score,
            precursor_only=precursor_score > 0.0 and active_attack_score == 0.0,
            threat_proximity_risk=0.0,
        )
        predicted = _inject_precursor_forecast(
            metric_names,
            trend_prediction,
            attack_state,
            precursor_score=precursor_score,
            horizon_steps=horizon_steps,
        )
        predicted = np.clip(predicted, lower, upper)
        predicted_distance = _rms((predicted - reference_vector) / scale_vector)
        predicted_attack_score = _attack_score(metric_names, predicted)
        hybrid_inputs = dict(
            deviation=predicted_distance,
            residual=residual / np.sqrt(float(horizon_steps)),
            operator_drift=operator_drift,
            attack_score=predicted_attack_score,
            precursor_only=precursor_score > 0.0 and active_attack_score == 0.0,
        )
        rule_risk = _forecast_risk(**hybrid_inputs, threat_proximity_risk=0.0)
        proximity = threat_proximity_risk if precursor_observed or active_attack_score > 0.0 else 0.0
        risk = _forecast_risk(**hybrid_inputs, threat_proximity_risk=proximity)
        forecasts.append(
            {
                "horizon_steps": horizon_steps,
                "horizon_seconds": horizon_steps * step_seconds,
                "target_step": step_index + horizon_steps,
                "target_time_seconds": time_seconds + horizon_steps * step_seconds,
                "predicted_vector": numeric.tolist(),
                "dmd_raw_vector": raw_dmd.tolist(),
                "dmd_projected_vector": dmd_prediction.tolist(),
                "physical_projection": {
                    "corrected_metric_count": len(changed),
                    "correction_by_metric": {
                        metric_names[i]: float(numeric[i] - numeric_candidate[i]) for i in changed
                    },
                },
                # Ряд риска используют для калибровки. Сохраняем то же
                # значение, которое ниже сравнивается с порогом предупреждения.
                "risk_score": risk,
                "koopman_dmd_only_risk_score": round(dmd_risk, 6),
                "koopman_plus_delay_trend_risk_score": round(trend_risk, 6),
                "precursor_classifier_risk_increment": round(max(0.0, rule_risk - trend_risk), 6),
                "predicted_attack_score": round(_attack_score(metric_names, numeric), 6),
                "predicted_reference_distance": round(_rms((numeric - reference_vector) / scale_vector), 6),
                "attack_kind": predicted_kind,
                "target_ids": target_ids,
                "alert_level": _alert_level(risk),
                "is_warning": bool(
                    precursor_observed
                    and precursor_confirmed
                    and risk >= warning_risk_threshold
                ),
                "prediction_origin": (
                    "repeated_koopman_projection_plus_observed_delay_trend_plus_causal_precursor_classifier"
                ),
                "risk_components": {
                    "delay_trend_increment": trend_risk - dmd_risk,
                    "precursor_rule_increment": rule_risk - trend_risk,
                    "target_proximity_increment": risk - rule_risk,
                    "hybrid_attack_score": predicted_attack_score,
                    "hybrid_reference_distance": predicted_distance,
                    "class_and_target_source": "observed_signature_or_current_event",
                },
            }
        )
    return forecasts


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
            if name in THREAT_METRICS:
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
    input_spectrum: dict[str, Any],
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
        "operator_estimator": "constrained_diagonal_dmd_with_synthetic_prior",
        "synthetic_input_spectrum": {**input_spectrum, "sample_count": training_sample_count - 1},
        "physical_domains": _physical_domains(metric_names),
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
        min(1.0, values.get("OBS.dns_query_pressure_ratio", 0.0)),
        min(1.0, values.get("OBS.application_work_pressure_ratio", 0.0)),
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
    # Изменение обученного оператора само по себе не является угрозой.
    # Его вклад уменьшается вместе с отклонением наблюдаемого состояния.
    drift_signal = min(1.0, operator_drift * deviation / 0.25) if active_deviation else 0.0
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
    match = match_attack_signature(signals)
    return match["kind"] if match else "unknown_degradation_or_attack"


def _predicted_attack_kind(attack_state: dict[str, Any], forecast_risk: float, precursor_kind: str) -> str:
    events = attack_state.get("events", []) if isinstance(attack_state, dict) else []
    precursors = attack_state.get("precursors", []) if isinstance(attack_state, dict) else []
    if events or precursors:
        kinds = {
            str(event.get("kind", "unknown"))
            for event in events
        }
        if precursor_kind != "none":
            kinds.update(precursor_kind.split("+"))
        return "+".join(sorted(kinds))
    if forecast_risk >= 0.50:
        return "unknown_degradation_or_attack"
    if forecast_risk >= 0.25:
        return "watch"
    return "none"


def _predicted_precursor_kind(attack_state: dict[str, Any], classified_records=None) -> str:
    precursors = attack_state.get("precursors", []) if isinstance(attack_state, dict) else []
    if not precursors:
        return "none"
    if classified_records is not None:
        return "+".join(sorted({item["kind"] for item in classified_records})) or "unknown_degradation_or_attack"
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
    warning_confirmed: bool,
    lyapunov_remap_pressure: float,
) -> dict[str, Any]:
    """Описать человеку проверку и возможные меры без исполнения в модели."""
    steps = []
    if warning_confirmed or attack_state.get("precursors"):
        steps.append("Проверьте показатели и журналы наблюдаемого узла. Сопоставьте соседние отсчёты.")
    primary_index = 0
    if warning_confirmed:
        class_review = "Оцените влияние на услуги: сначала золотой, затем серебряный и бронзовый классы."
        steps.append(class_review)
        specific_start = len(steps)
        if any(kind in predicted_attack.split("+") for kind in ("dos", "ddos")):
            steps.append("Сверьте рост входящего трафика с журналами узла и подтвердите источник до решения об ограничении.")
        if "ddos" in predicted_attack:
            steps.append("Согласуйте с оператором вышестоящей сети очистку атакующего трафика.")
        if "syn_flood" in predicted_attack:
            steps.append("Проверьте очередь незавершённых соединений и настройки защиты SYN.")
        if "dns_flood" in predicted_attack:
            steps.append("Проверьте частоту DNS-запросов и журнал службы; рассмотрите ограничения запросов после подтверждения источника.")
        if "dhcp_flood" in predicted_attack:
            steps.append("Проверьте журнал выдачи адресов, загрузку службы DHCP и точки, на которых наблюдаются запросы.")
        if "application_flood" in predicted_attack:
            steps.append("Проверьте тяжёлые запросы медиаприложения и загрузку процессора; рассмотрите лимиты дорогих операций после подтверждения источника.")
        if "brute_force" in predicted_attack:
            steps.append("Проверьте ошибки входа; рассмотрите многофакторную аутентификацию и правила блокировки учётных записей.")
        if "power_attack" in predicted_attack:
            steps.append("Проверьте журнал управления питанием, напряжение и запас энергии ИБП (источника бесперебойного питания).")
        for kind, instruction in {
            "bgp_hijack": "Сверьте происхождение префикса с RPKI (реестром полномочий на объявление адресов) и журналом маршрутизации.",
            "privilege_escalation": "Сверьте изменение привилегий с аудитом узла и разрешёнными действиями администратора.",
            "network_discovery": "Сопоставьте обращения к разным службам с расписанием разрешённых проверок.",
            "user_execution": "Сопоставьте время запуска процесса и рост загрузки CPU с действиями пользователя.",
            "exfiltration": "Проверьте объём исходящего потока, назначение и связь с известным каналом управления.",
            "mobile_network_dos": "Сравните доступность мобильного абонента с соседними линиями и картой покрытия.",
        }.items():
            if kind in predicted_attack.split("+"):
                steps.append(instruction)
        primary_index = specific_start if len(steps) > specific_start else specific_start - 1
        steps.append("Подтвердите источник по журналам до решения об ограничении доступа; решение и выполнение остаются за человеком.")
    if lyapunov_remap_pressure > 0.0:
        steps.append("Проверьте рост отклонения состояния по Ляпунову на следующих наблюдениях.")
    if not steps:
        steps.append("Продолжайте штатное наблюдение за качеством услуг.")
    return {
        "decision": "review_threat" if warning_confirmed else "observe",
        "decision_pressure": round(max(risk, lyapunov_remap_pressure), 6),
        "execution_mode": "human_only",
        "requires_human_action": warning_confirmed,
        "primary_action_ru": steps[primary_index],
        "steps_ru": steps,
        "sla_priority_order": ["gold", "silver", "bronze"],
        "sla_observation": _sla_restoration_plan(attack_state),
    }


def _forecast_entities(attack_state: dict[str, Any], history, step_index: int, step_seconds: int) -> dict[str, Any]:
    events = attack_state.get("events", []) if isinstance(attack_state, dict) else []
    precursors = attack_state.get("precursors", []) if isinstance(attack_state, dict) else []
    # Для раннего прогноза допустимы только идентификатор узла, на котором
    # измерен сигнал, и сами сигналы. attack_id/start_step намеренно не читаются.
    for channel, samples in list(history.items()):
        history[channel] = {step: sample for step, sample in samples.items()
                            if step_index - len(LAG_WEIGHTS) < step < step_index}
        if not history[channel]:
            del history[channel]
    next_records = _signature_records(precursors, "precursor", history, step_index, step_seconds)
    signature_records = _signature_records(attack_state.get("signature_observations", []),
                                            "observed_signature", history, step_index, step_seconds)
    current_records = [
        {
            "observation_type": "active_event",
            "attack_id": str(item.get("attack_id", "")),
            "target_id": str(item.get("target_id")),
            "kind": str(item.get("kind", "unknown")),
            **{key: item[key] for key in ("attack_family", "attack_mechanism", "attack_label_ru") if key in item},
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
        "signature_entity_records": signature_records,
    }


def _signature_records(observations, observation_type, history, step_index, step_seconds):
    records = []
    for item in observations:
        sensor_id = item.get("sensor_node_id")
        if not sensor_id:
            continue
        signals = item.get("signals", {})
        samples = history.setdefault((observation_type, str(sensor_id)), {})
        previous = [samples.get(step_index - lag) for lag in range(1, len(LAG_WEIGHTS))]
        match = match_attack_signature(signals, history=previous, step_seconds=step_seconds)
        # Два показания в одной точке в тот же шаг нельзя склеивать в одну сигнатуру.
        samples[step_index] = None if step_index in samples else dict(signals)
        if match is None:
            continue
        kind = match["kind"]
        records.append({
            "observation_type": observation_type,
            "target_id": str(sensor_id),
            "kind": kind,
            "feature_match": match,
            **{key: match[key] for key in ("attack_family", "attack_mechanism", "attack_label_ru") if key in match},
            "mitre_technique_candidates": [mapping.technique_id for mapping in MITRE_TENSOR_MAPPINGS
                                           if mapping.attack_kind.value == kind],
            "source_ids": sorted(str(source) for source in item.get("observed_source_node_ids", []) if source),
        })
    return records


def _sla_restoration_plan(attack_state: dict[str, Any]) -> dict[str, Any]:
    current_tiers = {
        item.get("sla_grade"): item
        for item in attack_state.get("sla_restoration", {}).get("tiers", [])
    }
    tiers = []
    actions = {
        "gold": "review_gold_first",
        "silver": "review_silver_after_gold",
        "bronze": "review_bronze_after_higher_priorities",
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
        "policy": "human_review_in_sla_priority_order",
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
