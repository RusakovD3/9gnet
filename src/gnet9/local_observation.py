"""Локальная очистка и диагностика текущей телеметрии без глобального тензора."""

from __future__ import annotations

import numpy as np

from .kalman import (cholesky_sigma_points, validate_noise_variances,
                     DEFAULT_PROCESS_VARIANCE, DEFAULT_MEASUREMENT_VARIANCE)
from .runtime_metrics import VERTEX_METRICS, EDGE_METRICS


# Маски — инженерные гипотезы стенда, а не сигнатуры, заданные MITRE.
# Нет измерений аутентификации/напряжения в этих осях: соответствующие
# техники остаются в отдельном канале наблюдений, без выдуманных признаков.
TRAFFIC_MASKS = {
    "T1498.001": {"normal_arrival_rate_pps", "utilization_ratio", "mean_queue_packets",
                  "mean_delay_ms", "delay_variance_ms2", "speed_mbps", "loss_probability",
                  "destroyed_packet_fraction", "file_drop_fraction"},
    "T1498.002": {"utilization_ratio", "mean_queue_packets", "mean_delay_ms",
                  "speed_mbps", "loss_probability", "destroyed_packet_fraction", "file_drop_fraction"},
    "T1499.001": {"utilization_ratio", "mean_queue_packets", "buffer_memory_mib",
                  "syn_ack_fraction", "mean_delay_ms", "destroyed_packet_fraction"},
}
PRESERVED = {"attack_profile", "negative_arrival_rate_pps", "destroyed_packet_fraction",
             "loss_probability", "file_drop_fraction", "syn_ack_fraction", "reliability_ratio"}
INNOVATION_CHI_SQUARE_THRESHOLD = 24.7
WINDOW_RESET_INTERVAL_STEPS = 10
MINIMUM_COVARIANCE_SAMPLES = 4
INNOVATION_COVARIANCE_RATIO_LIMIT = 9.0
FILTER_STATUS_CODES = {
    "updated": 0, "innovation_rejected": 1, "non_nominal_passthrough": 2, "covariance_reset": 3,
}


def is_smoothable_metric(name: str) -> bool:
    """Признаки угроз и качества услуги проходят точно, включая оси уровней."""
    metric = name.lower()
    return not PRESERVED.intersection(metric.split(".")) and not any(part in metric for part in (
        "loss", "drop", "destroy", "attack", "sla", "authentication", "syn_",
        "voltage", "power_control", "battery_discharge", "energy_reserve",
    ))


def _axes(state, kind):
    """Одинаковые оси для подробного и компактного экспорта пакетов."""
    if kind == "vertices":
        names = tuple(state["vertex_metric_names"])
        if "nodes" in state:
            rows = [(str(n["id"]), [n["metrics"][m] for m in names]) for n in state["nodes"]]
        else:
            rows = list(zip(state["node_ids"], state["vertex_vectors"]))
    else:
        names = tuple(state["edge_metric_names"])
        if "edges" in state:
            rows = [(tuple(sorted((e["source"], e["target"]))),
                     [e["metrics"][m] for m in names]) for e in state["edges"]]
        else:
            rows = [(tuple(sorted(e)), v) for e, v in zip(state["edge_ids"], state["edge_vectors"])]
    rows.sort(key=lambda row: row[0])
    values = np.asarray([row[1] for row in rows], dtype=float).reshape(len(rows), len(names))
    return names, tuple(row[0] for row in rows), values


def _reset_window(block, rows):
    """Начать оценку ковариации заново с последним исправным средним."""
    for field in ("variance", "window_count", "window_mean", "window_m2"):
        block[field][rows] = 0


def _innovation_covariance_ratio(block, whitened, accepted):
    """Максимальная диагональ выборочной ковариации белых невязок.

    Стандартизованные невязки уже делены на √S; при согласованной модели
    их дисперсия около 1. Например, [3, -3, 3, -3] даёт 12.
    """
    count = block["window_count"]
    mean = block["window_mean"]
    m2 = block["window_m2"]
    sample = whitened[accepted][:, block["smoothable"]]
    count[accepted] += 1
    delta = sample - mean[accepted]
    mean[accepted] += delta / count[accepted, None]
    m2[accepted] += delta * (sample - mean[accepted])
    ratio = np.max(m2, axis=1, initial=0.) / np.maximum(count - 1, 1)
    return np.where(accepted & (count >= MINIMUM_COVARIANCE_SAMPLES), ratio, 0.)


class LocalObservationFilter:
    """Банк независимых локальных фильтров с диагональной ковариацией.

    Нормировка относительно неизменяемого t0: масштаб max(|t0|, 1).
    Q=1e-4 и R=2.5e-3 относятся к квадрату нормированных единиц за шаг.
    Корреляции разных объектов не предполагаются; матрица сети не строится.
    """

    def __init__(self, reference, *, process_variance=DEFAULT_PROCESS_VARIANCE,
                 measurement_variance=DEFAULT_MEASUREMENT_VARIANCE):
        validate_noise_variances(process_variance, measurement_variance)
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.last_step = -1
        self.previous_mode = "nominal"
        self.previous_events = frozenset()
        self.blocks = {}
        for kind, expected in (("vertices", VERTEX_METRICS), ("edges", EDGE_METRICS)):
            names, ids, values = _axes(reference, kind)
            if names != expected or not np.isfinite(values).all():
                raise ValueError("Изменились имена или значения локальных показателей")
            smoothable = np.asarray([is_smoothable_metric(name) for name in names])
            window_shape = (len(ids), int(np.count_nonzero(smoothable)))
            self.blocks[kind] = dict(names=names, ids=ids, reference=values.copy(),
                                     scale=np.maximum(np.abs(values), 1.0),
                                     smoothable=smoothable,
                                     continuous=np.asarray([name != "attack_profile" for name in names]),
                                     mean=np.zeros_like(values), variance=np.zeros_like(values),
                                     window_count=np.zeros(len(ids), dtype=int),
                                     window_mean=np.zeros(window_shape),
                                     window_m2=np.zeros(window_shape))

    def analyze(self, observation, *, nominal: bool, step_index: int | None = None,
                event_active: bool = False, precursor_active: bool = False,
                observed_events: tuple[tuple[str, str, bool], ...] = ()):
        step_index = self.last_step + 1 if step_index is None else step_index
        if step_index <= self.last_step:
            raise ValueError("Шаг локального наблюдения должен расти")
        blocks = {}
        for kind, block in self.blocks.items():
            names, ids, raw = _axes(observation, kind)
            if names != block["names"] or ids != block["ids"] or not np.isfinite(raw).all():
                raise ValueError("Локальные оси должны совпадать с t0")
            blocks[kind] = raw
        if event_active:
            mode = "event"
        elif precursor_active:
            mode = "precursor"
        else:
            mode = "nominal" if nominal else "degraded"
        nominal = mode == "nominal"
        reset_reasons = []
        if step_index > 0 and step_index % WINDOW_RESET_INTERVAL_STEPS == 0:
            reset_reasons.append("timer")
        if mode != self.previous_mode:
            reset_reasons.append("observed_mode_change")
        events = frozenset(observed_events)
        if mode == self.previous_mode == "event" and events != self.previous_events:
            reset_reasons.append("observed_event_change")

        result = {"model": "local_diagonal_cholesky_random_walk",
                  "applied_to_nominal_observation": nominal,
                  "mask_origin": "engineering_hypotheses_not_mitre_calibration",
                  "decision_role": "diagnostic_only",
                  "unsupported_local_techniques": ["T1110.001", "T0831"],
                  "innovation_chi_square_threshold": INNOVATION_CHI_SQUARE_THRESHOLD,
                  "process_variance": self.process_variance,
                  "measurement_variance": self.measurement_variance,
                  "filter_status_codes": dict(FILTER_STATUS_CODES),
                  "window": {
                      "step_index": step_index,
                      "observation_mode": mode,
                      "previous_observation_mode": self.previous_mode,
                      "global_reset_reasons": reset_reasons,
                      "reset_interval_steps": WINDOW_RESET_INTERVAL_STEPS,
                      "minimum_covariance_samples": MINIMUM_COVARIANCE_SAMPLES,
                      "innovation_covariance_ratio_limit": INNOVATION_COVARIANCE_RATIO_LIMIT,
                  }}
        detailed = "nodes" in observation
        for kind, block in self.blocks.items():
            result[kind] = self._analyze_block(
                block, blocks[kind], nominal=nominal, reset=bool(reset_reasons), detailed=detailed,
            )
        self.last_step = step_index
        self.previous_mode = mode
        self.previous_events = events
        return result

    def _analyze_block(self, block, raw, *, nominal, reset, detailed):
        names, ids = block["names"], block["ids"]
        z = (raw - block["reference"]) / block["scale"]
        prior = block["variance"] + self.process_variance
        innovation = z - block["mean"]
        # L=diag(sqrt(P+Q+R)); махаланобисова длина ||L^-1 (M ⊙ y)||.
        root = np.sqrt(prior + self.measurement_variance)
        whitened = innovation / root
        mask_scores = {}
        scores = np.zeros(len(ids))
        for technique, metrics in TRAFFIC_MASKS.items():
            mask = np.asarray([name in metrics for name in names], dtype=bool)
            distances = np.linalg.norm(whitened * mask, axis=1)
            mask_scores[technique] = distances
            scores = np.maximum(scores, distances)
        enabled = block["smoothable"]
        chi_square = np.sum(whitened[:, enabled] ** 2, axis=1)
        rejected = nominal & (chi_square > INNOVATION_CHI_SQUARE_THRESHOLD)
        accepted = nominal & ~rejected
        if reset:
            _reset_window(block, slice(None))
        covariance_ratio = _innovation_covariance_ratio(block, whitened, accepted)
        covariance_reset = covariance_ratio > INNOVATION_COVARIANCE_RATIO_LIMIT
        _reset_window(block, covariance_reset)
        update = (accepted & ~covariance_reset)[:, None] & enabled
        # Текущий сигнал уже оценён с предыдущим окном. Новая ковариация
        # влияет только на обновление исправной оценки и следующий шаг.
        update_prior = block["variance"] + self.process_variance
        gain = update_prior / (update_prior + self.measurement_variance)
        filtered = np.where(update, block["mean"] + gain * innovation, z)
        # Отбрасывается обновление оценки, наблюдение остаётся в снимке.
        block["mean"][update] = filtered[update]
        block["variance"][update] = ((1 - gain) * update_prior)[update]
        # Общая диагональная ковариация внутри блока, одна для обеих выборок.
        # После центрирования все точки t0 имеют вектор 0. При положительной
        # цене несовпадения меток текущее состояние ближе всего к своей метке
        # в t0, обратное направленное расстояние не больше прямого.
        continuous = block["continuous"]
        state_distances = np.zeros(len(ids))
        if ids:
            common_variance = np.mean(prior[:, continuous], axis=0) + self.measurement_variance
            state_distances = np.linalg.norm(z[:, continuous] / np.sqrt(common_variance), axis=1)
        # Только изменившиеся объекты: ограниченный набор для разбора.
        priority = np.maximum(scores, np.sqrt(chi_square))
        priority = np.maximum(priority, np.where(covariance_reset, np.sqrt(covariance_ratio), 0.))
        selected = [i for i in np.argsort(-priority, kind="stable") if priority[i] > 1e-12][:5]
        top = []
        for i in selected:
            # Все оси сохранены; категорические/защищённые значения проходят
            # без сглаживания. R оставляет их неопределённость ненулевой.
            covariance = np.diag(np.where(update[i], block["variance"][i], 0.0)
                                 + self.measurement_variance)
            points, weights = cholesky_sigma_points(filtered[i], covariance)
            # Код категории не является непрерывным измерением. Проекция
            # фиксирует его во всех точках; соответствующая пара совпадёт
            # с центром, а дисперсия этой координаты станет нулевой.
            if "attack_profile" in names:
                category = names.index("attack_profile")
                points[:, category] = filtered[i, category]
            top.append({"entity_id": ids[i], "maximum_mask_distance": float(scores[i]),
                        "innovation_chi_square": float(chi_square[i]),
                        "innovation_update_rejected": bool(rejected[i]),
                        "innovation_covariance_ratio": float(covariance_ratio[i]),
                        "covariance_window_reset": bool(covariance_reset[i]),
                        "mask_distances": {k: float(v[i]) for k, v in mask_scores.items()},
                        "filtered_normalized_vector": filtered[i].tolist(),
                        "sigma_points": points.tolist(), "sigma_weights": weights.tolist()})
        status = np.full(len(ids), FILTER_STATUS_CODES["non_nominal_passthrough"], dtype=int)
        status[accepted] = FILTER_STATUS_CODES["updated"]
        status[rejected] = FILTER_STATUS_CODES["innovation_rejected"]
        status[covariance_reset] = FILTER_STATUS_CODES["covariance_reset"]
        result = {
            "entity_count": len(ids), "entity_ids": ids, "metric_names": list(names),
            "dimension": len(names), "sigma_point_count": 2 * len(names) + 1,
            "preserved_metric_names": [name for name, smooth in zip(names, enabled) if not smooth],
            "filter_status_by_entity": status.tolist(),
            "innovation_chi_square_by_entity": chi_square.round(6).tolist(),
            "innovation_rejected_entity_count": int(np.count_nonzero(rejected)),
            "covariance_reset_entity_count": int(np.count_nonzero(covariance_reset)),
            "hausdorff_mahalanobis_to_t0": float(np.max(state_distances, initial=0.0)),
            "maximum_mask_distance": float(np.max(scores, initial=0.0)),
            "top_deformed_entities": top,
            "filtered_normalized_rms": float(np.sqrt(np.mean(filtered ** 2))) if ids else 0.0,
        }
        if detailed:
            result["filtered_normalized_vectors"] = filtered.tolist()
        return result
