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
    return names, tuple(row[0] for row in rows), np.asarray([row[1] for row in rows], dtype=float)


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
        self.blocks = {}
        for kind, expected in (("vertices", VERTEX_METRICS), ("edges", EDGE_METRICS)):
            names, ids, values = _axes(reference, kind)
            if names != expected or not np.isfinite(values).all():
                raise ValueError("Изменились имена или значения локальных показателей")
            self.blocks[kind] = dict(names=names, ids=ids, reference=values.copy(),
                                     scale=np.maximum(np.abs(values), 1.0),
                                     mean=np.zeros_like(values), variance=np.zeros_like(values))

    def analyze(self, observation, *, nominal: bool):
        result = {"model": "local_diagonal_cholesky_random_walk",
                  "applied_to_nominal_observation": nominal,
                  "mask_origin": "engineering_hypotheses_not_mitre_calibration",
                  "decision_role": "diagnostic_only_not_remap_authorization",
                  "unsupported_local_techniques": ["T1110.001", "T0831"],
                  "process_variance": self.process_variance,
                  "measurement_variance": self.measurement_variance}
        for kind, block in self.blocks.items():
            names, ids, raw = _axes(observation, kind)
            if names != block["names"] or ids != block["ids"] or not np.isfinite(raw).all():
                raise ValueError("Локальные оси должны совпадать с t0")
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
            enabled = np.asarray([is_smoothable_metric(name) for name in names])
            gain = (prior / root) / root
            filtered = z.copy()
            if nominal:
                filtered[:, enabled] = (block["mean"] + gain * innovation)[:, enabled]
                block["mean"][:, enabled] = filtered[:, enabled]
                block["variance"][:, enabled] = ((1 - gain) * prior)[:, enabled]
            # Только изменившиеся объекты: ограниченный набор для разбора.
            selected = [i for i in np.argsort(-scores, kind="stable") if scores[i] > 1e-12][:5]
            top = []
            for i in selected:
                # Все оси сохранены; категорические/защищённые значения проходят
                # без сглаживания. R оставляет их неопределённость ненулевой.
                covariance = np.diag(np.where(enabled & nominal, block["variance"][i], 0.0)
                                     + self.measurement_variance)
                points, weights = cholesky_sigma_points(filtered[i], covariance)
                # Код категории не является непрерывным измерением. Проекция
                # фиксирует его во всех точках; соответствующая пара совпадёт
                # с центром, а дисперсия этой координаты станет нулевой.
                if "attack_profile" in names:
                    category = names.index("attack_profile")
                    points[:, category] = filtered[i, category]
                top.append({"entity_id": ids[i], "maximum_mask_distance": float(scores[i]),
                            "mask_distances": {k: float(v[i]) for k, v in mask_scores.items()},
                            "filtered_normalized_vector": filtered[i].tolist(),
                            "sigma_points": points.tolist(), "sigma_weights": weights.tolist()})
            result[kind] = {
                "entity_count": len(ids), "metric_names": list(names),
                "dimension": len(names), "sigma_point_count": 2 * len(names) + 1,
                "preserved_metric_names": [name for name in names if not is_smoothable_metric(name)],
                "maximum_mask_distance": float(np.max(scores, initial=0.0)),
                "top_deformed_entities": top,
                "filtered_normalized_rms": float(np.sqrt(np.mean(filtered ** 2))),
            }
        return result
