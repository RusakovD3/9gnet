"""Конечная очередь Геленбе: обычные заявки и уничтожающие заявки."""

from __future__ import annotations

import math

import numpy as np


def gelenbe_queue(arrival: float, service: float, negative: float, capacity: int) -> dict[str, float]:
    """λ, μ, ν в пакетах/с; ν удаляет одну заявку из начала непустой очереди.

    Вместимость K включает обслуживание. Стационарные вероятности
    p(n) ∝ (λ/(μ+ν))**n. Для K=1, λ=2, μ=ν=1: обслуживание и уничтожение
    по 0,5 пакета/с, переполнение 1 пакет/с, среднее пребывание 500 мс.
    Это локальная оценка при постоянных интенсивностях текущего шага.
    """
    if not all(math.isfinite(value) for value in (arrival, service, negative)):
        raise ValueError("Интенсивности должны быть конечными")
    if arrival < 0 or service <= 0 or negative < 0 or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("Нужны λ ≥ 0, μ > 0, ν ≥ 0 и целая вместимость K ≥ 1")
    states = np.arange(capacity + 1, dtype=float)
    removal = service + negative
    if not math.isfinite(removal):
        raise ValueError("Суммарная интенсивность завершений должна быть конечной")
    if arrival == 0:
        probability = np.zeros(capacity + 1)
        probability[0] = 1.0
    else:
        log_rho = math.log(arrival) - math.log(removal)
        weights = np.exp((states - (capacity if log_rho > 0 else 0)) * log_rho)
        probability = weights / weights.sum()
    busy = float(probability[1:].sum())
    admitted = removal * busy
    served, destroyed = service * busy, negative * busy
    population = float(states @ probability)
    # При поступлении в неполную очередь заявка видит n перед собой.
    if arrival:
        # Нормируем условие n < K отдельно: при сильной перегрузке все
        # вероятности неполной очереди могут округлиться до нуля, хотя
        # допущенная заявка всё ещё проходит K этапов обслуживания.
        admitted_states = states[:-1]
        offset = capacity - 1 if log_rho > 0 else 0
        admitted_weights = np.exp((admitted_states - offset) * log_rho)
        conditional = admitted_weights / admitted_weights.sum()
        stages = states[:-1] + 1.0
        mean_stages = float(stages @ conditional)
        variance_stages = float((stages - mean_stages) ** 2 @ conditional)
        delay = mean_stages / removal
        delay_variance = (mean_stages + variance_stages) / removal ** 2
    else:
        delay = delay_variance = 0.0
    return {
        "arrival_rate_pps": arrival, "service_rate_pps": service,
        "negative_arrival_rate_pps": negative,
        "blocking_probability": float(probability[-1]),
        "effective_arrival_rate_pps": admitted,
        "served_rate_pps": served, "destroyed_rate_pps": destroyed,
        "overflow_rate_pps": max(0.0, arrival - admitted),
        "destroyed_fraction": destroyed / arrival if arrival else 0.0,
        "busy_probability": busy, "utilization_rho": arrival / removal,
        "mean_system_packets": population,
        "mean_queue_packets": max(0.0, population - busy),
        "mean_system_time_ms": delay * 1000.0,
        "delay_variance_ms2": delay_variance * 1_000_000.0,
        "jitter_ms": math.sqrt(delay_variance) * 1000.0,
    }
