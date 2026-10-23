"""Обычные поступления: число заявок за шаг имеет распределение Пуассона."""

import hashlib

import numpy as np


def poisson_arrivals(expected_count: float, *, node_id: str, step: int, seed: int = 42) -> int:
    """Среднее и дисперсия числа заявок равны λ·Δt = expected_count.

    Поток случайных чисел зависит только от абонента, шага и seed;
    изменение будущего расписания атак не меняет обычный трафик.
    Шаг 0 использует ожидаемое ненулевое число как исправный эталон.
    """
    if not np.isfinite(expected_count) or expected_count < 0 or step < 0:
        raise ValueError("Нужны конечное неотрицательное ожидание и шаг ≥ 0")
    if step == 0:
        return max(1, round(expected_count))
    identity = f"{seed}:{node_id}:{step}".encode("utf-8")
    stream_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
    return int(np.random.default_rng(stream_seed).poisson(expected_count))
