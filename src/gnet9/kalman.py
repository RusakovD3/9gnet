"""Причинное сглаживание нормированных наблюдений без внешних библиотек."""

from __future__ import annotations

import numpy as np


DEFAULT_PROCESS_VARIANCE = 0.0001
DEFAULT_MEASUREMENT_VARIANCE = 0.0025


def validate_noise_variances(process_variance: float, measurement_variance: float) -> None:
    """Дисперсии нормированных данных: Q ≥ 0, R > 0, обе конечны."""
    if (not np.isfinite([process_variance, measurement_variance]).all()
            or process_variance < 0 or measurement_variance <= 0):
        raise ValueError("Нужны конечные дисперсии Q ≥ 0 и R > 0")


def cholesky_sigma_points(mean: np.ndarray, covariance: np.ndarray):
    """2n+1 точек с весами: центр 0, остальные 1/(2n), смещение √n L.

    Вход — конечное среднее и симметричная положительно определённая
    ковариация в квадрате единиц среднего. Например, для x=2, P=4
    точки [2, 4, 0] восстанавливают среднее 2 и дисперсию 4.
    Нулевая центральная масса соответствует alpha=1, kappa=beta=0.
    Это квадратура моментов, а не гарантия вероятности покрытия.
    """
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if (mean.ndim != 1 or not mean.size or covariance.shape != (mean.size, mean.size)
            or not np.isfinite(mean).all() or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14)):
        raise ValueError("Нужны конечное среднее и симметричная ковариация")
    # Строки смещений — столбцы нижнетреугольного множителя L.
    offsets = np.sqrt(mean.size) * np.linalg.cholesky(covariance).T
    points = np.vstack((mean, mean + offsets, mean - offsets))
    weights = np.full(2 * mean.size + 1, 1.0 / (2 * mean.size))
    weights[0] = 0.0
    return points, weights


class DiagonalKalmanFilter:
    """Независимые случайные блуждания: x(t)=x(t-1)+w, z(t)=x(t)+v.

    Все значения безразмерные, уже приведённые к масштабу показателя.
    Q=0,0001 — дисперсия изменения состояния; R=0,0025 — шум измерения.
    При P=0, R=Q=1 первое измерение имеет усиление 1/2.
    """

    def __init__(self, size: int, *, process_variance: float = DEFAULT_PROCESS_VARIANCE,
                 measurement_variance: float = DEFAULT_MEASUREMENT_VARIANCE) -> None:
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError("Размер фильтра должен быть положительным целым числом")
        validate_noise_variances(process_variance, measurement_variance)
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.state = np.zeros(size)
        self.variance = np.zeros(size)
        self.gain = np.zeros(size)

    def update(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        if observation.shape != self.state.shape or mask.shape != self.state.shape:
            raise ValueError("Размер наблюдения и маски должен совпадать с фильтром")
        if not np.isfinite(observation).all():
            raise ValueError("Наблюдение должно быть конечным")
        prior = self.variance + self.process_variance
        # Холецкий диагональной ковариации инновации: S=L Lᵀ.
        # Два деления на диагональ L заменяют обращение S.
        root = np.sqrt(prior + self.measurement_variance)
        self.gain = np.where(mask, (prior / root) / root, 1.0)
        self.state = np.where(mask, self.state + self.gain * (observation - self.state), observation)
        # Пропущенное через фильтр значение остаётся текущим наблюдением.
        self.variance = np.where(mask, (1.0 - self.gain) * prior, 0.0)
        return self.state.copy()
