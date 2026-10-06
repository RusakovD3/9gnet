"""Dense matrix views of every GNet9 state tensor.

``StateTensor`` is deliberately a small named one-dimensional vector attached
to one graph entity.  That is convenient for the graph and for GraphML.  This
module makes the complementary view needed by numerical algorithms: for every
level it creates a stable matrix ``entities x named metrics`` without dropping
any tensor value.  The matrices remain in memory; exports may contain either a
summary or the complete values, depending on the snapshot-detail setting.

The Koopman channel below uses a *factorised* Kronecker operator.  ``numpy.kron``
is used to verify the exact dense equivalent for small states, but the normal
propagation is ``A @ Z @ B.T``.  Materialising ``kron(A, B)`` for a large state
would multiply dimensions and defeat the purpose of the factorisation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np


TENSOR_MATRIX_LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "EDGE")
KRONECKER_FEATURE_NAMES = (
    "mean_signed_normalized_delta",
    "rms_normalized_delta",
    "maximum_absolute_normalized_delta",
    "changed_value_share",
)


@dataclass(frozen=True)
class TensorLevelMatrix:
    """One dense matrix with stable entity and metric axes."""

    level: str
    entity_ids: tuple[str, ...]
    metric_names: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class TensorMatrixView:
    """Matrix representation of all current graph tensors."""

    levels: tuple[str, ...]
    matrices: dict[str, TensorLevelMatrix]


def build_tensor_matrix_view(tensor_state: dict[str, Any]) -> TensorMatrixView:
    """Build ``entity x metric`` matrices from the complete tensor snapshot.

    Metric names are retained as columns, not position-only values.  A missing
    metric in a heterogeneous level is represented by zero, exactly as a
    tensor factory represents an omitted optional metric.
    """
    matrices: dict[str, TensorLevelMatrix] = {}
    levels = tuple(str(level) for level in tensor_state.get("levels", TENSOR_MATRIX_LEVELS))
    by_level = tensor_state.get("by_level", {})
    for level in levels:
        items = sorted(by_level.get(level, []), key=_tensor_item_identity)
        metric_names = _metric_axis(items)
        entity_ids = tuple(_tensor_item_identity(item) for item in items)
        values = np.zeros((len(items), len(metric_names)), dtype=float)
        column_index = {name: index for index, name in enumerate(metric_names)}
        for row_index, item in enumerate(items):
            for name, value in item.get("metrics", {}).items():
                if name in column_index:
                    values[row_index, column_index[name]] = float(value)
        values.setflags(write=False)
        matrices[level] = TensorLevelMatrix(
            level=level,
            entity_ids=entity_ids,
            metric_names=metric_names,
            values=values,
        )
    return TensorMatrixView(levels=levels, matrices=matrices)


def tensor_matrix_export(view: TensorMatrixView, *, include_values: bool) -> dict[str, Any]:
    """Return JSON-safe matrix metadata, optionally with every raw value."""
    payload: dict[str, Any] = {
        "representation": "dense_entity_by_named_metric_matrices",
        "all_tensor_values_preserved": True,
        "levels": {},
    }
    for level in view.levels:
        matrix = view.matrices[level]
        values = matrix.values
        item: dict[str, Any] = {
            "shape": [int(values.shape[0]), int(values.shape[1])],
            "entity_ids": list(matrix.entity_ids),
            "metric_names": list(matrix.metric_names),
            "value_count": int(values.size),
            "frobenius_norm": round(float(np.linalg.norm(values, ord="fro")), 8),
            "sha256": _array_fingerprint(values),
        }
        if include_values:
            item["values"] = [[round(float(value), 10) for value in row] for row in values]
        payload["levels"][level] = item
    return payload


def kronecker_observables(
    reference: TensorMatrixView,
    current: TensorMatrixView,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Reduce complete per-level matrices to a small, auditable observable grid.

    Every value participates in the four statistics.  The reduction is only a
    numerical observable for Koopman, not a replacement for stored matrices.
    """
    level_names = tuple(level for level in reference.levels if level in current.matrices)
    values = np.zeros((len(level_names), len(KRONECKER_FEATURE_NAMES)), dtype=float)
    for row_index, level in enumerate(level_names):
        baseline, observed = _aligned_level_values(
            reference.matrices[level], current.matrices[level]
        )
        if baseline.size == 0:
            continue
        normalized_delta = (observed - baseline) / np.maximum(np.abs(baseline), 1.0)
        absolute = np.abs(normalized_delta)
        values[row_index] = (
            float(np.mean(normalized_delta)),
            float(np.sqrt(np.mean(np.square(normalized_delta)))),
            float(np.max(absolute)),
            float(np.mean(absolute > 0.01)),
        )
    return level_names, values


def apply_factorised_kronecker(
    level_operator: np.ndarray,
    feature_operator: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Apply ``level_operator ⊗ feature_operator`` without building it."""
    return level_operator @ state @ feature_operator.T


def materialize_kronecker_operator(
    level_operator: np.ndarray,
    feature_operator: np.ndarray,
) -> np.ndarray:
    """Return the exact dense operator for diagnostics and small-state tests."""
    dimension = level_operator.shape[0] * feature_operator.shape[0]
    columns = level_operator.shape[1] * feature_operator.shape[1]
    if dimension * columns > 1_000_000:
        raise ValueError("Dense Kronecker diagnostic exceeds one million elements")
    return np.kron(level_operator, feature_operator)


@dataclass(frozen=True)
class LazyKroneckerOperator:
    """Действие A ⊗ B на вектор с порядком элементов по строкам (C)."""

    left: np.ndarray
    right: np.ndarray

    def __post_init__(self) -> None:
        for name in ("left", "right"):
            factor = np.array(getattr(self, name), dtype=float, copy=True)
            if factor.ndim != 2 or not all(factor.shape) or not np.isfinite(factor).all():
                raise ValueError("Kronecker factors must be finite nonempty matrices")
            factor.setflags(write=False)
            object.__setattr__(self, name, factor)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.left.shape[0] * self.right.shape[0],
                self.left.shape[1] * self.right.shape[1])

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.shape[1],):
            raise ValueError(f"Expected a vector of length {self.shape[1]}")
        state = vector.reshape(self.left.shape[1], self.right.shape[1])
        return apply_factorised_kronecker(self.left, self.right, state).reshape(-1)

    def rmatvec(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.shape[0],):
            raise ValueError(f"Expected a vector of length {self.shape[0]}")
        state = vector.reshape(self.left.shape[0], self.right.shape[0])
        return (self.left.T @ state @ self.right).reshape(-1)


@dataclass
class TensorKroneckerKoopman:
    """A bounded structured Koopman residual channel for tensor matrices.

    The existing compact DMD predictor remains the attack-warning channel.  This
    companion channel consumes the complete tensor-matrix observation and
    reports whether its structured evolution is consistent with the previous
    observation.  It is intentionally not trained on synthetic attacks.
    """

    reference: TensorMatrixView
    level_names: tuple[str, ...]
    level_operator: np.ndarray
    feature_operator: np.ndarray
    previous_observables: np.ndarray | None = None

    @classmethod
    def from_reference(cls, reference: TensorMatrixView, model=None) -> "TensorKroneckerKoopman":
        level_names, observables = kronecker_observables(reference, reference)
        level_count, feature_count = observables.shape
        # Связи уровней берутся из показателей одного объекта и концов
        # одной связи сети. Сила переноса 0,04 — настройка модели.
        level_operator = 0.93 * np.eye(level_count, dtype=float)
        if model is not None:
            coupling = np.zeros((level_count, level_count), dtype=float)
            index = {level: offset for offset, level in enumerate(level_names)}
            entities = [attrs for _, attrs in model.graph.nodes(data=True)]
            entities.extend(attrs for _, _, attrs in model.graph.edges(data=True))
            for attrs in entities:
                levels = {getattr(value, "level", None) for value in attrs.values()}
                present = [index[level] for level in levels if level in index]
                for source in present:
                    for target in present:
                        if source != target:
                            coupling[source, target] += 1.0
            row_sum = coupling.sum(axis=1, keepdims=True)
            coupling /= np.maximum(row_sum, 1.0)
            level_operator += 0.04 * coupling
        elif level_count:
            level_operator += 0.04 * np.ones((level_count, level_count), dtype=float) / level_count
        feature_operator = np.diag((0.96, 0.92, 0.90, 0.88)[:feature_count]).astype(float)
        if feature_count > 1:
            feature_operator += 0.01 * np.ones((feature_count, feature_count), dtype=float) / feature_count
        return cls(
            reference=reference,
            level_names=level_names,
            level_operator=level_operator,
            feature_operator=feature_operator,
        )

    def analyze(self, current: TensorMatrixView, *, verify_dense: bool = False) -> dict[str, Any]:
        """Evaluate a current full-matrix observation using factorised algebra."""
        current_levels, observables = kronecker_observables(self.reference, current)
        if current_levels != self.level_names:
            raise ValueError("Tensor matrix levels changed after Koopman reference was created")
        previous = self.previous_observables
        if previous is None:
            previous = np.zeros_like(observables)
        operator = LazyKroneckerOperator(self.level_operator, self.feature_operator)
        predicted = operator.matvec(previous.reshape(-1)).reshape(previous.shape)
        residual = observables - predicted
        state_dimension = int(observables.size)
        parameter_count = int(self.level_operator.size + self.feature_operator.size)
        dense_parameter_count = int(state_dimension * state_dimension)
        dense_equivalence_error = None
        # На небольшом состоянии сверяем два способа расчёта. Большую
        # матрицу не создаём: она нужна только для этой проверки.
        if verify_dense and state_dimension <= 256:
            dense = materialize_kronecker_operator(self.level_operator, self.feature_operator)
            dense_prediction = dense @ previous.reshape(-1)
            dense_equivalence_error = float(
                np.max(np.abs(dense_prediction - predicted.reshape(-1)))
            )
        self.previous_observables = observables.copy()
        return {
            "model": "factorised_kronecker_tensor_koopman",
            "input": "complete_entity_by_metric_tensor_matrices",
            "level_names": list(self.level_names),
            "level_operator": self.level_operator.tolist(),
            "feature_names": list(KRONECKER_FEATURE_NAMES),
            "observable_shape": [int(observables.shape[0]), int(observables.shape[1])],
            "state_dimension": state_dimension,
            "factorised_parameter_count": parameter_count,
            "dense_equivalent_parameter_count": dense_parameter_count,
            "parameter_reduction_ratio": round(
                1.0 - parameter_count / max(dense_parameter_count, 1), 6
            ),
            "propagation": "level_operator @ observable @ feature_operator.T",
            "numpy_kron_used_for_small_state_equivalence_check": dense_equivalence_error is not None,
            "dense_equivalence_max_abs_error": (
                None if dense_equivalence_error is None else round(dense_equivalence_error, 12)
            ),
            "reference_distance": round(float(np.linalg.norm(observables, ord="fro")), 8),
            "one_step_residual": round(float(np.sqrt(np.mean(np.square(residual)))), 8),
            "maximum_residual": round(float(np.max(np.abs(residual))) if residual.size else 0.0, 8),
            "per_level": {
                level: {
                    "state_vector": observables[index].tolist(),
                    "predicted_vector": predicted[index].tolist(),
                    "residual_rms": float(np.sqrt(np.mean(residual[index] ** 2))),
                }
                for index, level in enumerate(self.level_names)
            },
            "advisory_only": True,
            "semantics_ru": (
                "Структурированная проверка согласованности полных матриц тензоров; "
                "не является калиброванной вероятностью атаки и не отменяет подтверждение телеметрией."
            ),
        }


def _tensor_item_identity(item: dict[str, Any]) -> str:
    if item.get("scope") == "edge":
        source, target = sorted((str(item.get("source", "")), str(item.get("target", ""))))
        return f"edge:{source}--{target}:{item.get('tensor_name', 'tensor')}"
    return f"node:{item.get('node_id', '')}:{item.get('tensor_name', 'tensor')}"


def _metric_axis(items: list[dict[str, Any]]) -> tuple[str, ...]:
    result: list[str] = []
    for item in items:
        for name in item.get("metrics", {}):
            if name not in result:
                result.append(str(name))
    return tuple(result)


def _aligned_level_values(
    baseline: TensorLevelMatrix,
    observed: TensorLevelMatrix,
) -> tuple[np.ndarray, np.ndarray]:
    if baseline.entity_ids == observed.entity_ids and baseline.metric_names == observed.metric_names:
        return baseline.values, observed.values
    baseline_rows = {name: index for index, name in enumerate(baseline.entity_ids)}
    observed_rows = {name: index for index, name in enumerate(observed.entity_ids)}
    baseline_columns = {name: index for index, name in enumerate(baseline.metric_names)}
    observed_columns = {name: index for index, name in enumerate(observed.metric_names)}
    entity_ids = tuple(sorted(set(baseline_rows) | set(observed_rows)))
    metric_names = tuple(sorted(set(baseline_columns) | set(observed_columns)))
    reference_values = np.zeros((len(entity_ids), len(metric_names)), dtype=float)
    current_values = np.zeros_like(reference_values)
    for row_index, entity_id in enumerate(entity_ids):
        for column_index, metric_name in enumerate(metric_names):
            if entity_id in baseline_rows and metric_name in baseline_columns:
                reference_values[row_index, column_index] = baseline.values[
                    baseline_rows[entity_id], baseline_columns[metric_name]
                ]
            if entity_id in observed_rows and metric_name in observed_columns:
                current_values[row_index, column_index] = observed.values[
                    observed_rows[entity_id], observed_columns[metric_name]
                ]
    return reference_values, current_values


def _array_fingerprint(values: np.ndarray) -> str:
    payload = json.dumps(values.tolist(), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
