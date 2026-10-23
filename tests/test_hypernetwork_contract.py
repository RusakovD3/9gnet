"""Проверки направления вложения и ленивого оператора на ненулевых данных."""

from copy import deepcopy

import numpy as np
import pytest

from src.gnet9.dynamics import _tensor_state_snapshot
from src.gnet9.hypernetwork import hypernetwork_embedding
from src.gnet9.tensor_matrix import (
    LazyKroneckerOperator,
    TensorKroneckerKoopman,
    build_tensor_matrix_view,
)
from src.gnet9.topology_builder import GNetBaselineBuilder


@pytest.fixture(scope="module")
def model():
    return GNetBaselineBuilder().build()


# Требуемая схема задаётся независимо от констант реализации.
EMBEDDING = {
    ("L0", "L1"), ("L1", "L2"),
    ("L2", "L3"), ("L2", "L4"), ("L2", "L6"), ("L2", "L8"),
    ("L3", "L4"), ("L3", "L8"), ("L4", "L8"),
    ("L5", "L2"), ("L5", "L3"), ("L6", "L4"),
}
CONTROLS = {("L7", level) for level in ("L1", "L2", "L3", "L5", "L6")}


def test_hypernetwork_preserves_directed_embedding_and_sla_feedback(model):
    baseline = deepcopy(model.to_serializable())
    embedding = hypernetwork_embedding(model)
    by_kind = {}
    for relation in embedding["relations"]:
        levels = {level for _, level in relation["members"]}
        for dependency in relation["dependencies"]:
            pair = (dependency["source_level"], dependency["target_level"])
            assert set(pair) <= levels
            by_kind.setdefault(relation["kind"], set()).add(pair)
    assert by_kind["sla_observation"] == {("L0", "L7")}
    assert by_kind["control"] == CONTROLS
    actual_embedding = set().union(*(pairs for kind, pairs in by_kind.items()
                                    if kind not in {"control", "sla_observation"}))
    assert actual_embedding == EMBEDDING
    assert sum(r["kind"] == "service_dependency" for r in embedding["relations"]) == 480
    assert model.to_serializable() == baseline


@pytest.mark.parametrize("use_model", [True, False])
def test_numeric_embedding_uses_target_rows_and_no_forbidden_control(model, use_model):
    reference = build_tensor_matrix_view(_tensor_state_snapshot(model))
    channel = TensorKroneckerKoopman.from_reference(reference, model=model if use_model else None)
    index = {level: row for row, level in enumerate(channel.level_names)}
    allowed = EMBEDDING | CONTROLS | {("L0", "L7")}
    actual = {
        (source, target)
        for source, column in index.items()
        for target, row in index.items()
        if row != column and channel.level_operator[row, column] > 0
    }
    assert actual == allowed
    # Импульс управления L7 достигает только разрешённых уровней.
    state = np.zeros((len(index), 4))
    state[index["L7"], 0] = 1
    operator = LazyKroneckerOperator(channel.level_operator, np.eye(4))
    propagated = operator.matvec(state.reshape(-1)).reshape(state.shape)
    assert propagated[index["L1"], 0] > 0
    assert propagated[index["L4"], 0] == 0
    assert propagated[index["L8"], 0] == 0
    # Строковая сумма ограничивает перенос; EDGE не получает выдуманных связей.
    assert np.max(channel.level_operator.sum(axis=1)) <= 0.970000000001
    np.testing.assert_array_equal(channel.level_operator[index["EDGE"]],
                                  np.eye(len(index))[index["EDGE"]] * 0.93)


def test_lazy_kronecker_known_solution_without_materialisation(monkeypatch):
    operator = LazyKroneckerOperator(np.array([[1, 2], [3, 4]]),
                                     np.array([[2, 0], [1, -1]]))
    def forbidden(*args, **kwargs):
        raise AssertionError("Ленивый расчёт создал полную матрицу")
    monkeypatch.setattr(np, "kron", forbidden)
    np.testing.assert_array_equal(operator.matvec(np.array([1, 2, 3, 4])),
                                  [14, -3, 30, -7])


def test_lazy_operator_handles_independent_row_and_column_permutations(monkeypatch):
    left = np.array([[1, 2, -3], [4, -2, 1]], dtype=float)
    right = np.array([[2, 5], [1, -1], [3, 2]], dtype=float)
    # Перестановки независимо меняют выходные строки и входные столбцы.
    left_rows, left_columns = [1, 0], [2, 0, 1]
    right_rows, right_columns = [2, 0, 1], [1, 0]
    rows = np.arange(6).reshape(2, 3)[np.ix_(left_rows, right_rows)].reshape(-1)
    columns = np.arange(6).reshape(3, 2)[np.ix_(left_columns, right_columns)].reshape(-1)
    dense = np.kron(left, right)[np.ix_(rows, columns)]
    operator = LazyKroneckerOperator(left[np.ix_(left_rows, left_columns)],
                                     right[np.ix_(right_rows, right_columns)])
    vector = np.array([1, -2, 3, 4, -1, 2], dtype=float)
    def forbidden(*args, **kwargs):
        raise AssertionError("Ленивый расчёт создал полную матрицу")
    monkeypatch.setattr(np, "kron", forbidden)
    np.testing.assert_allclose(operator.matvec(vector), dense @ vector)
    np.testing.assert_allclose(operator.rmatvec(vector), dense.T @ vector)
