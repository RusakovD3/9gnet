"""Известные моменты, локальные маски и причинная оценка точности."""

import copy

import numpy as np
import pytest

from src.gnet9.kalman import cholesky_sigma_points
from src.gnet9.local_observation import LocalObservationFilter
from src.gnet9.runtime_metrics import VERTEX_METRICS, EDGE_METRICS
from src.gnet9.dynamics import DynamicsConfig, simulate_stationary_dynamics
from src.gnet9.topology_builder import GNetBaselineBuilder
from src.gnet9.koopman import KoopmanOnlineAnalyzer
from src.gnet9.tensor_matrix import TensorLevelMatrix, TensorMatrixView


def test_cholesky_correlated_moments_and_linear_forecast():
    # L=[[2,0],[1,3]], P=[[4,2],[2,10]]; Var(3x-y)=34.
    mean = np.array([2., -1.])
    covariance = np.array([[4., 2.], [2., 10.]])
    points, weights = cholesky_sigma_points(mean, covariance)
    assert points.shape == (5, 2)
    assert weights @ points == pytest.approx(mean)
    delta = points - mean
    assert (delta.T * weights) @ delta == pytest.approx(covariance)
    projected = points @ np.array([3., -1.])
    assert weights @ projected == pytest.approx(7.)
    assert weights @ (projected - 7.) ** 2 == pytest.approx(34.)


@pytest.mark.parametrize("covariance", [np.array([[1., 2.], [0., 1.]]),
    np.array([[1., 2.], [2., 1.]]), np.array([[1., 0.], [0., np.nan]])])
def test_invalid_covariance_is_not_silently_repaired(covariance):
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        cholesky_sigma_points(np.zeros(2), covariance)


def _state():
    return {"vertex_metric_names": list(VERTEX_METRICS),
            "edge_metric_names": list(EDGE_METRICS), "node_ids": ["b", "a"],
            "vertex_vectors": [[1.] * 11, [1.] * 11],
            "edge_ids": [["b", "a"]], "edge_vectors": [[1.] * 9]}


def test_local_mask_known_distance_freeze_and_preserved_metrics():
    reference = _state()
    saved = copy.deepcopy(reference)
    bank = LocalObservationFilter(reference)
    observation = copy.deepcopy(reference)
    observation["vertex_vectors"][0][VERTEX_METRICS.index("mean_delay_ms")] = 2.
    observation["vertex_vectors"][0][VERTEX_METRICS.index("attack_profile")] = 4.
    before = copy.deepcopy(bank.blocks)
    result = bank.analyze(observation, nominal=False)
    top = result["vertices"]["top_deformed_entities"][0]
    assert top["entity_id"] == "b"
    assert top["mask_distances"]["T1498.001"] == pytest.approx(1 / np.sqrt(.0026))
    assert top["filtered_normalized_vector"][8] == 3.
    assert np.asarray(top["sigma_points"])[:, 8] == pytest.approx(np.full(23, 3.))
    assert result["vertices"]["sigma_point_count"] == 23
    assert result["edges"]["sigma_point_count"] == 19
    for kind in bank.blocks:
        assert bank.blocks[kind]["mean"] == pytest.approx(before[kind]["mean"])
        assert bank.blocks[kind]["variance"] == pytest.approx(before[kind]["variance"])
    assert reference == saved


def test_local_causal_prefix_and_axis_permutations():
    reference = _state()
    first, second = LocalObservationFilter(reference), LocalObservationFilter(reference)
    changed = copy.deepcopy(reference)
    changed["vertex_vectors"][0][0] = 1.1
    permuted = copy.deepcopy(changed)
    permuted["node_ids"].reverse()
    permuted["vertex_vectors"].reverse()
    permuted["edge_ids"][0].reverse()
    result = first.analyze(changed, nominal=True)
    assert second.analyze(permuted, nominal=True) == result
    second.analyze(reference, nominal=True)
    assert result["vertices"]["maximum_mask_distance"] == pytest.approx(.1 / np.sqrt(.0026))


def test_pipeline_exports_local_precision_and_preserves_t0():
    model = GNetBaselineBuilder().build()
    original = copy.deepcopy(model.to_serializable())
    simulation = simulate_stationary_dynamics(model, DynamicsConfig(step_count=2, step_seconds=2))
    for snapshot in simulation["snapshots"]:
        analysis = snapshot["koopman"]
        kalman = analysis["kalman_filter"]
        assert kalman["local_observation"]["vertices"]["dimension"] == 11
        assert kalman["local_observation"]["edges"]["dimension"] == 9
        uncertainty = kalman["koopman_input_uncertainty"]
        assert uncertainty["operator_revision_used"] == analysis["operator_revision_used"]
        assert np.isfinite(uncertainty["normalized_std_vector"]).all()
        assert analysis["tensor_matrix_koopman"]["input_kalman_filtered"]
    assert model.to_serializable() == original


def test_tensor_filter_preserves_damage_and_ignores_future_metadata():
    model = GNetBaselineBuilder().build()
    state = {"metric_names": ["OBS.normal_arrival_rate_pps"], "vector": [100.]}
    def view(cpu, loss):
        matrix = TensorLevelMatrix("L2", ("router",),
                                   ("cpu_load_percent", "loss_probability"),
                                   np.array([[cpu, loss]]))
        return TensorMatrixView(("L2",), {"L2": matrix})
    reference = view(10., 0.)
    first = KoopmanOnlineAnalyzer.from_t0(model, state, tensor_matrices=reference, step_seconds=2)
    second = KoopmanOnlineAnalyzer.from_t0(model, state, tensor_matrices=reference, step_seconds=2)
    changed = view(20., .2)
    initial = changed.matrices["L2"].values.copy()
    for analyzer, future in ((first, 30), (second, 1000)):
        analyzer.observed_operator[:] = .5
        report = analyzer.analyze_step(model, state, tensor_matrices=changed,
            attack_state={"future_attack_start_step": future}, step_index=1, time_seconds=2)
        # Известный перенос σ через K=0.5: σ_out=0.5 sqrt(QR/(Q+R)).
        assert report["kalman_filter"]["koopman_input_uncertainty"]["normalized_std_vector"] == pytest.approx(
            [.5 * np.sqrt(.0001 * .0025 / .0026)])
        tensor_filter = analyzer.tensor_filters["L2"]
        assert tensor_filter.state[0] < 1.
        assert tensor_filter.state[1] == .2
    assert first.state_history == pytest.approx(second.state_history)
    before = first.tensor_filters["L2"].state.copy()
    report = first.analyze_step(model, state, tensor_matrices=changed,
        attack_state={"events": [{"kind": "dos"}]}, step_index=2, time_seconds=4)
    assert not report["tensor_matrix_koopman"]["input_kalman_filtered"]
    assert first.tensor_filters["L2"].state == pytest.approx(before)
    assert changed.matrices["L2"].values == pytest.approx(initial)
    assert reference.matrices["L2"].values == pytest.approx(np.array([[10., 0.]]))
