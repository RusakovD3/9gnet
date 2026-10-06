from copy import deepcopy

import numpy as np
import pytest

from main import _simulation_timing, parse_args
from src.gnet9.arbitrator import build_trigger_card, trigger_state
from src.gnet9.attacks import attack_catalog
from src.gnet9.dynamics import DynamicsConfig, simulate_stationary_dynamics
from src.gnet9.koopman import KoopmanOnlineAnalyzer
from src.gnet9.l1_d0sl import _mm1k_stationary_metrics
from src.gnet9.tensor_matrix import LazyKroneckerOperator, materialize_kronecker_operator
from src.gnet9.topology_builder import GNetBaselineBuilder
from src.gnet9.transport_queue import apply_transport_queues


@pytest.fixture(scope="module")
def model():
    return GNetBaselineBuilder().build()


def test_lazy_operator_nonzero_rectangular_vectors_and_transpose(monkeypatch):
    left = np.array([[1, 2, -3], [4, -2, 1]])
    right = np.array([[2, 0], [1, -1], [3, 2]])
    dense = materialize_kronecker_operator(left, right)
    vector = np.array([1, -2, 3, 4, -1, 2], dtype=float)
    expected = dense @ vector
    expected_transpose = dense.T @ vector
    operator = LazyKroneckerOperator(left, right)
    left[:] = 0
    def forbidden(*args, **kwargs):
        raise AssertionError("Ленивый расчёт создал полную матрицу")
    monkeypatch.setattr(np, "kron", forbidden)
    np.testing.assert_allclose(operator.matvec(vector), expected)
    np.testing.assert_allclose(operator.rmatvec(vector), expected_transpose)
    with pytest.raises(ValueError):
        operator.matvec(vector[:3])


def test_trigger_card_requires_confirmation_and_distinguishes_threats(monkeypatch):
    options = dict(hausdorff=0.9, risk=0.9, lyapunov_pressure=0.8, decision_pressure=0.9, source_confirmed=False)
    unconfirmed = build_trigger_card(**options, confirmed=False, attack_kind="ddos")
    assert unconfirmed["action"] == "OBSERVE_PRECURSOR"
    assert not unconfirmed["source_quarantine_allowed"]
    power = build_trigger_card(**options, confirmed=True, attack_kind="power_attack")
    syn = build_trigger_card(**options, confirmed=True, attack_kind="syn_flood")
    assert "prefer_nodes_with_energy_reserve" in power["controls"]
    assert "enable_syn_protection" in syn["controls"]
    assert not set(power["controls"]) & set(syn["controls"])
    assert trigger_state(0.349999, "koopman")["state"] == "watch"
    assert trigger_state(0.35, "koopman")["state"] == "alert"
    assert trigger_state(0.70, "koopman")["state"] == "critical"
    for pressure, stage in ((0.20, "observe"), (0.21, "prepare"), (0.50, "protect"), (0.75, "emergency")):
        card = build_trigger_card(**(options | {"decision_pressure": pressure}), confirmed=True, attack_kind="ddos")
        assert card["response_stage"] == stage
    power_with_source = build_trigger_card(**(options | {"source_confirmed": True}), confirmed=True, attack_kind="power_attack")
    assert not power_with_source["source_quarantine_allowed"]
    with pytest.raises(ValueError):
        trigger_state(float("nan"), "koopman")
    from src.gnet9.arbitrator import TRIGGER_BANDS
    monkeypatch.setitem(TRIGGER_BANDS, "decision", (0.30, 0.60, 0.90))
    card = build_trigger_card(**(options | {"decision_pressure": 0.25}), confirmed=True, attack_kind="ddos")
    assert card["action"] == "NO_REMAP"
    card = build_trigger_card(**(options | {"decision_pressure": 0.55}), confirmed=True, attack_kind="ddos")
    assert card["response_stage"] == "prepare"


def test_finite_queue_known_solution_and_heavy_overload():
    blocking, admitted, population, waiting, delay = _mm1k_stationary_metrics(2, 2, 1)
    assert (blocking, admitted, population, waiting, delay) == (0.5, 1.0, 0.5, 0.0, 500.0)
    result = _mm1k_stationary_metrics(1e9, 1e3, 128)
    assert np.isfinite(result).all()
    assert 0.999 < result[0] < 1.0
    assert result[2] <= 128
    assert result[1] == pytest.approx(1000, rel=1e-5)
    assert _mm1k_stationary_metrics(1e20, 1, 128) == (1.0, 1.0, 128.0, 127.0, 128000.0)
    assert _mm1k_stationary_metrics(0, 1, 128) == (0, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        _mm1k_stationary_metrics(float("nan"), 1, 128)


def test_tcp_recovery_and_udp_do_not_double_count_loss(model):
    client = next(node for node, attrs in model.graph.nodes(data=True) if attrs.get("level") == "L1")
    base = dict(client_node=client, route=[client, "C1"], packet_count=1000, payload_bytes=100000, interval_seconds=2, observed_dropped_packets=800, one_way_latency_ms=5)
    tcp = dict(base, transport="TCP", rtt_ms=10, observed_retransmissions=800)
    udp = dict(base, transport="UDP", rtt_ms=None, observed_retransmissions=0)
    apply_transport_queues(model, [tcp, udp])
    assert tcp["transport_queue"]["tcp_recovery_seconds"] > 0
    assert udp["transport_queue"]["tcp_recovery_seconds"] == 0
    assert tcp["goodput_mbps"] < udp["goodput_mbps"]
    assert tcp["one_way_latency_ms"] > 5
    assert tcp["observed_dropped_packets"] == udp["observed_dropped_packets"] == 800
    assert udp["observed_retransmissions"] == 0


def test_cli_schedule_and_training_before_step_50(model):
    args = parse_args(["--attack-scenario", "predictive-demo", "--dynamics-steps", "100", "--attack-start-step", "50", "--attack-interval-steps", "10"])
    count, period = _simulation_timing(args)
    config = DynamicsConfig(step_count=count, step_seconds=period, attack_scenario=args.attack_scenario, attack_start_step=args.attack_start_step, attack_interval_steps=args.attack_interval_steps, snapshot_detail="summary", packet_detail="summary")
    catalog = attack_catalog(model, "predictive-demo", config=config)
    starts = [item["temporal"]["start_step"] for item in catalog]
    assert starts == list(range(50, 50 + 10 * len(starts), 10))
    original = deepcopy(model.graph.nodes["C1"]["tensor"].data)
    # До предвестников — 39 чистых переходов. Проверяем обучение и начало
    # воздействия; полная серия проверяется обычным запуском.
    short = DynamicsConfig(step_count=53, step_seconds=2, attack_scenario="predictive-demo", attack_start_step=50, attack_interval_steps=10, snapshot_detail="summary", packet_detail="summary")
    result = simulate_stationary_dynamics(model, short)
    clean = [s for s in result["snapshots"] if 1 <= s["step_index"] < 40]
    assert len(clean) == 39
    assert all(s["training"]["operator_updated"] for s in clean)
    assert all(not s["attacks"]["active"] and not s["attacks"]["precursor_active"] for s in clean)
    assert all(s["arbitrator"]["remap"]["action"] == "NO_REMAP" for s in clean)
    assert result["training_summary"]["first_attack_step"] == 50
    assert result["ideal_t0"]["ok"]
    np.testing.assert_array_equal(model.graph.nodes["C1"]["tensor"].data, original)


def test_recovery_transition_is_not_used_for_training(model):
    baseline = simulate_stationary_dynamics(model, DynamicsConfig(step_count=0, snapshot_detail="summary", packet_detail="summary"))
    state = baseline["snapshots"][0]["state_vector"]
    analyzer = KoopmanOnlineAnalyzer.from_t0(model, state, step_seconds=2)
    analyzer.analyze_step(model, state, attack_state={}, step_index=0, time_seconds=0)
    attack = {"events": [{"attack_id": "test", "target_id": "C1", "kind": "dos"}]}
    analyzer.analyze_step(model, state, attack_state=attack, step_index=1, time_seconds=2)
    recovery = analyzer.analyze_step(model, state, attack_state={}, step_index=2, time_seconds=4)
    assert recovery["update_applied_for_next_step"] is False
    assert analyzer.nominal_observation_count == 0
