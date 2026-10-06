from src.gnet9.dynamics import DynamicsConfig, _tensor_state_snapshot, simulate_stationary_dynamics
from src.gnet9.attacks import attack_catalog
from src.gnet9.tensor_matrix import (
    TensorKroneckerKoopman,
    build_tensor_matrix_view,
    materialize_kronecker_operator,
)
from src.gnet9.topology_builder import GNetBaselineBuilder


MODEL = GNetBaselineBuilder().build()


def test_every_tensor_is_preserved_in_named_level_matrix() -> None:
    tensor_state = _tensor_state_snapshot(MODEL)
    matrices = build_tensor_matrix_view(tensor_state)
    l1 = matrices.matrices["L1"]
    assert l1.values.shape == (tensor_state["counts"]["L1"], 11)
    assert "authentication_failure_rate_per_second" in l1.metric_names
    assert l1.entity_ids[0].startswith("node:")
    assert sum(matrix.values.size for matrix in matrices.matrices.values()) > 0


def test_factorised_kronecker_channel_matches_dense_numpy_kron_for_t0() -> None:
    matrices = build_tensor_matrix_view(_tensor_state_snapshot(MODEL))
    channel = TensorKroneckerKoopman.from_reference(matrices)
    report = channel.analyze(matrices, verify_dense=True)
    dense = materialize_kronecker_operator(
        channel.level_operator,
        channel.feature_operator,
    )
    assert dense.shape == (report["state_dimension"], report["state_dimension"])
    assert report["numpy_kron_used_for_small_state_equivalence_check"] is True
    assert report["dense_equivalence_max_abs_error"] == 0.0
    assert report["factorised_parameter_count"] < report["dense_equivalent_parameter_count"]


def test_arbitrator_exports_kvu_game_and_safe_sdn_intent() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(
            step_count=2,
            snapshot_detail="summary",
            packet_detail="summary",
            attack_scenario="mitre-demo",
        ),
    )
    snapshot = dynamics["snapshots"][-1]
    hierarchy = snapshot["arbitrator"]["critical_node_hierarchy"]
    assert hierarchy
    assert hierarchy[0]["gold_subscriber_count"] >= hierarchy[-1]["gold_subscriber_count"]
    assert snapshot["koopman"]["tensor_matrix_koopman"]["input"] == (
        "complete_entity_by_metric_tensor_matrices"
    )
    game = snapshot["arbitrator"]["game_theory"]
    assert game["model"] == "finite_defender_attacker_normal_form_game"
    assert game["recommended_action"]
    sdn = snapshot["arbitrator"]["remap"]["sdn_intent"]
    assert sdn["execution_mode"] == "simulation_only_no_network_io"
    assert sdn["southbound_transport"] == "none"


def test_attack_catalog_binds_vvh_tth_to_target_capacity_profile() -> None:
    catalog = attack_catalog(MODEL, "mitre-demo")
    c1 = next(item for item in catalog if item["target_id"] == "C1")
    subscriber = next(item for item in catalog if item["target_id"] == "M1_01")
    assert c1["target_device_context"]["profile_name"] == "NCS5501_CORE"
    assert subscriber["target_device_context"]["binding"] == (
        "subscriber_target_inherits_home_access_profile"
    )
