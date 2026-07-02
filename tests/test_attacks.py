from src.gnet9.attacks import MITRE_DEMO_ATTACKS, attack_catalog
from src.gnet9.dynamics import DynamicsConfig, simulate_stationary_dynamics
from src.gnet9.topology_builder import GNetBaselineBuilder


MODEL = GNetBaselineBuilder().build()


def test_attack_catalog_uses_official_mitre_techniques_and_explicit_assumptions() -> None:
    catalog = attack_catalog()
    assert {item["mitre_technique_id"] for item in catalog} == {"T1498.001", "T1498.002", "T1499.001"}
    assert {item["kind"] for item in catalog} == {"dos", "ddos", "syn_flood"}
    assert all(item["mitre_url"].startswith("https://attack.mitre.org/techniques/") for item in catalog)
    assert all(item["technical"]["value_origin"] == "scenario_assumption" for item in catalog)
    assert all(item["temporal"]["duration_steps"] == 2 for item in catalog)


def test_gold_routes_mark_critical_nodes_and_safe_remap_limit() -> None:
    for node_id in ("C1", "SRV_MEDIA", "M1_01"):
        protection = MODEL.graph.nodes[node_id]["critical_protection"]
        assert protection["is_critical"] is True
        assert protection["protection_priority"] == 1
        assert protection["maximum_safe_utilization_percent"] == 80.0
    assert MODEL.graph.nodes["M1_01"]["sla_grade"] == "gold"


def test_mitre_demo_schedule_and_network_reaction() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(step_count=10, snapshot_detail="summary", packet_detail="flows", attack_scenario="mitre-demo"),
    )
    snapshots = dynamics["snapshots"]
    active_steps = [snapshot["step_index"] for snapshot in snapshots if snapshot["attacks"]["active"]]
    assert active_steps == [2, 3, 5, 6, 8, 9]
    assert snapshots[0]["attacks"]["attack_rate_mbps"] == 0.0
    assert snapshots[0]["attacks"]["legitimate_loss_ratio"] == 0.0
    assert snapshots[6]["attacks"]["attack_rate_mbps"] == 32_000.0
    assert snapshots[6]["attacks"]["legitimate_loss_ratio"] > 0.05
    assert snapshots[6]["attacks"]["impacted_gold_flow_count"] > 0
    assert snapshots[6]["arbitrator"]["analysis"]["attack_pressure"] >= 0.90
    assert snapshots[6]["arbitrator"]["remap"]["reason"] == "mitre_attack_observed"
    assert snapshots[7]["attacks"]["active"] is False


def test_attack_flows_are_visible_and_separate_from_legitimate_flows() -> None:
    dynamics = simulate_stationary_dynamics(
        MODEL,
        DynamicsConfig(step_count=3, snapshot_detail="summary", packet_detail="flows", attack_scenario="mitre-demo"),
    )
    flows = dynamics["snapshots"][3]["traffic"]["flows"]
    attack_flows = [flow for flow in flows if flow.get("is_attack_traffic")]
    legitimate_flows = [flow for flow in flows if not flow.get("is_attack_traffic")]
    assert len(legitimate_flows) == 240
    assert {flow["application"] for flow in attack_flows} == {"ATTACK_DOS"}
    assert all(flow["mitre_technique_id"] == "T1498.001" for flow in attack_flows)
    assert len(MITRE_DEMO_ATTACKS) == 3
