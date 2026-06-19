from collections import Counter
from pathlib import Path

from src.gnet9.l1_d0sl import TrafficKind, build_l1_queue_model, load_l1_d0sl_catalog
from src.gnet9.packet_simulator import simulate_packet_snapshot
from src.gnet9.flow_visualizer import _advance_playback, _particle_progress
from src.gnet9.topology_builder import GNetBaselineBuilder


def test_voice_d0sl_profiles_are_complete() -> None:
    catalog = load_l1_d0sl_catalog(Path(__file__).resolve().parents[1] / "policies" / "l1_policies.d0sl")
    for grade, bitrate, latency, loss, jitter in (
        ("gold", 32.0, 80.0, 1.0, 20.0),
        ("silver", 24.0, 120.0, 2.0, 30.0),
        ("bronze", 16.0, 150.0, 3.0, 40.0),
    ):
        policy = catalog.get(grade, TrafficKind.VOICE.value)
        assert policy.target_bitrate_kbps == bitrate
        assert policy.latency_budget_ms == latency
        assert policy.packet_loss_budget_percent == loss
        assert policy.jitter_budget_ms == jitter
        assert policy.packetization_ms == 20
        assert policy.rtp_clock_rate_hz == 48_000
        assert policy.rtp_payload_type == 111
        assert build_l1_queue_model(policy).arrival_rate_pps == 50.0


def test_voice_flows_use_opus_rtp_and_voice_service() -> None:
    model = GNetBaselineBuilder().build()
    traffic_kinds = Counter(
        attrs["traffic_kind"]
        for _, attrs in model.graph.nodes(data=True)
        if attrs.get("level") == "L1"
    )
    assert traffic_kinds["voice"] == 60

    traffic = simulate_packet_snapshot(
        model,
        step_index=0,
        time_seconds=0,
        step_seconds=5,
        detail="sample",
        packet_sample_limit=1000,
    )
    voice_flows = [flow for flow in traffic["flows"] if flow["application"] == "RTP_OPUS"]
    assert len(voice_flows) == 60
    assert all(flow["server_node"] == "SVC_VOICE" for flow in voice_flows)
    assert all(flow["packets_per_second"] == 50 for flow in voice_flows)
    assert all(flow["line_bitrate_kbps"] > flow["codec_bitrate_kbps"] for flow in voice_flows)

    packet = next(packet for packet in traffic["packet_sample"] if packet["packet_role"] == "rtp_voice")
    assert packet["rtp"]["codec"] == "Opus"
    assert packet["rtp"]["payload_type"] == 111
    assert packet["rtp"]["clock_rate_hz"] == 48_000


def test_l2_roles_distinguish_switches_and_routers() -> None:
    model = GNetBaselineBuilder().build()
    assert all(model.graph.nodes[f"A{index}"]["role"] == "aggregation-switch" for index in range(1, 7))
    assert all(model.graph.nodes[f"C{index}"]["role"] == "core-router" for index in range(1, 13))
    assert all(model.graph.nodes[f"A{index}"]["platform_family"] == "Catalyst C9500-24Y4C" for index in range(1, 7))
    assert all(model.graph.nodes[f"C{index}"]["platform_family"] == "NCS 5501" for index in range(1, 13))


def test_paused_t0_packets_are_at_route_sources() -> None:
    assert _particle_progress(step_index=0, frame=0, offset=7, particle_count=42, paused=True) == 0.0
    assert _particle_progress(step_index=0, frame=30, offset=7, particle_count=42, paused=True) == 0.0
    assert _particle_progress(step_index=1, frame=17, offset=7, particle_count=42, paused=True) > 0.0


def test_playback_advances_steps_and_finishes_on_last_snapshot() -> None:
    assert _advance_playback(step_index=4, frame_in_step=46, final_step=10, frames_per_step=48) == (4, 47, False)
    assert _advance_playback(step_index=4, frame_in_step=47, final_step=10, frames_per_step=48) == (5, 0, False)
    assert _advance_playback(step_index=10, frame_in_step=47, final_step=10, frames_per_step=48) == (10, 48, True)
