"""Интерактивные callbacks читают выбранный снимок и не придумывают трафик."""

from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent
import pytest

from src.gnet9.flow_details import _interactive_status_sections, _node_hover_text
from src.gnet9.flow_visualizer import ServiceFlowWindow, _point_on_route, _spread_sample, aggregate_flow_edges
from src.gnet9.packet_simulator import simulate_packet_snapshot
from src.gnet9.topology_builder import GNetBaselineBuilder


@pytest.fixture(scope="module")
def model():
    return GNetBaselineBuilder().build()


@pytest.fixture(scope="module")
def packet(model):
    return simulate_packet_snapshot(model, step_index=0, time_seconds=0, step_seconds=2, detail="flows")


def _snapshots(packet):
    snapshots = []
    for step in (0, 10, 20):
        snapshots.append({"step_index": step, "time_seconds": step * 2,
                          "traffic": deepcopy(packet), "attacks": deepcopy(packet["attack_state"]),
                          "koopman": {"forecast_risk_score": 0.05,
                                      "maximum_forecast_horizon_seconds": 20,
                                      "nominal_training": {"transition_accepted": step > 0, "observation_eligible": True}}})
    return snapshots


def test_loads_and_particles_exclude_unavailable_or_empty_flows():
    flow = dict(route=["a", "b"], packet_count=10, wire_bytes=1000, application="DNS")
    flows = [flow, dict(flow, isolated=True), dict(flow, quarantined=True),
             dict(flow, route=[]), dict(flow, packet_count=0), dict(flow, route_available=False)]
    loads = aggregate_flow_edges(flows)
    assert list(loads) == [frozenset(("a", "b"))]
    assert loads[frozenset(("a", "b"))]["wire_bytes"] == 1000
    assert loads[frozenset(("a", "b"))]["flow_count"] == 1
    assert _spread_sample(flows, 52, 0) == [flow]
    # Отсутствующее промежуточное положение не создаёт выдуманную прямую.
    assert _point_on_route(["a", "missing", "b"], {"a": (0, 0), "b": (2, 0)}, 0.5) is None


def test_cards_separate_prediction_from_observed_actions_and_all_sla_classes(model):
    snapshot = {"step_index": 20, "time_seconds": 40,
                "attacks": {"precursor_active": True, "sla_accounted_flow_count": 20,
                            "security_excluded_flow_count": 2,
                            "sla_restoration": {"tiers": [
                                {"sla_grade": "gold", "flow_count": 10, "sla_compliant_flow_count": 9},
                                {"sla_grade": "silver", "flow_count": 10, "sla_compliant_flow_count": 4},
                                {"sla_grade": "bronze", "flow_count": 0}]}},
                "arbitrator": {"remap": {"action": "ISOLATE_CONFIRMED_SOURCES_AND_REMAP"}},
                "koopman": {"forecast_is_early_warning": True, "forecast_risk_score": 0.85,
                            "forecast_window_start_seconds": 50, "forecast_window_end_seconds": 60,
                            "forecast_target_ids": ["C1"],
                            "nominal_training": {"transition_accepted": False, "observation_eligible": False}}}
    sections = _interactive_status_sections(snapshot, [], {}, True, model)
    assert "ПРИЗНАКИ УГРОЗЫ" in sections[0]["lines"][0]
    forecast = "\n".join(sections[1]["lines"])
    assert "+10…+20 с" in forecast and "50…60 с" in forecast and "не вероятность" in forecast
    quality = "\n".join(sections[2]["lines"])
    assert "Золотой: 90.00% · 9/10" in quality
    assert "Серебряный: 40.00% · 4/10" in quality
    assert "Бронзовый: нет данных" in quality
    assert "Учтено в SLA: 20 · исключено: 2" in quality
    actions = "\n".join(sections[3]["lines"])
    assert "Смена маршрута: 0" in actions and "карантин: 0" in actions
    assert "Решение на следующий шаг:" in actions
    assert "Обучение приостановлено" in sections[4]["lines"]
    # Будущее расписание из диагностического экспорта не меняет карточки.
    altered = deepcopy(snapshot)
    altered["future_attack_schedule"] = [{"start": 42, "target": "SECRET"}]
    assert _interactive_status_sections(altered, [], {}, True, model) == sections


@pytest.mark.parametrize("compact", [False, True])
def test_hover_reads_current_cpu_delay_and_loss_from_both_saved_formats(model, packet, compact):
    snapshot = _snapshots(packet)[1]
    state = snapshot["traffic"]["network_state"]
    current = next(node for node in state["nodes"] if node["id"] == "C1")
    current["metrics"].update(cpu_load_percent=77.7, mean_delay_ms=123.4,
                               delay_variance_ms2=25.0, destroyed_packet_fraction=0.125)
    if compact:
        snapshot["traffic"]["network_state"] = {
            "node_ids": ["C1"], "vertex_metric_names": list(current["metrics"]),
            "vertex_vectors": [list(current["metrics"].values())], "node_attack_arrival_rates_pps": [99.0],
        }
    text = _node_hover_text(model, "C1", snapshot)
    assert "77.7%" in text and "123.40 мс" in text and "25.00 мс²" in text and "12.50%" in text
    assert "ШАГ 10" in text and "Базовая загрузка" not in text
    fallback = _node_hover_text(model, "C1", {"step_index": 9})
    assert "Текущие показатели отсутствуют" in fallback
    assert "конфигурация эталона t0" in fallback


def test_keyboard_buttons_slider_and_hover_select_the_actual_snapshot(model, packet, tmp_path):
    snapshots = _snapshots(packet)
    window = ServiceFlowWindow(model, snapshots)
    try:
        window.animation.event_source.stop()
        window.fig.canvas.draw()
        assert len(window.hover_nodes) == model.graph.number_of_nodes()
        assert len(window.hover_nodes) > 510
        assert window.paused and window.step_index == 0
        window.fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", window.fig.canvas, key="right"))
        assert window.step_index == 1 and window.paused
        assert "Шаг 10" in window.playback_status.get_text()
        assert window.slider.valtext.get_text() == "10"
        # Настоящий callback кнопки, затем callback ползунка.
        window.previous_button._observers.process("clicked", None)
        assert window.step_index == 0
        window.slider.set_val(2)
        assert window.step_index == 2 and "Шаг 20" in window.playback_status.get_text()
        x, y = window.ax.transData.transform(window.pos["C1"])
        event = MouseEvent("motion_notify_event", window.fig.canvas, x, y)
        window.fig.canvas.callbacks.process("motion_notify_event", event)
        assert window.hover_note.get_visible()
        assert "ШАГ 20" in window.hover_note.get_text()
        window.next_button._observers.process("clicked", None)
        assert window.step_index == 2
        window.fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", window.fig.canvas, key="home"))
        assert window.step_index == 0 and not window.hover_note.get_visible()
        window.fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", window.fig.canvas, key=" "))
        assert not window.paused
        window.fig.savefig(tmp_path / "interactive_callbacks.png", dpi=70)
    finally:
        plt.close(window.fig)


@pytest.mark.parametrize("missing", [False, True])
def test_animation_stops_at_last_snapshot_and_empty_frames_do_not_disappear(model, packet, tmp_path, missing):
    snapshots = _snapshots(packet)
    snapshots[1]["traffic"]["flows"] = []
    if missing:
        snapshots[1]["traffic"].pop("flows")
    snapshots[2]["traffic"]["flows"] = []
    window = ServiceFlowWindow(model, snapshots)
    try:
        window.animation.event_source.stop()
        window.fig.canvas.draw()
        window.slider.set_val(1)
        assert window.step_index == 1 and len(window.snapshots) == 3
        assert window.particles.get_offsets().shape == (0, 2)
        assert all(line.get_alpha() == 0 for line in window.edge_artists.values())
        assert ("нет подробных потоков" in window.status.get_text()) == missing
        window.slider.set_val(2)
        window.frame_in_step = window.frames_per_step - 1
        window._toggle_pause(None)
        window._update(0)
        assert window.paused and window.step_index == 2
        assert "ЗАВЕРШЕНО" in window.playback_status.get_text()
        window._toggle_pause(None)
        assert window.step_index == 0 and not window.paused
        # Возврат к началу должен перерисовать линии, а не сохранить последний шаг.
        assert any(line.get_alpha() > 0 for line in window.edge_artists.values())
        window.fig.savefig(tmp_path / "interactive_final_step.png", dpi=70)
    finally:
        plt.close(window.fig)


def test_status_cards_and_legend_fit_their_panels(model, packet, tmp_path):
    snapshot = _snapshots(packet)[1]
    snapshot["koopman"].update(forecast_is_early_warning=True, forecast_window_start_seconds=30,
                               forecast_window_end_seconds=40, forecast_target_ids=["C1", "SRV_MEDIA", "RAN_A1"])
    snapshot["koopman"]["nominal_training"] = {"transition_accepted": False, "observation_eligible": False}
    window = ServiceFlowWindow(model, [snapshot])
    try:
        window.animation.event_source.stop()
        window.fig.canvas.draw()
        renderer = window.fig.canvas.get_renderer()
        for heading, body, panel in window.status_cards:
            bounds = panel.get_window_extent(renderer)
            for text in (heading, body):
                extent = text.get_window_extent(renderer)
                assert extent.x0 >= bounds.x0 and extent.x1 <= bounds.x1
                assert extent.y0 >= bounds.y0 and extent.y1 <= bounds.y1
        legend = window.legend_ax.get_legend().get_window_extent(renderer)
        assert legend.x1 <= window.fig.bbox.x1
        assert "выборка" in window.help_text.get_text()
        assert "точка — пакет" not in window.help_text.get_text()
        window.fig.savefig(tmp_path / "interactive_panels.png", dpi=70)
    finally:
        plt.close(window.fig)
