"""Интерактивные callbacks читают выбранный снимок и не придумывают трафик."""

from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent
import pytest

from src.gnet9.flow_details import _interactive_status_sections, _node_hover_text
from src.gnet9.flow_visualizer import ServiceFlowWindow, _point_on_route, _spread_sample, aggregate_flow_edges, draw_security_dashboard
from src.gnet9.packet_simulator import simulate_packet_snapshot
from src.gnet9.topology_builder import GNetBaselineBuilder
from src.gnet9.routing import shortest_data_path


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
                "arbitrator": {"recommendation": {"steps_ru": ["Проверьте источники"]}},
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
    assert "Проверьте источники" in actions
    assert "Структура сети и маршруты фиксированы." in actions
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
    snapshot["arbitrator"] = {"recommendation": {"steps_ru": ["Проверьте журнал", "Проверьте классы услуг",
        "Проверьте журнал управления питанием, напряжение и запас энергии ИБП (источника бесперебойного питания)."]}}
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
        labels = " ".join(text.get_text() for text in window.legend_ax.get_legend().get_texts())
        assert "Изолированный" not in labels and "Маршрут изменён" not in labels
        assert "Резервный сервер" not in labels
        legend = window.legend_ax.get_legend().get_window_extent(renderer)
        assert legend.x1 <= window.fig.bbox.x1
        assert "выборка" in window.help_text.get_text()
        assert "точка — пакет" not in window.help_text.get_text()
        window.fig.savefig(tmp_path / "interactive_panels.png", dpi=70)
    finally:
        plt.close(window.fig)


def _alarm_snapshots(model, packet):
    snapshots = _snapshots(packet)
    active = snapshots[1]
    targets = ["C1", "C2", "C3", "SRV_DATA", "SRV_MEDIA", "C5", "C9"]
    kinds = ["dos", "ddos", "syn_flood", "dns_flood", "application_flood", "brute_force", "power_attack"]
    ingress = next(node for node, attrs in model.graph.nodes(data=True) if attrs.get("role") == "radio-access-node")
    events = [{"attack_id": f"A{i}", "target_id": target, "kind": kind,
               "ingress_nodes": [ingress], "routes": [shortest_data_path(model, ingress, target)],
               "effective_packet_rate_pps": 1000, "effective_offered_rate_mbps": 10}
              for i, (target, kind) in enumerate(zip(targets, kinds))]
    active["attacks"].update(active=True, events=events, target_observations=[
        {"target_id": target, "service_availability_ratio": .75} for target in targets])
    active["arbitrator"] = {"recommendation": {"response_stage": "emergency", "requires_human_action": True,
        "steps_ru": ["Проверьте журналы и текущую загрузку наблюдаемого узла. " * 3 for _ in range(5)] +
                    ["Подтвердите источник до решения об ограничении доступа; решение принимает человек."]}}
    snapshots[2]["arbitrator"] = {"recommendation": {"response_stage": "observe", "steps_ru": ["Наблюдайте."]}}
    return snapshots


def _mouse(window, ax, x, y, name="button_press_event", **extra):
    px, py = ax.transData.transform((x, y))
    event = MouseEvent(name, window.fig.canvas, px, py, **extra)
    window.fig.canvas.callbacks.process(name, event)


def test_alarm_selection_paging_and_recommendations_are_linked_to_map(model, packet, tmp_path):
    snapshots = _alarm_snapshots(model, packet)
    before = deepcopy(snapshots)
    window = ServiceFlowWindow(model, snapshots)
    try:
        window.animation.event_source.stop()
        window._select_step(1)
        assert window.slider.val == 1
        window.fig.canvas.draw()
        panel = window.security_panel
        assert panel.data["severity_counts"]["critical"] == 7
        assert len(window.attack_routes.get_segments()) > 0
        _mouse(window, panel.axes["alarms"], .4, .57)
        assert panel.selected["target_id"] == "C2"
        assert tuple(window.selected_target.get_offsets()[0]) == window.pos["C2"]
        assert "C2" in window.attack_location.get_text()
        _mouse(window, panel.axes["alarms"], .95, .92)
        assert panel.page == 1
        _mouse(window, panel.axes["alarms"], .4, .75)
        assert panel.selected["target_id"] == panel.visible_alarms[0]["target_id"]
        for _ in range(20):
            _mouse(window, panel.axes["recommendations"], .5, .5, "scroll_event", button="down", step=-1)
        assert any("Подтвердите источник" in text.get_text() for text in panel.axes["recommendations"].texts)
        last_offset = panel.recommendation_offset
        _mouse(window, panel.axes["recommendations"], .93, .10)
        assert panel.recommendation_offset < last_offset
        _mouse(window, panel.axes["recommendations"], .97, .10)
        assert panel.recommendation_offset == last_offset
        window.metrics_button._observers.process("clicked", None)
        assert window.status_ax.get_visible() and not panel.axes["alarms"].get_visible()
        window.security_button._observers.process("clicked", None)
        assert not window.status_ax.get_visible() and panel.axes["alarms"].get_visible()
        window.fig.savefig(tmp_path / "dashboard_overlap.png", dpi=80)
        window._select_step(0)
        assert panel.data["alarms"] == []
        assert len(window.attack_routes.get_segments()) == 0
        assert len(window.selected_target.get_offsets()) == 0
        assert snapshots == before
    finally:
        plt.close(window.fig)


def test_dashboard_export_clears_previous_run_without_packet_snapshots(tmp_path):
    path = tmp_path / "security_dashboard.png"
    path.write_bytes(b"previous experiment")
    draw_security_dashboard(None, [], path)
    assert not path.exists()
    draw_security_dashboard(None, [], path)


def test_security_text_fits_full_hd_and_new_attack_replaces_automatic_history_selection(model, packet, tmp_path):
    snapshots = _alarm_snapshots(model, packet)
    # Длинная подпись восстановления проверяется вместе с максимальной срочностью.
    snapshots[1]["attacks"]["events"][-1]["is_recovery_effect"] = True
    snapshots[2]["attacks"].update(active=True, events=[{
        "attack_id": "new", "kind": "dns_flood", "target_id": "SRV_DATA", "routes": []}])
    window = ServiceFlowWindow(model, snapshots)
    try:
        window.animation.event_source.stop()
        window.fig.set_size_inches(19.2, 10.8)
        window._select_step(1)
        panel = window.security_panel
        panel.selected_key = next(a["key"] for a in panel.data["alarms"] if a["kind"] == "power_attack")
        panel.page = 1
        panel.draw()
        window.fig.canvas.draw()
        renderer = window.fig.canvas.get_renderer()
        for ax in panel.axes.values():
            bounds = ax.get_window_extent(renderer)
            for artist in ax.texts:
                extent = artist.get_window_extent(renderer)
                assert extent.x0 >= bounds.x0 and extent.x1 <= bounds.x1, artist.get_text()
                assert extent.y0 >= bounds.y0 and extent.y1 <= bounds.y1, artist.get_text()
        window.fig.savefig(tmp_path / "dashboard_full_hd.png", dpi=100)
        window._select_step(2)
        assert panel.selected["phase"] == "active"
        assert panel.selected["kind"] == "dns_flood"
    finally:
        plt.close(window.fig)
