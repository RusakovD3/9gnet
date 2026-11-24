"""Сохранённая сводка читается без действий в интерактивном окне."""

from copy import deepcopy

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text

from src.gnet9.flow_visualizer import draw_security_dashboard
from src.gnet9.security_dashboard import peak_security_step


def _export(snapshots, tmp_path, monkeypatch):
    figures = []
    save = Figure.savefig

    def capture(fig, *args, **kwargs):
        figures.append(fig)
        return save(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    path = tmp_path / "security_dashboard.png"
    draw_security_dashboard(None, snapshots, path)
    assert path.read_bytes().startswith(b"\x89PNG")
    fig, = figures
    FigureCanvasAgg(fig).draw()
    artists = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    return fig, artists, " ".join(" ".join(t.get_text().split()) for t in artists)


def test_static_dashboard_shows_the_selected_observation_quality_and_complete_action(tmp_path, monkeypatch):
    action = ("Проверьте журналы наблюдаемого узла и сопоставьте соседние отсчёты. "
              "Подтвердите источник по журналам до решения об ограничении доступа.")
    attack = {"step_index": 8, "time_seconds": 16, "traffic": {"flows": []},
              "attacks": {"events": [{"kind": "dos", "target_id": "C1",
                                      "routes": [["RAN_A1", "A1", "C1"]]}],
                         "precursors": [{"sensor_node_id": node} for node in ("A1", "A2", "A3")],
                         "sla_restoration": {"tiers": [
                             {"sla_grade": "gold", "flow_count": 17, "sla_compliant_flow_count": 14}]}},
              "arbitrator": {"recommendation": {"response_stage": "protect", "primary_action_ru": action,
                                                "steps_ru": ["Общий порядок проверки.", action]}}}
    later = deepcopy(attack)
    later["step_index"] = 9
    later["time_seconds"] = 18
    later["attacks"]["events"][0]["target_id"] = "FUTURE_TARGET"
    later["arbitrator"]["recommendation"]["response_stage"] = "prepare"
    snapshots = [attack, later]
    original = deepcopy(snapshots)
    fig, artists, text = _export(snapshots, tmp_path, monkeypatch)
    assert "колесо" not in text and "Продолжить" not in text and "Строки" not in text
    assert "Шаг 8" in text and "16 с" in text and "Высокий" in text
    assert "RAN_A1 → A1 → C1" in text and "FUTURE_TARGET" not in text
    assert "Другие наблюдения на шаге: 2" in text
    assert "14 из 17" in text and action in text
    renderer = fig.canvas.get_renderer()
    bounds = [artist.get_window_extent(renderer) for artist in artists]
    for i, box in enumerate(bounds):
        assert fig.bbox.contains(*box.min) and fig.bbox.contains(*box.max)
        assert all(not box.overlaps(other) for other in bounds[i + 1:])
    assert snapshots == original


def test_static_dashboard_distinguishes_absent_readings_from_no_alarms(tmp_path, monkeypatch):
    _, _, text = _export([{"step_index": 0, "time_seconds": 0, "attacks": {"events": [], "precursors": []},
                           "arbitrator": {"recommendation": {"response_stage": "observe"}}}],
                         tmp_path, monkeypatch)
    assert "Признаков угрозы нет" in text and "Наблюдение" in text
    assert "Нет данных о качестве" in text and "Рекомендация не сохранена" in text


def test_older_snapshot_without_observations_does_not_claim_a_healthy_network(tmp_path, monkeypatch):
    _, _, text = _export([{"step_index": 0, "time_seconds": 0,
                           "arbitrator": {"recommendation": {"steps_ru": ["Проверьте журналы."]}}}],
                         tmp_path, monkeypatch)
    assert "Нет данных о срочности" in text and "Наблюдения не сохранены" in text
    assert "Проверьте журналы." in text


def test_peak_security_step_prefers_observed_event_then_earlier_sample_at_equal_urgency():
    high = {"arbitrator": {"trigger_card": {"response_stage": "protect"}}}
    with_event = dict(high, attacks={"events": [{"target_id": "C1"}]})
    snapshots = [{}, high, with_event, with_event, {"arbitrator": {"recommendation": {"response_stage": "prepare"}}}]
    assert peak_security_step(snapshots) == 2
