"""Графики объясняют сохранённые результаты, не принимая новых решений."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.gnet9 import dynamics_charts as charts


def _snapshots():
    snapshots = []
    for step in range(4):
        snapshots.append({
            "step_index": step, "time_seconds": 2 * step,
            "koopman": {"nominal_training": {"transition_accepted": step == 1},
                        "forecast_is_early_warning": step == 2,
                        "warning_risk_threshold": .37},
            "traffic": {"network_state": {"edge_metric_names": ["capacity_mbps", "speed_mbps"],
                                             "edge_ids": [["a", "b"]], "edge_vectors": [[10, 2]]}},
            "attacks": {"events": [], "routing": {
                "maximum_projected_utilization_percent": 999,
                "maximum_projected_edge_utilization_percent": 140,
                "maximum_projected_node_utilization_percent": 35,
                "maximum_projected_server_cpu_utilization_percent": 240,
                "maximum_projected_server_ram_utilization_percent": 40,
                "maximum_projected_server_session_utilization_percent": 60,
                "maximum_safe_utilization_percent": 82.5,
            }},
        })
    return snapshots


def test_unknown_measurements_are_gaps_instead_of_zero_or_success():
    series = charts.extract_dynamics_chart_series({"snapshots": [{"time_seconds": 2}]})
    for name in ("gold_availability_percent", "legitimate_delivery_percent", "gold_sla_percent",
                 "sla_margin_min", "koopman_forecast_risk", "normal_arrival_rate_pps"):
        assert np.isnan(series[name][0]), name
    assert np.isnan(charts._snapshot_peak_load(None, {}))


def test_observed_windows_preserve_gaps_and_ignore_future_schedule():
    snapshots = _snapshots()
    event = {"attack_id": "one", "kind": "dos", "temporal": {"start_step": 100}}
    snapshots[0]["attacks"]["events"] = [event]
    snapshots[3]["attacks"]["events"] = [event]
    dynamics = {"config": {"step_seconds": 2}, "snapshots": snapshots,
                "training_summary": {"clean_training_steps": 17}}
    assert charts._accepted_transition_count(dynamics) == 1
    assert charts._training_windows(dynamics) == [(0, 2)]
    windows = charts._attack_windows(dynamics)
    assert [(w["start"], w["end"]) for w in windows] == [(0, 2), (6, 8)]
    # Плановый размер чистого участка не выдаётся за принятые переходы.
    assert charts._accepted_transition_count({"training_summary": {"clean_training_steps": 17}}) is None


def test_capacity_separates_resources_keeps_outliers_and_saved_admission_limit(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(charts, "save_figure", lambda figure, path: captured.append(figure))
    charts._plot_capacity_bottlenecks(None, _snapshots(), tmp_path / "capacity.png")
    figure = captured[0]
    try:
        _, network, servers = figure.axes
        assert network.get_ylim()[1] > 140
        assert servers.get_ylim()[1] > 240
        network_lines = {line.get_label(): np.asarray(line.get_ydata()) for line in network.lines}
        server_lines = {line.get_label(): np.asarray(line.get_ydata()) for line in servers.lines}
        np.testing.assert_array_equal(network_lines["максимальная нагрузка линии"], [20] * 4)
        np.testing.assert_array_equal(network_lines["линии: оценка резервирования"], [140] * 4)
        np.testing.assert_array_equal(server_lines["CPU (процессор)"], [240] * 4)
        np.testing.assert_array_equal(network_lines["сохранённый допуск нового маршрута"], [82.5] * 4)
        assert not any(999 in values for values in (*network_lines.values(), *server_lines.values()))
    finally:
        plt.close(figure)


def test_overview_labels_actual_training_and_prediction_issue_time(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(charts, "save_figure", lambda figure, path: captured.append(figure))
    dynamics = {"snapshots": _snapshots(), "config": {"step_count": 99, "step_seconds": 2},
                "training_summary": {"clean_training_steps": 17}}
    charts._plot_dynamics_overview(charts.extract_dynamics_chart_series(dynamics), dynamics, tmp_path / "overview.png")
    figure = captured[0]
    try:
        text = " ".join(item.get_text() for item in figure.texts)
        assert "Принято переходов в обучение: 1" in text
        assert "время выдачи прогноза" in text
        assert "Чистых переходов" not in text
        assert len(figure.axes) == 7
        assert max(figure.axes[3].get_yticks()) <= 100
        figure.canvas.draw()
        timeline_label = figure.axes[0].xaxis.label.get_window_extent()
        first_panel_title = figure.axes[1]._left_title.get_window_extent()
        assert timeline_label.y0 > first_panel_title.y1
    finally:
        plt.close(figure)


def test_two_existing_png_exports_render_for_empty_and_recorded_data(tmp_path):
    for name, dynamics in (("empty", {}), ("recorded", {"snapshots": _snapshots()})):
        exported = charts.export_dynamics_charts(dynamics, tmp_path / name)
        assert set(exported) == {"overview", "capacity"}
        assert {path.name for path in exported.values()} == {"dynamics_overview.png", "capacity_bottlenecks.png"}
        for path in exported.values():
            image = plt.imread(path)
            assert image.shape[0] > 1000 and image.shape[1] > 1000
    assert not plt.get_fignums()
