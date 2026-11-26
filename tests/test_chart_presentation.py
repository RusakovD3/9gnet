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
            "traffic": {"network_state": {"node_metric_maxima": {"cpu_load_percent": 240}, "edge_metric_names": ["capacity_mbps", "speed_mbps"],
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
    for name in ("legitimate_delivery_percent", "gold_sla_percent",
                 "sla_margin_min", "koopman_forecast_risk", "normal_arrival_rate_pps"):
        assert np.isnan(series[name][0]), name
    assert np.isnan(charts._link_load_summary(None, [{}])[1][0])


def test_observed_windows_preserve_gaps_and_ignore_future_schedule():
    snapshots = _snapshots()
    event = {"attack_id": "one", "kind": "dos", "temporal": {"start_step": 100}}
    snapshots[0]["attacks"]["events"] = [event]
    snapshots[3]["attacks"]["events"] = [event]
    dynamics = {"config": {"step_seconds": 2}, "snapshots": snapshots,
                "training_summary": {"clean_training_steps": 17}}
    assert charts._accepted_transition_count(dynamics) == 1
    windows = charts.RunTimeAxis.from_result(dynamics).windows
    assert [(w["start"], w["end"]) for w in windows] == [(0, 2), (6, 8)]
    # Плановый размер чистого участка не выдаётся за принятые переходы.
    assert charts._accepted_transition_count({"training_summary": {"clean_training_steps": 17}}) is None


def test_capacity_separates_resources_keeps_outliers_and_saved_admission_limit(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(charts, "save_figure", lambda figure, path, **kwargs: captured.append(figure))
    snapshots = _snapshots()
    snapshots[2]["traffic"]["network_state"]["edge_vectors"] = [[10, 15]]
    run = charts.RunTimeAxis.from_result({"snapshots": snapshots})
    rows, peaks = charts._link_load_summary(None, snapshots)
    charts._plot_capacity_bottlenecks(rows, tmp_path / "capacity.png")
    figure = captured[0]
    try:
        assert len(figure.axes) == 1
        charts._plot_link_utilization(list(run.by_step), peaks, run, tmp_path / "links.png")
        charts._plot_resource_utilization(_snapshots(), run, tmp_path / "resources.png")
        network, servers = captured[1].axes[0], captured[2].axes[0]
        assert figure.axes[0].patches[0].get_width() == 150
        assert figure.axes[0].get_xlim()[1] >= 150
        assert "предложенная нагрузка" in figure.axes[0].get_xlabel().lower()
        assert network.get_ylim()[1] >= 150
        assert servers.get_ylim()[1] > 240
        network_lines = {line.get_label(): np.asarray(line.get_ydata()) for line in network.lines}
        server_lines = {line.get_label(): np.asarray(line.get_ydata()) for line in servers.lines}
        np.testing.assert_array_equal(network_lines["максимальная предложенная нагрузка"], [20, 20, 150, 20])
        np.testing.assert_array_equal(network_lines["пропускная способность линии"], [100, 100])
        np.testing.assert_array_equal(server_lines["Максимум CPU по всем узлам"], [240] * 4)
        assert "сохранённый допуск нового маршрута" not in network_lines
        assert not any(999 in values for values in (*network_lines.values(), *server_lines.values()))
    finally:
        for rendered in captured:
            plt.close(rendered)


def test_capacity_shows_target_work_when_packet_queue_cpu_is_low(tmp_path, monkeypatch):
    captured = []
    snapshots = _snapshots()
    for snapshot, pressure in zip(snapshots, [0, 75, 100, 0]):
        snapshot["traffic"]["network_state"]["node_metric_maxima"]["cpu_load_percent"] = 22
        snapshot["attacks"]["maximum_target_utilization_percent"] = pressure
    monkeypatch.setattr(charts, "save_figure", lambda figure, path, **kwargs: captured.append(figure))
    charts._plot_resource_utilization(snapshots, charts.RunTimeAxis.from_result({"snapshots": snapshots}), tmp_path / "resources.png")
    figure = captured[0]
    try:
        lines = {line.get_label(): line.get_ydata() for line in figure.axes[0].lines}
        np.testing.assert_array_equal(lines["Максимум CPU по всем узлам"], [22] * 4)
        np.testing.assert_array_equal(lines["ресурс атакованной цели"], [0, 75, 100, 0])
    finally:
        plt.close(figure)


def test_timeline_contains_only_training_native_traffic_and_observed_attacks(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(charts, "save_figure", lambda fig, *args, **kwargs: captured.append(fig))
    snapshots = _snapshots()
    event = {"attack_id": "one", "attack_label_ru": "DoS", "target_id": "A"}
    for i in (0, 2):
        snapshots[i]["traffic"]["summary"] = {"normal_arrival_rate_pps": 100}
    snapshots[1]["traffic"]["summary"] = {"normal_arrival_rate_pps": 0, "attack_arrival_rate_pps": 100}
    snapshots[2]["attacks"]["events"] = [event]
    snapshots[3]["attacks"]["events"] = [dict(event, is_recovery_effect=True)]
    snapshots[2]["koopman"]["kalman_filter"] = {
        "local_observation": {"window": {"global_reset_reasons": ["timer"]}}}
    dynamics = {"snapshots": snapshots, "resilience_evaluation": {"episodes": [{"recovered_step": 3}]}}
    charts._plot_timeline(dynamics, charts.RunTimeAxis.from_result(dynamics), tmp_path / "timeline.png")
    fig, = captured
    try:
        ax, = fig.axes
        assert [label.get_text() for label in ax.get_yticklabels()] == [
            "Обучение Купмана", "Обычный трафик", "Атака · DoS · A"]
        assert [t.get_text() for t in fig.legends[0].get_texts()] == ["Начало атаки", "Конец атаки"]
        assert len(ax.collections) == 5
        intervals = [[(p.vertices[:, 0].min(), p.vertices[:, 0].max()) for p in row.get_paths()]
                     for row in ax.collections[:3]]
        assert intervals == [[(0, 2)], [(0, 2), (4, 6)], [(4, 6)]]
    finally:
        plt.close(fig)


def test_png_exports_render_for_empty_and_recorded_data(tmp_path):
    for name, dynamics in (("empty", {}), ("recorded", {"snapshots": _snapshots()})):
        exported = charts.export_dynamics_charts(dynamics, tmp_path / name)
        assert {path.name for path in exported.values()} == {
            "incident_overview.png", "koopman_risk.png", "lyapunov_growth.png", "native_traffic.png",
            "attack_traffic.png", "sla_by_class.png", "attacked_node_sla.png",
            "event_timeline.png", "arbitrator_signals.png", "capacity_bottlenecks.png",
            "link_utilization.png", "resource_utilization.png", "service_sla_forecast.png",
        }
        for path in exported.values():
            image = plt.imread(path)
            assert image.shape[0] > 400 and image.shape[1] > 1000
            np.testing.assert_allclose(image[0, 0, :3], [1, 1, 1])
    assert not plt.get_fignums()


def test_koopman_and_lyapunov_read_saved_values_and_use_seconds(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(charts, "save_figure", lambda figure, path, **kwargs: captured.append(figure))
    snapshots = _snapshots()
    for snapshot, derivative in zip(snapshots, [-.025, 0, .025, .05]):
        snapshot["attacks"]["precursor_confidence"] = .13
        snapshot["koopman"].update(
            forecast_risk_score=.8,
            lyapunov_derivative_per_second=derivative,
            lyapunov_remap_response_scale_per_second=.025,
        )
    dynamics = {"snapshots": snapshots}
    series = charts.extract_dynamics_chart_series(dynamics)
    run = charts.RunTimeAxis.from_result(dynamics)
    charts._plot_koopman(series, *charts._risk_series(series, dynamics), run, tmp_path / "risk.png")
    charts._plot_lyapunov(series, dynamics, run, tmp_path / "lyapunov.png")
    try:
        risk, growth = (figure.axes[0] for figure in captured)
        assert risk.get_ylim() == (0, 1)
        assert growth.get_ylim() == (-1, 1)
        np.testing.assert_array_equal(risk.lines[1].get_ydata(), [.8] * 4)
        np.testing.assert_array_equal(risk.lines[2].get_ydata(), [.37] * 4)
        np.testing.assert_array_equal(growth.lines[0].get_xdata(), [0, 2, 4, 6])
        np.testing.assert_allclose(growth.lines[0].get_ydata(), [-.5, 0, .5, 2 / 3])
    finally:
        for figure in captured:
            plt.close(figure)


def test_event_rates_and_target_sla_follow_observed_flows():
    snapshots = _snapshots()
    snapshots[1]["attacks"]["events"] = [
        {"attack_id": "one", "target_id": "r1", "kind": "dos", "effective_packet_rate_pps": 10},
        {"attack_id": "two", "target_id": "r2", "kind": "auth", "effective_packet_rate_pps": 0},
    ]
    snapshots[2]["attacks"]["events"] = [
        {"attack_id": "one", "target_id": "r1", "is_recovery_effect": True, "effective_packet_rate_pps": 0},
    ]
    flow = {"sla_grade": "gold", "route": ["a", "r1", "b"],
            "slo_evaluation": {"compliant": False}, "attack_ids": ["one"],
            "normal_expected_packet_rate_pps": 10.0}
    snapshots[1]["traffic"]["flows"] = [
        flow,
        flow | {"slo_evaluation": {"compliant": True}},
        flow | {"is_attack_traffic": True},
        flow | {"slo_evaluation": {}},
        flow | {"route": ["c", "r2", "d"], "attack_ids": [], "slo_evaluation": {"compliant": True}},
    ]
    rates = charts._event_packet_rates(snapshots)
    assert rates["one"]["rates"] == {1: 10}
    assert rates["two"]["rates"] == {1: 0}
    sla = charts._target_sla_rows(snapshots)
    assert sla["r1", "gold"] == {1: (1, 2)}
    assert sla["r2", "gold"] == {1: (1, 1)}
    expected = charts._expected_native_rates(snapshots)
    assert expected[1] == 40.0
    assert np.isnan(expected[0])


def test_timeline_merges_adjacent_samples_preserving_gaps(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(charts, "save_figure", lambda fig, *args, **kwargs: captured.append(fig))
    snapshots = [{"time_seconds": t, "traffic": {"summary": {"legitimate_packet_count": 1}},
                  "training": {"operator_updated": t > 0}} for t in (0, 2, 4, 10)]
    dynamics = {"snapshots": snapshots, "config": {"step_seconds": 2}}
    charts._plot_timeline(dynamics, charts.RunTimeAxis.from_result(dynamics), tmp_path / "timeline.png")
    fig, = captured
    try:
        training, traffic = fig.axes[0].collections
        assert [(p.vertices[:, 0].min(), p.vertices[:, 0].max()) for p in training.get_paths()] == [(0, 4)]
        assert [(p.vertices[:, 0].min(), p.vertices[:, 0].max()) for p in traffic.get_paths()] == [(0, 6), (10, 12)]
    finally:
        plt.close(fig)


def test_cpu_peak_rows_follow_winner_changes_and_ties(tmp_path, monkeypatch):
    states = [{"nodes": [{"id": "A", "metrics": {"cpu_load_percent": 70}},
                         {"id": "B", "metrics": {"cpu_load_percent": 30}}]},
              {"vertex_metric_names": ["cpu_load_percent"], "node_ids": ["B", "A"],
               "vertex_vectors": [[60], [60]]},
              {"nodes": [{"id": "B", "metrics": {"cpu_load_percent": 80}}]},
              {"node_metric_maxima": {"cpu_load_percent": 75}}]
    snapshots = [{"traffic": {"network_state": state}, "time_seconds": i * 5}
                 for i, state in enumerate(states)]
    captured = []
    monkeypatch.setattr(charts, "save_figure", lambda fig, *args, **kwargs: captured.append(fig))
    run = charts.RunTimeAxis.from_result({"snapshots": snapshots})
    charts._plot_resource_utilization(snapshots, run, tmp_path / "cpu.png")
    fig, = captured
    try:
        main, lanes = fig.axes
        np.testing.assert_array_equal(main.lines[0].get_ydata(), [70, 60, 80, 75])
        assert [tick.get_text() for tick in lanes.get_yticklabels()] == ["A", "B", "Узел не указан"]
        assert [[path.vertices[:, 0].min() for path in row.get_paths()]
                for row in lanes.collections] == [[0, 5], [5, 10], [15]]
    finally:
        plt.close(fig)


def test_sla_legend_counts_accounted_flows_per_step():
    tiers = [{"sla_grade": "gold", "flow_count": n, "sla_compliant_flow_count": passed,
              "subscriber_flow_count": 12} for n, passed in [(10, 5), (8, 6)]]
    series = charts.extract_dynamics_chart_series({"snapshots": [
        {"attacks": {"sla_restoration": {"tiers": [tier]}}} for tier in tiers]})
    fig, ax = plt.subplots()
    try:
        charts._sla_lines(ax, series, [0, 2])
        assert ax.lines[0].get_label() == "золотой · потоков: 8–10\nмин. 50 % в 0 с"
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [50, 75])
    finally:
        plt.close(fig)


def test_minimum_caption_uses_first_observed_minimum_and_handles_flat_or_missing_series():
    assert charts._minimum_caption([0, 2, 4, 6], [np.nan, 99.25, 100, 99.25]) == "мин. 99,25 % в 2 с"
    assert charts._minimum_caption([0, 2], [100, np.nan]) == "100 % по измерениям"
    assert charts._minimum_caption([0, 2], [100, 99.999]) == "мин. <100 % в 2 с"
    assert charts._minimum_caption([0], [np.nan]) == ""


def test_forecast_case_title_and_legend_fit_exported_canvas(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(charts, "save_figure", lambda fig, *args, **kwargs: figures.append(fig))
    case = {"service": "RTP_VLC_AV", "flow_id": "M1_01",
            "label_ru": "односторонняя задержка медиапотока", "unit": "мс",
            "measurements": [{"time_seconds": 0, "value": 100, "budget": 150}],
            "issue_time_seconds": 0, "target_time_seconds": 2, "actual_time_seconds": 2,
            "measured_value": 100, "predicted_value": 160, "trend_at_target": 110}
    charts._plot_service_forecast({"cases": {"hit": case}}, tmp_path / "forecast.png",
                                  charts.RunTimeAxis.from_result({"snapshots": _snapshots()}))
    fig, = figures
    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert "Тренд потока" not in [t.get_text() for t in legend.get_texts()]
        title = next(t for t in fig.findobj(plt.Text) if t.get_text().startswith("Предупреждённое нарушение"))
        for artist in (title, legend):
            bounds = artist.get_window_extent(fig.canvas.get_renderer())
            assert fig.bbox.contains(*bounds.min) and fig.bbox.contains(*bounds.max)
    finally:
        plt.close(fig)


def test_event_guides_use_observed_boundaries_and_saved_recovery():
    event = {"attack_id": "A"}
    states = [[], [event], [event, {"attack_id": "B"}],
              [dict(event, is_recovery_effect=True)], [], [event]]
    snapshots = [{"step_index": i, "time_seconds": 5 * i, "attacks": {"events": events}}
                 for i, events in enumerate(states)]
    result = {"snapshots": snapshots, "resilience_evaluation": {"episodes": [
        {"recovered_step": 4}, {"recovered_step": None}]}}
    axis = charts.RunTimeAxis.from_result(result)
    assert axis.starts == (5, 10, 25)
    assert axis.ends == (15,)
    assert axis.recoveries == (20,)
    # Конец окна наблюдения и отсутствующие снимки не подтверждают конец атаки.
    assert charts.RunTimeAxis.from_result({"snapshots": snapshots[2::2],
        "config": {"step_seconds": 5}}).ends == ()


def test_export_projects_same_second_to_same_pixel_across_time_charts(tmp_path, monkeypatch):
    snapshots = _snapshots()
    snapshots[1]["attacks"]["events"] = [{"attack_id": "one", "effective_packet_rate_pps": 0}]
    result = {"snapshots": snapshots, "resilience_evaluation": {"episodes": [{"recovered_step": 2}]}}
    projections = {}

    def check_figure(fig, path, *, tight=True):
        try:
            if path.name == "capacity_bottlenecks.png":
                return
            assert not tight
            for ax in fig.axes:
                assert ax.get_xlim() == (0, 8)
                projections.setdefault(path.name, []).append(ax.transData.transform((4, 0))[0])
                if path.name == "event_timeline.png":
                    start, end = [c.get_segments() for c in ax.collections[-2:]]
                    np.testing.assert_allclose(start, [[[2, 0], [2, 1]]])
                    np.testing.assert_allclose(end, [[[4, 0], [4, 1]]])
                    continue
                start, end, restored = [c.get_segments() for c in ax.collections[-3:]]
                np.testing.assert_allclose(start, [[[2, 0], [2, 1]]])
                np.testing.assert_allclose(end, [[[4, 0], [4, .5]]])
                np.testing.assert_allclose(restored, [[[4, .5], [4, 1]]])
        finally:
            plt.close(fig)

    monkeypatch.setattr(charts, "save_figure", check_figure)
    charts.export_dynamics_charts(result, tmp_path)
    assert "attack_traffic.png" in projections and "event_timeline.png" in projections
    assert len(projections["incident_overview.png"]) == 3
    pixels = [x for axes in projections.values() for x in axes]
    np.testing.assert_allclose(pixels, pixels[0])
