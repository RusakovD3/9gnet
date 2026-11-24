from src.gnet9.route_stress import METHOD, RouteStressMonitor, evaluate_route_stress


def dns(client, server, queries):
    return {"traffic_kind": "dns", "client_node": client, "server_node": server, "route": [client, server],
            "dns_queries": queries, "dns_timeout_count": 0, "slo_evaluation": {"metrics": {
                "dns_timeout_ratio_percent": {"required": True, "compliant": True, "comparison": "lte",
                                              "unit": "%", "budget_value": 1, "observed_value": 0.}}}}


def sample(value, onset=50):
    return [{"sensor_node_id": "S", "start_step": onset, "signals": {"dns_query_pressure_ratio": value}}]


def test_warning_needs_local_rising_readings_and_strict_one_loss_sensitivity():
    monitor = RouteStressMonitor(2)
    flows = [dns("A", "S", 10), dns("B", "X", 10), dns("C", "S", 100)]
    assert not monitor.observe(flows, sample(.45, 9), time_seconds=0)["warnings"]
    view = monitor.observe(flows, sample(.60, 999), time_seconds=2)
    assert [w["flow_id"] for w in view["warnings"]] == ["A"]
    warning = view["warnings"][0]
    assert warning["conditional_value_if_one_more_loss"] == 10.
    assert warning["measured_value"] == 0.
    assert warning["window_end_seconds"] == 22
    assert [key[1] for key in view["sensor_only"]] == ["A", "C"]
    assert [key[1] for key in view["one_drop_only"]] == ["A", "B"]
    assert not monitor.observe(flows, sample(.90), time_seconds=6)["warnings"]


def test_interval_scores_use_only_positive_leads_and_censor_incomplete_controls():
    monitor = RouteStressMonitor(2)
    flows = [dns("A", "S", 10), dns("B", "X", 10)]
    views = [monitor.observe(flows, sample(v), time_seconds=t) for v,t in ((.3,0),(.4,2),(.4,4))]
    onset = {"service": "dns", "flow_id": "A", "criterion": "dns_timeout_ratio_percent"}
    truth = [{"time_seconds": t, "onsets": [onset] if t in (2,4) else [],
              "measurement_gaps": []} for t in (0,2,4)]
    scores = evaluate_route_stress(views, truth)["methods"]
    assert scores[METHOD]["onsets"] == 2
    assert scores[METHOD]["hits"] == scores[METHOD]["misses"] == 1
    assert scores[METHOD]["lead_seconds"] == [2]
    assert scores[METHOD]["median_lead_seconds"] == 2
    assert scores[METHOD]["issued_samples"] == scores[METHOD]["issued_windows"] == 1
    assert scores["one_drop_only"]["censored"] > 0


def test_false_alarms_per_hour_uses_only_adjacent_measured_intervals():
    monitor = RouteStressMonitor(2)
    flow = dns("A", "S", 10)
    times = list(range(0, 30, 2))
    views = [monitor.observe([] if t == 24 else [flow], sample(.3 if t == 0 else .4), time_seconds=t)
             for t in times]
    truth = [{"time_seconds": t, "onsets": [],
              "measurement_gaps": [("dns", "A", "dns_timeout_ratio_percent")] if t == 24 else []}
             for t in times]
    result = evaluate_route_stress(views, truth)["methods"][METHOD]
    assert result["false_alarms"] == result["issued_windows"] == 1
    assert result["observed_hours"] == 24 / 3600.
    assert result["false_alarms_per_observed_hour"] == 150
    assert result["median_lead_seconds"] is None
    empty = evaluate_route_stress(views[:1], truth[:1])["methods"][METHOD]
    assert empty["issued_samples"] == 0
    assert empty["false_alarms_per_observed_hour"] is None
