from src.gnet9.route_stress import METHOD, RouteStressMonitor, evaluate_route_stress


def dns(client, server, queries):
    return {"traffic_kind": "dns", "client_node": client, "server_node": server, "route": [client, server],
            "dns_queries": queries, "dns_timeout_count": 0, "slo_evaluation": {"metrics": {
                "dns_timeout_ratio_percent": {"required": True, "compliant": True, "comparison": "lte",
                                              "unit": "%", "budget_value": 1, "observed_value": 0.}}}}


def sample(value, onset=50):
    return [{"sensor_node_id": "S", "start_step": onset, "signals": {"dns_query_pressure_ratio": value}}]


def test_operator_forecasts_next_sample_from_local_delay_and_measured_route_signal():
    def media(client, node, count, latency, loss=0):
        return {"client_node": client, "traffic_kind": "voice", "route": [client, node], "packet_count": count,
                "observed_dropped_packets": loss, "slo_evaluation": {"metrics": {
                    name: {"compliant": True, "required": True, "comparison": "lte", "unit": unit,
                           "observed_value": value, "effective_budget_value": limit}
                    for name, unit, value, limit in (("one_way_media_latency_ms", "ms", latency, 150),
                            ("network_packet_loss_percent", "%", 100 * loss / count, 1))}}}

    initial = [media("A", "S", 100, 90), media("B", "S", 50, 10), media("C", "X", 50, 10)]
    current = [media("A", "S", 100, 140, 1), *initial[1:]]
    def run(hidden):
        monitor = RouteStressMonitor(2)
        assert not monitor.observe(initial, [], time_seconds=0)["operator_warnings"]
        readings = [{"sensor_node_id": "S", "signals": {"traffic_acceleration_ratio": .6}, "start_step": hidden}]
        issued = monitor.observe(current, readings, time_seconds=2)["operator_warnings"]
        assert not monitor.observe(current, readings, time_seconds=4)["operator_warnings"]
        assert not monitor.observe(current, [], time_seconds=8)["operator_warnings"]
        return issued

    warnings = run(99)
    assert warnings == run(5)
    assert {(w["flow_id"], w["criterion"], w["predicted_value"]) for w in warnings} == {
        ("A", "one_way_media_latency_ms", 165), ("A", "network_packet_loss_percent", 2),
        ("B", "network_packet_loss_percent", 2)}
    assert all(w["issue_time_seconds"] == 2 and w["target_time_seconds"] == 4 for w in warnings)
    monitor = RouteStressMonitor(2)
    monitor.observe(initial, None, time_seconds=0)
    readings = [{"sensor_node_id": "S", "signals": {"traffic_acceleration_ratio": .6}}]
    assert not monitor.observe(initial, readings, time_seconds=2)["operator_warnings"]


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


def test_queue_delay_is_counted_once_and_bitrate_uses_comparable_offers():
    def media(name, delay, queue, loss, measured):
        def metric(value, limit, comparison, unit):
            return dict(observed_value=value, budget_value=limit, effective_budget_value=limit,
                        compliant=True, comparison=comparison, unit=unit, required=True)
        return {"client_node": name, "traffic_kind": "vlc_av", "sla_grade": "gold", "packet_count": 100,
                "observed_dropped_packets": loss, "payload_bytes": 25000, "interval_seconds": 2,
                "transport_queue": {"additional_delay_ms": queue, "utilization_rho": 2},
                "slo_evaluation": {"metrics": {
                    "one_way_media_latency_ms": metric(delay, 250, "lte", "ms"),
                    "media_bitrate_kbps": metric(measured, 82, "gte", "kbit/s")}}}
    first = [media(n, 10, 0, 0, 100) for n in "ABC"]
    now = [media(n, 200, 60, 16, 84) for n in "ABC"]
    for version in (2, 3, 4):
        monitor = RouteStressMonitor(2, operator_version=version)
        monitor.observe(first, [], time_seconds=0)
        warnings = monitor.observe(now, [], time_seconds=2)["operator_warnings"]
        if version == 2:
            assert {w["predicted_value"] for w in warnings} == {295}
        else:
            assert len(warnings) == 3
            assert {(w["criterion"], w["predicted_value"], w["predicted_headroom"]) for w in warnings} == {
                ("media_bitrate_kbps", 80, -2)}
            monitor.observe([], [], time_seconds=4)
            assert not monitor.observe(now, [], time_seconds=6)["operator_warnings"]

    # При среднем спросе 100 кбит/с разброс одновременных предложений 90, 100, 110
    # оставляет реалистичный исход выше предела 82 кбит/с даже при потере 20%.
    for row, demand in zip(now, (90, 100, 110)):
        row["payload_bytes"] = 250 * demand
        metric = row["slo_evaluation"]["metrics"]["media_bitrate_kbps"]
        metric.update(observed_value=.84 * demand, compliant=.84 * demand >= 82)
    for version, expected in ((3, ["B", "C"]), (4, [])):
        monitor = RouteStressMonitor(2, operator_version=version)
        monitor.observe(first, [], time_seconds=0)
        issued = monitor.observe(now, [], time_seconds=2)["operator_warnings"]
        assert [w["flow_id"] for w in issued if w["criterion"] == "media_bitrate_kbps"] == expected


def test_dns_warning_needs_delivered_pressure_and_observed_timeouts_on_same_server():
    def flows(lost):
        rows = [dns("A", "S", 4), dns("B", "S", 4), dns("C", "X", 4)]
        for row in rows:
            row["sla_grade"] = "gold"
        rows[0]["dns_timeout_count"] = lost
        if lost:
            rows[0]["slo_evaluation"]["metrics"]["dns_timeout_ratio_percent"].update(
                compliant=False, observed_value=25.)
        return rows
    for readings in (None, [], sample(.9)):
        monitor = RouteStressMonitor(2)
        monitor.observe(flows(0), [], time_seconds=0)
        issued = monitor.observe(flows(1), readings, time_seconds=2)["operator_warnings"]
        assert [(w["flow_id"], w["predicted_value"]) for w in issued] == ([("B", 12.5)] if readings else [])
