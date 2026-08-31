"""Предиктивное переназначение легитимного трафика без изменения топологии.

Модуль принимает уже построенные записи потоков и накладывает на них временный
маршрутный overlay.  Исходный :class:`networkx.Graph`, входные потоки и план
защиты не изменяются.  Это важно для сравнения состояния с эталоном t0 и для
последующего расчёта расстояния Хаусдорфа между исходными и активными маршрутами.

Решения принимаются строго в порядке Gold -> Silver -> Bronze.  Каждый новый
путь проверяется против 80-процентного предела как по каналам, так и по узлам.
При failover сервиса дополнительно проверяются прогнозные CPU, RAM и число сесий
резервного сервера с той же безопасной 80-процентной границей.
Числа здесь являются оценкой модели, а не конфигурацией реального оборудования.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import islice
import math
from typing import Any

import networkx as nx

from .models import NetworkModel, StateTensor
from .routing import DATA_PLANE_TRANSIT_ROLES, data_plane_routing_view


SLA_ORDER = {"gold": 0, "silver": 1, "bronze": 2}
MAX_REMAP_UTILIZATION_PERCENT = 80.0
NETWORK_ATTACK_KINDS = {"dos", "ddos", "syn_flood"}
NETWORK_DEVICE_ROLES = set(DATA_PLANE_TRANSIT_ROLES)
SERVER_ROLE = "service-server"
SUBSCRIBER_ROLES = {"mobile-subscriber", "fixed-subscriber"}
_MAX_PATH_CANDIDATES = 16


def apply_gold_first_remap(
    model: NetworkModel,
    flows: list[dict[str, Any]],
    events: list[dict[str, Any]] | None,
    defense_plan: dict[str, Any] | None,
    *,
    step_index: int,
    identities: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Применить подготовленный Купманом маршрутный overlay.

    ``defense_plan`` может содержать параметры непосредственно или во вложенном
    словаре ``routing_control``.  Поддерживаются как одношаговое поле
    ``valid_for_step``, так и полуинтервал ``valid_from_step`` / ``valid_until_step``.

    Возвращаются новые записи потоков и сериализуемая сводка.  Атакующий трафик
    намеренно пропускается: генератор атак вызывается после этого этапа.
    """
    copied_flows = [_copy_flow(flow) for flow in flows]
    control = _routing_control(defense_plan)
    plan_is_active, valid_from, valid_until = _plan_window(defense_plan, control, step_index)
    context = _control_context(model, events or [], defense_plan or {}, control)
    server_resource_state = _initial_server_resource_state(model)
    flow_resource_demands = _flow_server_resource_demands(
        copied_flows,
        server_resource_state,
    )

    for flow in copied_flows:
        _initialize_overlay_fields(flow)

    if not plan_is_active:
        return copied_flows, _build_summary(
            model,
            copied_flows,
            step_index=step_index,
            plan_active=False,
            valid_from=valid_from,
            valid_until=valid_until,
            actions=[],
            capacity_rejections=0,
            network_capacity_rejections=0,
            server_resource_rejections=0,
            server_resource_rejection_reasons={},
            projected_edge_loads=_initial_edge_loads(copied_flows),
            projected_node_loads=_initial_node_loads(copied_flows),
            projected_server_resources=server_resource_state,
            context=context,
        )

    # Резервирование строится заново в строгом SLA-порядке. Если заранее
    # занести сюда Silver/Bronze baseline, они смогут занять альтернативу до
    # Gold и формальный приоритет будет нарушен.
    edge_loads: dict[tuple[str, str], float] = {}
    node_loads: dict[str, float] = {}
    actions: list[dict[str, Any]] = []
    network_capacity_rejections = 0
    server_resource_rejections = 0
    server_resource_rejection_reasons: Counter[str] = Counter()

    # Стабильная сортировка сохраняет исходный порядок внутри одного SLA-класса.
    ordered = sorted(
        enumerate(copied_flows),
        key=lambda item: (SLA_ORDER.get(str(item[1].get("sla_grade", "bronze")).lower(), 3), item[0]),
    )

    for _, flow in ordered:
        if flow.get("is_attack_traffic"):
            continue

        client = str(flow.get("client_node", ""))
        grade = str(flow.get("sla_grade", "bronze")).lower()
        original_server = str(flow.get("original_server_node") or flow.get("server_node", ""))
        original_nodes = _flow_nodes(flow)
        rate_mbps = _flow_rate_mbps(flow)

        if client in context["quarantine_sources"]:
            reason = "источник сетевой атаки временно помещён в карантин"
            _isolate_flow(flow, "quarantine_attack_source", reason)
            actions.append(_action_record(flow, context, grade))
            continue

        endpoint_kinds = context["target_kinds"].get(client, set())
        if client in context["endpoint_targets"]:
            # SYN-proxy/ACL защищает жертву; отключение жертвы означало бы успех атаки.
            action = "protect_endpoint_syn_proxy" if "syn_flood" in endpoint_kinds else "protect_endpoint_rate_limit"
            reason = (
                "жертва остаётся подключённой; SYN-proxy и ACL ограничивают входящий flood"
                if action == "protect_endpoint_syn_proxy"
                else "жертва остаётся подключённой; ограничивается только вредоносный входящий поток"
            )
            flow["endpoint_protected"] = True
            flow["routing_action"] = action
            flow["routing_reason"] = reason
            flow["service_available"] = True
            if _path_has_capacity(
                model, list(flow.get("route") or []), rate_mbps, edge_loads, node_loads
            ):
                _reserve_path(list(flow.get("route") or []), rate_mbps, edge_loads, node_loads)
                actions.append(_action_record(flow, context, grade))
                continue
            network_capacity_rejections += 1

        service = str(flow.get("service_node", ""))
        server_is_target = original_server in context["server_targets"]
        service_is_target = service in context["service_targets"]
        route_hits_target = bool(original_nodes & context["avoid_nodes"])
        requires_safe_route = server_is_target or service_is_target or route_hits_target
        if not requires_safe_route:
            original_route = list(flow.get("route") or [])
            if _path_has_capacity(model, original_route, rate_mbps, edge_loads, node_loads):
                _reserve_path(original_route, rate_mbps, edge_loads, node_loads)
                continue
            # Gold уже зарезервирован первым; нижний SLA-класс ищет другой
            # маршрут либо структурированно изолируется вместо вытеснения Gold.
            network_capacity_rejections += 1

        server_candidates = [original_server]
        if server_is_target or service_is_target:
            server_candidates = _standby_servers(
                model,
                flow,
                control,
                context["avoid_nodes"],
            )
            if not server_candidates:
                _isolate_flow(
                    flow,
                    "isolate_no_standby_server",
                    "для поражённого сервиса нет доступного резервного сервера",
                )
                actions.append(_action_record(flow, context, grade))
                continue

        selected_server = ""
        selected_resource_demand: Mapping[str, float] | None = None
        candidate: list[str] = []
        for standby in server_candidates:
            resource_demand = flow_resource_demands.get(id(flow), {})
            if standby != original_server:
                admitted, rejection_reasons = _server_accepts_failover(
                    server_resource_state,
                    standby,
                    resource_demand,
                )
                if not admitted:
                    server_resource_rejections += 1
                    server_resource_rejection_reasons.update(rejection_reasons)
                    continue
            avoid_nodes = set(context["avoid_nodes"])
            # Клиент является конечной точкой, а не транзитным узлом; защищаем
            # его, но не удаляем из графа при поиске пути. Выбранный standby
            # тоже является допустимой конечной точкой.
            avoid_nodes.discard(client)
            avoid_nodes.discard(standby)
            candidate, rejected = _select_capacity_aware_path(
                model,
                standby,
                client,
                avoid_nodes,
                rate_mbps,
                edge_loads,
                node_loads,
            )
            network_capacity_rejections += rejected
            if candidate:
                selected_server = standby
                selected_resource_demand = resource_demand
                break

        if not candidate:
            power_aggregation = bool(
                context["power_targets"]
                & {node for node in original_nodes if _node_role(model, node) == "aggregation-switch"}
            )
            reason = _no_path_reason(model, client, grade, power_aggregation)
            _isolate_flow(flow, "isolate_no_safe_route", reason)
            actions.append(_action_record(flow, context, grade))
            continue

        failover = selected_server != original_server
        _reserve_path(candidate, rate_mbps, edge_loads, node_loads)
        if failover and selected_resource_demand is not None:
            _reserve_server_resources(
                server_resource_state,
                selected_server,
                selected_resource_demand,
            )
        _activate_route(
            model,
            flow,
            candidate,
            selected_server=selected_server,
            failover=failover,
            identities=identities,
        )
        actions.append(_action_record(flow, context, grade))

    return copied_flows, _build_summary(
        model,
        copied_flows,
        step_index=step_index,
        plan_active=True,
        valid_from=valid_from,
        valid_until=valid_until,
        actions=actions,
        capacity_rejections=network_capacity_rejections + server_resource_rejections,
        network_capacity_rejections=network_capacity_rejections,
        server_resource_rejections=server_resource_rejections,
        server_resource_rejection_reasons=server_resource_rejection_reasons,
        projected_edge_loads=edge_loads,
        projected_node_loads=node_loads,
        projected_server_resources=server_resource_state,
        context=context,
    )


def _copy_flow(flow: Mapping[str, Any]) -> dict[str, Any]:
    """Скопировать запись достаточно глубоко для безопасной замены маршрутов."""
    copied = dict(flow)
    for field in ("route", "reverse_route", "original_route", "original_reverse_route"):
        value = copied.get(field)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            copied[field] = list(value)
    return copied


def _initialize_overlay_fields(flow: dict[str, Any]) -> None:
    route = list(flow.get("route") or [])
    reverse = list(flow.get("reverse_route") or [])
    flow.setdefault("original_route", route)
    flow.setdefault("original_reverse_route", reverse)
    flow.setdefault("original_server_node", flow.get("server_node"))
    flow.setdefault("active_access_node", flow.get("home_access_node"))
    flow.setdefault("active_access_role", "primary")
    flow["rerouted"] = False
    flow["failover_active"] = False
    flow["isolated"] = False
    flow["endpoint_protected"] = False
    flow["security_excluded_from_sla_accounting"] = False
    flow["route_available"] = bool(route)
    flow["service_available"] = bool(route)
    flow["routing_action"] = "keep_baseline_route"
    flow["routing_reason"] = "маршрут не требует предиктивного переназначения"
    flow["route_stretch_ratio"] = 1.0
    flow["additional_latency_ms"] = 0.0


def _routing_control(defense_plan: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(defense_plan, Mapping):
        return {}
    nested = defense_plan.get("routing_control")
    return dict(nested) if isinstance(nested, Mapping) else {}


def _plan_window(
    defense_plan: Mapping[str, Any] | None,
    control: Mapping[str, Any],
    step_index: int,
) -> tuple[bool, int | None, int | None]:
    if not isinstance(defense_plan, Mapping) or not defense_plan:
        return False, None, None

    def first_int(name: str) -> int | None:
        for source in (control, defense_plan):
            value = source.get(name)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
        return None

    exact = first_int("valid_for_step")
    start = first_int("valid_from_step")
    end = first_int("valid_until_step")
    if exact is not None:
        start = exact if start is None else start
        end = exact if end is None else end
    if start is None and end is None:
        return True, None, None
    if start is None:
        start = step_index
    if end is None:
        end = start
    return start <= step_index <= end, start, end


def _control_context(
    model: NetworkModel,
    events: list[dict[str, Any]],
    defense_plan: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    targets = _string_set(control.get("target_ids")) | _string_set(defense_plan.get("target_ids"))
    explicit_avoid = _string_set(control.get("avoid_nodes")) | _string_set(defense_plan.get("avoid_nodes"))
    plan_attack_ids = _string_set(control.get("attack_ids")) | _string_set(defense_plan.get("attack_ids"))
    plan_kinds = _string_set(control.get("attack_kinds")) | _string_set(defense_plan.get("attack_kinds"))
    observed_sources = (
        _string_set(control.get("quarantine_sources"))
        | _string_set(control.get("source_ids"))
        | _string_set(control.get("ingress_nodes"))
        | _string_set(defense_plan.get("source_ids"))
        | _string_set(defense_plan.get("ingress_nodes"))
    )
    # Предсказательный источник — ещё не доказанный злоумышленник. В отличие от
    # observed_sources, этот набор используется только для уже разрешённого
    # карантина: явного подтверждённого плана или активного события.
    quarantine_candidates = (
        _string_set(control.get("quarantine_sources"))
        | _string_set(defense_plan.get("quarantine_sources"))
    )
    target_kinds: dict[str, set[str]] = defaultdict(set)
    transparent_power_targets: set[str] = set()
    nontransparent_event_targets: set[str] = set()
    active_event_observed = False
    entity_records: list[Mapping[str, Any]] = []
    for source in (control, defense_plan):
        value = source.get("entity_records", [])
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
            entity_records.extend(item for item in value if isinstance(item, Mapping))
    for record in entity_records:
        target = str(record.get("target_id", ""))
        kind = _kind_text(record.get("kind"))
        if target:
            targets.add(target)
            if kind:
                target_kinds[target].add(kind)
        observed_sources.update(_string_set(record.get("source_ids")))

    for event in events:
        event_id = str(event.get("attack_id", ""))
        target = str(event.get("target_id", ""))
        kind = _kind_text(event.get("kind"))
        matches = (
            not (plan_attack_ids or targets or plan_kinds)
            or event_id in plan_attack_ids
            or target in targets
            or kind in plan_kinds
        )
        if not matches:
            continue
        active_event_observed = True
        if target:
            targets.add(target)
            target_kinds[target].add(kind)
            if _is_service_transparent_power_event(event):
                transparent_power_targets.add(target)
            else:
                nontransparent_event_targets.add(target)
        if kind in NETWORK_ATTACK_KINDS:
            observed_sources.update(_string_set(event.get("ingress_nodes")))
            quarantine_candidates.update(_string_set(event.get("ingress_nodes")))

    # Фактическая телеметрия сильнее предварительного плана: если UPS держит
    # нагрузку и доступность сервиса равна 1, узел остаётся наблюдаемой целью,
    # но не выдаётся за отказавший data-plane элемент. Явный avoid_nodes или
    # другое одновременное воздействие по-прежнему требуют обхода.
    transparent_power_targets.difference_update(nontransparent_event_targets)

    # Для прогнозного шага активных events ещё нет, поэтому тип берётся из
    # классификатора плана, но сама дата начала атаки модулю не передаётся.
    for target in targets:
        if not target_kinds[target]:
            target_kinds[target].update(plan_kinds)

    service_targets: set[str] = set()
    server_targets: set[str] = set()
    endpoint_targets: set[str] = set()
    network_targets: set[str] = set()
    power_targets: set[str] = set()

    for target in targets | explicit_avoid:
        if target not in model.graph:
            continue
        role = _node_role(model, target)
        level = str(model.graph.nodes[target].get("level", ""))
        kinds = target_kinds.get(target, plan_kinds)
        requires_avoidance = target not in transparent_power_targets or target in explicit_avoid
        if "power_attack" in kinds and requires_avoidance:
            power_targets.add(target)
        if not requires_avoidance:
            continue
        if role == SERVER_ROLE:
            server_targets.add(target)
        elif role in SUBSCRIBER_ROLES or level == "L1":
            endpoint_targets.add(target)
        elif role == "service":
            service_targets.add(target)
        elif role in NETWORK_DEVICE_ROLES:
            network_targets.add(target)

    avoid_nodes = explicit_avoid | network_targets | server_targets | service_targets
    # Логический узел сервиса обычно не лежит на data-plane-маршруте; его
    # физический primary server исключается отдельно при выборе standby.
    for service in service_targets:
        primary = model.graph.nodes[service].get("hosted_on")
        if primary:
            server_targets.add(str(primary))
            avoid_nodes.add(str(primary))

    # Карантин применим только к абонентским источникам.  Маршрутизатор в поле
    # ingress означает точку наблюдения, а не заражённого абонента.
    quarantine_sources = {
        node for node in quarantine_candidates
        if node in model.graph
        and (model.graph.nodes[node].get("level") == "L1" or _node_role(model, node) in SUBSCRIBER_ROLES)
    }

    return {
        "targets": targets,
        "avoid_nodes": avoid_nodes,
        "server_targets": server_targets,
        "service_targets": service_targets,
        "endpoint_targets": endpoint_targets,
        "network_targets": network_targets,
        "power_targets": power_targets,
        "service_transparent_power_targets": transparent_power_targets,
        "observed_source_ids": observed_sources,
        "quarantine_sources": quarantine_sources,
        "protection_stage": (
            "confirmed_active_mitigation"
            if active_event_observed
            else str(control.get("protection_stage", "inactive"))
        ),
        "source_quarantine_policy": str(
            control.get("source_quarantine_policy", "explicit_plan_only")
        ),
        "target_kinds": target_kinds,
        "attack_kinds": plan_kinds,
        "sdn_intent": dict(defense_plan.get("sdn_intent", {})),
    }


def _kind_text(value: Any) -> str:
    if hasattr(value, "value"):
        value = value.value
    text = str(value or "").lower()
    return text.rsplit(".", 1)[-1]


def _is_service_transparent_power_event(event: Mapping[str, Any]) -> bool:
    """Вернуть True только для подтверждённого UPS ride-through без outage."""
    if _kind_text(event.get("kind")) != "power_attack":
        return False
    runtime = event.get("power_runtime")
    if isinstance(runtime, Mapping):
        return (
            bool(runtime.get("ride_through_active"))
            and float(runtime.get("service_availability_ratio", 0.0)) >= 0.999999
            and float(runtime.get("power_output_availability_ratio", 0.0)) >= 0.999999
        )
    return float(event.get("power_service_impact_ratio", 1.0)) <= 0.0


def _string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    if isinstance(value, Mapping):
        return {str(key) for key, enabled in value.items() if enabled}
    if isinstance(value, Iterable):
        return {str(item) for item in value if item is not None and str(item)}
    return {str(value)}


def _standby_servers(
    model: NetworkModel,
    flow: Mapping[str, Any],
    control: Mapping[str, Any],
    avoid_nodes: set[str],
) -> list[str]:
    service = str(flow.get("service_node", ""))
    primary = str(flow.get("original_server_node") or flow.get("server_node", ""))
    configured = control.get("service_failovers", {})
    candidates: list[str] = []
    if isinstance(configured, Mapping):
        candidates.extend(_server_candidates(configured.get(service)))
        candidates.extend(_server_candidates(configured.get(primary)))

    if service in model.graph:
        attrs = model.graph.nodes[service]
        for field in ("standby_hosts", "standby_servers", "backup_servers", "standby_server_ids"):
            candidates.extend(_server_candidates(attrs.get(field)))

    if primary in model.graph:
        attrs = model.graph.nodes[primary]
        for field in ("standby_hosts", "standby_servers", "backup_servers"):
            candidates.extend(_server_candidates(attrs.get(field)))

    selected: list[str] = []
    seen: set[str] = set()
    for server in candidates:
        if server in seen:
            continue
        seen.add(server)
        if server == primary or server in avoid_nodes or server not in model.graph:
            continue
        if _node_role(model, server) == SERVER_ROLE:
            selected.append(server)
    return selected


def _server_candidates(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Mapping):
        for field in ("server_id", "id", "host"):
            if value.get(field):
                return [str(value[field])]
        return [str(key) for key, enabled in value.items() if enabled]
    if isinstance(value, Iterable):
        result: list[str] = []
        for item in value:
            result.extend(_server_candidates(item))
        return result
    return [str(value)]


def _initial_server_resource_state(
    model: NetworkModel,
) -> dict[str, dict[str, float]]:
    """Скопировать паспортные и runtime-ресурсы серверов в локальный overlay.

    Ёмкость сессий отсутствует в паспорте Dell: она честно выводится из
    наблюдаемого числа сессий и наиболее загруженного baseline-ресурса. Поэтому
    это сценарная оценка headroom, а не заявленная производителем величина.
    """
    state: dict[str, dict[str, float]] = {}
    for server_id, attrs in model.graph.nodes(data=True):
        if str(attrs.get("role", "")) != SERVER_ROLE:
            continue
        runtime = attrs.get("runtime")
        profile = attrs.get("server_profile")
        if not isinstance(runtime, Mapping) or not isinstance(profile, Mapping):
            continue
        cores = max(0.0, _resource_number(profile.get("total_cores")))
        ram_gb = max(0.0, _resource_number(profile.get("ram_gb")))
        cpu_percent = max(0.0, _resource_number(runtime.get("cpu_util_percent")))
        ram_percent = max(0.0, _resource_number(runtime.get("ram_util_percent")))
        network_percent = max(0.0, _resource_number(runtime.get("network_util_percent")))
        sessions = max(0.0, _resource_number(runtime.get("active_sessions")))
        limiting_percent = max(cpu_percent, ram_percent, network_percent)
        if sessions > 0.0 and limiting_percent > 0.0:
            session_capacity = float(math.ceil(sessions * 100.0 / limiting_percent))
        else:
            # Для пустого baseline используем лишь воспроизводимый fallback;
            # текущий каталог всегда содержит измеренное число сессий.
            session_capacity = max(
                1.0,
                _resource_number(profile.get("threads_total"), cores * 2.0),
            )
        state[str(server_id)] = {
            "cpu_capacity_cores": cores,
            "ram_capacity_gb": ram_gb,
            "baseline_cpu_util_percent": cpu_percent,
            "baseline_ram_util_percent": ram_percent,
            "baseline_network_util_percent": network_percent,
            "baseline_active_sessions": sessions,
            "inferred_session_capacity": max(sessions, session_capacity),
            "baseline_cpu_busy_cores": cores * cpu_percent / 100.0,
            "baseline_ram_used_gb": ram_gb * ram_percent / 100.0,
            "additional_cpu_busy_cores": 0.0,
            "additional_ram_gb": 0.0,
            "additional_sessions": 0.0,
            "transferred_flow_count": 0.0,
        }
    return state


def _flow_server_resource_demands(
    flows: Sequence[Mapping[str, Any]],
    server_resource_state: Mapping[str, Mapping[str, float]],
) -> dict[int, dict[str, float]]:
    """Оценить переносимую CPU/RAM-нагрузку каждого легитимного потока.

    Baseline busy cores и RAM исходного сервера делятся между его активными
    сессиями. Вес потока на 50% является фиксированной стоимостью сессии и на
    50% зависит от его wire-rate. Нормировка сохраняет агрегатную runtime-
    нагрузку, когда в вызове присутствуют все наблюдаемые сессии.
    """
    by_server: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for flow in flows:
        if flow.get("is_attack_traffic"):
            continue
        server_id = str(flow.get("original_server_node") or flow.get("server_node", ""))
        if server_id in server_resource_state:
            by_server[server_id].append(flow)

    demands: dict[int, dict[str, float]] = {}
    for server_id, server_flows in by_server.items():
        state = server_resource_state[server_id]
        rates = [_flow_rate_mbps(flow) for flow in server_flows]
        average_rate = sum(rates) / max(len(rates), 1)
        weights = [
            1.0 if average_rate <= 0.0 else 0.5 + 0.5 * rate / average_rate
            for rate in rates
        ]
        denominator = max(
            float(state.get("baseline_active_sessions", 0.0)),
            sum(weights),
            1.0,
        )
        for flow, weight in zip(server_flows, weights):
            demands[id(flow)] = {
                "cpu_busy_cores": (
                    float(state.get("baseline_cpu_busy_cores", 0.0)) * weight / denominator
                ),
                "ram_gb": (
                    float(state.get("baseline_ram_used_gb", 0.0)) * weight / denominator
                ),
                "sessions": 1.0,
                "load_weight": weight,
            }
    return demands


def _server_accepts_failover(
    server_resource_state: Mapping[str, Mapping[str, float]],
    server_id: str,
    demand: Mapping[str, float],
) -> tuple[bool, list[str]]:
    state = server_resource_state.get(server_id)
    if state is None:
        return False, ["server_resource_profile_unavailable"]
    projection = _server_resource_projection(state, demand)
    reasons: list[str] = []
    if projection["cpu_utilization_percent"] > MAX_REMAP_UTILIZATION_PERCENT + 1e-9:
        reasons.append("cpu_above_80_percent")
    if projection["ram_utilization_percent"] > MAX_REMAP_UTILIZATION_PERCENT + 1e-9:
        reasons.append("ram_above_80_percent")
    if projection["session_utilization_percent"] > MAX_REMAP_UTILIZATION_PERCENT + 1e-9:
        reasons.append("sessions_above_80_percent")
    return not reasons, reasons


def _reserve_server_resources(
    server_resource_state: dict[str, dict[str, float]],
    server_id: str,
    demand: Mapping[str, float],
) -> None:
    state = server_resource_state[server_id]
    state["additional_cpu_busy_cores"] += max(
        0.0, _resource_number(demand.get("cpu_busy_cores"))
    )
    state["additional_ram_gb"] += max(0.0, _resource_number(demand.get("ram_gb")))
    state["additional_sessions"] += max(0.0, _resource_number(demand.get("sessions")))
    state["transferred_flow_count"] += 1.0


def _server_resource_projection(
    state: Mapping[str, float],
    demand: Mapping[str, float] | None = None,
) -> dict[str, float]:
    demand = demand or {}
    cores = max(0.0, float(state.get("cpu_capacity_cores", 0.0)))
    ram_gb = max(0.0, float(state.get("ram_capacity_gb", 0.0)))
    session_capacity = max(0.0, float(state.get("inferred_session_capacity", 0.0)))
    cpu_busy = (
        float(state.get("baseline_cpu_busy_cores", 0.0))
        + float(state.get("additional_cpu_busy_cores", 0.0))
        + max(0.0, _resource_number(demand.get("cpu_busy_cores")))
    )
    ram_used = (
        float(state.get("baseline_ram_used_gb", 0.0))
        + float(state.get("additional_ram_gb", 0.0))
        + max(0.0, _resource_number(demand.get("ram_gb")))
    )
    sessions = (
        float(state.get("baseline_active_sessions", 0.0))
        + float(state.get("additional_sessions", 0.0))
        + max(0.0, _resource_number(demand.get("sessions")))
    )
    return {
        "cpu_busy_cores": cpu_busy,
        "ram_used_gb": ram_used,
        "active_sessions": sessions,
        "cpu_utilization_percent": 100.0 if cores <= 0.0 else cpu_busy / cores * 100.0,
        "ram_utilization_percent": 100.0 if ram_gb <= 0.0 else ram_used / ram_gb * 100.0,
        "session_utilization_percent": (
            100.0 if session_capacity <= 0.0 else sessions / session_capacity * 100.0
        ),
    }


def _resource_number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _select_capacity_aware_path(
    model: NetworkModel,
    source: str,
    target: str,
    avoid_nodes: set[str],
    rate_mbps: float,
    edge_loads: dict[tuple[str, str], float],
    node_loads: dict[str, float],
) -> tuple[list[str], int]:
    if source not in model.graph or target not in model.graph:
        return [], 0
    view = data_plane_routing_view(
        model,
        endpoints=(source, target),
        excluded_nodes=avoid_nodes,
    )
    if source not in view or target not in view:
        return [], 0

    rejected = 0
    try:
        paths = nx.shortest_simple_paths(view, source, target, weight="latency_ms")
        for path in islice(paths, _MAX_PATH_CANDIDATES):
            candidate = list(path)
            if _path_has_capacity(model, candidate, rate_mbps, edge_loads, node_loads):
                return candidate, rejected
            rejected += 1
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [], rejected
    return [], rejected


def _path_has_capacity(
    model: NetworkModel,
    path: list[str],
    rate_mbps: float,
    edge_loads: Mapping[tuple[str, str], float],
    node_loads: Mapping[str, float],
) -> bool:
    for source, target in zip(path, path[1:]):
        edge = model.graph.edges[source, target]
        capacity = max(float(edge.get("capacity_mbps", 0.0)), 1e-9)
        projected = edge_loads.get(_edge_key(source, target), 0.0) + rate_mbps
        if projected / capacity * 100.0 > MAX_REMAP_UTILIZATION_PERCENT + 1e-9:
            return False
    for node in set(path):
        capacity = _node_capacity_mbps(model, node)
        if capacity <= 0.0:
            continue
        projected = node_loads.get(node, 0.0) + rate_mbps
        if projected / capacity * 100.0 > MAX_REMAP_UTILIZATION_PERCENT + 1e-9:
            return False
    return True


def _activate_route(
    model: NetworkModel,
    flow: dict[str, Any],
    server_to_client: list[str],
    *,
    selected_server: str,
    failover: bool,
    identities: Mapping[str, Any] | None,
) -> None:
    original_server = str(flow.get("original_server_node") or flow.get("server_node", ""))
    client = str(flow.get("client_node", ""))
    original_route = list(flow.get("original_route") or [])
    original_reverse = list(flow.get("original_reverse_route") or [])

    flow["route"] = _orient_path(server_to_client, original_route, original_server, client)
    flow["reverse_route"] = _orient_path(server_to_client, original_reverse, original_server, client)
    if not flow["reverse_route"]:
        flow["reverse_route"] = list(reversed(flow["route"]))

    old_latency = float(flow.get("one_way_latency_ms") or 0.0)
    packet_bytes = max(64.0, float(flow.get("wire_bytes", 0.0)) / max(int(flow.get("packet_count", 0)), 1))
    new_latency = _path_latency_ms(model, list(flow["route"]), packet_bytes)
    flow["one_way_latency_ms"] = round(new_latency, 4)
    if flow.get("rtt_ms") is not None:
        reverse_latency = _path_latency_ms(model, list(flow["reverse_route"]), packet_bytes)
        flow["rtt_ms"] = round(new_latency + reverse_latency, 4)
    flow["additional_latency_ms"] = round(new_latency - old_latency, 4)
    flow["route_stretch_ratio"] = round(new_latency / max(old_latency, 1e-9), 6)
    flow["hop_count"] = max(0, len(flow["route"]) - 1)
    flow["expected_loss_ratio"] = round(_path_expected_loss(model, flow["route"]), 9)
    flow["server_node"] = selected_server
    if identities is not None and selected_server in identities:
        identity = identities[selected_server]
        ip = identity.get("ip") if isinstance(identity, Mapping) else getattr(identity, "ip", None)
        if ip:
            flow["server_ip"] = ip
    flow["rerouted"] = list(flow["route"]) != original_route or failover
    flow["failover_active"] = failover
    flow["isolated"] = False
    flow["route_available"] = True
    flow["service_available"] = True
    active_access = _endpoint_access_node(server_to_client, client)
    home_access = model.graph.nodes[client].get("home_access") if client in model.graph else None
    access_role = "primary" if active_access == home_access else "unknown"
    flow["active_access_node"] = active_access
    flow["active_access_role"] = access_role
    if failover:
        flow["routing_action"] = "failover_server_and_reroute"
        flow["routing_reason"] = f"сервис переключён с {original_server} на резервный {selected_server}"
    else:
        flow["routing_action"] = "reroute_around_target"
        flow["routing_reason"] = "маршрут обходит прогнозируемую или активную цель атаки"


def _endpoint_access_node(path: Sequence[str], client: str) -> str | None:
    """Вернуть точку доступа, непосредственно соседнюю с клиентом в пути."""
    if len(path) < 2:
        return None
    if path[0] == client:
        return str(path[1])
    if path[-1] == client:
        return str(path[-2])
    return None


def _orient_path(
    server_to_client: list[str],
    original: list[str],
    original_server: str,
    client: str,
) -> list[str]:
    if not original:
        return list(server_to_client)
    if original[0] == client or original[-1] == original_server:
        return list(reversed(server_to_client))
    return list(server_to_client)


def _isolate_flow(flow: dict[str, Any], action: str, reason: str) -> None:
    packet_count = max(0, int(flow.get("packet_count", 0)))
    flow["route"] = []
    flow["reverse_route"] = []
    flow["hop_count"] = 0
    flow["rerouted"] = False
    flow["failover_active"] = False
    flow["isolated"] = True
    flow["route_available"] = False
    flow["service_available"] = False
    flow["routing_action"] = action
    flow["routing_reason"] = reason
    flow["security_excluded_from_sla_accounting"] = (
        action == "quarantine_attack_source"
    )
    flow["sla_exclusion_reason"] = (
        "скомпрометированный источник изолирован политикой безопасности"
        if action == "quarantine_attack_source"
        else None
    )
    flow["route_stretch_ratio"] = 0.0
    flow["additional_latency_ms"] = 0.0
    flow["expected_loss_ratio"] = 1.0
    flow["observed_dropped_packets"] = max(int(flow.get("observed_dropped_packets", 0)), packet_count)
    flow["delivered_packet_count"] = 0


def _no_path_reason(model: NetworkModel, client: str, grade: str, power_aggregation: bool) -> str:
    if power_aggregation:
        return (
            f"{grade.capitalize()} имеет единственную линию доступа; поток изолирован "
            "до восстановления питания"
        )
    return "безопасный маршрут не найден либо его прогнозная загрузка превысила 80%"


def _action_record(flow: Mapping[str, Any], context: Mapping[str, Any], grade: str) -> dict[str, Any]:
    return {
        "flow_id": flow.get("flow_id"),
        "sla_grade": grade,
        "client_node": flow.get("client_node"),
        "service_node": flow.get("service_node"),
        "server_node": flow.get("server_node"),
        "action": flow.get("routing_action"),
        "reason_ru": flow.get("routing_reason"),
        "target_ids": sorted(context.get("targets", set())),
    }


def _initial_edge_loads(flows: list[Mapping[str, Any]]) -> dict[tuple[str, str], float]:
    loads: dict[tuple[str, str], float] = defaultdict(float)
    for flow in flows:
        if flow.get("is_attack_traffic") or flow.get("isolated"):
            continue
        rate = _flow_rate_mbps(flow)
        for edge in _flow_edges(flow):
            loads[edge] += rate
    return dict(loads)


def _initial_node_loads(flows: list[Mapping[str, Any]]) -> dict[str, float]:
    loads: dict[str, float] = defaultdict(float)
    for flow in flows:
        if flow.get("is_attack_traffic") or flow.get("isolated"):
            continue
        rate = _flow_rate_mbps(flow)
        for node in _flow_nodes(flow):
            loads[node] += rate
    return dict(loads)


def _reserve_path(
    path: list[str],
    rate_mbps: float,
    edge_loads: dict[tuple[str, str], float],
    node_loads: dict[str, float],
) -> None:
    for source, target in zip(path, path[1:]):
        edge = _edge_key(source, target)
        edge_loads[edge] = edge_loads.get(edge, 0.0) + rate_mbps
    for node in set(path):
        node_loads[node] = node_loads.get(node, 0.0) + rate_mbps


def _flow_rate_mbps(flow: Mapping[str, Any]) -> float:
    explicit = flow.get("offered_rate_mbps")
    if explicit is not None:
        try:
            return max(0.0, float(explicit))
        except (TypeError, ValueError):
            pass
    interval = max(float(flow.get("interval_seconds", 1.0) or 1.0), 1e-9)
    return max(0.0, float(flow.get("wire_bytes", 0.0) or 0.0) * 8.0 / interval / 1_000_000.0)


def _flow_edges(flow: Mapping[str, Any]) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for field in ("route", "reverse_route"):
        route = list(flow.get(field) or [])
        edges.update(_edge_key(source, target) for source, target in zip(route, route[1:]))
    return edges


def _flow_nodes(flow: Mapping[str, Any]) -> set[str]:
    return set(flow.get("route") or []) | set(flow.get("reverse_route") or [])


def _edge_key(source: str, target: str) -> tuple[str, str]:
    return (source, target) if source <= target else (target, source)


def _node_role(model: NetworkModel, node: str) -> str:
    return str(model.graph.nodes[node].get("role", "")) if node in model.graph else ""


def _node_capacity_mbps(model: NetworkModel, node: str) -> float:
    if node not in model.graph:
        return 0.0
    attrs = model.graph.nodes[node]
    l2_profile = attrs.get("l2_profile")
    if isinstance(l2_profile, Mapping) and l2_profile.get("throughput_gbps") is not None:
        return max(0.0, float(l2_profile["throughput_gbps"]) * 1_000.0)
    server_profile = attrs.get("server_profile")
    if isinstance(server_profile, Mapping):
        ports = server_profile.get("network_ports_gbps", [])
        if isinstance(ports, Iterable) and not isinstance(ports, (str, bytes)):
            return max(0.0, sum(float(speed) for speed in ports) * 1_000.0)
    incident = [float(data.get("capacity_mbps", 0.0)) for *_, data in model.graph.edges(node, data=True)]
    return max(incident, default=0.0)


def _path_latency_ms(model: NetworkModel, path: list[str], packet_bytes: float) -> float:
    latency = 0.0
    for source, target in zip(path, path[1:]):
        attrs = model.graph.edges[source, target]
        capacity = max(float(attrs.get("capacity_mbps", 0.0)), 1e-9)
        latency += float(attrs.get("latency_ms", 0.0))
        latency += packet_bytes * 8.0 / (capacity * 1_000_000.0) * 1_000.0
    for node in path[1:-1]:
        attrs = model.graph.nodes[node]
        if attrs.get("level") == "L2":
            latency += _tensor_metric(attrs.get("tensor"), "packet_processing_time_ms")
    return latency


def _path_expected_loss(model: NetworkModel, path: list[str]) -> float:
    survival = 1.0
    for source, target in zip(path, path[1:]):
        tensor = model.graph.edges[source, target].get("tensor")
        survival *= 1.0 - _tensor_metric(tensor, "loss_probability")
    return max(0.0, min(1.0, 1.0 - survival))


def _tensor_metric(tensor: Any, metric_name: str) -> float:
    if not isinstance(tensor, StateTensor) or metric_name not in tensor.metric_index:
        return 0.0
    return float(tensor.data[tensor.metric_index[metric_name][0]])


def _utilization_views(
    model: NetworkModel,
    edge_loads: Mapping[tuple[str, str], float],
    node_loads: Mapping[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edges: list[dict[str, Any]] = []
    for (source, target), rate in edge_loads.items():
        if not model.graph.has_edge(source, target):
            continue
        capacity = max(float(model.graph.edges[source, target].get("capacity_mbps", 0.0)), 1e-9)
        edges.append({
            "source": source,
            "target": target,
            "load_mbps": round(rate, 6),
            "capacity_mbps": round(capacity, 6),
            "utilization_percent": round(rate / capacity * 100.0, 6),
        })
    nodes: list[dict[str, Any]] = []
    for node, rate in node_loads.items():
        capacity = _node_capacity_mbps(model, node)
        if capacity <= 0.0:
            continue
        nodes.append({
            "node_id": node,
            "load_mbps": round(rate, 6),
            "capacity_mbps": round(capacity, 6),
            "utilization_percent": round(rate / capacity * 100.0, 6),
        })
    return (
        sorted(edges, key=lambda item: item["utilization_percent"], reverse=True),
        sorted(nodes, key=lambda item: item["utilization_percent"], reverse=True),
    )


def _server_resource_views(
    server_resource_state: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for server_id, state in server_resource_state.items():
        projection = _server_resource_projection(state)
        utilizations = (
            projection["cpu_utilization_percent"],
            projection["ram_utilization_percent"],
            projection["session_utilization_percent"],
        )
        safe_session_limit = math.floor(
            float(state["inferred_session_capacity"])
            * MAX_REMAP_UTILIZATION_PERCENT
            / 100.0
            + 1e-9
        )
        views.append({
            "server_id": server_id,
            "baseline_cpu_utilization_percent": round(
                float(state["baseline_cpu_util_percent"]), 6
            ),
            "projected_cpu_utilization_percent": round(
                projection["cpu_utilization_percent"], 6
            ),
            "baseline_ram_utilization_percent": round(
                float(state["baseline_ram_util_percent"]), 6
            ),
            "projected_ram_utilization_percent": round(
                projection["ram_utilization_percent"], 6
            ),
            "baseline_active_sessions": int(round(float(state["baseline_active_sessions"]))),
            "projected_active_sessions": int(round(projection["active_sessions"])),
            "inferred_session_capacity": int(round(float(state["inferred_session_capacity"]))),
            "safe_session_limit_at_80_percent": safe_session_limit,
            "projected_session_utilization_percent": round(
                projection["session_utilization_percent"], 6
            ),
            "transferred_flow_count": int(round(float(state["transferred_flow_count"]))),
            "transferred_cpu_busy_cores": round(
                float(state["additional_cpu_busy_cores"]), 6
            ),
            "transferred_ram_gb": round(float(state["additional_ram_gb"]), 6),
            "maximum_projected_resource_utilization_percent": round(max(utilizations), 6),
            "minimum_headroom_to_safe_limit_percent": round(
                MAX_REMAP_UTILIZATION_PERCENT - max(utilizations), 6
            ),
        })
    return sorted(
        views,
        key=lambda item: item["maximum_projected_resource_utilization_percent"],
        reverse=True,
    )


def _build_summary(
    model: NetworkModel,
    flows: list[dict[str, Any]],
    *,
    step_index: int,
    plan_active: bool,
    valid_from: int | None,
    valid_until: int | None,
    actions: list[dict[str, Any]],
    capacity_rejections: int,
    network_capacity_rejections: int,
    server_resource_rejections: int,
    server_resource_rejection_reasons: Mapping[str, int],
    projected_edge_loads: Mapping[tuple[str, str], float],
    projected_node_loads: Mapping[str, float],
    projected_server_resources: Mapping[str, Mapping[str, float]],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    legitimate = [flow for flow in flows if not flow.get("is_attack_traffic")]
    tier_views: list[dict[str, Any]] = []
    for priority, grade in enumerate(("gold", "silver", "bronze"), start=1):
        tier = [flow for flow in legitimate if str(flow.get("sla_grade", "")).lower() == grade]
        acted = [flow for flow in tier if flow.get("routing_action") != "keep_baseline_route"]
        available = [flow for flow in tier if flow.get("service_available")]
        tier_views.append({
            "sla_grade": grade,
            "priority": priority,
            "flow_count": len(tier),
            "affected_flow_count": len(acted),
            "rerouted_flow_count": sum(bool(flow.get("rerouted")) for flow in tier),
            "failover_flow_count": sum(bool(flow.get("failover_active")) for flow in tier),
            "isolated_flow_count": sum(bool(flow.get("isolated")) for flow in tier),
            "protected_endpoint_flow_count": sum(bool(flow.get("endpoint_protected")) for flow in tier),
            "available_flow_count": len(available),
            "availability_ratio": round(len(available) / max(len(tier), 1), 6),
            "restoration_ratio": (
                1.0
                if not acted
                else round(sum(bool(flow.get("service_available")) for flow in acted) / len(acted), 6)
            ),
        })

    baseline_gold_routes = [
        list(flow.get("original_route") or [])
        for flow in legitimate
        if str(flow.get("sla_grade", "")).lower() == "gold"
    ]
    active_gold_routes = [
        list(flow.get("route") or [])
        for flow in legitimate
        if str(flow.get("sla_grade", "")).lower() == "gold" and flow.get("route_available")
    ]
    baseline_nodes, baseline_edges = _route_sets(baseline_gold_routes)
    active_nodes, active_edges = _route_sets(active_gold_routes)
    edge_utilization, node_utilization = _utilization_views(model, projected_edge_loads, projected_node_loads)
    maximum_edge_utilization = max(
        (item["utilization_percent"] for item in edge_utilization),
        default=0.0,
    )
    maximum_node_utilization = max(
        (item["utilization_percent"] for item in node_utilization),
        default=0.0,
    )
    server_resources = _server_resource_views(projected_server_resources)
    maximum_server_cpu_utilization = max(
        (item["projected_cpu_utilization_percent"] for item in server_resources),
        default=0.0,
    )
    maximum_server_ram_utilization = max(
        (item["projected_ram_utilization_percent"] for item in server_resources),
        default=0.0,
    )
    maximum_server_session_utilization = max(
        (item["projected_session_utilization_percent"] for item in server_resources),
        default=0.0,
    )
    maximum_server_resource_utilization = max(
        maximum_server_cpu_utilization,
        maximum_server_ram_utilization,
        maximum_server_session_utilization,
    )
    action_counts = Counter(str(action.get("action", "unknown")) for action in actions)

    return {
        "step_index": step_index,
        "plan_active": plan_active,
        "valid_from_step": valid_from,
        "valid_until_step": valid_until,
        "policy": "gold_then_silver_then_bronze",
        "sdn_control": context.get("sdn_intent", {}),
        "protection_stage": str(context.get("protection_stage", "inactive")),
        "source_quarantine_policy": str(
            context.get("source_quarantine_policy", "explicit_plan_only")
        ),
        "maximum_safe_utilization_percent": MAX_REMAP_UTILIZATION_PERCENT,
        "target_ids": sorted(context.get("targets", set())),
        "avoid_nodes": sorted(context.get("avoid_nodes", set())),
        "service_transparent_power_targets": sorted(
            context.get("service_transparent_power_targets", set())
        ),
        "observed_source_ids": sorted(context.get("observed_source_ids", set())),
        "quarantine_sources": sorted(context.get("quarantine_sources", set())),
        "rerouted_flow_count": sum(bool(flow.get("rerouted")) for flow in legitimate),
        "failover_flow_count": sum(bool(flow.get("failover_active")) for flow in legitimate),
        "isolated_flow_count": sum(bool(flow.get("isolated")) for flow in legitimate),
        "protected_endpoint_flow_count": sum(bool(flow.get("endpoint_protected")) for flow in legitimate),
        "capacity_rejected_candidate_count": capacity_rejections,
        "network_capacity_rejected_candidate_count": network_capacity_rejections,
        "server_resource_rejected_candidate_count": server_resource_rejections,
        "server_resource_rejection_reasons": dict(
            sorted(server_resource_rejection_reasons.items())
        ),
        "maximum_projected_utilization_percent": round(
            max(
                maximum_edge_utilization,
                maximum_node_utilization,
                maximum_server_resource_utilization,
            ),
            6,
        ),
        "maximum_projected_edge_utilization_percent": round(maximum_edge_utilization, 6),
        "maximum_projected_node_utilization_percent": round(maximum_node_utilization, 6),
        "maximum_projected_utilization_semantics": (
            "maximum_of_reserved_network_and_projected_server_cpu_ram_session_load"
        ),
        "maximum_projected_server_resource_utilization_percent": round(
            maximum_server_resource_utilization, 6
        ),
        "maximum_projected_server_cpu_utilization_percent": round(
            maximum_server_cpu_utilization, 6
        ),
        "maximum_projected_server_ram_utilization_percent": round(
            maximum_server_ram_utilization, 6
        ),
        "maximum_projected_server_session_utilization_percent": round(
            maximum_server_session_utilization, 6
        ),
        "server_resource_admission_semantics": {
            "safe_limit_percent": MAX_REMAP_UTILIZATION_PERCENT,
            "cpu": (
                "baseline runtime busy cores plus transferred flow share, "
                "normalized by target physical cores"
            ),
            "ram": (
                "baseline runtime used GiB plus transferred flow share, "
                "normalized by target installed RAM"
            ),
            "sessions": (
                "baseline active sessions plus one per transferred flow; capacity is "
                "inferred from baseline sessions and the maximum baseline CPU/RAM/network utilization"
            ),
            "flow_share": (
                "50_percent_fixed_session_cost_plus_50_percent_wire_rate_weight, "
                "normalized_to_source_server_runtime"
            ),
            "capacity_origin": (
                "cpu_and_ram_from_server_profile; utilization_and_sessions_from_runtime; "
                "session_capacity_is_a_gnet9_inference_not_a_vendor_limit"
            ),
            "mutation": "local_overlay_only; baseline_t0_graph_is_not_modified",
        },
        "tiers": tier_views,
        "action_counts": dict(sorted(action_counts.items())),
        "actions": actions,
        "top_projected_edges": edge_utilization[:12],
        "top_projected_nodes": node_utilization[:12],
        "projected_server_resources": server_resources,
        "baseline_gold_route_node_set": sorted(baseline_nodes),
        "active_gold_route_node_set": sorted(active_nodes),
        "baseline_gold_route_edge_set": sorted(_edge_label(edge) for edge in baseline_edges),
        "active_gold_route_edge_set": sorted(_edge_label(edge) for edge in active_edges),
        "gold_route_sets_changed": baseline_nodes != active_nodes or baseline_edges != active_edges,
        "calculation_note_ru": (
            "Сетевая нагрузка оценена по wire_bytes/interval_seconds; для failover также "
            "проверены прогнозные CPU, RAM и сессии сервера. Ресурсы выделялись "
            "последовательно по SLA без изменения исходного NetworkX-графа."
        ),
    }


def _route_sets(routes: Iterable[list[str]]) -> tuple[set[str], set[tuple[str, str]]]:
    nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()
    for route in routes:
        nodes.update(route)
        edges.update(_edge_key(source, target) for source, target in zip(route, route[1:]))
    return nodes, edges


def _edge_label(edge: tuple[str, str]) -> str:
    return f"{edge[0]}--{edge[1]}"
