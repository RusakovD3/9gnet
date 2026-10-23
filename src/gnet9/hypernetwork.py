"""Зависимости уровней: одно отношение может связывать несколько объектов."""

from __future__ import annotations


# Стрелка означает направление вложения, а не обратное влияние всех уровней.
EMBEDDING_LEVEL_PAIRS = (
    ("L0", "L1"), ("L1", "L2"),
    ("L2", "L3"), ("L2", "L4"), ("L2", "L6"), ("L2", "L8"),
    ("L3", "L4"), ("L3", "L8"), ("L4", "L8"),
    ("L5", "L2"), ("L5", "L3"), ("L6", "L4"),
)
CONTROLLED_LEVELS = ("L1", "L2", "L3", "L5", "L6")


def hypernetwork_embedding(model) -> dict:
    """Вершина вложения — пара (объект, уровень), связь — общее отношение.

    Не создаёт копию физической сети: хранит только ссылки на её объекты.
    Например, услуга связана одновременно с сервером, абонентом и доступом.
    ``dependencies`` задают ориентированные связи уровней внутри отношения.
    Совместное присутствие уровней само по себе не разрешает управление.
    """
    relations = []

    def add(kind, members, dependencies=None):
        unique = sorted(set(members))
        if len(unique) > 1:
            levels = {level for _, level in unique}
            pairs = EMBEDDING_LEVEL_PAIRS if dependencies is None else dependencies
            relations.append({
                "kind": kind,
                "members": [list(member) for member in unique],
                "dependencies": [
                    {"source_level": source, "target_level": target}
                    for source, target in pairs if source in levels and target in levels
                ],
            })

    entity_levels = {}
    coordinates = set()
    for node, attrs in sorted(model.graph.nodes(data=True)):
        levels = {value.level for value in attrs.values() if hasattr(value, "metric_names")}
        # Физический сервер имеет основной уровень L0, но отдельный тензор
        # L0 у него отсутствует: его измеряемые ресурсы хранятся в L6.
        if attrs.get("level"):
            levels.add(attrs["level"])
        entity_levels[node] = levels
        coordinates.update((node, level) for level in levels)
        add("shared_object", [(node, level) for level in levels])
    for source, target, attrs in sorted(model.graph.edges(data=True)):
        edge_id = "--".join(sorted((source, target)))
        members = [(edge_id, value.level) for value in attrs.values() if hasattr(value, "metric_names")]
        coordinates.update(members)
        members += [(node, level) for node in (source, target) for level in entity_levels[node]]
        add("line_and_endpoints", members)
    service_by_codec = {service.codec_profile_id: service for service in model.services}
    for node, attrs in sorted(model.graph.nodes(data=True)):
        if attrs.get("level") != "L1":
            continue
        service = service_by_codec.get(attrs.get("codec_profile_id"))
        if service is None:
            continue
        access = attrs.get("home_access")
        members = [(node, "L1"), (service.service_id, "L0"), (service.server_id, "L0")]
        members += [(access, level) for level in entity_levels.get(access, ())]
        add("service_dependency", members)
    arbitrators = sorted(member for member in coordinates if member[1] == "L7")
    if arbitrators:
        services = [(service.service_id, "L0") for service in model.services
                    if (service.service_id, "L0") in coordinates]
        add("sla_observation", services + arbitrators, (("L0", "L7"),))
        for level in CONTROLLED_LEVELS:
            targets = sorted(member for member in coordinates if member[1] == level)
            add("control", arbitrators + targets, (("L7", level),))
    return {"representation": "named_incidence_hypergraph", "relations": relations,
            "relation_count": len(relations),
            "member_semantics": ["existing_entity_id", "level"],
            "dependency_semantics": "source_level_to_target_level",
            "embedding_level_pairs": [list(pair) for pair in EMBEDDING_LEVEL_PAIRS],
            "sla_observation_level_pairs": [["L0", "L7"]],
            "control_level_pairs": [["L7", level] for level in CONTROLLED_LEVELS]}
