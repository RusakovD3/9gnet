"""Размещение объектов на картах без изменения координат модели."""

def network_positions(model) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    missing: list[str] = []
    for node_id, attrs in model.graph.nodes(data=True):
        if "pos" in attrs:
            x, y = attrs["pos"]
            positions[node_id] = (float(x) * 1.55, float(y) * 1.04)
        else:
            missing.append(node_id)
    for index, node_id in enumerate(missing):
        positions[node_id] = (-10.5 + index * 0.4, 8.0)

    # Координаты меняются только для рисунка; исходный граф остаётся прежним.
    access_groups: dict[str, list[str]] = {}
    for node_id, attrs in model.graph.nodes(data=True):
        if attrs.get("level") == "L1":
            access_groups.setdefault(attrs["home_access"], []).append(node_id)
    for access_node, node_ids in access_groups.items():
        center_x, access_y = positions[access_node]
        for index, node_id in enumerate(sorted(node_ids)):
            row, column = divmod(index, 4)
            positions[node_id] = (
                center_x + (column - 1.5) * 0.28,
                access_y - 1.28 - row * 0.42,
            )
    return positions
