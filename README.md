# G-Net 9

G-Net 9 - исследовательская модель телекоммуникационной сети. Проект строит
воспроизводимое baseline-состояние `t0`: топологию, абонентов, сервисы,
телеметрию, d0sl-политики и числовые тензоры состояния для дальнейших
экспериментов с отказами, атаками, ремаппингом, Koopman/DMD, Hausdorff и
Lyapunov-анализом.

## Быстрый старт

```bash
pip install -r requirements.txt
python main.py
pytest tests/test_topology.py -v
```

`python main.py` создает артефакты в `output/`.

## Что генерируется

- `baseline_topology.json` - полный граф NetworkX с атрибутами и тензорами.
- `baseline_topology.graphml` - GraphML для внешних инструментов.
- `baseline_summary.txt` - краткая сводка уровней, сервисов и slices.
- `network_logic.png` - визуализация логической сети.
- `layer_scheme.png` - схема уровней G-Net.
- `l1_monitoring.csv` - 240 абонентов x 30 секунд мониторинга.
- `l1_d0sl_profiles.json` - профили абонентов и примененные политики.
- `l1_d0sl_parsed.json` - уникальные d0sl-политики, реально использованные в baseline.
- `l2_equipment_profiles.json` - профили L2-оборудования и raw telemetry.
- `l1_policies.d0sl` - копия исходного файла политик.

## Текущая топология

Модель содержит 270 узлов и 274 ребра:

- `L0`: 4 сервиса: Voice, Video, FTP, Telemetry.
- `L1`: 240 абонентов: 120 mobile и 120 fixed.
- `L2`: 18 active equipment nodes: 12 core routers и 6 aggregation routers.
- `L7`: 1 arbitrator node.
- `L8`: 7 terrain anchors.

Важная модельная договоренность: `L3`, `L4`, `L5`, `L6` сейчас не являются
отдельными узлами графа. Они представлены тензорами на существующих объектах:

- `L3` и `L4` лежат на ребрах как `l3_tensor` и `l4_tensor`.
- `L5` и `L6` лежат на L2-оборудовании.
- `L6` также добавлен абонентам.
- `L8` добавлен всем размещенным non-L0 узлам как `l8_tensor`.

## Тензоры состояния

Раньше проект использовал фиксированный 5D-тензор `2x2x2x2x2`. Сейчас это
заменено на более простой и расчетно удобный `StateTensor`: числовой вектор
с явными `metric_names`, `units`, `metric_index` и `shape`.

Такой формат проще использовать как вектор состояния для:

- Koopman/DMD;
- Lyapunov-функций;
- сравнения baseline/degraded/repaired состояний;
- Hausdorff distance по координатам L8;
- cost-aware remapping и CAPEX/OPEX оценок.

### L0 - сервисы

Метрики:

- `service_code`
- `bitrate_mbps`
- `latency_budget_ms`
- `jitter_budget_ms`
- `availability_target`
- `priority_code`
- `demand_pressure`
- `service_health`

### L1 - абоненты

Метрики:

- `access_type_code` - fixed/mobile.
- `service_code` - связь с сервисом L0.
- `request_rate_pps`
- `response_rate_pps`
- `traffic_intensity_rho`
- `traffic_distribution_cv`
- `processing_speed_mbps`
- `processing_delay_ms`
- `capex_opex_cost`
- `sla_margin`

### L2 - активное оборудование

Метрики:

- `ram_used_gb`
- `ram_load_percent`
- `cpu_load_percent`
- `packet_processing_time_ms`
- `traffic_distribution_code`
- `port_delay_ms`
- `port_speed_mbps`
- `capex_opex_cost`
- `stability_margin`

### L3 - среда передачи

Метрики на ребрах:

- `medium_code`
- `line_rate_mbps`
- `distance_m`
- `frequency_mhz`
- `attenuation_db`
- `noise_interference_db`
- `snr_db`

### L4 - кабельная канализация / физический путь

Метрики на ребрах:

- `x_mid`
- `y_mid`
- `length_m`
- `cross_connect_present`
- `duct_capacity_used_ratio`
- `repair_time_hours`

### L5 - протоколы, маршрутизация, ремаппинг

Метрики:

- `protocol_code`
- `socket_binding_present`
- `routing_mode_code`
- `remap_algorithm_code`
- `percolation_threshold`
- `reconfiguration_time_s`

### L6 - питание и стоимость

Метрики:

- `power_supply_code`
- `nominal_power_kw`
- `backup_autonomy_hours`
- `energy_reserve_ratio`
- `capex_opex_cost`

### L7 - арбитр

Метрики:

- `hausdorff_distance`
- `lyapunov_value`
- `lyapunov_delta`
- `koopman_residual`
- `remap_pressure`
- `decision_confidence`
- `action_cost`

### L8 - размещение

Метрики:

- `x`
- `y`
- `coordinate_norm`
- `placement_role_code`
- `terrain_risk`

## Основные файлы

- `main.py` - входная точка и экспорт артефактов.
- `src/gnet9/topology_builder.py` - построение baseline-графа.
- `src/gnet9/tensors.py` - спецификации `StateTensor` для уровней L0-L8 и EDGE.
- `src/gnet9/models.py` - `StateTensor`, `NetworkModel`, профили сервисов и slices.
- `src/gnet9/l1_d0sl.py` - парсер d0sl, очереди L1, мониторинг L1.
- `src/gnet9/l2_equipment.py` - L2-профили оборудования и расчет L2 state metrics.
- `src/gnet9/metrics.py` - Hausdorff distance и centrality.
- `src/gnet9/visualizer.py` - PNG-визуализации.
- `policies/l1_policies.d0sl` - политики SLA/SLO для L1.
- `tests/test_topology.py` - базовые структурные проверки.

## Что уже реализовано

- Воспроизводимая baseline-топология.
- L0-сервисы, L1-абоненты, L2 core/aggregation routers.
- d0sl-политики для SLA/SLO/SLI.
- Синтетический L1-мониторинг.
- L2 raw telemetry и Cisco-like capacity profiles.
- Числовые state tensors для всех уровней модели.
- JSON, GraphML, CSV и PNG-экспорт.
- Базовые тесты структуры и тензоров.

## Что еще не реализовано

- Реальные сценарии атак и отказов.
- Полный decision engine для L7.
- Автоматический ремаппинг маршрутов и slices.
- Koopman/DMD pipeline по временным рядам.
- Lyapunov-анализ деградации и восстановления.
- SDN orchestration и auto-healing.

Текущая цель проекта - иметь чистый baseline `t0`, от которого можно строить
сценарии деградации, восстановления и прогнозирования.
