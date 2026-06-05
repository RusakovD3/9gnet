# G-Net 9

G-Net 9 - исследовательская модель телекоммуникационной сети. Проект строит
воспроизводимое здоровое baseline-состояние `t0`, а затем может "оживлять" сеть
дискретными шагами по 5 секунд: сохранять состояния тензоров, имитировать
TCP/IP-трафик внутри Python и передавать агрегированное состояние L7-арбитратору.

Модель не отправляет реальные пакеты в операционную систему. Packet simulation
строит детерминированные Ethernet/IPv4/TCP/UDP события в памяти.

## Быстрый Старт

```powershell
pip install -r requirements.txt
python main.py
```

Сгенерировать 12 шагов динамики после `t0`:

```powershell
python main.py --dynamics-steps 12
```

Сгенерировать компактную динамику для длинного анализа:

```powershell
python main.py --dynamics-steps 100 --snapshot-detail summary --packet-detail summary
```

## Основная Идея

Проект разделен на несколько простых слоев ответственности:

- `topology_builder.py` строит форму сети: узлы, связи, уровни и роли.
- `baseline.py` явно задает здоровые начальные значения тензоров `t0`.
- `tensors.py` описывает имена метрик, единицы измерения и порядок числовых векторов.
- `dynamics.py` двигает время шагами по 5 секунд и сохраняет snapshots.
- `packet_simulator.py` имитирует TCP/IP flow и representative packets.
- `arbitrator.py` читает тензоры всех уровней, строит агрегаты и готовит решение по remapping.

Так проще объяснять проект: топология отдельно, идеальное состояние отдельно,
динамика отдельно, анализ сети отдельно.

## Текущая Топология

Baseline содержит 270 узлов и 274 ребра:

- `L0`: 4 сервиса: Voice, Video, FTP, Telemetry.
- `L1`: 240 абонентов: 120 mobile и 120 fixed.
- `L2`: 18 активных устройств: 12 core routers и 6 aggregation routers.
- `L7`: 1 arbitrator node.
- `L8`: 7 terrain anchors.

`L3`, `L4`, `L5`, `L6` не являются отдельными узлами графа:

- `L3` и `L4` хранятся на ребрах как `l3_tensor` и `l4_tensor`.
- `L5` и `L6` хранятся на L2-оборудовании.
- `L6` также хранится на L1-абонентах.
- `L8` хранится на размещенных non-L0 узлах как `l8_tensor`.

Граф построен как `networkx.Graph`, поэтому связи неориентированные: ребро между
двумя узлами читается в обе стороны и не дублируется.

## Тензоры

Каждый `StateTensor` - это числовой вектор с явными `metric_names`, `units`,
`metric_index` и `vector`. Это сделано намеренно: для Koopman/DMD, Lyapunov,
remapping и сравнения состояний удобнее иметь компактные числовые признаки,
чем большой разреженный многомерный массив.

Где задаются значения:

- `baseline.py` - человекочитаемые значения здорового `t0`.
- `tensors.py` - порядок метрик и единицы измерения.
- `topology_builder.py` - прикрепление тензоров к узлам и ребрам.
- `dynamics.py` - сохранение тензоров на каждом шаге.

## Динамика

Текущий режим динамики называется `stationary_healthy_baseline`.

Это означает:

- сеть живет во времени;
- шаг по умолчанию равен 5 секундам;
- число шагов задается через `DynamicsConfig.step_count` или `--dynamics-steps`;
- топология не меняется;
- значения тензоров не деградируют;
- SLA/SLO не нарушаются;
- пакеты имитируются, но реальные сетевые сокеты не используются.

По умолчанию экспортируются snapshots:

```text
t = 0, 5, 10, 15, 20, 25, 30
```

Если задать `--dynamics-steps 12`, будут шаги от `t=0` до `t=60`.

## Режимы Экспорта Динамики

`--snapshot-detail` управляет размером snapshot:

- `full` - полный режим по умолчанию: узлы, ребра, все тензоры, арбитратор, traffic.
- `tensor` - без списков узлов и ребер, но со всеми тензорами в `tensor_state.by_level`.
- `summary` - компактный режим: counts, state vector, арбитратор и traffic summary.

`--packet-detail` управляет детализацией имитации трафика:

- `summary` - только aggregate-счетчики: число flow, packet count, latency, drops.
- `flows` - summary плюс 240 flow records, по одному на L1-абонента.
- `sample` - summary, flows и representative packet headers.

Для длинных прогонов лучше использовать:

```powershell
python main.py --dynamics-steps 100 --snapshot-detail summary --packet-detail summary
```

Для демонстрации TCP/IP headers лучше использовать default `full/sample`.

## Формат Snapshot

В полном режиме один snapshot содержит:

```json
{
  "step_index": 0,
  "time_seconds": 0,
  "level_summary": {},
  "nodes": [],
  "edges": [],
  "tensor_state": {},
  "state_vector": {},
  "arbitrator": {},
  "traffic": {}
}
```

`tensor_state` собирает все тензоры по уровням:

```json
{
  "levels": ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "EDGE"],
  "counts": {"L1": 240, "EDGE": 274},
  "by_level": {
    "L1": [{"node_id": "M1_01", "metrics": {}}],
    "EDGE": [{"source": "A1", "target": "M1_01", "metrics": {}}]
  }
}
```

`state_vector` - компактный числовой вектор для будущего Koopman/DMD и Lyapunov:

```json
{
  "metric_names": ["L1.sla_margin.min", "EDGE.stability_margin.min"],
  "vector": [0.68, 0.45]
}
```

## Арбитратор

`arbitrator.py` получает `tensor_state` каждого шага и строит:

- агрегаты по уровням: min/max/mean для выбранных метрик;
- компактный `state_vector`;
- baseline-оценки `lyapunov_value`, `koopman_residual`, `remap_pressure`;
- решение `remap`.

В здоровом baseline арбитратор должен выдавать:

```json
{
  "remap": {
    "needed": false,
    "action": "NO_REMAP"
  }
}
```

Позже, когда появятся атаки, перегрузки или отказы, тот же арбитратор сможет
использовать измененные тензоры и начать выдавать `PLAN_REMAP`.

## Packet Simulation

`packet_simulator.py` моделирует:

- deterministic private IPv4 и MAC для узлов;
- Ethernet MTU 1500;
- IPv4 header;
- TCP handshake на `t0`;
- TCP FTP data flow;
- UDP DNS query/response;
- UDP RTP-like media stream для `broadcast_mp3`;
- shortest path по latency;
- per-hop MAC rewrite;
- TTL decrement;
- serialization delay;
- expected path loss.

В healthy baseline:

- `observed_dropped_packets = 0`;
- `observed_retransmissions = 0`;
- `observed_loss_ratio = 0.0`.

## Основные Артефакты

`python main.py` пишет в `output/`:

- `baseline_topology.json` - полный граф с тензорами.
- `baseline_topology.graphml` - GraphML для внешних инструментов.
- `baseline_summary.txt` - краткая сводка уровней.
- `network_dynamics.json` - snapshots динамики.
- `l1_monitoring.csv` - синтетический L1-мониторинг.
- `l1_d0sl_profiles.json` - профили L1-абонентов.
- `l1_d0sl_parsed.json` - использованные d0sl-политики.
- `l2_equipment_profiles.json` - L2-профили и raw telemetry.
- `network_logic.png` и `layer_scheme.png` - визуализации.

## Что Уже Реализовано

- Воспроизводимая baseline-топология.
- Явные здоровые значения тензоров в `baseline.py`.
- d0sl-политики SLA/SLO для L1.
- Синтетический L1-мониторинг без baseline-нарушений.
- L2 Cisco-like capacity profiles.
- Динамика шагами по 5 секунд.
- Сохранение тензоров всех уровней на каждом шаге.
- Компактный state vector для будущего анализа.
- L7-арбитратор, который читает тензоры и пока принимает `NO_REMAP`.
- In-memory TCP/IP packet simulation.
- Режимы компактного и полного экспорта.

## Что Пока Не Реализовано

- Реальные сценарии атак и отказов.
- Изменение тензоров во времени при деградации.
- Реальный remapping маршрутов и slices.
- Koopman/DMD pipeline по временным рядам.
- Lyapunov-анализ восстановления.
- SDN orchestration и auto-healing.

## Тесты

```powershell
pytest tests/test_topology.py -v
```

Тесты проверяют:

- количество L0/L1/L2 узлов;
- наличие тензоров всех уровней;
- здоровый baseline;
- snapshots по 5 секунд;
- режимы детализации dynamics;
- работу арбитратора;
- in-memory TCP/IP headers.
