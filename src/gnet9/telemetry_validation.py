"""Офлайн-валидация порога предупреждений по внешней телеметрии.

На входе готовый обезличенный ряд риска и подтверждённые начала инцидентов.
Порог выбирается на ранних данных и проверяется на позднем участке того же ряда.
Источник численных оценок устанавливает поставщик CSV.
"""

from __future__ import annotations

from bisect import bisect_left
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable


REPORT_VERSION = 3
REQUIRED_COLUMNS = ("timestamp_seconds", "risk_score", "attack_onset")


class TelemetryValidationError(ValueError):
    """Внешняя телеметрия не соответствует контракту безопасной валидации."""


@dataclass(frozen=True)
class TelemetryPoint:
    """Один обезличенный отсчёт уже рассчитанного риска."""

    timestamp_seconds: float
    risk_score: float
    attack_onset: bool
    attack_id: str
    attack_kind: str


@dataclass(frozen=True)
class TelemetryValidationConfig:
    """Критерии допуска порога к пилотному применению.

    Значения не являются универсальными нормативами: их должен утвердить
    владелец сервиса с учётом допустимой цены пропуска и ложной тревоги.
    """

    training_fraction: float = 0.70
    minimum_lead_seconds: float = 10.0
    maximum_lead_seconds: float = 20.0
    minimum_events_per_partition: int = 5
    minimum_precision: float = 0.90
    minimum_recall: float = 0.90
    maximum_false_warnings_per_hour: float = 0.10
    # Бюджет внимания в пилоте: до 12 минут под тревогой за час наблюдения.
    maximum_warning_occupancy_ratio: float = 0.20

    def validate(self) -> None:
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise TelemetryValidationError("Параметры проверки должны быть конечными числами")
        if not 0.5 <= self.training_fraction < 1.0:
            raise TelemetryValidationError("training_fraction должен быть в диапазоне [0.5, 1.0)")
        if self.minimum_lead_seconds <= 0:
            raise TelemetryValidationError("minimum_lead_seconds должен быть положительным")
        if self.maximum_lead_seconds < self.minimum_lead_seconds:
            raise TelemetryValidationError("maximum_lead_seconds не может быть меньше minimum_lead_seconds")
        if self.minimum_events_per_partition < 1:
            raise TelemetryValidationError("minimum_events_per_partition должен быть не меньше 1")
        if not 0.0 < self.minimum_precision <= 1.0:
            raise TelemetryValidationError("minimum_precision должен быть в диапазоне (0, 1]")
        if not 0.0 < self.minimum_recall <= 1.0:
            raise TelemetryValidationError("minimum_recall должен быть в диапазоне (0, 1]")
        if self.maximum_false_warnings_per_hour < 0:
            raise TelemetryValidationError("maximum_false_warnings_per_hour не может быть отрицательным")
        if not 0.0 <= self.maximum_warning_occupancy_ratio <= 1.0:
            raise TelemetryValidationError("maximum_warning_occupancy_ratio должен быть в диапазоне [0, 1]")


def read_telemetry_csv(path: Path) -> list[TelemetryPoint]:
    """Прочитать строгий и минимальный CSV-контракт внешней телеметрии."""

    if not path.is_file():
        raise TelemetryValidationError(f"Файл телеметрии не найден: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise TelemetryValidationError("CSV не содержит строку заголовков")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise TelemetryValidationError("CSV содержит повторяющиеся имена столбцов")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise TelemetryValidationError(
                "В CSV отсутствуют обязательные поля: " + ", ".join(missing)
            )
        points: list[TelemetryPoint] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise TelemetryValidationError(f"Строка {row_number}: больше столбцов, чем в заголовке")
            try:
                timestamp = float(row["timestamp_seconds"])
                risk_score = float(row["risk_score"])
            except (TypeError, ValueError) as error:
                raise TelemetryValidationError(
                    f"Строка {row_number}: timestamp_seconds и risk_score должны быть числами"
                ) from error
            onset = _parse_bool(row["attack_onset"], row_number)
            attack_id = str(row.get("attack_id") or "").strip()
            attack_kind = str(row.get("attack_kind") or "").strip()
            if onset and not attack_id:
                attack_id = f"incident_at_{timestamp:g}"
            points.append(
                TelemetryPoint(
                    timestamp_seconds=timestamp,
                    risk_score=risk_score,
                    attack_onset=onset,
                    attack_id=attack_id,
                    attack_kind=attack_kind,
                )
            )
    _validate_points(points)
    return points


def _validate_points(points: list[TelemetryPoint]) -> None:
    if len(points) < 3:
        raise TelemetryValidationError("Нужно не менее трёх отсчётов телеметрии")
    previous_time = -math.inf
    for row_number, point in enumerate(points, start=2):
        if not math.isfinite(point.timestamp_seconds) or not math.isfinite(point.risk_score):
            raise TelemetryValidationError(f"Строка {row_number}: значения должны быть конечными")
        if not 0.0 <= point.risk_score <= 1.0:
            raise TelemetryValidationError(f"Строка {row_number}: risk_score должен находиться в диапазоне [0, 1]")
        if point.timestamp_seconds <= previous_time:
            raise TelemetryValidationError(
                f"Строка {row_number}: timestamp_seconds должен строго возрастать"
            )
        previous_time = point.timestamp_seconds


def validate_external_telemetry(
    points: list[TelemetryPoint],
    config: TelemetryValidationConfig | None = None,
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Калибровать порог на раннем отрезке и проверить его на holdout.

    Порог выбирается только по обучающей части. Последующая часть временного
    ряда не участвует в выборе и поэтому является независимой постпроверкой
    внутри предоставленного набора данных.
    """

    config = config or TelemetryValidationConfig()
    config.validate()
    _validate_points(points)
    split_index = int(len(points) * config.training_fraction)
    split_index = min(max(split_index, 1), len(points) - 1)
    training_points = points[:split_index]
    holdout_points = points[split_index:]
    candidates = _threshold_candidates(training_points)
    training_gap = _maximum_sample_gap(training_points)
    best_rank = None
    for evaluated_count, threshold in enumerate(candidates, 1):
        item = _evaluate_threshold(training_points, threshold, config, training_gap)
        rate = item["false_warnings_per_hour"]
        rank = (_meets_quality_gate(item, config), item["recall"], item["precision"],
                -rate if rate is not None else -math.inf, threshold)
        if best_rank is None or rank > best_rank:
            selected, best_rank = item, rank
        # Идём от большего порога. Лучший возможный результат уже нельзя превзойти.
        if rank[:4] == (True, 1.0, 1.0, 0.0):
            break
        # При снижении порога длительность под тревогой может только расти.
        if best_rank[0] and item["warning_occupancy_ratio"] > config.maximum_warning_occupancy_ratio:
            break
    threshold = float(selected["warning_risk_threshold"])
    previous = training_points[-1]
    first = holdout_points[0]
    continuing_warning = (first.timestamp_seconds - previous.timestamp_seconds <= training_gap
                          and min(previous.risk_score, first.risk_score) >= threshold)
    holdout = _evaluate_threshold(holdout_points, threshold, config, training_gap,
                                  continuing_warning=continuing_warning)
    training_gate = _meets_quality_gate(selected, config)
    holdout_gate = _meets_quality_gate(holdout, config)
    eligible = bool(training_gate and holdout_gate)
    source_digest = _source_digest(source_path) if source_path else None

    return {
        "report_version": REPORT_VERSION,
        "report_kind": "external_telemetry_chronological_holdout_validation",
        "validation_scope": "threshold_only_on_supplied_risk_scores",
        "data_contract": {
            "required_columns": list(REQUIRED_COLUMNS),
            "optional_columns": ["attack_id", "attack_kind"],
            "risk_score_semantics": "supplied scalar score, model provenance unverified",
            "raw_payloads_or_personal_data_expected": False,
            "source_sha256": source_digest,
        },
        "validation_protocol_ru": (
            "Хронологическое разделение без перемешивания: порог выбирается только на раннем "
            "участке, затем фиксируется и проверяется на последующем отложенном участке. "
            "Проверяется только порог готового ряда риска. Цель и вид угрозы в прогнозе "
            "CSV отсутствуют. Для допуска учитывается время первого сигнала эпизода. "
            "Совпадения с последующими отсчётами показаны отдельно. "
            "Лимит доли времени под тревогой действует в обеих частях. "
            "Разрыв более 1,5 медианного интервала обучающей части завершает участок."
        ),
        "configuration": asdict(config),
        "partition": {
            "training_row_count": len(training_points),
            "holdout_row_count": len(holdout_points),
            "training_time_range_seconds": _time_range(training_points),
            "holdout_time_range_seconds": _time_range(holdout_points),
            "split_timestamp_seconds": holdout_points[0].timestamp_seconds,
            "maximum_contiguous_sample_gap_seconds": training_gap,
        },
        "threshold_selection": {
            "candidate_count": len(candidates),
            "evaluated_candidate_count": evaluated_count,
            "selection_uses_holdout": False,
            "selection_status": "feasible" if training_gate else "no_training_threshold_meets_gate",
            "selected_warning_risk_threshold": threshold,
            "warning_threshold_origin": "chronological_external_telemetry_calibration",
            "training_metrics": selected,
        },
        "independent_holdout": {
            "performed": True,
            "metrics": holdout,
            "quality_gate_passed": holdout_gate,
        },
        "deployment_profile": {
            "warning_risk_threshold": threshold,
            "warning_threshold_origin": "chronological_external_telemetry_calibration",
            "eligible_for_pilot": eligible,
            "not_a_probability_calibration": True,
            "automatic_network_change_authorized": False,
        },
        "deployment_gate": {
            "status": "PASSED_FOR_SUPERVISED_PILOT" if eligible else "FAILED",
            "passed": eligible,
            "semantics_ru": (
                "Допуск относится к порогу предоставленного ряда риска. "
                "Происхождение ряда, прогноз показателей услуг и перенос на другие сети требуют отдельной проверки."
            ),
            "training_quality_gate_passed": training_gate,
            "holdout_quality_gate_passed": holdout_gate,
        },
    }


def write_validation_report(report: dict[str, Any], path: Path) -> None:
    """Записать воспроизводимый отчёт и профиль порога без сырых строк CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def load_pilot_calibration(path: Path) -> dict[str, Any]:
    """Прочитать только отчёт, прошедший независимый holdout-допуск."""

    if not path.is_file():
        raise TelemetryValidationError(f"Отчёт калибровки не найден: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TelemetryValidationError(f"Не удалось прочитать отчёт калибровки: {path}") from error
    if not isinstance(report, dict):
        raise TelemetryValidationError("Отчёт калибровки должен быть объектом JSON")
    if type(report.get("report_version")) is not int or report["report_version"] != REPORT_VERSION:
        raise TelemetryValidationError("Неподдерживаемая версия отчёта. Повторите проверку порога по исходному CSV.")
    if report.get("report_kind") != "external_telemetry_chronological_holdout_validation":
        raise TelemetryValidationError("Требуется отчёт хронологической проверки телеметрии")
    profile = _report_section(report, "deployment_profile")
    gate = _report_section(report, "deployment_gate")
    holdout = _report_section(report, "independent_holdout")
    selection = _report_section(report, "threshold_selection")
    # В JSON строка «false» и число 1 отличаются от логического true.
    if not all(flag is True for flag in (
        gate.get("passed"), gate.get("training_quality_gate_passed"),
        gate.get("holdout_quality_gate_passed"), profile.get("eligible_for_pilot"),
        holdout.get("performed"), holdout.get("quality_gate_passed"),
    )):
        raise TelemetryValidationError(
            "Отчёт не прошёл независимую проверку и не может применяться как профиль порога"
        )
    threshold = profile.get("warning_risk_threshold")
    training_metrics = _report_section(selection, "training_metrics")
    holdout_metrics = _report_section(holdout, "metrics")
    occupancy_limit = _report_section(report, "configuration").get("maximum_warning_occupancy_ratio")
    if type(occupancy_limit) not in (int, float) or not 0 <= occupancy_limit <= 1:
        raise TelemetryValidationError("В отчёте нужен предел доли времени под тревогой в диапазоне [0, 1]")
    for metrics in (training_metrics, holdout_metrics):
        occupancy = metrics.get("warning_occupancy_ratio")
        if type(occupancy) not in (int, float) or not 0 <= occupancy <= occupancy_limit:
            raise TelemetryValidationError("Доля времени под тревогой не соответствует условиям допуска")
    thresholds = (threshold, selection.get("selected_warning_risk_threshold"),
                  training_metrics.get("warning_risk_threshold"), holdout_metrics.get("warning_risk_threshold"))
    if not all(type(value) in (int, float) and 0 <= value <= 1 and value == threshold
               for value in thresholds):
        raise TelemetryValidationError("Порог профиля должен совпадать с проверенным порогом в диапазоне [0, 1]")
    return {
        "warning_risk_threshold": float(threshold),
        "warning_threshold_origin": str(profile.get("warning_threshold_origin")),
        "source_sha256": _report_section(report, "data_contract").get("source_sha256"),
    }


def _report_section(report: dict, name: str) -> dict:
    section = report.get(name)
    if not isinstance(section, dict):
        raise TelemetryValidationError(f"В отчёте нужен раздел {name} в формате объекта JSON")
    return section


def _parse_bool(value: str | None, row_number: int) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "да"}:
        return True
    if normalized in {"0", "false", "no", "нет"}:
        return False
    raise TelemetryValidationError(
        f"Строка {row_number}: attack_onset должен быть 0/1, true/false или да/нет"
    )


def _threshold_candidates(points: Iterable[TelemetryPoint]) -> list[float]:
    return sorted({point.risk_score for point in points}, reverse=True)


def _match_episode_times(episodes, onsets, config):
    """Каждый эпизод и каждое начало используют максимум один раз."""
    next_episode = 0
    matched = set()
    leads = []
    for onset in onsets:
        earliest = onset - config.maximum_lead_seconds
        latest = onset - config.minimum_lead_seconds
        while next_episode < len(episodes) and episodes[next_episode][-1] < earliest:
            next_episode += 1
        if next_episode == len(episodes):
            break
        samples = episodes[next_episode]
        first_valid = bisect_left(samples, earliest)
        if first_valid < len(samples) and samples[first_valid] <= latest:
            leads.append(onset - samples[first_valid])
            matched.add(next_episode)
            next_episode += 1
    return matched, leads


def _evaluate_threshold(
    points: list[TelemetryPoint],
    threshold: float,
    config: TelemetryValidationConfig,
    max_gap: float,
    *,
    continuing_warning: bool = False,
) -> dict[str, Any]:
    leads = []
    update_leads = []
    episode_count = onset_count = censored_count = 0
    left_censored_count = 0
    observed_seconds = warning_seconds = 0.0
    boundaries = [0]
    for index, (left, right) in enumerate(zip(points, points[1:]), 1):
        interval = right.timestamp_seconds - left.timestamp_seconds
        if interval > max_gap:
            boundaries.append(index)
        else:
            observed_seconds += interval
            if left.risk_score >= threshold:
                warning_seconds += interval
    boundaries.append(len(points))
    for start, stop in zip(boundaries, boundaries[1:]):
        segment = points[start:stop]
        episodes = _warning_episodes(segment, threshold)
        episode_count += len(episodes)
        if start == 0 and continuing_warning and episodes and segment[0].risk_score >= threshold:
            # Первая тревога уже действовала при выборе границы участков.
            # Её длительность входит в экспозицию, но новый эпизод не начался.
            left_censored_count = 1
            episodes = episodes[1:]
        onsets = [p.timestamp_seconds for p in segment if p.attack_onset]
        onset_count += len(onsets)
        starts = [[samples[0]] for samples in episodes]
        matched, initial_leads = _match_episode_times(starts, onsets, config)
        leads.extend(initial_leads)
        _, updated = _match_episode_times(episodes, onsets, config)
        update_leads.extend(updated)
        segment_end = segment[-1].timestamp_seconds
        censored_count += sum(index not in matched and start[0] + config.maximum_lead_seconds > segment_end
                              for index, start in enumerate(starts))
    predicted = len(leads)
    false_episodes = episode_count - predicted - censored_count - left_censored_count
    duration_seconds = points[-1].timestamp_seconds - points[0].timestamp_seconds
    precision = predicted / max(predicted + false_episodes, 1)
    recall = predicted / max(onset_count, 1)
    return {
        # Порог — часть правила решения, а не округлённая подпись графика.
        # Обучение, holdout и сохранённый профиль используют одно значение.
        "warning_risk_threshold": float(threshold),
        "row_count": len(points),
        "duration_seconds": duration_seconds,
        "attack_onset_count": onset_count,
        "predicted_onset_count": predicted,
        "missed_attack_count": onset_count - predicted,
        "observed_duration_seconds": observed_seconds,
        "gap_duration_seconds": duration_seconds - observed_seconds,
        "warning_duration_seconds": warning_seconds,
        "warning_occupancy_ratio": warning_seconds / observed_seconds if observed_seconds else None,
        "warning_episode_count": episode_count,
        "left_censored_warning_count": left_censored_count,
        "right_censored_warning_count": censored_count,
        "false_warning_episode_count": false_episodes,
        "false_warnings_per_hour": false_episodes * 3600.0 / observed_seconds if observed_seconds else None,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "minimum_observed_lead_seconds": min(leads, default=0.0),
        "median_observed_lead_seconds": median(leads) if leads else 0.0,
        "within_episode_updates": {
            "predicted_onset_count": len(update_leads),
            "minimum_observed_lead_seconds": min(update_leads, default=0.0),
            "median_observed_lead_seconds": median(update_leads) if update_leads else 0.0,
        },
    }


def _maximum_sample_gap(points: list[TelemetryPoint]) -> float:
    intervals = [right.timestamp_seconds - left.timestamp_seconds for left, right in zip(points, points[1:])]
    return 1.5 * median(intervals) if intervals else 0.0


def _warning_episodes(points: list[TelemetryPoint], threshold: float) -> list[list[float]]:
    episodes: list[list[float]] = []
    active = False
    for point in points:
        if point.risk_score < threshold:
            active = False
            continue
        if not active:
            episodes.append([])
        episodes[-1].append(point.timestamp_seconds)
        active = True
    return episodes


def _meets_quality_gate(metrics: dict[str, Any], config: TelemetryValidationConfig) -> bool:
    return bool(
        int(metrics["attack_onset_count"]) >= config.minimum_events_per_partition
        and metrics["right_censored_warning_count"] == 0
        and float(metrics["precision"]) >= config.minimum_precision
        and float(metrics["recall"]) >= config.minimum_recall
        and metrics["false_warnings_per_hour"] is not None
        and metrics["false_warnings_per_hour"] <= config.maximum_false_warnings_per_hour
        and metrics["warning_occupancy_ratio"] is not None
        and metrics["warning_occupancy_ratio"] <= config.maximum_warning_occupancy_ratio
    )


def _time_range(points: list[TelemetryPoint]) -> list[float]:
    return [points[0].timestamp_seconds, points[-1].timestamp_seconds]


def _source_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
