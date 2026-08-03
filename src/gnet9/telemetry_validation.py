"""Офлайн-валидация порога предупреждений по внешней телеметрии.

Модуль намеренно не подключается к оборудованию и не отправляет команды в
сеть. Он принимает уже обезличенную временную таблицу с риском, рассчитанным
тем же экземпляром модели Купмана, и с подтверждёнными началами инцидентов.
Так можно получить воспроизводимый профиль порога без утечки будущих меток в
обучающую часть и проверить его на более позднем, независимом участке ряда.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable


REPORT_VERSION = 1
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

    def validate(self) -> None:
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


def read_telemetry_csv(path: Path) -> list[TelemetryPoint]:
    """Прочитать строгий и минимальный CSV-контракт внешней телеметрии."""

    if not path.is_file():
        raise TelemetryValidationError(f"Файл телеметрии не найден: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise TelemetryValidationError("CSV не содержит строку заголовков")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise TelemetryValidationError(
                "В CSV отсутствуют обязательные поля: " + ", ".join(missing)
            )
        points: list[TelemetryPoint] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = float(row["timestamp_seconds"])
                risk_score = float(row["risk_score"])
            except (TypeError, ValueError) as error:
                raise TelemetryValidationError(
                    f"Строка {row_number}: timestamp_seconds и risk_score должны быть числами"
                ) from error
            if not math.isfinite(timestamp) or not math.isfinite(risk_score):
                raise TelemetryValidationError(f"Строка {row_number}: значения должны быть конечными")
            if not 0.0 <= risk_score <= 1.0:
                raise TelemetryValidationError(
                    f"Строка {row_number}: risk_score должен находиться в диапазоне [0, 1]"
                )
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
    if len(points) < 3:
        raise TelemetryValidationError("Нужно не менее трёх отсчётов телеметрии")
    for previous, current in zip(points, points[1:]):
        if current.timestamp_seconds <= previous.timestamp_seconds:
            raise TelemetryValidationError(
                "timestamp_seconds должен строго возрастать; сортировка входных данных не выполняется намеренно"
            )
    return points


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
    if len(points) < 3:
        raise TelemetryValidationError("Нужно не менее трёх отсчётов")
    split_index = int(len(points) * config.training_fraction)
    split_index = min(max(split_index, 1), len(points) - 1)
    training_points = points[:split_index]
    holdout_points = points[split_index:]
    candidates = _threshold_candidates(training_points)
    candidate_evaluations = [
        _evaluate_threshold(training_points, threshold, config)
        for threshold in candidates
    ]
    feasible = [
        item for item in candidate_evaluations if _meets_quality_gate(item, config)
    ]
    selected = max(
        feasible or candidate_evaluations,
        key=lambda item: (
            int(_meets_quality_gate(item, config)),
            float(item["recall"]),
            float(item["precision"]),
            -float(item["false_warnings_per_hour"]),
            float(item["warning_risk_threshold"]),
        ),
    )
    threshold = float(selected["warning_risk_threshold"])
    holdout = _evaluate_threshold(holdout_points, threshold, config)
    training_gate = _meets_quality_gate(selected, config)
    holdout_gate = _meets_quality_gate(holdout, config)
    eligible = bool(training_gate and holdout_gate)
    source_digest = _source_digest(source_path) if source_path else None

    return {
        "report_version": REPORT_VERSION,
        "report_kind": "external_telemetry_chronological_holdout_validation",
        "data_contract": {
            "required_columns": list(REQUIRED_COLUMNS),
            "optional_columns": ["attack_id", "attack_kind"],
            "risk_score_semantics": "risk score of the same Koopman model revision, not a probability",
            "raw_payloads_or_personal_data_expected": False,
            "source_sha256": source_digest,
        },
        "validation_protocol_ru": (
            "Хронологическое разделение без перемешивания: порог выбирается только на раннем "
            "участке, затем фиксируется и проверяется на последующем независимом holdout-участке."
        ),
        "configuration": asdict(config),
        "partition": {
            "training_row_count": len(training_points),
            "holdout_row_count": len(holdout_points),
            "training_time_range_seconds": _time_range(training_points),
            "holdout_time_range_seconds": _time_range(holdout_points),
            "split_timestamp_seconds": holdout_points[0].timestamp_seconds,
        },
        "threshold_selection": {
            "candidate_count": len(candidates),
            "selection_uses_holdout": False,
            "selection_status": "feasible" if feasible else "no_training_threshold_meets_gate",
            "selected_warning_risk_threshold": threshold,
            "warning_threshold_origin": "chronological_external_telemetry_calibration",
            "training_metrics": _public_metrics(selected),
        },
        "independent_holdout": {
            "performed": True,
            "metrics": _public_metrics(holdout),
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
                "Прохождение допускает только наблюдаемый пилот с ручным утверждением ремаппинга; "
                "оно не является гарантией отсутствия ошибок и не разрешает автоматически менять реальную сеть."
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
    except (OSError, json.JSONDecodeError) as error:
        raise TelemetryValidationError(f"Не удалось прочитать отчёт калибровки: {path}") from error
    if report.get("report_version") != REPORT_VERSION:
        raise TelemetryValidationError("Неподдерживаемая версия отчёта калибровки")
    profile = report.get("deployment_profile", {})
    gate = report.get("deployment_gate", {})
    threshold = profile.get("warning_risk_threshold")
    if not gate.get("passed") or not profile.get("eligible_for_pilot"):
        raise TelemetryValidationError(
            "Отчёт не прошёл независимую проверку и не может применяться как профиль порога"
        )
    if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
        raise TelemetryValidationError("В отчёте отсутствует корректный warning_risk_threshold")
    return {
        "warning_risk_threshold": float(threshold),
        "warning_threshold_origin": str(profile.get("warning_threshold_origin")),
        "source_sha256": report.get("data_contract", {}).get("source_sha256"),
    }


def _parse_bool(value: str | None, row_number: int) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "да"}:
        return True
    if normalized in {"0", "false", "no", "нет", ""}:
        return False
    raise TelemetryValidationError(
        f"Строка {row_number}: attack_onset должен быть 0/1, true/false или да/нет"
    )


def _threshold_candidates(points: Iterable[TelemetryPoint]) -> list[float]:
    unique_scores = sorted({point.risk_score for point in points})
    if len(unique_scores) <= 257:
        return unique_scores
    return sorted({unique_scores[round(index * (len(unique_scores) - 1) / 256)] for index in range(257)})


def _evaluate_threshold(
    points: list[TelemetryPoint],
    threshold: float,
    config: TelemetryValidationConfig,
) -> dict[str, Any]:
    onsets = [point for point in points if point.attack_onset]
    warning_episodes = _warning_episodes(points, threshold)
    used_episode_indexes: set[int] = set()
    records: list[dict[str, Any]] = []
    for onset in onsets:
        earliest = onset.timestamp_seconds - config.maximum_lead_seconds
        latest = onset.timestamp_seconds - config.minimum_lead_seconds
        candidates = [
            (index, sample_time)
            for index, episode in enumerate(warning_episodes)
            if index not in used_episode_indexes
            for sample_time in episode["sample_times"]
            if earliest <= sample_time <= latest
        ]
        if candidates:
            episode_index, warning_time = max(candidates, key=lambda item: item[1])
            used_episode_indexes.add(episode_index)
            lead = onset.timestamp_seconds - warning_time
            records.append(
                {
                    "attack_id": onset.attack_id,
                    "attack_kind": onset.attack_kind,
                    "onset_seconds": onset.timestamp_seconds,
                    "predicted": True,
                    "warning_seconds": warning_time,
                    "lead_seconds": lead,
                }
            )
        else:
            records.append(
                {
                    "attack_id": onset.attack_id,
                    "attack_kind": onset.attack_kind,
                    "onset_seconds": onset.timestamp_seconds,
                    "predicted": False,
                    "warning_seconds": None,
                    "lead_seconds": None,
                }
            )
    predicted = sum(record["predicted"] for record in records)
    false_episodes = len(warning_episodes) - len(used_episode_indexes)
    duration_seconds = max(points[-1].timestamp_seconds - points[0].timestamp_seconds, 1.0)
    precision = predicted / max(predicted + false_episodes, 1)
    recall = predicted / max(len(onsets), 1)
    leads = [float(record["lead_seconds"]) for record in records if record["lead_seconds"] is not None]
    return {
        "warning_risk_threshold": round(float(threshold), 6),
        "row_count": len(points),
        "duration_seconds": duration_seconds,
        "attack_onset_count": len(onsets),
        "predicted_onset_count": predicted,
        "missed_attack_ids": [record["attack_id"] for record in records if not record["predicted"]],
        "warning_episode_count": len(warning_episodes),
        "false_warning_episode_count": false_episodes,
        "false_warnings_per_hour": false_episodes * 3600.0 / duration_seconds,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "minimum_observed_lead_seconds": min(leads, default=0.0),
        "median_observed_lead_seconds": median(leads) if leads else 0.0,
        "event_records": records,
    }


def _warning_episodes(points: list[TelemetryPoint], threshold: float) -> list[dict[str, Any]]:
    selected = [point.timestamp_seconds for point in points if point.risk_score >= threshold]
    if not selected:
        return []
    intervals = [right.timestamp_seconds - left.timestamp_seconds for left, right in zip(points, points[1:])]
    sample_interval = median(intervals) if intervals else 1.0
    max_gap = max(sample_interval * 1.5, 1e-9)
    episodes: list[list[float]] = [[selected[0]]]
    for sample_time in selected[1:]:
        if sample_time - episodes[-1][-1] <= max_gap:
            episodes[-1].append(sample_time)
        else:
            episodes.append([sample_time])
    return [
        {"start_seconds": values[0], "end_seconds": values[-1], "sample_times": values}
        for values in episodes
    ]


def _meets_quality_gate(metrics: dict[str, Any], config: TelemetryValidationConfig) -> bool:
    return bool(
        int(metrics["attack_onset_count"]) >= config.minimum_events_per_partition
        and float(metrics["precision"]) >= config.minimum_precision
        and float(metrics["recall"]) >= config.minimum_recall
        and float(metrics["false_warnings_per_hour"]) <= config.maximum_false_warnings_per_hour
        and not metrics["missed_attack_ids"]
    )


def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Не переносить ID инцидентов и временные строки из CSV в отчёт."""

    result = {
        key: value
        for key, value in metrics.items()
        if key not in {"event_records", "missed_attack_ids"}
    }
    result["missed_attack_count"] = len(metrics["missed_attack_ids"])
    return result


def _time_range(points: list[TelemetryPoint]) -> list[float]:
    return [points[0].timestamp_seconds, points[-1].timestamp_seconds]


def _source_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
