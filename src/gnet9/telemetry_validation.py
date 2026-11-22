"""Офлайн-валидация порога предупреждений по внешней телеметрии.

Модуль намеренно не подключается к оборудованию и не отправляет команды в
сеть. Он принимает уже обезличенную временную таблицу с риском, рассчитанным
тем же экземпляром модели Купмана, и с подтверждёнными началами инцидентов.
Так можно получить воспроизводимый профиль порога без утечки будущих меток в
обучающую часть и проверить его на более позднем, независимом участке ряда.
"""

from __future__ import annotations

from bisect import bisect_right
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
    selected = max(
        (_evaluate_threshold(training_points, threshold, config, training_gap) for threshold in candidates),
        key=lambda item: (
            int(_meets_quality_gate(item, config)),
            float(item["recall"]),
            float(item["precision"]),
            -item["false_warnings_per_hour"] if item["false_warnings_per_hour"] is not None else -math.inf,
            float(item["warning_risk_threshold"]),
        ),
    )
    threshold = float(selected["warning_risk_threshold"])
    holdout = _evaluate_threshold(holdout_points, threshold, config, _maximum_sample_gap(holdout_points))
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
                "Прохождение допускает только наблюдаемый пилот с рекомендациями человеку; "
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
    except (OSError, ValueError) as error:
        raise TelemetryValidationError(f"Не удалось прочитать отчёт калибровки: {path}") from error
    if not isinstance(report, dict):
        raise TelemetryValidationError("Отчёт калибровки должен быть объектом JSON")
    if type(report.get("report_version")) is not int or report["report_version"] != REPORT_VERSION:
        raise TelemetryValidationError("Неподдерживаемая версия отчёта калибровки")
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
    thresholds = (threshold, selection.get("selected_warning_risk_threshold"),
                  _report_section(selection, "training_metrics").get("warning_risk_threshold"),
                  _report_section(holdout, "metrics").get("warning_risk_threshold"))
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
    unique_scores = sorted({point.risk_score for point in points})
    if len(unique_scores) <= 257:
        return unique_scores
    return sorted({unique_scores[round(index * (len(unique_scores) - 1) / 256)] for index in range(257)})


def _evaluate_threshold(
    points: list[TelemetryPoint],
    threshold: float,
    config: TelemetryValidationConfig,
    max_gap: float,
) -> dict[str, Any]:
    episodes = _warning_episodes(points, threshold, max_gap)
    next_episode = 0
    onset_count = 0
    leads: list[float] = []
    for onset in points:
        if not onset.attack_onset:
            continue
        onset_count += 1
        earliest = onset.timestamp_seconds - config.maximum_lead_seconds
        latest = onset.timestamp_seconds - config.minimum_lead_seconds
        # Окна движутся вперёд. Самый ранний подходящий эпизод оставляет
        # следующие эпизоды доступными для соседних начал инцидентов.
        while next_episode < len(episodes) and episodes[next_episode][-1] < earliest:
            next_episode += 1
        if next_episode < len(episodes):
            samples = episodes[next_episode]
            stop = bisect_right(samples, latest)
            if stop and samples[stop - 1] >= earliest:
                leads.append(onset.timestamp_seconds - samples[stop - 1])
                next_episode += 1
    predicted = len(leads)
    false_episodes = len(episodes) - predicted
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
        "warning_episode_count": len(episodes),
        "false_warning_episode_count": false_episodes,
        "false_warnings_per_hour": false_episodes * 3600.0 / duration_seconds if duration_seconds else None,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "minimum_observed_lead_seconds": min(leads, default=0.0),
        "median_observed_lead_seconds": median(leads) if leads else 0.0,
    }


def _maximum_sample_gap(points: list[TelemetryPoint]) -> float:
    intervals = [right.timestamp_seconds - left.timestamp_seconds for left, right in zip(points, points[1:])]
    return 1.5 * median(intervals) if intervals else 0.0


def _warning_episodes(points: list[TelemetryPoint], threshold: float, max_gap: float) -> list[list[float]]:
    episodes: list[list[float]] = []
    active = False
    for point in points:
        if point.risk_score < threshold:
            active = False
            continue
        if not active or point.timestamp_seconds - episodes[-1][-1] > max_gap:
            episodes.append([])
        episodes[-1].append(point.timestamp_seconds)
        active = True
    return episodes


def _meets_quality_gate(metrics: dict[str, Any], config: TelemetryValidationConfig) -> bool:
    return bool(
        int(metrics["attack_onset_count"]) >= config.minimum_events_per_partition
        and float(metrics["precision"]) >= config.minimum_precision
        and float(metrics["recall"]) >= config.minimum_recall
        and metrics["false_warnings_per_hour"] is not None
        and metrics["false_warnings_per_hour"] <= config.maximum_false_warnings_per_hour
    )


def _time_range(points: list[TelemetryPoint]) -> list[float]:
    return [points[0].timestamp_seconds, points[-1].timestamp_seconds]


def _source_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
