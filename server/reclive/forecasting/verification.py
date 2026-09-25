"""Score finished hours only against immutable forecasts published in advance.

The one-hour and one-day cohorts use publications 1–2 and 24–25 hours before
the target hour starts. The upper bound is exclusive, keeping outages from
silently substituting older forecasts. This module never fetches or writes data.
"""

from __future__ import annotations

import gzip
import json
import math
import zlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .publication import _ARCHIVE_NAME


UTC = timezone.utc
LEAD_HOURS = (1, 24)


@dataclass(frozen=True)
class HourlyPrediction:
    published_at: datetime
    generated_at: datetime
    archive: str
    predicted_count: int
    expected_pct: float | None
    thresholds: tuple[float, float] | None


PredictionIndex = dict[tuple[int, datetime], list[HourlyPrediction]]


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("time must be an aware datetime")
    return value.astimezone(UTC)


def _parse(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO string with an offset")
    return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _number(value: object, maximum: float | None = None) -> bool:
    try:
        return (
            type(value) in (int, float)
            and math.isfinite(value)
            and value >= 0
            and (maximum is None or value <= maximum)
        )
    except OverflowError:
        return False


def _js_round(value: float) -> int:
    return math.floor(value + .5)


def _thresholds(value: object) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        return None
    low, peak = value.get("lowMax"), value.get("peakMin")
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in (low, peak)):
        return None
    low = max(0, min(99, low))
    return low, max(low + 1, min(100, peak))


def _crowd(pct: float | None, thresholds: tuple[float, float] | None) -> str | None:
    if pct is None or thresholds is None:
        return None
    percent = pct * 100
    if percent <= thresholds[0]:
        return "low"
    return "medium" if percent < thresholds[1] else "peak"


def _archive_predictions(payload: object, name: str, now: datetime, min_hour: datetime | None):
    if not isinstance(payload, dict) or payload.get("schemaVersion") != 1:
        raise ValueError("unsupported archive")
    published = _parse(payload.get("publishedAt"))
    generated = _parse(payload.get("generatedAt"))
    if not generated <= published <= now:
        raise ValueError("invalid publication chronology")
    match = _ARCHIVE_NAME.fullmatch(name)
    if match and datetime.strptime(match[1], "%Y%m%dT%H%M%S%fZ").replace(tzinfo=UTC) != published:
        raise ValueError("archive filename and publication time disagree")
    zone = ZoneInfo(payload["timezone"])
    results = []
    facilities = payload["facilities"]
    if not isinstance(facilities, list):
        raise ValueError("facilities must be a list")
    seen_facilities = set()
    for facility in facilities:
        facility_id = facility["facilityId"]
        if type(facility_id) is not int or facility_id <= 0 or facility_id in seen_facilities:
            raise ValueError("invalid or duplicate facility")
        seen_facilities.add(facility_id)
        thresholds = _thresholds(facility.get("occupancyThresholds"))
        hours = defaultdict(dict)
        ambiguous = set()
        for day in facility["weeklyForecast"]:
            for point in day["totalHours"]:
                instant = _parse(point["hourStart"])
                if instant.second or instant.microsecond or instant.minute % 15:
                    raise ValueError("forecast point is not on a quarter hour")
                local = instant.astimezone(zone)
                if local.date().isoformat() != day["date"]:
                    raise ValueError("forecast point is on another local date")
                # Explicit offsets distinguish repeated fall-back hours; reject
                # impossible local offsets instead of shifting a malformed hour.
                raw = datetime.fromisoformat(point["hourStart"].replace("Z", "+00:00"))
                if raw.utcoffset() != timedelta(0) and raw.utcoffset() != local.utcoffset():
                    raise ValueError("forecast point offset disagrees with timezone")
                start = instant.replace(minute=0)
                if min_hour is not None and start < min_hour:
                    continue
                count = point.get("expectedCount")
                if not _number(count):
                    raise ValueError("invalid forecast count")
                pct = point.get("expectedPct")
                if pct is not None and not _number(pct, 1):
                    raise ValueError("invalid forecast percentage")
                item = (count, pct)
                previous = hours[start].get(instant.minute)
                if previous is not None and previous != item:
                    ambiguous.add(start)
                hours[start][instant.minute] = item
        for start, points in hours.items():
            if start in ambiguous or set(points) != {0, 15, 30, 45}:
                continue
            values = [points[minute] for minute in (0, 15, 30, 45)]
            pct = sum(item[1] for item in values) / 4 if all(item[1] is not None for item in values) else None
            results.append(((facility_id, start), HourlyPrediction(
                published_at=published,
                generated_at=generated,
                archive=name,
                predicted_count=_js_round(sum(item[0] for item in values) / 4),
                expected_pct=pct,
                thresholds=thresholds,
            )))
    return results


def load_predictions(archive_dir, now: datetime, *, min_hour: datetime | None = None) -> tuple[PredictionIndex, int]:
    """Read valid archives into a UTC-hour index, returning corrupt-file count.

    An optional minimum target hour bounds work for routine incremental runs.
    Missing directories are empty evidence, and symlinks are never followed.
    """
    now = _utc(now)
    minimum = _utc(min_hour) if min_hour is not None else None
    predictions: PredictionIndex = {}
    invalid = 0
    for path in sorted(Path(archive_dir).glob("*.json.gz")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            match = _ARCHIVE_NAME.fullmatch(path.name)
            if minimum is not None and match:
                filename_time = datetime.strptime(match[1], "%Y%m%dT%H%M%S%fZ").replace(tzinfo=UTC)
                if filename_time <= minimum - timedelta(hours=max(LEAD_HOURS) + 1):
                    continue
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            items = _archive_predictions(payload, path.name, now, minimum)
        except (OSError, EOFError, ValueError, TypeError, KeyError, AttributeError, OverflowError, ZoneInfoNotFoundError, zlib.error):
            invalid += 1
            continue
        for key, prediction in items:
            predictions.setdefault(key, []).append(prediction)
    for values in predictions.values():
        values.sort(key=lambda item: (item.published_at, item.generated_at, item.archive))
    return predictions, invalid


def _qualified_actual(actual: dict | None, hour_start: datetime) -> bool:
    if not isinstance(actual, dict):
        return False
    try:
        if _parse(actual.get("hourStart")) != hour_start:
            return False
    except (TypeError, ValueError):
        return False
    count = actual.get("actualCount")
    threshold = actual.get("coverageThreshold")
    if type(count) is not int or count < 0 or not _number(threshold, 1) or threshold <= 0:
        return False
    if not _number(actual.get("expectedCapacity")) or actual["expectedCapacity"] <= 0:
        return False
    return all(
        _number(actual.get(key), 1) and actual[key] + 1e-12 >= threshold
        for key in ("actualCoverage", "temporalCoverage")
    )


def evaluate_hour(
    predictions: PredictionIndex,
    facility_id: int,
    hour_start: datetime,
    actual: dict | None,
    now: datetime,
    evaluated_at: datetime | None = None,
) -> list[dict]:
    """Return one record per lead cohort; never score an unfinished hour."""
    start, now = _utc(hour_start), _utc(now)
    if start.minute or start.second or start.microsecond:
        raise ValueError("target must start on an exact hour")
    evaluated = _utc(evaluated_at) if evaluated_at is not None else now
    if start + timedelta(hours=1) > now:
        return []
    qualified = _qualified_actual(actual, start)
    actual = actual if isinstance(actual, dict) else {}
    count = actual.get("actualCount") if qualified else None
    pct = actual.get("actualPct") if qualified else None
    pct = pct if _number(pct, 1) else None
    rows = []
    for lead in LEAD_HOURS:
        cutoff = start - timedelta(hours=lead)
        candidates = [
            item for item in predictions.get((facility_id, start), [])
            if cutoff - timedelta(hours=1) < item.published_at <= cutoff
            and item.generated_at <= item.published_at <= now
        ]
        selected = max(candidates, key=lambda item: (item.published_at, item.generated_at, item.archive), default=None)
        if selected is not None:
            tied_values = {
                (item.generated_at, item.predicted_count, item.expected_pct, item.thresholds)
                for item in candidates if item.published_at == selected.published_at
            }
            # Identical retries are harmless; conflicting evidence with the
            # same publication instant cannot establish which was served.
            if len(tied_values) != 1:
                selected = None
        predicted = selected.predicted_count if selected else None
        error = predicted - count if predicted is not None and count is not None else None
        thresholds = selected.thresholds if selected else None
        predicted_crowd = _crowd(selected.expected_pct, thresholds) if selected else None
        actual_crowd = _crowd(pct, thresholds)
        row = {
            "facilityId": facility_id,
            "hourStart": hour_start.isoformat(),
            "leadHours": lead,
            "status": "no_forecast" if selected is None else "scored" if qualified else "insufficient_actual",
            "evaluatedAt": evaluated.isoformat(),
            "publishedAt": selected.published_at.isoformat() if selected else None,
            "generatedAt": selected.generated_at.isoformat() if selected else None,
            "archive": selected.archive if selected else None,
            "actualLeadHours": (start - selected.published_at).total_seconds() / 3600 if selected else None,
            "predictedCount": predicted,
            "actualCount": count,
            "errorPeople": error,
            "absoluteError": abs(error) if error is not None else None,
            "expectedPct": selected.expected_pct if selected else None,
            "actualPct": pct,
            "predictedCrowd": predicted_crowd,
            "actualCrowd": actual_crowd,
            "crowdMatch": predicted_crowd == actual_crowd if predicted_crowd is not None and actual_crowd is not None else None,
        }
        for key in ("actualCoverage", "temporalCoverage", "coverageThreshold", "observedCount", "observedCapacity", "expectedCapacity"):
            value = actual.get(key)
            row[key] = value if _number(value) else None
        rows.append(row)
    return rows
