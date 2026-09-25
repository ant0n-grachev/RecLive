"""Reporting-only metrics and transient evaluation evidence; no model imports."""

import math
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from numbers import Real
from typing import Mapping, Sequence


def finite_number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def reporting_utc(value, *, trusted_db=False):
    """Only trusted DB naive timestamps mean UTC; external timestamps need offsets."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        if not trusted_db:
            return None
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class ForecastMetrics:
    mae_people: float | None
    mae_capacity_percentage_points: float | None
    rmse_people: float | None
    prediction_interval_coverage: float | None
    simple_baseline_mae_people: float | None
    rolling_holdout_by_facility: Mapping[int, "FacilityHoldoutMetrics"]
    observation_counts: Mapping[str, int] = field(default_factory=dict)

    def to_payload(self):
        return {
            "maePeople": self.mae_people,
            "maeCapacityPercentagePoints": self.mae_capacity_percentage_points,
            "rmsePeople": self.rmse_people,
            "predictionIntervalCoverage": self.prediction_interval_coverage,
            "simpleBaselineMaePeople": self.simple_baseline_mae_people,
            "rollingHoldoutByFacility": {
                str(key): value.to_payload()
                for key, value in self.rolling_holdout_by_facility.items()
            },
        }


def compute_forecast_metrics(
    actual_people, predicted_people, capacity_people, lower_people, upper_people,
    baseline_people,
) -> ForecastMetrics:
    vectors = (actual_people, predicted_people, capacity_people, lower_people,
               upper_people, baseline_people)
    if len({len(vector) for vector in vectors}) != 1:
        raise ValueError("metric vectors must have equal lengths")
    errors, capacity_errors, coverage, baseline_errors = [], [], [], []
    for actual, predicted, capacity, lower, upper, baseline in zip(*vectors):
        if not finite_number(actual):
            continue
        if finite_number(predicted):
            error = abs(float(predicted) - float(actual))
            if math.isfinite(error):
                errors.append(error)
                if finite_number(capacity) and capacity > 0:
                    percentage_points = error / float(capacity) * 100.0
                    if math.isfinite(percentage_points):
                        capacity_errors.append(percentage_points)
        if finite_number(lower) and finite_number(upper) and lower <= upper:
            coverage.append(float(lower <= actual <= upper))
        if finite_number(baseline):
            error = abs(float(baseline) - float(actual))
            if math.isfinite(error):
                baseline_errors.append(error)

    def mean(values):
        return math.fsum(value / len(values) for value in values) if values else None

    # Scaling avoids overflow when squaring an otherwise finite people error.
    scale = max(errors, default=0.0)
    rmse = (scale * math.sqrt(mean([(error / scale) ** 2 for error in errors]))
            if scale else (0.0 if errors else None))
    return ForecastMetrics(
        mean(errors), mean(capacity_errors), rmse, mean(coverage),
        mean(baseline_errors), {}, {
            "maePeople": len(errors), "maeCapacityPercentagePoints": len(capacity_errors),
            "rmsePeople": len(errors), "predictionIntervalCoverage": len(coverage),
            "simpleBaselineMaePeople": len(baseline_errors),
        },
    )


@dataclass(frozen=True)
class RawBaselineObservation:
    observed_at: datetime
    available_at: datetime
    people: float


def raw_baseline_observation(observed_at, available_at, people):
    observed = reporting_utc(observed_at, trusted_db=True)
    available = reporting_utc(available_at, trusted_db=True)
    if observed is None or available is None or not finite_number(people) or people < 0:
        return None
    return RawBaselineObservation(observed, available, float(people))


def persistence_baseline(observations, start):
    start = reporting_utc(start)
    if start is None:
        return None
    eligible = [row for row in observations if isinstance(row, RawBaselineObservation)
                and reporting_utc(row.observed_at) is not None
                and reporting_utc(row.available_at) is not None
                and row.observed_at < start and row.available_at < start
                and finite_number(row.people) and row.people >= 0]
    if not eligible:
        return None
    latest = max(row.observed_at for row in eligible)
    values = {row.people for row in eligible if row.observed_at == latest}
    # No stable source ID exists: conflicting observations cannot be ordered safely.
    return next(iter(values)) if len(values) == 1 else None


@dataclass(frozen=True)
class ForecastEvaluationRow:
    facility_id: int
    location_id: int
    target: datetime
    actual_people: float | None
    predicted_people: float | None
    capacity_people: float | None
    lower_people: float | None = None
    upper_people: float | None = None
    baseline_people: float | None = None


@dataclass(frozen=True)
class RollingHoldoutWindow:
    facility_id: int
    start: datetime
    end: datetime


@dataclass(frozen=True)
class FacilityHoldoutMetrics:
    facility_id: int
    windows: Sequence[tuple[RollingHoldoutWindow, ForecastMetrics]]

    def to_payload(self):
        return {
            "method": "fixed_model_terminal_holdout",
            "independentBacktest": False,
            "windows": [{"start": window.start.isoformat(), "end": window.end.isoformat(),
                         "metrics": metrics.to_payload(),
                         "observationCounts": dict(metrics.observation_counts)}
                        for window, metrics in self.windows],
        }


def metrics_for_rows(rows):
    return compute_forecast_metrics(*[
        [getattr(row, name) for row in rows] for name in (
            "actual_people", "predicted_people", "capacity_people", "lower_people",
            "upper_people", "baseline_people",
        )
    ])


def compute_rolling_holdout_by_facility(rows, windows):
    grouped = {}
    for window in windows:
        start, end = reporting_utc(window.start), reporting_utc(window.end)
        if window.facility_id not in (1186, 1656) or start is None or end is None or start >= end:
            raise ValueError("holdout windows require supported facilities and aware ordered times")
        grouped.setdefault(window.facility_id, []).append(replace(window, start=start, end=end))
    seen = set()
    for row in rows:
        target = reporting_utc(row.target)
        identity = (row.facility_id, row.location_id, target)
        if target is None or identity in seen:
            raise ValueError("holdout rows require aware times and unique identities")
        seen.add(identity)
    result = {}
    for facility_id, facility_windows in grouped.items():
        facility_windows.sort(key=lambda window: window.start)
        if any(left.end > right.start for left, right in zip(facility_windows, facility_windows[1:])):
            raise ValueError("holdout windows must not overlap")
        result[facility_id] = FacilityHoldoutMetrics(facility_id, [
            (window, metrics_for_rows([row for row in rows if row.facility_id == facility_id
                                      and window.start <= row.target < window.end]))
            for window in facility_windows
        ])
    return result


def terminal_evaluation_evidence(model_key, metadata, times, indices, predicted, lower, upper,
                                 split_start, interval_minutes):
    """Capture existing pre-calibration predictions without changing their arrays."""
    start = reporting_utc(split_start)
    if (start is None or len(metadata) != len(times) or not indices
            or len({len(indices), len(predicted), len(lower), len(upper)}) != 1
            or not finite_number(interval_minutes) or interval_minutes <= 0):
        return None
    rows, raw_by_location = [], {}
    for index, prediction, low, high in zip(indices, predicted, lower, upper):
        if not isinstance(index, int) or index < 0 or index >= len(metadata):
            return None
        meta = metadata[index]
        if not isinstance(meta, dict) or meta.get("time_aligned") is not True:
            return None
        target = reporting_utc(meta.get("target"))
        capacity = meta.get("capacity")
        facility, location = meta.get("facility_id"), meta.get("location_id")
        if (target is None or target != reporting_utc(times[index]) or target < start
                or facility not in (1186, 1656) or not isinstance(location, int)
                or isinstance(location, bool) or not finite_number(capacity) or capacity <= 0):
            return None

        def people(value):
            product = float(value) * float(capacity) if finite_number(value) else None
            return product if finite_number(product) else None

        rows.append(ForecastEvaluationRow(facility, location, target, meta.get("actual_people"),
                                         people(prediction), float(capacity), people(low), people(high)))
        raw_by_location[location] = meta.get("raw_baseline", ())
    if len({row.facility_id for row in rows}) != 1:
        return None
    end = max(row.target for row in rows) + timedelta(minutes=float(interval_minutes))
    windows = []
    while start < end:
        window_end = min(start + timedelta(hours=24), end)
        windows.append(RollingHoldoutWindow(rows[0].facility_id, start, window_end))
        start = window_end
    baseline_by_window = {
        (location, window.start): persistence_baseline(observations, window.start)
        for location, observations in raw_by_location.items() for window in windows
    }
    rows = [replace(row, baseline_people=baseline_by_window[(row.location_id, window.start)])
            for row in rows for window in windows if window.start <= row.target < window.end]
    try:
        compute_rolling_holdout_by_facility(rows, windows)
    except ValueError:
        return None
    return {"model_key": model_key, "stage": "terminal_pre_calibration", "rows": rows,
            "windows": windows}


sys.modules.setdefault("server.reclive.forecasting.metrics", sys.modules[__name__])
sys.modules.setdefault("reclive.forecasting.metrics", sys.modules[__name__])
