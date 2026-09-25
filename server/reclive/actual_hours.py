from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


UTC = timezone.utc
CHICAGO = ZoneInfo("America/Chicago")
DATE_KEY_PATTERN = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
COVERAGE_COMPARISON_ABS_TOLERANCE = 1e-12


@dataclass(frozen=True)
class HistoryState:
    location_id: int
    is_closed: bool
    count: int
    capacity: int
    fetched_at: datetime
    id: int

    def __post_init__(self) -> None:
        _require_positive_integer(self.location_id, "location ID")
        if type(self.is_closed) is not bool:
            raise ValueError("is_closed must be a boolean")
        _require_non_negative_integer(self.count, "count")
        _require_non_negative_integer(self.capacity, "capacity")
        _require_aware_utc(self.fetched_at, "fetched_at")
        _require_positive_integer(self.id, "event ID")


@dataclass(frozen=True)
class IngestionHeartbeat:
    completed_at: datetime
    observed_location_ids: frozenset[int]

    def __post_init__(self) -> None:
        _require_aware_utc(self.completed_at, "completed_at")
        if not isinstance(self.observed_location_ids, frozenset) or any(
            not _is_positive_integer(location_id)
            for location_id in self.observed_location_ids
        ):
            raise ValueError(
                "observed location IDs must be a frozenset of positive integers"
            )


@dataclass(frozen=True)
class HourWindow:
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        _require_aware_utc(self.start, "hour window start")
        _require_aware_utc(self.end, "hour window end")
        if self.end <= self.start:
            raise ValueError("hour window must end after start")


@dataclass(frozen=True)
class ActualHourSummary:
    observed_count: int | None
    observed_capacity: int
    expected_capacity: int
    actual_coverage: float
    temporal_coverage: float
    coverage_threshold: float
    actual_count: int | None


@dataclass(frozen=True)
class _ConfirmedStateSegment:
    start: datetime
    end: datetime
    is_closed: bool
    count: int
    capacity: int


def calculate_actual_hour(
    location_ids: Sequence[int],
    expected_capacity: int,
    window: HourWindow,
    states: Sequence[HistoryState],
    heartbeats: Sequence[IngestionHeartbeat],
    coverage_threshold: float,
) -> ActualHourSummary:
    requested_location_ids = _validated_requested_location_ids(location_ids)
    if type(expected_capacity) is not int or expected_capacity < 0:
        raise ValueError("expected capacity must be a non-negative integer")
    if not isinstance(window, HourWindow):
        raise ValueError("window must be an HourWindow")
    state_items = _validated_states(states)
    heartbeat_items = _validated_heartbeats(heartbeats)
    threshold = _validated_coverage_threshold(coverage_threshold)

    event_ids = [state.id for state in state_items]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("history event IDs must be unique")

    observed_count_total = 0.0
    observed_capacity_total = 0.0
    confirmed_capacity_seconds = 0.0

    for location_id in requested_location_ids:
        location_known_seconds = 0.0
        location_count_seconds = 0.0
        location_capacity_seconds = 0.0

        for segment in _confirmed_state_segments(
            location_id,
            window,
            state_items,
            heartbeat_items,
        ):
            duration = (segment.end - segment.start).total_seconds()
            if duration <= 0 or segment.is_closed or segment.capacity <= 0:
                continue
            location_known_seconds += duration
            location_count_seconds += segment.count * duration
            location_capacity_seconds += segment.capacity * duration

        if location_known_seconds > 0:
            observed_count_total += location_count_seconds / location_known_seconds
            observed_capacity_total += (
                location_capacity_seconds / location_known_seconds
            )
            confirmed_capacity_seconds += location_capacity_seconds

    window_seconds = (window.end - window.start).total_seconds()
    observed_count = (
        round(observed_count_total) if observed_capacity_total > 0 else None
    )
    observed_capacity = round(observed_capacity_total)
    actual_coverage = (
        observed_capacity / expected_capacity if expected_capacity > 0 else 0.0
    )
    temporal_denominator = observed_capacity_total * window_seconds
    raw_temporal_coverage = (
        confirmed_capacity_seconds / temporal_denominator
        if temporal_denominator > 0
        else 0.0
    )
    temporal_coverage = min(1.0, max(0.0, raw_temporal_coverage))
    actual_count = (
        observed_count
        if observed_count is not None
        and expected_capacity > 0
        and _coverage_meets_threshold(actual_coverage, threshold)
        and _coverage_meets_threshold(temporal_coverage, threshold)
        else None
    )

    return ActualHourSummary(
        observed_count=observed_count,
        observed_capacity=observed_capacity,
        expected_capacity=expected_capacity,
        actual_coverage=actual_coverage,
        temporal_coverage=temporal_coverage,
        coverage_threshold=threshold,
        actual_count=actual_count,
    )


def build_chicago_hour_windows(date_key: str) -> list[HourWindow]:
    if not isinstance(date_key, str) or not DATE_KEY_PATTERN.fullmatch(date_key):
        raise ValueError("date key must use YYYY-MM-DD")
    try:
        local_date = date.fromisoformat(date_key)
        following_date = local_date + timedelta(days=1)
    except (OverflowError, ValueError) as exc:
        raise ValueError("date key must use YYYY-MM-DD") from exc
    if local_date.isoformat() != date_key:
        raise ValueError("date key must use YYYY-MM-DD")

    local_start = datetime.combine(local_date, time.min, tzinfo=CHICAGO)
    local_end = datetime.combine(following_date, time.min, tzinfo=CHICAGO)
    cursor = local_start.astimezone(UTC)
    end = local_end.astimezone(UTC)
    windows: list[HourWindow] = []
    while cursor < end:
        following = cursor + timedelta(hours=1)
        windows.append(HourWindow(cursor, following))
        cursor = following
    return windows


def _confirmed_state_segments(
    location_id: int,
    window: HourWindow,
    states: tuple[HistoryState, ...],
    heartbeats: tuple[IngestionHeartbeat, ...],
) -> list[_ConfirmedStateSegment]:
    timeline = _collapsed_location_states(location_id, states)
    if not timeline:
        return []

    seed_index: int | None = None
    active_indexes: list[int] = []
    for index, state in enumerate(timeline):
        if state.fetched_at < window.start:
            seed_index = index
        elif state.fetched_at < window.end:
            active_indexes.append(index)
        else:
            break
    if seed_index is not None:
        active_indexes.insert(0, seed_index)

    location_heartbeats = tuple(
        heartbeat.completed_at
        for heartbeat in heartbeats
        if location_id in heartbeat.observed_location_ids
    )
    segments: list[_ConfirmedStateSegment] = []
    for index in active_indexes:
        state = timeline[index]
        next_state_at = (
            timeline[index + 1].fetched_at
            if index + 1 < len(timeline)
            else None
        )
        interval_start = max(window.start, state.fetched_at)
        interval_end = window.end
        if next_state_at is not None:
            interval_end = min(interval_end, next_state_at)
        if interval_end <= interval_start:
            continue

        confirming_times = (
            completed_at
            for completed_at in location_heartbeats
            if completed_at >= state.fetched_at
            and (next_state_at is None or completed_at <= next_state_at)
        )
        latest_confirmation = max(confirming_times, default=None)
        if latest_confirmation is None:
            continue
        confirmed_end = min(interval_end, latest_confirmation)
        if confirmed_end <= interval_start:
            continue

        segments.append(
            _ConfirmedStateSegment(
                start=interval_start,
                end=confirmed_end,
                is_closed=state.is_closed,
                count=state.count,
                capacity=state.capacity,
            )
        )
    return segments


def _collapsed_location_states(
    location_id: int,
    states: tuple[HistoryState, ...],
) -> list[HistoryState]:
    by_timestamp: dict[datetime, HistoryState] = {}
    for state in states:
        if state.location_id != location_id:
            continue
        retained = by_timestamp.get(state.fetched_at)
        if retained is None or state.id > retained.id:
            by_timestamp[state.fetched_at] = state
    return sorted(by_timestamp.values(), key=lambda state: (state.fetched_at, state.id))


def _validated_requested_location_ids(
    location_ids: Sequence[int],
) -> tuple[int, ...]:
    try:
        items = tuple(location_ids)
    except TypeError as exc:
        raise ValueError(
            "location IDs must be non-negative integers; valid IDs are greater than zero"
        ) from exc
    if any(not _is_positive_integer(location_id) for location_id in items):
        raise ValueError(
            "location IDs must be non-negative integers; valid IDs are greater than zero"
        )
    if len(items) != len(set(items)):
        raise ValueError("location IDs must be unique")
    return items


def _validated_states(states: Sequence[HistoryState]) -> tuple[HistoryState, ...]:
    try:
        items = tuple(states)
    except TypeError as exc:
        raise ValueError("states must be a sequence of HistoryState values") from exc
    if any(not isinstance(state, HistoryState) for state in items):
        raise ValueError("states must be a sequence of HistoryState values")
    return items


def _validated_heartbeats(
    heartbeats: Sequence[IngestionHeartbeat],
) -> tuple[IngestionHeartbeat, ...]:
    try:
        items = tuple(heartbeats)
    except TypeError as exc:
        raise ValueError(
            "heartbeats must be a sequence of IngestionHeartbeat values"
        ) from exc
    if any(not isinstance(heartbeat, IngestionHeartbeat) for heartbeat in items):
        raise ValueError("heartbeats must be a sequence of IngestionHeartbeat values")
    return items


def _validated_coverage_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("coverage threshold must be a finite number between 0 and 1")
    if not 0.0 <= value <= 1.0 or not math.isfinite(value):
        raise ValueError("coverage threshold must be a finite number between 0 and 1")
    return float(value)


def _coverage_meets_threshold(coverage: float, threshold: float) -> bool:
    return coverage >= threshold or math.isclose(
        coverage,
        threshold,
        rel_tol=0.0,
        abs_tol=COVERAGE_COMPARISON_ABS_TOLERANCE,
    )


def _require_aware_utc(value: datetime, label: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
        or value.utcoffset() != timedelta(0)
    ):
        raise ValueError(f"{label} must be an aware UTC datetime")


def _require_positive_integer(value: int, label: str) -> None:
    if not _is_positive_integer(value):
        raise ValueError(f"{label} must be a positive integer")


def _require_non_negative_integer(value: int, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _is_positive_integer(value: object) -> bool:
    return type(value) is int and value > 0
