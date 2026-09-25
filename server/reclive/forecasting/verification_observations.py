"""Read qualified attendance for completed hours without bootstrapping the API.

The current forecast's rolling date range does not limit this reader. Every
returned row is an elapsed Chicago hour; excluded rows retain a reason instead
of turning an unavailable observation into a zero.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from server.reclive import facility_schedule, sections
from server.reclive.actual_hours import (
    CHICAGO,
    HourWindow,
    build_chicago_hour_windows,
    calculate_actual_hour,
)
from server.reclive.occupancy_repository import SnapshotRepository
from server.reclive.runtime import Runtime, runtime_scope
from server.reclive.settings import Settings


_OFFICIAL_EVENT_TITLES = {
    "thanksgiving weekend",
    "final exam week",
    "commencement weekend",
    "adjusted hours",
    "independence day",
    "nick",
    "bakke",
}
# Same area exclusions as the forecaster, without importing its model/config
# module (which reads environment and training configuration at import time).
_AREA_TITLE_WORDS = (
    "ice rink",
    "sub zero",
    "pool",
    "court",
    "track",
    "climbing",
    "esports",
    "simulator",
    "fitness",
    "room",
)


def _official_building_sections(raw_sections: list[dict]) -> list[dict]:
    """Adapt published RecWell headings to the strict pure schedule resolver.

    Date-only headings retain their full scope; only observed official event
    headings become unscoped building sections. Their explicit row dates still
    determine applicability. Unknown headings are never promoted to building
    hours, and unrelated area closures cannot close the entire facility.
    """
    normalized = []
    for section in raw_sections:
        title = section["title"]
        normalized_title = facility_schedule._normalize(title)
        if not normalized_title or any(
            word in normalized_title for word in _AREA_TITLE_WORDS
        ):
            continue
        if facility_schedule._section_kind(title) is not None:
            adapted_title = title
        elif facility_schedule._EXACT_DATE_RANGE.fullmatch(normalized_title):
            adapted_title = f"Building Hours: {title}"
        elif normalized_title in _OFFICIAL_EVENT_TITLES:
            adapted_title = "Building Hours"
        else:
            continue
        rows = []
        for row in section["rows"]:
            label = row["label"]
            date_label = re.sub(r"\s*\([^()]*\)\s*$", "", label)
            # e.g. December 13 (Commencement). Do not strip annotations from
            # weekday rules or promote text that is not itself a valid date.
            if (
                date_label != label
                and facility_schedule.parse_schedule_date_range(date_label, 2000)
                is not None
            ):
                label = date_label
            rows.append({**row, "label": label})
        normalized.append({**section, "title": adapted_title, "rows": rows})
    return normalized


def _schedule_sections(settings: Settings, now: datetime) -> dict[int, list[dict]]:
    try:
        payload = facility_schedule.validate_schedule_payload(
            json.loads(
                Path(settings.facility_hours_json_path).read_text(encoding="utf-8")
            ),
            now=now,
        )
    except (OSError, ValueError, TypeError):
        return {}
    generated_at = facility_schedule.parse_utc_timestamp(payload.get("generatedAt"))
    if (
        generated_at is None
        or not 0
        <= (now - generated_at).total_seconds()
        <= settings.schedule_stale_after_seconds
    ):
        return {}
    result = {}
    for facility in payload["facilities"]:
        successful_at = facility_schedule.parse_utc_timestamp(
            facility.get("lastSuccessfulAt")
        )
        if (
            facility.get("status") == "ok"
            and facility.get("stale") is False
            and successful_at is not None
            and 0
            <= (now - successful_at).total_seconds()
            <= settings.schedule_stale_after_seconds
        ):
            result[facility["facilityId"]] = _official_building_sections(
                facility["sections"]
            )
    return result


def _hour_status(schedule: list[dict] | None, window: HourWindow) -> str:
    if not schedule:
        return "schedule_unavailable"
    # The official schedule parser has minute precision. Inspect the complete
    # half-open hour, including partial opening/closing and overnight windows.
    states = {
        facility_schedule.get_facility_schedule_open_state(
            schedule, window.start + timedelta(minutes=minute)
        )
        for minute in range(60)
    }
    if None in states:
        return "schedule_unavailable"
    if states == {True}:
        return "ready"
    return "closed" if states == {False} else "partial_open"


def _row(
    facility_id: int, window: HourWindow, capacity: int, threshold: float, status: str
) -> dict:
    return {
        "facilityId": facility_id,
        "hourStart": window.start.astimezone(CHICAGO).isoformat(),
        "actualCount": None,
        "actualPct": None,
        "observedCount": None,
        "observedCapacity": 0,
        "expectedCapacity": capacity,
        "actualCoverage": 0.0,
        "temporalCoverage": 0.0,
        "coverageThreshold": threshold,
        "observationStatus": status,
        "skipReason": None if status == "ready" else status,
    }


def collect_actual_hours(
    settings: Settings, dates: list[str], now: datetime
) -> list[dict]:
    """Return actuals and exclusion reasons using only a read-only DB transaction.

    ``actualCount`` is populated only for fully open, completed hours with both
    capacity and temporal coverage at or above the existing configured minimum.
    ``actualPct`` is a 0..1 fraction, matching the public actual-hours endpoint.
    The DB is never migrated, committed to, or initialized by this function.
    """
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be an aware datetime")
    now = now.astimezone(timezone.utc)
    day_windows = {
        key: [window for window in build_chicago_hour_windows(key) if window.end <= now]
        for key in sorted(set(dates))
    }
    day_windows = {key: windows for key, windows in day_windows.items() if windows}
    if not day_windows:
        return []

    runtime = Runtime(settings=settings, clock=lambda: now)
    with runtime_scope(runtime):
        sections.ensure_runtime_facility_configuration()
    schedules = _schedule_sections(settings, now)
    facility_specs = {}
    for facility_id in sorted(runtime.section_ids):
        ids = sorted(
            {
                location_id
                for location_id in runtime.section_ids[facility_id].get("overall", [])
                if type(location_id) is int and location_id > 0
            }
        )
        capacity = sum(
            max(0, int(runtime.capacities.get(location_id, 0))) for location_id in ids
        )
        facility_specs[facility_id] = (ids, capacity)

    output = []
    pending = {}
    for date_key, windows in day_windows.items():
        for facility_id, (ids, capacity) in facility_specs.items():
            for window in windows:
                status = _hour_status(schedules.get(facility_id), window)
                row = _row(
                    facility_id,
                    window,
                    capacity,
                    settings.actual_hour_min_coverage,
                    status,
                )
                output.append(row)
                if status == "ready":
                    if not ids or capacity <= 0:
                        row.update(
                            observationStatus="missing_data",
                            skipReason="missing_capacity_configuration",
                        )
                    else:
                        pending.setdefault(date_key, []).append(
                            (row, window, ids, capacity)
                        )
    if not pending:
        return output

    connection = runtime.connect(autocommit=False)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION time_zone = '+00:00'")
            cursor.execute("START TRANSACTION READ ONLY")
        repository = SnapshotRepository(connection)
        for date_key, candidates in pending.items():
            windows = day_windows[date_key]
            all_ids = sorted(
                {location_id for _, _, ids, _ in candidates for location_id in ids}
            )
            # Include a following confirmation when already available, without
            # querying beyond this run's cutoff or retaining an unbounded range.
            range_end = min(now, windows[-1].end + timedelta(hours=1))
            states, heartbeats = repository.load_actual_hour_inputs(
                all_ids, windows[0].start, range_end
            )
            for row, window, ids, capacity in candidates:
                summary = calculate_actual_hour(
                    ids,
                    capacity,
                    window,
                    states,
                    heartbeats,
                    settings.actual_hour_min_coverage,
                )
                status = (
                    "ready"
                    if summary.actual_count is not None
                    else "missing_data"
                    if summary.observed_count is None
                    else "insufficient_coverage"
                )
                row.update(
                    actualCount=summary.actual_count,
                    actualPct=(
                        round(min(1.0, max(0.0, summary.actual_count / capacity)), 4)
                        if summary.actual_count is not None
                        else None
                    ),
                    observedCount=summary.observed_count,
                    observedCapacity=summary.observed_capacity,
                    actualCoverage=min(1.0, max(0.0, summary.actual_coverage)),
                    temporalCoverage=min(1.0, max(0.0, summary.temporal_coverage)),
                    observationStatus=status,
                    skipReason=None if status == "ready" else status,
                )
    finally:
        try:
            connection.rollback()
        finally:
            connection.close()
    return output
