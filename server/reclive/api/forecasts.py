from __future__ import annotations

import sys as _import_sys
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pytz
from fastapi import APIRouter, Depends, HTTPException, Query
from server.reclive.actual_hours import (
    CHICAGO as ACTUAL_HOURS_CHICAGO_TZ,
    ActualHourSummary,
    HourWindow,
    build_chicago_hour_windows,
    calculate_actual_hour,
)
from server.reclive.occupancy_repository import ActualHourReadProtocol
from server.reclive.runtime import current_runtime
from server.reclive.api.dependencies import get_actual_hour_repository
from server.reclive import sections as _owner_sections

CHICAGO_TZ = pytz.timezone("America/Chicago")


def database_timezone():
    try:
        return pytz.timezone(current_runtime().settings.database.timezone)
    except Exception:
        return pytz.utc


router = APIRouter()


def load_forecast() -> Dict[str, Any]:
    if not os.path.exists(current_runtime().settings.forecast_json_path):
        raise HTTPException(status_code=503, detail="Forecast not generated yet")
    try:
        with open(
            current_runtime().settings.forecast_json_path, "r", encoding="utf-8"
        ) as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Forecast file corrupted") from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail="Failed to read forecast file"
        ) from exc


def generated_age_seconds(payload: Dict[str, Any]) -> Optional[int]:
    generated_at = payload.get("generatedAt")
    if not generated_at:
        return None
    try:
        ts = datetime.fromisoformat(str(generated_at))
        if ts.tzinfo is None:
            ts = CHICAGO_TZ.localize(ts)
        else:
            ts = ts.astimezone(CHICAGO_TZ)
        now = current_runtime().clock().astimezone(CHICAGO_TZ)
        return int((now - ts).total_seconds())
    except Exception:
        return None


def compact_hour_payload(hour: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in (
        "hour",
        "hourStart",
        "expectedCount",
        "expectedPct",
        "actualCount",
        "actualPct",
        "actualSampleCount",
        "actualCoverage",
        "spikeAdjusted",
    ):
        if key in hour:
            compact[key] = hour.get(key)
    return compact


def compact_window_payload(window: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in (
        "start",
        "end",
        "startHour",
        "endHour",
        "windowHours",
        "expectedTotal",
        "expectedAvg",
    ):
        if key in window:
            compact[key] = window.get(key)
    return compact


def compact_day_payload(day: Dict[str, Any]) -> Dict[str, Any]:
    categories: List[Dict[str, Any]] = []
    for category in (
        day.get("categories", []) if isinstance(day.get("categories"), list) else []
    ):
        if not isinstance(category, dict):
            continue
        hours_raw = category.get("hours", [])
        hours = [
            compact_hour_payload(hour)
            for hour in (hours_raw if isinstance(hours_raw, list) else [])
            if isinstance(hour, dict)
        ]
        categories.append(
            {
                "key": category.get("key"),
                "title": category.get("title"),
                "maxCapacity": category.get("maxCapacity"),
                "hours": hours,
            }
        )
    total_hours_raw = day.get("totalHours", [])
    total_hours = [
        compact_hour_payload(hour)
        for hour in (total_hours_raw if isinstance(total_hours_raw, list) else [])
        if isinstance(hour, dict)
    ]
    avoid_windows = [
        compact_window_payload(window)
        for window in (
            day.get("avoidWindows", [])
            if isinstance(day.get("avoidWindows"), list)
            else []
        )
        if isinstance(window, dict)
    ]
    best_windows = [
        compact_window_payload(window)
        for window in (
            day.get("bestWindows", [])
            if isinstance(day.get("bestWindows"), list)
            else []
        )
        if isinstance(window, dict)
    ]
    compact_day: Dict[str, Any] = {
        "dayName": day.get("dayName"),
        "date": day.get("date"),
        "categories": categories,
        "totalHours": total_hours,
        "avoidWindows": avoid_windows,
        "bestWindows": best_windows,
        "crowdBands": day.get("crowdBands", []),
    }
    return compact_day


def compact_facility_payload(facility: Dict[str, Any]) -> Dict[str, Any]:
    weekly = facility.get("weeklyForecast", [])
    compact_weekly = [
        compact_day_payload(day)
        for day in (weekly if isinstance(weekly, list) else [])
        if isinstance(day, dict)
    ]
    return {
        "facilityId": facility.get("facilityId"),
        "facilityName": facility.get("facilityName"),
        "occupancyThresholds": facility.get("occupancyThresholds"),
        "sectionOccupancyThresholds": facility.get("sectionOccupancyThresholds"),
        "locationOccupancyThresholds": facility.get("locationOccupancyThresholds"),
        "weeklyForecast": compact_weekly,
    }


def parse_chicago_date_key(value: str) -> Tuple[datetime, datetime]:
    text = str(value or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="date is required (YYYY-MM-DD)")
    try:
        start_naive = datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="date must be in YYYY-MM-DD format"
        ) from exc
    start_local = CHICAGO_TZ.localize(start_naive)
    end_local = start_local + timedelta(days=1)
    return (start_local, end_local)


def to_chicago_datetime(raw_value: Any) -> Optional[datetime]:
    if raw_value is None:
        return None
    if isinstance(raw_value, datetime):
        dt = raw_value
    else:
        text = str(raw_value).strip()
        if not text:
            return None
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        dt = None
        try:
            dt = datetime.fromisoformat(normalized)
        except Exception:
            for fmt in (
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
            ):
                try:
                    dt = datetime.strptime(normalized, fmt)
                    break
                except Exception:
                    continue
        if dt is None:
            return None
    if dt.tzinfo is None:
        try:
            dt = database_timezone().localize(dt)
        except Exception:
            dt = pytz.utc.localize(dt)
    else:
        dt = dt.astimezone(database_timezone())
    return dt.astimezone(CHICAGO_TZ)


@router.get("/api/forecast")
def forecast() -> Dict[str, Any]:
    return load_forecast()


@router.get("/api/forecast/facilities")
def facilities() -> List[Dict[str, Any]]:
    payload = load_forecast()
    items = []
    for facility in payload.get("facilities", []):
        items.append(
            {
                "facilityId": facility.get("facilityId"),
                "facilityName": facility.get("facilityName"),
                "days": len(facility.get("weeklyForecast", [])),
            }
        )
    return items


@router.get("/api/forecast/facilities/{facility_id}")
def facility_forecast(
    facility_id: int,
    date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    compact: bool = Query(False, description="Return only app-required fields"),
) -> Dict[str, Any]:
    payload = load_forecast()
    facilities_data = payload.get("facilities", [])
    facility = next(
        (row for row in facilities_data if row.get("facilityId") == facility_id), None
    )
    if not facility:
        raise HTTPException(status_code=404, detail="Facility not found")
    forecast_start_hour = payload.get("forecastDayStartHour")
    forecast_end_hour = payload.get("forecastDayEndHour")
    if not date:
        facility_payload = compact_facility_payload(facility) if compact else facility
        return {
            **facility_payload,
            "forecastDayStartHour": forecast_start_hour,
            "forecastDayEndHour": forecast_end_hour,
        }
    day = next(
        (row for row in facility.get("weeklyForecast", []) if row.get("date") == date),
        None,
    )
    if not day:
        raise HTTPException(status_code=404, detail="Date not found for facility")
    return {
        "facilityId": facility.get("facilityId"),
        "facilityName": facility.get("facilityName"),
        "forecastDayStartHour": forecast_start_hour,
        "forecastDayEndHour": forecast_end_hour,
        "occupancyThresholds": facility.get("occupancyThresholds"),
        "sectionOccupancyThresholds": facility.get("sectionOccupancyThresholds"),
        "locationOccupancyThresholds": facility.get("locationOccupancyThresholds"),
        "day": compact_day_payload(day) if compact else day,
    }


def serialize_actual_hour(
    window: HourWindow, summary: ActualHourSummary
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "hourStart": window.start.astimezone(ACTUAL_HOURS_CHICAGO_TZ).isoformat(),
        "observedCount": summary.observed_count,
        "observedCapacity": summary.observed_capacity,
        "expectedCapacity": summary.expected_capacity,
        "actualCoverage": round(min(1.0, max(0.0, summary.actual_coverage)), 4),
        "temporalCoverage": round(min(1.0, max(0.0, summary.temporal_coverage)), 4),
        "coverageThreshold": round(summary.coverage_threshold, 4),
        "actualCount": summary.actual_count,
    }
    if summary.actual_count is not None and summary.expected_capacity > 0:
        payload["actualPct"] = round(
            min(1.0, max(0.0, summary.actual_count / float(summary.expected_capacity))),
            4,
        )
    return payload


@router.get("/api/forecast/facilities/{facility_id}/actual-hours")
def facility_actual_hours(
    facility_id: int,
    date: str = Query(..., description="YYYY-MM-DD"),
    repository: ActualHourReadProtocol = Depends(get_actual_hour_repository),
) -> Dict[str, Any]:
    try:
        hour_windows = build_chicago_hour_windows(date)
    except (ValueError, OverflowError) as exc:
        raise HTTPException(
            status_code=400, detail="date must be in YYYY-MM-DD format"
        ) from exc
    payload = load_forecast()
    facilities_data = payload.get("facilities", [])
    facility = next(
        (row for row in facilities_data if row.get("facilityId") == facility_id), None
    )
    if not facility:
        raise HTTPException(status_code=404, detail="Facility not found")
    day = next(
        (row for row in facility.get("weeklyForecast", []) if row.get("date") == date),
        None,
    )
    if not day:
        raise HTTPException(status_code=404, detail="Date not found for facility")
    categories_raw = day.get("categories", [])
    if not isinstance(categories_raw, list):
        categories_raw = []
    section_map = current_runtime().section_ids.get(facility_id, {})
    overall_location_ids: List[int] = []
    overall_seen: set[int] = set()
    for raw_location_id in (
        section_map.get("overall", []) if isinstance(section_map, dict) else []
    ):
        try:
            location_id = int(raw_location_id)
        except (TypeError, ValueError):
            continue
        if location_id > 0 and location_id not in overall_seen:
            overall_seen.add(location_id)
            overall_location_ids.append(location_id)
    category_specs: List[Dict[str, Any]] = []
    all_location_ids: List[int] = list(overall_location_ids)
    seen_location_ids = set(overall_location_ids)
    for category in categories_raw:
        if not isinstance(category, dict):
            continue
        location_ids: List[int] = []
        category_seen: set[int] = set()
        for location_id in _owner_sections.category_location_ids_for_forecast(
            facility_id, category
        ):
            if location_id <= 0 or location_id in category_seen:
                continue
            category_seen.add(location_id)
            location_ids.append(location_id)
        if not location_ids:
            continue
        for location_id in location_ids:
            if location_id in seen_location_ids:
                continue
            seen_location_ids.add(location_id)
            all_location_ids.append(location_id)
        category_max = _owner_sections._int_or_default(category.get("maxCapacity"), 0)
        if category_max <= 0:
            category_max = sum(
                (
                    max(0, int(current_runtime().capacities.get(location_id, 0)))
                    for location_id in location_ids
                )
            )
        category_specs.append(
            {
                "key": category.get("key"),
                "title": category.get("title"),
                "locationIds": location_ids,
                "maxCapacity": max(0, category_max),
            }
        )
    if not all_location_ids:
        return {
            "facilityId": facility_id,
            "date": date,
            "categories": [],
            "totalHours": [],
        }
    range_start = hour_windows[0].start
    range_end = hour_windows[-1].end
    try:
        states, heartbeats = repository.load_actual_hour_inputs(
            all_location_ids, range_start, range_end
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Actual-hour DB query failed"
        ) from exc
    categories_payload: List[Dict[str, Any]] = []
    for spec in category_specs:
        location_ids = spec["locationIds"]
        category_max = int(spec["maxCapacity"])
        hours_payload = [
            serialize_actual_hour(
                window,
                calculate_actual_hour(
                    location_ids,
                    category_max,
                    window,
                    states,
                    heartbeats,
                    current_runtime().settings.actual_hour_min_coverage,
                ),
            )
            for window in hour_windows
        ]
        categories_payload.append(
            {"key": spec["key"], "title": spec["title"], "hours": hours_payload}
        )
    total_hours_payload: List[Dict[str, Any]] = []
    facility_max_capacity = sum(
        (
            max(0, int(current_runtime().capacities.get(location_id, 0)))
            for location_id in all_location_ids
        )
    )
    for window in hour_windows:
        total_hours_payload.append(
            serialize_actual_hour(
                window,
                calculate_actual_hour(
                    all_location_ids,
                    facility_max_capacity,
                    window,
                    states,
                    heartbeats,
                    current_runtime().settings.actual_hour_min_coverage,
                ),
            )
        )
    return {
        "facilityId": facility_id,
        "date": date,
        "categories": categories_payload,
        "totalHours": total_hours_payload,
    }


_import_sys.modules.setdefault(
    "server.reclive.api.forecasts", _import_sys.modules[__name__]
)
_import_sys.modules.setdefault("reclive.api.forecasts", _import_sys.modules[__name__])
