import asyncio
import json
import math
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence, Tuple

import pytz
import pymysql
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from pywebpush import WebPushException, webpush
from env_loader import load_project_dotenv

try:
    from forecast_shared import normalize_section_key
    from facility_capacities import load_facility_capacities
    from reclive.actual_hours import (
        CHICAGO as ACTUAL_HOURS_CHICAGO_TZ,
        ActualHourSummary,
        HistoryState,
        HourWindow,
        IngestionHeartbeat,
        build_chicago_hour_windows,
        calculate_actual_hour,
    )
    from reclive.ingestion import safe_close
    from reclive.occupancy_repository import (
        ActualHourReadProtocol,
        RepositoryFactory,
        SnapshotReadProtocol,
        SnapshotRepository,
        SnapshotRow,
    )
except ImportError:
    from server.forecast_shared import normalize_section_key
    from server.facility_capacities import load_facility_capacities
    from server.reclive.actual_hours import (
        CHICAGO as ACTUAL_HOURS_CHICAGO_TZ,
        ActualHourSummary,
        HistoryState,
        HourWindow,
        IngestionHeartbeat,
        build_chicago_hour_windows,
        calculate_actual_hour,
    )
    from server.reclive.ingestion import safe_close
    from server.reclive.occupancy_repository import (
        ActualHourReadProtocol,
        RepositoryFactory,
        SnapshotReadProtocol,
        SnapshotRepository,
        SnapshotRow,
    )

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
load_project_dotenv()

def _read_env(name: str, aliases: Sequence[str] = ()) -> Optional[str]:
    for key in (name, *aliases):
        value = os.getenv(key)
        if value is None:
            continue
        normalized = value.strip()
        if normalized:
            return normalized
    return None


def require_env(name: str, aliases: Sequence[str] = ()) -> str:
    value = _read_env(name, aliases=aliases)
    if value is None:
        alias_text = f" (aliases: {', '.join(aliases)})" if aliases else ""
        raise RuntimeError(f"Missing required env var: {name}{alias_text}")
    return value


def env_with_default(name: str, default: str, aliases: Sequence[str] = ()) -> str:
    value = _read_env(name, aliases=aliases)
    if value is None:
        return default
    return value


def int_with_default(name: str, default: int, aliases: Sequence[str] = ()) -> int:
    raw = _read_env(name, aliases=aliases)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer for env var {name}: {raw}") from exc


def bool_with_default(name: str, default: bool, aliases: Sequence[str] = ()) -> bool:
    raw = _read_env(name, aliases=aliases)
    if raw is None:
        return default
    value = raw.lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Invalid boolean for env var {name}: {raw}")


def path_with_default(name: str, default: str, aliases: Sequence[str] = ()) -> str:
    raw = env_with_default(name, default, aliases=aliases)
    return resolve_path(raw)


def resolve_path(raw: str) -> str:
    if os.path.isabs(raw):
        return raw

    # Prefer script-local relative paths for deployments that keep .env + scripts together.
    script_candidate = os.path.abspath(os.path.join(SCRIPT_DIR, raw))
    return script_candidate


FORECAST_JSON_PATH = path_with_default(
    "FORECAST_JSON_PATH",
    os.path.join(os.path.dirname(__file__), "forecast.json"),
)

API_HOST = env_with_default("FORECAST_API_HOST", "0.0.0.0")
API_PORT = int_with_default("FORECAST_API_PORT", 8000)
CHICAGO_TZ = pytz.timezone("America/Chicago")
DB_TIMEZONE_NAME = env_with_default("GYM_DB_TIMEZONE", "UTC")
try:
    DB_TZ = pytz.timezone(DB_TIMEZONE_NAME)
except Exception:
    DB_TZ = pytz.utc
ACTUAL_HOUR_MIN_COVERAGE = float(env_with_default("ACTUAL_HOUR_MIN_COVERAGE", "0.75"))

FACILITY_SECTION_CONFIG_PATH = path_with_default(
    "FACILITY_SECTION_CONFIG_PATH",
    os.path.join(SCRIPT_DIR, "facility_sections.json"),
)
FACILITY_HOURS_JSON_PATH = path_with_default(
    "FACILITY_HOURS_JSON_PATH",
    os.path.join(SCRIPT_DIR, "facility_hours.json"),
)
PUSH_RULES_TABLE = env_with_default("PUSH_RULES_TABLE", "push_rules")
EVALUATOR_INTERVAL_SECONDS = int_with_default("PUSH_EVALUATOR_INTERVAL_SECONDS", 180)
DEFAULT_NOTIFICATION_URL = env_with_default("PUSH_DEFAULT_NOTIFICATION_URL", "/")
PUSH_EVALUATOR_DB_LOCK_NAME = env_with_default("PUSH_EVALUATOR_DB_LOCK_NAME", "reclive_push_eval")

PUSH_VAPID_PUBLIC_KEY = env_with_default("PUSH_VAPID_PUBLIC_KEY", "")
PUSH_VAPID_PRIVATE_KEY = env_with_default("PUSH_VAPID_PRIVATE_KEY", "")
PUSH_VAPID_SUBJECT = env_with_default("PUSH_VAPID_SUBJECT", "")
PUSH_ADMIN_TOKEN = env_with_default("PUSH_ADMIN_TOKEN", "")

STORE_LOCK = threading.Lock()
EVALUATOR_TASK: Optional[asyncio.Task] = None


MAX_CAP = load_facility_capacities()

def load_facility_sections() -> Tuple[Dict[int, str], Dict[int, Dict[str, List[int]]]]:
    try:
        with open(FACILITY_SECTION_CONFIG_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to read facility section config at {FACILITY_SECTION_CONFIG_PATH}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Facility section config is not valid JSON: {FACILITY_SECTION_CONFIG_PATH}"
        ) from exc

    facilities_raw = payload.get("facilities")
    if not isinstance(facilities_raw, dict):
        raise RuntimeError("Facility section config must contain an object at key 'facilities'")

    facility_names: Dict[int, str] = {}
    section_ids: Dict[int, Dict[str, List[int]]] = {}

    for facility_id_text, facility_payload in facilities_raw.items():
        if not isinstance(facility_payload, dict):
            continue

        try:
            facility_id = int(facility_id_text)
        except (TypeError, ValueError):
            continue

        short_name = str(facility_payload.get("shortName", "")).strip()
        if not short_name:
            short_name = str(facility_payload.get("facilityName", "")).strip()
        if not short_name:
            short_name = f"Facility {facility_id}"
        facility_names[facility_id] = short_name

        sections_raw = facility_payload.get("sections")
        if not isinstance(sections_raw, list):
            sections_raw = []

        by_section: Dict[str, List[int]] = {}
        overall: List[int] = []
        overall_seen = set()

        for section in sections_raw:
            if not isinstance(section, dict):
                continue

            key = normalize_section_key(str(section.get("key", "")))
            ids_raw = section.get("ids")
            if not key or not isinstance(ids_raw, list):
                continue

            location_ids: List[int] = []
            for location_id_raw in ids_raw:
                try:
                    location_id = int(location_id_raw)
                except (TypeError, ValueError):
                    continue
                location_ids.append(location_id)
                if location_id not in overall_seen:
                    overall_seen.add(location_id)
                    overall.append(location_id)

            if location_ids:
                by_section[key] = location_ids

        by_section["overall"] = overall
        section_ids[facility_id] = by_section

    if not facility_names or not section_ids:
        raise RuntimeError("Facility section config did not produce any facilities")
    return facility_names, section_ids


FACILITY_NAMES, SECTION_IDS = load_facility_sections()


def parse_allowed_origins() -> List[str]:
    raw = env_with_default("FORECAST_API_ALLOW_ORIGINS", "*")
    parsed = [item.strip() for item in raw.split(",") if item.strip()]
    if not parsed:
        return ["*"]
    return parsed


def load_forecast() -> Dict[str, Any]:
    if not os.path.exists(FORECAST_JSON_PATH):
        raise HTTPException(status_code=503, detail="Forecast not generated yet")

    try:
        with open(FORECAST_JSON_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Forecast file corrupted") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to read forecast file") from exc


def load_facility_hours() -> Dict[str, Any]:
    if not os.path.exists(FACILITY_HOURS_JSON_PATH):
        raise HTTPException(status_code=503, detail="Facility hours not generated yet")

    try:
        with open(FACILITY_HOURS_JSON_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Facility hours file corrupted") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to read facility hours file") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Facility hours payload shape is invalid")

    facilities = payload.get("facilities", [])
    if not isinstance(facilities, list):
        raise HTTPException(status_code=500, detail="Facility hours payload facilities shape is invalid")

    return payload


def _parse_facility_id(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_facility_hours_entry(payload: Dict[str, Any], facility_id: int) -> Dict[str, Any]:
    facilities = payload.get("facilities", [])
    for item in facilities:
        if not isinstance(item, dict):
            continue
        row_facility_id = _parse_facility_id(item.get("facilityId"))
        if row_facility_id == facility_id:
            return item
    raise HTTPException(status_code=404, detail="Facility schedule not found")


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
        now = datetime.now(CHICAGO_TZ)
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
    for key in ("start", "end", "startHour", "endHour", "windowHours", "expectedTotal", "expectedAvg"):
        if key in window:
            compact[key] = window.get(key)
    return compact


def compact_day_payload(day: Dict[str, Any]) -> Dict[str, Any]:
    categories: List[Dict[str, Any]] = []
    for category in day.get("categories", []) if isinstance(day.get("categories"), list) else []:
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
        for window in (day.get("avoidWindows", []) if isinstance(day.get("avoidWindows"), list) else [])
        if isinstance(window, dict)
    ]
    best_windows = [
        compact_window_payload(window)
        for window in (day.get("bestWindows", []) if isinstance(day.get("bestWindows"), list) else [])
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
        raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format") from exc
    start_local = CHICAGO_TZ.localize(start_naive)
    end_local = start_local + timedelta(days=1)
    return start_local, end_local


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
            dt = DB_TZ.localize(dt)
        except Exception:
            dt = pytz.utc.localize(dt)
    else:
        dt = dt.astimezone(DB_TZ)
    return dt.astimezone(CHICAGO_TZ)


def category_location_ids_for_forecast(
    facility_id: int,
    category: Dict[str, Any],
) -> List[int]:
    section_map = SECTION_IDS.get(facility_id, {})
    key_raw = _str_or_none(category.get("key"))
    title_raw = _str_or_none(category.get("title"))

    candidates: List[str] = []
    if key_raw:
        candidates.append(normalize_section_key(key_raw))
        candidates.append(normalize_section_key(key_raw.replace("_", " ")))
    if title_raw:
        candidates.append(normalize_section_key(title_raw))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        ids = section_map.get(candidate)
        if not isinstance(ids, list):
            continue
        output: List[int] = []
        for raw_id in ids:
            try:
                output.append(int(raw_id))
            except (TypeError, ValueError):
                continue
        if output:
            return output

    return []

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global EVALUATOR_TASK
    if evaluator_enabled() and (EVALUATOR_TASK is None or EVALUATOR_TASK.done()):
        EVALUATOR_TASK = asyncio.create_task(evaluator_loop())
    try:
        yield
    finally:
        if EVALUATOR_TASK is not None:
            EVALUATOR_TASK.cancel()
            try:
                await EVALUATOR_TASK
            except asyncio.CancelledError:
                pass
            EVALUATOR_TASK = None


app = FastAPI(title="RecLive Forecast API", version="1.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_allowed_origins(),
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    payload = load_forecast()
    return {
        "status": "ok",
        "generatedAt": payload.get("generatedAt"),
        "generatedAgeSeconds": generated_age_seconds(payload),
        "facilities": len(payload.get("facilities", [])),
        "modelStatus": payload.get("modelInfo", {}).get("status"),
    }


@app.get("/api/forecast")
def forecast() -> Dict[str, Any]:
    return load_forecast()


@app.get("/api/forecast/facilities")
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


@app.get("/api/forecast/facilities/{facility_id}")
def facility_forecast(
    facility_id: int,
    date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    compact: bool = Query(False, description="Return only app-required fields"),
) -> Dict[str, Any]:
    payload = load_forecast()
    facilities_data = payload.get("facilities", [])
    facility = next((row for row in facilities_data if row.get("facilityId") == facility_id), None)
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

    day = next((row for row in facility.get("weeklyForecast", []) if row.get("date") == date), None)
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


class OwnedActualHourRepository:
    def load_actual_hour_inputs(
        self,
        location_ids: Sequence[int],
        range_start: datetime,
        range_end: datetime,
    ) -> tuple[list[HistoryState], list[IngestionHeartbeat]]:
        connection = None
        try:
            connection = open_db_connection(autocommit=False)
            repository = SnapshotRepository(connection)
            return repository.load_actual_hour_inputs(
                location_ids,
                range_start,
                range_end,
            )
        finally:
            safe_close(connection)


def get_actual_hour_repository() -> ActualHourReadProtocol:
    return OwnedActualHourRepository()


def serialize_actual_hour(
    window: HourWindow,
    summary: ActualHourSummary,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "hourStart": window.start.astimezone(
            ACTUAL_HOURS_CHICAGO_TZ
        ).isoformat(),
        "observedCount": summary.observed_count,
        "observedCapacity": summary.observed_capacity,
        "expectedCapacity": summary.expected_capacity,
        "actualCoverage": round(
            min(1.0, max(0.0, summary.actual_coverage)), 4
        ),
        "temporalCoverage": round(
            min(1.0, max(0.0, summary.temporal_coverage)), 4
        ),
        "coverageThreshold": round(summary.coverage_threshold, 4),
        "actualCount": summary.actual_count,
    }
    if summary.actual_count is not None and summary.expected_capacity > 0:
        payload["actualPct"] = round(
            min(
                1.0,
                max(
                    0.0,
                    summary.actual_count / float(summary.expected_capacity),
                ),
            ),
            4,
        )
    return payload


@app.get("/api/forecast/facilities/{facility_id}/actual-hours")
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
    facility = next((row for row in facilities_data if row.get("facilityId") == facility_id), None)
    if not facility:
        raise HTTPException(status_code=404, detail="Facility not found")

    day = next((row for row in facility.get("weeklyForecast", []) if row.get("date") == date), None)
    if not day:
        raise HTTPException(status_code=404, detail="Date not found for facility")

    categories_raw = day.get("categories", [])
    if not isinstance(categories_raw, list):
        categories_raw = []

    section_map = SECTION_IDS.get(facility_id, {})
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
        for location_id in category_location_ids_for_forecast(
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

        category_max = _int_or_default(category.get("maxCapacity"), 0)
        if category_max <= 0:
            category_max = sum(
                max(0, int(MAX_CAP.get(location_id, 0)))
                for location_id in location_ids
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
            all_location_ids,
            range_start,
            range_end,
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
                    ACTUAL_HOUR_MIN_COVERAGE,
                ),
            )
            for window in hour_windows
        ]

        categories_payload.append(
            {
                "key": spec["key"],
                "title": spec["title"],
                "hours": hours_payload,
            }
        )

    total_hours_payload: List[Dict[str, Any]] = []
    facility_max_capacity = sum(
        max(0, int(MAX_CAP.get(location_id, 0)))
        for location_id in all_location_ids
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
                    ACTUAL_HOUR_MIN_COVERAGE,
                ),
            )
        )

    return {
        "facilityId": facility_id,
        "date": date,
        "categories": categories_payload,
        "totalHours": total_hours_payload,
    }


def get_snapshot_repository() -> Iterator[SnapshotRepository]:
    try:
        connection = open_db_connection(autocommit=False)
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Live occupancy DB is unavailable"
        ) from exc
    try:
        yield SnapshotRepository(connection)
    finally:
        safe_close(connection)


@app.get("/api/live-counts")
def live_counts(
    repository: SnapshotRepository = Depends(get_snapshot_repository),
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    try:
        snapshot = repository.fetch_live_snapshot(now)
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Failed to query live occupancy snapshot"
        ) from exc

    if not snapshot.rows:
        raise HTTPException(status_code=503, detail="Live occupancy snapshot is empty")

    last_successful_fetch_at = snapshot.last_successful_fetch_at
    if last_successful_fetch_at is None:
        ingestion = {
            "lastSuccessfulFetchAt": None,
            "ageSeconds": None,
            "status": "unavailable",
        }
    else:
        elapsed_seconds = max(
            0.0, (now - last_successful_fetch_at).total_seconds()
        )
        age_seconds = int(elapsed_seconds)
        ingestion = {
            "lastSuccessfulFetchAt": utc_iso(last_successful_fetch_at),
            "ageSeconds": age_seconds,
            "status": "healthy" if elapsed_seconds <= 600 else "stale",
        }
    try:
        rows = [
            {
                "LocationId": row.location_id,
                "IsClosed": row.is_closed,
                "LastCount": row.current_capacity,
                "LastUpdatedDateAndTime": (
                    utc_iso(row.source_updated_at)
                    if row.source_updated_at is not None
                    else None
                ),
                "FetchedAt": utc_iso(row.fetched_at),
            }
            for row in snapshot.rows
        ]
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=503, detail="Failed to query live occupancy snapshot"
        ) from exc
    return {"ingestion": ingestion, "rows": rows}


def utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


@app.get("/api/facility-hours")
def facility_hours() -> Dict[str, Any]:
    return load_facility_hours()


@app.get("/api/facility-hours/facilities")
def facility_hours_facilities() -> List[Dict[str, Any]]:
    payload = load_facility_hours()
    facilities = payload.get("facilities", [])
    items: List[Dict[str, Any]] = []
    for item in facilities:
        if not isinstance(item, dict):
            continue
        sections = item.get("sections", [])
        items.append(
            {
                "facilityId": _parse_facility_id(item.get("facilityId")),
                "facilityName": str(item.get("facilityName", "")).strip(),
                "status": str(item.get("status", "")).strip(),
                "sections": len(sections) if isinstance(sections, list) else 0,
            }
        )
    return items


@app.get("/api/facility-hours/facilities/{facility_id}")
def facility_hours_for_facility(facility_id: int) -> Dict[str, Any]:
    payload = load_facility_hours()
    facility = get_facility_hours_entry(payload, facility_id)
    return {
        "generatedAt": payload.get("generatedAt"),
        "sourceSite": payload.get("sourceSite"),
        **facility,
    }


def evaluator_enabled() -> bool:
    return bool_with_default("PUSH_EVALUATOR_ENABLED", True) and push_vapid_configured()


def push_vapid_configured() -> bool:
    return bool(PUSH_VAPID_PUBLIC_KEY and PUSH_VAPID_PRIVATE_KEY and PUSH_VAPID_SUBJECT)


def push_admin_configured() -> bool:
    return bool(PUSH_ADMIN_TOKEN)


def require_admin_token(
    x_reclive_admin_token: Optional[str] = Header(default=None, alias="X-RecLive-Admin-Token"),
) -> None:
    if not push_admin_configured():
        raise HTTPException(status_code=503, detail="Admin token is not configured")
    if x_reclive_admin_token != PUSH_ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Admin token is required")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_sql_identifier(value: str, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"{name} must not be empty")
    if not (text[0].isalpha() or text[0] == "_"):
        raise RuntimeError(f"{name} must start with a letter or underscore")
    for ch in text:
        if not (ch.isalnum() or ch == "_"):
            raise RuntimeError(f"{name} contains invalid characters: {text}")
    return text


def push_rules_table_name() -> str:
    return safe_sql_identifier(PUSH_RULES_TABLE, "PUSH_RULES_TABLE")


def open_db_connection(*, autocommit: bool = True) -> Any:
    return pymysql.connect(
        host=require_env("GYM_DB_HOST"),
        port=int(require_env("GYM_DB_PORT")),
        user=require_env("GYM_DB_USER"),
        password=require_env("GYM_DB_PASSWORD"),
        database=require_env("GYM_DB_NAME"),
        autocommit=autocommit,
        charset="utf8mb4",
        connect_timeout=10,
        read_timeout=20,
        write_timeout=20,
    )


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def load_store_from_db() -> Dict[str, Any]:
    table_name = push_rules_table_name()
    sql = f"""
    SELECT
        id,
        endpoint,
        subscription_json,
        facility_id,
        section_key,
        threshold,
        created_at
    FROM {table_name}
    """

    conn = None
    try:
        conn = open_db_connection()
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Push rule store DB is unavailable") from exc
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    rules: List[Dict[str, Any]] = []
    for row in rows:
        (
            rule_id,
            endpoint,
            subscription_json,
            facility_id,
            section_key,
            threshold,
            created_at,
        ) = row

        try:
            parsed_subscription = json.loads(subscription_json or "{}")
            if not isinstance(parsed_subscription, dict):
                parsed_subscription = {}
        except Exception:
            parsed_subscription = {}

        rules.append(
            {
                "_id": _int_or_default(rule_id, 0),
                "endpoint": str(endpoint or "").strip(),
                "subscription": parsed_subscription,
                "facilityId": _int_or_default(facility_id, 0),
                "sectionKey": canonical_section_key(str(section_key or "")),
                "threshold": _int_or_default(threshold, 0),
                "createdAt": _str_or_none(created_at),
            }
        )
    return {"rules": rules}


def canonical_section_key(value: str) -> str:
    key = normalize_section_key(value)
    if key in {"overall", "entire facility", "facility", "all", "all sections", "whole gym"}:
        return "overall"
    return key


def location_ids_for_section(facility_id: int, section_key: str) -> List[int]:
    section_map = SECTION_IDS.get(facility_id, {})
    normalized_key = canonical_section_key(section_key)

    if normalized_key == "overall":
        output: List[int] = []
        seen = set()
        for ids in section_map.values():
            if not isinstance(ids, list):
                continue
            for raw_id in ids:
                try:
                    location_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if location_id in seen:
                    continue
                seen.add(location_id)
                output.append(location_id)
        return output

    ids = section_map.get(normalized_key)
    if not isinstance(ids, list):
        return []

    output = []
    for raw_id in ids:
        try:
            output.append(int(raw_id))
        except (TypeError, ValueError):
            continue
    return output


def db_rules_count() -> int:
    table_name = push_rules_table_name()
    conn = None
    try:
        conn = open_db_connection()
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            row = cur.fetchone()
            return _int_or_default(row[0] if row else 0, 0)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Push rule store DB is unavailable") from exc
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def db_delete_rule_by_id(rule_id: int) -> int:
    table_name = push_rules_table_name()
    conn = None
    try:
        conn = open_db_connection()
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table_name} WHERE id = %s", (int(rule_id),))
            return int(cur.rowcount or 0)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Push rule store DB is unavailable") from exc
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def db_delete_rules_by_endpoint(endpoint: str) -> int:
    table_name = push_rules_table_name()
    conn = None
    try:
        conn = open_db_connection()
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table_name} WHERE endpoint = %s", (str(endpoint),))
            return int(cur.rowcount or 0)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Push rule store DB is unavailable") from exc
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def db_rule_exists(endpoint: str, facility_id: int, section_key: str, threshold: int) -> bool:
    table_name = push_rules_table_name()
    normalized_key = canonical_section_key(section_key)
    conn = None
    try:
        conn = open_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT 1
                FROM {table_name}
                WHERE endpoint = %s
                  AND facility_id = %s
                  AND section_key = %s
                  AND threshold = %s
                LIMIT 1
                """,
                (
                    str(endpoint),
                    int(facility_id),
                    normalized_key,
                    int(threshold),
                ),
            )
            row = cur.fetchone()
            return row is not None
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Push rule store DB is unavailable") from exc
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def db_upsert_rule(
    endpoint: str,
    subscription: Dict[str, Any],
    facility_id: int,
    section_key: str,
    threshold: int,
) -> int:
    table_name = push_rules_table_name()
    now = now_iso()
    normalized_key = canonical_section_key(section_key)
    subscription_json = json.dumps(subscription, separators=(",", ":"), ensure_ascii=False)
    conn = None

    try:
        conn = open_db_connection(autocommit=False)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {table_name}
                    (endpoint, subscription_json, facility_id, section_key, threshold, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    subscription_json = VALUES(subscription_json),
                    facility_id = VALUES(facility_id),
                    section_key = VALUES(section_key),
                    created_at = VALUES(created_at),
                    threshold = VALUES(threshold)
                """,
                (
                    str(endpoint),
                    subscription_json,
                    int(facility_id),
                    normalized_key,
                    int(threshold),
                    now,
                ),
            )
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            row = cur.fetchone()
            rules_count = _int_or_default(row[0] if row else 0, 0)
        conn.commit()
        return rules_count
    except HTTPException:
        raise
    except Exception as exc:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        raise HTTPException(status_code=503, detail="Push rule store DB is unavailable") from exc
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def db_acquire_evaluator_lock() -> Optional[Any]:
    conn = None
    try:
        conn = open_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT GET_LOCK(%s, 0)", (PUSH_EVALUATOR_DB_LOCK_NAME,))
            row = cur.fetchone()
            if row and _int_or_default(row[0], 0) == 1:
                return conn
    except Exception:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        return None

    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    return None


def db_release_evaluator_lock(conn: Any) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT RELEASE_LOCK(%s)", (PUSH_EVALUATOR_DB_LOCK_NAME,))
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_vapid_public_key() -> str:
    if not PUSH_VAPID_PUBLIC_KEY:
        raise HTTPException(status_code=503, detail="Push VAPID public key is not configured")
    return PUSH_VAPID_PUBLIC_KEY


def get_vapid_private_key() -> str:
    if not PUSH_VAPID_PRIVATE_KEY:
        raise HTTPException(status_code=503, detail="Push VAPID private key is not configured")
    return PUSH_VAPID_PRIVATE_KEY


def get_vapid_claims() -> Dict[str, str]:
    if not PUSH_VAPID_SUBJECT:
        raise HTTPException(status_code=503, detail="Push VAPID subject is not configured")
    return {"sub": PUSH_VAPID_SUBJECT}


def extract_endpoint(subscription: Dict[str, Any]) -> str:
    endpoint = str(subscription.get("endpoint", "")).strip()
    if not endpoint:
        raise HTTPException(status_code=400, detail="Push subscription endpoint is missing")
    return endpoint


def send_notification(
    subscription: Dict[str, Any],
    title: str,
    body: str,
    url: str,
    sent_at: str | None = None,
) -> None:
    payload = json.dumps({
        "title": title,
        "body": body,
        "url": url or DEFAULT_NOTIFICATION_URL,
        "sentAt": sent_at or now_iso(),
    })

    try:
        webpush(
            subscription_info=subscription,
            data=payload,
            vapid_private_key=get_vapid_private_key(),
            vapid_claims=get_vapid_claims(),
            ttl=120,
        )
    except WebPushException as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code in {404, 410}:
            raise HTTPException(status_code=410, detail="Push subscription is no longer valid") from exc
        raise HTTPException(status_code=502, detail="Failed to send push notification") from exc


def push_db_available() -> bool:
    table_name = push_rules_table_name()
    conn = None
    try:
        conn = open_db_connection()
        with conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
            cur.fetchone()
        return True
    except Exception:
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def index_live_rows(rows: Sequence[SnapshotRow]) -> Dict[int, SnapshotRow]:
    output: Dict[int, SnapshotRow] = {}
    for row in rows:
        output[row.location_id] = row
    return output


OCCUPANCY_FRESHNESS = timedelta(minutes=10)


def is_aware_utc_datetime(value: Any) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() == timedelta(0)
    )


def is_fresh_snapshot(row: SnapshotRow, now: datetime) -> bool:
    fetched_at = row.fetched_at
    if not is_aware_utc_datetime(fetched_at):
        return False
    elapsed = now - fetched_at
    return timedelta(0) <= elapsed <= OCCUPANCY_FRESHNESS


def round_nonnegative_percent(value: float) -> int:
    return max(0, math.floor(value + 0.5))


def compute_section_metrics(
    facility_id: int,
    section_key: str,
    live_index: Dict[int, SnapshotRow],
    now: datetime | None = None,
) -> Optional[Dict[str, Any]]:
    location_ids = location_ids_for_section(facility_id, section_key)
    if not location_ids:
        return None

    metric_now = now if now is not None else datetime.now(timezone.utc)
    if not is_aware_utc_datetime(metric_now):
        raise ValueError("section metric time must be an aware UTC datetime")

    configured_locations = 0
    closed_locations = 0
    expected_open_capacity = 0
    observed_locations = 0
    observed_capacity = 0
    observed_count = 0
    for location_id in location_ids:
        configured_capacity = MAX_CAP.get(location_id)
        if type(configured_capacity) is not int or configured_capacity <= 0:
            continue

        configured_locations += 1
        row = live_index.get(location_id)
        fresh = row is not None and is_fresh_snapshot(row, metric_now)

        if fresh and row is not None and row.is_closed is True:
            closed_locations += 1
            continue

        expected_open_capacity += configured_capacity

        if (
            fresh
            and row is not None
            and row.is_closed is False
            and type(row.current_capacity) is int
            and row.current_capacity >= 0
        ):
            observed_locations += 1
            observed_capacity += configured_capacity
            observed_count += row.current_capacity

    coverage = (
        observed_capacity / expected_open_capacity
        if expected_open_capacity > 0
        else 0
    )
    all_configured_locations_closed = (
        configured_locations > 0
        and closed_locations == configured_locations
    )

    if all_configured_locations_closed:
        status = "closed"
    elif expected_open_capacity <= 0:
        status = "unknown"
    elif coverage >= 0.8:
        status = "live"
    elif coverage >= 0.5:
        status = "partial"
    else:
        status = "insufficient"

    percent = (
        (observed_count / observed_capacity) * 100
        if status in {"live", "partial"} and observed_capacity > 0
        else None
    )
    return {
        "total": observed_count if observed_locations > 0 else None,
        "max": observed_capacity,
        "expectedOpenCapacity": expected_open_capacity,
        "coverage": coverage,
        "percent": percent,
        "status": status,
    }


def evaluate_rules_once(
    snapshot_reader: SnapshotReadProtocol | None = None,
    repository_factory: RepositoryFactory = SnapshotRepository,
) -> Dict[str, Any]:
    snapshot_connection = None
    try:
        with STORE_LOCK:
            rules = list(load_store_from_db().get("rules", []))
            if not rules:
                return {"status": "ok", "rules": 0, "sent": 0, "failed": 0}

            evaluation_now = datetime.now(timezone.utc)
            evaluation_now_iso = evaluation_now.isoformat()

            if snapshot_reader is None:
                try:
                    snapshot_connection = open_db_connection(autocommit=False)
                    snapshot_reader = repository_factory(snapshot_connection)
                except Exception as exc:
                    raise HTTPException(
                        status_code=503, detail="Live occupancy DB is unavailable"
                    ) from exc
            try:
                live_rows = snapshot_reader.fetch_live_snapshot_rows()
            except Exception as exc:
                raise HTTPException(
                    status_code=503, detail="Failed to query live occupancy snapshot"
                ) from exc
            live_index = index_live_rows(live_rows)

            sent = 0
            failed = 0
            skipped_threshold = 0
            skipped_missing = 0
            evaluator_lock_conn = db_acquire_evaluator_lock()
            if evaluator_lock_conn is None:
                return {
                    "status": "ok",
                    "rules": db_rules_count(),
                    "sent": 0,
                    "failed": 0,
                    "skippedThreshold": 0,
                    "skippedCooldown": 0,
                    "skippedMissingSection": 0,
                    "skippedInactive": 0,
                    "skippedLocked": 1,
                    "evaluatedAt": evaluation_now_iso,
                }

            try:
                for rule in rules:
                    rule_id = _int_or_default(rule.get("_id"), 0)
                    try:
                        facility_id = int(rule.get("facilityId", 0))
                        section_key = canonical_section_key(
                            str(rule.get("sectionKey", ""))
                        )
                        threshold = int(rule.get("threshold", 0))

                        metrics = compute_section_metrics(
                            facility_id,
                            section_key,
                            live_index,
                            now=evaluation_now,
                        )
                        if metrics is None:
                            skipped_missing += 1
                            continue

                        coverage_value = metrics.get("coverage")
                        percent_value = metrics.get("percent")
                        if (
                            metrics.get("status") != "live"
                            or type(coverage_value) not in {int, float}
                            or float(coverage_value) < 0.8
                            or type(percent_value) not in {int, float}
                        ):
                            skipped_missing += 1
                            continue

                        percent = round_nonnegative_percent(float(percent_value))
                        if percent > threshold:
                            skipped_threshold += 1
                            continue

                        section_label = (
                            "entire facility"
                            if section_key == "overall"
                            else str(section_key or "Selected area")
                        )
                        facility_label = FACILITY_NAMES.get(facility_id, "Gym")
                        notification_title = "RecLive Alert"
                        notification_body = (
                            f"{facility_label} {section_label} is {percent}% full "
                            f"(at or below your {threshold}% alert)."
                        )
                        notification_url = f"/?facility={facility_id}"

                        send_notification(
                            subscription=rule.get("subscription", {}),
                            title=notification_title,
                            body=notification_body,
                            url=notification_url,
                            sent_at=evaluation_now_iso,
                        )
                        sent += 1
                        if rule_id > 0:
                            db_delete_rule_by_id(rule_id)
                    except HTTPException as exc:
                        failed += 1
                        if exc.status_code == 410 and rule_id > 0:
                            db_delete_rule_by_id(rule_id)
                    except Exception:
                        failed += 1
            finally:
                db_release_evaluator_lock(evaluator_lock_conn)

        final_rules = db_rules_count()
        return {
            "status": "ok",
            "rules": final_rules,
            "sent": sent,
            "failed": failed,
            "skippedThreshold": skipped_threshold,
            "skippedCooldown": 0,
            "skippedMissingSection": skipped_missing,
            "skippedInactive": 0,
            "evaluatedAt": evaluation_now_iso,
        }
    finally:
        safe_close(snapshot_connection)


async def evaluator_loop() -> None:
    while True:
        try:
            if hasattr(asyncio, "to_thread"):
                await asyncio.to_thread(evaluate_rules_once)
            else:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, evaluate_rules_once)
        except Exception as exc:
            print(f"[push-evaluator] error: {exc}")
        await asyncio.sleep(max(30, EVALUATOR_INTERVAL_SECONDS))


class IgnoreExtraModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class PushRuleRequest(IgnoreExtraModel):
    subscription: Dict[str, Any]
    facilityId: int
    sectionKey: str
    threshold: int = Field(ge=1, le=1000)


class UnsubscribeRequest(IgnoreExtraModel):
    endpoint: str


class PushRuleExistsRequest(IgnoreExtraModel):
    subscription: Dict[str, Any]
    facilityId: int
    sectionKey: str
    threshold: int = Field(ge=1, le=1000)


class PushDispatchRequest(IgnoreExtraModel):
    facilityId: Optional[int] = None
    sectionKey: Optional[str] = None
    title: str = "RecLive Alert"
    body: str = "Your occupancy threshold was met."
    url: str = "/"


@app.get("/health/push")
def push_health(_admin: None = Depends(require_admin_token)) -> Dict[str, Any]:
    rules_count = db_rules_count()
    return {
        "status": "ok",
        "rules": rules_count,
        "vapidConfigured": push_vapid_configured(),
        "evaluatorEnabled": evaluator_enabled(),
        "evaluatorIntervalSeconds": EVALUATOR_INTERVAL_SECONDS,
    }


@app.get("/api/push/public-key")
def public_key() -> Dict[str, str]:
    return {"publicKey": get_vapid_public_key()}


@app.get("/api/push/availability")
def push_availability() -> Dict[str, Any]:
    db_available = push_db_available()
    vapid_configured = push_vapid_configured()
    alerts_available = db_available and vapid_configured
    reason: Optional[str] = None
    if not db_available:
        reason = "push_rules_db_unavailable"
    elif not vapid_configured:
        reason = "push_vapid_unconfigured"
    return {
        "apiAvailable": True,
        "dbAvailable": db_available,
        "alertsAvailable": alerts_available,
        "reason": reason,
        "storeBackend": "db",
    }


@app.post("/api/push/rules/exists")
def push_rule_exists(payload: PushRuleExistsRequest) -> Dict[str, bool]:
    endpoint = extract_endpoint(payload.subscription)
    section_key = canonical_section_key(payload.sectionKey)
    return {
        "exists": db_rule_exists(
            endpoint=endpoint,
            facility_id=payload.facilityId,
            section_key=section_key,
            threshold=payload.threshold,
        )
    }


@app.post("/api/push/subscribe")
def subscribe(payload: PushRuleRequest) -> Dict[str, Any]:
    endpoint = extract_endpoint(payload.subscription)
    section_key = canonical_section_key(payload.sectionKey)
    if not location_ids_for_section(payload.facilityId, section_key):
        raise HTTPException(status_code=400, detail="Unknown section for facility")
    with STORE_LOCK:
        rules_count = db_upsert_rule(
            endpoint=endpoint,
            subscription=payload.subscription,
            facility_id=payload.facilityId,
            section_key=section_key,
            threshold=payload.threshold,
        )
        return {"status": "ok", "rules": rules_count}


@app.post("/api/push/unsubscribe")
def unsubscribe(payload: UnsubscribeRequest) -> Dict[str, Any]:
    endpoint = payload.endpoint.strip()
    if not endpoint:
        raise HTTPException(status_code=400, detail="Endpoint is required")

    with STORE_LOCK:
        removed = db_delete_rules_by_endpoint(endpoint)
    return {"status": "ok", "removed": removed}


@app.post("/api/push/dispatch")
def dispatch(
    payload: PushDispatchRequest,
    _admin: None = Depends(require_admin_token),
) -> Dict[str, Any]:
    with STORE_LOCK:
        rules = load_store_from_db().get("rules", [])

        targets = []
        payload_section_key = canonical_section_key(payload.sectionKey) if payload.sectionKey else None
        for rule in rules:
            if payload.facilityId is not None and int(rule.get("facilityId", -1)) != payload.facilityId:
                continue
            if payload_section_key and canonical_section_key(str(rule.get("sectionKey", ""))) != payload_section_key:
                continue
            targets.append(rule)

        if not targets:
            return {"status": "ok", "sent": 0, "failed": 0}

        sent = 0
        failed = 0
        for rule in targets:
            try:
                send_notification(
                    subscription=rule.get("subscription", {}),
                    title=payload.title,
                    body=payload.body,
                    url=payload.url,
                )
                sent += 1
            except HTTPException as exc:
                failed += 1
                if exc.status_code == 410:
                    rule_id = _int_or_default(rule.get("_id"), 0)
                    if rule_id > 0:
                        db_delete_rule_by_id(rule_id)
        return {"status": "ok", "sent": sent, "failed": failed}


@app.post("/api/push/evaluate")
def evaluate(_admin: None = Depends(require_admin_token)) -> Dict[str, Any]:
    return evaluate_rules_once()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("forecast_api:app", host=API_HOST, port=API_PORT, reload=False)
