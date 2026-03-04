import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

import pytz
import pymysql
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from pywebpush import WebPushException, webpush
from env_loader import load_project_dotenv

try:
    from forecast_shared import normalize_section_key
except ImportError:
    from server.forecast_shared import normalize_section_key

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


MAX_CAP = {
    # Nick
    5761: 140, 5764: 230, 5760: 150, 7089: 24, 5762: 100,
    5758: 200, 7090: 48, 5766: 24, 5753: 6, 5754: 6, 5763: 100,
    # Bakke
    8718: 30, 8717: 130, 8720: 24, 8714: 24, 8716: 116, 10550: 200,
    8705: 65, 8708: 27, 8712: 12, 8700: 246, 8698: 48, 8701: 39,
    8699: 75, 8696: 46, 8694: 100, 8695: 18,
}

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


@app.get("/api/forecast/facilities/{facility_id}/actual-hours")
def facility_actual_hours(
    facility_id: int,
    date: str = Query(..., description="YYYY-MM-DD"),
) -> Dict[str, Any]:
    start_local, end_local = parse_chicago_date_key(date)

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

    category_specs: List[Dict[str, Any]] = []
    all_location_ids: List[int] = []
    seen_location_ids = set()

    for category in categories_raw:
        if not isinstance(category, dict):
            continue
        location_ids = category_location_ids_for_forecast(facility_id, category)
        if not location_ids:
            continue

        for location_id in location_ids:
            if location_id in seen_location_ids:
                continue
            seen_location_ids.add(location_id)
            all_location_ids.append(location_id)

        hours_raw = category.get("hours", [])
        hour_points: List[Tuple[datetime, str]] = []
        for hour in hours_raw if isinstance(hours_raw, list) else []:
            if not isinstance(hour, dict):
                continue
            hour_start = _str_or_none(hour.get("hourStart"))
            if not hour_start:
                continue
            hour_dt = to_chicago_datetime(hour_start)
            if hour_dt is None:
                continue
            hour_bucket = hour_dt.replace(minute=0, second=0, microsecond=0)
            if not (start_local <= hour_bucket < end_local):
                continue
            hour_points.append((hour_bucket, hour_start))

        if not hour_points:
            continue

        category_max = _int_or_default(category.get("maxCapacity"), 0)
        if category_max <= 0:
            category_max = sum(max(0, int(MAX_CAP.get(location_id, 0))) for location_id in location_ids)

        category_specs.append(
            {
                "key": category.get("key"),
                "title": category.get("title"),
                "locationIds": location_ids,
                "maxCapacity": max(0, category_max),
                "hours": hour_points,
            }
        )

    if not category_specs or not all_location_ids:
        return {
            "facilityId": facility_id,
            "date": date,
            "coverageThreshold": ACTUAL_HOUR_MIN_COVERAGE,
            "categories": [],
        }

    placeholders = ",".join(["%s"] * len(all_location_ids))
    start_db = start_local.astimezone(DB_TZ).replace(tzinfo=None) - timedelta(hours=2)
    end_db = end_local.astimezone(DB_TZ).replace(tzinfo=None) + timedelta(hours=2)
    sql = f"""
    SELECT location_id, last_updated, fetched_at, current_capacity
    FROM location_history
    WHERE current_capacity IS NOT NULL
      AND (is_closed = 0 OR is_closed IS NULL)
      AND location_id IN ({placeholders})
      AND (
        (fetched_at IS NOT NULL AND fetched_at >= %s AND fetched_at < %s)
        OR
        (last_updated IS NOT NULL AND last_updated >= %s AND last_updated < %s)
      )
    """
    params: Tuple[Any, ...] = (
        *all_location_ids,
        start_db,
        end_db,
        start_db,
        end_db,
    )

    conn = None
    try:
        conn = open_db_connection()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Actual-hour DB query failed") from exc
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    by_location_hour: Dict[Tuple[int, datetime], List[float]] = {}
    for row in rows:
        try:
            location_id_raw, last_updated, fetched_at, current_capacity = row
            location_id = int(location_id_raw)
            count = float(current_capacity)
        except Exception:
            continue
        if count < 0:
            continue

        observed_at = to_chicago_datetime(last_updated) or to_chicago_datetime(fetched_at)
        if observed_at is None:
            continue

        hour_bucket = observed_at.replace(minute=0, second=0, microsecond=0)
        if not (start_local <= hour_bucket < end_local):
            continue

        key = (location_id, hour_bucket)
        by_location_hour.setdefault(key, []).append(count)

    by_location_hour_summary: Dict[Tuple[int, datetime], Tuple[float, int]] = {}
    for key, values in by_location_hour.items():
        if not values:
            continue
        samples = len(values)
        by_location_hour_summary[key] = (sum(values) / samples, samples)

    categories_payload: List[Dict[str, Any]] = []
    for spec in category_specs:
        location_ids = spec["locationIds"]
        category_max = int(spec["maxCapacity"])
        hours_payload: List[Dict[str, Any]] = []

        for hour_bucket, hour_start in spec["hours"]:
            observed_total = 0.0
            observed_capacity = 0
            observed_samples = 0

            for location_id in location_ids:
                entry = by_location_hour_summary.get((location_id, hour_bucket))
                if entry is None:
                    continue
                mean_count, samples = entry
                observed_total += max(0.0, float(mean_count))
                observed_samples += int(samples)
                observed_capacity += max(0, int(MAX_CAP.get(location_id, 0)))

            if observed_capacity <= 0:
                continue

            actual_coverage: Optional[float] = None
            adjusted_total = observed_total
            if category_max > 0:
                actual_coverage = max(0.0, min(observed_capacity / float(category_max), 1.0))
                if observed_capacity < category_max:
                    # Scale partial instrumentation up to the forecast category capacity so
                    # past hours can still surface observed counts instead of dropping back
                    # to predictions when only some linked locations reported.
                    adjusted_total = adjusted_total * (category_max / float(observed_capacity))

            actual_count = max(0, int(round(adjusted_total)))
            actual_pct = round(min(actual_count / float(category_max), 1.0), 4) if category_max > 0 else None
            hour_payload: Dict[str, Any] = {
                "hourStart": hour_start,
                "actualCount": actual_count,
                "actualPct": actual_pct,
                "actualSampleCount": observed_samples,
            }
            if actual_coverage is not None:
                hour_payload["actualCoverage"] = round(actual_coverage, 4)
            hours_payload.append(hour_payload)

        categories_payload.append(
            {
                "key": spec["key"],
                "title": spec["title"],
                "hours": hours_payload,
            }
        )

    return {
        "facilityId": facility_id,
        "date": date,
        "coverageThreshold": ACTUAL_HOUR_MIN_COVERAGE,
        "categories": categories_payload,
    }


@app.get("/api/live-counts")
def live_counts() -> List[Dict[str, Any]]:
    return fetch_live_counts()


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
) -> None:
    payload = json.dumps({
        "title": title,
        "body": body,
        "url": url or DEFAULT_NOTIFICATION_URL,
        "sentAt": now_iso(),
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


def fetch_live_counts() -> List[Dict[str, Any]]:
    latest_sql = """
    SELECT h.location_id, h.is_closed, h.current_capacity, h.last_updated
    FROM location_history h
    JOIN (
        SELECT location_id, MAX(id) AS max_id
        FROM location_history
        GROUP BY location_id
    ) x ON x.location_id = h.location_id AND x.max_id = h.id
    ORDER BY h.location_id;
    """

    try:
        conn = open_db_connection()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Live occupancy DB is unavailable") from exc

    try:
        with conn.cursor() as cur:
            cur.execute(latest_sql)
            rows = cur.fetchall()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Failed to query live occupancy snapshot") from exc
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not rows:
        raise HTTPException(status_code=503, detail="Live occupancy snapshot is empty")

    payload: List[Dict[str, Any]] = []
    for row in rows:
        try:
            location_id, is_closed, current_capacity, last_updated = row
        except Exception:
            continue

        if isinstance(last_updated, datetime):
            last_updated_text = last_updated.strftime("%Y-%m-%d %H:%M:%S")
        elif last_updated is None:
            last_updated_text = None
        else:
            last_updated_text = str(last_updated)

        payload.append(
            {
                "LocationId": int(location_id),
                "IsClosed": None if is_closed is None else bool(is_closed),
                "LastCount": None if current_capacity is None else int(current_capacity),
                "LastUpdatedDateAndTime": last_updated_text,
            }
        )
    return payload


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


def index_live_rows(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    output: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        try:
            location_id = int(row.get("LocationId"))
        except (TypeError, ValueError):
            continue
        output[location_id] = row
    return output


def compute_section_metrics(
    facility_id: int,
    section_key: str,
    live_index: Dict[int, Dict[str, Any]],
) -> Optional[Dict[str, int]]:
    location_ids = location_ids_for_section(facility_id, section_key)
    if not location_ids:
        return None

    total = 0
    max_capacity = 0
    for location_id in location_ids:
        row = live_index.get(location_id, {})
        count_value = row.get("LastCount")
        try:
            current = int(count_value) if count_value is not None else 0
        except (TypeError, ValueError):
            current = 0
        max_cap = int(MAX_CAP.get(location_id, 0))
        total += max(0, current)
        max_capacity += max(0, max_cap)

    if max_capacity <= 0:
        percent = 0
    else:
        percent = int(round((total / max_capacity) * 100))

    percent = max(0, percent)
    return {"total": total, "max": max_capacity, "percent": percent}


def evaluate_rules_once() -> Dict[str, Any]:
    with STORE_LOCK:
        rules = list(load_store_from_db().get("rules", []))
        if not rules:
            return {"status": "ok", "rules": 0, "sent": 0, "failed": 0}

        live_rows = fetch_live_counts()
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
                "evaluatedAt": now_iso(),
            }

        try:
            for rule in rules:
                rule_id = _int_or_default(rule.get("_id"), 0)
                try:
                    facility_id = int(rule.get("facilityId", 0))
                    section_key = canonical_section_key(str(rule.get("sectionKey", "")))
                    threshold = int(rule.get("threshold", 0))

                    metrics = compute_section_metrics(facility_id, section_key, live_index)
                    if metrics is None:
                        skipped_missing += 1
                        continue

                    percent = int(metrics["percent"])
                    if percent > threshold:
                        skipped_threshold += 1
                        continue

                    section_label = "entire facility" if section_key == "overall" else str(section_key or "Selected area")
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
        "evaluatedAt": now_iso(),
    }


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


class PushTestRequest(IgnoreExtraModel):
    endpoint: Optional[str] = None
    title: str = "RecLive Alert"
    body: str = "This is a test notification."
    url: str = "/"


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


@app.post("/api/push/test")
def test_push(
    payload: PushTestRequest,
    _admin: None = Depends(require_admin_token),
) -> Dict[str, Any]:
    with STORE_LOCK:
        rules = load_store_from_db().get("rules", [])
        if not rules:
            raise HTTPException(status_code=404, detail="No stored push subscriptions")

        if payload.endpoint:
            endpoint = payload.endpoint.strip()
            rule = next((item for item in rules if item.get("endpoint") == endpoint), None)
            if not rule:
                raise HTTPException(status_code=404, detail="Subscription not found")
        else:
            rule = rules[0]

        send_notification(
            subscription=rule.get("subscription", {}),
            title=payload.title,
            body=payload.body,
            url=payload.url,
        )
    return {"status": "sent", "endpoint": rule.get("endpoint")}


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
