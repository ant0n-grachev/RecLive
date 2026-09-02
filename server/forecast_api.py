import asyncio
import base64
import binascii
import hmac
import http.client
import ipaddress
import json
import math
import os
import queue
import re
import socket
import ssl
import threading
import time
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)
from urllib.parse import urlsplit

import pytz
import pymysql
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from pywebpush import WebPushException, webpush
from env_loader import load_project_dotenv

try:
    from facility_schedule import official_facility_is_open
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
        SnapshotRepository,
        SnapshotRow,
    )
    from reclive.push_identity import (
        endpoint_hash,
        normalize_push_endpoint,  # noqa: F401 - consumed by Phase 5 routes
        rate_limit_subject_hash,  # noqa: F401 - consumed by Phase 5 routes
    )
except ImportError:
    from server.facility_schedule import official_facility_is_open
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
        SnapshotRepository,
        SnapshotRow,
    )
    from server.reclive.push_identity import (
        endpoint_hash,
        normalize_push_endpoint,  # noqa: F401 - consumed by Phase 5 routes
        rate_limit_subject_hash,  # noqa: F401 - consumed by Phase 5 routes
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
PUSH_BODY_MAX_BYTES = 16 * 1024
PUSH_DEFAULT_RULE_TTL_SECONDS = int_with_default(
    "PUSH_RULE_DEFAULT_TTL_SECONDS", 86_400
)
PUSH_MAX_RULE_TTL_SECONDS = int_with_default(
    "PUSH_RULE_MAX_TTL_SECONDS", 604_800
)
PUSH_MAX_ACTIVE_RULES_PER_ENDPOINT = int_with_default(
    "PUSH_MAX_ACTIVE_RULES_PER_ENDPOINT", 10
)
PUSH_WRITE_RATE_LIMIT = int_with_default("PUSH_WRITE_RATE_LIMIT", 20)
PUSH_WRITE_RATE_WINDOW_SECONDS = int_with_default(
    "PUSH_WRITE_RATE_WINDOW_SECONDS", 600
)
EVALUATOR_INTERVAL_SECONDS = int_with_default("PUSH_EVALUATOR_INTERVAL_SECONDS", 180)
PUSH_EVALUATOR_DB_LOCK_NAME = env_with_default("PUSH_EVALUATOR_DB_LOCK_NAME", "reclive_push_eval")

PUSH_VAPID_PUBLIC_KEY = env_with_default("PUSH_VAPID_PUBLIC_KEY", "")
PUSH_VAPID_PRIVATE_KEY = env_with_default("PUSH_VAPID_PRIVATE_KEY", "")
PUSH_VAPID_SUBJECT = env_with_default("PUSH_VAPID_SUBJECT", "")
PUSH_ADMIN_TOKEN = env_with_default("PUSH_ADMIN_TOKEN", "")

EVALUATOR_TASK: Optional[asyncio.Task] = None
APP_ENVIRONMENTS = frozenset({"development", "test", "production"})
PUSH_ADMIN_TOKEN_MIN_BYTES = 32
PUSH_ADMIN_TOKEN_MAX_BYTES = 512


def app_environment() -> str:
    value = env_with_default("APP_ENV", "development").lower()
    if value not in APP_ENVIRONMENTS:
        raise RuntimeError("APP_ENV must be development, test, or production")
    return value


def push_admin_routes_enabled() -> bool:
    return bool_with_default("PUSH_ADMIN_ROUTES_ENABLED", False)


def _validated_admin_token_bytes() -> bytes:
    try:
        token = PUSH_ADMIN_TOKEN.encode("ascii")
    except UnicodeEncodeError:
        raise RuntimeError(
            "PUSH_ADMIN_TOKEN must contain only ASCII characters"
        ) from None
    if len(token) < PUSH_ADMIN_TOKEN_MIN_BYTES:
        raise RuntimeError(
            "PUSH_ADMIN_TOKEN must be at least 32 bytes when admin routes are enabled"
        )
    if len(token) > PUSH_ADMIN_TOKEN_MAX_BYTES:
        raise RuntimeError(
            "PUSH_ADMIN_TOKEN must not exceed 512 bytes when admin routes are enabled"
        )
    return token


def validate_push_configuration() -> None:
    if PUSH_DEFAULT_RULE_TTL_SECONDS <= 0:
        raise RuntimeError("Push default rule TTL must be positive")
    if PUSH_MAX_RULE_TTL_SECONDS <= 0:
        raise RuntimeError("Push maximum rule TTL must be positive")
    if PUSH_MAX_ACTIVE_RULES_PER_ENDPOINT <= 0:
        raise RuntimeError("Push active-rule maximum must be positive")
    if PUSH_WRITE_RATE_LIMIT <= 0:
        raise RuntimeError("Push write rate limit must be positive")
    if PUSH_WRITE_RATE_WINDOW_SECONDS <= 0:
        raise RuntimeError("Push rate-limit window must be positive")
    if PUSH_DEFAULT_RULE_TTL_SECONDS > PUSH_MAX_RULE_TTL_SECONDS:
        raise RuntimeError("Push default rule TTL must not exceed the maximum rule TTL")
    if PUSH_MAX_RULE_TTL_SECONDS > 604_800:
        raise RuntimeError("Push maximum rule TTL must not exceed 604800 seconds")
    if PUSH_MAX_ACTIVE_RULES_PER_ENDPOINT > 10:
        raise RuntimeError("Push active-rule maximum must not exceed 10")
    if PUSH_WRITE_RATE_LIMIT > 20:
        raise RuntimeError("Push write rate limit must not exceed 20")

    environment = app_environment()
    if environment == "production":
        endpoint_hash("https://push.reclive.app/startup-check")
    if push_admin_routes_enabled():
        _validated_admin_token_bytes()


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
    validate_push_configuration()
    started_task: Optional[asyncio.Task] = None
    if evaluator_enabled() and (EVALUATOR_TASK is None or EVALUATOR_TASK.done()):
        started_task = asyncio.create_task(evaluator_loop())
        EVALUATOR_TASK = started_task
    try:
        yield
    finally:
        if started_task is not None:
            started_task.cancel()
            try:
                await started_task
            except asyncio.CancelledError:
                pass
            if EVALUATOR_TASK is started_task:
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
    try:
        _validated_admin_token_bytes()
    except RuntimeError:
        return False
    return True


def require_admin_token(
    x_reclive_admin_token: Optional[str] = Header(default=None, alias="X-RecLive-Admin-Token"),
) -> None:
    if not push_admin_routes_enabled():
        raise HTTPException(status_code=503, detail="Admin routes are disabled")
    try:
        configured = _validated_admin_token_bytes()
    except RuntimeError:
        raise HTTPException(
            status_code=503, detail="Admin token is not configured"
        ) from None
    try:
        supplied = (x_reclive_admin_token or "").encode("ascii")
    except UnicodeEncodeError:
        raise HTTPException(status_code=401, detail="Admin token is required") from None
    if not hmac.compare_digest(supplied, configured):
        raise HTTPException(status_code=401, detail="Admin token is required")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return now_utc().isoformat()


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


PUSH_RULE_SELECT_COLUMNS = """
    id,
    endpoint_hash,
    subscription_json,
    facility_id,
    section_key,
    threshold,
    created_at,
    expires_at,
    status,
    active_identity
"""


@dataclass(frozen=True)
class PushRuleRecord:
    id: int
    endpoint_hash: bytes
    subscription_json: object
    facility_id: int
    section_key: str
    threshold: int
    created_at: datetime
    expires_at: datetime
    status: str
    active_identity: int | None


class PushRuleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: int = Field(gt=0)
    facility_id: Literal[1186, 1656] = Field(alias="facilityId")
    section_key: str = Field(alias="sectionKey", min_length=1, max_length=80)
    threshold: int = Field(ge=1, le=100)
    created_at: datetime = Field(alias="createdAt")
    expires_at: datetime = Field(alias="expiresAt")
    status: Literal["pending"]


def _mysql_utc_datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("push rule timestamp must be UTC")
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    if value.utcoffset() != timedelta(0):
        raise ValueError("push rule timestamp must be UTC")
    return value.astimezone(timezone.utc)


def _mysql_utc_bind(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
        or value.utcoffset() != timedelta(0)
    ):
        raise ValueError("push rule timestamp must be aware UTC")
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _push_rule_from_row(row: Sequence[object]) -> PushRuleRecord:
    if len(row) != 10:
        raise ValueError("invalid push rule row")
    (
        rule_id,
        stored_digest,
        subscription_json,
        facility_id,
        section_key,
        threshold,
        created_at,
        expires_at,
        status,
        active_identity,
    ) = row
    if type(rule_id) is not int or rule_id <= 0:
        raise ValueError("invalid push rule row")
    try:
        digest = bytes(stored_digest)
    except (TypeError, ValueError):
        raise ValueError("invalid push rule row") from None
    if len(digest) != 32:
        raise ValueError("invalid push rule row")
    if type(facility_id) is not int or facility_id not in {1186, 1656}:
        raise ValueError("invalid push rule row")
    if type(section_key) is not str:
        raise ValueError("invalid push rule row")
    if type(threshold) is not int or not 1 <= threshold <= 100:
        raise ValueError("invalid push rule row")
    if type(status) is not str:
        raise ValueError("invalid push rule row")
    if active_identity is not None and type(active_identity) is not int:
        raise ValueError("invalid push rule row")
    return PushRuleRecord(
        id=rule_id,
        endpoint_hash=digest,
        subscription_json=subscription_json,
        facility_id=facility_id,
        section_key=section_key,
        threshold=threshold,
        created_at=_mysql_utc_datetime(created_at),
        expires_at=_mysql_utc_datetime(expires_at),
        status=status,
        active_identity=active_identity,
    )


def push_rule_response(rule: PushRuleRecord) -> Dict[str, Any]:
    created_at = _mysql_utc_datetime(rule.created_at)
    expires_at = _mysql_utc_datetime(rule.expires_at)
    if (
        rule.status != "pending"
        or rule.active_identity is None
        or canonical_section_key(rule.section_key) != rule.section_key
        or not location_ids_for_section(rule.facility_id, rule.section_key)
    ):
        raise ValueError("invalid pending push rule")
    try:
        response = PushRuleResponse(
            id=rule.id,
            facilityId=rule.facility_id,
            sectionKey=rule.section_key,
            threshold=rule.threshold,
            createdAt=created_at,
            expiresAt=expires_at,
            status="pending",
        )
    except ValidationError:
        raise ValueError("invalid pending push rule") from None
    return response.model_dump(mode="json", by_alias=True)


def _decode_stored_subscription(value: object) -> Dict[str, Any] | None:
    try:
        if isinstance(value, bytes):
            decoded: object = json.loads(value.decode("utf-8"))
        elif isinstance(value, str):
            decoded = json.loads(value)
        else:
            decoded = value
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _canonical_stored_endpoint(value: object) -> str | None:
    subscription = _decode_stored_subscription(value)
    if subscription is None or set(subscription) != {"endpoint", "keys"}:
        return None
    try:
        return validate_push_subscription(subscription).endpoint
    except (HTTPException, TypeError, ValueError):
        return None


def _rule_is_owned(
    rule: PushRuleRecord,
    supplied_endpoint: str,
    supplied_digest: bytes,
) -> bool:
    if not hmac.compare_digest(rule.endpoint_hash, supplied_digest):
        return False
    stored_endpoint = _canonical_stored_endpoint(rule.subscription_json)
    if stored_endpoint is None:
        return False
    return hmac.compare_digest(
        stored_endpoint.encode("utf-8"),
        supplied_endpoint.encode("utf-8"),
    )


def _canonical_subscription_json(subscription: "ValidatedSubscription") -> str:
    return json.dumps(
        subscription.subscription,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _safe_rollback(connection: Any) -> None:
    try:
        connection.rollback()
    except Exception:
        pass


def _safe_close_connection(connection: Any) -> None:
    try:
        connection.close()
    except Exception:
        pass


def _rule_store_unavailable() -> HTTPException:
    return _push_http_error(503, "push_rule_store_unavailable")


def _endpoint_lock_name(digest: bytes) -> str:
    if len(digest) != 32:
        raise ValueError("invalid push endpoint digest")
    return f"reclive:push:{digest.hex()[:48]}"


def _acquire_endpoint_lock(cursor: Any, lock_name: str) -> None:
    cursor.execute("SELECT GET_LOCK(%s, 2)", (lock_name,))
    row = cursor.fetchone()
    if row and type(row[0]) is int and row[0] == 1:
        return
    if row and type(row[0]) is int and row[0] == 0:
        raise _push_http_error(503, "push_rule_store_busy")
    raise _rule_store_unavailable()


def _release_endpoint_lock(connection: Any, lock_name: str) -> None:
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT RELEASE_LOCK(%s)", (lock_name,))
    except Exception:
        pass


def db_select_rule_by_id(cursor: Any, rule_id: int) -> PushRuleRecord | None:
    table_name = push_rules_table_name()
    cursor.execute(
        f"SELECT {PUSH_RULE_SELECT_COLUMNS} FROM {table_name} "
        "WHERE id = %s LIMIT 1",
        (int(rule_id),),
    )
    row = cursor.fetchone()
    return _push_rule_from_row(row) if row is not None else None


def _select_active_identity_rule(
    cursor: Any,
    digest: bytes,
    facility_id: int,
    section_key: str,
    threshold: int,
) -> PushRuleRecord | None:
    table_name = push_rules_table_name()
    cursor.execute(
        f"SELECT {PUSH_RULE_SELECT_COLUMNS} FROM {table_name} "
        "WHERE endpoint_hash = %s AND facility_id = %s "
        "AND section_key = %s AND threshold = %s "
        "AND active_identity IS NOT NULL ORDER BY id DESC LIMIT 1 FOR UPDATE",
        (digest, facility_id, section_key, threshold),
    )
    row = cursor.fetchone()
    return _push_rule_from_row(row) if row is not None else None


def _expire_owned_pending_rules(
    cursor: Any,
    subscription: "ValidatedSubscription",
    digest: bytes,
    now_bound: datetime,
) -> int:
    table_name = push_rules_table_name()
    cursor.execute(
        f"SELECT {PUSH_RULE_SELECT_COLUMNS} FROM {table_name} "
        "WHERE endpoint_hash = %s AND status = 'pending' "
        "AND active_identity IS NOT NULL AND expires_at <= %s FOR UPDATE",
        (digest, now_bound),
    )
    expired = 0
    for row in cursor.fetchall():
        try:
            rule = _push_rule_from_row(row)
        except ValueError:
            continue
        if not _rule_is_owned(rule, subscription.endpoint, digest):
            continue
        cursor.execute(
            f"UPDATE {table_name} SET status = 'expired', finalized_at = %s "
            "WHERE id = %s AND status = 'pending' AND expires_at <= %s",
            (now_bound, rule.id, now_bound),
        )
        expired += int(cursor.rowcount or 0)
    return expired


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


def db_subscribe_rule(
    subscription: "ValidatedSubscription",
    facility_id: int,
    section_key: str,
    threshold: int,
    ttl_seconds: int | None,
) -> tuple[bool, PushRuleRecord]:
    table_name = push_rules_table_name()
    canonical_endpoint = normalize_push_endpoint(subscription.endpoint)
    if not hmac.compare_digest(
        canonical_endpoint.encode("utf-8"),
        subscription.endpoint.encode("utf-8"),
    ):
        raise _push_http_error(422, "invalid_push_subscription")
    normalized_key = canonical_section_key(section_key)
    if (
        type(facility_id) is not int
        or facility_id not in {1186, 1656}
        or type(threshold) is not int
        or not 1 <= threshold <= 100
        or normalized_key != section_key
        or not location_ids_for_section(facility_id, normalized_key)
    ):
        raise _push_http_error(422, "invalid_push_request")
    ttl = PUSH_DEFAULT_RULE_TTL_SECONDS if ttl_seconds is None else ttl_seconds
    if type(ttl) is not int or not 1 <= ttl <= PUSH_MAX_RULE_TTL_SECONDS:
        raise _push_http_error(422, "invalid_push_request")
    try:
        now = now_utc()
        now_bound = _mysql_utc_bind(now)
        expires_bound = _mysql_utc_bind(now + timedelta(seconds=ttl))
        digest = endpoint_hash(canonical_endpoint)
        lock_name = _endpoint_lock_name(digest)
        subscription_json = _canonical_subscription_json(subscription)
    except HTTPException:
        raise
    except Exception:
        raise _rule_store_unavailable() from None

    conn = None
    locked = False
    committed = False

    try:
        conn = open_db_connection(autocommit=False)
        with conn.cursor() as cur:
            _acquire_endpoint_lock(cur, lock_name)
            locked = True
            _expire_owned_pending_rules(
                cur,
                subscription,
                digest,
                now_bound,
            )
            existing = _select_active_identity_rule(
                cur,
                digest,
                facility_id,
                normalized_key,
                threshold,
            )
            if existing is not None:
                if not _rule_is_owned(existing, canonical_endpoint, digest):
                    raise _push_http_error(409, "push_identity_conflict")
                if existing.status == "claimed":
                    raise _push_http_error(409, "push_rule_in_progress")
                if existing.status != "pending" or existing.expires_at <= now:
                    raise _push_http_error(409, "push_identity_conflict")
                conn.commit()
                committed = True
                return False, existing

            cur.execute(
                f"SELECT COUNT(*) FROM {table_name} "
                "WHERE endpoint_hash = %s AND active_identity IS NOT NULL "
                "AND status IN ('pending', 'claimed')",
                (digest,),
            )
            count_row = cur.fetchone()
            if (
                not count_row
                or type(count_row[0]) is not int
                or count_row[0] < 0
            ):
                raise RuntimeError("invalid active push rule count")
            if count_row[0] >= PUSH_MAX_ACTIVE_RULES_PER_ENDPOINT:
                raise _push_http_error(409, "push_rule_limit_reached")

            try:
                cur.execute(
                    f"""
                    INSERT INTO {table_name}
                        (endpoint_hash, subscription_json, facility_id,
                         section_key, threshold, created_at, expires_at, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending')
                    """,
                    (
                        digest,
                        subscription_json,
                        facility_id,
                        normalized_key,
                        threshold,
                        now_bound,
                        expires_bound,
                    ),
                )
            except pymysql.err.IntegrityError as exc:
                if not exc.args or exc.args[0] != 1062:
                    raise
                recovered = _select_active_identity_rule(
                    cur,
                    digest,
                    facility_id,
                    normalized_key,
                    threshold,
                )
                if recovered is None:
                    raise RuntimeError("active push identity unavailable") from None
                if not _rule_is_owned(recovered, canonical_endpoint, digest):
                    raise _push_http_error(409, "push_identity_conflict")
                if recovered.status == "claimed":
                    raise _push_http_error(409, "push_rule_in_progress")
                if recovered.status != "pending" or recovered.expires_at <= now:
                    raise _push_http_error(409, "push_identity_conflict")
                conn.commit()
                committed = True
                return False, recovered

            inserted_id = int(cur.lastrowid or 0)
            inserted = db_select_rule_by_id(cur, inserted_id)
            if inserted is None or not _rule_is_owned(
                inserted,
                canonical_endpoint,
                digest,
            ):
                raise RuntimeError("inserted push rule unavailable")
        conn.commit()
        committed = True
        return True, inserted
    except HTTPException:
        if conn is not None and not committed:
            _safe_rollback(conn)
        raise
    except Exception:
        if conn is not None and not committed:
            _safe_rollback(conn)
        raise _rule_store_unavailable() from None
    finally:
        if conn is not None:
            if locked:
                _release_endpoint_lock(conn, lock_name)
            _safe_close_connection(conn)


def _db_list_owned_rule_records(
    subscription: "ValidatedSubscription",
) -> List[PushRuleRecord]:
    table_name = push_rules_table_name()
    try:
        canonical_endpoint = normalize_push_endpoint(subscription.endpoint)
        digest = endpoint_hash(canonical_endpoint)
        now_bound = _mysql_utc_bind(now_utc())
    except Exception:
        raise _rule_store_unavailable() from None
    conn = None
    try:
        conn = open_db_connection(autocommit=False)
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {PUSH_RULE_SELECT_COLUMNS} FROM {table_name} "
                "WHERE endpoint_hash = %s AND status = 'pending' "
                "AND active_identity IS NOT NULL AND expires_at > %s "
                "ORDER BY created_at, id",
                (
                    digest,
                    now_bound,
                ),
            )
            rows = cur.fetchall()
        owned: List[PushRuleRecord] = []
        for row in rows:
            try:
                rule = _push_rule_from_row(row)
            except ValueError:
                continue
            if _rule_is_owned(rule, canonical_endpoint, digest):
                owned.append(rule)
        conn.commit()
        return owned
    except Exception:
        if conn is not None:
            _safe_rollback(conn)
        raise _rule_store_unavailable() from None
    finally:
        if conn is not None:
            _safe_close_connection(conn)


def resolve_owned_rule(endpoint: str, rule_id: int) -> PushRuleRecord:
    try:
        canonical_endpoint = normalize_push_endpoint(endpoint)
        digest = endpoint_hash(canonical_endpoint)
    except Exception:
        raise _push_http_error(404, "push_rule_not_found") from None
    conn = None
    try:
        conn = open_db_connection(autocommit=False)
        with conn.cursor() as cur:
            table_name = push_rules_table_name()
            cur.execute(
                f"SELECT {PUSH_RULE_SELECT_COLUMNS} FROM {table_name} "
                "WHERE endpoint_hash = %s AND id = %s LIMIT 1",
                (digest, int(rule_id)),
            )
            row = cur.fetchone()
            rule = _push_rule_from_row(row) if row is not None else None
        if rule is None or not _rule_is_owned(rule, canonical_endpoint, digest):
            raise _push_http_error(404, "push_rule_not_found")
        conn.commit()
        return rule
    except HTTPException:
        if conn is not None:
            _safe_rollback(conn)
        raise
    except Exception:
        if conn is not None:
            _safe_rollback(conn)
        raise _rule_store_unavailable() from None
    finally:
        if conn is not None:
            _safe_close_connection(conn)


def db_cancel_owned_rule(
    subscription: "ValidatedSubscription",
    rule_id: int,
) -> int:
    table_name = push_rules_table_name()
    try:
        canonical_endpoint = normalize_push_endpoint(subscription.endpoint)
        digest = endpoint_hash(canonical_endpoint)
        lock_name = _endpoint_lock_name(digest)
        now = now_utc()
        now_bound = _mysql_utc_bind(now)
    except Exception:
        raise _rule_store_unavailable() from None
    conn = None
    locked = False
    committed = False
    try:
        conn = open_db_connection(autocommit=False)
        with conn.cursor() as cur:
            _acquire_endpoint_lock(cur, lock_name)
            locked = True
            _expire_owned_pending_rules(
                cur,
                subscription,
                digest,
                now_bound,
            )
            cur.execute(
                f"SELECT {PUSH_RULE_SELECT_COLUMNS} FROM {table_name} "
                "WHERE endpoint_hash = %s AND id = %s LIMIT 1 FOR UPDATE",
                (digest, int(rule_id)),
            )
            row = cur.fetchone()
            try:
                rule = _push_rule_from_row(row) if row is not None else None
            except ValueError:
                rule = None
            if (
                rule is None
                or rule.status != "pending"
                or rule.expires_at <= now
                or not _rule_is_owned(rule, canonical_endpoint, digest)
            ):
                raise _push_http_error(404, "push_rule_not_found")
            cur.execute(
                f"UPDATE {table_name} SET status = 'cancelled', "
                "finalized_at = %s WHERE id = %s AND status = 'pending' "
                "AND expires_at > %s",
                (now_bound, rule.id, now_bound),
            )
            cancelled = int(cur.rowcount or 0)
            if cancelled != 1:
                raise _push_http_error(404, "push_rule_not_found")
        conn.commit()
        committed = True
    except HTTPException:
        if conn is not None and not committed:
            _safe_rollback(conn)
        raise
    except Exception:
        if conn is not None and not committed:
            _safe_rollback(conn)
        raise _rule_store_unavailable() from None
    finally:
        if conn is not None:
            if locked:
                _release_endpoint_lock(conn, lock_name)
            _safe_close_connection(conn)
    return cancelled


def db_cancel_all_owned_rules(
    subscription: "ValidatedSubscription",
) -> int:
    table_name = push_rules_table_name()
    try:
        canonical_endpoint = normalize_push_endpoint(subscription.endpoint)
        digest = endpoint_hash(canonical_endpoint)
        lock_name = _endpoint_lock_name(digest)
        now = now_utc()
        now_bound = _mysql_utc_bind(now)
    except Exception:
        raise _rule_store_unavailable() from None
    conn = None
    locked = False
    committed = False
    try:
        conn = open_db_connection(autocommit=False)
        with conn.cursor() as cur:
            _acquire_endpoint_lock(cur, lock_name)
            locked = True
            _expire_owned_pending_rules(
                cur,
                subscription,
                digest,
                now_bound,
            )
            cur.execute(
                f"SELECT {PUSH_RULE_SELECT_COLUMNS} FROM {table_name} "
                "WHERE endpoint_hash = %s AND status = 'pending' "
                "AND active_identity IS NOT NULL AND expires_at > %s "
                "ORDER BY id FOR UPDATE",
                (digest, now_bound),
            )
            cancelled = 0
            for row in cur.fetchall():
                try:
                    rule = _push_rule_from_row(row)
                except ValueError:
                    continue
                if not _rule_is_owned(rule, canonical_endpoint, digest):
                    continue
                cur.execute(
                    f"UPDATE {table_name} SET status = 'cancelled', "
                    "finalized_at = %s WHERE id = %s AND status = 'pending' "
                    "AND expires_at > %s",
                    (now_bound, rule.id, now_bound),
                )
                cancelled += int(cur.rowcount or 0)
        conn.commit()
        committed = True
        return cancelled
    except HTTPException:
        if conn is not None and not committed:
            _safe_rollback(conn)
        raise
    except Exception:
        if conn is not None and not committed:
            _safe_rollback(conn)
        raise _rule_store_unavailable() from None
    finally:
        if conn is not None:
            if locked:
                _release_endpoint_lock(conn, lock_name)
            _safe_close_connection(conn)


class PushEvaluatorStoreError(RuntimeError):
    pass


def db_acquire_evaluator_lock() -> Optional[Any]:
    conn = None
    try:
        conn = open_db_connection(autocommit=False)
        with conn.cursor() as cur:
            cur.execute("SELECT GET_LOCK(%s, 0)", (PUSH_EVALUATOR_DB_LOCK_NAME,))
            row = cur.fetchone()
            valid_row = (
                isinstance(row, (list, tuple))
                and len(row) == 1
                and type(row[0]) is int
            )
            if valid_row and row[0] == 1:
                return conn
            if valid_row and row[0] == 0:
                _safe_close_connection(conn)
                return None
    except Exception:
        if conn is not None:
            _safe_close_connection(conn)
        raise PushEvaluatorStoreError("push evaluator lock unavailable") from None

    if conn is not None:
        _safe_close_connection(conn)
    raise PushEvaluatorStoreError("push evaluator lock result invalid")


def db_release_evaluator_lock(conn: Any) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT RELEASE_LOCK(%s)", (PUSH_EVALUATOR_DB_LOCK_NAME,))
            row = cur.fetchone()
            if (
                not isinstance(row, (list, tuple))
                or len(row) != 1
                or type(row[0]) is not int
                or row[0] != 1
            ):
                raise PushEvaluatorStoreError("push evaluator lock release failed")
    except PushEvaluatorStoreError:
        raise
    except Exception:
        raise PushEvaluatorStoreError("push evaluator lock release failed") from None
    finally:
        _safe_close_connection(conn)


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


PUSH_PROVIDER_RESPONSE_MAX_BYTES = 4_096
PUSH_TRANSPORT_TIMEOUT_SECONDS = 10
PUSH_TRANSPORT_DEADLINE_SECONDS = 10.0
PUSH_TRANSPORT_WORKER_COUNT = 2
PUSH_TRANSPORT_QUEUE_CAPACITY = 2


class SafePushDispatchError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("push_dispatch_failed")


class PushProviderStatusError(SafePushDispatchError):
    def __init__(self, status_code: int) -> None:
        super().__init__()
        self.status_code = status_code


class _PushAttemptDeadline:
    def __init__(self, timeout_seconds: float) -> None:
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise SafePushDispatchError()
        self._deadline = time.monotonic() + timeout
        self._cancelled = threading.Event()
        self._resource_lock = threading.Lock()
        self._resources: list[Any] = []

    def remaining(self) -> float:
        if self._cancelled.is_set():
            raise SafePushDispatchError()
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise SafePushDispatchError()
        return remaining

    def check(self) -> None:
        self.remaining()

    def register(self, resource: Any) -> Any:
        close_immediately = False
        with self._resource_lock:
            if self._cancelled.is_set():
                close_immediately = True
            else:
                self._resources.append(resource)
        if close_immediately:
            self._close_resource(resource)
            raise SafePushDispatchError()
        self.check()
        return resource

    def cancel(self) -> None:
        self._cancelled.set()
        with self._resource_lock:
            resources = tuple(reversed(self._resources))
            self._resources.clear()
        for resource in resources:
            self._close_resource(resource)

    @staticmethod
    def _close_resource(resource: Any) -> None:
        try:
            resource.close()
        except Exception:
            pass


class _BoundedPushExecutor:
    def __init__(self, worker_count: int, queue_capacity: int) -> None:
        if worker_count < 1 or queue_capacity < 1:
            raise ValueError("invalid push executor bounds")
        self._tasks: queue.Queue[Callable[[], None]] = queue.Queue(
            maxsize=queue_capacity
        )
        self._workers = tuple(
            threading.Thread(
                target=self._run,
                name=f"reclive-push-worker-{index + 1}",
                daemon=True,
            )
            for index in range(worker_count)
        )
        for worker in self._workers:
            worker.start()

    def submit(self, task: Callable[[], None]) -> bool:
        try:
            self._tasks.put_nowait(task)
        except queue.Full:
            return False
        return True

    def _run(self) -> None:
        while True:
            task = self._tasks.get()
            try:
                task()
            except BaseException:
                pass
            finally:
                self._tasks.task_done()


_push_executor_lock = threading.Lock()
_push_executor: _BoundedPushExecutor | None = None


def _get_push_executor() -> _BoundedPushExecutor:
    global _push_executor
    if _push_executor is None:
        with _push_executor_lock:
            if _push_executor is None:
                _push_executor = _BoundedPushExecutor(
                    PUSH_TRANSPORT_WORKER_COUNT,
                    PUSH_TRANSPORT_QUEUE_CAPACITY,
                )
    return _push_executor


@dataclass(frozen=True)
class PinnedPushTarget:
    endpoint: str
    connect_ip: str
    tls_server_hostname: str
    host_header: str
    port: int
    request_target: str


@dataclass(frozen=True)
class SafePushResponse:
    status_code: int
    reason: str = ""
    text: str = ""


def resolve_endpoint_host(host: str, port: int) -> List[str]:
    answers = socket.getaddrinfo(
        host,
        int(port),
        socket.AF_UNSPEC,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
    )
    return [str(answer[4][0]) for answer in answers]


def _is_safe_push_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    if isinstance(address, ipaddress.IPv6Address) and (
        address.scope_id is not None
        or address.ipv4_mapped is not None
        or address.sixtofour is not None
        or address.teredo is not None
    ):
        return False
    return bool(
        address.is_global
        and not address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_unspecified
        and not address.is_reserved
    )


def resolve_public_push_addresses(
    endpoint: str,
    resolver: Any | None = None,
    *,
    attempt: _PushAttemptDeadline | None = None,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    try:
        if attempt is not None:
            attempt.check()
        canonical = normalize_push_endpoint(endpoint)
        parsed = urlsplit(canonical)
        host = parsed.hostname
        port = parsed.port or 443
        if not host:
            raise ValueError("missing push host")
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            raw_answers = (resolver or resolve_endpoint_host)(host, port)
        else:
            raw_answers = [literal.compressed]
        if attempt is not None:
            attempt.check()
        if not isinstance(raw_answers, (list, tuple)) or not raw_answers:
            raise ValueError("empty push address set")
        addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
        for raw_answer in raw_answers:
            if not isinstance(raw_answer, str):
                raise ValueError("invalid push address")
            address = ipaddress.ip_address(raw_answer)
            if not _is_safe_push_address(address):
                raise ValueError("unsafe push address")
            addresses.add(address)
        if not addresses:
            raise ValueError("empty push address set")
        return tuple(
            sorted(
                addresses,
                key=lambda address: (
                    0 if isinstance(address, ipaddress.IPv6Address) else 1,
                    int(address),
                ),
            )
        )
    except SafePushDispatchError:
        raise
    except Exception:
        raise SafePushDispatchError() from None


def build_pinned_push_target(
    endpoint: str,
    *,
    attempt: _PushAttemptDeadline | None = None,
) -> PinnedPushTarget:
    try:
        if attempt is not None:
            attempt.check()
        canonical = normalize_push_endpoint(endpoint)
        parsed = urlsplit(canonical)
        host = parsed.hostname
        if not host:
            raise ValueError("missing push host")
        port = parsed.port or 443
        addresses = resolve_public_push_addresses(
            canonical,
            attempt=attempt,
        )
        if attempt is not None:
            attempt.check()
        selected = addresses[0]
        try:
            host_address = ipaddress.ip_address(host)
        except ValueError:
            host_header_name = host
        else:
            host_header_name = (
                f"[{host_address.compressed}]"
                if isinstance(host_address, ipaddress.IPv6Address)
                else host_address.compressed
            )
        host_header = (
            host_header_name
            if port == 443
            else f"{host_header_name}:{port}"
        )
        request_target = parsed.path or "/"
        if parsed.query:
            request_target = f"{request_target}?{parsed.query}"
        request_target.encode("ascii")
        host_header.encode("ascii")
        host.encode("ascii")
        return PinnedPushTarget(
            endpoint=canonical,
            connect_ip=selected.compressed,
            tls_server_hostname=host,
            host_header=host_header,
            port=port,
            request_target=request_target,
        )
    except SafePushDispatchError:
        raise
    except Exception:
        raise SafePushDispatchError() from None


_PROTECTED_PUSH_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "cookie",
        "expect",
        "host",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


class PinnedPushSession:
    def __init__(
        self,
        target: PinnedPushTarget,
        attempt: _PushAttemptDeadline | None = None,
    ) -> None:
        self.target = target
        self.attempt = attempt or _PushAttemptDeadline(
            PUSH_TRANSPORT_DEADLINE_SECONDS
        )

    def post(
        self,
        url: str,
        *,
        timeout: float = PUSH_TRANSPORT_TIMEOUT_SECONDS,
        data: bytes,
        headers: Mapping[str, Any],
        **kwargs: Any,
    ) -> SafePushResponse:
        raw_socket: Any = None
        tls_socket: Any = None
        response: Any = None
        try:
            self.attempt.check()
            if kwargs or timeout != PUSH_TRANSPORT_TIMEOUT_SECONDS:
                raise ValueError("invalid push request options")
            canonical = normalize_push_endpoint(url)
            if not hmac.compare_digest(
                canonical.encode("utf-8"),
                self.target.endpoint.encode("utf-8"),
            ):
                raise ValueError("push endpoint mismatch")
            if not isinstance(data, bytes) or not isinstance(headers, Mapping):
                raise ValueError("invalid prepared push request")
            serialized_headers: list[tuple[str, str]] = []
            for raw_name, raw_value in headers.items():
                if not isinstance(raw_name, str):
                    raise ValueError("invalid push header")
                name = raw_name.strip()
                value = str(raw_value).strip()
                lowered = name.lower()
                if (
                    not name
                    or re.fullmatch(
                        r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+",
                        name,
                    )
                    is None
                    or lowered in _PROTECTED_PUSH_HEADERS
                    or lowered.startswith("proxy-")
                    or "\r" in name
                    or "\n" in name
                    or ":" in name
                    or "\r" in value
                    or "\n" in value
                ):
                    raise ValueError("invalid push header")
                name.encode("ascii")
                value.encode("latin-1")
                serialized_headers.append((name, value))

            request_lines = [
                f"POST {self.target.request_target} HTTP/1.1",
                f"Host: {self.target.host_header}",
                "Connection: close",
                f"Content-Length: {len(data)}",
                *(f"{name}: {value}" for name, value in serialized_headers),
                "",
                "",
            ]
            request_head = "\r\n".join(request_lines).encode("latin-1")
            connect_timeout = min(float(timeout), self.attempt.remaining())
            raw_socket = socket.create_connection(
                (self.target.connect_ip, self.target.port),
                timeout=connect_timeout,
            )
            self.attempt.register(raw_socket)
            self.attempt.check()
            context = ssl.create_default_context()
            if not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
                raise ValueError("unsafe TLS context")
            tls_socket = context.wrap_socket(
                raw_socket,
                server_hostname=self.target.tls_server_hostname,
            )
            self.attempt.register(tls_socket)
            self.attempt.check()
            tls_socket.sendall(request_head + data)
            self.attempt.check()
            response = http.client.HTTPResponse(tls_socket)
            self.attempt.register(response)
            response.begin()
            self.attempt.check()
            status = response.status
            if type(status) is not int or not 200 <= status <= 599:
                raise ValueError("invalid push response status")
            response.read(PUSH_PROVIDER_RESPONSE_MAX_BYTES + 1)
            self.attempt.check()
            return SafePushResponse(status_code=status)
        except SafePushDispatchError:
            raise
        except Exception:
            raise SafePushDispatchError() from None
        finally:
            for resource in (response, tls_socket, raw_socket):
                if resource is None:
                    continue
                try:
                    resource.close()
                except Exception:
                    pass


def _send_notification_pinned_attempt(
    subscription: Mapping[str, Any],
    title: str,
    body: str,
    url: str,
    sent_at: str | None,
    attempt: _PushAttemptDeadline,
) -> None:
    attempt.check()
    validated = validate_push_subscription(subscription)
    target = build_pinned_push_target(
        validated.endpoint,
        attempt=attempt,
    )
    attempt.check()
    payload = json.dumps(
        {
            "title": str(title)[:80],
            "body": str(body)[:240],
            "url": url,
            "sentAt": sent_at or now_iso(),
        },
        separators=(",", ":"),
        ensure_ascii=True,
    )
    attempt.check()
    response = webpush(
        subscription_info=validated.subscription,
        data=payload,
        vapid_private_key=get_vapid_private_key(),
        vapid_claims=dict(get_vapid_claims()),
        ttl=120,
        timeout=PUSH_TRANSPORT_TIMEOUT_SECONDS,
        requests_session=PinnedPushSession(target, attempt),
    )
    attempt.check()
    status_code = getattr(response, "status_code", None)
    if type(status_code) is not int or not 200 <= status_code <= 202:
        if type(status_code) is int and status_code in {404, 410}:
            raise PushProviderStatusError(status_code)
        raise SafePushDispatchError()


def send_notification_pinned(
    subscription: Mapping[str, Any],
    title: str,
    body: str,
    url: str,
    sent_at: str | None = None,
) -> None:
    attempt: _PushAttemptDeadline | None = None
    try:
        attempt = _PushAttemptDeadline(PUSH_TRANSPORT_DEADLINE_SECONDS)
        outcome: list[BaseException | None] = []
        finished = threading.Event()

        def worker() -> None:
            try:
                _send_notification_pinned_attempt(
                    subscription,
                    title,
                    body,
                    url,
                    sent_at,
                    attempt,
                )
            except BaseException as exc:
                outcome.append(exc)
            else:
                outcome.append(None)
            finally:
                finished.set()

        if not _get_push_executor().submit(worker):
            attempt.cancel()
            raise SafePushDispatchError()
        if not finished.wait(attempt.remaining()):
            attempt.cancel()
            raise SafePushDispatchError()
        if not outcome:
            raise SafePushDispatchError()
        error = outcome[0]
        if error is None:
            return
        if isinstance(error, PushProviderStatusError):
            raise PushProviderStatusError(error.status_code) from None
        if isinstance(error, WebPushException):
            status_code = getattr(
                getattr(error, "response", None),
                "status_code",
                None,
            )
            if type(status_code) is int and status_code in {404, 410}:
                raise PushProviderStatusError(status_code) from None
        raise SafePushDispatchError() from None
    except PushProviderStatusError:
        raise
    except SafePushDispatchError:
        raise
    except Exception:
        raise SafePushDispatchError() from None
    finally:
        if attempt is not None:
            attempt.cancel()


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


def push_identity_configured() -> bool:
    try:
        endpoint_hash("https://push.reclive.app/availability-check")
    except Exception:
        return False
    return True


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


def load_evaluator_candidates(
    conn: Any,
    now: datetime,
) -> List[PushRuleRecord]:
    table_name = push_rules_table_name()
    now_bound = _mysql_utc_bind(now)
    with conn.cursor() as cursor:
        cursor.execute(
            f"UPDATE {table_name} SET status = 'expired', finalized_at = %s "
            "WHERE status = 'pending' AND expires_at <= %s",
            (now_bound, now_bound),
        )
        cursor.execute(
            f"SELECT {PUSH_RULE_SELECT_COLUMNS} FROM {table_name} "
            "WHERE status = 'pending' AND active_identity IS NOT NULL "
            "AND expires_at > %s ORDER BY id",
            (now_bound,),
        )
        rows = cursor.fetchall()
    candidates: List[PushRuleRecord] = []
    for row in rows:
        try:
            rule = _push_rule_from_row(row)
        except (TypeError, ValueError):
            continue
        if (
            rule.status == "pending"
            and rule.active_identity is not None
            and rule.expires_at > now
        ):
            candidates.append(rule)
    return candidates


def claim_pending_rule(conn: Any, rule_id: int, now: datetime) -> bool:
    table_name = push_rules_table_name()
    now_bound = _mysql_utc_bind(now)
    with conn.cursor() as cursor:
        cursor.execute(
            f"UPDATE {table_name} SET status = 'claimed', claimed_at = %s "
            "WHERE id = %s AND status = 'pending' AND expires_at > %s",
            (now_bound, int(rule_id), now_bound),
        )
        return int(cursor.rowcount or 0) == 1


def finalize_claimed_rule(
    conn: Any,
    rule_id: int,
    now: datetime,
    status: str,
    failure_code: str | None = None,
) -> bool:
    if status not in {"sent", "failed", "invalid_subscription"}:
        raise ValueError("invalid push terminal status")
    if status == "sent" and failure_code is not None:
        raise ValueError("sent rule cannot have a failure code")
    if status != "sent" and failure_code not in {
        "webpush_failed",
        "webpush_404",
        "webpush_410",
    }:
        raise ValueError("invalid push failure code")
    table_name = push_rules_table_name()
    now_bound = _mysql_utc_bind(now)
    sent_at = now_bound if status == "sent" else None
    with conn.cursor() as cursor:
        cursor.execute(
            f"UPDATE {table_name} SET status = %s, sent_at = %s, "
            "finalized_at = %s, failure_code = %s "
            "WHERE id = %s AND status = 'claimed'",
            (
                status,
                sent_at,
                now_bound,
                failure_code,
                int(rule_id),
            ),
        )
        return int(cursor.rowcount or 0) == 1


def compute_fresh_section_metrics(
    facility_id: int,
    section_key: str,
    snapshots: Mapping[int, SnapshotRow],
    now: datetime,
) -> Optional[Dict[str, Any]]:
    metrics = compute_section_metrics(
        facility_id,
        section_key,
        dict(snapshots),
        now=now,
    )
    if metrics is None:
        return None
    coverage = metrics.get("coverage")
    percent = metrics.get("percent")
    if (
        metrics.get("status") != "live"
        or type(coverage) not in {int, float}
        or not math.isfinite(float(coverage))
        or float(coverage) < 0.8
        or type(percent) not in {int, float}
        or not math.isfinite(float(percent))
    ):
        return None
    return metrics


def facility_notification_url(facility_id: int) -> str:
    if facility_id == 1186:
        return "/nick"
    if facility_id == 1656:
        return "/bakke"
    raise ValueError("unsupported facility")


def _evaluator_store_unavailable() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="push_evaluator_store_unavailable",
    )


def _base_evaluator_result(now: datetime) -> Dict[str, Any]:
    return {
        "status": "ok",
        "rules": 0,
        "sent": 0,
        "failed": 0,
        "skippedThreshold": 0,
        "skippedCooldown": 0,
        "skippedMissingSection": 0,
        "skippedInactive": 0,
        "skippedLocked": 0,
        "evaluatedAt": now.isoformat(),
    }


def _sample_evaluator_now(
    fixed_now: datetime | None,
    *,
    not_before: datetime | None = None,
) -> datetime:
    sample = fixed_now if fixed_now is not None else now_utc()
    if (
        not is_aware_utc_datetime(sample)
        or (not_before is not None and sample < not_before)
    ):
        raise ValueError("invalid evaluator time")
    return sample


def _fresh_ingestion_at(value: object, now: datetime) -> bool:
    if not is_aware_utc_datetime(value):
        return False
    age = now - value
    return timedelta(0) <= age <= OCCUPANCY_FRESHNESS


def _rule_schedule_gate(
    rule: PushRuleRecord,
    schedule_payload: Mapping[str, Any],
    at: datetime,
) -> bool:
    return (
        rule.expires_at > at
        and canonical_section_key(rule.section_key) == rule.section_key
        and bool(location_ids_for_section(rule.facility_id, rule.section_key))
        and official_facility_is_open(
            schedule_payload,
            rule.facility_id,
            at,
        )
    )


def _rule_snapshot_gate(
    rule: PushRuleRecord,
    schedule_payload: Mapping[str, Any],
    snapshot: Any,
    live_index: Mapping[int, SnapshotRow],
    at: datetime,
) -> tuple[str, int | None]:
    if not _rule_schedule_gate(rule, schedule_payload, at):
        return "missing", None
    if not _fresh_ingestion_at(snapshot.last_successful_fetch_at, at):
        return "missing", None
    metrics = compute_fresh_section_metrics(
        rule.facility_id,
        rule.section_key,
        live_index,
        at,
    )
    if metrics is None:
        return "missing", None
    coverage_value = metrics.get("coverage")
    percent_value = metrics.get("percent")
    if (
        metrics.get("status") != "live"
        or type(coverage_value) not in {int, float}
        or not math.isfinite(float(coverage_value))
        or float(coverage_value) < 0.8
        or type(percent_value) not in {int, float}
        or not math.isfinite(float(percent_value))
    ):
        return "missing", None
    percent = round_nonnegative_percent(float(percent_value))
    if percent > rule.threshold:
        return "threshold", None
    return "eligible", percent


def _provider_failure_state(exc: Exception) -> tuple[str, str]:
    status_code: object = None
    if isinstance(exc, PushProviderStatusError):
        status_code = exc.status_code
    elif isinstance(exc, WebPushException):
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if type(status_code) is int and status_code in {404, 410}:
        return "invalid_subscription", f"webpush_{status_code}"
    return "failed", "webpush_failed"


def evaluate_rules_once(
    now: datetime | None = None,
    snapshot_reader: Any | None = None,
    repository_factory: RepositoryFactory = SnapshotRepository,
    *,
    facility_filter: int | None = None,
    section_filter: str | None = None,
) -> Dict[str, Any]:
    try:
        evaluation_now = _sample_evaluator_now(now)
    except Exception:
        raise _evaluator_store_unavailable() from None
    result = _base_evaluator_result(evaluation_now)
    connection = None
    try:
        try:
            connection = db_acquire_evaluator_lock()
        except Exception:
            raise _evaluator_store_unavailable() from None
        if connection is None:
            result["skippedLocked"] = 1
            return result

        try:
            rules = load_evaluator_candidates(connection, evaluation_now)
            result["rules"] = len(rules)
            if not rules:
                connection.commit()
                return result
            schedule_payload = load_facility_hours()
            reader = (
                snapshot_reader
                if snapshot_reader is not None
                else repository_factory(connection)
            )
        except Exception:
            _safe_rollback(connection)
            raise _evaluator_store_unavailable() from None

        claimed_any = False
        latest_evaluator_now = evaluation_now
        for rule in rules:
            if facility_filter is not None and rule.facility_id != facility_filter:
                continue
            if section_filter is not None and rule.section_key != section_filter:
                continue
            subscription = _decode_stored_subscription(rule.subscription_json)
            if subscription is None or _canonical_stored_endpoint(subscription) is None:
                result["skippedMissingSection"] += 1
                continue
            try:
                gate_now = _sample_evaluator_now(
                    now,
                    not_before=latest_evaluator_now,
                )
            except Exception:
                result["failed"] += 1
                continue
            latest_evaluator_now = gate_now
            if not _rule_schedule_gate(rule, schedule_payload, gate_now):
                result["skippedMissingSection"] += 1
                continue
            try:
                connection.commit()
                snapshot = reader.fetch_live_snapshot(gate_now)
                live_index = index_live_rows(snapshot.rows)
            except Exception:
                _safe_rollback(connection)
                raise _evaluator_store_unavailable() from None
            try:
                claim_now = _sample_evaluator_now(
                    now,
                    not_before=latest_evaluator_now,
                )
            except Exception:
                result["failed"] += 1
                continue
            latest_evaluator_now = claim_now
            final_status, percent = _rule_snapshot_gate(
                rule,
                schedule_payload,
                snapshot,
                live_index,
                claim_now,
            )
            if final_status == "missing":
                result["skippedMissingSection"] += 1
                continue
            if final_status == "threshold":
                result["skippedThreshold"] += 1
                continue
            if percent is None:
                result["failed"] += 1
                continue
            try:
                claimed = claim_pending_rule(
                    connection,
                    rule.id,
                    claim_now,
                )
            except Exception:
                _safe_rollback(connection)
                result["failed"] += 1
                continue
            if not claimed:
                continue
            try:
                connection.commit()
                claimed_any = True
            except Exception:
                _safe_rollback(connection)
                result["failed"] += 1
                continue

            section_label = (
                "entire facility"
                if rule.section_key == "overall"
                else rule.section_key
            )
            facility_label = FACILITY_NAMES.get(rule.facility_id, "Gym")
            terminal_status = "sent"
            failure_code = None
            try:
                send_notification_pinned(
                    subscription,
                    title="RecLive Alert",
                    body=(
                        f"{facility_label} {section_label} is {percent}% full "
                        f"(at or below your {rule.threshold}% alert)."
                    ),
                    url=facility_notification_url(rule.facility_id),
                    sent_at=claim_now.isoformat(),
                )
            except Exception as exc:
                terminal_status, failure_code = _provider_failure_state(exc)

            try:
                terminal_now = _sample_evaluator_now(
                    now,
                    not_before=latest_evaluator_now,
                )
            except Exception:
                result["failed"] += 1
                continue
            latest_evaluator_now = terminal_now
            try:
                if not finalize_claimed_rule(
                    connection,
                    rule.id,
                    terminal_now,
                    terminal_status,
                    failure_code,
                ):
                    raise RuntimeError("push terminal transition lost")
                connection.commit()
            except Exception:
                _safe_rollback(connection)
                result["failed"] += 1
                continue
            if terminal_status == "sent":
                result["sent"] += 1
            else:
                result["failed"] += 1

        if not claimed_any:
            try:
                connection.commit()
            except Exception:
                _safe_rollback(connection)
                raise _evaluator_store_unavailable() from None
        return result
    finally:
        if connection is not None:
            try:
                db_release_evaluator_lock(connection)
            except Exception:
                raise _evaluator_store_unavailable() from None


async def evaluator_loop() -> None:
    while True:
        try:
            if hasattr(asyncio, "to_thread"):
                await asyncio.to_thread(evaluate_rules_once)
            else:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, evaluate_rules_once)
        except Exception:
            print("[push-evaluator] error=push_evaluation_failed")
        await asyncio.sleep(max(30, EVALUATOR_INTERVAL_SECONDS))


class IgnoreExtraModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class StrictPushModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        populate_by_name=True,
    )


class PushKeysInput(StrictPushModel):
    p256dh: str = Field(min_length=1, max_length=512)
    auth: str = Field(min_length=1, max_length=512)


def _is_safe_expiration_time(value: object) -> bool:
    if value is None:
        return True
    if type(value) is int:
        return value >= 0
    if type(value) is float:
        return math.isfinite(value) and value >= 0
    return False


class PushSubscriptionInput(StrictPushModel):
    endpoint: str = Field(min_length=1)
    keys: PushKeysInput
    expiration_time: int | float | None = Field(
        default=None,
        alias="expirationTime",
    )

    @field_validator("expiration_time", mode="before")
    @classmethod
    def validate_expiration_time(cls, value: object) -> object:
        if not _is_safe_expiration_time(value):
            raise ValueError("expirationTime must be finite and nonnegative")
        return value


class PushRuleRequest(StrictPushModel):
    subscription: PushSubscriptionInput
    facility_id: Literal[1186, 1656] = Field(alias="facilityId")
    section_key: str = Field(
        alias="sectionKey",
        min_length=1,
        max_length=80,
    )
    threshold: int = Field(ge=1, le=100)
    ttl_seconds: int | None = Field(
        default=None,
        alias="ttlSeconds",
        ge=1,
        le=PUSH_MAX_RULE_TTL_SECONDS,
    )

    @model_validator(mode="after")
    def validate_configured_section(self) -> "PushRuleRequest":
        if normalize_section_key(self.section_key) != self.section_key:
            raise ValueError("sectionKey must already be canonical")
        if canonical_section_key(self.section_key) != self.section_key:
            raise ValueError("sectionKey must already be canonical")
        if not location_ids_for_section(self.facility_id, self.section_key):
            raise ValueError("sectionKey is not configured for the facility")
        return self


class PushOwnershipRequest(StrictPushModel):
    subscription: PushSubscriptionInput


@dataclass(frozen=True)
class ValidatedSubscription:
    endpoint: str
    subscription: Dict[str, Any]


PushModelT = TypeVar("PushModelT", bound=BaseModel)
BASE64URL_VALUE = re.compile(r"^[A-Za-z0-9_-]+={0,2}$")
MYSQL_UNSIGNED_BIGINT_MAX_TEXT = "18446744073709551615"


def _push_http_error(status_code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail)


def _content_length_value(request: Request) -> int | None:
    raw_values = [
        value
        for name, value in request.scope.get("headers", [])
        if bytes(name).lower() == b"content-length"
    ]
    if not raw_values:
        return None
    if len(raw_values) != 1:
        raise _push_http_error(422, "invalid_push_content_length")
    try:
        text = bytes(raw_values[0]).decode("ascii")
    except UnicodeDecodeError:
        raise _push_http_error(422, "invalid_push_content_length") from None
    if not re.fullmatch(r"[0-9]+", text):
        raise _push_http_error(422, "invalid_push_content_length")
    normalized = text.lstrip("0") or "0"
    maximum = str(PUSH_BODY_MAX_BYTES)
    if len(normalized) > len(maximum) or (
        len(normalized) == len(maximum) and normalized > maximum
    ):
        raise _push_http_error(413, "push_request_too_large")
    return int(normalized, 10)


def _reject_json_constant(_value: str) -> None:
    raise ValueError("nonstandard JSON constant")


async def _read_limited_push_json(request: Request) -> object:
    declared_size = _content_length_value(request)
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > PUSH_BODY_MAX_BYTES:
            raise _push_http_error(413, "push_request_too_large")
        chunks.append(chunk)
    if declared_size is not None and declared_size != size:
        raise _push_http_error(422, "invalid_push_content_length")
    try:
        text = b"".join(chunks).decode("utf-8")
        return json.loads(text, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise _push_http_error(422, "invalid_push_request") from None


def _validate_push_model(
    decoded: object,
    model_type: type[PushModelT],
) -> PushModelT:
    try:
        return model_type.model_validate(decoded)
    except (RecursionError, ValidationError):
        raise _push_http_error(422, "invalid_push_request") from None


async def parse_limited_push_body(
    request: Request,
    model_type: type[PushModelT],
) -> PushModelT:
    decoded = await _read_limited_push_json(request)
    return _validate_push_model(decoded, model_type)


def _decoded_subscription_endpoint(decoded: object) -> str | None:
    if not isinstance(decoded, Mapping):
        return None
    subscription = decoded.get("subscription")
    if isinstance(subscription, Mapping):
        endpoint = subscription.get("endpoint")
    else:
        endpoint = decoded.get("endpoint")
    return endpoint if type(endpoint) is str else None


def rate_limit_public_push_write(request: Request, decoded: object | None) -> None:
    endpoint = _decoded_subscription_endpoint(decoded)
    try:
        if endpoint is not None:
            try:
                subject_hash = rate_limit_subject_hash("endpoint", endpoint)
            except ValueError:
                subject_hash = None
        else:
            subject_hash = None
        if subject_hash is None:
            client_host = (
                request.client.host.strip()
                if request.client is not None and request.client.host.strip()
                else "unavailable"
            )
            subject_hash = rate_limit_subject_hash("client", client_host)
    except Exception:
        raise _push_http_error(503, "push_rate_limit_store_unavailable") from None

    conn = None
    try:
        sample = now_utc()
        if sample.tzinfo is None or sample.utcoffset() is None:
            raise ValueError("push clock must be timezone-aware")
        sample_utc = sample.astimezone(timezone.utc)
        epoch = int(sample_utc.timestamp())
        window_epoch = epoch - (epoch % PUSH_WRITE_RATE_WINDOW_SECONDS)
        window_started_at = datetime.fromtimestamp(
            window_epoch,
            tz=timezone.utc,
        ).replace(tzinfo=None)
        updated_at = sample_utc.replace(tzinfo=None)

        conn = open_db_connection(autocommit=False)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO push_rate_limits
                    (subject_hash, window_started_at, request_count, updated_at)
                VALUES (%s, %s, 1, %s)
                ON DUPLICATE KEY UPDATE
                    request_count = request_count + 1,
                    updated_at = VALUES(updated_at)
                """,
                (subject_hash, window_started_at, updated_at),
            )
            cur.execute(
                """
                SELECT request_count
                FROM push_rate_limits
                WHERE subject_hash = %s AND window_started_at = %s
                """,
                (subject_hash, window_started_at),
            )
            row = cur.fetchone()
            if not row or type(row[0]) is not int or row[0] < 1:
                raise RuntimeError("invalid push rate-limit counter result")
            request_count = row[0]
        conn.commit()
    except Exception:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        raise _push_http_error(503, "push_rate_limit_store_unavailable") from None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    if request_count > PUSH_WRITE_RATE_LIMIT:
        raise _push_http_error(429, "push_write_rate_limited")


async def _parse_limited_push_write(
    request: Request,
    model_type: type[PushModelT],
) -> PushModelT:
    decoded: object | None = None
    parse_error: HTTPException | None = None
    try:
        decoded = await _read_limited_push_json(request)
    except HTTPException as exc:
        parse_error = exc

    rate_limit_public_push_write(request, decoded)
    if parse_error is not None:
        raise parse_error
    return _validate_push_model(decoded, model_type)


def _parse_push_rule_id(raw_rule_id: str) -> int:
    if (
        type(raw_rule_id) is not str
        or re.fullmatch(r"[1-9][0-9]{0,19}", raw_rule_id) is None
        or (
            len(raw_rule_id) == len(MYSQL_UNSIGNED_BIGINT_MAX_TEXT)
            and raw_rule_id > MYSQL_UNSIGNED_BIGINT_MAX_TEXT
        )
    ):
        raise _push_http_error(422, "invalid_push_request")
    return int(raw_rule_id, 10)


def _strict_base64url_decode(value: object) -> bytes:
    if type(value) is not str or not 1 <= len(value) <= 512:
        raise ValueError("invalid base64url value")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError("invalid base64url value") from None
    if BASE64URL_VALUE.fullmatch(value) is None:
        raise ValueError("invalid base64url value")
    unpadded = encoded.rstrip(b"=")
    supplied_padding = len(encoded) - len(unpadded)
    required_padding = (-len(unpadded)) % 4
    if required_padding > 2 or supplied_padding not in {0, required_padding}:
        raise ValueError("invalid base64url padding")
    try:
        return base64.b64decode(
            unpadded + (b"=" * required_padding),
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError):
        raise ValueError("invalid base64url value") from None


def validate_push_subscription(
    value: Mapping[str, Any],
) -> ValidatedSubscription:
    allowed_fields = {"endpoint", "keys", "expirationTime"}
    if set(value) - allowed_fields or not {"endpoint", "keys"} <= set(value):
        raise _push_http_error(422, "invalid_push_subscription")
    expiration_time = value.get("expirationTime")
    if not _is_safe_expiration_time(expiration_time):
        raise _push_http_error(422, "invalid_push_subscription")

    endpoint_value = value.get("endpoint")
    keys_value = value.get("keys")
    if type(endpoint_value) is not str or not isinstance(keys_value, Mapping):
        raise _push_http_error(422, "invalid_push_subscription")
    if set(keys_value) != {"p256dh", "auth"}:
        raise _push_http_error(422, "invalid_push_subscription")
    p256dh_value = keys_value.get("p256dh")
    auth_value = keys_value.get("auth")
    try:
        endpoint = normalize_push_endpoint(endpoint_value)
        p256dh = _strict_base64url_decode(p256dh_value)
        auth = _strict_base64url_decode(auth_value)
    except (TypeError, ValueError):
        raise _push_http_error(422, "invalid_push_subscription") from None
    if len(p256dh) != 65 or p256dh[0] != 0x04 or len(auth) != 16:
        raise _push_http_error(422, "invalid_push_subscription")

    return ValidatedSubscription(
        endpoint=endpoint,
        subscription={
            "endpoint": endpoint,
            "keys": {
                "p256dh": p256dh_value,
                "auth": auth_value,
            },
        },
    )


def _validated_subscription_from_model(
    subscription: PushSubscriptionInput,
) -> ValidatedSubscription:
    return validate_push_subscription(subscription.model_dump(by_alias=True))


def subscribe_owned_push_rule(
    subscription: ValidatedSubscription,
    facility_id: int,
    section_key: str,
    threshold: int,
    ttl_seconds: int | None,
) -> Dict[str, Any]:
    created, rule = db_subscribe_rule(
        subscription,
        facility_id=facility_id,
        section_key=section_key,
        threshold=threshold,
        ttl_seconds=ttl_seconds,
    )
    try:
        response = push_rule_response(rule)
    except ValueError:
        raise _rule_store_unavailable() from None
    return {"status": "ok", "created": created, "rule": response}


def list_owned_push_rules(subscription: ValidatedSubscription) -> Dict[str, Any]:
    rules: List[Dict[str, Any]] = []
    for record in _db_list_owned_rule_records(subscription):
        try:
            rules.append(push_rule_response(record))
        except ValueError:
            continue
    return {"status": "ok", "rules": rules}


def cancel_owned_push_rule(
    subscription: ValidatedSubscription,
    rule_id: int,
) -> Dict[str, Any]:
    if type(rule_id) is not int or rule_id <= 0:
        raise _push_http_error(404, "push_rule_not_found")
    cancelled = db_cancel_owned_rule(subscription, rule_id)
    return {"status": "ok", "cancelled": cancelled}


def cancel_all_owned_push_rules(
    subscription: ValidatedSubscription,
) -> Dict[str, Any]:
    cancelled = db_cancel_all_owned_rules(subscription)
    return {"status": "ok", "cancelled": cancelled}


class PushDispatchRequest(StrictPushModel):
    facilityId: Optional[Literal[1186, 1656]] = None
    sectionKey: Optional[str] = Field(default=None, min_length=1, max_length=80)

    @model_validator(mode="after")
    def validate_filter(self) -> "PushDispatchRequest":
        if self.sectionKey is None:
            return self
        if self.facilityId is None:
            raise ValueError("facilityId is required with sectionKey")
        normalized = canonical_section_key(self.sectionKey)
        if (
            normalized != self.sectionKey
            or not location_ids_for_section(self.facilityId, normalized)
        ):
            raise ValueError("sectionKey must be configured")
        return self


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
    identity_configured = push_identity_configured()
    alerts_available = db_available and vapid_configured and identity_configured
    reason: Optional[str] = None
    if not db_available:
        reason = "push_rules_db_unavailable"
    elif not vapid_configured:
        reason = "push_vapid_unconfigured"
    elif not identity_configured:
        reason = "push_identity_unconfigured"
    return {
        "apiAvailable": True,
        "dbAvailable": db_available,
        "alertsAvailable": alerts_available,
        "reason": reason,
        "storeBackend": "db",
    }


@app.post("/api/push/subscribe")
async def subscribe(request: Request) -> Dict[str, Any]:
    payload = await _parse_limited_push_write(request, PushRuleRequest)
    subscription = _validated_subscription_from_model(payload.subscription)
    return subscribe_owned_push_rule(
        subscription=subscription,
        facility_id=payload.facility_id,
        section_key=payload.section_key,
        threshold=payload.threshold,
        ttl_seconds=payload.ttl_seconds,
    )


@app.post("/api/push/rules/list")
async def push_rules_list(request: Request) -> Dict[str, Any]:
    payload = await parse_limited_push_body(request, PushOwnershipRequest)
    subscription = _validated_subscription_from_model(payload.subscription)
    return list_owned_push_rules(subscription)


@app.delete("/api/push/rules/{rule_id}")
async def push_rule_cancel(rule_id: str, request: Request) -> Dict[str, Any]:
    payload = await _parse_limited_push_write(request, PushOwnershipRequest)
    parsed_rule_id = _parse_push_rule_id(rule_id)
    subscription = _validated_subscription_from_model(payload.subscription)
    return cancel_owned_push_rule(subscription, parsed_rule_id)


@app.post("/api/push/rules/cancel-all")
async def push_rules_cancel_all(request: Request) -> Dict[str, Any]:
    payload = await _parse_limited_push_write(request, PushOwnershipRequest)
    subscription = _validated_subscription_from_model(payload.subscription)
    return cancel_all_owned_push_rules(subscription)


@app.post("/api/push/dispatch")
async def dispatch(
    request: Request,
    _admin: None = Depends(require_admin_token),
) -> Dict[str, Any]:
    payload = await parse_limited_push_body(request, PushDispatchRequest)
    return evaluate_rules_once(
        facility_filter=payload.facilityId,
        section_filter=payload.sectionKey,
    )


@app.post("/api/push/evaluate")
def evaluate(_admin: None = Depends(require_admin_token)) -> Dict[str, Any]:
    return evaluate_rules_once()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("forecast_api:app", host=API_HOST, port=API_PORT, reload=False)
