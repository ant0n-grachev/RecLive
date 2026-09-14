from __future__ import annotations

import sys as _import_sys
import json
import copy
import html as html_lib
import os
import tempfile
from urllib.parse import urlsplit
import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from server.reclive.settings import (
    SERVER_ROOT,
    Settings,
    validate_command_environment,
    resolve_command_path as resolve_path,  # noqa: F401 - historical export
)
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from fastapi import HTTPException
from server.reclive.runtime import current_runtime
from server.reclive import runtime as _owner_runtime
from datetime import date
from zoneinfo import ZoneInfo

SCRIPT_DIR = str(SERVER_ROOT)
CHICAGO_TZ = ZoneInfo("America/Chicago")
MINUTES_PER_DAY = 24 * 60
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_DATE_LIKE_PATTERN = "(?:\\b\\d{4}-\\d{2}-\\d{2}\\b|\\b\\d{1,2}/\\d{1,2}(?:/\\d{2,4})?\\b|\\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\\s+\\d{1,2}\\b)"
_DATE_LIKE = re.compile(_DATE_LIKE_PATTERN)
_DAY_NAME_PATTERN = "mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:r|rs|rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?"
_DAY_TOKEN = re.compile(f"\\b({_DAY_NAME_PATTERN})\\b")
_DAY_RANGE = re.compile(f"\\b({_DAY_NAME_PATTERN})\\s*-\\s*({_DAY_NAME_PATTERN})\\b")
_FACILITY_NAME_PATTERN = "(?:nick|nicholas recreation center|bakke|bakke recreation (?:&|and) wellbeing center)"
_TITLE_DATE_TOKEN_PATTERN = "(?:\\d{4}-\\d{2}-\\d{2}|\\d{1,2}/\\d{1,2}(?:/\\d{2,4})?|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\\s+\\d{1,2}(?:,\\s*\\d{4})?)"
_TITLE_DATE_RANGE_PATTERN = f"{_TITLE_DATE_TOKEN_PATTERN}(?:\\s*-\\s*(?:{_TITLE_DATE_TOKEN_PATTERN}|\\d{{1,2}}))?"
_EXACT_DATE_RANGE = re.compile(
    f"^({_TITLE_DATE_TOKEN_PATTERN})(?:\\s*-\\s*({_TITLE_DATE_TOKEN_PATTERN}|\\d{{1,2}}))?$"
)
_TITLE_SUFFIX_PATTERN = f"(?:\\s*(?::|-)\\s*{_TITLE_DATE_RANGE_PATTERN}|\\s*\\({_TITLE_DATE_RANGE_PATTERN}\\))?"
_BUILDING_SECTION_TITLE = re.compile(
    f"^(?:(?:{_FACILITY_NAME_PATTERN})\\s+)?(?:building hours|facility hours|hours of operation|facility schedule){_TITLE_SUFFIX_PATTERN}$|^(?:{_FACILITY_NAME_PATTERN})\\s+hours{_TITLE_SUFFIX_PATTERN}$"
)
_MAINTENANCE_SECTION_TITLE = re.compile(
    f"^(?:(?:{_FACILITY_NAME_PATTERN})\\s+)?maintenance closures?{_TITLE_SUFFIX_PATTERN}$"
)


def _normalize(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return re.sub(
        "\\s+", " ", value.lower().replace("–", "-").replace("—", "-")
    ).strip()


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except (TypeError, ValueError, OverflowError):
        return None


def _parse_date_token(
    token: str, fallback_year: int, fallback_month: int | None = None
) -> date | None:
    normalized = _normalize(token)
    if not normalized:
        return None
    match = re.fullmatch("(\\d{4})-(\\d{2})-(\\d{2})", normalized)
    if match:
        return _safe_date(*(int(part) for part in match.groups()))
    match = re.fullmatch("(\\d{1,2})/(\\d{1,2})(?:/(\\d{2,4}))?", normalized)
    if match:
        year = fallback_year
        if match.group(3):
            year = int(match.group(3))
            if year < 100:
                year += 2000
        return _safe_date(year, int(match.group(1)), int(match.group(2)))
    match = re.fullmatch("([a-z]+)\\s+(\\d{1,2})(?:,\\s*(\\d{4}))?", normalized)
    if match:
        month = _MONTHS.get(match.group(1))
        if month is None:
            return None
        year = int(match.group(3)) if match.group(3) else fallback_year
        return _safe_date(year, month, int(match.group(2)))
    match = re.fullmatch("(\\d{1,2})", normalized)
    if match and fallback_month is not None:
        return _safe_date(fallback_year, fallback_month, int(match.group(1)))
    return None


def parse_schedule_date_range(
    value: str, fallback_year: int
) -> Optional[tuple[date, date, int]]:
    normalized = _normalize(value)
    if (
        not normalized
        or type(fallback_year) is not int
        or (not 1 <= fallback_year <= 9999)
        or (not _DATE_LIKE.search(normalized))
    ):
        return None
    first_date = _DATE_LIKE.search(normalized)
    if first_date is None:
        return None
    date_text = normalized[first_date.start() :].strip(" :()[]")
    match = _EXACT_DATE_RANGE.fullmatch(date_text)
    if match is None:
        return None
    start_token, end_token = match.groups()
    start = _parse_date_token(start_token, fallback_year)
    if start is None:
        return None
    end = start
    if end_token is not None:
        end = _parse_date_token(end_token, start.year, start.month)
        if end is None:
            end = _parse_date_token(end_token, fallback_year)
        if end is None:
            return None
    if end < start:
        if end_token is None:
            return None
        shifted = _parse_date_token(end_token, start.year + 1, start.month)
        if shifted is None or shifted < start:
            return None
        end = shifted
    span = (end - start).days + 1
    return (start, end, span) if span > 0 else None


def _weekday_index(token: str) -> int | None:
    normalized = _normalize(token)
    if normalized is None:
        return None
    for prefix, index in (
        ("mon", 0),
        ("tue", 1),
        ("wed", 2),
        ("thu", 3),
        ("fri", 4),
        ("sat", 5),
        ("sun", 6),
    ):
        if normalized.startswith(prefix):
            return index
    return None


def parse_schedule_weekday_set(label: str) -> Optional[set[int]]:
    normalized = _normalize(label)
    if not normalized:
        return None
    if re.fullmatch("(?:open\\s+)?daily", normalized):
        return set(range(7))
    if re.fullmatch("(?:open\\s+)?weekdays", normalized):
        return {0, 1, 2, 3, 4}
    if re.fullmatch("(?:open\\s+)?weekends", normalized):
        return {5, 6}
    range_match = _DAY_RANGE.fullmatch(normalized)
    if range_match:
        start = _weekday_index(range_match.group(1))
        end = _weekday_index(range_match.group(2))
        if start is None or end is None:
            return None
        output: set[int] = set()
        index = start
        while True:
            output.add(index)
            if index == end:
                break
            index = (index + 1) % 7
        return output
    if (
        re.fullmatch(
            f"(?:{_DAY_NAME_PATTERN})(?:\\s*(?:,|/|&|\\band\\b)\\s*(?:{_DAY_NAME_PATTERN}))*",
            normalized,
        )
        is None
    ):
        return None
    output = {
        index
        for match in _DAY_TOKEN.finditer(normalized)
        if (index := _weekday_index(match.group(1))) is not None
    }
    return output or None


def _parse_clock(token: str, *, is_end: bool) -> int | None:
    normalized = _normalize(token)
    if not normalized:
        return None
    normalized = normalized.replace(".", "")
    if normalized == "midnight":
        return MINUTES_PER_DAY if is_end else 0
    if normalized == "noon":
        return 12 * 60
    match = re.fullmatch("(\\d{1,2})(?::(\\d{2}))?\\s*(am|pm)?", normalized)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    suffix = match.group(3)
    if minute > 59:
        return None
    if suffix is not None:
        if not 1 <= hour <= 12:
            return None
        if hour == 12:
            hour = 0
        if suffix == "pm":
            hour += 12
    elif hour > 24 or (hour == 24 and minute != 0):
        return None
    return hour * 60 + minute


def parse_schedule_hours_window(value: str) -> Optional[tuple[int, int, bool]]:
    normalized = _normalize(value)
    if not normalized:
        return None
    if re.fullmatch("closed(?:\\s+.*)?", normalized):
        return (0, 0, True)
    if re.fullmatch("(?:open\\s+)?24\\s*(?:hours?|hrs?)", normalized):
        return (0, MINUTES_PER_DAY, False)
    parts = [part.strip() for part in re.split("\\s*-\\s*", normalized)]
    if len(parts) != 2 and " to " in normalized:
        parts = [part.strip() for part in normalized.split(" to ", 1)]
    if len(parts) != 2 or any((not part for part in parts)):
        return None
    start = _parse_clock(parts[0], is_end=False)
    end = _parse_clock(parts[1], is_end=True)
    if (
        start is None
        or end is None
        or start >= MINUTES_PER_DAY
        or (start % MINUTES_PER_DAY == end % MINUTES_PER_DAY)
    ):
        return None
    if end < start:
        end += MINUTES_PER_DAY
    if end - start > MINUTES_PER_DAY:
        return None
    return (start, end, False)


@dataclass(frozen=True)
class _ScheduleCandidate:
    rank: tuple[int, int, int, int]
    window: tuple[int, int, bool] | None
    explicit_date: bool
    maintenance: bool


def _section_kind(title: object) -> str | None:
    normalized = _normalize(title)
    if not normalized:
        return None
    if _MAINTENANCE_SECTION_TITLE.fullmatch(normalized):
        return "maintenance"
    if _BUILDING_SECTION_TITLE.fullmatch(normalized):
        return "building"
    return None


def _ranges(value: str, anchor: date) -> list[tuple[date, date, int]]:
    output: list[tuple[date, date, int]] = []
    for year in (anchor.year, anchor.year - 1):
        parsed = parse_schedule_date_range(value, year)
        if parsed is not None and parsed not in output:
            output.append(parsed)
    return output


def _select_candidate(
    sections: list[dict[str, object]], anchor: date, *, section_kind: str
) -> _ScheduleCandidate | None:
    candidates: list[_ScheduleCandidate] = []
    for section_index, section in enumerate(sections):
        if not isinstance(section, Mapping):
            continue
        kind = _section_kind(section.get("title"))
        if kind != section_kind:
            continue
        title = section.get("title")
        if not isinstance(title, str):
            continue
        title_ranges = _ranges(title, anchor)
        normalized_title = _normalize(title)
        if (
            normalized_title is not None
            and _DATE_LIKE.search(normalized_title)
            and (not title_ranges)
        ):
            candidates.append(
                _ScheduleCandidate(
                    rank=(-3, 0, section_index, -1),
                    window=None,
                    explicit_date=True,
                    maintenance=kind == "maintenance",
                )
            )
            continue
        title_match = next(
            (value for value in title_ranges if value[0] <= anchor <= value[1]), None
        )
        if title_ranges and title_match is None:
            continue
        rows = section.get("rows")
        if not isinstance(rows, list):
            continue
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            label = row.get("label")
            hours = row.get("hours")
            if not isinstance(label, str) or not isinstance(hours, str):
                continue
            row_ranges = _ranges(label, anchor)
            normalized_label = _normalize(label)
            if (
                normalized_label is not None
                and _DATE_LIKE.search(normalized_label)
                and (not row_ranges)
            ):
                candidates.append(
                    _ScheduleCandidate(
                        rank=(-3, 0, section_index, row_index),
                        window=None,
                        explicit_date=True,
                        maintenance=kind == "maintenance",
                    )
                )
                continue
            row_match = next(
                (value for value in row_ranges if value[0] <= anchor <= value[1]), None
            )
            if row_ranges:
                if row_match is None:
                    continue
            else:
                weekdays = parse_schedule_weekday_set(label)
                if weekdays is None:
                    candidates.append(
                        _ScheduleCandidate(
                            rank=(-3, 0, section_index, row_index),
                            window=None,
                            explicit_date=False,
                            maintenance=kind == "maintenance",
                        )
                    )
                    continue
                if anchor.weekday() not in weekdays:
                    continue
            date_scopes = [
                value for value in (title_match, row_match) if value is not None
            ]
            if date_scopes:
                effective_start = max((value[0] for value in date_scopes))
                effective_end = min((value[1] for value in date_scopes))
                if effective_end < effective_start:
                    continue
                specificity = 2
                span = (effective_end - effective_start).days + 1
                explicit_date = True
            else:
                specificity = 1
                span = 999999
                explicit_date = False
            window = parse_schedule_hours_window(hours)
            if kind == "maintenance" and (window is None or window[2] is not True):
                window = None
            candidates.append(
                _ScheduleCandidate(
                    rank=(-specificity, span, section_index, row_index),
                    window=window,
                    explicit_date=explicit_date,
                    maintenance=kind == "maintenance",
                )
            )
    return min(candidates, key=lambda candidate: candidate.rank, default=None)


def get_facility_schedule_open_state(
    sections: list[dict[str, object]], at: datetime
) -> Optional[bool]:
    if (
        not isinstance(sections, list)
        or not sections
        or (not isinstance(at, datetime))
        or (at.tzinfo is None)
        or (at.utcoffset() is None)
    ):
        return None
    for section in sections:
        if (
            not isinstance(section, Mapping)
            or not isinstance(section.get("title"), str)
            or (not isinstance(section.get("rows"), list))
        ):
            return None
        rows = section.get("rows")
        if not isinstance(rows, list):
            return None
        if any(
            (
                not isinstance(row, Mapping)
                or not isinstance(row.get("label"), str)
                or (not isinstance(row.get("hours"), str))
                for row in rows
            )
        ):
            return None
    local = at.astimezone(timezone.utc).astimezone(CHICAGO_TZ)
    minute = local.hour * 60 + local.minute
    current_maintenance = _select_candidate(
        sections, local.date(), section_kind="maintenance"
    )
    if current_maintenance is not None:
        if current_maintenance.window is None:
            return None
        if current_maintenance.window[2]:
            return False
        return None
    current = _select_candidate(sections, local.date(), section_kind="building")
    previous = _select_candidate(
        sections, local.date() - timedelta(days=1), section_kind="building"
    )
    if current is not None:
        if current.window is None:
            return None
        start, end, closed = current.window
        if closed and current.explicit_date:
            return False
        if not closed and start <= minute < min(end, MINUTES_PER_DAY):
            return True
    if previous is not None:
        if previous.window is None:
            return None if current is None else False
        _start, end, closed = previous.window
        if not closed and end > MINUTES_PER_DAY and (minute < end - MINUTES_PER_DAY):
            return True
    if current is not None:
        return False
    return None


def parse_utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except (ValueError, TypeError, OverflowError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def official_facility_is_open(
    payload: Mapping[str, Any],
    facility_id: int,
    at: datetime,
    *,
    stale_after_seconds: int = 21600,
) -> bool:
    if (
        not isinstance(payload, Mapping)
        or type(facility_id) is not int
        or facility_id not in {1186, 1656}
        or (not isinstance(at, datetime))
        or (at.tzinfo is None)
        or (at.utcoffset() is None)
        or (type(stale_after_seconds) is not int)
        or (stale_after_seconds <= 0)
    ):
        return False
    facilities = payload.get("facilities")
    if (
        not isinstance(facilities, list)
        or not facilities
        or any(
            (
                not isinstance(item, Mapping) or type(item.get("facilityId")) is not int
                for item in facilities
            )
        )
    ):
        return False
    matches = [
        item
        for item in facilities
        if isinstance(item, Mapping)
        and type(item.get("facilityId")) is int
        and (item.get("facilityId") == facility_id)
    ]
    if len(matches) != 1:
        return False
    facility = matches[0]
    if facility.get("status") != "ok" or facility.get("stale") is not False:
        return False
    now = at.astimezone(timezone.utc)
    for raw_timestamp in (payload.get("generatedAt"), facility.get("lastSuccessfulAt")):
        observed = parse_utc_timestamp(raw_timestamp)
        if observed is None:
            return False
        age = (now - observed).total_seconds()
        if age < 0 or age > stale_after_seconds:
            return False
    sections = facility.get("sections")
    if not isinstance(sections, list) or not sections:
        return False
    return get_facility_schedule_open_state(sections, at) is True


def load_facility_hours() -> Dict[str, Any]:
    try:
        with open(
            current_runtime().settings.facility_hours_json_path, "r", encoding="utf-8"
        ) as handle:
            payload = json.load(handle)
        return validate_schedule_payload(payload, now=_owner_runtime.now_utc())
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        raise HTTPException(status_code=503, detail="schedule_unavailable") from None


def schedule_health(payload: Mapping[str, Any], now: datetime) -> Dict[str, Any]:
    generated_at = parse_utc_timestamp(payload.get("generatedAt"))
    raw_age_seconds = (
        None
        if generated_at is None
        else max(0.0, (now.astimezone(timezone.utc) - generated_at).total_seconds())
    )
    age_seconds = None if raw_age_seconds is None else int(raw_age_seconds)
    facilities = payload.get("facilities")
    statuses = (
        {
            str(row["facilityId"]): str(row["status"])
            for row in facilities
            if isinstance(row, Mapping) and "facilityId" in row and ("status" in row)
        }
        if isinstance(facilities, list)
        else {}
    )
    if age_seconds is None or (
        statuses and all((status == "error" for status in statuses.values()))
    ):
        state = "unavailable"
    elif (
        raw_age_seconds <= current_runtime().settings.schedule_stale_after_seconds
        and statuses
        and all((status == "ok" for status in statuses.values()))
    ):
        state = "healthy"
    else:
        state = "stale"
    return {"state": state, "ageSeconds": age_seconds, "facilities": statuses}


def _parse_facility_id(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_facility_hours_entry(
    payload: Dict[str, Any], facility_id: int
) -> Dict[str, Any]:
    facilities = payload.get("facilities", [])
    for item in facilities:
        if not isinstance(item, dict):
            continue
        row_facility_id = _parse_facility_id(item.get("facilityId"))
        if row_facility_id == facility_id:
            return item
    raise HTTPException(status_code=404, detail="Facility schedule not found")


# Official schedule collection, validation and atomic publication.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

DEFAULT_OUTPUT_FILE = "facility_hours.json"
DEFAULT_SITE_BASE = "https://recwell.wisc.edu"
DEFAULT_FACILITIES = [
    {
        "facilityId": 1186,
        "slug": "nick",
        "facilityName": "Nick",
        "url": "https://recwell.wisc.edu/locations/nick/",
    },
    {
        "facilityId": 1656,
        "slug": "bakke",
        "facilityName": "Bakke",
        "url": "https://recwell.wisc.edu/locations/bakke/",
    },
]
SUPPORTED_FACILITY_IDENTITIES = {
    1186: ("Nick", "nick"),
    1656: ("Bakke", "bakke"),
}

DAY_HINT_RE = re.compile(
    r"\b("
    r"mon(day)?|tue(s|sday)?|wed(nesday)?|thu(r|rs|rsday)?|fri(day)?|"
    r"sat(urday)?|sun(day)?|daily|weekdays?|weekends?"
    r")\b",
    re.IGNORECASE,
)
DATE_LABEL_RE = re.compile(
    r"\b("
    r"date|dates|"
    r"jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|"
    r"aug(ust)?|sep(t|tember)?|oct(ober)?|nov(ember)?|dec(ember)?|"
    r"\d{1,2}/\d{1,2}(/\d{2,4})?|"
    r"\d{4}-\d{2}-\d{2}"
    r")\b",
    re.IGNORECASE,
)
HOURS_HINT_RE = re.compile(
    r"(am|pm|closed|24\s*hours?|noon|midnight|\d{1,2}(:\d{2})?\s*(am|pm)?)",
    re.IGNORECASE,
)
NOTICE_HOURS_RE = re.compile(
    r"check back later for [^.]+? hours\.",
    re.IGNORECASE,
)
NOTICE_MAINTENANCE_RE = re.compile(
    r"(?:no scheduled maintenance closures at this time\.?|"
    r"maintenance closure:\s*[^.\n]{1,240}\.)",
    re.IGNORECASE,
)
TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
DEFINITION_LIST_RE = re.compile(
    r"<dl\b[^>]*>(.*?)</dl>",
    re.IGNORECASE | re.DOTALL,
)
DEFINITION_PAIR_RE = re.compile(
    r"<dt\b[^>]*>(.*?)</dt>(?:(?!<dt\b).)*?<dd\b[^>]*>(.*?)</dd>",
    re.IGNORECASE | re.DOTALL,
)
ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
CELL_RE = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.IGNORECASE | re.DOTALL)
HEADING_RE = re.compile(r"<h([2-5])[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")

SCHEDULE_ERROR_CATEGORIES = frozenset(
    {
        "anti_bot",
        "upstream_timeout",
        "upstream_http",
        "wp_payload_invalid",
        "parse_empty",
        "schema_invalid",
        "io_error",
    }
)
SCHEDULE_SOURCES = frozenset({"direct_html", "wp_json"})
SCHEDULE_REFRESH_ERROR = "Official hours could not be refreshed."
FACILITY_RECORD_KEYS = frozenset(
    {
        "facilityId",
        "facilityName",
        "slug",
        "url",
        "resolvedUrl",
        "status",
        "source",
        "sourceModifiedGmt",
        "sections",
        "sourceFetchedAt",
        "lastSuccessfulAt",
        "stale",
        "error",
        "errorCategory",
        "updatedAt",
    }
)
FACILITY_RECORD_REQUIRED_KEYS = FACILITY_RECORD_KEYS - {"sourceModifiedGmt"}
SCHEDULE_PAYLOAD_KEYS = frozenset(
    {"generatedAt", "sourceSite", "facilities", "okCount", "totalCount"}
)
MAX_SCHEDULE_SECTIONS = 32
MAX_SCHEDULE_ROWS = 128
MAX_SCHEDULE_URL_LENGTH = 2_048
UNSAFE_URL_CHARACTER_RE = re.compile(r"[\x00-\x20\x7f\\]")
UNSAFE_ENCODED_URL_CHARACTER_RE = re.compile(
    r"%(?:0[0-9a-f]|1[0-9a-f]|7f|5c)",
    re.IGNORECASE,
)
LOCAL_ISO_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?$"
)


class ScheduleFetchError(RuntimeError):
    def __init__(self, category: str) -> None:
        if not isinstance(category, str) or category not in SCHEDULE_ERROR_CATEGORIES:
            raise ValueError("unsupported schedule error category")
        self.category = category
        super().__init__(category)


def safe_error_message(category: str) -> str:
    """Return the sole user-visible refresh error without reflecting input."""

    return SCHEDULE_REFRESH_ERROR


def _bounded_text(value: object, maximum: int) -> bool:
    return (
        isinstance(value, str)
        and value == value.strip()
        and 0 < len(value) <= maximum
        and "<" not in value
        and ">" not in value
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _https_origin(value: object) -> Optional[Tuple[str, int]]:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_SCHEDULE_URL_LENGTH
        or UNSAFE_URL_CHARACTER_RE.search(value)
        or UNSAFE_ENCODED_URL_CHARACTER_RE.search(value)
        or "?" in value
        or "#" in value
    ):
        return None
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError):
        return None
    if (
        parsed.scheme.lower() != "https"
        or not parsed.netloc
        or parsed.netloc.rsplit("@", 1)[-1].endswith(":")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return None
    hostname = parsed.hostname
    if (
        not isinstance(hostname, str)
        or not hostname
        or len(hostname) > 253
        or hostname.startswith(".")
        or hostname.endswith(".")
        or ".." in hostname
        or not re.fullmatch(r"[a-zA-Z0-9.-]+", hostname)
    ):
        return None
    labels = hostname.split(".")
    if any(
        not label
        or label.startswith("-")
        or label.endswith("-")
        or len(label) > 63
        for label in labels
    ):
        return None
    return hostname.lower(), port if port is not None else 443


def safe_same_origin_https_url(value: object, site_base: object) -> Optional[str]:
    """Return an unchanged credential-free URL only for the configured origin."""

    value_origin = _https_origin(value)
    base_origin = _https_origin(site_base)
    if value_origin is None or base_origin is None or value_origin != base_origin:
        return None
    return value if isinstance(value, str) else None


def _safe_current_public_url(value: object, current_url: str) -> Optional[str]:
    safe_url = safe_same_origin_https_url(value, current_url)
    if safe_url is None:
        return None
    if urlsplit(safe_url).path != urlsplit(current_url).path:
        return None
    return safe_url


def _valid_local_iso_timestamp(value: object) -> bool:
    if (
        not isinstance(value, str)
        or len(value) > 48
        or LOCAL_ISO_TIMESTAMP_RE.fullmatch(value) is None
    ):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is None


def _copy_valid_sections(value: object) -> Optional[List[Dict[str, object]]]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or len(value) > MAX_SCHEDULE_SECTIONS
    ):
        return None

    copied_sections: List[Dict[str, object]] = []
    for section in value:
        if not isinstance(section, Mapping):
            return None
        section_keys = set(section.keys())
        if not {"title", "rows"}.issubset(section_keys) or not section_keys.issubset(
            {"title", "rows", "note"}
        ):
            return None
        title = section.get("title")
        rows = section.get("rows")
        note = section.get("note") if "note" in section else None
        if (
            not _bounded_text(title, 160)
            or not isinstance(rows, (list, tuple))
            or len(rows) > MAX_SCHEDULE_ROWS
            or (note is not None and not _bounded_text(note, 500))
            or (not rows and note is None)
        ):
            return None

        copied_rows: List[Dict[str, str]] = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row.keys()) != {"label", "hours"}:
                return None
            label = row.get("label")
            hours_value = row.get("hours")
            if not _bounded_text(label, 160) or not _bounded_text(hours_value, 240):
                return None
            copied_rows.append({"label": label, "hours": hours_value})

        copied_section: Dict[str, object] = {
            "title": title,
            "rows": copied_rows,
        }
        if "note" in section:
            copied_section["note"] = note
        copied_sections.append(copied_section)
    return copied_sections


def _parse_canonical_utc_timestamp(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    normalized = parsed.astimezone(timezone.utc)
    if normalized.isoformat().replace("+00:00", "Z") != value:
        return None
    return normalized


def _aware_datetime(value: object) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() is not None
    )


@dataclass(frozen=True)
class FacilityCandidate:
    facility_id: int
    facility_name: str
    slug: str
    public_url: str
    source: Optional[str]
    resolved_url: Optional[str]
    source_modified_gmt: Optional[str]
    sections: Tuple[Dict[str, object], ...]
    fetched_at: datetime
    error_category: Optional[str]

    def __post_init__(self) -> None:
        if (
            type(self.facility_id) is not int
            or self.facility_id not in SUPPORTED_FACILITY_IDENTITIES
            or not _bounded_text(self.facility_name, 160)
            or not _bounded_text(self.slug, 80)
            or (
                self.facility_name,
                self.slug,
            )
            != SUPPORTED_FACILITY_IDENTITIES.get(self.facility_id)
            or _https_origin(self.public_url) is None
            or not _aware_datetime(self.fetched_at)
            or (
                self.error_category is not None
                and self.error_category not in SCHEDULE_ERROR_CATEGORIES
            )
        ):
            raise ValueError("invalid facility candidate")

        copied_sections = _copy_valid_sections(self.sections) if self.sections else []
        if self.source in SCHEDULE_SOURCES:
            if (
                self.error_category is not None
                or not copied_sections
                or (
                    self.resolved_url is not None
                    and safe_same_origin_https_url(
                        self.resolved_url,
                        self.public_url,
                    )
                    is None
                )
                or (
                    self.source == "direct_html"
                    and self.source_modified_gmt is not None
                )
                or (
                    self.source_modified_gmt is not None
                    and not _valid_local_iso_timestamp(self.source_modified_gmt)
                )
            ):
                raise ValueError("invalid facility candidate")
        elif (
            self.source is not None
            or self.error_category not in SCHEDULE_ERROR_CATEGORIES
            or self.resolved_url is not None
            or self.source_modified_gmt is not None
            or self.sections
        ):
            raise ValueError("invalid facility candidate")

        object.__setattr__(
            self,
            "sections",
            tuple(copy.deepcopy(copied_sections)),
        )

    @property
    def ok(self) -> bool:
        return (
            self.error_category is None
            and self.source in SCHEDULE_SOURCES
            and bool(self.sections)
        )

    def as_public_record(self) -> Dict[str, object]:
        if not self.ok:
            raise ValueError("failed schedule candidate has no public record")
        return {
            "facilityId": self.facility_id,
            "facilityName": self.facility_name,
            "slug": self.slug,
            "url": self.public_url,
            "resolvedUrl": self.resolved_url,
            "source": self.source,
            "sourceModifiedGmt": self.source_modified_gmt,
            "sections": copy.deepcopy(list(self.sections)),
        }


def env_with_default(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip()
    return value if value else default


def iso_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return iso_utc(now_utc())


def _copy_valid_previous_facility(
    previous: object,
    candidate: FacilityCandidate,
    generated_at: datetime,
) -> Optional[Dict[str, object]]:
    if (
        not isinstance(previous, Mapping)
        or not FACILITY_RECORD_REQUIRED_KEYS.issubset(previous.keys())
        or not set(previous.keys()).issubset(FACILITY_RECORD_KEYS)
        or type(previous.get("facilityId")) is not int
        or previous.get("facilityId") != candidate.facility_id
        or previous.get("facilityName") != candidate.facility_name
        or previous.get("slug") != candidate.slug
        or not _bounded_text(previous.get("facilityName"), 160)
    ):
        return None

    public_url = _safe_current_public_url(
        previous.get("url"),
        candidate.public_url,
    )
    resolved_value = previous.get("resolvedUrl")
    resolved_url = (
        None
        if resolved_value is None
        else safe_same_origin_https_url(resolved_value, candidate.public_url)
    )
    if public_url is None or (resolved_value is not None and resolved_url is None):
        return None

    status = previous.get("status")
    stale = previous.get("stale")
    error = previous.get("error")
    error_category = previous.get("errorCategory")
    source = previous.get("source")
    source_modified_gmt = previous.get("sourceModifiedGmt")
    sections = _copy_valid_sections(previous.get("sections"))
    if (
        status not in {"ok", "stale"}
        or source not in SCHEDULE_SOURCES
        or sections is None
        or (
            source == "direct_html"
            and source_modified_gmt is not None
        )
        or (
            source_modified_gmt is not None
            and not _valid_local_iso_timestamp(source_modified_gmt)
        )
        or (
            status == "ok"
            and (
                stale is not False
                or error is not None
                or error_category is not None
            )
        )
        or (
            status == "stale"
            and (
                stale is not True
                or error != SCHEDULE_REFRESH_ERROR
                or error_category not in SCHEDULE_ERROR_CATEGORIES
            )
        )
    ):
        return None

    generated_utc = generated_at.astimezone(timezone.utc)
    parsed_timestamps = [
        _parse_canonical_utc_timestamp(previous.get(key))
        for key in ("sourceFetchedAt", "lastSuccessfulAt", "updatedAt")
    ]
    if any(
        parsed is None or parsed > generated_utc
        for parsed in parsed_timestamps
    ):
        return None

    return {
        "facilityId": candidate.facility_id,
        "facilityName": previous["facilityName"],
        "slug": candidate.slug,
        "url": public_url,
        "resolvedUrl": resolved_url,
        "status": status,
        "source": source,
        "sourceModifiedGmt": source_modified_gmt,
        "sections": sections,
        "sourceFetchedAt": previous["sourceFetchedAt"],
        "lastSuccessfulAt": previous["lastSuccessfulAt"],
        "stale": stale,
        "error": error,
        "errorCategory": error_category,
        "updatedAt": previous["updatedAt"],
    }


def previous_is_valid_facility(
    previous: object,
    facility_id: int,
) -> bool:
    if type(facility_id) is not int or facility_id not in SUPPORTED_FACILITY_IDENTITIES:
        return False
    if not isinstance(previous, Mapping):
        return False
    facility_name, slug = SUPPORTED_FACILITY_IDENTITIES[facility_id]
    public_url = previous.get("url")
    if _https_origin(public_url) is None or not isinstance(public_url, str):
        return False
    try:
        candidate = FacilityCandidate(
            facility_id=facility_id,
            facility_name=facility_name,
            slug=slug,
            public_url=public_url,
            source=None,
            resolved_url=None,
            source_modified_gmt=None,
            sections=(),
            fetched_at=datetime.now(timezone.utc),
            error_category="schema_invalid",
        )
    except ValueError:
        return False
    return (
        _copy_valid_previous_facility(
            previous,
            candidate,
            datetime.now(timezone.utc),
        )
        is not None
    )


def merge_facility_candidate(
    candidate: FacilityCandidate,
    previous: Optional[Mapping[str, Any]],
    generated_at: datetime,
) -> Dict[str, object]:
    if not isinstance(candidate, FacilityCandidate) or not _aware_datetime(generated_at):
        raise ValueError("invalid schedule merge input")
    generated_utc = generated_at.astimezone(timezone.utc)
    candidate_fetched_utc = candidate.fetched_at.astimezone(timezone.utc)

    if candidate.ok and candidate_fetched_utc <= generated_utc:
        return {
            **candidate.as_public_record(),
            "status": "ok",
            "sourceFetchedAt": iso_utc(candidate.fetched_at),
            "lastSuccessfulAt": iso_utc(candidate.fetched_at),
            "stale": False,
            "error": None,
            "errorCategory": None,
            "updatedAt": iso_utc(generated_at),
        }

    error_category = (
        candidate.error_category
        if candidate.error_category in SCHEDULE_ERROR_CATEGORIES
        else "schema_invalid"
    )
    previous_copy = _copy_valid_previous_facility(
        previous,
        candidate,
        generated_at,
    )
    if previous_copy is not None:
        previous_copy.update(
            {
                "status": "stale",
                "stale": True,
                "error": safe_error_message(error_category),
                "errorCategory": error_category,
                "updatedAt": iso_utc(generated_at),
            }
        )
        return previous_copy

    return {
        "facilityId": candidate.facility_id,
        "facilityName": candidate.facility_name,
        "slug": candidate.slug,
        "url": candidate.public_url,
        "resolvedUrl": None,
        "status": "error",
        "source": None,
        "sourceModifiedGmt": None,
        "sections": [],
        "sourceFetchedAt": None,
        "lastSuccessfulAt": None,
        "stale": False,
        "error": safe_error_message(error_category),
        "errorCategory": error_category,
        "updatedAt": iso_utc(generated_at),
    }


def _copy_json_schedule_sections(value: object) -> Optional[List[Dict[str, object]]]:
    if not isinstance(value, list):
        return None
    for section in value:
        if (
            not isinstance(section, dict)
            or not isinstance(section.get("rows"), list)
            or any(not isinstance(row, dict) for row in section.get("rows", []))
        ):
            return None
    return _copy_valid_sections(value)


def _safe_source_site(value: object) -> Optional[str]:
    if _https_origin(value) is None or not isinstance(value, str):
        return None
    if urlsplit(value).path not in {"", "/"}:
        return None
    return value


def _normalized_facility_record(
    record: object,
    expected_id: int,
    source_site: str,
    generated_at: datetime,
) -> Optional[Dict[str, object]]:
    if not isinstance(record, dict) or set(record.keys()) != FACILITY_RECORD_KEYS:
        return None
    expected_name, expected_slug = SUPPORTED_FACILITY_IDENTITIES[expected_id]
    if (
        type(record.get("facilityId")) is not int
        or record.get("facilityId") != expected_id
        or record.get("facilityName") != expected_name
        or record.get("slug") != expected_slug
    ):
        return None

    public_url = safe_same_origin_https_url(record.get("url"), source_site)
    if public_url is None:
        return None
    public_path = urlsplit(public_url).path.rstrip("/")
    if not public_path or public_path.rsplit("/", 1)[-1] != expected_slug:
        return None

    resolved_value = record.get("resolvedUrl")
    resolved_url = (
        None
        if resolved_value is None
        else safe_same_origin_https_url(resolved_value, source_site)
    )
    if resolved_value is not None and resolved_url is None:
        return None

    updated_at = _parse_canonical_utc_timestamp(record.get("updatedAt"))
    if updated_at is None or updated_at > generated_at:
        return None

    status = record.get("status")
    stale = record.get("stale")
    source = record.get("source")
    source_modified_gmt = record.get("sourceModifiedGmt")
    error = record.get("error")
    error_category = record.get("errorCategory")
    source_fetched_at = _parse_canonical_utc_timestamp(
        record.get("sourceFetchedAt")
    )
    last_successful_at = _parse_canonical_utc_timestamp(
        record.get("lastSuccessfulAt")
    )

    if status == "error":
        if (
            stale is not False
            or source is not None
            or source_modified_gmt is not None
            or resolved_url is not None
            or record.get("sections") != []
            or record.get("sourceFetchedAt") is not None
            or record.get("lastSuccessfulAt") is not None
            or error != SCHEDULE_REFRESH_ERROR
            or error_category not in SCHEDULE_ERROR_CATEGORIES
        ):
            return None
        sections: List[Dict[str, object]] = []
    elif status in {"ok", "stale"}:
        sections_result = _copy_json_schedule_sections(record.get("sections"))
        if (
            source not in SCHEDULE_SOURCES
            or sections_result is None
            or source_fetched_at is None
            or last_successful_at is None
            or source_fetched_at > generated_at
            or last_successful_at > generated_at
            or (
                source == "direct_html"
                and source_modified_gmt is not None
            )
            or (
                source_modified_gmt is not None
                and not _valid_local_iso_timestamp(source_modified_gmt)
            )
            or (
                status == "ok"
                and (
                    stale is not False
                    or error is not None
                    or error_category is not None
                )
            )
            or (
                status == "stale"
                and (
                    stale is not True
                    or error != SCHEDULE_REFRESH_ERROR
                    or error_category not in SCHEDULE_ERROR_CATEGORIES
                )
            )
        ):
            return None
        sections = sections_result
    else:
        return None

    return {
        "facilityId": expected_id,
        "facilityName": expected_name,
        "slug": expected_slug,
        "url": public_url,
        "resolvedUrl": resolved_url,
        "status": status,
        "source": source,
        "sourceModifiedGmt": source_modified_gmt,
        "sections": sections,
        "sourceFetchedAt": record.get("sourceFetchedAt"),
        "lastSuccessfulAt": record.get("lastSuccessfulAt"),
        "stale": stale,
        "error": error,
        "errorCategory": error_category,
        "updatedAt": record.get("updatedAt"),
    }


def validate_schedule_payload(
    payload: Mapping[str, Any],
    now: Optional[datetime] = None,
) -> Dict[str, object]:
    try:
        reference_now = datetime.now(timezone.utc) if now is None else now
        if not _aware_datetime(reference_now):
            raise ValueError("invalid schedule payload")
        reference_utc = reference_now.astimezone(timezone.utc)
        if not isinstance(payload, Mapping) or set(payload.keys()) != SCHEDULE_PAYLOAD_KEYS:
            raise ValueError("invalid schedule payload")

        generated_value = payload.get("generatedAt")
        generated_at = _parse_canonical_utc_timestamp(generated_value)
        source_site = _safe_source_site(payload.get("sourceSite"))
        facilities = payload.get("facilities")
        ok_count = payload.get("okCount")
        total_count = payload.get("totalCount")
        if (
            generated_at is None
            or generated_at > reference_utc
            or source_site is None
            or not isinstance(facilities, list)
            or len(facilities) != 2
            or type(ok_count) is not int
            or type(total_count) is not int
            or total_count != 2
        ):
            raise ValueError("invalid schedule payload")

        normalized_facilities: List[Dict[str, object]] = []
        for expected_id, record in zip((1186, 1656), facilities):
            normalized = _normalized_facility_record(
                record,
                expected_id,
                source_site,
                generated_at,
            )
            if normalized is None:
                raise ValueError("invalid schedule payload")
            normalized_facilities.append(normalized)
        truthful_ok_count = sum(
            1 for record in normalized_facilities if record["status"] == "ok"
        )
        if ok_count != truthful_ok_count:
            raise ValueError("invalid schedule payload")

        return {
            "generatedAt": generated_value,
            "sourceSite": source_site,
            "facilities": normalized_facilities,
            "okCount": ok_count,
            "totalCount": total_count,
        }
    except ValueError:
        raise ValueError("invalid schedule payload") from None
    except Exception:
        raise ValueError("invalid schedule payload") from None


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def is_header_row(left: str, right: str) -> bool:
    left_norm = clean_text(left).lower()
    right_norm = clean_text(right).lower()
    if left_norm in {"day", "days", "date", "dates"} and "hour" in right_norm:
        return True
    if left_norm == "hours" and right_norm in {"", "time"}:
        return True
    return False


def looks_like_hours_row(label: str, hours_value: str) -> bool:
    label_norm = clean_text(label)
    hours_norm = clean_text(hours_value)
    if not label_norm or not hours_norm:
        return False
    has_label_hint = bool(DAY_HINT_RE.search(label_norm) or DATE_LABEL_RE.search(label_norm))
    has_hours_hint = bool(HOURS_HINT_RE.search(hours_norm))
    return has_label_hint and has_hours_hint


def strip_html(value: str) -> str:
    with_breaks = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    without_tags = TAG_RE.sub(" ", with_breaks)
    return clean_text(html_lib.unescape(without_tags))


def extract_rows_from_table(table: Any) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for tr in table.find_all("tr"):
        cells = [clean_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
        if len(cells) < 2:
            continue

        left = cells[0]
        right = cells[1]
        if is_header_row(left, right):
            continue
        if not left or not right:
            continue

        rows.append({"label": left, "hours": right})
    return rows


def find_table_heading(table: Any) -> str:
    preferred_tags = {"h2", "h3", "h4", "h5"}
    fallback: Optional[str] = None
    for prev in table.find_all_previous(limit=16):
        if not getattr(prev, "name", None):
            continue
        text = clean_text(prev.get_text(" ", strip=True))
        if not text:
            continue
        if len(text) > 140:
            continue
        tag_name = str(prev.name).lower()

        if tag_name in preferred_tags:
            if any(keyword in text.lower() for keyword in ("hour", "schedule", "building")):
                return text
            if fallback is None:
                fallback = text
    return fallback or "Hours"


def dedupe_sections(sections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output: List[Dict[str, Any]] = []
    for section in sections:
        key = (
            clean_text(str(section.get("title", ""))).lower(),
            json.dumps(section.get("rows", []), sort_keys=True, ensure_ascii=False),
            clean_text(str(section.get("note", ""))).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(section)
    return output


def parse_hours_sections_with_bs4(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    sections: List[Dict[str, Any]] = []

    for table in soup.find_all("table"):
        rows = extract_rows_from_table(table)
        if len(rows) < 1:
            continue

        matching_rows = [row for row in rows if looks_like_hours_row(row["label"], row["hours"])]
        if len(matching_rows) < 1:
            continue

        sections.append(
            {
                "title": find_table_heading(table),
                "rows": matching_rows,
                "note": None,
            }
        )

    for definition_list in soup.find_all("dl"):
        rows: List[Dict[str, str]] = []
        for term in definition_list.find_all("dt"):
            description = term.find_next_sibling("dd")
            if description is None or description.find_parent("dl") is not definition_list:
                continue
            label = clean_text(term.get_text(" ", strip=True))
            hours = clean_text(description.get_text(" ", strip=True))
            if looks_like_hours_row(label, hours):
                rows.append({"label": label, "hours": hours})
        if rows:
            sections.append(
                {
                    "title": find_table_heading(definition_list),
                    "rows": rows,
                    "note": None,
                }
            )

    for match in NOTICE_HOURS_RE.finditer(soup.get_text("\n", strip=True)):
        notice = clean_text(match.group(0))
        if not notice:
            continue
        sections.append(
            {
                "title": "Seasonal Notice",
                "rows": [],
                "note": notice,
            }
        )

    for match in NOTICE_MAINTENANCE_RE.finditer(soup.get_text("\n", strip=True)):
        notice = clean_text(match.group(0))
        if not notice:
            continue
        sections.append(
            {
                "title": "Maintenance Closures",
                "rows": [],
                "note": notice,
            }
        )

    return dedupe_sections(sections)


def find_heading_before_table(html: str, table_start: int) -> str:
    fallback: Optional[str] = None
    for match in HEADING_RE.finditer(html):
        if match.start() >= table_start:
            break

        text = strip_html(match.group(2))
        if not text:
            continue
        if len(text) > 140:
            continue
        fallback = text
        if any(keyword in text.lower() for keyword in ("hour", "schedule", "building")):
            return text
    return fallback or "Hours"


def parse_hours_sections_with_regex(html: str) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    for table_match in TABLE_RE.finditer(html):
        table_html = table_match.group(1)
        rows: List[Dict[str, str]] = []

        for row_match in ROW_RE.finditer(table_html):
            cells_raw = [strip_html(cell) for cell in CELL_RE.findall(row_match.group(1))]
            if len(cells_raw) < 2:
                continue

            left = cells_raw[0]
            right = cells_raw[1]
            if is_header_row(left, right):
                continue
            if not left or not right:
                continue
            rows.append({"label": left, "hours": right})

        if len(rows) < 1:
            continue

        matching_rows = [row for row in rows if looks_like_hours_row(row["label"], row["hours"])]
        if len(matching_rows) < 1:
            continue

        sections.append(
            {
                "title": find_heading_before_table(html, table_match.start()),
                "rows": matching_rows,
                "note": None,
            }
        )

    for definition_list_match in DEFINITION_LIST_RE.finditer(html):
        rows = []
        for pair_match in DEFINITION_PAIR_RE.finditer(
            definition_list_match.group(1)
        ):
            label = strip_html(pair_match.group(1))
            hours = strip_html(pair_match.group(2))
            if looks_like_hours_row(label, hours):
                rows.append({"label": label, "hours": hours})
        if rows:
            sections.append(
                {
                    "title": find_heading_before_table(
                        html,
                        definition_list_match.start(),
                    ),
                    "rows": rows,
                    "note": None,
                }
            )

    for match in NOTICE_HOURS_RE.finditer(strip_html(html)):
        notice = clean_text(match.group(0))
        if not notice:
            continue
        sections.append(
            {
                "title": "Seasonal Notice",
                "rows": [],
                "note": notice,
            }
        )

    for match in NOTICE_MAINTENANCE_RE.finditer(strip_html(html)):
        notice = clean_text(match.group(0))
        if not notice:
            continue
        sections.append(
            {
                "title": "Maintenance Closures",
                "rows": [],
                "note": notice,
            }
        )

    return dedupe_sections(sections)


def parse_schedule_sections(html: str) -> List[Dict[str, Any]]:
    if not isinstance(html, str) or not html.strip():
        return []
    if BeautifulSoup is not None:
        return parse_hours_sections_with_bs4(html)
    return parse_hours_sections_with_regex(html)


def parse_hours_sections(html: str) -> List[Dict[str, Any]]:
    return parse_schedule_sections(html)


def looks_like_bot_challenge(html: str) -> bool:
    lowered = html.lower()
    markers = (
        "please enable javascript",
        "checking your browser",
        "challenge-platform",
        "just a moment...",
        "cf-chl-",
    )
    return any(marker in lowered for marker in markers)


def parse_direct_response(
    html: str,
    resolved_url: str,
) -> Tuple[List[Dict[str, Any]], str]:
    if (
        not isinstance(html, str)
        or not isinstance(resolved_url, str)
        or not resolved_url.strip()
    ):
        raise ScheduleFetchError("schema_invalid")
    if looks_like_bot_challenge(html):
        raise ScheduleFetchError("anti_bot")
    sections = parse_schedule_sections(html)
    if not sections:
        raise ScheduleFetchError("parse_empty")
    return sections, resolved_url.strip()


def parse_wp_page_payload(
    payload: object,
) -> Tuple[str, Optional[str], Optional[str]]:
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise ScheduleFetchError("wp_payload_invalid")
    page = payload[0]
    content = page.get("content")
    if not isinstance(content, dict):
        raise ScheduleFetchError("wp_payload_invalid")
    rendered = content.get("rendered")
    if not isinstance(rendered, str) or not rendered.strip():
        raise ScheduleFetchError("wp_payload_invalid")

    modified_gmt = page.get("modified_gmt")
    if modified_gmt is not None and (
        not isinstance(modified_gmt, str) or not modified_gmt.strip()
    ):
        raise ScheduleFetchError("wp_payload_invalid")
    link = page.get("link")
    if link is not None and (not isinstance(link, str) or not link.strip()):
        raise ScheduleFetchError("wp_payload_invalid")

    return (
        rendered,
        modified_gmt if isinstance(modified_gmt, str) else None,
        link.strip() if isinstance(link, str) else None,
    )


def fetch_direct_html(url: str) -> Tuple[str, str]:
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            timeout=25,
        )
        response.raise_for_status()
    except requests.Timeout:
        raise ScheduleFetchError("upstream_timeout") from None
    except requests.RequestException:
        raise ScheduleFetchError("upstream_http") from None
    return response.text, response.url


def fetch_wp_json_html(site_base: str, slug: str) -> Tuple[str, Optional[str], Optional[str]]:
    endpoint = site_base.rstrip("/") + "/wp-json/wp/v2/pages"
    try:
        response = requests.get(
            endpoint,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            params={
                "slug": slug,
                "_fields": "slug,link,title.rendered,content.rendered,modified_gmt",
            },
            timeout=25,
        )
        response.raise_for_status()
    except requests.Timeout:
        raise ScheduleFetchError("upstream_timeout") from None
    except requests.RequestException:
        raise ScheduleFetchError("upstream_http") from None
    try:
        payload = response.json()
    except ValueError:
        raise ScheduleFetchError("wp_payload_invalid") from None
    return parse_wp_page_payload(payload)


def _validated_candidate_identity(
    facility: Mapping[str, Any],
    site_base: str,
    fetched_at: datetime,
) -> Tuple[int, str, str, str]:
    if (
        not isinstance(facility, Mapping)
        or not _aware_datetime(fetched_at)
        or _https_origin(site_base) is None
    ):
        raise ValueError("invalid facility configuration")
    facility_id = facility.get("facilityId")
    if type(facility_id) is not int or facility_id not in SUPPORTED_FACILITY_IDENTITIES:
        raise ValueError("invalid facility configuration")
    expected_name, expected_slug = SUPPORTED_FACILITY_IDENTITIES[facility_id]
    if (
        facility.get("facilityName") != expected_name
        or facility.get("slug") != expected_slug
    ):
        raise ValueError("invalid facility configuration")
    public_url = safe_same_origin_https_url(facility.get("url"), site_base)
    if public_url is None:
        raise ValueError("invalid facility configuration")
    return facility_id, expected_name, expected_slug, public_url


def collect_facility_candidate(
    facility: Mapping[str, Any],
    site_base: str,
    fetched_at: datetime,
) -> FacilityCandidate:
    facility_id, facility_name, slug, public_url = _validated_candidate_identity(
        facility,
        site_base,
        fetched_at,
    )

    try:
        direct_html, direct_resolved_url = fetch_direct_html(public_url)
        direct_sections, _ = parse_direct_response(
            direct_html,
            direct_resolved_url,
        )
        safe_resolved_url = safe_same_origin_https_url(
            direct_resolved_url,
            site_base,
        )
        if safe_resolved_url is None:
            raise ScheduleFetchError("schema_invalid")
        return FacilityCandidate(
            facility_id=facility_id,
            facility_name=facility_name,
            slug=slug,
            public_url=public_url,
            source="direct_html",
            resolved_url=safe_resolved_url,
            source_modified_gmt=None,
            sections=tuple(direct_sections),
            fetched_at=fetched_at,
            error_category=None,
        )
    except ScheduleFetchError:
        pass
    except Exception:
        pass

    final_category = "schema_invalid"
    try:
        wp_html, source_modified_gmt, wp_link = fetch_wp_json_html(
            site_base=site_base,
            slug=slug,
        )
        if not isinstance(wp_html, str):
            raise ScheduleFetchError("wp_payload_invalid")
        if looks_like_bot_challenge(wp_html):
            raise ScheduleFetchError("anti_bot")
        wp_sections = parse_schedule_sections(wp_html)
        if not wp_sections:
            raise ScheduleFetchError("parse_empty")
        if source_modified_gmt is not None and not _valid_local_iso_timestamp(
            source_modified_gmt
        ):
            raise ScheduleFetchError("wp_payload_invalid")
        if wp_link is not None and not isinstance(wp_link, str):
            raise ScheduleFetchError("wp_payload_invalid")
        safe_wp_link = (
            safe_same_origin_https_url(wp_link, site_base)
            if wp_link is not None
            else None
        )
        return FacilityCandidate(
            facility_id=facility_id,
            facility_name=facility_name,
            slug=slug,
            public_url=public_url,
            source="wp_json",
            resolved_url=safe_wp_link,
            source_modified_gmt=source_modified_gmt,
            sections=tuple(wp_sections),
            fetched_at=fetched_at,
            error_category=None,
        )
    except ScheduleFetchError as exc:
        final_category = exc.category
    except Exception:
        final_category = "schema_invalid"

    return FacilityCandidate(
        facility_id=facility_id,
        facility_name=facility_name,
        slug=slug,
        public_url=public_url,
        source=None,
        resolved_url=None,
        source_modified_gmt=None,
        sections=(),
        fetched_at=fetched_at,
        error_category=final_category,
    )


def build_combined_payload(
    facilities: Sequence[Mapping[str, Any]],
    previous_payload: Optional[Mapping[str, Any]],
    site_base: str,
    generated_at: datetime,
) -> Dict[str, object]:
    if not _aware_datetime(generated_at) or _safe_source_site(site_base) is None:
        raise ValueError("invalid schedule payload")
    if (
        not isinstance(facilities, Sequence)
        or isinstance(facilities, (str, bytes))
        or len(facilities) != 2
    ):
        raise ValueError("invalid schedule payload")

    validated_identities: List[Tuple[int, str, str, str]] = []
    try:
        for expected_id, facility in zip((1186, 1656), facilities):
            identity = _validated_candidate_identity(
                facility,
                site_base,
                generated_at,
            )
            if identity[0] != expected_id:
                raise ValueError("invalid schedule payload")
            validated_identities.append(identity)
    except Exception:
        raise ValueError("invalid schedule payload") from None

    normalized_previous: Optional[Dict[str, object]] = None
    if previous_payload is not None:
        try:
            normalized_previous = validate_schedule_payload(
                previous_payload,
                now=generated_at,
            )
        except ValueError:
            normalized_previous = None
    if normalized_previous is not None:
        previous_records = normalized_previous.get("facilities")
        if not isinstance(previous_records, list):
            normalized_previous = None
        else:
            for record, identity in zip(previous_records, validated_identities):
                facility_id, facility_name, slug, public_url = identity
                # Payload validation already checked status/content. An error
                # record has no reusable hours, but cannot invalidate its sibling.
                if (
                    record.get("facilityId") != facility_id
                    or record.get("facilityName") != facility_name
                    or record.get("slug") != slug
                    or _safe_current_public_url(record.get("url"), public_url) is None
                    or (
                        record.get("resolvedUrl") is not None
                        and safe_same_origin_https_url(record.get("resolvedUrl"), public_url) is None
                    )
                ):
                    normalized_previous = None
                    break
    previous_by_id = {
        record["facilityId"]: record
        for record in (
            normalized_previous.get("facilities", [])
            if normalized_previous is not None
            else []
        )
        if isinstance(record, Mapping)
    }

    merged_facilities: List[Dict[str, object]] = []
    for facility, identity in zip(facilities, validated_identities):
        facility_id, facility_name, slug, public_url = identity
        try:
            candidate = collect_facility_candidate(
                facility,
                site_base,
                generated_at,
            )
            if not isinstance(candidate, FacilityCandidate):
                raise ValueError("invalid facility candidate")
        except Exception:
            candidate = FacilityCandidate(
                facility_id=facility_id,
                facility_name=facility_name,
                slug=slug,
                public_url=public_url,
                source=None,
                resolved_url=None,
                source_modified_gmt=None,
                sections=(),
                fetched_at=generated_at,
                error_category="schema_invalid",
            )
        merged_facilities.append(
            merge_facility_candidate(
                candidate,
                previous_by_id.get(facility_id),
                generated_at,
            )
        )

    payload = {
        "generatedAt": iso_utc(generated_at),
        "sourceSite": site_base,
        "facilities": merged_facilities,
        "okCount": sum(
            1 for record in merged_facilities if record.get("status") == "ok"
        ),
        "totalCount": 2,
    }
    return validate_schedule_payload(payload, now=generated_at)


def collect_facility_hours(facility: Dict[str, Any], site_base: str) -> Dict[str, Any]:
    url = str(facility.get("url", "")).strip()
    slug = str(facility.get("slug", "")).strip()

    output: Dict[str, Any] = {
        "facilityId": int(facility.get("facilityId")),
        "facilityName": str(facility.get("facilityName", "")).strip(),
        "slug": slug,
        "url": url,
        "status": "error",
        "source": None,
        "sections": [],
        "error": None,
        "updatedAt": now_iso(),
    }
    errors: List[str] = []

    try:
        html, resolved_url = fetch_direct_html(url)
        sections, resolved_url = parse_direct_response(html, resolved_url)
        output["status"] = "ok"
        output["source"] = "direct_html"
        output["resolvedUrl"] = resolved_url
        output["sections"] = sections
        return output
    except ScheduleFetchError as exc:
        errors.append("direct_html: " + exc.category)
    except Exception:
        errors.append("direct_html: schema_invalid")

    try:
        html, modified_gmt, resolved_url = fetch_wp_json_html(site_base=site_base, slug=slug)
        sections = parse_schedule_sections(html)
        if not sections:
            raise ScheduleFetchError("parse_empty")
        output["status"] = "ok"
        output["source"] = "wp_json"
        output["resolvedUrl"] = resolved_url or url
        output["sourceModifiedGmt"] = modified_gmt
        output["sections"] = sections
        return output
    except ScheduleFetchError as exc:
        errors.append("wp_json: " + exc.category)
    except Exception:
        errors.append("wp_json: schema_invalid")

    output["error"] = "; ".join(errors) if errors else "schema_invalid"
    return output


def build_facilities(settings: Settings | None = None) -> List[Dict[str, Any]]:
    nick_url = (
        settings.recwell_nick_url
        if settings is not None
        else env_with_default("RECWELL_NICK_URL", DEFAULT_FACILITIES[0]["url"])
    )
    bakke_url = (
        settings.recwell_bakke_url
        if settings is not None
        else env_with_default("RECWELL_BAKKE_URL", DEFAULT_FACILITIES[1]["url"])
    )
    return [
        {
            "facilityId": DEFAULT_FACILITIES[0]["facilityId"],
            "facilityName": DEFAULT_FACILITIES[0]["facilityName"],
            "slug": DEFAULT_FACILITIES[0]["slug"],
            "url": nick_url,
        },
        {
            "facilityId": DEFAULT_FACILITIES[1]["facilityId"],
            "facilityName": DEFAULT_FACILITIES[1]["facilityName"],
            "slug": DEFAULT_FACILITIES[1]["slug"],
            "url": bakke_url,
        },
    ]


def _cleanup_atomic_file(fd: Optional[int], temporary_path: Optional[str]) -> None:
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
    if temporary_path is not None:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    normalized = validate_schedule_payload(payload)
    fd: Optional[int] = None
    temporary_path: Optional[str] = None
    try:
        serialized = json.dumps(normalized, indent=2, ensure_ascii=False)
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=".facility-hours-",
            suffix=".json",
            dir=directory,
        )
        handle = os.fdopen(fd, "w", encoding="utf-8")
        fd = None
        with handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except BaseException as exc:
        _cleanup_atomic_file(fd, temporary_path)
        if not isinstance(exc, Exception):
            raise
        raise ScheduleFetchError("io_error") from None


def write_json(path: str, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, payload)


def _reject_duplicate_json_keys(
    pairs: Sequence[Tuple[str, object]],
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("invalid schedule payload")
        result[key] = value
    return result


def load_previous_schedule(
    path: str,
    now: Optional[datetime] = None,
) -> Optional[Dict[str, object]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
        return validate_schedule_payload(payload, now=now)
    except Exception:
        return None


def main_for_output(output_path: str, *, settings: Settings | None = None) -> int:
    generated_at = now_utc()
    previous_payload = load_previous_schedule(output_path, now=generated_at)
    payload = build_combined_payload(
        build_facilities(settings) if settings is not None else build_facilities(),
        previous_payload,
        settings.recwell_site_base
        if settings is not None
        else env_with_default("RECWELL_SITE_BASE", DEFAULT_SITE_BASE),
        generated_at,
    )
    atomic_write_json(output_path, payload)
    return 0 if payload["okCount"] == payload["totalCount"] else 1


def run_facility_hours_fetch(settings: Settings) -> int:
    validate_command_environment(settings, ("FACILITY_HOURS_JSON_PATH",))
    return main_for_output(settings.facility_hours_json_path, settings=settings)


_import_sys.modules.setdefault(
    "server.reclive.facility_schedule", _import_sys.modules[__name__]
)
_import_sys.modules.setdefault(
    "reclive.facility_schedule", _import_sys.modules[__name__]
)
