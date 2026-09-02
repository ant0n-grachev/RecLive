from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Optional
from zoneinfo import ZoneInfo


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
_DATE_LIKE_PATTERN = (
    r"(?:\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|"
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}\b)"
)
_DATE_LIKE = re.compile(_DATE_LIKE_PATTERN)
_DAY_NAME_PATTERN = (
    r"mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:r|rs|rsday)?|"
    r"fri(?:day)?|sat(?:urday)?|sun(?:day)?"
)
_DAY_TOKEN = re.compile(rf"\b({_DAY_NAME_PATTERN})\b")
_DAY_RANGE = re.compile(
    rf"\b({_DAY_NAME_PATTERN})\s*-\s*({_DAY_NAME_PATTERN})\b"
)
_FACILITY_NAME_PATTERN = (
    r"(?:nick|nicholas recreation center|bakke|"
    r"bakke recreation (?:&|and) wellbeing center)"
)
_TITLE_DATE_TOKEN_PATTERN = (
    r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?|"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?)"
)
_TITLE_DATE_RANGE_PATTERN = (
    rf"{_TITLE_DATE_TOKEN_PATTERN}"
    rf"(?:\s*-\s*(?:{_TITLE_DATE_TOKEN_PATTERN}|\d{{1,2}}))?"
)
_EXACT_DATE_RANGE = re.compile(
    rf"^({_TITLE_DATE_TOKEN_PATTERN})"
    rf"(?:\s*-\s*({_TITLE_DATE_TOKEN_PATTERN}|\d{{1,2}}))?$"
)
_TITLE_SUFFIX_PATTERN = (
    rf"(?:\s*(?::|-)\s*{_TITLE_DATE_RANGE_PATTERN}"
    rf"|\s*\({_TITLE_DATE_RANGE_PATTERN}\))?"
)
_BUILDING_SECTION_TITLE = re.compile(
    rf"^(?:(?:{_FACILITY_NAME_PATTERN})\s+)?"
    rf"(?:building hours|facility hours|hours of operation|facility schedule)"
    rf"{_TITLE_SUFFIX_PATTERN}$"
    rf"|^(?:{_FACILITY_NAME_PATTERN})\s+hours{_TITLE_SUFFIX_PATTERN}$"
)
_MAINTENANCE_SECTION_TITLE = re.compile(
    rf"^(?:(?:{_FACILITY_NAME_PATTERN})\s+)?"
    rf"maintenance closures?{_TITLE_SUFFIX_PATTERN}$"
)


def _normalize(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return re.sub(
        r"\s+",
        " ",
        value.lower().replace("–", "-").replace("—", "-"),
    ).strip()


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except (TypeError, ValueError, OverflowError):
        return None


def _parse_date_token(
    token: str,
    fallback_year: int,
    fallback_month: int | None = None,
) -> date | None:
    normalized = _normalize(token)
    if not normalized:
        return None
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", normalized)
    if match:
        return _safe_date(*(int(part) for part in match.groups()))
    match = re.fullmatch(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", normalized)
    if match:
        year = fallback_year
        if match.group(3):
            year = int(match.group(3))
            if year < 100:
                year += 2000
        return _safe_date(year, int(match.group(1)), int(match.group(2)))
    match = re.fullmatch(r"([a-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?", normalized)
    if match:
        month = _MONTHS.get(match.group(1))
        if month is None:
            return None
        year = int(match.group(3)) if match.group(3) else fallback_year
        return _safe_date(year, month, int(match.group(2)))
    match = re.fullmatch(r"(\d{1,2})", normalized)
    if match and fallback_month is not None:
        return _safe_date(fallback_year, fallback_month, int(match.group(1)))
    return None


def parse_schedule_date_range(
    value: str,
    fallback_year: int,
) -> Optional[tuple[date, date, int]]:
    normalized = _normalize(value)
    if (
        not normalized
        or type(fallback_year) is not int
        or not 1 <= fallback_year <= 9999
        or not _DATE_LIKE.search(normalized)
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
    if re.fullmatch(r"(?:open\s+)?daily", normalized):
        return set(range(7))
    if re.fullmatch(r"(?:open\s+)?weekdays", normalized):
        return {0, 1, 2, 3, 4}
    if re.fullmatch(r"(?:open\s+)?weekends", normalized):
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
    if re.fullmatch(
        rf"(?:{_DAY_NAME_PATTERN})"
        rf"(?:\s*(?:,|/|&|\band\b)\s*(?:{_DAY_NAME_PATTERN}))*",
        normalized,
    ) is None:
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
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", normalized)
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


def parse_schedule_hours_window(
    value: str,
) -> Optional[tuple[int, int, bool]]:
    normalized = _normalize(value)
    if not normalized:
        return None
    if re.fullmatch(r"closed(?:\s+.*)?", normalized):
        return (0, 0, True)
    if re.fullmatch(r"(?:open\s+)?24\s*(?:hours?|hrs?)", normalized):
        return (0, MINUTES_PER_DAY, False)
    parts = [part.strip() for part in re.split(r"\s*-\s*", normalized)]
    if len(parts) != 2 and " to " in normalized:
        parts = [part.strip() for part in normalized.split(" to ", 1)]
    if len(parts) != 2 or any(not part for part in parts):
        return None
    start = _parse_clock(parts[0], is_end=False)
    end = _parse_clock(parts[1], is_end=True)
    if (
        start is None
        or end is None
        or start >= MINUTES_PER_DAY
        or start % MINUTES_PER_DAY == end % MINUTES_PER_DAY
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
    sections: list[dict[str, object]],
    anchor: date,
    *,
    section_kind: str,
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
            and not title_ranges
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
            (value for value in title_ranges if value[0] <= anchor <= value[1]),
            None,
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
                and not row_ranges
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
                (value for value in row_ranges if value[0] <= anchor <= value[1]),
                None,
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
                effective_start = max(value[0] for value in date_scopes)
                effective_end = min(value[1] for value in date_scopes)
                if effective_end < effective_start:
                    continue
                specificity = 2
                span = (effective_end - effective_start).days + 1
                explicit_date = True
            else:
                specificity = 1
                span = 999_999
                explicit_date = False
            window = parse_schedule_hours_window(hours)
            if kind == "maintenance" and (
                window is None or window[2] is not True
            ):
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
    sections: list[dict[str, object]],
    at: datetime,
) -> Optional[bool]:
    if (
        not isinstance(sections, list)
        or not sections
        or not isinstance(at, datetime)
        or at.tzinfo is None
        or at.utcoffset() is None
    ):
        return None
    for section in sections:
        if (
            not isinstance(section, Mapping)
            or not isinstance(section.get("title"), str)
            or not isinstance(section.get("rows"), list)
        ):
            return None
        rows = section.get("rows")
        if not isinstance(rows, list):
            return None
        if any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("label"), str)
            or not isinstance(row.get("hours"), str)
            for row in rows
        ):
            return None
    local = at.astimezone(timezone.utc).astimezone(CHICAGO_TZ)
    minute = local.hour * 60 + local.minute
    current_maintenance = _select_candidate(
        sections,
        local.date(),
        section_kind="maintenance",
    )
    if current_maintenance is not None:
        if current_maintenance.window is None:
            return None
        if current_maintenance.window[2]:
            return False
        return None

    current = _select_candidate(
        sections,
        local.date(),
        section_kind="building",
    )
    previous = _select_candidate(
        sections,
        local.date() - timedelta(days=1),
        section_kind="building",
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
        if not closed and end > MINUTES_PER_DAY and minute < end - MINUTES_PER_DAY:
            return True

    if current is not None:
        return False
    return None


def _parse_aware_timestamp(value: object) -> datetime | None:
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
    stale_after_seconds: int = 21_600,
) -> bool:
    if (
        not isinstance(payload, Mapping)
        or type(facility_id) is not int
        or facility_id not in {1186, 1656}
        or not isinstance(at, datetime)
        or at.tzinfo is None
        or at.utcoffset() is None
        or type(stale_after_seconds) is not int
        or stale_after_seconds <= 0
    ):
        return False
    facilities = payload.get("facilities")
    if (
        not isinstance(facilities, list)
        or not facilities
        or any(
            not isinstance(item, Mapping)
            or type(item.get("facilityId")) is not int
            for item in facilities
        )
    ):
        return False
    matches = [
        item
        for item in facilities
        if isinstance(item, Mapping)
        and type(item.get("facilityId")) is int
        and item.get("facilityId") == facility_id
    ]
    if len(matches) != 1:
        return False
    facility = matches[0]
    if facility.get("status") != "ok" or facility.get("stale") is not False:
        return False
    now = at.astimezone(timezone.utc)
    for raw_timestamp in (
        payload.get("generatedAt"),
        facility.get("lastSuccessfulAt"),
    ):
        observed = _parse_aware_timestamp(raw_timestamp)
        if observed is None:
            return False
        age = (now - observed).total_seconds()
        if age < 0 or age > stale_after_seconds:
            return False
    sections = facility.get("sections")
    if not isinstance(sections, list) or not sections:
        return False
    return get_facility_schedule_open_state(sections, at) is True
