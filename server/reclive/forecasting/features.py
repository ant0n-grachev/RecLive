"""Forecasting features owner; mechanically transferred definitions."""

import sys as _sys
from server.reclive.forecasting import config
import bisect
import math
import re
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


from server.facility_schedule import parse_schedule_date_range
from server.facility_schedule import parse_schedule_hours_window
from server.facility_schedule import parse_schedule_weekday_set




def normalize_schedule_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).lower().replace("–", "-").replace("—", "-")).strip()


def is_schedule_facility_wide_section(section_title: str) -> bool:
    title = normalize_schedule_text(section_title)
    if not title:
        return False
    if title == "seasonal notice":
        return False
    if "maintenance closures" in title:
        return True

    area_keywords = [
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
    ]
    return not any(keyword in title for keyword in area_keywords)


def schedule_open_for_window(window: Tuple[int, int, bool], minute_of_day: int) -> bool:
    start_minutes, end_minutes, is_closed = window
    if is_closed:
        return False

    if end_minutes > config.SCHEDULE_MINUTES_PER_DAY:
        return minute_of_day >= start_minutes or minute_of_day < (end_minutes - config.SCHEDULE_MINUTES_PER_DAY)
    return start_minutes <= minute_of_day < end_minutes


def schedule_boundary_for_window(
    window: Tuple[int, int, bool],
    minute_of_day: int,
) -> Tuple[bool, bool]:
    start_minutes, end_minutes, is_closed = window
    if is_closed:
        return False, False

    open_exact = int(minute_of_day) == int(start_minutes)
    close_minute = (
        int(end_minutes - config.SCHEDULE_MINUTES_PER_DAY)
        if end_minutes > config.SCHEDULE_MINUTES_PER_DAY
        else int(end_minutes % config.SCHEDULE_MINUTES_PER_DAY)
    )
    close_exact = int(minute_of_day) == int(close_minute)
    return bool(open_exact), bool(close_exact)


def get_facility_schedule_boundary_state(
    sections: List[Dict[str, object]],
    ts: datetime,
    date_range_cache: Dict[Tuple[str, int], Optional[Tuple[date, date, int]]],
    weekday_cache: Dict[str, Optional[Set[int]]],
    hours_window_cache: Dict[str, Optional[Tuple[int, int, bool]]],
) -> Tuple[bool, bool]:
    if not sections:
        return False, False

    current_day = ts.date()
    current_year = current_day.year
    current_weekday = int(ts.weekday())
    minute_of_day = int(ts.hour) * 60 + int(ts.minute)
    candidates: List[Tuple[int, int, int, int, bool, bool]] = []

    for section_idx, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        section_title = str(section.get("title", "")).strip()
        if not is_schedule_facility_wide_section(section_title):
            continue

        section_title_norm = normalize_schedule_text(section_title)
        title_range_key = (section_title_norm, current_year)
        if title_range_key not in date_range_cache:
            date_range_cache[title_range_key] = parse_schedule_date_range(section_title, current_year)
        title_range = date_range_cache.get(title_range_key)

        rows = section.get("rows")
        if not isinstance(rows, list):
            continue

        for row_idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            label = str(row.get("label", "")).strip()
            hours = str(row.get("hours", "")).strip()
            if not label or not hours:
                continue

            label_norm = normalize_schedule_text(label)
            row_range_key = (label_norm, current_year)
            if row_range_key not in date_range_cache:
                date_range_cache[row_range_key] = parse_schedule_date_range(label, current_year)
            row_range = date_range_cache.get(row_range_key)

            specificity = 1
            span_days = 9999
            is_date_match = False

            if row_range is not None:
                start_day, end_day, row_span_days = row_range
                is_date_match = start_day <= current_day <= end_day
                specificity = 3
                span_days = row_span_days
            else:
                if label_norm not in weekday_cache:
                    weekday_cache[label_norm] = parse_schedule_weekday_set(label)
                weekday_set = weekday_cache.get(label_norm)
                if not weekday_set:
                    continue
                if title_range is not None:
                    title_start, title_end, title_span_days = title_range
                    if not (title_start <= current_day <= title_end):
                        continue
                    specificity = 2
                    span_days = title_span_days
                else:
                    specificity = 1
                    span_days = 9998
                is_date_match = current_weekday in weekday_set

            if not is_date_match:
                continue

            hours_norm = normalize_schedule_text(hours)
            if hours_norm not in hours_window_cache:
                hours_window_cache[hours_norm] = parse_schedule_hours_window(hours)
            hours_window = hours_window_cache.get(hours_norm)
            if hours_window is None:
                continue

            open_exact, close_exact = schedule_boundary_for_window(hours_window, minute_of_day)
            candidates.append((-specificity, span_days, section_idx, row_idx, open_exact, close_exact))

    if not candidates:
        return False, False

    candidates.sort()
    _spec, _span, _section_idx, _row_idx, open_exact, close_exact = candidates[0]
    return bool(open_exact), bool(close_exact)


def get_facility_schedule_window_for_timestamp(
    sections: List[Dict[str, object]],
    ts: datetime,
    date_range_cache: Dict[Tuple[str, int], Optional[Tuple[date, date, int]]],
    weekday_cache: Dict[str, Optional[Set[int]]],
    hours_window_cache: Dict[str, Optional[Tuple[int, int, bool]]],
) -> Optional[Tuple[int, int, bool]]:
    if not sections:
        return None

    current_day = ts.date()
    current_year = current_day.year
    current_weekday = int(ts.weekday())
    candidates: List[Tuple[int, int, int, int, Tuple[int, int, bool]]] = []

    for section_idx, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        section_title = str(section.get("title", "")).strip()
        if not is_schedule_facility_wide_section(section_title):
            continue

        section_title_norm = normalize_schedule_text(section_title)
        title_range_key = (section_title_norm, current_year)
        if title_range_key not in date_range_cache:
            date_range_cache[title_range_key] = parse_schedule_date_range(section_title, current_year)
        title_range = date_range_cache.get(title_range_key)

        rows = section.get("rows")
        if not isinstance(rows, list):
            continue

        for row_idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            label = str(row.get("label", "")).strip()
            hours = str(row.get("hours", "")).strip()
            if not label or not hours:
                continue

            label_norm = normalize_schedule_text(label)
            row_range_key = (label_norm, current_year)
            if row_range_key not in date_range_cache:
                date_range_cache[row_range_key] = parse_schedule_date_range(label, current_year)
            row_range = date_range_cache.get(row_range_key)

            specificity = 1
            span_days = 9999
            is_date_match = False

            if row_range is not None:
                start_day, end_day, row_span_days = row_range
                is_date_match = start_day <= current_day <= end_day
                specificity = 3
                span_days = row_span_days
            else:
                if label_norm not in weekday_cache:
                    weekday_cache[label_norm] = parse_schedule_weekday_set(label)
                weekday_set = weekday_cache.get(label_norm)
                if not weekday_set:
                    continue
                if title_range is not None:
                    title_start, title_end, title_span_days = title_range
                    if not (title_start <= current_day <= title_end):
                        continue
                    specificity = 2
                    span_days = title_span_days
                else:
                    specificity = 1
                    span_days = 9998
                is_date_match = current_weekday in weekday_set

            if not is_date_match:
                continue

            hours_norm = normalize_schedule_text(hours)
            if hours_norm not in hours_window_cache:
                hours_window_cache[hours_norm] = parse_schedule_hours_window(hours)
            hours_window = hours_window_cache.get(hours_norm)
            if hours_window is None:
                continue

            candidates.append(
                (
                    -specificity,
                    span_days,
                    section_idx,
                    row_idx,
                    hours_window,
                )
            )

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][4]


def schedule_phase_features_for_target(
    sections: List[Dict[str, object]],
    ts: datetime,
    date_range_cache: Dict[Tuple[str, int], Optional[Tuple[date, date, int]]],
    weekday_cache: Dict[str, Optional[Set[int]]],
    hours_window_cache: Dict[str, Optional[Tuple[int, int, bool]]],
) -> List[float]:
    zeros = [0.0] * config.SCHEDULE_PHASE_FEATURE_COUNT
    window = get_facility_schedule_window_for_timestamp(
        sections=sections,
        ts=ts,
        date_range_cache=date_range_cache,
        weekday_cache=weekday_cache,
        hours_window_cache=hours_window_cache,
    )
    if window is None:
        return zeros

    start_minutes, end_minutes, is_closed = window
    if is_closed:
        return zeros

    minute_of_day = int(ts.hour) * 60 + int(ts.minute)
    close_boundary = False
    _open_boundary, close_boundary = schedule_boundary_for_window(window, minute_of_day)

    minute_abs = int(minute_of_day)
    if end_minutes > config.SCHEDULE_MINUTES_PER_DAY and minute_abs < int(start_minutes):
        minute_abs += config.SCHEDULE_MINUTES_PER_DAY

    duration = max(1.0, float(end_minutes - start_minutes))
    is_open = bool(start_minutes <= minute_abs < end_minutes)

    if is_open:
        since_open = max(0.0, float(minute_abs - start_minutes))
        until_close = max(0.0, float(end_minutes - minute_abs))
        progress = max(0.0, min(1.0, since_open / duration))
        return [
            1.0,
            float(progress),
            float(max(0.0, min(1.0, since_open / 720.0))),
            float(max(0.0, min(1.0, until_close / 720.0))),
            1.0 if since_open <= 60.0 else 0.0,
            1.0 if until_close <= 60.0 else 0.0,
        ]

    if close_boundary:
        return [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]

    return zeros


def schedule_transition_weight_from_phase(phase_features: List[float]) -> float:
    if not config.SCHEDULE_TRANSITION_WEIGHT_ENABLED:
        return 1.0
    if not isinstance(phase_features, list) or len(phase_features) < config.SCHEDULE_PHASE_FEATURE_COUNT:
        return 1.0
    near_open = max(0.0, min(1.0, float(to_float_or_none(phase_features[4]) or 0.0)))
    near_close = max(0.0, min(1.0, float(to_float_or_none(phase_features[5]) or 0.0)))
    edge_strength = max(near_open, near_close)
    if edge_strength <= 0.0:
        return 1.0
    mult = max(1.0, float(config.SCHEDULE_TRANSITION_WEIGHT_MULTIPLIER))
    return float(1.0 + (mult - 1.0) * edge_strength)


def schedule_phase_features_for_location_target(
    loc_data: Dict[str, object],
    target: datetime,
) -> List[float]:
    zeros = [0.0] * config.SCHEDULE_PHASE_FEATURE_COUNT
    schedule_sections_raw = loc_data.get("schedule_sections", [])
    schedule_sections = schedule_sections_raw if isinstance(schedule_sections_raw, list) else []
    if not schedule_sections:
        return zeros

    minute_of_day = int(target.hour) * 60 + int(target.minute)
    cache_key = (target.date(), minute_of_day)
    phase_cache = loc_data.get("schedule_phase_cache")
    if not isinstance(phase_cache, dict):
        phase_cache = {}
        loc_data["schedule_phase_cache"] = phase_cache

    cached_phase = phase_cache.get(cache_key)
    if (
        isinstance(cached_phase, (tuple, list))
        and len(cached_phase) == config.SCHEDULE_PHASE_FEATURE_COUNT
    ):
        return [float(to_float_or_none(v) or 0.0) for v in cached_phase]

    schedule_date_cache = loc_data.get("schedule_date_range_cache")
    if not isinstance(schedule_date_cache, dict):
        schedule_date_cache = {}
        loc_data["schedule_date_range_cache"] = schedule_date_cache
    schedule_weekday_cache = loc_data.get("schedule_weekday_cache")
    if not isinstance(schedule_weekday_cache, dict):
        schedule_weekday_cache = {}
        loc_data["schedule_weekday_cache"] = schedule_weekday_cache
    schedule_hours_cache = loc_data.get("schedule_hours_cache")
    if not isinstance(schedule_hours_cache, dict):
        schedule_hours_cache = {}
        loc_data["schedule_hours_cache"] = schedule_hours_cache

    phase = schedule_phase_features_for_target(
        sections=schedule_sections,
        ts=target,
        date_range_cache=schedule_date_cache,
        weekday_cache=schedule_weekday_cache,
        hours_window_cache=schedule_hours_cache,
    )
    phase_cache[cache_key] = tuple(float(v) for v in phase)
    return [float(v) for v in phase]


def schedule_transition_weight_for_location_target(
    loc_data: Dict[str, object],
    target: datetime,
) -> float:
    phase = schedule_phase_features_for_location_target(loc_data, target)
    return schedule_transition_weight_from_phase(phase)


def get_facility_schedule_open_state(
    sections: List[Dict[str, object]],
    ts: datetime,
    date_range_cache: Dict[Tuple[str, int], Optional[Tuple[date, date, int]]],
    weekday_cache: Dict[str, Optional[Set[int]]],
    hours_window_cache: Dict[str, Optional[Tuple[int, int, bool]]],
) -> Optional[bool]:
    if not sections:
        return None

    current_day = ts.date()
    current_year = current_day.year
    current_weekday = int(ts.weekday())
    minute_of_day = int(ts.hour) * 60 + int(ts.minute)
    candidates: List[Tuple[int, int, int, int, bool]] = []

    for section_idx, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        section_title = str(section.get("title", "")).strip()
        if not is_schedule_facility_wide_section(section_title):
            continue

        section_title_norm = normalize_schedule_text(section_title)
        title_range_key = (section_title_norm, current_year)
        if title_range_key not in date_range_cache:
            date_range_cache[title_range_key] = parse_schedule_date_range(section_title, current_year)
        title_range = date_range_cache.get(title_range_key)

        rows = section.get("rows")
        if not isinstance(rows, list):
            continue

        for row_idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            label = str(row.get("label", "")).strip()
            hours = str(row.get("hours", "")).strip()
            if not label or not hours:
                continue

            label_norm = normalize_schedule_text(label)
            row_range_key = (label_norm, current_year)
            if row_range_key not in date_range_cache:
                date_range_cache[row_range_key] = parse_schedule_date_range(label, current_year)
            row_range = date_range_cache.get(row_range_key)

            specificity = 1
            span_days = 9999
            is_date_match = False

            if row_range is not None:
                start_day, end_day, row_span_days = row_range
                is_date_match = start_day <= current_day <= end_day
                specificity = 3
                span_days = row_span_days
            else:
                if label_norm not in weekday_cache:
                    weekday_cache[label_norm] = parse_schedule_weekday_set(label)
                weekday_set = weekday_cache.get(label_norm)
                if not weekday_set:
                    continue
                if title_range is not None:
                    title_start, title_end, title_span_days = title_range
                    if not (title_start <= current_day <= title_end):
                        continue
                    specificity = 2
                    span_days = title_span_days
                else:
                    specificity = 1
                    span_days = 9998
                is_date_match = current_weekday in weekday_set

            if not is_date_match:
                continue

            hours_norm = normalize_schedule_text(hours)
            if hours_norm not in hours_window_cache:
                hours_window_cache[hours_norm] = parse_schedule_hours_window(hours)
            hours_window = hours_window_cache.get(hours_norm)
            if hours_window is None:
                continue

            is_open = schedule_open_for_window(hours_window, minute_of_day)
            candidates.append((-specificity, span_days, section_idx, row_idx, is_open))

    if not candidates:
        return None
    candidates.sort()
    return bool(candidates[0][4])


def to_local(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = config.DB_TZ.localize(dt)
    else:
        dt = dt.astimezone(config.DB_TZ)
    return dt.astimezone(config.TZ)


def floor_time(dt: datetime, minutes: int) -> datetime:
    minutes = max(1, minutes)
    dt = dt.replace(second=0, microsecond=0)
    return dt.replace(minute=dt.minute - (dt.minute % minutes))


def aggregate_sum_count(store: Dict, key, value: float) -> None:
    if key in store:
        store[key][0] += value
        store[key][1] += 1.0
    else:
        store[key] = [value, 1.0]


def finalize_averages(store: Dict) -> Dict:
    return {k: (v[0] / v[1], int(v[1])) for k, v in store.items()}


def parse_iso_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except Exception:
        return None
    if dt.tzinfo is None:
        return config.TZ.localize(dt)
    return dt.astimezone(config.TZ)


def parse_observed_at_value(raw_value) -> Optional[datetime]:
    if raw_value is None:
        return None

    if isinstance(raw_value, datetime):
        return to_local(raw_value)

    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return None

        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        parsed = None
        try:
            parsed = datetime.fromisoformat(normalized)
        except Exception:
            for fmt in (
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
            ):
                try:
                    parsed = datetime.strptime(normalized, fmt)
                    break
                except Exception:
                    continue

        if parsed is None:
            return None

        if parsed.tzinfo is None:
            parsed = config.DB_TZ.localize(parsed)
        else:
            parsed = parsed.astimezone(config.DB_TZ)
        return parsed.astimezone(config.TZ)

    return None


def model_feature_count(loc_count: int) -> int:
    # time/cycle + calendar + lags/sensor + schedule-phase + weather + one-hot
    return (
        17
        + config.CALENDAR_FEATURE_COUNT
        + config.LAG_TREND_SENSOR_FEATURE_COUNT
        + config.SCHEDULE_PHASE_FEATURE_COUNT
        + (len(config.WEATHER_KEYS) * 3)
        + len(config.WEATHER_ROLLING_KEYS)
        + config.WEATHER_DERIVED_FEATURE_COUNT
        + config.WEATHER_QUALITY_FEATURE_COUNT
        + loc_count
    )


def last_before(times: List[datetime], values: List[float], target: datetime) -> Tuple[float, float]:
    idx = bisect.bisect_left(times, target) - 1
    if idx >= 0:
        return values[idx], (target - times[idx]).total_seconds() / 60.0
    return float("nan"), float("nan")


def last_before_with_overrides(
    times: List[datetime],
    values: List[float],
    target: datetime,
    lag_ratio_override: Optional[Dict[datetime, float]] = None,
) -> Tuple[float, float]:
    base_value, base_minutes = last_before(times, values, target)
    if not lag_ratio_override:
        return base_value, base_minutes

    latest_ts = None
    latest_val = None
    for ts, raw in lag_ratio_override.items():
        if ts >= target:
            continue
        numeric = to_float_or_none(raw)
        if numeric is None:
            continue
        if latest_ts is None or ts > latest_ts:
            latest_ts = ts
            latest_val = float(numeric)

    if latest_ts is None or latest_val is None:
        return base_value, base_minutes

    override_minutes = (target - latest_ts).total_seconds() / 60.0
    if math.isnan(base_minutes) or override_minutes < base_minutes:
        return latest_val, float(override_minutes)
    return base_value, base_minutes


def date_in_ranges(target_date: date, ranges: List[Tuple[date, date]]) -> bool:
    for start, end in ranges:
        if start <= target_date <= end:
            return True
    return False


def range_progress(target_date: date, ranges: List[Tuple[date, date]]) -> float:
    for start, end in ranges:
        if start <= target_date <= end:
            total = max(1, (end - start).days)
            return max(0.0, min(1.0, float((target_date - start).days) / float(total)))
    return -1.0


def most_recent_date_distance(target_date: date, anchors: List[date], max_days: int) -> float:
    latest = None
    for anchor in anchors:
        if anchor <= target_date:
            latest = anchor
        else:
            break
    if latest is None:
        return -1.0
    delta = (target_date - latest).days
    return max(0.0, min(1.0, float(delta) / float(max(1, max_days))))


def is_pre_exam_week(target_date: date) -> bool:
    for exam_start in config.ACADEMIC_EXAM_START_DATES:
        gap = (exam_start - target_date).days
        if 1 <= gap <= 7:
            return True
    return False


def build_calendar_features(dt: datetime) -> List[float]:
    d = dt.date()
    in_fall = date_in_ranges(d, config.ACADEMIC_FALL_INSTRUCTION)
    in_spring = date_in_ranges(d, config.ACADEMIC_SPRING_INSTRUCTION)
    in_exams = date_in_ranges(d, config.ACADEMIC_EXAMS)
    in_thanksgiving = date_in_ranges(d, config.ACADEMIC_THANKSGIVING_RECESS)
    in_spring_recess = date_in_ranges(d, config.ACADEMIC_SPRING_RECESS)
    in_summer = date_in_ranges(d, config.ACADEMIC_SUMMER_SESSION)

    term_progress = max(
        range_progress(d, config.ACADEMIC_FALL_INSTRUCTION),
        range_progress(d, config.ACADEMIC_SPRING_INSTRUCTION),
        range_progress(d, config.ACADEMIC_SUMMER_SESSION),
    )
    since_term_start = most_recent_date_distance(d, config.ACADEMIC_TERM_START_DATES, max_days=150)

    features = [
        float(in_fall),
        float(in_spring),
        float(in_exams),
        float(d in config.ACADEMIC_STUDY_DAYS),
        float(in_thanksgiving),
        float(in_spring_recess),
        float(in_summer),
        float(d in config.ACADEMIC_HOLIDAYS),
        float(d in config.ACADEMIC_COMMENCEMENT_DAYS),
        float(d in config.ACADEMIC_GRADING_DEADLINES),
        float(is_pre_exam_week(d)),
        float(term_progress if term_progress >= 0.0 else since_term_start),
    ]
    return features


def build_time_features(dt: datetime) -> List[float]:
    hour = dt.hour
    minute = dt.minute
    quarter_slot = minute // max(1, config.RESAMPLE_MINUTES)
    dow = dt.weekday()
    month = dt.month
    day_of_year = dt.timetuple().tm_yday
    is_weekend = 1 if dow >= 5 else 0

    hour_rad = 2 * math.pi * hour / 24
    minute_of_day = hour * 60 + minute
    minute_of_day_rad = 2 * math.pi * minute_of_day / 1440.0
    dow_rad = 2 * math.pi * dow / 7
    month_rad = 2 * math.pi * (month - 1) / 12
    doy_rad = 2 * math.pi * day_of_year / 365.25

    return [
        float(hour),
        float(minute),
        float(quarter_slot),
        float(dow),
        float(month),
        float(day_of_year),
        float(is_weekend),
        math.sin(hour_rad),
        math.cos(hour_rad),
        math.sin(minute_of_day_rad),
        math.cos(minute_of_day_rad),
        math.sin(dow_rad),
        math.cos(dow_rad),
        math.sin(month_rad),
        math.cos(month_rad),
        math.sin(doy_rad),
        math.cos(doy_rad),
    ] + build_calendar_features(dt)


def ratio_value_from_maps(
    bucket_map: Dict[datetime, float],
    ts: datetime,
    lag_ratio_override: Optional[Dict[datetime, float]] = None,
) -> Optional[float]:
    if lag_ratio_override is not None:
        override = lag_ratio_override.get(ts)
        numeric = to_float_or_none(override)
        if numeric is not None:
            return float(numeric)
    raw = bucket_map.get(ts)
    numeric = to_float_or_none(raw)
    return float(numeric) if numeric is not None else None


def rolling_mean(
    bucket_map: Dict[datetime, float],
    target: datetime,
    steps: int,
    lag_ratio_override: Optional[Dict[datetime, float]] = None,
) -> float:
    values = []
    for i in range(1, steps + 1):
        ts = target - timedelta(minutes=config.RESAMPLE_MINUTES * i)
        value = ratio_value_from_maps(
            bucket_map=bucket_map,
            ts=ts,
            lag_ratio_override=lag_ratio_override,
        )
        if value is not None:
            values.append(value)
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def rolling_std(
    bucket_map: Dict[datetime, float],
    target: datetime,
    steps: int,
    lag_ratio_override: Optional[Dict[datetime, float]] = None,
) -> float:
    values: List[float] = []
    for i in range(1, steps + 1):
        ts = target - timedelta(minutes=config.RESAMPLE_MINUTES * i)
        value = ratio_value_from_maps(
            bucket_map=bucket_map,
            ts=ts,
            lag_ratio_override=lag_ratio_override,
        )
        if value is not None:
            values.append(float(value))
    if len(values) < 2:
        return float("nan")
    return float(np.std(np.array(values, dtype=np.float32), ddof=0))


def rolling_range(
    bucket_map: Dict[datetime, float],
    target: datetime,
    steps: int,
    lag_ratio_override: Optional[Dict[datetime, float]] = None,
) -> float:
    values: List[float] = []
    for i in range(1, steps + 1):
        ts = target - timedelta(minutes=config.RESAMPLE_MINUTES * i)
        value = ratio_value_from_maps(
            bucket_map=bucket_map,
            ts=ts,
            lag_ratio_override=lag_ratio_override,
        )
        if value is not None:
            values.append(float(value))
    if not values:
        return float("nan")
    return float(max(values) - min(values))


def count_consecutive_flatline_steps(
    bucket_map: Dict[datetime, float],
    target: datetime,
    steps: int,
    tolerance: float = 0.002,
) -> int:
    prev = None
    count = 0
    for i in range(1, max(1, steps) + 1):
        ts = target - timedelta(minutes=config.RESAMPLE_MINUTES * i)
        value = bucket_map.get(ts)
        if value is None:
            break
        value = float(value)
        if prev is None:
            prev = value
            count = 1
            continue
        if abs(value - prev) <= tolerance:
            count += 1
            prev = value
            continue
        break
    return count


def recent_missing_ratio(
    bucket_map: Dict[datetime, float],
    target: datetime,
    steps: int,
) -> float:
    total = max(1, int(steps))
    missing = 0
    for i in range(1, total + 1):
        ts = target - timedelta(minutes=config.RESAMPLE_MINUTES * i)
        if bucket_map.get(ts) is None:
            missing += 1
    return float(missing) / float(total)


def sensor_quality_signals(
    target: datetime,
    bucket_map: Dict[datetime, float],
    raw_times: List[datetime],
    raw_values: List[float],
) -> Tuple[float, float, float]:
    _last_raw, minutes_since_raw = last_before(raw_times, raw_values, target)
    flatline_steps = count_consecutive_flatline_steps(
        bucket_map=bucket_map,
        target=target,
        steps=max(2, int(120 / max(1, config.RESAMPLE_MINUTES))),
    )
    missing_ratio = recent_missing_ratio(
        bucket_map=bucket_map,
        target=target,
        steps=max(2, int(120 / max(1, config.RESAMPLE_MINUTES))),
    )
    return (
        float(flatline_steps),
        float(missing_ratio),
        float(minutes_since_raw),
    )


def sensor_quality_weight(
    flatline_steps: float,
    missing_ratio: float,
    minutes_since_raw: float,
) -> float:
    steps = max(0.0, float(flatline_steps))
    missing = max(0.0, min(1.0, float(missing_ratio)))
    age = max(0.0, float(minutes_since_raw))

    flatline_penalty = max(0.0, min(0.6, (steps / 8.0) * 0.6))
    missing_penalty = max(0.0, min(0.5, missing * 0.5))
    age_penalty = max(0.0, min(0.4, max(0.0, age - 120.0) / 360.0))

    weight = 1.0 - flatline_penalty - missing_penalty - age_penalty
    return max(float(config.SENSOR_WEIGHT_MIN), min(1.0, float(weight)))


def feature_missing_rate(feature_row: List[float]) -> float:
    arr = np.asarray(feature_row, dtype=np.float32).reshape(-1)
    if arr.size <= 0:
        return 1.0
    finite = np.isfinite(arr)
    return float(1.0 - (float(np.count_nonzero(finite)) / float(arr.size)))


def feature_quality_weight(feature_row: List[float]) -> float:
    if not config.MODEL_FEATURE_QUALITY_WEIGHT_ENABLED:
        return 1.0

    missing = max(0.0, min(1.0, feature_missing_rate(feature_row)))
    completeness = 1.0 - missing
    min_w = max(0.05, min(1.0, float(config.MODEL_FEATURE_QUALITY_WEIGHT_MIN)))
    power = max(0.25, min(4.0, float(config.MODEL_FEATURE_QUALITY_WEIGHT_POWER)))
    weight = min_w + (1.0 - min_w) * (completeness ** power)
    return max(min_w, min(1.0, float(weight)))


def to_float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def configured_direct_horizon_hours() -> List[int]:
    cache_key = (bool(config.MODEL_DIRECT_HORIZON_ENABLED), str(config.MODEL_DIRECT_HORIZON_HOURS_RAW or ""))
    cached = getattr(configured_direct_horizon_hours, "_cache", None)
    if isinstance(cached, dict) and cache_key in cached:
        return list(cached[cache_key])

    if not config.MODEL_DIRECT_HORIZON_ENABLED:
        parsed_out: List[int] = []
        if not isinstance(cached, dict):
            cached = {}
        cached[cache_key] = parsed_out
        setattr(configured_direct_horizon_hours, "_cache", cached)
        return parsed_out

    raw = cache_key[1]
    tokens = [part.strip() for part in re.split(r"[,\s;]+", raw) if part.strip()]
    parsed: List[int] = []
    seen: Set[int] = set()
    for token in tokens:
        try:
            hours = int(float(token))
        except Exception:
            continue
        if hours <= 0:
            continue
        hours = min(72, hours)
        if hours in seen:
            continue
        seen.add(hours)
        parsed.append(hours)
    if not parsed:
        parsed = [1, 2, 3, 6, 12]
    else:
        parsed = sorted(parsed)

    if not isinstance(cached, dict):
        cached = {}
    cached[cache_key] = parsed
    setattr(configured_direct_horizon_hours, "_cache", cached)
    return list(parsed)


def profile_ratio_for_location_target(loc_data: Dict[str, object], target: datetime) -> float:
    dow_hour = loc_data.get("fallback_avg_dow_hour")
    if isinstance(dow_hour, dict):
        value = dow_hour.get((target.weekday(), target.hour))
        if isinstance(value, (tuple, list)) and value:
            ratio = to_float_or_none(value[0])
            if ratio is not None:
                return float(ratio)

    hourly = loc_data.get("fallback_avg_hour")
    if isinstance(hourly, dict):
        value = hourly.get(target.hour)
        if isinstance(value, (tuple, list)) and value:
            ratio = to_float_or_none(value[0])
            if ratio is not None:
                return float(ratio)

    overall = loc_data.get("fallback_avg_overall")
    if isinstance(overall, (tuple, list)) and overall:
        ratio = to_float_or_none(overall[0])
        if ratio is not None:
            return float(ratio)

    return float("nan")


def weather_last_before(
    times: List[datetime],
    weather_map: Dict[datetime, Dict[str, float]],
    target: datetime,
    key: str,
) -> float:
    idx = bisect.bisect_left(times, target) - 1
    while idx >= 0:
        row = weather_map.get(times[idx], {})
        value = row.get(key)
        numeric = to_float_or_none(value)
        if numeric is not None:
            return float(numeric)
        idx -= 1
    return float("nan")


def weather_value_at_or_before(
    times: List[datetime],
    weather_map: Dict[datetime, Dict[str, float]],
    target: datetime,
    key: str,
    cache: Optional[Dict[Tuple[datetime, str], float]] = None,
) -> float:
    cache_key = None
    if cache is not None:
        cache_key = (target, key)
        cached = cache.get(cache_key)
        if cached is not None:
            return float(cached)

    value = float("nan")
    row = weather_map.get(target)
    if row is not None:
        numeric = to_float_or_none(row.get(key))
        if numeric is not None:
            value = float(numeric)

    if math.isnan(value):
        value = weather_last_before(times, weather_map, target, key)

    if cache is not None and cache_key is not None:
        cache[cache_key] = float(value)
    return value


def weather_rolling_mean(
    times: List[datetime],
    weather_map: Dict[datetime, Dict[str, float]],
    target: datetime,
    steps: int,
    key: str,
    cache: Optional[Dict[Tuple[datetime, str], float]] = None,
) -> float:
    values = []
    for i in range(1, steps + 1):
        ts = target - timedelta(minutes=config.RESAMPLE_MINUTES * i)
        value = weather_value_at_or_before(times, weather_map, ts, key, cache=cache)
        if not math.isnan(value):
            values.append(value)
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def weather_code_family_flags(weather_code: Optional[float]) -> List[float]:
    numeric = to_float_or_none(weather_code)
    if numeric is None:
        return [0.0] * 6
    code = int(round(float(numeric)))
    clear = 1.0 if code in {0, 1} else 0.0
    cloudy = 1.0 if code in {2, 3} else 0.0
    fog = 1.0 if code in {45, 48} else 0.0
    rain = 1.0 if (51 <= code <= 67) or (80 <= code <= 82) else 0.0
    snow = 1.0 if (71 <= code <= 77) or (85 <= code <= 86) else 0.0
    storm = 1.0 if 95 <= code <= 99 else 0.0
    return [clear, cloudy, fog, rain, snow, storm]


def build_weather_derived_features(weather_now: Dict[str, float]) -> List[float]:
    precip_mm = to_float_or_none(weather_now.get("precip_mm"))
    rain_mm = to_float_or_none(weather_now.get("rain_mm"))
    snow_cm = to_float_or_none(weather_now.get("snow_cm"))
    temp_c = to_float_or_none(weather_now.get("temp_c"))
    wind_mps = to_float_or_none(weather_now.get("wind_mps"))
    wind_gust_mps = to_float_or_none(weather_now.get("wind_gust_mps"))
    humidity_pct = to_float_or_none(weather_now.get("humidity_pct"))
    weather_code = to_float_or_none(weather_now.get("weather_code"))

    precip_any = 1.0 if any(v is not None and v > 0.05 for v in (precip_mm, rain_mm, snow_cm)) else 0.0
    heavy_precip = 1.0 if any(v is not None and v >= 1.5 for v in (precip_mm, rain_mm)) else 0.0
    if snow_cm is not None and snow_cm >= 0.8:
        heavy_precip = 1.0
    cold = 1.0 if temp_c is not None and temp_c <= 2.0 else 0.0
    hot = 1.0 if temp_c is not None and temp_c >= 27.0 else 0.0
    windy = 1.0 if (wind_mps is not None and wind_mps >= 6.0) else 0.0
    if wind_gust_mps is not None and wind_gust_mps >= 10.0:
        windy = 1.0
    humid = 1.0 if humidity_pct is not None and humidity_pct >= 80.0 else 0.0
    return [
        float(precip_any),
        float(heavy_precip),
        float(cold),
        float(hot),
        float(windy),
        float(humid),
    ] + weather_code_family_flags(weather_code)


def build_weather_quality_features(
    missing_now: int,
    missing_1h: int,
    missing_delta: int,
    missing_roll: int,
) -> List[float]:
    now_den = max(1, len(config.WEATHER_KEYS))
    roll_den = max(1, len(config.WEATHER_ROLLING_KEYS))
    return [
        float(max(0.0, min(1.0, float(missing_now) / float(now_den)))),
        float(max(0.0, min(1.0, float(missing_1h) / float(now_den)))),
        float(max(0.0, min(1.0, float(missing_delta) / float(now_den)))),
        float(max(0.0, min(1.0, float(missing_roll) / float(roll_den)))),
    ]


def finite_missing_ratio(values: List[float]) -> float:
    if not values:
        return 1.0
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size <= 0:
        return 1.0
    finite = np.isfinite(arr)
    return float(1.0 - (float(np.count_nonzero(finite)) / float(arr.size)))


def build_features(
    target: datetime,
    loc_data: Dict[str, object],
    onehot: List[float],
    weather_source: Optional[Dict[str, object]] = None,
    weather_lookup_cache: Optional[Dict[Tuple[datetime, str], float]] = None,
    lag_ratio_override: Optional[Dict[datetime, float]] = None,
) -> List[float]:
    bucket_map = loc_data["bucket_map"]
    bucket_times = loc_data["bucket_times"]
    bucket_values = loc_data["bucket_values"]
    raw_times = loc_data["raw_times"]
    raw_values = loc_data["raw_values"]
    if weather_source is not None:
        weather_bucket_map = weather_source.get("map", {})
        weather_bucket_times = weather_source.get("times", [])
    else:
        weather_bucket_map = loc_data.get("weather_bucket_map", {})
        weather_bucket_times = loc_data.get("weather_bucket_times", [])

    lag_15m = ratio_value_from_maps(
        bucket_map,
        target - timedelta(minutes=config.RESAMPLE_MINUTES),
        lag_ratio_override=lag_ratio_override,
    )
    lag_1h = ratio_value_from_maps(
        bucket_map,
        target - timedelta(hours=1),
        lag_ratio_override=lag_ratio_override,
    )
    lag_2h = ratio_value_from_maps(
        bucket_map,
        target - timedelta(hours=2),
        lag_ratio_override=lag_ratio_override,
    )
    lag_3h = ratio_value_from_maps(
        bucket_map,
        target - timedelta(hours=3),
        lag_ratio_override=lag_ratio_override,
    )
    lag_12h = ratio_value_from_maps(
        bucket_map,
        target - timedelta(hours=12),
        lag_ratio_override=lag_ratio_override,
    )
    lag_24h = ratio_value_from_maps(
        bucket_map,
        target - timedelta(hours=24),
        lag_ratio_override=lag_ratio_override,
    )
    lag_7d = ratio_value_from_maps(
        bucket_map,
        target - timedelta(days=7),
        lag_ratio_override=lag_ratio_override,
    )

    if lag_15m is None:
        lag_15m = profile_ratio_for_location_target(
            loc_data,
            target - timedelta(minutes=config.RESAMPLE_MINUTES),
        )
    if lag_1h is None:
        lag_1h = profile_ratio_for_location_target(
            loc_data,
            target - timedelta(hours=1),
        )
    if lag_2h is None:
        lag_2h = profile_ratio_for_location_target(
            loc_data,
            target - timedelta(hours=2),
        )
    if lag_3h is None:
        lag_3h = profile_ratio_for_location_target(
            loc_data,
            target - timedelta(hours=3),
        )
    if lag_12h is None:
        lag_12h = profile_ratio_for_location_target(
            loc_data,
            target - timedelta(hours=12),
        )
    if lag_24h is None:
        lag_24h = profile_ratio_for_location_target(
            loc_data,
            target - timedelta(hours=24),
        )
    if lag_7d is None:
        lag_7d = profile_ratio_for_location_target(
            loc_data,
            target - timedelta(days=7),
        )

    last_bucket, minutes_since_bucket = last_before_with_overrides(
        bucket_times,
        bucket_values,
        target,
        lag_ratio_override=lag_ratio_override,
    )
    flatline_steps_recent, missing_ratio_recent, minutes_since_raw = sensor_quality_signals(
        target=target,
        bucket_map=bucket_map,
        raw_times=raw_times,
        raw_values=raw_values,
    )

    delta_1h = float("nan")
    delta_3h = float("nan")
    delta_12h = float("nan")
    delta_24h = float("nan")
    delta_7d = float("nan")
    if not math.isnan(lag_15m) and not math.isnan(lag_1h):
        delta_1h = lag_15m - lag_1h
    if not math.isnan(lag_15m) and not math.isnan(lag_3h):
        delta_3h = lag_15m - lag_3h
    if not math.isnan(lag_15m) and not math.isnan(lag_12h):
        delta_12h = lag_15m - lag_12h
    if not math.isnan(lag_15m) and not math.isnan(lag_24h):
        delta_24h = lag_15m - lag_24h
    if not math.isnan(lag_15m) and not math.isnan(lag_7d):
        delta_7d = lag_15m - lag_7d

    roll_1h = rolling_mean(
        bucket_map,
        target,
        steps=max(1, int(60 / config.RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    roll_2h = rolling_mean(
        bucket_map,
        target,
        steps=max(1, int(120 / config.RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    roll_6h = rolling_mean(
        bucket_map,
        target,
        steps=max(1, int(360 / config.RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    roll_12h = rolling_mean(
        bucket_map,
        target,
        steps=max(1, int(720 / config.RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    roll_24h = rolling_mean(
        bucket_map,
        target,
        steps=max(1, int(1440 / config.RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    vol_1h = rolling_std(
        bucket_map,
        target,
        steps=max(2, int(60 / config.RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    vol_3h = rolling_std(
        bucket_map,
        target,
        steps=max(2, int(180 / config.RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    vol_6h = rolling_std(
        bucket_map,
        target,
        steps=max(2, int(360 / config.RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    range_1h = rolling_range(
        bucket_map,
        target,
        steps=max(2, int(60 / config.RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    range_3h = rolling_range(
        bucket_map,
        target,
        steps=max(2, int(180 / config.RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    range_6h = rolling_range(
        bucket_map,
        target,
        steps=max(2, int(360 / config.RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    trend_short = float("nan")
    trend_mid = float("nan")
    trend_long = float("nan")
    trend_accel_short = float("nan")
    trend_accel_mid = float("nan")
    if not math.isnan(roll_1h) and not math.isnan(roll_2h):
        trend_short = roll_1h - roll_2h
    if not math.isnan(roll_2h) and not math.isnan(roll_6h):
        trend_mid = roll_2h - roll_6h
    if not math.isnan(roll_2h) and not math.isnan(roll_24h):
        trend_long = roll_2h - roll_24h
    if not math.isnan(trend_short) and not math.isnan(trend_mid):
        trend_accel_short = trend_short - trend_mid
    if not math.isnan(trend_mid) and not math.isnan(trend_long):
        trend_accel_mid = trend_mid - trend_long
    lag_missing_ratio = finite_missing_ratio(
        [
            float(lag_15m),
            float(lag_1h),
            float(lag_2h),
            float(lag_3h),
            float(lag_12h),
            float(lag_24h),
            float(lag_7d),
        ]
    )
    delta_missing_ratio = finite_missing_ratio(
        [
            float(delta_1h),
            float(delta_3h),
            float(delta_12h),
            float(delta_24h),
            float(delta_7d),
        ]
    )
    trend_missing_ratio = finite_missing_ratio(
        [
            float(trend_short),
            float(trend_mid),
            float(trend_long),
            float(trend_accel_short),
            float(trend_accel_mid),
        ]
    )
    roll_missing_ratio = finite_missing_ratio(
        [
            float(roll_1h),
            float(roll_2h),
            float(roll_6h),
            float(roll_12h),
            float(roll_24h),
            float(vol_1h),
            float(vol_3h),
            float(vol_6h),
            float(range_1h),
            float(range_3h),
            float(range_6h),
        ]
    )
    dynamics_missing_ratio = max(
        0.0,
        min(
            1.0,
            float(
                (
                    lag_missing_ratio
                    + delta_missing_ratio
                    + trend_missing_ratio
                    + roll_missing_ratio
                )
                / 4.0
            ),
        ),
    )
    weather_features: List[float] = []
    weather_now_values: Dict[str, float] = {}
    missing_weather_now = 0
    missing_weather_1h = 0
    missing_weather_delta = 0
    missing_weather_roll = 0
    schedule_phase_features = schedule_phase_features_for_location_target(loc_data, target)
    weather_ts = target
    lag_ts = target - timedelta(hours=1)

    for key in config.WEATHER_KEYS:
        weather_now = weather_value_at_or_before(
            weather_bucket_times,
            weather_bucket_map,
            weather_ts,
            key,
            cache=weather_lookup_cache,
        )
        weather_1h = weather_value_at_or_before(
            weather_bucket_times,
            weather_bucket_map,
            lag_ts,
            key,
            cache=weather_lookup_cache,
        )
        weather_delta_1h = float("nan")
        if not math.isnan(weather_now) and not math.isnan(weather_1h):
            weather_delta_1h = weather_now - weather_1h
        if math.isnan(weather_now):
            missing_weather_now += 1
        if math.isnan(weather_1h):
            missing_weather_1h += 1
        if math.isnan(weather_delta_1h):
            missing_weather_delta += 1

        weather_now_values[key] = float(weather_now)
        weather_features.extend(
            [
                float(weather_now),
                float(weather_1h),
                float(weather_delta_1h),
            ]
        )

    for key in config.WEATHER_ROLLING_KEYS:
        weather_roll = weather_rolling_mean(
            weather_bucket_times,
            weather_bucket_map,
            target,
            steps=max(1, int(180 / config.RESAMPLE_MINUTES)),
            key=key,
            cache=weather_lookup_cache,
        )
        weather_features.append(float(weather_roll))
        if math.isnan(weather_roll):
            missing_weather_roll += 1
    weather_derived_features = build_weather_derived_features(weather_now_values)
    weather_quality_features = build_weather_quality_features(
        missing_now=missing_weather_now,
        missing_1h=missing_weather_1h,
        missing_delta=missing_weather_delta,
        missing_roll=missing_weather_roll,
    )

    return (
        build_time_features(target)
        + [
            float(lag_15m),
            float(lag_1h),
            float(lag_2h),
            float(lag_3h),
            float(lag_12h),
            float(lag_24h),
            float(lag_7d),
            float(delta_1h),
            float(delta_3h),
            float(delta_12h),
            float(delta_24h),
            float(delta_7d),
            float(roll_1h),
            float(roll_2h),
            float(roll_6h),
            float(roll_12h),
            float(roll_24h),
            float(trend_short),
            float(trend_mid),
            float(trend_long),
            float(trend_accel_short),
            float(trend_accel_mid),
            float(vol_1h),
            float(vol_3h),
            float(vol_6h),
            float(range_1h),
            float(range_3h),
            float(range_6h),
            float(last_bucket),
            float(minutes_since_bucket),
            float(minutes_since_raw),
            float(flatline_steps_recent),
            float(missing_ratio_recent),
            float(lag_missing_ratio),
            float(delta_missing_ratio),
            float(trend_missing_ratio),
            float(roll_missing_ratio),
            float(dynamics_missing_ratio),
            float(schedule_phase_features[0]),
            float(schedule_phase_features[1]),
            float(schedule_phase_features[2]),
            float(schedule_phase_features[3]),
            float(schedule_phase_features[4]),
            float(schedule_phase_features[5]),
        ]
        + weather_features
        + weather_derived_features
        + weather_quality_features
        + onehot
    )


def dedupe_exact_timestamps(entries: List[Tuple[datetime, float]]) -> Tuple[List[Tuple[datetime, float]], int]:
    if not entries:
        return [], 0
    entries.sort(key=lambda row: row[0])
    deduped = []
    removed = 0
    idx = 0
    n = len(entries)
    while idx < n:
        ts = entries[idx][0]
        values = [entries[idx][1]]
        idx += 1
        while idx < n and entries[idx][0] == ts:
            values.append(entries[idx][1])
            idx += 1
        if len(values) > 1:
            removed += len(values) - 1
        deduped.append((ts, float(sum(values) / len(values))))
    return deduped, removed


def drop_impossible_jumps(
    entries: List[Tuple[datetime, float]],
    max_cap: int,
) -> Tuple[List[Tuple[datetime, float]], int]:
    if not entries:
        return [], 0

    cleaned = [entries[0]]
    removed = 0
    max_jump = max_cap * config.IMPOSSIBLE_JUMP_PCT

    for ts, value in entries[1:]:
        prev_ts, prev_value = cleaned[-1]
        gap_min = (ts - prev_ts).total_seconds() / 60.0
        jump = abs(value - prev_value)
        if gap_min <= config.IMPOSSIBLE_JUMP_MAX_GAP_MIN and jump > max_jump:
            removed += 1
            continue
        cleaned.append((ts, value))

    return cleaned, removed


def drop_flatline_plateaus(
    entries: List[Tuple[datetime, float]],
    max_cap: int,
) -> Tuple[List[Tuple[datetime, float]], int, int]:
    if not entries:
        return [], 0, 0

    tolerance = max(0.0, float(max_cap) * max(0.0, config.SENSOR_FLATLINE_TOLERANCE_PCT))
    max_gap = max(1.0, float(config.SENSOR_FLATLINE_MAX_GAP_MIN))
    min_duration = max(1.0, float(config.SENSOR_FLATLINE_MIN_DURATION_MIN))
    keep_interval = max(1.0, float(config.SENSOR_FLATLINE_KEEP_INTERVAL_MIN))

    cleaned: List[Tuple[datetime, float]] = []
    removed = 0
    runs_detected = 0

    def flush_run(run: List[Tuple[datetime, float]]) -> None:
        nonlocal removed, runs_detected, cleaned
        if not run:
            return

        start = run[0][0]
        end = run[-1][0]
        duration_min = (end - start).total_seconds() / 60.0
        run_value = float(run[-1][1])
        likely_stuck = run_value > 1.0 and run_value < max(1.0, float(max_cap - 1))
        is_flatline = duration_min >= min_duration and likely_stuck
        if not is_flatline:
            cleaned.extend(run)
            return

        runs_detected += 1
        kept: List[Tuple[datetime, float]] = []
        last_kept_ts = None
        for ts, value in run:
            if last_kept_ts is None or (ts - last_kept_ts).total_seconds() / 60.0 >= keep_interval:
                kept.append((ts, value))
                last_kept_ts = ts
            else:
                removed += 1
        cleaned.extend(kept)

    run: List[Tuple[datetime, float]] = [entries[0]]
    for ts, value in entries[1:]:
        prev_ts, prev_value = run[-1]
        gap_min = (ts - prev_ts).total_seconds() / 60.0
        same_level = abs(float(value) - float(prev_value)) <= tolerance
        if gap_min <= max_gap and same_level:
            run.append((ts, value))
            continue
        flush_run(run)
        run = [(ts, value)]
    flush_run(run)

    return cleaned, removed, runs_detected


def build_onehot(loc_ids: Iterable[int]) -> Dict[int, List[float]]:
    unique = sorted(set(loc_ids))
    vectors: Dict[int, List[float]] = {}
    for idx, loc_id in enumerate(unique):
        vec = [0.0] * len(unique)
        vec[idx] = 1.0
        vectors[loc_id] = vec
    return vectors


def build_recency_weights(times: List[datetime]) -> np.ndarray:
    if not times:
        return np.array([], dtype=np.float32)

    if not config.RECENCY_WEIGHT_ENABLED:
        return np.ones(len(times), dtype=np.float32)

    half_life_days = max(1.0, float(config.RECENCY_HALFLIFE_DAYS))
    min_weight = max(0.0, min(1.0, float(config.RECENCY_MIN_WEIGHT)))
    reference = max(times)
    weights: List[float] = []

    for ts in times:
        age_days = max(0.0, (reference - ts).total_seconds() / 86400.0)
        weight = 2.0 ** (-age_days / half_life_days)
        weights.append(max(min_weight, float(weight)))

    return np.array(weights, dtype=np.float32)


def build_occupancy_weights(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return np.array([], dtype=np.float32)
    if not config.OCCUPANCY_WEIGHT_ENABLED:
        return np.ones(y.size, dtype=np.float32)

    alpha = max(0.0, float(config.OCCUPANCY_WEIGHT_ALPHA))
    gamma = max(1.0, float(config.OCCUPANCY_WEIGHT_GAMMA))
    clipped = np.clip(np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=1.2, neginf=0.0), 0.0, 1.2)
    boosted = np.power(clipped, gamma)
    return (1.0 + alpha * boosted).astype(np.float32)


def build_location_balance_weight_map_from_counts(
    counts_by_loc: Dict[int, int],
) -> Dict[int, float]:
    if not isinstance(counts_by_loc, dict):
        return {}

    normalized_counts: Dict[int, int] = {}
    for raw_loc_id, raw_count in counts_by_loc.items():
        try:
            loc_id = int(raw_loc_id)
        except Exception:
            continue
        count = max(0, int(raw_count or 0))
        normalized_counts[loc_id] = count

    if not normalized_counts:
        return {}
    if not config.LOCATION_BALANCE_WEIGHT_ENABLED:
        return {loc_id: 1.0 for loc_id in normalized_counts}

    positive = [count for count in normalized_counts.values() if count > 0]
    if not positive:
        return {loc_id: 1.0 for loc_id in normalized_counts}

    reference = float(np.median(np.array(positive, dtype=np.float32)))
    if not math.isfinite(reference) or reference <= 0.0:
        return {loc_id: 1.0 for loc_id in normalized_counts}

    power = max(0.0, min(1.5, float(config.LOCATION_BALANCE_WEIGHT_POWER)))
    min_w = max(0.05, float(config.LOCATION_BALANCE_WEIGHT_MIN))
    max_w = max(min_w, float(config.LOCATION_BALANCE_WEIGHT_MAX))
    weight_map: Dict[int, float] = {}
    for loc_id, count in normalized_counts.items():
        safe_count = max(1, int(count))
        ratio = reference / float(safe_count)
        weight = float(ratio ** power) if power > 0.0 else 1.0
        weight_map[loc_id] = float(max(min_w, min(max_w, weight)))
    return weight_map


def build_location_balance_weight_map_from_loc_data(
    loc_ids: Iterable[int],
    loc_data: Dict[int, Dict[str, object]],
    since: Optional[datetime] = None,
) -> Dict[int, float]:
    counts_by_loc: Dict[int, int] = {}
    for raw_loc_id in loc_ids:
        try:
            loc_id = int(raw_loc_id)
        except Exception:
            continue
        data = loc_data.get(loc_id)
        if not isinstance(data, dict):
            continue
        bucket_times = data.get("bucket_times", [])
        if not isinstance(bucket_times, list):
            continue
        if isinstance(since, datetime):
            count = sum(1 for ts in bucket_times if isinstance(ts, datetime) and ts >= since)
        else:
            count = len(bucket_times)
        counts_by_loc[loc_id] = int(max(0, count))
    return build_location_balance_weight_map_from_counts(counts_by_loc)


def location_balance_weight_for_loc(
    weight_map: Optional[Dict[int, float]],
    loc_id: int,
) -> float:
    if not isinstance(weight_map, dict):
        return 1.0
    return float(weight_map.get(int(loc_id), 1.0))


def build_model_observation_dataset(
    loc_ids: Iterable[int],
    loc_data: Dict[int, Dict[str, object]],
    onehot: Dict[int, List[float]],
    weather_source: Optional[Dict[str, object]],
    location_balance_map: Optional[Dict[int, float]] = None,
    since: Optional[datetime] = None,
    loc_samples: Optional[Dict[int, int]] = None,
    require_min_samples: bool = False,
    exclude_stale: bool = False,
    include_direct_horizon_pairs: bool = False,
    weather_lookup_cache: Optional[Dict[Tuple[datetime, str], float]] = None,
    core_feature_cache: Optional[Dict[Tuple[int, datetime], List[float]]] = None,
    dataset_cache: Optional[Dict[Tuple[object, ...], Dict[str, object]]] = None,
    cache_key: Optional[Tuple[object, ...]] = None,
) -> Dict[str, object]:
    if isinstance(dataset_cache, dict) and cache_key is not None:
        cached = dataset_cache.get(cache_key)
        if isinstance(cached, dict):
            return cached

    direct_horizon_pairs: Dict[int, List[Tuple[float, float, float]]] = {}
    direct_horizon_hours = (
        configured_direct_horizon_hours()
        if include_direct_horizon_pairs
        else []
    )
    pair_half_life_days = max(1.0, float(config.RECENCY_HALFLIFE_DAYS))
    pair_min_recency_weight = max(0.0, min(1.0, float(config.RECENCY_MIN_WEIGHT)))

    features_rows: List[List[float]] = []
    reporting_rows = []
    labels: List[float] = []
    times: List[datetime] = []
    hours: List[int] = []
    row_quality_weights: List[float] = []
    sensor_quality_weights: List[float] = []
    feature_quality_weights: List[float] = []
    transition_weights: List[float] = []
    location_balance_weights: List[float] = []

    for raw_loc_id in loc_ids:
        try:
            loc_id = int(raw_loc_id)
        except Exception:
            continue

        data = loc_data.get(loc_id)
        if not isinstance(data, dict):
            continue
        if require_min_samples and isinstance(loc_samples, dict):
            if int(loc_samples.get(loc_id, 0) or 0) < config.MIN_SAMPLES_PER_LOC:
                continue
        if exclude_stale and bool(data.get("is_stale")):
            continue

        onehot_vec = onehot.get(loc_id)
        if onehot_vec is None:
            continue

        loc_balance_weight = location_balance_weight_for_loc(location_balance_map, loc_id)
        bucket_map = data.get("bucket_map", {})
        bucket_times = data.get("bucket_times", [])
        bucket_values = data.get("bucket_values", [])
        loc_reference_ts = bucket_times[-1] if bucket_times else None

        for target, label in zip(bucket_times, bucket_values):
            if not isinstance(target, datetime):
                continue
            if isinstance(since, datetime) and target < since:
                continue

            core_cache_key = (int(loc_id), target)
            core_feature_vec = None
            if isinstance(core_feature_cache, dict):
                cached_core = core_feature_cache.get(core_cache_key)
                if isinstance(cached_core, list):
                    core_feature_vec = cached_core
            if core_feature_vec is None:
                core_feature_vec = build_features(
                    target,
                    data,
                    [],
                    weather_source=weather_source,
                    weather_lookup_cache=weather_lookup_cache,
                )
                if isinstance(core_feature_cache, dict):
                    core_feature_cache[core_cache_key] = list(core_feature_vec)
            feature_vec = list(core_feature_vec) + list(onehot_vec)
            flatline_steps_recent, missing_ratio_recent, minutes_since_raw = sensor_quality_signals(
                target=target,
                bucket_map=data["bucket_map"],
                raw_times=data["raw_times"],
                raw_values=data["raw_values"],
            )
            sensor_weight = sensor_quality_weight(
                flatline_steps=flatline_steps_recent,
                missing_ratio=missing_ratio_recent,
                minutes_since_raw=minutes_since_raw,
            )
            feature_weight = feature_quality_weight(feature_vec)
            transition_weight = schedule_transition_weight_for_location_target(data, target)
            row_quality_weight = (
                float(sensor_weight)
                * float(feature_weight)
                * float(transition_weight)
                * float(loc_balance_weight)
            )

            features_rows.append(feature_vec)
            labels.append(float(label))
            times.append(target)
            reporting_rows.append({
                "location_id": loc_id, "facility_id": data.get("reporting_facility_id"),
                "target": target, "capacity": data.get("max_cap"),
                "time_aligned": data.get("reporting_time_aligned") is True,
                "actual_people": data.get("reporting_bucket_people", {}).get(target),
                "raw_baseline": data.get("reporting_raw_baseline", ()),
            })
            hours.append(int(target.hour))
            row_quality_weights.append(float(row_quality_weight))
            sensor_quality_weights.append(float(sensor_weight))
            feature_quality_weights.append(float(feature_weight))
            transition_weights.append(float(transition_weight))
            location_balance_weights.append(float(loc_balance_weight))

            if not direct_horizon_hours:
                continue

            source_ratio = to_float_or_none(label)
            if source_ratio is None:
                continue
            if isinstance(loc_reference_ts, datetime):
                age_days = max(0.0, (loc_reference_ts - target).total_seconds() / 86400.0)
            else:
                age_days = 0.0
            pair_recency_weight = max(
                pair_min_recency_weight,
                2.0 ** (-age_days / pair_half_life_days),
            )
            for horizon_hours in direct_horizon_hours:
                future_ratio = ratio_value_from_maps(
                    bucket_map=bucket_map,
                    ts=target + timedelta(hours=int(horizon_hours)),
                )
                if future_ratio is None:
                    continue
                occupancy_boost = (
                    1.0
                    + max(0.0, float(config.OCCUPANCY_WEIGHT_ALPHA))
                    * (
                        max(0.0, min(1.2, float(future_ratio)))
                        ** max(1.0, float(config.OCCUPANCY_WEIGHT_GAMMA))
                    )
                )
                pair_weight = max(
                    float(config.SENSOR_WEIGHT_MIN),
                    float(row_quality_weight) * float(occupancy_boost) * float(pair_recency_weight),
                )
                direct_horizon_pairs.setdefault(int(horizon_hours), []).append(
                    (float(source_ratio), float(future_ratio), float(pair_weight))
                )

    if features_rows:
        X = sanitize_feature_matrix(np.array(features_rows, dtype=np.float32))
    else:
        X = np.zeros((0, 0), dtype=np.float32)

    dataset = {
        "X": X,
        "y": np.array(labels, dtype=np.float32),
        "times": list(times),
        "hours": list(hours),
        "rowQualityWeights": np.array(row_quality_weights, dtype=np.float32),
        "sensorQualityWeights": np.array(sensor_quality_weights, dtype=np.float32),
        "featureQualityWeights": np.array(feature_quality_weights, dtype=np.float32),
        "transitionWeights": np.array(transition_weights, dtype=np.float32),
        "locationBalanceWeights": np.array(location_balance_weights, dtype=np.float32),
        "directHorizonPairs": direct_horizon_pairs,
        "reportingRows": reporting_rows,
    }
    if isinstance(dataset_cache, dict) and cache_key is not None:
        dataset_cache[cache_key] = dataset
    return dataset


def stabilize_sample_weights(weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr

    out = np.array(arr, copy=True)
    out[~np.isfinite(out)] = 0.0
    out = np.maximum(out, 0.0)

    min_w = max(1e-6, float(config.MODEL_WEIGHT_CLIP_MIN))
    max_w = max(min_w, float(config.MODEL_WEIGHT_CLIP_MAX))

    positive = out[out > 0.0]
    if positive.size <= 0:
        return np.ones(out.size, dtype=np.float32)

    if config.MODEL_WEIGHT_STABILIZATION_ENABLED:
        low_q = max(0.0, min(0.49, float(config.MODEL_WEIGHT_CLIP_LOWER_Q)))
        high_q = max(low_q + 0.01, min(1.0, float(config.MODEL_WEIGHT_CLIP_UPPER_Q)))
        q_low = float(np.quantile(positive, low_q))
        q_high = float(np.quantile(positive, high_q))
        lower_bound = max(min_w, q_low)
        upper_bound = min(max_w, max(lower_bound, q_high))
        out = np.clip(out, lower_bound, upper_bound)
    else:
        out = np.clip(out, min_w, max_w)

    if config.MODEL_WEIGHT_NORMALIZE_MEAN:
        positive = out[out > 0.0]
        mean_w = float(np.mean(positive)) if positive.size > 0 else 0.0
        if math.isfinite(mean_w) and mean_w > 0.0:
            out = out / mean_w

    out = np.nan_to_num(out, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
    out = np.maximum(out, 1e-6)
    return out


def weighted_average(values: np.ndarray, weights: Optional[np.ndarray]) -> float:
    values_arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if values_arr.size == 0:
        return 0.0

    if weights is None:
        finite = values_arr[np.isfinite(values_arr)]
        if finite.size == 0:
            return 0.0
        return float(np.mean(finite))

    weights_arr = np.asarray(weights, dtype=np.float32).reshape(-1)
    if weights_arr.size != values_arr.size:
        finite = values_arr[np.isfinite(values_arr)]
        if finite.size == 0:
            return 0.0
        return float(np.mean(finite))

    mask = np.isfinite(values_arr) & np.isfinite(weights_arr) & (weights_arr > 0.0)
    if not np.any(mask):
        return 0.0

    v = values_arr[mask]
    w = weights_arr[mask]
    total_weight = float(np.sum(w))
    if total_weight <= 0.0:
        return float(np.mean(v))
    return float(np.average(v, weights=w))


def weighted_quantile(
    values: np.ndarray,
    quantile: float,
    weights: Optional[np.ndarray] = None,
) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    q = max(0.0, min(1.0, float(quantile)))

    if weights is None:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return 0.0
        return float(np.quantile(finite, q))

    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    if w.size != arr.size:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return 0.0
        return float(np.quantile(finite, q))

    mask = np.isfinite(arr) & np.isfinite(w) & (w > 0.0)
    if not np.any(mask):
        return 0.0

    v = arr[mask]
    ww = w[mask]
    order = np.argsort(v)
    v = v[order]
    ww = ww[order]
    total = float(np.sum(ww))
    if total <= 0.0:
        return float(np.quantile(v, q))

    cutoff = q * total
    cumsum = np.cumsum(ww, dtype=np.float64)
    idx = int(np.searchsorted(cumsum, cutoff, side="left"))
    idx = max(0, min(idx, v.size - 1))
    if idx <= 0:
        return float(v[0])

    prev_cum = float(cumsum[idx - 1])
    cur_cum = float(cumsum[idx])
    if not math.isfinite(prev_cum) or not math.isfinite(cur_cum) or cur_cum <= prev_cum:
        return float(v[idx])

    alpha = max(0.0, min(1.0, (float(cutoff) - prev_cum) / max(1e-9, cur_cum - prev_cum)))
    prev_val = float(v[idx - 1])
    cur_val = float(v[idx])
    return float(prev_val * (1.0 - alpha) + cur_val * alpha)


def sanitize_feature_matrix(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if arr.size == 0:
        return arr
    invalid = ~np.isfinite(arr)
    max_abs = max(0.0, float(config.MODEL_FEATURE_ABS_MAX))
    if max_abs > 0.0:
        finite = np.isfinite(arr)
        if np.any(finite):
            invalid = invalid | (finite & (np.abs(arr) > max_abs))
    if np.any(invalid):
        arr = np.array(arr, copy=True)
        arr[invalid] = np.nan
    return arr


def compute_feature_fill_values(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] <= 0:
        return np.array([], dtype=np.float32)

    medians = np.zeros(arr.shape[1], dtype=np.float32)
    for col in range(arr.shape[1]):
        col_vals = arr[:, col]
        finite = col_vals[np.isfinite(col_vals)]
        if finite.size > 0:
            medians[col] = float(np.median(finite))
    return medians


def coerce_feature_fill_values(
    values: object,
    expected_cols: Optional[int] = None,
) -> Optional[np.ndarray]:
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
    elif isinstance(values, (list, tuple)):
        parsed: List[float] = []
        for raw in values:
            numeric = to_float_or_none(raw)
            parsed.append(float(numeric) if numeric is not None else 0.0)
        arr = np.array(parsed, dtype=np.float32)
    else:
        return None

    if arr.size == 0:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if expected_cols is not None and int(expected_cols) > 0 and arr.size != int(expected_cols):
        return None
    return arr


def apply_feature_fill_values(
    X: np.ndarray,
    fill_values: Optional[np.ndarray],
) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    fills = coerce_feature_fill_values(fill_values, expected_cols=arr.shape[1])
    if fills is None:
        return arr
    invalid = ~np.isfinite(arr)
    if not np.any(invalid):
        return arr
    out = np.array(arr, copy=True)
    row_idx, col_idx = np.where(invalid)
    out[row_idx, col_idx] = fills[col_idx]
    return out


def compute_feature_clip_bounds(
    X: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not config.MODEL_FEATURE_CLIP_ENABLED:
        return None

    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] <= 0:
        return None

    lower_q = max(0.0, min(0.49, float(config.MODEL_FEATURE_CLIP_LOWER_Q)))
    upper_q = max(lower_q + 0.01, min(1.0, float(config.MODEL_FEATURE_CLIP_UPPER_Q)))
    min_spread = max(0.0, float(config.MODEL_FEATURE_CLIP_MIN_SPREAD))

    lower = np.full(arr.shape[1], -np.inf, dtype=np.float32)
    upper = np.full(arr.shape[1], np.inf, dtype=np.float32)
    has_any = False

    for col in range(arr.shape[1]):
        col_vals = arr[:, col]
        finite = col_vals[np.isfinite(col_vals)]
        if finite.size <= 0:
            continue

        lo = float(np.quantile(finite, lower_q))
        hi = float(np.quantile(finite, upper_q))
        if not math.isfinite(lo) or not math.isfinite(hi):
            continue
        if hi < lo:
            lo, hi = hi, lo

        if min_spread > 0.0 and (hi - lo) < min_spread:
            center = float(np.median(finite))
            half = 0.5 * min_spread
            lo = center - half
            hi = center + half

        lower[col] = float(lo)
        upper[col] = float(hi)
        has_any = True

    if not has_any:
        return None
    return lower, upper


def coerce_feature_clip_bounds(
    lower_values: object,
    upper_values: object,
    expected_cols: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if lower_values is None or upper_values is None:
        return None

    def parse(raw: object) -> Optional[np.ndarray]:
        if isinstance(raw, np.ndarray):
            return np.asarray(raw, dtype=np.float32).reshape(-1)
        if isinstance(raw, (list, tuple)):
            parsed: List[float] = []
            for value in raw:
                numeric = to_float_or_none(value)
                parsed.append(float(numeric) if numeric is not None else float("nan"))
            return np.array(parsed, dtype=np.float32)
        return None

    lower_arr = parse(lower_values)
    upper_arr = parse(upper_values)
    if lower_arr is None or upper_arr is None:
        return None
    if lower_arr.size <= 0 or upper_arr.size <= 0:
        return None
    if lower_arr.size != upper_arr.size:
        return None
    if expected_cols is not None and int(expected_cols) > 0 and lower_arr.size != int(expected_cols):
        return None

    lower_arr = np.asarray(lower_arr, dtype=np.float32).reshape(-1)
    upper_arr = np.asarray(upper_arr, dtype=np.float32).reshape(-1)
    lower_arr = np.where(np.isfinite(lower_arr), lower_arr, -np.inf).astype(np.float32)
    upper_arr = np.where(np.isfinite(upper_arr), upper_arr, np.inf).astype(np.float32)

    swap_mask = lower_arr > upper_arr
    if np.any(swap_mask):
        low_copy = np.array(lower_arr, copy=True)
        lower_arr[swap_mask] = upper_arr[swap_mask]
        upper_arr[swap_mask] = low_copy[swap_mask]

    bounded_mask = np.isfinite(lower_arr) | np.isfinite(upper_arr)
    if not np.any(bounded_mask):
        return None
    return lower_arr, upper_arr


def apply_feature_clip_bounds(
    X: np.ndarray,
    clip_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if not config.MODEL_FEATURE_CLIP_ENABLED:
        return arr
    if arr.ndim != 2 or arr.size == 0:
        return arr
    if not isinstance(clip_bounds, (tuple, list)) or len(clip_bounds) < 2:
        return arr

    bounds = coerce_feature_clip_bounds(
        clip_bounds[0],
        clip_bounds[1],
        expected_cols=arr.shape[1],
    )
    if bounds is None:
        return arr
    lower, upper = bounds

    out = np.array(arr, copy=True)
    for col in range(out.shape[1]):
        lo = float(lower[col])
        hi = float(upper[col])
        if not math.isfinite(lo) and not math.isfinite(hi):
            continue

        col_vals = out[:, col]
        finite_mask = np.isfinite(col_vals)
        if not np.any(finite_mask):
            continue
        vals = col_vals[finite_mask]
        if math.isfinite(lo):
            vals = np.maximum(vals, lo)
        if math.isfinite(hi):
            vals = np.minimum(vals, hi)
        col_vals[finite_mask] = vals
        out[:, col] = col_vals

    return out


def supervised_row_mask(
    X: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
    mask = np.isfinite(y_arr)

    if weights is not None:
        w_arr = np.asarray(weights, dtype=np.float32).reshape(-1)
        if w_arr.size != y_arr.size:
            return np.zeros(y_arr.size, dtype=bool)
        mask &= np.isfinite(w_arr) & (w_arr > 0.0)

    x_arr = np.asarray(X, dtype=np.float32)
    if x_arr.ndim == 2 and x_arr.shape[0] == y_arr.size:
        finite = np.isfinite(x_arr)
        min_ratio = max(0.0, min(1.0, float(config.MODEL_MIN_FEATURE_FINITE_RATIO)))
        if min_ratio <= 0.0:
            mask &= finite.any(axis=1)
        else:
            finite_ratio = np.mean(finite, axis=1)
            mask &= finite_ratio >= min_ratio
    elif x_arr.ndim != 2:
        return np.zeros(y_arr.size, dtype=bool)
    return mask


def ordered_prediction_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    low = np.asarray(lower, dtype=np.float32)
    high = np.asarray(upper, dtype=np.float32)
    return np.minimum(low, high), np.maximum(low, high)


def parse_weather_hourly_payload(
    payload: Dict[str, object],
    min_dt: Optional[datetime] = None,
    max_dt: Optional[datetime] = None,
) -> Dict[str, object]:
    hourly = payload.get("hourly") or {}
    times_raw = hourly.get("time") or []
    if not times_raw:
        return {"times": [], "map": {}}

    weather_map: Dict[datetime, Dict[str, float]] = {}
    for idx, ts_raw in enumerate(times_raw):
        try:
            parsed = datetime.fromisoformat(str(ts_raw))
        except Exception:
            continue

        if parsed.tzinfo is None:
            parsed = config.TZ.localize(parsed)
        else:
            parsed = parsed.astimezone(config.TZ)

        if min_dt is not None and parsed < min_dt:
            continue
        if max_dt is not None and parsed > max_dt:
            continue

        row: Dict[str, float] = {}
        for key, api_key in config.WEATHER_API_HOURLY_MAP.items():
            series = hourly.get(api_key) or []
            value = series[idx] if idx < len(series) else None
            numeric = to_float_or_none(value)
            if numeric is not None:
                row[key] = numeric

        if row:
            weather_map[parsed] = row

    times = sorted(weather_map.keys())
    return {"times": times, "map": weather_map}


def merge_weather_series(*series_list: Dict[str, object]) -> Dict[str, object]:
    merged_map: Dict[datetime, Dict[str, float]] = {}

    for series in series_list:
        weather_map = series.get("map", {})
        if not isinstance(weather_map, dict):
            continue
        for ts, row in weather_map.items():
            if not isinstance(ts, datetime) or not isinstance(row, dict):
                continue
            combined = merged_map.get(ts, {}).copy()
            combined.update(row)
            merged_map[ts] = combined

    times = sorted(merged_map.keys())
    return {"times": times, "map": merged_map}


_sys.modules.setdefault("server.reclive.forecasting.features", _sys.modules[__name__])
_sys.modules.setdefault("reclive.forecasting.features", _sys.modules[__name__])
