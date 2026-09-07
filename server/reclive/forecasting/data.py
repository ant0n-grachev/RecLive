"""Forecasting data owner; mechanically transferred definitions."""

import sys as _sys
from server.reclive.forecasting import config, features, metrics, reporting
import json
import math
import os
import re
import shutil
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import pymysql
import requests
import xgboost as xgb








def load_schedule_sections_by_facility() -> Dict[int, Dict[str, object]]:
    if not config.SCHEDULE_FILTER_ENABLED:
        return {}
    if not os.path.exists(config.FACILITY_HOURS_JSON_PATH):
        return {}

    try:
        with open(config.FACILITY_HOURS_JSON_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}

    facilities_raw = payload.get("facilities")
    if not isinstance(facilities_raw, list):
        return {}

    out: Dict[int, Dict[str, object]] = {}
    for row in facilities_raw:
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "")).strip().lower() != "ok":
            continue
        raw_id = row.get("facilityId")
        try:
            facility_id = int(raw_id)
        except Exception:
            continue
        sections = row.get("sections")
        if not isinstance(sections, list) or not sections:
            continue
        out[facility_id] = {"sections": sections}
    return out


def db_connect():
    host = config.require_env("GYM_DB_HOST")
    port = config.require_int_env("GYM_DB_PORT")
    user = config.require_env("GYM_DB_USER")
    password = config.require_env("GYM_DB_PASSWORD")
    database = config.require_env("GYM_DB_NAME")

    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        autocommit=True,
        charset="utf8mb4",
        connect_timeout=10,
        read_timeout=20,
        write_timeout=20,
    )


def model_artifact_paths(model_key: str) -> Tuple[str, str, str, str]:
    safe_key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(model_key)).strip("._")
    if not safe_key:
        safe_key = "default"
    stem = os.path.join(config.MODEL_ARTIFACT_DIR, f"{config.MODEL_BASENAME}_{safe_key}")
    p50_path = f"{stem}.p50.xgb.json"
    p10_path = f"{stem}.p10.xgb.json"
    p90_path = f"{stem}.p90.xgb.json"
    meta_path = f"{stem}.meta.json"
    return p50_path, p10_path, p90_path, meta_path


def model_previous_artifact_paths(model_key: str) -> Tuple[str, str, str, str]:
    p50_path, p10_path, p90_path, meta_path = model_artifact_paths(model_key)
    return (
        p50_path + ".prev",
        p10_path + ".prev",
        p90_path + ".prev",
        meta_path + ".prev",
    )


def _copy_if_exists(src: str, dst: str) -> bool:
    try:
        if not os.path.exists(src):
            return False
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def backup_current_artifacts(model_key: str) -> bool:
    p50_path, p10_path, p90_path, meta_path = model_artifact_paths(model_key)
    prev_p50, prev_p10, prev_p90, prev_meta = model_previous_artifact_paths(model_key)
    ensure_dir(config.MODEL_ARTIFACT_DIR)

    if not os.path.exists(p50_path) or not os.path.exists(meta_path):
        return False

    copied_any = False
    copied_any = _copy_if_exists(p50_path, prev_p50) or copied_any
    copied_any = _copy_if_exists(p10_path, prev_p10) or copied_any
    copied_any = _copy_if_exists(p90_path, prev_p90) or copied_any
    copied_any = _copy_if_exists(meta_path, prev_meta) or copied_any
    return copied_any


def load_saved_meta_only(model_key: str) -> Optional[Dict[str, object]]:
    _p50_path, _p10_path, _p90_path, meta_path = model_artifact_paths(model_key)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    return meta


def collect_saved_meta_snapshots() -> Dict[str, Dict[str, object]]:
    output: Dict[str, Dict[str, object]] = {}
    for model_key in config.iter_model_unit_keys():
        meta = load_saved_meta_only(model_key)
        if isinstance(meta, dict):
            output[model_key] = meta
    return output


def load_history(
    conn,
    facility_schedule_by_id: Optional[Dict[int, Dict[str, object]]] = None,
):
    raw_by_loc: Dict[int, List[Tuple[datetime, float]]] = {}
    reporting_raw_by_loc = {}
    reporting_time_aligned_by_loc = {}
    max_caps: Dict[int, int] = {}
    location_facility_map = config.location_to_facility_map()
    active_schedule_map = facility_schedule_by_id or {}
    schedule_eval_cache: Dict[Tuple[int, date, int], Optional[bool]] = {}
    schedule_boundary_cache: Dict[Tuple[int, date, int], Tuple[bool, bool]] = {}
    schedule_date_range_cache: Dict[Tuple[str, int], Optional[Tuple[date, date, int]]] = {}
    schedule_weekday_cache: Dict[str, Optional[Set[int]]] = {}
    schedule_hours_cache: Dict[str, Optional[Tuple[int, int, bool]]] = {}

    quality = {
        "rowsRead": 0,
        "rowsDroppedInvalid": 0,
        "rowsDroppedScheduleClosed": 0,
        "rowsScheduleEvaluated": 0,
        "rowsScheduleUnknown": 0,
        "rowsScheduleBoundaryZeroed": 0,
        "rowsScheduleBoundaryClosedKept": 0,
        "scheduleFilterEnabled": bool(config.SCHEDULE_FILTER_ENABLED),
        "scheduleBoundaryZeroEnabled": bool(config.SCHEDULE_BOUNDARY_ZERO_ENABLED),
        "scheduleFacilitiesLoaded": len(active_schedule_map),
        "duplicatesRemoved": 0,
        "impossibleJumpsRemoved": 0,
        "flatlineRowsPruned": 0,
        "flatlineRunsDetected": 0,
        "boundaryBucketsZeroed": 0,
        "flatlineLocations": [],
        "staleLocations": [],
    }

    sql = config.SQL_HISTORY_BASE
    params: Tuple[object, ...] = ()
    if config.HISTORY_DAYS > 0:
        sql += " AND fetched_at >= %s"
        since = datetime.now(config.DB_TZ) - timedelta(days=config.HISTORY_DAYS)
        params = (since,)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        for (
            loc_id,
            last_updated,
            fetched_at,
            current_capacity,
            _is_closed,
            max_cap,
        ) in cur.fetchall():
            reporting_original_count = current_capacity
            quality["rowsRead"] += 1
            if current_capacity is None:
                quality["rowsDroppedInvalid"] += 1
                continue

            try:
                loc_id = int(loc_id)
                current_capacity = float(current_capacity)
            except Exception:
                quality["rowsDroppedInvalid"] += 1
                continue
            if not math.isfinite(current_capacity) or current_capacity < 0:
                quality["rowsDroppedInvalid"] += 1
                continue

            local_dt = features.parse_observed_at_value(last_updated)
            if local_dt is None and fetched_at is not None:
                local_dt = features.to_local(fetched_at)
            if local_dt is None:
                quality["rowsDroppedInvalid"] += 1
                continue

            reporting_canonical_observed = metrics.reporting_utc(last_updated, trusted_db=True)
            reporting_time_aligned_by_loc[loc_id] = (
                reporting_time_aligned_by_loc.get(loc_id, True)
                and reporting_canonical_observed is not None
                and reporting_canonical_observed == metrics.reporting_utc(local_dt)
            )
            reporting_observation = metrics.raw_baseline_observation(
                last_updated, fetched_at, reporting_original_count,
            )
            if reporting_observation is not None:
                reporting_raw_by_loc.setdefault(loc_id, []).append(reporting_observation)

            if active_schedule_map:
                facility_id = location_facility_map.get(loc_id)
                if facility_id is not None and facility_id in active_schedule_map:
                    minute_of_day = int(local_dt.hour) * 60 + int(local_dt.minute)
                    schedule_cache_key = (int(facility_id), local_dt.date(), minute_of_day)
                    if schedule_cache_key not in schedule_eval_cache:
                        sections_raw = active_schedule_map.get(facility_id, {}).get("sections", [])
                        sections = sections_raw if isinstance(sections_raw, list) else []
                        schedule_eval_cache[schedule_cache_key] = features.get_facility_schedule_open_state(
                            sections=sections,
                            ts=local_dt,
                            date_range_cache=schedule_date_range_cache,
                            weekday_cache=schedule_weekday_cache,
                            hours_window_cache=schedule_hours_cache,
                        )
                    if (
                        config.SCHEDULE_BOUNDARY_ZERO_ENABLED
                        and schedule_cache_key not in schedule_boundary_cache
                    ):
                        sections_raw = active_schedule_map.get(facility_id, {}).get("sections", [])
                        sections = sections_raw if isinstance(sections_raw, list) else []
                        schedule_boundary_cache[schedule_cache_key] = features.get_facility_schedule_boundary_state(
                            sections=sections,
                            ts=local_dt,
                            date_range_cache=schedule_date_range_cache,
                            weekday_cache=schedule_weekday_cache,
                            hours_window_cache=schedule_hours_cache,
                        )

                    quality["rowsScheduleEvaluated"] += 1
                    open_state = schedule_eval_cache.get(schedule_cache_key)
                    boundary_open = False
                    boundary_close = False
                    if config.SCHEDULE_BOUNDARY_ZERO_ENABLED:
                        boundary_open, boundary_close = schedule_boundary_cache.get(
                            schedule_cache_key,
                            (False, False),
                        )
                        if boundary_open or boundary_close:
                            current_capacity = 0.0
                            quality["rowsScheduleBoundaryZeroed"] += 1
                    if open_state is None:
                        quality["rowsScheduleUnknown"] += 1
                    elif not open_state and not (config.SCHEDULE_BOUNDARY_ZERO_ENABLED and boundary_close):
                        quality["rowsDroppedScheduleClosed"] += 1
                        continue
                    elif not open_state and config.SCHEDULE_BOUNDARY_ZERO_ENABLED and boundary_close:
                        quality["rowsScheduleBoundaryClosedKept"] += 1

            raw_by_loc.setdefault(loc_id, []).append((local_dt, current_capacity))

            if max_cap is not None:
                try:
                    cap = int(max_cap)
                    if cap > 0:
                        max_caps[loc_id] = max(cap, max_caps.get(loc_id, 0))
                except Exception:
                    pass

    avg_dow_hour_sum: Dict[Tuple[int, int, int], List[float]] = {}
    avg_hour_sum: Dict[Tuple[int, int], List[float]] = {}
    avg_overall_sum: Dict[int, List[float]] = {}

    loc_data: Dict[int, Dict[str, object]] = {}
    loc_samples: Dict[int, int] = {}
    now_local = datetime.now(config.TZ)

    for loc_id, entries in raw_by_loc.items():
        max_cap = max_caps.get(loc_id, 0)
        if max_cap <= 0:
            continue
        loc_facility_id = location_facility_map.get(loc_id)

        deduped, dup_removed = features.dedupe_exact_timestamps(entries)
        quality["duplicatesRemoved"] += dup_removed

        cleaned, jump_removed = features.drop_impossible_jumps(deduped, max_cap=max_cap)
        quality["impossibleJumpsRemoved"] += jump_removed

        cleaned, flatline_removed, flatline_runs = features.drop_flatline_plateaus(cleaned, max_cap=max_cap)
        quality["flatlineRowsPruned"] += flatline_removed
        quality["flatlineRunsDetected"] += flatline_runs
        if flatline_runs > 0:
            quality["flatlineLocations"].append(loc_id)

        if not cleaned:
            continue

        latest_ts = cleaned[-1][0]
        age_hours = (now_local - latest_ts).total_seconds() / 3600.0
        is_stale = age_hours > config.STALE_SENSOR_HOURS
        if is_stale:
            quality["staleLocations"].append(loc_id)

        raw_times = [row[0] for row in cleaned]
        raw_values = [float(row[1]) for row in cleaned]

        bucket_counts: Dict[datetime, List[float]] = {}
        for ts, count in cleaned:
            if not math.isfinite(float(count)):
                continue
            bucket = features.floor_time(ts, config.RESAMPLE_MINUTES)
            bucket_counts.setdefault(bucket, []).append(float(count))

        if not bucket_counts:
            continue

        bucket_times = sorted(bucket_counts.keys())
        bucket_values: List[float] = []
        bucket_map: Dict[datetime, float] = {}
        bucket_people_map = {}
        loc_avg_dow_hour_sum: Dict[Tuple[int, int], List[float]] = {}
        loc_avg_hour_sum: Dict[int, List[float]] = {}
        loc_avg_overall_sum = [0.0, 0.0]

        for bt in bucket_times:
            avg_count = float(sum(bucket_counts[bt]) / len(bucket_counts[bt]))
            if not math.isfinite(avg_count):
                continue
            bucket_people_map[bt] = avg_count
            ratio = avg_count / max_cap
            ratio = max(0.0, min(ratio, 1.2))
            if (
                config.SCHEDULE_BOUNDARY_ZERO_ENABLED
                and loc_facility_id is not None
                and loc_facility_id in active_schedule_map
            ):
                minute_of_day = int(bt.hour) * 60 + int(bt.minute)
                schedule_cache_key = (int(loc_facility_id), bt.date(), minute_of_day)
                if schedule_cache_key not in schedule_boundary_cache:
                    sections_raw = active_schedule_map.get(loc_facility_id, {}).get("sections", [])
                    sections = sections_raw if isinstance(sections_raw, list) else []
                    schedule_boundary_cache[schedule_cache_key] = features.get_facility_schedule_boundary_state(
                        sections=sections,
                        ts=bt,
                        date_range_cache=schedule_date_range_cache,
                        weekday_cache=schedule_weekday_cache,
                        hours_window_cache=schedule_hours_cache,
                    )
                boundary_open, boundary_close = schedule_boundary_cache.get(
                    schedule_cache_key,
                    (False, False),
                )
                if boundary_open or boundary_close:
                    ratio = 0.0
                    quality["boundaryBucketsZeroed"] += 1
            bucket_values.append(ratio)
            bucket_map[bt] = ratio

            features.aggregate_sum_count(avg_dow_hour_sum, (loc_id, bt.weekday(), bt.hour), ratio)
            features.aggregate_sum_count(avg_hour_sum, (loc_id, bt.hour), ratio)
            features.aggregate_sum_count(avg_overall_sum, loc_id, ratio)
            features.aggregate_sum_count(loc_avg_dow_hour_sum, (bt.weekday(), bt.hour), ratio)
            features.aggregate_sum_count(loc_avg_hour_sum, bt.hour, ratio)
            loc_avg_overall_sum[0] += ratio
            loc_avg_overall_sum[1] += 1.0

        if not bucket_values:
            continue

        schedule_sections: List[Dict[str, object]] = []
        if loc_facility_id is not None and loc_facility_id in active_schedule_map:
            sections_raw = active_schedule_map.get(loc_facility_id, {}).get("sections", [])
            if isinstance(sections_raw, list):
                schedule_sections = sections_raw

        loc_data[loc_id] = {
            "raw_times": raw_times,
            "raw_values": raw_values,
            "bucket_times": bucket_times,
            "bucket_values": bucket_values,
            "bucket_map": bucket_map,
            "reporting_bucket_people": bucket_people_map,
            "reporting_facility_id": loc_facility_id,
            "reporting_time_aligned": reporting_time_aligned_by_loc.get(loc_id, False),
            "reporting_raw_baseline": tuple(reporting_raw_by_loc.get(loc_id, ())),
            "max_cap": max_cap,
            "is_stale": is_stale,
            "latest_ts": latest_ts.isoformat(),
            # Per-location priors used to impute missing lag features for future horizons.
            "fallback_avg_dow_hour": features.finalize_averages(loc_avg_dow_hour_sum),
            "fallback_avg_hour": features.finalize_averages(loc_avg_hour_sum),
            "fallback_avg_overall": (
                (
                    loc_avg_overall_sum[0] / loc_avg_overall_sum[1],
                    int(loc_avg_overall_sum[1]),
                )
                if loc_avg_overall_sum[1] > 0
                else None
            ),
            "schedule_sections": schedule_sections,
            "schedule_phase_cache": {},
            "schedule_date_range_cache": {},
            "schedule_weekday_cache": {},
            "schedule_hours_cache": {},
        }
        loc_samples[loc_id] = len(bucket_times)

    quality["locationsWithHistory"] = len(loc_data)
    quality["staleLocations"] = sorted(set(quality["staleLocations"]))
    quality["staleLocationsCount"] = len(quality["staleLocations"])
    quality["flatlineLocations"] = sorted(set(quality["flatlineLocations"]))
    quality["flatlineLocationsCount"] = len(quality["flatlineLocations"])

    avg_dow_hour = features.finalize_averages(avg_dow_hour_sum)
    avg_hour = features.finalize_averages(avg_hour_sum)
    avg_overall = features.finalize_averages(avg_overall_sum)

    return loc_data, avg_dow_hour, avg_hour, avg_overall, max_caps, loc_samples, quality


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_saved_model(
    model_key: str,
    expected_loc_ids: List[int],
    expected_feature_count: int,
):
    p50_path, p10_path, p90_path, meta_path = model_artifact_paths(model_key)
    if not os.path.exists(p50_path) or not os.path.exists(meta_path):
        return None, None

    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except Exception:
        return None, None

    if meta.get("schemaVersion") != config.MODEL_SCHEMA_VERSION:
        return None, None
    if meta.get("locIds") != expected_loc_ids:
        return None, None
    if int(meta.get("featureCount", -1)) != expected_feature_count:
        return None, None
    if str(meta.get("modelKey", "")) != str(model_key):
        return None, None

    p50 = xgb.Booster()
    try:
        p50.load_model(p50_path)
    except Exception:
        return None, None

    p10 = None
    p90 = None
    if os.path.exists(p10_path):
        try:
            p10 = xgb.Booster()
            p10.load_model(p10_path)
        except Exception:
            p10 = None
    if os.path.exists(p90_path):
        try:
            p90 = xgb.Booster()
            p90.load_model(p90_path)
        except Exception:
            p90 = None

    bundle = {
        "p50": p50,
        "p10": p10,
        "p90": p90,
        "quantileDirect": bool(p10 is not None and p90 is not None),
    }
    return bundle, meta


def load_saved_previous_model(
    model_key: str,
    expected_loc_ids: List[int],
    expected_feature_count: int,
):
    p50_path, p10_path, p90_path, meta_path = model_previous_artifact_paths(model_key)
    if not os.path.exists(p50_path) or not os.path.exists(meta_path):
        return None, None

    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except Exception:
        return None, None

    if meta.get("schemaVersion") != config.MODEL_SCHEMA_VERSION:
        return None, None
    if meta.get("locIds") != expected_loc_ids:
        return None, None
    if int(meta.get("featureCount", -1)) != expected_feature_count:
        return None, None
    if str(meta.get("modelKey", "")) != str(model_key):
        return None, None

    p50 = xgb.Booster()
    try:
        p50.load_model(p50_path)
    except Exception:
        return None, None

    p10 = None
    p90 = None
    if os.path.exists(p10_path):
        try:
            p10 = xgb.Booster()
            p10.load_model(p10_path)
        except Exception:
            p10 = None
    if os.path.exists(p90_path):
        try:
            p90 = xgb.Booster()
            p90.load_model(p90_path)
        except Exception:
            p90 = None

    bundle = {
        "p50": p50,
        "p10": p10,
        "p90": p90,
        "quantileDirect": bool(p10 is not None and p90 is not None),
    }
    return bundle, meta


def rollback_to_previous_model(
    model_key: str,
    expected_loc_ids: List[int],
    expected_feature_count: int,
    now: datetime,
) -> Optional[Dict[str, object]]:
    previous_bundle, previous_meta = load_saved_previous_model(
        model_key=model_key,
        expected_loc_ids=expected_loc_ids,
        expected_feature_count=expected_feature_count,
    )
    if previous_bundle is None or previous_meta is None:
        return None

    restored_meta = dict(previous_meta)
    restored_meta["rolledBackAt"] = now.isoformat()
    restored_meta["rolledBackFromDrift"] = True
    restored_meta["driftAlertStreak"] = 0
    restored_meta["forceRetrain"] = True
    restored_meta["forceRetrainUntil"] = (now + timedelta(hours=max(1, config.DRIFT_ACTION_FORCE_HOURS))).isoformat()
    save_model_artifacts(previous_bundle, restored_meta, model_key=model_key)
    return restored_meta


def _safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def save_model_artifacts(
    model_bundle: Dict[str, object],
    meta: Dict[str, object],
    model_key: str,
) -> None:
    p50_path, p10_path, p90_path, meta_path = model_artifact_paths(model_key)
    ensure_dir(config.MODEL_ARTIFACT_DIR)
    tmp_p50 = p50_path + ".tmp"
    tmp_meta = meta_path + ".tmp"

    p50 = model_bundle["p50"]
    p50.save_model(tmp_p50)
    with open(tmp_meta, "w", encoding="utf-8") as handle:
        json.dump(reporting.sanitize_for_json(meta), handle, ensure_ascii=False, allow_nan=False)

    os.replace(tmp_p50, p50_path)
    os.replace(tmp_meta, meta_path)

    p10 = model_bundle.get("p10")
    p90 = model_bundle.get("p90")

    if p10 is not None:
        tmp_p10 = p10_path + ".tmp"
        p10.save_model(tmp_p10)
        os.replace(tmp_p10, p10_path)
    else:
        _safe_remove(p10_path)

    if p90 is not None:
        tmp_p90 = p90_path + ".tmp"
        p90.save_model(tmp_p90)
        os.replace(tmp_p90, p90_path)
    else:
        _safe_remove(p90_path)


def save_model_meta_only(meta: Dict[str, object], model_key: str) -> None:
    _p50_path, _p10_path, _p90_path, meta_path = model_artifact_paths(model_key)
    ensure_dir(config.MODEL_ARTIFACT_DIR)
    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as handle:
        json.dump(reporting.sanitize_for_json(meta), handle, ensure_ascii=False, allow_nan=False)
    os.replace(tmp_meta, meta_path)


def weather_history_start(loc_data: Dict[int, Dict[str, object]], now: datetime) -> datetime:
    floor_start = now - timedelta(days=max(1, config.WEATHER_HISTORY_MAX_DAYS))
    earliest = None
    for data in loc_data.values():
        bucket_times = data.get("bucket_times", [])
        if not bucket_times:
            continue
        first = bucket_times[0]
        if earliest is None or first < earliest:
            earliest = first

    if earliest is None:
        return floor_start

    return max(earliest - timedelta(hours=3), floor_start)


def fetch_weather_history_series(start_dt: datetime, end_dt: datetime) -> Dict[str, object]:
    if end_dt < start_dt:
        return {"times": [], "map": {}}

    params = {
        "latitude": config.WEATHER_LAT,
        "longitude": config.WEATHER_LON,
        "hourly": ",".join(config.WEATHER_API_HOURLY_MAP.values()),
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "timezone": config.TZ_NAME,
        "start_date": start_dt.date().isoformat(),
        "end_date": end_dt.date().isoformat(),
    }

    try:
        resp = requests.get(config.WEATHER_ARCHIVE_URL, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        return features.parse_weather_hourly_payload(
            payload,
            min_dt=start_dt,
            max_dt=end_dt + timedelta(hours=3),
        )
    except Exception:
        return {"times": [], "map": {}}


def fetch_weather_forecast_series(now: datetime) -> Dict[str, object]:
    params = {
        "latitude": config.WEATHER_LAT,
        "longitude": config.WEATHER_LON,
        "hourly": ",".join(config.WEATHER_API_HOURLY_MAP.values()),
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "timezone": config.TZ_NAME,
        "forecast_days": max(1, config.WEATHER_FORECAST_DAYS),
        "past_days": 2,
    }

    try:
        resp = requests.get(config.WEATHER_URL, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        return features.parse_weather_hourly_payload(payload, min_dt=now - timedelta(hours=12))
    except Exception:
        return {"times": [], "map": {}}


_sys.modules.setdefault("server.reclive.forecasting.data", _sys.modules[__name__])
_sys.modules.setdefault("reclive.forecasting.data", _sys.modules[__name__])
