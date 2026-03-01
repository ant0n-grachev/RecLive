import bisect
import json
import math
import os
import random
import re
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pymysql
import pytz
import requests
import xgboost as xgb

TZ_NAME = "America/Chicago"
TZ = pytz.timezone(TZ_NAME)

RESAMPLE_MINUTES = 15
WINDOW_RESAMPLE_MINUTES = 30
WINDOW_MERGE_GAP_MIN = 90
MIN_SAMPLES_PER_LOC = int(os.getenv("MIN_SAMPLES_PER_LOC", "80"))
MIN_SAMPLES_PER_HOUR_TOTAL = int(os.getenv("MIN_SAMPLES_PER_HOUR_TOTAL", "5"))
MIN_TRAIN_SAMPLES = int(os.getenv("MIN_TRAIN_SAMPLES", "300"))
TRAIN_SPLIT = float(os.getenv("MODEL_TRAIN_SPLIT", "0.8"))
EARLY_STOPPING_ROUNDS = int(os.getenv("MODEL_EARLY_STOPPING", "50"))
MODEL_MAX_DEPTH = int(os.getenv("MODEL_MAX_DEPTH", "4"))
MODEL_NUM_BOOST_ROUND = int(os.getenv("MODEL_NUM_BOOST_ROUND", "500"))
MODEL_NTHREAD = int(os.getenv("MODEL_NTHREAD", "2"))
MODEL_ETA = float(os.getenv("MODEL_ETA", "0.05"))
MODEL_SUBSAMPLE = float(os.getenv("MODEL_SUBSAMPLE", "0.9"))
MODEL_COLSAMPLE_BYTREE = float(os.getenv("MODEL_COLSAMPLE_BYTREE", "0.9"))
MODEL_MIN_CHILD_WEIGHT = float(os.getenv("MODEL_MIN_CHILD_WEIGHT", "1.0"))
MODEL_TREE_METHOD = os.getenv("MODEL_TREE_METHOD", "hist").strip() or "hist"
MODEL_MAX_BIN = int(os.getenv("MODEL_MAX_BIN", "256"))
MODEL_RETRAIN_HOURS = int(os.getenv("MODEL_RETRAIN_HOURS", "24"))
MODEL_GUARDRAIL_MAX_MAE_DEGRADE = float(os.getenv("MODEL_GUARDRAIL_MAX_MAE_DEGRADE", "0.05"))
MODEL_GUARDRAIL_MIN_VAL_ROWS = int(os.getenv("MODEL_GUARDRAIL_MIN_VAL_ROWS", "200"))
MODEL_TUNING_ENABLED = os.getenv("MODEL_TUNING_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_TUNING_MIN_ROWS = int(os.getenv("MODEL_TUNING_MIN_ROWS", "1200"))
MODEL_TUNING_CV_FOLDS = int(os.getenv("MODEL_TUNING_CV_FOLDS", "3"))
MODEL_TUNING_MAX_CANDIDATES = int(os.getenv("MODEL_TUNING_MAX_CANDIDATES", "16"))
MODEL_TUNING_BOOST_ROUND = int(os.getenv("MODEL_TUNING_BOOST_ROUND", "250"))
MODEL_TUNING_RANDOM_SEED = int(os.getenv("MODEL_TUNING_RANDOM_SEED", "7"))
MODEL_PARALLEL_WORKERS = int(os.getenv("MODEL_PARALLEL_WORKERS", "2"))
CHAMPION_GATE_ENABLED = os.getenv("MODEL_CHAMPION_GATE_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
CHAMPION_GATE_RECENT_DAYS = int(os.getenv("MODEL_CHAMPION_GATE_RECENT_DAYS", "14"))
CHAMPION_GATE_MIN_ROWS = int(os.getenv("MODEL_CHAMPION_GATE_MIN_ROWS", "240"))
CHAMPION_GATE_MIN_MAE_IMPROVEMENT = float(
    os.getenv("MODEL_CHAMPION_GATE_MIN_MAE_IMPROVEMENT", "0.0015")
)
CHAMPION_GATE_MAX_RMSE_DEGRADE = float(os.getenv("MODEL_CHAMPION_GATE_MAX_RMSE_DEGRADE", "0.01"))
CHAMPION_GATE_MAX_INTERVAL_ERR_DEGRADE = float(
    os.getenv("MODEL_CHAMPION_GATE_MAX_INTERVAL_ERR_DEGRADE", "0.01")
)
CHAMPION_ROLLBACK_ENABLED = os.getenv("MODEL_CHAMPION_ROLLBACK_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
CHAMPION_ROLLBACK_DRIFT_STREAK = int(os.getenv("MODEL_CHAMPION_ROLLBACK_DRIFT_STREAK", "3"))
DIRECT_QUANTILE_ENABLED = os.getenv("MODEL_DIRECT_QUANTILE_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
OCCUPANCY_WEIGHT_ENABLED = os.getenv("MODEL_OCCUPANCY_WEIGHT_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
OCCUPANCY_WEIGHT_ALPHA = float(os.getenv("MODEL_OCCUPANCY_WEIGHT_ALPHA", "1.2"))
OCCUPANCY_WEIGHT_GAMMA = float(os.getenv("MODEL_OCCUPANCY_WEIGHT_GAMMA", "1.4"))
RECENCY_WEIGHT_ENABLED = os.getenv("MODEL_RECENCY_WEIGHT_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
RECENCY_HALFLIFE_DAYS = float(os.getenv("MODEL_RECENCY_HALFLIFE_DAYS", "45"))
RECENCY_MIN_WEIGHT = float(os.getenv("MODEL_RECENCY_MIN_WEIGHT", "0.2"))
DRIFT_ENABLED = os.getenv("MODEL_DRIFT_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
DRIFT_RECENT_DAYS = int(os.getenv("MODEL_DRIFT_RECENT_DAYS", "14"))
DRIFT_MIN_POINTS = int(os.getenv("MODEL_DRIFT_MIN_POINTS", "120"))
DRIFT_ALERT_MULTIPLIER = float(os.getenv("MODEL_DRIFT_ALERT_MULTIPLIER", "0.2"))
DRIFT_ACTIONS_ENABLED = os.getenv("MODEL_DRIFT_ACTIONS_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
DRIFT_ACTION_STREAK_FOR_RETRAIN = int(os.getenv("MODEL_DRIFT_ACTION_STREAK_FOR_RETRAIN", "2"))
DRIFT_ACTION_FORCE_HOURS = int(os.getenv("MODEL_DRIFT_ACTION_FORCE_HOURS", "24"))
DRIFT_ACTION_INTERVAL_MULTIPLIER_STEP = float(
    os.getenv("MODEL_DRIFT_ACTION_INTERVAL_MULTIPLIER_STEP", "0.15")
)
DRIFT_ACTION_INTERVAL_MAX_MULTIPLIER = float(
    os.getenv("MODEL_DRIFT_ACTION_INTERVAL_MAX_MULTIPLIER", "1.7")
)
ADAPTIVE_CONTROLS_ENABLED = os.getenv("MODEL_ADAPTIVE_CONTROLS_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
ADAPTIVE_HISTORY_MAX_POINTS = int(os.getenv("MODEL_ADAPTIVE_HISTORY_MAX_POINTS", "60"))
ADAPTIVE_RETRAIN_MIN_HOURS = int(os.getenv("MODEL_ADAPTIVE_RETRAIN_MIN_HOURS", "8"))
ADAPTIVE_RETRAIN_MAX_HOURS = int(os.getenv("MODEL_ADAPTIVE_RETRAIN_MAX_HOURS", "72"))
ADAPTIVE_DRIFT_DAYS_MIN = int(os.getenv("MODEL_ADAPTIVE_DRIFT_DAYS_MIN", "7"))
ADAPTIVE_DRIFT_DAYS_MAX = int(os.getenv("MODEL_ADAPTIVE_DRIFT_DAYS_MAX", "21"))
ADAPTIVE_DRIFT_MULT_MIN = float(os.getenv("MODEL_ADAPTIVE_DRIFT_MULT_MIN", "0.12"))
ADAPTIVE_DRIFT_MULT_MAX = float(os.getenv("MODEL_ADAPTIVE_DRIFT_MULT_MAX", "0.35"))
ADAPTIVE_ALERT_RATE_STABLE_MAX = float(os.getenv("MODEL_ADAPTIVE_ALERT_RATE_STABLE_MAX", "0.15"))
ADAPTIVE_ALERT_RATE_UNSTABLE_MIN = float(os.getenv("MODEL_ADAPTIVE_ALERT_RATE_UNSTABLE_MIN", "0.40"))
FORCE_RETRAIN = os.getenv("MODEL_FORCE_RETRAIN", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_SCHEMA_VERSION = 9

INTERVAL_MIN_SAMPLES_PER_HOUR = int(os.getenv("INTERVAL_MIN_SAMPLES_PER_HOUR", "30"))
INTERVAL_Q_LOW = float(os.getenv("INTERVAL_Q_LOW", "0.10"))
INTERVAL_Q_HIGH = float(os.getenv("INTERVAL_Q_HIGH", "0.90"))
INTERVAL_CONFORMAL_ENABLED = os.getenv("INTERVAL_CONFORMAL_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
INTERVAL_CONFORMAL_RECENT_DAYS = int(os.getenv("INTERVAL_CONFORMAL_RECENT_DAYS", "21"))
INTERVAL_CONFORMAL_ALPHA = float(os.getenv("INTERVAL_CONFORMAL_ALPHA", "0.20"))
INTERVAL_CONFORMAL_MIN_POINTS = int(os.getenv("INTERVAL_CONFORMAL_MIN_POINTS", "120"))
INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR = int(
    os.getenv("INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR", "20")
)
INTERVAL_CONFORMAL_MAX_MARGIN = float(os.getenv("INTERVAL_CONFORMAL_MAX_MARGIN", "0.35"))
FEATURE_ABLATION_ENABLED = os.getenv("MODEL_FEATURE_ABLATION_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
FEATURE_ABLATION_MIN_VAL_ROWS = int(os.getenv("MODEL_FEATURE_ABLATION_MIN_VAL_ROWS", "120"))
DATA_QUALITY_ALERTS_ENABLED = os.getenv("DATA_QUALITY_ALERTS_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
DATA_QUALITY_MAX_INVALID_ROW_RATE = float(os.getenv("DATA_QUALITY_MAX_INVALID_ROW_RATE", "0.35"))
DATA_QUALITY_MAX_STALE_LOC_RATE = float(os.getenv("DATA_QUALITY_MAX_STALE_LOC_RATE", "0.6"))
DATA_QUALITY_MAX_FLATLINE_LOC_RATE = float(os.getenv("DATA_QUALITY_MAX_FLATLINE_LOC_RATE", "0.4"))
DATA_QUALITY_MIN_LOCATIONS_MODELED = int(os.getenv("DATA_QUALITY_MIN_LOCATIONS_MODELED", "6"))

HISTORY_DAYS = int(os.getenv("GYM_MODEL_HISTORY_DAYS", "0"))
DB_TIMEZONE_NAME = os.getenv("GYM_DB_TIMEZONE", TZ_NAME)
DB_TZ = pytz.timezone(DB_TIMEZONE_NAME)

FORECAST_DAY_START_HOUR = int(os.getenv("FORECAST_DAY_START_HOUR", "6"))
FORECAST_DAY_END_HOUR = int(os.getenv("FORECAST_DAY_END_HOUR", "23"))

PEAK_WINDOW_HOURS = int(os.getenv("PEAK_WINDOW_HOURS", "2"))
PEAK_Z_THRESHOLD = float(os.getenv("PEAK_Z_THRESHOLD", "0.4"))
LOW_Z_THRESHOLD = float(os.getenv("LOW_Z_THRESHOLD", "0.4"))
SMOOTH_WINDOW = int(os.getenv("SMOOTH_WINDOW", "1"))
CROWD_BASELINE_MIN_COVERAGE = float(os.getenv("CROWD_BASELINE_MIN_COVERAGE", "0.6"))
CROWD_BASELINE_MIN_POINTS = int(os.getenv("CROWD_BASELINE_MIN_POINTS", "120"))
CROWD_BASELINE_LOW_QUANTILE = float(os.getenv("CROWD_BASELINE_LOW_QUANTILE", "0.3"))
CROWD_BASELINE_PEAK_QUANTILE = float(os.getenv("CROWD_BASELINE_PEAK_QUANTILE", "0.7"))

STALE_SENSOR_HOURS = float(os.getenv("STALE_SENSOR_HOURS", "24"))
IMPOSSIBLE_JUMP_PCT = float(os.getenv("IMPOSSIBLE_JUMP_PCT", "0.60"))
IMPOSSIBLE_JUMP_MAX_GAP_MIN = float(os.getenv("IMPOSSIBLE_JUMP_MAX_GAP_MIN", "120"))
SENSOR_FLATLINE_MAX_GAP_MIN = float(os.getenv("SENSOR_FLATLINE_MAX_GAP_MIN", "20"))
SENSOR_FLATLINE_MIN_DURATION_MIN = float(os.getenv("SENSOR_FLATLINE_MIN_DURATION_MIN", "360"))
SENSOR_FLATLINE_KEEP_INTERVAL_MIN = float(os.getenv("SENSOR_FLATLINE_KEEP_INTERVAL_MIN", "60"))
SENSOR_FLATLINE_TOLERANCE_PCT = float(os.getenv("SENSOR_FLATLINE_TOLERANCE_PCT", "0.01"))
SENSOR_WEIGHT_MIN = float(os.getenv("SENSOR_WEIGHT_MIN", "0.25"))

SPIKE_AWARE_ENABLED = os.getenv("SPIKE_AWARE_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
SPIKE_AWARE_MAX_AGE_MIN = float(os.getenv("SPIKE_AWARE_MAX_AGE_MIN", "90"))
SPIKE_AWARE_HORIZON_HOURS = int(os.getenv("SPIKE_AWARE_HORIZON_HOURS", "6"))
SPIKE_AWARE_DECAY = float(os.getenv("SPIKE_AWARE_DECAY", "0.55"))
SPIKE_AWARE_MAX_CAP_MULTIPLIER = float(os.getenv("SPIKE_AWARE_MAX_CAP_MULTIPLIER", "1.35"))
WEATHER_URL = os.getenv("GYM_WEATHER_URL", "https://api.open-meteo.com/v1/forecast")
WEATHER_ARCHIVE_URL = os.getenv(
    "GYM_WEATHER_ARCHIVE_URL",
    "https://archive-api.open-meteo.com/v1/archive",
)
WEATHER_LAT = float(os.getenv("GYM_WEATHER_LAT", "43.0731"))
WEATHER_LON = float(os.getenv("GYM_WEATHER_LON", "-89.4012"))
WEATHER_FORECAST_DAYS = int(os.getenv("GYM_WEATHER_FORECAST_DAYS", "7"))
WEATHER_HISTORY_MAX_DAYS = int(os.getenv("GYM_WEATHER_HISTORY_MAX_DAYS", "180"))

FORECAST_JSON_PATH = os.getenv(
    "FORECAST_JSON_PATH",
    os.path.join(os.path.dirname(__file__), "forecast.json"),
)

MODEL_ARTIFACT_DIR = os.getenv(
    "MODEL_ARTIFACT_DIR",
    os.path.join(os.path.dirname(__file__), "model_artifacts"),
)
MODEL_BASENAME = os.getenv("MODEL_BASENAME", "forecast_model")

WEATHER_KEYS = (
    "temp_c",
    "feels_like_c",
    "precip_mm",
    "rain_mm",
    "snow_cm",
    "wind_mps",
    "wind_gust_mps",
    "humidity_pct",
    "weather_code",
)
WEATHER_ROLLING_KEYS = (
    "temp_c",
    "precip_mm",
    "wind_mps",
)
WEATHER_API_HOURLY_MAP = {
    "temp_c": "temperature_2m",
    "feels_like_c": "apparent_temperature",
    "precip_mm": "precipitation",
    "rain_mm": "rain",
    "snow_cm": "snowfall",
    "wind_mps": "wind_speed_10m",
    "wind_gust_mps": "wind_gusts_10m",
    "humidity_pct": "relative_humidity_2m",
    "weather_code": "weather_code",
}

FACILITIES = {
    1186: {
        "name": "Nicholas Recreation Center",
        "categories": [
            {
                "key": "fitness_floors",
                "title": "Fitness Floors",
                "location_ids": [5761, 5760, 5762, 5758],
            },
            {
                "key": "basketball_courts",
                "title": "Basketball Courts",
                "location_ids": [7089, 7090, 5766],
            },
            {
                "key": "running_track",
                "title": "Running Track",
                "location_ids": [5763],
            },
            {
                "key": "swimming_pool",
                "title": "Swimming Pool",
                "location_ids": [5764],
            },
            {
                "key": "racquetball_courts",
                "title": "Racquetball Courts",
                "location_ids": [5753, 5754],
            },
        ],
    },
    1656: {
        "name": "Bakke Recreation & Wellbeing Center",
        "categories": [
            {
                "key": "fitness_floors",
                "title": "Fitness Floors",
                "location_ids": [8718, 8717, 8705, 8700, 8699, 8696],
            },
            {
                "key": "basketball_courts",
                "title": "Basketball Courts",
                "location_ids": [8720, 8714, 8698],
            },
            {
                "key": "running_track",
                "title": "Running Track",
                "location_ids": [8694],
            },
            {
                "key": "swimming_pool",
                "title": "Swimming Pool",
                "location_ids": [8716],
            },
            {
                "key": "rock_climbing",
                "title": "Rock Climbing",
                "location_ids": [8701],
            },
            {
                "key": "ice_skating",
                "title": "Ice Skating",
                "location_ids": [10550],
            },
            {
                "key": "esports_room",
                "title": "Esports Room",
                "location_ids": [8712],
            },
            {
                "key": "sports_simulators",
                "title": "Sports Simulators",
                "location_ids": [8695],
            },
        ],
    },
}

FORECAST_CATEGORY_KEYS = {"fitness_floors", "basketball_courts"}


def should_train_category(category_key: str) -> bool:
    return str(category_key) in FORECAST_CATEGORY_KEYS


def _d(year: int, month: int, day: int) -> date:
    return date(year, month, day)


ACADEMIC_FALL_INSTRUCTION = [
    (_d(2025, 9, 3), _d(2025, 12, 10)),
    (_d(2026, 9, 2), _d(2026, 12, 9)),
    (_d(2027, 9, 8), _d(2027, 12, 15)),
    (_d(2028, 9, 6), _d(2028, 12, 13)),
    (_d(2029, 9, 5), _d(2029, 12, 12)),
]
ACADEMIC_SPRING_INSTRUCTION = [
    (_d(2026, 1, 20), _d(2026, 5, 1)),
    (_d(2027, 1, 19), _d(2027, 4, 30)),
    (_d(2028, 1, 25), _d(2028, 5, 5)),
    (_d(2029, 1, 23), _d(2029, 5, 4)),
    (_d(2030, 1, 22), _d(2030, 5, 3)),
]
ACADEMIC_EXAMS = [
    (_d(2025, 12, 12), _d(2025, 12, 18)),
    (_d(2026, 5, 3), _d(2026, 5, 8)),
    (_d(2026, 12, 11), _d(2026, 12, 17)),
    (_d(2027, 5, 2), _d(2027, 5, 7)),
    (_d(2027, 12, 17), _d(2027, 12, 23)),
    (_d(2028, 5, 7), _d(2028, 5, 12)),
    (_d(2028, 12, 15), _d(2028, 12, 21)),
    (_d(2029, 5, 6), _d(2029, 5, 11)),
    (_d(2029, 12, 14), _d(2029, 12, 20)),
    (_d(2030, 5, 5), _d(2030, 5, 10)),
]
ACADEMIC_STUDY_DAYS = {
    _d(2025, 12, 11),
    _d(2026, 5, 2),
    _d(2026, 12, 10),
    _d(2027, 5, 1),
    _d(2027, 12, 16),
    _d(2028, 5, 6),
    _d(2028, 12, 14),
    _d(2029, 5, 5),
    _d(2029, 12, 13),
    _d(2030, 5, 4),
}
ACADEMIC_THANKSGIVING_RECESS = [
    (_d(2025, 11, 27), _d(2025, 11, 30)),
    (_d(2026, 11, 26), _d(2026, 11, 29)),
    (_d(2027, 11, 25), _d(2027, 11, 28)),
    (_d(2028, 11, 23), _d(2028, 11, 26)),
    (_d(2029, 11, 22), _d(2029, 11, 25)),
]
ACADEMIC_SPRING_RECESS = [
    (_d(2026, 3, 28), _d(2026, 4, 5)),
    (_d(2027, 3, 20), _d(2027, 3, 28)),
    (_d(2028, 3, 25), _d(2028, 4, 2)),
    (_d(2029, 3, 24), _d(2029, 4, 1)),
    (_d(2030, 3, 23), _d(2030, 3, 31)),
]
ACADEMIC_SUMMER_SESSION = [
    (_d(2026, 5, 18), _d(2026, 8, 9)),
    (_d(2027, 5, 17), _d(2027, 8, 8)),
    (_d(2028, 5, 22), _d(2028, 8, 13)),
    (_d(2029, 5, 21), _d(2029, 8, 12)),
    (_d(2030, 5, 20), _d(2030, 8, 11)),
]
ACADEMIC_HOLIDAYS = {
    _d(2025, 9, 1),
    _d(2026, 1, 19),
    _d(2026, 5, 25),
    _d(2026, 7, 4),
    _d(2026, 9, 7),
    _d(2027, 1, 18),
    _d(2027, 5, 31),
    _d(2027, 7, 4),
    _d(2027, 9, 6),
    _d(2028, 1, 17),
    _d(2028, 5, 29),
    _d(2028, 7, 4),
    _d(2028, 9, 4),
    _d(2029, 1, 15),
    _d(2029, 5, 28),
    _d(2029, 7, 4),
    _d(2029, 9, 3),
    _d(2030, 1, 21),
    _d(2030, 5, 27),
    _d(2030, 7, 4),
}
ACADEMIC_COMMENCEMENT_DAYS = {
    _d(2025, 12, 14),
    _d(2026, 5, 8),
    _d(2026, 5, 9),
    _d(2026, 12, 13),
    _d(2027, 5, 7),
    _d(2027, 5, 8),
    _d(2027, 12, 19),
    _d(2028, 5, 12),
    _d(2028, 5, 13),
    _d(2028, 12, 17),
    _d(2029, 5, 11),
    _d(2029, 5, 12),
    _d(2029, 12, 16),
    _d(2030, 5, 10),
    _d(2030, 5, 11),
}
ACADEMIC_GRADING_DEADLINES = {
    _d(2025, 12, 21),
    _d(2026, 5, 11),
    _d(2026, 8, 12),
    _d(2026, 12, 20),
    _d(2027, 5, 10),
    _d(2027, 8, 11),
    _d(2027, 12, 26),
    _d(2028, 5, 15),
    _d(2028, 8, 16),
    _d(2028, 12, 24),
    _d(2029, 5, 14),
    _d(2029, 8, 15),
    _d(2029, 12, 23),
    _d(2030, 5, 13),
    _d(2030, 8, 14),
}
ACADEMIC_TERM_START_DATES = sorted(
    {
        start
        for start, _end in (ACADEMIC_FALL_INSTRUCTION + ACADEMIC_SPRING_INSTRUCTION + ACADEMIC_SUMMER_SESSION)
    }
)
ACADEMIC_EXAM_START_DATES = sorted(start for start, _end in ACADEMIC_EXAMS)
CALENDAR_FEATURE_COUNT = 12
LAG_TREND_SENSOR_FEATURE_COUNT = 19

SQL_HISTORY_BASE = """
SELECT
    location_id,
    last_updated,
    fetched_at,
    current_capacity,
    is_closed,
    max_capacity
FROM location_history
WHERE current_capacity IS NOT NULL
  AND (is_closed = 0 OR is_closed IS NULL)
"""


def db_connect():
    host = os.getenv("GYM_DB_HOST", "localhost")
    port = int(os.getenv("GYM_DB_PORT", "3306"))
    user = os.getenv("GYM_DB_USER", "root")
    password = os.getenv("GYM_DB_PASSWORD", "faDpud-vydqys-batto9")
    database = os.getenv("GYM_DB_NAME", "gym_data")

    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        autocommit=True,
        charset="utf8mb4",
    )


def all_location_ids() -> List[int]:
    ids = []
    for facility in FACILITIES.values():
        for category in facility["categories"]:
            ids.extend(category["location_ids"])
    return sorted(set(ids))


def facility_location_ids(facility: Dict[str, object]) -> List[int]:
    ids = []
    for category in facility.get("categories", []):
        ids.extend(category.get("location_ids", []))
    return sorted(set(int(loc_id) for loc_id in ids))


def location_to_facility_map() -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for facility_id, facility in FACILITIES.items():
        for loc_id in facility_location_ids(facility):
            mapping[int(loc_id)] = int(facility_id)
    return mapping


def model_unit_key(facility_id: int, category_key: str) -> str:
    return f"{int(facility_id)}::{category_key}"


def iter_model_unit_keys() -> List[str]:
    keys: List[str] = []
    for facility_id, facility in FACILITIES.items():
        keys.append(model_unit_key(facility_id, "__all__"))
        for category in facility.get("categories", []):
            category_key = str(category.get("key"))
            if category_key and should_train_category(category_key):
                keys.append(model_unit_key(facility_id, category_key))
    return sorted(set(keys))


def model_artifact_paths(model_key: str) -> Tuple[str, str, str, str]:
    safe_key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(model_key)).strip("._")
    if not safe_key:
        safe_key = "default"
    stem = os.path.join(MODEL_ARTIFACT_DIR, f"{MODEL_BASENAME}_{safe_key}")
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
    ensure_dir(MODEL_ARTIFACT_DIR)

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
    for model_key in iter_model_unit_keys():
        meta = load_saved_meta_only(model_key)
        if isinstance(meta, dict):
            output[model_key] = meta
    return output


def derive_adaptive_runtime_controls(
    meta_snapshots: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    controls = {
        "enabled": ADAPTIVE_CONTROLS_ENABLED,
        "mode": "default",
        "retrainHours": max(1, int(MODEL_RETRAIN_HOURS)),
        "driftRecentDays": max(1, int(DRIFT_RECENT_DAYS)),
        "driftAlertMultiplier": float(DRIFT_ALERT_MULTIPLIER),
        "driftActionStreakForRetrain": max(1, int(DRIFT_ACTION_STREAK_FOR_RETRAIN)),
        "samples": 0,
        "alertRate": None,
        "medianMaeRatio": None,
    }
    if not ADAPTIVE_CONTROLS_ENABLED:
        return controls

    history_rows: List[Dict[str, object]] = []
    for meta in meta_snapshots.values():
        rows = meta.get("driftHistory")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    history_rows.append(row)
    if not history_rows:
        return controls

    history_rows = history_rows[-max(10, ADAPTIVE_HISTORY_MAX_POINTS) :]
    alert_values: List[float] = []
    mae_ratios: List[float] = []
    for row in history_rows:
        alert_values.append(1.0 if bool(row.get("alert")) else 0.0)
        recent_mae = row.get("recentMae")
        baseline_mae = row.get("baselineMae")
        try:
            recent = float(recent_mae) if recent_mae is not None else None
            baseline = float(baseline_mae) if baseline_mae is not None else None
        except Exception:
            recent = None
            baseline = None
        if recent is not None and baseline is not None and baseline > 0:
            mae_ratios.append(recent / baseline)

    if not alert_values:
        return controls

    alert_rate = float(sum(alert_values) / len(alert_values))
    median_ratio = float(np.median(np.array(mae_ratios, dtype=np.float32))) if mae_ratios else None
    controls["samples"] = len(alert_values)
    controls["alertRate"] = round(alert_rate, 4)
    controls["medianMaeRatio"] = round(median_ratio, 4) if median_ratio is not None else None

    retrain_hours = max(1, int(MODEL_RETRAIN_HOURS))
    drift_days = max(1, int(DRIFT_RECENT_DAYS))
    drift_mult = float(DRIFT_ALERT_MULTIPLIER)
    action_streak = max(1, int(DRIFT_ACTION_STREAK_FOR_RETRAIN))

    stable = alert_rate <= ADAPTIVE_ALERT_RATE_STABLE_MAX and (
        median_ratio is None or median_ratio <= 1.03
    )
    unstable = alert_rate >= ADAPTIVE_ALERT_RATE_UNSTABLE_MIN or (
        median_ratio is not None and median_ratio >= 1.12
    )

    if stable:
        controls["mode"] = "stable_relax"
        retrain_hours = int(round(retrain_hours * 1.5))
        drift_days = drift_days + 3
        drift_mult = drift_mult + 0.05
        action_streak = action_streak + 1
    elif unstable:
        controls["mode"] = "unstable_tighten"
        retrain_hours = int(round(retrain_hours * 0.6))
        drift_days = drift_days - 3
        drift_mult = drift_mult - 0.05
        action_streak = action_streak - 1

    retrain_hours = max(int(ADAPTIVE_RETRAIN_MIN_HOURS), min(int(ADAPTIVE_RETRAIN_MAX_HOURS), retrain_hours))
    drift_days = max(int(ADAPTIVE_DRIFT_DAYS_MIN), min(int(ADAPTIVE_DRIFT_DAYS_MAX), drift_days))
    drift_mult = max(float(ADAPTIVE_DRIFT_MULT_MIN), min(float(ADAPTIVE_DRIFT_MULT_MAX), drift_mult))
    action_streak = max(1, min(6, int(action_streak)))

    controls["retrainHours"] = retrain_hours
    controls["driftRecentDays"] = drift_days
    controls["driftAlertMultiplier"] = round(drift_mult, 4)
    controls["driftActionStreakForRetrain"] = int(action_streak)
    return controls


def to_local(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = DB_TZ.localize(dt)
    else:
        dt = dt.astimezone(DB_TZ)
    return dt.astimezone(TZ)


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
        return TZ.localize(dt)
    return dt.astimezone(TZ)


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
            parsed = DB_TZ.localize(parsed)
        else:
            parsed = parsed.astimezone(DB_TZ)
        return parsed.astimezone(TZ)

    return None


def model_feature_count(loc_count: int) -> int:
    # time/cycle + calendar + lags + weather stats + rolling weather + one-hot
    return (
        17
        + CALENDAR_FEATURE_COUNT
        + LAG_TREND_SENSOR_FEATURE_COUNT
        + (len(WEATHER_KEYS) * 3)
        + len(WEATHER_ROLLING_KEYS)
        + loc_count
    )


def last_before(times: List[datetime], values: List[float], target: datetime) -> Tuple[float, float]:
    idx = bisect.bisect_left(times, target) - 1
    if idx >= 0:
        return values[idx], (target - times[idx]).total_seconds() / 60.0
    return float("nan"), float("nan")


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
    for exam_start in ACADEMIC_EXAM_START_DATES:
        gap = (exam_start - target_date).days
        if 1 <= gap <= 7:
            return True
    return False


def build_calendar_features(dt: datetime) -> List[float]:
    d = dt.date()
    in_fall = date_in_ranges(d, ACADEMIC_FALL_INSTRUCTION)
    in_spring = date_in_ranges(d, ACADEMIC_SPRING_INSTRUCTION)
    in_exams = date_in_ranges(d, ACADEMIC_EXAMS)
    in_thanksgiving = date_in_ranges(d, ACADEMIC_THANKSGIVING_RECESS)
    in_spring_recess = date_in_ranges(d, ACADEMIC_SPRING_RECESS)
    in_summer = date_in_ranges(d, ACADEMIC_SUMMER_SESSION)

    term_progress = max(
        range_progress(d, ACADEMIC_FALL_INSTRUCTION),
        range_progress(d, ACADEMIC_SPRING_INSTRUCTION),
        range_progress(d, ACADEMIC_SUMMER_SESSION),
    )
    since_term_start = most_recent_date_distance(d, ACADEMIC_TERM_START_DATES, max_days=150)

    features = [
        float(in_fall),
        float(in_spring),
        float(in_exams),
        float(d in ACADEMIC_STUDY_DAYS),
        float(in_thanksgiving),
        float(in_spring_recess),
        float(in_summer),
        float(d in ACADEMIC_HOLIDAYS),
        float(d in ACADEMIC_COMMENCEMENT_DAYS),
        float(d in ACADEMIC_GRADING_DEADLINES),
        float(is_pre_exam_week(d)),
        float(term_progress if term_progress >= 0.0 else since_term_start),
    ]
    return features


def build_time_features(dt: datetime) -> List[float]:
    hour = dt.hour
    minute = dt.minute
    quarter_slot = minute // max(1, RESAMPLE_MINUTES)
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


def rolling_mean(
    bucket_map: Dict[datetime, float],
    target: datetime,
    steps: int,
) -> float:
    values = []
    for i in range(1, steps + 1):
        ts = target - timedelta(minutes=RESAMPLE_MINUTES * i)
        value = bucket_map.get(ts)
        if value is not None:
            values.append(value)
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def count_consecutive_flatline_steps(
    bucket_map: Dict[datetime, float],
    target: datetime,
    steps: int,
    tolerance: float = 0.002,
) -> int:
    prev = None
    count = 0
    for i in range(1, max(1, steps) + 1):
        ts = target - timedelta(minutes=RESAMPLE_MINUTES * i)
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
        ts = target - timedelta(minutes=RESAMPLE_MINUTES * i)
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
        steps=max(2, int(120 / max(1, RESAMPLE_MINUTES))),
    )
    missing_ratio = recent_missing_ratio(
        bucket_map=bucket_map,
        target=target,
        steps=max(2, int(120 / max(1, RESAMPLE_MINUTES))),
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
    return max(float(SENSOR_WEIGHT_MIN), min(1.0, float(weight)))


def to_float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


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
        if value is not None:
            return float(value)
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
        direct = row.get(key)
        if direct is not None:
            value = float(direct)

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
        ts = target - timedelta(minutes=RESAMPLE_MINUTES * i)
        value = weather_value_at_or_before(times, weather_map, ts, key, cache=cache)
        if not math.isnan(value):
            values.append(value)
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def build_features(
    target: datetime,
    loc_data: Dict[str, object],
    onehot: List[float],
    weather_source: Optional[Dict[str, object]] = None,
    weather_lookup_cache: Optional[Dict[Tuple[datetime, str], float]] = None,
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

    lag_15m = bucket_map.get(target - timedelta(minutes=RESAMPLE_MINUTES))
    lag_1h = bucket_map.get(target - timedelta(hours=1))
    lag_2h = bucket_map.get(target - timedelta(hours=2))
    lag_24h = bucket_map.get(target - timedelta(hours=24))
    lag_7d = bucket_map.get(target - timedelta(days=7))

    if lag_15m is None:
        lag_15m = float("nan")
    if lag_1h is None:
        lag_1h = float("nan")
    if lag_2h is None:
        lag_2h = float("nan")
    if lag_24h is None:
        lag_24h = float("nan")
    if lag_7d is None:
        lag_7d = float("nan")

    last_bucket, minutes_since_bucket = last_before(bucket_times, bucket_values, target)
    flatline_steps_recent, missing_ratio_recent, minutes_since_raw = sensor_quality_signals(
        target=target,
        bucket_map=bucket_map,
        raw_times=raw_times,
        raw_values=raw_values,
    )

    delta_1h = float("nan")
    delta_24h = float("nan")
    delta_7d = float("nan")
    if not math.isnan(lag_15m) and not math.isnan(lag_1h):
        delta_1h = lag_15m - lag_1h
    if not math.isnan(lag_15m) and not math.isnan(lag_24h):
        delta_24h = lag_15m - lag_24h
    if not math.isnan(lag_15m) and not math.isnan(lag_7d):
        delta_7d = lag_15m - lag_7d

    roll_1h = rolling_mean(
        bucket_map,
        target,
        steps=max(1, int(60 / RESAMPLE_MINUTES)),
    )
    roll_2h = rolling_mean(
        bucket_map,
        target,
        steps=max(1, int(120 / RESAMPLE_MINUTES)),
    )
    roll_6h = rolling_mean(
        bucket_map,
        target,
        steps=max(1, int(360 / RESAMPLE_MINUTES)),
    )
    roll_24h = rolling_mean(
        bucket_map,
        target,
        steps=max(1, int(1440 / RESAMPLE_MINUTES)),
    )
    trend_short = float("nan")
    trend_long = float("nan")
    if not math.isnan(roll_1h) and not math.isnan(roll_2h):
        trend_short = roll_1h - roll_2h
    if not math.isnan(roll_2h) and not math.isnan(roll_24h):
        trend_long = roll_2h - roll_24h
    weather_features: List[float] = []
    weather_ts = target
    lag_ts = target - timedelta(hours=1)

    for key in WEATHER_KEYS:
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

        weather_features.extend(
            [
                float(weather_now),
                float(weather_1h),
                float(weather_delta_1h),
            ]
        )

    for key in WEATHER_ROLLING_KEYS:
        weather_roll = weather_rolling_mean(
            weather_bucket_times,
            weather_bucket_map,
            target,
            steps=max(1, int(180 / RESAMPLE_MINUTES)),
            key=key,
            cache=weather_lookup_cache,
        )
        weather_features.append(float(weather_roll))

    return (
        build_time_features(target)
        + [
            float(lag_15m),
            float(lag_1h),
            float(lag_2h),
            float(lag_24h),
            float(lag_7d),
            float(delta_1h),
            float(delta_24h),
            float(delta_7d),
            float(roll_1h),
            float(roll_2h),
            float(roll_6h),
            float(roll_24h),
            float(trend_short),
            float(trend_long),
            float(last_bucket),
            float(minutes_since_bucket),
            float(minutes_since_raw),
            float(flatline_steps_recent),
            float(missing_ratio_recent),
        ]
        + weather_features
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
    max_jump = max_cap * IMPOSSIBLE_JUMP_PCT

    for ts, value in entries[1:]:
        prev_ts, prev_value = cleaned[-1]
        gap_min = (ts - prev_ts).total_seconds() / 60.0
        jump = abs(value - prev_value)
        if gap_min <= IMPOSSIBLE_JUMP_MAX_GAP_MIN and jump > max_jump:
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

    tolerance = max(0.0, float(max_cap) * max(0.0, SENSOR_FLATLINE_TOLERANCE_PCT))
    max_gap = max(1.0, float(SENSOR_FLATLINE_MAX_GAP_MIN))
    min_duration = max(1.0, float(SENSOR_FLATLINE_MIN_DURATION_MIN))
    keep_interval = max(1.0, float(SENSOR_FLATLINE_KEEP_INTERVAL_MIN))

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


def load_history(conn):
    raw_by_loc: Dict[int, List[Tuple[datetime, float]]] = {}
    max_caps: Dict[int, int] = {}

    quality = {
        "rowsRead": 0,
        "rowsDroppedInvalid": 0,
        "duplicatesRemoved": 0,
        "impossibleJumpsRemoved": 0,
        "flatlineRowsPruned": 0,
        "flatlineRunsDetected": 0,
        "flatlineLocations": [],
        "staleLocations": [],
    }

    sql = SQL_HISTORY_BASE
    params: Tuple[object, ...] = ()
    if HISTORY_DAYS > 0:
        sql += " AND fetched_at >= %s"
        since = datetime.now(DB_TZ) - timedelta(days=HISTORY_DAYS)
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

            local_dt = parse_observed_at_value(last_updated)
            if local_dt is None and fetched_at is not None:
                local_dt = to_local(fetched_at)
            if local_dt is None:
                quality["rowsDroppedInvalid"] += 1
                continue

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
    now_local = datetime.now(TZ)

    for loc_id, entries in raw_by_loc.items():
        max_cap = max_caps.get(loc_id, 0)
        if max_cap <= 0:
            continue

        deduped, dup_removed = dedupe_exact_timestamps(entries)
        quality["duplicatesRemoved"] += dup_removed

        cleaned, jump_removed = drop_impossible_jumps(deduped, max_cap=max_cap)
        quality["impossibleJumpsRemoved"] += jump_removed

        cleaned, flatline_removed, flatline_runs = drop_flatline_plateaus(cleaned, max_cap=max_cap)
        quality["flatlineRowsPruned"] += flatline_removed
        quality["flatlineRunsDetected"] += flatline_runs
        if flatline_runs > 0:
            quality["flatlineLocations"].append(loc_id)

        if not cleaned:
            continue

        latest_ts = cleaned[-1][0]
        age_hours = (now_local - latest_ts).total_seconds() / 3600.0
        is_stale = age_hours > STALE_SENSOR_HOURS
        if is_stale:
            quality["staleLocations"].append(loc_id)

        raw_times = [row[0] for row in cleaned]
        raw_values = [float(row[1]) for row in cleaned]

        bucket_counts: Dict[datetime, List[float]] = {}
        for ts, count in cleaned:
            bucket = floor_time(ts, RESAMPLE_MINUTES)
            bucket_counts.setdefault(bucket, []).append(count)

        if not bucket_counts:
            continue

        bucket_times = sorted(bucket_counts.keys())
        bucket_values: List[float] = []
        bucket_map: Dict[datetime, float] = {}

        for bt in bucket_times:
            avg_count = float(sum(bucket_counts[bt]) / len(bucket_counts[bt]))
            ratio = avg_count / max_cap
            ratio = max(0.0, min(ratio, 1.2))
            bucket_values.append(ratio)
            bucket_map[bt] = ratio

            aggregate_sum_count(avg_dow_hour_sum, (loc_id, bt.weekday(), bt.hour), ratio)
            aggregate_sum_count(avg_hour_sum, (loc_id, bt.hour), ratio)
            aggregate_sum_count(avg_overall_sum, loc_id, ratio)

        loc_data[loc_id] = {
            "raw_times": raw_times,
            "raw_values": raw_values,
            "bucket_times": bucket_times,
            "bucket_values": bucket_values,
            "bucket_map": bucket_map,
            "max_cap": max_cap,
            "is_stale": is_stale,
            "latest_ts": latest_ts.isoformat(),
        }
        loc_samples[loc_id] = len(bucket_times)

    quality["locationsWithHistory"] = len(loc_data)
    quality["staleLocations"] = sorted(set(quality["staleLocations"]))
    quality["staleLocationsCount"] = len(quality["staleLocations"])
    quality["flatlineLocations"] = sorted(set(quality["flatlineLocations"]))
    quality["flatlineLocationsCount"] = len(quality["flatlineLocations"])

    avg_dow_hour = finalize_averages(avg_dow_hour_sum)
    avg_hour = finalize_averages(avg_hour_sum)
    avg_overall = finalize_averages(avg_overall_sum)

    return loc_data, avg_dow_hour, avg_hour, avg_overall, max_caps, loc_samples, quality


def build_data_quality_alerts(
    quality: Dict[str, object],
    loc_data: Dict[int, Dict[str, object]],
    loc_samples: Dict[int, int],
) -> Dict[str, object]:
    rows_read = int(quality.get("rowsRead", 0) or 0)
    rows_dropped = int(quality.get("rowsDroppedInvalid", 0) or 0)
    locations_with_history = max(1, int(quality.get("locationsWithHistory", 0) or 0))
    stale_count = int(quality.get("staleLocationsCount", 0) or 0)
    flatline_count = int(quality.get("flatlineLocationsCount", 0) or 0)

    invalid_rate = float(rows_dropped) / float(rows_read) if rows_read > 0 else 0.0
    stale_rate = float(stale_count) / float(locations_with_history)
    flatline_rate = float(flatline_count) / float(locations_with_history)
    modeled_locations = sum(
        1
        for loc_id, data in loc_data.items()
        if loc_samples.get(loc_id, 0) >= MIN_SAMPLES_PER_LOC and not data.get("is_stale")
    )

    warnings: List[str] = []
    critical: List[str] = []

    if invalid_rate > float(DATA_QUALITY_MAX_INVALID_ROW_RATE):
        critical.append("invalid_row_rate_high")
    elif invalid_rate > float(DATA_QUALITY_MAX_INVALID_ROW_RATE) * 0.75:
        warnings.append("invalid_row_rate_elevated")

    if stale_rate > float(DATA_QUALITY_MAX_STALE_LOC_RATE):
        critical.append("stale_location_rate_high")
    elif stale_rate > float(DATA_QUALITY_MAX_STALE_LOC_RATE) * 0.75:
        warnings.append("stale_location_rate_elevated")

    if flatline_rate > float(DATA_QUALITY_MAX_FLATLINE_LOC_RATE):
        critical.append("flatline_location_rate_high")
    elif flatline_rate > float(DATA_QUALITY_MAX_FLATLINE_LOC_RATE) * 0.75:
        warnings.append("flatline_location_rate_elevated")

    if modeled_locations < int(DATA_QUALITY_MIN_LOCATIONS_MODELED):
        critical.append("modeled_locations_too_low")

    severity = "ok"
    if critical:
        severity = "critical"
    elif warnings:
        severity = "warning"

    block_training = bool(DATA_QUALITY_ALERTS_ENABLED and severity == "critical")
    return {
        "enabled": DATA_QUALITY_ALERTS_ENABLED,
        "severity": severity,
        "blockTraining": block_training,
        "rowsRead": rows_read,
        "rowsDroppedInvalid": rows_dropped,
        "invalidRowRate": round(invalid_rate, 4),
        "staleLocationRate": round(stale_rate, 4),
        "flatlineLocationRate": round(flatline_rate, 4),
        "modeledLocations": int(modeled_locations),
        "warnings": warnings,
        "critical": critical,
    }


def build_onehot(loc_ids: Iterable[int]) -> Dict[int, List[float]]:
    unique = sorted(set(loc_ids))
    vectors: Dict[int, List[float]] = {}
    for idx, loc_id in enumerate(unique):
        vec = [0.0] * len(unique)
        vec[idx] = 1.0
        vectors[loc_id] = vec
    return vectors


def build_interval_profile(
    val_times: List[datetime],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, object]:
    residuals = y_true - y_pred
    profile: Dict[str, object] = {
        "global": {
            "q10": float(np.quantile(residuals, INTERVAL_Q_LOW)),
            "q90": float(np.quantile(residuals, INTERVAL_Q_HIGH)),
            "count": int(len(residuals)),
        },
        "byHour": {},
    }

    for hour in range(24):
        idx = [i for i, ts in enumerate(val_times) if ts.hour == hour]
        if len(idx) < INTERVAL_MIN_SAMPLES_PER_HOUR:
            continue
        hour_res = residuals[idx]
        profile["byHour"][str(hour)] = {
            "q10": float(np.quantile(hour_res, INTERVAL_Q_LOW)),
            "q90": float(np.quantile(hour_res, INTERVAL_Q_HIGH)),
            "count": int(len(hour_res)),
        }

    return profile


def build_xgb_params(overrides: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    params: Dict[str, object] = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": max(1, MODEL_MAX_DEPTH),
        "eta": max(0.001, float(MODEL_ETA)),
        "subsample": max(0.1, min(1.0, float(MODEL_SUBSAMPLE))),
        "colsample_bytree": max(0.1, min(1.0, float(MODEL_COLSAMPLE_BYTREE))),
        "min_child_weight": max(0.0, float(MODEL_MIN_CHILD_WEIGHT)),
        "lambda": 1.0,
        "alpha": 0.0,
        "verbosity": 0,
        "nthread": max(1, MODEL_NTHREAD),
    }
    if MODEL_TREE_METHOD:
        params["tree_method"] = MODEL_TREE_METHOD
        if MODEL_TREE_METHOD in {"hist", "approx"}:
            params["max_bin"] = max(64, MODEL_MAX_BIN)
    if overrides:
        params.update(overrides)
    return params


def build_recency_weights(times: List[datetime]) -> np.ndarray:
    if not times:
        return np.array([], dtype=np.float32)

    if not RECENCY_WEIGHT_ENABLED:
        return np.ones(len(times), dtype=np.float32)

    half_life_days = max(1.0, float(RECENCY_HALFLIFE_DAYS))
    min_weight = max(0.0, min(1.0, float(RECENCY_MIN_WEIGHT)))
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
    if not OCCUPANCY_WEIGHT_ENABLED:
        return np.ones(y.size, dtype=np.float32)

    alpha = max(0.0, float(OCCUPANCY_WEIGHT_ALPHA))
    gamma = max(1.0, float(OCCUPANCY_WEIGHT_GAMMA))
    clipped = np.clip(y.astype(np.float32), 0.0, 1.2)
    boosted = np.power(clipped, gamma)
    return (1.0 + alpha * boosted).astype(np.float32)


def weighted_average(values: np.ndarray, weights: Optional[np.ndarray]) -> float:
    if values.size == 0:
        return 0.0
    if weights is None or weights.size == 0:
        return float(np.mean(values))
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def feature_group_slices(loc_count: int) -> Dict[str, Tuple[int, int]]:
    time_end = 17 + CALENDAR_FEATURE_COUNT
    lag_end = time_end + LAG_TREND_SENSOR_FEATURE_COUNT
    weather_end = lag_end + (len(WEATHER_KEYS) * 3) + len(WEATHER_ROLLING_KEYS)
    onehot_end = weather_end + max(0, int(loc_count))
    return {
        "time_calendar": (0, time_end),
        "lags_trends_sensor": (time_end, lag_end),
        "weather": (lag_end, weather_end),
        "onehot_location": (weather_end, onehot_end),
    }


def evaluate_feature_ablation(
    p50_model: xgb.Booster,
    X_val: np.ndarray,
    y_val: np.ndarray,
    w_val: np.ndarray,
    loc_count: int,
    baseline_mae: float,
) -> Optional[Dict[str, object]]:
    if not FEATURE_ABLATION_ENABLED:
        return None
    if len(y_val) < max(1, FEATURE_ABLATION_MIN_VAL_ROWS):
        return None
    if X_val.ndim != 2 or X_val.shape[0] == 0 or X_val.shape[1] == 0:
        return None

    slices = feature_group_slices(loc_count)
    groups: Dict[str, Dict[str, float]] = {}

    for name, (start, end) in slices.items():
        if start >= end or start < 0 or end > X_val.shape[1]:
            continue
        X_ab = np.array(X_val, copy=True)
        if name == "onehot_location":
            X_ab[:, start:end] = 0.0
        else:
            X_ab[:, start:end] = np.nan
        preds = p50_model.predict(xgb.DMatrix(X_ab))
        mae = weighted_average(np.abs(preds - y_val), w_val)
        groups[name] = {
            "mae": round(float(mae), 6),
            "deltaMae": round(float(mae - baseline_mae), 6),
            "start": int(start),
            "end": int(end),
            "features": int(end - start),
        }

    if not groups:
        return None

    ranking = sorted(groups.items(), key=lambda item: float(item[1].get("deltaMae", 0.0)), reverse=True)
    return {
        "baselineMae": round(float(baseline_mae), 6),
        "groups": groups,
        "ranking": [name for name, _ in ranking],
    }


def sorted_time_indices(times: List[datetime]) -> List[int]:
    return sorted(range(len(times)), key=lambda idx: times[idx])


def build_time_series_cv_splits(times: List[datetime], folds: int) -> List[Tuple[List[int], List[int]]]:
    n = len(times)
    if n < 60:
        return []

    sorted_idx = sorted_time_indices(times)
    folds = max(2, int(folds))
    splits: List[Tuple[List[int], List[int]]] = []

    for fold in range(1, folds + 1):
        train_end = int(round(n * (fold / float(folds + 1))))
        val_end = int(round(n * ((fold + 1) / float(folds + 1))))
        train_end = max(20, min(train_end, n - 20))
        val_end = max(train_end + 10, min(val_end, n))
        if val_end - train_end < 10:
            continue
        train_idx = sorted_idx[:train_end]
        val_idx = sorted_idx[train_end:val_end]
        if len(train_idx) >= 20 and len(val_idx) >= 10:
            splits.append((train_idx, val_idx))

    return splits


def evaluate_params_cv(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    times: List[datetime],
    params: Dict[str, object],
) -> float:
    splits = build_time_series_cv_splits(times, MODEL_TUNING_CV_FOLDS)
    if not splits:
        return float("inf")

    cv_errors: List[float] = []
    tune_rounds = max(20, min(MODEL_TUNING_BOOST_ROUND, MODEL_NUM_BOOST_ROUND))

    for train_idx, val_idx in splits:
        X_train = X[train_idx]
        y_train = y[train_idx]
        w_train = weights[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        w_val = weights[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
        dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)
        try:
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=tune_rounds,
                evals=[(dval, "val")],
                early_stopping_rounds=min(EARLY_STOPPING_ROUNDS, 30),
                verbose_eval=False,
            )
        except Exception:
            return float("inf")

        preds = model.predict(dval)
        fold_mae = weighted_average(np.abs(preds - y_val), w_val)
        cv_errors.append(float(fold_mae))

    if not cv_errors:
        return float("inf")
    return float(sum(cv_errors) / len(cv_errors))


def normalize_tuning_overrides(overrides: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not overrides or not isinstance(overrides, dict):
        return None
    out: Dict[str, object] = {}
    try:
        if "max_depth" in overrides:
            out["max_depth"] = int(overrides["max_depth"])
        if "eta" in overrides:
            out["eta"] = float(overrides["eta"])
        if "min_child_weight" in overrides:
            out["min_child_weight"] = float(overrides["min_child_weight"])
        if "subsample" in overrides:
            out["subsample"] = float(overrides["subsample"])
        if "colsample_bytree" in overrides:
            out["colsample_bytree"] = float(overrides["colsample_bytree"])
        if "max_bin" in overrides:
            out["max_bin"] = int(overrides["max_bin"])
    except Exception:
        return None
    return out or None


def params_for_meta(params: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not params:
        return None
    try:
        return {
            "max_depth": int(params.get("max_depth", MODEL_MAX_DEPTH)),
            "eta": float(params.get("eta", MODEL_ETA)),
            "min_child_weight": float(params.get("min_child_weight", MODEL_MIN_CHILD_WEIGHT)),
            "subsample": float(params.get("subsample", MODEL_SUBSAMPLE)),
            "colsample_bytree": float(params.get("colsample_bytree", MODEL_COLSAMPLE_BYTREE)),
            "max_bin": int(params.get("max_bin", MODEL_MAX_BIN)),
        }
    except Exception:
        return None


def build_tuning_candidates(
    base_params: Dict[str, object],
    preferred_params: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    limit = max(1, MODEL_TUNING_MAX_CANDIDATES)
    rng = random.Random(MODEL_TUNING_RANDOM_SEED)
    candidates: List[Dict[str, object]] = []
    seen = set()

    def candidate_key(cand: Dict[str, object]) -> Tuple[object, ...]:
        return (
            int(cand.get("max_depth", MODEL_MAX_DEPTH)),
            round(float(cand.get("eta", MODEL_ETA)), 6),
            round(float(cand.get("min_child_weight", MODEL_MIN_CHILD_WEIGHT)), 6),
            round(float(cand.get("subsample", MODEL_SUBSAMPLE)), 6),
            round(float(cand.get("colsample_bytree", MODEL_COLSAMPLE_BYTREE)), 6),
            int(cand.get("max_bin", MODEL_MAX_BIN)),
        )

    def add_candidate(overrides: Optional[Dict[str, object]] = None) -> None:
        if len(candidates) >= limit:
            return
        cand = dict(base_params)
        if overrides:
            cand.update(overrides)
        cand["max_depth"] = max(2, int(cand.get("max_depth", MODEL_MAX_DEPTH)))
        cand["eta"] = max(0.005, min(0.2, float(cand.get("eta", MODEL_ETA))))
        cand["min_child_weight"] = max(0.0, float(cand.get("min_child_weight", MODEL_MIN_CHILD_WEIGHT)))
        cand["subsample"] = max(0.5, min(1.0, float(cand.get("subsample", MODEL_SUBSAMPLE))))
        cand["colsample_bytree"] = max(0.5, min(1.0, float(cand.get("colsample_bytree", MODEL_COLSAMPLE_BYTREE))))
        if "max_bin" in cand:
            cand["max_bin"] = max(64, int(cand.get("max_bin", MODEL_MAX_BIN)))
        key = candidate_key(cand)
        if key in seen:
            return
        seen.add(key)
        candidates.append(cand)

    preferred = normalize_tuning_overrides(preferred_params)
    add_candidate(preferred)
    add_candidate()

    anchors: List[Dict[str, object]] = []
    if preferred:
        anchors.append(dict(base_params, **preferred))
    anchors.append(base_params)

    for anchor in anchors:
        depth0 = int(anchor.get("max_depth", MODEL_MAX_DEPTH))
        eta0 = float(anchor.get("eta", MODEL_ETA))
        child0 = float(anchor.get("min_child_weight", MODEL_MIN_CHILD_WEIGHT))
        sub0 = float(anchor.get("subsample", MODEL_SUBSAMPLE))
        col0 = float(anchor.get("colsample_bytree", MODEL_COLSAMPLE_BYTREE))
        bin0 = int(anchor.get("max_bin", MODEL_MAX_BIN))

        depths = sorted(set([max(2, depth0 - 2), max(2, depth0 - 1), depth0, depth0 + 1]))
        etas = sorted(set([max(0.01, eta0 * 0.7), max(0.01, eta0 * 0.85), eta0, min(0.2, eta0 * 1.15)]))
        childs = sorted(set([max(0.0, child0 * 0.5), child0, max(1.0, child0 * 2.0)]))
        subsamples = sorted(set([max(0.6, sub0 - 0.15), sub0, min(1.0, sub0 + 0.1)]))
        colsamples = sorted(set([max(0.6, col0 - 0.15), col0, min(1.0, col0 + 0.1)]))
        bins = sorted(set([max(64, int(round(bin0 * 0.75))), bin0, max(64, int(round(bin0 * 1.25)))]))

        grid = []
        for depth in depths:
            for eta in etas:
                for child in childs:
                    for subsample in subsamples:
                        for colsample in colsamples:
                            for max_bin in bins:
                                grid.append(
                                    {
                                        "max_depth": depth,
                                        "eta": eta,
                                        "min_child_weight": child,
                                        "subsample": subsample,
                                        "colsample_bytree": colsample,
                                        "max_bin": max_bin,
                                    }
                                )
        rng.shuffle(grid)
        for overrides in grid:
            add_candidate(overrides)
            if len(candidates) >= limit:
                return candidates

    attempts = 0
    max_attempts = limit * 20
    while len(candidates) < limit and attempts < max_attempts:
        attempts += 1
        add_candidate(
            {
                "max_depth": rng.randint(2, 8),
                "eta": rng.uniform(0.01, 0.15),
                "min_child_weight": rng.uniform(0.0, 4.0),
                "subsample": rng.uniform(0.6, 1.0),
                "colsample_bytree": rng.uniform(0.6, 1.0),
                "max_bin": rng.choice([96, 128, 192, 256, 320, 384]),
            }
        )

    return candidates


def choose_best_params(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    times: List[datetime],
    preferred_params: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, object], Optional[float], int]:
    normalized_preferred = normalize_tuning_overrides(preferred_params)
    base = build_xgb_params(normalized_preferred)
    if not MODEL_TUNING_ENABLED or len(times) < MODEL_TUNING_MIN_ROWS:
        return base, None, 0

    candidates = build_tuning_candidates(base, preferred_params=normalized_preferred)
    best_params = base
    best_score = float("inf")
    tested = 0

    for candidate in candidates:
        score = evaluate_params_cv(X, y, weights, times, candidate)
        tested += 1
        if score < best_score:
            best_score = score
            best_params = candidate

    if not math.isfinite(best_score):
        return base, None, tested
    return best_params, best_score, tested


def train_quantile_model(
    alpha: float,
    params: Dict[str, object],
    dtrain: xgb.DMatrix,
    dval: Optional[xgb.DMatrix],
) -> Optional[xgb.Booster]:
    if not DIRECT_QUANTILE_ENABLED:
        return None

    qparams = dict(params)
    qparams.update(
        {
            "objective": "reg:quantileerror",
            "quantile_alpha": float(alpha),
            "eval_metric": "quantile",
        }
    )

    try:
        if dval is not None:
            return xgb.train(
                qparams,
                dtrain,
                num_boost_round=MODEL_NUM_BOOST_ROUND,
                evals=[(dval, "val")],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
            )
        return xgb.train(
            qparams,
            dtrain,
            num_boost_round=MODEL_NUM_BOOST_ROUND,
            verbose_eval=False,
        )
    except Exception:
        return None


def train_model_unit(
    model_key: str,
    loc_ids: Iterable[int],
    loc_data: Dict[int, Dict[str, object]],
    onehot: Dict[int, List[float]],
    loc_samples: Dict[int, int],
    weather_source: Optional[Dict[str, object]],
    preferred_params: Optional[Dict[str, object]] = None,
):
    rows = []
    loc_ids = sorted(set(int(loc_id) for loc_id in loc_ids))
    weather_lookup_cache: Optional[Dict[Tuple[datetime, str], float]] = (
        {} if weather_source is not None else None
    )

    for loc_id in loc_ids:
        data = loc_data.get(loc_id)
        if not data:
            continue
        if loc_samples.get(loc_id, 0) < MIN_SAMPLES_PER_LOC:
            continue
        if data.get("is_stale"):
            continue
        onehot_vec = onehot.get(loc_id)
        if onehot_vec is None:
            continue
        for target, label in zip(data["bucket_times"], data["bucket_values"]):
            features = build_features(
                target,
                data,
                onehot_vec,
                weather_source=weather_source,
                weather_lookup_cache=weather_lookup_cache,
            )
            flatline_steps_recent, missing_ratio_recent, minutes_since_raw = sensor_quality_signals(
                target=target,
                bucket_map=data["bucket_map"],
                raw_times=data["raw_times"],
                raw_values=data["raw_values"],
            )
            quality_weight = sensor_quality_weight(
                flatline_steps=flatline_steps_recent,
                missing_ratio=missing_ratio_recent,
                minutes_since_raw=minutes_since_raw,
            )
            rows.append((target, features, label, quality_weight))

    if len(rows) < MIN_TRAIN_SAMPLES:
        return None, {
            "model_key": model_key,
            "train_rows": len(rows),
            "val_rows": 0,
            "val_mae": None,
            "val_rmse": None,
            "tuning_cv_mae": None,
            "tuned_candidates": 0,
            "quantile_direct": False,
            "best_params": params_for_meta(normalize_tuning_overrides(preferred_params)),
            "feature_ablation": None,
        }, None

    times = [row[0] for row in rows]
    X = np.array([row[1] for row in rows], dtype=np.float32)
    y = np.array([row[2] for row in rows], dtype=np.float32)
    quality_weights = np.array([row[3] for row in rows], dtype=np.float32)
    weights = build_recency_weights(times) * quality_weights * build_occupancy_weights(y)

    params, tuning_score, tuned_candidates = choose_best_params(
        X,
        y,
        weights,
        times,
        preferred_params=preferred_params,
    )

    times_sorted = sorted(times)
    split_idx = int(len(times_sorted) * TRAIN_SPLIT)
    split_idx = max(1, min(split_idx, len(times_sorted) - 1))
    split_time = times_sorted[split_idx]

    train_idx = [i for i, ts in enumerate(times) if ts < split_time]
    val_idx = [i for i, ts in enumerate(times) if ts >= split_time]

    if len(train_idx) < 10 or len(val_idx) < 10:
        dtrain = xgb.DMatrix(X, label=y, weight=weights)
        p50_model = xgb.train(
            params,
            dtrain,
            num_boost_round=MODEL_NUM_BOOST_ROUND,
            verbose_eval=False,
        )
        model_bundle = {
            "p50": p50_model,
            "p10": None,
            "p90": None,
            "quantileDirect": False,
        }
        return model_bundle, {
            "model_key": model_key,
            "train_rows": len(rows),
            "val_rows": 0,
            "val_mae": None,
            "val_rmse": None,
            "tuning_cv_mae": tuning_score,
            "tuned_candidates": tuned_candidates,
            "quantile_direct": False,
            "best_params": params_for_meta(params),
            "feature_ablation": None,
        }, None

    X_train = X[train_idx]
    y_train = y[train_idx]
    w_train = weights[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    w_val = weights[val_idx]
    val_times = [times[i] for i in val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)

    p50_model = xgb.train(
        params,
        dtrain,
        num_boost_round=MODEL_NUM_BOOST_ROUND,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

    preds = p50_model.predict(dval)
    abs_errors = np.abs(preds - y_val)
    sq_errors = (preds - y_val) ** 2
    mae = weighted_average(abs_errors, w_val)
    rmse = math.sqrt(weighted_average(sq_errors, w_val))
    interval_profile = build_interval_profile(val_times, y_val, preds)
    feature_ablation = evaluate_feature_ablation(
        p50_model=p50_model,
        X_val=X_val,
        y_val=y_val,
        w_val=w_val,
        loc_count=len(loc_ids),
        baseline_mae=float(mae),
    )

    p10_model = train_quantile_model(0.10, params, dtrain, dval)
    p90_model = train_quantile_model(0.90, params, dtrain, dval)
    quantile_direct = bool(p10_model is not None and p90_model is not None)

    model_bundle = {
        "p50": p50_model,
        "p10": p10_model,
        "p90": p90_model,
        "quantileDirect": quantile_direct,
    }
    return model_bundle, {
        "model_key": model_key,
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
        "val_mae": mae,
        "val_rmse": rmse,
        "tuning_cv_mae": tuning_score,
        "tuned_candidates": tuned_candidates,
        "quantile_direct": quantile_direct,
        "best_params": params_for_meta(params),
        "feature_ablation": feature_ablation,
    }, interval_profile


def evaluate_model_bundle_on_recent_window(
    model_bundle: Dict[str, object],
    residual_profile: Optional[Dict[str, object]],
    loc_ids: List[int],
    loc_data: Dict[int, Dict[str, object]],
    onehot: Dict[int, List[float]],
    weather_source: Optional[Dict[str, object]],
    since: datetime,
) -> Optional[Dict[str, object]]:
    p50_model = model_bundle.get("p50")
    if p50_model is None:
        return None

    target_coverage = max(0.1, min(0.98, float(INTERVAL_Q_HIGH - INTERVAL_Q_LOW)))
    weather_lookup_cache: Optional[Dict[Tuple[datetime, str], float]] = (
        {} if weather_source is not None else None
    )

    points = 0
    sum_weight = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    sum_within = 0.0

    for loc_id in loc_ids:
        data = loc_data.get(loc_id)
        if not data:
            continue
        onehot_vec = onehot.get(loc_id)
        if onehot_vec is None:
            continue

        features_rows: List[List[float]] = []
        labels: List[float] = []
        times: List[datetime] = []
        hours: List[int] = []
        quality_weights: List[float] = []

        for ts, label in zip(data.get("bucket_times", []), data.get("bucket_values", [])):
            if ts < since:
                continue
            features_rows.append(
                build_features(
                    ts,
                    data,
                    onehot_vec,
                    weather_source=weather_source,
                    weather_lookup_cache=weather_lookup_cache,
                )
            )
            labels.append(float(label))
            times.append(ts)
            hours.append(int(ts.hour))
            flatline_steps_recent, missing_ratio_recent, minutes_since_raw = sensor_quality_signals(
                target=ts,
                bucket_map=data["bucket_map"],
                raw_times=data["raw_times"],
                raw_values=data["raw_values"],
            )
            quality_weights.append(
                sensor_quality_weight(
                    flatline_steps=flatline_steps_recent,
                    missing_ratio=missing_ratio_recent,
                    minutes_since_raw=minutes_since_raw,
                )
            )

        if not features_rows:
            continue

        features_arr = np.array(features_rows, dtype=np.float32)
        labels_arr = np.array(labels, dtype=np.float32)
        dmat = xgb.DMatrix(features_arr)

        p50 = p50_model.predict(dmat).astype(np.float32)
        p10_model = model_bundle.get("p10")
        p90_model = model_bundle.get("p90")

        if p10_model is not None and p90_model is not None:
            p10 = p10_model.predict(dmat).astype(np.float32)
            p90 = p90_model.predict(dmat).astype(np.float32)
        else:
            p10_vals: List[float] = []
            p90_vals: List[float] = []
            for pred, hour in zip(p50.tolist(), hours):
                b10, _mid, b90 = interval_bounds(
                    point_ratio=float(pred),
                    hour=int(hour),
                    residual_profile=residual_profile,
                )
                p10_vals.append(float(b10))
                p90_vals.append(float(b90))
            p10 = np.array(p10_vals, dtype=np.float32)
            p90 = np.array(p90_vals, dtype=np.float32)

        p10 = np.minimum(p10, p90)
        p90 = np.maximum(p10, p90)

        recency_w = build_recency_weights(times)
        occupancy_w = build_occupancy_weights(labels_arr)
        quality_w = np.array(quality_weights, dtype=np.float32)
        weights = recency_w * occupancy_w * quality_w
        total_w = float(np.sum(weights))
        if total_w <= 0.0:
            weights = np.ones_like(labels_arr, dtype=np.float32)
            total_w = float(np.sum(weights))

        abs_err = np.abs(p50 - labels_arr)
        sq_err = (p50 - labels_arr) ** 2
        within = ((labels_arr >= p10) & (labels_arr <= p90)).astype(np.float32)

        sum_weight += total_w
        sum_abs += float(np.sum(abs_err * weights))
        sum_sq += float(np.sum(sq_err * weights))
        sum_within += float(np.sum(within * weights))
        points += int(labels_arr.size)

    if points <= 0 or sum_weight <= 0.0:
        return None

    mae = sum_abs / sum_weight
    rmse = math.sqrt(sum_sq / sum_weight)
    interval_coverage = sum_within / sum_weight
    interval_err = abs(interval_coverage - target_coverage)
    return {
        "points": int(points),
        "mae": float(mae),
        "rmse": float(rmse),
        "intervalCoverage": float(interval_coverage),
        "targetCoverage": float(target_coverage),
        "intervalCoverageError": float(interval_err),
    }


def champion_gate_decision(
    champion_eval: Optional[Dict[str, object]],
    challenger_eval: Optional[Dict[str, object]],
) -> Dict[str, object]:
    decision = {
        "enabled": CHAMPION_GATE_ENABLED,
        "promote": True,
        "reason": "gate_disabled",
        "minRows": max(1, CHAMPION_GATE_MIN_ROWS),
        "minMaeImprovement": float(CHAMPION_GATE_MIN_MAE_IMPROVEMENT),
        "maxRmseDegrade": float(CHAMPION_GATE_MAX_RMSE_DEGRADE),
        "maxIntervalErrDegrade": float(CHAMPION_GATE_MAX_INTERVAL_ERR_DEGRADE),
        "champion": champion_eval,
        "challenger": challenger_eval,
    }
    if not CHAMPION_GATE_ENABLED:
        return decision
    if champion_eval is None:
        decision["reason"] = "no_champion_eval"
        decision["promote"] = True
        return decision
    if challenger_eval is None:
        decision["reason"] = "no_challenger_eval"
        decision["promote"] = False
        return decision

    champion_points = int(champion_eval.get("points", 0) or 0)
    challenger_points = int(challenger_eval.get("points", 0) or 0)
    if champion_points < CHAMPION_GATE_MIN_ROWS or challenger_points < CHAMPION_GATE_MIN_ROWS:
        decision["reason"] = "insufficient_eval_rows"
        decision["promote"] = True
        return decision

    champion_mae = float(champion_eval.get("mae", float("inf")))
    challenger_mae = float(challenger_eval.get("mae", float("inf")))
    champion_rmse = float(champion_eval.get("rmse", float("inf")))
    challenger_rmse = float(challenger_eval.get("rmse", float("inf")))
    champion_interval_err = float(champion_eval.get("intervalCoverageError", float("inf")))
    challenger_interval_err = float(challenger_eval.get("intervalCoverageError", float("inf")))

    mae_gain = champion_mae - challenger_mae
    rmse_degrade = challenger_rmse - champion_rmse
    interval_err_degrade = challenger_interval_err - champion_interval_err

    decision["maeGain"] = float(mae_gain)
    decision["rmseDegrade"] = float(rmse_degrade)
    decision["intervalErrDegrade"] = float(interval_err_degrade)

    promote = (
        mae_gain >= float(CHAMPION_GATE_MIN_MAE_IMPROVEMENT)
        and rmse_degrade <= float(CHAMPION_GATE_MAX_RMSE_DEGRADE)
        and interval_err_degrade <= float(CHAMPION_GATE_MAX_INTERVAL_ERR_DEGRADE)
    )
    decision["promote"] = bool(promote)
    decision["reason"] = "promote" if promote else "champion_kept"
    return decision


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

    if meta.get("schemaVersion") != MODEL_SCHEMA_VERSION:
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

    if meta.get("schemaVersion") != MODEL_SCHEMA_VERSION:
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
    restored_meta["forceRetrainUntil"] = (now + timedelta(hours=max(1, DRIFT_ACTION_FORCE_HOURS))).isoformat()
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
    ensure_dir(MODEL_ARTIFACT_DIR)
    tmp_p50 = p50_path + ".tmp"
    tmp_meta = meta_path + ".tmp"

    p50 = model_bundle["p50"]
    p50.save_model(tmp_p50)
    with open(tmp_meta, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False)

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
    ensure_dir(MODEL_ARTIFACT_DIR)
    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False)
    os.replace(tmp_meta, meta_path)


def should_retrain_model(
    meta: Optional[Dict[str, object]],
    now: datetime,
    retrain_hours: Optional[int] = None,
) -> bool:
    if FORCE_RETRAIN:
        return True
    if not meta:
        return True

    force_until = parse_iso_datetime(str(meta.get("forceRetrainUntil", "")))
    if bool(meta.get("forceRetrain")):
        if force_until is None:
            return True
        if now <= force_until:
            return True
    elif force_until is not None and now <= force_until:
        return True

    trained_at = parse_iso_datetime(meta.get("trainedAt", ""))
    if not trained_at:
        return True

    effective_hours = max(1, int(retrain_hours if retrain_hours is not None else MODEL_RETRAIN_HOURS))
    return (now - trained_at) >= timedelta(hours=effective_hours)


def passes_guardrail(
    baseline_meta: Optional[Dict[str, object]],
    candidate_metrics: Dict[str, object],
) -> bool:
    if not baseline_meta:
        return True

    baseline_mae = baseline_meta.get("valMae")
    baseline_rows = int(baseline_meta.get("valRows", 0) or 0)
    candidate_mae = candidate_metrics.get("val_mae")
    candidate_rows = int(candidate_metrics.get("val_rows", 0) or 0)

    if baseline_mae is None or candidate_mae is None:
        return True
    if baseline_rows < MODEL_GUARDRAIL_MIN_VAL_ROWS:
        return True
    if candidate_rows < MODEL_GUARDRAIL_MIN_VAL_ROWS:
        return True

    return float(candidate_mae) <= float(baseline_mae) * (1.0 + MODEL_GUARDRAIL_MAX_MAE_DEGRADE)


def prepare_model(
    now: datetime,
    model_key: str,
    facility_id: int,
    category_key: str,
    expected_loc_ids: List[int],
    onehot: Dict[int, List[float]],
    loc_data: Dict[int, Dict[str, object]],
    loc_samples: Dict[int, int],
    weather_series: Optional[Dict[str, object]],
    allow_retrain: bool = True,
    adaptive_controls: Optional[Dict[str, object]] = None,
):
    feature_count = model_feature_count(len(expected_loc_ids))
    saved_bundle, saved_meta = load_saved_model(
        model_key=model_key,
        expected_loc_ids=expected_loc_ids,
        expected_feature_count=feature_count,
    )

    status = "using_saved_model" if saved_bundle is not None else "no_saved_model"
    run_metrics: Dict[str, object] = {}
    adaptive_controls = adaptive_controls or {}
    effective_retrain_hours = max(
        1,
        int(adaptive_controls.get("retrainHours", MODEL_RETRAIN_HOURS)),
    )
    run_metrics["effective_retrain_hours"] = int(effective_retrain_hours)

    if not allow_retrain:
        run_metrics["retrain_blocked_by_quality"] = True
        if saved_bundle is not None:
            status = "using_saved_model_quality_block"
            return saved_bundle, saved_meta, status, run_metrics
        status = "no_model_quality_block"
        return None, None, status, run_metrics

    if should_retrain_model(saved_meta, now, retrain_hours=effective_retrain_hours):
        preferred_params = normalize_tuning_overrides(
            saved_meta.get("bestParams") if isinstance(saved_meta, dict) else None
        )
        candidate_bundle, candidate_metrics, interval_profile = train_model_unit(
            model_key=model_key,
            loc_ids=expected_loc_ids,
            loc_data=loc_data,
            onehot=onehot,
            loc_samples=loc_samples,
            weather_source=weather_series,
            preferred_params=preferred_params,
        )
        run_metrics = candidate_metrics

        if candidate_bundle is None:
            if saved_bundle is not None:
                status = "using_saved_model_train_skipped"
                return saved_bundle, saved_meta, status, run_metrics
            status = "no_model_train_skipped"
            return None, None, status, run_metrics

        candidate_meta = {
            "schemaVersion": MODEL_SCHEMA_VERSION,
            "facilityId": int(facility_id),
            "categoryKey": str(category_key),
            "modelKey": model_key,
            "trainedAt": now.isoformat(),
            "locIds": expected_loc_ids,
            "featureCount": feature_count,
            "trainRows": int(candidate_metrics.get("train_rows", 0)),
            "valRows": int(candidate_metrics.get("val_rows", 0)),
            "valMae": candidate_metrics.get("val_mae"),
            "valRmse": candidate_metrics.get("val_rmse"),
            "residualProfile": interval_profile,
            "quantileDirect": bool(candidate_metrics.get("quantile_direct")),
            "tuningCvMae": candidate_metrics.get("tuning_cv_mae"),
            "tunedCandidates": int(candidate_metrics.get("tuned_candidates", 0)),
            "bestParams": candidate_metrics.get("best_params"),
            "featureAblation": candidate_metrics.get("feature_ablation"),
            "driftAlertStreak": 0,
            "forceRetrain": False,
            "forceRetrainUntil": None,
            "adaptiveControls": {
                "retrainHours": int(effective_retrain_hours),
                "driftRecentDays": int(adaptive_controls.get("driftRecentDays", DRIFT_RECENT_DAYS)),
                "driftAlertMultiplier": float(
                    adaptive_controls.get("driftAlertMultiplier", DRIFT_ALERT_MULTIPLIER)
                ),
                "driftActionStreakForRetrain": int(
                    adaptive_controls.get(
                        "driftActionStreakForRetrain",
                        DRIFT_ACTION_STREAK_FOR_RETRAIN,
                    )
                ),
            },
        }

        if saved_bundle is not None:
            since = now - timedelta(days=max(1, CHAMPION_GATE_RECENT_DAYS))
            champion_eval = evaluate_model_bundle_on_recent_window(
                model_bundle=saved_bundle,
                residual_profile=saved_meta.get("residualProfile") if isinstance(saved_meta, dict) else None,
                loc_ids=expected_loc_ids,
                loc_data=loc_data,
                onehot=onehot,
                weather_source=weather_series,
                since=since,
            )
            challenger_eval = evaluate_model_bundle_on_recent_window(
                model_bundle=candidate_bundle,
                residual_profile=interval_profile,
                loc_ids=expected_loc_ids,
                loc_data=loc_data,
                onehot=onehot,
                weather_source=weather_series,
                since=since,
            )
            gate = champion_gate_decision(champion_eval=champion_eval, challenger_eval=challenger_eval)
            run_metrics["champion_gate"] = gate
            candidate_meta["championGate"] = gate

            if not bool(gate.get("promote")):
                status = "champion_kept_by_gate"
                if isinstance(saved_meta, dict):
                    updated_saved = dict(saved_meta)
                    updated_saved["lastChampionGate"] = gate
                    updated_saved["lastChampionGateAt"] = now.isoformat()
                    try:
                        save_model_meta_only(updated_saved, model_key=model_key)
                    except Exception:
                        traceback.print_exc()
                    saved_meta = updated_saved
                return saved_bundle, saved_meta, status, run_metrics

        if saved_bundle is not None and not passes_guardrail(saved_meta, candidate_metrics):
            status = "guardrail_kept_previous"
            return saved_bundle, saved_meta, status, run_metrics

        backup_written = False
        if saved_bundle is not None:
            backup_written = backup_current_artifacts(model_key=model_key)
        candidate_meta["previousBackupWritten"] = bool(backup_written)

        save_model_artifacts(
            candidate_bundle,
            candidate_meta,
            model_key=model_key,
        )
        status = "trained_and_saved"
        return candidate_bundle, candidate_meta, status, run_metrics

    return saved_bundle, saved_meta, status, run_metrics


def prepare_models(
    now: datetime,
    loc_data: Dict[int, Dict[str, object]],
    loc_samples: Dict[int, int],
    weather_series: Optional[Dict[str, object]],
    allow_retrain: bool = True,
    adaptive_controls: Optional[Dict[str, object]] = None,
):
    models_by_key: Dict[str, Dict[str, object]] = {}
    model_meta_by_key: Dict[str, Dict[str, object]] = {}
    model_status_by_key: Dict[str, str] = {}
    run_metrics_by_key: Dict[str, Dict[str, object]] = {}
    onehot_by_key: Dict[str, Dict[int, List[float]]] = {}
    unit_loc_ids: Dict[str, List[int]] = {}
    loc_to_model_key: Dict[int, str] = {}
    loc_to_fallback_key: Dict[int, str] = {}
    unit_specs: List[Dict[str, object]] = []

    for facility_id, facility in FACILITIES.items():
        all_loc_ids = facility_location_ids(facility)
        all_key = model_unit_key(facility_id, "__all__")
        all_onehot = build_onehot(all_loc_ids)
        onehot_by_key[all_key] = all_onehot
        unit_loc_ids[all_key] = all_loc_ids
        for loc_id in all_loc_ids:
            loc_to_fallback_key[loc_id] = all_key
        unit_specs.append(
            {
                "model_key": all_key,
                "facility_id": int(facility_id),
                "category_key": "__all__",
                "loc_ids": all_loc_ids,
            }
        )

        for category in facility.get("categories", []):
            category_key = str(category.get("key"))
            if not should_train_category(category_key):
                continue
            category_loc_ids = sorted(set(int(loc_id) for loc_id in category.get("location_ids", [])))
            if not category_loc_ids:
                continue

            key = model_unit_key(facility_id, category_key)
            onehot = build_onehot(category_loc_ids)
            onehot_by_key[key] = onehot
            unit_loc_ids[key] = category_loc_ids
            unit_specs.append(
                {
                    "model_key": key,
                    "facility_id": int(facility_id),
                    "category_key": category_key,
                    "loc_ids": category_loc_ids,
                }
            )

            for loc_id in category_loc_ids:
                loc_to_model_key[loc_id] = key

    def run_unit(spec: Dict[str, object]):
        key = str(spec["model_key"])
        try:
            model_bundle, meta, status, run_metrics = prepare_model(
                now=now,
                model_key=key,
                facility_id=int(spec["facility_id"]),
                category_key=str(spec["category_key"]),
                expected_loc_ids=list(spec["loc_ids"]),
                onehot=onehot_by_key[key],
                loc_data=loc_data,
                loc_samples=loc_samples,
                weather_series=weather_series,
                allow_retrain=allow_retrain,
                adaptive_controls=adaptive_controls,
            )
            return key, model_bundle, meta, status, run_metrics
        except Exception as exc:
            traceback.print_exc()
            return key, None, None, "error", {"error": str(exc)}

    use_parallel = max(1, int(MODEL_PARALLEL_WORKERS)) > 1 and len(unit_specs) > 1
    if use_parallel:
        max_workers = min(len(unit_specs), max(1, int(MODEL_PARALLEL_WORKERS)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_unit, spec) for spec in unit_specs]
            for fut in as_completed(futures):
                key, model_bundle, meta, status, run_metrics = fut.result()
                if model_bundle is not None:
                    models_by_key[key] = model_bundle
                if meta is not None:
                    model_meta_by_key[key] = meta
                model_status_by_key[key] = status
                run_metrics_by_key[key] = run_metrics
    else:
        for spec in unit_specs:
            key, model_bundle, meta, status, run_metrics = run_unit(spec)
            if model_bundle is not None:
                models_by_key[key] = model_bundle
            if meta is not None:
                model_meta_by_key[key] = meta
            model_status_by_key[key] = status
            run_metrics_by_key[key] = run_metrics

    return (
        models_by_key,
        model_meta_by_key,
        model_status_by_key,
        run_metrics_by_key,
        onehot_by_key,
        loc_to_model_key,
        loc_to_fallback_key,
        unit_loc_ids,
    )


def summarize_model_results(
    model_meta_by_key: Dict[str, Dict[str, object]],
    model_status_by_key: Dict[str, str],
    run_metrics_by_key: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    statuses = [status for status in model_status_by_key.values() if status]
    aggregate_status = statuses[0] if statuses and len(set(statuses)) == 1 else "mixed"

    total_train_rows = 0
    total_val_rows = 0
    weighted_mae_sum = 0.0
    weighted_rmse_sum = 0.0
    weighted_mae_rows = 0
    weighted_rmse_rows = 0
    trained_at_latest: Optional[datetime] = None
    by_facility: Dict[str, Dict[str, object]] = {}
    by_model: Dict[str, Dict[str, object]] = {}

    for facility_id, facility in FACILITIES.items():
        all_key = model_unit_key(facility_id, "__all__")
        meta = model_meta_by_key.get(all_key)
        run_metrics = run_metrics_by_key.get(all_key, {})
        status = model_status_by_key.get(all_key)

        train_rows = int((meta.get("trainRows") if meta else run_metrics.get("train_rows")) or 0)
        val_rows = int((meta.get("valRows") if meta else run_metrics.get("val_rows")) or 0)
        val_mae = meta.get("valMae") if meta else run_metrics.get("val_mae")
        val_rmse = meta.get("valRmse") if meta else run_metrics.get("val_rmse")
        trained_at = meta.get("trainedAt") if meta else None

        total_train_rows += max(0, train_rows)
        total_val_rows += max(0, val_rows)

        if val_rows > 0 and val_mae is not None:
            weighted_mae_sum += float(val_mae) * float(val_rows)
            weighted_mae_rows += int(val_rows)
        if val_rows > 0 and val_rmse is not None:
            weighted_rmse_sum += float(val_rmse) * float(val_rows)
            weighted_rmse_rows += int(val_rows)

        parsed_trained_at = parse_iso_datetime(str(trained_at)) if trained_at else None
        if parsed_trained_at and (trained_at_latest is None or parsed_trained_at > trained_at_latest):
            trained_at_latest = parsed_trained_at

        by_facility[str(facility_id)] = {
            "facilityName": facility.get("name"),
            "status": status,
            "trainedAt": trained_at,
            "trainRows": train_rows,
            "valRows": val_rows,
            "valMae": val_mae,
            "valRmse": val_rmse,
            "quantileDirect": bool(meta.get("quantileDirect")) if meta else False,
        }

    for key in sorted(set(model_status_by_key.keys()) | set(model_meta_by_key.keys())):
        meta = model_meta_by_key.get(key)
        run_metrics = run_metrics_by_key.get(key, {})
        by_model[key] = {
            "status": model_status_by_key.get(key),
            "trainedAt": meta.get("trainedAt") if meta else None,
            "trainRows": int((meta.get("trainRows") if meta else run_metrics.get("train_rows")) or 0),
            "valRows": int((meta.get("valRows") if meta else run_metrics.get("val_rows")) or 0),
            "valMae": meta.get("valMae") if meta else run_metrics.get("val_mae"),
            "valRmse": meta.get("valRmse") if meta else run_metrics.get("val_rmse"),
            "quantileDirect": bool(meta.get("quantileDirect")) if meta else False,
            "bestParams": (meta.get("bestParams") if meta else run_metrics.get("best_params")),
            "featureAblation": (meta.get("featureAblation") if meta else run_metrics.get("feature_ablation")),
        }

    val_mae = (weighted_mae_sum / float(weighted_mae_rows)) if weighted_mae_rows > 0 else None
    val_rmse = (weighted_rmse_sum / float(weighted_rmse_rows)) if weighted_rmse_rows > 0 else None

    return {
        "status": aggregate_status,
        "trainedAt": trained_at_latest.isoformat() if trained_at_latest else None,
        "trainRows": total_train_rows,
        "valRows": total_val_rows,
        "valMae": val_mae,
        "valRmse": val_rmse,
        "byFacility": by_facility,
        "byModel": by_model,
    }


def compute_model_drift(
    now: datetime,
    models_by_key: Dict[str, Dict[str, object]],
    onehot_by_key: Dict[str, Dict[int, List[float]]],
    unit_loc_ids: Dict[str, List[int]],
    loc_data: Dict[int, Dict[str, object]],
    loc_samples: Dict[int, int],
    weather_series: Optional[Dict[str, object]],
    model_meta_by_key: Dict[str, Dict[str, object]],
    drift_recent_days: Optional[int] = None,
    drift_alert_multiplier: Optional[float] = None,
) -> Dict[str, object]:
    recent_days = max(1, int(drift_recent_days if drift_recent_days is not None else DRIFT_RECENT_DAYS))
    alert_multiplier = float(
        drift_alert_multiplier if drift_alert_multiplier is not None else DRIFT_ALERT_MULTIPLIER
    )
    summary = {
        "enabled": DRIFT_ENABLED,
        "recentDays": recent_days,
        "minPoints": max(1, DRIFT_MIN_POINTS),
        "alertMultiplier": alert_multiplier,
        "modelsEvaluated": 0,
        "modelsAlerting": 0,
        "byModel": {},
    }
    if not DRIFT_ENABLED:
        return summary

    since = now - timedelta(days=recent_days)
    weather_lookup_cache: Dict[Tuple[datetime, str], float] = {}

    for model_key, bundle in models_by_key.items():
        p50_model = bundle.get("p50")
        if p50_model is None:
            continue
        loc_ids = unit_loc_ids.get(model_key) or []
        if not loc_ids:
            continue

        abs_errors: List[float] = []
        points = 0
        for loc_id in loc_ids:
            data = loc_data.get(loc_id)
            if not data:
                continue
            if loc_samples.get(loc_id, 0) < MIN_SAMPLES_PER_LOC:
                continue
            if data.get("is_stale"):
                continue

            onehot_vec = onehot_by_key.get(model_key, {}).get(loc_id)
            if onehot_vec is None:
                continue

            features_rows: List[List[float]] = []
            labels: List[float] = []
            for ts, label in zip(data.get("bucket_times", []), data.get("bucket_values", [])):
                if ts < since:
                    continue
                features_rows.append(
                    build_features(
                        ts,
                        data,
                        onehot_vec,
                        weather_source=weather_series,
                        weather_lookup_cache=weather_lookup_cache,
                    )
                )
                labels.append(float(label))

            if not features_rows:
                continue

            preds = p50_model.predict(xgb.DMatrix(np.array(features_rows, dtype=np.float32)))
            errors = np.abs(preds - np.array(labels, dtype=np.float32))
            abs_errors.extend(float(v) for v in errors.tolist())
            points += len(labels)

        if points < max(1, DRIFT_MIN_POINTS):
            continue

        recent_mae = float(sum(abs_errors) / max(1, len(abs_errors)))
        baseline_mae_raw = (model_meta_by_key.get(model_key) or {}).get("valMae")
        baseline_mae = float(baseline_mae_raw) if baseline_mae_raw is not None else None
        threshold = (
            baseline_mae * (1.0 + alert_multiplier)
            if baseline_mae is not None
            else None
        )
        alert = bool(
            threshold is not None
            and recent_mae > threshold
        )

        summary["modelsEvaluated"] += 1
        if alert:
            summary["modelsAlerting"] += 1

        summary["byModel"][model_key] = {
            "points": points,
            "recentMae": recent_mae,
            "baselineMae": baseline_mae,
            "alertThresholdMae": threshold,
            "alert": alert,
        }

    return summary


def drift_interval_multiplier(streak: int) -> float:
    step = max(0.0, float(DRIFT_ACTION_INTERVAL_MULTIPLIER_STEP))
    max_mult = max(1.0, float(DRIFT_ACTION_INTERVAL_MAX_MULTIPLIER))
    if step <= 0.0:
        return 1.0
    return max(1.0, min(max_mult, 1.0 + float(max(0, int(streak))) * step))


def apply_drift_actions(
    now: datetime,
    drift_summary: Dict[str, object],
    model_meta_by_key: Dict[str, Dict[str, object]],
    action_streak_for_retrain: Optional[int] = None,
    action_force_hours: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    trigger_streak = max(
        1,
        int(action_streak_for_retrain if action_streak_for_retrain is not None else DRIFT_ACTION_STREAK_FOR_RETRAIN),
    )
    force_hours = max(
        1,
        int(action_force_hours if action_force_hours is not None else DRIFT_ACTION_FORCE_HOURS),
    )
    summary = {
        "enabled": DRIFT_ACTIONS_ENABLED,
        "triggerStreak": trigger_streak,
        "forceHours": force_hours,
        "intervalStep": max(0.0, DRIFT_ACTION_INTERVAL_MULTIPLIER_STEP),
        "intervalMax": max(1.0, DRIFT_ACTION_INTERVAL_MAX_MULTIPLIER),
        "modelsEvaluated": 0,
        "modelsForcedRetrain": 0,
        "modelsRolledBack": 0,
        "byModel": {},
    }
    interval_multiplier_by_key: Dict[str, float] = {}
    if not DRIFT_ACTIONS_ENABLED:
        return interval_multiplier_by_key, summary

    drift_by_model = drift_summary.get("byModel", {}) if isinstance(drift_summary, dict) else {}
    if not isinstance(drift_by_model, dict):
        drift_by_model = {}

    all_model_keys = sorted(set(model_meta_by_key.keys()) | set(drift_by_model.keys()))
    for model_key in all_model_keys:
        meta = model_meta_by_key.get(model_key)
        if not isinstance(meta, dict):
            continue

        evaluated = model_key in drift_by_model
        if not evaluated:
            prev_streak = int(meta.get("driftAlertStreak", 0) or 0)
            prev_force = bool(meta.get("forceRetrain"))
            multiplier = drift_interval_multiplier(prev_streak) if prev_force else 1.0
            interval_multiplier_by_key[model_key] = multiplier
            summary["byModel"][model_key] = {
                "evaluated": False,
                "alert": None,
                "streak": int(prev_streak),
                "forceRetrain": prev_force,
                "forceRetrainUntil": meta.get("forceRetrainUntil"),
                "intervalMultiplier": round(float(multiplier), 4),
                "rolledBack": False,
            }
            continue

        drift_row = drift_by_model.get(model_key, {})
        alert = bool(drift_row.get("alert")) if isinstance(drift_row, dict) else False

        prev_streak = int(meta.get("driftAlertStreak", 0) or 0)
        streak = prev_streak + 1 if alert else 0
        multiplier = drift_interval_multiplier(streak)
        interval_multiplier_by_key[model_key] = multiplier

        history = list(meta.get("driftHistory") or [])
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "at": now.isoformat(),
                "alert": bool(alert),
                "recentMae": drift_row.get("recentMae"),
                "baselineMae": drift_row.get("baselineMae"),
                "alertThresholdMae": drift_row.get("alertThresholdMae"),
            }
        )
        history = history[-max(10, int(ADAPTIVE_HISTORY_MAX_POINTS)) :]

        rolled_back = False
        if CHAMPION_ROLLBACK_ENABLED and alert and streak >= max(1, CHAMPION_ROLLBACK_DRIFT_STREAK):
            loc_ids_raw = meta.get("locIds") or []
            loc_ids = []
            if isinstance(loc_ids_raw, list):
                for loc_id in loc_ids_raw:
                    try:
                        loc_ids.append(int(loc_id))
                    except Exception:
                        continue
            feature_count = int(meta.get("featureCount", -1) or -1)
            if loc_ids and feature_count > 0:
                restored_meta = rollback_to_previous_model(
                    model_key=model_key,
                    expected_loc_ids=loc_ids,
                    expected_feature_count=feature_count,
                    now=now,
                )
                if restored_meta is not None:
                    restored_meta["driftHistory"] = history
                    restored_meta["lastRollbackReason"] = "drift_streak"
                    restored_meta["lastRollbackSource"] = "previous_artifact"
                    try:
                        save_model_meta_only(restored_meta, model_key=model_key)
                    except Exception:
                        traceback.print_exc()
                    model_meta_by_key[model_key] = restored_meta
                    interval_multiplier_by_key[model_key] = 1.0
                    summary["modelsEvaluated"] += 1
                    summary["modelsRolledBack"] += 1
                    summary["byModel"][model_key] = {
                        "evaluated": True,
                        "alert": alert,
                        "streak": int(streak),
                        "forceRetrain": True,
                        "forceRetrainUntil": restored_meta.get("forceRetrainUntil"),
                        "intervalMultiplier": 1.0,
                        "rolledBack": True,
                    }
                    rolled_back = True

        if rolled_back:
            continue

        force_retrain = False
        force_until_iso = None
        if streak >= trigger_streak:
            force_retrain = True
            force_until_iso = (now + timedelta(hours=force_hours)).isoformat()
            summary["modelsForcedRetrain"] += 1
        else:
            prev_force_until = parse_iso_datetime(str(meta.get("forceRetrainUntil", "")))
            if bool(meta.get("forceRetrain")) and prev_force_until and now <= prev_force_until:
                force_retrain = True
                force_until_iso = prev_force_until.isoformat()

        updated = dict(meta)
        updated["driftAlertStreak"] = int(streak)
        updated["lastDriftEvaluatedAt"] = now.isoformat()
        if alert:
            updated["lastDriftAlertAt"] = now.isoformat()
        updated["forceRetrain"] = bool(force_retrain)
        updated["forceRetrainUntil"] = force_until_iso
        updated["driftHistory"] = history

        model_meta_by_key[model_key] = updated
        try:
            save_model_meta_only(updated, model_key=model_key)
        except Exception:
            traceback.print_exc()

        summary["modelsEvaluated"] += 1
        summary["byModel"][model_key] = {
            "evaluated": True,
            "alert": alert,
            "streak": int(streak),
            "forceRetrain": bool(force_retrain),
            "forceRetrainUntil": force_until_iso,
            "intervalMultiplier": round(float(multiplier), 4),
            "rolledBack": False,
        }

    return interval_multiplier_by_key, summary


def conformal_margin_from_scores(scores: np.ndarray) -> float:
    if scores.size == 0:
        return 0.0
    alpha = max(0.01, min(0.5, float(INTERVAL_CONFORMAL_ALPHA)))
    quantile = max(0.0, min(1.0, 1.0 - alpha))
    margin = float(np.quantile(scores, quantile))
    return max(0.0, min(float(INTERVAL_CONFORMAL_MAX_MARGIN), margin))


def hour_block_key(hour: int) -> str:
    hour = int(hour) % 24
    if 0 <= hour < 6:
        return "overnight"
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "midday"
    if 17 <= hour < 21:
        return "evening"
    return "late"


def conformal_margin_for_hour(
    hour: int,
    profile: Optional[Dict[str, object]],
) -> float:
    if not profile:
        return 0.0

    by_hour = profile.get("byHour", {})
    hour_stats = by_hour.get(str(int(hour))) if isinstance(by_hour, dict) else None
    if hour_stats and int(hour_stats.get("count", 0) or 0) >= INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR:
        return max(0.0, float(hour_stats.get("margin", 0.0) or 0.0))

    by_block = profile.get("byHourBlock", {})
    block_stats = by_block.get(hour_block_key(hour)) if isinstance(by_block, dict) else None
    if block_stats and int(block_stats.get("count", 0) or 0) >= INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR:
        return max(0.0, float(block_stats.get("margin", 0.0) or 0.0))

    global_stats = profile.get("global", {})
    return max(0.0, float(global_stats.get("margin", 0.0) or 0.0))


def compute_interval_conformal_profiles(
    now: datetime,
    models_by_key: Dict[str, Dict[str, object]],
    onehot_by_key: Dict[str, Dict[int, List[float]]],
    unit_loc_ids: Dict[str, List[int]],
    loc_data: Dict[int, Dict[str, object]],
    loc_samples: Dict[int, int],
    weather_series: Optional[Dict[str, object]],
    interval_profile_by_key: Dict[str, Optional[Dict[str, object]]],
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    summary = {
        "enabled": INTERVAL_CONFORMAL_ENABLED,
        "recentDays": max(1, INTERVAL_CONFORMAL_RECENT_DAYS),
        "alpha": max(0.01, min(0.5, float(INTERVAL_CONFORMAL_ALPHA))),
        "minPoints": max(1, INTERVAL_CONFORMAL_MIN_POINTS),
        "minPointsPerHour": max(1, INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR),
        "maxMargin": float(INTERVAL_CONFORMAL_MAX_MARGIN),
        "modelsCalibrated": 0,
        "byModel": {},
    }
    if not INTERVAL_CONFORMAL_ENABLED:
        return {}, summary

    since = now - timedelta(days=max(1, INTERVAL_CONFORMAL_RECENT_DAYS))
    weather_lookup_cache: Dict[Tuple[datetime, str], float] = {}
    profiles: Dict[str, Dict[str, object]] = {}

    for model_key, bundle in models_by_key.items():
        p50_model = bundle.get("p50")
        if p50_model is None:
            continue

        loc_ids = unit_loc_ids.get(model_key) or []
        if not loc_ids:
            continue

        global_scores: List[float] = []
        by_hour_scores: Dict[int, List[float]] = {}
        by_block_scores: Dict[str, List[float]] = {}
        points = 0

        for loc_id in loc_ids:
            data = loc_data.get(loc_id)
            if not data:
                continue
            if loc_samples.get(loc_id, 0) < MIN_SAMPLES_PER_LOC:
                continue
            if data.get("is_stale"):
                continue

            onehot_vec = onehot_by_key.get(model_key, {}).get(loc_id)
            if onehot_vec is None:
                continue

            features_rows: List[List[float]] = []
            labels: List[float] = []
            hours: List[int] = []
            for ts, label in zip(data.get("bucket_times", []), data.get("bucket_values", [])):
                if ts < since:
                    continue
                features_rows.append(
                    build_features(
                        ts,
                        data,
                        onehot_vec,
                        weather_source=weather_series,
                        weather_lookup_cache=weather_lookup_cache,
                    )
                )
                labels.append(float(label))
                hours.append(int(ts.hour))

            if not features_rows:
                continue

            dmat = xgb.DMatrix(np.array(features_rows, dtype=np.float32))
            p50 = p50_model.predict(dmat)
            p10_model = bundle.get("p10")
            p90_model = bundle.get("p90")

            if p10_model is not None and p90_model is not None:
                p10 = p10_model.predict(dmat)
                p90 = p90_model.predict(dmat)
            else:
                p10_vals = []
                p90_vals = []
                residual_profile = interval_profile_by_key.get(model_key)
                for pred, hour in zip(p50, hours):
                    b10, _mid, b90 = interval_bounds(
                        point_ratio=float(pred),
                        hour=int(hour),
                        residual_profile=residual_profile,
                    )
                    p10_vals.append(float(b10))
                    p90_vals.append(float(b90))
                p10 = np.array(p10_vals, dtype=np.float32)
                p90 = np.array(p90_vals, dtype=np.float32)

            labels_arr = np.array(labels, dtype=np.float32)
            scores = np.maximum.reduce(
                [
                    np.zeros_like(labels_arr),
                    p10 - labels_arr,
                    labels_arr - p90,
                ]
            )

            for idx, score in enumerate(scores.tolist()):
                score_val = float(score)
                hour = int(hours[idx])
                global_scores.append(score_val)
                by_hour_scores.setdefault(hour, []).append(score_val)
                by_block_scores.setdefault(hour_block_key(hour), []).append(score_val)
            points += len(labels)

        if points < max(1, INTERVAL_CONFORMAL_MIN_POINTS) or not global_scores:
            continue

        global_margin = conformal_margin_from_scores(np.array(global_scores, dtype=np.float32))
        by_hour_payload: Dict[str, Dict[str, object]] = {}
        for hour, scores in sorted(by_hour_scores.items()):
            if len(scores) < INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR:
                continue
            by_hour_payload[str(int(hour))] = {
                "margin": conformal_margin_from_scores(np.array(scores, dtype=np.float32)),
                "count": int(len(scores)),
            }

        by_block_payload: Dict[str, Dict[str, object]] = {}
        for block, scores in sorted(by_block_scores.items()):
            if len(scores) < INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR:
                continue
            by_block_payload[str(block)] = {
                "margin": conformal_margin_from_scores(np.array(scores, dtype=np.float32)),
                "count": int(len(scores)),
            }

        profile = {
            "global": {
                "margin": global_margin,
                "count": int(len(global_scores)),
            },
            "byHour": by_hour_payload,
            "byHourBlock": by_block_payload,
        }
        profiles[model_key] = profile
        summary["modelsCalibrated"] += 1
        summary["byModel"][model_key] = {
            "points": points,
            "globalMargin": global_margin,
            "hoursCalibrated": len(by_hour_payload),
            "blocksCalibrated": len(by_block_payload),
        }

    return profiles, summary


def interval_bounds(
    point_ratio: float,
    hour: int,
    residual_profile: Optional[Dict[str, object]],
) -> Tuple[float, float, float]:
    point_ratio = max(0.0, min(point_ratio, 1.0))
    if not residual_profile:
        return point_ratio, point_ratio, point_ratio

    hour_stats = residual_profile.get("byHour", {}).get(str(hour))
    if hour_stats and int(hour_stats.get("count", 0)) >= INTERVAL_MIN_SAMPLES_PER_HOUR:
        q10 = float(hour_stats.get("q10", 0.0))
        q90 = float(hour_stats.get("q90", 0.0))
    else:
        global_stats = residual_profile.get("global", {})
        q10 = float(global_stats.get("q10", 0.0))
        q90 = float(global_stats.get("q90", 0.0))

    p10 = max(0.0, min(point_ratio + q10, 1.0))
    p90 = max(0.0, min(point_ratio + q90, 1.0))
    if p10 > p90:
        p10, p90 = p90, p10
    return p10, point_ratio, p90


def get_sample_count(
    loc_id: int,
    target: datetime,
    avg_dow_hour: Dict[Tuple[int, int, int], Tuple[float, int]],
    avg_hour: Dict[Tuple[int, int], Tuple[float, int]],
    avg_overall: Dict[int, Tuple[float, int]],
) -> int:
    dow = target.weekday()
    hour = target.hour

    value = avg_dow_hour.get((loc_id, dow, hour))
    if value:
        return int(value[1])

    value = avg_hour.get((loc_id, hour))
    if value:
        return int(value[1])

    value = avg_overall.get(loc_id)
    if value:
        return int(value[1])

    return 0


def clamp_ratio(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(float(value), 1.2))


def fallback_ratio_for_location(
    loc_id: int,
    target: datetime,
    avg_dow_hour: Dict[Tuple[int, int, int], Tuple[float, int]],
    avg_hour: Dict[Tuple[int, int], Tuple[float, int]],
    avg_overall: Dict[int, Tuple[float, int]],
) -> float:
    dow = target.weekday()
    hour = target.hour

    value = avg_dow_hour.get((loc_id, dow, hour))
    if value:
        return float(value[0])

    value = avg_hour.get((loc_id, hour))
    if value:
        return float(value[0])

    value = avg_overall.get(loc_id)
    if value:
        return float(value[0])

    return 0.0


def estimate_location(
    loc_id: int,
    target: datetime,
    ctx: Dict[str, object],
) -> Dict[str, float]:
    max_caps = ctx["max_caps"]
    max_cap = max_caps.get(loc_id, 0)
    if max_cap <= 0:
        return {
            "countP10": 0.0,
            "countP50": 0.0,
            "countP90": 0.0,
            "sampleCount": 0.0,
        }

    active_key = None
    loc_to_model_key = ctx.get("loc_to_model_key", {})
    loc_to_fallback_key = ctx.get("loc_to_fallback_key", {})
    models_by_key = ctx.get("models_by_key", {})
    primary_key = loc_to_model_key.get(loc_id)
    fallback_key = loc_to_fallback_key.get(loc_id)
    model_bundle = models_by_key.get(primary_key)
    if model_bundle is not None:
        active_key = primary_key
    elif fallback_key:
        model_bundle = models_by_key.get(fallback_key)
        active_key = fallback_key

    loc_data = ctx["loc_data"]
    loc_samples = ctx["loc_samples"]
    loc_entry = loc_data.get(loc_id)
    onehot_by_key = ctx.get("onehot_by_key", {})
    onehot_vec = onehot_by_key.get(active_key, {}).get(loc_id) if active_key else None
    interval_profile_by_key = ctx.get("interval_profile_by_key", {})
    interval_profile = interval_profile_by_key.get(active_key)
    conformal_by_key = ctx.get("conformal_by_key", {})
    conformal_profile = conformal_by_key.get(active_key)
    interval_multiplier_by_key = ctx.get("interval_multiplier_by_key", {})
    interval_multiplier = float(interval_multiplier_by_key.get(active_key, 1.0) or 1.0)

    prediction_cache = ctx.get("prediction_cache")
    cache_key = (active_key, int(loc_id), target.isoformat())
    if isinstance(prediction_cache, dict):
        cached = prediction_cache.get(cache_key)
        if cached is not None:
            p10_ratio, p50_ratio, p90_ratio = cached
            sample_count = get_sample_count(
                loc_id,
                target,
                ctx["avg_dow_hour"],
                ctx["avg_hour"],
                ctx["avg_overall"],
            )
            return {
                "countP10": clamp_ratio(p10_ratio) * max_cap,
                "countP50": clamp_ratio(p50_ratio) * max_cap,
                "countP90": clamp_ratio(p90_ratio) * max_cap,
                "sampleCount": float(sample_count),
            }

    p10_ratio = None
    p50_ratio = None
    p90_ratio = None

    if (
        model_bundle is not None
        and model_bundle.get("p50") is not None
        and loc_entry is not None
        and onehot_vec is not None
        and loc_samples.get(loc_id, 0) >= MIN_SAMPLES_PER_LOC
        and not loc_entry.get("is_stale")
    ):
        weather_source = ctx.get("weather_series")
        weather_lookup_cache = ctx.get("weather_lookup_cache")
        feature_cache = ctx.get("feature_cache")
        feature_cache_key = (active_key, int(loc_id), target.isoformat())
        feature_vec = None
        if isinstance(feature_cache, dict):
            feature_vec = feature_cache.get(feature_cache_key)
        if feature_vec is None:
            feature_vec = build_features(
                target,
                loc_entry,
                onehot_vec,
                weather_source=weather_source,
                weather_lookup_cache=weather_lookup_cache,
            )
            if isinstance(feature_cache, dict):
                feature_cache[feature_cache_key] = feature_vec

        features_arr = np.array([feature_vec], dtype=np.float32)
        dmatrix = xgb.DMatrix(features_arr)

        p50_model = model_bundle.get("p50")
        p50_ratio = float(p50_model.predict(dmatrix)[0]) if p50_model is not None else None

        p10_model = model_bundle.get("p10")
        p90_model = model_bundle.get("p90")
        if p10_model is not None and p90_model is not None:
            p10_ratio = float(p10_model.predict(dmatrix)[0])
            p90_ratio = float(p90_model.predict(dmatrix)[0])
        else:
            p10_ratio, p50_from_interval, p90_ratio = interval_bounds(
                point_ratio=float(p50_ratio if p50_ratio is not None else 0.0),
                hour=target.hour,
                residual_profile=interval_profile,
            )
            p50_ratio = p50_from_interval

    if p50_ratio is None or math.isnan(p50_ratio):
        fallback = fallback_ratio_for_location(
            loc_id=loc_id,
            target=target,
            avg_dow_hour=ctx["avg_dow_hour"],
            avg_hour=ctx["avg_hour"],
            avg_overall=ctx["avg_overall"],
        )
        p10_ratio, p50_ratio, p90_ratio = interval_bounds(
            point_ratio=float(fallback),
            hour=target.hour,
            residual_profile=interval_profile,
        )

    margin = conformal_margin_for_hour(target.hour, conformal_profile)
    if margin > 0.0:
        p10_ratio = float(p10_ratio if p10_ratio is not None else p50_ratio) - margin
        p90_ratio = float(p90_ratio if p90_ratio is not None else p50_ratio) + margin

    if interval_multiplier > 1.0 and p50_ratio is not None:
        center = float(p50_ratio)
        cur_p10 = float(p10_ratio if p10_ratio is not None else center)
        cur_p90 = float(p90_ratio if p90_ratio is not None else center)
        low_width = max(0.0, center - cur_p10)
        high_width = max(0.0, cur_p90 - center)
        p10_ratio = center - low_width * interval_multiplier
        p90_ratio = center + high_width * interval_multiplier

    p10_ratio = clamp_ratio(float(p10_ratio if p10_ratio is not None else p50_ratio))
    p50_ratio = clamp_ratio(float(p50_ratio))
    p90_ratio = clamp_ratio(float(p90_ratio if p90_ratio is not None else p50_ratio))
    if p10_ratio > p90_ratio:
        p10_ratio, p90_ratio = p90_ratio, p10_ratio
    p50_ratio = min(max(p50_ratio, p10_ratio), p90_ratio)

    if isinstance(prediction_cache, dict):
        prediction_cache[cache_key] = (p10_ratio, p50_ratio, p90_ratio)

    sample_count = get_sample_count(
        loc_id,
        target,
        ctx["avg_dow_hour"],
        ctx["avg_hour"],
        ctx["avg_overall"],
    )

    return {
        "countP10": clamp_ratio(p10_ratio) * max_cap,
        "countP50": clamp_ratio(p50_ratio) * max_cap,
        "countP90": clamp_ratio(p90_ratio) * max_cap,
        "sampleCount": float(sample_count),
    }


def sum_max_caps(max_caps: Dict[int, int], loc_ids: Iterable[int]) -> int:
    return sum(max_caps.get(loc_id, 0) for loc_id in loc_ids)


def round_count(value: float) -> int:
    return max(0, int(round(value)))


def to_hour_payload(
    target: datetime,
    sum_p10: float,
    sum_p50: float,
    sum_p90: float,
    category_max: int,
    samples: int,
) -> Dict[str, object]:
    expected_p10 = round_count(sum_p10)
    expected = round_count(sum_p50)
    expected_p90 = round_count(sum_p90)

    pct = round(min(expected / category_max, 1.0), 4) if category_max else None
    pct_p10 = round(min(expected_p10 / category_max, 1.0), 4) if category_max else None
    pct_p90 = round(min(expected_p90 / category_max, 1.0), 4) if category_max else None

    return {
        "hour": target.hour,
        "hourStart": target.isoformat(),
        "expectedCount": expected,
        "expectedPct": pct,
        "expectedCountP10": expected_p10,
        "expectedCountP90": expected_p90,
        "expectedPctP10": pct_p10,
        "expectedPctP90": pct_p90,
        "sampleCount": samples,
    }


def safe_parse_hour_start(value: object) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def latest_live_count_for_location(loc_entry: Optional[Dict[str, object]], now: datetime) -> Optional[float]:
    if not loc_entry:
        return None
    raw_times = loc_entry.get("raw_times") or []
    raw_values = loc_entry.get("raw_values") or []
    if not raw_times or not raw_values:
        return None

    ts = raw_times[-1]
    value = raw_values[-1]
    if ts is None or value is None:
        return None

    age_min = (now - ts).total_seconds() / 60.0
    if age_min < 0:
        age_min = 0.0
    if age_min > SPIKE_AWARE_MAX_AGE_MIN:
        return None

    return max(0.0, float(value))


def category_live_total(
    loc_ids: Iterable[int],
    loc_data: Dict[int, Dict[str, object]],
    now: datetime,
) -> Tuple[float, int]:
    total = 0.0
    observed_locs = 0

    for loc_id in loc_ids:
        live = latest_live_count_for_location(loc_data.get(loc_id), now)
        if live is None:
            continue
        total += live
        observed_locs += 1

    return total, observed_locs


def max_allowed_category_count(category_max: int) -> float:
    if not category_max:
        return float("inf")
    return max(float(category_max), float(category_max) * SPIKE_AWARE_MAX_CAP_MULTIPLIER)


def spike_weight(hours_ahead: int) -> float:
    if hours_ahead <= 0:
        return 1.0
    return math.exp(-SPIKE_AWARE_DECAY * max(0, hours_ahead - 1))


def apply_spike_adjustment_to_category_hours(
    day_hours: List[Dict[str, object]],
    category_max: int,
    live_total: float,
    now: datetime,
) -> bool:
    if not SPIKE_AWARE_ENABLED or not day_hours:
        return False

    parsed_starts = [safe_parse_hour_start(item.get("hourStart")) for item in day_hours]
    current_idx = None
    for idx, dt in enumerate(parsed_starts):
        if dt is None:
            continue
        if dt <= now:
            current_idx = idx
        else:
            break

    if current_idx is None:
        return False

    baseline_now = float(day_hours[current_idx].get("expectedCount", 0))
    drift = float(live_total) - baseline_now
    if abs(drift) < 1.0:
        return False

    max_allowed = max_allowed_category_count(category_max)
    adjusted = False

    for idx in range(current_idx + 1, len(day_hours)):
        dt = parsed_starts[idx]
        if dt is None:
            continue
        hours_ahead = idx - current_idx
        if hours_ahead > SPIKE_AWARE_HORIZON_HOURS:
            break

        weight = spike_weight(hours_ahead)
        delta = drift * weight
        if abs(delta) < 0.5:
            continue

        point = day_hours[idx]
        raw_mid = float(point.get("expectedCount", 0))
        raw_p10 = float(point.get("expectedCountP10", raw_mid))
        raw_p90 = float(point.get("expectedCountP90", raw_mid))

        new_mid = max(0.0, min(raw_mid + delta, max_allowed))
        new_p10 = max(0.0, min(raw_p10 + delta, max_allowed))
        new_p90 = max(0.0, min(raw_p90 + delta, max_allowed))
        if new_p10 > new_p90:
            new_p10, new_p90 = new_p90, new_p10
        new_mid = min(max(new_mid, new_p10), new_p90)

        expected = round_count(new_mid)
        expected_p10 = round_count(new_p10)
        expected_p90 = round_count(new_p90)

        point["expectedCountRaw"] = int(point.get("expectedCount", 0))
        point["expectedCount"] = expected
        point["expectedCountP10"] = expected_p10
        point["expectedCountP90"] = expected_p90
        point["spikeAdjusted"] = True

        if category_max:
            point["expectedPct"] = round(min(expected / category_max, 1.0), 4)
            point["expectedPctP10"] = round(min(expected_p10 / category_max, 1.0), 4)
            point["expectedPctP90"] = round(min(expected_p90 / category_max, 1.0), 4)
        else:
            point["expectedPct"] = None
            point["expectedPctP10"] = None
            point["expectedPctP90"] = None

        adjusted = True

    return adjusted


def build_total_series_for_targets(
    loc_ids: Iterable[int],
    targets: List[datetime],
    ctx: Dict[str, object],
) -> List[Dict[str, object]]:
    rows = []
    now = ctx.get("now")

    for target in targets:
        sum_p10 = 0.0
        sum_p50 = 0.0
        sum_p90 = 0.0
        samples = 0

        for loc_id in loc_ids:
            result = estimate_location(
                loc_id,
                target,
                ctx,
            )
            sum_p10 += result["countP10"]
            sum_p50 += result["countP50"]
            sum_p90 += result["countP90"]
            samples += int(result["sampleCount"])

        rows.append(
            {
                "hour": target.hour,
                "hourStart": target.isoformat(),
                "expectedTotal": round_count(sum_p50),
                "expectedTotalP10": round_count(sum_p10),
                "expectedTotalP50": round_count(sum_p50),
                "expectedTotalP90": round_count(sum_p90),
                "sampleCount": samples,
                "isFuture": bool(now and target > now),
            }
        )

    return rows


def build_category_hours_for_targets(
    loc_ids: Iterable[int],
    targets: List[datetime],
    category_max: int,
    ctx: Dict[str, object],
) -> List[Dict[str, object]]:
    outputs = []

    for target in targets:
        sum_p10 = 0.0
        sum_p50 = 0.0
        sum_p90 = 0.0
        samples = 0

        for loc_id in loc_ids:
            result = estimate_location(
                loc_id,
                target,
                ctx,
            )
            sum_p10 += result["countP10"]
            sum_p50 += result["countP50"]
            sum_p90 += result["countP90"]
            samples += int(result["sampleCount"])

        outputs.append(
            to_hour_payload(
                target=target,
                sum_p10=sum_p10,
                sum_p50=sum_p50,
                sum_p90=sum_p90,
                category_max=category_max,
                samples=samples,
            )
        )

    return outputs


def get_targets_for_date(day_date: date) -> List[datetime]:
    start_hour = max(0, min(23, FORECAST_DAY_START_HOUR))
    end_hour = max(0, min(23, FORECAST_DAY_END_HOUR))
    if end_hour < start_hour:
        start_hour, end_hour = end_hour, start_hour

    targets = []
    for hour in range(start_hour, end_hour + 1):
        naive = datetime(day_date.year, day_date.month, day_date.day, hour, 0, 0)
        targets.append(TZ.localize(naive))
    return targets


def get_window_targets_for_date(day_date: date) -> List[datetime]:
    start_hour = max(0, min(23, FORECAST_DAY_START_HOUR))
    end_hour = max(0, min(23, FORECAST_DAY_END_HOUR))
    if end_hour < start_hour:
        start_hour, end_hour = end_hour, start_hour

    start_dt = TZ.localize(datetime(day_date.year, day_date.month, day_date.day, start_hour, 0, 0))
    end_exclusive = TZ.localize(
        datetime(day_date.year, day_date.month, day_date.day, end_hour, 0, 0)
    ) + timedelta(hours=1)

    step = timedelta(minutes=max(1, WINDOW_RESAMPLE_MINUTES))
    targets = []
    current = start_dt
    while current < end_exclusive:
        targets.append(current)
        current += step
    return targets


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
            parsed = TZ.localize(parsed)
        else:
            parsed = parsed.astimezone(TZ)

        if min_dt is not None and parsed < min_dt:
            continue
        if max_dt is not None and parsed > max_dt:
            continue

        row: Dict[str, float] = {}
        for key, api_key in WEATHER_API_HOURLY_MAP.items():
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


def weather_history_start(loc_data: Dict[int, Dict[str, object]], now: datetime) -> datetime:
    floor_start = now - timedelta(days=max(1, WEATHER_HISTORY_MAX_DAYS))
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
        "latitude": WEATHER_LAT,
        "longitude": WEATHER_LON,
        "hourly": ",".join(WEATHER_API_HOURLY_MAP.values()),
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "timezone": TZ_NAME,
        "start_date": start_dt.date().isoformat(),
        "end_date": end_dt.date().isoformat(),
    }

    try:
        resp = requests.get(WEATHER_ARCHIVE_URL, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        return parse_weather_hourly_payload(
            payload,
            min_dt=start_dt,
            max_dt=end_dt + timedelta(hours=3),
        )
    except Exception:
        return {"times": [], "map": {}}


def fetch_weather_forecast_series(now: datetime) -> Dict[str, object]:
    params = {
        "latitude": WEATHER_LAT,
        "longitude": WEATHER_LON,
        "hourly": ",".join(WEATHER_API_HOURLY_MAP.values()),
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "timezone": TZ_NAME,
        "forecast_days": max(1, WEATHER_FORECAST_DAYS),
        "past_days": 2,
    }

    try:
        resp = requests.get(WEATHER_URL, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        return parse_weather_hourly_payload(payload, min_dt=now - timedelta(hours=12))
    except Exception:
        return {"times": [], "map": {}}


def build_weather_hours_for_targets(
    targets: List[datetime],
    weather_series: Dict[str, object],
) -> List[Dict[str, object]]:
    times = weather_series.get("times", [])
    weather_map = weather_series.get("map", {})
    rows = []

    for target in targets:
        def value(key: str):
            val = weather_value_at_or_before(times, weather_map, target, key)
            if math.isnan(val):
                return None
            if key == "weather_code":
                return int(round(val))
            return round(float(val), 2)

        rows.append(
            {
                "hour": target.hour,
                "hourStart": target.isoformat(),
                "tempC": value("temp_c"),
                "feelsLikeC": value("feels_like_c"),
                "precipMm": value("precip_mm"),
                "rainMm": value("rain_mm"),
                "snowCm": value("snow_cm"),
                "windMps": value("wind_mps"),
                "windGustMps": value("wind_gust_mps"),
                "humidityPct": value("humidity_pct"),
                "weatherCode": value("weather_code"),
            }
        )

    return rows


def build_weather_day_summary(weather_hours: List[Dict[str, object]]) -> Dict[str, object]:
    temps = [row["tempC"] for row in weather_hours if row.get("tempC") is not None]
    feels = [row["feelsLikeC"] for row in weather_hours if row.get("feelsLikeC") is not None]
    precip = [row["precipMm"] for row in weather_hours if row.get("precipMm") is not None]
    rain = [row["rainMm"] for row in weather_hours if row.get("rainMm") is not None]
    snow = [row["snowCm"] for row in weather_hours if row.get("snowCm") is not None]
    wind = [row["windMps"] for row in weather_hours if row.get("windMps") is not None]
    gust = [row["windGustMps"] for row in weather_hours if row.get("windGustMps") is not None]
    humidity = [row["humidityPct"] for row in weather_hours if row.get("humidityPct") is not None]
    weather_codes = [row["weatherCode"] for row in weather_hours if row.get("weatherCode") is not None]

    return {
        "avgTempC": round(sum(temps) / len(temps), 2) if temps else None,
        "avgFeelsLikeC": round(sum(feels) / len(feels), 2) if feels else None,
        "totalPrecipMm": round(sum(precip), 2) if precip else None,
        "totalRainMm": round(sum(rain), 2) if rain else None,
        "totalSnowCm": round(sum(snow), 2) if snow else None,
        "maxWindMps": round(max(wind), 2) if wind else None,
        "maxWindGustMps": round(max(gust), 2) if gust else None,
        "avgHumidityPct": round(sum(humidity) / len(humidity), 2) if humidity else None,
        "weatherCodes": sorted(set(int(code) for code in weather_codes)) if weather_codes else [],
    }


def normalize_series(series: List[Dict[str, object]]) -> List[Tuple[datetime, int, int]]:
    entries: List[Tuple[datetime, int, int]] = []
    for item in series:
        hour_start = item.get("hourStart")
        if not hour_start:
            continue
        entries.append(
            (
                datetime.fromisoformat(hour_start),
                int(item.get("expectedTotal", 0)),
                int(item.get("sampleCount", 0)),
            )
        )
    entries.sort(key=lambda row: row[0])
    return entries


def infer_series_step_minutes(series: List[Dict[str, object]]) -> int:
    if len(series) < 2:
        return max(1, RESAMPLE_MINUTES)

    times: List[datetime] = []
    for item in series:
        raw = item.get("hourStart")
        if not raw:
            continue
        try:
            times.append(datetime.fromisoformat(str(raw)))
        except Exception:
            continue

    if len(times) < 2:
        return max(1, RESAMPLE_MINUTES)

    diffs = []
    for idx in range(len(times) - 1):
        diff_min = int(round((times[idx + 1] - times[idx]).total_seconds() / 60.0))
        if diff_min > 0:
            diffs.append(diff_min)

    if not diffs:
        return max(1, RESAMPLE_MINUTES)
    return max(1, min(diffs))


def compute_range_metrics(
    series_entries: List[Tuple[datetime, int, int]],
    start: datetime,
    end: datetime,
) -> Tuple[int, float, int, int]:
    total_sum = 0
    hours = 0
    min_samples = None

    for dt, expected, samples in series_entries:
        if start <= dt < end:
            total_sum += expected
            hours += 1
            min_samples = samples if min_samples is None else min(min_samples, samples)

    if hours == 0:
        hours = max(1, int(round((end - start).total_seconds() / 3600)))

    return total_sum, round(total_sum / hours, 2), min_samples or 0, hours


def merge_windows_with_series(
    windows: List[Dict[str, object]],
    series: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    if not windows:
        return []

    ranges = []
    for window in windows:
        if not window or not window.get("start") or not window.get("end"):
            continue
        ranges.append(
            (
                datetime.fromisoformat(window["start"]),
                datetime.fromisoformat(window["end"]),
            )
        )

    if not ranges:
        return []

    ranges.sort(key=lambda row: row[0])
    merged_ranges = []
    cur_start, cur_end = ranges[0]
    allowed_gap = timedelta(minutes=WINDOW_MERGE_GAP_MIN)

    for start, end in ranges[1:]:
        if start <= (cur_end + allowed_gap):
            if end > cur_end:
                cur_end = end
        else:
            merged_ranges.append((cur_start, cur_end))
            cur_start, cur_end = start, end

    merged_ranges.append((cur_start, cur_end))

    series_entries = normalize_series(series)
    merged = []
    for start, end in merged_ranges:
        total_sum, avg, min_samples, points = compute_range_metrics(series_entries, start, end)
        window_hours = round((end - start).total_seconds() / 3600.0, 2)
        merged.append(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "startHour": start.hour,
                "endHour": end.hour,
                "windowHours": window_hours,
                "expectedTotal": total_sum,
                "expectedAvg": avg,
                "sampleCountMin": min_samples,
                "windowPoints": points,
            }
        )

    return merged


def smooth_values(values: List[float], window: int) -> List[float]:
    if not values:
        return []
    window = max(1, window)
    if window == 1:
        return values[:]

    half = window // 2
    smoothed = []
    n = len(values)

    for idx in range(n):
        start = max(0, idx - half)
        end = min(n, idx + half + 1)
        smoothed.append(sum(values[start:end]) / (end - start))

    return smoothed


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def build_facility_crowd_baseline(
    facility_loc_ids: Iterable[int],
    loc_data: Dict[int, Dict[str, object]],
    max_caps: Dict[int, int],
) -> Optional[Dict[str, float]]:
    facility_loc_ids = list(dict.fromkeys(int(loc_id) for loc_id in facility_loc_ids))
    if not facility_loc_ids:
        return None

    facility_max_cap = sum_max_caps(max_caps, facility_loc_ids)
    if facility_max_cap <= 0:
        return None

    totals_by_bucket: Dict[datetime, float] = {}
    observed_cap_by_bucket: Dict[datetime, int] = {}

    for loc_id in facility_loc_ids:
        cap = int(max_caps.get(loc_id, 0) or 0)
        if cap <= 0:
            continue

        data = loc_data.get(loc_id) or {}
        bucket_map = data.get("bucket_map") or {}
        if not isinstance(bucket_map, dict):
            continue

        for bucket_ts, ratio in bucket_map.items():
            if not isinstance(bucket_ts, datetime):
                continue
            try:
                ratio_f = float(ratio)
            except Exception:
                continue

            ratio_f = max(0.0, min(ratio_f, 1.2))
            totals_by_bucket[bucket_ts] = totals_by_bucket.get(bucket_ts, 0.0) + ratio_f * cap
            observed_cap_by_bucket[bucket_ts] = observed_cap_by_bucket.get(bucket_ts, 0) + cap

    if not totals_by_bucket:
        return None

    def collect_scaled_totals(require_coverage: bool) -> List[float]:
        scaled: List[float] = []
        for bucket_ts, total in totals_by_bucket.items():
            observed_cap = observed_cap_by_bucket.get(bucket_ts, 0)
            if observed_cap <= 0:
                continue

            coverage = observed_cap / float(facility_max_cap)
            if require_coverage and coverage < CROWD_BASELINE_MIN_COVERAGE:
                continue

            adjusted_total = float(total)
            if observed_cap < facility_max_cap:
                adjusted_total = adjusted_total * (facility_max_cap / float(observed_cap))
            scaled.append(max(0.0, adjusted_total))
        return scaled

    baseline_values = collect_scaled_totals(require_coverage=True)
    if len(baseline_values) < CROWD_BASELINE_MIN_POINTS:
        baseline_values = collect_scaled_totals(require_coverage=False)
    if len(baseline_values) < CROWD_BASELINE_MIN_POINTS:
        return None

    low_q = max(0.0, min(1.0, CROWD_BASELINE_LOW_QUANTILE))
    peak_q = max(0.0, min(1.0, CROWD_BASELINE_PEAK_QUANTILE))
    if peak_q < low_q:
        low_q, peak_q = peak_q, low_q

    mean, std = mean_std(baseline_values)
    low_ceiling = float(np.quantile(baseline_values, low_q))
    peak_floor = float(np.quantile(baseline_values, peak_q))
    if peak_floor < low_ceiling:
        peak_floor = low_ceiling

    return {
        "mean": float(mean),
        "std": float(std),
        "lowCeiling": low_ceiling,
        "peakFloor": peak_floor,
        "sampleCount": float(len(baseline_values)),
    }


def detect_extrema_indices(
    series: List[Dict[str, object]],
    smoothed: List[float],
    mean: float,
    std: float,
    threshold: float,
    mode: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> List[int]:
    if not series or not smoothed:
        return []

    indices = []
    n = len(smoothed)
    peak_cutoff = mean + threshold * std
    if min_value is not None:
        peak_cutoff = max(peak_cutoff, float(min_value))
    low_cutoff = mean - threshold * std
    if max_value is not None:
        low_cutoff = min(low_cutoff, float(max_value))

    for idx in range(n):
        if series[idx]["sampleCount"] < MIN_SAMPLES_PER_HOUR_TOTAL:
            continue

        left = smoothed[idx - 1] if idx - 1 >= 0 else smoothed[idx]
        right = smoothed[idx + 1] if idx + 1 < n else smoothed[idx]

        if mode == "peak":
            if smoothed[idx] < left or smoothed[idx] < right:
                continue
            if smoothed[idx] < peak_cutoff:
                continue
            indices.append(idx)

        if mode == "low":
            if smoothed[idx] > left or smoothed[idx] > right:
                continue
            if smoothed[idx] > low_cutoff:
                continue
            indices.append(idx)

    return indices


def window_indices_for_center(index: int, window: int, length: int) -> Tuple[int, int]:
    window = max(1, min(window, length))
    half = window // 2
    start = max(0, index - half)
    if start + window > length:
        start = length - window
    return start, start + window


def build_windows_for_indices(
    series: List[Dict[str, object]],
    indices: List[int],
    window_points: int,
) -> List[Dict[str, object]]:
    if not series or not indices:
        return []

    windows = {}
    n = len(series)
    step_minutes = infer_series_step_minutes(series)

    for idx in indices:
        start_idx, end_idx = window_indices_for_center(idx, window_points, n)
        window_slice = series[start_idx:end_idx]
        if not window_slice:
            continue

        total_sum = sum(item["expectedTotal"] for item in window_slice)
        min_samples = min(item["sampleCount"] for item in window_slice)

        start_iso = window_slice[0]["hourStart"]
        start_dt = datetime.fromisoformat(start_iso)
        end_dt = start_dt + timedelta(minutes=step_minutes * len(window_slice))
        window_hours = round((step_minutes * len(window_slice)) / 60.0, 2)

        windows[start_iso] = {
            "start": start_iso,
            "end": end_dt.isoformat(),
            "startHour": window_slice[0]["hour"],
            "endHour": end_dt.hour,
            "windowHours": window_hours,
            "expectedTotal": total_sum,
            "expectedAvg": round(total_sum / len(window_slice), 2),
            "sampleCountMin": min_samples,
            "windowPoints": len(window_slice),
            "centerHour": series[idx]["hour"],
        }

    return sorted(windows.values(), key=lambda item: item["start"])


def find_peak_and_low_windows(
    series: List[Dict[str, object]],
    window_hours: int,
    global_baseline: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not series:
        return [], []

    values = [item["expectedTotal"] for item in series]
    smoothed = smooth_values(values, SMOOTH_WINDOW)
    mean, std = mean_std(smoothed)
    peak_floor = None
    low_ceiling = None
    allow_fallback = True

    if global_baseline:
        mean = float(global_baseline.get("mean", mean))
        std = float(global_baseline.get("std", std))
        peak_floor = global_baseline.get("peakFloor")
        low_ceiling = global_baseline.get("lowCeiling")
        allow_fallback = False

    peak_indices = detect_extrema_indices(
        series,
        smoothed,
        mean,
        std,
        PEAK_Z_THRESHOLD,
        "peak",
        min_value=peak_floor,
    )
    low_indices = detect_extrema_indices(
        series,
        smoothed,
        mean,
        std,
        LOW_Z_THRESHOLD,
        "low",
        max_value=low_ceiling,
    )

    if allow_fallback and not peak_indices:
        peak_indices = detect_extrema_indices(series, smoothed, mean, std, 0.0, "peak")
    if allow_fallback and not low_indices:
        low_indices = detect_extrema_indices(series, smoothed, mean, std, 0.0, "low")

    step_minutes = infer_series_step_minutes(series)
    window_points = max(1, int(round((window_hours * 60.0) / step_minutes)))

    return (
        build_windows_for_indices(series, peak_indices, window_points),
        build_windows_for_indices(series, low_indices, window_points),
    )


def parse_window_ranges(windows: List[Dict[str, object]]) -> List[Tuple[datetime, datetime]]:
    ranges: List[Tuple[datetime, datetime]] = []
    for window in windows:
        raw_start = window.get("start")
        raw_end = window.get("end")
        if not raw_start or not raw_end:
            continue
        try:
            start = datetime.fromisoformat(str(raw_start))
            end = datetime.fromisoformat(str(raw_end))
        except Exception:
            continue
        if end > start:
            ranges.append((start, end))
    return ranges


def build_crowd_bands(
    series: List[Dict[str, object]],
    low_windows: List[Dict[str, object]],
    peak_windows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    if not series:
        return []

    step_minutes = infer_series_step_minutes(series)
    step = timedelta(minutes=step_minutes)

    low_ranges = parse_window_ranges(low_windows)
    peak_ranges = parse_window_ranges(peak_windows)
    entries = normalize_series(series)
    if not entries:
        return []

    def in_ranges(ts: datetime, ranges: List[Tuple[datetime, datetime]]) -> bool:
        for start, end in ranges:
            if start <= ts < end:
                return True
        return False

    labels: List[Tuple[datetime, str]] = []
    for ts, _expected, _samples in entries:
        label = "medium"
        # Priority: peak > low, so bands can never overlap.
        if in_ranges(ts, peak_ranges):
            label = "peak"
        elif in_ranges(ts, low_ranges):
            label = "low"
        labels.append((ts, label))

    if not labels:
        return []

    bands: List[Dict[str, object]] = []
    cur_start, cur_label = labels[0]
    prev_ts = labels[0][0]

    for ts, label in labels[1:]:
        if label != cur_label:
            end = prev_ts + step
            bands.append(
                {
                    "start": cur_start.isoformat(),
                    "end": end.isoformat(),
                    "level": cur_label,
                }
            )
            cur_start = ts
            cur_label = label
        prev_ts = ts

    bands.append(
        {
            "start": cur_start.isoformat(),
            "end": (prev_ts + step).isoformat(),
            "level": cur_label,
        }
    )

    return bands


def build_windows_from_bands(
    bands: List[Dict[str, object]],
    level: str,
    series: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    if not bands:
        return []

    series_entries = normalize_series(series)
    windows: List[Dict[str, object]] = []
    for band in bands:
        if band.get("level") != level:
            continue

        raw_start = band.get("start")
        raw_end = band.get("end")
        if not raw_start or not raw_end:
            continue

        try:
            start = datetime.fromisoformat(str(raw_start))
            end = datetime.fromisoformat(str(raw_end))
        except Exception:
            continue

        if end <= start:
            continue

        total_sum, avg, min_samples, points = compute_range_metrics(series_entries, start, end)
        windows.append(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "startHour": start.hour,
                "endHour": end.hour,
                "windowHours": round((end - start).total_seconds() / 3600.0, 2),
                "expectedTotal": total_sum,
                "expectedAvg": avg,
                "sampleCountMin": min_samples,
                "windowPoints": points,
            }
        )

    return windows


def build_forecast():
    now = datetime.now(TZ)
    week_dates = [(now + timedelta(days=offset)).date() for offset in range(7)]
    saved_meta_snapshot = collect_saved_meta_snapshots()
    adaptive_controls = derive_adaptive_runtime_controls(saved_meta_snapshot)

    conn = db_connect()
    try:
        (
            loc_data,
            avg_dow_hour,
            avg_hour,
            avg_overall,
            max_caps,
            loc_samples,
            quality,
        ) = load_history(conn)
    finally:
        conn.close()

    history_start = weather_history_start(loc_data, now)
    weather_history_series = fetch_weather_history_series(history_start, now)
    future_weather_series = fetch_weather_forecast_series(now)
    weather_series = merge_weather_series(weather_history_series, future_weather_series)
    quality["weatherHistoryHours"] = len(weather_history_series.get("times", []))
    quality["weatherForecastHours"] = len(future_weather_series.get("times", []))
    quality["weatherMergedHours"] = len(weather_series.get("times", []))
    quality["weatherAvailable"] = bool(weather_series.get("times"))
    quality_alerts = build_data_quality_alerts(
        quality=quality,
        loc_data=loc_data,
        loc_samples=loc_samples,
    )
    quality["alerts"] = quality_alerts
    quality["adaptiveControls"] = adaptive_controls

    (
        models_by_key,
        model_meta_by_key,
        model_status_by_key,
        run_metrics_by_key,
        onehot_by_key,
        loc_to_model_key,
        loc_to_fallback_key,
        unit_loc_ids,
    ) = prepare_models(
        now=now,
        loc_data=loc_data,
        loc_samples=loc_samples,
        weather_series=weather_series,
        allow_retrain=not bool(quality_alerts.get("blockTraining")),
        adaptive_controls=adaptive_controls,
    )
    interval_profile_by_key: Dict[str, Optional[Dict[str, object]]] = {}
    for model_key, meta in model_meta_by_key.items():
        interval_profile_by_key[model_key] = meta.get("residualProfile") if meta else None

    model_summary = summarize_model_results(
        model_meta_by_key=model_meta_by_key,
        model_status_by_key=model_status_by_key,
        run_metrics_by_key=run_metrics_by_key,
    )
    drift_summary = compute_model_drift(
        now=now,
        models_by_key=models_by_key,
        onehot_by_key=onehot_by_key,
        unit_loc_ids=unit_loc_ids,
        loc_data=loc_data,
        loc_samples=loc_samples,
        weather_series=weather_series,
        model_meta_by_key=model_meta_by_key,
        drift_recent_days=int(adaptive_controls.get("driftRecentDays", DRIFT_RECENT_DAYS)),
        drift_alert_multiplier=float(
            adaptive_controls.get("driftAlertMultiplier", DRIFT_ALERT_MULTIPLIER)
        ),
    )
    interval_multiplier_by_key, drift_actions_summary = apply_drift_actions(
        now=now,
        drift_summary=drift_summary,
        model_meta_by_key=model_meta_by_key,
        action_streak_for_retrain=int(
            adaptive_controls.get("driftActionStreakForRetrain", DRIFT_ACTION_STREAK_FOR_RETRAIN)
        ),
        action_force_hours=max(1, int(DRIFT_ACTION_FORCE_HOURS)),
    )
    for model_key, row in (drift_actions_summary.get("byModel") or {}).items():
        if not isinstance(row, dict) or not bool(row.get("rolledBack")):
            continue
        meta = model_meta_by_key.get(model_key)
        if not isinstance(meta, dict):
            continue
        loc_ids = meta.get("locIds") or []
        feature_count = int(meta.get("featureCount", -1) or -1)
        if not isinstance(loc_ids, list) or feature_count <= 0:
            continue
        try:
            expected_loc_ids = [int(loc_id) for loc_id in loc_ids]
        except Exception:
            continue
        restored_bundle, restored_meta = load_saved_model(
            model_key=model_key,
            expected_loc_ids=expected_loc_ids,
            expected_feature_count=feature_count,
        )
        if restored_bundle is not None:
            models_by_key[model_key] = restored_bundle
        if isinstance(restored_meta, dict):
            model_meta_by_key[model_key] = restored_meta
            interval_profile_by_key[model_key] = restored_meta.get("residualProfile")
    conformal_by_key, conformal_summary = compute_interval_conformal_profiles(
        now=now,
        models_by_key=models_by_key,
        onehot_by_key=onehot_by_key,
        unit_loc_ids=unit_loc_ids,
        loc_data=loc_data,
        loc_samples=loc_samples,
        weather_series=weather_series,
        interval_profile_by_key=interval_profile_by_key,
    )

    ctx = {
        "now": now,
        "models_by_key": models_by_key,
        "interval_profile_by_key": interval_profile_by_key,
        "conformal_by_key": conformal_by_key,
        "interval_multiplier_by_key": interval_multiplier_by_key,
        "weather_series": weather_series,
        "weather_lookup_cache": {},
        "feature_cache": {},
        "prediction_cache": {},
        "loc_data": loc_data,
        "onehot_by_key": onehot_by_key,
        "loc_to_model_key": loc_to_model_key,
        "loc_to_fallback_key": loc_to_fallback_key,
        "avg_dow_hour": avg_dow_hour,
        "avg_hour": avg_hour,
        "avg_overall": avg_overall,
        "loc_samples": loc_samples,
        "max_caps": max_caps,
    }

    facilities_payload = []
    spike_adjusted_categories = 0

    for facility_id, facility in FACILITIES.items():
        weekly_forecast = []
        facility_loc_ids = facility_location_ids(facility)
        forecast_categories = [
            category
            for category in facility["categories"]
            if str(category.get("key")) in FORECAST_CATEGORY_KEYS
        ]
        facility_crowd_baseline = build_facility_crowd_baseline(
            facility_loc_ids=facility_loc_ids,
            loc_data=loc_data,
            max_caps=max_caps,
        )

        for day_date in week_dates:
            targets = get_targets_for_date(day_date)
            window_targets = get_window_targets_for_date(day_date)
            day_weather = build_weather_hours_for_targets(targets, weather_series)
            day_weather_summary = build_weather_day_summary(day_weather)
            day_categories = []

            for category in forecast_categories:
                category_max = sum_max_caps(max_caps, category["location_ids"])
                day_hours = build_category_hours_for_targets(
                    loc_ids=category["location_ids"],
                    targets=targets,
                    category_max=category_max,
                    ctx=ctx,
                )

                live_total, observed_locs = category_live_total(
                    category["location_ids"],
                    loc_data,
                    now,
                )
                if observed_locs > 0:
                    if apply_spike_adjustment_to_category_hours(
                        day_hours=day_hours,
                        category_max=category_max,
                        live_total=live_total,
                        now=now,
                    ):
                        spike_adjusted_categories += 1

                day_categories.append(
                    {
                        "key": category["key"],
                        "title": category["title"],
                        "maxCapacity": category_max or None,
                        "hours": day_hours,
                    }
                )

            totals_day = build_total_series_for_targets(
                loc_ids=facility_loc_ids,
                targets=window_targets,
                ctx=ctx,
            )

            avoid_windows_day, best_windows_day = find_peak_and_low_windows(
                totals_day,
                PEAK_WINDOW_HOURS,
                global_baseline=facility_crowd_baseline,
            )
            avoid_windows_day = merge_windows_with_series(avoid_windows_day, totals_day)
            best_windows_day = merge_windows_with_series(best_windows_day, totals_day)
            crowd_bands_day = build_crowd_bands(
                totals_day,
                low_windows=best_windows_day,
                peak_windows=avoid_windows_day,
            )
            best_windows_day = build_windows_from_bands(crowd_bands_day, "low", totals_day)
            avoid_windows_day = build_windows_from_bands(crowd_bands_day, "peak", totals_day)

            weekly_forecast.append(
                {
                    "dayName": day_date.strftime("%A"),
                    "date": day_date.isoformat(),
                    "weatherHours": day_weather,
                    "weatherSummary": day_weather_summary,
                    "categories": day_categories,
                    "avoidWindows": avoid_windows_day,
                    "bestWindows": best_windows_day,
                    "crowdBands": crowd_bands_day,
                }
            )

        facilities_payload.append(
            {
                "facilityId": facility_id,
                "facilityName": facility["name"],
                "weeklyForecast": weekly_forecast,
            }
        )

    quality["locationsModeled"] = sum(
        1
        for loc_id, data in loc_data.items()
        if loc_samples.get(loc_id, 0) >= MIN_SAMPLES_PER_LOC and not data.get("is_stale")
    )
    quality["spikeAwareEnabled"] = SPIKE_AWARE_ENABLED
    quality["spikeAwareCategoriesAdjusted"] = spike_adjusted_categories
    quality["drift"] = drift_summary
    quality["driftActions"] = drift_actions_summary
    quality["intervalConformal"] = conformal_summary

    val_mae_for_payload = model_summary.get("valMae")
    precision_pct = None
    if val_mae_for_payload is not None:
        precision_pct = max(0.0, min(100.0, (1.0 - float(val_mae_for_payload)) * 100.0))

    payload = {
        "generatedAt": now.isoformat(),
        "timezone": TZ_NAME,
        "model": "xgboost",
        "peakWindowHours": max(1, PEAK_WINDOW_HOURS),
        "peakDetection": {
            "smoothWindow": max(1, SMOOTH_WINDOW),
            "peakZ": PEAK_Z_THRESHOLD,
            "lowZ": LOW_Z_THRESHOLD,
        },
        "modelInfo": {
            "status": model_summary.get("status"),
            "trainedAt": model_summary.get("trainedAt"),
            "trainRows": model_summary.get("trainRows"),
            "valRows": model_summary.get("valRows"),
            "valMae": val_mae_for_payload,
            "valRmse": model_summary.get("valRmse"),
            "precisionPct": precision_pct,
            "byFacility": model_summary.get("byFacility"),
            "byModel": model_summary.get("byModel"),
            "drift": drift_summary,
            "driftActions": drift_actions_summary,
            "intervalConformal": conformal_summary,
            "retrainHours": int(adaptive_controls.get("retrainHours", MODEL_RETRAIN_HOURS)),
            "guardrailMaxMaeDegrade": MODEL_GUARDRAIL_MAX_MAE_DEGRADE,
            "adaptiveControls": adaptive_controls,
            "spikeAwareEnabled": SPIKE_AWARE_ENABLED,
            "spikeAwareHorizonHours": SPIKE_AWARE_HORIZON_HOURS,
            "spikeAwareMaxAgeMin": SPIKE_AWARE_MAX_AGE_MIN,
        },
        "dataQuality": quality,
        "facilities": facilities_payload,
    }

    return payload


def write_forecast(payload: Dict[str, object]) -> None:
    out_dir = os.path.dirname(FORECAST_JSON_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_path = FORECAST_JSON_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(tmp_path, FORECAST_JSON_PATH)


def main() -> int:
    start = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    try:
        payload = build_forecast()
        write_forecast(payload)

        facilities_count = len(payload.get("facilities", []))
        model_info = payload.get("modelInfo", {})
        status = model_info.get("status")
        val_mae = model_info.get("valMae")
        val_rmse = model_info.get("valRmse")
        val_rows = model_info.get("valRows")
        precision_pct = model_info.get("precisionPct")

        metric_part = ""
        if val_mae is not None and val_rmse is not None:
            metric_part = f" | valRows {val_rows} | valMAE {float(val_mae):.4f} | valRMSE {float(val_rmse):.4f}"
        if precision_pct is not None:
            metric_part += f" | precision {float(precision_pct):.2f}%"
        else:
            metric_part += " | precision n/a"

        print(
            f"{start} OK: facilities {facilities_count} | modelStatus {status}{metric_part}"
        )
        return 0
    except Exception as exc:
        print(start, "ERROR:", "{}: {}".format(type(exc).__name__, exc))
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
