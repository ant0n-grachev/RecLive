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
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pymysql
import pytz
import requests
import xgboost as xgb
from env_loader import load_project_dotenv

try:
    from forecast_shared import normalize_section_key
except ImportError:  # pragma: no cover - supports package-style imports.
    from server.forecast_shared import normalize_section_key

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
load_project_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Missing required env var: {name}")

    normalized = value.strip()
    if not normalized:
        raise RuntimeError(f"Missing required env var: {name}")
    return normalized


def require_int_env(name: str) -> int:
    raw = require_env(name)
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer for env var {name}: {raw}") from exc


def require_float_env(name: str) -> float:
    raw = require_env(name)
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid float for env var {name}: {raw}") from exc


def require_path_env(name: str) -> str:
    raw = require_env(name)
    return resolve_path(raw)


def resolve_path(raw: str) -> str:
    if os.path.isabs(raw):
        return raw

    # Prefer script-local relative paths for deployments that keep .env + scripts together.
    script_candidate = os.path.abspath(os.path.join(SCRIPT_DIR, raw))
    return script_candidate

TZ_NAME = "America/Chicago"
TZ = pytz.timezone(TZ_NAME)

RESAMPLE_MINUTES = max(5, int(os.getenv("GYM_RESAMPLE_MINUTES", "15")))
WINDOW_RESAMPLE_MINUTES = max(5, int(os.getenv("GYM_WINDOW_RESAMPLE_MINUTES", "30")))
CROWD_BAND_BRIDGE_MIN = max(
    0,
    int(os.getenv("CROWD_BAND_BRIDGE_MIN", os.getenv("CROWD_BAND_MEDIUM_BRIDGE_MIN", "90"))),
)
HISTORY_DAYS = int(os.getenv("GYM_MODEL_HISTORY_DAYS", "0"))
MIN_SAMPLES_PER_LOC = int(os.getenv("MIN_SAMPLES_PER_LOC", "80"))
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
MODEL_GAMMA = float(os.getenv("MODEL_GAMMA", "0.0"))
MODEL_REG_LAMBDA = float(os.getenv("MODEL_REG_LAMBDA", "1.0"))
MODEL_REG_ALPHA = float(os.getenv("MODEL_REG_ALPHA", "0.0"))
MODEL_TREE_METHOD = os.getenv("MODEL_TREE_METHOD", "hist").strip() or "hist"
MODEL_MAX_BIN = int(os.getenv("MODEL_MAX_BIN", "256"))
MODEL_RETRAIN_HOURS = int(os.getenv("MODEL_RETRAIN_HOURS", "24"))
MODEL_GUARDRAIL_MAX_MAE_DEGRADE = float(os.getenv("MODEL_GUARDRAIL_MAX_MAE_DEGRADE", "0.05"))
MODEL_GUARDRAIL_MIN_VAL_ROWS = int(os.getenv("MODEL_GUARDRAIL_MIN_VAL_ROWS", "200"))
MODEL_HOLDOUT_SPLIT = float(os.getenv("MODEL_HOLDOUT_SPLIT", "0.12"))
MODEL_HOLDOUT_MIN_ROWS = int(os.getenv("MODEL_HOLDOUT_MIN_ROWS", "120"))
MODEL_GUARDRAIL_MAX_HOLDOUT_MAE_DEGRADE = float(
    os.getenv("MODEL_GUARDRAIL_MAX_HOLDOUT_MAE_DEGRADE", "0.03")
)
MODEL_GUARDRAIL_MAX_HOLDOUT_INTERVAL_ERR_DEGRADE = float(
    os.getenv("MODEL_GUARDRAIL_MAX_HOLDOUT_INTERVAL_ERR_DEGRADE", "0.02")
)
MODEL_GUARDRAIL_MAX_VAL_INTERVAL_ERR_DEGRADE = float(
    os.getenv("MODEL_GUARDRAIL_MAX_VAL_INTERVAL_ERR_DEGRADE", "0.015")
)
MODEL_FEATURE_MISSING_GUARD_ENABLED = os.getenv("MODEL_FEATURE_MISSING_GUARD_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_FEATURE_MISSING_MIN_ROWS = int(os.getenv("MODEL_FEATURE_MISSING_MIN_ROWS", "240"))
MODEL_MAX_FEATURE_MISSING_RATE = float(os.getenv("MODEL_MAX_FEATURE_MISSING_RATE", "0.45"))
MODEL_MAX_LAG_MISSING_RATE = float(os.getenv("MODEL_MAX_LAG_MISSING_RATE", "0.55"))
MODEL_MAX_WEATHER_MISSING_RATE = float(os.getenv("MODEL_MAX_WEATHER_MISSING_RATE", "0.65"))
MODEL_FEATURE_CLIP_ENABLED = os.getenv("MODEL_FEATURE_CLIP_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_FEATURE_CLIP_LOWER_Q = float(os.getenv("MODEL_FEATURE_CLIP_LOWER_Q", "0.002"))
MODEL_FEATURE_CLIP_UPPER_Q = float(os.getenv("MODEL_FEATURE_CLIP_UPPER_Q", "0.998"))
MODEL_FEATURE_CLIP_MIN_SPREAD = float(os.getenv("MODEL_FEATURE_CLIP_MIN_SPREAD", "0.01"))
MODEL_FEATURE_ABS_MAX = float(os.getenv("MODEL_FEATURE_ABS_MAX", "500.0"))
MODEL_MIN_FEATURE_FINITE_RATIO = float(os.getenv("MODEL_MIN_FEATURE_FINITE_RATIO", "0.30"))
MODEL_ENSEMBLE_BLEND_ENABLED = os.getenv("MODEL_ENSEMBLE_BLEND_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_ENSEMBLE_DEFAULT_PRIMARY_WEIGHT = float(
    os.getenv("MODEL_ENSEMBLE_DEFAULT_PRIMARY_WEIGHT", "0.72")
)
MODEL_ENSEMBLE_MIN_PRIMARY_WEIGHT = float(os.getenv("MODEL_ENSEMBLE_MIN_PRIMARY_WEIGHT", "0.15"))
MODEL_ENSEMBLE_MAX_PRIMARY_WEIGHT = float(os.getenv("MODEL_ENSEMBLE_MAX_PRIMARY_WEIGHT", "0.9"))
MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_ENABLED = os.getenv(
    "MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_ENABLED",
    "1",
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_STRENGTH = float(
    os.getenv("MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_STRENGTH", "0.45")
)
MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_EXP = float(
    os.getenv("MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_EXP", "1.6")
)
MODEL_ENSEMBLE_SAMPLE_SUPPORT_ADJUST_ENABLED = os.getenv(
    "MODEL_ENSEMBLE_SAMPLE_SUPPORT_ADJUST_ENABLED",
    "1",
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_ENSEMBLE_SAMPLE_SUPPORT_TARGET = int(
    os.getenv("MODEL_ENSEMBLE_SAMPLE_SUPPORT_TARGET", "80")
)
MODEL_ENSEMBLE_SAMPLE_SUPPORT_MAX_SHIFT = float(
    os.getenv("MODEL_ENSEMBLE_SAMPLE_SUPPORT_MAX_SHIFT", "0.22")
)
LIVE_BIAS_ENABLED = os.getenv("MODEL_LIVE_BIAS_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
LIVE_BIAS_MAX_AGE_MIN = float(os.getenv("MODEL_LIVE_BIAS_MAX_AGE_MIN", "45"))
LIVE_BIAS_MAX_HORIZON_HOURS = float(os.getenv("MODEL_LIVE_BIAS_MAX_HORIZON_HOURS", "4"))
LIVE_BIAS_BASE_WEIGHT = float(os.getenv("MODEL_LIVE_BIAS_BASE_WEIGHT", "0.55"))
LIVE_BIAS_HORIZON_DECAY = float(os.getenv("MODEL_LIVE_BIAS_HORIZON_DECAY", "1.6"))
LIVE_BIAS_AGE_DECAY_MIN = float(os.getenv("MODEL_LIVE_BIAS_AGE_DECAY_MIN", "35"))
MODEL_LOW_SAMPLE_BLEND_ENABLED = os.getenv("MODEL_LOW_SAMPLE_BLEND_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_LOW_SAMPLE_TARGET_COUNT = int(os.getenv("MODEL_LOW_SAMPLE_TARGET_COUNT", "60"))
MODEL_LOW_SAMPLE_MAX_BLEND = float(os.getenv("MODEL_LOW_SAMPLE_MAX_BLEND", "0.35"))
MODEL_MISSING_FEATURE_BLEND_ENABLED = os.getenv("MODEL_MISSING_FEATURE_BLEND_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_MISSING_FEATURE_BLEND_START = float(os.getenv("MODEL_MISSING_FEATURE_BLEND_START", "0.25"))
MODEL_MISSING_FEATURE_BLEND_FULL = float(os.getenv("MODEL_MISSING_FEATURE_BLEND_FULL", "0.60"))
MODEL_MISSING_FEATURE_BLEND_MAX_WEIGHT = float(os.getenv("MODEL_MISSING_FEATURE_BLEND_MAX_WEIGHT", "0.45"))
MODEL_MISSING_FEATURE_INTERVAL_WIDEN_ENABLED = os.getenv(
    "MODEL_MISSING_FEATURE_INTERVAL_WIDEN_ENABLED",
    "1",
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_MISSING_FEATURE_INTERVAL_WIDEN_START = float(
    os.getenv("MODEL_MISSING_FEATURE_INTERVAL_WIDEN_START", "0.20")
)
MODEL_MISSING_FEATURE_INTERVAL_WIDEN_FULL = float(
    os.getenv("MODEL_MISSING_FEATURE_INTERVAL_WIDEN_FULL", "0.60")
)
MODEL_MISSING_FEATURE_INTERVAL_WIDEN_MAX_MULT = float(
    os.getenv("MODEL_MISSING_FEATURE_INTERVAL_WIDEN_MAX_MULT", "1.55")
)
MODEL_SAMPLE_SUPPORT_INTERVAL_WIDEN_ENABLED = os.getenv(
    "MODEL_SAMPLE_SUPPORT_INTERVAL_WIDEN_ENABLED",
    "1",
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_SAMPLE_SUPPORT_INTERVAL_TARGET = int(
    os.getenv("MODEL_SAMPLE_SUPPORT_INTERVAL_TARGET", "80")
)
MODEL_SAMPLE_SUPPORT_INTERVAL_MAX_MULT = float(
    os.getenv("MODEL_SAMPLE_SUPPORT_INTERVAL_MAX_MULT", "1.28")
)
MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_ENABLED = os.getenv(
    "MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_ENABLED",
    "1",
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_MIN_DIFF = float(
    os.getenv("MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_MIN_DIFF", "0.03")
)
MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_SCALE = float(
    os.getenv("MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_SCALE", "0.75")
)
MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_MAX_MULT = float(
    os.getenv("MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_MAX_MULT", "1.38")
)
MODEL_POINT_BIAS_CORRECTION_ENABLED = os.getenv("MODEL_POINT_BIAS_CORRECTION_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_POINT_BIAS_MIN_POINTS_PER_SEGMENT = int(
    os.getenv("MODEL_POINT_BIAS_MIN_POINTS_PER_SEGMENT", "30")
)
MODEL_POINT_BIAS_MAX_ABS = float(os.getenv("MODEL_POINT_BIAS_MAX_ABS", "0.25"))
MODEL_POINT_BIAS_OCCUPANCY_ENABLED = os.getenv("MODEL_POINT_BIAS_OCCUPANCY_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_POINT_BIAS_MIN_POINTS_PER_OCCUPANCY = int(
    os.getenv("MODEL_POINT_BIAS_MIN_POINTS_PER_OCCUPANCY", "24")
)
MODEL_POINT_BIAS_SUPPORT_TARGET_MULT = float(
    os.getenv("MODEL_POINT_BIAS_SUPPORT_TARGET_MULT", "3.0")
)
MODEL_TUNING_INTERVAL_ERR_WEIGHT = float(os.getenv("MODEL_TUNING_INTERVAL_ERR_WEIGHT", "0.2"))
MODEL_TUNING_TAIL_ERR_WEIGHT = float(os.getenv("MODEL_TUNING_TAIL_ERR_WEIGHT", "0.15"))
MODEL_TUNING_TAIL_QUANTILE = float(os.getenv("MODEL_TUNING_TAIL_QUANTILE", "0.9"))
MODEL_TUNING_COMPLEXITY_WEIGHT = float(os.getenv("MODEL_TUNING_COMPLEXITY_WEIGHT", "0.03"))
MODEL_TUNING_COMPLEXITY_DEPTH_REF = int(os.getenv("MODEL_TUNING_COMPLEXITY_DEPTH_REF", "4"))
MODEL_LONG_HORIZON_BLEND_ENABLED = os.getenv("MODEL_LONG_HORIZON_BLEND_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_LONG_HORIZON_BLEND_START_HOURS = float(os.getenv("MODEL_LONG_HORIZON_BLEND_START_HOURS", "4"))
MODEL_LONG_HORIZON_BLEND_FULL_HOURS = float(os.getenv("MODEL_LONG_HORIZON_BLEND_FULL_HOURS", "12"))
MODEL_LONG_HORIZON_BLEND_MAX_WEIGHT = float(os.getenv("MODEL_LONG_HORIZON_BLEND_MAX_WEIGHT", "0.28"))
MODEL_DIRECT_HORIZON_ENABLED = os.getenv("MODEL_DIRECT_HORIZON_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_DIRECT_HORIZON_HOURS_RAW = os.getenv("MODEL_DIRECT_HORIZON_HOURS", "1,2,3,6,12")
MODEL_DIRECT_HORIZON_MIN_PAIRS = int(os.getenv("MODEL_DIRECT_HORIZON_MIN_PAIRS", "100"))
MODEL_DIRECT_HORIZON_SEGMENT_MIN_PAIRS = int(
    os.getenv("MODEL_DIRECT_HORIZON_SEGMENT_MIN_PAIRS", "30")
)
MODEL_DIRECT_HORIZON_MAX_BLEND = float(os.getenv("MODEL_DIRECT_HORIZON_MAX_BLEND", "0.45"))
MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENTS_ENABLED = os.getenv(
    "MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENTS_ENABLED",
    "1",
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENT_MIN_PAIRS = int(
    os.getenv("MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENT_MIN_PAIRS", "18")
)
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
LOCATION_BALANCE_WEIGHT_ENABLED = os.getenv("MODEL_LOCATION_BALANCE_WEIGHT_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
LOCATION_BALANCE_WEIGHT_POWER = float(os.getenv("MODEL_LOCATION_BALANCE_WEIGHT_POWER", "0.5"))
LOCATION_BALANCE_WEIGHT_MIN = float(os.getenv("MODEL_LOCATION_BALANCE_WEIGHT_MIN", "0.55"))
LOCATION_BALANCE_WEIGHT_MAX = float(os.getenv("MODEL_LOCATION_BALANCE_WEIGHT_MAX", "1.8"))
MODEL_WEIGHT_STABILIZATION_ENABLED = os.getenv("MODEL_WEIGHT_STABILIZATION_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_WEIGHT_CLIP_LOWER_Q = float(os.getenv("MODEL_WEIGHT_CLIP_LOWER_Q", "0.02"))
MODEL_WEIGHT_CLIP_UPPER_Q = float(os.getenv("MODEL_WEIGHT_CLIP_UPPER_Q", "0.98"))
MODEL_WEIGHT_CLIP_MIN = float(os.getenv("MODEL_WEIGHT_CLIP_MIN", "0.08"))
MODEL_WEIGHT_CLIP_MAX = float(os.getenv("MODEL_WEIGHT_CLIP_MAX", "6.0"))
MODEL_WEIGHT_NORMALIZE_MEAN = os.getenv("MODEL_WEIGHT_NORMALIZE_MEAN", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
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
RECENT_DRIFT_BIAS_ENABLED = os.getenv("MODEL_RECENT_DRIFT_BIAS_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
RECENT_DRIFT_BIAS_MAX_ABS = float(os.getenv("MODEL_RECENT_DRIFT_BIAS_MAX_ABS", "0.12"))
RECENT_DRIFT_BIAS_MIN_POINTS_PER_HOUR = int(os.getenv("MODEL_RECENT_DRIFT_BIAS_MIN_POINTS_PER_HOUR", "20"))
RECENT_DRIFT_BIAS_MIN_POINTS_PER_OCCUPANCY = int(
    os.getenv("MODEL_RECENT_DRIFT_BIAS_MIN_POINTS_PER_OCCUPANCY", "24")
)
RECENT_DRIFT_BIAS_HORIZON_DECAY_HOURS = float(os.getenv("MODEL_RECENT_DRIFT_BIAS_HORIZON_DECAY_HOURS", "8"))
RECENT_DRIFT_BIAS_BLEND = float(os.getenv("MODEL_RECENT_DRIFT_BIAS_BLEND", "0.6"))
RECENT_DRIFT_BIAS_SUPPORT_TARGET_MULT = float(os.getenv("MODEL_RECENT_DRIFT_BIAS_SUPPORT_TARGET_MULT", "3.0"))
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
MODEL_SCHEMA_VERSION = 10

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
INTERVAL_CONFORMAL_SEGMENT_BLEND_TARGET_MULT = float(
    os.getenv("INTERVAL_CONFORMAL_SEGMENT_BLEND_TARGET_MULT", "3.0")
)
INTERVAL_SEGMENT_BLEND_TARGET_MULT = float(os.getenv("INTERVAL_SEGMENT_BLEND_TARGET_MULT", "3.0"))
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

DB_TIMEZONE_NAME = require_env("GYM_DB_TIMEZONE")
DB_TZ = pytz.timezone(DB_TIMEZONE_NAME)

FORECAST_DAY_START_HOUR = require_int_env("FORECAST_DAY_START_HOUR")
FORECAST_DAY_END_HOUR = require_int_env("FORECAST_DAY_END_HOUR")

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
MODEL_FEATURE_QUALITY_WEIGHT_ENABLED = os.getenv("MODEL_FEATURE_QUALITY_WEIGHT_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_FEATURE_QUALITY_WEIGHT_MIN = float(os.getenv("MODEL_FEATURE_QUALITY_WEIGHT_MIN", "0.35"))
MODEL_FEATURE_QUALITY_WEIGHT_POWER = float(os.getenv("MODEL_FEATURE_QUALITY_WEIGHT_POWER", "1.25"))

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
WEATHER_URL = require_env("GYM_WEATHER_URL")
WEATHER_ARCHIVE_URL = require_env("GYM_WEATHER_ARCHIVE_URL")
WEATHER_LAT = require_float_env("GYM_WEATHER_LAT")
WEATHER_LON = require_float_env("GYM_WEATHER_LON")
WEATHER_FORECAST_DAYS = require_int_env("GYM_WEATHER_FORECAST_DAYS")
WEATHER_HISTORY_MAX_DAYS = require_int_env("GYM_WEATHER_HISTORY_MAX_DAYS")

FORECAST_JSON_PATH = require_path_env("FORECAST_JSON_PATH")

MODEL_ARTIFACT_DIR = require_path_env("MODEL_ARTIFACT_DIR")
MODEL_BASENAME = require_env("MODEL_BASENAME")
FACILITY_HOURS_JSON_PATH = resolve_path(
    os.getenv("FACILITY_HOURS_JSON_PATH", "facility_hours.json").strip() or "facility_hours.json"
)
SCHEDULE_FILTER_ENABLED = os.getenv("SCHEDULE_FILTER_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
SCHEDULE_BOUNDARY_ZERO_ENABLED = os.getenv("SCHEDULE_BOUNDARY_ZERO_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
SCHEDULE_TRANSITION_WEIGHT_ENABLED = os.getenv("SCHEDULE_TRANSITION_WEIGHT_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
SCHEDULE_TRANSITION_WEIGHT_MULTIPLIER = max(
    1.0,
    float(os.getenv("SCHEDULE_TRANSITION_WEIGHT_MULTIPLIER", "1.35")),
)
FORECAST_OUTPUT_INCLUDE_WEATHER = os.getenv("FORECAST_OUTPUT_INCLUDE_WEATHER", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
FORECAST_OUTPUT_INCLUDE_INTERVAL_FIELDS = os.getenv("FORECAST_OUTPUT_INCLUDE_INTERVAL_FIELDS", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

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
WEATHER_DERIVED_FEATURE_COUNT = 12
WEATHER_QUALITY_FEATURE_COUNT = 4
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


SCHEDULE_MINUTES_PER_DAY = 24 * 60
SCHEDULE_DAY_TOKEN_PATTERN = (
    r"mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:r|rs|rsday)?|"
    r"fri(?:day)?|sat(?:urday)?|sun(?:day)?"
)
SCHEDULE_DAY_TOKEN_RE = re.compile(rf"\b({SCHEDULE_DAY_TOKEN_PATTERN})\b", re.IGNORECASE)
SCHEDULE_DAY_RANGE_RE = re.compile(
    rf"\b({SCHEDULE_DAY_TOKEN_PATTERN})\s*-\s*({SCHEDULE_DAY_TOKEN_PATTERN})\b",
    re.IGNORECASE,
)
SCHEDULE_DATE_LIKE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})|(\d{1,2}/\d{1,2}(?:/\d{2,4})?)|([a-z]{3,9}\s+\d{1,2})",
    re.IGNORECASE,
)
SCHEDULE_MONTH_BY_NAME = {
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


def normalize_schedule_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).lower().replace("–", "-").replace("—", "-")).strip()


def schedule_day_token_to_index(token: str) -> Optional[int]:
    text = normalize_schedule_text(token)
    if text.startswith("mon"):
        return 0
    if text.startswith("tue"):
        return 1
    if text.startswith("wed"):
        return 2
    if text.startswith("thu"):
        return 3
    if text.startswith("fri"):
        return 4
    if text.startswith("sat"):
        return 5
    if text.startswith("sun"):
        return 6
    return None


def safe_build_date(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(int(year), int(month), int(day))
    except Exception:
        return None


def parse_schedule_token_date(
    token: str,
    fallback_year: int,
    fallback_month: Optional[int] = None,
) -> Optional[date]:
    normalized = normalize_schedule_text(token)
    if not normalized:
        return None

    iso_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", normalized)
    if iso_match:
        return safe_build_date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))

    slash_match = re.match(r"^(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?$", normalized)
    if slash_match:
        month = int(slash_match.group(1))
        day = int(slash_match.group(2))
        year_raw = slash_match.group(3)
        year = int(fallback_year)
        if year_raw:
            parsed_year = int(year_raw)
            if parsed_year < 100:
                parsed_year += 2000
            year = parsed_year
        return safe_build_date(year, month, day)

    month_day_match = re.match(r"^([a-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?$", normalized)
    if month_day_match:
        month = SCHEDULE_MONTH_BY_NAME.get(month_day_match.group(1))
        if month is None:
            return None
        day = int(month_day_match.group(2))
        year = int(month_day_match.group(3)) if month_day_match.group(3) else int(fallback_year)
        return safe_build_date(year, month, day)

    day_only_match = re.match(r"^(\d{1,2})$", normalized)
    if day_only_match and fallback_month is not None:
        return safe_build_date(int(fallback_year), int(fallback_month), int(day_only_match.group(1)))

    return None


def parse_schedule_date_range(value: str, fallback_year: int) -> Optional[Tuple[date, date, int]]:
    normalized = normalize_schedule_text(value)
    if not normalized:
        return None
    if not SCHEDULE_DATE_LIKE_RE.search(normalized):
        return None

    parts = [part.strip() for part in re.split(r"\s+-\s+", normalized) if part.strip()]
    if not parts:
        return None

    start = parse_schedule_token_date(parts[0], fallback_year)
    if start is None:
        return None

    end_token = parts[-1]
    end = parse_schedule_token_date(end_token, start.year, start.month)
    if end is None:
        end = parse_schedule_token_date(end_token, fallback_year)
    if end is None:
        return None

    if end < start:
        shifted_end = parse_schedule_token_date(end_token, start.year + 1, start.month)
        if shifted_end is None or shifted_end < start:
            return None
        end = shifted_end

    span_days = (end - start).days + 1
    if span_days <= 0:
        return None
    return start, end, span_days


def parse_schedule_weekday_set(label: str) -> Optional[Set[int]]:
    normalized = normalize_schedule_text(label)
    if not normalized:
        return None

    if "daily" in normalized:
        return set(range(7))
    if "weekdays" in normalized:
        return {0, 1, 2, 3, 4}
    if "weekends" in normalized:
        return {5, 6}

    range_match = SCHEDULE_DAY_RANGE_RE.search(normalized)
    if range_match:
        start = schedule_day_token_to_index(range_match.group(1))
        end = schedule_day_token_to_index(range_match.group(2))
        if start is not None and end is not None:
            output: Set[int] = set()
            if start <= end:
                for idx in range(start, end + 1):
                    output.add(idx)
            else:
                for idx in range(start, 7):
                    output.add(idx)
                for idx in range(0, end + 1):
                    output.add(idx)
            return output

    output: Set[int] = set()
    for match in SCHEDULE_DAY_TOKEN_RE.finditer(normalized):
        idx = schedule_day_token_to_index(match.group(1))
        if idx is not None:
            output.add(idx)
    return output if output else None


def parse_schedule_clock_token(token: str, is_end: bool) -> Optional[int]:
    normalized = normalize_schedule_text(token).replace(".", "")
    if not normalized:
        return None
    if normalized == "midnight":
        return SCHEDULE_MINUTES_PER_DAY if is_end else 0
    if normalized == "noon":
        return 12 * 60

    match = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", normalized)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or "0")
    suffix = match.group(3) or ""

    if minute < 0 or minute > 59:
        return None

    if suffix == "am":
        if hour == 12:
            hour = 0
    elif suffix == "pm":
        if hour < 12:
            hour += 12

    if hour < 0 or hour > 24:
        return None
    if hour == 24 and minute > 0:
        return None
    return hour * 60 + minute


def parse_schedule_hours_window(value: str) -> Optional[Tuple[int, int, bool]]:
    normalized = normalize_schedule_text(value)
    if not normalized:
        return None

    if "closed" in normalized:
        return 0, 0, True
    if "24 hours" in normalized:
        return 0, SCHEDULE_MINUTES_PER_DAY, False

    parts = [part for part in re.split(r"\s*-\s*", normalized) if part]
    if len(parts) != 2 and " to " in normalized:
        parts = [part.strip() for part in normalized.split(" to ", 1) if part.strip()]
    if len(parts) != 2:
        return None

    start_minutes = parse_schedule_clock_token(parts[0], is_end=False)
    end_minutes = parse_schedule_clock_token(parts[1], is_end=True)
    if start_minutes is None or end_minutes is None:
        return None

    if end_minutes <= start_minutes:
        end_minutes += SCHEDULE_MINUTES_PER_DAY
    return int(start_minutes), int(end_minutes), False


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

    if end_minutes > SCHEDULE_MINUTES_PER_DAY:
        return minute_of_day >= start_minutes or minute_of_day < (end_minutes - SCHEDULE_MINUTES_PER_DAY)
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
        int(end_minutes - SCHEDULE_MINUTES_PER_DAY)
        if end_minutes > SCHEDULE_MINUTES_PER_DAY
        else int(end_minutes % SCHEDULE_MINUTES_PER_DAY)
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
    zeros = [0.0] * SCHEDULE_PHASE_FEATURE_COUNT
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
    if end_minutes > SCHEDULE_MINUTES_PER_DAY and minute_abs < int(start_minutes):
        minute_abs += SCHEDULE_MINUTES_PER_DAY

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
    if not SCHEDULE_TRANSITION_WEIGHT_ENABLED:
        return 1.0
    if not isinstance(phase_features, list) or len(phase_features) < SCHEDULE_PHASE_FEATURE_COUNT:
        return 1.0
    near_open = max(0.0, min(1.0, float(to_float_or_none(phase_features[4]) or 0.0)))
    near_close = max(0.0, min(1.0, float(to_float_or_none(phase_features[5]) or 0.0)))
    edge_strength = max(near_open, near_close)
    if edge_strength <= 0.0:
        return 1.0
    mult = max(1.0, float(SCHEDULE_TRANSITION_WEIGHT_MULTIPLIER))
    return float(1.0 + (mult - 1.0) * edge_strength)


def schedule_phase_features_for_location_target(
    loc_data: Dict[str, object],
    target: datetime,
) -> List[float]:
    zeros = [0.0] * SCHEDULE_PHASE_FEATURE_COUNT
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
        and len(cached_phase) == SCHEDULE_PHASE_FEATURE_COUNT
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


def load_schedule_sections_by_facility() -> Dict[int, Dict[str, object]]:
    if not SCHEDULE_FILTER_ENABLED:
        return {}
    if not os.path.exists(FACILITY_HOURS_JSON_PATH):
        return {}

    try:
        with open(FACILITY_HOURS_JSON_PATH, "r", encoding="utf-8") as handle:
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
LAG_TREND_SENSOR_FEATURE_COUNT = 38
SCHEDULE_PHASE_FEATURE_COUNT = 6

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
    host = require_env("GYM_DB_HOST")
    port = require_int_env("GYM_DB_PORT")
    user = require_env("GYM_DB_USER")
    password = require_env("GYM_DB_PASSWORD")
    database = require_env("GYM_DB_NAME")

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
    # time/cycle + calendar + lags/sensor + schedule-phase + weather + one-hot
    return (
        17
        + CALENDAR_FEATURE_COUNT
        + LAG_TREND_SENSOR_FEATURE_COUNT
        + SCHEDULE_PHASE_FEATURE_COUNT
        + (len(WEATHER_KEYS) * 3)
        + len(WEATHER_ROLLING_KEYS)
        + WEATHER_DERIVED_FEATURE_COUNT
        + WEATHER_QUALITY_FEATURE_COUNT
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
        ts = target - timedelta(minutes=RESAMPLE_MINUTES * i)
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
        ts = target - timedelta(minutes=RESAMPLE_MINUTES * i)
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
        ts = target - timedelta(minutes=RESAMPLE_MINUTES * i)
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


def feature_missing_rate(feature_row: List[float]) -> float:
    arr = np.asarray(feature_row, dtype=np.float32).reshape(-1)
    if arr.size <= 0:
        return 1.0
    finite = np.isfinite(arr)
    return float(1.0 - (float(np.count_nonzero(finite)) / float(arr.size)))


def feature_quality_weight(feature_row: List[float]) -> float:
    if not MODEL_FEATURE_QUALITY_WEIGHT_ENABLED:
        return 1.0

    missing = max(0.0, min(1.0, feature_missing_rate(feature_row)))
    completeness = 1.0 - missing
    min_w = max(0.05, min(1.0, float(MODEL_FEATURE_QUALITY_WEIGHT_MIN)))
    power = max(0.25, min(4.0, float(MODEL_FEATURE_QUALITY_WEIGHT_POWER)))
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
    cache_key = (bool(MODEL_DIRECT_HORIZON_ENABLED), str(MODEL_DIRECT_HORIZON_HOURS_RAW or ""))
    cached = getattr(configured_direct_horizon_hours, "_cache", None)
    if isinstance(cached, dict) and cache_key in cached:
        return list(cached[cache_key])

    if not MODEL_DIRECT_HORIZON_ENABLED:
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
        ts = target - timedelta(minutes=RESAMPLE_MINUTES * i)
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
    now_den = max(1, len(WEATHER_KEYS))
    roll_den = max(1, len(WEATHER_ROLLING_KEYS))
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
        target - timedelta(minutes=RESAMPLE_MINUTES),
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
            target - timedelta(minutes=RESAMPLE_MINUTES),
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
        steps=max(1, int(60 / RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    roll_2h = rolling_mean(
        bucket_map,
        target,
        steps=max(1, int(120 / RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    roll_6h = rolling_mean(
        bucket_map,
        target,
        steps=max(1, int(360 / RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    roll_12h = rolling_mean(
        bucket_map,
        target,
        steps=max(1, int(720 / RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    roll_24h = rolling_mean(
        bucket_map,
        target,
        steps=max(1, int(1440 / RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    vol_1h = rolling_std(
        bucket_map,
        target,
        steps=max(2, int(60 / RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    vol_3h = rolling_std(
        bucket_map,
        target,
        steps=max(2, int(180 / RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    vol_6h = rolling_std(
        bucket_map,
        target,
        steps=max(2, int(360 / RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    range_1h = rolling_range(
        bucket_map,
        target,
        steps=max(2, int(60 / RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    range_3h = rolling_range(
        bucket_map,
        target,
        steps=max(2, int(180 / RESAMPLE_MINUTES)),
        lag_ratio_override=lag_ratio_override,
    )
    range_6h = rolling_range(
        bucket_map,
        target,
        steps=max(2, int(360 / RESAMPLE_MINUTES)),
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


def load_history(
    conn,
    facility_schedule_by_id: Optional[Dict[int, Dict[str, object]]] = None,
):
    raw_by_loc: Dict[int, List[Tuple[datetime, float]]] = {}
    max_caps: Dict[int, int] = {}
    location_facility_map = location_to_facility_map()
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
        "scheduleFilterEnabled": bool(SCHEDULE_FILTER_ENABLED),
        "scheduleBoundaryZeroEnabled": bool(SCHEDULE_BOUNDARY_ZERO_ENABLED),
        "scheduleFacilitiesLoaded": len(active_schedule_map),
        "duplicatesRemoved": 0,
        "impossibleJumpsRemoved": 0,
        "flatlineRowsPruned": 0,
        "flatlineRunsDetected": 0,
        "boundaryBucketsZeroed": 0,
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
            if not math.isfinite(current_capacity) or current_capacity < 0:
                quality["rowsDroppedInvalid"] += 1
                continue

            local_dt = parse_observed_at_value(last_updated)
            if local_dt is None and fetched_at is not None:
                local_dt = to_local(fetched_at)
            if local_dt is None:
                quality["rowsDroppedInvalid"] += 1
                continue

            if active_schedule_map:
                facility_id = location_facility_map.get(loc_id)
                if facility_id is not None and facility_id in active_schedule_map:
                    minute_of_day = int(local_dt.hour) * 60 + int(local_dt.minute)
                    schedule_cache_key = (int(facility_id), local_dt.date(), minute_of_day)
                    if schedule_cache_key not in schedule_eval_cache:
                        sections_raw = active_schedule_map.get(facility_id, {}).get("sections", [])
                        sections = sections_raw if isinstance(sections_raw, list) else []
                        schedule_eval_cache[schedule_cache_key] = get_facility_schedule_open_state(
                            sections=sections,
                            ts=local_dt,
                            date_range_cache=schedule_date_range_cache,
                            weekday_cache=schedule_weekday_cache,
                            hours_window_cache=schedule_hours_cache,
                        )
                    if (
                        SCHEDULE_BOUNDARY_ZERO_ENABLED
                        and schedule_cache_key not in schedule_boundary_cache
                    ):
                        sections_raw = active_schedule_map.get(facility_id, {}).get("sections", [])
                        sections = sections_raw if isinstance(sections_raw, list) else []
                        schedule_boundary_cache[schedule_cache_key] = get_facility_schedule_boundary_state(
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
                    if SCHEDULE_BOUNDARY_ZERO_ENABLED:
                        boundary_open, boundary_close = schedule_boundary_cache.get(
                            schedule_cache_key,
                            (False, False),
                        )
                        if boundary_open or boundary_close:
                            current_capacity = 0.0
                            quality["rowsScheduleBoundaryZeroed"] += 1
                    if open_state is None:
                        quality["rowsScheduleUnknown"] += 1
                    elif not open_state and not (SCHEDULE_BOUNDARY_ZERO_ENABLED and boundary_close):
                        quality["rowsDroppedScheduleClosed"] += 1
                        continue
                    elif not open_state and SCHEDULE_BOUNDARY_ZERO_ENABLED and boundary_close:
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
    now_local = datetime.now(TZ)

    for loc_id, entries in raw_by_loc.items():
        max_cap = max_caps.get(loc_id, 0)
        if max_cap <= 0:
            continue
        loc_facility_id = location_facility_map.get(loc_id)

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
            if not math.isfinite(float(count)):
                continue
            bucket = floor_time(ts, RESAMPLE_MINUTES)
            bucket_counts.setdefault(bucket, []).append(float(count))

        if not bucket_counts:
            continue

        bucket_times = sorted(bucket_counts.keys())
        bucket_values: List[float] = []
        bucket_map: Dict[datetime, float] = {}
        loc_avg_dow_hour_sum: Dict[Tuple[int, int], List[float]] = {}
        loc_avg_hour_sum: Dict[int, List[float]] = {}
        loc_avg_overall_sum = [0.0, 0.0]

        for bt in bucket_times:
            avg_count = float(sum(bucket_counts[bt]) / len(bucket_counts[bt]))
            if not math.isfinite(avg_count):
                continue
            ratio = avg_count / max_cap
            ratio = max(0.0, min(ratio, 1.2))
            if (
                SCHEDULE_BOUNDARY_ZERO_ENABLED
                and loc_facility_id is not None
                and loc_facility_id in active_schedule_map
            ):
                minute_of_day = int(bt.hour) * 60 + int(bt.minute)
                schedule_cache_key = (int(loc_facility_id), bt.date(), minute_of_day)
                if schedule_cache_key not in schedule_boundary_cache:
                    sections_raw = active_schedule_map.get(loc_facility_id, {}).get("sections", [])
                    sections = sections_raw if isinstance(sections_raw, list) else []
                    schedule_boundary_cache[schedule_cache_key] = get_facility_schedule_boundary_state(
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

            aggregate_sum_count(avg_dow_hour_sum, (loc_id, bt.weekday(), bt.hour), ratio)
            aggregate_sum_count(avg_hour_sum, (loc_id, bt.hour), ratio)
            aggregate_sum_count(avg_overall_sum, loc_id, ratio)
            aggregate_sum_count(loc_avg_dow_hour_sum, (bt.weekday(), bt.hour), ratio)
            aggregate_sum_count(loc_avg_hour_sum, bt.hour, ratio)
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
            "max_cap": max_cap,
            "is_stale": is_stale,
            "latest_ts": latest_ts.isoformat(),
            # Per-location priors used to impute missing lag features for future horizons.
            "fallback_avg_dow_hour": finalize_averages(loc_avg_dow_hour_sum),
            "fallback_avg_hour": finalize_averages(loc_avg_hour_sum),
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
    weights: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    default = {
        "global": {"q10": 0.0, "q90": 0.0, "count": 0},
        "byHour": {},
        "byHourBlock": {},
        "byDayType": {},
        "byOccupancy": {},
    }
    if y_true.size == 0 or y_pred.size == 0 or not val_times:
        return default

    y_true_arr = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    n = min(y_true_arr.size, y_pred_arr.size, len(val_times))
    if weights is not None:
        n = min(n, np.asarray(weights, dtype=np.float32).reshape(-1).size)
    if n <= 0:
        return default

    y_true_arr = y_true_arr[:n]
    y_pred_arr = y_pred_arr[:n]
    times_n = val_times[:n]
    if weights is None:
        residual_weights = np.ones(n, dtype=np.float32)
    else:
        residual_weights = np.asarray(weights, dtype=np.float32).reshape(-1)[:n]

    residuals = (y_true_arr - y_pred_arr).astype(np.float32)
    point_ratios = y_pred_arr.astype(np.float32)
    finite_mask = (
        np.isfinite(residuals)
        & np.isfinite(point_ratios)
        & np.isfinite(residual_weights)
        & (residual_weights > 0.0)
    )
    if not np.any(finite_mask):
        return default

    residuals = residuals[finite_mask]
    point_ratios = point_ratios[finite_mask]
    residual_weights = residual_weights[finite_mask]
    filtered_times = [times_n[idx] for idx, keep in enumerate(finite_mask.tolist()) if keep]
    if residuals.size == 0 or not filtered_times:
        return default

    profile: Dict[str, object] = {
        "global": {
            "q10": float(weighted_quantile(residuals, INTERVAL_Q_LOW, residual_weights)),
            "q90": float(weighted_quantile(residuals, INTERVAL_Q_HIGH, residual_weights)),
            "count": int(len(residuals)),
        },
        "byHour": {},
        "byHourBlock": {},
        "byDayType": {},
    }

    for hour in range(24):
        idx = [i for i, ts in enumerate(filtered_times) if ts.hour == hour]
        if len(idx) < INTERVAL_MIN_SAMPLES_PER_HOUR:
            continue
        hour_res = residuals[idx]
        hour_w = residual_weights[idx]
        profile["byHour"][str(hour)] = {
            "q10": float(weighted_quantile(hour_res, INTERVAL_Q_LOW, hour_w)),
            "q90": float(weighted_quantile(hour_res, INTERVAL_Q_HIGH, hour_w)),
            "count": int(len(hour_res)),
        }

    by_hour_block = {}
    for block_key in ("overnight", "morning", "midday", "evening", "late"):
        idx = [
            i
            for i, ts in enumerate(filtered_times)
            if hour_block_key(int(ts.hour)) == block_key
        ]
        if len(idx) < INTERVAL_MIN_SAMPLES_PER_HOUR:
            continue
        block_res = residuals[idx]
        block_w = residual_weights[idx]
        by_hour_block[block_key] = {
            "q10": float(weighted_quantile(block_res, INTERVAL_Q_LOW, block_w)),
            "q90": float(weighted_quantile(block_res, INTERVAL_Q_HIGH, block_w)),
            "count": int(len(block_res)),
        }
    profile["byHourBlock"] = by_hour_block

    by_day_type = {}
    for day_key, is_weekend in (("weekday", False), ("weekend", True)):
        idx = [
            i
            for i, ts in enumerate(filtered_times)
            if (int(ts.weekday()) >= 5) == bool(is_weekend)
        ]
        if len(idx) < INTERVAL_MIN_SAMPLES_PER_HOUR:
            continue
        day_res = residuals[idx]
        day_w = residual_weights[idx]
        by_day_type[day_key] = {
            "q10": float(weighted_quantile(day_res, INTERVAL_Q_LOW, day_w)),
            "q90": float(weighted_quantile(day_res, INTERVAL_Q_HIGH, day_w)),
            "count": int(len(day_res)),
        }
    profile["byDayType"] = by_day_type

    by_occupancy = {}
    for occ_key in ("low", "mid", "high"):
        idx = [
            i
            for i, ratio in enumerate(point_ratios.tolist())
            if occupancy_bucket_key_from_ratio(float(ratio)) == occ_key
        ]
        if len(idx) < INTERVAL_MIN_SAMPLES_PER_HOUR:
            continue
        occ_res = residuals[idx]
        occ_w = residual_weights[idx]
        by_occupancy[occ_key] = {
            "q10": float(weighted_quantile(occ_res, INTERVAL_Q_LOW, occ_w)),
            "q90": float(weighted_quantile(occ_res, INTERVAL_Q_HIGH, occ_w)),
            "count": int(len(occ_res)),
        }
    profile["byOccupancy"] = by_occupancy

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
        "gamma": max(0.0, float(MODEL_GAMMA)),
        "lambda": max(0.0, float(MODEL_REG_LAMBDA)),
        "alpha": max(0.0, float(MODEL_REG_ALPHA)),
        "verbosity": 0,
        "nthread": max(1, MODEL_NTHREAD),
        "seed": int(MODEL_TUNING_RANDOM_SEED),
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
    if not LOCATION_BALANCE_WEIGHT_ENABLED:
        return {loc_id: 1.0 for loc_id in normalized_counts}

    positive = [count for count in normalized_counts.values() if count > 0]
    if not positive:
        return {loc_id: 1.0 for loc_id in normalized_counts}

    reference = float(np.median(np.array(positive, dtype=np.float32)))
    if not math.isfinite(reference) or reference <= 0.0:
        return {loc_id: 1.0 for loc_id in normalized_counts}

    power = max(0.0, min(1.5, float(LOCATION_BALANCE_WEIGHT_POWER)))
    min_w = max(0.05, float(LOCATION_BALANCE_WEIGHT_MIN))
    max_w = max(min_w, float(LOCATION_BALANCE_WEIGHT_MAX))
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
    pair_half_life_days = max(1.0, float(RECENCY_HALFLIFE_DAYS))
    pair_min_recency_weight = max(0.0, min(1.0, float(RECENCY_MIN_WEIGHT)))

    features_rows: List[List[float]] = []
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
            if int(loc_samples.get(loc_id, 0) or 0) < MIN_SAMPLES_PER_LOC:
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
                    + max(0.0, float(OCCUPANCY_WEIGHT_ALPHA))
                    * (
                        max(0.0, min(1.2, float(future_ratio)))
                        ** max(1.0, float(OCCUPANCY_WEIGHT_GAMMA))
                    )
                )
                pair_weight = max(
                    float(SENSOR_WEIGHT_MIN),
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

    min_w = max(1e-6, float(MODEL_WEIGHT_CLIP_MIN))
    max_w = max(min_w, float(MODEL_WEIGHT_CLIP_MAX))

    positive = out[out > 0.0]
    if positive.size <= 0:
        return np.ones(out.size, dtype=np.float32)

    if MODEL_WEIGHT_STABILIZATION_ENABLED:
        low_q = max(0.0, min(0.49, float(MODEL_WEIGHT_CLIP_LOWER_Q)))
        high_q = max(low_q + 0.01, min(1.0, float(MODEL_WEIGHT_CLIP_UPPER_Q)))
        q_low = float(np.quantile(positive, low_q))
        q_high = float(np.quantile(positive, high_q))
        lower_bound = max(min_w, q_low)
        upper_bound = min(max_w, max(lower_bound, q_high))
        out = np.clip(out, lower_bound, upper_bound)
    else:
        out = np.clip(out, min_w, max_w)

    if MODEL_WEIGHT_NORMALIZE_MEAN:
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
    max_abs = max(0.0, float(MODEL_FEATURE_ABS_MAX))
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
    if not MODEL_FEATURE_CLIP_ENABLED:
        return None

    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] <= 0:
        return None

    lower_q = max(0.0, min(0.49, float(MODEL_FEATURE_CLIP_LOWER_Q)))
    upper_q = max(lower_q + 0.01, min(1.0, float(MODEL_FEATURE_CLIP_UPPER_Q)))
    min_spread = max(0.0, float(MODEL_FEATURE_CLIP_MIN_SPREAD))

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
    if not MODEL_FEATURE_CLIP_ENABLED:
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
        min_ratio = max(0.0, min(1.0, float(MODEL_MIN_FEATURE_FINITE_RATIO)))
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


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, np.floating):
        value = float(value)
    elif isinstance(value, np.integer):
        value = int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_point_bias_profile(
    times: List[datetime],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray],
) -> Optional[Dict[str, object]]:
    if not MODEL_POINT_BIAS_CORRECTION_ENABLED:
        return None
    if y_true.size == 0 or y_pred.size == 0:
        return None

    y_true_arr = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    n = min(y_true_arr.size, y_pred_arr.size, len(times))
    if n <= 0:
        return None

    residuals = (y_true_arr[:n] - y_pred_arr[:n]).astype(np.float32)
    point_ratios = y_pred_arr[:n].astype(np.float32)
    if weights is None:
        weight_arr = np.ones(n, dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        if w.size < n:
            n = min(n, w.size)
            residuals = residuals[:n]
            point_ratios = point_ratios[:n]
        weight_arr = w[:n] if w.size >= n else np.ones(n, dtype=np.float32)

    filtered_times = times[:n]
    finite_mask = (
        np.isfinite(residuals)
        & np.isfinite(point_ratios)
        & np.isfinite(weight_arr)
        & (weight_arr > 0.0)
    )
    if not np.any(finite_mask):
        return None

    residuals = residuals[finite_mask]
    point_ratios = point_ratios[finite_mask]
    weight_arr = weight_arr[finite_mask]
    filtered_times = [filtered_times[i] for i, keep in enumerate(finite_mask.tolist()) if keep]
    if not filtered_times:
        return None

    def summarize(values: List[float], ws: List[float]) -> Dict[str, object]:
        arr = np.array(values, dtype=np.float32)
        warr = np.array(ws, dtype=np.float32)
        bias = weighted_average(arr, warr)
        max_abs = max(0.0, float(MODEL_POINT_BIAS_MAX_ABS))
        if max_abs > 0.0:
            bias = max(-max_abs, min(max_abs, float(bias)))
        return {
            "bias": float(bias),
            "count": int(len(values)),
        }

    by_hour_values: Dict[str, List[float]] = {}
    by_hour_weights: Dict[str, List[float]] = {}
    by_hour_block_values: Dict[str, List[float]] = {}
    by_hour_block_weights: Dict[str, List[float]] = {}
    by_horizon_values: Dict[str, List[float]] = {}
    by_horizon_weights: Dict[str, List[float]] = {}
    by_day_type_values: Dict[str, List[float]] = {"weekday": [], "weekend": []}
    by_day_type_weights: Dict[str, List[float]] = {"weekday": [], "weekend": []}
    by_occupancy_values: Dict[str, List[float]] = {"low": [], "mid": [], "high": []}
    by_occupancy_weights: Dict[str, List[float]] = {"low": [], "mid": [], "high": []}

    for idx, ts in enumerate(filtered_times):
        residual = float(residuals[idx])
        ratio = float(point_ratios[idx]) if idx < point_ratios.size else 0.0
        w = float(weight_arr[idx])

        hour_key = str(int(ts.hour))
        by_hour_values.setdefault(hour_key, []).append(residual)
        by_hour_weights.setdefault(hour_key, []).append(w)
        block_key = hour_block_key(int(ts.hour))
        by_hour_block_values.setdefault(block_key, []).append(residual)
        by_hour_block_weights.setdefault(block_key, []).append(w)

        horizon_key = horizon_bucket_key(0)
        by_horizon_values.setdefault(horizon_key, []).append(residual)
        by_horizon_weights.setdefault(horizon_key, []).append(w)

        day_type = "weekend" if int(ts.weekday()) >= 5 else "weekday"
        by_day_type_values[day_type].append(residual)
        by_day_type_weights[day_type].append(w)
        occ_key = occupancy_bucket_key_from_ratio(ratio)
        by_occupancy_values.setdefault(occ_key, []).append(residual)
        by_occupancy_weights.setdefault(occ_key, []).append(w)

    min_points = max(1, int(MODEL_POINT_BIAS_MIN_POINTS_PER_SEGMENT))
    min_points_occ = max(1, int(MODEL_POINT_BIAS_MIN_POINTS_PER_OCCUPANCY))
    by_hour = {}
    for key in sorted(by_hour_values.keys(), key=lambda x: int(x)):
        if len(by_hour_values[key]) >= min_points:
            by_hour[key] = summarize(by_hour_values[key], by_hour_weights[key])

    by_hour_block = {}
    for key in ("overnight", "morning", "midday", "evening", "late"):
        if len(by_hour_block_values.get(key, [])) >= min_points:
            by_hour_block[key] = summarize(
                by_hour_block_values[key],
                by_hour_block_weights[key],
            )

    by_horizon = {}
    for key in sorted(by_horizon_values.keys()):
        if len(by_horizon_values[key]) >= min_points:
            by_horizon[key] = summarize(by_horizon_values[key], by_horizon_weights[key])

    by_day_type = {}
    for key in ("weekday", "weekend"):
        if len(by_day_type_values[key]) >= min_points:
            by_day_type[key] = summarize(by_day_type_values[key], by_day_type_weights[key])

    by_occupancy = {}
    for key in ("low", "mid", "high"):
        if len(by_occupancy_values.get(key, [])) >= min_points_occ:
            by_occupancy[key] = summarize(by_occupancy_values[key], by_occupancy_weights[key])

    return {
        "global": summarize(residuals.astype(float).tolist(), weight_arr.astype(float).tolist()),
        "byHour": by_hour,
        "byHourBlock": by_hour_block,
        "byHorizon": by_horizon,
        "byDayType": by_day_type,
        "byOccupancy": by_occupancy,
    }


def point_bias_for_target(
    target: datetime,
    profile: Optional[Dict[str, object]],
    hours_ahead: Optional[float] = None,
    point_ratio: Optional[float] = None,
) -> float:
    if not MODEL_POINT_BIAS_CORRECTION_ENABLED or not profile:
        return 0.0

    min_points = max(1, int(MODEL_POINT_BIAS_MIN_POINTS_PER_SEGMENT))
    min_points_occ = max(1, int(MODEL_POINT_BIAS_MIN_POINTS_PER_OCCUPANCY))
    support_mult = max(1.0, float(MODEL_POINT_BIAS_SUPPORT_TARGET_MULT))
    by_horizon = profile.get("byHorizon", {}) if isinstance(profile, dict) else {}
    by_hour = profile.get("byHour", {}) if isinstance(profile, dict) else {}
    by_hour_block = profile.get("byHourBlock", {}) if isinstance(profile, dict) else {}
    by_day_type = profile.get("byDayType", {}) if isinstance(profile, dict) else {}
    by_occupancy = profile.get("byOccupancy", {}) if isinstance(profile, dict) else {}
    global_stats = profile.get("global", {}) if isinstance(profile, dict) else {}
    global_bias = float(global_stats.get("bias", 0.0) or 0.0) if isinstance(global_stats, dict) else 0.0

    def blended_bias(stats: Optional[Dict[str, object]], required_points: int) -> Optional[float]:
        if not isinstance(stats, dict):
            return None
        count = int(stats.get("count", 0) or 0)
        if count < max(1, int(required_points)):
            return None
        seg_bias = to_float_or_none(stats.get("bias"))
        if seg_bias is None:
            return None
        support_target = max(1.0, float(max(1, int(required_points))) * support_mult)
        support = max(0.0, min(1.0, float(count) / support_target))
        return float(support * float(seg_bias) + (1.0 - support) * float(global_bias))

    if MODEL_POINT_BIAS_OCCUPANCY_ENABLED and point_ratio is not None and isinstance(by_occupancy, dict):
        occ_key = occupancy_bucket_key_from_ratio(float(point_ratio))
        occ_bias = blended_bias(by_occupancy.get(occ_key), min_points_occ)
        if occ_bias is not None:
            return occ_bias

    horizon_hours = forecast_horizon_hours(target, hours_ahead=hours_ahead)
    horizon_stats = (
        by_horizon.get(horizon_bucket_key(horizon_hours))
        if isinstance(by_horizon, dict)
        else None
    )
    horizon_bias = blended_bias(horizon_stats if isinstance(horizon_stats, dict) else None, min_points)
    if horizon_bias is not None:
        return horizon_bias

    hour_stats = by_hour.get(str(int(target.hour))) if isinstance(by_hour, dict) else None
    hour_bias = blended_bias(hour_stats if isinstance(hour_stats, dict) else None, min_points)
    if hour_bias is not None:
        return hour_bias

    block_stats = by_hour_block.get(hour_block_key(int(target.hour))) if isinstance(by_hour_block, dict) else None
    block_bias = blended_bias(block_stats if isinstance(block_stats, dict) else None, min_points)
    if block_bias is not None:
        return block_bias

    day_type = "weekend" if int(target.weekday()) >= 5 else "weekday"
    day_stats = by_day_type.get(day_type) if isinstance(by_day_type, dict) else None
    day_bias = blended_bias(day_stats if isinstance(day_stats, dict) else None, min_points)
    if day_bias is not None:
        return day_bias

    return float(global_bias)


def apply_point_bias_shift(
    p10_ratio: float,
    p50_ratio: float,
    p90_ratio: float,
    bias: float,
) -> Tuple[float, float, float]:
    if not MODEL_POINT_BIAS_CORRECTION_ENABLED:
        return p10_ratio, p50_ratio, p90_ratio
    shift = to_float_or_none(bias)
    if shift is None or shift == 0.0:
        return p10_ratio, p50_ratio, p90_ratio

    p10 = float(p10_ratio) + float(shift)
    p50 = float(p50_ratio) + float(shift)
    p90 = float(p90_ratio) + float(shift)
    if p10 > p90:
        p10, p90 = p90, p10
    p50 = min(max(p50, p10), p90)
    return float(p10), float(p50), float(p90)


def recent_drift_bias_for_target(
    target: datetime,
    profile: Optional[Dict[str, object]],
    hours_ahead: Optional[float] = None,
    point_ratio: Optional[float] = None,
) -> float:
    if not RECENT_DRIFT_BIAS_ENABLED or not isinstance(profile, dict):
        return 0.0

    min_points = max(1, int(RECENT_DRIFT_BIAS_MIN_POINTS_PER_HOUR))
    min_points_occ = max(1, int(RECENT_DRIFT_BIAS_MIN_POINTS_PER_OCCUPANCY))
    max_abs = max(0.0, float(RECENT_DRIFT_BIAS_MAX_ABS))
    blend = max(0.0, min(1.0, float(RECENT_DRIFT_BIAS_BLEND)))
    decay_h = max(0.5, float(RECENT_DRIFT_BIAS_HORIZON_DECAY_HOURS))
    support_mult = max(1.0, float(RECENT_DRIFT_BIAS_SUPPORT_TARGET_MULT))

    raw_bias = 0.0
    chosen_count = 0
    required_points = min_points
    by_hour = profile.get("byHour", {}) if isinstance(profile, dict) else {}
    by_day_type = profile.get("byDayType", {}) if isinstance(profile, dict) else {}
    by_occupancy = profile.get("byOccupancy", {}) if isinstance(profile, dict) else {}

    if point_ratio is not None and isinstance(by_occupancy, dict):
        occ_key = occupancy_bucket_key_from_ratio(float(point_ratio))
        occ_stats = by_occupancy.get(occ_key)
        if isinstance(occ_stats, dict) and int(occ_stats.get("count", 0) or 0) >= min_points_occ:
            raw_bias = float(occ_stats.get("bias", 0.0) or 0.0)
            chosen_count = int(occ_stats.get("count", 0) or 0)
            required_points = min_points_occ

    if chosen_count <= 0:
        hour_stats = by_hour.get(str(int(target.hour))) if isinstance(by_hour, dict) else None
        if isinstance(hour_stats, dict) and int(hour_stats.get("count", 0) or 0) >= min_points:
            raw_bias = float(hour_stats.get("bias", 0.0) or 0.0)
            chosen_count = int(hour_stats.get("count", 0) or 0)
        else:
            day_key = "weekend" if int(target.weekday()) >= 5 else "weekday"
            day_stats = by_day_type.get(day_key) if isinstance(by_day_type, dict) else None
            if isinstance(day_stats, dict) and int(day_stats.get("count", 0) or 0) >= min_points:
                raw_bias = float(day_stats.get("bias", 0.0) or 0.0)
                chosen_count = int(day_stats.get("count", 0) or 0)
            else:
                global_stats = profile.get("global", {}) if isinstance(profile, dict) else {}
                raw_bias = float(global_stats.get("bias", 0.0) or 0.0)
                chosen_count = int(global_stats.get("count", 0) or 0)

    if max_abs > 0.0:
        raw_bias = max(-max_abs, min(max_abs, raw_bias))
    if raw_bias == 0.0 or blend <= 0.0:
        return 0.0

    horizon_hours = max(0.0, forecast_horizon_hours(target, hours_ahead=hours_ahead))
    decay = math.exp(-horizon_hours / decay_h)
    support_target = max(1.0, float(required_points) * support_mult)
    support = max(0.0, min(1.0, float(chosen_count) / support_target))
    return float(raw_bias * blend * decay * support)


def build_regime_mae_profile(
    times: List[datetime],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray],
) -> Optional[Dict[str, object]]:
    if y_true.size == 0 or y_pred.size == 0:
        return None

    y_true_arr = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    n = min(y_true_arr.size, y_pred_arr.size, len(times))
    if n <= 0:
        return None

    abs_err = np.abs(y_true_arr[:n] - y_pred_arr[:n]).astype(np.float32)
    if weights is None:
        weight_arr = np.ones(n, dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        if w.size < n:
            n = min(n, w.size)
            abs_err = abs_err[:n]
        weight_arr = w[:n] if w.size >= n else np.ones(n, dtype=np.float32)

    filtered_times = times[:n]
    mask = np.isfinite(abs_err) & np.isfinite(weight_arr) & (weight_arr > 0.0)
    if not np.any(mask):
        return None

    abs_err = abs_err[mask]
    weight_arr = weight_arr[mask]
    filtered_times = [filtered_times[i] for i, keep in enumerate(mask.tolist()) if keep]
    if not filtered_times:
        return None

    def summarize(vals: List[float], ws: List[float]) -> Dict[str, object]:
        return {
            "mae": float(weighted_average(np.array(vals, dtype=np.float32), np.array(ws, dtype=np.float32))),
            "count": int(len(vals)),
        }

    by_hour_block_values: Dict[str, List[float]] = {}
    by_hour_block_weights: Dict[str, List[float]] = {}
    by_horizon_values: Dict[str, List[float]] = {}
    by_horizon_weights: Dict[str, List[float]] = {}
    by_day_type_values: Dict[str, List[float]] = {"weekday": [], "weekend": []}
    by_day_type_weights: Dict[str, List[float]] = {"weekday": [], "weekend": []}

    for i, ts in enumerate(filtered_times):
        err = float(abs_err[i])
        w = float(weight_arr[i])
        block_key = hour_block_key(int(ts.hour))
        by_hour_block_values.setdefault(block_key, []).append(err)
        by_hour_block_weights.setdefault(block_key, []).append(w)

        horizon_key = horizon_bucket_key(0)
        by_horizon_values.setdefault(horizon_key, []).append(err)
        by_horizon_weights.setdefault(horizon_key, []).append(w)

        day_type = "weekend" if int(ts.weekday()) >= 5 else "weekday"
        by_day_type_values[day_type].append(err)
        by_day_type_weights[day_type].append(w)

    min_points = max(1, int(MODEL_POINT_BIAS_MIN_POINTS_PER_SEGMENT))
    by_hour_block = {}
    for key, vals in by_hour_block_values.items():
        if len(vals) >= min_points:
            by_hour_block[key] = summarize(vals, by_hour_block_weights[key])

    by_horizon = {}
    for key, vals in by_horizon_values.items():
        if len(vals) >= min_points:
            by_horizon[key] = summarize(vals, by_horizon_weights[key])

    by_day_type = {}
    for key in ("weekday", "weekend"):
        vals = by_day_type_values[key]
        if len(vals) >= min_points:
            by_day_type[key] = summarize(vals, by_day_type_weights[key])

    return {
        "global": summarize(abs_err.astype(float).tolist(), weight_arr.astype(float).tolist()),
        "byHourBlock": by_hour_block,
        "byHorizon": by_horizon,
        "byDayType": by_day_type,
    }


def fit_direct_horizon_segment(
    source_ratio: np.ndarray,
    target_ratio: np.ndarray,
    weights: np.ndarray,
) -> Optional[Dict[str, object]]:
    if source_ratio.size == 0 or target_ratio.size == 0 or weights.size == 0:
        return None

    x = np.asarray(source_ratio, dtype=np.float32).reshape(-1)
    y = np.asarray(target_ratio, dtype=np.float32).reshape(-1)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    n = min(x.size, y.size, w.size)
    if n <= 0:
        return None

    x = x[:n]
    y = y[:n]
    w = w[:n]
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    if not np.any(mask):
        return None

    x = x[mask]
    y = y[mask]
    w = w[mask]
    w = stabilize_sample_weights(w)
    n = int(x.size)
    if n <= 0:
        return None

    x_mean = weighted_average(x, w)
    y_mean = weighted_average(y, w)
    cov = weighted_average((x - x_mean) * (y - y_mean), w)
    var = weighted_average((x - x_mean) ** 2, w)

    if var <= 1e-6:
        slope = 1.0
    else:
        slope = cov / var
    slope = max(0.0, min(2.5, float(slope)))

    intercept = float(y_mean) - float(slope) * float(x_mean)
    intercept = max(-0.5, min(0.5, float(intercept)))

    baseline = np.clip(x, 0.0, 1.2)
    adjusted = np.clip(slope * x + intercept, 0.0, 1.2)
    baseline_mae = weighted_average(np.abs(baseline - y), w)
    adjusted_mae = weighted_average(np.abs(adjusted - y), w)
    improvement = max(0.0, float(baseline_mae) - float(adjusted_mae))

    min_pairs = max(1, int(MODEL_DIRECT_HORIZON_MIN_PAIRS))
    support = max(0.0, min(1.0, float(n) / float(max(1, min_pairs * 3))))
    if baseline_mae > 1e-4 and improvement > 0.0:
        quality = max(0.0, min(1.0, float(improvement) / float(baseline_mae)))
    else:
        quality = 0.0

    max_blend = max(0.0, min(1.0, float(MODEL_DIRECT_HORIZON_MAX_BLEND)))
    blend = max_blend * support * quality

    return {
        "count": int(n),
        "slope": float(slope),
        "intercept": float(intercept),
        "baselineMae": float(baseline_mae),
        "mae": float(adjusted_mae),
        "improvement": float(improvement),
        "blend": float(blend),
    }


def build_direct_horizon_profile(
    pairs_by_hours: Dict[int, List[Tuple[float, float, float]]],
) -> Optional[Dict[str, object]]:
    if not MODEL_DIRECT_HORIZON_ENABLED:
        return None
    if not pairs_by_hours:
        return None

    min_pairs = max(1, int(MODEL_DIRECT_HORIZON_MIN_PAIRS))
    segment_min_pairs = max(
        10,
        min(
            min_pairs,
            int(MODEL_DIRECT_HORIZON_SEGMENT_MIN_PAIRS),
        ),
    )
    occupancy_segment_min_pairs = max(
        8,
        min(
            segment_min_pairs,
            int(MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENT_MIN_PAIRS),
        ),
    )
    by_hours: Dict[str, Dict[str, object]] = {}
    all_x: List[float] = []
    all_y: List[float] = []
    all_w: List[float] = []
    all_occ_x: Dict[str, List[float]] = {"low": [], "mid": [], "high": []}
    all_occ_y: Dict[str, List[float]] = {"low": [], "mid": [], "high": []}
    all_occ_w: Dict[str, List[float]] = {"low": [], "mid": [], "high": []}

    for hours in sorted(pairs_by_hours.keys()):
        rows = pairs_by_hours.get(hours) or []
        if not rows:
            continue
        src_vals: List[float] = []
        dst_vals: List[float] = []
        ws_vals: List[float] = []
        occ_x: Dict[str, List[float]] = {"low": [], "mid": [], "high": []}
        occ_y: Dict[str, List[float]] = {"low": [], "mid": [], "high": []}
        occ_w: Dict[str, List[float]] = {"low": [], "mid": [], "high": []}
        for src, dst, w in rows:
            src_num = to_float_or_none(src)
            dst_num = to_float_or_none(dst)
            w_num = to_float_or_none(w)
            if src_num is None or dst_num is None or w_num is None:
                continue
            if w_num <= 0.0:
                continue
            src_vals.append(float(src_num))
            dst_vals.append(float(dst_num))
            ws_vals.append(float(w_num))
            occ_key = occupancy_bucket_key_from_ratio(float(src_num))
            occ_x.setdefault(occ_key, []).append(float(src_num))
            occ_y.setdefault(occ_key, []).append(float(dst_num))
            occ_w.setdefault(occ_key, []).append(float(w_num))

        if len(src_vals) < segment_min_pairs:
            continue

        stats = fit_direct_horizon_segment(
            source_ratio=np.array(src_vals, dtype=np.float32),
            target_ratio=np.array(dst_vals, dtype=np.float32),
            weights=np.array(ws_vals, dtype=np.float32),
        )
        if stats is None:
            continue

        if MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENTS_ENABLED:
            by_occ_payload: Dict[str, Dict[str, object]] = {}
            for occ_key in ("low", "mid", "high"):
                occ_src = occ_x.get(occ_key, [])
                occ_dst = occ_y.get(occ_key, [])
                occ_ws = occ_w.get(occ_key, [])
                if len(occ_src) < occupancy_segment_min_pairs:
                    continue
                occ_stats = fit_direct_horizon_segment(
                    source_ratio=np.array(occ_src, dtype=np.float32),
                    target_ratio=np.array(occ_dst, dtype=np.float32),
                    weights=np.array(occ_ws, dtype=np.float32),
                )
                if occ_stats is not None:
                    by_occ_payload[occ_key] = occ_stats
            if by_occ_payload:
                stats = dict(stats)
                stats["byOccupancy"] = by_occ_payload

        by_hours[str(int(hours))] = stats
        all_x.extend(src_vals)
        all_y.extend(dst_vals)
        all_w.extend(ws_vals)
        for occ_key in ("low", "mid", "high"):
            all_occ_x.setdefault(occ_key, []).extend(occ_x.get(occ_key, []))
            all_occ_y.setdefault(occ_key, []).extend(occ_y.get(occ_key, []))
            all_occ_w.setdefault(occ_key, []).extend(occ_w.get(occ_key, []))

    if not by_hours:
        return None

    global_stats = fit_direct_horizon_segment(
        source_ratio=np.array(all_x, dtype=np.float32),
        target_ratio=np.array(all_y, dtype=np.float32),
        weights=np.array(all_w, dtype=np.float32),
    )
    if MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENTS_ENABLED and isinstance(global_stats, dict):
        global_by_occ: Dict[str, Dict[str, object]] = {}
        for occ_key in ("low", "mid", "high"):
            occ_src = all_occ_x.get(occ_key, [])
            occ_dst = all_occ_y.get(occ_key, [])
            occ_ws = all_occ_w.get(occ_key, [])
            if len(occ_src) < occupancy_segment_min_pairs:
                continue
            occ_stats = fit_direct_horizon_segment(
                source_ratio=np.array(occ_src, dtype=np.float32),
                target_ratio=np.array(occ_dst, dtype=np.float32),
                weights=np.array(occ_ws, dtype=np.float32),
            )
            if occ_stats is not None:
                global_by_occ[occ_key] = occ_stats
        if global_by_occ:
            global_stats = dict(global_stats)
            global_stats["byOccupancy"] = global_by_occ

    return {
        "minPairs": int(min_pairs),
        "segmentMinPairs": int(segment_min_pairs),
        "occupancySegmentsEnabled": bool(MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENTS_ENABLED),
        "occupancySegmentMinPairs": int(occupancy_segment_min_pairs),
        "maxBlend": max(0.0, min(1.0, float(MODEL_DIRECT_HORIZON_MAX_BLEND))),
        "byHours": by_hours,
        "global": global_stats,
    }


def feature_group_slices(loc_count: int) -> Dict[str, Tuple[int, int]]:
    time_end = 17 + CALENDAR_FEATURE_COUNT
    lag_end = time_end + LAG_TREND_SENSOR_FEATURE_COUNT + SCHEDULE_PHASE_FEATURE_COUNT
    weather_end = (
        lag_end
        + (len(WEATHER_KEYS) * 3)
        + len(WEATHER_ROLLING_KEYS)
        + WEATHER_DERIVED_FEATURE_COUNT
        + WEATHER_QUALITY_FEATURE_COUNT
    )
    onehot_end = weather_end + max(0, int(loc_count))
    return {
        "time_calendar": (0, time_end),
        "lags_trends_sensor": (time_end, lag_end),
        "weather": (lag_end, weather_end),
        "onehot_location": (weather_end, onehot_end),
    }


def feature_missingness_summary(
    X: np.ndarray,
    loc_count: int,
) -> Dict[str, object]:
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        return {
            "rows": int(X.shape[0]) if X.ndim == 2 else 0,
            "features": int(X.shape[1]) if X.ndim == 2 else 0,
            "globalMissingRate": 1.0,
            "allMissingRowsRate": 1.0,
            "groups": {},
        }

    finite = np.isfinite(X)
    total_cells = max(1, int(finite.size))
    finite_cells = int(np.count_nonzero(finite))
    global_missing = 1.0 - (float(finite_cells) / float(total_cells))
    row_has_signal = np.any(finite, axis=1)
    all_missing_rows_rate = 1.0 - (
        float(np.count_nonzero(row_has_signal)) / float(max(1, finite.shape[0]))
    )

    groups: Dict[str, Dict[str, object]] = {}
    for name, (start, end) in feature_group_slices(loc_count).items():
        if start >= end or start < 0 or end > X.shape[1]:
            continue
        part = finite[:, start:end]
        part_total = max(1, int(part.size))
        part_finite = int(np.count_nonzero(part))
        groups[name] = {
            "start": int(start),
            "end": int(end),
            "features": int(end - start),
            "missingRate": float(1.0 - (float(part_finite) / float(part_total))),
        }

    return {
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "globalMissingRate": float(global_missing),
        "allMissingRowsRate": float(all_missing_rows_rate),
        "groups": groups,
    }


def feature_missingness_block_reason(
    summary: Dict[str, object],
) -> Optional[str]:
    if not MODEL_FEATURE_MISSING_GUARD_ENABLED:
        return None

    rows = int(summary.get("rows", 0) or 0)
    if rows < max(1, MODEL_FEATURE_MISSING_MIN_ROWS):
        return None

    global_missing = to_float_or_none(summary.get("globalMissingRate"))
    if global_missing is not None and global_missing > float(MODEL_MAX_FEATURE_MISSING_RATE):
        return "global_missing_rate_high"

    groups = summary.get("groups", {})
    if not isinstance(groups, dict):
        return None

    lag_missing = to_float_or_none((groups.get("lags_trends_sensor") or {}).get("missingRate"))
    if lag_missing is not None and lag_missing > float(MODEL_MAX_LAG_MISSING_RATE):
        return "lag_feature_missing_rate_high"

    weather_missing = to_float_or_none((groups.get("weather") or {}).get("missingRate"))
    if weather_missing is not None and weather_missing > float(MODEL_MAX_WEATHER_MISSING_RATE):
        return "weather_feature_missing_rate_high"

    return None


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
    if not math.isfinite(float(baseline_mae)):
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
        if not math.isfinite(float(mae)):
            continue
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


def tuning_params_complexity_score(params: Dict[str, object]) -> float:
    depth_ref = max(1, int(MODEL_TUNING_COMPLEXITY_DEPTH_REF))
    depth = max(1.0, float(params.get("max_depth", MODEL_MAX_DEPTH)))
    eta = max(0.001, float(params.get("eta", MODEL_ETA)))
    min_child = max(0.0, float(params.get("min_child_weight", MODEL_MIN_CHILD_WEIGHT)))
    subsample = max(0.1, min(1.0, float(params.get("subsample", MODEL_SUBSAMPLE))))
    colsample = max(0.1, min(1.0, float(params.get("colsample_bytree", MODEL_COLSAMPLE_BYTREE))))
    gamma = max(0.0, float(params.get("gamma", MODEL_GAMMA)))
    reg_lambda = max(0.0, float(params.get("lambda", MODEL_REG_LAMBDA)))
    reg_alpha = max(0.0, float(params.get("alpha", MODEL_REG_ALPHA)))
    max_bin = max(64.0, float(params.get("max_bin", MODEL_MAX_BIN)))

    depth_term = max(0.0, (depth - float(depth_ref)) / float(depth_ref))
    eta_term = max(0.0, (eta - float(MODEL_ETA)) / max(0.01, float(MODEL_ETA)))
    child_term = max(0.0, (1.0 - min_child) / 1.0)
    subsample_term = max(0.0, (0.9 - subsample) / 0.4)
    colsample_term = max(0.0, (0.9 - colsample) / 0.4)
    gamma_term = max(0.0, (0.2 - gamma) / 0.2)
    lambda_term = max(0.0, (1.0 - reg_lambda) / 1.0)
    alpha_term = max(0.0, (0.1 - reg_alpha) / 0.1)
    bin_term = max(0.0, (max_bin - float(MODEL_MAX_BIN)) / max(64.0, float(MODEL_MAX_BIN)))

    score = (
        0.30 * depth_term
        + 0.25 * eta_term
        + 0.20 * child_term
        + 0.10 * subsample_term
        + 0.10 * colsample_term
        + 0.07 * gamma_term
        + 0.07 * lambda_term
        + 0.07 * alpha_term
        + 0.04 * bin_term
    )
    if not math.isfinite(score):
        return 0.0
    return max(0.0, float(score))


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

    interval_weight = max(0.0, float(MODEL_TUNING_INTERVAL_ERR_WEIGHT))
    tail_weight = max(0.0, float(MODEL_TUNING_TAIL_ERR_WEIGHT))
    tail_q = max(0.5, min(0.99, float(MODEL_TUNING_TAIL_QUANTILE)))
    complexity_weight = max(0.0, float(MODEL_TUNING_COMPLEXITY_WEIGHT))
    complexity_penalty = tuning_params_complexity_score(params)
    target_coverage = max(0.1, min(0.98, float(INTERVAL_Q_HIGH - INTERVAL_Q_LOW)))

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
        abs_err = np.abs(preds - y_val)
        fold_mae = weighted_average(abs_err, w_val)
        if not math.isfinite(float(fold_mae)):
            return float("inf")

        train_preds = model.predict(dtrain)
        train_residuals = (y_train - train_preds).astype(np.float32)
        finite_train_mask = np.isfinite(train_residuals) & np.isfinite(w_train) & (w_train > 0.0)
        if int(np.count_nonzero(finite_train_mask)) >= 20:
            q10 = float(
                weighted_quantile(
                    train_residuals[finite_train_mask],
                    INTERVAL_Q_LOW,
                    w_train[finite_train_mask],
                )
            )
            q90 = float(
                weighted_quantile(
                    train_residuals[finite_train_mask],
                    INTERVAL_Q_HIGH,
                    w_train[finite_train_mask],
                )
            )
        else:
            q10 = 0.0
            q90 = 0.0

        p10 = preds + q10
        p90 = preds + q90
        p10, p90 = ordered_prediction_bounds(p10, p90)
        within = ((y_val >= p10) & (y_val <= p90)).astype(np.float32)
        coverage = weighted_average(within, w_val)
        interval_err = abs(float(coverage) - target_coverage)

        tail_cut = float(weighted_quantile(abs_err, tail_q, w_val)) if abs_err.size > 0 else 0.0
        tail_mask = abs_err >= tail_cut
        tail_mae = weighted_average(abs_err[tail_mask], w_val[tail_mask]) if np.any(tail_mask) else 0.0

        fold_score = float(fold_mae) + interval_weight * float(interval_err) + tail_weight * float(tail_mae)
        cv_errors.append(float(fold_score))

    if not cv_errors:
        return float("inf")
    score = float(sum(cv_errors) / len(cv_errors))
    if complexity_weight > 0.0 and math.isfinite(complexity_penalty):
        score += float(complexity_weight) * float(complexity_penalty)
    return score


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
        if "gamma" in overrides:
            out["gamma"] = float(overrides["gamma"])
        if "lambda" in overrides:
            out["lambda"] = float(overrides["lambda"])
        elif "reg_lambda" in overrides:
            out["lambda"] = float(overrides["reg_lambda"])
        if "alpha" in overrides:
            out["alpha"] = float(overrides["alpha"])
        elif "reg_alpha" in overrides:
            out["alpha"] = float(overrides["reg_alpha"])
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
            "gamma": float(params.get("gamma", MODEL_GAMMA)),
            "lambda": float(params.get("lambda", MODEL_REG_LAMBDA)),
            "alpha": float(params.get("alpha", MODEL_REG_ALPHA)),
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
            round(float(cand.get("gamma", MODEL_GAMMA)), 6),
            round(float(cand.get("lambda", MODEL_REG_LAMBDA)), 6),
            round(float(cand.get("alpha", MODEL_REG_ALPHA)), 6),
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
        cand["gamma"] = max(0.0, min(8.0, float(cand.get("gamma", MODEL_GAMMA))))
        cand["lambda"] = max(0.0, min(20.0, float(cand.get("lambda", MODEL_REG_LAMBDA))))
        cand["alpha"] = max(0.0, min(10.0, float(cand.get("alpha", MODEL_REG_ALPHA))))
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
        gamma0 = float(anchor.get("gamma", MODEL_GAMMA))
        lambda0 = float(anchor.get("lambda", MODEL_REG_LAMBDA))
        alpha0 = float(anchor.get("alpha", MODEL_REG_ALPHA))
        bin0 = int(anchor.get("max_bin", MODEL_MAX_BIN))

        depths = sorted(set([max(2, depth0 - 2), max(2, depth0 - 1), depth0, depth0 + 1]))
        etas = sorted(set([max(0.01, eta0 * 0.7), max(0.01, eta0 * 0.85), eta0, min(0.2, eta0 * 1.15)]))
        childs = sorted(set([max(0.0, child0 * 0.5), child0, max(1.0, child0 * 2.0)]))
        subsamples = sorted(set([max(0.6, sub0 - 0.15), sub0, min(1.0, sub0 + 0.1)]))
        colsamples = sorted(set([max(0.6, col0 - 0.15), col0, min(1.0, col0 + 0.1)]))
        gammas = sorted(set([max(0.0, gamma0 * 0.5), gamma0, min(8.0, gamma0 + 0.8)]))
        lambdas = sorted(set([max(0.0, lambda0 * 0.6), lambda0, min(20.0, max(0.5, lambda0 * 1.8))]))
        alphas = sorted(set([max(0.0, alpha0 * 0.5), alpha0, min(10.0, alpha0 + 0.6)]))
        bins = sorted(set([max(64, int(round(bin0 * 0.75))), bin0, max(64, int(round(bin0 * 1.25)))]))

        grid = []
        for depth in depths:
            for eta in etas:
                for child in childs:
                    for subsample in subsamples:
                        for colsample in colsamples:
                            for gamma in gammas:
                                for reg_lambda in lambdas:
                                    for reg_alpha in alphas:
                                        for max_bin in bins:
                                            grid.append(
                                                {
                                                    "max_depth": depth,
                                                    "eta": eta,
                                                    "min_child_weight": child,
                                                    "subsample": subsample,
                                                    "colsample_bytree": colsample,
                                                    "gamma": gamma,
                                                    "lambda": reg_lambda,
                                                    "alpha": reg_alpha,
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
                "gamma": rng.uniform(0.0, 2.0),
                "lambda": rng.uniform(0.0, 8.0),
                "alpha": rng.uniform(0.0, 4.0),
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
    num_boost_round_override: Optional[int] = None,
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
        rounds = max(1, int(num_boost_round_override or MODEL_NUM_BOOST_ROUND))
        if dval is not None and num_boost_round_override is None:
            return xgb.train(
                qparams,
                dtrain,
                num_boost_round=rounds,
                evals=[(dval, "val")],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
            )
        return xgb.train(
            qparams,
            dtrain,
            num_boost_round=rounds,
            verbose_eval=False,
        )
    except Exception:
        return None


def selected_boost_rounds(
    model: Optional[xgb.Booster],
    fallback_rounds: int,
) -> int:
    rounds = max(1, int(fallback_rounds))
    if model is None:
        return rounds

    try:
        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is not None:
            rounds = max(1, int(best_iteration) + 1)
    except Exception:
        pass

    if rounds <= 1:
        try:
            attrs = model.attributes()
            raw = attrs.get("best_iteration") if isinstance(attrs, dict) else None
            if raw is not None:
                rounds = max(1, int(raw) + 1)
        except Exception:
            pass

    return max(1, min(int(MODEL_NUM_BOOST_ROUND), int(rounds)))


def train_model_unit(
    model_key: str,
    loc_ids: Iterable[int],
    loc_data: Dict[int, Dict[str, object]],
    onehot: Dict[int, List[float]],
    loc_samples: Dict[int, int],
    weather_source: Optional[Dict[str, object]],
    preferred_params: Optional[Dict[str, object]] = None,
    dataset_cache: Optional[Dict[Tuple[object, ...], Dict[str, object]]] = None,
    core_feature_cache: Optional[Dict[Tuple[int, datetime], List[float]]] = None,
):
    loc_ids = sorted(set(int(loc_id) for loc_id in loc_ids))
    location_balance_map = build_location_balance_weight_map_from_counts(
        {loc_id: int(loc_samples.get(loc_id, 0) or 0) for loc_id in loc_ids}
    )
    weather_lookup_cache: Optional[Dict[Tuple[datetime, str], float]] = (
        {} if weather_source is not None else None
    )
    train_dataset = build_model_observation_dataset(
        loc_ids=loc_ids,
        loc_data=loc_data,
        onehot=onehot,
        weather_source=weather_source,
        location_balance_map=location_balance_map,
        since=None,
        loc_samples=loc_samples,
        require_min_samples=True,
        exclude_stale=True,
        include_direct_horizon_pairs=True,
        weather_lookup_cache=weather_lookup_cache,
        core_feature_cache=core_feature_cache,
        dataset_cache=dataset_cache,
        cache_key=(str(model_key), "__all__", "train"),
    )
    times = list(train_dataset.get("times", []))
    X = np.asarray(train_dataset.get("X"), dtype=np.float32)
    y = np.asarray(train_dataset.get("y"), dtype=np.float32).reshape(-1)
    quality_weights = np.asarray(train_dataset.get("rowQualityWeights"), dtype=np.float32).reshape(-1)
    direct_horizon_pairs = (
        train_dataset.get("directHorizonPairs", {})
        if isinstance(train_dataset.get("directHorizonPairs", {}), dict)
        else {}
    )

    base_skip_metrics = {
        "model_key": model_key,
        "val_rows": 0,
        "val_mae": None,
        "val_rmse": None,
        "val_interval_coverage": None,
        "val_interval_coverage_error": None,
        "selected_boost_rounds": None,
        "tuning_cv_mae": None,
        "tuning_cv_objective": None,
        "tuned_candidates": 0,
        "quantile_direct": False,
        "feature_ablation": None,
        "holdout_rows": 0,
        "holdout_mae": None,
        "holdout_rmse": None,
        "holdout_interval_coverage": None,
        "holdout_interval_coverage_error": None,
        "feature_missingness": None,
        "feature_quality_blocked": False,
        "feature_quality_reason": None,
        "feature_fill_values": None,
        "feature_clip_lower": None,
        "feature_clip_upper": None,
        "point_bias_profile": None,
        "regime_profile": None,
        "direct_horizon_profile": None,
    }

    if len(times) < MIN_TRAIN_SAMPLES:
        return None, {
            **base_skip_metrics,
            "train_rows": len(times),
            "best_params": params_for_meta(normalize_tuning_overrides(preferred_params)),
            "invalid_rows_dropped": 0,
        }, None

    weights = stabilize_sample_weights(
        build_recency_weights(times) * quality_weights * build_occupancy_weights(y)
    )
    valid_mask = supervised_row_mask(X, y, weights)
    invalid_rows_dropped = int(valid_mask.size - int(np.count_nonzero(valid_mask)))
    if invalid_rows_dropped > 0:
        X = X[valid_mask]
        y = y[valid_mask]
        weights = weights[valid_mask]
        keep_indices = np.flatnonzero(valid_mask).tolist()
        times = [times[idx] for idx in keep_indices]

    if len(times) < MIN_TRAIN_SAMPLES:
        return None, {
            **base_skip_metrics,
            "train_rows": len(times),
            "best_params": params_for_meta(normalize_tuning_overrides(preferred_params)),
            "invalid_rows_dropped": invalid_rows_dropped,
        }, None

    feature_missingness = feature_missingness_summary(X, len(loc_ids))
    missing_block_reason = feature_missingness_block_reason(feature_missingness)
    if missing_block_reason:
        return None, {
            **base_skip_metrics,
            "train_rows": len(times),
            "best_params": params_for_meta(normalize_tuning_overrides(preferred_params)),
            "invalid_rows_dropped": invalid_rows_dropped,
            "feature_missingness": feature_missingness,
            "feature_quality_blocked": True,
            "feature_quality_reason": missing_block_reason,
        }, None

    n_rows = len(times)
    holdout_idx: List[int] = []
    model_idx: List[int] = list(range(n_rows))
    holdout_fraction = max(0.0, min(0.45, float(MODEL_HOLDOUT_SPLIT)))
    holdout_target = max(0, int(round(n_rows * holdout_fraction)))
    min_holdout_rows = max(1, int(MODEL_HOLDOUT_MIN_ROWS))
    if holdout_fraction > 0.0 and holdout_target > 0:
        holdout_target = max(holdout_target, min_holdout_rows)
        if n_rows >= (MIN_TRAIN_SAMPLES + holdout_target):
            times_sorted = sorted(times)
            holdout_start = max(1, min(n_rows - 1, n_rows - holdout_target))
            holdout_time = times_sorted[holdout_start]
            candidate_model_idx = [i for i, ts in enumerate(times) if ts < holdout_time]
            candidate_holdout_idx = [i for i, ts in enumerate(times) if ts >= holdout_time]
            if (
                len(candidate_model_idx) >= MIN_TRAIN_SAMPLES
                and len(candidate_holdout_idx) >= min_holdout_rows
            ):
                model_idx = candidate_model_idx
                holdout_idx = candidate_holdout_idx

    fill_source_idx = model_idx if model_idx else list(range(n_rows))
    feature_fill_values = compute_feature_fill_values(X[fill_source_idx])
    X = apply_feature_fill_values(X, feature_fill_values)
    feature_clip_bounds = compute_feature_clip_bounds(X[fill_source_idx])
    X = apply_feature_clip_bounds(X, feature_clip_bounds)

    X_model = X[model_idx]
    y_model = y[model_idx]
    w_model = weights[model_idx]
    times_model = [times[i] for i in model_idx]

    params, tuning_score, tuned_candidates = choose_best_params(
        X_model,
        y_model,
        w_model,
        times_model,
        preferred_params=preferred_params,
    )

    times_sorted = sorted(times_model)
    split_idx = int(len(times_sorted) * TRAIN_SPLIT)
    split_idx = max(1, min(split_idx, len(times_sorted) - 1))
    split_time = times_sorted[split_idx]

    train_idx = [i for i, ts in enumerate(times_model) if ts < split_time]
    val_idx = [i for i, ts in enumerate(times_model) if ts >= split_time]

    val_mae = None
    val_rmse = None
    val_rows = 0
    val_interval_coverage = None
    val_interval_coverage_error = None
    feature_ablation = None
    interval_profile = None
    point_bias_profile = None
    regime_profile = None
    direct_horizon_profile = None
    selected_rounds = max(1, int(MODEL_NUM_BOOST_ROUND))
    dtrain_full = xgb.DMatrix(X_model, label=y_model, weight=w_model)

    dval = None
    if len(train_idx) < 10 or len(val_idx) < 10:
        p50_model = xgb.train(
            params,
            dtrain_full,
            num_boost_round=MODEL_NUM_BOOST_ROUND,
            verbose_eval=False,
        )
        selected_rounds = selected_boost_rounds(p50_model, MODEL_NUM_BOOST_ROUND)
        train_rows = len(times_model)
    else:
        X_train = X_model[train_idx]
        y_train = y_model[train_idx]
        w_train = w_model[train_idx]
        X_val = X_model[val_idx]
        y_val = y_model[val_idx]
        w_val = w_model[val_idx]
        val_times = [times_model[i] for i in val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
        dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)

        p50_eval_model = xgb.train(
            params,
            dtrain,
            num_boost_round=MODEL_NUM_BOOST_ROUND,
            evals=[(dval, "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        selected_rounds = selected_boost_rounds(p50_eval_model, MODEL_NUM_BOOST_ROUND)

        preds = p50_eval_model.predict(dval)
        abs_errors = np.abs(preds - y_val)
        sq_errors = (preds - y_val) ** 2
        val_mae = weighted_average(abs_errors, w_val)
        val_rmse = math.sqrt(weighted_average(sq_errors, w_val))
        val_rows = len(val_idx)

        train_preds_for_interval = p50_eval_model.predict(dtrain)
        train_residuals_for_interval = (y_train - train_preds_for_interval).astype(np.float32)
        train_residual_mask = np.isfinite(train_residuals_for_interval) & np.isfinite(w_train) & (w_train > 0.0)
        if int(np.count_nonzero(train_residual_mask)) >= 20:
            q10_val = float(
                weighted_quantile(
                    train_residuals_for_interval[train_residual_mask],
                    INTERVAL_Q_LOW,
                    w_train[train_residual_mask],
                )
            )
            q90_val = float(
                weighted_quantile(
                    train_residuals_for_interval[train_residual_mask],
                    INTERVAL_Q_HIGH,
                    w_train[train_residual_mask],
                )
            )
        else:
            q10_val = 0.0
            q90_val = 0.0
        p10_val = preds + q10_val
        p90_val = preds + q90_val
        p10_val, p90_val = ordered_prediction_bounds(p10_val, p90_val)
        target_coverage = max(0.1, min(0.98, float(INTERVAL_Q_HIGH - INTERVAL_Q_LOW)))
        within_val = ((y_val >= p10_val) & (y_val <= p90_val)).astype(np.float32)
        val_interval_coverage = weighted_average(within_val, w_val)
        val_interval_coverage_error = abs(float(val_interval_coverage) - float(target_coverage))

        interval_profile = build_interval_profile(val_times, y_val, preds, weights=w_val)
        point_bias_profile = build_point_bias_profile(
            times=val_times,
            y_true=y_val,
            y_pred=preds,
            weights=w_val,
        )
        regime_profile = build_regime_mae_profile(
            times=val_times,
            y_true=y_val,
            y_pred=preds,
            weights=w_val,
        )
        feature_ablation = evaluate_feature_ablation(
            p50_model=p50_eval_model,
            X_val=X_val,
            y_val=y_val,
            w_val=w_val,
            loc_count=len(loc_ids),
            baseline_mae=float(val_mae),
        )
        p50_model = xgb.train(
            params,
            dtrain_full,
            num_boost_round=selected_rounds,
            verbose_eval=False,
        )
        train_rows = len(times_model)

    p10_model = train_quantile_model(
        0.10,
        params,
        dtrain_full,
        None,
        num_boost_round_override=selected_rounds,
    )
    p90_model = train_quantile_model(
        0.90,
        params,
        dtrain_full,
        None,
        num_boost_round_override=selected_rounds,
    )
    quantile_direct = bool(p10_model is not None and p90_model is not None)

    holdout_rows = 0
    holdout_mae = None
    holdout_rmse = None
    holdout_interval_coverage = None
    holdout_interval_coverage_error = None
    if holdout_idx:
        X_holdout = X[holdout_idx]
        y_holdout = y[holdout_idx]
        w_holdout = weights[holdout_idx]
        holdout_times = [times[i] for i in holdout_idx]
        holdout_hours = [int(ts.hour) for ts in holdout_times]
        dholdout = xgb.DMatrix(X_holdout)
        p50_holdout = p50_model.predict(dholdout).astype(np.float32)
        if p10_model is not None and p90_model is not None:
            p10_holdout = p10_model.predict(dholdout).astype(np.float32)
            p90_holdout = p90_model.predict(dholdout).astype(np.float32)
        else:
            p10_vals: List[float] = []
            p90_vals: List[float] = []
            for pred, hour, ts in zip(p50_holdout.tolist(), holdout_hours, holdout_times):
                b10, _mid, b90 = interval_bounds(
                    point_ratio=float(pred),
                    hour=int(hour),
                    residual_profile=interval_profile,
                    target=ts,
                )
                p10_vals.append(float(b10))
                p90_vals.append(float(b90))
            p10_holdout = np.array(p10_vals, dtype=np.float32)
            p90_holdout = np.array(p90_vals, dtype=np.float32)

        p10_holdout, p90_holdout = ordered_prediction_bounds(p10_holdout, p90_holdout)
        finite_mask = (
            np.isfinite(y_holdout)
            & np.isfinite(w_holdout)
            & np.isfinite(p50_holdout)
            & np.isfinite(p10_holdout)
            & np.isfinite(p90_holdout)
            & (w_holdout > 0.0)
        )
        if np.any(finite_mask):
            y_holdout = y_holdout[finite_mask]
            w_holdout = w_holdout[finite_mask]
            p50_holdout = p50_holdout[finite_mask]
            p10_holdout = p10_holdout[finite_mask]
            p90_holdout = p90_holdout[finite_mask]
            filtered_holdout_times = [
                holdout_times[idx]
                for idx, keep in enumerate(finite_mask.tolist())
                if keep
            ]

            holdout_rows = int(y_holdout.size)
            holdout_abs = np.abs(p50_holdout - y_holdout)
            holdout_sq = (p50_holdout - y_holdout) ** 2
            holdout_mae = weighted_average(holdout_abs, w_holdout)
            holdout_rmse = math.sqrt(weighted_average(holdout_sq, w_holdout))
            target_coverage = max(0.1, min(0.98, float(INTERVAL_Q_HIGH - INTERVAL_Q_LOW)))
            holdout_within = ((y_holdout >= p10_holdout) & (y_holdout <= p90_holdout)).astype(np.float32)
            holdout_interval_coverage = weighted_average(holdout_within, w_holdout)
            holdout_interval_coverage_error = abs(
                float(holdout_interval_coverage) - float(target_coverage)
            )

            if interval_profile is None:
                interval_profile = build_interval_profile(
                    filtered_holdout_times,
                    y_holdout,
                    p50_holdout,
                    weights=w_holdout,
                )
            holdout_bias_profile = build_point_bias_profile(
                times=filtered_holdout_times,
                y_true=y_holdout,
                y_pred=p50_holdout,
                weights=w_holdout,
            )
            if holdout_bias_profile is not None:
                point_bias_profile = holdout_bias_profile
            holdout_regime_profile = build_regime_mae_profile(
                times=filtered_holdout_times,
                y_true=y_holdout,
                y_pred=p50_holdout,
                weights=w_holdout,
            )
            if holdout_regime_profile is not None:
                regime_profile = holdout_regime_profile

    direct_horizon_profile = build_direct_horizon_profile(direct_horizon_pairs)

    model_bundle = {
        "p50": p50_model,
        "p10": p10_model,
        "p90": p90_model,
        "quantileDirect": quantile_direct,
    }
    return model_bundle, {
        "model_key": model_key,
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_interval_coverage": val_interval_coverage,
        "val_interval_coverage_error": val_interval_coverage_error,
        "selected_boost_rounds": int(selected_rounds),
        "tuning_cv_mae": tuning_score,
        "tuning_cv_objective": tuning_score,
        "tuned_candidates": tuned_candidates,
        "quantile_direct": quantile_direct,
        "best_params": params_for_meta(params),
        "feature_ablation": feature_ablation,
        "invalid_rows_dropped": invalid_rows_dropped,
        "holdout_rows": int(holdout_rows),
        "holdout_mae": holdout_mae,
        "holdout_rmse": holdout_rmse,
        "holdout_interval_coverage": holdout_interval_coverage,
        "holdout_interval_coverage_error": holdout_interval_coverage_error,
        "feature_missingness": feature_missingness,
        "feature_quality_blocked": False,
        "feature_quality_reason": None,
        "feature_fill_values": feature_fill_values.astype(np.float32).tolist(),
        "feature_clip_lower": (
            feature_clip_bounds[0].astype(np.float32).tolist()
            if isinstance(feature_clip_bounds, (tuple, list)) and len(feature_clip_bounds) >= 2
            else None
        ),
        "feature_clip_upper": (
            feature_clip_bounds[1].astype(np.float32).tolist()
            if isinstance(feature_clip_bounds, (tuple, list)) and len(feature_clip_bounds) >= 2
            else None
        ),
        "point_bias_profile": point_bias_profile,
        "regime_profile": regime_profile,
        "direct_horizon_profile": direct_horizon_profile,
    }, interval_profile


def predict_model_bundle_on_feature_matrix(
    model_bundle: Dict[str, object],
    X: np.ndarray,
    times: List[datetime],
    hours: List[int],
    residual_profile: Optional[Dict[str, object]],
    point_bias_profile: Optional[Dict[str, object]],
    direct_horizon_profile: Optional[Dict[str, object]],
    feature_fill_values: Optional[np.ndarray],
    feature_clip_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    include_intervals: bool = True,
) -> Optional[Dict[str, np.ndarray]]:
    p50_model = model_bundle.get("p50")
    if p50_model is None:
        return None

    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim != 2 or X_arr.shape[0] <= 0:
        return None

    row_count = min(int(X_arr.shape[0]), len(times), len(hours))
    if row_count <= 0:
        return None

    X_arr = X_arr[:row_count]
    times_n = times[:row_count]
    hours_n = hours[:row_count]
    features_arr = apply_feature_fill_values(X_arr, feature_fill_values)
    features_arr = apply_feature_clip_bounds(features_arr, feature_clip_bounds)
    dmat = xgb.DMatrix(features_arr)
    p50 = p50_model.predict(dmat).astype(np.float32)
    p50 = p50[:row_count]

    p10: Optional[np.ndarray] = None
    p90: Optional[np.ndarray] = None
    if include_intervals:
        p10_model = model_bundle.get("p10")
        p90_model = model_bundle.get("p90")
        if p10_model is not None and p90_model is not None:
            p10 = p10_model.predict(dmat).astype(np.float32)[:row_count]
            p90 = p90_model.predict(dmat).astype(np.float32)[:row_count]
        else:
            p10_vals: List[float] = []
            p90_vals: List[float] = []
            for pred, hour, ts in zip(p50.tolist(), hours_n, times_n):
                b10, _mid, b90 = interval_bounds(
                    point_ratio=float(pred),
                    hour=int(hour),
                    residual_profile=residual_profile,
                    target=ts,
                )
                p10_vals.append(float(b10))
                p90_vals.append(float(b90))
            p10 = np.array(p10_vals, dtype=np.float32)
            p90 = np.array(p90_vals, dtype=np.float32)

        if p10 is not None and p90 is not None:
            p10, p90 = ordered_prediction_bounds(p10, p90)

    if direct_horizon_profile:
        if include_intervals and p10 is not None and p90 is not None:
            adj_p10: List[float] = []
            adj_p50: List[float] = []
            adj_p90: List[float] = []
            for idx, ts in enumerate(times_n):
                cur_p10 = float(p10[idx]) if idx < p10.size else float(p50[idx])
                cur_p50 = float(p50[idx]) if idx < p50.size else float(cur_p10)
                cur_p90 = float(p90[idx]) if idx < p90.size else float(cur_p50)
                cur_p10, cur_p50, cur_p90 = apply_direct_horizon_adjustment(
                    cur_p10,
                    cur_p50,
                    cur_p90,
                    target=ts,
                    profile=direct_horizon_profile,
                    hours_ahead=0.0,
                )
                adj_p10.append(float(cur_p10))
                adj_p50.append(float(cur_p50))
                adj_p90.append(float(cur_p90))
            p10 = np.array(adj_p10, dtype=np.float32)
            p50 = np.array(adj_p50, dtype=np.float32)
            p90 = np.array(adj_p90, dtype=np.float32)
        else:
            adjusted = []
            for pred, ts in zip(p50.tolist(), times_n):
                _lo, center, _hi = apply_direct_horizon_adjustment(
                    float(pred),
                    float(pred),
                    float(pred),
                    target=ts,
                    profile=direct_horizon_profile,
                    hours_ahead=0.0,
                )
                adjusted.append(float(center))
            p50 = np.array(adjusted, dtype=np.float32)

    if point_bias_profile:
        bias_vals = np.array(
            [
                point_bias_for_target(
                    ts,
                    point_bias_profile,
                    hours_ahead=0.0,
                    point_ratio=float(p50[idx]) if idx < p50.size else None,
                )
                for idx, ts in enumerate(times_n)
            ],
            dtype=np.float32,
        )
        p50 = np.clip(p50 + bias_vals, 0.0, 1.2)
        if include_intervals and p10 is not None and p90 is not None:
            p10 = np.clip(p10 + bias_vals, 0.0, 1.2)
            p90 = np.clip(p90 + bias_vals, 0.0, 1.2)
            p10, p90 = ordered_prediction_bounds(p10, p90)
            p50 = np.minimum(np.maximum(p50, p10), p90)

    out: Dict[str, np.ndarray] = {
        "p50": np.asarray(p50, dtype=np.float32),
    }
    if include_intervals and p10 is not None and p90 is not None:
        out["p10"] = np.asarray(p10, dtype=np.float32)
        out["p90"] = np.asarray(p90, dtype=np.float32)
    return out


def evaluate_model_bundle_on_recent_window(
    model_bundle: Dict[str, object],
    residual_profile: Optional[Dict[str, object]],
    point_bias_profile: Optional[Dict[str, object]],
    direct_horizon_profile: Optional[Dict[str, object]],
    feature_fill_values: Optional[np.ndarray],
    feature_clip_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    loc_ids: List[int],
    loc_data: Dict[int, Dict[str, object]],
    onehot: Dict[int, List[float]],
    weather_source: Optional[Dict[str, object]],
    since: datetime,
    dataset: Optional[Dict[str, object]] = None,
) -> Optional[Dict[str, object]]:
    target_coverage = max(0.1, min(0.98, float(INTERVAL_Q_HIGH - INTERVAL_Q_LOW)))
    eval_dataset = dataset
    if not isinstance(eval_dataset, dict):
        weather_lookup_cache: Optional[Dict[Tuple[datetime, str], float]] = (
            {} if weather_source is not None else None
        )
        location_balance_map = build_location_balance_weight_map_from_loc_data(
            loc_ids,
            loc_data,
            since=since,
        )
        eval_dataset = build_model_observation_dataset(
            loc_ids=loc_ids,
            loc_data=loc_data,
            onehot=onehot,
            weather_source=weather_source,
            location_balance_map=location_balance_map,
            since=since,
            require_min_samples=False,
            exclude_stale=False,
            include_direct_horizon_pairs=False,
            weather_lookup_cache=weather_lookup_cache,
        )

    X = np.asarray(eval_dataset.get("X"), dtype=np.float32)
    labels_arr = np.asarray(eval_dataset.get("y"), dtype=np.float32).reshape(-1)
    times = list(eval_dataset.get("times", []))
    hours = list(eval_dataset.get("hours", []))
    quality_weights = np.asarray(eval_dataset.get("rowQualityWeights"), dtype=np.float32).reshape(-1)
    if X.ndim != 2 or X.shape[0] <= 0 or labels_arr.size <= 0 or not times:
        return None

    preds = predict_model_bundle_on_feature_matrix(
        model_bundle=model_bundle,
        X=X,
        times=times,
        hours=hours,
        residual_profile=residual_profile,
        point_bias_profile=point_bias_profile,
        direct_horizon_profile=direct_horizon_profile,
        feature_fill_values=feature_fill_values,
        feature_clip_bounds=feature_clip_bounds,
        include_intervals=True,
    )
    if not isinstance(preds, dict):
        return None

    p50 = np.asarray(preds.get("p50"), dtype=np.float32).reshape(-1)
    p10 = np.asarray(preds.get("p10"), dtype=np.float32).reshape(-1)
    p90 = np.asarray(preds.get("p90"), dtype=np.float32).reshape(-1)
    row_count = min(labels_arr.size, p50.size, p10.size, p90.size, len(times), quality_weights.size)
    if row_count <= 0:
        return None

    labels_arr = labels_arr[:row_count]
    p50 = p50[:row_count]
    p10 = p10[:row_count]
    p90 = p90[:row_count]
    times = times[:row_count]
    quality_weights = quality_weights[:row_count]
    finite_mask = np.isfinite(labels_arr) & np.isfinite(p50) & np.isfinite(p10) & np.isfinite(p90)
    if not np.any(finite_mask):
        return None

    keep_indices = np.flatnonzero(finite_mask).tolist()
    labels_arr = labels_arr[finite_mask]
    p50 = p50[finite_mask]
    p10 = p10[finite_mask]
    p90 = p90[finite_mask]
    filtered_times = [times[idx] for idx in keep_indices]
    filtered_quality = quality_weights[finite_mask]
    recency_w = build_recency_weights(filtered_times)
    occupancy_w = build_occupancy_weights(labels_arr)
    weights = stabilize_sample_weights(recency_w * occupancy_w * filtered_quality)
    total_w = float(np.sum(weights))
    if total_w <= 0.0:
        weights = np.ones_like(labels_arr, dtype=np.float32)
        total_w = float(np.sum(weights))

    abs_err = np.abs(p50 - labels_arr)
    sq_err = (p50 - labels_arr) ** 2
    within = ((labels_arr >= p10) & (labels_arr <= p90)).astype(np.float32)
    points = int(labels_arr.size)
    sum_weight = total_w
    sum_abs = float(np.sum(abs_err * weights))
    sum_sq = float(np.sum(sq_err * weights))
    sum_within = float(np.sum(within * weights))

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
    if not (
        math.isfinite(champion_mae)
        and math.isfinite(challenger_mae)
        and math.isfinite(champion_rmse)
        and math.isfinite(challenger_rmse)
        and math.isfinite(champion_interval_err)
        and math.isfinite(challenger_interval_err)
    ):
        decision["promote"] = False
        decision["reason"] = "invalid_eval_metrics"
        return decision

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
        json.dump(sanitize_for_json(meta), handle, ensure_ascii=False, allow_nan=False)

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
        json.dump(sanitize_for_json(meta), handle, ensure_ascii=False, allow_nan=False)
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

    baseline_holdout_rows = int(baseline_meta.get("holdoutRows", 0) or 0)
    candidate_holdout_rows = int(candidate_metrics.get("holdout_rows", 0) or 0)
    baseline_holdout_mae = to_float_or_none(baseline_meta.get("holdoutMae"))
    candidate_holdout_mae = to_float_or_none(candidate_metrics.get("holdout_mae"))
    baseline_holdout_int_err = to_float_or_none(baseline_meta.get("holdoutIntervalCoverageError"))
    candidate_holdout_int_err = to_float_or_none(candidate_metrics.get("holdout_interval_coverage_error"))

    if (
        baseline_holdout_rows >= MODEL_HOLDOUT_MIN_ROWS
        and candidate_holdout_rows >= MODEL_HOLDOUT_MIN_ROWS
        and baseline_holdout_mae is not None
        and candidate_holdout_mae is not None
    ):
        if candidate_holdout_mae > baseline_holdout_mae * (1.0 + MODEL_GUARDRAIL_MAX_HOLDOUT_MAE_DEGRADE):
            return False
        if (
            baseline_holdout_int_err is not None
            and candidate_holdout_int_err is not None
            and (candidate_holdout_int_err - baseline_holdout_int_err)
            > MODEL_GUARDRAIL_MAX_HOLDOUT_INTERVAL_ERR_DEGRADE
        ):
            return False

    baseline_mae = to_float_or_none(baseline_meta.get("valMae"))
    baseline_rows = int(baseline_meta.get("valRows", 0) or 0)
    candidate_mae = to_float_or_none(candidate_metrics.get("val_mae"))
    candidate_rows = int(candidate_metrics.get("val_rows", 0) or 0)
    baseline_val_int_err = to_float_or_none(baseline_meta.get("valIntervalCoverageError"))
    candidate_val_int_err = to_float_or_none(candidate_metrics.get("val_interval_coverage_error"))

    if baseline_mae is None or candidate_mae is None:
        return True
    if baseline_rows < MODEL_GUARDRAIL_MIN_VAL_ROWS:
        return True
    if candidate_rows < MODEL_GUARDRAIL_MIN_VAL_ROWS:
        return True

    if float(candidate_mae) > float(baseline_mae) * (1.0 + MODEL_GUARDRAIL_MAX_MAE_DEGRADE):
        return False

    if (
        baseline_val_int_err is not None
        and candidate_val_int_err is not None
        and (candidate_val_int_err - baseline_val_int_err) > MODEL_GUARDRAIL_MAX_VAL_INTERVAL_ERR_DEGRADE
    ):
        return False

    return True


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
    feature_dataset_cache: Optional[Dict[Tuple[object, ...], Dict[str, object]]] = None,
    feature_core_cache: Optional[Dict[Tuple[int, datetime], List[float]]] = None,
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
            dataset_cache=feature_dataset_cache,
            core_feature_cache=feature_core_cache,
        )
        run_metrics = candidate_metrics

        if candidate_bundle is None:
            blocked_by_features = bool(candidate_metrics.get("feature_quality_blocked"))
            if saved_bundle is not None:
                status = (
                    "using_saved_model_feature_quality_block"
                    if blocked_by_features
                    else "using_saved_model_train_skipped"
                )
                return saved_bundle, saved_meta, status, run_metrics
            status = "no_model_feature_quality_block" if blocked_by_features else "no_model_train_skipped"
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
            "valMae": to_float_or_none(candidate_metrics.get("val_mae")),
            "valRmse": to_float_or_none(candidate_metrics.get("val_rmse")),
            "valIntervalCoverage": to_float_or_none(
                candidate_metrics.get("val_interval_coverage")
            ),
            "valIntervalCoverageError": to_float_or_none(
                candidate_metrics.get("val_interval_coverage_error")
            ),
            "selectedBoostRounds": int(candidate_metrics.get("selected_boost_rounds", 0) or 0),
            "holdoutRows": int(candidate_metrics.get("holdout_rows", 0)),
            "holdoutMae": to_float_or_none(candidate_metrics.get("holdout_mae")),
            "holdoutRmse": to_float_or_none(candidate_metrics.get("holdout_rmse")),
            "holdoutIntervalCoverage": to_float_or_none(
                candidate_metrics.get("holdout_interval_coverage")
            ),
            "holdoutIntervalCoverageError": to_float_or_none(
                candidate_metrics.get("holdout_interval_coverage_error")
            ),
            "residualProfile": interval_profile,
            "quantileDirect": bool(candidate_metrics.get("quantile_direct")),
            "tuningCvMae": candidate_metrics.get("tuning_cv_mae"),
            "tuningCvObjective": candidate_metrics.get("tuning_cv_objective"),
            "tunedCandidates": int(candidate_metrics.get("tuned_candidates", 0)),
            "bestParams": candidate_metrics.get("best_params"),
            "featureAblation": candidate_metrics.get("feature_ablation"),
            "featureMissingness": candidate_metrics.get("feature_missingness"),
            "featureFillValues": candidate_metrics.get("feature_fill_values"),
            "featureClipLower": candidate_metrics.get("feature_clip_lower"),
            "featureClipUpper": candidate_metrics.get("feature_clip_upper"),
            "pointBiasProfile": candidate_metrics.get("point_bias_profile"),
            "regimeProfile": candidate_metrics.get("regime_profile"),
            "directHorizonProfile": candidate_metrics.get("direct_horizon_profile"),
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
            recent_eval_dataset = build_model_observation_dataset(
                loc_ids=expected_loc_ids,
                loc_data=loc_data,
                onehot=onehot,
                weather_source=weather_series,
                location_balance_map=build_location_balance_weight_map_from_loc_data(
                    expected_loc_ids,
                    loc_data,
                    since=since,
                ),
                since=since,
                require_min_samples=False,
                exclude_stale=False,
                include_direct_horizon_pairs=False,
                weather_lookup_cache={} if weather_series is not None else None,
                core_feature_cache=feature_core_cache,
                dataset_cache=feature_dataset_cache,
                cache_key=("recent_dataset", "recent_balance", str(model_key), since.isoformat()),
            )
            champion_eval = evaluate_model_bundle_on_recent_window(
                model_bundle=saved_bundle,
                residual_profile=saved_meta.get("residualProfile") if isinstance(saved_meta, dict) else None,
                point_bias_profile=saved_meta.get("pointBiasProfile") if isinstance(saved_meta, dict) else None,
                direct_horizon_profile=saved_meta.get("directHorizonProfile") if isinstance(saved_meta, dict) else None,
                feature_fill_values=coerce_feature_fill_values(
                    saved_meta.get("featureFillValues") if isinstance(saved_meta, dict) else None,
                    expected_cols=feature_count,
                ),
                feature_clip_bounds=coerce_feature_clip_bounds(
                    saved_meta.get("featureClipLower") if isinstance(saved_meta, dict) else None,
                    saved_meta.get("featureClipUpper") if isinstance(saved_meta, dict) else None,
                    expected_cols=feature_count,
                ),
                loc_ids=expected_loc_ids,
                loc_data=loc_data,
                onehot=onehot,
                weather_source=weather_series,
                since=since,
                dataset=recent_eval_dataset,
            )
            challenger_eval = evaluate_model_bundle_on_recent_window(
                model_bundle=candidate_bundle,
                residual_profile=interval_profile,
                point_bias_profile=candidate_metrics.get("point_bias_profile"),
                direct_horizon_profile=candidate_metrics.get("direct_horizon_profile"),
                feature_fill_values=coerce_feature_fill_values(
                    candidate_metrics.get("feature_fill_values"),
                    expected_cols=feature_count,
                ),
                feature_clip_bounds=coerce_feature_clip_bounds(
                    candidate_metrics.get("feature_clip_lower"),
                    candidate_metrics.get("feature_clip_upper"),
                    expected_cols=feature_count,
                ),
                loc_ids=expected_loc_ids,
                loc_data=loc_data,
                onehot=onehot,
                weather_source=weather_series,
                since=since,
                dataset=recent_eval_dataset,
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
            feature_dataset_cache: Dict[Tuple[object, ...], Dict[str, object]] = {}
            feature_core_cache: Dict[Tuple[int, datetime], List[float]] = {}
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
                feature_dataset_cache=feature_dataset_cache,
                feature_core_cache=feature_core_cache,
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
        holdout_rows = int((meta.get("holdoutRows") if meta else run_metrics.get("holdout_rows")) or 0)
        val_mae = to_float_or_none(meta.get("valMae") if meta else run_metrics.get("val_mae"))
        val_rmse = to_float_or_none(meta.get("valRmse") if meta else run_metrics.get("val_rmse"))
        val_interval_cov = to_float_or_none(
            meta.get("valIntervalCoverage")
            if meta
            else run_metrics.get("val_interval_coverage")
        )
        val_interval_cov_err = to_float_or_none(
            meta.get("valIntervalCoverageError")
            if meta
            else run_metrics.get("val_interval_coverage_error")
        )
        holdout_mae = to_float_or_none(meta.get("holdoutMae") if meta else run_metrics.get("holdout_mae"))
        holdout_rmse = to_float_or_none(meta.get("holdoutRmse") if meta else run_metrics.get("holdout_rmse"))
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
            "holdoutRows": holdout_rows,
            "valMae": val_mae,
            "valRmse": val_rmse,
            "valIntervalCoverage": val_interval_cov,
            "valIntervalCoverageError": val_interval_cov_err,
            "holdoutMae": holdout_mae,
            "holdoutRmse": holdout_rmse,
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
            "holdoutRows": int((meta.get("holdoutRows") if meta else run_metrics.get("holdout_rows")) or 0),
            "valMae": to_float_or_none(meta.get("valMae") if meta else run_metrics.get("val_mae")),
            "valRmse": to_float_or_none(meta.get("valRmse") if meta else run_metrics.get("val_rmse")),
            "valIntervalCoverage": to_float_or_none(
                meta.get("valIntervalCoverage")
                if meta
                else run_metrics.get("val_interval_coverage")
            ),
            "valIntervalCoverageError": to_float_or_none(
                meta.get("valIntervalCoverageError")
                if meta
                else run_metrics.get("val_interval_coverage_error")
            ),
            "selectedBoostRounds": int(
                (meta.get("selectedBoostRounds") if meta else run_metrics.get("selected_boost_rounds"))
                or 0
            ),
            "holdoutMae": to_float_or_none(meta.get("holdoutMae") if meta else run_metrics.get("holdout_mae")),
            "holdoutRmse": to_float_or_none(meta.get("holdoutRmse") if meta else run_metrics.get("holdout_rmse")),
            "holdoutIntervalCoverage": to_float_or_none(
                meta.get("holdoutIntervalCoverage")
                if meta
                else run_metrics.get("holdout_interval_coverage")
            ),
            "holdoutIntervalCoverageError": to_float_or_none(
                meta.get("holdoutIntervalCoverageError")
                if meta
                else run_metrics.get("holdout_interval_coverage_error")
            ),
            "quantileDirect": bool(meta.get("quantileDirect")) if meta else False,
            "bestParams": (meta.get("bestParams") if meta else run_metrics.get("best_params")),
            "tuningCvObjective": (
                meta.get("tuningCvObjective")
                if meta
                else run_metrics.get("tuning_cv_objective")
            ),
            "featureAblation": (meta.get("featureAblation") if meta else run_metrics.get("feature_ablation")),
            "featureMissingness": (
                meta.get("featureMissingness")
                if meta
                else run_metrics.get("feature_missingness")
            ),
            "pointBiasProfile": (
                meta.get("pointBiasProfile")
                if meta
                else run_metrics.get("point_bias_profile")
            ),
            "regimeProfile": (
                meta.get("regimeProfile")
                if meta
                else run_metrics.get("regime_profile")
            ),
            "directHorizonProfile": (
                meta.get("directHorizonProfile")
                if meta
                else run_metrics.get("direct_horizon_profile")
            ),
            "featureQualityBlocked": (
                False if meta else bool(run_metrics.get("feature_quality_blocked"))
            ),
            "featureQualityReason": (
                None if meta else run_metrics.get("feature_quality_reason")
            ),
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
    recent_dataset_cache: Optional[Dict[Tuple[object, ...], Dict[str, object]]] = None,
    core_feature_cache: Optional[Dict[Tuple[int, datetime], List[float]]] = None,
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
        "recentBiasEnabled": RECENT_DRIFT_BIAS_ENABLED,
        "recentBiasMaxAbs": float(RECENT_DRIFT_BIAS_MAX_ABS),
        "recentBiasMinPointsPerHour": max(1, int(RECENT_DRIFT_BIAS_MIN_POINTS_PER_HOUR)),
        "recentBiasMinPointsPerOccupancy": max(1, int(RECENT_DRIFT_BIAS_MIN_POINTS_PER_OCCUPANCY)),
        "recentBiasHorizonDecayHours": max(0.5, float(RECENT_DRIFT_BIAS_HORIZON_DECAY_HOURS)),
        "recentBiasBlend": max(0.0, min(1.0, float(RECENT_DRIFT_BIAS_BLEND))),
        "recentBiasSupportTargetMult": max(1.0, float(RECENT_DRIFT_BIAS_SUPPORT_TARGET_MULT)),
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
        location_balance_map = build_location_balance_weight_map_from_counts(
            {int(loc_id): int(loc_samples.get(int(loc_id), 0) or 0) for loc_id in loc_ids}
        )
        model_meta = model_meta_by_key.get(model_key) if isinstance(model_meta_by_key, dict) else None
        expected_feature_count = (
            int(model_meta.get("featureCount", -1) or -1)
            if isinstance(model_meta, dict)
            else -1
        )
        point_bias_profile = model_meta.get("pointBiasProfile") if isinstance(model_meta, dict) else None
        direct_horizon_profile = model_meta.get("directHorizonProfile") if isinstance(model_meta, dict) else None
        feature_fill_values = coerce_feature_fill_values(
            model_meta.get("featureFillValues") if isinstance(model_meta, dict) else None,
            expected_cols=expected_feature_count if expected_feature_count > 0 else None,
        )
        feature_clip_bounds = coerce_feature_clip_bounds(
            model_meta.get("featureClipLower") if isinstance(model_meta, dict) else None,
            model_meta.get("featureClipUpper") if isinstance(model_meta, dict) else None,
            expected_cols=expected_feature_count if expected_feature_count > 0 else None,
        )

        weighted_abs_sum = 0.0
        weighted_signed_sum = 0.0
        weight_sum = 0.0
        points = 0
        bias_hour_sum: Dict[int, float] = {}
        bias_hour_weight: Dict[int, float] = {}
        bias_hour_count: Dict[int, int] = {}
        bias_day_sum: Dict[str, float] = {"weekday": 0.0, "weekend": 0.0}
        bias_day_weight: Dict[str, float] = {"weekday": 0.0, "weekend": 0.0}
        bias_day_count: Dict[str, int] = {"weekday": 0, "weekend": 0}
        bias_occ_sum: Dict[str, float] = {"low": 0.0, "mid": 0.0, "high": 0.0}
        bias_occ_weight: Dict[str, float] = {"low": 0.0, "mid": 0.0, "high": 0.0}
        bias_occ_count: Dict[str, int] = {"low": 0, "mid": 0, "high": 0}
        drift_dataset = build_model_observation_dataset(
            loc_ids=loc_ids,
            loc_data=loc_data,
            onehot=onehot_by_key.get(model_key, {}),
            weather_source=weather_series,
            location_balance_map=location_balance_map,
            since=since,
            loc_samples=loc_samples,
            require_min_samples=True,
            exclude_stale=True,
            include_direct_horizon_pairs=False,
            weather_lookup_cache=weather_lookup_cache,
            core_feature_cache=core_feature_cache,
            dataset_cache=recent_dataset_cache,
            cache_key=("recent_dataset", "counts_balance", str(model_key), since.isoformat()),
        )
        X = np.asarray(drift_dataset.get("X"), dtype=np.float32)
        labels_arr = np.asarray(drift_dataset.get("y"), dtype=np.float32).reshape(-1)
        times = list(drift_dataset.get("times", []))
        hours = list(drift_dataset.get("hours", []))
        feature_quality_weights = np.asarray(
            drift_dataset.get("featureQualityWeights"),
            dtype=np.float32,
        ).reshape(-1)
        transition_weights = np.asarray(
            drift_dataset.get("transitionWeights"),
            dtype=np.float32,
        ).reshape(-1)
        location_balance_weights = np.asarray(
            drift_dataset.get("locationBalanceWeights"),
            dtype=np.float32,
        ).reshape(-1)
        row_count = min(
            int(X.shape[0]) if X.ndim == 2 else 0,
            labels_arr.size,
            len(times),
            len(hours),
            feature_quality_weights.size,
            transition_weights.size,
            location_balance_weights.size,
        )
        if row_count <= 0:
            continue

        preds = predict_model_bundle_on_feature_matrix(
            model_bundle=bundle,
            X=X[:row_count],
            times=times[:row_count],
            hours=hours[:row_count],
            residual_profile=None,
            point_bias_profile=point_bias_profile if isinstance(point_bias_profile, dict) else None,
            direct_horizon_profile=direct_horizon_profile if isinstance(direct_horizon_profile, dict) else None,
            feature_fill_values=feature_fill_values,
            feature_clip_bounds=feature_clip_bounds,
            include_intervals=False,
        )
        if not isinstance(preds, dict):
            continue

        preds_arr = np.asarray(preds.get("p50"), dtype=np.float32).reshape(-1)[:row_count]
        labels_arr = labels_arr[:row_count]
        times = times[:row_count]
        feature_quality_weights = feature_quality_weights[:row_count]
        transition_weights = transition_weights[:row_count]
        location_balance_weights = location_balance_weights[:row_count]
        finite_mask = np.isfinite(labels_arr) & np.isfinite(preds_arr)
        if not np.any(finite_mask):
            continue
        keep_list = np.flatnonzero(finite_mask).tolist()
        errors = np.abs(preds_arr[finite_mask] - labels_arr[finite_mask]).astype(np.float32)
        residuals = (labels_arr[finite_mask] - preds_arr[finite_mask]).astype(np.float32)
        filtered_preds = preds_arr[finite_mask].astype(np.float32)
        filtered_times = [times[idx] for idx in keep_list]
        filtered_feature_quality = feature_quality_weights[finite_mask]
        filtered_transition_weights = transition_weights[finite_mask]
        filtered_location_balance = location_balance_weights[finite_mask]
        filtered_labels = labels_arr[finite_mask]
        recency_w = build_recency_weights(filtered_times)
        occupancy_w = build_occupancy_weights(filtered_labels)
        drift_w = stabilize_sample_weights(
            recency_w
            * occupancy_w
            * filtered_transition_weights
            * filtered_feature_quality
            * filtered_location_balance
        )
        if float(np.sum(drift_w)) <= 0.0:
            drift_w = np.ones_like(errors, dtype=np.float32)

        weighted_abs_sum += float(np.sum(errors * drift_w))
        weighted_signed_sum += float(np.sum(residuals * drift_w))
        weight_sum += float(np.sum(drift_w))
        points += int(errors.size)
        for idx, ts in enumerate(filtered_times):
            if idx >= drift_w.size or idx >= residuals.size:
                break
            wv = float(drift_w[idx])
            if not math.isfinite(wv) or wv <= 0.0:
                continue
            rv = float(residuals[idx])
            hour = int(ts.hour)
            bias_hour_sum[hour] = float(bias_hour_sum.get(hour, 0.0) + rv * wv)
            bias_hour_weight[hour] = float(bias_hour_weight.get(hour, 0.0) + wv)
            bias_hour_count[hour] = int(bias_hour_count.get(hour, 0) + 1)
            day_key = "weekend" if int(ts.weekday()) >= 5 else "weekday"
            bias_day_sum[day_key] = float(bias_day_sum.get(day_key, 0.0) + rv * wv)
            bias_day_weight[day_key] = float(bias_day_weight.get(day_key, 0.0) + wv)
            bias_day_count[day_key] = int(bias_day_count.get(day_key, 0) + 1)
            occ_ratio = float(filtered_preds[idx]) if idx < filtered_preds.size else 0.0
            occ_key = occupancy_bucket_key_from_ratio(occ_ratio)
            bias_occ_sum[occ_key] = float(bias_occ_sum.get(occ_key, 0.0) + rv * wv)
            bias_occ_weight[occ_key] = float(bias_occ_weight.get(occ_key, 0.0) + wv)
            bias_occ_count[occ_key] = int(bias_occ_count.get(occ_key, 0) + 1)

        if points < max(1, DRIFT_MIN_POINTS):
            continue

        recent_mae = float(weighted_abs_sum / max(1e-6, weight_sum))
        raw_bias = float(weighted_signed_sum / max(1e-6, weight_sum))
        max_bias_abs = max(0.0, float(RECENT_DRIFT_BIAS_MAX_ABS))
        if max_bias_abs > 0.0:
            raw_bias = max(-max_bias_abs, min(max_bias_abs, raw_bias))
        min_points_hour = max(1, int(RECENT_DRIFT_BIAS_MIN_POINTS_PER_HOUR))
        bias_by_hour_payload: Dict[str, Dict[str, object]] = {}
        for hour in sorted(bias_hour_sum.keys()):
            count = int(bias_hour_count.get(hour, 0))
            total_w = float(bias_hour_weight.get(hour, 0.0))
            if count < min_points_hour or total_w <= 0.0:
                continue
            hbias = float(bias_hour_sum[hour] / max(1e-6, total_w))
            if max_bias_abs > 0.0:
                hbias = max(-max_bias_abs, min(max_bias_abs, hbias))
            bias_by_hour_payload[str(int(hour))] = {
                "bias": float(hbias),
                "count": count,
            }
        bias_by_day_payload: Dict[str, Dict[str, object]] = {}
        for day_key in ("weekday", "weekend"):
            count = int(bias_day_count.get(day_key, 0))
            total_w = float(bias_day_weight.get(day_key, 0.0))
            if count < min_points_hour or total_w <= 0.0:
                continue
            dbias = float(bias_day_sum.get(day_key, 0.0) / max(1e-6, total_w))
            if max_bias_abs > 0.0:
                dbias = max(-max_bias_abs, min(max_bias_abs, dbias))
            bias_by_day_payload[day_key] = {
                "bias": float(dbias),
                "count": count,
            }
        min_points_occ = max(1, int(RECENT_DRIFT_BIAS_MIN_POINTS_PER_OCCUPANCY))
        bias_by_occ_payload: Dict[str, Dict[str, object]] = {}
        for occ_key in ("low", "mid", "high"):
            count = int(bias_occ_count.get(occ_key, 0))
            total_w = float(bias_occ_weight.get(occ_key, 0.0))
            if count < min_points_occ or total_w <= 0.0:
                continue
            obias = float(bias_occ_sum.get(occ_key, 0.0) / max(1e-6, total_w))
            if max_bias_abs > 0.0:
                obias = max(-max_bias_abs, min(max_bias_abs, obias))
            bias_by_occ_payload[occ_key] = {
                "bias": float(obias),
                "count": count,
            }
        recent_bias_profile = {
            "global": {"bias": float(raw_bias), "count": int(points)},
            "byHour": bias_by_hour_payload,
            "byDayType": bias_by_day_payload,
            "byOccupancy": bias_by_occ_payload,
        }
        baseline_mae, _baseline_rows = preferred_model_error_and_rows(model_meta)
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
            "recentBias": float(raw_bias),
            "recentBiasProfile": recent_bias_profile,
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


def conformal_margin_from_scores(
    scores: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    if scores.size == 0:
        return 0.0
    alpha = max(0.01, min(0.5, float(INTERVAL_CONFORMAL_ALPHA)))
    quantile = max(0.0, min(1.0, 1.0 - alpha))
    margin = float(weighted_quantile(scores, quantile, weights))
    return max(0.0, min(float(INTERVAL_CONFORMAL_MAX_MARGIN), margin))


def forecast_horizon_hours(
    target: datetime,
    reference: Optional[datetime] = None,
    hours_ahead: Optional[float] = None,
) -> int:
    if hours_ahead is not None:
        parsed = to_float_or_none(hours_ahead)
        if parsed is not None:
            return max(0, int(parsed))

    if isinstance(reference, datetime):
        delta_h = (target - reference).total_seconds() / 3600.0
        if math.isfinite(delta_h):
            return max(0, int(delta_h))

    start_hour, _end_hour = normalized_forecast_hour_bounds()
    return max(0, int(target.hour) - int(start_hour))


def horizon_bucket_key(hours_ahead: int) -> str:
    h = max(0, int(hours_ahead))
    if h <= 2:
        return "0_2h"
    if h <= 6:
        return "3_6h"
    if h <= 12:
        return "7_12h"
    if h <= 24:
        return "13_24h"
    return "25h_plus"


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


def occupancy_bucket_key_from_ratio(ratio: float) -> str:
    val = max(0.0, min(1.2, float(ratio)))
    if val < 0.35:
        return "low"
    if val < 0.75:
        return "mid"
    return "high"


def conformal_margin_for_hour(
    hour: int,
    hours_ahead: int,
    profile: Optional[Dict[str, object]],
    point_ratio: Optional[float] = None,
    target: Optional[datetime] = None,
) -> float:
    if not profile:
        return 0.0

    min_points = max(1, int(INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR))
    global_stats = profile.get("global", {})
    global_margin = (
        max(0.0, float(global_stats.get("margin", 0.0) or 0.0))
        if isinstance(global_stats, dict)
        else 0.0
    )
    support_target = max(
        float(min_points),
        float(min_points) * max(1.0, float(INTERVAL_CONFORMAL_SEGMENT_BLEND_TARGET_MULT)),
    )

    def blended_margin(stats: Optional[Dict[str, object]]) -> Optional[float]:
        if not isinstance(stats, dict):
            return None
        count = int(stats.get("count", 0) or 0)
        margin = to_float_or_none(stats.get("margin"))
        if margin is None or count < min_points:
            return None
        support = max(0.0, min(1.0, float(count) / support_target))
        value = float(support * float(margin) + (1.0 - support) * float(global_margin))
        return max(0.0, value)

    by_occupancy = profile.get("byOccupancy", {})
    if point_ratio is not None and isinstance(by_occupancy, dict):
        occ_key = occupancy_bucket_key_from_ratio(float(point_ratio))
        occ_stats = by_occupancy.get(occ_key)
        margin = blended_margin(occ_stats if isinstance(occ_stats, dict) else None)
        if margin is not None:
            return margin

    by_horizon = profile.get("byHorizon", {})
    horizon_stats = (
        by_horizon.get(horizon_bucket_key(hours_ahead))
        if isinstance(by_horizon, dict)
        else None
    )
    margin = blended_margin(horizon_stats if isinstance(horizon_stats, dict) else None)
    if margin is not None:
        return margin

    by_hour = profile.get("byHour", {})
    hour_stats = by_hour.get(str(int(hour))) if isinstance(by_hour, dict) else None
    margin = blended_margin(hour_stats if isinstance(hour_stats, dict) else None)
    if margin is not None:
        return margin

    by_block = profile.get("byHourBlock", {})
    block_stats = by_block.get(hour_block_key(hour)) if isinstance(by_block, dict) else None
    margin = blended_margin(block_stats if isinstance(block_stats, dict) else None)
    if margin is not None:
        return margin

    if isinstance(target, datetime):
        day_key = "weekend" if int(target.weekday()) >= 5 else "weekday"
        by_day_type = profile.get("byDayType", {})
        day_stats = by_day_type.get(day_key) if isinstance(by_day_type, dict) else None
        margin = blended_margin(day_stats if isinstance(day_stats, dict) else None)
        if margin is not None:
            return margin

    return float(global_margin)


def compute_interval_conformal_profiles(
    now: datetime,
    models_by_key: Dict[str, Dict[str, object]],
    onehot_by_key: Dict[str, Dict[int, List[float]]],
    unit_loc_ids: Dict[str, List[int]],
    loc_data: Dict[int, Dict[str, object]],
    loc_samples: Dict[int, int],
    weather_series: Optional[Dict[str, object]],
    interval_profile_by_key: Dict[str, Optional[Dict[str, object]]],
    model_meta_by_key: Optional[Dict[str, Dict[str, object]]] = None,
    recent_dataset_cache: Optional[Dict[Tuple[object, ...], Dict[str, object]]] = None,
    core_feature_cache: Optional[Dict[Tuple[int, datetime], List[float]]] = None,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    summary = {
        "enabled": INTERVAL_CONFORMAL_ENABLED,
        "recentDays": max(1, INTERVAL_CONFORMAL_RECENT_DAYS),
        "alpha": max(0.01, min(0.5, float(INTERVAL_CONFORMAL_ALPHA))),
        "minPoints": max(1, INTERVAL_CONFORMAL_MIN_POINTS),
        "minPointsPerHour": max(1, INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR),
        "maxMargin": float(INTERVAL_CONFORMAL_MAX_MARGIN),
        "segmentBlendTargetMult": max(1.0, float(INTERVAL_CONFORMAL_SEGMENT_BLEND_TARGET_MULT)),
        "modelsCalibrated": 0,
        "byModel": {},
    }
    if not INTERVAL_CONFORMAL_ENABLED:
        return {}, summary

    since = now - timedelta(days=max(1, INTERVAL_CONFORMAL_RECENT_DAYS))
    weather_lookup_cache: Dict[Tuple[datetime, str], float] = {}
    profiles: Dict[str, Dict[str, object]] = {}
    horizon_key_zero = horizon_bucket_key(0)

    for model_key, bundle in models_by_key.items():
        p50_model = bundle.get("p50")
        if p50_model is None:
            continue
        model_meta = (
            model_meta_by_key.get(model_key)
            if isinstance(model_meta_by_key, dict)
            else None
        )
        expected_feature_count = (
            int(model_meta.get("featureCount", -1) or -1)
            if isinstance(model_meta, dict)
            else -1
        )
        point_bias_profile = model_meta.get("pointBiasProfile") if isinstance(model_meta, dict) else None
        direct_horizon_profile = model_meta.get("directHorizonProfile") if isinstance(model_meta, dict) else None
        feature_fill_values = coerce_feature_fill_values(
            model_meta.get("featureFillValues") if isinstance(model_meta, dict) else None,
            expected_cols=expected_feature_count if expected_feature_count > 0 else None,
        )
        feature_clip_bounds = coerce_feature_clip_bounds(
            model_meta.get("featureClipLower") if isinstance(model_meta, dict) else None,
            model_meta.get("featureClipUpper") if isinstance(model_meta, dict) else None,
            expected_cols=expected_feature_count if expected_feature_count > 0 else None,
        )

        loc_ids = unit_loc_ids.get(model_key) or []
        if not loc_ids:
            continue
        location_balance_map = build_location_balance_weight_map_from_counts(
            {int(loc_id): int(loc_samples.get(int(loc_id), 0) or 0) for loc_id in loc_ids}
        )

        global_scores: List[float] = []
        global_weights: List[float] = []
        by_hour_scores: Dict[int, List[float]] = {}
        by_hour_weights: Dict[int, List[float]] = {}
        by_block_scores: Dict[str, List[float]] = {}
        by_block_weights: Dict[str, List[float]] = {}
        by_day_type_scores: Dict[str, List[float]] = {"weekday": [], "weekend": []}
        by_day_type_weights: Dict[str, List[float]] = {"weekday": [], "weekend": []}
        by_horizon_scores: Dict[str, List[float]] = {}
        by_horizon_weights: Dict[str, List[float]] = {}
        by_occupancy_scores: Dict[str, List[float]] = {}
        by_occupancy_weights: Dict[str, List[float]] = {}
        points = 0
        conformal_dataset = build_model_observation_dataset(
            loc_ids=loc_ids,
            loc_data=loc_data,
            onehot=onehot_by_key.get(model_key, {}),
            weather_source=weather_series,
            location_balance_map=location_balance_map,
            since=since,
            loc_samples=loc_samples,
            require_min_samples=True,
            exclude_stale=True,
            include_direct_horizon_pairs=False,
            weather_lookup_cache=weather_lookup_cache,
            core_feature_cache=core_feature_cache,
            dataset_cache=recent_dataset_cache,
            cache_key=("recent_dataset", "counts_balance", str(model_key), since.isoformat()),
        )
        X = np.asarray(conformal_dataset.get("X"), dtype=np.float32)
        labels_arr = np.asarray(conformal_dataset.get("y"), dtype=np.float32).reshape(-1)
        times = list(conformal_dataset.get("times", []))
        hours = list(conformal_dataset.get("hours", []))
        quality_weights = np.asarray(
            conformal_dataset.get("rowQualityWeights"),
            dtype=np.float32,
        ).reshape(-1)
        row_count = min(
            int(X.shape[0]) if X.ndim == 2 else 0,
            labels_arr.size,
            len(times),
            len(hours),
            quality_weights.size,
        )
        if row_count <= 0:
            continue

        preds = predict_model_bundle_on_feature_matrix(
            model_bundle=bundle,
            X=X[:row_count],
            times=times[:row_count],
            hours=hours[:row_count],
            residual_profile=interval_profile_by_key.get(model_key),
            point_bias_profile=point_bias_profile if isinstance(point_bias_profile, dict) else None,
            direct_horizon_profile=direct_horizon_profile if isinstance(direct_horizon_profile, dict) else None,
            feature_fill_values=feature_fill_values,
            feature_clip_bounds=feature_clip_bounds,
            include_intervals=True,
        )
        if not isinstance(preds, dict):
            continue

        p50 = np.asarray(preds.get("p50"), dtype=np.float32).reshape(-1)[:row_count]
        p10 = np.asarray(preds.get("p10"), dtype=np.float32).reshape(-1)[:row_count]
        p90 = np.asarray(preds.get("p90"), dtype=np.float32).reshape(-1)[:row_count]
        labels_arr = labels_arr[:row_count]
        times = times[:row_count]
        hours = hours[:row_count]
        quality_weights = quality_weights[:row_count]
        finite_mask = np.isfinite(labels_arr) & np.isfinite(p50) & np.isfinite(p10) & np.isfinite(p90)
        if not np.any(finite_mask):
            continue

        keep_list = np.flatnonzero(finite_mask).tolist()
        labels_arr = labels_arr[finite_mask]
        p50 = p50[finite_mask]
        p10 = p10[finite_mask]
        p90 = p90[finite_mask]
        filtered_p50 = p50.copy()
        filtered_times = [times[idx] for idx in keep_list]
        filtered_hours = [hours[idx] for idx in keep_list]
        filtered_quality = quality_weights[finite_mask]
        scores = np.maximum.reduce(
            [
                np.zeros_like(labels_arr),
                p10 - labels_arr,
                labels_arr - p90,
            ]
        )
        recency_w = build_recency_weights(filtered_times)
        occupancy_w = build_occupancy_weights(labels_arr)
        score_weights = stabilize_sample_weights(recency_w * occupancy_w * filtered_quality)
        if float(np.sum(score_weights)) <= 0.0:
            score_weights = np.ones_like(scores, dtype=np.float32)

        for idx, score in enumerate(scores.tolist()):
            score_val = float(score)
            weight_val = float(score_weights[idx]) if idx < score_weights.size else 1.0
            hour = int(filtered_hours[idx])
            occupancy_key = occupancy_bucket_key_from_ratio(
                float(filtered_p50[idx]) if idx < filtered_p50.size else 0.0
            )
            global_scores.append(score_val)
            global_weights.append(weight_val)
            by_hour_scores.setdefault(hour, []).append(score_val)
            by_hour_weights.setdefault(hour, []).append(weight_val)
            by_block_scores.setdefault(hour_block_key(hour), []).append(score_val)
            by_block_weights.setdefault(hour_block_key(hour), []).append(weight_val)
            if idx < len(filtered_times):
                day_key = "weekend" if int(filtered_times[idx].weekday()) >= 5 else "weekday"
                by_day_type_scores.setdefault(day_key, []).append(score_val)
                by_day_type_weights.setdefault(day_key, []).append(weight_val)
            by_horizon_scores.setdefault(horizon_key_zero, []).append(score_val)
            by_horizon_weights.setdefault(horizon_key_zero, []).append(weight_val)
            by_occupancy_scores.setdefault(occupancy_key, []).append(score_val)
            by_occupancy_weights.setdefault(occupancy_key, []).append(weight_val)
        points += len(filtered_hours)

        if points < max(1, INTERVAL_CONFORMAL_MIN_POINTS) or not global_scores:
            continue

        global_margin = conformal_margin_from_scores(
            np.array(global_scores, dtype=np.float32),
            np.array(global_weights, dtype=np.float32),
        )
        by_hour_payload: Dict[str, Dict[str, object]] = {}
        for hour, scores in sorted(by_hour_scores.items()):
            if len(scores) < INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR:
                continue
            by_hour_payload[str(int(hour))] = {
                "margin": conformal_margin_from_scores(
                    np.array(scores, dtype=np.float32),
                    np.array(by_hour_weights.get(hour, []), dtype=np.float32),
                ),
                "count": int(len(scores)),
            }

        by_block_payload: Dict[str, Dict[str, object]] = {}
        for block, scores in sorted(by_block_scores.items()):
            if len(scores) < INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR:
                continue
            by_block_payload[str(block)] = {
                "margin": conformal_margin_from_scores(
                    np.array(scores, dtype=np.float32),
                    np.array(by_block_weights.get(block, []), dtype=np.float32),
                ),
                "count": int(len(scores)),
            }

        by_day_type_payload: Dict[str, Dict[str, object]] = {}
        for day_key in ("weekday", "weekend"):
            scores = by_day_type_scores.get(day_key, [])
            if len(scores) < INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR:
                continue
            by_day_type_payload[day_key] = {
                "margin": conformal_margin_from_scores(
                    np.array(scores, dtype=np.float32),
                    np.array(by_day_type_weights.get(day_key, []), dtype=np.float32),
                ),
                "count": int(len(scores)),
            }

        by_horizon_payload: Dict[str, Dict[str, object]] = {}
        for horizon_key, scores in sorted(by_horizon_scores.items()):
            if len(scores) < INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR:
                continue
            by_horizon_payload[str(horizon_key)] = {
                "margin": conformal_margin_from_scores(
                    np.array(scores, dtype=np.float32),
                    np.array(by_horizon_weights.get(horizon_key, []), dtype=np.float32),
                ),
                "count": int(len(scores)),
            }

        by_occupancy_payload: Dict[str, Dict[str, object]] = {}
        for occ_key in ("low", "mid", "high"):
            scores = by_occupancy_scores.get(occ_key, [])
            if len(scores) < INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR:
                continue
            by_occupancy_payload[str(occ_key)] = {
                "margin": conformal_margin_from_scores(
                    np.array(scores, dtype=np.float32),
                    np.array(by_occupancy_weights.get(occ_key, []), dtype=np.float32),
                ),
                "count": int(len(scores)),
            }

        profile = {
            "global": {
                "margin": global_margin,
                "count": int(len(global_scores)),
            },
            "byHour": by_hour_payload,
            "byHourBlock": by_block_payload,
            "byDayType": by_day_type_payload,
            "byHorizon": by_horizon_payload,
            "byOccupancy": by_occupancy_payload,
        }
        profiles[model_key] = profile
        summary["modelsCalibrated"] += 1
        summary["byModel"][model_key] = {
            "points": points,
            "globalMargin": global_margin,
            "hoursCalibrated": len(by_hour_payload),
            "blocksCalibrated": len(by_block_payload),
            "dayTypesCalibrated": len(by_day_type_payload),
            "horizonsCalibrated": len(by_horizon_payload),
            "occupancyBucketsCalibrated": len(by_occupancy_payload),
        }

    return profiles, summary


def interval_bounds(
    point_ratio: float,
    hour: int,
    residual_profile: Optional[Dict[str, object]],
    target: Optional[datetime] = None,
) -> Tuple[float, float, float]:
    point_ratio = max(0.0, min(point_ratio, 1.2))
    if not residual_profile:
        return point_ratio, point_ratio, point_ratio

    min_samples = max(1, int(INTERVAL_MIN_SAMPLES_PER_HOUR))
    blend_target = max(float(min_samples), float(min_samples) * max(1.0, float(INTERVAL_SEGMENT_BLEND_TARGET_MULT)))
    global_stats = residual_profile.get("global", {}) if isinstance(residual_profile, dict) else {}
    global_q10 = float(global_stats.get("q10", 0.0) or 0.0) if isinstance(global_stats, dict) else 0.0
    global_q90 = float(global_stats.get("q90", 0.0) or 0.0) if isinstance(global_stats, dict) else 0.0

    def blended_quantiles(stats: Optional[Dict[str, object]]) -> Optional[Tuple[float, float]]:
        if not isinstance(stats, dict):
            return None
        count = int(stats.get("count", 0) or 0)
        if count < min_samples:
            return None
        q10 = to_float_or_none(stats.get("q10"))
        q90 = to_float_or_none(stats.get("q90"))
        if q10 is None or q90 is None:
            return None
        support = max(0.0, min(1.0, float(count) / blend_target))
        bq10 = float(support * float(q10) + (1.0 - support) * float(global_q10))
        bq90 = float(support * float(q90) + (1.0 - support) * float(global_q90))
        return bq10, bq90

    occupancy_stats = (
        residual_profile.get("byOccupancy", {}).get(occupancy_bucket_key_from_ratio(point_ratio))
        if isinstance(residual_profile, dict)
        else None
    )
    quantiles = blended_quantiles(occupancy_stats if isinstance(occupancy_stats, dict) else None)
    if quantiles is None:
        hour_stats = residual_profile.get("byHour", {}).get(str(hour)) if isinstance(residual_profile, dict) else None
        quantiles = blended_quantiles(hour_stats if isinstance(hour_stats, dict) else None)
    if quantiles is None:
        block_stats = (
            residual_profile.get("byHourBlock", {}).get(hour_block_key(hour))
            if isinstance(residual_profile, dict)
            else None
        )
        quantiles = blended_quantiles(block_stats if isinstance(block_stats, dict) else None)
    if quantiles is None and isinstance(target, datetime):
        day_key = "weekend" if int(target.weekday()) >= 5 else "weekday"
        day_stats = (
            residual_profile.get("byDayType", {}).get(day_key)
            if isinstance(residual_profile, dict)
            else None
        )
        quantiles = blended_quantiles(day_stats if isinstance(day_stats, dict) else None)
    if quantiles is None:
        quantiles = (float(global_q10), float(global_q90))

    q10, q90 = quantiles
    p10 = max(0.0, min(point_ratio + q10, 1.2))
    p90 = max(0.0, min(point_ratio + q90, 1.2))
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
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(float(value), 1.2))


def direct_horizon_stats_for_hours(
    hours_ahead: float,
    profile: Optional[Dict[str, object]],
    point_ratio: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    if not MODEL_DIRECT_HORIZON_ENABLED:
        return None
    if not isinstance(profile, dict):
        return None

    by_hours = profile.get("byHours", {})
    if not isinstance(by_hours, dict) or not by_hours:
        return None

    min_pairs = max(
        1,
        int(
            profile.get(
                "segmentMinPairs",
                profile.get("minPairs", MODEL_DIRECT_HORIZON_MIN_PAIRS),
            )
            or MODEL_DIRECT_HORIZON_MIN_PAIRS
        ),
    )
    occupancy_min_pairs = max(
        1,
        int(profile.get("occupancySegmentMinPairs", min_pairs) or min_pairs),
    )
    points: List[Tuple[float, Dict[str, float]]] = []
    for key, raw_stats in by_hours.items():
        if not isinstance(raw_stats, dict):
            continue
        try:
            horizon_hour = float(int(key))
        except Exception:
            continue
        stats_source = raw_stats
        occ_fallback_applied = False
        if MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENTS_ENABLED and point_ratio is not None:
            occ_key = occupancy_bucket_key_from_ratio(float(point_ratio))
            by_occ = raw_stats.get("byOccupancy", {})
            if isinstance(by_occ, dict):
                occ_stats = by_occ.get(occ_key)
                if isinstance(occ_stats, dict):
                    stats_source = occ_stats
                    occ_fallback_applied = True
        count = int(stats_source.get("count", 0) or 0)
        if occ_fallback_applied and count < occupancy_min_pairs:
            stats_source = raw_stats
            count = int(raw_stats.get("count", 0) or 0)
            occ_fallback_applied = False
        if count < min_pairs:
            continue
        slope = to_float_or_none(stats_source.get("slope"))
        intercept = to_float_or_none(stats_source.get("intercept"))
        blend = to_float_or_none(stats_source.get("blend"))
        if slope is None or intercept is None:
            continue
        points.append(
            (
                float(horizon_hour),
                {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "blend": max(0.0, min(1.0, float(blend) if blend is not None else 0.0)),
                    "count": float(count),
                },
            )
        )

    if not points:
        global_stats = profile.get("global", {}) if isinstance(profile, dict) else {}
        if not isinstance(global_stats, dict):
            return None
        stats_source = global_stats
        if MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENTS_ENABLED and point_ratio is not None:
            occ_key = occupancy_bucket_key_from_ratio(float(point_ratio))
            by_occ = global_stats.get("byOccupancy", {})
            if isinstance(by_occ, dict):
                occ_stats = by_occ.get(occ_key)
                if isinstance(occ_stats, dict) and int(occ_stats.get("count", 0) or 0) >= occupancy_min_pairs:
                    stats_source = occ_stats
        slope = to_float_or_none(stats_source.get("slope"))
        intercept = to_float_or_none(stats_source.get("intercept"))
        blend = to_float_or_none(stats_source.get("blend"))
        count = int(stats_source.get("count", 0) or 0)
        if slope is None or intercept is None or count < min_pairs:
            return None
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "blend": max(0.0, min(1.0, float(blend) if blend is not None else 0.0)),
            "count": float(count),
        }

    points = sorted(points, key=lambda item: item[0])
    h = max(0.0, float(hours_ahead))
    if h <= points[0][0]:
        return points[0][1]
    if h >= points[-1][0]:
        return points[-1][1]

    lower = points[0]
    upper = points[-1]
    for idx in range(1, len(points)):
        candidate = points[idx]
        if h <= candidate[0]:
            lower = points[idx - 1]
            upper = candidate
            break

    lh, lstats = lower
    uh, ustats = upper
    span = max(1e-6, float(uh - lh))
    alpha = max(0.0, min(1.0, (h - lh) / span))
    return {
        "slope": float(lstats["slope"] * (1.0 - alpha) + ustats["slope"] * alpha),
        "intercept": float(lstats["intercept"] * (1.0 - alpha) + ustats["intercept"] * alpha),
        "blend": float(lstats["blend"] * (1.0 - alpha) + ustats["blend"] * alpha),
        "count": float(lstats["count"] * (1.0 - alpha) + ustats["count"] * alpha),
    }


def apply_direct_horizon_adjustment(
    p10_ratio: float,
    p50_ratio: float,
    p90_ratio: float,
    target: datetime,
    profile: Optional[Dict[str, object]],
    hours_ahead: Optional[float] = None,
) -> Tuple[float, float, float]:
    if not MODEL_DIRECT_HORIZON_ENABLED:
        return p10_ratio, p50_ratio, p90_ratio
    if hours_ahead is None:
        horizon_hours = 0.0
    else:
        horizon_hours = max(0.0, float(hours_ahead))
    stats = direct_horizon_stats_for_hours(
        hours_ahead=horizon_hours,
        profile=profile,
        point_ratio=float(p50_ratio),
    )
    if not stats:
        return p10_ratio, p50_ratio, p90_ratio

    blend = max(
        0.0,
        min(
            float(MODEL_DIRECT_HORIZON_MAX_BLEND),
            float(stats.get("blend", 0.0) or 0.0),
        ),
    )
    by_hours = profile.get("byHours", {}) if isinstance(profile, dict) else {}
    if isinstance(by_hours, dict) and by_hours:
        horizons: List[float] = []
        for key in by_hours.keys():
            try:
                horizons.append(float(int(key)))
            except Exception:
                continue
        if horizons:
            min_h = max(1e-6, min(horizons))
            max_h = max(min_h, max(horizons))
            if horizon_hours > max_h:
                blend *= max(0.15, min(1.0, max_h / max(1e-6, horizon_hours)))
            elif horizon_hours < min_h:
                blend *= max(0.3, min(1.0, horizon_hours / min_h))
    if blend <= 0.0:
        return p10_ratio, p50_ratio, p90_ratio

    slope = float(stats.get("slope", 1.0) or 1.0)
    intercept = float(stats.get("intercept", 0.0) or 0.0)
    center = float(p50_ratio)
    low_width = max(0.0, center - float(p10_ratio))
    high_width = max(0.0, float(p90_ratio) - center)
    direct_center = clamp_ratio(slope * center + intercept)

    adjusted_center = (1.0 - blend) * center + blend * direct_center
    adjusted_p10 = adjusted_center - low_width
    adjusted_p90 = adjusted_center + high_width
    if adjusted_p10 > adjusted_p90:
        adjusted_p10, adjusted_p90 = adjusted_p90, adjusted_p10
    adjusted_center = min(max(adjusted_center, adjusted_p10), adjusted_p90)
    return float(adjusted_p10), float(adjusted_center), float(adjusted_p90)


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


def seed_recursive_ratio_overrides(
    lag_ratio_override: Optional[Dict[datetime, float]],
    target: datetime,
    ratio: float,
) -> None:
    if not isinstance(lag_ratio_override, dict):
        return
    numeric = to_float_or_none(ratio)
    if numeric is None:
        return

    lag_ratio_override[target] = float(numeric)
    step = max(1, int(RESAMPLE_MINUTES))
    for offset in range(step, 60, step):
        lag_ratio_override[target + timedelta(minutes=offset)] = float(numeric)


def preferred_model_error_and_rows(meta: Optional[Dict[str, object]]) -> Tuple[Optional[float], int]:
    if not isinstance(meta, dict):
        return None, 0

    holdout_mae = to_float_or_none(meta.get("holdoutMae"))
    holdout_rows = int(meta.get("holdoutRows", 0) or 0)
    if holdout_mae is not None and holdout_rows > 0:
        return float(holdout_mae), int(holdout_rows)

    val_mae = to_float_or_none(meta.get("valMae"))
    val_rows = int(meta.get("valRows", 0) or 0)
    if val_mae is not None and val_rows > 0:
        return float(val_mae), int(val_rows)

    return None, 0


def model_quality_score_for_blend(meta: Optional[Dict[str, object]]) -> float:
    mae, rows = preferred_model_error_and_rows(meta)
    if mae is None:
        return 0.0
    safe_mae = max(0.005, float(mae))
    support = math.log1p(max(1, int(rows)))
    base_score = float(support / safe_mae)
    if isinstance(meta, dict):
        holdout_int_err = to_float_or_none(meta.get("holdoutIntervalCoverageError"))
        if holdout_int_err is None:
            holdout_int_err = to_float_or_none(meta.get("valIntervalCoverageError"))
        if holdout_int_err is not None:
            interval_penalty = 1.0 / (1.0 + 8.0 * max(0.0, float(holdout_int_err)))
            base_score *= interval_penalty

        missing_summary = meta.get("featureMissingness")
        if isinstance(missing_summary, dict):
            missing_rate = to_float_or_none(missing_summary.get("globalMissingRate"))
            if missing_rate is not None:
                missing_penalty = max(0.35, 1.0 - max(0.0, min(0.8, float(missing_rate))))
                base_score *= missing_penalty

        drift_streak = max(0, int(meta.get("driftAlertStreak", 0) or 0))
        drift_penalty = 1.0 / (1.0 + 0.25 * float(drift_streak))
        retrain_penalty = 0.8 if bool(meta.get("forceRetrain")) else 1.0
        base_score *= drift_penalty * retrain_penalty
    return float(base_score)


def ensemble_primary_weight(
    primary_meta: Optional[Dict[str, object]],
    fallback_meta: Optional[Dict[str, object]],
) -> float:
    min_w = max(0.0, min(1.0, float(MODEL_ENSEMBLE_MIN_PRIMARY_WEIGHT)))
    max_w = max(min_w, min(1.0, float(MODEL_ENSEMBLE_MAX_PRIMARY_WEIGHT)))
    default_w = max(min_w, min(max_w, float(MODEL_ENSEMBLE_DEFAULT_PRIMARY_WEIGHT)))

    if not MODEL_ENSEMBLE_BLEND_ENABLED:
        return 1.0

    primary_score = model_quality_score_for_blend(primary_meta)
    fallback_score = model_quality_score_for_blend(fallback_meta)

    if primary_score <= 0.0 and fallback_score <= 0.0:
        return default_w
    if primary_score > 0.0 and fallback_score <= 0.0:
        return max(default_w, min(0.9, max_w))
    if primary_score <= 0.0 and fallback_score > 0.0:
        return max(min_w, min(max_w, default_w * 0.75))

    ratio = primary_score / max(1e-6, primary_score + fallback_score)
    blended = 0.5 * default_w + 0.5 * float(ratio)
    return max(min_w, min(max_w, blended))


def regime_mae_for_target(
    target: datetime,
    profile: Optional[Dict[str, object]],
    hours_ahead: Optional[float] = None,
) -> Optional[float]:
    if not isinstance(profile, dict):
        return None
    min_points = max(1, int(MODEL_POINT_BIAS_MIN_POINTS_PER_SEGMENT))

    by_horizon = profile.get("byHorizon", {})
    horizon_hours = forecast_horizon_hours(target, hours_ahead=hours_ahead)
    horizon_stats = (
        by_horizon.get(horizon_bucket_key(horizon_hours))
        if isinstance(by_horizon, dict)
        else None
    )
    if isinstance(horizon_stats, dict) and int(horizon_stats.get("count", 0) or 0) >= min_points:
        mae = to_float_or_none(horizon_stats.get("mae"))
        if mae is not None and mae > 0:
            return float(mae)

    by_hour_block = profile.get("byHourBlock", {})
    block_stats = by_hour_block.get(hour_block_key(int(target.hour))) if isinstance(by_hour_block, dict) else None
    if isinstance(block_stats, dict) and int(block_stats.get("count", 0) or 0) >= min_points:
        mae = to_float_or_none(block_stats.get("mae"))
        if mae is not None and mae > 0:
            return float(mae)

    by_day_type = profile.get("byDayType", {})
    day_key = "weekend" if int(target.weekday()) >= 5 else "weekday"
    day_stats = by_day_type.get(day_key) if isinstance(by_day_type, dict) else None
    if isinstance(day_stats, dict) and int(day_stats.get("count", 0) or 0) >= min_points:
        mae = to_float_or_none(day_stats.get("mae"))
        if mae is not None and mae > 0:
            return float(mae)

    global_stats = profile.get("global", {})
    mae = to_float_or_none(global_stats.get("mae")) if isinstance(global_stats, dict) else None
    if mae is not None and mae > 0:
        return float(mae)
    return None


def ensemble_primary_weight_for_target(
    target: datetime,
    primary_meta: Optional[Dict[str, object]],
    fallback_meta: Optional[Dict[str, object]],
    hours_ahead: Optional[float] = None,
) -> float:
    base_weight = ensemble_primary_weight(primary_meta, fallback_meta)
    min_w = max(0.0, min(1.0, float(MODEL_ENSEMBLE_MIN_PRIMARY_WEIGHT)))
    max_w = max(min_w, min(1.0, float(MODEL_ENSEMBLE_MAX_PRIMARY_WEIGHT)))

    primary_profile = primary_meta.get("regimeProfile") if isinstance(primary_meta, dict) else None
    fallback_profile = fallback_meta.get("regimeProfile") if isinstance(fallback_meta, dict) else None
    primary_mae = regime_mae_for_target(target, primary_profile, hours_ahead=hours_ahead)
    fallback_mae = regime_mae_for_target(target, fallback_profile, hours_ahead=hours_ahead)
    if primary_mae is None or fallback_mae is None:
        return max(min_w, min(max_w, float(base_weight)))

    primary_score = 1.0 / max(1e-4, float(primary_mae))
    fallback_score = 1.0 / max(1e-4, float(fallback_mae))
    ratio = primary_score / max(1e-6, primary_score + fallback_score)
    target_weight = 0.6 * float(base_weight) + 0.4 * float(ratio)
    return max(min_w, min(max_w, target_weight))


def cached_ensemble_primary_weight(
    cache: Optional[Dict[Tuple[str, str, datetime], float]],
    primary_key: Optional[str],
    fallback_key: Optional[str],
    target: datetime,
    primary_meta: Optional[Dict[str, object]],
    fallback_meta: Optional[Dict[str, object]],
    hours_ahead: float,
) -> float:
    cache_key = (
        str(primary_key or ""),
        str(fallback_key or ""),
        target,
    )
    if isinstance(cache, dict):
        cached_weight = to_float_or_none(cache.get(cache_key))
        if cached_weight is not None:
            return float(max(0.0, min(1.0, cached_weight)))
    weight = ensemble_primary_weight_for_target(
        target=target,
        primary_meta=primary_meta,
        fallback_meta=fallback_meta,
        hours_ahead=hours_ahead,
    )
    if isinstance(cache, dict):
        cache[cache_key] = float(weight)
    return float(weight)


def cached_point_bias_value(
    cache: Optional[Dict[Tuple[str, int, str, str, str], float]],
    model_key: Optional[str],
    target: datetime,
    horizon_key: str,
    day_type_key: str,
    profile: Optional[Dict[str, object]],
    point_ratio: Optional[float],
    hours_ahead: float,
) -> float:
    occ_key = (
        occupancy_bucket_key_from_ratio(float(point_ratio))
        if point_ratio is not None
        else "__none__"
    )
    cache_key = (
        str(model_key or ""),
        int(target.hour),
        str(horizon_key),
        str(day_type_key),
        str(occ_key),
    )
    if isinstance(cache, dict):
        cached_bias = to_float_or_none(cache.get(cache_key))
        if cached_bias is not None:
            return float(cached_bias)
    bias = point_bias_for_target(
        target,
        profile,
        hours_ahead=hours_ahead,
        point_ratio=point_ratio,
    )
    if isinstance(cache, dict):
        cache[cache_key] = float(bias)
    return float(bias)


def cached_recent_drift_bias_value(
    cache: Optional[Dict[Tuple[str, int, str, str, str], float]],
    model_key: Optional[str],
    target: datetime,
    horizon_key: str,
    day_type_key: str,
    profile: Optional[Dict[str, object]],
    point_ratio: Optional[float],
    hours_ahead: float,
) -> float:
    occ_key = (
        occupancy_bucket_key_from_ratio(float(point_ratio))
        if point_ratio is not None
        else "__none__"
    )
    cache_key = (
        str(model_key or ""),
        int(target.hour),
        str(horizon_key),
        str(day_type_key),
        str(occ_key),
    )
    if isinstance(cache, dict):
        cached_bias = to_float_or_none(cache.get(cache_key))
        if cached_bias is not None:
            return float(cached_bias)
    bias = recent_drift_bias_for_target(
        target,
        profile if isinstance(profile, dict) else None,
        hours_ahead=hours_ahead,
        point_ratio=point_ratio,
    )
    if isinstance(cache, dict):
        cache[cache_key] = float(bias)
    return float(bias)


def cached_conformal_margin_value(
    cache: Optional[Dict[Tuple[str, int, str, str, str], float]],
    model_key: Optional[str],
    target: datetime,
    horizon_hours: int,
    horizon_key: str,
    day_type_key: str,
    profile: Optional[Dict[str, object]],
    point_ratio: Optional[float],
) -> float:
    occ_key = (
        occupancy_bucket_key_from_ratio(float(point_ratio))
        if point_ratio is not None
        else "__none__"
    )
    cache_key = (
        str(model_key or ""),
        int(target.hour),
        str(horizon_key),
        str(day_type_key),
        str(occ_key),
    )
    if isinstance(cache, dict):
        cached_margin = to_float_or_none(cache.get(cache_key))
        if cached_margin is not None:
            return float(max(0.0, cached_margin))
    margin_value = conformal_margin_for_hour(
        int(target.hour),
        int(horizon_hours),
        profile,
        point_ratio=point_ratio,
        target=target,
    )
    if isinstance(cache, dict):
        cache[cache_key] = float(margin_value)
    return float(max(0.0, margin_value))


def adjust_primary_weight_for_feature_quality(
    primary_weight: float,
    primary_missing_ratio: Optional[float],
    fallback_missing_ratio: Optional[float],
) -> float:
    min_w = max(0.0, min(1.0, float(MODEL_ENSEMBLE_MIN_PRIMARY_WEIGHT)))
    max_w = max(min_w, min(1.0, float(MODEL_ENSEMBLE_MAX_PRIMARY_WEIGHT)))
    base = max(min_w, min(max_w, float(primary_weight)))
    if not MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_ENABLED:
        return float(base)
    if primary_missing_ratio is None or fallback_missing_ratio is None:
        return float(base)

    mp = max(0.0, min(1.0, float(primary_missing_ratio)))
    mf = max(0.0, min(1.0, float(fallback_missing_ratio)))
    exp = max(0.5, min(4.0, float(MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_EXP)))
    strength = max(0.0, min(1.0, float(MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_STRENGTH)))
    p_score = max(0.0, 1.0 - mp) ** exp
    f_score = max(0.0, 1.0 - mf) ** exp
    if p_score <= 0.0 and f_score <= 0.0:
        return float(base)

    quality_ratio = p_score / max(1e-6, p_score + f_score)
    adjusted = (1.0 - strength) * float(base) + strength * float(quality_ratio)
    return float(max(min_w, min(max_w, adjusted)))


def adjust_primary_weight_for_sample_support(
    primary_weight: float,
    sample_count: int,
) -> float:
    min_w = max(0.0, min(1.0, float(MODEL_ENSEMBLE_MIN_PRIMARY_WEIGHT)))
    max_w = max(min_w, min(1.0, float(MODEL_ENSEMBLE_MAX_PRIMARY_WEIGHT)))
    base = max(min_w, min(max_w, float(primary_weight)))
    if not MODEL_ENSEMBLE_SAMPLE_SUPPORT_ADJUST_ENABLED:
        return float(base)

    target = max(1, int(MODEL_ENSEMBLE_SAMPLE_SUPPORT_TARGET))
    if int(sample_count) >= target:
        return float(base)
    max_shift = max(0.0, min(1.0, float(MODEL_ENSEMBLE_SAMPLE_SUPPORT_MAX_SHIFT)))
    if max_shift <= 0.0:
        return float(base)

    support = max(0.0, min(1.0, float(max(0, int(sample_count))) / float(target)))
    shift = max_shift * (1.0 - support)
    adjusted = (1.0 - shift) * float(base) + shift * float(min_w)
    return float(max(min_w, min(max_w, adjusted)))


def blend_prediction_triples(
    primary_pred: Tuple[float, float, float],
    fallback_pred: Tuple[float, float, float],
    primary_weight: float,
) -> Tuple[float, float, float]:
    w = max(0.0, min(1.0, float(primary_weight)))
    wf = 1.0 - w
    p10 = w * float(primary_pred[0]) + wf * float(fallback_pred[0])
    p50 = w * float(primary_pred[1]) + wf * float(fallback_pred[1])
    p90 = w * float(primary_pred[2]) + wf * float(fallback_pred[2])

    if not math.isfinite(p50):
        p50 = 0.0
    if not math.isfinite(p10):
        p10 = p50
    if not math.isfinite(p90):
        p90 = p50
    if p10 > p90:
        p10, p90 = p90, p10
    p50 = min(max(p50, p10), p90)
    return float(p10), float(p50), float(p90)


def latest_live_ratio_and_age_minutes(
    loc_entry: Optional[Dict[str, object]],
    max_cap: int,
    now: datetime,
) -> Tuple[Optional[float], Optional[float]]:
    if max_cap <= 0:
        return None, None
    live_count = latest_live_count_for_location(loc_entry, now)
    if live_count is None:
        return None, None
    raw_times = (loc_entry or {}).get("raw_times") or []
    if not raw_times:
        return None, None
    latest_ts = raw_times[-1]
    if not isinstance(latest_ts, datetime):
        return None, None
    age_min = max(0.0, (now - latest_ts).total_seconds() / 60.0)
    return clamp_ratio(float(live_count) / float(max_cap)), float(age_min)


def apply_live_bias_correction(
    p10_ratio: float,
    p50_ratio: float,
    p90_ratio: float,
    live_ratio: Optional[float],
    age_min: Optional[float],
    hours_ahead: float,
) -> Tuple[float, float, float]:
    if not LIVE_BIAS_ENABLED:
        return p10_ratio, p50_ratio, p90_ratio
    if live_ratio is None or age_min is None:
        return p10_ratio, p50_ratio, p90_ratio
    if age_min > max(0.0, float(LIVE_BIAS_MAX_AGE_MIN)):
        return p10_ratio, p50_ratio, p90_ratio
    if hours_ahead < 0.0 or hours_ahead > max(0.0, float(LIVE_BIAS_MAX_HORIZON_HOURS)):
        return p10_ratio, p50_ratio, p90_ratio

    base = max(0.0, min(1.0, float(LIVE_BIAS_BASE_WEIGHT)))
    horizon_decay = max(0.05, float(LIVE_BIAS_HORIZON_DECAY))
    age_decay = max(1.0, float(LIVE_BIAS_AGE_DECAY_MIN))
    weight = base * math.exp(-hours_ahead / horizon_decay) * math.exp(-age_min / age_decay)
    weight = max(0.0, min(0.9, float(weight)))
    if weight <= 0.0:
        return p10_ratio, p50_ratio, p90_ratio

    center = float(p50_ratio)
    low_width = max(0.0, center - float(p10_ratio))
    high_width = max(0.0, float(p90_ratio) - center)
    shifted_center = (1.0 - weight) * center + weight * float(live_ratio)
    shifted_p10 = shifted_center - low_width
    shifted_p90 = shifted_center + high_width
    if shifted_p10 > shifted_p90:
        shifted_p10, shifted_p90 = shifted_p90, shifted_p10
    shifted_center = min(max(shifted_center, shifted_p10), shifted_p90)
    return float(shifted_p10), float(shifted_center), float(shifted_p90)


def apply_low_sample_blend(
    p10_ratio: float,
    p50_ratio: float,
    p90_ratio: float,
    fallback_ratio: float,
    sample_count: int,
) -> Tuple[float, float, float]:
    if not MODEL_LOW_SAMPLE_BLEND_ENABLED:
        return p10_ratio, p50_ratio, p90_ratio

    target_count = max(1, int(MODEL_LOW_SAMPLE_TARGET_COUNT))
    max_blend = max(0.0, min(1.0, float(MODEL_LOW_SAMPLE_MAX_BLEND)))
    if sample_count >= target_count or max_blend <= 0.0:
        return p10_ratio, p50_ratio, p90_ratio

    support = max(0.0, min(1.0, float(sample_count) / float(target_count)))
    blend_weight = (1.0 - support) * max_blend
    center = float(p50_ratio)
    low_width = max(0.0, center - float(p10_ratio))
    high_width = max(0.0, float(p90_ratio) - center)
    blended_center = (1.0 - blend_weight) * center + blend_weight * float(fallback_ratio)
    blended_p10 = blended_center - low_width
    blended_p90 = blended_center + high_width
    if blended_p10 > blended_p90:
        blended_p10, blended_p90 = blended_p90, blended_p10
    blended_center = min(max(blended_center, blended_p10), blended_p90)
    return float(blended_p10), float(blended_center), float(blended_p90)


def apply_missing_feature_blend(
    p10_ratio: float,
    p50_ratio: float,
    p90_ratio: float,
    fallback_ratio: float,
    missing_ratio: Optional[float],
) -> Tuple[float, float, float]:
    if not MODEL_MISSING_FEATURE_BLEND_ENABLED:
        return p10_ratio, p50_ratio, p90_ratio
    if missing_ratio is None:
        return p10_ratio, p50_ratio, p90_ratio

    mr = max(0.0, min(1.0, float(missing_ratio)))
    start = max(0.0, min(1.0, float(MODEL_MISSING_FEATURE_BLEND_START)))
    full = max(start + 0.01, min(1.0, float(MODEL_MISSING_FEATURE_BLEND_FULL)))
    max_w = max(0.0, min(1.0, float(MODEL_MISSING_FEATURE_BLEND_MAX_WEIGHT)))
    if mr <= start or max_w <= 0.0:
        return p10_ratio, p50_ratio, p90_ratio

    progress = max(0.0, min(1.0, (mr - start) / max(0.01, (full - start))))
    weight = max_w * progress
    center = float(p50_ratio)
    low_width = max(0.0, center - float(p10_ratio))
    high_width = max(0.0, float(p90_ratio) - center)
    blended_center = (1.0 - weight) * center + weight * float(fallback_ratio)
    blended_p10 = blended_center - low_width
    blended_p90 = blended_center + high_width
    if blended_p10 > blended_p90:
        blended_p10, blended_p90 = blended_p90, blended_p10
    blended_center = min(max(blended_center, blended_p10), blended_p90)
    return float(blended_p10), float(blended_center), float(blended_p90)


def missing_feature_interval_multiplier(missing_ratio: Optional[float]) -> float:
    if not MODEL_MISSING_FEATURE_INTERVAL_WIDEN_ENABLED:
        return 1.0
    if missing_ratio is None:
        return 1.0

    mr = max(0.0, min(1.0, float(missing_ratio)))
    start = max(0.0, min(1.0, float(MODEL_MISSING_FEATURE_INTERVAL_WIDEN_START)))
    full = max(start + 0.01, min(1.0, float(MODEL_MISSING_FEATURE_INTERVAL_WIDEN_FULL)))
    max_mult = max(1.0, float(MODEL_MISSING_FEATURE_INTERVAL_WIDEN_MAX_MULT))
    if mr <= start or max_mult <= 1.0:
        return 1.0

    progress = max(0.0, min(1.0, (mr - start) / max(0.01, (full - start))))
    return float(1.0 + (max_mult - 1.0) * progress)


def ensemble_disagreement_interval_multiplier(disagreement: Optional[float]) -> float:
    if not MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_ENABLED:
        return 1.0
    if disagreement is None:
        return 1.0

    diff = max(0.0, float(disagreement))
    min_diff = max(0.0, float(MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_MIN_DIFF))
    scale = max(0.0, float(MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_SCALE))
    max_mult = max(1.0, float(MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_MAX_MULT))
    if diff <= min_diff or max_mult <= 1.0 or scale <= 0.0:
        return 1.0

    raw = 1.0 + (diff - min_diff) * scale
    return float(max(1.0, min(max_mult, raw)))


def sample_support_interval_multiplier(sample_count: int) -> float:
    if not MODEL_SAMPLE_SUPPORT_INTERVAL_WIDEN_ENABLED:
        return 1.0

    target = max(1, int(MODEL_SAMPLE_SUPPORT_INTERVAL_TARGET))
    max_mult = max(1.0, float(MODEL_SAMPLE_SUPPORT_INTERVAL_MAX_MULT))
    sc = max(0, int(sample_count))
    if sc >= target or max_mult <= 1.0:
        return 1.0

    support = max(0.0, min(1.0, float(sc) / float(target)))
    return float(1.0 + (max_mult - 1.0) * (1.0 - support))


def apply_long_horizon_stability_blend(
    p10_ratio: float,
    p50_ratio: float,
    p90_ratio: float,
    fallback_ratio: float,
    hours_ahead: float,
) -> Tuple[float, float, float]:
    if not MODEL_LONG_HORIZON_BLEND_ENABLED:
        return p10_ratio, p50_ratio, p90_ratio

    start_h = max(0.0, float(MODEL_LONG_HORIZON_BLEND_START_HOURS))
    full_h = max(start_h + 0.5, float(MODEL_LONG_HORIZON_BLEND_FULL_HOURS))
    max_w = max(0.0, min(1.0, float(MODEL_LONG_HORIZON_BLEND_MAX_WEIGHT)))
    if hours_ahead <= start_h or max_w <= 0.0:
        return p10_ratio, p50_ratio, p90_ratio

    progress = max(0.0, min(1.0, (float(hours_ahead) - start_h) / max(0.5, (full_h - start_h))))
    weight = max_w * progress
    center = float(p50_ratio)
    low_width = max(0.0, center - float(p10_ratio))
    high_width = max(0.0, float(p90_ratio) - center)
    blended_center = (1.0 - weight) * center + weight * float(fallback_ratio)
    blended_p10 = blended_center - low_width
    blended_p90 = blended_center + high_width
    if blended_p10 > blended_p90:
        blended_p10, blended_p90 = blended_p90, blended_p10
    blended_center = min(max(blended_center, blended_p10), blended_p90)
    return float(blended_p10), float(blended_center), float(blended_p90)


def recursive_override_map_for_model(
    recursive_ratio_cache: Optional[Dict[Tuple[str, int], Dict[datetime, float]]],
    model_key: Optional[str],
    loc_id: int,
    create: bool = True,
) -> Optional[Dict[datetime, float]]:
    if model_key is None or not isinstance(recursive_ratio_cache, dict):
        return None
    cache_key = (str(model_key), int(loc_id))
    existing = recursive_ratio_cache.get(cache_key)
    if isinstance(existing, dict):
        return existing
    if not create:
        return None
    created: Dict[datetime, float] = {}
    recursive_ratio_cache[cache_key] = created
    return created


def estimate_location(
    loc_id: int,
    target: datetime,
    ctx: Dict[str, object],
) -> Dict[str, float]:
    loc_id_int = int(loc_id)
    target_key = target
    location_estimate_cache = ctx.get("location_estimate_cache")
    estimate_cache_key = (int(loc_id_int), target_key)
    if isinstance(location_estimate_cache, dict):
        cached_estimate = location_estimate_cache.get(estimate_cache_key)
        if isinstance(cached_estimate, dict):
            return cached_estimate

    def cache_and_return(result: Dict[str, float]) -> Dict[str, float]:
        if isinstance(location_estimate_cache, dict):
            location_estimate_cache[estimate_cache_key] = result
        return result

    max_caps = ctx["max_caps"]
    max_cap = max_caps.get(loc_id_int, 0)
    if max_cap <= 0:
        return cache_and_return(
            {
                "countP10": 0.0,
                "countP50": 0.0,
                "countP90": 0.0,
                "sampleCount": 0.0,
            }
        )

    loc_to_model_key = ctx.get("loc_to_model_key", {})
    loc_to_fallback_key = ctx.get("loc_to_fallback_key", {})
    models_by_key = ctx.get("models_by_key", {})
    model_meta_by_key = ctx.get("model_meta_by_key", {})
    primary_key = loc_to_model_key.get(loc_id_int)
    fallback_key = loc_to_fallback_key.get(loc_id_int)
    now = ctx.get("now")
    hours_ahead = 0.0
    if isinstance(now, datetime):
        hours_ahead = max(0.0, (target - now).total_seconds() / 3600.0)

    loc_data = ctx["loc_data"]
    loc_samples = ctx["loc_samples"]
    loc_entry = loc_data.get(loc_id_int)
    interval_profile_by_key = ctx.get("interval_profile_by_key", {})
    conformal_by_key = ctx.get("conformal_by_key", {})
    interval_multiplier_by_key = ctx.get("interval_multiplier_by_key", {})
    recursive_ratio_cache = ctx.get("recursive_ratio_cache")
    feature_cache = ctx.get("feature_cache")
    feature_matrix_cache = ctx.get("feature_matrix_cache")
    ensemble_weight_cache = ctx.get("ensemble_weight_cache")
    point_bias_value_cache = ctx.get("point_bias_value_cache")
    recent_drift_bias_value_cache = ctx.get("recent_drift_bias_value_cache")
    conformal_margin_cache = ctx.get("conformal_margin_cache")
    weather_source = ctx.get("weather_series")
    weather_lookup_cache = ctx.get("weather_lookup_cache")
    onehot_by_key = ctx.get("onehot_by_key", {})
    feature_fill_values_by_key = ctx.get("feature_fill_values_by_key", {})
    feature_clip_bounds_by_key = ctx.get("feature_clip_bounds_by_key", {})
    recent_drift_bias_by_key = ctx.get("recent_drift_bias_by_key", {})
    model_prediction_cache = ctx.get("model_prediction_cache")
    horizon_hours = forecast_horizon_hours(target, hours_ahead=hours_ahead)
    horizon_key = horizon_bucket_key(horizon_hours)
    day_type_key = "weekend" if int(target.weekday()) >= 5 else "weekday"

    def seed_boundary_zero_override(model_key: Optional[str]) -> None:
        existing = recursive_override_map_for_model(
            recursive_ratio_cache if isinstance(recursive_ratio_cache, dict) else None,
            model_key,
            loc_id_int,
            create=True,
        )
        if not isinstance(existing, dict):
            return
        seed_recursive_ratio_overrides(existing, target, 0.0)

    sample_count: Optional[int] = None

    def resolve_sample_count() -> int:
        nonlocal sample_count
        if sample_count is None:
            sample_count = int(
                get_sample_count(
                    loc_id_int,
                    target,
                    ctx["avg_dow_hour"],
                    ctx["avg_hour"],
                    ctx["avg_overall"],
                )
            )
        return int(sample_count)

    if SCHEDULE_BOUNDARY_ZERO_ENABLED:
        facility_schedule_by_id = ctx.get("facility_schedule_by_id", {})
        location_facility_map = ctx.get("location_facility_map", {})
        schedule_boundary_cache = ctx.get("schedule_boundary_cache")
        schedule_date_range_cache = ctx.get("schedule_date_range_cache")
        schedule_weekday_cache = ctx.get("schedule_weekday_cache")
        schedule_hours_cache = ctx.get("schedule_hours_cache")
        if (
            isinstance(facility_schedule_by_id, dict)
            and isinstance(location_facility_map, dict)
            and isinstance(schedule_boundary_cache, dict)
            and isinstance(schedule_date_range_cache, dict)
            and isinstance(schedule_weekday_cache, dict)
            and isinstance(schedule_hours_cache, dict)
        ):
            facility_id_raw = location_facility_map.get(int(loc_id_int))
            try:
                facility_id = int(facility_id_raw) if facility_id_raw is not None else None
            except Exception:
                facility_id = None
            if facility_id is not None and facility_id in facility_schedule_by_id:
                sections_raw = facility_schedule_by_id.get(facility_id, {}).get("sections", [])
                sections = sections_raw if isinstance(sections_raw, list) else []
                if sections:
                    minute_of_day = int(target.hour) * 60 + int(target.minute)
                    schedule_cache_key = (int(facility_id), target.date(), minute_of_day)
                    if schedule_cache_key not in schedule_boundary_cache:
                        schedule_boundary_cache[schedule_cache_key] = get_facility_schedule_boundary_state(
                            sections,
                            target,
                            date_range_cache=schedule_date_range_cache,
                            weekday_cache=schedule_weekday_cache,
                            hours_window_cache=schedule_hours_cache,
                        )
                    boundary_open, boundary_close = schedule_boundary_cache.get(
                        schedule_cache_key,
                        (False, False),
                    )
                    if boundary_open or boundary_close:
                        seed_boundary_zero_override(primary_key)
                        if fallback_key and fallback_key != primary_key:
                            seed_boundary_zero_override(fallback_key)
                        return cache_and_return(
                            {
                            "countP10": 0.0,
                            "countP50": 0.0,
                            "countP90": 0.0,
                            "sampleCount": float(resolve_sample_count()),
                            }
                        )

    def override_map_for_model(model_key: Optional[str]) -> Optional[Dict[datetime, float]]:
        return recursive_override_map_for_model(
            recursive_ratio_cache if isinstance(recursive_ratio_cache, dict) else None,
            model_key,
            loc_id_int,
            create=True,
        )

    def predict_for_model(
        model_key: Optional[str],
        lag_ratio_override: Optional[Dict[datetime, float]],
    ) -> Optional[Tuple[float, float, float, float]]:
        if not model_key:
            return None
        batch_cache_key = (str(model_key), int(loc_id_int), target_key)
        if isinstance(model_prediction_cache, dict):
            cached_model_pred = model_prediction_cache.get(batch_cache_key)
            if isinstance(cached_model_pred, tuple) and len(cached_model_pred) >= 4:
                cached_p10 = to_float_or_none(cached_model_pred[0])
                cached_p50 = to_float_or_none(cached_model_pred[1])
                cached_p90 = to_float_or_none(cached_model_pred[2])
                cached_missing = to_float_or_none(cached_model_pred[3])
                if (
                    cached_p10 is not None
                    and cached_p50 is not None
                    and cached_p90 is not None
                ):
                    seed_recursive_ratio_overrides(lag_ratio_override, target, float(cached_p50))
                    return (
                        float(cached_p10),
                        float(cached_p50),
                        float(cached_p90),
                        float(cached_missing if cached_missing is not None else 0.0),
                    )
        model_bundle = models_by_key.get(model_key)
        if (
            model_bundle is None
            or model_bundle.get("p50") is None
            or loc_entry is None
            or loc_samples.get(loc_id_int, 0) < MIN_SAMPLES_PER_LOC
            or bool(loc_entry.get("is_stale"))
        ):
            return None

        onehot_vec = onehot_by_key.get(model_key, {}).get(loc_id_int)
        if onehot_vec is None:
            return None

        feature_vec = None
        missing_ratio = None
        feature_cache_key = (str(model_key), int(loc_id_int), target_key)
        if isinstance(feature_cache, dict):
            cached_feature = feature_cache.get(feature_cache_key)
            if isinstance(cached_feature, tuple) and len(cached_feature) >= 1:
                feature_vec = cached_feature[0]
                if len(cached_feature) >= 2:
                    missing_ratio = to_float_or_none(cached_feature[1])
            else:
                feature_vec = cached_feature
        if feature_vec is None:
            feature_vec = build_features(
                target,
                loc_entry,
                onehot_vec,
                weather_source=weather_source,
                weather_lookup_cache=weather_lookup_cache,
                lag_ratio_override=lag_ratio_override,
            )
        if missing_ratio is None and isinstance(feature_vec, list):
            missing_ratio = feature_missing_rate(feature_vec)
        if isinstance(feature_cache, dict):
            feature_cache[feature_cache_key] = (
                feature_vec,
                float(missing_ratio) if missing_ratio is not None else 0.0,
            )

        dmatrix = None
        if isinstance(feature_matrix_cache, dict):
            cached_dmatrix = feature_matrix_cache.get(feature_cache_key)
            if cached_dmatrix is not None:
                dmatrix = cached_dmatrix
        if dmatrix is None:
            features_arr = sanitize_feature_matrix(np.array([feature_vec], dtype=np.float32))
            if isinstance(feature_fill_values_by_key, dict):
                features_arr = apply_feature_fill_values(
                    features_arr,
                    feature_fill_values_by_key.get(model_key),
                )
            if isinstance(feature_clip_bounds_by_key, dict):
                features_arr = apply_feature_clip_bounds(
                    features_arr,
                    feature_clip_bounds_by_key.get(model_key),
                )
            dmatrix = xgb.DMatrix(features_arr)
            if isinstance(feature_matrix_cache, dict):
                feature_matrix_cache[feature_cache_key] = dmatrix
        p50_model = model_bundle.get("p50")
        p10_model = model_bundle.get("p10")
        p90_model = model_bundle.get("p90")
        p50_ratio = float(p50_model.predict(dmatrix)[0]) if p50_model is not None else None
        if p50_ratio is None or not math.isfinite(p50_ratio):
            return None

        if p10_model is not None and p90_model is not None:
            p10_ratio = float(p10_model.predict(dmatrix)[0])
            p90_ratio = float(p90_model.predict(dmatrix)[0])
            if not math.isfinite(p10_ratio) or not math.isfinite(p90_ratio):
                interval_profile = interval_profile_by_key.get(model_key)
                p10_ratio, p50_from_interval, p90_ratio = interval_bounds(
                    point_ratio=float(p50_ratio),
                    hour=target.hour,
                    residual_profile=interval_profile,
                    target=target,
                )
                p50_ratio = float(p50_from_interval)
        else:
            interval_profile = interval_profile_by_key.get(model_key)
            p10_ratio, p50_from_interval, p90_ratio = interval_bounds(
                point_ratio=float(p50_ratio),
                hour=target.hour,
                residual_profile=interval_profile,
                target=target,
            )
            p50_ratio = float(p50_from_interval)

        p10_ratio = clamp_ratio(float(p10_ratio))
        p50_ratio = clamp_ratio(float(p50_ratio))
        p90_ratio = clamp_ratio(float(p90_ratio))
        if p10_ratio > p90_ratio:
            p10_ratio, p90_ratio = p90_ratio, p10_ratio
        p50_ratio = min(max(p50_ratio, p10_ratio), p90_ratio)
        missing_ratio = float(missing_ratio) if missing_ratio is not None else 0.0
        if not math.isfinite(missing_ratio):
            missing_ratio = 0.0
        if isinstance(model_prediction_cache, dict):
            model_prediction_cache[batch_cache_key] = (
                float(p10_ratio),
                float(p50_ratio),
                float(p90_ratio),
                float(missing_ratio),
            )
        return float(p10_ratio), float(p50_ratio), float(p90_ratio), float(missing_ratio)

    prediction_cache = ctx.get("prediction_cache")
    cache_key = (str(primary_key or ""), str(fallback_key or ""), int(loc_id_int), target_key)
    if isinstance(prediction_cache, dict):
        cached = prediction_cache.get(cache_key)
        if cached is not None:
            cached_sample_count = None
            if isinstance(cached, tuple) and len(cached) >= 4:
                p10_ratio, p50_ratio, p90_ratio, cached_sample_count = cached
            else:
                p10_ratio, p50_ratio, p90_ratio = cached
            seed_recursive_ratio_overrides(override_map_for_model(primary_key), target, float(p50_ratio))
            if fallback_key and fallback_key != primary_key:
                seed_recursive_ratio_overrides(override_map_for_model(fallback_key), target, float(p50_ratio))
            out_sample_count = to_float_or_none(cached_sample_count)
            if out_sample_count is None:
                out_sample_count = float(resolve_sample_count())
            return cache_and_return(
                {
                "countP10": clamp_ratio(p10_ratio) * max_cap,
                "countP50": clamp_ratio(p50_ratio) * max_cap,
                "countP90": clamp_ratio(p90_ratio) * max_cap,
                "sampleCount": float(out_sample_count),
                }
            )

    primary_override = override_map_for_model(primary_key)
    fallback_override = override_map_for_model(fallback_key) if fallback_key != primary_key else primary_override

    primary_pred = predict_for_model(primary_key, primary_override)
    fallback_pred = None
    need_fallback_pred = bool(
        fallback_key
        and fallback_key != primary_key
        and (MODEL_ENSEMBLE_BLEND_ENABLED or primary_pred is None)
    )
    if need_fallback_pred:
        fallback_pred = predict_for_model(fallback_key, fallback_override)

    if MODEL_DIRECT_HORIZON_ENABLED and isinstance(model_meta_by_key, dict):
        if primary_pred is not None and primary_key:
            primary_profile = (model_meta_by_key.get(primary_key) or {}).get("directHorizonProfile")
            primary_adjusted = apply_direct_horizon_adjustment(
                float(primary_pred[0]),
                float(primary_pred[1]),
                float(primary_pred[2]),
                target=target,
                profile=primary_profile if isinstance(primary_profile, dict) else None,
                hours_ahead=hours_ahead,
            )
            primary_pred = (
                float(primary_adjusted[0]),
                float(primary_adjusted[1]),
                float(primary_adjusted[2]),
                float(primary_pred[3]),
            )
        if fallback_pred is not None and fallback_key and fallback_key != primary_key:
            fallback_profile = (model_meta_by_key.get(fallback_key) or {}).get("directHorizonProfile")
            fallback_adjusted = apply_direct_horizon_adjustment(
                float(fallback_pred[0]),
                float(fallback_pred[1]),
                float(fallback_pred[2]),
                target=target,
                profile=fallback_profile if isinstance(fallback_profile, dict) else None,
                hours_ahead=hours_ahead,
            )
            fallback_pred = (
                float(fallback_adjusted[0]),
                float(fallback_adjusted[1]),
                float(fallback_adjusted[2]),
                float(fallback_pred[3]),
            )

    sample_count = resolve_sample_count()
    active_key = None
    blend_primary_weight = None
    blend_disagreement = None
    p10_ratio = None
    p50_ratio = None
    p90_ratio = None
    prediction_missing_ratio = None
    if primary_pred is not None and fallback_pred is not None and MODEL_ENSEMBLE_BLEND_ENABLED:
        primary_meta = model_meta_by_key.get(primary_key) if isinstance(model_meta_by_key, dict) else None
        fallback_meta = model_meta_by_key.get(fallback_key) if isinstance(model_meta_by_key, dict) else None
        weight_primary = cached_ensemble_primary_weight(
            cache=ensemble_weight_cache if isinstance(ensemble_weight_cache, dict) else None,
            primary_key=primary_key,
            fallback_key=fallback_key,
            target=target,
            primary_meta=primary_meta,
            fallback_meta=fallback_meta,
            hours_ahead=hours_ahead,
        )
        weight_primary = adjust_primary_weight_for_feature_quality(
            primary_weight=float(weight_primary),
            primary_missing_ratio=float(primary_pred[3]) if primary_pred is not None else None,
            fallback_missing_ratio=float(fallback_pred[3]) if fallback_pred is not None else None,
        )
        weight_primary = adjust_primary_weight_for_sample_support(
            primary_weight=float(weight_primary),
            sample_count=int(sample_count),
        )
        blend_primary_weight = float(weight_primary)
        blend_disagreement = abs(float(primary_pred[1]) - float(fallback_pred[1]))
        p10_ratio, p50_ratio, p90_ratio = blend_prediction_triples(
            primary_pred=(float(primary_pred[0]), float(primary_pred[1]), float(primary_pred[2])),
            fallback_pred=(float(fallback_pred[0]), float(fallback_pred[1]), float(fallback_pred[2])),
            primary_weight=weight_primary,
        )
        prediction_missing_ratio = float(weight_primary) * float(primary_pred[3]) + (
            1.0 - float(weight_primary)
        ) * float(fallback_pred[3])
        active_key = primary_key or fallback_key
    elif primary_pred is not None:
        p10_ratio, p50_ratio, p90_ratio = float(primary_pred[0]), float(primary_pred[1]), float(primary_pred[2])
        prediction_missing_ratio = float(primary_pred[3])
        active_key = primary_key
    elif fallback_pred is not None:
        p10_ratio, p50_ratio, p90_ratio = float(fallback_pred[0]), float(fallback_pred[1]), float(fallback_pred[2])
        prediction_missing_ratio = float(fallback_pred[3])
        active_key = fallback_key

    interval_profile = interval_profile_by_key.get(active_key) if active_key else None
    conformal_profile = conformal_by_key.get(active_key) if active_key else None
    interval_multiplier = float(interval_multiplier_by_key.get(active_key, 1.0) or 1.0)
    fallback_ratio: Optional[float] = None

    if p50_ratio is None:
        fallback_ratio = fallback_ratio_for_location(
            loc_id=loc_id_int,
            target=target,
            avg_dow_hour=ctx["avg_dow_hour"],
            avg_hour=ctx["avg_hour"],
            avg_overall=ctx["avg_overall"],
        )
        p10_ratio, p50_ratio, p90_ratio = interval_bounds(
            point_ratio=float(fallback_ratio),
            hour=target.hour,
            residual_profile=interval_profile,
            target=target,
        )
        prediction_missing_ratio = 1.0

    point_bias = 0.0
    if isinstance(model_meta_by_key, dict):
        if (
            blend_primary_weight is not None
            and fallback_key
            and fallback_key != primary_key
        ):
            primary_profile = (model_meta_by_key.get(primary_key) or {}).get("pointBiasProfile")
            fallback_profile = (model_meta_by_key.get(fallback_key) or {}).get("pointBiasProfile")
            primary_bias = cached_point_bias_value(
                cache=point_bias_value_cache if isinstance(point_bias_value_cache, dict) else None,
                model_key=primary_key,
                target=target,
                horizon_key=horizon_key,
                day_type_key=day_type_key,
                profile=primary_profile if isinstance(primary_profile, dict) else None,
                point_ratio=float(primary_pred[1]) if primary_pred is not None else None,
                hours_ahead=hours_ahead,
            )
            fallback_bias = cached_point_bias_value(
                cache=point_bias_value_cache if isinstance(point_bias_value_cache, dict) else None,
                model_key=fallback_key,
                target=target,
                horizon_key=horizon_key,
                day_type_key=day_type_key,
                profile=fallback_profile if isinstance(fallback_profile, dict) else None,
                point_ratio=float(fallback_pred[1]) if fallback_pred is not None else None,
                hours_ahead=hours_ahead,
            )
            point_bias = float(blend_primary_weight) * float(primary_bias) + (
                1.0 - float(blend_primary_weight)
            ) * float(fallback_bias)
        elif active_key:
            active_profile = (model_meta_by_key.get(active_key) or {}).get("pointBiasProfile")
            point_bias = cached_point_bias_value(
                cache=point_bias_value_cache if isinstance(point_bias_value_cache, dict) else None,
                model_key=active_key,
                target=target,
                horizon_key=horizon_key,
                day_type_key=day_type_key,
                profile=active_profile if isinstance(active_profile, dict) else None,
                point_ratio=float(p50_ratio) if p50_ratio is not None else None,
                hours_ahead=hours_ahead,
            )

    recent_drift_bias = 0.0
    if isinstance(recent_drift_bias_by_key, dict):
        if (
            blend_primary_weight is not None
            and fallback_key
            and fallback_key != primary_key
        ):
            primary_recent_profile = recent_drift_bias_by_key.get(primary_key)
            fallback_recent_profile = recent_drift_bias_by_key.get(fallback_key)
            primary_recent_bias = cached_recent_drift_bias_value(
                cache=recent_drift_bias_value_cache if isinstance(recent_drift_bias_value_cache, dict) else None,
                model_key=primary_key,
                target=target,
                horizon_key=horizon_key,
                day_type_key=day_type_key,
                profile=primary_recent_profile if isinstance(primary_recent_profile, dict) else None,
                point_ratio=float(primary_pred[1]) if primary_pred is not None else None,
                hours_ahead=hours_ahead,
            )
            fallback_recent_bias = cached_recent_drift_bias_value(
                cache=recent_drift_bias_value_cache if isinstance(recent_drift_bias_value_cache, dict) else None,
                model_key=fallback_key,
                target=target,
                horizon_key=horizon_key,
                day_type_key=day_type_key,
                profile=fallback_recent_profile if isinstance(fallback_recent_profile, dict) else None,
                point_ratio=float(fallback_pred[1]) if fallback_pred is not None else None,
                hours_ahead=hours_ahead,
            )
            recent_drift_bias = float(blend_primary_weight) * float(primary_recent_bias) + (
                1.0 - float(blend_primary_weight)
            ) * float(fallback_recent_bias)
        elif active_key:
            active_recent_profile = recent_drift_bias_by_key.get(active_key)
            recent_drift_bias = cached_recent_drift_bias_value(
                cache=recent_drift_bias_value_cache if isinstance(recent_drift_bias_value_cache, dict) else None,
                model_key=active_key,
                target=target,
                horizon_key=horizon_key,
                day_type_key=day_type_key,
                profile=active_recent_profile if isinstance(active_recent_profile, dict) else None,
                point_ratio=float(p50_ratio) if p50_ratio is not None else None,
                hours_ahead=hours_ahead,
            )

    if fallback_ratio is None:
        fallback_ratio = fallback_ratio_for_location(
            loc_id=loc_id_int,
            target=target,
            avg_dow_hour=ctx["avg_dow_hour"],
            avg_hour=ctx["avg_hour"],
            avg_overall=ctx["avg_overall"],
        )
    p10_ratio, p50_ratio, p90_ratio = apply_low_sample_blend(
        float(p10_ratio if p10_ratio is not None else p50_ratio),
        float(p50_ratio),
        float(p90_ratio if p90_ratio is not None else p50_ratio),
        fallback_ratio=float(fallback_ratio),
        sample_count=int(sample_count),
    )
    p10_ratio, p50_ratio, p90_ratio = apply_point_bias_shift(
        float(p10_ratio),
        float(p50_ratio),
        float(p90_ratio),
        bias=float(point_bias) + float(recent_drift_bias),
    )
    p10_ratio, p50_ratio, p90_ratio = apply_missing_feature_blend(
        float(p10_ratio),
        float(p50_ratio),
        float(p90_ratio),
        fallback_ratio=float(fallback_ratio),
        missing_ratio=prediction_missing_ratio,
    )

    if isinstance(now, datetime):
        live_ratio, live_age_min = latest_live_ratio_and_age_minutes(loc_entry, max_cap=max_cap, now=now)
        p10_ratio, p50_ratio, p90_ratio = apply_live_bias_correction(
            float(p10_ratio if p10_ratio is not None else p50_ratio),
            float(p50_ratio),
            float(p90_ratio if p90_ratio is not None else p50_ratio),
            live_ratio=live_ratio,
            age_min=live_age_min,
            hours_ahead=hours_ahead,
        )
    p10_ratio, p50_ratio, p90_ratio = apply_long_horizon_stability_blend(
        float(p10_ratio if p10_ratio is not None else p50_ratio),
        float(p50_ratio),
        float(p90_ratio if p90_ratio is not None else p50_ratio),
        fallback_ratio=float(fallback_ratio),
        hours_ahead=float(hours_ahead),
    )

    margin = cached_conformal_margin_value(
        cache=conformal_margin_cache if isinstance(conformal_margin_cache, dict) else None,
        model_key=active_key,
        target=target,
        horizon_hours=int(horizon_hours),
        horizon_key=horizon_key,
        day_type_key=day_type_key,
        profile=conformal_profile,
        point_ratio=float(p50_ratio) if p50_ratio is not None else None,
    )
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

    missing_interval_mult = missing_feature_interval_multiplier(prediction_missing_ratio)
    if missing_interval_mult > 1.0 and p50_ratio is not None:
        center = float(p50_ratio)
        cur_p10 = float(p10_ratio if p10_ratio is not None else center)
        cur_p90 = float(p90_ratio if p90_ratio is not None else center)
        low_width = max(0.0, center - cur_p10)
        high_width = max(0.0, cur_p90 - center)
        p10_ratio = center - low_width * float(missing_interval_mult)
        p90_ratio = center + high_width * float(missing_interval_mult)

    disagreement_interval_mult = ensemble_disagreement_interval_multiplier(blend_disagreement)
    if disagreement_interval_mult > 1.0 and p50_ratio is not None:
        center = float(p50_ratio)
        cur_p10 = float(p10_ratio if p10_ratio is not None else center)
        cur_p90 = float(p90_ratio if p90_ratio is not None else center)
        low_width = max(0.0, center - cur_p10)
        high_width = max(0.0, cur_p90 - center)
        p10_ratio = center - low_width * float(disagreement_interval_mult)
        p90_ratio = center + high_width * float(disagreement_interval_mult)

    sample_support_interval_mult = sample_support_interval_multiplier(int(sample_count))
    if sample_support_interval_mult > 1.0 and p50_ratio is not None:
        center = float(p50_ratio)
        cur_p10 = float(p10_ratio if p10_ratio is not None else center)
        cur_p90 = float(p90_ratio if p90_ratio is not None else center)
        low_width = max(0.0, center - cur_p10)
        high_width = max(0.0, cur_p90 - center)
        p10_ratio = center - low_width * float(sample_support_interval_mult)
        p90_ratio = center + high_width * float(sample_support_interval_mult)

    p10_ratio = clamp_ratio(float(p10_ratio if p10_ratio is not None else p50_ratio))
    p50_ratio = clamp_ratio(float(p50_ratio))
    p90_ratio = clamp_ratio(float(p90_ratio if p90_ratio is not None else p50_ratio))
    if p10_ratio > p90_ratio:
        p10_ratio, p90_ratio = p90_ratio, p10_ratio
    p50_ratio = min(max(p50_ratio, p10_ratio), p90_ratio)

    if isinstance(prediction_cache, dict):
        prediction_cache[cache_key] = (p10_ratio, p50_ratio, p90_ratio, float(sample_count))
    seed_recursive_ratio_overrides(primary_override, target, float(p50_ratio))
    if fallback_key and fallback_key != primary_key:
        seed_recursive_ratio_overrides(fallback_override, target, float(p50_ratio))

    return cache_and_return(
        {
            "countP10": clamp_ratio(p10_ratio) * max_cap,
            "countP50": clamp_ratio(p50_ratio) * max_cap,
            "countP90": clamp_ratio(p90_ratio) * max_cap,
            "sampleCount": float(sample_count),
        }
    )


def unique_location_ids(loc_ids: Iterable[int]) -> List[int]:
    seen: Set[int] = set()
    normalized: List[int] = []
    for raw_loc_id in loc_ids:
        try:
            loc_id = int(raw_loc_id)
        except Exception:
            continue
        if loc_id in seen:
            continue
        seen.add(loc_id)
        normalized.append(loc_id)
    return normalized


def prime_model_prediction_cache_for_targets(
    loc_ids: Iterable[int],
    targets: Iterable[datetime],
    ctx: Dict[str, object],
) -> None:
    model_prediction_cache = ctx.get("model_prediction_cache")
    if not isinstance(model_prediction_cache, dict):
        return

    models_by_key = ctx.get("models_by_key", {})
    loc_to_model_key = ctx.get("loc_to_model_key", {})
    loc_to_fallback_key = ctx.get("loc_to_fallback_key", {})
    loc_data = ctx.get("loc_data", {})
    loc_samples = ctx.get("loc_samples", {})
    max_caps = ctx.get("max_caps", {})
    onehot_by_key = ctx.get("onehot_by_key", {})
    recursive_ratio_cache = ctx.get("recursive_ratio_cache")
    feature_cache = ctx.get("feature_cache")
    weather_source = ctx.get("weather_series")
    weather_lookup_cache = ctx.get("weather_lookup_cache")
    feature_fill_values_by_key = ctx.get("feature_fill_values_by_key", {})
    feature_clip_bounds_by_key = ctx.get("feature_clip_bounds_by_key", {})
    interval_profile_by_key = ctx.get("interval_profile_by_key", {})

    normalized_targets = sorted(unique_targets_by_iso(targets))
    if not normalized_targets:
        return

    model_to_loc_ids: Dict[str, List[int]] = {}
    for loc_id in unique_location_ids(loc_ids):
        for model_key in (
            loc_to_model_key.get(int(loc_id)),
            loc_to_fallback_key.get(int(loc_id)),
        ):
            if not model_key or model_key not in models_by_key:
                continue
            model_to_loc_ids.setdefault(str(model_key), []).append(int(loc_id))

    if not model_to_loc_ids:
        return

    for target in normalized_targets:
        for model_key, model_loc_ids in model_to_loc_ids.items():
            model_bundle = models_by_key.get(model_key)
            if not isinstance(model_bundle, dict) or model_bundle.get("p50") is None:
                continue
            onehot_map = onehot_by_key.get(model_key, {}) if isinstance(onehot_by_key, dict) else {}
            residual_profile = interval_profile_by_key.get(model_key) if isinstance(interval_profile_by_key, dict) else None
            feature_rows: List[List[float]] = []
            batch_loc_ids: List[int] = []
            batch_missing_ratio: List[float] = []
            batch_override_maps: List[Optional[Dict[datetime, float]]] = []

            for loc_id in model_loc_ids:
                cache_key = (str(model_key), int(loc_id), target)
                override_map = recursive_override_map_for_model(
                    recursive_ratio_cache if isinstance(recursive_ratio_cache, dict) else None,
                    model_key,
                    int(loc_id),
                    create=True,
                )
                cached_pred = model_prediction_cache.get(cache_key)
                if isinstance(cached_pred, tuple) and len(cached_pred) >= 2:
                    cached_p50 = to_float_or_none(cached_pred[1])
                    if cached_p50 is not None:
                        seed_recursive_ratio_overrides(override_map, target, float(cached_p50))
                    continue

                loc_entry = loc_data.get(int(loc_id)) if isinstance(loc_data, dict) else None
                if (
                    not isinstance(loc_entry, dict)
                    or int(max_caps.get(int(loc_id), 0) or 0) <= 0
                    or int(loc_samples.get(int(loc_id), 0) or 0) < MIN_SAMPLES_PER_LOC
                    or bool(loc_entry.get("is_stale"))
                ):
                    continue
                onehot_vec = onehot_map.get(int(loc_id)) if isinstance(onehot_map, dict) else None
                if onehot_vec is None:
                    continue

                feature_vec = None
                missing_ratio = None
                if isinstance(feature_cache, dict):
                    cached_feature = feature_cache.get(cache_key)
                    if isinstance(cached_feature, tuple) and len(cached_feature) >= 1:
                        feature_vec = cached_feature[0]
                        if len(cached_feature) >= 2:
                            missing_ratio = to_float_or_none(cached_feature[1])
                    else:
                        feature_vec = cached_feature
                if feature_vec is None:
                    feature_vec = build_features(
                        target,
                        loc_entry,
                        onehot_vec,
                        weather_source=weather_source,
                        weather_lookup_cache=weather_lookup_cache,
                        lag_ratio_override=override_map,
                    )
                if missing_ratio is None and isinstance(feature_vec, list):
                    missing_ratio = feature_missing_rate(feature_vec)
                if isinstance(feature_cache, dict):
                    feature_cache[cache_key] = (
                        feature_vec,
                        float(missing_ratio) if missing_ratio is not None else 0.0,
                    )

                feature_rows.append(feature_vec)
                batch_loc_ids.append(int(loc_id))
                batch_missing_ratio.append(float(missing_ratio) if missing_ratio is not None else 0.0)
                batch_override_maps.append(override_map)

            if not feature_rows:
                continue

            features_arr = sanitize_feature_matrix(np.array(feature_rows, dtype=np.float32))
            if isinstance(feature_fill_values_by_key, dict):
                features_arr = apply_feature_fill_values(
                    features_arr,
                    feature_fill_values_by_key.get(model_key),
                )
            if isinstance(feature_clip_bounds_by_key, dict):
                features_arr = apply_feature_clip_bounds(
                    features_arr,
                    feature_clip_bounds_by_key.get(model_key),
                )
            dmatrix = xgb.DMatrix(features_arr)
            p50_model = model_bundle.get("p50")
            p10_model = model_bundle.get("p10")
            p90_model = model_bundle.get("p90")
            p50_arr = p50_model.predict(dmatrix).astype(np.float32) if p50_model is not None else np.array([], dtype=np.float32)
            p10_arr = p10_model.predict(dmatrix).astype(np.float32) if p10_model is not None else None
            p90_arr = p90_model.predict(dmatrix).astype(np.float32) if p90_model is not None else None

            for idx, loc_id in enumerate(batch_loc_ids):
                p50_ratio = float(p50_arr[idx]) if idx < p50_arr.size else float("nan")
                if not math.isfinite(p50_ratio):
                    continue
                if p10_arr is not None and p90_arr is not None:
                    p10_ratio = float(p10_arr[idx]) if idx < p10_arr.size else float("nan")
                    p90_ratio = float(p90_arr[idx]) if idx < p90_arr.size else float("nan")
                    if not math.isfinite(p10_ratio) or not math.isfinite(p90_ratio):
                        p10_ratio, p50_from_interval, p90_ratio = interval_bounds(
                            point_ratio=float(p50_ratio),
                            hour=target.hour,
                            residual_profile=residual_profile,
                            target=target,
                        )
                        p50_ratio = float(p50_from_interval)
                else:
                    p10_ratio, p50_from_interval, p90_ratio = interval_bounds(
                        point_ratio=float(p50_ratio),
                        hour=target.hour,
                        residual_profile=residual_profile,
                        target=target,
                    )
                    p50_ratio = float(p50_from_interval)

                p10_ratio = clamp_ratio(float(p10_ratio))
                p50_ratio = clamp_ratio(float(p50_ratio))
                p90_ratio = clamp_ratio(float(p90_ratio))
                if p10_ratio > p90_ratio:
                    p10_ratio, p90_ratio = p90_ratio, p10_ratio
                p50_ratio = min(max(p50_ratio, p10_ratio), p90_ratio)
                model_prediction_cache[(str(model_key), int(loc_id), target)] = (
                    float(p10_ratio),
                    float(p50_ratio),
                    float(p90_ratio),
                    float(batch_missing_ratio[idx]) if idx < len(batch_missing_ratio) else 0.0,
                )
                seed_recursive_ratio_overrides(
                    batch_override_maps[idx] if idx < len(batch_override_maps) else None,
                    target,
                    float(p50_ratio),
                )


def sum_max_caps(max_caps: Dict[int, int], loc_ids: Iterable[int]) -> int:
    return sum(max_caps.get(loc_id, 0) for loc_id in unique_location_ids(loc_ids))


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

    payload: Dict[str, object] = {
        "hour": target.hour,
        "hourStart": target.isoformat(),
        "expectedCount": expected,
        "expectedPct": pct,
    }
    if FORECAST_OUTPUT_INCLUDE_INTERVAL_FIELDS:
        payload["expectedCountP10"] = expected_p10
        payload["expectedCountP90"] = expected_p90
        payload["expectedPctP10"] = pct_p10
        payload["expectedPctP90"] = pct_p90
        payload["sampleCount"] = samples
    return payload


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
    hour_starts: Optional[List[datetime]] = None,
) -> bool:
    if not SPIKE_AWARE_ENABLED or not day_hours:
        return False

    parsed_starts: List[Optional[datetime]] = []
    if (
        isinstance(hour_starts, list)
        and len(hour_starts) == len(day_hours)
        and all(isinstance(ts, datetime) for ts in hour_starts)
    ):
        parsed_starts = [ts for ts in hour_starts]
    else:
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
    estimate_lookup: Optional[Dict[Tuple[int, datetime], Dict[str, float]]] = None,
    target_vectors: Optional[Dict[datetime, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None,
    loc_index_lookup: Optional[Dict[int, int]] = None,
    loc_index_array_cache: Optional[Dict[Tuple[int, ...], np.ndarray]] = None,
    target_index_lookup: Optional[Dict[datetime, int]] = None,
    target_matrices: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    target_row_indices: Optional[np.ndarray] = None,
    target_rows: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> List[Dict[str, object]]:
    rows = []
    now = ctx.get("now")
    normalized_loc_ids = unique_location_ids(loc_ids)
    loc_idx_arr: Optional[np.ndarray] = None
    if isinstance(loc_index_lookup, dict) and (
        isinstance(target_vectors, dict) or target_matrices is not None
    ):
        loc_key = tuple(normalized_loc_ids)
        if isinstance(loc_index_array_cache, dict):
            cached_idx_arr = loc_index_array_cache.get(loc_key)
            if isinstance(cached_idx_arr, np.ndarray):
                loc_idx_arr = cached_idx_arr
        if loc_idx_arr is None:
            loc_indices: List[int] = []
            for loc_id in normalized_loc_ids:
                idx = loc_index_lookup.get(int(loc_id))
                if idx is None:
                    continue
                loc_indices.append(int(idx))
            loc_idx_arr = np.array(loc_indices, dtype=np.int32)
            if isinstance(loc_index_array_cache, dict):
                loc_index_array_cache[loc_key] = loc_idx_arr

    if (
        target_matrices is not None
        and isinstance(target_index_lookup, dict)
        and loc_idx_arr is not None
        and targets
    ):
        p10_mat, p50_mat, p90_mat, sample_mat = target_matrices
        p10_rows: Optional[np.ndarray] = None
        p50_rows: Optional[np.ndarray] = None
        p90_rows: Optional[np.ndarray] = None
        sample_rows: Optional[np.ndarray] = None
        valid = True
        if (
            isinstance(target_rows, tuple)
            and len(target_rows) == 4
            and all(isinstance(arr, np.ndarray) for arr in target_rows)
            and all(arr.shape[0] == len(targets) for arr in target_rows)
        ):
            p10_rows, p50_rows, p90_rows, sample_rows = target_rows
        else:
            row_idx: Optional[np.ndarray] = None
            if (
                isinstance(target_row_indices, np.ndarray)
                and target_row_indices.ndim == 1
                and target_row_indices.size == len(targets)
            ):
                row_idx = np.asarray(target_row_indices, dtype=np.int32)
            else:
                target_indices: List[int] = []
                for target in targets:
                    idx = target_index_lookup.get(target)
                    if idx is None:
                        valid = False
                        break
                    target_indices.append(int(idx))
                if valid and target_indices:
                    row_idx = np.array(target_indices, dtype=np.int32)
                elif valid and not target_indices:
                    row_idx = np.array([], dtype=np.int32)
            if valid and row_idx is not None and row_idx.size > 0:
                if (
                    np.any(row_idx < 0)
                    or np.any(row_idx >= int(p10_mat.shape[0]))
                ):
                    valid = False
            if valid and row_idx is not None:
                p10_rows = p10_mat[row_idx]
                p50_rows = p50_mat[row_idx]
                p90_rows = p90_mat[row_idx]
                sample_rows = sample_mat[row_idx]
        if (
            valid
            and p10_rows is not None
            and p50_rows is not None
            and p90_rows is not None
            and sample_rows is not None
        ):
            p10_arr = np.sum(p10_rows[:, loc_idx_arr], axis=1, dtype=np.float64)
            p50_arr = np.sum(p50_rows[:, loc_idx_arr], axis=1, dtype=np.float64)
            p90_arr = np.sum(p90_rows[:, loc_idx_arr], axis=1, dtype=np.float64)
            sample_arr = np.sum(sample_rows[:, loc_idx_arr], axis=1, dtype=np.int64)
            for idx, target in enumerate(targets):
                sum_p10 = float(p10_arr[idx])
                sum_p50 = float(p50_arr[idx])
                sum_p90 = float(p90_arr[idx])
                samples = int(sample_arr[idx])
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

    for target in targets:
        target_key = target
        sum_p10 = 0.0
        sum_p50 = 0.0
        sum_p90 = 0.0
        samples = 0

        vectors = target_vectors.get(target_key) if isinstance(target_vectors, dict) else None
        if vectors is not None and loc_idx_arr is not None:
            p10_vec, p50_vec, p90_vec, sample_vec = vectors
            sum_p10 = float(np.sum(p10_vec[loc_idx_arr], dtype=np.float64))
            sum_p50 = float(np.sum(p50_vec[loc_idx_arr], dtype=np.float64))
            sum_p90 = float(np.sum(p90_vec[loc_idx_arr], dtype=np.float64))
            samples = int(np.rint(np.sum(sample_vec[loc_idx_arr], dtype=np.float64)))
        else:
            for loc_id in normalized_loc_ids:
                result = None
                if isinstance(estimate_lookup, dict):
                    cached_result = estimate_lookup.get((int(loc_id), target_key))
                    if isinstance(cached_result, dict):
                        result = cached_result
                if result is None:
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
    estimate_lookup: Optional[Dict[Tuple[int, datetime], Dict[str, float]]] = None,
    target_vectors: Optional[Dict[datetime, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None,
    loc_index_lookup: Optional[Dict[int, int]] = None,
    loc_index_array_cache: Optional[Dict[Tuple[int, ...], np.ndarray]] = None,
    target_index_lookup: Optional[Dict[datetime, int]] = None,
    target_matrices: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    target_row_indices: Optional[np.ndarray] = None,
    target_rows: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> List[Dict[str, object]]:
    outputs = []
    normalized_loc_ids = unique_location_ids(loc_ids)
    loc_idx_arr: Optional[np.ndarray] = None
    if isinstance(loc_index_lookup, dict) and (
        isinstance(target_vectors, dict) or target_matrices is not None
    ):
        loc_key = tuple(normalized_loc_ids)
        if isinstance(loc_index_array_cache, dict):
            cached_idx_arr = loc_index_array_cache.get(loc_key)
            if isinstance(cached_idx_arr, np.ndarray):
                loc_idx_arr = cached_idx_arr
        if loc_idx_arr is None:
            loc_indices: List[int] = []
            for loc_id in normalized_loc_ids:
                idx = loc_index_lookup.get(int(loc_id))
                if idx is None:
                    continue
                loc_indices.append(int(idx))
            loc_idx_arr = np.array(loc_indices, dtype=np.int32)
            if isinstance(loc_index_array_cache, dict):
                loc_index_array_cache[loc_key] = loc_idx_arr

    if (
        target_matrices is not None
        and isinstance(target_index_lookup, dict)
        and loc_idx_arr is not None
        and targets
    ):
        p10_mat, p50_mat, p90_mat, sample_mat = target_matrices
        p10_rows: Optional[np.ndarray] = None
        p50_rows: Optional[np.ndarray] = None
        p90_rows: Optional[np.ndarray] = None
        sample_rows: Optional[np.ndarray] = None
        valid = True
        if (
            isinstance(target_rows, tuple)
            and len(target_rows) == 4
            and all(isinstance(arr, np.ndarray) for arr in target_rows)
            and all(arr.shape[0] == len(targets) for arr in target_rows)
        ):
            p10_rows, p50_rows, p90_rows, sample_rows = target_rows
        else:
            row_idx: Optional[np.ndarray] = None
            if (
                isinstance(target_row_indices, np.ndarray)
                and target_row_indices.ndim == 1
                and target_row_indices.size == len(targets)
            ):
                row_idx = np.asarray(target_row_indices, dtype=np.int32)
            else:
                target_indices: List[int] = []
                for target in targets:
                    idx = target_index_lookup.get(target)
                    if idx is None:
                        valid = False
                        break
                    target_indices.append(int(idx))
                if valid and target_indices:
                    row_idx = np.array(target_indices, dtype=np.int32)
                elif valid and not target_indices:
                    row_idx = np.array([], dtype=np.int32)
            if valid and row_idx is not None and row_idx.size > 0:
                if (
                    np.any(row_idx < 0)
                    or np.any(row_idx >= int(p10_mat.shape[0]))
                ):
                    valid = False
            if valid and row_idx is not None:
                p10_rows = p10_mat[row_idx]
                p50_rows = p50_mat[row_idx]
                p90_rows = p90_mat[row_idx]
                sample_rows = sample_mat[row_idx]
        if (
            valid
            and p10_rows is not None
            and p50_rows is not None
            and p90_rows is not None
            and sample_rows is not None
        ):
            p10_arr = np.sum(p10_rows[:, loc_idx_arr], axis=1, dtype=np.float64)
            p50_arr = np.sum(p50_rows[:, loc_idx_arr], axis=1, dtype=np.float64)
            p90_arr = np.sum(p90_rows[:, loc_idx_arr], axis=1, dtype=np.float64)
            sample_arr = np.sum(sample_rows[:, loc_idx_arr], axis=1, dtype=np.int64)
            for idx, target in enumerate(targets):
                outputs.append(
                    to_hour_payload(
                        target=target,
                        sum_p10=float(p10_arr[idx]),
                        sum_p50=float(p50_arr[idx]),
                        sum_p90=float(p90_arr[idx]),
                        category_max=category_max,
                        samples=int(sample_arr[idx]),
                    )
                )
            return outputs

    for target in targets:
        target_key = target
        sum_p10 = 0.0
        sum_p50 = 0.0
        sum_p90 = 0.0
        samples = 0

        vectors = target_vectors.get(target_key) if isinstance(target_vectors, dict) else None
        if vectors is not None and loc_idx_arr is not None:
            p10_vec, p50_vec, p90_vec, sample_vec = vectors
            sum_p10 = float(np.sum(p10_vec[loc_idx_arr], dtype=np.float64))
            sum_p50 = float(np.sum(p50_vec[loc_idx_arr], dtype=np.float64))
            sum_p90 = float(np.sum(p90_vec[loc_idx_arr], dtype=np.float64))
            samples = int(np.rint(np.sum(sample_vec[loc_idx_arr], dtype=np.float64)))
        else:
            for loc_id in normalized_loc_ids:
                result = None
                if isinstance(estimate_lookup, dict):
                    cached_result = estimate_lookup.get((int(loc_id), target_key))
                    if isinstance(cached_result, dict):
                        result = cached_result
                if result is None:
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


def unique_targets_by_iso(targets: Iterable[datetime]) -> List[datetime]:
    seen: Set[datetime] = set()
    ordered: List[datetime] = []
    for target in targets:
        if not isinstance(target, datetime):
            continue
        if target in seen:
            continue
        seen.add(target)
        ordered.append(target)
    return ordered


def precompute_target_estimate_matrices_for_locations(
    loc_ids: Iterable[int],
    targets: Iterable[datetime],
    ctx: Dict[str, object],
) -> Tuple[
    List[int],
    List[datetime],
    Dict[datetime, int],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    normalized_loc_ids = unique_location_ids(loc_ids)
    normalized_targets = sorted(unique_targets_by_iso(targets))
    target_index_lookup = {
        target: int(idx) for idx, target in enumerate(normalized_targets)
    }
    if not normalized_loc_ids or not normalized_targets:
        empty_f = np.zeros((0, 0), dtype=np.float32)
        empty_i = np.zeros((0, 0), dtype=np.int32)
        return normalized_loc_ids, normalized_targets, target_index_lookup, (
            empty_f,
            empty_f,
            empty_f,
            empty_i,
        )

    prime_model_prediction_cache_for_targets(
        loc_ids=normalized_loc_ids,
        targets=normalized_targets,
        ctx=ctx,
    )

    height = len(normalized_targets)
    width = len(normalized_loc_ids)
    p10 = np.zeros((height, width), dtype=np.float32)
    p50 = np.zeros((height, width), dtype=np.float32)
    p90 = np.zeros((height, width), dtype=np.float32)
    samples = np.zeros((height, width), dtype=np.int32)

    for t_idx, target in enumerate(normalized_targets):
        for l_idx, loc_id in enumerate(normalized_loc_ids):
            result = estimate_location(
                int(loc_id),
                target,
                ctx,
            )
            if not isinstance(result, dict):
                continue
            p10[t_idx, l_idx] = float(to_float_or_none(result.get("countP10")) or 0.0)
            p50[t_idx, l_idx] = float(to_float_or_none(result.get("countP50")) or 0.0)
            p90[t_idx, l_idx] = float(to_float_or_none(result.get("countP90")) or 0.0)
            samples[t_idx, l_idx] = int(to_float_or_none(result.get("sampleCount")) or 0.0)

    return normalized_loc_ids, normalized_targets, target_index_lookup, (
        p10,
        p50,
        p90,
        samples,
    )


def normalized_forecast_hour_bounds() -> Tuple[int, int]:
    start_hour = max(0, min(23, FORECAST_DAY_START_HOUR))
    end_hour = max(0, min(23, FORECAST_DAY_END_HOUR))
    if end_hour < start_hour:
        start_hour, end_hour = end_hour, start_hour
    return start_hour, end_hour


def get_targets_for_date(day_date: date) -> List[datetime]:
    start_hour, end_hour = normalized_forecast_hour_bounds()

    start_dt = TZ.localize(datetime(day_date.year, day_date.month, day_date.day, start_hour, 0, 0))
    end_exclusive = TZ.localize(
        datetime(day_date.year, day_date.month, day_date.day, end_hour, 0, 0)
    ) + timedelta(hours=1)

    step = timedelta(minutes=max(1, RESAMPLE_MINUTES))
    targets = []
    current = start_dt
    while current < end_exclusive:
        targets.append(current)
        current += step
    return targets


def get_window_targets_for_date(day_date: date) -> List[datetime]:
    start_hour, end_hour = normalized_forecast_hour_bounds()

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


def filter_targets_by_schedule(
    facility_id: int,
    targets: List[datetime],
    facility_schedule_by_id: Dict[int, Dict[str, object]],
    schedule_eval_cache: Dict[Tuple[int, date, int], Optional[bool]],
    schedule_boundary_cache: Dict[Tuple[int, date, int], Tuple[bool, bool]],
    date_range_cache: Dict[Tuple[str, int], Optional[Tuple[date, date, int]]],
    weekday_cache: Dict[str, Optional[Set[int]]],
    hours_window_cache: Dict[str, Optional[Tuple[int, int, bool]]],
) -> Tuple[List[datetime], List[bool]]:
    if not targets:
        return [], []

    if not facility_schedule_by_id:
        return list(targets), [False] * len(targets)

    facility_schedule = facility_schedule_by_id.get(int(facility_id), {})
    sections_raw = facility_schedule.get("sections", []) if isinstance(facility_schedule, dict) else []
    sections = sections_raw if isinstance(sections_raw, list) else []
    if not sections:
        return list(targets), [False] * len(targets)

    filtered: List[datetime] = []
    boundary_flags: List[bool] = []

    for target in targets:
        minute_of_day = int(target.hour) * 60 + int(target.minute)
        cache_key = (int(facility_id), target.date(), minute_of_day)
        if cache_key not in schedule_eval_cache:
            schedule_eval_cache[cache_key] = get_facility_schedule_open_state(
                sections=sections,
                ts=target,
                date_range_cache=date_range_cache,
                weekday_cache=weekday_cache,
                hours_window_cache=hours_window_cache,
            )

        open_state = schedule_eval_cache.get(cache_key)
        open_exact = False
        close_exact = False
        if SCHEDULE_BOUNDARY_ZERO_ENABLED:
            if cache_key not in schedule_boundary_cache:
                schedule_boundary_cache[cache_key] = get_facility_schedule_boundary_state(
                    sections=sections,
                    ts=target,
                    date_range_cache=date_range_cache,
                    weekday_cache=weekday_cache,
                    hours_window_cache=hours_window_cache,
                )
            open_exact, close_exact = schedule_boundary_cache.get(cache_key, (False, False))

        if open_state is True:
            filtered.append(target)
            boundary_flags.append(bool(open_exact or close_exact))
            continue

        # Keep the exact close boundary so downstream output can be hard-zeroed.
        if SCHEDULE_BOUNDARY_ZERO_ENABLED:
            if close_exact:
                filtered.append(target)
                boundary_flags.append(True)

    # Unknown/closed rows are excluded except the exact close boundary row.
    return filtered, boundary_flags


def apply_schedule_boundary_zero_to_series(
    facility_id: int,
    series: List[Dict[str, object]],
    facility_schedule_by_id: Dict[int, Dict[str, object]],
    schedule_boundary_cache: Dict[Tuple[int, date, int], Tuple[bool, bool]],
    date_range_cache: Dict[Tuple[str, int], Optional[Tuple[date, date, int]]],
    weekday_cache: Dict[str, Optional[Set[int]]],
    hours_window_cache: Dict[str, Optional[Tuple[int, int, bool]]],
    boundary_flags: Optional[List[bool]] = None,
    hour_starts: Optional[List[datetime]] = None,
) -> int:
    if not SCHEDULE_BOUNDARY_ZERO_ENABLED:
        return 0
    if not series:
        return 0

    use_boundary_flags = isinstance(boundary_flags, list) and len(boundary_flags) >= len(series)
    sections: List[object] = []
    if not use_boundary_flags:
        if not facility_schedule_by_id:
            return 0
        facility_schedule = facility_schedule_by_id.get(int(facility_id), {})
        sections_raw = facility_schedule.get("sections", []) if isinstance(facility_schedule, dict) else []
        sections = sections_raw if isinstance(sections_raw, list) else []
        if not sections:
            return 0

    adjusted = 0
    for idx, row in enumerate(series):
        if not isinstance(row, dict):
            continue
        is_boundary = False
        if use_boundary_flags:
            is_boundary = bool(boundary_flags[idx]) if idx < len(boundary_flags) else False
        else:
            hour_start = None
            if (
                isinstance(hour_starts, list)
                and idx < len(hour_starts)
                and isinstance(hour_starts[idx], datetime)
            ):
                hour_start = hour_starts[idx]
            else:
                hour_start = safe_parse_hour_start(row.get("hourStart"))
            if hour_start is None:
                continue

            minute_of_day = int(hour_start.hour) * 60 + int(hour_start.minute)
            cache_key = (int(facility_id), hour_start.date(), minute_of_day)
            if cache_key not in schedule_boundary_cache:
                schedule_boundary_cache[cache_key] = get_facility_schedule_boundary_state(
                    sections=sections,
                    ts=hour_start,
                    date_range_cache=date_range_cache,
                    weekday_cache=weekday_cache,
                    hours_window_cache=hours_window_cache,
                )
            open_exact, close_exact = schedule_boundary_cache.get(cache_key, (False, False))
            is_boundary = bool(open_exact or close_exact)
        if not is_boundary:
            continue

        changed = False
        if "expectedCount" in row:
            row["expectedCount"] = 0
            changed = True
            if "expectedCountP10" in row:
                row["expectedCountP10"] = 0
            if "expectedCountP90" in row:
                row["expectedCountP90"] = 0
            if "expectedPct" in row:
                row["expectedPct"] = 0.0
            if "expectedPctP10" in row:
                row["expectedPctP10"] = 0.0
            if "expectedPctP90" in row:
                row["expectedPctP90"] = 0.0

        if "expectedTotal" in row:
            row["expectedTotal"] = 0
            changed = True
            if "expectedTotalP10" in row:
                row["expectedTotalP10"] = 0
            if "expectedTotalP50" in row:
                row["expectedTotalP50"] = 0
            if "expectedTotalP90" in row:
                row["expectedTotalP90"] = 0

        if changed:
            row["scheduleBoundaryZeroed"] = True
            adjusted += 1

    return int(adjusted)


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


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def build_location_bucket_total_cache(
    loc_ids: Iterable[int],
    loc_data: Dict[int, Dict[str, object]],
    max_caps: Dict[int, int],
) -> Dict[int, Dict[datetime, float]]:
    bucket_totals_by_location: Dict[int, Dict[datetime, float]] = {}
    for loc_id in unique_location_ids(loc_ids):
        cap = int(max_caps.get(loc_id, 0) or 0)
        if cap <= 0:
            continue

        data = loc_data.get(loc_id) or {}
        bucket_map = data.get("bucket_map") or {}
        if not isinstance(bucket_map, dict):
            continue

        location_bucket_totals: Dict[datetime, float] = {}
        for bucket_ts, ratio in bucket_map.items():
            if not isinstance(bucket_ts, datetime):
                continue
            try:
                ratio_f = float(ratio)
            except Exception:
                continue
            location_bucket_totals[bucket_ts] = max(0.0, min(ratio_f, 1.2)) * cap

        if location_bucket_totals:
            bucket_totals_by_location[int(loc_id)] = location_bucket_totals

    return bucket_totals_by_location


def build_facility_crowd_baseline(
    facility_loc_ids: Iterable[int],
    loc_data: Dict[int, Dict[str, object]],
    max_caps: Dict[int, int],
    location_bucket_totals: Optional[Dict[int, Dict[datetime, float]]] = None,
) -> Optional[Dict[str, float]]:
    facility_loc_ids = list(dict.fromkeys(int(loc_id) for loc_id in facility_loc_ids))
    if not facility_loc_ids:
        return None

    facility_max_cap = sum_max_caps(max_caps, facility_loc_ids)
    if facility_max_cap <= 0:
        return None

    totals_by_bucket: Dict[datetime, float] = {}
    observed_cap_by_bucket: Dict[datetime, int] = {}
    bucket_totals_source = (
        location_bucket_totals
        if isinstance(location_bucket_totals, dict)
        else build_location_bucket_total_cache(
            facility_loc_ids,
            loc_data=loc_data,
            max_caps=max_caps,
        )
    )

    for loc_id in facility_loc_ids:
        cap = int(max_caps.get(loc_id, 0) or 0)
        if cap <= 0:
            continue

        bucket_map = bucket_totals_source.get(int(loc_id)) or {}
        if not bucket_map:
            continue

        for bucket_ts, total in bucket_map.items():
            if not isinstance(bucket_ts, datetime):
                continue
            totals_by_bucket[bucket_ts] = totals_by_bucket.get(bucket_ts, 0.0) + float(total)
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
    required_points = max(1, CROWD_BASELINE_MIN_POINTS)
    if len(baseline_values) < required_points:
        return None
    if not baseline_values:
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
    }


def occupancy_thresholds_from_baseline(
    baseline: Optional[Dict[str, float]],
    max_capacity: int,
) -> Optional[Dict[str, float]]:
    if not isinstance(baseline, dict) or max_capacity <= 0:
        return None

    try:
        low_ceiling = float(baseline.get("lowCeiling"))
        peak_floor = float(baseline.get("peakFloor"))
    except (TypeError, ValueError):
        return None

    if not math.isfinite(low_ceiling) or not math.isfinite(peak_floor):
        return None

    low_max = int(round(max(0.0, min(99.0, (low_ceiling / float(max_capacity)) * 100.0))))
    peak_min = int(round(max(float(low_max + 1), min(100.0, (peak_floor / float(max_capacity)) * 100.0))))

    payload: Dict[str, float] = {
        "lowMax": float(low_max),
        "peakMin": float(peak_min),
    }

    return payload


def build_crowd_bands_from_labels(
    labels: List[Tuple[datetime, str]],
    step: timedelta,
) -> List[Dict[str, object]]:
    if not labels:
        return []

    bridge_gap = timedelta(minutes=CROWD_BAND_BRIDGE_MIN)
    if bridge_gap > timedelta(0) and len(labels) >= 3:
        smoothed = list(labels)
        runs: List[Tuple[int, int, str]] = []
        run_start = 0
        run_label = smoothed[0][1]
        prev_ts = smoothed[0][0]

        for idx in range(1, len(smoothed)):
            ts, label = smoothed[idx]
            contiguous = (ts - prev_ts) <= step
            prev_ts = ts
            if label == run_label and contiguous:
                continue
            runs.append((run_start, idx - 1, run_label))
            run_start = idx
            run_label = label
        runs.append((run_start, len(smoothed) - 1, run_label))

        for run_idx in range(1, len(runs) - 1):
            start_idx, end_idx, _label = runs[run_idx]
            prev_label = runs[run_idx - 1][2]
            next_label = runs[run_idx + 1][2]
            if prev_label != next_label:
                continue

            run_duration = (smoothed[end_idx][0] + step) - smoothed[start_idx][0]
            if run_duration > bridge_gap:
                continue
            if (smoothed[start_idx][0] - smoothed[start_idx - 1][0]) > step:
                continue
            if (smoothed[end_idx + 1][0] - smoothed[end_idx][0]) > step:
                continue

            for idx in range(start_idx, end_idx + 1):
                ts, _ = smoothed[idx]
                smoothed[idx] = (ts, prev_label)

        labels = smoothed

    bands: List[Dict[str, object]] = []
    cur_start, cur_label = labels[0]
    prev_ts = labels[0][0]

    for ts, label in labels[1:]:
        contiguous = (ts - prev_ts) <= step
        if label != cur_label or not contiguous:
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


def build_threshold_crowd_bands(
    series: List[Dict[str, object]],
    max_capacity: int,
    occupancy_thresholds: Optional[Dict[str, float]],
) -> List[Dict[str, object]]:
    if not series or max_capacity <= 0 or not isinstance(occupancy_thresholds, dict):
        return []

    entries = normalize_series(series)
    if not entries:
        return []

    low_max = to_float_or_none(occupancy_thresholds.get("lowMax"))
    peak_min = to_float_or_none(occupancy_thresholds.get("peakMin"))
    if low_max is None and peak_min is None:
        return []
    if low_max is None:
        low_max = peak_min
    if peak_min is None:
        peak_min = low_max
    if low_max is None or peak_min is None:
        return []
    low_max = max(0.0, min(99.0, float(low_max)))
    peak_min = max(low_max + 1.0, min(100.0, float(peak_min)))

    step_minutes = infer_series_step_minutes(series)
    step = timedelta(minutes=step_minutes)
    labels: List[Tuple[datetime, str]] = []
    for ts, expected, _samples in entries:
        percent = max(0.0, min(100.0, (float(expected) / float(max_capacity)) * 100.0))
        label = "medium"
        if percent >= peak_min:
            label = "peak"
        elif percent <= low_max:
            label = "low"
        labels.append((ts, label))

    return build_crowd_bands_from_labels(labels, step)

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
    forecast_start_hour, forecast_end_hour = normalized_forecast_hour_bounds()
    week_dates = [(now + timedelta(days=offset)).date() for offset in range(7)]
    day_targets_by_date: Dict[date, List[datetime]] = {
        day_date: get_targets_for_date(day_date) for day_date in week_dates
    }
    window_targets_by_date: Dict[date, List[datetime]] = {
        day_date: get_window_targets_for_date(day_date) for day_date in week_dates
    }
    saved_meta_snapshot = collect_saved_meta_snapshots()
    adaptive_controls = derive_adaptive_runtime_controls(saved_meta_snapshot)
    facility_schedule_by_id = load_schedule_sections_by_facility()

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
        ) = load_history(
            conn,
            facility_schedule_by_id=facility_schedule_by_id,
        )
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
    quality["scheduleHoursPath"] = FACILITY_HOURS_JSON_PATH
    quality["scheduleHoursAvailable"] = bool(facility_schedule_by_id)
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
    feature_fill_values_by_key: Dict[str, np.ndarray] = {}
    feature_clip_bounds_by_key: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for model_key, meta in model_meta_by_key.items():
        interval_profile_by_key[model_key] = meta.get("residualProfile") if meta else None
        expected_feature_count = (
            int(meta.get("featureCount", -1) or -1)
            if isinstance(meta, dict)
            else -1
        )
        fills = coerce_feature_fill_values(
            meta.get("featureFillValues") if isinstance(meta, dict) else None,
            expected_cols=expected_feature_count if expected_feature_count > 0 else None,
        )
        if fills is not None:
            feature_fill_values_by_key[model_key] = fills
        clip_bounds = coerce_feature_clip_bounds(
            meta.get("featureClipLower") if isinstance(meta, dict) else None,
            meta.get("featureClipUpper") if isinstance(meta, dict) else None,
            expected_cols=expected_feature_count if expected_feature_count > 0 else None,
        )
        if clip_bounds is not None:
            feature_clip_bounds_by_key[model_key] = clip_bounds

    model_summary = summarize_model_results(
        model_meta_by_key=model_meta_by_key,
        model_status_by_key=model_status_by_key,
        run_metrics_by_key=run_metrics_by_key,
    )
    recent_model_dataset_cache: Dict[Tuple[object, ...], Dict[str, object]] = {}
    recent_core_feature_cache: Dict[Tuple[int, datetime], List[float]] = {}
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
        recent_dataset_cache=recent_model_dataset_cache,
        core_feature_cache=recent_core_feature_cache,
    )
    recent_drift_bias_by_key: Dict[str, Dict[str, object]] = {}
    for model_key, row in (drift_summary.get("byModel") or {}).items():
        if not isinstance(row, dict):
            continue
        profile = row.get("recentBiasProfile")
        if isinstance(profile, dict):
            recent_drift_bias_by_key[str(model_key)] = profile
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
        recent_drift_bias_by_key.pop(str(model_key), None)
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
            restored_fills = coerce_feature_fill_values(
                restored_meta.get("featureFillValues"),
                expected_cols=feature_count,
            )
            if restored_fills is not None:
                feature_fill_values_by_key[model_key] = restored_fills
            else:
                feature_fill_values_by_key.pop(model_key, None)
            restored_clip = coerce_feature_clip_bounds(
                restored_meta.get("featureClipLower"),
                restored_meta.get("featureClipUpper"),
                expected_cols=feature_count,
            )
            if restored_clip is not None:
                feature_clip_bounds_by_key[model_key] = restored_clip
            else:
                feature_clip_bounds_by_key.pop(model_key, None)
    conformal_by_key, conformal_summary = compute_interval_conformal_profiles(
        now=now,
        models_by_key=models_by_key,
        onehot_by_key=onehot_by_key,
        unit_loc_ids=unit_loc_ids,
        loc_data=loc_data,
        loc_samples=loc_samples,
        weather_series=weather_series,
        interval_profile_by_key=interval_profile_by_key,
        model_meta_by_key=model_meta_by_key,
        recent_dataset_cache=recent_model_dataset_cache,
        core_feature_cache=recent_core_feature_cache,
    )
    location_facility_map = location_to_facility_map()
    schedule_eval_cache: Dict[Tuple[int, date, int], Optional[bool]] = {}
    schedule_boundary_cache: Dict[Tuple[int, date, int], Tuple[bool, bool]] = {}
    schedule_date_range_cache: Dict[Tuple[str, int], Optional[Tuple[date, date, int]]] = {}
    schedule_weekday_cache: Dict[str, Optional[Set[int]]] = {}
    schedule_hours_cache: Dict[str, Optional[Tuple[int, int, bool]]] = {}
    ctx = {
        "now": now,
        "models_by_key": models_by_key,
        "model_meta_by_key": model_meta_by_key,
        "interval_profile_by_key": interval_profile_by_key,
        "conformal_by_key": conformal_by_key,
        "interval_multiplier_by_key": interval_multiplier_by_key,
        "recent_drift_bias_by_key": recent_drift_bias_by_key,
        "weather_series": weather_series,
        "weather_lookup_cache": {},
        "feature_cache": {},
        "feature_matrix_cache": {},
        "model_prediction_cache": {},
        "prediction_cache": {},
        "location_estimate_cache": {},
        "ensemble_weight_cache": {},
        "point_bias_value_cache": {},
        "recent_drift_bias_value_cache": {},
        "conformal_margin_cache": {},
        "recursive_ratio_cache": {},
        "loc_data": loc_data,
        "onehot_by_key": onehot_by_key,
        "feature_fill_values_by_key": feature_fill_values_by_key,
        "feature_clip_bounds_by_key": feature_clip_bounds_by_key,
        "loc_to_model_key": loc_to_model_key,
        "loc_to_fallback_key": loc_to_fallback_key,
        "avg_dow_hour": avg_dow_hour,
        "avg_hour": avg_hour,
        "avg_overall": avg_overall,
        "loc_samples": loc_samples,
        "max_caps": max_caps,
        "facility_schedule_by_id": facility_schedule_by_id,
        "location_facility_map": location_facility_map,
        "schedule_eval_cache": schedule_eval_cache,
        "schedule_boundary_cache": schedule_boundary_cache,
        "schedule_date_range_cache": schedule_date_range_cache,
        "schedule_weekday_cache": schedule_weekday_cache,
        "schedule_hours_cache": schedule_hours_cache,
    }
    location_bucket_totals = build_location_bucket_total_cache(
        all_location_ids(),
        loc_data=loc_data,
        max_caps=max_caps,
    )

    facilities_payload = []
    spike_adjusted_categories = 0
    schedule_boundary_forecast_rows_zeroed = 0

    for facility_id, facility in FACILITIES.items():
        weekly_forecast = []
        facility_loc_ids = facility_location_ids(facility)
        crowd_baseline_cache: Dict[Tuple[int, ...], Optional[Dict[str, float]]] = {}

        def baseline_for_locations(loc_ids: Iterable[int]) -> Optional[Dict[str, float]]:
            cache_key = tuple(unique_location_ids(loc_ids))
            if not cache_key:
                return None
            if cache_key not in crowd_baseline_cache:
                crowd_baseline_cache[cache_key] = build_facility_crowd_baseline(
                    facility_loc_ids=cache_key,
                    loc_data=loc_data,
                    max_caps=max_caps,
                    location_bucket_totals=location_bucket_totals,
                )
            return crowd_baseline_cache[cache_key]

        category_definitions: List[Dict[str, object]] = []
        for raw_category in facility["categories"]:
            category_key = str(raw_category.get("key") or "")
            category_title = str(raw_category.get("title") or category_key)
            category_loc_ids = unique_location_ids(raw_category.get("location_ids", []))
            category_definitions.append(
                {
                    "key": category_key,
                    "title": category_title,
                    "location_ids": category_loc_ids,
                    "maxCapacity": sum_max_caps(max_caps, category_loc_ids),
                    "thresholdKey": normalize_section_key(category_title or category_key),
                    "forecastEnabled": category_key in FORECAST_CATEGORY_KEYS,
                }
            )

        facility_max_capacity = sum_max_caps(max_caps, facility_loc_ids)
        facility_crowd_baseline = baseline_for_locations(facility_loc_ids)
        facility_occupancy_thresholds = occupancy_thresholds_from_baseline(
            facility_crowd_baseline,
            facility_max_capacity,
        )
        section_occupancy_thresholds: Dict[str, Dict[str, float]] = {}
        location_occupancy_thresholds: Dict[str, Dict[str, float]] = {}
        for category in category_definitions:
            category_loc_ids = category.get("location_ids", [])
            if not category_loc_ids:
                continue

            category_max_capacity = int(category.get("maxCapacity", 0) or 0)
            category_baseline = baseline_for_locations(category_loc_ids)
            category_thresholds = occupancy_thresholds_from_baseline(
                category_baseline,
                category_max_capacity,
            )
            category_key = str(category.get("thresholdKey") or "")
            if category_key and category_thresholds:
                section_occupancy_thresholds[category_key] = category_thresholds
        for loc_id in facility_loc_ids:
            location_max_capacity = int(max_caps.get(loc_id, 0) or 0)
            if location_max_capacity <= 0:
                continue
            location_baseline = baseline_for_locations([loc_id])
            location_thresholds = occupancy_thresholds_from_baseline(
                location_baseline,
                location_max_capacity,
            )
            if location_thresholds:
                location_occupancy_thresholds[str(int(loc_id))] = location_thresholds

        forecast_categories: List[Dict[str, object]] = [
            category
            for category in category_definitions
            if bool(category.get("forecastEnabled"))
        ]
        category_live_cache: Dict[str, Tuple[int, int]] = {}
        for category in forecast_categories:
            category_key = str(category.get("key"))
            category_loc_ids = category.get("location_ids", [])
            live_total, observed_locs = category_live_total(
                category_loc_ids if isinstance(category_loc_ids, list) else [],
                loc_data,
                now,
            )
            category_live_cache[category_key] = (int(live_total), int(observed_locs))

        facility_targets_by_day: Dict[date, List[datetime]] = {}
        facility_window_targets_by_day: Dict[date, List[datetime]] = {}
        facility_targets_boundary_flags_by_day: Dict[date, List[bool]] = {}
        facility_window_targets_boundary_flags_by_day: Dict[date, List[bool]] = {}
        facility_targets_has_boundary_by_day: Dict[date, bool] = {}
        facility_window_targets_has_boundary_by_day: Dict[date, bool] = {}
        facility_all_targets: List[datetime] = []
        for day_date in week_dates:
            day_targets, day_target_boundary_flags = filter_targets_by_schedule(
                facility_id=facility_id,
                targets=day_targets_by_date.get(day_date, []),
                facility_schedule_by_id=facility_schedule_by_id,
                schedule_eval_cache=schedule_eval_cache,
                schedule_boundary_cache=schedule_boundary_cache,
                date_range_cache=schedule_date_range_cache,
                weekday_cache=schedule_weekday_cache,
                hours_window_cache=schedule_hours_cache,
            )
            day_window_targets, day_window_target_boundary_flags = filter_targets_by_schedule(
                facility_id=facility_id,
                targets=window_targets_by_date.get(day_date, []),
                facility_schedule_by_id=facility_schedule_by_id,
                schedule_eval_cache=schedule_eval_cache,
                schedule_boundary_cache=schedule_boundary_cache,
                date_range_cache=schedule_date_range_cache,
                weekday_cache=schedule_weekday_cache,
                hours_window_cache=schedule_hours_cache,
            )
            facility_targets_by_day[day_date] = day_targets
            facility_window_targets_by_day[day_date] = day_window_targets
            facility_targets_boundary_flags_by_day[day_date] = day_target_boundary_flags
            facility_window_targets_boundary_flags_by_day[day_date] = day_window_target_boundary_flags
            facility_targets_has_boundary_by_day[day_date] = bool(any(day_target_boundary_flags))
            facility_window_targets_has_boundary_by_day[day_date] = bool(any(day_window_target_boundary_flags))
            facility_all_targets.extend(day_targets)
            facility_all_targets.extend(day_window_targets)

        facility_combined_targets = unique_targets_by_iso(facility_all_targets)
        (
            vector_loc_ids,
            _facility_target_order,
            facility_target_index_lookup,
            facility_target_matrices,
        ) = precompute_target_estimate_matrices_for_locations(
            loc_ids=facility_loc_ids,
            targets=facility_combined_targets,
            ctx=ctx,
        )
        facility_loc_index_lookup = {
            int(loc_id): int(idx) for idx, loc_id in enumerate(vector_loc_ids)
        }
        facility_loc_index_array_cache: Dict[Tuple[int, ...], np.ndarray] = {}

        for day_date in week_dates:
            targets = facility_targets_by_day.get(day_date, [])
            window_targets = facility_window_targets_by_day.get(day_date, [])
            targets_boundary_flags = facility_targets_boundary_flags_by_day.get(day_date, [])
            window_targets_boundary_flags = facility_window_targets_boundary_flags_by_day.get(day_date, [])
            targets_has_boundary = bool(facility_targets_has_boundary_by_day.get(day_date, False))
            window_targets_has_boundary = bool(
                facility_window_targets_has_boundary_by_day.get(day_date, False)
            )
            day_target_row_indices = np.array(
                [int(facility_target_index_lookup.get(target, -1)) for target in targets],
                dtype=np.int32,
            )
            if np.any(day_target_row_indices < 0):
                day_target_row_indices = np.array([], dtype=np.int32)
            window_target_row_indices = np.array(
                [int(facility_target_index_lookup.get(target, -1)) for target in window_targets],
                dtype=np.int32,
            )
            if np.any(window_target_row_indices < 0):
                window_target_row_indices = np.array([], dtype=np.int32)
            day_target_rows: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
            if day_target_row_indices.size == len(targets) and targets:
                p10_mat, p50_mat, p90_mat, sample_mat = facility_target_matrices
                day_target_rows = (
                    p10_mat[day_target_row_indices],
                    p50_mat[day_target_row_indices],
                    p90_mat[day_target_row_indices],
                    sample_mat[day_target_row_indices],
                )
            window_target_rows: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
            if window_target_row_indices.size == len(window_targets) and window_targets:
                p10_mat, p50_mat, p90_mat, sample_mat = facility_target_matrices
                window_target_rows = (
                    p10_mat[window_target_row_indices],
                    p50_mat[window_target_row_indices],
                    p90_mat[window_target_row_indices],
                    sample_mat[window_target_row_indices],
                )
            day_weather = build_weather_hours_for_targets(targets, weather_series) if FORECAST_OUTPUT_INCLUDE_WEATHER else []
            day_weather_summary = build_weather_day_summary(day_weather) if FORECAST_OUTPUT_INCLUDE_WEATHER else {}
            total_day_series = build_total_series_for_targets(
                loc_ids=facility_loc_ids,
                targets=targets,
                ctx=ctx,
                estimate_lookup=None,
                target_vectors=None,
                loc_index_lookup=facility_loc_index_lookup,
                loc_index_array_cache=facility_loc_index_array_cache,
                target_index_lookup=facility_target_index_lookup,
                target_matrices=facility_target_matrices,
                target_row_indices=day_target_row_indices if day_target_row_indices.size == len(targets) else None,
                target_rows=day_target_rows,
            )
            if targets_has_boundary:
                schedule_boundary_forecast_rows_zeroed += apply_schedule_boundary_zero_to_series(
                    facility_id=facility_id,
                    series=total_day_series,
                    facility_schedule_by_id=facility_schedule_by_id,
                    schedule_boundary_cache=schedule_boundary_cache,
                    date_range_cache=schedule_date_range_cache,
                    weekday_cache=schedule_weekday_cache,
                    hours_window_cache=schedule_hours_cache,
                    boundary_flags=targets_boundary_flags,
                    hour_starts=targets,
                )
            total_day_hours = [
                {
                    "hour": row.get("hour"),
                    "hourStart": row.get("hourStart"),
                    "expectedCount": int(row.get("expectedTotal", 0) or 0),
                    "expectedPct": (
                        round(
                            min(float(row.get("expectedTotal", 0) or 0) / float(facility_max_capacity), 1.0),
                            4,
                        )
                        if facility_max_capacity > 0
                        else None
                    ),
                }
                for row in total_day_series
                if isinstance(row, dict)
            ]
            day_categories = []

            for category in forecast_categories:
                category_loc_ids = category["location_ids"]
                category_max = int(category.get("maxCapacity", 0) or 0)
                day_hours = build_category_hours_for_targets(
                    loc_ids=category_loc_ids,
                    targets=targets,
                    category_max=category_max,
                    ctx=ctx,
                    estimate_lookup=None,
                    target_vectors=None,
                    loc_index_lookup=facility_loc_index_lookup,
                    loc_index_array_cache=facility_loc_index_array_cache,
                    target_index_lookup=facility_target_index_lookup,
                    target_matrices=facility_target_matrices,
                    target_row_indices=day_target_row_indices if day_target_row_indices.size == len(targets) else None,
                    target_rows=day_target_rows,
                )

                category_key = str(category.get("key"))
                live_total, observed_locs = category_live_cache.get(category_key, (0, 0))
                if observed_locs > 0:
                    if apply_spike_adjustment_to_category_hours(
                        day_hours=day_hours,
                        category_max=category_max,
                        live_total=live_total,
                        now=now,
                        hour_starts=targets,
                    ):
                        spike_adjusted_categories += 1

                if targets_has_boundary:
                    schedule_boundary_forecast_rows_zeroed += apply_schedule_boundary_zero_to_series(
                        facility_id=facility_id,
                        series=day_hours,
                        facility_schedule_by_id=facility_schedule_by_id,
                        schedule_boundary_cache=schedule_boundary_cache,
                        date_range_cache=schedule_date_range_cache,
                        weekday_cache=schedule_weekday_cache,
                        hours_window_cache=schedule_hours_cache,
                        boundary_flags=targets_boundary_flags,
                        hour_starts=targets,
                    )

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
                estimate_lookup=None,
                target_vectors=None,
                loc_index_lookup=facility_loc_index_lookup,
                loc_index_array_cache=facility_loc_index_array_cache,
                target_index_lookup=facility_target_index_lookup,
                target_matrices=facility_target_matrices,
                target_row_indices=(
                    window_target_row_indices
                    if window_target_row_indices.size == len(window_targets)
                    else None
                ),
                target_rows=window_target_rows,
            )
            if window_targets_has_boundary:
                schedule_boundary_forecast_rows_zeroed += apply_schedule_boundary_zero_to_series(
                    facility_id=facility_id,
                    series=totals_day,
                    facility_schedule_by_id=facility_schedule_by_id,
                    schedule_boundary_cache=schedule_boundary_cache,
                    date_range_cache=schedule_date_range_cache,
                    weekday_cache=schedule_weekday_cache,
                    hours_window_cache=schedule_hours_cache,
                    boundary_flags=window_targets_boundary_flags,
                    hour_starts=window_targets,
                )

            crowd_bands_day = build_threshold_crowd_bands(
                totals_day,
                facility_max_capacity,
                facility_occupancy_thresholds,
            )
            best_windows_day = build_windows_from_bands(crowd_bands_day, "low", totals_day)
            avoid_windows_day = build_windows_from_bands(crowd_bands_day, "peak", totals_day)

            day_payload: Dict[str, object] = {
                "dayName": day_date.strftime("%A"),
                "date": day_date.isoformat(),
                "categories": day_categories,
                "totalHours": total_day_hours,
                "avoidWindows": avoid_windows_day,
                "bestWindows": best_windows_day,
                "crowdBands": crowd_bands_day,
            }
            if FORECAST_OUTPUT_INCLUDE_WEATHER:
                day_payload["weatherHours"] = day_weather
                day_payload["weatherSummary"] = day_weather_summary
            weekly_forecast.append(day_payload)

        facilities_payload.append(
            {
                "facilityId": facility_id,
                "facilityName": facility["name"],
                "occupancyThresholds": facility_occupancy_thresholds,
                "sectionOccupancyThresholds": section_occupancy_thresholds,
                "locationOccupancyThresholds": location_occupancy_thresholds,
                "weeklyForecast": weekly_forecast,
            }
        )
        for cache_name in (
            "feature_cache",
            "feature_matrix_cache",
            "model_prediction_cache",
            "prediction_cache",
            "location_estimate_cache",
            "recursive_ratio_cache",
            "ensemble_weight_cache",
            "point_bias_value_cache",
            "recent_drift_bias_value_cache",
            "conformal_margin_cache",
        ):
            cache_obj = ctx.get(cache_name)
            if isinstance(cache_obj, dict):
                cache_obj.clear()

    quality["locationsModeled"] = sum(
        1
        for loc_id, data in loc_data.items()
        if loc_samples.get(loc_id, 0) >= MIN_SAMPLES_PER_LOC and not data.get("is_stale")
    )
    quality["spikeAwareEnabled"] = SPIKE_AWARE_ENABLED
    quality["spikeAwareCategoriesAdjusted"] = spike_adjusted_categories
    quality["scheduleBoundaryForecastRowsZeroed"] = schedule_boundary_forecast_rows_zeroed
    quality["drift"] = drift_summary
    quality["driftActions"] = drift_actions_summary
    quality["intervalConformal"] = conformal_summary

    val_mae_for_payload = model_summary.get("valMae")
    precision_pct = None
    if val_mae_for_payload is not None:
        val_mae_float = float(val_mae_for_payload)
        if math.isfinite(val_mae_float):
            precision_pct = max(0.0, min(100.0, (1.0 - val_mae_float) * 100.0))

    payload = {
        "generatedAt": now.isoformat(),
        "timezone": TZ_NAME,
        "forecastDayStartHour": forecast_start_hour,
        "forecastDayEndHour": forecast_end_hour,
        "model": "xgboost",
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
            "guardrailMaxHoldoutMaeDegrade": MODEL_GUARDRAIL_MAX_HOLDOUT_MAE_DEGRADE,
            "guardrailMaxHoldoutIntervalErrDegrade": MODEL_GUARDRAIL_MAX_HOLDOUT_INTERVAL_ERR_DEGRADE,
            "guardrailMaxValIntervalErrDegrade": MODEL_GUARDRAIL_MAX_VAL_INTERVAL_ERR_DEGRADE,
            "featureMissingGuardEnabled": MODEL_FEATURE_MISSING_GUARD_ENABLED,
            "featureMissingMinRows": MODEL_FEATURE_MISSING_MIN_ROWS,
            "featureMissingMaxGlobalRate": MODEL_MAX_FEATURE_MISSING_RATE,
            "featureMissingMaxLagRate": MODEL_MAX_LAG_MISSING_RATE,
            "featureMissingMaxWeatherRate": MODEL_MAX_WEATHER_MISSING_RATE,
            "featureImputationEnabled": True,
            "featureImputationStrategy": "median_by_feature",
            "featureClipEnabled": MODEL_FEATURE_CLIP_ENABLED,
            "featureClipLowerQ": MODEL_FEATURE_CLIP_LOWER_Q,
            "featureClipUpperQ": MODEL_FEATURE_CLIP_UPPER_Q,
            "featureClipMinSpread": MODEL_FEATURE_CLIP_MIN_SPREAD,
            "featureAbsMax": MODEL_FEATURE_ABS_MAX,
            "minFeatureFiniteRatio": MODEL_MIN_FEATURE_FINITE_RATIO,
            "ensembleBlendEnabled": MODEL_ENSEMBLE_BLEND_ENABLED,
            "ensembleTargetAwareEnabled": MODEL_ENSEMBLE_BLEND_ENABLED,
            "ensembleFeatureQualityAdjustEnabled": MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_ENABLED,
            "ensembleFeatureQualityAdjustStrength": MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_STRENGTH,
            "ensembleFeatureQualityAdjustExp": MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_EXP,
            "ensembleSampleSupportAdjustEnabled": MODEL_ENSEMBLE_SAMPLE_SUPPORT_ADJUST_ENABLED,
            "ensembleSampleSupportTarget": MODEL_ENSEMBLE_SAMPLE_SUPPORT_TARGET,
            "ensembleSampleSupportMaxShift": MODEL_ENSEMBLE_SAMPLE_SUPPORT_MAX_SHIFT,
            "ensembleDisagreementIntervalEnabled": MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_ENABLED,
            "ensembleDisagreementIntervalMinDiff": MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_MIN_DIFF,
            "ensembleDisagreementIntervalScale": MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_SCALE,
            "ensembleDisagreementIntervalMaxMult": MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_MAX_MULT,
            "sampleSupportIntervalWidenEnabled": MODEL_SAMPLE_SUPPORT_INTERVAL_WIDEN_ENABLED,
            "sampleSupportIntervalTarget": MODEL_SAMPLE_SUPPORT_INTERVAL_TARGET,
            "sampleSupportIntervalMaxMult": MODEL_SAMPLE_SUPPORT_INTERVAL_MAX_MULT,
            "liveBiasEnabled": LIVE_BIAS_ENABLED,
            "lowSampleBlendEnabled": MODEL_LOW_SAMPLE_BLEND_ENABLED,
            "missingFeatureBlendEnabled": MODEL_MISSING_FEATURE_BLEND_ENABLED,
            "missingFeatureBlendStart": MODEL_MISSING_FEATURE_BLEND_START,
            "missingFeatureBlendFull": MODEL_MISSING_FEATURE_BLEND_FULL,
            "missingFeatureBlendMaxWeight": MODEL_MISSING_FEATURE_BLEND_MAX_WEIGHT,
            "missingFeatureIntervalWidenEnabled": MODEL_MISSING_FEATURE_INTERVAL_WIDEN_ENABLED,
            "missingFeatureIntervalWidenStart": MODEL_MISSING_FEATURE_INTERVAL_WIDEN_START,
            "missingFeatureIntervalWidenFull": MODEL_MISSING_FEATURE_INTERVAL_WIDEN_FULL,
            "missingFeatureIntervalWidenMaxMult": MODEL_MISSING_FEATURE_INTERVAL_WIDEN_MAX_MULT,
            "locationEstimateCacheEnabled": True,
            "featureMatrixCacheEnabled": True,
            "dayEstimatePrecomputeEnabled": True,
            "dayEstimateVectorAggregationEnabled": True,
            "dayEstimateMatrixPrecomputeEnabled": True,
            "dayTargetRowIndexPrecomputeEnabled": True,
            "dayCategoryLocIndexCacheEnabled": True,
            "scheduleBoundaryFlagPrecomputeEnabled": True,
            "scheduleBoundaryVectorApplyEnabled": True,
            "scheduleBoundaryNoOpSkipEnabled": True,
            "dayTargetRowsPrecomputeEnabled": True,
            "datetimeCacheKeysEnabled": True,
            "facilityCacheResetEnabled": True,
            "sharedObservationDatasetEnabled": True,
            "sharedObservationDatasetCacheEnabled": True,
            "coreFeaturePrefixCacheEnabled": True,
            "sharedPredictionMatrixHelperEnabled": True,
            "batchedModelPredictionPrimeEnabled": True,
            "chronologicalTargetPrecomputeEnabled": True,
            "ensembleWeightCacheEnabled": True,
            "biasMemoizationEnabled": True,
            "conformalMemoizationEnabled": True,
            "driftTransitionWeightReuseEnabled": True,
            "conformalHorizonZeroFastPathEnabled": True,
            "finiteMaskIndexReuseEnabled": True,
            "longHorizonBlendEnabled": MODEL_LONG_HORIZON_BLEND_ENABLED,
            "directHorizonEnabled": MODEL_DIRECT_HORIZON_ENABLED,
            "directHorizonHours": configured_direct_horizon_hours(),
            "directHorizonMinPairs": MODEL_DIRECT_HORIZON_MIN_PAIRS,
            "directHorizonSegmentMinPairs": MODEL_DIRECT_HORIZON_SEGMENT_MIN_PAIRS,
            "directHorizonOccupancySegmentsEnabled": MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENTS_ENABLED,
            "directHorizonOccupancySegmentMinPairs": MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENT_MIN_PAIRS,
            "directHorizonMaxBlend": MODEL_DIRECT_HORIZON_MAX_BLEND,
            "pointBiasCorrectionEnabled": MODEL_POINT_BIAS_CORRECTION_ENABLED,
            "pointBiasMinPointsPerSegment": MODEL_POINT_BIAS_MIN_POINTS_PER_SEGMENT,
            "pointBiasOccupancyEnabled": MODEL_POINT_BIAS_OCCUPANCY_ENABLED,
            "pointBiasMinPointsPerOccupancy": MODEL_POINT_BIAS_MIN_POINTS_PER_OCCUPANCY,
            "pointBiasSupportTargetMult": MODEL_POINT_BIAS_SUPPORT_TARGET_MULT,
            "pointBiasMaxAbs": MODEL_POINT_BIAS_MAX_ABS,
            "intervalHourBlockFallbackEnabled": True,
            "intervalDayTypeFallbackEnabled": True,
            "intervalConformalDayTypeFallbackEnabled": True,
            "intervalSegmentBlendTargetMult": INTERVAL_SEGMENT_BLEND_TARGET_MULT,
            "intervalConformalSegmentBlendTargetMult": INTERVAL_CONFORMAL_SEGMENT_BLEND_TARGET_MULT,
            "tuningIntervalErrWeight": MODEL_TUNING_INTERVAL_ERR_WEIGHT,
            "tuningTailErrWeight": MODEL_TUNING_TAIL_ERR_WEIGHT,
            "tuningTailQuantile": MODEL_TUNING_TAIL_QUANTILE,
            "tuningComplexityWeight": MODEL_TUNING_COMPLEXITY_WEIGHT,
            "tuningComplexityDepthRef": MODEL_TUNING_COMPLEXITY_DEPTH_REF,
            "lagTrendSensorFeatureCount": LAG_TREND_SENSOR_FEATURE_COUNT,
            "locationBalanceWeightEnabled": LOCATION_BALANCE_WEIGHT_ENABLED,
            "locationBalanceWeightPower": LOCATION_BALANCE_WEIGHT_POWER,
            "locationBalanceWeightMin": LOCATION_BALANCE_WEIGHT_MIN,
            "locationBalanceWeightMax": LOCATION_BALANCE_WEIGHT_MAX,
            "weightStabilizationEnabled": MODEL_WEIGHT_STABILIZATION_ENABLED,
            "weightClipLowerQ": MODEL_WEIGHT_CLIP_LOWER_Q,
            "weightClipUpperQ": MODEL_WEIGHT_CLIP_UPPER_Q,
            "weightClipMin": MODEL_WEIGHT_CLIP_MIN,
            "weightClipMax": MODEL_WEIGHT_CLIP_MAX,
            "weightNormalizeMean": MODEL_WEIGHT_NORMALIZE_MEAN,
            "featureQualityWeightEnabled": MODEL_FEATURE_QUALITY_WEIGHT_ENABLED,
            "featureQualityWeightMin": MODEL_FEATURE_QUALITY_WEIGHT_MIN,
            "featureQualityWeightPower": MODEL_FEATURE_QUALITY_WEIGHT_POWER,
            "recentDriftBiasEnabled": RECENT_DRIFT_BIAS_ENABLED,
            "recentDriftBiasMaxAbs": RECENT_DRIFT_BIAS_MAX_ABS,
            "recentDriftBiasMinPointsPerHour": RECENT_DRIFT_BIAS_MIN_POINTS_PER_HOUR,
            "recentDriftBiasMinPointsPerOccupancy": RECENT_DRIFT_BIAS_MIN_POINTS_PER_OCCUPANCY,
            "recentDriftBiasHorizonDecayHours": RECENT_DRIFT_BIAS_HORIZON_DECAY_HOURS,
            "recentDriftBiasBlend": RECENT_DRIFT_BIAS_BLEND,
            "recentDriftBiasSupportTargetMult": RECENT_DRIFT_BIAS_SUPPORT_TARGET_MULT,
            "schedulePhaseFeatureCount": SCHEDULE_PHASE_FEATURE_COUNT,
            "scheduleTransitionWeightEnabled": SCHEDULE_TRANSITION_WEIGHT_ENABLED,
            "scheduleTransitionWeightMultiplier": SCHEDULE_TRANSITION_WEIGHT_MULTIPLIER,
            "weatherDerivedFeatureCount": WEATHER_DERIVED_FEATURE_COUNT,
            "weatherQualityFeatureCount": WEATHER_QUALITY_FEATURE_COUNT,
            "outputIncludeWeather": FORECAST_OUTPUT_INCLUDE_WEATHER,
            "outputIncludeIntervalFields": FORECAST_OUTPUT_INCLUDE_INTERVAL_FIELDS,
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
        json.dump(sanitize_for_json(payload), handle, ensure_ascii=False, allow_nan=False)
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
