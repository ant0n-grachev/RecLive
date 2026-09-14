"""Forecasting config owner; mechanically transferred definitions."""

import sys as _sys
import os
from datetime import date
from typing import Dict, List

import pytz
from server.reclive.observability import ForecastConfigurationError
from server.env_loader import (
    load_project_dotenv,
)


from server.facility_schedule import MINUTES_PER_DAY as SHARED_SCHEDULE_MINUTES_PER_DAY



SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


load_project_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise ForecastConfigurationError(name, "missing")

    normalized = value.strip()
    if not normalized:
        raise ForecastConfigurationError(name, "missing")
    return normalized


def require_int_env(name: str) -> int:
    raw = require_env(name)
    try:
        return int(raw)
    except ValueError:
        raise ForecastConfigurationError(name, "invalid_integer") from None


def require_float_env(name: str) -> float:
    raw = require_env(name)
    try:
        return float(raw)
    except ValueError:
        raise ForecastConfigurationError(name, "invalid_float") from None


def int_env_with_default(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        raise ForecastConfigurationError(name, "invalid_integer") from None


def float_env_with_default(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        raise ForecastConfigurationError(name, "invalid_float") from None


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


RESAMPLE_MINUTES = max(5, int_env_with_default("GYM_RESAMPLE_MINUTES", "15"))


WINDOW_RESAMPLE_MINUTES = max(5, int_env_with_default("GYM_WINDOW_RESAMPLE_MINUTES", "30"))


CROWD_BAND_BRIDGE_MIN = max(
    0,
    int_env_with_default("CROWD_BAND_BRIDGE_MIN", os.getenv("CROWD_BAND_MEDIUM_BRIDGE_MIN", "90")),
)


HISTORY_DAYS = int_env_with_default("GYM_MODEL_HISTORY_DAYS", "0")


MIN_SAMPLES_PER_LOC = int_env_with_default("MIN_SAMPLES_PER_LOC", "80")


MIN_TRAIN_SAMPLES = int_env_with_default("MIN_TRAIN_SAMPLES", "300")


TRAIN_SPLIT = float_env_with_default("MODEL_TRAIN_SPLIT", "0.8")


EARLY_STOPPING_ROUNDS = int_env_with_default("MODEL_EARLY_STOPPING", "50")


MODEL_MAX_DEPTH = int_env_with_default("MODEL_MAX_DEPTH", "4")


MODEL_NUM_BOOST_ROUND = int_env_with_default("MODEL_NUM_BOOST_ROUND", "500")


MODEL_NTHREAD = int_env_with_default("MODEL_NTHREAD", "2")


MODEL_ETA = float_env_with_default("MODEL_ETA", "0.05")


MODEL_SUBSAMPLE = float_env_with_default("MODEL_SUBSAMPLE", "0.9")


MODEL_COLSAMPLE_BYTREE = float_env_with_default("MODEL_COLSAMPLE_BYTREE", "0.9")


MODEL_MIN_CHILD_WEIGHT = float_env_with_default("MODEL_MIN_CHILD_WEIGHT", "1.0")


MODEL_GAMMA = float_env_with_default("MODEL_GAMMA", "0.0")


MODEL_REG_LAMBDA = float_env_with_default("MODEL_REG_LAMBDA", "1.0")


MODEL_REG_ALPHA = float_env_with_default("MODEL_REG_ALPHA", "0.0")


MODEL_TREE_METHOD = os.getenv("MODEL_TREE_METHOD", "hist").strip() or "hist"


MODEL_MAX_BIN = int_env_with_default("MODEL_MAX_BIN", "256")


MODEL_RETRAIN_HOURS = int_env_with_default("MODEL_RETRAIN_HOURS", "24")


MODEL_GUARDRAIL_MAX_MAE_DEGRADE = float_env_with_default("MODEL_GUARDRAIL_MAX_MAE_DEGRADE", "0.05")


MODEL_GUARDRAIL_MIN_VAL_ROWS = int_env_with_default("MODEL_GUARDRAIL_MIN_VAL_ROWS", "200")


MODEL_HOLDOUT_SPLIT = float_env_with_default("MODEL_HOLDOUT_SPLIT", "0.12")


MODEL_HOLDOUT_MIN_ROWS = int_env_with_default("MODEL_HOLDOUT_MIN_ROWS", "120")


MODEL_GUARDRAIL_MAX_HOLDOUT_MAE_DEGRADE = float_env_with_default("MODEL_GUARDRAIL_MAX_HOLDOUT_MAE_DEGRADE", "0.03")


MODEL_GUARDRAIL_MAX_HOLDOUT_INTERVAL_ERR_DEGRADE = float_env_with_default("MODEL_GUARDRAIL_MAX_HOLDOUT_INTERVAL_ERR_DEGRADE", "0.02")


MODEL_GUARDRAIL_MAX_VAL_INTERVAL_ERR_DEGRADE = float_env_with_default("MODEL_GUARDRAIL_MAX_VAL_INTERVAL_ERR_DEGRADE", "0.015")


MODEL_FEATURE_MISSING_GUARD_ENABLED = os.getenv("MODEL_FEATURE_MISSING_GUARD_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_FEATURE_MISSING_MIN_ROWS = int_env_with_default("MODEL_FEATURE_MISSING_MIN_ROWS", "240")


MODEL_MAX_FEATURE_MISSING_RATE = float_env_with_default("MODEL_MAX_FEATURE_MISSING_RATE", "0.45")


MODEL_MAX_LAG_MISSING_RATE = float_env_with_default("MODEL_MAX_LAG_MISSING_RATE", "0.55")


MODEL_MAX_WEATHER_MISSING_RATE = float_env_with_default("MODEL_MAX_WEATHER_MISSING_RATE", "0.65")


MODEL_FEATURE_CLIP_ENABLED = os.getenv("MODEL_FEATURE_CLIP_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_FEATURE_CLIP_LOWER_Q = float_env_with_default("MODEL_FEATURE_CLIP_LOWER_Q", "0.002")


MODEL_FEATURE_CLIP_UPPER_Q = float_env_with_default("MODEL_FEATURE_CLIP_UPPER_Q", "0.998")


MODEL_FEATURE_CLIP_MIN_SPREAD = float_env_with_default("MODEL_FEATURE_CLIP_MIN_SPREAD", "0.01")


MODEL_FEATURE_ABS_MAX = float_env_with_default("MODEL_FEATURE_ABS_MAX", "500.0")


MODEL_MIN_FEATURE_FINITE_RATIO = float_env_with_default("MODEL_MIN_FEATURE_FINITE_RATIO", "0.30")


MODEL_ENSEMBLE_BLEND_ENABLED = os.getenv("MODEL_ENSEMBLE_BLEND_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_ENSEMBLE_DEFAULT_PRIMARY_WEIGHT = float_env_with_default("MODEL_ENSEMBLE_DEFAULT_PRIMARY_WEIGHT", "0.72")


MODEL_ENSEMBLE_MIN_PRIMARY_WEIGHT = float_env_with_default("MODEL_ENSEMBLE_MIN_PRIMARY_WEIGHT", "0.15")


MODEL_ENSEMBLE_MAX_PRIMARY_WEIGHT = float_env_with_default("MODEL_ENSEMBLE_MAX_PRIMARY_WEIGHT", "0.9")


MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_ENABLED = os.getenv(
    "MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_ENABLED",
    "1",
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_STRENGTH = float_env_with_default("MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_STRENGTH", "0.45")


MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_EXP = float_env_with_default("MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_EXP", "1.6")


MODEL_ENSEMBLE_SAMPLE_SUPPORT_ADJUST_ENABLED = os.getenv(
    "MODEL_ENSEMBLE_SAMPLE_SUPPORT_ADJUST_ENABLED",
    "1",
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_ENSEMBLE_SAMPLE_SUPPORT_TARGET = int_env_with_default("MODEL_ENSEMBLE_SAMPLE_SUPPORT_TARGET", "80")


MODEL_ENSEMBLE_SAMPLE_SUPPORT_MAX_SHIFT = float_env_with_default("MODEL_ENSEMBLE_SAMPLE_SUPPORT_MAX_SHIFT", "0.22")


LIVE_BIAS_ENABLED = os.getenv("MODEL_LIVE_BIAS_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


LIVE_BIAS_MAX_AGE_MIN = float_env_with_default("MODEL_LIVE_BIAS_MAX_AGE_MIN", "45")


LIVE_BIAS_MAX_HORIZON_HOURS = float_env_with_default("MODEL_LIVE_BIAS_MAX_HORIZON_HOURS", "4")


LIVE_BIAS_BASE_WEIGHT = float_env_with_default("MODEL_LIVE_BIAS_BASE_WEIGHT", "0.55")


LIVE_BIAS_HORIZON_DECAY = float_env_with_default("MODEL_LIVE_BIAS_HORIZON_DECAY", "1.6")


LIVE_BIAS_AGE_DECAY_MIN = float_env_with_default("MODEL_LIVE_BIAS_AGE_DECAY_MIN", "35")


MODEL_LOW_SAMPLE_BLEND_ENABLED = os.getenv("MODEL_LOW_SAMPLE_BLEND_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_LOW_SAMPLE_TARGET_COUNT = int_env_with_default("MODEL_LOW_SAMPLE_TARGET_COUNT", "60")


MODEL_LOW_SAMPLE_MAX_BLEND = float_env_with_default("MODEL_LOW_SAMPLE_MAX_BLEND", "0.35")


MODEL_MISSING_FEATURE_BLEND_ENABLED = os.getenv("MODEL_MISSING_FEATURE_BLEND_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_MISSING_FEATURE_BLEND_START = float_env_with_default("MODEL_MISSING_FEATURE_BLEND_START", "0.25")


MODEL_MISSING_FEATURE_BLEND_FULL = float_env_with_default("MODEL_MISSING_FEATURE_BLEND_FULL", "0.60")


MODEL_MISSING_FEATURE_BLEND_MAX_WEIGHT = float_env_with_default("MODEL_MISSING_FEATURE_BLEND_MAX_WEIGHT", "0.45")


MODEL_MISSING_FEATURE_INTERVAL_WIDEN_ENABLED = os.getenv(
    "MODEL_MISSING_FEATURE_INTERVAL_WIDEN_ENABLED",
    "1",
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_MISSING_FEATURE_INTERVAL_WIDEN_START = float_env_with_default("MODEL_MISSING_FEATURE_INTERVAL_WIDEN_START", "0.20")


MODEL_MISSING_FEATURE_INTERVAL_WIDEN_FULL = float_env_with_default("MODEL_MISSING_FEATURE_INTERVAL_WIDEN_FULL", "0.60")


MODEL_MISSING_FEATURE_INTERVAL_WIDEN_MAX_MULT = float_env_with_default("MODEL_MISSING_FEATURE_INTERVAL_WIDEN_MAX_MULT", "1.55")


MODEL_SAMPLE_SUPPORT_INTERVAL_WIDEN_ENABLED = os.getenv(
    "MODEL_SAMPLE_SUPPORT_INTERVAL_WIDEN_ENABLED",
    "1",
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_SAMPLE_SUPPORT_INTERVAL_TARGET = int_env_with_default("MODEL_SAMPLE_SUPPORT_INTERVAL_TARGET", "80")


MODEL_SAMPLE_SUPPORT_INTERVAL_MAX_MULT = float_env_with_default("MODEL_SAMPLE_SUPPORT_INTERVAL_MAX_MULT", "1.28")


MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_ENABLED = os.getenv(
    "MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_ENABLED",
    "1",
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_MIN_DIFF = float_env_with_default("MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_MIN_DIFF", "0.03")


MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_SCALE = float_env_with_default("MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_SCALE", "0.75")


MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_MAX_MULT = float_env_with_default("MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_MAX_MULT", "1.38")


MODEL_POINT_BIAS_CORRECTION_ENABLED = os.getenv("MODEL_POINT_BIAS_CORRECTION_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_POINT_BIAS_MIN_POINTS_PER_SEGMENT = int_env_with_default("MODEL_POINT_BIAS_MIN_POINTS_PER_SEGMENT", "30")


MODEL_POINT_BIAS_MAX_ABS = float_env_with_default("MODEL_POINT_BIAS_MAX_ABS", "0.25")


MODEL_POINT_BIAS_OCCUPANCY_ENABLED = os.getenv("MODEL_POINT_BIAS_OCCUPANCY_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_POINT_BIAS_MIN_POINTS_PER_OCCUPANCY = int_env_with_default("MODEL_POINT_BIAS_MIN_POINTS_PER_OCCUPANCY", "24")


MODEL_POINT_BIAS_SUPPORT_TARGET_MULT = float_env_with_default("MODEL_POINT_BIAS_SUPPORT_TARGET_MULT", "3.0")


MODEL_TUNING_INTERVAL_ERR_WEIGHT = float_env_with_default("MODEL_TUNING_INTERVAL_ERR_WEIGHT", "0.2")


MODEL_TUNING_TAIL_ERR_WEIGHT = float_env_with_default("MODEL_TUNING_TAIL_ERR_WEIGHT", "0.15")


MODEL_TUNING_TAIL_QUANTILE = float_env_with_default("MODEL_TUNING_TAIL_QUANTILE", "0.9")


MODEL_TUNING_COMPLEXITY_WEIGHT = float_env_with_default("MODEL_TUNING_COMPLEXITY_WEIGHT", "0.03")


MODEL_TUNING_COMPLEXITY_DEPTH_REF = int_env_with_default("MODEL_TUNING_COMPLEXITY_DEPTH_REF", "4")


MODEL_LONG_HORIZON_BLEND_ENABLED = os.getenv("MODEL_LONG_HORIZON_BLEND_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_LONG_HORIZON_BLEND_START_HOURS = float_env_with_default("MODEL_LONG_HORIZON_BLEND_START_HOURS", "4")


MODEL_LONG_HORIZON_BLEND_FULL_HOURS = float_env_with_default("MODEL_LONG_HORIZON_BLEND_FULL_HOURS", "12")


MODEL_LONG_HORIZON_BLEND_MAX_WEIGHT = float_env_with_default("MODEL_LONG_HORIZON_BLEND_MAX_WEIGHT", "0.28")


MODEL_DIRECT_HORIZON_ENABLED = os.getenv("MODEL_DIRECT_HORIZON_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_DIRECT_HORIZON_HOURS_RAW = os.getenv("MODEL_DIRECT_HORIZON_HOURS", "1,2,3,6,12")


MODEL_DIRECT_HORIZON_MIN_PAIRS = int_env_with_default("MODEL_DIRECT_HORIZON_MIN_PAIRS", "100")


MODEL_DIRECT_HORIZON_SEGMENT_MIN_PAIRS = int_env_with_default("MODEL_DIRECT_HORIZON_SEGMENT_MIN_PAIRS", "30")


MODEL_DIRECT_HORIZON_MAX_BLEND = float_env_with_default("MODEL_DIRECT_HORIZON_MAX_BLEND", "0.45")


MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENTS_ENABLED = os.getenv(
    "MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENTS_ENABLED",
    "1",
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENT_MIN_PAIRS = int_env_with_default("MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENT_MIN_PAIRS", "18")


MODEL_TUNING_ENABLED = os.getenv("MODEL_TUNING_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_TUNING_MIN_ROWS = int_env_with_default("MODEL_TUNING_MIN_ROWS", "1200")


MODEL_TUNING_CV_FOLDS = int_env_with_default("MODEL_TUNING_CV_FOLDS", "3")


MODEL_TUNING_MAX_CANDIDATES = int_env_with_default("MODEL_TUNING_MAX_CANDIDATES", "16")


MODEL_TUNING_BOOST_ROUND = int_env_with_default("MODEL_TUNING_BOOST_ROUND", "250")


MODEL_TUNING_RANDOM_SEED = int_env_with_default("MODEL_TUNING_RANDOM_SEED", "7")


MODEL_PARALLEL_WORKERS = int_env_with_default("MODEL_PARALLEL_WORKERS", "2")


CHAMPION_GATE_ENABLED = os.getenv("MODEL_CHAMPION_GATE_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


CHAMPION_GATE_RECENT_DAYS = int_env_with_default("MODEL_CHAMPION_GATE_RECENT_DAYS", "14")


CHAMPION_GATE_MIN_ROWS = int_env_with_default("MODEL_CHAMPION_GATE_MIN_ROWS", "240")


CHAMPION_GATE_MIN_MAE_IMPROVEMENT = float_env_with_default("MODEL_CHAMPION_GATE_MIN_MAE_IMPROVEMENT", "0.0015")


CHAMPION_GATE_MAX_RMSE_DEGRADE = float_env_with_default("MODEL_CHAMPION_GATE_MAX_RMSE_DEGRADE", "0.01")


CHAMPION_GATE_MAX_INTERVAL_ERR_DEGRADE = float_env_with_default("MODEL_CHAMPION_GATE_MAX_INTERVAL_ERR_DEGRADE", "0.01")


CHAMPION_ROLLBACK_ENABLED = os.getenv("MODEL_CHAMPION_ROLLBACK_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


CHAMPION_ROLLBACK_DRIFT_STREAK = int_env_with_default("MODEL_CHAMPION_ROLLBACK_DRIFT_STREAK", "3")


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


OCCUPANCY_WEIGHT_ALPHA = float_env_with_default("MODEL_OCCUPANCY_WEIGHT_ALPHA", "1.2")


OCCUPANCY_WEIGHT_GAMMA = float_env_with_default("MODEL_OCCUPANCY_WEIGHT_GAMMA", "1.4")


LOCATION_BALANCE_WEIGHT_ENABLED = os.getenv("MODEL_LOCATION_BALANCE_WEIGHT_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


LOCATION_BALANCE_WEIGHT_POWER = float_env_with_default("MODEL_LOCATION_BALANCE_WEIGHT_POWER", "0.5")


LOCATION_BALANCE_WEIGHT_MIN = float_env_with_default("MODEL_LOCATION_BALANCE_WEIGHT_MIN", "0.55")


LOCATION_BALANCE_WEIGHT_MAX = float_env_with_default("MODEL_LOCATION_BALANCE_WEIGHT_MAX", "1.8")


MODEL_WEIGHT_STABILIZATION_ENABLED = os.getenv("MODEL_WEIGHT_STABILIZATION_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_WEIGHT_CLIP_LOWER_Q = float_env_with_default("MODEL_WEIGHT_CLIP_LOWER_Q", "0.02")


MODEL_WEIGHT_CLIP_UPPER_Q = float_env_with_default("MODEL_WEIGHT_CLIP_UPPER_Q", "0.98")


MODEL_WEIGHT_CLIP_MIN = float_env_with_default("MODEL_WEIGHT_CLIP_MIN", "0.08")


MODEL_WEIGHT_CLIP_MAX = float_env_with_default("MODEL_WEIGHT_CLIP_MAX", "6.0")


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


RECENCY_HALFLIFE_DAYS = float_env_with_default("MODEL_RECENCY_HALFLIFE_DAYS", "45")


RECENCY_MIN_WEIGHT = float_env_with_default("MODEL_RECENCY_MIN_WEIGHT", "0.2")


DRIFT_ENABLED = os.getenv("MODEL_DRIFT_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


DRIFT_RECENT_DAYS = int_env_with_default("MODEL_DRIFT_RECENT_DAYS", "14")


DRIFT_MIN_POINTS = int_env_with_default("MODEL_DRIFT_MIN_POINTS", "120")


DRIFT_ALERT_MULTIPLIER = float_env_with_default("MODEL_DRIFT_ALERT_MULTIPLIER", "0.2")


DRIFT_ACTIONS_ENABLED = os.getenv("MODEL_DRIFT_ACTIONS_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


DRIFT_ACTION_STREAK_FOR_RETRAIN = int_env_with_default("MODEL_DRIFT_ACTION_STREAK_FOR_RETRAIN", "2")


DRIFT_ACTION_FORCE_HOURS = int_env_with_default("MODEL_DRIFT_ACTION_FORCE_HOURS", "24")


DRIFT_ACTION_INTERVAL_MULTIPLIER_STEP = float_env_with_default("MODEL_DRIFT_ACTION_INTERVAL_MULTIPLIER_STEP", "0.15")


DRIFT_ACTION_INTERVAL_MAX_MULTIPLIER = float_env_with_default("MODEL_DRIFT_ACTION_INTERVAL_MAX_MULTIPLIER", "1.7")


RECENT_DRIFT_BIAS_ENABLED = os.getenv("MODEL_RECENT_DRIFT_BIAS_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


RECENT_DRIFT_BIAS_MAX_ABS = float_env_with_default("MODEL_RECENT_DRIFT_BIAS_MAX_ABS", "0.12")


RECENT_DRIFT_BIAS_MIN_POINTS_PER_HOUR = int_env_with_default("MODEL_RECENT_DRIFT_BIAS_MIN_POINTS_PER_HOUR", "20")


RECENT_DRIFT_BIAS_MIN_POINTS_PER_OCCUPANCY = int_env_with_default("MODEL_RECENT_DRIFT_BIAS_MIN_POINTS_PER_OCCUPANCY", "24")


RECENT_DRIFT_BIAS_HORIZON_DECAY_HOURS = float_env_with_default("MODEL_RECENT_DRIFT_BIAS_HORIZON_DECAY_HOURS", "8")


RECENT_DRIFT_BIAS_BLEND = float_env_with_default("MODEL_RECENT_DRIFT_BIAS_BLEND", "0.6")


RECENT_DRIFT_BIAS_SUPPORT_TARGET_MULT = float_env_with_default("MODEL_RECENT_DRIFT_BIAS_SUPPORT_TARGET_MULT", "3.0")


ADAPTIVE_CONTROLS_ENABLED = os.getenv("MODEL_ADAPTIVE_CONTROLS_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


ADAPTIVE_HISTORY_MAX_POINTS = int_env_with_default("MODEL_ADAPTIVE_HISTORY_MAX_POINTS", "60")


ADAPTIVE_RETRAIN_MIN_HOURS = int_env_with_default("MODEL_ADAPTIVE_RETRAIN_MIN_HOURS", "8")


ADAPTIVE_RETRAIN_MAX_HOURS = int_env_with_default("MODEL_ADAPTIVE_RETRAIN_MAX_HOURS", "72")


ADAPTIVE_DRIFT_DAYS_MIN = int_env_with_default("MODEL_ADAPTIVE_DRIFT_DAYS_MIN", "7")


ADAPTIVE_DRIFT_DAYS_MAX = int_env_with_default("MODEL_ADAPTIVE_DRIFT_DAYS_MAX", "21")


ADAPTIVE_DRIFT_MULT_MIN = float_env_with_default("MODEL_ADAPTIVE_DRIFT_MULT_MIN", "0.12")


ADAPTIVE_DRIFT_MULT_MAX = float_env_with_default("MODEL_ADAPTIVE_DRIFT_MULT_MAX", "0.35")


ADAPTIVE_ALERT_RATE_STABLE_MAX = float_env_with_default("MODEL_ADAPTIVE_ALERT_RATE_STABLE_MAX", "0.15")


ADAPTIVE_ALERT_RATE_UNSTABLE_MIN = float_env_with_default("MODEL_ADAPTIVE_ALERT_RATE_UNSTABLE_MIN", "0.40")


FORCE_RETRAIN = os.getenv("MODEL_FORCE_RETRAIN", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_SCHEMA_VERSION = 10


INTERVAL_MIN_SAMPLES_PER_HOUR = int_env_with_default("INTERVAL_MIN_SAMPLES_PER_HOUR", "30")


INTERVAL_Q_LOW = float_env_with_default("INTERVAL_Q_LOW", "0.10")


INTERVAL_Q_HIGH = float_env_with_default("INTERVAL_Q_HIGH", "0.90")


INTERVAL_CONFORMAL_ENABLED = os.getenv("INTERVAL_CONFORMAL_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


INTERVAL_CONFORMAL_RECENT_DAYS = int_env_with_default("INTERVAL_CONFORMAL_RECENT_DAYS", "21")


INTERVAL_CONFORMAL_ALPHA = float_env_with_default("INTERVAL_CONFORMAL_ALPHA", "0.20")


INTERVAL_CONFORMAL_MIN_POINTS = int_env_with_default("INTERVAL_CONFORMAL_MIN_POINTS", "120")


INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR = int_env_with_default("INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR", "20")


INTERVAL_CONFORMAL_MAX_MARGIN = float_env_with_default("INTERVAL_CONFORMAL_MAX_MARGIN", "0.35")


INTERVAL_CONFORMAL_SEGMENT_BLEND_TARGET_MULT = float_env_with_default("INTERVAL_CONFORMAL_SEGMENT_BLEND_TARGET_MULT", "3.0")


INTERVAL_SEGMENT_BLEND_TARGET_MULT = float_env_with_default("INTERVAL_SEGMENT_BLEND_TARGET_MULT", "3.0")


FEATURE_ABLATION_ENABLED = os.getenv("MODEL_FEATURE_ABLATION_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


FEATURE_ABLATION_MIN_VAL_ROWS = int_env_with_default("MODEL_FEATURE_ABLATION_MIN_VAL_ROWS", "120")


DATA_QUALITY_ALERTS_ENABLED = os.getenv("DATA_QUALITY_ALERTS_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


DATA_QUALITY_MAX_INVALID_ROW_RATE = float_env_with_default("DATA_QUALITY_MAX_INVALID_ROW_RATE", "0.35")


DATA_QUALITY_MAX_STALE_LOC_RATE = float_env_with_default("DATA_QUALITY_MAX_STALE_LOC_RATE", "0.6")


DATA_QUALITY_MAX_FLATLINE_LOC_RATE = float_env_with_default("DATA_QUALITY_MAX_FLATLINE_LOC_RATE", "0.4")


DATA_QUALITY_MIN_LOCATIONS_MODELED = int_env_with_default("DATA_QUALITY_MIN_LOCATIONS_MODELED", "6")


DB_TIMEZONE_NAME = require_env("GYM_DB_TIMEZONE")


try:
    DB_TZ = pytz.timezone(DB_TIMEZONE_NAME)
except pytz.UnknownTimeZoneError:
    raise ForecastConfigurationError("GYM_DB_TIMEZONE", "invalid_timezone") from None


FORECAST_DAY_START_HOUR = require_int_env("FORECAST_DAY_START_HOUR")


FORECAST_DAY_END_HOUR = require_int_env("FORECAST_DAY_END_HOUR")


CROWD_BASELINE_MIN_COVERAGE = float_env_with_default("CROWD_BASELINE_MIN_COVERAGE", "0.6")


CROWD_BASELINE_MIN_POINTS = int_env_with_default("CROWD_BASELINE_MIN_POINTS", "120")


CROWD_BASELINE_LOW_QUANTILE = float_env_with_default("CROWD_BASELINE_LOW_QUANTILE", "0.3")


CROWD_BASELINE_PEAK_QUANTILE = float_env_with_default("CROWD_BASELINE_PEAK_QUANTILE", "0.7")


STALE_SENSOR_HOURS = float_env_with_default("STALE_SENSOR_HOURS", "24")


IMPOSSIBLE_JUMP_PCT = float_env_with_default("IMPOSSIBLE_JUMP_PCT", "0.60")


IMPOSSIBLE_JUMP_MAX_GAP_MIN = float_env_with_default("IMPOSSIBLE_JUMP_MAX_GAP_MIN", "120")


SENSOR_FLATLINE_MAX_GAP_MIN = float_env_with_default("SENSOR_FLATLINE_MAX_GAP_MIN", "20")


SENSOR_FLATLINE_MIN_DURATION_MIN = float_env_with_default("SENSOR_FLATLINE_MIN_DURATION_MIN", "360")


SENSOR_FLATLINE_KEEP_INTERVAL_MIN = float_env_with_default("SENSOR_FLATLINE_KEEP_INTERVAL_MIN", "60")


SENSOR_FLATLINE_TOLERANCE_PCT = float_env_with_default("SENSOR_FLATLINE_TOLERANCE_PCT", "0.01")


SENSOR_WEIGHT_MIN = float_env_with_default("SENSOR_WEIGHT_MIN", "0.25")


MODEL_FEATURE_QUALITY_WEIGHT_ENABLED = os.getenv("MODEL_FEATURE_QUALITY_WEIGHT_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


MODEL_FEATURE_QUALITY_WEIGHT_MIN = float_env_with_default("MODEL_FEATURE_QUALITY_WEIGHT_MIN", "0.35")


MODEL_FEATURE_QUALITY_WEIGHT_POWER = float_env_with_default("MODEL_FEATURE_QUALITY_WEIGHT_POWER", "1.25")


SPIKE_AWARE_ENABLED = os.getenv("SPIKE_AWARE_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


SPIKE_AWARE_MAX_AGE_MIN = float_env_with_default("SPIKE_AWARE_MAX_AGE_MIN", "90")


SPIKE_AWARE_HORIZON_HOURS = int_env_with_default("SPIKE_AWARE_HORIZON_HOURS", "6")


SPIKE_AWARE_DECAY = float_env_with_default("SPIKE_AWARE_DECAY", "0.55")


SPIKE_AWARE_MAX_CAP_MULTIPLIER = float_env_with_default("SPIKE_AWARE_MAX_CAP_MULTIPLIER", "1.35")


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
    float_env_with_default("SCHEDULE_TRANSITION_WEIGHT_MULTIPLIER", "1.35"),
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


SCHEDULE_MINUTES_PER_DAY = SHARED_SCHEDULE_MINUTES_PER_DAY


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


_sys.modules.setdefault("server.reclive.forecasting.config", _sys.modules[__name__])
_sys.modules.setdefault("reclive.forecasting.config", _sys.modules[__name__])
