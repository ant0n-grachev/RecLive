"""Closed operational events: values and exception details never become diagnostics."""

import json
import sys
from datetime import datetime, timezone

from server.env_loader import EnvironmentConfigurationError

EVENT_FIELDS = {
    "ingestion.completed": {"receivedCount", "historyInsertedCount", "unchangedCount"},
    "ingestion.failed": {"errorCategory"},
    "forecast.completed": {"generatedFacilities"},
    "forecast.failed": set(),
    "forecast.configuration_failed": {"configurationName", "reason"},
    "forecast.model_metadata_write_failed": {"errorCategory"},
    "forecast.model_unit_failed": set(),
    "schedules.completed": {"facilityCount", "failedFacilityCount"},
    "schedules.failed": {"errorCategory"},
    "push.evaluator_failed": {"errorCategory"},
}
ERROR_CATEGORIES = frozenset({
    "database_unavailable", "file_unavailable", "network_error",
    "push_unavailable", "validation_error",
})
CONFIGURATION_NAMES = frozenset({
    "APP_ENV",
    "CROWD_BAND_BRIDGE_MIN",
    "CROWD_BAND_MEDIUM_BRIDGE_MIN",
    "CROWD_BASELINE_LOW_QUANTILE",
    "CROWD_BASELINE_MIN_COVERAGE",
    "CROWD_BASELINE_MIN_POINTS",
    "CROWD_BASELINE_PEAK_QUANTILE",
    "DATA_QUALITY_MAX_FLATLINE_LOC_RATE",
    "DATA_QUALITY_MAX_INVALID_ROW_RATE",
    "DATA_QUALITY_MAX_STALE_LOC_RATE",
    "DATA_QUALITY_MIN_LOCATIONS_MODELED",
    "FORECAST_DAY_END_HOUR",
    "FORECAST_DAY_START_HOUR",
    "FORECAST_JSON_PATH",
    "GYM_DB_HOST",
    "GYM_DB_NAME",
    "GYM_DB_PASSWORD",
    "GYM_DB_PORT",
    "GYM_DB_TIMEZONE",
    "GYM_DB_USER",
    "GYM_MODEL_HISTORY_DAYS",
    "GYM_RESAMPLE_MINUTES",
    "GYM_WEATHER_ARCHIVE_URL",
    "GYM_WEATHER_FORECAST_DAYS",
    "GYM_WEATHER_HISTORY_MAX_DAYS",
    "GYM_WEATHER_LAT",
    "GYM_WEATHER_LON",
    "GYM_WEATHER_URL",
    "GYM_WINDOW_RESAMPLE_MINUTES",
    "IMPOSSIBLE_JUMP_MAX_GAP_MIN",
    "IMPOSSIBLE_JUMP_PCT",
    "INTERVAL_CONFORMAL_ALPHA",
    "INTERVAL_CONFORMAL_MAX_MARGIN",
    "INTERVAL_CONFORMAL_MIN_POINTS",
    "INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR",
    "INTERVAL_CONFORMAL_RECENT_DAYS",
    "INTERVAL_CONFORMAL_SEGMENT_BLEND_TARGET_MULT",
    "INTERVAL_MIN_SAMPLES_PER_HOUR",
    "INTERVAL_Q_HIGH",
    "INTERVAL_Q_LOW",
    "INTERVAL_SEGMENT_BLEND_TARGET_MULT",
    "MIN_SAMPLES_PER_LOC",
    "MIN_TRAIN_SAMPLES",
    "MODEL_ADAPTIVE_ALERT_RATE_STABLE_MAX",
    "MODEL_ADAPTIVE_ALERT_RATE_UNSTABLE_MIN",
    "MODEL_ADAPTIVE_DRIFT_DAYS_MAX",
    "MODEL_ADAPTIVE_DRIFT_DAYS_MIN",
    "MODEL_ADAPTIVE_DRIFT_MULT_MAX",
    "MODEL_ADAPTIVE_DRIFT_MULT_MIN",
    "MODEL_ADAPTIVE_HISTORY_MAX_POINTS",
    "MODEL_ADAPTIVE_RETRAIN_MAX_HOURS",
    "MODEL_ADAPTIVE_RETRAIN_MIN_HOURS",
    "MODEL_ARTIFACT_DIR",
    "MODEL_BASENAME",
    "MODEL_CHAMPION_GATE_MAX_INTERVAL_ERR_DEGRADE",
    "MODEL_CHAMPION_GATE_MAX_RMSE_DEGRADE",
    "MODEL_CHAMPION_GATE_MIN_MAE_IMPROVEMENT",
    "MODEL_CHAMPION_GATE_MIN_ROWS",
    "MODEL_CHAMPION_GATE_RECENT_DAYS",
    "MODEL_CHAMPION_ROLLBACK_DRIFT_STREAK",
    "MODEL_COLSAMPLE_BYTREE",
    "MODEL_DIRECT_HORIZON_MAX_BLEND",
    "MODEL_DIRECT_HORIZON_MIN_PAIRS",
    "MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENT_MIN_PAIRS",
    "MODEL_DIRECT_HORIZON_SEGMENT_MIN_PAIRS",
    "MODEL_DRIFT_ACTION_FORCE_HOURS",
    "MODEL_DRIFT_ACTION_INTERVAL_MAX_MULTIPLIER",
    "MODEL_DRIFT_ACTION_INTERVAL_MULTIPLIER_STEP",
    "MODEL_DRIFT_ACTION_STREAK_FOR_RETRAIN",
    "MODEL_DRIFT_ALERT_MULTIPLIER",
    "MODEL_DRIFT_MIN_POINTS",
    "MODEL_DRIFT_RECENT_DAYS",
    "MODEL_EARLY_STOPPING",
    "MODEL_ENSEMBLE_DEFAULT_PRIMARY_WEIGHT",
    "MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_MAX_MULT",
    "MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_MIN_DIFF",
    "MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_SCALE",
    "MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_EXP",
    "MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_STRENGTH",
    "MODEL_ENSEMBLE_MAX_PRIMARY_WEIGHT",
    "MODEL_ENSEMBLE_MIN_PRIMARY_WEIGHT",
    "MODEL_ENSEMBLE_SAMPLE_SUPPORT_MAX_SHIFT",
    "MODEL_ENSEMBLE_SAMPLE_SUPPORT_TARGET",
    "MODEL_ETA",
    "MODEL_FEATURE_ABLATION_MIN_VAL_ROWS",
    "MODEL_FEATURE_ABS_MAX",
    "MODEL_FEATURE_CLIP_LOWER_Q",
    "MODEL_FEATURE_CLIP_MIN_SPREAD",
    "MODEL_FEATURE_CLIP_UPPER_Q",
    "MODEL_FEATURE_MISSING_MIN_ROWS",
    "MODEL_FEATURE_QUALITY_WEIGHT_MIN",
    "MODEL_FEATURE_QUALITY_WEIGHT_POWER",
    "MODEL_GAMMA",
    "MODEL_GUARDRAIL_MAX_HOLDOUT_INTERVAL_ERR_DEGRADE",
    "MODEL_GUARDRAIL_MAX_HOLDOUT_MAE_DEGRADE",
    "MODEL_GUARDRAIL_MAX_MAE_DEGRADE",
    "MODEL_GUARDRAIL_MAX_VAL_INTERVAL_ERR_DEGRADE",
    "MODEL_GUARDRAIL_MIN_VAL_ROWS",
    "MODEL_HOLDOUT_MIN_ROWS",
    "MODEL_HOLDOUT_SPLIT",
    "MODEL_LIVE_BIAS_AGE_DECAY_MIN",
    "MODEL_LIVE_BIAS_BASE_WEIGHT",
    "MODEL_LIVE_BIAS_HORIZON_DECAY",
    "MODEL_LIVE_BIAS_MAX_AGE_MIN",
    "MODEL_LIVE_BIAS_MAX_HORIZON_HOURS",
    "MODEL_LOCATION_BALANCE_WEIGHT_MAX",
    "MODEL_LOCATION_BALANCE_WEIGHT_MIN",
    "MODEL_LOCATION_BALANCE_WEIGHT_POWER",
    "MODEL_LONG_HORIZON_BLEND_FULL_HOURS",
    "MODEL_LONG_HORIZON_BLEND_MAX_WEIGHT",
    "MODEL_LONG_HORIZON_BLEND_START_HOURS",
    "MODEL_LOW_SAMPLE_MAX_BLEND",
    "MODEL_LOW_SAMPLE_TARGET_COUNT",
    "MODEL_MAX_BIN",
    "MODEL_MAX_DEPTH",
    "MODEL_MAX_FEATURE_MISSING_RATE",
    "MODEL_MAX_LAG_MISSING_RATE",
    "MODEL_MAX_WEATHER_MISSING_RATE",
    "MODEL_MIN_CHILD_WEIGHT",
    "MODEL_MIN_FEATURE_FINITE_RATIO",
    "MODEL_MISSING_FEATURE_BLEND_FULL",
    "MODEL_MISSING_FEATURE_BLEND_MAX_WEIGHT",
    "MODEL_MISSING_FEATURE_BLEND_START",
    "MODEL_MISSING_FEATURE_INTERVAL_WIDEN_FULL",
    "MODEL_MISSING_FEATURE_INTERVAL_WIDEN_MAX_MULT",
    "MODEL_MISSING_FEATURE_INTERVAL_WIDEN_START",
    "MODEL_NTHREAD",
    "MODEL_NUM_BOOST_ROUND",
    "MODEL_OCCUPANCY_WEIGHT_ALPHA",
    "MODEL_OCCUPANCY_WEIGHT_GAMMA",
    "MODEL_PARALLEL_WORKERS",
    "MODEL_POINT_BIAS_MAX_ABS",
    "MODEL_POINT_BIAS_MIN_POINTS_PER_OCCUPANCY",
    "MODEL_POINT_BIAS_MIN_POINTS_PER_SEGMENT",
    "MODEL_POINT_BIAS_SUPPORT_TARGET_MULT",
    "MODEL_RECENCY_HALFLIFE_DAYS",
    "MODEL_RECENCY_MIN_WEIGHT",
    "MODEL_RECENT_DRIFT_BIAS_BLEND",
    "MODEL_RECENT_DRIFT_BIAS_HORIZON_DECAY_HOURS",
    "MODEL_RECENT_DRIFT_BIAS_MAX_ABS",
    "MODEL_RECENT_DRIFT_BIAS_MIN_POINTS_PER_HOUR",
    "MODEL_RECENT_DRIFT_BIAS_MIN_POINTS_PER_OCCUPANCY",
    "MODEL_RECENT_DRIFT_BIAS_SUPPORT_TARGET_MULT",
    "MODEL_REG_ALPHA",
    "MODEL_REG_LAMBDA",
    "MODEL_RETRAIN_HOURS",
    "MODEL_SAMPLE_SUPPORT_INTERVAL_MAX_MULT",
    "MODEL_SAMPLE_SUPPORT_INTERVAL_TARGET",
    "MODEL_SUBSAMPLE",
    "MODEL_TRAIN_SPLIT",
    "MODEL_TUNING_BOOST_ROUND",
    "MODEL_TUNING_COMPLEXITY_DEPTH_REF",
    "MODEL_TUNING_COMPLEXITY_WEIGHT",
    "MODEL_TUNING_CV_FOLDS",
    "MODEL_TUNING_INTERVAL_ERR_WEIGHT",
    "MODEL_TUNING_MAX_CANDIDATES",
    "MODEL_TUNING_MIN_ROWS",
    "MODEL_TUNING_RANDOM_SEED",
    "MODEL_TUNING_TAIL_ERR_WEIGHT",
    "MODEL_TUNING_TAIL_QUANTILE",
    "MODEL_WEIGHT_CLIP_LOWER_Q",
    "MODEL_WEIGHT_CLIP_MAX",
    "MODEL_WEIGHT_CLIP_MIN",
    "MODEL_WEIGHT_CLIP_UPPER_Q",
    "SCHEDULE_TRANSITION_WEIGHT_MULTIPLIER",
    "SENSOR_FLATLINE_KEEP_INTERVAL_MIN",
    "SENSOR_FLATLINE_MAX_GAP_MIN",
    "SENSOR_FLATLINE_MIN_DURATION_MIN",
    "SENSOR_FLATLINE_TOLERANCE_PCT",
    "SENSOR_WEIGHT_MIN",
    "SPIKE_AWARE_DECAY",
    "SPIKE_AWARE_HORIZON_HOURS",
    "SPIKE_AWARE_MAX_AGE_MIN",
    "SPIKE_AWARE_MAX_CAP_MULTIPLIER",
    "STALE_SENSOR_HOURS",
})
CONFIGURATION_REASONS = frozenset({
    "missing", "invalid_integer", "invalid_float", "invalid_timezone",
    "unsafe_configuration",
})


def format_event(event: str, **fields: object) -> str:
    """Validate the entire record before writing to a caller-owned sink."""
    if type(event) is not str or event not in EVENT_FIELDS:
        raise ValueError("unknown operational event")
    if set(fields) - EVENT_FIELDS[event]:
        raise ValueError("invalid operational event fields")
    for name, value in fields.items():
        if isinstance(value, str) and len(value) > 240:
            raise ValueError("invalid operational event fields")
        if name == "errorCategory":
            valid = type(value) is str and value in ERROR_CATEGORIES
            if event == "forecast.model_metadata_write_failed":
                valid = valid and value == "file_unavailable"
        elif name == "configurationName":
            valid = type(value) is str and value in CONFIGURATION_NAMES
        elif name == "reason":
            valid = type(value) is str and value in CONFIGURATION_REASONS
        else:
            valid = type(value) is int and value >= 0
        if not valid:
            raise ValueError("invalid operational event fields")
    return json.dumps(
        {"event": event, "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), **fields},
        separators=(",", ":"), sort_keys=True, allow_nan=False,
    )


def log_event(event: str, **fields: object) -> None:
    print(format_event(event, **fields), flush=True)


def best_effort_event(event: str, *, stderr: bool = False, **fields: object) -> None:
    """A broken output stream must not prevent optional work from continuing."""
    try:
        print(format_event(event, **fields), file=sys.stderr if stderr else sys.stdout, flush=True)
    except Exception:
        pass


class ForecastConfigurationError(EnvironmentConfigurationError):
    """Name-only failure with no retained rejected value in the public message."""

    def __init__(self, name: str, reason: str):
        self.configuration_name = name
        self.reason = reason
        label = {
            "missing": "Missing required env var",
            "invalid_integer": "Invalid integer for env var",
            "invalid_float": "Invalid float for env var",
            "invalid_timezone": "Invalid timezone for env var",
        }[reason]
        super().__init__(f"{label}: {name}")


# Exact matching of existing trusted error forms, never extraction of free text.
_LEGACY_CONFIGURATION_ERRORS = {
    f"Unsafe production configuration: {name}": name
    for name in CONFIGURATION_NAMES
}
_LEGACY_CONFIGURATION_ERRORS["Unsafe environment configuration: APP_ENV"] = "APP_ENV"


def log_configuration_failure(error: EnvironmentConfigurationError) -> None:
    if isinstance(error, ForecastConfigurationError):
        name, reason = error.configuration_name, error.reason
    else:
        # Inspect only a builtin string argument; do not invoke exception __str__.
        message = error.args[0] if len(error.args) == 1 else None
        name = _LEGACY_CONFIGURATION_ERRORS.get(message) if type(message) is str else None
        reason = "unsafe_configuration"
    if type(name) is str and name in CONFIGURATION_NAMES and type(reason) is str and reason in CONFIGURATION_REASONS:
        log_event("forecast.configuration_failed", configurationName=name, reason=reason)
    else:
        log_event("forecast.failed")
