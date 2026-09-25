"""Forecasting reporting owner; mechanically transferred definitions."""

import sys as _sys
from server.reclive.forecasting import config, features, metrics
from dataclasses import replace
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np







def build_public_metrics(models_by_key, model_status_by_key, run_metrics_by_key, unit_loc_ids):
    """Publish only aligned fresh accepted facility-wide terminal observations."""
    rows, windows = [], []
    for facility_id in (1186, 1656):
        key = config.model_unit_key(facility_id, "__all__")
        run = run_metrics_by_key.get(key, {})
        evidence = run.get("reporting_evidence")
        if (model_status_by_key.get(key) != "trained_and_saved"
                or models_by_key.get(key) is None
                or run.get("reporting_selected_bundle") is not models_by_key[key]
                or not isinstance(evidence, dict)
                or evidence.get("model_key") != key
                or evidence.get("stage") != "terminal_pre_calibration"):
            continue
        candidate_rows = evidence.get("rows", [])
        candidate_windows = evidence.get("windows", [])
        membership = set(unit_loc_ids.get(key, ()))
        if (not candidate_rows or not candidate_windows
                or any(not isinstance(row, metrics.ForecastEvaluationRow) or row.facility_id != facility_id
                       or row.location_id not in membership for row in candidate_rows)
                or any(not isinstance(window, metrics.RollingHoldoutWindow)
                       or window.facility_id != facility_id for window in candidate_windows)):
            continue
        try:
            metrics.compute_rolling_holdout_by_facility(candidate_rows, candidate_windows)
            if any(sum(window.start <= row.target < window.end for window in candidate_windows) != 1
                   for row in candidate_rows):
                continue
            # Actual stage boundaries must remain canonical UTC fixed 24-hour windows.
            ordered = sorted(candidate_windows, key=lambda window: window.start)
            if (any(window.start != metrics.reporting_utc(window.start) for window in ordered)
                    or any((window.end - window.start).total_seconds() > 86400 for window in ordered)
                    or any(left.end != right.start or (left.end - left.start).total_seconds() != 86400
                           for left, right in zip(ordered, ordered[1:]))):
                continue
        except (TypeError, ValueError):
            continue
        rows.extend(candidate_rows)
        windows.extend(candidate_windows)
    result = replace(metrics.metrics_for_rows(rows), rolling_holdout_by_facility=
                      metrics.compute_rolling_holdout_by_facility(rows, windows))
    context = {
        "method": "fixed_model_terminal_holdout",
        "independentBacktest": False,
        "observationScope": "location observations from freshly accepted facility-wide __all__ models; not facility totals or final blended forecasts",
        "evaluationStage": "terminal holdout predictions before holdout-derived calibration and profiles",
        "actualPeopleBasis": "cleaned schedule-adjusted observed bucket mean retained before ratio clipping",
        "capacityBasis": "per-location normalization capacity: maximum encountered in loaded model history; not historical as-of capacity",
        "baselineMethod": "raw last-observation persistence frozen at each UTC window start; observed and fetched availability times must both precede start",
        "baselineSourceLimit": "existing SQL-filtered history before schedule rewriting, jump and flatline cleaning; missing, late or ambiguous availability gives null baseline",
        "windowMethod": "non-overlapping 24-hour UTC windows anchored at the actual terminal split; last ends after the last target plus resample interval",
        "timestampAlignment": "reporting requires canonical DB observation instants to match preserved model instants for every contributing location; absent, unparseable or mixed alignment suppresses that model's metrics and windows",
        "limitations": [
            "one fitted model across terminal windows; no rolling-origin retraining",
            "retrospective preprocessing and full-history priors can use later observations",
            "validation affects tuning and selection; subsequent calibration and champion selection may reuse holdout information",
            "saved-only, skipped, rejected, replaced or misaligned models provide no reporting evidence",
        ],
        "legacyTelemetryUnits": "byFacility/byModel valMae, valRmse, holdoutMae and holdoutRmse are weighted occupancy ratios; guardrail, drift and blend telemetry retain algorithm units",
        "compatibilityAliases": {"valMae": "metrics.maePeople", "valRmse": "metrics.rmsePeople"},
        "observationCounts": dict(result.observation_counts),
    }
    return result, context


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
        if loc_samples.get(loc_id, 0) >= config.MIN_SAMPLES_PER_LOC and not data.get("is_stale")
    )

    warnings: List[str] = []
    critical: List[str] = []

    if invalid_rate > float(config.DATA_QUALITY_MAX_INVALID_ROW_RATE):
        critical.append("invalid_row_rate_high")
    elif invalid_rate > float(config.DATA_QUALITY_MAX_INVALID_ROW_RATE) * 0.75:
        warnings.append("invalid_row_rate_elevated")

    if stale_rate > float(config.DATA_QUALITY_MAX_STALE_LOC_RATE):
        critical.append("stale_location_rate_high")
    elif stale_rate > float(config.DATA_QUALITY_MAX_STALE_LOC_RATE) * 0.75:
        warnings.append("stale_location_rate_elevated")

    if flatline_rate > float(config.DATA_QUALITY_MAX_FLATLINE_LOC_RATE):
        critical.append("flatline_location_rate_high")
    elif flatline_rate > float(config.DATA_QUALITY_MAX_FLATLINE_LOC_RATE) * 0.75:
        warnings.append("flatline_location_rate_elevated")

    if modeled_locations < int(config.DATA_QUALITY_MIN_LOCATIONS_MODELED):
        critical.append("modeled_locations_too_low")

    severity = "ok"
    if critical:
        severity = "critical"
    elif warnings:
        severity = "warning"

    block_training = bool(config.DATA_QUALITY_ALERTS_ENABLED and severity == "critical")
    return {
        "enabled": config.DATA_QUALITY_ALERTS_ENABLED,
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

    for facility_id, facility in config.FACILITIES.items():
        all_key = config.model_unit_key(facility_id, "__all__")
        meta = model_meta_by_key.get(all_key)
        run_metrics = run_metrics_by_key.get(all_key, {})
        status = model_status_by_key.get(all_key)

        train_rows = int((meta.get("trainRows") if meta else run_metrics.get("train_rows")) or 0)
        val_rows = int((meta.get("valRows") if meta else run_metrics.get("val_rows")) or 0)
        holdout_rows = int((meta.get("holdoutRows") if meta else run_metrics.get("holdout_rows")) or 0)
        val_mae = features.to_float_or_none(meta.get("valMae") if meta else run_metrics.get("val_mae"))
        val_rmse = features.to_float_or_none(meta.get("valRmse") if meta else run_metrics.get("val_rmse"))
        val_interval_cov = features.to_float_or_none(
            meta.get("valIntervalCoverage")
            if meta
            else run_metrics.get("val_interval_coverage")
        )
        val_interval_cov_err = features.to_float_or_none(
            meta.get("valIntervalCoverageError")
            if meta
            else run_metrics.get("val_interval_coverage_error")
        )
        holdout_mae = features.to_float_or_none(meta.get("holdoutMae") if meta else run_metrics.get("holdout_mae"))
        holdout_rmse = features.to_float_or_none(meta.get("holdoutRmse") if meta else run_metrics.get("holdout_rmse"))
        trained_at = meta.get("trainedAt") if meta else None

        total_train_rows += max(0, train_rows)
        total_val_rows += max(0, val_rows)

        if val_rows > 0 and val_mae is not None:
            weighted_mae_sum += float(val_mae) * float(val_rows)
            weighted_mae_rows += int(val_rows)
        if val_rows > 0 and val_rmse is not None:
            weighted_rmse_sum += float(val_rmse) * float(val_rows)
            weighted_rmse_rows += int(val_rows)

        parsed_trained_at = features.parse_iso_datetime(str(trained_at)) if trained_at else None
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
            "valMae": features.to_float_or_none(meta.get("valMae") if meta else run_metrics.get("val_mae")),
            "valRmse": features.to_float_or_none(meta.get("valRmse") if meta else run_metrics.get("val_rmse")),
            "valIntervalCoverage": features.to_float_or_none(
                meta.get("valIntervalCoverage")
                if meta
                else run_metrics.get("val_interval_coverage")
            ),
            "valIntervalCoverageError": features.to_float_or_none(
                meta.get("valIntervalCoverageError")
                if meta
                else run_metrics.get("val_interval_coverage_error")
            ),
            "selectedBoostRounds": int(
                (meta.get("selectedBoostRounds") if meta else run_metrics.get("selected_boost_rounds"))
                or 0
            ),
            "holdoutMae": features.to_float_or_none(meta.get("holdoutMae") if meta else run_metrics.get("holdout_mae")),
            "holdoutRmse": features.to_float_or_none(meta.get("holdoutRmse") if meta else run_metrics.get("holdout_rmse")),
            "holdoutIntervalCoverage": features.to_float_or_none(
                meta.get("holdoutIntervalCoverage")
                if meta
                else run_metrics.get("holdout_interval_coverage")
            ),
            "holdoutIntervalCoverageError": features.to_float_or_none(
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


_sys.modules.setdefault("server.reclive.forecasting.reporting", _sys.modules[__name__])
_sys.modules.setdefault("reclive.forecasting.reporting", _sys.modules[__name__])
