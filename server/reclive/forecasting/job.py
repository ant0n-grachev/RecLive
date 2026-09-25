"""Forecasting job owner; mechanically transferred definitions."""

import sys as _sys
from server.reclive.forecasting import config, data, features, prediction, publication, reporting, training
import json
import os
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from server.env_loader import (
    EnvironmentConfigurationError,
    validate_production_environment,
)

from server.forecast_shared import normalize_section_key
from server.reclive.observability import log_event, log_configuration_failure










def build_forecast():
    saved_meta_snapshot = data.collect_saved_meta_snapshots()
    adaptive_controls = training.derive_adaptive_runtime_controls(saved_meta_snapshot)
    facility_schedule_by_id = data.load_schedule_sections_by_facility()

    conn = data.db_connect()
    try:
        (
            loc_data,
            avg_dow_hour,
            avg_hour,
            avg_overall,
            max_caps,
            loc_samples,
            quality,
        ) = data.load_history(
            conn,
            facility_schedule_by_id=facility_schedule_by_id,
        )
        live_snapshots = data.load_live_snapshots(conn)
    finally:
        conn.close()

    for loc_id, entry in loc_data.items():
        entry["live_snapshot"] = live_snapshots.get(loc_id)

    # A poll can complete while history is loading. Anchor the run after its inputs
    # have been read so a valid new snapshot is not rejected as a future reading.
    now = datetime.now(config.TZ)
    forecast_start_hour, forecast_end_hour = prediction.normalized_forecast_hour_bounds()
    week_dates = [(now + timedelta(days=offset)).date() for offset in range(7)]
    day_targets_by_date: Dict[date, List[datetime]] = {
        day_date: prediction.get_targets_for_date(day_date) for day_date in week_dates
    }
    window_targets_by_date: Dict[date, List[datetime]] = {
        day_date: prediction.get_window_targets_for_date(day_date) for day_date in week_dates
    }

    history_start = data.weather_history_start(loc_data, now)
    weather_history_series = data.fetch_weather_history_series(history_start, now)
    future_weather_series = data.fetch_weather_forecast_series(now)
    weather_series = features.merge_weather_series(weather_history_series, future_weather_series)
    quality["weatherHistoryHours"] = len(weather_history_series.get("times", []))
    quality["weatherForecastHours"] = len(future_weather_series.get("times", []))
    quality["weatherMergedHours"] = len(weather_series.get("times", []))
    quality["weatherAvailable"] = bool(weather_series.get("times"))
    quality["scheduleHoursPath"] = config.FACILITY_HOURS_JSON_PATH
    quality["scheduleHoursAvailable"] = bool(facility_schedule_by_id)
    quality_alerts = reporting.build_data_quality_alerts(
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
    ) = training.prepare_models(
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
        fills = features.coerce_feature_fill_values(
            meta.get("featureFillValues") if isinstance(meta, dict) else None,
            expected_cols=expected_feature_count if expected_feature_count > 0 else None,
        )
        if fills is not None:
            feature_fill_values_by_key[model_key] = fills
        clip_bounds = features.coerce_feature_clip_bounds(
            meta.get("featureClipLower") if isinstance(meta, dict) else None,
            meta.get("featureClipUpper") if isinstance(meta, dict) else None,
            expected_cols=expected_feature_count if expected_feature_count > 0 else None,
        )
        if clip_bounds is not None:
            feature_clip_bounds_by_key[model_key] = clip_bounds

    model_summary = reporting.summarize_model_results(
        model_meta_by_key=model_meta_by_key,
        model_status_by_key=model_status_by_key,
        run_metrics_by_key=run_metrics_by_key,
    )
    recent_model_dataset_cache: Dict[Tuple[object, ...], Dict[str, object]] = {}
    recent_core_feature_cache: Dict[Tuple[int, datetime], List[float]] = {}
    drift_summary = prediction.compute_model_drift(
        now=now,
        models_by_key=models_by_key,
        onehot_by_key=onehot_by_key,
        unit_loc_ids=unit_loc_ids,
        loc_data=loc_data,
        loc_samples=loc_samples,
        weather_series=weather_series,
        model_meta_by_key=model_meta_by_key,
        drift_recent_days=int(adaptive_controls.get("driftRecentDays", config.DRIFT_RECENT_DAYS)),
        drift_alert_multiplier=float(
            adaptive_controls.get("driftAlertMultiplier", config.DRIFT_ALERT_MULTIPLIER)
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
    interval_multiplier_by_key, drift_actions_summary = prediction.apply_drift_actions(
        now=now,
        drift_summary=drift_summary,
        model_meta_by_key=model_meta_by_key,
        action_streak_for_retrain=int(
            adaptive_controls.get("driftActionStreakForRetrain", config.DRIFT_ACTION_STREAK_FOR_RETRAIN)
        ),
        action_force_hours=max(1, int(config.DRIFT_ACTION_FORCE_HOURS)),
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
        restored_bundle, restored_meta = data.load_saved_model(
            model_key=model_key,
            expected_loc_ids=expected_loc_ids,
            expected_feature_count=feature_count,
        )
        if restored_bundle is not None:
            models_by_key[model_key] = restored_bundle
        if isinstance(restored_meta, dict):
            model_meta_by_key[model_key] = restored_meta
            interval_profile_by_key[model_key] = restored_meta.get("residualProfile")
            restored_fills = features.coerce_feature_fill_values(
                restored_meta.get("featureFillValues"),
                expected_cols=feature_count,
            )
            if restored_fills is not None:
                feature_fill_values_by_key[model_key] = restored_fills
            else:
                feature_fill_values_by_key.pop(model_key, None)
            restored_clip = features.coerce_feature_clip_bounds(
                restored_meta.get("featureClipLower"),
                restored_meta.get("featureClipUpper"),
                expected_cols=feature_count,
            )
            if restored_clip is not None:
                feature_clip_bounds_by_key[model_key] = restored_clip
            else:
                feature_clip_bounds_by_key.pop(model_key, None)
    conformal_by_key, conformal_summary = prediction.compute_interval_conformal_profiles(
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
    location_facility_map = config.location_to_facility_map()
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
    location_bucket_totals = prediction.build_location_bucket_total_cache(
        config.all_location_ids(),
        loc_data=loc_data,
        max_caps=max_caps,
    )

    facilities_payload = []
    spike_adjusted_categories = 0
    schedule_boundary_forecast_rows_zeroed = 0

    for facility_id, facility in config.FACILITIES.items():
        weekly_forecast = []
        facility_loc_ids = config.facility_location_ids(facility)
        crowd_baseline_cache: Dict[Tuple[int, ...], Optional[Dict[str, float]]] = {}

        def baseline_for_locations(loc_ids: Iterable[int]) -> Optional[Dict[str, float]]:
            cache_key = tuple(prediction.unique_location_ids(loc_ids))
            if not cache_key:
                return None
            if cache_key not in crowd_baseline_cache:
                crowd_baseline_cache[cache_key] = prediction.build_facility_crowd_baseline(
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
            category_loc_ids = prediction.unique_location_ids(raw_category.get("location_ids", []))
            category_definitions.append(
                {
                    "key": category_key,
                    "title": category_title,
                    "location_ids": category_loc_ids,
                    "maxCapacity": prediction.sum_max_caps(max_caps, category_loc_ids),
                    "thresholdKey": normalize_section_key(category_title or category_key),
                    "forecastEnabled": category_key in config.FORECAST_CATEGORY_KEYS,
                }
            )

        facility_max_capacity = prediction.sum_max_caps(max_caps, facility_loc_ids)
        facility_crowd_baseline = baseline_for_locations(facility_loc_ids)
        facility_occupancy_thresholds = prediction.occupancy_thresholds_from_baseline(
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
            category_thresholds = prediction.occupancy_thresholds_from_baseline(
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
            location_thresholds = prediction.occupancy_thresholds_from_baseline(
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
            live_total, observed_locs = prediction.category_live_total(
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
            day_targets, day_target_boundary_flags = prediction.filter_targets_by_schedule(
                facility_id=facility_id,
                targets=day_targets_by_date.get(day_date, []),
                facility_schedule_by_id=facility_schedule_by_id,
                schedule_eval_cache=schedule_eval_cache,
                schedule_boundary_cache=schedule_boundary_cache,
                date_range_cache=schedule_date_range_cache,
                weekday_cache=schedule_weekday_cache,
                hours_window_cache=schedule_hours_cache,
            )
            day_window_targets, day_window_target_boundary_flags = prediction.filter_targets_by_schedule(
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

        facility_combined_targets = prediction.unique_targets_by_iso(facility_all_targets)
        (
            vector_loc_ids,
            _facility_target_order,
            facility_target_index_lookup,
            facility_target_matrices,
        ) = prediction.precompute_target_estimate_matrices_for_locations(
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
            day_weather = prediction.build_weather_hours_for_targets(targets, weather_series) if config.FORECAST_OUTPUT_INCLUDE_WEATHER else []
            day_weather_summary = reporting.build_weather_day_summary(day_weather) if config.FORECAST_OUTPUT_INCLUDE_WEATHER else {}
            total_day_series = prediction.build_total_series_for_targets(
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
                schedule_boundary_forecast_rows_zeroed += prediction.apply_schedule_boundary_zero_to_series(
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
                day_hours = prediction.build_category_hours_for_targets(
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
                    if prediction.apply_spike_adjustment_to_category_hours(
                        day_hours=day_hours,
                        category_max=category_max,
                        live_total=live_total,
                        now=now,
                        hour_starts=targets,
                    ):
                        spike_adjusted_categories += 1

                if targets_has_boundary:
                    schedule_boundary_forecast_rows_zeroed += prediction.apply_schedule_boundary_zero_to_series(
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

            totals_day = prediction.build_total_series_for_targets(
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
                schedule_boundary_forecast_rows_zeroed += prediction.apply_schedule_boundary_zero_to_series(
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

            crowd_bands_day = prediction.build_threshold_crowd_bands(
                totals_day,
                facility_max_capacity,
                facility_occupancy_thresholds,
            )
            best_windows_day = prediction.build_windows_from_bands(crowd_bands_day, "low", totals_day)
            avoid_windows_day = prediction.build_windows_from_bands(crowd_bands_day, "peak", totals_day)

            day_payload: Dict[str, object] = {
                "dayName": day_date.strftime("%A"),
                "date": day_date.isoformat(),
                "categories": day_categories,
                "totalHours": total_day_hours,
                "avoidWindows": avoid_windows_day,
                "bestWindows": best_windows_day,
                "crowdBands": crowd_bands_day,
            }
            if config.FORECAST_OUTPUT_INCLUDE_WEATHER:
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
        if loc_samples.get(loc_id, 0) >= config.MIN_SAMPLES_PER_LOC and not data.get("is_stale")
    )
    quality["spikeAwareEnabled"] = config.SPIKE_AWARE_ENABLED
    quality["spikeAwareCategoriesAdjusted"] = spike_adjusted_categories
    quality["scheduleBoundaryForecastRowsZeroed"] = schedule_boundary_forecast_rows_zeroed
    quality["drift"] = drift_summary
    quality["driftActions"] = drift_actions_summary
    quality["intervalConformal"] = conformal_summary

    public_metrics, metric_context = reporting.build_public_metrics(
        models_by_key, model_status_by_key, run_metrics_by_key, unit_loc_ids,
    )

    payload = {
        "generatedAt": now.isoformat(),
        "timezone": config.TZ_NAME,
        "forecastDayStartHour": forecast_start_hour,
        "forecastDayEndHour": forecast_end_hour,
        "model": "xgboost",
        "modelInfo": {
            "status": model_summary.get("status"),
            "trainedAt": model_summary.get("trainedAt"),
            "trainRows": model_summary.get("trainRows"),
            "valRows": model_summary.get("valRows"),
            "valMae": public_metrics.mae_people,
            "valRmse": public_metrics.rmse_people,
            "metrics": public_metrics.to_payload(),
            "metricContext": metric_context,
            "byFacility": model_summary.get("byFacility"),
            "byModel": model_summary.get("byModel"),
            "drift": drift_summary,
            "driftActions": drift_actions_summary,
            "intervalConformal": conformal_summary,
            "retrainHours": int(adaptive_controls.get("retrainHours", config.MODEL_RETRAIN_HOURS)),
            "guardrailMaxMaeDegrade": config.MODEL_GUARDRAIL_MAX_MAE_DEGRADE,
            "guardrailMaxHoldoutMaeDegrade": config.MODEL_GUARDRAIL_MAX_HOLDOUT_MAE_DEGRADE,
            "guardrailMaxHoldoutIntervalErrDegrade": config.MODEL_GUARDRAIL_MAX_HOLDOUT_INTERVAL_ERR_DEGRADE,
            "guardrailMaxValIntervalErrDegrade": config.MODEL_GUARDRAIL_MAX_VAL_INTERVAL_ERR_DEGRADE,
            "featureMissingGuardEnabled": config.MODEL_FEATURE_MISSING_GUARD_ENABLED,
            "featureMissingMinRows": config.MODEL_FEATURE_MISSING_MIN_ROWS,
            "featureMissingMaxGlobalRate": config.MODEL_MAX_FEATURE_MISSING_RATE,
            "featureMissingMaxLagRate": config.MODEL_MAX_LAG_MISSING_RATE,
            "featureMissingMaxWeatherRate": config.MODEL_MAX_WEATHER_MISSING_RATE,
            "featureImputationEnabled": True,
            "featureImputationStrategy": "median_by_feature",
            "featureClipEnabled": config.MODEL_FEATURE_CLIP_ENABLED,
            "featureClipLowerQ": config.MODEL_FEATURE_CLIP_LOWER_Q,
            "featureClipUpperQ": config.MODEL_FEATURE_CLIP_UPPER_Q,
            "featureClipMinSpread": config.MODEL_FEATURE_CLIP_MIN_SPREAD,
            "featureAbsMax": config.MODEL_FEATURE_ABS_MAX,
            "minFeatureFiniteRatio": config.MODEL_MIN_FEATURE_FINITE_RATIO,
            "ensembleBlendEnabled": config.MODEL_ENSEMBLE_BLEND_ENABLED,
            "ensembleTargetAwareEnabled": config.MODEL_ENSEMBLE_BLEND_ENABLED,
            "ensembleFeatureQualityAdjustEnabled": config.MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_ENABLED,
            "ensembleFeatureQualityAdjustStrength": config.MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_STRENGTH,
            "ensembleFeatureQualityAdjustExp": config.MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_EXP,
            "ensembleSampleSupportAdjustEnabled": config.MODEL_ENSEMBLE_SAMPLE_SUPPORT_ADJUST_ENABLED,
            "ensembleSampleSupportTarget": config.MODEL_ENSEMBLE_SAMPLE_SUPPORT_TARGET,
            "ensembleSampleSupportMaxShift": config.MODEL_ENSEMBLE_SAMPLE_SUPPORT_MAX_SHIFT,
            "ensembleDisagreementIntervalEnabled": config.MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_ENABLED,
            "ensembleDisagreementIntervalMinDiff": config.MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_MIN_DIFF,
            "ensembleDisagreementIntervalScale": config.MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_SCALE,
            "ensembleDisagreementIntervalMaxMult": config.MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_MAX_MULT,
            "sampleSupportIntervalWidenEnabled": config.MODEL_SAMPLE_SUPPORT_INTERVAL_WIDEN_ENABLED,
            "sampleSupportIntervalTarget": config.MODEL_SAMPLE_SUPPORT_INTERVAL_TARGET,
            "sampleSupportIntervalMaxMult": config.MODEL_SAMPLE_SUPPORT_INTERVAL_MAX_MULT,
            "liveBiasEnabled": config.LIVE_BIAS_ENABLED,
            "lowSampleBlendEnabled": config.MODEL_LOW_SAMPLE_BLEND_ENABLED,
            "missingFeatureBlendEnabled": config.MODEL_MISSING_FEATURE_BLEND_ENABLED,
            "missingFeatureBlendStart": config.MODEL_MISSING_FEATURE_BLEND_START,
            "missingFeatureBlendFull": config.MODEL_MISSING_FEATURE_BLEND_FULL,
            "missingFeatureBlendMaxWeight": config.MODEL_MISSING_FEATURE_BLEND_MAX_WEIGHT,
            "missingFeatureIntervalWidenEnabled": config.MODEL_MISSING_FEATURE_INTERVAL_WIDEN_ENABLED,
            "missingFeatureIntervalWidenStart": config.MODEL_MISSING_FEATURE_INTERVAL_WIDEN_START,
            "missingFeatureIntervalWidenFull": config.MODEL_MISSING_FEATURE_INTERVAL_WIDEN_FULL,
            "missingFeatureIntervalWidenMaxMult": config.MODEL_MISSING_FEATURE_INTERVAL_WIDEN_MAX_MULT,
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
            "longHorizonBlendEnabled": config.MODEL_LONG_HORIZON_BLEND_ENABLED,
            "directHorizonEnabled": config.MODEL_DIRECT_HORIZON_ENABLED,
            "directHorizonHours": features.configured_direct_horizon_hours(),
            "directHorizonMinPairs": config.MODEL_DIRECT_HORIZON_MIN_PAIRS,
            "directHorizonSegmentMinPairs": config.MODEL_DIRECT_HORIZON_SEGMENT_MIN_PAIRS,
            "directHorizonOccupancySegmentsEnabled": config.MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENTS_ENABLED,
            "directHorizonOccupancySegmentMinPairs": config.MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENT_MIN_PAIRS,
            "directHorizonMaxBlend": config.MODEL_DIRECT_HORIZON_MAX_BLEND,
            "pointBiasCorrectionEnabled": config.MODEL_POINT_BIAS_CORRECTION_ENABLED,
            "pointBiasMinPointsPerSegment": config.MODEL_POINT_BIAS_MIN_POINTS_PER_SEGMENT,
            "pointBiasOccupancyEnabled": config.MODEL_POINT_BIAS_OCCUPANCY_ENABLED,
            "pointBiasMinPointsPerOccupancy": config.MODEL_POINT_BIAS_MIN_POINTS_PER_OCCUPANCY,
            "pointBiasSupportTargetMult": config.MODEL_POINT_BIAS_SUPPORT_TARGET_MULT,
            "pointBiasMaxAbs": config.MODEL_POINT_BIAS_MAX_ABS,
            "intervalHourBlockFallbackEnabled": True,
            "intervalDayTypeFallbackEnabled": True,
            "intervalConformalDayTypeFallbackEnabled": True,
            "intervalSegmentBlendTargetMult": config.INTERVAL_SEGMENT_BLEND_TARGET_MULT,
            "intervalConformalSegmentBlendTargetMult": config.INTERVAL_CONFORMAL_SEGMENT_BLEND_TARGET_MULT,
            "tuningIntervalErrWeight": config.MODEL_TUNING_INTERVAL_ERR_WEIGHT,
            "tuningTailErrWeight": config.MODEL_TUNING_TAIL_ERR_WEIGHT,
            "tuningTailQuantile": config.MODEL_TUNING_TAIL_QUANTILE,
            "tuningComplexityWeight": config.MODEL_TUNING_COMPLEXITY_WEIGHT,
            "tuningComplexityDepthRef": config.MODEL_TUNING_COMPLEXITY_DEPTH_REF,
            "lagTrendSensorFeatureCount": config.LAG_TREND_SENSOR_FEATURE_COUNT,
            "locationBalanceWeightEnabled": config.LOCATION_BALANCE_WEIGHT_ENABLED,
            "locationBalanceWeightPower": config.LOCATION_BALANCE_WEIGHT_POWER,
            "locationBalanceWeightMin": config.LOCATION_BALANCE_WEIGHT_MIN,
            "locationBalanceWeightMax": config.LOCATION_BALANCE_WEIGHT_MAX,
            "weightStabilizationEnabled": config.MODEL_WEIGHT_STABILIZATION_ENABLED,
            "weightClipLowerQ": config.MODEL_WEIGHT_CLIP_LOWER_Q,
            "weightClipUpperQ": config.MODEL_WEIGHT_CLIP_UPPER_Q,
            "weightClipMin": config.MODEL_WEIGHT_CLIP_MIN,
            "weightClipMax": config.MODEL_WEIGHT_CLIP_MAX,
            "weightNormalizeMean": config.MODEL_WEIGHT_NORMALIZE_MEAN,
            "featureQualityWeightEnabled": config.MODEL_FEATURE_QUALITY_WEIGHT_ENABLED,
            "featureQualityWeightMin": config.MODEL_FEATURE_QUALITY_WEIGHT_MIN,
            "featureQualityWeightPower": config.MODEL_FEATURE_QUALITY_WEIGHT_POWER,
            "recentDriftBiasEnabled": config.RECENT_DRIFT_BIAS_ENABLED,
            "recentDriftBiasMaxAbs": config.RECENT_DRIFT_BIAS_MAX_ABS,
            "recentDriftBiasMinPointsPerHour": config.RECENT_DRIFT_BIAS_MIN_POINTS_PER_HOUR,
            "recentDriftBiasMinPointsPerOccupancy": config.RECENT_DRIFT_BIAS_MIN_POINTS_PER_OCCUPANCY,
            "recentDriftBiasHorizonDecayHours": config.RECENT_DRIFT_BIAS_HORIZON_DECAY_HOURS,
            "recentDriftBiasBlend": config.RECENT_DRIFT_BIAS_BLEND,
            "recentDriftBiasSupportTargetMult": config.RECENT_DRIFT_BIAS_SUPPORT_TARGET_MULT,
            "schedulePhaseFeatureCount": config.SCHEDULE_PHASE_FEATURE_COUNT,
            "scheduleTransitionWeightEnabled": config.SCHEDULE_TRANSITION_WEIGHT_ENABLED,
            "scheduleTransitionWeightMultiplier": config.SCHEDULE_TRANSITION_WEIGHT_MULTIPLIER,
            "weatherDerivedFeatureCount": config.WEATHER_DERIVED_FEATURE_COUNT,
            "weatherQualityFeatureCount": config.WEATHER_QUALITY_FEATURE_COUNT,
            "outputIncludeWeather": config.FORECAST_OUTPUT_INCLUDE_WEATHER,
            "outputIncludeIntervalFields": config.FORECAST_OUTPUT_INCLUDE_INTERVAL_FIELDS,
            "adaptiveControls": adaptive_controls,
            "spikeAwareEnabled": config.SPIKE_AWARE_ENABLED,
            "spikeAwareHorizonHours": config.SPIKE_AWARE_HORIZON_HOURS,
            "spikeAwareMaxAgeMin": config.SPIKE_AWARE_MAX_AGE_MIN,
        },
        "dataQuality": quality,
        "facilities": facilities_payload,
    }

    return payload


def write_forecast(payload: Dict[str, object]) -> None:
    out_dir = os.path.dirname(config.FORECAST_JSON_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_path = config.FORECAST_JSON_PATH + ".tmp"
    sanitized = reporting.sanitize_for_json(payload)
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(sanitized, handle, ensure_ascii=False, allow_nan=False)
    os.replace(tmp_path, config.FORECAST_JSON_PATH)
    publication.archive_published_forecast(sanitized, config.FORECAST_JSON_PATH)


def main() -> int:
    try:
        validate_production_environment(
            os.environ,
            required_names=(
                "GYM_DB_HOST",
                "GYM_DB_PORT",
                "GYM_DB_USER",
                "GYM_DB_PASSWORD",
                "GYM_DB_NAME",
                "MODEL_ARTIFACT_DIR",
                "MODEL_BASENAME",
                "FORECAST_JSON_PATH",
            ),
            cors_name=None,
            admin_enabled=False,
        )
        payload = build_forecast()
        write_forecast(payload)

        log_event("forecast.completed", generatedFacilities=len(payload.get("facilities", [])))
        return 0
    except EnvironmentConfigurationError as exc:
        log_configuration_failure(exc)
        return 1
    except Exception:
        log_event("forecast.failed")
        return 1


_sys.modules.setdefault("server.reclive.forecasting.job", _sys.modules[__name__])
_sys.modules.setdefault("reclive.forecasting.job", _sys.modules[__name__])
