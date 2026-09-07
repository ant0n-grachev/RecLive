"""Forecasting prediction owner; mechanically transferred definitions."""

import sys as _sys
from server.reclive.forecasting import config, data, features, reporting
import math
import traceback
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import xgboost as xgb









def point_bias_for_target(
    target: datetime,
    profile: Optional[Dict[str, object]],
    hours_ahead: Optional[float] = None,
    point_ratio: Optional[float] = None,
) -> float:
    if not config.MODEL_POINT_BIAS_CORRECTION_ENABLED or not profile:
        return 0.0

    min_points = max(1, int(config.MODEL_POINT_BIAS_MIN_POINTS_PER_SEGMENT))
    min_points_occ = max(1, int(config.MODEL_POINT_BIAS_MIN_POINTS_PER_OCCUPANCY))
    support_mult = max(1.0, float(config.MODEL_POINT_BIAS_SUPPORT_TARGET_MULT))
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
        seg_bias = features.to_float_or_none(stats.get("bias"))
        if seg_bias is None:
            return None
        support_target = max(1.0, float(max(1, int(required_points))) * support_mult)
        support = max(0.0, min(1.0, float(count) / support_target))
        return float(support * float(seg_bias) + (1.0 - support) * float(global_bias))

    if config.MODEL_POINT_BIAS_OCCUPANCY_ENABLED and point_ratio is not None and isinstance(by_occupancy, dict):
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
    if not config.MODEL_POINT_BIAS_CORRECTION_ENABLED:
        return p10_ratio, p50_ratio, p90_ratio
    shift = features.to_float_or_none(bias)
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
    if not config.RECENT_DRIFT_BIAS_ENABLED or not isinstance(profile, dict):
        return 0.0

    min_points = max(1, int(config.RECENT_DRIFT_BIAS_MIN_POINTS_PER_HOUR))
    min_points_occ = max(1, int(config.RECENT_DRIFT_BIAS_MIN_POINTS_PER_OCCUPANCY))
    max_abs = max(0.0, float(config.RECENT_DRIFT_BIAS_MAX_ABS))
    blend = max(0.0, min(1.0, float(config.RECENT_DRIFT_BIAS_BLEND)))
    decay_h = max(0.5, float(config.RECENT_DRIFT_BIAS_HORIZON_DECAY_HOURS))
    support_mult = max(1.0, float(config.RECENT_DRIFT_BIAS_SUPPORT_TARGET_MULT))

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
    features_arr = features.apply_feature_fill_values(X_arr, feature_fill_values)
    features_arr = features.apply_feature_clip_bounds(features_arr, feature_clip_bounds)
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
            p10, p90 = features.ordered_prediction_bounds(p10, p90)

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
            p10, p90 = features.ordered_prediction_bounds(p10, p90)
            p50 = np.minimum(np.maximum(p50, p10), p90)

    out: Dict[str, np.ndarray] = {
        "p50": np.asarray(p50, dtype=np.float32),
    }
    if include_intervals and p10 is not None and p90 is not None:
        out["p10"] = np.asarray(p10, dtype=np.float32)
        out["p90"] = np.asarray(p90, dtype=np.float32)
    return out


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
    recent_days = max(1, int(drift_recent_days if drift_recent_days is not None else config.DRIFT_RECENT_DAYS))
    alert_multiplier = float(
        drift_alert_multiplier if drift_alert_multiplier is not None else config.DRIFT_ALERT_MULTIPLIER
    )
    summary = {
        "enabled": config.DRIFT_ENABLED,
        "recentDays": recent_days,
        "minPoints": max(1, config.DRIFT_MIN_POINTS),
        "alertMultiplier": alert_multiplier,
        "recentBiasEnabled": config.RECENT_DRIFT_BIAS_ENABLED,
        "recentBiasMaxAbs": float(config.RECENT_DRIFT_BIAS_MAX_ABS),
        "recentBiasMinPointsPerHour": max(1, int(config.RECENT_DRIFT_BIAS_MIN_POINTS_PER_HOUR)),
        "recentBiasMinPointsPerOccupancy": max(1, int(config.RECENT_DRIFT_BIAS_MIN_POINTS_PER_OCCUPANCY)),
        "recentBiasHorizonDecayHours": max(0.5, float(config.RECENT_DRIFT_BIAS_HORIZON_DECAY_HOURS)),
        "recentBiasBlend": max(0.0, min(1.0, float(config.RECENT_DRIFT_BIAS_BLEND))),
        "recentBiasSupportTargetMult": max(1.0, float(config.RECENT_DRIFT_BIAS_SUPPORT_TARGET_MULT)),
        "modelsEvaluated": 0,
        "modelsAlerting": 0,
        "byModel": {},
    }
    if not config.DRIFT_ENABLED:
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
        location_balance_map = features.build_location_balance_weight_map_from_counts(
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
        feature_fill_values = features.coerce_feature_fill_values(
            model_meta.get("featureFillValues") if isinstance(model_meta, dict) else None,
            expected_cols=expected_feature_count if expected_feature_count > 0 else None,
        )
        feature_clip_bounds = features.coerce_feature_clip_bounds(
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
        drift_dataset = features.build_model_observation_dataset(
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
        recency_w = features.build_recency_weights(filtered_times)
        occupancy_w = features.build_occupancy_weights(filtered_labels)
        drift_w = features.stabilize_sample_weights(
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

        if points < max(1, config.DRIFT_MIN_POINTS):
            continue

        recent_mae = float(weighted_abs_sum / max(1e-6, weight_sum))
        raw_bias = float(weighted_signed_sum / max(1e-6, weight_sum))
        max_bias_abs = max(0.0, float(config.RECENT_DRIFT_BIAS_MAX_ABS))
        if max_bias_abs > 0.0:
            raw_bias = max(-max_bias_abs, min(max_bias_abs, raw_bias))
        min_points_hour = max(1, int(config.RECENT_DRIFT_BIAS_MIN_POINTS_PER_HOUR))
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
        min_points_occ = max(1, int(config.RECENT_DRIFT_BIAS_MIN_POINTS_PER_OCCUPANCY))
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
    step = max(0.0, float(config.DRIFT_ACTION_INTERVAL_MULTIPLIER_STEP))
    max_mult = max(1.0, float(config.DRIFT_ACTION_INTERVAL_MAX_MULTIPLIER))
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
        int(action_streak_for_retrain if action_streak_for_retrain is not None else config.DRIFT_ACTION_STREAK_FOR_RETRAIN),
    )
    force_hours = max(
        1,
        int(action_force_hours if action_force_hours is not None else config.DRIFT_ACTION_FORCE_HOURS),
    )
    summary = {
        "enabled": config.DRIFT_ACTIONS_ENABLED,
        "triggerStreak": trigger_streak,
        "forceHours": force_hours,
        "intervalStep": max(0.0, config.DRIFT_ACTION_INTERVAL_MULTIPLIER_STEP),
        "intervalMax": max(1.0, config.DRIFT_ACTION_INTERVAL_MAX_MULTIPLIER),
        "modelsEvaluated": 0,
        "modelsForcedRetrain": 0,
        "modelsRolledBack": 0,
        "byModel": {},
    }
    interval_multiplier_by_key: Dict[str, float] = {}
    if not config.DRIFT_ACTIONS_ENABLED:
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
        history = history[-max(10, int(config.ADAPTIVE_HISTORY_MAX_POINTS)) :]

        rolled_back = False
        if config.CHAMPION_ROLLBACK_ENABLED and alert and streak >= max(1, config.CHAMPION_ROLLBACK_DRIFT_STREAK):
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
                restored_meta = data.rollback_to_previous_model(
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
                        data.save_model_meta_only(restored_meta, model_key=model_key)
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
            prev_force_until = features.parse_iso_datetime(str(meta.get("forceRetrainUntil", "")))
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
            data.save_model_meta_only(updated, model_key=model_key)
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
    alpha = max(0.01, min(0.5, float(config.INTERVAL_CONFORMAL_ALPHA)))
    quantile = max(0.0, min(1.0, 1.0 - alpha))
    margin = float(features.weighted_quantile(scores, quantile, weights))
    return max(0.0, min(float(config.INTERVAL_CONFORMAL_MAX_MARGIN), margin))


def forecast_horizon_hours(
    target: datetime,
    reference: Optional[datetime] = None,
    hours_ahead: Optional[float] = None,
) -> int:
    if hours_ahead is not None:
        parsed = features.to_float_or_none(hours_ahead)
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

    min_points = max(1, int(config.INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR))
    global_stats = profile.get("global", {})
    global_margin = (
        max(0.0, float(global_stats.get("margin", 0.0) or 0.0))
        if isinstance(global_stats, dict)
        else 0.0
    )
    support_target = max(
        float(min_points),
        float(min_points) * max(1.0, float(config.INTERVAL_CONFORMAL_SEGMENT_BLEND_TARGET_MULT)),
    )

    def blended_margin(stats: Optional[Dict[str, object]]) -> Optional[float]:
        if not isinstance(stats, dict):
            return None
        count = int(stats.get("count", 0) or 0)
        margin = features.to_float_or_none(stats.get("margin"))
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
        "enabled": config.INTERVAL_CONFORMAL_ENABLED,
        "recentDays": max(1, config.INTERVAL_CONFORMAL_RECENT_DAYS),
        "alpha": max(0.01, min(0.5, float(config.INTERVAL_CONFORMAL_ALPHA))),
        "minPoints": max(1, config.INTERVAL_CONFORMAL_MIN_POINTS),
        "minPointsPerHour": max(1, config.INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR),
        "maxMargin": float(config.INTERVAL_CONFORMAL_MAX_MARGIN),
        "segmentBlendTargetMult": max(1.0, float(config.INTERVAL_CONFORMAL_SEGMENT_BLEND_TARGET_MULT)),
        "modelsCalibrated": 0,
        "byModel": {},
    }
    if not config.INTERVAL_CONFORMAL_ENABLED:
        return {}, summary

    since = now - timedelta(days=max(1, config.INTERVAL_CONFORMAL_RECENT_DAYS))
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
        feature_fill_values = features.coerce_feature_fill_values(
            model_meta.get("featureFillValues") if isinstance(model_meta, dict) else None,
            expected_cols=expected_feature_count if expected_feature_count > 0 else None,
        )
        feature_clip_bounds = features.coerce_feature_clip_bounds(
            model_meta.get("featureClipLower") if isinstance(model_meta, dict) else None,
            model_meta.get("featureClipUpper") if isinstance(model_meta, dict) else None,
            expected_cols=expected_feature_count if expected_feature_count > 0 else None,
        )

        loc_ids = unit_loc_ids.get(model_key) or []
        if not loc_ids:
            continue
        location_balance_map = features.build_location_balance_weight_map_from_counts(
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
        conformal_dataset = features.build_model_observation_dataset(
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
        recency_w = features.build_recency_weights(filtered_times)
        occupancy_w = features.build_occupancy_weights(labels_arr)
        score_weights = features.stabilize_sample_weights(recency_w * occupancy_w * filtered_quality)
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

        if points < max(1, config.INTERVAL_CONFORMAL_MIN_POINTS) or not global_scores:
            continue

        global_margin = conformal_margin_from_scores(
            np.array(global_scores, dtype=np.float32),
            np.array(global_weights, dtype=np.float32),
        )
        by_hour_payload: Dict[str, Dict[str, object]] = {}
        for hour, scores in sorted(by_hour_scores.items()):
            if len(scores) < config.INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR:
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
            if len(scores) < config.INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR:
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
            if len(scores) < config.INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR:
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
            if len(scores) < config.INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR:
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
            if len(scores) < config.INTERVAL_CONFORMAL_MIN_POINTS_PER_HOUR:
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

    min_samples = max(1, int(config.INTERVAL_MIN_SAMPLES_PER_HOUR))
    blend_target = max(float(min_samples), float(min_samples) * max(1.0, float(config.INTERVAL_SEGMENT_BLEND_TARGET_MULT)))
    global_stats = residual_profile.get("global", {}) if isinstance(residual_profile, dict) else {}
    global_q10 = float(global_stats.get("q10", 0.0) or 0.0) if isinstance(global_stats, dict) else 0.0
    global_q90 = float(global_stats.get("q90", 0.0) or 0.0) if isinstance(global_stats, dict) else 0.0

    def blended_quantiles(stats: Optional[Dict[str, object]]) -> Optional[Tuple[float, float]]:
        if not isinstance(stats, dict):
            return None
        count = int(stats.get("count", 0) or 0)
        if count < min_samples:
            return None
        q10 = features.to_float_or_none(stats.get("q10"))
        q90 = features.to_float_or_none(stats.get("q90"))
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
    if not config.MODEL_DIRECT_HORIZON_ENABLED:
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
                profile.get("minPairs", config.MODEL_DIRECT_HORIZON_MIN_PAIRS),
            )
            or config.MODEL_DIRECT_HORIZON_MIN_PAIRS
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
        if config.MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENTS_ENABLED and point_ratio is not None:
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
        slope = features.to_float_or_none(stats_source.get("slope"))
        intercept = features.to_float_or_none(stats_source.get("intercept"))
        blend = features.to_float_or_none(stats_source.get("blend"))
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
        if config.MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENTS_ENABLED and point_ratio is not None:
            occ_key = occupancy_bucket_key_from_ratio(float(point_ratio))
            by_occ = global_stats.get("byOccupancy", {})
            if isinstance(by_occ, dict):
                occ_stats = by_occ.get(occ_key)
                if isinstance(occ_stats, dict) and int(occ_stats.get("count", 0) or 0) >= occupancy_min_pairs:
                    stats_source = occ_stats
        slope = features.to_float_or_none(stats_source.get("slope"))
        intercept = features.to_float_or_none(stats_source.get("intercept"))
        blend = features.to_float_or_none(stats_source.get("blend"))
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
    if not config.MODEL_DIRECT_HORIZON_ENABLED:
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
            float(config.MODEL_DIRECT_HORIZON_MAX_BLEND),
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
    numeric = features.to_float_or_none(ratio)
    if numeric is None:
        return

    lag_ratio_override[target] = float(numeric)
    step = max(1, int(config.RESAMPLE_MINUTES))
    for offset in range(step, 60, step):
        lag_ratio_override[target + timedelta(minutes=offset)] = float(numeric)


def preferred_model_error_and_rows(meta: Optional[Dict[str, object]]) -> Tuple[Optional[float], int]:
    if not isinstance(meta, dict):
        return None, 0

    holdout_mae = features.to_float_or_none(meta.get("holdoutMae"))
    holdout_rows = int(meta.get("holdoutRows", 0) or 0)
    if holdout_mae is not None and holdout_rows > 0:
        return float(holdout_mae), int(holdout_rows)

    val_mae = features.to_float_or_none(meta.get("valMae"))
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
        holdout_int_err = features.to_float_or_none(meta.get("holdoutIntervalCoverageError"))
        if holdout_int_err is None:
            holdout_int_err = features.to_float_or_none(meta.get("valIntervalCoverageError"))
        if holdout_int_err is not None:
            interval_penalty = 1.0 / (1.0 + 8.0 * max(0.0, float(holdout_int_err)))
            base_score *= interval_penalty

        missing_summary = meta.get("featureMissingness")
        if isinstance(missing_summary, dict):
            missing_rate = features.to_float_or_none(missing_summary.get("globalMissingRate"))
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
    min_w = max(0.0, min(1.0, float(config.MODEL_ENSEMBLE_MIN_PRIMARY_WEIGHT)))
    max_w = max(min_w, min(1.0, float(config.MODEL_ENSEMBLE_MAX_PRIMARY_WEIGHT)))
    default_w = max(min_w, min(max_w, float(config.MODEL_ENSEMBLE_DEFAULT_PRIMARY_WEIGHT)))

    if not config.MODEL_ENSEMBLE_BLEND_ENABLED:
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
    min_points = max(1, int(config.MODEL_POINT_BIAS_MIN_POINTS_PER_SEGMENT))

    by_horizon = profile.get("byHorizon", {})
    horizon_hours = forecast_horizon_hours(target, hours_ahead=hours_ahead)
    horizon_stats = (
        by_horizon.get(horizon_bucket_key(horizon_hours))
        if isinstance(by_horizon, dict)
        else None
    )
    if isinstance(horizon_stats, dict) and int(horizon_stats.get("count", 0) or 0) >= min_points:
        mae = features.to_float_or_none(horizon_stats.get("mae"))
        if mae is not None and mae > 0:
            return float(mae)

    by_hour_block = profile.get("byHourBlock", {})
    block_stats = by_hour_block.get(hour_block_key(int(target.hour))) if isinstance(by_hour_block, dict) else None
    if isinstance(block_stats, dict) and int(block_stats.get("count", 0) or 0) >= min_points:
        mae = features.to_float_or_none(block_stats.get("mae"))
        if mae is not None and mae > 0:
            return float(mae)

    by_day_type = profile.get("byDayType", {})
    day_key = "weekend" if int(target.weekday()) >= 5 else "weekday"
    day_stats = by_day_type.get(day_key) if isinstance(by_day_type, dict) else None
    if isinstance(day_stats, dict) and int(day_stats.get("count", 0) or 0) >= min_points:
        mae = features.to_float_or_none(day_stats.get("mae"))
        if mae is not None and mae > 0:
            return float(mae)

    global_stats = profile.get("global", {})
    mae = features.to_float_or_none(global_stats.get("mae")) if isinstance(global_stats, dict) else None
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
    min_w = max(0.0, min(1.0, float(config.MODEL_ENSEMBLE_MIN_PRIMARY_WEIGHT)))
    max_w = max(min_w, min(1.0, float(config.MODEL_ENSEMBLE_MAX_PRIMARY_WEIGHT)))

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
        cached_weight = features.to_float_or_none(cache.get(cache_key))
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
        cached_bias = features.to_float_or_none(cache.get(cache_key))
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
        cached_bias = features.to_float_or_none(cache.get(cache_key))
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
        cached_margin = features.to_float_or_none(cache.get(cache_key))
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
    min_w = max(0.0, min(1.0, float(config.MODEL_ENSEMBLE_MIN_PRIMARY_WEIGHT)))
    max_w = max(min_w, min(1.0, float(config.MODEL_ENSEMBLE_MAX_PRIMARY_WEIGHT)))
    base = max(min_w, min(max_w, float(primary_weight)))
    if not config.MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_ENABLED:
        return float(base)
    if primary_missing_ratio is None or fallback_missing_ratio is None:
        return float(base)

    mp = max(0.0, min(1.0, float(primary_missing_ratio)))
    mf = max(0.0, min(1.0, float(fallback_missing_ratio)))
    exp = max(0.5, min(4.0, float(config.MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_EXP)))
    strength = max(0.0, min(1.0, float(config.MODEL_ENSEMBLE_FEATURE_QUALITY_ADJUST_STRENGTH)))
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
    min_w = max(0.0, min(1.0, float(config.MODEL_ENSEMBLE_MIN_PRIMARY_WEIGHT)))
    max_w = max(min_w, min(1.0, float(config.MODEL_ENSEMBLE_MAX_PRIMARY_WEIGHT)))
    base = max(min_w, min(max_w, float(primary_weight)))
    if not config.MODEL_ENSEMBLE_SAMPLE_SUPPORT_ADJUST_ENABLED:
        return float(base)

    target = max(1, int(config.MODEL_ENSEMBLE_SAMPLE_SUPPORT_TARGET))
    if int(sample_count) >= target:
        return float(base)
    max_shift = max(0.0, min(1.0, float(config.MODEL_ENSEMBLE_SAMPLE_SUPPORT_MAX_SHIFT)))
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
    if not config.LIVE_BIAS_ENABLED:
        return p10_ratio, p50_ratio, p90_ratio
    if live_ratio is None or age_min is None:
        return p10_ratio, p50_ratio, p90_ratio
    if age_min > max(0.0, float(config.LIVE_BIAS_MAX_AGE_MIN)):
        return p10_ratio, p50_ratio, p90_ratio
    if hours_ahead < 0.0 or hours_ahead > max(0.0, float(config.LIVE_BIAS_MAX_HORIZON_HOURS)):
        return p10_ratio, p50_ratio, p90_ratio

    base = max(0.0, min(1.0, float(config.LIVE_BIAS_BASE_WEIGHT)))
    horizon_decay = max(0.05, float(config.LIVE_BIAS_HORIZON_DECAY))
    age_decay = max(1.0, float(config.LIVE_BIAS_AGE_DECAY_MIN))
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
    if not config.MODEL_LOW_SAMPLE_BLEND_ENABLED:
        return p10_ratio, p50_ratio, p90_ratio

    target_count = max(1, int(config.MODEL_LOW_SAMPLE_TARGET_COUNT))
    max_blend = max(0.0, min(1.0, float(config.MODEL_LOW_SAMPLE_MAX_BLEND)))
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
    if not config.MODEL_MISSING_FEATURE_BLEND_ENABLED:
        return p10_ratio, p50_ratio, p90_ratio
    if missing_ratio is None:
        return p10_ratio, p50_ratio, p90_ratio

    mr = max(0.0, min(1.0, float(missing_ratio)))
    start = max(0.0, min(1.0, float(config.MODEL_MISSING_FEATURE_BLEND_START)))
    full = max(start + 0.01, min(1.0, float(config.MODEL_MISSING_FEATURE_BLEND_FULL)))
    max_w = max(0.0, min(1.0, float(config.MODEL_MISSING_FEATURE_BLEND_MAX_WEIGHT)))
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
    if not config.MODEL_MISSING_FEATURE_INTERVAL_WIDEN_ENABLED:
        return 1.0
    if missing_ratio is None:
        return 1.0

    mr = max(0.0, min(1.0, float(missing_ratio)))
    start = max(0.0, min(1.0, float(config.MODEL_MISSING_FEATURE_INTERVAL_WIDEN_START)))
    full = max(start + 0.01, min(1.0, float(config.MODEL_MISSING_FEATURE_INTERVAL_WIDEN_FULL)))
    max_mult = max(1.0, float(config.MODEL_MISSING_FEATURE_INTERVAL_WIDEN_MAX_MULT))
    if mr <= start or max_mult <= 1.0:
        return 1.0

    progress = max(0.0, min(1.0, (mr - start) / max(0.01, (full - start))))
    return float(1.0 + (max_mult - 1.0) * progress)


def ensemble_disagreement_interval_multiplier(disagreement: Optional[float]) -> float:
    if not config.MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_ENABLED:
        return 1.0
    if disagreement is None:
        return 1.0

    diff = max(0.0, float(disagreement))
    min_diff = max(0.0, float(config.MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_MIN_DIFF))
    scale = max(0.0, float(config.MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_SCALE))
    max_mult = max(1.0, float(config.MODEL_ENSEMBLE_DISAGREEMENT_INTERVAL_MAX_MULT))
    if diff <= min_diff or max_mult <= 1.0 or scale <= 0.0:
        return 1.0

    raw = 1.0 + (diff - min_diff) * scale
    return float(max(1.0, min(max_mult, raw)))


def sample_support_interval_multiplier(sample_count: int) -> float:
    if not config.MODEL_SAMPLE_SUPPORT_INTERVAL_WIDEN_ENABLED:
        return 1.0

    target = max(1, int(config.MODEL_SAMPLE_SUPPORT_INTERVAL_TARGET))
    max_mult = max(1.0, float(config.MODEL_SAMPLE_SUPPORT_INTERVAL_MAX_MULT))
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
    if not config.MODEL_LONG_HORIZON_BLEND_ENABLED:
        return p10_ratio, p50_ratio, p90_ratio

    start_h = max(0.0, float(config.MODEL_LONG_HORIZON_BLEND_START_HOURS))
    full_h = max(start_h + 0.5, float(config.MODEL_LONG_HORIZON_BLEND_FULL_HOURS))
    max_w = max(0.0, min(1.0, float(config.MODEL_LONG_HORIZON_BLEND_MAX_WEIGHT)))
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

    if config.SCHEDULE_BOUNDARY_ZERO_ENABLED:
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
                        schedule_boundary_cache[schedule_cache_key] = features.get_facility_schedule_boundary_state(
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
                cached_p10 = features.to_float_or_none(cached_model_pred[0])
                cached_p50 = features.to_float_or_none(cached_model_pred[1])
                cached_p90 = features.to_float_or_none(cached_model_pred[2])
                cached_missing = features.to_float_or_none(cached_model_pred[3])
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
            or loc_samples.get(loc_id_int, 0) < config.MIN_SAMPLES_PER_LOC
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
                    missing_ratio = features.to_float_or_none(cached_feature[1])
            else:
                feature_vec = cached_feature
        if feature_vec is None:
            feature_vec = features.build_features(
                target,
                loc_entry,
                onehot_vec,
                weather_source=weather_source,
                weather_lookup_cache=weather_lookup_cache,
                lag_ratio_override=lag_ratio_override,
            )
        if missing_ratio is None and isinstance(feature_vec, list):
            missing_ratio = features.feature_missing_rate(feature_vec)
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
            features_arr = features.sanitize_feature_matrix(np.array([feature_vec], dtype=np.float32))
            if isinstance(feature_fill_values_by_key, dict):
                features_arr = features.apply_feature_fill_values(
                    features_arr,
                    feature_fill_values_by_key.get(model_key),
                )
            if isinstance(feature_clip_bounds_by_key, dict):
                features_arr = features.apply_feature_clip_bounds(
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
            out_sample_count = features.to_float_or_none(cached_sample_count)
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
        and (config.MODEL_ENSEMBLE_BLEND_ENABLED or primary_pred is None)
    )
    if need_fallback_pred:
        fallback_pred = predict_for_model(fallback_key, fallback_override)

    if config.MODEL_DIRECT_HORIZON_ENABLED and isinstance(model_meta_by_key, dict):
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
    if primary_pred is not None and fallback_pred is not None and config.MODEL_ENSEMBLE_BLEND_ENABLED:
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
                    cached_p50 = features.to_float_or_none(cached_pred[1])
                    if cached_p50 is not None:
                        seed_recursive_ratio_overrides(override_map, target, float(cached_p50))
                    continue

                loc_entry = loc_data.get(int(loc_id)) if isinstance(loc_data, dict) else None
                if (
                    not isinstance(loc_entry, dict)
                    or int(max_caps.get(int(loc_id), 0) or 0) <= 0
                    or int(loc_samples.get(int(loc_id), 0) or 0) < config.MIN_SAMPLES_PER_LOC
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
                            missing_ratio = features.to_float_or_none(cached_feature[1])
                    else:
                        feature_vec = cached_feature
                if feature_vec is None:
                    feature_vec = features.build_features(
                        target,
                        loc_entry,
                        onehot_vec,
                        weather_source=weather_source,
                        weather_lookup_cache=weather_lookup_cache,
                        lag_ratio_override=override_map,
                    )
                if missing_ratio is None and isinstance(feature_vec, list):
                    missing_ratio = features.feature_missing_rate(feature_vec)
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

            features_arr = features.sanitize_feature_matrix(np.array(feature_rows, dtype=np.float32))
            if isinstance(feature_fill_values_by_key, dict):
                features_arr = features.apply_feature_fill_values(
                    features_arr,
                    feature_fill_values_by_key.get(model_key),
                )
            if isinstance(feature_clip_bounds_by_key, dict):
                features_arr = features.apply_feature_clip_bounds(
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
    if config.FORECAST_OUTPUT_INCLUDE_INTERVAL_FIELDS:
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
    if age_min > config.SPIKE_AWARE_MAX_AGE_MIN:
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
    return max(float(category_max), float(category_max) * config.SPIKE_AWARE_MAX_CAP_MULTIPLIER)


def spike_weight(hours_ahead: int) -> float:
    if hours_ahead <= 0:
        return 1.0
    return math.exp(-config.SPIKE_AWARE_DECAY * max(0, hours_ahead - 1))


def apply_spike_adjustment_to_category_hours(
    day_hours: List[Dict[str, object]],
    category_max: int,
    live_total: float,
    now: datetime,
    hour_starts: Optional[List[datetime]] = None,
) -> bool:
    if not config.SPIKE_AWARE_ENABLED or not day_hours:
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
        if hours_ahead > config.SPIKE_AWARE_HORIZON_HOURS:
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
            p10[t_idx, l_idx] = float(features.to_float_or_none(result.get("countP10")) or 0.0)
            p50[t_idx, l_idx] = float(features.to_float_or_none(result.get("countP50")) or 0.0)
            p90[t_idx, l_idx] = float(features.to_float_or_none(result.get("countP90")) or 0.0)
            samples[t_idx, l_idx] = int(features.to_float_or_none(result.get("sampleCount")) or 0.0)

    return normalized_loc_ids, normalized_targets, target_index_lookup, (
        p10,
        p50,
        p90,
        samples,
    )


def normalized_forecast_hour_bounds() -> Tuple[int, int]:
    start_hour = max(0, min(23, config.FORECAST_DAY_START_HOUR))
    end_hour = max(0, min(23, config.FORECAST_DAY_END_HOUR))
    if end_hour < start_hour:
        start_hour, end_hour = end_hour, start_hour
    return start_hour, end_hour


def get_targets_for_date(day_date: date) -> List[datetime]:
    start_hour, end_hour = normalized_forecast_hour_bounds()

    start_dt = config.TZ.localize(datetime(day_date.year, day_date.month, day_date.day, start_hour, 0, 0))
    end_exclusive = config.TZ.localize(
        datetime(day_date.year, day_date.month, day_date.day, end_hour, 0, 0)
    ) + timedelta(hours=1)

    step = timedelta(minutes=max(1, config.RESAMPLE_MINUTES))
    targets = []
    current = start_dt
    while current < end_exclusive:
        targets.append(current)
        current += step
    return targets


def get_window_targets_for_date(day_date: date) -> List[datetime]:
    start_hour, end_hour = normalized_forecast_hour_bounds()

    start_dt = config.TZ.localize(datetime(day_date.year, day_date.month, day_date.day, start_hour, 0, 0))
    end_exclusive = config.TZ.localize(
        datetime(day_date.year, day_date.month, day_date.day, end_hour, 0, 0)
    ) + timedelta(hours=1)

    step = timedelta(minutes=max(1, config.WINDOW_RESAMPLE_MINUTES))
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
            schedule_eval_cache[cache_key] = features.get_facility_schedule_open_state(
                sections=sections,
                ts=target,
                date_range_cache=date_range_cache,
                weekday_cache=weekday_cache,
                hours_window_cache=hours_window_cache,
            )

        open_state = schedule_eval_cache.get(cache_key)
        open_exact = False
        close_exact = False
        if config.SCHEDULE_BOUNDARY_ZERO_ENABLED:
            if cache_key not in schedule_boundary_cache:
                schedule_boundary_cache[cache_key] = features.get_facility_schedule_boundary_state(
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
        if config.SCHEDULE_BOUNDARY_ZERO_ENABLED:
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
    if not config.SCHEDULE_BOUNDARY_ZERO_ENABLED:
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
                schedule_boundary_cache[cache_key] = features.get_facility_schedule_boundary_state(
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


def build_weather_hours_for_targets(
    targets: List[datetime],
    weather_series: Dict[str, object],
) -> List[Dict[str, object]]:
    times = weather_series.get("times", [])
    weather_map = weather_series.get("map", {})
    rows = []

    for target in targets:
        def value(key: str):
            val = features.weather_value_at_or_before(times, weather_map, target, key)
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
        return max(1, config.RESAMPLE_MINUTES)

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
        return max(1, config.RESAMPLE_MINUTES)

    diffs = []
    for idx in range(len(times) - 1):
        diff_min = int(round((times[idx + 1] - times[idx]).total_seconds() / 60.0))
        if diff_min > 0:
            diffs.append(diff_min)

    if not diffs:
        return max(1, config.RESAMPLE_MINUTES)
    return max(1, min(diffs))


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
            if require_coverage and coverage < config.CROWD_BASELINE_MIN_COVERAGE:
                continue

            adjusted_total = float(total)
            if observed_cap < facility_max_cap:
                adjusted_total = adjusted_total * (facility_max_cap / float(observed_cap))
            scaled.append(max(0.0, adjusted_total))
        return scaled

    baseline_values = collect_scaled_totals(require_coverage=True)
    required_points = max(1, config.CROWD_BASELINE_MIN_POINTS)
    if len(baseline_values) < required_points:
        return None
    if not baseline_values:
        return None

    low_q = max(0.0, min(1.0, config.CROWD_BASELINE_LOW_QUANTILE))
    peak_q = max(0.0, min(1.0, config.CROWD_BASELINE_PEAK_QUANTILE))
    if peak_q < low_q:
        low_q, peak_q = peak_q, low_q

    mean, std = reporting.mean_std(baseline_values)
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

    bridge_gap = timedelta(minutes=config.CROWD_BAND_BRIDGE_MIN)
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

    low_max = features.to_float_or_none(occupancy_thresholds.get("lowMax"))
    peak_min = features.to_float_or_none(occupancy_thresholds.get("peakMin"))
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

        total_sum, avg, min_samples, points = reporting.compute_range_metrics(series_entries, start, end)
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


_sys.modules.setdefault("server.reclive.forecasting.prediction", _sys.modules[__name__])
_sys.modules.setdefault("reclive.forecasting.prediction", _sys.modules[__name__])
