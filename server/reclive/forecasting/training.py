"""Forecasting training owner; mechanically transferred definitions."""

import sys as _sys
from server.reclive.forecasting import config, data, features, metrics, prediction
import math
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import xgboost as xgb









def derive_adaptive_runtime_controls(
    meta_snapshots: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    controls = {
        "enabled": config.ADAPTIVE_CONTROLS_ENABLED,
        "mode": "default",
        "retrainHours": max(1, int(config.MODEL_RETRAIN_HOURS)),
        "driftRecentDays": max(1, int(config.DRIFT_RECENT_DAYS)),
        "driftAlertMultiplier": float(config.DRIFT_ALERT_MULTIPLIER),
        "driftActionStreakForRetrain": max(1, int(config.DRIFT_ACTION_STREAK_FOR_RETRAIN)),
        "samples": 0,
        "alertRate": None,
        "medianMaeRatio": None,
    }
    if not config.ADAPTIVE_CONTROLS_ENABLED:
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

    history_rows = history_rows[-max(10, config.ADAPTIVE_HISTORY_MAX_POINTS) :]
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

    retrain_hours = max(1, int(config.MODEL_RETRAIN_HOURS))
    drift_days = max(1, int(config.DRIFT_RECENT_DAYS))
    drift_mult = float(config.DRIFT_ALERT_MULTIPLIER)
    action_streak = max(1, int(config.DRIFT_ACTION_STREAK_FOR_RETRAIN))

    stable = alert_rate <= config.ADAPTIVE_ALERT_RATE_STABLE_MAX and (
        median_ratio is None or median_ratio <= 1.03
    )
    unstable = alert_rate >= config.ADAPTIVE_ALERT_RATE_UNSTABLE_MIN or (
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

    retrain_hours = max(int(config.ADAPTIVE_RETRAIN_MIN_HOURS), min(int(config.ADAPTIVE_RETRAIN_MAX_HOURS), retrain_hours))
    drift_days = max(int(config.ADAPTIVE_DRIFT_DAYS_MIN), min(int(config.ADAPTIVE_DRIFT_DAYS_MAX), drift_days))
    drift_mult = max(float(config.ADAPTIVE_DRIFT_MULT_MIN), min(float(config.ADAPTIVE_DRIFT_MULT_MAX), drift_mult))
    action_streak = max(1, min(6, int(action_streak)))

    controls["retrainHours"] = retrain_hours
    controls["driftRecentDays"] = drift_days
    controls["driftAlertMultiplier"] = round(drift_mult, 4)
    controls["driftActionStreakForRetrain"] = int(action_streak)
    return controls


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
            "q10": float(features.weighted_quantile(residuals, config.INTERVAL_Q_LOW, residual_weights)),
            "q90": float(features.weighted_quantile(residuals, config.INTERVAL_Q_HIGH, residual_weights)),
            "count": int(len(residuals)),
        },
        "byHour": {},
        "byHourBlock": {},
        "byDayType": {},
    }

    for hour in range(24):
        idx = [i for i, ts in enumerate(filtered_times) if ts.hour == hour]
        if len(idx) < config.INTERVAL_MIN_SAMPLES_PER_HOUR:
            continue
        hour_res = residuals[idx]
        hour_w = residual_weights[idx]
        profile["byHour"][str(hour)] = {
            "q10": float(features.weighted_quantile(hour_res, config.INTERVAL_Q_LOW, hour_w)),
            "q90": float(features.weighted_quantile(hour_res, config.INTERVAL_Q_HIGH, hour_w)),
            "count": int(len(hour_res)),
        }

    by_hour_block = {}
    for block_key in ("overnight", "morning", "midday", "evening", "late"):
        idx = [
            i
            for i, ts in enumerate(filtered_times)
            if prediction.hour_block_key(int(ts.hour)) == block_key
        ]
        if len(idx) < config.INTERVAL_MIN_SAMPLES_PER_HOUR:
            continue
        block_res = residuals[idx]
        block_w = residual_weights[idx]
        by_hour_block[block_key] = {
            "q10": float(features.weighted_quantile(block_res, config.INTERVAL_Q_LOW, block_w)),
            "q90": float(features.weighted_quantile(block_res, config.INTERVAL_Q_HIGH, block_w)),
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
        if len(idx) < config.INTERVAL_MIN_SAMPLES_PER_HOUR:
            continue
        day_res = residuals[idx]
        day_w = residual_weights[idx]
        by_day_type[day_key] = {
            "q10": float(features.weighted_quantile(day_res, config.INTERVAL_Q_LOW, day_w)),
            "q90": float(features.weighted_quantile(day_res, config.INTERVAL_Q_HIGH, day_w)),
            "count": int(len(day_res)),
        }
    profile["byDayType"] = by_day_type

    by_occupancy = {}
    for occ_key in ("low", "mid", "high"):
        idx = [
            i
            for i, ratio in enumerate(point_ratios.tolist())
            if prediction.occupancy_bucket_key_from_ratio(float(ratio)) == occ_key
        ]
        if len(idx) < config.INTERVAL_MIN_SAMPLES_PER_HOUR:
            continue
        occ_res = residuals[idx]
        occ_w = residual_weights[idx]
        by_occupancy[occ_key] = {
            "q10": float(features.weighted_quantile(occ_res, config.INTERVAL_Q_LOW, occ_w)),
            "q90": float(features.weighted_quantile(occ_res, config.INTERVAL_Q_HIGH, occ_w)),
            "count": int(len(occ_res)),
        }
    profile["byOccupancy"] = by_occupancy

    return profile


def build_xgb_params(overrides: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    params: Dict[str, object] = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": max(1, config.MODEL_MAX_DEPTH),
        "eta": max(0.001, float(config.MODEL_ETA)),
        "subsample": max(0.1, min(1.0, float(config.MODEL_SUBSAMPLE))),
        "colsample_bytree": max(0.1, min(1.0, float(config.MODEL_COLSAMPLE_BYTREE))),
        "min_child_weight": max(0.0, float(config.MODEL_MIN_CHILD_WEIGHT)),
        "gamma": max(0.0, float(config.MODEL_GAMMA)),
        "lambda": max(0.0, float(config.MODEL_REG_LAMBDA)),
        "alpha": max(0.0, float(config.MODEL_REG_ALPHA)),
        "verbosity": 0,
        "nthread": max(1, config.MODEL_NTHREAD),
        "seed": int(config.MODEL_TUNING_RANDOM_SEED),
    }
    if config.MODEL_TREE_METHOD:
        params["tree_method"] = config.MODEL_TREE_METHOD
        if config.MODEL_TREE_METHOD in {"hist", "approx"}:
            params["max_bin"] = max(64, config.MODEL_MAX_BIN)
    if overrides:
        params.update(overrides)
    return params


def build_point_bias_profile(
    times: List[datetime],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray],
) -> Optional[Dict[str, object]]:
    if not config.MODEL_POINT_BIAS_CORRECTION_ENABLED:
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
        bias = features.weighted_average(arr, warr)
        max_abs = max(0.0, float(config.MODEL_POINT_BIAS_MAX_ABS))
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
        block_key = prediction.hour_block_key(int(ts.hour))
        by_hour_block_values.setdefault(block_key, []).append(residual)
        by_hour_block_weights.setdefault(block_key, []).append(w)

        horizon_key = prediction.horizon_bucket_key(0)
        by_horizon_values.setdefault(horizon_key, []).append(residual)
        by_horizon_weights.setdefault(horizon_key, []).append(w)

        day_type = "weekend" if int(ts.weekday()) >= 5 else "weekday"
        by_day_type_values[day_type].append(residual)
        by_day_type_weights[day_type].append(w)
        occ_key = prediction.occupancy_bucket_key_from_ratio(ratio)
        by_occupancy_values.setdefault(occ_key, []).append(residual)
        by_occupancy_weights.setdefault(occ_key, []).append(w)

    min_points = max(1, int(config.MODEL_POINT_BIAS_MIN_POINTS_PER_SEGMENT))
    min_points_occ = max(1, int(config.MODEL_POINT_BIAS_MIN_POINTS_PER_OCCUPANCY))
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
            "mae": float(features.weighted_average(np.array(vals, dtype=np.float32), np.array(ws, dtype=np.float32))),
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
        block_key = prediction.hour_block_key(int(ts.hour))
        by_hour_block_values.setdefault(block_key, []).append(err)
        by_hour_block_weights.setdefault(block_key, []).append(w)

        horizon_key = prediction.horizon_bucket_key(0)
        by_horizon_values.setdefault(horizon_key, []).append(err)
        by_horizon_weights.setdefault(horizon_key, []).append(w)

        day_type = "weekend" if int(ts.weekday()) >= 5 else "weekday"
        by_day_type_values[day_type].append(err)
        by_day_type_weights[day_type].append(w)

    min_points = max(1, int(config.MODEL_POINT_BIAS_MIN_POINTS_PER_SEGMENT))
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
    w = features.stabilize_sample_weights(w)
    n = int(x.size)
    if n <= 0:
        return None

    x_mean = features.weighted_average(x, w)
    y_mean = features.weighted_average(y, w)
    cov = features.weighted_average((x - x_mean) * (y - y_mean), w)
    var = features.weighted_average((x - x_mean) ** 2, w)

    if var <= 1e-6:
        slope = 1.0
    else:
        slope = cov / var
    slope = max(0.0, min(2.5, float(slope)))

    intercept = float(y_mean) - float(slope) * float(x_mean)
    intercept = max(-0.5, min(0.5, float(intercept)))

    baseline = np.clip(x, 0.0, 1.2)
    adjusted = np.clip(slope * x + intercept, 0.0, 1.2)
    baseline_mae = features.weighted_average(np.abs(baseline - y), w)
    adjusted_mae = features.weighted_average(np.abs(adjusted - y), w)
    improvement = max(0.0, float(baseline_mae) - float(adjusted_mae))

    min_pairs = max(1, int(config.MODEL_DIRECT_HORIZON_MIN_PAIRS))
    support = max(0.0, min(1.0, float(n) / float(max(1, min_pairs * 3))))
    if baseline_mae > 1e-4 and improvement > 0.0:
        quality = max(0.0, min(1.0, float(improvement) / float(baseline_mae)))
    else:
        quality = 0.0

    max_blend = max(0.0, min(1.0, float(config.MODEL_DIRECT_HORIZON_MAX_BLEND)))
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
    if not config.MODEL_DIRECT_HORIZON_ENABLED:
        return None
    if not pairs_by_hours:
        return None

    min_pairs = max(1, int(config.MODEL_DIRECT_HORIZON_MIN_PAIRS))
    segment_min_pairs = max(
        10,
        min(
            min_pairs,
            int(config.MODEL_DIRECT_HORIZON_SEGMENT_MIN_PAIRS),
        ),
    )
    occupancy_segment_min_pairs = max(
        8,
        min(
            segment_min_pairs,
            int(config.MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENT_MIN_PAIRS),
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
            src_num = features.to_float_or_none(src)
            dst_num = features.to_float_or_none(dst)
            w_num = features.to_float_or_none(w)
            if src_num is None or dst_num is None or w_num is None:
                continue
            if w_num <= 0.0:
                continue
            src_vals.append(float(src_num))
            dst_vals.append(float(dst_num))
            ws_vals.append(float(w_num))
            occ_key = prediction.occupancy_bucket_key_from_ratio(float(src_num))
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

        if config.MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENTS_ENABLED:
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
    if config.MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENTS_ENABLED and isinstance(global_stats, dict):
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
        "occupancySegmentsEnabled": bool(config.MODEL_DIRECT_HORIZON_OCCUPANCY_SEGMENTS_ENABLED),
        "occupancySegmentMinPairs": int(occupancy_segment_min_pairs),
        "maxBlend": max(0.0, min(1.0, float(config.MODEL_DIRECT_HORIZON_MAX_BLEND))),
        "byHours": by_hours,
        "global": global_stats,
    }


def feature_group_slices(loc_count: int) -> Dict[str, Tuple[int, int]]:
    time_end = 17 + config.CALENDAR_FEATURE_COUNT
    lag_end = time_end + config.LAG_TREND_SENSOR_FEATURE_COUNT + config.SCHEDULE_PHASE_FEATURE_COUNT
    weather_end = (
        lag_end
        + (len(config.WEATHER_KEYS) * 3)
        + len(config.WEATHER_ROLLING_KEYS)
        + config.WEATHER_DERIVED_FEATURE_COUNT
        + config.WEATHER_QUALITY_FEATURE_COUNT
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
    if not config.MODEL_FEATURE_MISSING_GUARD_ENABLED:
        return None

    rows = int(summary.get("rows", 0) or 0)
    if rows < max(1, config.MODEL_FEATURE_MISSING_MIN_ROWS):
        return None

    global_missing = features.to_float_or_none(summary.get("globalMissingRate"))
    if global_missing is not None and global_missing > float(config.MODEL_MAX_FEATURE_MISSING_RATE):
        return "global_missing_rate_high"

    groups = summary.get("groups", {})
    if not isinstance(groups, dict):
        return None

    lag_missing = features.to_float_or_none((groups.get("lags_trends_sensor") or {}).get("missingRate"))
    if lag_missing is not None and lag_missing > float(config.MODEL_MAX_LAG_MISSING_RATE):
        return "lag_feature_missing_rate_high"

    weather_missing = features.to_float_or_none((groups.get("weather") or {}).get("missingRate"))
    if weather_missing is not None and weather_missing > float(config.MODEL_MAX_WEATHER_MISSING_RATE):
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
    if not config.FEATURE_ABLATION_ENABLED:
        return None
    if not math.isfinite(float(baseline_mae)):
        return None
    if len(y_val) < max(1, config.FEATURE_ABLATION_MIN_VAL_ROWS):
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
        mae = features.weighted_average(np.abs(preds - y_val), w_val)
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
    depth_ref = max(1, int(config.MODEL_TUNING_COMPLEXITY_DEPTH_REF))
    depth = max(1.0, float(params.get("max_depth", config.MODEL_MAX_DEPTH)))
    eta = max(0.001, float(params.get("eta", config.MODEL_ETA)))
    min_child = max(0.0, float(params.get("min_child_weight", config.MODEL_MIN_CHILD_WEIGHT)))
    subsample = max(0.1, min(1.0, float(params.get("subsample", config.MODEL_SUBSAMPLE))))
    colsample = max(0.1, min(1.0, float(params.get("colsample_bytree", config.MODEL_COLSAMPLE_BYTREE))))
    gamma = max(0.0, float(params.get("gamma", config.MODEL_GAMMA)))
    reg_lambda = max(0.0, float(params.get("lambda", config.MODEL_REG_LAMBDA)))
    reg_alpha = max(0.0, float(params.get("alpha", config.MODEL_REG_ALPHA)))
    max_bin = max(64.0, float(params.get("max_bin", config.MODEL_MAX_BIN)))

    depth_term = max(0.0, (depth - float(depth_ref)) / float(depth_ref))
    eta_term = max(0.0, (eta - float(config.MODEL_ETA)) / max(0.01, float(config.MODEL_ETA)))
    child_term = max(0.0, (1.0 - min_child) / 1.0)
    subsample_term = max(0.0, (0.9 - subsample) / 0.4)
    colsample_term = max(0.0, (0.9 - colsample) / 0.4)
    gamma_term = max(0.0, (0.2 - gamma) / 0.2)
    lambda_term = max(0.0, (1.0 - reg_lambda) / 1.0)
    alpha_term = max(0.0, (0.1 - reg_alpha) / 0.1)
    bin_term = max(0.0, (max_bin - float(config.MODEL_MAX_BIN)) / max(64.0, float(config.MODEL_MAX_BIN)))

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
    splits = build_time_series_cv_splits(times, config.MODEL_TUNING_CV_FOLDS)
    if not splits:
        return float("inf")

    interval_weight = max(0.0, float(config.MODEL_TUNING_INTERVAL_ERR_WEIGHT))
    tail_weight = max(0.0, float(config.MODEL_TUNING_TAIL_ERR_WEIGHT))
    tail_q = max(0.5, min(0.99, float(config.MODEL_TUNING_TAIL_QUANTILE)))
    complexity_weight = max(0.0, float(config.MODEL_TUNING_COMPLEXITY_WEIGHT))
    complexity_penalty = tuning_params_complexity_score(params)
    target_coverage = max(0.1, min(0.98, float(config.INTERVAL_Q_HIGH - config.INTERVAL_Q_LOW)))

    cv_errors: List[float] = []
    tune_rounds = max(20, min(config.MODEL_TUNING_BOOST_ROUND, config.MODEL_NUM_BOOST_ROUND))

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
                early_stopping_rounds=min(config.EARLY_STOPPING_ROUNDS, 30),
                verbose_eval=False,
            )
        except Exception:
            return float("inf")

        preds = model.predict(dval)
        abs_err = np.abs(preds - y_val)
        fold_mae = features.weighted_average(abs_err, w_val)
        if not math.isfinite(float(fold_mae)):
            return float("inf")

        train_preds = model.predict(dtrain)
        train_residuals = (y_train - train_preds).astype(np.float32)
        finite_train_mask = np.isfinite(train_residuals) & np.isfinite(w_train) & (w_train > 0.0)
        if int(np.count_nonzero(finite_train_mask)) >= 20:
            q10 = float(
                features.weighted_quantile(
                    train_residuals[finite_train_mask],
                    config.INTERVAL_Q_LOW,
                    w_train[finite_train_mask],
                )
            )
            q90 = float(
                features.weighted_quantile(
                    train_residuals[finite_train_mask],
                    config.INTERVAL_Q_HIGH,
                    w_train[finite_train_mask],
                )
            )
        else:
            q10 = 0.0
            q90 = 0.0

        p10 = preds + q10
        p90 = preds + q90
        p10, p90 = features.ordered_prediction_bounds(p10, p90)
        within = ((y_val >= p10) & (y_val <= p90)).astype(np.float32)
        coverage = features.weighted_average(within, w_val)
        interval_err = abs(float(coverage) - target_coverage)

        tail_cut = float(features.weighted_quantile(abs_err, tail_q, w_val)) if abs_err.size > 0 else 0.0
        tail_mask = abs_err >= tail_cut
        tail_mae = features.weighted_average(abs_err[tail_mask], w_val[tail_mask]) if np.any(tail_mask) else 0.0

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
            "max_depth": int(params.get("max_depth", config.MODEL_MAX_DEPTH)),
            "eta": float(params.get("eta", config.MODEL_ETA)),
            "min_child_weight": float(params.get("min_child_weight", config.MODEL_MIN_CHILD_WEIGHT)),
            "subsample": float(params.get("subsample", config.MODEL_SUBSAMPLE)),
            "colsample_bytree": float(params.get("colsample_bytree", config.MODEL_COLSAMPLE_BYTREE)),
            "gamma": float(params.get("gamma", config.MODEL_GAMMA)),
            "lambda": float(params.get("lambda", config.MODEL_REG_LAMBDA)),
            "alpha": float(params.get("alpha", config.MODEL_REG_ALPHA)),
            "max_bin": int(params.get("max_bin", config.MODEL_MAX_BIN)),
        }
    except Exception:
        return None


def build_tuning_candidates(
    base_params: Dict[str, object],
    preferred_params: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    limit = max(1, config.MODEL_TUNING_MAX_CANDIDATES)
    rng = random.Random(config.MODEL_TUNING_RANDOM_SEED)
    candidates: List[Dict[str, object]] = []
    seen = set()

    def candidate_key(cand: Dict[str, object]) -> Tuple[object, ...]:
        return (
            int(cand.get("max_depth", config.MODEL_MAX_DEPTH)),
            round(float(cand.get("eta", config.MODEL_ETA)), 6),
            round(float(cand.get("min_child_weight", config.MODEL_MIN_CHILD_WEIGHT)), 6),
            round(float(cand.get("subsample", config.MODEL_SUBSAMPLE)), 6),
            round(float(cand.get("colsample_bytree", config.MODEL_COLSAMPLE_BYTREE)), 6),
            round(float(cand.get("gamma", config.MODEL_GAMMA)), 6),
            round(float(cand.get("lambda", config.MODEL_REG_LAMBDA)), 6),
            round(float(cand.get("alpha", config.MODEL_REG_ALPHA)), 6),
            int(cand.get("max_bin", config.MODEL_MAX_BIN)),
        )

    def add_candidate(overrides: Optional[Dict[str, object]] = None) -> None:
        if len(candidates) >= limit:
            return
        cand = dict(base_params)
        if overrides:
            cand.update(overrides)
        cand["max_depth"] = max(2, int(cand.get("max_depth", config.MODEL_MAX_DEPTH)))
        cand["eta"] = max(0.005, min(0.2, float(cand.get("eta", config.MODEL_ETA))))
        cand["min_child_weight"] = max(0.0, float(cand.get("min_child_weight", config.MODEL_MIN_CHILD_WEIGHT)))
        cand["subsample"] = max(0.5, min(1.0, float(cand.get("subsample", config.MODEL_SUBSAMPLE))))
        cand["colsample_bytree"] = max(0.5, min(1.0, float(cand.get("colsample_bytree", config.MODEL_COLSAMPLE_BYTREE))))
        cand["gamma"] = max(0.0, min(8.0, float(cand.get("gamma", config.MODEL_GAMMA))))
        cand["lambda"] = max(0.0, min(20.0, float(cand.get("lambda", config.MODEL_REG_LAMBDA))))
        cand["alpha"] = max(0.0, min(10.0, float(cand.get("alpha", config.MODEL_REG_ALPHA))))
        if "max_bin" in cand:
            cand["max_bin"] = max(64, int(cand.get("max_bin", config.MODEL_MAX_BIN)))
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
        depth0 = int(anchor.get("max_depth", config.MODEL_MAX_DEPTH))
        eta0 = float(anchor.get("eta", config.MODEL_ETA))
        child0 = float(anchor.get("min_child_weight", config.MODEL_MIN_CHILD_WEIGHT))
        sub0 = float(anchor.get("subsample", config.MODEL_SUBSAMPLE))
        col0 = float(anchor.get("colsample_bytree", config.MODEL_COLSAMPLE_BYTREE))
        gamma0 = float(anchor.get("gamma", config.MODEL_GAMMA))
        lambda0 = float(anchor.get("lambda", config.MODEL_REG_LAMBDA))
        alpha0 = float(anchor.get("alpha", config.MODEL_REG_ALPHA))
        bin0 = int(anchor.get("max_bin", config.MODEL_MAX_BIN))

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
    if not config.MODEL_TUNING_ENABLED or len(times) < config.MODEL_TUNING_MIN_ROWS:
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
    if not config.DIRECT_QUANTILE_ENABLED:
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
        rounds = max(1, int(num_boost_round_override or config.MODEL_NUM_BOOST_ROUND))
        if dval is not None and num_boost_round_override is None:
            return xgb.train(
                qparams,
                dtrain,
                num_boost_round=rounds,
                evals=[(dval, "val")],
                early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
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

    return max(1, min(int(config.MODEL_NUM_BOOST_ROUND), int(rounds)))


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
    location_balance_map = features.build_location_balance_weight_map_from_counts(
        {loc_id: int(loc_samples.get(loc_id, 0) or 0) for loc_id in loc_ids}
    )
    weather_lookup_cache: Optional[Dict[Tuple[datetime, str], float]] = (
        {} if weather_source is not None else None
    )
    train_dataset = features.build_model_observation_dataset(
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
    reporting_rows = train_dataset.get("reportingRows", [])
    if len(reporting_rows) != len(times):
        reporting_rows = []
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

    if len(times) < config.MIN_TRAIN_SAMPLES:
        return None, {
            **base_skip_metrics,
            "train_rows": len(times),
            "best_params": params_for_meta(normalize_tuning_overrides(preferred_params)),
            "invalid_rows_dropped": 0,
        }, None

    weights = features.stabilize_sample_weights(
        features.build_recency_weights(times) * quality_weights * features.build_occupancy_weights(y)
    )
    valid_mask = features.supervised_row_mask(X, y, weights)
    invalid_rows_dropped = int(valid_mask.size - int(np.count_nonzero(valid_mask)))
    if invalid_rows_dropped > 0:
        X = X[valid_mask]
        y = y[valid_mask]
        weights = weights[valid_mask]
        keep_indices = np.flatnonzero(valid_mask).tolist()
        times = [times[idx] for idx in keep_indices]
        if reporting_rows:
            reporting_rows = [reporting_rows[idx] for idx in keep_indices]

    if len(times) < config.MIN_TRAIN_SAMPLES:
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
    holdout_fraction = max(0.0, min(0.45, float(config.MODEL_HOLDOUT_SPLIT)))
    holdout_target = max(0, int(round(n_rows * holdout_fraction)))
    min_holdout_rows = max(1, int(config.MODEL_HOLDOUT_MIN_ROWS))
    if holdout_fraction > 0.0 and holdout_target > 0:
        holdout_target = max(holdout_target, min_holdout_rows)
        if n_rows >= (config.MIN_TRAIN_SAMPLES + holdout_target):
            times_sorted = sorted(times)
            holdout_start = max(1, min(n_rows - 1, n_rows - holdout_target))
            holdout_time = times_sorted[holdout_start]
            candidate_model_idx = [i for i, ts in enumerate(times) if ts < holdout_time]
            candidate_holdout_idx = [i for i, ts in enumerate(times) if ts >= holdout_time]
            if (
                len(candidate_model_idx) >= config.MIN_TRAIN_SAMPLES
                and len(candidate_holdout_idx) >= min_holdout_rows
            ):
                model_idx = candidate_model_idx
                holdout_idx = candidate_holdout_idx

    fill_source_idx = model_idx if model_idx else list(range(n_rows))
    feature_fill_values = features.compute_feature_fill_values(X[fill_source_idx])
    X = features.apply_feature_fill_values(X, feature_fill_values)
    feature_clip_bounds = features.compute_feature_clip_bounds(X[fill_source_idx])
    X = features.apply_feature_clip_bounds(X, feature_clip_bounds)

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
    split_idx = int(len(times_sorted) * config.TRAIN_SPLIT)
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
    selected_rounds = max(1, int(config.MODEL_NUM_BOOST_ROUND))
    dtrain_full = xgb.DMatrix(X_model, label=y_model, weight=w_model)

    dval = None
    if len(train_idx) < 10 or len(val_idx) < 10:
        p50_model = xgb.train(
            params,
            dtrain_full,
            num_boost_round=config.MODEL_NUM_BOOST_ROUND,
            verbose_eval=False,
        )
        selected_rounds = selected_boost_rounds(p50_model, config.MODEL_NUM_BOOST_ROUND)
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
            num_boost_round=config.MODEL_NUM_BOOST_ROUND,
            evals=[(dval, "val")],
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        selected_rounds = selected_boost_rounds(p50_eval_model, config.MODEL_NUM_BOOST_ROUND)

        preds = p50_eval_model.predict(dval)
        abs_errors = np.abs(preds - y_val)
        sq_errors = (preds - y_val) ** 2
        val_mae = features.weighted_average(abs_errors, w_val)
        val_rmse = math.sqrt(features.weighted_average(sq_errors, w_val))
        val_rows = len(val_idx)

        train_preds_for_interval = p50_eval_model.predict(dtrain)
        train_residuals_for_interval = (y_train - train_preds_for_interval).astype(np.float32)
        train_residual_mask = np.isfinite(train_residuals_for_interval) & np.isfinite(w_train) & (w_train > 0.0)
        if int(np.count_nonzero(train_residual_mask)) >= 20:
            q10_val = float(
                features.weighted_quantile(
                    train_residuals_for_interval[train_residual_mask],
                    config.INTERVAL_Q_LOW,
                    w_train[train_residual_mask],
                )
            )
            q90_val = float(
                features.weighted_quantile(
                    train_residuals_for_interval[train_residual_mask],
                    config.INTERVAL_Q_HIGH,
                    w_train[train_residual_mask],
                )
            )
        else:
            q10_val = 0.0
            q90_val = 0.0
        p10_val = preds + q10_val
        p90_val = preds + q90_val
        p10_val, p90_val = features.ordered_prediction_bounds(p10_val, p90_val)
        target_coverage = max(0.1, min(0.98, float(config.INTERVAL_Q_HIGH - config.INTERVAL_Q_LOW)))
        within_val = ((y_val >= p10_val) & (y_val <= p90_val)).astype(np.float32)
        val_interval_coverage = features.weighted_average(within_val, w_val)
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
    reporting_evidence = None
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
                b10, _mid, b90 = prediction.interval_bounds(
                    point_ratio=float(pred),
                    hour=int(hour),
                    residual_profile=interval_profile,
                    target=ts,
                )
                p10_vals.append(float(b10))
                p90_vals.append(float(b90))
            p10_holdout = np.array(p10_vals, dtype=np.float32)
            p90_holdout = np.array(p90_vals, dtype=np.float32)

        p10_holdout, p90_holdout = features.ordered_prediction_bounds(p10_holdout, p90_holdout)
        reporting_evidence = metrics.terminal_evaluation_evidence(
            model_key, reporting_rows, times, holdout_idx, p50_holdout, p10_holdout,
            p90_holdout, holdout_time, config.RESAMPLE_MINUTES,
        )
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
            holdout_mae = features.weighted_average(holdout_abs, w_holdout)
            holdout_rmse = math.sqrt(features.weighted_average(holdout_sq, w_holdout))
            target_coverage = max(0.1, min(0.98, float(config.INTERVAL_Q_HIGH - config.INTERVAL_Q_LOW)))
            holdout_within = ((y_holdout >= p10_holdout) & (y_holdout <= p90_holdout)).astype(np.float32)
            holdout_interval_coverage = features.weighted_average(holdout_within, w_holdout)
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
        "reporting_evidence": reporting_evidence,
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
    target_coverage = max(0.1, min(0.98, float(config.INTERVAL_Q_HIGH - config.INTERVAL_Q_LOW)))
    eval_dataset = dataset
    if not isinstance(eval_dataset, dict):
        weather_lookup_cache: Optional[Dict[Tuple[datetime, str], float]] = (
            {} if weather_source is not None else None
        )
        location_balance_map = features.build_location_balance_weight_map_from_loc_data(
            loc_ids,
            loc_data,
            since=since,
        )
        eval_dataset = features.build_model_observation_dataset(
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

    preds = prediction.predict_model_bundle_on_feature_matrix(
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
    recency_w = features.build_recency_weights(filtered_times)
    occupancy_w = features.build_occupancy_weights(labels_arr)
    weights = features.stabilize_sample_weights(recency_w * occupancy_w * filtered_quality)
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
        "enabled": config.CHAMPION_GATE_ENABLED,
        "promote": True,
        "reason": "gate_disabled",
        "minRows": max(1, config.CHAMPION_GATE_MIN_ROWS),
        "minMaeImprovement": float(config.CHAMPION_GATE_MIN_MAE_IMPROVEMENT),
        "maxRmseDegrade": float(config.CHAMPION_GATE_MAX_RMSE_DEGRADE),
        "maxIntervalErrDegrade": float(config.CHAMPION_GATE_MAX_INTERVAL_ERR_DEGRADE),
        "champion": champion_eval,
        "challenger": challenger_eval,
    }
    if not config.CHAMPION_GATE_ENABLED:
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
    if champion_points < config.CHAMPION_GATE_MIN_ROWS or challenger_points < config.CHAMPION_GATE_MIN_ROWS:
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
        mae_gain >= float(config.CHAMPION_GATE_MIN_MAE_IMPROVEMENT)
        and rmse_degrade <= float(config.CHAMPION_GATE_MAX_RMSE_DEGRADE)
        and interval_err_degrade <= float(config.CHAMPION_GATE_MAX_INTERVAL_ERR_DEGRADE)
    )
    decision["promote"] = bool(promote)
    decision["reason"] = "promote" if promote else "champion_kept"
    return decision


def should_retrain_model(
    meta: Optional[Dict[str, object]],
    now: datetime,
    retrain_hours: Optional[int] = None,
) -> bool:
    if config.FORCE_RETRAIN:
        return True
    if not meta:
        return True

    force_until = features.parse_iso_datetime(str(meta.get("forceRetrainUntil", "")))
    if bool(meta.get("forceRetrain")):
        if force_until is None:
            return True
        if now <= force_until:
            return True
    elif force_until is not None and now <= force_until:
        return True

    trained_at = features.parse_iso_datetime(meta.get("trainedAt", ""))
    if not trained_at:
        return True

    effective_hours = max(1, int(retrain_hours if retrain_hours is not None else config.MODEL_RETRAIN_HOURS))
    return (now - trained_at) >= timedelta(hours=effective_hours)


def passes_guardrail(
    baseline_meta: Optional[Dict[str, object]],
    candidate_metrics: Dict[str, object],
) -> bool:
    if not baseline_meta:
        return True

    baseline_holdout_rows = int(baseline_meta.get("holdoutRows", 0) or 0)
    candidate_holdout_rows = int(candidate_metrics.get("holdout_rows", 0) or 0)
    baseline_holdout_mae = features.to_float_or_none(baseline_meta.get("holdoutMae"))
    candidate_holdout_mae = features.to_float_or_none(candidate_metrics.get("holdout_mae"))
    baseline_holdout_int_err = features.to_float_or_none(baseline_meta.get("holdoutIntervalCoverageError"))
    candidate_holdout_int_err = features.to_float_or_none(candidate_metrics.get("holdout_interval_coverage_error"))

    if (
        baseline_holdout_rows >= config.MODEL_HOLDOUT_MIN_ROWS
        and candidate_holdout_rows >= config.MODEL_HOLDOUT_MIN_ROWS
        and baseline_holdout_mae is not None
        and candidate_holdout_mae is not None
    ):
        if candidate_holdout_mae > baseline_holdout_mae * (1.0 + config.MODEL_GUARDRAIL_MAX_HOLDOUT_MAE_DEGRADE):
            return False
        if (
            baseline_holdout_int_err is not None
            and candidate_holdout_int_err is not None
            and (candidate_holdout_int_err - baseline_holdout_int_err)
            > config.MODEL_GUARDRAIL_MAX_HOLDOUT_INTERVAL_ERR_DEGRADE
        ):
            return False

    baseline_mae = features.to_float_or_none(baseline_meta.get("valMae"))
    baseline_rows = int(baseline_meta.get("valRows", 0) or 0)
    candidate_mae = features.to_float_or_none(candidate_metrics.get("val_mae"))
    candidate_rows = int(candidate_metrics.get("val_rows", 0) or 0)
    baseline_val_int_err = features.to_float_or_none(baseline_meta.get("valIntervalCoverageError"))
    candidate_val_int_err = features.to_float_or_none(candidate_metrics.get("val_interval_coverage_error"))

    if baseline_mae is None or candidate_mae is None:
        return True
    if baseline_rows < config.MODEL_GUARDRAIL_MIN_VAL_ROWS:
        return True
    if candidate_rows < config.MODEL_GUARDRAIL_MIN_VAL_ROWS:
        return True

    if float(candidate_mae) > float(baseline_mae) * (1.0 + config.MODEL_GUARDRAIL_MAX_MAE_DEGRADE):
        return False

    if (
        baseline_val_int_err is not None
        and candidate_val_int_err is not None
        and (candidate_val_int_err - baseline_val_int_err) > config.MODEL_GUARDRAIL_MAX_VAL_INTERVAL_ERR_DEGRADE
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
    feature_count = features.model_feature_count(len(expected_loc_ids))
    saved_bundle, saved_meta = data.load_saved_model(
        model_key=model_key,
        expected_loc_ids=expected_loc_ids,
        expected_feature_count=feature_count,
    )

    status = "using_saved_model" if saved_bundle is not None else "no_saved_model"
    run_metrics: Dict[str, object] = {}
    adaptive_controls = adaptive_controls or {}
    effective_retrain_hours = max(
        1,
        int(adaptive_controls.get("retrainHours", config.MODEL_RETRAIN_HOURS)),
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
            "schemaVersion": config.MODEL_SCHEMA_VERSION,
            "facilityId": int(facility_id),
            "categoryKey": str(category_key),
            "modelKey": model_key,
            "trainedAt": now.isoformat(),
            "locIds": expected_loc_ids,
            "featureCount": feature_count,
            "trainRows": int(candidate_metrics.get("train_rows", 0)),
            "valRows": int(candidate_metrics.get("val_rows", 0)),
            "valMae": features.to_float_or_none(candidate_metrics.get("val_mae")),
            "valRmse": features.to_float_or_none(candidate_metrics.get("val_rmse")),
            "valIntervalCoverage": features.to_float_or_none(
                candidate_metrics.get("val_interval_coverage")
            ),
            "valIntervalCoverageError": features.to_float_or_none(
                candidate_metrics.get("val_interval_coverage_error")
            ),
            "selectedBoostRounds": int(candidate_metrics.get("selected_boost_rounds", 0) or 0),
            "holdoutRows": int(candidate_metrics.get("holdout_rows", 0)),
            "holdoutMae": features.to_float_or_none(candidate_metrics.get("holdout_mae")),
            "holdoutRmse": features.to_float_or_none(candidate_metrics.get("holdout_rmse")),
            "holdoutIntervalCoverage": features.to_float_or_none(
                candidate_metrics.get("holdout_interval_coverage")
            ),
            "holdoutIntervalCoverageError": features.to_float_or_none(
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
                "driftRecentDays": int(adaptive_controls.get("driftRecentDays", config.DRIFT_RECENT_DAYS)),
                "driftAlertMultiplier": float(
                    adaptive_controls.get("driftAlertMultiplier", config.DRIFT_ALERT_MULTIPLIER)
                ),
                "driftActionStreakForRetrain": int(
                    adaptive_controls.get(
                        "driftActionStreakForRetrain",
                        config.DRIFT_ACTION_STREAK_FOR_RETRAIN,
                    )
                ),
            },
        }

        if saved_bundle is not None:
            since = now - timedelta(days=max(1, config.CHAMPION_GATE_RECENT_DAYS))
            recent_eval_dataset = features.build_model_observation_dataset(
                loc_ids=expected_loc_ids,
                loc_data=loc_data,
                onehot=onehot,
                weather_source=weather_series,
                location_balance_map=features.build_location_balance_weight_map_from_loc_data(
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
                feature_fill_values=features.coerce_feature_fill_values(
                    saved_meta.get("featureFillValues") if isinstance(saved_meta, dict) else None,
                    expected_cols=feature_count,
                ),
                feature_clip_bounds=features.coerce_feature_clip_bounds(
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
                feature_fill_values=features.coerce_feature_fill_values(
                    candidate_metrics.get("feature_fill_values"),
                    expected_cols=feature_count,
                ),
                feature_clip_bounds=features.coerce_feature_clip_bounds(
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
                        data.save_model_meta_only(updated_saved, model_key=model_key)
                    except Exception:
                        traceback.print_exc()
                    saved_meta = updated_saved
                return saved_bundle, saved_meta, status, run_metrics

        if saved_bundle is not None and not passes_guardrail(saved_meta, candidate_metrics):
            status = "guardrail_kept_previous"
            return saved_bundle, saved_meta, status, run_metrics

        backup_written = False
        if saved_bundle is not None:
            backup_written = data.backup_current_artifacts(model_key=model_key)
        candidate_meta["previousBackupWritten"] = bool(backup_written)

        data.save_model_artifacts(
            candidate_bundle,
            candidate_meta,
            model_key=model_key,
        )
        status = "trained_and_saved"
        # Transient identity check prevents attribution after rollback/replacement.
        run_metrics["reporting_selected_bundle"] = candidate_bundle
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

    for facility_id, facility in config.FACILITIES.items():
        all_loc_ids = config.facility_location_ids(facility)
        all_key = config.model_unit_key(facility_id, "__all__")
        all_onehot = features.build_onehot(all_loc_ids)
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
            if not config.should_train_category(category_key):
                continue
            category_loc_ids = sorted(set(int(loc_id) for loc_id in category.get("location_ids", [])))
            if not category_loc_ids:
                continue

            key = config.model_unit_key(facility_id, category_key)
            onehot = features.build_onehot(category_loc_ids)
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

    use_parallel = max(1, int(config.MODEL_PARALLEL_WORKERS)) > 1 and len(unit_specs) > 1
    if use_parallel:
        max_workers = min(len(unit_specs), max(1, int(config.MODEL_PARALLEL_WORKERS)))
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


_sys.modules.setdefault("server.reclive.forecasting.training", _sys.modules[__name__])
_sys.modules.setdefault("reclive.forecasting.training", _sys.modules[__name__])
