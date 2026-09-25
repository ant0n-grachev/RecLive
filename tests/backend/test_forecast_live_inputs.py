"""Serving regressions with synthetic observations; no database or training."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from server.reclive.forecasting import config, data, features, prediction


NOW = datetime(2026, 9, 24, 17, 20, tzinfo=timezone.utc)


class SnapshotConnection:
    def __init__(self, rows):
        self.rows = rows

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params=()):
        if "FROM location_snapshot" not in sql or params:
            raise AssertionError("forecast live observations must read location snapshots")

    def fetchall(self):
        return self.rows


def live_location(*, fetched_at=NOW, source_updated_at=None, count=90.0):
    return {
        "raw_times": [NOW - timedelta(hours=2)],
        "raw_values": [70.0],
        "live_snapshot": {
            "count": count,
            "fetched_at": fetched_at,
            "source_updated_at": source_updated_at,
        },
    }


def estimate_context(location):
    return {
        "now": NOW,
        "max_caps": {11: 100},
        "loc_data": {11: location},
        "loc_samples": {11: 1000},
        "avg_dow_hour": {},
        "avg_hour": {},
        "avg_overall": {11: (0.2, 1000)},
        "loc_to_model_key": {11: "test"},
        "recursive_ratio_cache": {},
    }


def test_live_snapshot_loader_preserves_fetch_and_source_times_separately():
    fetched = datetime(2026, 9, 24, 17, 19)
    source = datetime(2026, 9, 24, 12)
    connection = SnapshotConnection([
        (11, False, 90, 100, source, fetched),
        (22, True, 70, 100, None, fetched),
    ])

    snapshots = data.load_live_snapshots(connection)

    assert snapshots == {
        11: {
            "count": 90.0,
            "fetched_at": datetime(2026, 9, 24, 17, 19, tzinfo=timezone.utc),
            "source_updated_at": datetime(2026, 9, 24, 12, tzinfo=timezone.utc),
        },
        22: {
            "count": 0.0,
            "fetched_at": datetime(2026, 9, 24, 17, 19, tzinfo=timezone.utc),
            "source_updated_at": None,
        },
    }


@pytest.mark.parametrize("count,fetched", [(None, NOW), (-1, NOW), (float("nan"), NOW), (20, None)])
def test_invalid_live_snapshots_are_unavailable(count, fetched):
    snapshots = data.load_live_snapshots(SnapshotConnection([(11, False, count, 100, None, fetched)]))
    assert snapshots == {}


@pytest.mark.parametrize("source", [None, NOW - timedelta(hours=4)])
def test_live_freshness_uses_successful_fetch_when_state_has_not_changed(source):
    location = live_location(fetched_at=NOW - timedelta(minutes=1), source_updated_at=source)

    assert prediction.latest_live_ratio_and_age_minutes(location, 100, NOW) == (0.9, 1.0)


@pytest.mark.parametrize("snapshot", [None, {}, {"count": 80.0, "fetched_at": NOW + timedelta(minutes=1)}])
def test_unavailable_or_future_snapshot_does_not_revive_prior_raw_count(snapshot):
    location = {"raw_times": [NOW - timedelta(minutes=1)], "raw_values": [70.0], "live_snapshot": snapshot}
    assert prediction.latest_live_ratio_and_age_minutes(location, 100, NOW) == (None, None)


def test_stalled_collection_keeps_live_correction_disabled(monkeypatch):
    monkeypatch.setattr(config, "LIVE_BIAS_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_BIAS_MAX_AGE_MIN", 45.0)
    location = live_location(fetched_at=NOW - timedelta(minutes=46))

    result = prediction.estimate_location(11, NOW, estimate_context(location))

    assert result["countP50"] == pytest.approx(20.0)


@pytest.mark.parametrize("target", [NOW - timedelta(minutes=20), NOW - timedelta(hours=5)])
def test_live_value_never_corrects_a_past_target(monkeypatch, target):
    monkeypatch.setattr(config, "LIVE_BIAS_ENABLED", True)
    location = {"raw_times": [NOW], "raw_values": [90.0]}

    result = prediction.estimate_location(11, target, estimate_context(location))

    assert result["countP50"] == pytest.approx(20.0)


def test_fresh_unchanged_snapshot_corrects_current_forecast(monkeypatch):
    monkeypatch.setattr(config, "LIVE_BIAS_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_BIAS_BASE_WEIGHT", 0.5)

    result = prediction.estimate_location(11, NOW, estimate_context(live_location()))

    assert result["countP50"] == pytest.approx(55.0)


@pytest.mark.parametrize("observed,predicted,want", [(0.2, 0.9, 0.2), (0.0, 0.9, 0.0), (None, 0.9, 0.9)])
def test_observed_lags_take_precedence_over_recursive_forecasts(observed, predicted, want):
    assert features.ratio_value_from_maps({NOW: observed}, NOW, {NOW: predicted}) == want


@pytest.mark.parametrize("target", [NOW - timedelta(hours=1), NOW - timedelta(minutes=20), NOW])
def test_recursive_seed_does_not_synthesize_known_or_current_history(target):
    overrides = {}
    prediction.seed_recursive_ratio_overrides(overrides, target, 0.9, as_of=NOW)
    assert overrides == {}


def test_recursive_seed_keeps_future_quarter_hours(monkeypatch):
    monkeypatch.setattr(config, "RESAMPLE_MINUTES", 15)
    overrides = {}
    target = datetime(2026, 9, 24, 18, tzinfo=timezone.utc)

    prediction.seed_recursive_ratio_overrides(overrides, target, 0.3, as_of=NOW)

    assert overrides == {
        datetime(2026, 9, 24, 18, minute, tzinfo=timezone.utc): 0.3
        for minute in (0, 15, 30, 45)
    }


class LagFollowingModel:
    """Deterministic inference boundary: return the real one-hour lag feature."""

    def predict(self, matrix):
        # The feature contract has 29 time/calendar values, then 15m and 1h lags.
        return np.asarray(matrix.get_data().toarray()[:, 30], dtype=np.float32)


@pytest.mark.parametrize("cached_past", [False, True])
def test_precomputed_future_estimate_uses_observed_trajectory(monkeypatch, cached_past):
    monkeypatch.setattr(config, "LIVE_BIAS_ENABLED", False)
    past = datetime(2026, 9, 24, 17, tzinfo=timezone.utc)
    future = datetime(2026, 9, 24, 18, tzinfo=timezone.utc)
    times = [past - timedelta(hours=1), past]
    location = {
        "bucket_map": dict(zip(times, [0.8, 0.2])),
        "bucket_times": times,
        "bucket_values": [0.8, 0.2],
        "raw_times": times,
        "raw_values": [80.0, 20.0],
        "max_cap": 100,
        "fallback_avg_overall": (0.2, 1000),
        "is_stale": False,
    }
    context = estimate_context(location)
    context.update({
        "models_by_key": {"test": {"p50": LagFollowingModel()}},
        "onehot_by_key": {"test": {11: [1.0]}},
        "model_prediction_cache": {("test", 11, past): (0.7, 0.8, 0.9, 0.0)} if cached_past else {},
    })

    _, _, _, (_, medians, _, _) = prediction.precompute_target_estimate_matrices_for_locations(
        [11], [past, future], context,
    )

    assert medians[1, 0] == pytest.approx(20.0)
    assert all(timestamp > NOW for timestamp in context["recursive_ratio_cache"][("test", 11)])


class AdvancingLagModel:
    """Deterministic inference that exposes which trajectory the features use."""

    def __init__(self, lag_column):
        self.lag_column = lag_column
        self.batch_sizes = []

    def predict(self, matrix):
        self.batch_sizes.append(matrix.num_row())
        return np.asarray(matrix.get_data().toarray()[:, self.lag_column] + 0.1, dtype=np.float32)


def recursive_context(now, model, location_ids=(11,)):
    location = {
        "bucket_map": {now: 0.2}, "bucket_times": [now], "bucket_values": [0.2],
        "raw_times": [now], "raw_values": [20.0], "max_cap": 100,
        "fallback_avg_overall": (0.2, 1000), "is_stale": False,
    }
    context = estimate_context(location)
    context.update({
        "now": now,
        "max_caps": {loc_id: 100 for loc_id in location_ids},
        "loc_data": {loc_id: dict(location) for loc_id in location_ids},
        "loc_samples": {loc_id: 1000 for loc_id in location_ids},
        "avg_overall": {loc_id: (0.2, 1000) for loc_id in location_ids},
        "loc_to_model_key": {loc_id: "test" for loc_id in location_ids},
        "models_by_key": {"test": {"p50": model}},
        "onehot_by_key": {"test": {loc_id: [1.0] for loc_id in location_ids}},
        "model_prediction_cache": {}, "feature_cache": {},
    })
    return context


def test_recursive_batch_uses_corrected_prior_forecast_and_keeps_location_batching(monkeypatch):
    monkeypatch.setattr(config, "LIVE_BIAS_ENABLED", False)
    monkeypatch.setattr(config, "MODEL_MISSING_FEATURE_BLEND_ENABLED", False)
    monkeypatch.setattr(config, "MODEL_LONG_HORIZON_BLEND_ENABLED", True)
    monkeypatch.setattr(config, "MODEL_LONG_HORIZON_BLEND_START_HOURS", 4.0)
    monkeypatch.setattr(config, "MODEL_LONG_HORIZON_BLEND_FULL_HOURS", 12.0)
    monkeypatch.setattr(config, "MODEL_LONG_HORIZON_BLEND_MAX_WEIGHT", 0.28)
    now = datetime(2026, 9, 24, 17, tzinfo=timezone.utc)
    model = AdvancingLagModel(lag_column=30)
    context = recursive_context(now, model, location_ids=(11, 22))
    targets = [now + timedelta(hours=hours) for hours in (12, 13, 14)]

    _, _, _, (_, medians, _, _) = prediction.precompute_target_estimate_matrices_for_locations(
        [11, 22], list(reversed(targets)), context,
    )

    # First corrected count is .72 * 30 + .28 * 20 = 27.2. Each next
    # raw prediction adds ten to the corrected count, then blends again.
    for column in range(2):
        assert medians[:, column] == pytest.approx([27.2, 32.384, 36.11648])
    assert model.batch_sizes == [2, 2, 2]


@pytest.mark.parametrize("boundary,lag_column,target_hours,want", [
    ("opening", 29, (10.0, 10.25, 10.5), [0.0, 10.0, 20.0]),
    ("closing", 33, (2.0, 14.0), [0.0, 10.0]),
])
def test_recursive_batch_uses_schedule_boundary_zero_for_later_targets(
    monkeypatch, boundary, lag_column, target_hours, want,
):
    monkeypatch.setattr(config, "LIVE_BIAS_ENABLED", False)
    monkeypatch.setattr(config, "MODEL_MISSING_FEATURE_BLEND_ENABLED", False)
    monkeypatch.setattr(config, "MODEL_LONG_HORIZON_BLEND_ENABLED", False)
    monkeypatch.setattr(config, "SCHEDULE_BOUNDARY_ZERO_ENABLED", True)
    now = datetime(2026, 9, 24, 20, tzinfo=timezone.utc)
    sections = [{"title": "Building Hours", "rows": [{"label": "Mon-Fri", "hours": "6:00 am - 10:00 pm"}]}]
    context = recursive_context(now, AdvancingLagModel(lag_column))
    context["loc_data"][11]["schedule_sections"] = sections
    context.update({
        "facility_schedule_by_id": {1186: {"sections": sections}},
        "location_facility_map": {11: 1186}, "schedule_boundary_cache": {},
        "schedule_date_range_cache": {}, "schedule_weekday_cache": {},
        "schedule_hours_cache": {},
    })
    targets = [now + timedelta(hours=hours) for hours in target_hours]

    _, _, _, (_, medians, _, _) = prediction.precompute_target_estimate_matrices_for_locations(
        [11], targets, context,
    )

    assert medians[:, 0] == pytest.approx(want), boundary
