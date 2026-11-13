from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from server.reclive.forecasting import config, features, training


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def assert_temporal_folds(times, splits, eligible=None):
    if eligible is None:
        eligible = range(len(times))
    previous_validation = set()
    for train_idx, val_idx in splits:
        train_times = {times[index] for index in train_idx}
        val_times = {times[index] for index in val_idx}
        assert len(train_idx) >= 20
        assert len(val_idx) >= 10
        assert max(train_times) < min(val_times)
        assert previous_validation <= set(train_idx)
        assert train_idx == sorted(
            (index for index in eligible if times[index] < min(val_times)),
            key=times.__getitem__,
        )
        assert val_idx == sorted(
            (index for index in eligible if min(val_times) <= times[index] <= max(val_times)),
            key=times.__getitem__,
        )
        previous_validation = set(val_idx)


@pytest.mark.parametrize("folds", [1, 2, 3, 5])
def test_tuning_keeps_whole_timestamps_in_unsorted_uneven_rows(folds):
    # Repeated locations have uneven coverage, including a group spanning several
    # nominal row cutoffs. Later groups still have enough rows for all folds.
    sizes = [20, 400, 12, 12, 12, 12, 12, 12]
    times = [
        START + timedelta(minutes=15 * offset)
        for location in range(max(sizes))
        for offset, size in reversed(list(enumerate(sizes)))
        if location < size
    ]
    splits = training.build_time_series_cv_splits(times, folds)
    assert len(splits) == folds
    assert_temporal_folds(times, splits)
    assert splits == training.build_time_series_cv_splits(times, folds)
    assert len(splits[-1][0]) + len(splits[-1][1]) == len(times)


@pytest.mark.parametrize(
    ("sizes", "expected"),
    [
        ([], 0),
        ([59], 0),
        ([150], 0),
        ([19, 120], 0),
        ([120, 9], 0),
        ([30, 30], 1),
        ([40, 12, 18], 2),
    ],
)
def test_tuning_returns_only_folds_with_enough_time_groups_and_rows(sizes, expected):
    times = [
        START + timedelta(minutes=15 * offset)
        for offset, size in enumerate(sizes)
        for _ in range(size)
    ]
    splits = training.build_time_series_cv_splits(times, 5)
    assert len(splits) == expected
    assert_temporal_folds(times, splits)


def test_facility_training_passes_aligned_rows_to_temporal_tuning(monkeypatch):
    location_ids = [11, 22, 33]
    times = [
        START + timedelta(minutes=15 * offset)
        for _location in location_ids
        for offset in range(479)
    ]
    row_ids = np.arange(len(times), dtype=np.float32)
    X = np.column_stack([row_ids, np.ones_like(row_ids)])
    y = (row_ids % 100) / 100
    y[0] = np.nan
    quality = 0.5 + row_ids / len(times)
    weights = features.stabilize_sample_weights(
        features.build_recency_weights(times) * quality * features.build_occupancy_weights(y)
    )
    dataset = {"X": X, "y": y, "times": times, "rowQualityWeights": quality}
    monkeypatch.setattr(features, "build_model_observation_dataset", lambda **kwargs: dataset)
    monkeypatch.setattr(config, "MODEL_FEATURE_CLIP_ENABLED", False)
    monkeypatch.setattr(config, "MODEL_TUNING_MAX_CANDIDATES", 1)
    monkeypatch.setattr(config, "DIRECT_QUANTILE_ENABLED", False)

    class Model:
        best_iteration = 0

        def predict(self, matrix):
            return np.full(len(matrix.values), 0.5, dtype=np.float32)

    def matrix(values, label=None, weight=None):
        return SimpleNamespace(values=values, label=label, weight=weight)

    splits = []
    model_rows = []

    def fit(params, dtrain, *, evals=None, **kwargs):
        if len(splits) < config.MODEL_TUNING_CV_FOLDS:
            dval = evals[0][0]
            indices = []
            for matrix in (dtrain, dval):
                rows = matrix.values[:, 0].astype(int)
                assert 0 not in rows
                np.testing.assert_array_equal(X[rows], matrix.values)
                np.testing.assert_array_equal(y[rows], matrix.label)
                np.testing.assert_allclose(weights[rows], matrix.weight, rtol=1e-6)
                indices.append(rows.tolist())
            splits.append(tuple(indices))
        elif evals is None and not model_rows:
            model_rows.extend(dtrain.values[:, 0].astype(int).tolist())
        return Model()

    monkeypatch.setattr(training, "xgb", SimpleNamespace(DMatrix=matrix, train=fit))
    _, metrics, _ = training.train_model_unit(
        "1186:__all__", location_ids, {}, {}, dict.fromkeys(location_ids, 479), None
    )

    assert len(model_rows) >= config.MODEL_TUNING_MIN_ROWS
    assert metrics["holdout_rows"] > 0
    assert metrics["tuned_candidates"] == 1
    assert metrics["tuning_cv_objective"] is not None
    assert len(splits) == config.MODEL_TUNING_CV_FOLDS
    final_train_idx, final_val_idx = splits[-1]
    assert set(final_train_idx + final_val_idx) == set(model_rows)
    assert_temporal_folds(times, splits, eligible=model_rows)
