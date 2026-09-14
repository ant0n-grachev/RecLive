"""Deterministic reporting checks: no training, database or provider access."""

import importlib
import math
import os
import subprocess
import sys
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def compute(**overrides):
    from server.reclive.forecasting.metrics import compute_forecast_metrics

    values = dict(
        actual_people=[100.0, 50.0], predicted_people=[90.0, 70.0],
        capacity_people=[200.0, 100.0], lower_people=[80.0, 40.0],
        upper_people=[110.0, 80.0], baseline_people=[110.0, 40.0],
    )
    return compute_forecast_metrics(**(values | overrides))


def test_explicit_metrics_have_correct_units_and_no_precision_pct():
    metrics = compute()
    assert metrics.mae_people == 15.0
    assert metrics.mae_capacity_percentage_points == 12.5
    assert metrics.rmse_people == pytest.approx(math.sqrt(250.0))
    assert metrics.prediction_interval_coverage == 1.0
    assert metrics.simple_baseline_mae_people == 10.0
    assert set(metrics.to_payload()) == {
        "maePeople", "maeCapacityPercentagePoints", "rmsePeople",
        "predictionIntervalCoverage", "simpleBaselineMaePeople",
        "rollingHoldoutByFacility",
    }


def test_optional_populations_do_not_erase_valid_people_errors():
    metrics = compute(
        capacity_people=[0.0, 100.0], lower_people=[None, 80.0],
        upper_people=[None, 40.0], baseline_people=[None, 40.0],
    )
    assert metrics.mae_people == 15.0
    assert metrics.mae_capacity_percentage_points == 20.0
    assert metrics.prediction_interval_coverage is None
    assert metrics.simple_baseline_mae_people == 10.0
    assert metrics.observation_counts == {
        "maePeople": 2, "maeCapacityPercentagePoints": 1, "rmsePeople": 2,
        "predictionIntervalCoverage": 0, "simpleBaselineMaePeople": 1,
    }


@pytest.mark.parametrize("invalid", [None, True, False, float("nan"), float("inf"), "100"])
def test_invalid_values_are_excluded_per_metric(invalid):
    metrics = compute(predicted_people=[invalid, 70.0])
    assert metrics.mae_people == 20.0
    assert metrics.prediction_interval_coverage == 1.0
    assert metrics.simple_baseline_mae_people == 10.0
    metrics = compute(actual_people=[invalid, 50.0])
    assert metrics.mae_people == 20.0
    assert metrics.observation_counts["predictionIntervalCoverage"] == 1


@pytest.mark.parametrize("name", ["actual_people", "predicted_people", "capacity_people", "lower_people", "upper_people", "baseline_people"])
def test_misaligned_vectors_are_rejected(name):
    with pytest.raises(ValueError, match="equal lengths"):
        compute(**{name: [1.0]})


def test_empty_metrics_are_null_and_bounds_are_inclusive():
    metrics = compute(**{name: [] for name in ["actual_people", "predicted_people", "capacity_people", "lower_people", "upper_people", "baseline_people"]})
    assert metrics.mae_people is None
    assert metrics.to_payload()["rollingHoldoutByFacility"] == {}
    assert compute(lower_people=[100.0, 0.0], upper_people=[100.0, 50.0]).prediction_interval_coverage == 1.0


def test_importing_metrics_is_inert_in_invalid_model_environment(tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, " + repr(str(root)) + "); from server.reclive.forecasting import metrics; assert 'xgboost' not in sys.modules; assert 'server.reclive.forecasting.config' not in sys.modules; print('inert')"],
        cwd=tmp_path, env={**os.environ, "GYM_RESAMPLE_MINUTES": "invalid"},
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "inert\n"


def test_forecasting_has_real_definition_owners():
    owners = {
        "config": "require_env", "data": "load_history",
        "features": "build_features", "training": "train_model_unit",
        "prediction": "predict_model_bundle_on_feature_matrix",
        "reporting": "sanitize_for_json", "job": "build_forecast",
    }
    for owner, name in owners.items():
        module = importlib.import_module("server.reclive.forecasting." + owner)
        assert getattr(module, name).__module__ == module.__name__


UTC = timezone.utc
START = datetime(2026, 1, 2, tzinfo=UTC)


def terminal_fixture(model_key="1186:__all__", facility_id=1186):
    from server.reclive.forecasting.metrics import raw_baseline_observation, terminal_evaluation_evidence

    times = [START, START, START + timedelta(hours=24)]
    metadata = [
        {"facility_id": facility_id, "location_id": location, "target": target,
         "time_aligned": True,
         "capacity": capacity, "actual_people": actual,
         "raw_baseline": [raw_baseline_observation(START - timedelta(hours=1),
                                                    START - timedelta(minutes=30), baseline)]}
        for location, target, capacity, actual, baseline in [
            (11, times[0], 200., 260., 110.), (22, times[1], 100., 50., 40.),
            (11, times[2], 200., 100., 110.),
        ]
    ]
    evidence = terminal_evaluation_evidence(model_key, metadata, times, [0, 1, 2],
                                             [1.2, .7, .45], [1.1, None, .4],
                                             [1.4, None, .55], START, 15)
    assert evidence is not None
    return evidence


def test_terminal_evidence_uses_unclipped_observations_and_location_capacities():
    from server.reclive.forecasting.metrics import metrics_for_rows, compute_rolling_holdout_by_facility

    evidence = terminal_fixture()
    rows = evidence["rows"]
    result = metrics_for_rows(rows)
    assert [row.actual_people for row in rows] == [260., 50., 100.]
    assert [row.predicted_people for row in rows] == [240., 70., 90.]
    assert result.mae_people == pytest.approx(50 / 3)
    assert result.mae_capacity_percentage_points == pytest.approx(35 / 3)
    assert result.rmse_people == pytest.approx(math.sqrt(300))
    assert result.prediction_interval_coverage == 1
    assert result.simple_baseline_mae_people == pytest.approx(170 / 3)
    assert result.observation_counts["predictionIntervalCoverage"] == 2
    grouped = compute_rolling_holdout_by_facility(rows, evidence["windows"])[1186].to_payload()
    assert grouped["method"] == "fixed_model_terminal_holdout"
    assert grouped["independentBacktest"] is False
    assert [window["observationCounts"]["maePeople"] for window in grouped["windows"]] == [2, 1]
    assert grouped["windows"][-1]["end"] == (START + timedelta(hours=24, minutes=15)).isoformat()


def test_baseline_requires_prior_observation_and_availability_and_rejects_conflicts():
    from server.reclive.forecasting.metrics import raw_baseline_observation, persistence_baseline

    early = raw_baseline_observation(START - timedelta(hours=2), START - timedelta(hours=1), 40.)
    late = raw_baseline_observation(START - timedelta(minutes=30), START + timedelta(seconds=1), 90.)
    boundary = raw_baseline_observation(START - timedelta(minutes=1), START, 100.)
    future = raw_baseline_observation(START, START - timedelta(seconds=1), 110.)
    assert persistence_baseline([early, late, boundary, future], START) == 40
    assert raw_baseline_observation(START, None, 10) is None
    assert raw_baseline_observation(START, START, True) is None
    assert raw_baseline_observation(START, START, -1) is None
    conflict = raw_baseline_observation(early.observed_at, START - timedelta(minutes=1), 41.)
    assert persistence_baseline([early, conflict], START) is None
    assert persistence_baseline([conflict, early], START) is None


def test_reporting_timestamps_use_utc_db_semantics_and_aware_external_inputs():
    from server.reclive.forecasting.metrics import reporting_utc, raw_baseline_observation
    from zoneinfo import ZoneInfo

    naive = datetime(2026, 11, 1, 7, 30)
    assert reporting_utc(naive) is None
    assert reporting_utc("2026-11-01T07:30:00") is None
    assert reporting_utc(naive, trusted_db=True) == naive.replace(tzinfo=UTC)
    mountain = datetime(2026, 11, 1, 1, 30, tzinfo=ZoneInfo("America/Denver"), fold=0)
    assert reporting_utc(mountain) == naive.replace(tzinfo=UTC)
    observation = raw_baseline_observation(naive, naive, 10.)
    assert observation.observed_at == naive.replace(tzinfo=UTC)
    assert reporting_utc("2026-11-01T01:30:00-07:00") == datetime(2026, 11, 1, 8, 30, tzinfo=UTC)


def test_holdout_rejects_overlap_duplicates_and_naive_times():
    from dataclasses import replace
    from server.reclive.forecasting.metrics import compute_rolling_holdout_by_facility

    evidence = terminal_fixture()
    rows, windows = evidence["rows"], evidence["windows"]
    with pytest.raises(ValueError, match="overlap"):
        compute_rolling_holdout_by_facility(rows, windows + [windows[0]])
    with pytest.raises(ValueError, match="unique"):
        compute_rolling_holdout_by_facility(rows + [rows[0]], windows)
    with pytest.raises(ValueError, match="aware"):
        compute_rolling_holdout_by_facility(rows, [replace(windows[0], start=START.replace(tzinfo=None))])


@pytest.mark.parametrize("damage", ["saved", "rejected", "replaced", "stage", "membership", "duplicate", "model_key", "absent"])
def test_public_report_omits_unqualified_evidence(damage):
    from server.reclive.forecasting import config, reporting

    key = config.model_unit_key(1186, "__all__")
    bundle = {"p50": object()}
    evidence = terminal_fixture(key)
    run = {"reporting_evidence": evidence, "reporting_selected_bundle": bundle}
    status, members = "trained_and_saved", [11, 22]
    if damage == "saved":
        status = "using_saved_model"
    elif damage == "rejected":
        status = "champion_kept_by_gate"
    elif damage == "replaced":
        bundle = {"p50": object()}
    elif damage == "stage":
        evidence["stage"] = "recent_window"
    elif damage == "membership":
        members = [11]
    elif damage == "duplicate":
        evidence["rows"].append(evidence["rows"][0])
    elif damage == "model_key":
        evidence["model_key"] = "wrong-model"
    elif damage == "absent":
        run = {"val_mae": .1, "val_rmse": .2}
    result, context = reporting.build_public_metrics({key: bundle}, {key: status}, {key: run}, {key: members})
    assert result.mae_people is None
    assert result.rolling_holdout_by_facility == {}
    assert context["observationCounts"]["maePeople"] == 0


def test_public_report_selects_each_facility_all_model_once_and_keeps_ratio_telemetry():
    from server.reclive.forecasting import config, reporting

    models, statuses, runs, members = {}, {}, {}, {}
    for facility in (1186, 1656):
        for category in ("__all__", "fitness"):
            key = config.model_unit_key(facility, category)
            models[key] = {"p50": object()}
            statuses[key] = "trained_and_saved"
            runs[key] = {"reporting_evidence": terminal_fixture(key, facility),
                         "reporting_selected_bundle": models[key], "val_mae": .1, "val_rmse": .2}
            members[key] = [11, 22]
    result, context = reporting.build_public_metrics(models, statuses, runs, members)
    assert result.observation_counts["maePeople"] == 6
    assert set(result.rolling_holdout_by_facility) == {1186, 1656}
    assert all(run["val_mae"] == .1 and run["val_rmse"] == .2 for run in runs.values())
    assert "weighted occupancy ratios" in context["legacyTelemetryUnits"]
    assert "full-history priors" in " ".join(context["limitations"])


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, *args):
        pass

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return None


class FakeConnection:
    def __init__(self, rows=()):
        self.rows = rows
        self.closed = False

    def cursor(self):
        return FakeCursor(self.rows)

    def close(self):
        self.closed = True


def test_history_sidecars_precede_schedule_and_cleaning_without_changing_model_values(monkeypatch):
    from server.reclive.forecasting import config, data, features

    monkeypatch.setattr(config, "location_to_facility_map", lambda: {11: 1186})
    monkeypatch.setattr(config, "SCHEDULE_BOUNDARY_ZERO_ENABLED", True)
    monkeypatch.setattr(config, "RESAMPLE_MINUTES", 15)
    monkeypatch.setattr(features, "get_facility_schedule_open_state", lambda **kwargs: True)
    monkeypatch.setattr(features, "get_facility_schedule_boundary_state", lambda **kwargs: (True, False))
    monkeypatch.setattr(features, "drop_impossible_jumps", lambda rows, **kwargs: (rows, 0))
    monkeypatch.setattr(features, "drop_flatline_plateaus", lambda rows, **kwargs: (rows, 0, 0))
    observed = START - timedelta(hours=1)
    conn = FakeConnection([(11, observed, observed + timedelta(minutes=1), 260., False, 200)])
    loc_data, *_ = data.load_history(conn, {1186: {"sections": []}})
    location = loc_data[11]
    assert location["raw_values"] == [0.]
    assert location["bucket_values"] == [0.]
    assert list(location["reporting_bucket_people"].values()) == [0.]
    assert location["reporting_raw_baseline"][0].people == 260.
    loc_data, *_ = data.load_history(conn)
    assert loc_data[11]["bucket_values"] == [1.2]
    assert list(loc_data[11]["reporting_bucket_people"].values()) == [260.]


def test_dataset_sidecar_aligns_to_location_major_rows_and_keeps_features_identical(monkeypatch):
    import numpy as np
    from server.reclive.forecasting import features

    monkeypatch.setattr(features, "build_features", lambda *args, **kwargs: [3., 4.])
    monkeypatch.setattr(features, "sensor_quality_signals", lambda **kwargs: (0., 0., 0.))
    monkeypatch.setattr(features, "schedule_transition_weight_for_location_target", lambda *args: 1.)
    times = [START, START + timedelta(hours=1)]
    locations = {location: {"bucket_map": dict.fromkeys(times, .5), "bucket_times": times,
                           "bucket_values": [.5, .5], "raw_times": times, "raw_values": [50., 50.],
                           "max_cap": capacity, "reporting_facility_id": 1186,
                           "reporting_bucket_people": dict.fromkeys(times, capacity / 2)}
                 for location, capacity in ((11, 200), (22, 100))}
    enriched = features.build_model_observation_dataset([11, 22], locations, {11: [1., 0.], 22: [0., 1.]}, None)
    bare_locations = {location: {key: value for key, value in row.items() if not key.startswith("reporting_")}
                      for location, row in locations.items()}
    bare = features.build_model_observation_dataset([11, 22], bare_locations, {11: [1., 0.], 22: [0., 1.]}, None)
    for key in enriched.keys() - {"reportingRows"}:
        if isinstance(enriched[key], np.ndarray):
            np.testing.assert_array_equal(enriched[key], bare[key])
        else:
            assert enriched[key] == bare[key]
    assert [(row["location_id"], row["capacity"], row["actual_people"]) for row in enriched["reportingRows"]] == [
        (11, 200, 100), (11, 200, 100), (22, 100, 50), (22, 100, 50)]


def fake_model_training(monkeypatch):
    """Execute the existing orchestration with fake matrices/predictors, never XGBoost."""
    import numpy as np
    from types import SimpleNamespace
    from server.reclive.forecasting import config, features, training

    times = [START + timedelta(hours=hour) for _location in (11, 22) for hour in range(6)]
    locations = [11] * 6 + [22] * 6
    labels = np.array([.5] * 12, dtype=np.float32)
    labels[0] = np.nan
    metadata = [{"location_id": location, "facility_id": 1186, "capacity": 200 if location == 11 else 100,
                 "time_aligned": True,
                 "actual_people": 260. if location == 11 else 50., "target": target}
                for location, target in zip(locations, times)]
    dataset = {"X": np.ones((12, 4), dtype=np.float32), "y": labels,
               "rowQualityWeights": np.ones(12, dtype=np.float32), "times": times,
               "reportingRows": metadata}
    monkeypatch.setattr(features, "build_model_observation_dataset", lambda **kwargs: dataset)
    monkeypatch.setattr(config, "MIN_TRAIN_SAMPLES", 3)
    monkeypatch.setattr(config, "MODEL_HOLDOUT_MIN_ROWS", 2)
    monkeypatch.setattr(config, "MODEL_HOLDOUT_SPLIT", .3)
    monkeypatch.setattr(config, "MODEL_FEATURE_MISSING_GUARD_ENABLED", False)
    monkeypatch.setattr(config, "MODEL_FEATURE_CLIP_ENABLED", False)
    monkeypatch.setattr(training, "choose_best_params", lambda *args, **kwargs: ({}, None, 0))
    calls = []

    def matrix(values, label=None, weight=None):
        calls.append((np.array(values), None if label is None else np.array(label)))
        return np.array(values)

    class Predictor:
        best_iteration = 0

        def predict(self, values):
            return np.full(len(values), .4, dtype=np.float32)

    monkeypatch.setattr(training, "xgb", SimpleNamespace(DMatrix=matrix, train=lambda *args, **kwargs: Predictor()))
    monkeypatch.setattr(training, "train_quantile_model", lambda *args, **kwargs: None)
    for name in ("build_interval_profile", "build_point_bias_profile", "build_regime_mae_profile", "build_direct_horizon_profile"):
        monkeypatch.setattr(training, name, lambda *args, **kwargs: None)
    return training, dataset, calls


def test_terminal_capture_uses_same_mask_and_split_without_training(monkeypatch):
    import numpy as np
    training, dataset, calls = fake_model_training(monkeypatch)
    bundle, result, _ = training.train_model_unit("1186:__all__", [11, 22], {}, {}, {11: 6, 22: 6}, None)
    assert bundle is not None
    assert result["invalid_rows_dropped"] == 1
    evidence = result["reporting_evidence"]
    assert evidence is not None
    assert [(row.location_id, row.target.hour) for row in evidence["rows"]] == [(11, 4), (11, 5), (22, 4), (22, 5)]
    assert evidence["windows"][0].start == START + timedelta(hours=4)
    assert [row.actual_people for row in evidence["rows"]] == [260., 260., 50., 50.]
    assert [row.predicted_people for row in evidence["rows"]] == pytest.approx([80., 80., 40., 40.])
    assert calls[0][0].shape[0] == 7
    np.testing.assert_array_equal(calls[0][1], np.full(7, .5, dtype=np.float32))
    assert result["holdout_mae"] == pytest.approx(.1)
    assert dataset["y"].shape == (12,) and np.isnan(dataset["y"][0])


@pytest.mark.parametrize("bad", ["length", "timestamp", "capacity", "facility"])
def test_misaligned_terminal_metadata_gives_no_evidence(monkeypatch, bad):
    training, dataset, _ = fake_model_training(monkeypatch)
    if bad == "length":
        dataset["reportingRows"].pop()
    elif bad == "timestamp":
        dataset["reportingRows"][-1]["target"] = START
    elif bad == "capacity":
        dataset["reportingRows"][-1]["capacity"] = 0
    elif bad == "facility":
        dataset["reportingRows"][-1]["facility_id"] = 999
    _, result, _ = training.train_model_unit("1186:__all__", [11, 22], {}, {}, {11: 6, 22: 6}, None)
    assert result["reporting_evidence"] is None


@pytest.mark.parametrize("mode", ["accepted", "saved", "rejected", "skipped", "rollback"])
def test_build_forecast_propagates_real_prepare_paths_and_serializes_public_units(monkeypatch, tmp_path, mode):
    from server.reclive.forecasting import config, data, job, prediction, reporting, training

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return START.astimezone(tz or UTC)

    monkeypatch.setattr(job, "datetime", Clock)
    monkeypatch.setattr(config, "FACILITIES", {
        1186: {"name": "Fixture facility", "categories": [
            {"key": "fixture", "title": "Fixture", "location_ids": [11, 22]},
        ]},
    })
    monkeypatch.setattr(config, "FORECAST_CATEGORY_KEYS", set())
    monkeypatch.setattr(config, "MODEL_PARALLEL_WORKERS", 1)
    monkeypatch.setattr(config, "CHAMPION_GATE_ENABLED", False)
    monkeypatch.setattr(config, "FORECAST_JSON_PATH", str(tmp_path / "forecast.json"))
    monkeypatch.setattr(config, "should_train_category", lambda category: False)
    monkeypatch.setattr(data, "collect_saved_meta_snapshots", lambda: {})
    monkeypatch.setattr(data, "load_schedule_sections_by_facility", lambda: {})
    connection = FakeConnection()
    monkeypatch.setattr(data, "db_connect", lambda: connection)
    monkeypatch.setattr(data, "load_history", lambda *args, **kwargs: ({}, {}, {}, {}, {}, {}, {}))
    monkeypatch.setattr(data, "weather_history_start", lambda *args: START)
    monkeypatch.setattr(data, "fetch_weather_history_series", lambda *args: {"times": [], "map": {}})
    monkeypatch.setattr(data, "fetch_weather_forecast_series", lambda *args: {"times": [], "map": {}})
    monkeypatch.setattr(prediction, "get_targets_for_date", lambda *args: [])
    monkeypatch.setattr(prediction, "get_window_targets_for_date", lambda *args: [])
    monkeypatch.setattr(reporting, "build_data_quality_alerts", lambda **kwargs: {"blockTraining": False})
    monkeypatch.setattr(prediction, "compute_model_drift", lambda **kwargs: {})
    monkeypatch.setattr(prediction, "compute_interval_conformal_profiles", lambda **kwargs: ({}, {}))
    key = config.model_unit_key(1186, "__all__")
    previous_bundle = {"p50": object()}
    previous_meta = {"valMae": .1, "valRmse": .2, "valRows": 20, "locIds": [11, 22],
                     "featureCount": 4, "trainedAt": START.isoformat()}
    saved_writes = []
    monkeypatch.setattr(data, "save_model_artifacts", lambda bundle, meta, **kwargs: saved_writes.append(dict(meta)))
    monkeypatch.setattr(data, "backup_current_artifacts", lambda **kwargs: True)
    monkeypatch.setattr(data, "load_saved_model", lambda **kwargs: (None, None) if mode in ("accepted", "skipped")
                        else (previous_bundle, dict(previous_meta)))
    monkeypatch.setattr(training, "should_retrain_model", lambda *args, **kwargs: mode != "saved")
    monkeypatch.setattr(training, "passes_guardrail", lambda *args: mode != "rejected")
    train_calls = []

    def candidate(**kwargs):
        train_calls.append(kwargs["model_key"])
        return (None if mode == "skipped" else {"p50": object()}), {
            "train_rows": 100, "val_rows": 20, "val_mae": .05, "val_rmse": .08,
            "reporting_evidence": terminal_fixture(kwargs["model_key"]),
        }, None

    monkeypatch.setattr(training, "train_model_unit", candidate)
    monkeypatch.setattr(prediction, "apply_drift_actions", lambda **kwargs: ({}, {
        "byModel": {key: {"rolledBack": True}} if mode == "rollback" else {},
    }))
    payload = job.build_forecast()
    job.write_forecast(payload)
    serialized = (tmp_path / "forecast.json").read_text()
    decoded = json.loads(serialized)
    assert "precisionPct" not in serialized
    assert "reporting_evidence" not in serialized and "reporting_selected_bundle" not in serialized
    info = decoded["modelInfo"]
    assert info["metricContext"]["independentBacktest"] is False
    assert "absent, unparseable or mixed alignment suppresses" in info["metricContext"]["timestampAlignment"]
    assert connection.closed
    if mode == "accepted":
        assert info["metrics"]["maePeople"] == pytest.approx(50 / 3)
        assert info["valMae"] == info["metrics"]["maePeople"]
        assert info["valRmse"] == info["metrics"]["rmsePeople"]
        assert info["byFacility"]["1186"]["valMae"] == .05
        assert saved_writes[0]["valMae"] == .05 and saved_writes[0]["valRmse"] == .08
        assert all(not name.startswith("reporting") for name in saved_writes[0])
    else:
        assert info["metrics"]["maePeople"] is None
        assert info["metrics"]["rollingHoldoutByFacility"] == {}
        assert info["valMae"] is None and info["valRmse"] is None
    assert train_calls == ([] if mode == "saved" else [key])


def test_main_logs_explicit_units_through_owner_seams(monkeypatch, capsys):
    import forecast_job
    from server.reclive.forecasting import job

    writes = []
    monkeypatch.setattr(job, "validate_production_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr(job, "build_forecast", lambda: {"facilities": [], "modelInfo": {"metrics": compute().to_payload()}})
    monkeypatch.setattr(job, "write_forecast", lambda payload: writes.append(payload))
    assert forecast_job.main() == 0
    output = capsys.readouterr().out
    event = json.loads(output)
    assert event.pop("timestamp").endswith("Z")
    assert event == {"event": "forecast.completed", "generatedFacilities": 0}
    assert len(writes) == 1
    metrics = writes[0]["modelInfo"]["metrics"]
    assert metrics["maePeople"] == 15.0
    assert metrics["maeCapacityPercentagePoints"] == 12.5
    assert metrics["rmsePeople"] == pytest.approx(math.sqrt(250.0))
    assert metrics["simpleBaselineMaePeople"] == 10.0


def test_saved_artifact_loader_preserves_ratio_metadata_and_legacy_paths(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from server.reclive.forecasting import config, data, reporting

    key = config.model_unit_key(1186, "__all__")
    monkeypatch.setattr(config, "MODEL_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "MODEL_BASENAME", "fixture")
    paths = data.model_artifact_paths(key)
    assert paths[0] == str(tmp_path / "fixture_1186___all.p50.xgb.json")
    meta = {"schemaVersion": config.MODEL_SCHEMA_VERSION, "locIds": [11, 22],
            "featureCount": 4, "modelKey": key, "valMae": .031, "valRmse": .052}
    # File contents are synthetic; the fake loader ensures there is no booster runtime.
    (tmp_path / Path(paths[0]).name).write_text("{}")
    (tmp_path / Path(paths[3]).name).write_text(json.dumps(meta))
    loaded = []
    monkeypatch.setattr(data, "xgb", SimpleNamespace(Booster=lambda: SimpleNamespace(load_model=lambda path: loaded.append(path))))
    bundle, saved = data.load_saved_model(key, [11, 22], 4)
    assert loaded == [paths[0]]
    assert saved == meta
    assert bundle["p50"] is not None
    result, _ = reporting.build_public_metrics({key: bundle}, {key: "using_saved_model"}, {}, {key: [11, 22]})
    assert result.mae_people is None and result.rolling_holdout_by_facility == {}


@pytest.mark.parametrize("source_kind,db_zone,aligned", [
    ("naive", "America/Chicago", False),
    ("aware", "America/Chicago", True),
    ("naive", "UTC", True),
    ("missing", "UTC", False),
    ("unparseable", "UTC", False),
    ("mixed", "America/Chicago", False),
])
def test_canonical_source_alignment_gates_terminal_reporting_without_changing_model_arrays(
    monkeypatch, source_kind, db_zone, aligned,
):
    import numpy as np
    import pytz
    from server.reclive.forecasting import config, data, features, metrics, reporting

    monkeypatch.setattr(config, "DB_TZ", pytz.timezone(db_zone))
    monkeypatch.setattr(config, "TZ", pytz.timezone("America/Chicago"))
    monkeypatch.setattr(config, "RESAMPLE_MINUTES", 15)
    monkeypatch.setattr(config, "SCHEDULE_BOUNDARY_ZERO_ENABLED", False)
    monkeypatch.setattr(config, "location_to_facility_map", lambda: {11: 1186})
    monkeypatch.setattr(features, "drop_impossible_jumps", lambda rows, **kwargs: (rows, 0))
    monkeypatch.setattr(features, "drop_flatline_plateaus", lambda rows, **kwargs: (rows, 0, 0))
    monkeypatch.setattr(features, "build_features", lambda *args, **kwargs: [3., 4.])
    monkeypatch.setattr(features, "sensor_quality_signals", lambda **kwargs: (0., 0., 0.))
    monkeypatch.setattr(features, "schedule_transition_weight_for_location_target", lambda *args: 1.)
    baseline_observed = START - timedelta(hours=1)
    target_observed = START
    if source_kind in ("naive", "mixed"):
        baseline_observed = baseline_observed.replace(tzinfo=None)
    if source_kind == "naive":
        target_observed = target_observed.replace(tzinfo=None)
    elif source_kind == "missing":
        baseline_observed = None
    elif source_kind == "unparseable":
        baseline_observed = "unparseable-synthetic-timestamp"
    # The baseline was backdated but first received after the true midnight start.
    baseline_fetched = (START + timedelta(hours=1)).replace(tzinfo=None)
    target_fetched = (START + timedelta(minutes=5)).replace(tzinfo=None)
    source_rows = [(11, baseline_observed, baseline_fetched, 40., False, 100),
                   (11, target_observed, target_fetched, 50., False, 100)]
    location = data.load_history(FakeConnection(source_rows))[0][11]
    expected_pairs = sorted([
        (features.parse_observed_at_value(observed) if features.parse_observed_at_value(observed) is not None
         else features.to_local(fetched), count)
        for _, observed, fetched, count, _, _ in source_rows
    ])
    expected_times = [features.floor_time(target, 15) for target, _ in expected_pairs]
    assert location["raw_times"] == [target for target, _ in expected_pairs]
    assert location["raw_values"] == [count for _, count in expected_pairs]
    assert location["bucket_times"] == expected_times
    assert location["bucket_values"] == [count / 100 for _, count in expected_pairs]
    enriched = features.build_model_observation_dataset([11], {11: location}, {11: [1.]}, None)
    bare_location = {key: value for key, value in location.items() if not key.startswith("reporting_")}
    bare = features.build_model_observation_dataset([11], {11: bare_location}, {11: [1.]}, None)
    for key in enriched.keys() - {"reportingRows"}:
        if isinstance(enriched[key], np.ndarray):
            np.testing.assert_array_equal(enriched[key], bare[key])
        else:
            assert enriched[key] == bare[key]
    selected_time = features.floor_time(features.parse_observed_at_value(target_observed), 15)
    selected_index = enriched["times"].index(selected_time)
    evidence = metrics.terminal_evaluation_evidence(
        "1186::__all__", enriched["reportingRows"], enriched["times"], [selected_index],
        [.5], [.4], [.6], selected_time, 15,
    )
    assert (evidence is not None) is aligned
    assert location["reporting_time_aligned"] is aligned
    assert all(row["time_aligned"] is aligned for row in enriched["reportingRows"])
    if evidence is not None:
        assert evidence["windows"][0].start == START
        assert evidence["rows"][0].baseline_people is None
    key = "1186::__all__"
    selected_bundle = {"p50": object()}
    public, _ = reporting.build_public_metrics(
        {key: selected_bundle}, {key: "trained_and_saved"},
        {key: {"reporting_evidence": evidence, "reporting_selected_bundle": selected_bundle}},
        {key: [11]},
    )
    if aligned:
        assert public.mae_people == 0.
    else:
        assert public.mae_people is None
        assert public.rolling_holdout_by_facility == {}


@pytest.mark.parametrize("provenance", [None, False, "true", 1])
def test_terminal_metadata_requires_explicit_true_alignment(provenance):
    from server.reclive.forecasting.metrics import terminal_evaluation_evidence

    metadata = [{"facility_id": 1186, "location_id": 11, "target": START,
                 "capacity": 100., "actual_people": 50., "time_aligned": provenance}]
    assert terminal_evaluation_evidence("1186::__all__", metadata, [START], [0],
                                        [.5], [.4], [.6], START, 15) is None
