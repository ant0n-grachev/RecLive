"""Real, tiny XGBoost serialization checks in isolated temporary directories."""

import json
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from server.reclive.forecasting import config, data


@pytest.fixture()
def artifact_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_ARTIFACT_DIR", str(tmp_path))
    matrix = xgb.DMatrix(np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32), label=[0.1, 0.2, 0.7, 0.9])
    model = xgb.train({"objective": "reg:squarederror", "max_depth": 1, "nthread": 1}, matrix, num_boost_round=2)
    meta = {"schemaVersion": config.MODEL_SCHEMA_VERSION, "locIds": [11], "featureCount": 1, "modelKey": "test"}
    return model, matrix, meta


def test_atomic_model_save_produces_reloadable_json_for_all_quantiles(artifact_model):
    model, matrix, meta = artifact_model
    bundle = {"p50": model, "p10": model, "p90": model}

    data.save_model_artifacts(bundle, meta, model_key="test")
    loaded, loaded_meta = data.load_saved_model("test", [11], 1)

    assert loaded is not None
    assert loaded_meta == meta
    for quantile, path in zip(("p50", "p10", "p90"), data.model_artifact_paths("test")[:3]):
        assert isinstance(json.loads(Path(path).read_text()), dict)
        np.testing.assert_array_equal(loaded[quantile].predict(matrix), model.predict(matrix))


@pytest.mark.parametrize("previous", [False, True])
@pytest.mark.parametrize("encoding", ["json", "ubj"])
def test_existing_artifacts_load_by_encoding_with_current_and_previous_names(artifact_model, previous, encoding):
    model, matrix, meta = artifact_model
    paths = data.model_previous_artifact_paths("test") if previous else data.model_artifact_paths("test")
    # Existing releases could write UBJSON bytes, then rename the file to .json.
    encoded = model.save_raw(raw_format=encoding)
    for path in paths[:3]:
        Path(path).write_bytes(encoded)
    Path(paths[3]).write_text(json.dumps(meta))
    loader = data.load_saved_previous_model if previous else data.load_saved_model

    loaded, loaded_meta = loader("test", [11], 1)

    assert loaded is not None
    assert loaded_meta == meta
    assert loaded["quantileDirect"] is True
    for quantile in ("p50", "p10", "p90"):
        np.testing.assert_array_equal(loaded[quantile].predict(matrix), model.predict(matrix))


def test_invalid_artifact_bytes_do_not_load_as_a_model(artifact_model):
    _model, _matrix, meta = artifact_model
    p50, _p10, _p90, meta_path = data.model_artifact_paths("test")
    Path(p50).write_bytes(b"invalid model")
    Path(meta_path).write_text(json.dumps(meta))

    assert data.load_saved_model("test", [11], 1) == (None, None)
