"""Synthetic configuration and failure-boundary controls; no model training or I/O."""

import ast
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MARKER = "private-logging-marker"


def isolated(code, overrides=None):
    from server.env_loader import DEFAULT_ENV

    setup = (
        "import sys; sys.path.insert(0, " + repr(str(ROOT)) + "); "
        "from server import env_loader; env_loader._DOTENV_STATE.loaded = True; "
    )
    return subprocess.run(
        [sys.executable, "-c", setup + code], cwd=ROOT,
        env={"PATH": os.defpath, "APP_ENV": "test", **DEFAULT_ENV, **(overrides or {})},
        capture_output=True, text=True, timeout=20,
    )


@pytest.mark.parametrize("name,reason", [
    ("GYM_RESAMPLE_MINUTES", "invalid_integer"),
    ("MODEL_ETA", "invalid_float"),
    ("CROWD_BAND_MEDIUM_BRIDGE_MIN", "invalid_integer"),
    ("GYM_WEATHER_LAT", "invalid_float"),
    ("GYM_DB_TIMEZONE", "invalid_timezone"),
])
@pytest.mark.parametrize("value", [MARKER, ""])
def test_direct_configuration_errors_are_one_private_value_free_event(name, reason, value):
    result = isolated("import runpy; runpy.run_path('server/forecast_job.py', run_name='__main__')", {name: value})
    assert result.returncode == 1
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload.pop("timestamp").endswith("Z")
    expected_name = "CROWD_BAND_BRIDGE_MIN" if name == "CROWD_BAND_MEDIUM_BRIDGE_MIN" else name
    if value == "" and name in {"GYM_DB_TIMEZONE", "GYM_WEATHER_LAT"}:
        reason = "missing"
    assert payload == {"event": "forecast.configuration_failed", "configurationName": expected_name, "reason": reason}
    assert MARKER not in result.stdout + result.stderr


def test_library_import_still_fails_and_suppresses_rejected_float_cause():
    result = isolated("""
import traceback
try:
    import server.forecast_job
except RuntimeError as error:
    print('caught')
    print('private-logging-marker' in ''.join(traceback.format_exception(error)))
    print(error.__suppress_context__)
else:
    raise AssertionError('import unexpectedly succeeded')
""", {"GYM_WEATHER_LAT": MARKER})
    assert result.returncode == 0
    assert result.stdout == "caught\nFalse\nTrue\n"
    assert result.stderr == ""


@pytest.mark.parametrize("overrides,expected", [
    ({}, [15, 30, 90, .05, 1.35, "UTC"]),
    ({"GYM_RESAMPLE_MINUTES": " +2 ", "GYM_WINDOW_RESAMPLE_MINUTES": "-1", "CROWD_BAND_MEDIUM_BRIDGE_MIN": "-1", "MODEL_ETA": " 1e-2 ", "SCHEDULE_TRANSITION_WEIGHT_MULTIPLIER": ".5"}, [5, 5, 0, .01, 1.0, "UTC"]),
    ({"CROWD_BAND_BRIDGE_MIN": "12", "CROWD_BAND_MEDIUM_BRIDGE_MIN": MARKER, "GYM_DB_TIMEZONE": "America/Chicago"}, [15, 30, 12, .05, 1.35, "America/Chicago"]),
])
def test_configuration_defaults_alias_clamps_and_timezone_are_preserved(overrides, expected):
    result = isolated("""
import json, inspect
from server.reclive.forecasting import config as c
import server.forecast_job as wrapper
for name in ('require_env','require_int_env','require_float_env','require_path_env'):
    assert getattr(wrapper, name) is getattr(c, name)
    assert list(inspect.signature(getattr(c, name)).parameters) == ['name']
print(json.dumps([c.RESAMPLE_MINUTES,c.WINDOW_RESAMPLE_MINUTES,c.CROWD_BAND_BRIDGE_MIN,c.MODEL_ETA,c.SCHEDULE_TRANSITION_WEIGHT_MULTIPLIER,c.DB_TZ.zone]))
""", {"GYM_DB_TIMEZONE": "UTC", **overrides})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected
    assert result.stderr == ""


def normalized_ast_dump(expression):
    """Ignore only Call's empty-keyword dump spelling, never literal contents."""
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Call":
            node.keywords = [
                keyword for keyword in node.keywords
                if not (keyword.arg == "keywords" and isinstance(keyword.value, ast.List) and not keyword.value.elts)
            ]
    return ast.dump(tree)


def test_ast_dump_normalization_accepts_empty_keyword_representations():
    omitted = "Call(func=Name(id='f', ctx=Load()), args=[Constant(value='value')])"
    explicit = "Call(func=Name(id='f', ctx=Load()), args=[Constant(value='value')], keywords=[])"
    assert normalized_ast_dump(omitted) == normalized_ast_dump(explicit)


def test_ast_dump_normalization_preserves_semantic_changes():
    original = "max(5, int(os.getenv('RATE', '15')))"
    for changed in (
        "min(5, int(os.getenv('RATE', '15')))",
        "max(6, int(os.getenv('RATE', '15')))",
        "max(5, float(os.getenv('RATE', '15')))",
        "max(5, int(os.getenv('OTHER_RATE', '15')))",
        "max(5, int(os.getenv('RATE', '16')))",
        "max(5, int(os.getenv('RATE', '15'), base=10))",
    ):
        assert normalized_ast_dump(ast.dump(ast.parse(original, mode="eval"))) != normalized_ast_dump(ast.dump(ast.parse(changed, mode="eval")))
    for left, right in (
        ("f(option=[])", "f()"),
        ("'literal, keywords=[]'", "'literal'"),
    ):
        assert normalized_ast_dump(ast.dump(ast.parse(left, mode="eval"))) != normalized_ast_dump(ast.dump(ast.parse(right, mode="eval")))


def test_all_numeric_initializers_preserve_original_ast_contract():
    baseline = json.loads((ROOT / "tests/fixtures/forecast_config_numeric_baseline.json").read_text())
    tree = ast.parse((ROOT / "server/reclive/forecasting/config.py").read_text())
    assignments = {node.targets[0].id: node.value for node in tree.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}

    class RestoreParser(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in {"int_env_with_default", "float_env_with_default"}:
                self.count += 1
                return ast.Call(func=ast.Name(id=node.func.id.split("_")[0], ctx=ast.Load()), args=[ast.Call(func=ast.Attribute(value=ast.Name(id="os", ctx=ast.Load()), attr="getenv", ctx=ast.Load()), args=node.args, keywords=[])], keywords=[])
            return node

    restore = RestoreParser()
    for expression in assignments.values():
        for node in ast.walk(expression):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"int", "float"}:
                assert not any(isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute) and arg.func.attr == "getenv" for arg in node.args)
    for row in baseline:
        assert normalized_ast_dump(ast.dump(restore.visit(assignments[row["symbol"]]))) == normalized_ast_dump(row["expression"])
    assert len(baseline) == restore.count == 156


def test_numeric_parsers_keep_python_acceptance_for_nonfinite_values():
    result = isolated("""
import math
from server.reclive.forecasting import config as c
assert math.isnan(c.MODEL_ETA)
assert math.isinf(c.MODEL_SUBSAMPLE) and c.MODEL_SUBSAMPLE < 0
print('accepted')
""", {"MODEL_ETA": "NaN", "MODEL_SUBSAMPLE": "-Infinity"})
    assert result.returncode == 0, result.stderr
    assert result.stdout == "accepted\n"


@pytest.fixture
def owners(monkeypatch):
    from server import env_loader
    for name, value in env_loader.DEFAULT_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(env_loader._DOTENV_STATE, "loaded", True)
    from server.reclive.forecasting import config, training, prediction
    return config, training, prediction


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("exception", [RuntimeError(MARKER), OSError(MARKER)])
def test_model_unit_errors_are_private_and_other_units_continue(owners, monkeypatch, capsys, workers, exception):
    config, training, _ = owners
    monkeypatch.setattr(config, "FACILITIES", {1186: {"categories": []}, 1656: {"categories": []}})
    monkeypatch.setattr(config, "MODEL_PARALLEL_WORKERS", workers)
    calls = []
    bundle, metadata = {"fixture": True}, {"valMae": .05}

    def prepare(**kwargs):
        calls.append(kwargs["facility_id"])
        if kwargs["facility_id"] == 1186:
            raise exception
        return bundle, metadata, "using_saved_model", {"val_mae": .05}

    monkeypatch.setattr(training, "prepare_model", prepare)
    result = training.prepare_models(datetime.now(timezone.utc), {}, {}, None)
    failed_key, good_key = config.model_unit_key(1186, "__all__"), config.model_unit_key(1656, "__all__")
    assert sorted(calls) == [1186, 1656]
    assert result[0] == {good_key: bundle}
    assert result[1] == {good_key: metadata}
    assert result[2] == {failed_key: "error", good_key: "using_saved_model"}
    assert result[3] == {failed_key: {}, good_key: {"val_mae": .05}}
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert set(payload) == {"event", "timestamp"}
    assert payload["event"] == "forecast.model_unit_failed"
    assert MARKER not in captured.err


@pytest.mark.parametrize("rollback", [False, True])
@pytest.mark.parametrize("exception", [RuntimeError(MARKER), OSError(MARKER)])
def test_drift_metadata_failure_preserves_decisions_and_memory(owners, monkeypatch, capsys, rollback, exception):
    config, _, prediction = owners
    monkeypatch.setattr(config, "DRIFT_ACTIONS_ENABLED", True)
    monkeypatch.setattr(config, "CHAMPION_ROLLBACK_ENABLED", rollback)
    monkeypatch.setattr(config, "CHAMPION_ROLLBACK_DRIFT_STREAK", 1)
    monkeypatch.setattr(prediction.data, "rollback_to_previous_model", lambda **kwargs: {"restored": True})
    def fail(*args, **kwargs):
        raise exception
    monkeypatch.setattr(prediction.data, "save_model_meta_only", fail)
    metadata = {"private-model-key": {"locIds": [1], "featureCount": 1}}
    multipliers, summary = prediction.apply_drift_actions(
        datetime(2026, 9, 1, tzinfo=timezone.utc), {"byModel": {"private-model-key": {"alert": True}}},
        metadata, action_streak_for_retrain=1,
    )
    assert summary["modelsEvaluated"] == 1
    assert summary["modelsRolledBack"] == int(rollback)
    assert summary["modelsForcedRetrain"] == int(not rollback)
    assert summary["byModel"]["private-model-key"]["rolledBack"] is rollback
    assert len(metadata["private-model-key"]["driftHistory"]) == 1
    if rollback:
        assert metadata["private-model-key"]["restored"] is True
        assert multipliers == {"private-model-key": 1.0}
    else:
        assert metadata["private-model-key"]["forceRetrain"] is True
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload.pop("timestamp").endswith("Z")
    assert payload == {"event": "forecast.model_metadata_write_failed", "errorCategory": "file_unavailable"}
    assert MARKER not in captured.err and "private-model-key" not in captured.err


@pytest.mark.parametrize("exception", [RuntimeError(MARKER), OSError(MARKER)])
def test_champion_metadata_failure_keeps_champion_and_gate(owners, monkeypatch, capsys, exception):
    _, training, _ = owners
    saved, candidate = {"saved": True}, {"candidate": True}
    monkeypatch.setattr(training.data, "load_saved_model", lambda **kwargs: (saved, {"valMae": .05}))
    monkeypatch.setattr(training, "should_retrain_model", lambda *args, **kwargs: True)
    monkeypatch.setattr(training, "train_model_unit", lambda **kwargs: (candidate, {"val_mae": .06}, {}))
    monkeypatch.setattr(training.features, "build_model_observation_dataset", lambda **kwargs: {})
    monkeypatch.setattr(training, "evaluate_model_bundle_on_recent_window", lambda **kwargs: {})
    monkeypatch.setattr(training, "champion_gate_decision", lambda **kwargs: {"promote": False})
    def fail(*args, **kwargs):
        raise exception
    monkeypatch.setattr(training.data, "save_model_meta_only", fail)
    now = datetime(2026, 9, 1, tzinfo=timezone.utc)
    bundle, meta, status, metrics = training.prepare_model(now, "private-model-key", 1186, "__all__", [1], {1: [1.]}, {}, {}, None)
    assert bundle is saved
    assert status == "champion_kept_by_gate"
    assert meta == {"valMae": .05, "lastChampionGate": {"promote": False}, "lastChampionGateAt": now.isoformat()}
    assert metrics == {"val_mae": .06, "champion_gate": {"promote": False}}
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload.pop("timestamp").endswith("Z")
    assert payload == {"event": "forecast.model_metadata_write_failed", "errorCategory": "file_unavailable"}
    assert MARKER not in captured.err and "private-model-key" not in captured.err
