import json
from datetime import datetime, timezone

import pytest

from server.reclive.observability import log_event


def test_log_event_emits_only_allowlisted_scalar_fields(capsys):
    log_event("ingestion.completed", receivedCount=8, historyInsertedCount=7, unchangedCount=1)
    payload = json.loads(capsys.readouterr().out)
    assert payload.pop("event") == "ingestion.completed"
    stamp = payload.pop("timestamp")
    assert stamp.endswith("Z")
    assert datetime.fromisoformat(stamp).utcoffset() == timezone.utc.utcoffset(None)
    assert payload == dict(receivedCount=8, historyInsertedCount=7, unchangedCount=1)


@pytest.mark.parametrize("fields", [
    dict(endpoint="https://private.invalid", password="private", requestBody={}, errorCategory="network_error"),
    dict(errorCategory=[]), dict(errorCategory={}), dict(errorCategory="private"),
    dict(errorCategory="x" * 241), dict(errorCategory=True),
])
def test_log_event_rejects_sensitive_or_structured_fields(capsys, fields):
    with pytest.raises(ValueError, match="^invalid operational event fields$"):
        log_event("ingestion.failed", **fields)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("event", ["dbHost.changed", [], {}, None, 1])
def test_log_event_rejects_unknown_event_names(capsys, event):
    with pytest.raises(ValueError, match="^unknown operational event$"):
        log_event(event)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("value", [True, -1, "8", None, [], {}, float("nan"), float("inf"), -float("inf"), 1.2])
def test_log_event_rejects_invalid_counts_without_partial_output(capsys, value):
    with pytest.raises(ValueError, match="^invalid operational event fields$"):
        log_event("ingestion.completed", receivedCount=2, unchangedCount=value)
    assert capsys.readouterr() == ("", "")


def test_configuration_diagnostics_are_closed(capsys):
    log_event("forecast.configuration_failed", configurationName="GYM_DB_TIMEZONE", reason="invalid_timezone")
    assert json.loads(capsys.readouterr().out)["reason"] == "invalid_timezone"
    for fields in [dict(configurationName="private", reason="invalid_float"), dict(configurationName="MODEL_ETA", reason="private"), dict(configurationName=[], reason="missing")]:
        with pytest.raises(ValueError, match="^invalid operational event fields$"):
            log_event("forecast.configuration_failed", **fields)
        assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("snapshot,history,expected", [(3, 2, 1), (1, 2, 0)])
def test_ingestion_unchanged_count_uses_valid_snapshot_heartbeats(snapshot, history, expected):
    from server.reclive.ingestion import IngestionRunResult, finish_ingestion_result
    events = []
    result = IngestionRunResult("succeeded", 8, 3, history, snapshot, None)
    assert finish_ingestion_result(result, None, lambda: None, events.append) is result
    assert len(events) == 1
    event = json.loads(events[0])
    assert event["receivedCount"] == 8
    assert event["historyInsertedCount"] == history
    assert event["unchangedCount"] == expected


@pytest.mark.parametrize("category,expected", [
    ("network", "network_error"), ("http", "network_error"),
    ("payload_not_list", "validation_error"), ("validation", "validation_error"),
    ("database", "database_unavailable"), ("transaction", "database_unavailable"),
])
def test_ingestion_failure_category_mapping(category, expected):
    from server.reclive.ingestion import IngestionRunResult, finish_ingestion_result
    events = []
    result = IngestionRunResult("failed", 0, 0, 0, 0, category)
    assert finish_ingestion_result(result, None, lambda: None, events.append) is result
    event = json.loads(events[0])
    assert event.pop("timestamp").endswith("Z")
    assert event == {"event": "ingestion.failed", "errorCategory": expected}


def test_failed_forecast_publication_does_not_emit_completion(monkeypatch, capsys):
    from server import env_loader
    for name, value in env_loader.DEFAULT_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(env_loader._DOTENV_STATE, "loaded", True)
    from server.reclive.forecasting import job
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setattr(job, "build_forecast", lambda: {"facilities": [{}, {}]})
    def fail(payload):
        assert payload == {"facilities": [{}, {}]}
        assert capsys.readouterr() == ("", "")
        raise OSError("private-artifact-marker")
    monkeypatch.setattr(job, "write_forecast", fail)
    assert job.main() == 1
    captured = capsys.readouterr()
    event = json.loads(captured.out)
    assert event.pop("timestamp").endswith("Z")
    assert event == {"event": "forecast.failed"}
    assert captured.err == ""


@pytest.mark.parametrize("stderr", [False, True])
def test_best_effort_event_preserves_continuation_when_stream_breaks(monkeypatch, stderr):
    from server.reclive import observability
    def fail(*args, **kwargs):
        raise OSError("private-output-error")
    monkeypatch.setattr("builtins.print", fail)
    observability.best_effort_event("forecast.model_unit_failed", stderr=stderr)


def test_unknown_configuration_error_uses_fieldless_event_without_stringifying(capsys):
    from server.env_loader import EnvironmentConfigurationError
    from server.reclive.observability import log_configuration_failure
    class Error(EnvironmentConfigurationError):
        def __str__(self):
            raise AssertionError("must not render exceptions")
    log_configuration_failure(Error("private-marker"))
    payload = json.loads(capsys.readouterr().out)
    assert payload.pop("timestamp").endswith("Z")
    assert payload == {"event": "forecast.failed"}


@pytest.mark.parametrize("category,expected", [
    ("anti_bot", "network_error"), ("upstream_timeout", "network_error"),
    ("upstream_http", "network_error"), ("wp_payload_invalid", "validation_error"),
    ("parse_empty", "validation_error"), ("schema_invalid", "validation_error"),
    ("io_error", "file_unavailable"),
])
def test_schedule_failures_use_closed_categories_on_stderr(monkeypatch, capsys, category, expected):
    import sys
    from server import facility_hours_fetch as command
    monkeypatch.setattr(command, "load_project_dotenv", lambda: None)
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setattr(sys, "argv", ["facility_hours_fetch.py", "--output", "synthetic.json"])
    def fail(settings):
        raise command.ScheduleFetchError(category) from RuntimeError("private-marker")
    monkeypatch.setattr(command, "run_facility_hours_fetch", fail)
    assert command.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    event = json.loads(captured.err)
    assert event.pop("timestamp").endswith("Z")
    assert event == {"event": "schedules.failed", "errorCategory": expected}
