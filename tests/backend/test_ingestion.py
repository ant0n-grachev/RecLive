import os
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
import pymysql
import requests

import reclive.ingestion as ingestion_module
from reclive.ingestion import (
    IngestionRunResult,
    NormalizedLiveRow,
    run_ingestion,
    sanitize_ingestion_exception,
    sanitize_ingestion_error,
    validate_and_deduplicate_rows,
)
from reclive.occupancy_repository import (
    IngestionWriteCounts,
    SnapshotRepository,
    as_mysql_utc,
    to_aware_utc,
)
from tests.fixtures.live_counts import LIVE_ROWS
from tests.fixtures.reclive_fakes import LifecycleRepositoryFactory


FIXED_UTC_NOW = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)


def fixed_utc_now() -> datetime:
    return FIXED_UTC_NOW


def run_lifecycle(
    factory: LifecycleRepositoryFactory, payload_loader
):
    return run_ingestion(
        payload_loader,
        factory.connect,
        {5761: 100},
        fixed_utc_now,
        repository_factory=factory,
        event_sink=factory.event_lines.append,
    )


def test_empty_valid_set_records_failure_without_snapshot_write() -> None:
    factory = LifecycleRepositoryFactory()

    result = run_lifecycle(
        factory,
        lambda: [{"LocationId": "invalid", "LastCount": -1}],
    )

    assert result.status == "failed"
    assert result.error_category == "validation"
    assert factory.snapshot_writes == []
    assert factory.failed_runs == [
        ("validation", "No valid live rows were received")
    ]


def test_fetch_failure_uses_separate_failure_transaction() -> None:
    factory = LifecycleRepositoryFactory()

    result = run_lifecycle(
        factory,
        lambda: (_ for _ in ()).throw(RuntimeError("upstream failed")),
    )

    assert result.status == "failed"
    assert result.error_category == "network"
    assert factory.failure_transaction_count == 1
    assert factory.failed_runs == [("network", "Ingestion failure")]


def test_exception_sanitization_emits_only_fixed_safe_fields() -> None:
    factory = LifecycleRepositoryFactory()
    error = RuntimeError("https://host.test/path?password=secret")
    error.sentinel = object()

    result = run_lifecycle(
        factory,
        lambda: (_ for _ in ()).throw(error),
    )

    assert result.error_category == "network"
    assert len(factory.event_lines) == 1
    event = json.loads(factory.event_lines[0])
    assert event.pop("timestamp").endswith("Z")
    assert event == {"event": "ingestion.failed", "errorCategory": "network_error"}
    event_output = " ".join(factory.event_lines)
    assert all(
        token not in event_output
        for token in ("host.test", "https://", "password", "secret", "object at")
    )


def test_work_connection_commits_only_after_success_is_complete() -> None:
    factory = LifecycleRepositoryFactory()

    result = run_lifecycle(factory, lambda: LIVE_ROWS)

    assert result.status == "succeeded"
    assert factory.run_events == ["start", "commit", "close"]
    assert factory.work_events == ["persist", "complete_success", "commit", "close"]


def test_success_emits_the_same_fixed_field_schema() -> None:
    factory = LifecycleRepositoryFactory()

    result = run_lifecycle(factory, lambda: LIVE_ROWS)

    assert result == IngestionRunResult("succeeded", 1, 1, 1, 1, None)
    assert len(factory.event_lines) == 1
    event = json.loads(factory.event_lines[0])
    assert event.pop("timestamp").endswith("Z")
    assert event == {"event": "ingestion.completed", "receivedCount": 1, "historyInsertedCount": 1, "unchangedCount": 0}


def test_non_list_payload_uses_a_sanitized_payload_category() -> None:
    factory = LifecycleRepositoryFactory()

    result = run_lifecycle(factory, lambda: {"payload": "marker"})

    assert result.error_category == "payload_not_list"
    assert factory.failed_runs == [("payload_not_list", "Ingestion failure")]
    assert "marker" not in " ".join(factory.event_lines)


@pytest.mark.parametrize(
    ("failure_point", "expected_category"),
    [
        ("run_connect", "database"),
        ("run_repository", "database"),
        ("start", "database"),
        ("run_commit", "transaction"),
        ("run_close", "transaction"),
    ],
)
def test_initial_connection_run_and_close_failures_are_sanitized(
    failure_point: str, expected_category: str
) -> None:
    factory = LifecycleRepositoryFactory()
    factory.raise_on(failure_point, RuntimeError("credential-like detail"))

    result = run_lifecycle(factory, lambda: LIVE_ROWS)

    assert result.status == "failed"
    assert result.error_category == expected_category
    assert factory.all_connections_closed
    assert "credential-like" not in " ".join(factory.event_lines)


@pytest.mark.parametrize(
    ("failure_point", "expected_category"),
    [
        ("work_connect", "database"),
        ("work_repository", "database"),
        ("persist", "database"),
        ("complete_success", "database"),
        ("work_commit", "transaction"),
    ],
)
def test_work_failures_rollback_and_record_only_a_safe_category(
    failure_point: str, expected_category: str
) -> None:
    factory = LifecycleRepositoryFactory()
    factory.raise_on(failure_point, RuntimeError("credential-like detail"))

    result = run_lifecycle(factory, lambda: LIVE_ROWS)

    assert result.status == "failed"
    assert result.error_category == expected_category
    assert factory.all_connections_closed
    assert "credential-like" not in " ".join(factory.event_lines)


@pytest.mark.parametrize(
    "failure_point",
    [
        "rollback",
        "failure_connect",
        "failure_repository",
        "complete_failure",
        "failure_commit",
        "work_close",
        "failure_close",
    ],
)
def test_primary_failure_wins_when_cleanup_or_failure_recording_also_fails(
    failure_point: str,
) -> None:
    factory = LifecycleRepositoryFactory()
    factory.raise_on(failure_point, RuntimeError("credential-like detail"))

    result = run_lifecycle(
        factory,
        lambda: (_ for _ in ()).throw(RuntimeError("upstream failed")),
    )

    assert result.error_category == "network"
    assert factory.all_connections_closed
    assert "credential-like" not in " ".join(factory.event_lines)


def test_failed_work_is_closed_before_the_fresh_failure_transaction() -> None:
    factory = LifecycleRepositoryFactory()

    result = run_lifecycle(
        factory,
        lambda: (_ for _ in ()).throw(RuntimeError("upstream failed")),
    )

    assert result.error_category == "network"
    lifecycle = [
        event
        for event in factory.trace
        if event
        in {
            "work.rollback",
            "work.close",
            "failure.connect",
            "failure.complete_failure",
            "failure.commit",
        }
    ]
    assert lifecycle == [
        "work.rollback",
        "work.close",
        "failure.connect",
        "failure.complete_failure",
        "failure.commit",
    ]


def test_uncertain_work_release_never_starts_a_self_blocking_failure_update() -> None:
    factory = LifecycleRepositoryFactory()
    factory.raise_on("rollback", RuntimeError("controlled rollback failure"))
    factory.raise_on("work_close", RuntimeError("controlled close failure"))

    result = run_lifecycle(
        factory,
        lambda: (_ for _ in ()).throw(RuntimeError("upstream failed")),
    )

    assert result.error_category == "network"
    assert factory.trace.count("work.close") == 1
    assert "failure.connect" not in factory.trace
    assert "failure.self_block" not in factory.trace


def test_exception_sanitization_never_stringifies_the_exception() -> None:
    class UnstringableFailure(RuntimeError):
        def __str__(self) -> str:
            raise AssertionError("exception text must not be accessed")

    factory = LifecycleRepositoryFactory()
    error = UnstringableFailure()

    result = run_lifecycle(
        factory,
        lambda: (_ for _ in ()).throw(error),
    )

    assert sanitize_ingestion_exception(error) == ("network", "Ingestion failure")
    assert result.error_category == "network"


@pytest.mark.parametrize(
    ("error", "expected_category"),
    [
        (requests.HTTPError("controlled"), "http"),
        (requests.Timeout("controlled"), "network"),
        (pymysql.MySQLError("controlled"), "database"),
    ],
)
def test_exception_classification_uses_types_not_exception_details(
    error: BaseException, expected_category: str
) -> None:
    assert sanitize_ingestion_exception(error) == (
        expected_category,
        "Ingestion failure",
    )


def test_default_repository_factory_failure_is_a_sanitized_database_result(
    monkeypatch,
) -> None:
    event_lines: list[str] = []

    def fail_factory_resolution():
        raise RuntimeError("credential-like detail")

    monkeypatch.setattr(
        ingestion_module, "default_repository_factory", fail_factory_resolution
    )

    result = run_ingestion(
        lambda: LIVE_ROWS,
        lambda: object(),
        {5761: 100},
        fixed_utc_now,
        event_sink=event_lines.append,
    )

    assert result.error_category == "database"
    assert len(event_lines) == 1
    event = json.loads(event_lines[0])
    assert event.pop("timestamp").endswith("Z")
    assert event == {"event": "ingestion.failed", "errorCategory": "database_unavailable"}
    assert "credential-like" not in " ".join(event_lines)


def test_fetch_live_raises_http_then_decodes_once_with_bounded_timeouts(
    monkeypatch,
) -> None:
    import gym_fetch

    events: list[str] = []
    payload = {"controlled": True}

    class Response:
        def raise_for_status(self) -> None:
            events.append("raise")

        def json(self):
            events.append("json")
            return payload

    calls: list[tuple[str, object]] = []

    def fake_get(url: str, *, timeout: object):
        calls.append((url, timeout))
        return Response()

    monkeypatch.setattr(ingestion_module, "LIVE_COUNTS_URL", "https://controlled.invalid/live")
    monkeypatch.setattr(ingestion_module.requests, "get", fake_get)

    assert gym_fetch.fetch_live() is payload
    assert calls == [("https://controlled.invalid/live", (5, 20))]
    assert events == ["raise", "json"]


def test_db_connect_explicitly_disables_autocommit(monkeypatch) -> None:
    import gym_fetch

    for name, value in {
        "GYM_DB_HOST": "127.0.0.1",
        "GYM_DB_PORT": "3306",
        "GYM_DB_USER": "controlled-user",
        "GYM_DB_PASSWORD": "controlled-value",
        "GYM_DB_NAME": "controlled-db",
    }.items():
        monkeypatch.setenv(name, value)

    observed: list[bool] = []
    connection = object()

    def fake_connect(**settings):
        observed.append(settings["autocommit"])
        return connection

    monkeypatch.setattr(ingestion_module.pymysql, "connect", fake_connect)

    assert gym_fetch.db_connect() is connection
    assert observed == [False]


def test_direct_script_entry_reaches_injected_runner_without_sensitive_output() -> None:
    root = Path(__file__).resolve().parents[2]
    smoke = textwrap.dedent(
        f"""
        import runpy
        import sys

        sys.path.insert(0, {str(root / 'server')!r})
        import pymysql
        import requests
        import reclive.ingestion as ingestion
        import env_loader
        env_loader._DOTENV_STATE.loaded = True

        calls = []
        fake_connection = object()

        class Response:
            def raise_for_status(self):
                return None
            def json(self):
                return [{{"controlled": "payload-marker"}}]

        requests.get = lambda *args, **kwargs: Response()
        pymysql.connect = lambda **kwargs: fake_connection

        def injected_runner(fetch_payload, connect, capacities, now, **kwargs):
            assert fetch_payload() == [{{"controlled": "payload-marker"}}]
            assert connect() is fake_connection
            assert capacities
            assert now().utcoffset().total_seconds() == 0
            calls.append("called")
            return ingestion.IngestionRunResult("succeeded", 1, 1, 1, 1, None)

        ingestion.run_ingestion = injected_runner
        try:
            runpy.run_path({str(root / 'server' / 'gym_fetch.py')!r}, run_name="__main__")
        except SystemExit as exc:
            exit_code = exc.code
        else:
            exit_code = 0

        if exit_code != 0 or calls != ["called"]:
            raise SystemExit(91)
        print("smoke-ok")
        """
    )
    environment = {
        **os.environ,
        "LIVE_COUNTS_URL": "https://url-marker.invalid/live",
        "GYM_DB_HOST": "host-marker",
        "GYM_DB_PORT": "3306",
        "GYM_DB_USER": "user-marker",
        "GYM_DB_PASSWORD": "password-marker",
        "GYM_DB_NAME": "database-marker",
    }

    completed = subprocess.run(
        [sys.executable, "-c", smoke],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "smoke-ok"
    combined_output = completed.stdout + completed.stderr
    assert all(
        marker not in combined_output
        for marker in (
            "url-marker",
            "host-marker",
            "user-marker",
            "password-marker",
            "database-marker",
            "payload-marker",
        )
    )


def test_direct_script_bootstrap_failure_is_fixed_safe_output_only() -> None:
    root = Path(__file__).resolve().parents[2]
    smoke = textwrap.dedent(
        f"""
        import runpy
        import sys

        sys.path.insert(0, {str(root / 'server')!r})
        import env_loader
        env_loader._DOTENV_STATE.loaded = True
        from server.reclive import ingestion
        import pymysql
        import requests

        def fail_capacity_bootstrap(*args, **kwargs):
            raise RuntimeError("bootstrap-path-marker")

        def unexpected_io(*args, **kwargs):
            raise AssertionError("unexpected database/provider access")

        ingestion.load_facility_capacities = fail_capacity_bootstrap
        pymysql.connect = unexpected_io
        requests.get = unexpected_io
        runpy.run_path({str(root / 'server' / 'gym_fetch.py')!r}, run_name="__main__")
        """
    )
    environment = {
        **os.environ,
        "LIVE_COUNTS_URL": "https://controlled.invalid/live",
    }

    completed = subprocess.run(
        [sys.executable, "-c", smoke],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 1
    event = json.loads(completed.stdout)
    assert event.pop("timestamp").endswith("Z")
    assert event == {"event": "ingestion.failed", "errorCategory": "validation_error"}
    assert completed.stderr == ""
    assert "bootstrap-path-marker" not in completed.stdout + completed.stderr
    assert "Traceback" not in completed.stdout + completed.stderr


def test_blank_source_timestamp_first_observation_is_valid() -> None:
    result = validate_and_deduplicate_rows(
        [{"LocationId": "5761", "IsClosed": False, "LastCount": "47"}], {5761: 100}
    )

    assert result.received_count == 1
    assert result.invalid_count == 0
    assert result.rows[0].location_id == 5761
    assert result.rows[0].source_updated_at is None
    assert result.rows[0].current_capacity == 47


def test_offset_source_timestamp_normalizes_and_persists_as_naive_utc(
    fake_db, fixed_utc_clock
) -> None:
    result = validate_and_deduplicate_rows(
        [
            {
                "LocationId": 5761,
                "IsClosed": False,
                "LastCount": 46,
                "LastUpdatedDateAndTime": "2026-08-31T12:00:00",
            },
            {
                "LocationId": 5761,
                "IsClosed": False,
                "LastCount": 48,
                "LastUpdatedDateAndTime": "2026-08-31T07:00:00-05:00",
            },
        ],
        {5761: 100},
    )

    assert result.received_count == 2
    assert result.invalid_count == 1
    assert len(result.rows) == 1
    assert result.rows[0].source_updated_at == datetime(
        2026, 8, 31, 12, 0, tzinfo=timezone.utc
    )
    assert result.rows[0].source_updated_at.tzinfo is timezone.utc

    counts = SnapshotRepository(fake_db).persist_successful_poll(
        7, result.rows, fixed_utc_clock
    )

    expected_mysql_utc = datetime(2026, 8, 31, 12, 0)
    assert counts == IngestionWriteCounts(history_inserted=1, snapshot_updated=1)
    assert fake_db.snapshot_updates[0]["source_updated_at"] == expected_mysql_utc
    assert fake_db.history_inserts[0]["source_updated_at"] == expected_mysql_utc
    assert fake_db.history_inserts[0]["last_updated"] == expected_mysql_utc


def test_newest_source_timestamp_wins_and_last_valid_tie_wins() -> None:
    result = validate_and_deduplicate_rows(
        [
            {"LocationId": 5761, "IsClosed": False, "LastCount": 10, "LastUpdatedDateAndTime": "2026-08-31T10:00:00Z"},
            {"LocationId": 5761, "IsClosed": False, "LastCount": 11, "LastUpdatedDateAndTime": "2026-08-31T10:01:00Z"},
            {"LocationId": 5761, "IsClosed": True, "LastCount": 0, "LastUpdatedDateAndTime": "2026-08-31T10:01:00Z"},
        ],
        {5761: 100},
    )

    assert [(row.current_capacity, row.is_closed) for row in result.rows] == [(0, True)]


def test_dated_duplicate_outranks_blank_duplicate_regardless_of_input_order() -> None:
    result = validate_and_deduplicate_rows(
        [
            {"LocationId": 5761, "IsClosed": False, "LastCount": 10, "LastUpdatedDateAndTime": ""},
            {"LocationId": 5761, "IsClosed": False, "LastCount": 11, "LastUpdatedDateAndTime": "2026-08-31T10:01:00Z"},
            {"LocationId": 5761, "IsClosed": False, "LastCount": 12, "LastUpdatedDateAndTime": None},
        ],
        {5761: 100},
    )

    assert result.rows[0].current_capacity == 11


def test_python_booleans_are_not_integer_location_ids_or_counts() -> None:
    result = validate_and_deduplicate_rows(
        [
            {"LocationId": True, "IsClosed": False, "LastCount": 1},
            {"LocationId": 5761, "IsClosed": False, "LastCount": False},
            {"LocationId": 5761, "IsClosed": False, "LastCount": 2},
        ],
        {5761: 100},
    )

    assert result.invalid_count == 2
    assert [(row.location_id, row.current_capacity) for row in result.rows] == [(5761, 2)]


@pytest.mark.parametrize("field", ["LocationId", "LastCount"])
def test_oversized_decimal_field_invalidates_only_its_row(field: str) -> None:
    oversized_decimal = "9" * 5000
    malformed_row = {"LocationId": 5761, "IsClosed": False, "LastCount": 2}
    malformed_row[field] = oversized_decimal

    result = validate_and_deduplicate_rows(
        [malformed_row, {"LocationId": 5761, "IsClosed": False, "LastCount": 3}],
        {5761: 100},
    )

    assert result.received_count == 2
    assert result.invalid_count == 1
    assert [(row.location_id, row.current_capacity) for row in result.rows] == [(5761, 3)]


@pytest.mark.parametrize("invalid_closed", [0, 1, "false", "true", None, ""])
def test_is_closed_requires_an_actual_python_boolean(invalid_closed: object) -> None:
    result = validate_and_deduplicate_rows(
        [{"LocationId": 5761, "IsClosed": invalid_closed, "LastCount": 2}],
        {5761: 100},
    )

    assert result.invalid_count == 1
    assert result.rows == ()


@pytest.mark.parametrize("is_closed", [False, True])
def test_is_closed_accepts_only_actual_python_booleans(is_closed: bool) -> None:
    result = validate_and_deduplicate_rows(
        [{"LocationId": 5761, "IsClosed": is_closed, "LastCount": 2}],
        {5761: 100},
    )

    assert result.invalid_count == 0
    assert result.rows[0].is_closed is is_closed


def test_invalid_rows_do_not_discard_another_valid_row() -> None:
    result = validate_and_deduplicate_rows(
        [None, {"LocationId": "bad", "LastCount": 2}, {"LocationId": 5761, "IsClosed": False, "LastCount": 2}],
        {5761: 100},
    )

    assert result.invalid_count == 2
    assert [row.location_id for row in result.rows] == [5761]


def test_sanitize_ingestion_error_allows_known_category_and_bounded_safe_detail() -> None:
    category, message = sanitize_ingestion_error("http", "  upstream   status  " + "x" * 300)

    assert category == "http"
    assert message == ("upstream status " + "x" * 300)[:240]


def test_sanitize_ingestion_error_uses_safe_default_without_reflecting_sensitive_detail() -> None:
    sentinel = object()
    category, message = sanitize_ingestion_error("not-allowlisted", "bad input")
    _, url_message = sanitize_ingestion_error(
        "http", "https://user:password@example.test/?token=secret"
    )
    sentinel_category, sentinel_message = sanitize_ingestion_error("network", sentinel)

    assert (category, message) == ("validation", "bad input")
    assert sentinel_category == "network"
    assert sentinel_message == "Ingestion failure"
    assert url_message == "Ingestion failure"
    assert "example.test" not in url_message
    assert "password" not in url_message


def test_unchanged_state_advances_snapshot_heartbeat_without_history(fake_db) -> None:
    counts = SnapshotRepository(fake_db).persist_successful_poll(
        7,
        [NormalizedLiveRow(5761, False, 47, 100, None)],
        datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc),
    )

    assert counts.snapshot_updated == 1
    assert counts.history_inserted == 0
    assert fake_db.snapshot_updates[0]["fetched_at"] == datetime(2026, 8, 31, 12, 0)


def test_changed_count_with_same_source_timestamp_inserts_history(fake_db) -> None:
    source_updated_at = datetime(2026, 8, 31, 11, tzinfo=timezone.utc)
    fake_db.snapshots[5761] = (5761, False, 47, 100, source_updated_at.replace(tzinfo=None), datetime(2026, 8, 31, 11, 59))

    counts = SnapshotRepository(fake_db).persist_successful_poll(
        7,
        [NormalizedLiveRow(5761, False, 48, 100, source_updated_at)],
        datetime(2026, 8, 31, 12, 1, tzinfo=timezone.utc),
    )

    assert counts.history_inserted == 1
    assert fake_db.history_inserts[0]["current_capacity"] == 48


def test_first_success_preserves_legacy_wall_time_and_inserts_utc_baseline(fake_db) -> None:
    fake_db.legacy_history_bytes = b"2026-11-01 01:30:00.000000"
    fake_db.succeeded_run_count = 0

    counts = SnapshotRepository(fake_db).persist_successful_poll(
        7,
        [NormalizedLiveRow(5761, False, 48, 100, None)],
        datetime(2026, 11, 1, 7, 0, tzinfo=timezone.utc),
    )

    assert fake_db.legacy_history_bytes == b"2026-11-01 01:30:00.000000"
    assert counts.history_inserted == 1
    assert fake_db.history_inserts[0]["fetched_at"] == datetime(2026, 11, 1, 7, 0)


def test_mysql_datetime_boundary_maps_naive_utc_binds_and_tuples() -> None:
    aware = datetime(2026, 11, 1, 7, 0, tzinfo=timezone.utc)

    assert as_mysql_utc(aware) == datetime(2026, 11, 1, 7, 0)
    assert to_aware_utc(datetime(2026, 11, 1, 7, 0)) == aware


def test_start_and_completion_bind_utc_values_sort_ids_and_do_not_own_transactions(fake_db) -> None:
    repository = SnapshotRepository(fake_db)
    started_at = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
    run_id = repository.start_run(started_at)
    repository.complete_success(
        run_id,
        started_at + timedelta(minutes=1),
        3,
        2,
        IngestionWriteCounts(history_inserted=1, snapshot_updated=2),
        [5761, 12, 5761],
    )

    assert run_id == 7
    assert fake_db.run_rows[7]["completed_at"] == datetime(2026, 8, 31, 12, 1)
    assert fake_db.run_rows[7]["observed_location_ids"] == "[12, 5761]"
    assert fake_db.transactions == []


def test_completion_requires_exactly_one_running_run(fake_db, fixed_utc_clock) -> None:
    fake_db.run_rows[7]["status"] = "failed"

    with pytest.raises(RuntimeError, match="running ingestion run"):
        SnapshotRepository(fake_db).complete_failure(
            7, fixed_utc_clock, "database", "Ingestion failure"
        )


@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 8, 31, 12, 0),
        datetime(2026, 8, 31, 7, 0, tzinfo=timezone(timedelta(hours=-5))),
    ],
)
def test_repository_rejects_non_utc_application_timestamps(fake_db, value) -> None:
    with pytest.raises(ValueError, match="aware UTC"):
        SnapshotRepository(fake_db).start_run(value)


def test_persist_rejects_invalid_dynamic_location_ids(fake_db, fixed_utc_clock) -> None:
    with pytest.raises(ValueError, match="location IDs"):
        SnapshotRepository(fake_db).persist_successful_poll(
            7,
            [NormalizedLiveRow(True, False, 47, 100, None)],
            fixed_utc_clock,
        )


@pytest.mark.parametrize(
    ("run_id", "status"),
    [(7, "failed"), (7, "succeeded"), (999, None)],
)
def test_persist_requires_the_supplied_running_run_before_any_write(
    fake_db, fixed_utc_clock, run_id, status
) -> None:
    if status is None:
        fake_db.run_rows.pop(7)
    else:
        fake_db.run_rows[7]["status"] = status

    with pytest.raises(RuntimeError, match="running ingestion run"):
        SnapshotRepository(fake_db).persist_successful_poll(
            run_id,
            [NormalizedLiveRow(5761, False, 48, 100, None)],
            fixed_utc_clock,
        )

    assert fake_db.snapshot_updates == []
    assert fake_db.history_inserts == []


def test_older_running_run_cannot_succeed_after_a_later_cutover(fake_db) -> None:
    first_success_at = datetime(2026, 11, 1, 7, 1, tzinfo=timezone.utc)
    fake_db.run_rows.pop(6)
    fake_db.run_rows[7]["started_at"] = datetime(2026, 11, 1, 6, 59)
    fake_db.run_rows[8] = {
        "status": "succeeded",
        "started_at": first_success_at.replace(tzinfo=None),
    }

    with pytest.raises(RuntimeError, match="predates the first succeeded run"):
        SnapshotRepository(fake_db).persist_successful_poll(
            7,
            [NormalizedLiveRow(5761, False, 48, 100, None)],
            datetime(2026, 11, 1, 7, 2, tzinfo=timezone.utc),
        )

    assert fake_db.snapshot_updates == []
    assert fake_db.history_inserts == []


def test_lower_id_same_time_running_run_cannot_precede_the_cutover(fake_db) -> None:
    cutover_at = datetime(2026, 11, 1, 7, 0, tzinfo=timezone.utc)
    fake_db.run_rows = {
        6: {"status": "running", "started_at": cutover_at.replace(tzinfo=None)},
        7: {"status": "succeeded", "started_at": cutover_at.replace(tzinfo=None)},
    }

    with pytest.raises(RuntimeError, match="predates the first succeeded run"):
        SnapshotRepository(fake_db).persist_successful_poll(
            6,
            [NormalizedLiveRow(5761, False, 48, 100, None)],
            cutover_at + timedelta(minutes=1),
        )

    assert fake_db.snapshot_updates == []
    assert fake_db.history_inserts == []


@pytest.mark.mysql
def test_mysql_snapshot_repository_persists_cutover_contract(clean_test_database) -> None:
    import json
    from pathlib import Path
    from threading import Event, Thread
    from time import sleep

    import pymysql

    from reclive.database_dialect import detect_database_dialect
    from reclive.migrations import execution_snapshot, snapshot_migration, split_statements

    migrations = Path(__file__).resolve().parents[2] / "server" / "migrations"
    connection = pymysql.connect(**clean_test_database)
    try:
        dialect = detect_database_dialect(connection)
        with connection.cursor() as cursor:
            for migration_name in ("0001_core_history.sql", "0002_snapshot_and_ingestion.sql"):
                migration = execution_snapshot(
                    snapshot_migration(migrations / migration_name), dialect
                ).sql_bytes.decode("utf-8")
                for statement in split_statements(migration):
                    cursor.execute(statement)
        connection.commit()
    finally:
        connection.close()

    settings = {**clean_test_database, "autocommit": False}
    writer = pymysql.connect(**settings)
    observer = pymysql.connect(**clean_test_database)
    try:
        started_at = datetime(2026, 11, 1, 7, 0, tzinfo=timezone.utc)
        rows = validate_and_deduplicate_rows(LIVE_ROWS, {5761: 100}).rows
        baseline_rows = [replace(rows[0], current_capacity=47, source_updated_at=None)]
        writer_repository = SnapshotRepository(writer)

        rolled_back_run = writer_repository.start_run(started_at)
        rolled_back_counts = writer_repository.persist_successful_poll(
            rolled_back_run, baseline_rows, started_at
        )
        writer_repository.complete_success(
            rolled_back_run, started_at, 1, 1, rolled_back_counts, [5761]
        )
        assert_observer_counts(observer, runs=0, snapshots=0, history=0)
        writer.rollback()
        assert_observer_counts(observer, runs=0, snapshots=0, history=0)

        committed_run = writer_repository.start_run(started_at)
        committed_counts = writer_repository.persist_successful_poll(
            committed_run, baseline_rows, started_at
        )
        writer_repository.complete_success(
            committed_run, started_at, 1, 1, committed_counts, [5761]
        )
        assert_observer_counts(observer, runs=0, snapshots=0, history=0)
        writer.commit()
        assert_observer_counts(observer, runs=1, snapshots=1, history=1)

        with observer.cursor() as cursor:
            cursor.execute("DELETE FROM location_history")
            cursor.execute("DELETE FROM location_snapshot")
            cursor.execute("DELETE FROM ingestion_runs")
            cursor.execute(
                "INSERT INTO location_snapshot("
                "location_id, is_closed, current_capacity, max_capacity, "
                "source_updated_at, fetched_at, created_at, updated_at"
                ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    7777,
                    False,
                    12,
                    100,
                    None,
                    datetime(2026, 11, 1, 6, 59),
                    datetime(2026, 11, 1, 6, 59),
                    datetime(2026, 11, 1, 6, 59),
                ),
            )

        setup = pymysql.connect(**settings)
        try:
            setup_repository = SnapshotRepository(setup)
            equal_time_late_run = setup_repository.start_run(started_at)
            first_run = setup_repository.start_run(started_at)
            second_run = setup_repository.start_run(started_at + timedelta(minutes=1))
            abandoned_run = setup_repository.start_run(
                started_at + timedelta(minutes=10)
            )
            setup.commit()
        finally:
            setup.close()

        first_connection = pymysql.connect(**settings)
        second_connection = pymysql.connect(**settings)
        try:
            first_repository = SnapshotRepository(first_connection)
            second_repository = SnapshotRepository(second_connection)
            first_counts = first_repository.persist_successful_poll(
                first_run, baseline_rows, started_at
            )

            second_started = Event()
            second_result: dict[str, object] = {}

            def persist_second_poll() -> None:
                second_started.set()
                try:
                    second_result["counts"] = second_repository.persist_successful_poll(
                        second_run,
                        [NormalizedLiveRow(7777, False, 12, 100, None)],
                        started_at + timedelta(minutes=1),
                    )
                except BaseException as exc:
                    second_result["error"] = exc

            second_thread = Thread(target=persist_second_poll)
            second_thread.start()
            assert second_started.wait(timeout=1)
            sleep(0.1)
            assert second_thread.is_alive()
            assert second_result == {}
            assert_observer_counts(observer, runs=4, snapshots=1, history=0)

            first_repository.complete_success(
                first_run, started_at, 1, 1, first_counts, [5761]
            )
            first_connection.commit()
            second_thread.join(timeout=5)
            assert not second_thread.is_alive()
            assert "error" not in second_result
            second_counts = second_result["counts"]
            assert second_counts == IngestionWriteCounts(
                history_inserted=0, snapshot_updated=1
            )
            second_repository.complete_success(
                second_run,
                started_at + timedelta(minutes=1),
                1,
                1,
                second_counts,
                [7777, 12, 7777],
            )
            second_connection.commit()
        finally:
            first_connection.close()
            second_connection.close()

        late_connection = pymysql.connect(**settings)
        try:
            late_repository = SnapshotRepository(late_connection)
            with pytest.raises(RuntimeError, match="predates the first succeeded run"):
                late_repository.persist_successful_poll(
                    equal_time_late_run,
                    [NormalizedLiveRow(8888, False, 3, 100, None)],
                    started_at + timedelta(minutes=2),
                )
            with pytest.raises(RuntimeError, match="predates the first succeeded run"):
                late_repository.complete_success(
                    equal_time_late_run,
                    started_at + timedelta(minutes=2),
                    1,
                    1,
                    IngestionWriteCounts(history_inserted=0, snapshot_updated=0),
                    [8888],
                )
            late_connection.rollback()

            older_run = late_repository.start_run(started_at - timedelta(minutes=1))
            late_connection.commit()
            with pytest.raises(RuntimeError, match="predates the first succeeded run"):
                late_repository.persist_successful_poll(
                    older_run,
                    [NormalizedLiveRow(9999, False, 3, 100, None)],
                    started_at + timedelta(minutes=2),
                )
            with pytest.raises(RuntimeError, match="predates the first succeeded run"):
                late_repository.complete_success(
                    older_run,
                    started_at + timedelta(minutes=2),
                    1,
                    1,
                    IngestionWriteCounts(history_inserted=0, snapshot_updated=0),
                    [9999],
                )
            late_connection.rollback()

            failed_run = late_repository.start_run(started_at + timedelta(minutes=3))
            late_repository.complete_failure(
                failed_run,
                started_at + timedelta(minutes=3),
                "database",
                "Ingestion failure",
            )
            late_connection.commit()

            for rejected_run in (failed_run, 999999):
                with pytest.raises(RuntimeError, match="running ingestion run"):
                    late_repository.persist_successful_poll(
                        rejected_run,
                        [NormalizedLiveRow(9999, False, 3, 100, None)],
                        started_at + timedelta(minutes=3),
                    )
                late_connection.rollback()
        finally:
            late_connection.close()

        with observer.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM ingestion_runs WHERE status='succeeded' "
                "ORDER BY started_at, id"
            )
            assert [row[0] for row in cursor.fetchall()] == [first_run, second_run]
            cursor.execute("SELECT COUNT(*) FROM location_history")
            assert cursor.fetchone()[0] == 1
            cursor.execute("SELECT COUNT(*) FROM location_snapshot")
            assert cursor.fetchone()[0] == 2
            cursor.execute(
                "SELECT observed_location_ids FROM ingestion_runs WHERE id = %s",
                (second_run,),
            )
            assert json.loads(cursor.fetchone()[0]) == [12, 7777]
            cursor.execute("SELECT status FROM ingestion_runs WHERE id = %s", (older_run,))
            assert cursor.fetchone()[0] == "running"
            cursor.execute(
                "SELECT status FROM ingestion_runs WHERE id = %s", (abandoned_run,)
            )
            assert cursor.fetchone()[0] == "running"
            cursor.execute(
                "SELECT status FROM ingestion_runs WHERE id = %s",
                (equal_time_late_run,),
            )
            assert cursor.fetchone()[0] == "running"
            cursor.execute("SELECT status FROM ingestion_runs WHERE id = %s", (failed_run,))
            assert cursor.fetchone()[0] == "failed"
            cursor.execute(
                "SELECT COUNT(*) FROM location_snapshot WHERE location_id IN (8888, 9999)"
            )
            assert cursor.fetchone()[0] == 0
    finally:
        writer.close()
        observer.close()


def assert_observer_counts(connection, *, runs: int, snapshots: int, history: int) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM ingestion_runs")
        assert cursor.fetchone()[0] == runs
        cursor.execute("SELECT COUNT(*) FROM location_snapshot")
        assert cursor.fetchone()[0] == snapshots
        cursor.execute("SELECT COUNT(*) FROM location_history")
        assert cursor.fetchone()[0] == history
