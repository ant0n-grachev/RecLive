from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Sequence

from fastapi.testclient import TestClient
from freezegun import freeze_time
import pytest

import forecast_api
from reclive.occupancy_repository import SnapshotRepository, SnapshotRow


class RouteSnapshotRepository:
    def __init__(
        self,
        *,
        rows: Sequence[SnapshotRow] | None = None,
        last_successful_fetch_at: datetime | None = datetime(
            2026, 8, 31, 12, 0, tzinfo=timezone.utc
        ),
        error: BaseException | None = None,
    ) -> None:
        self.rows = tuple(rows) if rows is not None else (snapshot_row(),)
        self.last_successful_fetch_at = last_successful_fetch_at
        self.error = error

    def fetch_live_snapshot(self, now: datetime) -> SimpleNamespace:
        assert now.tzinfo is timezone.utc
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            last_successful_fetch_at=self.last_successful_fetch_at,
            rows=self.rows,
        )


class EvaluatorSnapshotReader:
    def __init__(self, rows: Sequence[SnapshotRow]) -> None:
        self.rows = list(rows)
        self.internal_row_read_count = 0
        self.public_envelope_read_count = 0

    def fetch_live_snapshot_rows(self) -> list[SnapshotRow]:
        self.internal_row_read_count += 1
        return list(self.rows)

    def fetch_live_snapshot(self, now: datetime) -> SimpleNamespace:
        del now
        self.public_envelope_read_count += 1
        return SimpleNamespace(last_successful_fetch_at=None, rows=list(self.rows))


class EvaluatorConnection:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def snapshot_row(
    *,
    source_updated_at: datetime | None = datetime(
        2026, 8, 31, 11, 59, tzinfo=timezone.utc
    ),
    fetched_at: datetime = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc),
) -> SnapshotRow:
    return SnapshotRow(
        location_id=5761,
        is_closed=False,
        current_capacity=47,
        max_capacity=100,
        source_updated_at=source_updated_at,
        fetched_at=fetched_at,
    )


def get_live_counts(repository: object):
    dependency = getattr(forecast_api, "get_snapshot_repository", None)
    assert dependency is not None, "live-counts must expose its repository dependency"

    def override_repository() -> object:
        return repository

    forecast_api.app.dependency_overrides[dependency] = override_repository
    try:
        return TestClient(forecast_api.app).get("/api/live-counts")
    finally:
        forecast_api.app.dependency_overrides.clear()


def configure_evaluator_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        forecast_api,
        "load_store_from_db",
        lambda: {
            "rules": [
                {
                    "_id": 0,
                    "facilityId": 1186,
                    "sectionKey": "fitness floors",
                    "threshold": 1,
                    "subscription": {},
                }
            ]
        },
    )
    lock = object()
    monkeypatch.setattr(forecast_api, "db_acquire_evaluator_lock", lambda: lock)
    monkeypatch.setattr(
        forecast_api, "db_release_evaluator_lock", lambda connection: None
    )
    monkeypatch.setattr(forecast_api, "db_rules_count", lambda: 1)


@freeze_time("2026-08-31 12:05:00")
def test_live_counts_returns_snapshot_envelope_and_fetched_at() -> None:
    response = get_live_counts(RouteSnapshotRepository())

    assert response.status_code == 200
    assert response.json()["ingestion"]["status"] == "healthy"
    assert response.json()["rows"] == [
        {
            "LocationId": 5761,
            "IsClosed": False,
            "LastCount": 47,
            "LastUpdatedDateAndTime": "2026-08-31T11:59:00+00:00",
            "FetchedAt": "2026-08-31T12:00:00+00:00",
        }
    ]


def test_repository_reads_snapshot_and_latest_success_with_utc_mapping(fake_db) -> None:
    fake_db.snapshots = {
        5761: (
            5761,
            False,
            47,
            100,
            datetime(2026, 8, 31, 11, 59),
            datetime(2026, 8, 31, 12, 0),
        )
    }
    fake_db.run_rows[6]["completed_at"] = datetime(2026, 8, 31, 12, 1)

    snapshot = SnapshotRepository(fake_db).fetch_live_snapshot(
        datetime(2026, 8, 31, 12, 5, tzinfo=timezone.utc)
    )

    assert snapshot.last_successful_fetch_at == datetime(
        2026, 8, 31, 12, 1, tzinfo=timezone.utc
    )
    assert snapshot.rows == [
        SnapshotRow(
            location_id=5761,
            is_closed=False,
            current_capacity=47,
            max_capacity=100,
            source_updated_at=datetime(2026, 8, 31, 11, 59, tzinfo=timezone.utc),
            fetched_at=datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc),
        )
    ]
    statements = [" ".join(query.statement.split()) for query in fake_db.queries]
    assert any("FROM location_snapshot ORDER BY location_id" in sql for sql in statements)
    assert any(
        "FROM ingestion_runs WHERE status = 'succeeded'" in sql
        for sql in statements
    )
    assert all("location_history" not in sql for sql in statements)


@freeze_time("2026-08-31 12:05:00")
def test_live_counts_route_executes_only_snapshot_backed_reads(fake_db) -> None:
    fake_db.run_rows[6]["completed_at"] = datetime(2026, 8, 31, 12, 1)

    response = get_live_counts(SnapshotRepository(fake_db))

    assert response.status_code == 200
    statements = [" ".join(query.statement.split()) for query in fake_db.queries]
    assert any("FROM location_snapshot ORDER BY location_id" in sql for sql in statements)
    assert any("FROM ingestion_runs" in sql for sql in statements)
    assert all("location_history" not in sql for sql in statements)


@freeze_time("2026-08-31 12:05:00")
def test_live_counts_reports_no_success_as_unavailable_without_dropping_rows() -> None:
    response = get_live_counts(
        RouteSnapshotRepository(
            rows=[snapshot_row(source_updated_at=None)],
            last_successful_fetch_at=None,
        )
    )

    assert response.status_code == 200
    assert response.json()["ingestion"] == {
        "lastSuccessfulFetchAt": None,
        "ageSeconds": None,
        "status": "unavailable",
    }
    assert response.json()["rows"][0]["LocationId"] == 5761
    assert response.json()["rows"][0]["LastUpdatedDateAndTime"] is None
    assert response.json()["rows"][0]["FetchedAt"] == "2026-08-31T12:00:00+00:00"


@freeze_time("2026-08-31 12:05:00")
def test_live_counts_serializes_provenance_and_fetch_time_as_utc() -> None:
    plus_two = timezone(timedelta(hours=2))
    response = get_live_counts(
        RouteSnapshotRepository(
            rows=[
                snapshot_row(
                    source_updated_at=datetime(2026, 8, 31, 13, 59, tzinfo=plus_two),
                    fetched_at=datetime(2026, 8, 31, 14, 0, tzinfo=plus_two),
                )
            ]
        )
    )

    assert response.status_code == 200
    assert response.json()["rows"][0]["LastUpdatedDateAndTime"] == (
        "2026-08-31T11:59:00+00:00"
    )
    assert response.json()["rows"][0]["FetchedAt"] == "2026-08-31T12:00:00+00:00"


@pytest.mark.parametrize(
    ("last_successful_fetch_at", "expected_age", "expected_status"),
    [
        (datetime(2026, 8, 31, 11, 50, tzinfo=timezone.utc), 600, "healthy"),
        (
            datetime(2026, 8, 31, 11, 49, 59, 999999, tzinfo=timezone.utc),
            600,
            "stale",
        ),
        (datetime(2026, 8, 31, 11, 49, 59, tzinfo=timezone.utc), 601, "stale"),
        (datetime(2026, 8, 31, 12, 1, tzinfo=timezone.utc), 0, "healthy"),
    ],
)
@freeze_time("2026-08-31 12:00:00")
def test_live_counts_floors_reported_age_but_classifies_precise_elapsed_time(
    last_successful_fetch_at: datetime,
    expected_age: int,
    expected_status: str,
) -> None:
    response = get_live_counts(
        RouteSnapshotRepository(
            last_successful_fetch_at=last_successful_fetch_at
        )
    )

    assert response.status_code == 200
    assert response.json()["ingestion"]["ageSeconds"] == expected_age
    assert response.json()["ingestion"]["status"] == expected_status


@freeze_time("2026-08-31 12:05:00")
def test_live_counts_returns_503_only_for_empty_or_unavailable_snapshot() -> None:
    empty = get_live_counts(RouteSnapshotRepository(rows=[]))
    unavailable = get_live_counts(
        RouteSnapshotRepository(error=RuntimeError("private database detail"))
    )

    assert empty.status_code == 503
    assert empty.json() == {"detail": "Live occupancy snapshot is empty"}
    assert unavailable.status_code == 503
    assert unavailable.json() == {
        "detail": "Failed to query live occupancy snapshot"
    }
    assert "private database detail" not in unavailable.text


def test_evaluator_uses_internal_snapshot_rows_without_opening_a_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_evaluator_rule(monkeypatch)
    reader = EvaluatorSnapshotReader([snapshot_row()])

    def reject_connection(**kwargs: object) -> object:
        raise AssertionError(f"injected evaluator opened a connection: {kwargs}")

    monkeypatch.setattr(forecast_api, "open_db_connection", reject_connection)

    result = forecast_api.evaluate_rules_once(snapshot_reader=reader)

    assert result["skippedThreshold"] == 1
    assert reader.internal_row_read_count == 1
    assert reader.public_envelope_read_count == 0


def test_evaluator_factory_path_closes_its_connection_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_evaluator_rule(monkeypatch)
    connection = EvaluatorConnection()
    reader = EvaluatorSnapshotReader([snapshot_row()])
    factory_connections: list[object] = []

    def open_connection(*, autocommit: bool = True) -> EvaluatorConnection:
        assert autocommit is False
        return connection

    def repository_factory(candidate: object) -> EvaluatorSnapshotReader:
        factory_connections.append(candidate)
        return reader

    monkeypatch.setattr(forecast_api, "open_db_connection", open_connection)

    result = forecast_api.evaluate_rules_once(repository_factory=repository_factory)

    assert result["skippedThreshold"] == 1
    assert factory_connections == [connection]
    assert reader.internal_row_read_count == 1
    assert reader.public_envelope_read_count == 0
    assert connection.close_count == 1
