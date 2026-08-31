from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from fastapi.testclient import TestClient
from freezegun import freeze_time
import pytest

import forecast_api
from reclive.occupancy_repository import SnapshotRepository, SnapshotRow


PARITY_FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "occupancy_summary_parity.json").read_text(
        encoding="utf-8"
    )
)


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
    location_id: int = 5761,
    source_updated_at: datetime | None = datetime(
        2026, 8, 31, 11, 59, tzinfo=timezone.utc
    ),
    fetched_at: datetime = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc),
) -> SnapshotRow:
    return SnapshotRow(
        location_id=location_id,
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


def configure_evaluator_rule(
    monkeypatch: pytest.MonkeyPatch,
    *,
    facility_id: int = 1186,
    section_key: str = "fitness floors",
    threshold: int = 1,
) -> None:
    monkeypatch.setattr(
        forecast_api,
        "load_store_from_db",
        lambda: {
            "rules": [
                {
                    "_id": 0,
                    "facilityId": facility_id,
                    "sectionKey": section_key,
                    "threshold": threshold,
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


def utc_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def configure_parity_case(
    monkeypatch: pytest.MonkeyPatch,
    fixture_case: dict[str, Any],
) -> list[SnapshotRow]:
    location_ids = [entry["locationId"] for entry in fixture_case["locations"]]
    capacities = {
        entry["locationId"]: entry["maxCapacity"]
        for entry in fixture_case["locations"]
    }
    monkeypatch.setattr(
        forecast_api,
        "location_ids_for_section",
        lambda facility_id, section_key: location_ids,
    )
    monkeypatch.setattr(forecast_api, "MAX_CAP", capacities)

    rows: list[SnapshotRow] = []
    for entry in fixture_case["locations"]:
        row = entry["row"]
        if row is None:
            continue
        rows.append(
            SnapshotRow(
                location_id=entry["locationId"],
                is_closed=row["isClosed"],
                current_capacity=row["currentCapacity"],
                max_capacity=entry["maxCapacity"],
                source_updated_at=None,
                fetched_at=utc_datetime(row["fetchedAt"]),
            )
        )
    return rows


@pytest.mark.parametrize(
    "fixture_case",
    PARITY_FIXTURE["cases"],
    ids=[case["name"] for case in PARITY_FIXTURE["cases"]],
)
@freeze_time("2026-08-31 12:00:00")
def test_section_metrics_match_shared_cross_layer_summary_cases(
    monkeypatch: pytest.MonkeyPatch,
    fixture_case: dict[str, Any],
) -> None:
    rows = configure_parity_case(monkeypatch, fixture_case)

    metrics = forecast_api.compute_section_metrics(
        9999,
        "fixture section",
        forecast_api.index_live_rows(rows),
    )

    assert metrics is not None
    expected = fixture_case["expected"]
    actual = {
        "count": metrics.get("total"),
        "observedCapacity": metrics.get("max"),
        "expectedOpenCapacity": metrics.get("expectedOpenCapacity"),
        "coverage": metrics.get("coverage"),
        "percent": metrics.get("percent"),
        "status": metrics.get("status"),
    }
    assert actual["count"] == expected["count"]
    assert actual["observedCapacity"] == expected["observedCapacity"]
    assert actual["expectedOpenCapacity"] == expected["expectedOpenCapacity"]
    assert actual["coverage"] == pytest.approx(expected["coverage"])
    assert actual["percent"] == expected["percent"]
    assert actual["status"] == expected["status"]
    assert (
        actual["status"] == "live" and actual["coverage"] >= 0.8
    ) is fixture_case["evaluatorEligible"]


@pytest.mark.parametrize(
    ("capacity", "is_closed", "current_capacity", "fetched_at"),
    [
        (100, None, 83, datetime(2026, 8, 31, 11, 55, tzinfo=timezone.utc)),
        (100, False, True, datetime(2026, 8, 31, 11, 55, tzinfo=timezone.utc)),
        (100, False, 83, datetime(2026, 8, 31, 11, 55)),
        (True, False, 1, datetime(2026, 8, 31, 11, 55, tzinfo=timezone.utc)),
    ],
    ids=[
        "indeterminate-closure",
        "boolean-count",
        "naive-fetched-at",
        "boolean-configured-capacity",
    ],
)
@freeze_time("2026-08-31 12:00:00")
def test_section_metrics_reject_non_explicit_or_non_integer_observations(
    monkeypatch: pytest.MonkeyPatch,
    capacity: Any,
    is_closed: Any,
    current_capacity: Any,
    fetched_at: datetime,
) -> None:
    monkeypatch.setattr(
        forecast_api,
        "location_ids_for_section",
        lambda facility_id, section_key: [91001],
    )
    monkeypatch.setattr(forecast_api, "MAX_CAP", {91001: capacity})
    row = SnapshotRow(
        location_id=91001,
        is_closed=is_closed,
        current_capacity=current_capacity,
        max_capacity=100,
        source_updated_at=None,
        fetched_at=fetched_at,
    )

    metrics = forecast_api.compute_section_metrics(
        9999,
        "fixture section",
        forecast_api.index_live_rows([row]),
    )

    assert metrics is not None
    expected_status = "unknown" if capacity is True else "insufficient"
    assert metrics.get("total") is None
    assert metrics.get("percent") is None
    assert metrics.get("status") == expected_status


@pytest.mark.parametrize(
    "case_name",
    [
        "just-stale-row",
        "future-row",
        "fresh-explicit-closure",
        "missing-row",
        "partial-uses-observed-denominator",
        "no-positive-configured-capacity",
        "indeterminate-closure",
    ],
)
@freeze_time("2026-08-31 12:00:00")
def test_evaluator_skips_untrusted_or_above_threshold_shared_cases(
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
) -> None:
    fixture_case = next(
        case for case in PARITY_FIXTURE["cases"] if case["name"] == case_name
    )
    rows = configure_parity_case(monkeypatch, fixture_case)
    configure_evaluator_rule(
        monkeypatch,
        facility_id=9999,
        section_key="fixture section",
        threshold=70,
    )
    notifications: list[dict[str, Any]] = []
    monkeypatch.setattr(
        forecast_api,
        "send_notification",
        lambda **notification: notifications.append(notification),
    )

    result = forecast_api.evaluate_rules_once(
        snapshot_reader=EvaluatorSnapshotReader(rows)
    )

    assert result["sent"] == 0
    assert result["failed"] == 0
    assert notifications == []


@freeze_time("2026-08-31 12:00:00")
def test_evaluator_never_sends_for_partial_summary_even_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_case = next(
        case
        for case in PARITY_FIXTURE["cases"]
        if case["name"] == "partial-uses-observed-denominator"
    )
    rows = configure_parity_case(monkeypatch, fixture_case)
    configure_evaluator_rule(
        monkeypatch,
        facility_id=9999,
        section_key="fixture section",
        threshold=85,
    )
    notifications: list[dict[str, Any]] = []
    monkeypatch.setattr(
        forecast_api,
        "send_notification",
        lambda **notification: notifications.append(notification),
    )

    result = forecast_api.evaluate_rules_once(
        snapshot_reader=EvaluatorSnapshotReader(rows)
    )

    assert fixture_case["evaluatorEligible"] is False
    assert result["sent"] == 0
    assert result["failed"] == 0
    assert notifications == []


@freeze_time("2026-08-31 12:00:00")
def test_evaluator_can_send_for_live_summary_at_coverage_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_case = next(
        case
        for case in PARITY_FIXTURE["cases"]
        if case["name"] == "exact-live-coverage-boundary"
    )
    rows = configure_parity_case(monkeypatch, fixture_case)
    configure_evaluator_rule(
        monkeypatch,
        facility_id=9999,
        section_key="fixture section",
        threshold=20,
    )
    notifications: list[dict[str, Any]] = []
    monkeypatch.setattr(
        forecast_api,
        "send_notification",
        lambda **notification: notifications.append(notification),
    )

    result = forecast_api.evaluate_rules_once(
        snapshot_reader=EvaluatorSnapshotReader(rows)
    )

    assert fixture_case["evaluatorEligible"] is True
    assert fixture_case["expected"]["coverage"] == 0.8
    assert result["sent"] == 1
    assert result["failed"] == 0
    assert len(notifications) == 1
    assert "20% full" in notifications[0]["body"]


@freeze_time("2026-08-31 12:00:00")
def test_evaluator_rounds_half_percent_up_for_comparison_and_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_case = next(
        case
        for case in PARITY_FIXTURE["cases"]
        if case["name"] == "raw-half-percent"
    )
    rows = configure_parity_case(monkeypatch, fixture_case)
    notifications: list[dict[str, Any]] = []
    monkeypatch.setattr(
        forecast_api,
        "send_notification",
        lambda **notification: notifications.append(notification),
    )

    configure_evaluator_rule(
        monkeypatch,
        facility_id=9999,
        section_key="fixture section",
        threshold=12,
    )
    below_result = forecast_api.evaluate_rules_once(
        snapshot_reader=EvaluatorSnapshotReader(rows)
    )

    assert fixture_case["expected"]["percent"] == 12.5
    assert below_result["sent"] == 0
    assert below_result["skippedThreshold"] == 1
    assert notifications == []

    configure_evaluator_rule(
        monkeypatch,
        facility_id=9999,
        section_key="fixture section",
        threshold=13,
    )
    at_result = forecast_api.evaluate_rules_once(
        snapshot_reader=EvaluatorSnapshotReader(rows)
    )

    assert at_result["sent"] == 1
    assert len(notifications) == 1
    assert "13% full" in notifications[0]["body"]


def test_evaluator_samples_one_aware_utc_now_for_metrics_and_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CountingDateTime(datetime):
        calls = 0

        @classmethod
        def now(cls, tz: timezone | None = None) -> "CountingDateTime":
            assert tz is timezone.utc
            cls.calls += 1
            return cls(2026, 8, 31, 12, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(forecast_api, "datetime", CountingDateTime)
    monkeypatch.setattr(
        forecast_api,
        "location_ids_for_section",
        lambda facility_id, section_key: [91001],
    )
    monkeypatch.setattr(forecast_api, "MAX_CAP", {91001: 100})
    configure_evaluator_rule(
        monkeypatch,
        facility_id=9999,
        section_key="fixture section",
        threshold=85,
    )
    payloads: list[dict[str, Any]] = []
    monkeypatch.setattr(forecast_api, "get_vapid_private_key", lambda: "private")
    monkeypatch.setattr(
        forecast_api,
        "get_vapid_claims",
        lambda: {"sub": "mailto:test@example.com"},
    )
    monkeypatch.setattr(
        forecast_api,
        "webpush",
        lambda **kwargs: payloads.append(json.loads(kwargs["data"])),
    )
    row = SnapshotRow(
        location_id=91001,
        is_closed=False,
        current_capacity=83,
        max_capacity=100,
        source_updated_at=None,
        fetched_at=CountingDateTime(
            2026,
            8,
            31,
            11,
            55,
            tzinfo=timezone.utc,
        ),
    )

    result = forecast_api.evaluate_rules_once(
        snapshot_reader=EvaluatorSnapshotReader([row])
    )

    assert result["sent"] == 1
    assert CountingDateTime.calls == 1
    assert payloads[0]["sentAt"] == "2026-08-31T12:00:00+00:00"


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


@freeze_time("2026-08-31 12:05:00")
def test_evaluator_uses_internal_snapshot_rows_without_opening_a_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_evaluator_rule(monkeypatch, section_key="running track")
    reader = EvaluatorSnapshotReader([snapshot_row(location_id=5763)])

    def reject_connection(**kwargs: object) -> object:
        raise AssertionError(f"injected evaluator opened a connection: {kwargs}")

    monkeypatch.setattr(forecast_api, "open_db_connection", reject_connection)

    result = forecast_api.evaluate_rules_once(snapshot_reader=reader)

    assert result["skippedThreshold"] == 1
    assert reader.internal_row_read_count == 1
    assert reader.public_envelope_read_count == 0


@freeze_time("2026-08-31 12:05:00")
def test_evaluator_factory_path_closes_its_connection_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_evaluator_rule(monkeypatch, section_key="running track")
    connection = EvaluatorConnection()
    reader = EvaluatorSnapshotReader([snapshot_row(location_id=5763)])
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
