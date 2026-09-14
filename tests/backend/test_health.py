from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone, tzinfo
import io
import json
from math import inf, nan
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from server.reclive.api.app import create_app
from server.reclive.api.health import create_health_router
from server.reclive.health import (
    REQUIRED_COMPONENTS,
    ComponentHealth,
    HealthReport,
    classify_age,
    overall_status,
    utc_iso,
)
from server.reclive.health_repository import (
    MAX_HEALTH_ARTIFACT_BYTES,
    HealthEvidence,
    HealthRepository,
    generated_at,
    healthy_schedule_generated_at,
)
from server.reclive.migrations import migration_files, snapshot_migration
from server.reclive.settings import Settings, build_settings_from_environment
from tests.fixtures.health_payloads import (
    BOUNDARY_OBSERVED_AT,
    FUTURE_OBSERVED_AT,
    HEALTHY_OBSERVED_AT,
    JUST_STALE_OBSERVED_AT,
    NOW,
    STALE_OBSERVED_AT,
    healthy_schedule_payload,
)


READY_STATUSES = dict.fromkeys(REQUIRED_COMPONENTS, "ready")


class StaticHealthRepository:
    def __init__(self, evidence: HealthEvidence) -> None:
        self.evidence = evidence

    def collect(self, _now: datetime) -> HealthEvidence:
        return self.evidence


class MissingOffsetTimezone(tzinfo):
    def utcoffset(self, _dt: datetime | None) -> None:
        return None

    def dst(self, _dt: datetime | None) -> None:
        return None


def test_health_route_returns_sanitized_degraded_data_quality_signals() -> None:
    app = FastAPI()
    app.include_router(
        create_health_router(
            repository=StaticHealthRepository(
                HealthEvidence(
                    database="ready",
                    migrations="stale",
                    ingestion_observed_at=STALE_OBSERVED_AT,
                    forecast_observed_at=HEALTHY_OBSERVED_AT,
                    schedule_observed_at=None,
                    push="unavailable",
                )
            ),
            now=lambda: NOW,
            ingestion_stale_after_seconds=600,
            forecast_stale_after_seconds=21_600,
            schedule_stale_after_seconds=21_600,
        )
    )

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "checkedAt": "2026-08-31T12:00:00Z",
        "components": {
            "api": {
                "status": "ready",
                "observedAt": None,
                "ageSeconds": None,
                "detail": None,
            },
            "database": {
                "status": "ready",
                "observedAt": None,
                "ageSeconds": None,
                "detail": None,
            },
            "migrations": {
                "status": "stale",
                "observedAt": None,
                "ageSeconds": None,
                "detail": None,
            },
            "ingestion": {
                "status": "stale",
                "observedAt": "2026-08-31T11:00:00Z",
                "ageSeconds": 3600,
                "detail": None,
            },
            "forecast": {
                "status": "ready",
                "observedAt": "2026-08-31T11:55:00Z",
                "ageSeconds": 300,
                "detail": None,
            },
            "schedules": {
                "status": "missing",
                "observedAt": None,
                "ageSeconds": None,
                "detail": None,
            },
            "push": {
                "status": "unavailable",
                "observedAt": None,
                "ageSeconds": None,
                "detail": None,
            },
        },
    }
    body = response.text.lower()
    for forbidden in (
        "password",
        "token",
        "vapid",
        "endpoint",
        "host",
        "live_counts_url",
    ):
        assert forbidden not in body


def test_health_route_preserves_explicit_evidence_statuses_and_future_detail() -> None:
    app = FastAPI()
    app.include_router(
        create_health_router(
            repository=StaticHealthRepository(
                HealthEvidence(
                    database="ready",
                    migrations="ready",
                    ingestion_observed_at=HEALTHY_OBSERVED_AT,
                    forecast_observed_at=None,
                    schedule_observed_at=HEALTHY_OBSERVED_AT,
                    push="ready",
                    ingestion_status="unavailable",
                    forecast_status="missing",
                    schedule_status="stale",
                )
            ),
            now=lambda: NOW,
            ingestion_stale_after_seconds=600,
            forecast_stale_after_seconds=21_600,
            schedule_stale_after_seconds=21_600,
        )
    )

    components = TestClient(app).get("/health").json()["components"]

    assert components["ingestion"] == {
        "status": "unavailable",
        "observedAt": "2026-08-31T11:55:00Z",
        "ageSeconds": 300,
        "detail": None,
    }
    assert components["forecast"] == {
        "status": "missing",
        "observedAt": None,
        "ageSeconds": None,
        "detail": None,
    }
    assert components["schedules"] == {
        "status": "stale",
        "observedAt": "2026-08-31T11:55:00Z",
        "ageSeconds": 300,
        "detail": None,
    }


def test_health_route_preserves_future_timestamp_from_unavailable_evidence() -> None:
    app = FastAPI()
    app.include_router(
        create_health_router(
            repository=StaticHealthRepository(
                HealthEvidence(
                    database="ready",
                    migrations="ready",
                    ingestion_observed_at=HEALTHY_OBSERVED_AT,
                    forecast_observed_at=FUTURE_OBSERVED_AT,
                    schedule_observed_at=HEALTHY_OBSERVED_AT,
                    push="ready",
                    ingestion_status="ready",
                    forecast_status="unavailable",
                    schedule_status="ready",
                )
            ),
            now=lambda: NOW,
            ingestion_stale_after_seconds=600,
            forecast_stale_after_seconds=21_600,
            schedule_stale_after_seconds=21_600,
        )
    )

    forecast = TestClient(app).get("/health").json()["components"]["forecast"]

    assert forecast == {
        "status": "unavailable",
        "observedAt": "2026-08-31T12:00:00.000001Z",
        "ageSeconds": None,
        "detail": "future_timestamp",
    }


class SerializationFailingDatetime(datetime):
    def astimezone(self, *_args: object, **_kwargs: object) -> datetime:
        raise RuntimeError("private serialization failure")


@pytest.mark.parametrize("failure_stage", ["clock", "collect", "classify", "serialize"])
def test_health_route_returns_sanitized_503_for_any_report_failure(
    failure_stage: str,
) -> None:
    evidence = HealthEvidence(
        database="ready",
        migrations="ready",
        ingestion_observed_at=None,
        forecast_observed_at=None,
        schedule_observed_at=None,
        push="ready",
    )
    repository: object = StaticHealthRepository(evidence)

    def now() -> datetime:
        return NOW

    if failure_stage == "clock":

        def now() -> datetime:
            raise RuntimeError("private clock failure password=must-not-appear")

    elif failure_stage == "collect":

        class RaisingRepository:
            def collect(self, _now: datetime) -> HealthEvidence:
                raise RuntimeError("db.example.invalid password=must-not-appear")

        repository = RaisingRepository()
    elif failure_stage == "classify":

        def now() -> datetime:
            return NOW.replace(tzinfo=None)

    elif failure_stage == "serialize":

        def now() -> datetime:
            return SerializationFailingDatetime(
                2026, 8, 31, 12, 0, tzinfo=timezone.utc
            )

    app = FastAPI()
    app.include_router(
        create_health_router(
            repository=repository,  # type: ignore[arg-type]
            now=now,
            ingestion_stale_after_seconds=600,
            forecast_stale_after_seconds=21_600,
            schedule_stale_after_seconds=21_600,
        )
    )

    response = TestClient(app, raise_server_exceptions=False).get("/health")

    assert response.status_code == 503
    assert response.json() == {"detail": "health_unavailable"}
    assert "private" not in response.text.lower()
    assert "password" not in response.text.lower()


def test_health_threshold_settings_use_defaults_and_canonical_schedule_precedence() -> None:
    defaults = build_settings_from_environment({"APP_ENV": "test"})
    canonical = build_settings_from_environment(
        {
            "APP_ENV": "test",
            "INGESTION_STALE_AFTER_SECONDS": "601",
            "FORECAST_STALE_AFTER_SECONDS": "21601",
            "SCHEDULE_STALE_AFTER_SECONDS": "21602",
            "SCHEDULE_MAX_AGE_SECONDS": "private-invalid-legacy-value",
        }
    )

    assert (
        defaults.ingestion_stale_after_seconds,
        defaults.forecast_stale_after_seconds,
        defaults.schedule_stale_after_seconds,
    ) == (600, 21_600, 21_600)
    assert (
        canonical.ingestion_stale_after_seconds,
        canonical.forecast_stale_after_seconds,
        canonical.schedule_stale_after_seconds,
    ) == (601, 21_601, 21_602)


def test_health_threshold_settings_use_schedule_legacy_name_only_when_absent() -> None:
    legacy = build_settings_from_environment(
        {"APP_ENV": "test", "SCHEDULE_MAX_AGE_SECONDS": "7200"}
    )

    assert legacy.schedule_stale_after_seconds == 7200

    with pytest.raises(RuntimeError) as captured:
        build_settings_from_environment(
            {
                "APP_ENV": "test",
                "SCHEDULE_STALE_AFTER_SECONDS": "   ",
                "SCHEDULE_MAX_AGE_SECONDS": "7200",
            }
        )

    assert "SCHEDULE_STALE_AFTER_SECONDS" in str(captured.value)
    assert "7200" not in str(captured.value)


@pytest.mark.parametrize(
    ("environment_name", "threshold_name"),
    [
        ("development", "INGESTION_STALE_AFTER_SECONDS"),
        ("test", "FORECAST_STALE_AFTER_SECONDS"),
        ("production", "SCHEDULE_STALE_AFTER_SECONDS"),
    ],
)
def test_health_threshold_environment_values_are_positive_in_every_mode(
    environment_name: str,
    threshold_name: str,
) -> None:
    with pytest.raises(RuntimeError) as captured:
        build_settings_from_environment(
            {"APP_ENV": environment_name, threshold_name: "0"}
        )

    assert threshold_name in str(captured.value)
    assert "0" not in str(captured.value)


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("ingestion_stale_after_seconds", 0),
        ("forecast_stale_after_seconds", -1),
        ("schedule_stale_after_seconds", True),
        ("ingestion_stale_after_seconds", 600.0),
        ("forecast_stale_after_seconds", "21600"),
    ],
)
def test_explicit_health_threshold_settings_require_positive_nonboolean_integers(
    field_name: str,
    invalid: object,
) -> None:
    with pytest.raises(RuntimeError) as captured:
        Settings(**{field_name: invalid})  # type: ignore[arg-type]

    assert field_name.upper() in str(captured.value)
    assert str(invalid) not in str(captured.value)


def test_create_app_mounts_one_runtime_owned_health_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from server.reclive import push as push_owner
    from server.reclive import settings as settings_owner
    from server.reclive.repositories import push_rules as push_rules_owner
    from server.reclive.runtime import current_runtime

    forecast_path = tmp_path / "forecast.json"
    schedule_path = tmp_path / "facility_hours.json"
    settings = replace(
        Settings.for_test(forecast_json_path=str(forecast_path)),
        facility_hours_json_path=str(schedule_path),
    )
    app = create_app(settings)
    runtime = app.state.runtime
    checked_at = NOW + timedelta(seconds=30)
    runtime.clock = lambda: checked_at
    connection = object()
    connect_calls: list[bool] = []
    readiness_calls: list[str] = []

    def connect(*, autocommit: bool = True) -> object:
        connect_calls.append(autocommit)
        return connection

    runtime.connect = connect

    def readiness(name: str) -> bool:
        assert current_runtime() is runtime
        readiness_calls.append(name)
        return True

    monkeypatch.setattr(
        push_rules_owner, "push_db_available", lambda: readiness("database")
    )
    monkeypatch.setattr(
        settings_owner, "push_vapid_configured", lambda: readiness("vapid")
    )
    monkeypatch.setattr(
        push_owner, "push_identity_configured", lambda: readiness("identity")
    )

    def collect(self: HealthRepository, now: datetime) -> HealthEvidence:
        assert now == checked_at
        assert self.migration_dir == Path(__file__).resolve().parents[2] / "server/migrations"
        assert self.forecast_path == forecast_path
        assert self.schedule_path == schedule_path
        assert self.connect() is connection
        return HealthEvidence(
            database="ready",
            migrations="ready",
            ingestion_observed_at=NOW,
            forecast_observed_at=NOW,
            schedule_observed_at=NOW,
            push="ready" if self.push_status() else "unavailable",
            ingestion_status="ready",
            forecast_status="ready",
            schedule_status="ready",
        )

    monkeypatch.setattr(HealthRepository, "collect", collect)

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert set(response.json()) == {"status", "checkedAt", "components"}
    assert response.json()["checkedAt"] == "2026-08-31T12:00:30Z"
    assert response.json()["components"]["push"]["status"] == "ready"
    assert "schedule" not in response.json()
    assert connect_calls == [False]
    assert readiness_calls == ["database", "vapid", "identity"]
    health_routes = [
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == "/health"
    ]
    assert len(health_routes) == 1


def test_create_app_push_health_status_requires_every_local_readiness_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.reclive import push as push_owner
    from server.reclive import settings as settings_owner
    from server.reclive.repositories import push_rules as push_rules_owner

    statuses: list[str] = []

    def collect(self: HealthRepository, _now: datetime) -> HealthEvidence:
        statuses.append(self.push_status())
        return HealthEvidence(
            database="ready",
            migrations="ready",
            ingestion_observed_at=NOW,
            forecast_observed_at=NOW,
            schedule_observed_at=NOW,
            push=statuses[-1],
            ingestion_status="ready",
            forecast_status="ready",
            schedule_status="ready",
        )

    monkeypatch.setattr(HealthRepository, "collect", collect)
    monkeypatch.setattr(push_rules_owner, "push_db_available", lambda: True)
    monkeypatch.setattr(settings_owner, "push_vapid_configured", lambda: True)
    monkeypatch.setattr(push_owner, "push_identity_configured", lambda: False)
    app = create_app(Settings.for_test())
    app.state.runtime.clock = lambda: NOW

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["components"]["push"]["status"] == "unavailable"
    assert statuses == ["unavailable"]


def test_classify_age_returns_ready_with_utc_timestamp_and_exact_age() -> None:
    observed_at = HEALTHY_OBSERVED_AT.astimezone(timezone(timedelta(hours=-5)))

    component = classify_age(observed_at, NOW, 600)

    assert component == ComponentHealth(
        status="ready",
        observedAt="2026-08-31T11:55:00Z",
        ageSeconds=300,
        detail=None,
    )


def test_classify_age_marks_missing_and_stale_without_zero_age_substitution() -> None:
    missing = classify_age(None, NOW, 600)
    stale = classify_age(STALE_OBSERVED_AT, NOW, 600)

    assert missing == ComponentHealth(
        status="missing",
        observedAt=None,
        ageSeconds=None,
        detail=None,
    )
    assert stale == ComponentHealth(
        status="stale",
        observedAt="2026-08-31T11:00:00Z",
        ageSeconds=3600,
        detail=None,
    )


def test_classify_age_compares_exact_elapsed_time_before_public_integer_age() -> None:
    at_boundary = classify_age(BOUNDARY_OBSERVED_AT, NOW, 600)
    just_stale = classify_age(JUST_STALE_OBSERVED_AT, NOW, 600)

    assert (at_boundary.status, at_boundary.ageSeconds) == ("ready", 600)
    assert (just_stale.status, just_stale.ageSeconds) == ("stale", 600)


def test_classify_age_rejects_future_observation_as_unavailable() -> None:
    component = classify_age(FUTURE_OBSERVED_AT, NOW, 600)

    assert component == ComponentHealth(
        status="unavailable",
        observedAt="2026-08-31T12:00:00.000001Z",
        ageSeconds=None,
        detail="future_timestamp",
    )


@pytest.mark.parametrize(
    ("observed_at", "now"),
    [
        (datetime(2026, 8, 31, 11, 55), NOW),
        (HEALTHY_OBSERVED_AT, datetime(2026, 8, 31, 12, 0)),
        (datetime(2026, 8, 31, 11, 55, tzinfo=MissingOffsetTimezone()), NOW),
        (HEALTHY_OBSERVED_AT, datetime(2026, 8, 31, 12, 0, tzinfo=MissingOffsetTimezone())),
    ],
)
def test_classify_age_rejects_timestamps_without_usable_utc_offsets(
    observed_at: datetime,
    now: datetime,
) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        classify_age(observed_at, now, 600)


@pytest.mark.parametrize("threshold", [0, -1, True, False, nan, inf, -inf])
def test_classify_age_rejects_nonpositive_nonfinite_or_boolean_thresholds(
    threshold: int | float,
) -> None:
    with pytest.raises(ValueError, match="finite positive number"):
        classify_age(HEALTHY_OBSERVED_AT, NOW, threshold)


def test_utc_iso_requires_a_usable_offset() -> None:
    invalid = datetime(2026, 8, 31, 12, 0, tzinfo=MissingOffsetTimezone())

    with pytest.raises(ValueError, match="timezone-aware"):
        utc_iso(invalid)


def test_overall_status_requires_exactly_seven_ready_components() -> None:
    assert overall_status(READY_STATUSES) == "ready"
    assert overall_status({**READY_STATUSES, "database": "unavailable"}) == "degraded"
    assert overall_status({}) == "degraded"
    assert overall_status({name: "ready" for name in REQUIRED_COMPONENTS[:-1]}) == "degraded"
    assert overall_status({**READY_STATUSES, "extra": "ready"}) == "degraded"


def test_public_health_types_reject_open_status_and_detail_values() -> None:
    with pytest.raises(ValueError, match="component status"):
        ComponentHealth(status="ok", observedAt=None, ageSeconds=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="component detail"):
        ComponentHealth(
            status="unavailable",
            observedAt=None,
            ageSeconds=None,
            detail="database host failed",  # type: ignore[arg-type]
        )


def test_public_health_types_reject_unhashable_status_and_detail_objects() -> None:
    with pytest.raises(ValueError, match="component status"):
        ComponentHealth([], None, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="component detail"):
        ComponentHealth(
            "unavailable",
            None,
            None,
            [],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="overall status"):
        HealthReport(  # type: ignore[arg-type]
            status="ok",
            checkedAt="2026-08-31T12:00:00Z",
            components={
                name: ComponentHealth("ready", None, None) for name in REQUIRED_COMPONENTS
            },
        )


@pytest.mark.parametrize(
    ("observed_at", "age_seconds"),
    [
        ("2026-08-31T07:00:00-05:00", 300),
        ("not-a-timestamp", 300),
        ("2026-08-31T11:55:00Z", -1),
        ("2026-08-31T11:55:00Z", True),
    ],
)
def test_component_health_rejects_noncanonical_timestamps_and_invalid_ages(
    observed_at: str,
    age_seconds: int,
) -> None:
    with pytest.raises(ValueError):
        ComponentHealth("ready", observed_at, age_seconds)


def test_future_timestamp_detail_is_closed_to_its_unavailable_shape() -> None:
    with pytest.raises(ValueError, match="future_timestamp"):
        ComponentHealth(
            "ready",
            "2026-08-31T12:00:00.000001Z",
            None,
            "future_timestamp",
        )
    with pytest.raises(ValueError, match="future_timestamp"):
        ComponentHealth("unavailable", None, None, "future_timestamp")
    with pytest.raises(ValueError, match="future_timestamp"):
        ComponentHealth(
            "unavailable",
            "2026-08-31T12:00:00.000001Z",
            0,
            "future_timestamp",
        )


def test_health_report_rejects_noncanonical_checked_at() -> None:
    with pytest.raises(ValueError, match="checkedAt"):
        HealthReport(
            status="ready",
            checkedAt="2026-08-31T07:00:00-05:00",
            components={
                name: ComponentHealth("ready", None, None) for name in REQUIRED_COMPONENTS
            },
        )


def test_health_report_has_exact_shape_and_immutable_component_mapping() -> None:
    source = {
        name: ComponentHealth(status="ready", observedAt=None, ageSeconds=None)
        for name in REQUIRED_COMPONENTS
    }
    report = HealthReport(
        status="ready",
        checkedAt="2026-08-31T12:00:00Z",
        components=source,
    )
    source["api"] = ComponentHealth("unavailable", None, None)

    assert report.to_dict() == {
        "status": "ready",
        "checkedAt": "2026-08-31T12:00:00Z",
        "components": {
            name: {
                "status": "ready",
                "observedAt": None,
                "ageSeconds": None,
                "detail": None,
            }
            for name in REQUIRED_COMPONENTS
        },
    }
    assert report.components["api"].status == "ready"
    with pytest.raises(TypeError):
        report.components["api"] = ComponentHealth("unavailable", None, None)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        report.status = "degraded"  # type: ignore[misc]


def test_health_report_rejects_missing_extra_or_noncomponent_entries() -> None:
    ready_components = {
        name: ComponentHealth("ready", None, None) for name in REQUIRED_COMPONENTS
    }

    with pytest.raises(ValueError, match="exact required components"):
        HealthReport(
            status="degraded",
            checkedAt="2026-08-31T12:00:00Z",
            components={name: ready_components[name] for name in REQUIRED_COMPONENTS[:-1]},
        )
    with pytest.raises(ValueError, match="exact required components"):
        HealthReport(
            status="degraded",
            checkedAt="2026-08-31T12:00:00Z",
            components={**ready_components, "extra": ComponentHealth("ready", None, None)},
        )
    with pytest.raises(ValueError, match="ComponentHealth"):
        HealthReport(
            status="degraded",
            checkedAt="2026-08-31T12:00:00Z",
            components={**ready_components, "api": object()},  # type: ignore[dict-item]
        )


def test_health_report_rejects_status_inconsistent_with_its_components() -> None:
    ready_components = {
        name: ComponentHealth("ready", None, None) for name in REQUIRED_COMPONENTS
    }
    degraded_components = {
        **ready_components,
        "database": ComponentHealth("unavailable", None, None),
    }

    with pytest.raises(ValueError, match="match component readiness"):
        HealthReport(
            status="degraded",
            checkedAt="2026-08-31T12:00:00Z",
            components=ready_components,
        )
    with pytest.raises(ValueError, match="match component readiness"):
        HealthReport(
            status="ready",
            checkedAt="2026-08-31T12:00:00Z",
            components=degraded_components,
        )


class HealthCursor:
    def __init__(
        self,
        *,
        migration_rows: list[tuple[object, ...]],
        ingestion_row: tuple[object, ...] | None,
        failing_queries: frozenset[str] = frozenset(),
    ) -> None:
        self.migration_rows = migration_rows
        self.ingestion_row = ingestion_row
        self.failing_queries = failing_queries
        self.executed: list[tuple[str, object]] = []
        self.active_query = ""

    def execute(self, sql: str, params: object = None) -> None:
        self.executed.append((sql, params))
        if "schema_migrations" in sql:
            self.active_query = "migrations"
        elif "ingestion_runs" in sql:
            self.active_query = "ingestion"
        else:
            raise AssertionError("unexpected health query")
        if self.active_query in self.failing_queries:
            raise RuntimeError("private query failure must not escape")

    def fetchone(self) -> tuple[object, ...] | None:
        assert self.active_query == "ingestion"
        return self.ingestion_row

    def fetchall(self) -> list[tuple[object, ...]]:
        assert self.active_query == "migrations"
        return list(self.migration_rows)

    def __enter__(self) -> "HealthCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class HealthConnection:
    def __init__(self, cursor: HealthCursor) -> None:
        self.cursor_instance = cursor
        self.closed = False

    def cursor(self) -> HealthCursor:
        return self.cursor_instance

    def close(self) -> None:
        self.closed = True


def write_repository_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    forecast = tmp_path / "forecast.json"
    forecast.write_text('{"generatedAt":"2026-08-31T11:55:00Z"}', encoding="utf-8")
    schedule = tmp_path / "facility_hours.json"
    schedule.write_text(json.dumps(healthy_schedule_payload()), encoding="utf-8")
    migration_dir = tmp_path / "migrations"
    migration_dir.mkdir()
    migration = migration_dir / "0001_core_history.sql"
    migration.write_text("SELECT 1;\n", encoding="utf-8")
    return forecast, schedule, migration_dir, snapshot_migration(migration).checksum


def test_repository_collects_only_sanitized_aggregate_health_evidence(
    tmp_path: Path,
) -> None:
    forecast, schedule, migration_dir, checksum = write_repository_artifacts(tmp_path)
    cursor = HealthCursor(
        migration_rows=[("0001_core_history.sql", checksum)],
        ingestion_row=(datetime(2026, 8, 31, 11, 58),),
    )
    connection = HealthConnection(cursor)
    repository = HealthRepository(
        connect=lambda: connection,
        migration_dir=migration_dir,
        forecast_path=forecast,
        schedule_path=schedule,
        push_status=lambda: "ready",
    )

    evidence = repository.collect(NOW)

    assert (evidence.database, evidence.migrations, evidence.push) == (
        "ready",
        "ready",
        "ready",
    )
    assert evidence.ingestion_observed_at == datetime(
        2026, 8, 31, 11, 58, tzinfo=timezone.utc
    )
    assert evidence.forecast_observed_at == HEALTHY_OBSERVED_AT
    assert evidence.schedule_observed_at == datetime(
        2026, 8, 31, 11, 50, tzinfo=timezone.utc
    )
    assert (
        evidence.ingestion_status,
        evidence.forecast_status,
        evidence.schedule_status,
    ) == ("ready", "ready", "ready")
    assert all(sql.lstrip().upper().startswith("SELECT ") for sql, _ in cursor.executed)
    assert all(
        "endpoint" not in sql.lower() and "subscription" not in sql.lower()
        for sql, _ in cursor.executed
    )
    assert connection.closed is True


def test_repository_keeps_database_migrations_ingestion_and_push_probes_independent(
    tmp_path: Path,
) -> None:
    forecast, schedule, migration_dir, _checksum = write_repository_artifacts(tmp_path)
    forecast.write_text("{malformed-private-marker", encoding="utf-8")
    cursor = HealthCursor(
        migration_rows=[],
        ingestion_row=(datetime(2026, 8, 31, 11, 58),),
        failing_queries=frozenset({"migrations"}),
    )
    connection = HealthConnection(cursor)
    repository = HealthRepository(
        connect=lambda: connection,
        migration_dir=migration_dir,
        forecast_path=forecast,
        schedule_path=schedule,
        push_status=lambda: (_ for _ in ()).throw(
            RuntimeError("private push failure must not escape")
        ),
    )

    evidence = repository.collect(NOW)

    assert evidence.database == "ready"
    assert evidence.migrations == "unavailable"
    assert evidence.ingestion_status == "ready"
    assert evidence.ingestion_observed_at == datetime(
        2026, 8, 31, 11, 58, tzinfo=timezone.utc
    )
    assert evidence.forecast_status == "unavailable"
    assert evidence.schedule_status == "ready"
    assert evidence.push == "unavailable"
    assert [query for query, _params in cursor.executed] == [
        "SELECT filename, checksum FROM schema_migrations ORDER BY filename",
        "SELECT completed_at FROM ingestion_runs WHERE status = 'succeeded' "
        "ORDER BY completed_at DESC, id DESC LIMIT 1",
    ]
    assert "private" not in repr(evidence).lower()


def test_repository_distinguishes_unreachable_database_from_missing_artifacts(
    tmp_path: Path,
) -> None:
    repository = HealthRepository(
        connect=lambda: (_ for _ in ()).throw(
            RuntimeError("db.example.invalid password=not-for-output")
        ),
        migration_dir=tmp_path,
        forecast_path=tmp_path / "forecast.json",
        schedule_path=tmp_path / "facility_hours.json",
        push_status=lambda: "invalid-private-status",
    )

    evidence = repository.collect(NOW)

    assert (evidence.database, evidence.migrations, evidence.push) == (
        "unavailable",
        "unavailable",
        "unavailable",
    )
    assert (evidence.ingestion_status, evidence.ingestion_observed_at) == (
        "unavailable",
        None,
    )
    assert (evidence.forecast_status, evidence.forecast_observed_at) == (
        "missing",
        None,
    )
    assert (evidence.schedule_status, evidence.schedule_observed_at) == (
        "missing",
        None,
    )
    assert "password" not in repr(evidence).lower()


def test_repository_uses_effective_0003_checksum_and_detects_ledger_mismatch() -> None:
    migration_dir = Path(__file__).resolve().parents[2] / "server" / "migrations"
    recorded = [
        (path.name, snapshot_migration(path).checksum)
        for path in migration_files(migration_dir)
    ]

    ready_connection = HealthConnection(
        HealthCursor(migration_rows=recorded, ingestion_row=None)
    )
    ready = HealthRepository(
        connect=lambda: ready_connection,
        migration_dir=migration_dir,
        forecast_path=Path("absent-forecast.json"),
        schedule_path=Path("absent-schedule.json"),
        push_status=lambda: "unavailable",
    ).collect(NOW)

    mismatched = [
        (filename, "0" * 64 if filename == "0003_push_rule_lifecycle.sql" else checksum)
        for filename, checksum in recorded
    ]
    stale_connection = HealthConnection(
        HealthCursor(migration_rows=mismatched, ingestion_row=None)
    )
    stale = HealthRepository(
        connect=lambda: stale_connection,
        migration_dir=migration_dir,
        forecast_path=Path("absent-forecast.json"),
        schedule_path=Path("absent-schedule.json"),
        push_status=lambda: "unavailable",
    ).collect(NOW)

    assert ready.migrations == "ready"
    assert stale.database == "ready"
    assert stale.migrations == "stale"


@pytest.mark.parametrize(
    "payload",
    [
        b'{"generatedAt":"2026-08-31T11:55:00"}',
        b'{"generatedAt":"not-a-time"}',
        b'{"generatedAt":"2026-08-31T11:55:00Z","generatedAt":"2026-08-31T11:54:00Z"}',
    ],
)
def test_forecast_artifact_rejects_naive_malformed_or_ambiguous_timestamps(
    tmp_path: Path,
    payload: bytes,
) -> None:
    path = tmp_path / "forecast.json"
    path.write_bytes(payload)

    repository = HealthRepository(
        connect=lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
        migration_dir=tmp_path,
        forecast_path=path,
        schedule_path=tmp_path / "missing-schedule.json",
        push_status=lambda: "unavailable",
    )

    evidence = repository.collect(NOW)

    assert generated_at(path) is None
    assert (evidence.forecast_status, evidence.forecast_observed_at) == (
        "unavailable",
        None,
    )


def test_health_artifact_reader_is_byte_bounded_not_stat_then_unbounded() -> None:
    class BoundedOnlyStream(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            if size != MAX_HEALTH_ARTIFACT_BYTES + 1:
                raise AssertionError("health artifact read was not byte bounded")
            return super().read(size)

    class BoundedOnlyPath:
        def open(self, mode: str) -> BoundedOnlyStream:
            assert mode == "rb"
            return BoundedOnlyStream(
                b'{"generatedAt":"2026-08-31T11:55:00Z"}'
            )

    assert generated_at(BoundedOnlyPath()) == HEALTHY_OBSERVED_AT  # type: ignore[arg-type]


def test_oversized_forecast_artifact_is_unavailable_not_missing(tmp_path: Path) -> None:
    path = tmp_path / "forecast.json"
    path.write_bytes(b" " * (MAX_HEALTH_ARTIFACT_BYTES + 1))
    repository = HealthRepository(
        connect=lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
        migration_dir=tmp_path,
        forecast_path=path,
        schedule_path=tmp_path / "missing-schedule.json",
        push_status=lambda: "unavailable",
    )

    evidence = repository.collect(NOW)

    assert (evidence.forecast_status, evidence.forecast_observed_at) == (
        "unavailable",
        None,
    )


def test_future_forecast_artifact_is_unavailable_with_bounded_timestamp_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "forecast.json"
    path.write_text(
        '{"generatedAt":"2026-08-31T12:00:00.000001Z"}',
        encoding="utf-8",
    )
    repository = HealthRepository(
        connect=lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
        migration_dir=tmp_path,
        forecast_path=path,
        schedule_path=tmp_path / "missing-schedule.json",
        push_status=lambda: "unavailable",
    )

    evidence = repository.collect(NOW)

    assert (evidence.forecast_status, evidence.forecast_observed_at) == (
        "unavailable",
        FUTURE_OBSERVED_AT,
    )


def test_unreadable_forecast_artifact_is_unavailable_not_missing(tmp_path: Path) -> None:
    class UnreadablePath:
        def open(self, _mode: str) -> io.BytesIO:
            raise PermissionError("private unreadable path must not escape")

    repository = HealthRepository(
        connect=lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
        migration_dir=tmp_path,
        forecast_path=UnreadablePath(),  # type: ignore[arg-type]
        schedule_path=tmp_path / "missing-schedule.json",
        push_status=lambda: "unavailable",
    )

    evidence = repository.collect(NOW)

    assert (evidence.forecast_status, evidence.forecast_observed_at) == (
        "unavailable",
        None,
    )
    assert "private" not in repr(evidence).lower()


def test_local_migration_snapshot_failure_does_not_hide_database_or_ingestion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    forecast, schedule, migration_dir, checksum = write_repository_artifacts(tmp_path)
    migration_path = migration_dir / "0001_core_history.sql"
    original_read_bytes = Path.read_bytes

    def fail_selected_snapshot(path: Path) -> bytes:
        if path == migration_path:
            raise PermissionError("private migration path must not escape")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_selected_snapshot)
    cursor = HealthCursor(
        migration_rows=[("0001_core_history.sql", checksum)],
        ingestion_row=(datetime(2026, 8, 31, 11, 58),),
    )
    connection = HealthConnection(cursor)
    repository = HealthRepository(
        connect=lambda: connection,
        migration_dir=migration_dir,
        forecast_path=forecast,
        schedule_path=schedule,
        push_status=lambda: "ready",
    )

    evidence = repository.collect(NOW)

    assert evidence.database == "ready"
    assert evidence.migrations == "unavailable"
    assert evidence.ingestion_status == "ready"
    assert evidence.ingestion_observed_at == datetime(
        2026, 8, 31, 11, 58, tzinfo=timezone.utc
    )


def test_schedule_observation_uses_oldest_successful_source_not_generation_time(
    tmp_path: Path,
) -> None:
    path = tmp_path / "facility_hours.json"
    path.write_text(
        json.dumps(
            healthy_schedule_payload(
                generated_at="2026-08-31T11:59:59Z",
                nick_observed_at="2026-08-31T10:00:00Z",
                bakke_observed_at="2026-08-31T11:59:00Z",
            )
        ),
        encoding="utf-8",
    )

    repository = HealthRepository(
        connect=lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
        migration_dir=tmp_path,
        forecast_path=tmp_path / "missing-forecast.json",
        schedule_path=path,
        push_status=lambda: "unavailable",
    )
    evidence = repository.collect(NOW)

    assert healthy_schedule_generated_at(path, now=NOW) == datetime(
        2026, 8, 31, 10, 0, tzinfo=timezone.utc
    )
    assert evidence.schedule_status == "ready"
    assert evidence.schedule_observed_at == datetime(
        2026, 8, 31, 10, 0, tzinfo=timezone.utc
    )


def test_schedule_artifact_preserves_valid_stale_state(tmp_path: Path) -> None:
    payload = healthy_schedule_payload(nick_observed_at="2026-08-31T09:00:00Z")
    facilities = payload["facilities"]
    assert isinstance(facilities, list)
    nick = facilities[0]
    assert isinstance(nick, dict)
    nick.update(
        {
            "status": "stale",
            "stale": True,
            "error": "Official hours could not be refreshed.",
            "errorCategory": "upstream_timeout",
        }
    )
    payload["okCount"] = 1
    path = tmp_path / "facility_hours.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    repository = HealthRepository(
        connect=lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
        migration_dir=tmp_path,
        forecast_path=tmp_path / "missing-forecast.json",
        schedule_path=path,
        push_status=lambda: "unavailable",
    )
    evidence = repository.collect(NOW)

    assert evidence.schedule_status == "stale"
    assert evidence.schedule_observed_at == datetime(
        2026, 8, 31, 9, 0, tzinfo=timezone.utc
    )


@pytest.mark.parametrize("damage", ["incomplete", "duplicate", "oversized"])
def test_invalid_or_ambiguous_schedule_artifact_is_unavailable_not_missing(
    tmp_path: Path,
    damage: str,
) -> None:
    path = tmp_path / "facility_hours.json"
    if damage == "incomplete":
        payload = healthy_schedule_payload()
        payload["facilities"] = []
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif damage == "duplicate":
        serialized = json.dumps(healthy_schedule_payload())
        path.write_text(serialized[:-1] + ', "okCount": 2}', encoding="utf-8")
    else:
        path.write_bytes(b" " * (MAX_HEALTH_ARTIFACT_BYTES + 1))

    repository = HealthRepository(
        connect=lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
        migration_dir=tmp_path,
        forecast_path=tmp_path / "missing-forecast.json",
        schedule_path=path,
        push_status=lambda: "unavailable",
    )
    evidence = repository.collect(NOW)

    assert healthy_schedule_generated_at(path, now=NOW) is None
    assert (evidence.schedule_status, evidence.schedule_observed_at) == (
        "unavailable",
        None,
    )


def collect_with_forecast_payload(
    tmp_path: Path,
    payload: bytes,
) -> tuple[HealthEvidence, HealthConnection, list[str]]:
    forecast, schedule, migration_dir, checksum = write_repository_artifacts(tmp_path)
    forecast.write_bytes(payload)
    connection = HealthConnection(
        HealthCursor(
            migration_rows=[("0001_core_history.sql", checksum)],
            ingestion_row=(datetime(2026, 8, 31, 11, 58),),
        )
    )
    push_events: list[str] = []
    repository = HealthRepository(
        connect=lambda: connection,
        migration_dir=migration_dir,
        forecast_path=forecast,
        schedule_path=schedule,
        push_status=lambda: push_events.append("push") or "ready",
    )
    return repository.collect(NOW), connection, push_events


@pytest.mark.parametrize("constant", [b"NaN", b"Infinity", b"-Infinity"])
def test_repository_rejects_nonstandard_json_constants_without_stopping_later_probes(
    tmp_path: Path,
    constant: bytes,
) -> None:
    evidence, connection, push_events = collect_with_forecast_payload(
        tmp_path,
        b'{"generatedAt":"2026-08-31T11:55:00Z","diagnostic":'
        + constant
        + b"}",
    )

    assert evidence.forecast_status == "unavailable"
    assert evidence.forecast_observed_at is None
    assert evidence.schedule_status == "ready"
    assert evidence.push == "ready"
    assert connection.closed is True
    assert push_events == ["push"]


def test_repository_contains_deep_json_recursion_and_continues_later_probes(
    tmp_path: Path,
) -> None:
    nested_value = b"[" * 10_000 + b"0" + b"]" * 10_000
    evidence, connection, push_events = collect_with_forecast_payload(
        tmp_path,
        b'{"generatedAt":"2026-08-31T11:55:00Z","nested":'
        + nested_value
        + b"}",
    )

    assert evidence.forecast_status == "unavailable"
    assert evidence.forecast_observed_at is None
    assert evidence.schedule_status == "ready"
    assert evidence.push == "ready"
    assert connection.closed is True
    assert push_events == ["push"]


@pytest.mark.parametrize("value", [b"0", b"-1.25", b"1e300", b'"NaN"'])
def test_repository_accepts_standard_finite_json_controls(
    tmp_path: Path,
    value: bytes,
) -> None:
    evidence, connection, push_events = collect_with_forecast_payload(
        tmp_path,
        b'{"generatedAt":"2026-08-31T11:55:00Z","diagnostic":'
        + value
        + b"}",
    )

    assert evidence.forecast_status == "ready"
    assert evidence.forecast_observed_at == HEALTHY_OBSERVED_AT
    assert evidence.schedule_status == "ready"
    assert evidence.push == "ready"
    assert connection.closed is True
    assert push_events == ["push"]


@pytest.mark.parametrize(
    "timestamp",
    [
        "0001-01-01T00:00:00+14:00",
        "9999-12-31T23:59:59-14:00",
    ],
)
def test_repository_contains_utc_normalization_overflow_and_continues_later_probes(
    tmp_path: Path,
    timestamp: str,
) -> None:
    evidence, connection, push_events = collect_with_forecast_payload(
        tmp_path,
        json.dumps({"generatedAt": timestamp}).encode("utf-8"),
    )

    assert evidence.forecast_status == "unavailable"
    assert evidence.forecast_observed_at is None
    assert evidence.schedule_status == "ready"
    assert evidence.push == "ready"
    assert connection.closed is True
    assert push_events == ["push"]


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        (
            "2026-08-31T14:00:00+14:00",
            datetime(2026, 8, 31, 0, 0, tzinfo=timezone.utc),
        ),
        ("2026-08-30T22:00:00-14:00", NOW),
    ],
)
def test_repository_accepts_ordinary_aware_offset_controls(
    tmp_path: Path,
    timestamp: str,
    expected: datetime,
) -> None:
    evidence, connection, push_events = collect_with_forecast_payload(
        tmp_path,
        json.dumps({"generatedAt": timestamp}).encode("utf-8"),
    )

    assert evidence.forecast_status == "ready"
    assert evidence.forecast_observed_at == expected
    assert evidence.schedule_status == "ready"
    assert evidence.push == "ready"
    assert connection.closed is True
    assert push_events == ["push"]
