# RecLive Operations and Health Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a sanitized, evidence-based operational health contract plus secure configuration, deployment, security, and incident documentation for the fully hardened RecLive application.

**Architecture:** Health is a read-only backend boundary. Pure helpers classify timestamps and component states; repository probes gather only migration, ingestion, database, forecast, schedule, and push readiness facts; one API route serializes those facts into a stable public-safe response. Structured event logging uses an allowlist of scalar fields so compatibility entry points continue to run without writing secrets or customer data to logs. Documentation is treated as an executable operator interface: tests assert every required command, privacy boundary, and release condition.

**Tech Stack:** FastAPI, PyMySQL, Python 3.12, pytest, HTTPX, Ruff, React 19, Vite 7, Vitest, Playwright Chromium, MySQL 8.4, GitHub Actions, Gitleaks.

**Spec:** `docs/superpowers/specs/2026-08-31-reclive-security-data-trust-design.md`

## Global Constraints

- This is Phase 10 only and begins only after Phase 1 through Phase 9 are green on `hardening/reclive-security-data-trust`.
- The master plan's phase-level commit policy is authoritative: task commit snippets describe staging/review scope only; make the single Phase 10 source commit after Task 7.
- Preserve IDs `1186` and `1656`, routes `/nick` and `/bakke`, product behavior, forecasting, push notifications, installation, and executable entry points `server/gym_fetch.py`, `server/forecast_api.py`, `server/forecast_job.py`, and `server/facility_hours_fetch.py`.
- Health endpoints are read-only and expose only readiness/state, UTC ISO-8601 timestamps, ages, counts, and bounded category strings. They never expose database hosts, database names, credentials, upstream URLs, API keys, VAPID private material, admin tokens, push endpoints, request bodies, exception traces, or environment-variable values.
- Runtime configuration validation reports variable names and reasons only. It does not print the rejected values.
- Production fails closed for absent required configuration, `change_me`, `YOUR_ACCOUNT_API_KEY`, wildcard CORS, or enabled admin routes without a strong token; no task changes current production credentials or contacts providers.
- Default staleness thresholds are configuration owned: ingestion `600` seconds, forecast `21600` seconds, and schedules `21600` seconds. `SCHEDULE_STALE_AFTER_SECONDS` is the active name; `SCHEDULE_MAX_AGE_SECONDS` is accepted only as its backward-compatible alias. A nonpositive threshold is invalid in every environment.
- All database timestamps are UTC with microsecond precision. Chicago time remains a presentation/business-time concern and is not emitted by health status.
- Backend tests belong in `tests/backend/`, reusable fixture payloads in `tests/fixtures/`, frontend unit/component tests in `src/**/*.test.ts(x)`, and browser tests in `tests/e2e/`. This phase has no UI behavior change, so it adds no Playwright test unless a later implementation changes the public dashboard.
- Use red-green-refactor: each behavior begins with the named focused failing test, ends with the named focused passing test, and does not train XGBoost in tests or CI.
- Do not log environment dictionaries, SQL values, raw provider responses, raw push subscriptions, or generated secrets. Do not merge, deploy, rewrite history, force-push, edit `.env`, or commit credentials, dumps, artifacts, or private backup bundles.

## File Map and Contract Boundaries

| File(s) | Responsibility | Interface produced |
| --- | --- | --- |
| `server/reclive/health.py` | Pure timestamp/overall-state classification and public response types | `classify_age`, `component`, `overall_status`, `HealthReport` |
| `server/reclive/health_repository.py` | Read-only MySQL/file probes for migrations and ingestion state | `HealthRepository.collect(now) -> HealthEvidence` |
| `server/reclive/api/health.py` | FastAPI dependency and `GET /health` route | `get_health_report() -> HealthReport` |
| `server/reclive/observability.py` | Allowlisted JSON event logger | `log_event(event: str, **fields: Scalar) -> None` |
| `server/forecast_api.py` | Compatibility composition root that mounts the health route and uses the Phase 9 repository/settings modules | unchanged executable API entry point |
| `server/gym_fetch.py`, `server/forecast_job.py`, `server/facility_hours_fetch.py` | Compatibility entry points emitting sanitized lifecycle events | event names `ingestion.completed`, `forecast.completed`, `schedules.completed`, and bounded failure categories |
| `tests/fixtures/health_payloads.py` | Deterministic timestamp/evidence fixtures | `healthy_evidence`, `stale_evidence`, `unavailable_database_evidence` |
| `tests/backend/test_health.py` | Pure health classification and FastAPI response contracts | regression suite with no provider or XGBoost invocation |
| `tests/backend/test_observability.py` | Logging privacy/allowlist contract | JSON-line log contract |
| `tests/backend/test_operations_docs.py` | Documentation and configuration safety contract | validates required operator material without reading `.env` |
| `.env.example` | Placeholder-only public/private configuration reference | documented names without usable values |
| `README.md` | Local setup, deployment, scheduled jobs, health check, tests, and re-clone guidance | copy-paste-safe operator commands |
| `SECURITY.md` | Private reporting, secret handling, rotation, and history-rewrite limitations | repository security policy |
| `docs/operations/runbook.md` | Preflight, release, routine checks, incident response, and rollback-safe operational procedure | on-call runbook |

---

### Task 1: Define and test the pure health-state contract

**Files:**
- Create: `server/reclive/health.py`
- Create: `tests/fixtures/health_payloads.py`
- Create: `tests/backend/test_health.py`

**Interfaces:**
- Consumes: timezone-aware UTC `datetime` values, configured positive threshold seconds, and privacy-safe component evidence.
- Produces: `ComponentHealth(status: Literal["ready", "stale", "missing", "unavailable"], observedAt: str | None, ageSeconds: int | None, detail: str | None)`, `HealthReport(status: Literal["ready", "degraded"], checkedAt: str, components: dict[str, ComponentHealth])`, and `classify_age(observed_at, now, stale_after_seconds) -> ComponentHealth`.

- [ ] **Step 1: Write failing pure-contract tests and deterministic evidence fixtures**

```python
# tests/fixtures/health_payloads.py
from datetime import datetime, timezone

NOW = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
HEALTHY_OBSERVED_AT = datetime(2026, 8, 31, 11, 55, tzinfo=timezone.utc)
STALE_OBSERVED_AT = datetime(2026, 8, 31, 11, 0, tzinfo=timezone.utc)
```

```python
# tests/backend/test_health.py
from datetime import datetime, timezone

import pytest

from server.reclive.health import classify_age, overall_status
from tests.fixtures.health_payloads import HEALTHY_OBSERVED_AT, NOW, STALE_OBSERVED_AT


def test_classify_age_returns_ready_with_utc_timestamp_and_exact_age() -> None:
    component = classify_age(HEALTHY_OBSERVED_AT, NOW, 600)
    assert component.status == "ready"
    assert component.observedAt == "2026-08-31T11:55:00Z"
    assert component.ageSeconds == 300
    assert component.detail is None


def test_classify_age_marks_missing_and_stale_without_substituting_a_zero_age() -> None:
    missing = classify_age(None, NOW, 600)
    stale = classify_age(STALE_OBSERVED_AT, NOW, 600)
    assert (missing.status, missing.observedAt, missing.ageSeconds) == ("missing", None, None)
    assert (stale.status, stale.ageSeconds) == ("stale", 3600)


def test_classify_age_rejects_naive_or_nonpositive_configuration() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        classify_age(datetime(2026, 8, 31, 12, 0), NOW, 600)
    with pytest.raises(ValueError, match="positive"):
        classify_age(HEALTHY_OBSERVED_AT, NOW, 0)


def test_overall_status_is_degraded_if_any_required_component_is_not_ready() -> None:
    assert overall_status({"api": "ready", "database": "ready", "migrations": "ready", "ingestion": "ready", "forecast": "ready", "schedules": "ready", "push": "ready"}) == "ready"
    assert overall_status({"api": "ready", "database": "unavailable", "migrations": "unavailable", "ingestion": "missing", "forecast": "stale", "schedules": "ready", "push": "ready"}) == "degraded"
```

- [ ] **Step 2: Run the focused test to observe the missing health module**

Run: `python -m pytest tests/backend/test_health.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'server.reclive.health'`.

- [ ] **Step 3: Implement immutable public-safe health types and classification**

```python
# server/reclive/health.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Literal

ComponentStatus = Literal["ready", "stale", "missing", "unavailable"]
OverallStatus = Literal["ready", "degraded"]


def utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ComponentHealth:
    status: ComponentStatus
    observedAt: str | None
    ageSeconds: int | None
    detail: str | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        return asdict(self)


@dataclass(frozen=True)
class HealthReport:
    status: OverallStatus
    checkedAt: str
    components: dict[str, ComponentHealth]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "checkedAt": self.checkedAt,
            "components": {name: component.to_dict() for name, component in self.components.items()},
        }


def classify_age(observed_at: datetime | None, now: datetime, stale_after_seconds: int) -> ComponentHealth:
    if now.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    if stale_after_seconds <= 0:
        raise ValueError("stale threshold must be positive")
    if observed_at is None:
        return ComponentHealth(status="missing", observedAt=None, ageSeconds=None)
    if observed_at.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    age_seconds = max(0, int((now.astimezone(timezone.utc) - observed_at.astimezone(timezone.utc)).total_seconds()))
    status: ComponentStatus = "ready" if age_seconds <= stale_after_seconds else "stale"
    return ComponentHealth(status=status, observedAt=utc_iso(observed_at), ageSeconds=age_seconds)


def overall_status(component_statuses: dict[str, str]) -> OverallStatus:
    return "ready" if all(status == "ready" for status in component_statuses.values()) else "degraded"
```

- [ ] **Step 4: Run the pure health test and static lint check**

Run: `python -m pytest tests/backend/test_health.py -q && ruff check server/reclive/health.py tests/backend/test_health.py`

Expected: PASS; four tests pass and no Ruff finding is emitted.

- [ ] **Step 5: Commit the pure data-quality boundary**

```bash
git add server/reclive/health.py tests/fixtures/health_payloads.py tests/backend/test_health.py
git commit -m "feat: define operational health contract"
```

### Task 2: Collect read-only readiness, freshness, and migration evidence

**Files:**
- Create: `server/reclive/health_repository.py`
- Modify: `tests/backend/test_health.py`

**Interfaces:**
- Consumes: the Phase 1 `schema_migrations`, Phase 2 `ingestion_runs`, the configured forecast JSON path, the configured facility-hours JSON path, and Phase 5 push readiness provider. No probe accepts or returns connection-string text.
- Produces: `HealthEvidence(database: str, migrations: str, ingestion_observed_at: datetime | None, forecast_observed_at: datetime | None, schedule_observed_at: datetime | None, push: str)` and `HealthRepository.collect(now: datetime) -> HealthEvidence`. Migration readiness compares each checked-in numeric SQL filename and SHA-256 checksum to `schema_migrations`, not merely migration counts. A facility-hours artifact with any facility `status` other than `ok` is unhealthy even when top-level `generatedAt` is fresh.

- [ ] **Step 1: Add failing read-only evidence tests using fakes instead of MySQL or XGBoost**

```python
# tests/backend/test_health.py: append
from hashlib import sha256

from server.reclive.health_repository import HealthRepository, healthy_schedule_generated_at


class Cursor:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows
        self.executed: list[str] = []

    def execute(self, sql: str, _params: object = None) -> None:
        self.executed.append(sql)

    def fetchone(self) -> tuple[object, ...] | None:
        return self.rows.pop(0) if self.rows else None

    def fetchall(self) -> list[tuple[object, ...]]:
        return [self.rows.pop(0)] if self.rows else []

    def __enter__(self) -> "Cursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class Connection:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.cursor_instance = Cursor(rows)
        self.closed = False

    def cursor(self) -> Cursor:
        return self.cursor_instance

    def close(self) -> None:
        self.closed = True


def test_repository_collects_only_aggregate_health_evidence(tmp_path) -> None:
    forecast = tmp_path / "forecast.json"
    forecast.write_text('{"generatedAt":"2026-08-31T11:55:00Z"}', encoding="utf-8")
    schedules = tmp_path / "facility_hours.json"
    schedules.write_text('{"generatedAt":"2026-08-31T11:50:00Z","facilities":[{"facilityId":1186,"status":"ok","stale":false},{"facilityId":1656,"status":"ok","stale":false}]}', encoding="utf-8")
    migration = tmp_path / "0001_core_history.sql"
    migration.write_text("SELECT 1;\n", encoding="utf-8")
    digest = sha256(migration.read_bytes()).hexdigest()
    connection = Connection([("0001_core_history.sql", digest), ("2026-08-31 11:58:00",)])
    repository = HealthRepository(
        connect=lambda: connection,
        migration_dir=tmp_path,
        forecast_path=forecast,
        schedule_path=schedules,
        push_status=lambda: "ready",
    )

    evidence = repository.collect(NOW)

    assert evidence.database == "ready"
    assert evidence.migrations == "ready"
    assert evidence.ingestion_observed_at == datetime(2026, 8, 31, 11, 58, tzinfo=timezone.utc)
    assert evidence.forecast_observed_at == HEALTHY_OBSERVED_AT
    assert evidence.schedule_observed_at == datetime(2026, 8, 31, 11, 50, tzinfo=timezone.utc)
    assert evidence.push == "ready"
    assert all("endpoint" not in sql.lower() and "subscription" not in sql.lower() for sql in connection.cursor_instance.executed)
    assert connection.closed is True


def test_repository_reports_unavailable_without_serializing_exception_text(tmp_path) -> None:
    repository = HealthRepository(
        connect=lambda: (_ for _ in ()).throw(RuntimeError("db.example.invalid password=not-for-output")),
        migration_dir=tmp_path,
        forecast_path=tmp_path / "forecast.json",
        schedule_path=tmp_path / "facility_hours.json",
        push_status=lambda: "unavailable",
    )
    evidence = repository.collect(NOW)
    assert (evidence.database, evidence.migrations, evidence.push) == ("unavailable", "unavailable", "unavailable")


def test_repository_treats_a_partially_failed_schedule_artifact_as_unhealthy(tmp_path) -> None:
    schedules = tmp_path / "facility_hours.json"
    schedules.write_text('{"generatedAt":"2026-08-31T11:59:00Z","facilities":[{"facilityId":1186,"status":"ok","stale":false},{"facilityId":1656,"status":"stale","stale":true}]}', encoding="utf-8")
    assert healthy_schedule_generated_at(schedules) is None


def test_repository_rejects_empty_or_incomplete_schedule_artifacts(tmp_path) -> None:
    schedules = tmp_path / "facility_hours.json"
    schedules.write_text('{"generatedAt":"2026-08-31T11:59:00Z","facilities":[]}', encoding="utf-8")
    assert healthy_schedule_generated_at(schedules) is None
    schedules.write_text('{"generatedAt":"2026-08-31T11:59:00Z","facilities":[{"facilityId":1186,"status":"ok","stale":false}]}', encoding="utf-8")
    assert healthy_schedule_generated_at(schedules) is None
```

- [ ] **Step 2: Run the focused tests to observe the absent repository**

Run: `python -m pytest tests/backend/test_health.py::test_repository_collects_only_aggregate_health_evidence tests/backend/test_health.py::test_repository_reports_unavailable_without_serializing_exception_text -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'server.reclive.health_repository'`.

- [ ] **Step 3: Implement evidence collection with bounded, read-only probes**

```python
# server/reclive/health_repository.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable, Protocol


class Cursor(Protocol):
    def execute(self, sql: str, params: object = None) -> None: ...
    def fetchone(self) -> tuple[object, ...] | None: ...
    def fetchall(self) -> list[tuple[object, ...]]: ...
    def __enter__(self) -> "Cursor": ...
    def __exit__(self, *args: object) -> None: ...


class Connection(Protocol):
    def cursor(self) -> Cursor: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class HealthEvidence:
    database: str
    migrations: str
    ingestion_observed_at: datetime | None
    forecast_observed_at: datetime | None
    schedule_observed_at: datetime | None
    push: str


def parse_utc(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def generated_at(path: Path) -> datetime | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return parse_utc(payload.get("generatedAt")) if isinstance(payload, dict) else None


def healthy_schedule_generated_at(path: Path) -> datetime | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    facilities = payload.get("facilities") if isinstance(payload, dict) else None
    if (
        not isinstance(facilities, list)
        or len(facilities) != 2
        or {item.get("facilityId") for item in facilities if isinstance(item, dict)} != {1186, 1656}
        or any(not isinstance(item, dict) or item.get("status") != "ok" or item.get("stale") is not False for item in facilities)
    ):
        return None
    return parse_utc(payload.get("generatedAt"))


class HealthRepository:
    def __init__(
        self,
        *,
        connect: Callable[[], Connection],
        migration_dir: Path,
        forecast_path: Path,
        schedule_path: Path,
        push_status: Callable[[], str],
    ) -> None:
        self.connect = connect
        self.migration_dir = migration_dir
        self.forecast_path = forecast_path
        self.schedule_path = schedule_path
        self.push_status = push_status

    def collect(self, _now: datetime) -> HealthEvidence:
        try:
            connection = self.connect()
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT filename, checksum FROM schema_migrations ORDER BY filename")
                    recorded_migrations = {str(filename): str(checksum) for filename, checksum in cursor.fetchall()}
                    cursor.execute("SELECT completed_at FROM ingestion_runs WHERE status = %s ORDER BY completed_at DESC LIMIT 1", ("succeeded",))
                    latest_ingestion = cursor.fetchone()
            finally:
                connection.close()
            expected_migrations = {
                path.name: sha256(path.read_bytes()).hexdigest()
                for path in self.migration_dir.glob("[0-9][0-9][0-9][0-9]_*.sql")
            }
            migrations = "ready" if expected_migrations == recorded_migrations else "stale"
            ingestion_observed_at = parse_utc(latest_ingestion[0]) if latest_ingestion else None
            database = "ready"
        except Exception:
            database = "unavailable"
            migrations = "unavailable"
            ingestion_observed_at = None
        try:
            push = self.push_status()
        except Exception:
            push = "unavailable"
        return HealthEvidence(
            database=database,
            migrations=migrations,
            ingestion_observed_at=ingestion_observed_at,
            forecast_observed_at=generated_at(self.forecast_path),
            schedule_observed_at=healthy_schedule_generated_at(self.schedule_path),
            push=push if push in {"ready", "unavailable"} else "unavailable",
        )
```

- [ ] **Step 4: Run focused repository tests**

Run: `python -m pytest tests/backend/test_health.py::test_repository_collects_only_aggregate_health_evidence tests/backend/test_health.py::test_repository_reports_unavailable_without_serializing_exception_text -q`

Expected: PASS; both tests pass without a database process, no SQL query contains push subscription fields, and the fake exception string cannot enter evidence.

- [ ] **Step 5: Commit read-only evidence collection**

```bash
git add server/reclive/health_repository.py tests/backend/test_health.py
git commit -m "feat: collect sanitized health evidence"
```

### Task 3: Expose the stable `/health` API response and configuration-backed thresholds

**Files:**
- Modify: `server/reclive/api/health.py`
- Modify: `server/reclive/settings.py`
- Modify: `server/reclive/api/app.py`
- Modify: `tests/backend/test_health.py`

**Interfaces:**
- Consumes: `HealthRepository.collect(now)`, Phase 9 `Settings`, and FastAPI dependency injection.
- Produces: unauthenticated `GET /health` returning HTTP `200` for ready/degraded dependency states, HTTP `503` only when the health service itself cannot produce a sanitized report; its body is exactly `{"status", "checkedAt", "components"}` and component names are `api`, `database`, `migrations`, `ingestion`, `forecast`, `schedules`, and `push`. `api: ready` is a tested statement that routing and report serialization completed.

- [ ] **Step 1: Write the failing FastAPI response and config-validation tests**

```python
# tests/backend/test_health.py: append
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.reclive.api.health import create_health_router
from server.reclive.health_repository import HealthEvidence


class StaticHealthRepository:
    def __init__(self, evidence: HealthEvidence) -> None:
        self.evidence = evidence

    def collect(self, _now: datetime) -> HealthEvidence:
        return self.evidence


def test_health_route_returns_sanitized_degraded_data_quality_signals() -> None:
    app = FastAPI()
    app.include_router(create_health_router(
        repository=StaticHealthRepository(HealthEvidence(
            database="ready",
            migrations="stale",
            ingestion_observed_at=STALE_OBSERVED_AT,
            forecast_observed_at=HEALTHY_OBSERVED_AT,
            schedule_observed_at=None,
            push="unavailable",
        )),
        now=lambda: NOW,
        ingestion_stale_after_seconds=600,
        forecast_stale_after_seconds=21_600,
        schedule_stale_after_seconds=21_600,
    ))
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "checkedAt": "2026-08-31T12:00:00Z",
        "components": {
            "api": {"status": "ready", "observedAt": None, "ageSeconds": None, "detail": None},
            "database": {"status": "ready", "observedAt": None, "ageSeconds": None, "detail": None},
            "migrations": {"status": "stale", "observedAt": None, "ageSeconds": None, "detail": None},
            "ingestion": {"status": "stale", "observedAt": "2026-08-31T11:00:00Z", "ageSeconds": 3600, "detail": None},
            "forecast": {"status": "ready", "observedAt": "2026-08-31T11:55:00Z", "ageSeconds": 300, "detail": None},
            "schedules": {"status": "missing", "observedAt": None, "ageSeconds": None, "detail": None},
            "push": {"status": "unavailable", "observedAt": None, "ageSeconds": None, "detail": None},
        },
    }
    body = response.text.lower()
    for forbidden in ("password", "token", "vapid", "endpoint", "host", "live_counts_url"):
        assert forbidden not in body


def test_health_route_returns_sanitized_503_when_report_collection_itself_fails() -> None:
    class RaisingRepository:
        def collect(self, _now: datetime) -> HealthEvidence:
            raise RuntimeError("db.example.invalid password=must-not-appear")

    app = FastAPI()
    app.include_router(create_health_router(
        repository=RaisingRepository(),
        now=lambda: NOW,
        ingestion_stale_after_seconds=600,
        forecast_stale_after_seconds=21_600,
        schedule_stale_after_seconds=21_600,
    ))
    response = TestClient(app).get("/health")
    assert response.status_code == 503
    assert response.json() == {"detail": "health_unavailable"}
```

- [ ] **Step 2: Run the response contract to observe the missing health router**

Run: `python -m pytest tests/backend/test_health.py::test_health_route_returns_sanitized_degraded_data_quality_signals -q`

Expected: FAIL because Phase 9's transitional `server.reclive.api.health` router has no `create_health_router` factory and still emits the old top-level schedule evidence.

- [ ] **Step 3: Extend settings with positive health thresholds only**

```python
# server/reclive/settings.py: add to Settings and its validated constructor
ingestion_stale_after_seconds: int = 600
forecast_stale_after_seconds: int = 21_600
schedule_stale_after_seconds: int = 21_600
```

```python
# server/reclive/settings.py: add to the existing validation loop
for name, value in {
    "INGESTION_STALE_AFTER_SECONDS": settings.ingestion_stale_after_seconds,
    "FORECAST_STALE_AFTER_SECONDS": settings.forecast_stale_after_seconds,
    "SCHEDULE_STALE_AFTER_SECONDS": settings.schedule_stale_after_seconds,
}.items():
    if value <= 0:
        raise RuntimeError(f"Invalid configuration: {name} must be a positive integer")
```

The existing Phase 9 `Settings` parser must convert these values using the same integer parser it uses for ports, reading `SCHEDULE_STALE_AFTER_SECONDS` first and `SCHEDULE_MAX_AGE_SECONDS` only when the active name is absent. Its exception must retain only the variable name and reason, never the input value. Add a settings test covering canonical-name precedence, legacy fallback, and nonpositive rejection.

- [ ] **Step 4: Implement the health router and compose it through the compatibility API entry point**

```python
# server/reclive/api/health.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Protocol

from fastapi import APIRouter, HTTPException

from server.reclive.health import ComponentHealth, HealthReport, classify_age, overall_status, utc_iso
from server.reclive.health_repository import HealthEvidence


class EvidenceRepository(Protocol):
    def collect(self, now: datetime) -> HealthEvidence: ...


def fixed_component(status: str) -> ComponentHealth:
    safe_status = status if status in {"ready", "stale", "missing", "unavailable"} else "unavailable"
    return ComponentHealth(status=safe_status, observedAt=None, ageSeconds=None)


def create_health_router(
    *,
    repository: EvidenceRepository,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ingestion_stale_after_seconds: int,
    forecast_stale_after_seconds: int,
    schedule_stale_after_seconds: int,
) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def get_health_report() -> dict[str, object]:
        checked_at = now()
        try:
            evidence = repository.collect(checked_at)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="health_unavailable") from exc
        components = {
            "api": fixed_component("ready"),
            "database": fixed_component(evidence.database),
            "migrations": fixed_component(evidence.migrations),
            "ingestion": classify_age(evidence.ingestion_observed_at, checked_at, ingestion_stale_after_seconds),
            "forecast": classify_age(evidence.forecast_observed_at, checked_at, forecast_stale_after_seconds),
            "schedules": classify_age(evidence.schedule_observed_at, checked_at, schedule_stale_after_seconds),
            "push": fixed_component(evidence.push),
        }
        report = HealthReport(
            status=overall_status({name: component.status for name, component in components.items()}),
            checkedAt=utc_iso(checked_at),
            components=components,
        )
        return report.to_dict()

    return router
```

```python
# server/reclive/api/app.py: replace the Phase 9 transitional health-router import/composition
from pathlib import Path

from server.reclive.api.health import create_health_router
from server.reclive.api.push import push_db_available, push_vapid_configured
from server.reclive.db import open_db_connection
from server.reclive.health_repository import HealthRepository

# inside create_app(), after effective_settings and app are initialized
app.include_router(create_health_router(
    repository=HealthRepository(
        connect=lambda: open_db_connection(effective_settings.database),
        migration_dir=Path(__file__).resolve().parents[2] / "migrations",
        forecast_path=Path(effective_settings.forecast_json_path),
        schedule_path=Path(effective_settings.facility_hours_json_path),
        push_status=lambda: "ready" if push_db_available() and push_vapid_configured() else "unavailable",
    ),
    ingestion_stale_after_seconds=effective_settings.ingestion_stale_after_seconds,
    forecast_stale_after_seconds=effective_settings.forecast_stale_after_seconds,
    schedule_stale_after_seconds=effective_settings.schedule_stale_after_seconds,
))
```

Use the actual Phase 9 `settings`, `open_db_connection`, push-readiness provider, and `Path(__file__).resolve().parents[2] / "migrations"` location inside `create_app`. Remove `from server.reclive.api.health import router as health_router` and include exactly one router returned by `create_health_router`; do not append a second route after the module-level `app` has already been created. Leave `server/forecast_api.py` as the Phase 9 thin delegator. The final route body must omit Phase 7's transitional top-level `schedule` key and expose that same sanitized artifact state only as `components.schedules`; add a route assertion for both conditions.

- [ ] **Step 5: Run focused route/config tests and the backend suite**

Run: `python -m pytest tests/backend/test_health.py -q && ruff check server tests && python -m pytest -q`

Expected: PASS; the route returns a stable sanitized degraded response, pure classification remains green, no XGBoost training starts, and the complete backend suite passes.

- [ ] **Step 6: Commit the operational API contract**

```bash
git add server/reclive/api/health.py server/reclive/settings.py server/reclive/health_repository.py server/reclive/api/app.py tests/backend/test_health.py
git commit -m "feat: expose sanitized operational health"
```

### Task 4: Make operational logs structured and safe at compatibility entry points

**Files:**
- Create: `server/reclive/observability.py`
- Modify: `server/gym_fetch.py`
- Modify: `server/forecast_job.py`
- Modify: `server/facility_hours_fetch.py`
- Modify: `server/forecast_api.py`
- Create: `tests/backend/test_observability.py`

**Interfaces:**
- Consumes: event names and scalar metrics known at the call site.
- Produces: one JSON object per log line with `event`, `timestamp`, and allowlisted scalar fields. Valid field values are `str`, `int`, `float`, `bool`, and `None`; strings longer than 240 characters and every unknown field are rejected before any output. Error fields use a fixed category such as `network_error`, `validation_error`, `database_unavailable`, `file_unavailable`, or `push_unavailable`.

- [ ] **Step 1: Write failing structured-log privacy tests**

```python
# tests/backend/test_observability.py
import json

import pytest

from server.reclive.observability import log_event


def test_log_event_emits_only_allowlisted_scalar_fields(capsys) -> None:
    log_event("ingestion.completed", receivedCount=8, historyInsertedCount=7, unchangedCount=1)
    payload = json.loads(capsys.readouterr().out)
    assert payload["event"] == "ingestion.completed"
    assert payload["receivedCount"] == 8
    assert payload["historyInsertedCount"] == 7
    assert payload["unchangedCount"] == 1
    assert set(payload) == {"event", "timestamp", "receivedCount", "historyInsertedCount", "unchangedCount"}


def test_log_event_rejects_sensitive_or_structured_fields(capsys) -> None:
    with pytest.raises(ValueError, match="invalid operational event fields"):
        log_event("ingestion.failed", endpoint="https://not-permitted.example", password="not-permitted", requestBody={"key": "not-permitted"}, errorCategory="network_error")
    assert capsys.readouterr().out == ""


def test_log_event_rejects_unknown_event_names(capsys) -> None:
    with pytest.raises(ValueError, match="unknown operational event"):
        log_event("dbHost.changed", dbHost="not-allowed")
    assert capsys.readouterr().out == ""
```

- [ ] **Step 2: Run the focused tests to observe the missing logger**

Run: `python -m pytest tests/backend/test_observability.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'server.reclive.observability'`.

- [ ] **Step 3: Implement the allowlisted JSON-line logger**

```python
# server/reclive/observability.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

EVENT_FIELDS = {
    "ingestion.completed": {"receivedCount", "historyInsertedCount", "unchangedCount"},
    "ingestion.failed": {"errorCategory"},
    "forecast.completed": {"generatedFacilities"},
    "schedules.completed": {"facilityCount", "failedFacilityCount"},
    "push.evaluator_failed": {"errorCategory"},
}
ERROR_CATEGORIES = {"database_unavailable", "file_unavailable", "network_error", "push_unavailable", "validation_error"}


def allowed_value(name: str, value: Any) -> bool:
    if name == "errorCategory":
        return value is None or value in ERROR_CATEGORIES
    return isinstance(value, (int, float, bool, type(None))) or (isinstance(value, str) and len(value) <= 240)


def log_event(event: str, **fields: Any) -> None:
    if event not in EVENT_FIELDS:
        raise ValueError("unknown operational event")
    if set(fields) - EVENT_FIELDS[event] or any(not allowed_value(name, value) for name, value in fields.items()):
        raise ValueError("invalid operational event fields")
    payload: dict[str, Any] = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    for name, value in fields.items():
        payload[name] = value
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True), flush=True)
```

- [ ] **Step 4: Replace compatibility-entry-point diagnostic output with sanitized lifecycle events**

```python
# server/gym_fetch.py: at the successful end of main()
log_event("ingestion.completed", receivedCount=len(live), historyInsertedCount=inserted, unchangedCount=skipped)
```

```python
# server/gym_fetch.py: in the exception path of main()
log_event("ingestion.failed", errorCategory="database_unavailable" if isinstance(error, pymysql.MySQLError) else "network_error")
```

```python
# server/forecast_job.py: after atomic forecast publication
log_event("forecast.completed", generatedFacilities=len(forecast_payload["facilities"]))
```

```python
# server/facility_hours_fetch.py: after the atomic output publication
log_event("schedules.completed", facilityCount=len(facility_payloads), failedFacilityCount=sum(item["status"] != "ok" for item in facility_payloads))
```

```python
# server/forecast_api.py: evaluator exception branch
log_event("push.evaluator_failed", errorCategory="push_unavailable")
```

Import `log_event` from `server.reclive.observability` in each compatibility entry point. Preserve exit codes and public HTTP responses; remove `print(error)`, `print(traceback.format_exc())`, and any statement that interpolates an exception object or raw schedule/live payload.

- [ ] **Step 5: Run log tests and a narrow static scan**

Run: `python -m pytest tests/backend/test_observability.py -q && ! rg -n "print\(.*(error|exc|traceback|LIVE_COUNTS_URL|GYM_DB_PASSWORD|endpoint|subscription)" server/gym_fetch.py server/forecast_job.py server/facility_hours_fetch.py server/forecast_api.py`

Expected: PASS; two logger tests pass and `rg` returns no matches.

- [ ] **Step 6: Commit observability hardening**

```bash
git add server/reclive/observability.py server/gym_fetch.py server/forecast_job.py server/facility_hours_fetch.py server/forecast_api.py tests/backend/test_observability.py
git commit -m "feat: add sanitized operational events"
```

### Task 5: Document safe configuration and complete local/deployment operations

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `tests/backend/test_operations_docs.py`

**Interfaces:**
- Consumes: Phase 6 public/private configuration split, Phase 1 migration command, Phase 7 facility-hours command, Phase 8 PWA requirements, Phase 10 `/health` response, and existing compatibility entry points.
- Produces: a placeholder-only `.env.example` and README sections that name every required command and distinguish variables that can enter a browser bundle from backend-only values.

- [ ] **Step 1: Write the failing documentation/configuration contract test**

```python
# tests/backend/test_operations_docs.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_readme_documents_safe_operations_without_secret_material() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        "python server/migrate.py",
        "python server/gym_fetch.py",
        "python server/facility_hours_fetch.py",
        "python server/forecast_job.py",
        "uvicorn server.forecast_api:app",
        "curl -fsS http://127.0.0.1:8000/health",
        "npm run test:run",
        "npm run test:e2e",
        "npm audit --omit=dev",
        "re-clone",
    ):
        assert required in readme
    assert "VITE_API_BASE_URL" in readme
    assert "LIVE_COUNTS_URL" in readme
    assert "backend-only" in readme.lower()
    assert "public frontend" in readme.lower()


def test_env_example_uses_only_safe_placeholders_and_has_no_legacy_public_feed_name() -> None:
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    values = {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in env_example.splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }
    preserved = {
        "GYM_DB_HOST": "localhost", "GYM_DB_PORT": "3306", "GYM_DB_USER": "root", "GYM_DB_NAME": "gym_data", "GYM_DB_TIMEZONE": "America/Chicago",
        "FORECAST_DAY_START_HOUR": "6", "FORECAST_DAY_END_HOUR": "23", "GYM_RESAMPLE_MINUTES": "15", "GYM_WINDOW_RESAMPLE_MINUTES": "30",
        "GYM_WEATHER_URL": "https://api.open-meteo.com/v1/forecast", "GYM_WEATHER_ARCHIVE_URL": "https://archive-api.open-meteo.com/v1/archive",
        "GYM_WEATHER_LAT": "43.0731", "GYM_WEATHER_LON": "-89.4012", "GYM_WEATHER_FORECAST_DAYS": "7", "GYM_WEATHER_HISTORY_MAX_DAYS": "180",
        "MODEL_ARTIFACT_DIR": "model_artifacts", "MODEL_BASENAME": "forecast_model", "SCHEDULE_FILTER_ENABLED": "1",
        "FACILITY_HOURS_JSON_PATH": "facility_hours.json", "FORECAST_OUTPUT_INCLUDE_WEATHER": "1", "FORECAST_OUTPUT_INCLUDE_INTERVAL_FIELDS": "1",
        "FACILITY_CAPACITIES_JSON_PATH": "shared/facility_capacities.json", "FORECAST_JSON_PATH": "forecast.json", "FACILITY_SECTION_CONFIG_PATH": "facility_sections.json",
        "FORECAST_API_HOST": "0.0.0.0", "FORECAST_API_PORT": "8000", "ACTUAL_HOUR_MIN_COVERAGE": "0.75",
        "PUSH_RULES_TABLE": "push_rules", "PUSH_EVALUATOR_ENABLED": "true", "PUSH_EVALUATOR_INTERVAL_SECONDS": "180",
        "PUSH_EVALUATOR_DB_LOCK_NAME": "reclive_push_eval", "PUSH_DEFAULT_NOTIFICATION_URL": "/", "PUSH_VAPID_SUBJECT": "mailto:alerts@example.com",
        "PUSH_ADMIN_ROUTES_ENABLED": "false", "PUSH_RULE_DEFAULT_TTL_SECONDS": "86400", "PUSH_RULE_MAX_TTL_SECONDS": "604800",
        "PUSH_MAX_ACTIVE_RULES_PER_ENDPOINT": "10", "PUSH_WRITE_RATE_LIMIT": "20", "PUSH_WRITE_RATE_WINDOW_SECONDS": "600",
    }
    assert {name: values.get(name) for name in preserved} == preserved
    assert "VITE_LIVE_COUNTS_URL" not in env_example
    assert "VITE_FORECAST_API_BASE_URL" not in env_example
    assert "VITE_PUSH_API_BASE_URL" not in env_example
    assert "VITE_API_BASE_URL=change_me" in env_example
    assert "LIVE_COUNTS_URL=https://" not in env_example
    assert "LIVE_COUNTS_URL=change_me" in env_example
    for name in ("LIVE_COUNTS_URL", "GYM_DB_PASSWORD", "PUSH_VAPID_PUBLIC_KEY", "PUSH_VAPID_PRIVATE_KEY", "PUSH_ENDPOINT_HASH_KEY", "PUSH_ADMIN_TOKEN"):
        assert values[name] == "change_me"
```

- [ ] **Step 2: Run the focused documentation test to observe current gaps**

Run: `python -m pytest tests/backend/test_operations_docs.py -q`

Expected: FAIL because the current README lacks setup/deployment/runbook content and `.env.example` still exposes legacy public feed configuration names.

- [ ] **Step 3: Replace `.env.example` with grouped placeholder-only configuration**

```dotenv
# Public frontend variables. These values are embedded in the browser build.
VITE_API_BASE_URL=change_me
VITE_SITE_URL=change_me

# Runtime mode
APP_ENV=development

# Private backend integration. Never prefix these names with VITE_.
LIVE_COUNTS_URL=change_me
GYM_DB_HOST=localhost
GYM_DB_PORT=3306
GYM_DB_USER=root
GYM_DB_PASSWORD=change_me
GYM_DB_NAME=gym_data
GYM_DB_TIMEZONE=America/Chicago

# API and CORS. Production origins must be explicit, comma-separated HTTPS origins.
FORECAST_API_HOST=0.0.0.0
FORECAST_API_PORT=8000
FORECAST_API_ALLOW_ORIGINS=http://127.0.0.1:4173
FORECAST_JSON_PATH=forecast.json
FACILITY_HOURS_JSON_PATH=facility_hours.json
FACILITY_SECTION_CONFIG_PATH=facility_sections.json

# Health freshness thresholds, in seconds.
INGESTION_STALE_AFTER_SECONDS=600
FORECAST_STALE_AFTER_SECONDS=21600
SCHEDULE_STALE_AFTER_SECONDS=21600

# Forecast and schedule jobs
FORECAST_DAY_START_HOUR=6
FORECAST_DAY_END_HOUR=23
GYM_RESAMPLE_MINUTES=15
GYM_WINDOW_RESAMPLE_MINUTES=30
GYM_WEATHER_URL=https://api.open-meteo.com/v1/forecast
GYM_WEATHER_ARCHIVE_URL=https://archive-api.open-meteo.com/v1/archive
GYM_WEATHER_LAT=43.0731
GYM_WEATHER_LON=-89.4012
GYM_WEATHER_FORECAST_DAYS=7
GYM_WEATHER_HISTORY_MAX_DAYS=180
MODEL_ARTIFACT_DIR=model_artifacts
MODEL_BASENAME=forecast_model
FACILITY_CAPACITIES_JSON_PATH=shared/facility_capacities.json
SCHEDULE_FILTER_ENABLED=1
FORECAST_OUTPUT_INCLUDE_WEATHER=1
FORECAST_OUTPUT_INCLUDE_INTERVAL_FIELDS=1
ACTUAL_HOUR_MIN_COVERAGE=0.75

# Push notifications. Keep private keys, endpoint-hash keys, and admin tokens outside source control.
PUSH_VAPID_PUBLIC_KEY=change_me
PUSH_VAPID_PRIVATE_KEY=change_me
PUSH_VAPID_SUBJECT=mailto:alerts@example.com
PUSH_ENDPOINT_HASH_KEY=change_me
PUSH_ADMIN_TOKEN=change_me
PUSH_ADMIN_ROUTES_ENABLED=false
PUSH_EVALUATOR_ENABLED=true
PUSH_RULES_TABLE=push_rules
PUSH_EVALUATOR_INTERVAL_SECONDS=180
PUSH_EVALUATOR_DB_LOCK_NAME=reclive_push_eval
PUSH_DEFAULT_NOTIFICATION_URL=/
PUSH_RULE_DEFAULT_TTL_SECONDS=86400
PUSH_RULE_MAX_TTL_SECONDS=604800
PUSH_MAX_ACTIVE_RULES_PER_ENDPOINT=10
PUSH_WRITE_RATE_LIMIT=20
PUSH_WRITE_RATE_WINDOW_SECONDS=600
```

- [ ] **Step 4: Add the complete README operations reference**

```markdown
<!-- README.md: replace the product-only document with this operational structure after the existing introduction -->
## Architecture

The browser is a React/Vite PWA. The FastAPI service reads current snapshots, forecasts, schedules, and push state from backend-only configuration. MySQL retains schema migrations, current occupancy snapshots, history, ingestion runs, alert rules, and rate-limit counters. Browser code receives only `VITE_API_BASE_URL` and `VITE_SITE_URL`; `LIVE_COUNTS_URL`, database settings, admin tokens, and private push material stay backend-only.

## Local setup

```bash
npm ci
python -m pip install -r server/requirements.txt
python -m pip install -r server/requirements-dev.txt
cp .env.example .env
```

Set only local, non-production values in `.env`. In production, provide private environment variables through the host's secret manager or process manager; do not upload `.env` to the frontend host.

## MySQL and migrations

Create a least-privilege MySQL account that can read/write RecLive tables but cannot administer other schemas. Apply migrations before starting an application process that requires the new schema:

```bash
python server/migrate.py
```

Never edit an applied file under `server/migrations/`; the runner records each filename and SHA-256 checksum in `schema_migrations` and stops on drift.

## Runtime commands

```bash
python server/gym_fetch.py
python server/facility_hours_fetch.py
python server/forecast_job.py
uvicorn server.forecast_api:app --host 127.0.0.1 --port 8000
curl -fsS http://127.0.0.1:8000/health
```

Recommended cadences are live ingestion at least every 90 seconds, forecast generation every 15 minutes, facility-hours ingestion every 4 hours, and push evaluation at its configured interval. Run only one scheduler for each command; the database-backed evaluator lock protects cross-process alert dispatch, not duplicate cron entries for ingestion or forecast generation.

## Deployment

Build the frontend with only public frontend variables present:

```bash
npm ci
npm run build
```

Serve `dist/` from the frontend host with SPA fallback for `/nick` and `/bakke`. Deploy the backend with private variables injected by the backend host, run `python server/migrate.py` once before starting new backend processes, then verify the local backend endpoint and the externally routed endpoint return a sanitized `/health` response. Do not deploy `.env`, `server/forecast.json`, model artifacts, database dumps, push subscriptions, or private backup bundles to the frontend host.

## PWA and push

The PWA requires HTTPS in production. The browser build may contain the public VAPID key only when push is enabled; the VAPID private key, endpoint-hash key, admin token, and subscription endpoints remain backend-only. `/health` reports push readiness without exposing any push material.

## Testing and security checks

```bash
npm run lint
npm run build
npm run test:run
npm run test:e2e
ruff check server tests
python -m pytest -q
npm audit --omit=dev
git diff --check
gitleaks git --redact --log-opts="--all"
```

`gitleaks git --redact --log-opts="--all"` scans all reachable history while redacting suspected secret values from output. GitHub Actions also runs the same full-history gate and dependency review. A failed external provider check is reported as unexecuted; it is never inferred from a mock.

## Re-cloning after the history rewrite

The repository history was rewritten before this implementation branch. Collaborators must re-clone the repository or carefully rebase a clean local branch onto the rewritten remote history. Rewriting published references does not erase copies retained in forks, caches, old clones, or backup bundles.
```

- [ ] **Step 5: Run docs/config tests and inspect for non-placeholder private values**

Run: `python -m pytest tests/backend/test_operations_docs.py -q && ! rg -n "^(VITE_LIVE_COUNTS_URL|VITE_FORECAST_API_BASE_URL|VITE_PUSH_API_BASE_URL)=" .env.example`

Expected: PASS; both tests pass and `rg` returns no matches.

- [ ] **Step 6: Commit operational configuration documentation**

```bash
git add .env.example README.md tests/backend/test_operations_docs.py
git commit -m "docs: add secure configuration and deployment guide"
```

### Task 6: Publish SECURITY policy and the operator runbook

**Files:**
- Create: `SECURITY.md`
- Create: `docs/operations/runbook.md`
- Modify: `tests/backend/test_operations_docs.py`

**Interfaces:**
- Consumes: the README commands, `/health` response schema, security workflow, and the project's no-secret constraints.
- Produces: security reporting guidance plus an operator procedure that has preflight, deployment, health interpretation, incident, and routine-maintenance sections with only safe commands and evidence requirements.

- [ ] **Step 1: Extend documentation tests with failing policy and runbook contracts**

```python
# tests/backend/test_operations_docs.py: append
def test_security_policy_and_runbook_cover_required_controls() -> None:
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    runbook = (ROOT / "docs/operations/runbook.md").read_text(encoding="utf-8")
    for token in ("private", ".env", "credential rotation", "history rewrite", "Gitleaks", "do not include secrets"):
        assert token.lower() in security.lower()
    for token in ("Preflight", "Deploy", "Health", "Incident", "Routine maintenance", "schema_migrations", "curl -fsS", "re-clone"):
        assert token.lower() in runbook.lower()
    assert "curl -fsS http://127.0.0.1:8000/health" in runbook
    assert "git reset --hard" not in runbook
    assert "printenv" not in runbook
```

- [ ] **Step 2: Run the focused test to observe missing policy/runbook files**

Run: `python -m pytest tests/backend/test_operations_docs.py::test_security_policy_and_runbook_cover_required_controls -q`

Expected: FAIL with `FileNotFoundError` for `SECURITY.md`.

- [ ] **Step 3: Create the security policy**

```markdown
# Security Policy

## Reporting a vulnerability

Report suspected vulnerabilities privately to the repository maintainers. Include the affected route or component, reproducible steps, impact, and a safe redacted proof. Do not open a public issue containing credentials, access tokens, push subscriptions, database data, private provider URLs, or exploit payloads.

## Secret handling

Use ignored `.env` files or the deployment platform's secret manager for private settings. Never commit credentials, database dumps, VAPID private keys, endpoint-hash keys, admin tokens, provider URLs containing credentials, browser push endpoints, test recordings with private data, or the private history-rewrite backup bundle. `.env.example` contains names and non-working placeholders only.

## Credential rotation

If a credential may have been exposed, rotate or revoke it through the owning provider or secret manager, update the deployment secret out of band, and verify the application without printing the new value. Do not paste an old or new credential into commits, issues, chat transcripts, test fixtures, logs, shell commands, or documentation.

## History rewrite limitations

Published history was rewritten to remove the historical values from reachable repository branches and tags. That process does not erase values retained in forks, caches, old clones, backup bundles, or third-party indexes. Collaborators must re-clone or carefully rebase onto the rewritten history and must not push old references back to the remote.

## Security verification

GitHub Actions runs full-history Gitleaks and dependency review. Before a release, run the documented test suite, `npm audit --omit=dev`, and `git diff --check`; investigate scanner findings without copying suspected secret material into tickets or logs.
```

- [ ] **Step 4: Create the operator runbook**

````markdown
<!-- docs/operations/runbook.md -->
# RecLive Operator Runbook

## Preflight

1. Confirm the working tree contains only intended changes:

   ```bash
   git status --short
   git diff --check
   ```

2. Run the documented frontend and backend checks:

   ```bash
   npm run lint
   npm run build
   npm run test:run
   npm run test:e2e
   ruff check server tests
   python -m pytest -q
   npm audit --omit=dev
   ```

3. Confirm the deployment secret manager provides the required private settings. Do not print values, run `printenv`, or copy `.env` to a frontend host.

## Deploy

1. Build and publish the frontend artifact using only `VITE_API_BASE_URL` and `VITE_SITE_URL`.
2. Inject backend-only settings through the backend host.
3. Run migrations once, before starting application processes that require their tables:

   ```bash
   python server/migrate.py
   ```

4. Start or restart the backend process using the host's process manager. Keep the compatibility command available for local diagnosis:

   ```bash
   uvicorn server.forecast_api:app --host 127.0.0.1 --port 8000
   ```

5. Query the health endpoint without headers that contain credentials:

   ```bash
   curl -fsS http://127.0.0.1:8000/health
   ```

6. Verify the deployed `/nick` and `/bakke` routes and a fresh rendered dashboard. Record the deployed commit SHA, migration result, health JSON status, and HTTP status; do not record secrets or response bodies from private upstreams.

## Health interpretation

- `status: ready` means database, migrations, ingestion, forecast, schedules, and push are all ready under configured thresholds.
- `status: degraded` is an operational signal, not proof that a user-facing route is down. Inspect the named component's `status`, `observedAt`, and `ageSeconds`.
- `ingestion: stale` means the most recent successful ingestion is older than `INGESTION_STALE_AFTER_SECONDS`; do not describe live occupancy as fresh.
- `forecast: stale` or `schedules: stale` means their generated artifact is older than its configured threshold; preserve available data but investigate the corresponding scheduled job.
- `migrations: stale` means the recorded `schema_migrations` count does not match checked-in migration files. Stop application rollout and reconcile through the migration runner; do not edit schema_migrations manually.
- `database: unavailable` or `push: unavailable` is intentionally category-only. Check host-side service logs and secret-manager bindings without copying connection strings or push material into incident notes.

## Incident response

1. Preserve the error category, timestamp, affected component, route, and deployed SHA.
2. For a stale job, run the corresponding safe local command through the approved scheduler or process manager and re-check `/health`:

   ```bash
   python server/gym_fetch.py
   python server/facility_hours_fetch.py
   python server/forecast_job.py
   ```

3. For migration mismatch, stop rollout, compare migration filenames and checksums through `python server/migrate.py`, and restore service only after the runner reports success. Do not alter migration files that have been applied.
4. For suspected secret exposure, follow `SECURITY.md`: contain access through the owning secret manager/provider, rotate out of band, avoid printing values, and report privately.
5. If release rollback is required, return the application artifact to the previously verified release through the deployment platform. Do not use destructive Git commands against a shared checkout and do not roll database schema backwards unless an explicitly tested rollback migration exists.

## Routine maintenance

- Live ingestion cadence: at least every 90 seconds.
- Forecast generation cadence: every 15 minutes.
- Facility-hours cadence: every 4 hours.
- Review `/health` after scheduler changes and after any migration.
- Review `schema_migrations` via the runner, GitHub Actions results, Gitleaks, dependency review, and Dependabot updates weekly.
- Retain the history-rewrite backup bundle privately and untracked. New collaborators must re-clone after the rewrite; old clones must not republish original references.
````

- [ ] **Step 5: Run all documentation tests and safety scans**

Run: `python -m pytest tests/backend/test_operations_docs.py -q && ! rg -n "AccountAPIKey=|YOUR_ACCOUNT_API_KEY|replace_with_" README.md SECURITY.md docs/operations/runbook.md .env.example`

Expected: PASS; documentation tests pass and `rg` returns no matches.

- [ ] **Step 6: Commit operator policy and runbook**

```bash
git add SECURITY.md docs/operations/runbook.md tests/backend/test_operations_docs.py
git commit -m "docs: add security policy and operator runbook"
```

### Task 7: Verify the Phase 10 release gate without deployment

**Files:**
- Modify: `README.md`
- Modify: `docs/operations/runbook.md`
- Modify: `tests/backend/test_operations_docs.py`

**Interfaces:**
- Consumes: Phase 1 CI/security scripts and Tasks 1–6 health/documentation contracts.
- Produces: one final, ordered verification checklist that distinguishes commands actually run from provider-dependent checks that remain unexecuted.

- [ ] **Step 1: Add a failing test for the truthful verification boundary**

```python
# tests/backend/test_operations_docs.py: append
def test_release_docs_require_actual_evidence_and_prohibit_automatic_deploy() -> None:
    combined = "\n".join(
        (ROOT / name).read_text(encoding="utf-8")
        for name in ("README.md", "docs/operations/runbook.md")
    ).lower()
    for required in ("actual result", "unexecuted", "do not merge", "do not deploy", "gitleaks", "git diff --check"):
        assert required in combined
```

- [ ] **Step 2: Run the test to observe the absent explicit evidence policy**

Run: `python -m pytest tests/backend/test_operations_docs.py::test_release_docs_require_actual_evidence_and_prohibit_automatic_deploy -q`

Expected: FAIL because the release documents do not yet state every evidence and no-deployment condition.

- [ ] **Step 3: Append the exact evidence requirement to README and runbook**

```markdown
<!-- README.md: append -->
## Release evidence

This repository does not merge or deploy automatically. Before proposing a release, record the actual result of every command that was run, including `git diff --check`, frontend build/tests, backend tests, migration tests on clean and already-migrated MySQL 8.4, Gitleaks, and `npm audit --omit=dev`. Mark an unavailable credential-backed or provider-backed verification as **unexecuted** with its reason; never infer success from a mock, upload, workflow configuration, or deployment command alone.
```

```markdown
<!-- docs/operations/runbook.md: append -->
## Evidence record

Do not merge or deploy automatically. For every release candidate, retain the source commit SHA, migration output, exact commands, actual result, and fresh health/route checks. List every unavailable external verification as unexecuted and explain why it could not run. `gitleaks git --redact --log-opts="--all"`, migration tests on clean and already-migrated MySQL 8.4, and `git diff --check` are release gates; do not paste scanner match text into the evidence record.
```

- [ ] **Step 4: Run the complete Phase 10 verification set**

Run: `npm ci && npm run lint && npm run build && npm run test:run && npm run test:e2e && python -m pip install -r server/requirements.txt && python -m pip install -r server/requirements-dev.txt && ruff check server tests && python -m pytest -v && npm audit --omit=dev && git diff --check`

Run: `brew install mysql@8.4 gitleaks && brew services start mysql@8.4 && RECLIVE_MYSQL_BIN="$(brew --prefix mysql@8.4)/bin" && "$RECLIVE_MYSQL_BIN/mysqladmin" ping && "$RECLIVE_MYSQL_BIN/mysql" -uroot -e "CREATE USER IF NOT EXISTS 'reclive'@'127.0.0.1' IDENTIFIED BY 'reclive-ci-password'; GRANT ALL PRIVILEGES ON reclive_test.* TO 'reclive'@'127.0.0.1'; FLUSH PRIVILEGES;" && TEST_MYSQL_HOST=127.0.0.1 TEST_MYSQL_PORT=3306 TEST_MYSQL_USER=reclive TEST_MYSQL_PASSWORD=reclive-ci-password TEST_MYSQL_DATABASE=reclive_test TEST_MYSQL_ADMIN_USER=root TEST_MYSQL_ADMIN_PASSWORD= python -m pytest tests/backend/test_migrate.py -v && gitleaks git --redact --log-opts="--all"`

Expected: PASS; all local checks pass, the migration suite exercises a clean database and its unchanged second run, and Gitleaks scans every reachable ref without printing secret values. The Homebrew install/start command is required when MySQL 8.4 or Gitleaks is not already available; only a failed install/start may be recorded as unexecuted with its exact output.

- [ ] **Step 5: Commit the truthful release gate**

```bash
git add README.md docs/operations/runbook.md tests/backend/test_operations_docs.py
git commit -m "docs: require operational release evidence"
```

## Phase 10 Completion Check

- [ ] `GET /health` returns only `status`, `checkedAt`, and the seven documented component states, never configuration values or exception details.
- [ ] Database, migration, ingestion, forecast, schedule, and push signals distinguish `ready`, `stale`, `missing`, and `unavailable` honestly; a degraded response remains parseable and safe.
- [ ] Compatibility entry points emit allowlisted JSON events and no longer interpolate exception text, provider URLs, or secret-bearing payloads.
- [ ] `.env.example`, README, SECURITY policy, and runbook contain only non-working placeholders and never show real credentials or private URLs.
- [ ] README gives exact local, migration, ingestion, schedule, forecast, API, testing, PWA/push, deployment, and re-clone commands; the runbook requires a fresh health and rendered-route check after a deployment.
- [ ] Final report lists exact commands and actual results, along with every unexecuted external verification and its reason. No merge or deployment occurs.
