# RecLive Data Trust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Make RecLive current and historical occupancy truthful with validated ingestion, heartbeat-backed snapshots, a shared frontend summary, and qualified time-weighted actual hours.

**Architecture:** Keep `server/gym_fetch.py` as the executable poll entry point and delegate validation, persistence, and actual-hour calculations into small `server/reclive/` modules. Read current values only from `location_snapshot`, retain `location_history` as state changes, and use successful `ingestion_runs` heartbeats to bound historical observation. Parse `FetchedAt` into each frontend `Location`, then derive every live display from a single `OccupancySummary`.

**Tech Stack:** Python 3, FastAPI, PyMySQL, MySQL 8.4, pytest; React 19, TypeScript 5.9, Vitest, React Testing Library.

**Spec:** `docs/superpowers/specs/2026-08-31-reclive-security-data-trust-design.md`

## Global Constraints

- Implement Phases 2-4 only. Do not merge, deploy, contact providers, change current production credentials, or run full XGBoost training.
- Preserve facility IDs `1186` and `1656`, routes `/nick` and `/bakke`, forecasting, push notifications, install behavior, product identity, and executable entry points `server/gym_fetch.py`, `server/forecast_api.py`, `server/forecast_job.py`, and `server/facility_hours_fetch.py`.
- Store database timestamps in UTC with microsecond precision and serialize timezone-aware ISO 8601. Use `America/Chicago` for schedules, date grouping, and forecast display.
- Preserve every Phase 1 `location_history.fetched_at` byte unchanged: legacy values are ambiguous naive `America/Chicago` wall times and must never be reinterpreted or rewritten as UTC. The UTC `started_at` of the first Phase 2 `succeeded` ingestion run is the history trust cutover; Phase 4 excludes all earlier history and reports hours without a post-cutover UTC baseline plus successful heartbeat as unavailable.
- Never store, return, print, or log credentials, upstream URLs, response bodies, database values, or unsanitized exceptions. Persist allowlisted failure categories and whitespace-normalized messages capped at 240 characters.
- A successful poll updates every valid snapshot row’s `fetched_at`, including unchanged state. An empty or wholly invalid payload cannot modify a healthy snapshot.
- Preserve `LocationId`, `IsClosed`, `LastCount`, and `LastUpdatedDateAndTime`; add `FetchedAt`. Backend returns `{ingestion, rows}`; frontend temporarily accepts legacy array and `{data: []}` forms.
- Repository snapshot reads expose a row-list helper for internal consumers; the evaluator consumes only that internal list.
- Only the `/api/live-counts` HTTP handler wraps a snapshot list in the public `{ingestion, rows}` envelope.
- Use a centralized ten-minute frontend freshness threshold. Coverage is `live` at >= 0.8, `partial` at >= 0.5 and < 0.8, and `insufficient` below 0.5 with expected capacity; hide an insufficient percentage.
- Preserve `ACTUAL_HOUR_MIN_COVERAGE=0.75`. Do not scale partial observations or put a low-coverage estimate in `actualCount`.
- New runtime imports use `reclive.*` when `server/` is the Python path; avoid repository/ingestion cycles with `TYPE_CHECKING` or a shared model boundary. Preserve direct `python server/*.py` entry compatibility with an explicit smoke test.
- Use focused backend tests in `tests/backend/`, shared fixtures in `tests/fixtures/`, and colocated frontend tests in `src/**/*.test.ts(x)`.

## Execution and Commit Discipline

- This dedicated fresh clone is the isolated Phase 1 workspace; do not create a second worktree.
- Task 0 is a plan-only temporary commit, `docs: reconcile data trust interfaces`. Tasks 1-4, 5-6, and 7-9 may use temporary task commits for review. After each phase-wide gate, squash the unpublished task commits into exactly these logical commits: `feat: trust occupancy ingestion snapshots`, `feat: show trustworthy occupancy coverage`, and `fix: qualify time-weighted actual occupancy`. Task 0 is included in the Phase 2 squash.

---

## File Structure

- Create: `server/reclive/ingestion.py` — normalized upstream rows, deterministic validation/deduplication, bounded failure sanitization, and poll orchestration.
- Create: `server/reclive/occupancy_repository.py` — snapshot/history/run transaction operations plus current/historical reads.
- Create: `server/reclive/actual_hours.py` — pure heartbeat-bounded step-function integration.
- Modify: `server/gym_fetch.py` and `server/forecast_api.py`.
- Create: `tests/backend/test_ingestion.py`, `tests/backend/test_live_counts_api.py`, and `tests/backend/test_actual_hours.py`.
- Modify: `tests/backend/conftest.py`; create `tests/fixtures/reclive_fakes.py` for deterministic fake connection, repository factory, FastAPI client/repository override, SQL recorder, forecast-day builder, and UTC clock seams. It defines `fake_db`, `fake_repository`, `fake_repository_factory`, `fixed_utc_now`, `client`, `snapshot_repository`, `repository`, and `sql_recorder` before any task test uses them.
- Create: `src/shared/occupancy/computeOccupancySummary.ts` and `src/shared/occupancy/computeOccupancySummary.test.ts`.
- Modify: `src/lib/types/facility.ts`, `src/lib/api/facilityParser.ts`, `src/lib/storage/facilityCache.ts`, `src/app/App.tsx`, `src/app/warningStatus.ts`, `src/facilities/OccupancyHero.tsx`, `src/facilities/SectionCommandCenter.tsx`, `src/facilities/SectionSummaryOther.tsx`, `src/facilities/FloorHeatMapCard.tsx`, and `src/facilities/CrowdAlertSubscriptionCard.tsx`.
- Create: `src/lib/api/facilityParser.test.ts`, `src/lib/storage/facilityCache.test.ts`, and `src/app/warningStatus.test.ts`.
- Modify: `src/lib/types/forecast.ts` and `src/lib/api/forecastParser.ts`; create `src/lib/api/forecastParser.test.ts`.
- Modify: every static layout object in `src/lib/data/nick.ts` and `src/lib/data/bakke.ts` when `Location.fetchedAt` becomes required.

### Task 1: Define normalized live-row validation and deterministic duplicate selection

**Files:**
- Create: `server/reclive/ingestion.py`
- Create: `tests/backend/test_ingestion.py`

**Interfaces:**
- Consumes: decoded upstream `object` and `Mapping[int, int]` configured capacities.
- Produces: `NormalizedLiveRow`, `ValidationResult`, `validate_and_deduplicate_rows(payload, capacities)`, and `sanitize_ingestion_error(category, detail)`.

- [ ] **Step 1: Write the failing validation tests**

```python
import pytest

from reclive.ingestion import sanitize_ingestion_error, validate_and_deduplicate_rows


def test_blank_source_timestamp_first_observation_is_valid() -> None:
    result = validate_and_deduplicate_rows(
        [{"LocationId": "5761", "IsClosed": False, "LastCount": "47"}], {5761: 100}
    )

    assert result.received_count == 1
    assert result.invalid_count == 0
    assert result.rows[0].location_id == 5761
    assert result.rows[0].source_updated_at is None
    assert result.rows[0].current_capacity == 47


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/backend/test_ingestion.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'reclive.ingestion'`.

- [ ] **Step 3: Write the minimal validation implementation**

```python
@dataclass(frozen=True)
class NormalizedLiveRow:
    location_id: int
    is_closed: bool
    current_capacity: int
    max_capacity: int
    source_updated_at: datetime | None


@dataclass(frozen=True)
class ValidationResult:
    received_count: int
    invalid_count: int
    rows: Sequence[NormalizedLiveRow]


def validate_and_deduplicate_rows(
    payload: object, capacities: Mapping[int, int]
) -> ValidationResult:
    if not isinstance(payload, list):
        raise IngestionValidationError("payload_not_list", "Live feed payload is not a list")
    retained: dict[int, tuple[int, NormalizedLiveRow]] = {}
    invalid_count = 0
    for index, raw in enumerate(payload):
        row = parse_live_row(raw, capacities)
        if row is None:
            invalid_count += 1
            continue
        previous = retained.get(row.location_id)
        if previous is None or source_timestamp_sort_key(row.source_updated_at, index) >= source_timestamp_sort_key(previous[1].source_updated_at, previous[0]):
            retained[row.location_id] = (index, row)
    return ValidationResult(
        received_count=len(payload),
        invalid_count=invalid_count,
        rows=tuple(row for _, row in sorted(retained.values(), key=lambda item: item[1].location_id)),
    )
```

Require a list payload; validate every row independently; accept only configured IDs, actual booleans for `IsClosed`, nonnegative integer counts, positive configured capacity, and optional timezone-aware source timestamps. Numeric IDs/counts may be parsed from documented decimal input, but never accept Python `bool` as an integer in either field. Blank timestamp becomes `None`; malformed timestamp makes only its row invalid. Sort retained rows by `location_id`. For each ID, an aware timestamp always outranks a blank timestamp regardless of order; newest aware timestamp wins; equal aware timestamps and all-blank duplicates retain the last valid input.

Implement `sanitize_ingestion_error(category: str, detail: object) -> tuple[str, str]` with only `network`, `http`, `payload_not_list`, `validation`, `database`, and `transaction` as allowed categories; an unknown category becomes `validation`. It normalizes and caps an explicitly safe detail at 240 characters, but returns the fixed `Ingestion failure` text for non-string detail or a URL, credential-like, or otherwise unsafe string. It must never call `str()` on arbitrary objects or reflect raw URLs, credentials, upstream response data, or exception text.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/backend/test_ingestion.py -q`

Expected: PASS with all validation, deterministic-deduplication, strict-boolean, and sanitization tests green.

### Task 2: Persist snapshots, state-change history, and ingestion runs atomically

**Files:**
- Read: `server/migrations/0002_snapshot_and_ingestion.sql`
- Create: `server/reclive/occupancy_repository.py`
- Modify: `tests/backend/test_ingestion.py`
- Modify: `tests/backend/conftest.py`
- Create: `tests/fixtures/reclive_fakes.py`

**Interfaces:**
- Consumes: `Sequence[NormalizedLiveRow]`, UTC `datetime`, and a PyMySQL connection with `autocommit=False`.
- Produces: `SnapshotRow`, `SnapshotReadProtocol`, `RepositoryFactory`, `IngestionWriteCounts`, `SnapshotRepository.start_run`, `complete_success`, `complete_failure`, and `persist_successful_poll`.

- [ ] **Step 1: Write failing repository behavior tests**

```python
from datetime import datetime, timezone

from reclive.ingestion import NormalizedLiveRow
from reclive.occupancy_repository import SnapshotRepository


def test_unchanged_state_advances_snapshot_heartbeat_without_history(fake_db) -> None:
    counts = SnapshotRepository(fake_db).persist_successful_poll(
        7, [NormalizedLiveRow(5761, False, 47, 100, None)],
        datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc),
    )

    assert counts.snapshot_updated == 1
    assert counts.history_inserted == 0
    assert fake_db.snapshot_updates[0]["fetched_at"] == datetime(2026, 8, 31, 12, 0)


def test_changed_count_with_same_source_timestamp_inserts_history(fake_db) -> None:
    counts = SnapshotRepository(fake_db).persist_successful_poll(
        7, [NormalizedLiveRow(5761, False, 48, 100, datetime(2026, 8, 31, 11, tzinfo=timezone.utc))],
        datetime(2026, 8, 31, 12, 1, tzinfo=timezone.utc),
    )

    assert counts.history_inserted == 1
    assert fake_db.history_inserts[0]["current_capacity"] == 48


def test_first_success_preserves_legacy_wall_time_and_inserts_utc_baseline(fake_db) -> None:
    fake_db.legacy_history_bytes = b"2026-11-01 01:30:00.000000"
    fake_db.succeeded_run_count = 0

    counts = SnapshotRepository(fake_db).persist_successful_poll(
        7, [NormalizedLiveRow(5761, False, 48, 100, None)],
        datetime(2026, 11, 1, 7, 0, tzinfo=timezone.utc),
    )

    assert fake_db.legacy_history_bytes == b"2026-11-01 01:30:00.000000"
    assert counts.history_inserted == 1
    assert fake_db.history_inserts[0]["fetched_at"] == datetime(2026, 11, 1, 7, 0)


def test_mysql_datetime_boundary_maps_naive_utc_binds_and_tuples() -> None:
    aware = datetime(2026, 11, 1, 7, 0, tzinfo=timezone.utc)

    assert as_mysql_utc(aware) == datetime(2026, 11, 1, 7, 0)
    assert to_aware_utc(datetime(2026, 11, 1, 7, 0)) == aware
```

Add `FakeConnection`/`FakeCursor` fixtures that record transactions and typed bind values without a database, plus a fixed UTC clock fixture. The fake must model `commit`, `rollback`, and `close` failures independently so later lifecycle tests assert sanitized cleanup and no leaked exception detail. Use `tests/fixtures/live_counts.py` for reusable valid upstream rows; do not refer to nonexistent `fake_db`, `fake_repository`, `client`, `repository`, or `sql_recorder` fixtures without defining them in `conftest.py` or `reclive_fakes.py`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/backend/test_ingestion.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'reclive.occupancy_repository'`.

- [ ] **Step 3: Write the migration and transactional repository implementation**

```python
@dataclass(frozen=True)
class IngestionWriteCounts:
    history_inserted: int
    snapshot_updated: int


@dataclass(frozen=True)
class SnapshotRow:
    location_id: int
    is_closed: bool
    current_capacity: int
    max_capacity: int
    source_updated_at: datetime | None
    fetched_at: datetime


class SnapshotReadProtocol(Protocol):
    def fetch_live_snapshot_rows(self) -> Sequence[SnapshotRow]: ...


RepositoryFactory = Callable[[Any], "SnapshotRepository"]


class SnapshotRepository:
    def start_run(self, started_at: datetime) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO ingestion_runs(started_at, status, observed_location_ids) VALUES (%s, 'running', JSON_ARRAY())",
                (as_mysql_utc(started_at),),
            )
            return int(cursor.lastrowid)
    def complete_success(self, run_id: int, completed_at: datetime, received_count: int,
                         valid_count: int, counts: IngestionWriteCounts,
                         observed_location_ids: Sequence[int]) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "UPDATE ingestion_runs SET completed_at=%s, status='succeeded', received_count=%s, valid_count=%s, history_inserted_count=%s, snapshot_updated_count=%s, observed_location_ids=%s WHERE id=%s AND status='running'",
                (as_mysql_utc(completed_at), received_count, valid_count, counts.history_inserted, counts.snapshot_updated, json.dumps(sorted(set(observed_location_ids))), run_id),
            )
    def complete_failure(self, run_id: int, completed_at: datetime,
                         category: str, message: str) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "UPDATE ingestion_runs SET completed_at=%s, status='failed', error_category=%s, error_message=%s WHERE id=%s AND status='running'",
                (as_mysql_utc(completed_at), category, message, run_id),
            )
    def persist_successful_poll(self, run_id: int, rows: Sequence[NormalizedLiveRow],
                                fetched_at: datetime) -> IngestionWriteCounts:
        previous = self.lock_snapshots([row.location_id for row in rows])
        requires_utc_baseline = not self.has_succeeded_ingestion_run()
        inserted = 0
        for row in rows:
            self.upsert_snapshot(row, fetched_at)
            if requires_utc_baseline or previous.get(row.location_id) != state_fingerprint(row):
                self.insert_history(row, fetched_at)
                inserted += 1
        return IngestionWriteCounts(history_inserted=inserted, snapshot_updated=len(rows))
```

Consume the already-applied `0002_snapshot_and_ingestion.sql` from Phase 1; do not edit its checksum-controlled contents. Verify it supplies `location_snapshot`, `ingestion_runs`, and indexes on `location_snapshot(fetched_at)`, `ingestion_runs(status, completed_at)`, and `location_history(location_id, fetched_at)`. Phase 1 `0001_core_history.sql` must provide nullable `location_history.source_updated_at DATETIME(6)` with a safe backfill from legacy `last_updated`; new code reads/writes `source_updated_at` and writes `last_updated` as a compatibility alias.

Implement `lock_snapshots` with a dynamic validated-ID placeholder query, `upsert_snapshot` with an explicit `INSERT INTO location_snapshot(location_id, is_closed, current_capacity, max_capacity, source_updated_at, fetched_at, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE is_closed=VALUES(is_closed), current_capacity=VALUES(current_capacity), max_capacity=VALUES(max_capacity), source_updated_at=VALUES(source_updated_at), fetched_at=VALUES(fetched_at), updated_at=VALUES(updated_at)` statement, and `insert_history` with identical provenance values in `source_updated_at` and legacy `last_updated`. Compare exactly `(is_closed, current_capacity, max_capacity, source_updated_at)`. While the first new run is still `running`, detect that no earlier `succeeded` run exists and insert one UTC history baseline for every observed location even when its snapshot fingerprint is unchanged; only then mark that run `succeeded` and store sorted integer observed IDs before transaction commit. This run's UTC `started_at` is the durable trust cutover. Do not update, reinterpret, or timezone-shift any pre-existing history value.

`as_mysql_utc` accepts only aware UTC datetimes and strips `tzinfo` only at the MySQL `DATETIME(6)` bind boundary. `to_aware_utc` maps every naive datetime returned by the UTC-configured database to `value.replace(tzinfo=timezone.utc)`; it never applies Chicago time or local process time. Apply this mapping to ingestion `started_at`/`completed_at` cutover rows, history `fetched_at` tuples, snapshot timestamps, and succeeded-heartbeat `completed_at` tuples before comparison or serialization. Add the bind/tuple fixture test above and reject non-UTC-aware application inputs.

Define the narrow repository protocol/factory seam before route work: `RepositoryFactory` creates `SnapshotRepository` from an explicit connection, and `SnapshotReadProtocol` is the evaluator's internal row-list contract. Ingestion depends on the factory/protocol rather than importing the runtime implementation in a cycle. Keep shared row dataclasses in a cycle-free module or use `TYPE_CHECKING` for type-only references.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/backend/test_ingestion.py -q`

Expected: PASS; the first successful new poll inserts one UTC baseline per observed location without changing legacy bytes, later unchanged state advances `fetched_at` without history, and a later count change inserts exactly one event.

### Task 3: Delegate the durable poll lifecycle from the preserved script entry point

**Files:**
- Modify: `server/reclive/ingestion.py`
- Modify: `server/gym_fetch.py`
- Modify: `tests/backend/test_ingestion.py`
- Modify: `tests/fixtures/reclive_fakes.py`

**Interfaces:**
- Consumes: `Callable[[], object]` payload loader, `Callable[[], Any]` connector that creates `autocommit=False` connections, capacities, and UTC clock.
- Produces: `run_ingestion(fetch_payload, connect, capacities, now) -> IngestionRunResult`; `gym_fetch.main() -> int` remains executable unchanged.

- [ ] **Step 1: Write failing poll lifecycle tests**

```python
import pytest

from reclive.ingestion import run_ingestion
from tests.fixtures.live_counts import LIVE_ROWS


def test_empty_valid_set_records_failure_without_snapshot_write(fake_repository) -> None:
    result = run_ingestion(
        lambda: [{"LocationId": "invalid", "LastCount": -1}],
        lambda: fake_repository.connection, {5761: 100}, fixed_utc_now,
    )

    assert result.status == "failed"
    assert result.error_category == "validation"
    assert fake_repository.snapshot_writes == []
    assert fake_repository.failed_runs == [("validation", "No valid live rows were received")]


def test_fetch_failure_uses_separate_failure_transaction(fake_repository) -> None:
    result = run_ingestion(
        lambda: (_ for _ in ()).throw(RuntimeError("upstream failed")),
        lambda: fake_repository.connection, {5761: 100}, fixed_utc_now,
    )

    assert result.status == "failed"
    assert result.error_category == "network"
    assert fake_repository.failure_transaction_count == 1


def test_exception_sanitization_persists_and_emits_only_fixed_safe_fields(
    fake_repository_factory
) -> None:
    sentinel = object()
    error = RuntimeError("https://host.test/path?password=secret")
    error.sentinel = sentinel

    result = run_ingestion(
        lambda: (_ for _ in ()).throw(error), fake_repository_factory.connect,
        {5761: 100}, fixed_utc_now, repository_factory=fake_repository_factory,
    )

    assert result.error_category == "network"
    assert fake_repository_factory.failed_runs == [("network", "Ingestion failure")]
    assert fake_repository_factory.event_lines == [
        "status=failed received=0 valid=0 history=0 snapshot=0 durationMs=0 category=network"
    ]
    event_output = " ".join(fake_repository_factory.event_lines)
    assert all(token not in event_output for token in ("host.test", "https://", "password", "secret"))
    assert "object at" not in event_output


def test_initial_connection_and_cleanup_failures_are_sanitized(fake_repository_factory) -> None:
    fake_repository_factory.raise_on_initial_connect = RuntimeError("credential-like detail")

    result = run_ingestion(
        lambda: [], fake_repository_factory.connect, {5761: 100}, fixed_utc_now,
        repository_factory=fake_repository_factory,
    )

    assert result.status == "failed"
    assert result.error_category == "database"
    assert "credential-like" not in fake_repository_factory.event_lines[0]


@pytest.mark.parametrize("failure_point", ["rollback", "failure_commit", "work_close", "failure_close"])
def test_primary_failure_wins_when_cleanup_or_failure_recording_also_fails(
    fake_repository_factory, failure_point
) -> None:
    fake_repository_factory.raise_on(failure_point, RuntimeError("credential-like detail"))

    result = run_ingestion(
        lambda: (_ for _ in ()).throw(RuntimeError("upstream failed")),
        fake_repository_factory.connect, {5761: 100}, fixed_utc_now,
        repository_factory=fake_repository_factory,
    )

    assert result.error_category == "network"
    assert fake_repository_factory.all_connections_closed
    assert all("credential-like" not in line for line in fake_repository_factory.event_lines)


def test_work_connection_does_not_commit_before_complete_success(fake_repository_factory) -> None:
    run_ingestion(lambda: LIVE_ROWS, fake_repository_factory.connect, {5761: 100}, fixed_utc_now,
                  repository_factory=fake_repository_factory)

    assert fake_repository_factory.run_events == ["start", "commit", "close"]
    assert fake_repository_factory.work_events == ["persist", "complete_success", "commit", "close"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/backend/test_ingestion.py -q`

Expected: FAIL with `ImportError: cannot import name 'run_ingestion'`.

- [ ] **Step 3: Write the minimal orchestration**

```python
@dataclass(frozen=True)
class IngestionRunResult:
    status: Literal["succeeded", "failed"]
    received_count: int
    valid_count: int
    history_inserted: int
    snapshot_updated: int
    error_category: str | None


def sanitize_ingestion_exception(exc: BaseException) -> tuple[str, str]:
    return sanitize_ingestion_error(classify_ingestion_exception(exc), exc)


def run_ingestion(
    fetch_payload: Callable[[], object], connect: Callable[[], Any],
    capacities: Mapping[int, int], now: Callable[[], datetime],
    repository_factory: RepositoryFactory = SnapshotRepository,
) -> IngestionRunResult:
    run_connection = None
    work_connection = None
    run_id = None
    try:
        started_at = require_aware_utc(now())
        run_connection = connect()
        run_repository = repository_factory(run_connection)
        run_id = run_repository.start_run(started_at)
        run_connection.commit()
        safe_close(run_connection)
        run_connection = None
        work_connection = connect()
        repository = repository_factory(work_connection)
        validated = validate_and_deduplicate_rows(fetch_payload(), capacities)
        if not validated.rows:
            raise IngestionValidationError("validation", "No valid live rows were received")
        counts = repository.persist_successful_poll(run_id, validated.rows, now())
        repository.complete_success(run_id, now(), validated.received_count, len(validated.rows), counts, [row.location_id for row in validated.rows])
        work_connection.commit()
        return IngestionRunResult("succeeded", validated.received_count, len(validated.rows), counts.history_inserted, counts.snapshot_updated, None)
    except Exception as exc:
        safe_rollback(work_connection)
        category, message = sanitize_ingestion_exception(exc)
        failure_connection = safe_connect(connect)
        try:
            if run_id is not None and failure_connection is not None:
                repository_factory(failure_connection).complete_failure(run_id, require_aware_utc(now()), category, message)
                failure_connection.commit()
        finally:
            safe_close(failure_connection)
        return IngestionRunResult("failed", 0, 0, 0, 0, category)
    finally:
        safe_close(run_connection)
        safe_close(work_connection)
```

Make `db_connect()` explicitly return `pymysql.connect(..., autocommit=False)`. Make `fetch_live()` use `requests.get(LIVE_COUNTS_URL, timeout=(5, 20))`, raise for HTTP failure, decode JSON once, and return `object`. `sanitize_ingestion_exception` must classify the exception and delegate all category/message handling to `sanitize_ingestion_error`; it must not construct a parallel sanitizer or access exception text/attributes. Guard the initial clock/connection/repository/start-run path before the main fetch transaction; create and commit the `running` run through a short-lived run connection before fetching so a later work rollback cannot erase it. Close that connection before opening the work connection. The work connection has no commit before `complete_success`; persist snapshots/history, mark the run successful, then commit once. Empty valid input completes as `validation` without touching snapshot rows. After a recorded run exists, rollback failed work and record a sanitized failure through a fresh transaction. Guard rollback, failure-record commit, and the run/work/failure close paths so cleanup errors never mask the primary result or emit exception details. If a database connection cannot be opened, return failure without claiming a run was recorded. Emit exactly `status=<status> received=<n> valid=<n> history=<n> snapshot=<n> durationMs=<n> category=<category>` and no other fields; the URL/credential/sentinel exception fixture must persist `Ingestion failure` and emit only that fixed-field line.

Retain direct script compatibility: `server/gym_fetch.py` imports its runtime collaborators as `reclive.*` when launched with `python server/gym_fetch.py`, while package imports work under the test runner. Add a subprocess/import smoke test using controlled non-secret configuration and a fake connector/payload path; assert an executable script entry reaches the injected runner and never prints URL, payload, credentials, or raw exceptions.

- [ ] **Step 4: Run focused verification**

Run: `pytest tests/backend/test_ingestion.py -q && ruff check server/gym_fetch.py server/reclive/ingestion.py server/reclive/occupancy_repository.py`

Expected: PASS; assertions and output contain no raw response, exception, URL, or credential data.

### Task 4: Return the snapshot-backed live-count envelope

**Files:**
- Modify: `server/reclive/occupancy_repository.py`
- Modify: `server/forecast_api.py`
- Create: `tests/backend/test_live_counts_api.py`
- Modify: `tests/backend/conftest.py` and `tests/fixtures/reclive_fakes.py`

**Interfaces:**
- Consumes: `SnapshotRepository.fetch_live_snapshot(now: datetime) -> LiveSnapshotRead`.
- Produces: `SnapshotRepository.fetch_live_snapshot_rows() -> list[SnapshotRow]` for internal consumers.
- Produces: unchanged `GET /api/live-counts` returning `{ingestion, rows}` with `FetchedAt` on each public row.

- [ ] **Step 1: Write failing route contract tests**

```python
from forecast_api import evaluate_rules_once


def test_live_counts_returns_snapshot_envelope_and_fetched_at(client, snapshot_repository) -> None:
    snapshot_repository.live_snapshot = [
        {"location_id": 5761, "is_closed": False, "current_capacity": 47,
         "source_updated_at": "2026-08-31T11:59:00Z", "fetched_at": "2026-08-31T12:00:00Z"}
    ]
    snapshot_repository.last_successful_fetch_at = "2026-08-31T12:00:00Z"

    response = client.get("/api/live-counts")

    assert response.status_code == 200
    assert response.json()["ingestion"]["status"] == "healthy"
    assert response.json()["rows"][0]["FetchedAt"] == "2026-08-31T12:00:00+00:00"


def test_live_counts_never_reads_location_history(client, sql_recorder) -> None:
    assert client.get("/api/live-counts").status_code == 200
    assert any("location_snapshot" in query for query in sql_recorder.queries)
    assert not any("location_history" in query for query in sql_recorder.queries)


def test_evaluator_uses_internal_snapshot_rows(snapshot_repository) -> None:
    snapshot_repository.live_snapshot = [
        {"location_id": 5761, "is_closed": False, "current_capacity": 47,
         "max_capacity": 100, "source_updated_at": None, "fetched_at": "2026-08-31T12:00:00Z"}
    ]

    evaluate_rules_once(snapshot_reader=snapshot_repository)

    assert snapshot_repository.internal_row_read_count == 1
    assert snapshot_repository.public_envelope_read_count == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/backend/test_live_counts_api.py -q`

Expected: FAIL because the current route returns a list and reads `location_history` through `MAX(id)`.

- [ ] **Step 3: Implement snapshot-only API reads**

```python
@dataclass(frozen=True)
class LiveSnapshotRead:
    last_successful_fetch_at: datetime | None
    rows: Sequence[SnapshotRow]


def fetch_live_snapshot(self, now: datetime) -> LiveSnapshotRead:
    with self.connection.cursor() as cursor:
        cursor.execute("SELECT location_id, is_closed, current_capacity, max_capacity, source_updated_at, fetched_at FROM location_snapshot ORDER BY location_id")
        rows = tuple(SnapshotRow.from_db(row) for row in cursor.fetchall())
        cursor.execute("SELECT completed_at FROM ingestion_runs WHERE status='succeeded' ORDER BY completed_at DESC LIMIT 1")
        latest = cursor.fetchone()
    return LiveSnapshotRead(to_aware_utc(latest[0]) if latest else None, rows)


def get_snapshot_repository() -> Iterator[SnapshotRepository]:
    connection = open_db_connection(autocommit=False)
    try:
        yield SnapshotRepository(connection)
    finally:
        safe_close(connection)


def evaluate_rules_once(
    snapshot_reader: SnapshotReadProtocol | None = None,
    repository_factory: RepositoryFactory = SnapshotRepository,
) -> Dict[str, Any]:
    connection = None
    try:
        if snapshot_reader is None:
            connection = open_db_connection(autocommit=False)
            snapshot_reader = repository_factory(connection)
        live_rows = snapshot_reader.fetch_live_snapshot_rows()
        return evaluate_live_rows(live_rows)
    finally:
        safe_close(connection)
```

Read `location_snapshot` ordered by `location_id` and separately the newest `ingestion_runs` row with `status = 'succeeded'`. `fetch_live_snapshot_rows()` returns the internal `SnapshotRow` list without HTTP metadata and is the only replacement for the evaluator's current list consumer. `fetch_live_snapshot()` may compose that list with ingestion metadata. At the HTTP boundary, map the tuple/db row fields explicitly: `source_updated_at` -> `LastUpdatedDateAndTime` and `fetched_at` -> `FetchedAt`; serialize both as aware UTC `isoformat()` values. Do not substitute fetch time for source provenance.

Calculate nonnegative `ageSeconds`; return `healthy` through ten minutes, `stale` afterwards, and `unavailable` with null success time when no run succeeded. Return 503 only for unavailable database or empty snapshot. Define `get_snapshot_repository()` as an HTTP-only FastAPI dependency and override `app.dependency_overrides[get_snapshot_repository]` in route tests; do not monkeypatch a raw module-global connection. The HTTP handler receives `repository: SnapshotRepository = Depends(get_snapshot_repository)`. `evaluate_rules_once(snapshot_reader: SnapshotReadProtocol | None = None, repository_factory: RepositoryFactory = SnapshotRepository)` uses direct injection for tests and opens/closes its own connection only when no reader was supplied; the evaluator never uses FastAPI `Depends` or receives an HTTP response shape. Add factory-path and injected-reader tests that assert the internally opened evaluator connection is closed exactly once.

- [ ] **Step 4: Run Phase 2 checks and create its logical commit**

Run: `pytest tests/backend/test_ingestion.py tests/backend/test_live_counts_api.py -q && ruff check server/gym_fetch.py server/forecast_api.py server/reclive`

Expected: PASS; no live-count SQL uses `location_history`.

After this gate, squash the Task 0 `docs: reconcile data trust interfaces` temporary commit and reviewed Tasks 1-4 commits into the single Phase 2 logical commit `feat: trust occupancy ingestion snapshots`. Its staged file list is exactly `docs/superpowers/plans/2026-08-31-reclive-data-trust.md`, `server/reclive/ingestion.py`, `server/reclive/occupancy_repository.py`, `server/gym_fetch.py`, `server/forecast_api.py`, `tests/backend/conftest.py`, `tests/backend/test_ingestion.py`, `tests/backend/test_live_counts_api.py`, and `tests/fixtures/reclive_fakes.py`.

### Task 5: Establish the sole shared occupancy-summary contract

**Files:**
- Create: `src/shared/occupancy/computeOccupancySummary.ts`
- Create: `src/shared/occupancy/computeOccupancySummary.test.ts`
- Modify: `src/lib/types/facility.ts`
- Modify: `src/lib/data/nick.ts` and `src/lib/data/bakke.ts`

**Interfaces:**
- Consumes: `readonly Location[]` and `ComputeOccupancySummaryOptions`.
- Produces: `computeOccupancySummary(locations, options): OccupancySummary` and `OCCUPANCY_FRESHNESS_MS`.

- [ ] **Step 1: Write failing coverage boundary tests**

```ts
import {computeOccupancySummary} from "./computeOccupancySummary";

it("does not turn a stale room into zero", () => {
    const result = computeOccupancySummary([
        location({locationId: 1, currentCapacity: 20, maxCapacity: 100, fetchedAt: "2026-08-31T12:00:00Z"}),
        location({locationId: 2, currentCapacity: 80, maxCapacity: 100, fetchedAt: "2026-08-31T11:49:59Z"}),
    ], {nowMs: Date.parse("2026-08-31T12:00:00Z")});

    expect(result).toMatchObject({count: 20, observedCapacity: 100, expectedOpenCapacity: 200, coverage: 0.5, percent: 20, status: "partial"});
});

it("treats future fetch timestamp as untrusted", () => {
    const result = computeOccupancySummary([location({fetchedAt: "2026-08-31T12:00:01Z"})], {nowMs: Date.parse("2026-08-31T12:00:00Z")});

    expect(result).toMatchObject({count: null, percent: null, status: "insufficient"});
});

it("excludes only fresh confirmed closure from expected capacity", () => {
    const result = computeOccupancySummary([location({isClosed: true, fetchedAt: "2026-08-31T12:00:00Z"})], {nowMs: Date.parse("2026-08-31T12:00:00Z")});

    expect(result).toMatchObject({expectedOpenCapacity: 0, percent: null, status: "closed"});
});
```

Add table-driven cases for missing `fetchedAt`, invalid/negative counts, invalid capacities, all-missing locations, and exactly 0.8 coverage.

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm run test:run -- src/shared/occupancy/computeOccupancySummary.test.ts`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the exact shared helper**

```ts
export interface OccupancySummary {
    count: number | null;
    observedCapacity: number;
    expectedOpenCapacity: number;
    coverage: number;
    percent: number | null;
    observedLocations: number;
    expectedLocations: number;
    latestFetchedAt: string | null;
    oldestFetchedAt: string | null;
    status: "live" | "partial" | "insufficient" | "closed" | "unknown";
}

export interface ComputeOccupancySummaryOptions {
    nowMs?: number;
    freshnessMs?: number;
}

export const OCCUPANCY_FRESHNESS_MS = 10 * 60 * 1000;

export declare function computeOccupancySummary(
    locations: readonly Location[], options: ComputeOccupancySummaryOptions = {}
): OccupancySummary;
```

Add required `fetchedAt: string | null` to `Location`, then add `fetchedAt: null` to every static location literal in `src/lib/data/nick.ts` and `src/lib/data/bakke.ts`. A row is fresh only if its parsed timestamp is finite, no later than `nowMs`, and within `freshnessMs`. Fresh closed rows exclude configured positive capacity from expected open capacity. Missing, stale, invalid, or future rows retain expected capacity but add neither count nor observed capacity. Valid fresh open rows require finite nonnegative count and positive finite capacity. Return `unknown` where no location establishes a trustworthy fresh-open or fresh-closed conclusion; return `closed` only when a nonempty expected set is all fresh confirmed closures.

- [ ] **Step 4: Run the test and type check to verify it passes**

Run: `npm run test:run -- src/shared/occupancy/computeOccupancySummary.test.ts && npm run build`

Expected: PASS; no null count is coerced to zero.

### Task 6: Parse FetchedAt, reject future cache values, and migrate all occupancy consumers

**Files:**
- Modify: `src/lib/api/facilityParser.ts`
- Modify: `src/lib/storage/facilityCache.ts`
- Modify: `src/app/App.tsx` and `src/app/warningStatus.ts`
- Modify: `src/facilities/OccupancyHero.tsx`, `src/facilities/SectionCommandCenter.tsx`, `src/facilities/SectionSummaryOther.tsx`, `src/facilities/FloorHeatMapCard.tsx`, and `src/facilities/CrowdAlertSubscriptionCard.tsx`
- Create: `src/lib/api/facilityParser.test.ts`, `src/lib/storage/facilityCache.test.ts`, `src/app/warningStatus.test.ts`, and `src/facilities/CrowdAlertSubscriptionCard.test.tsx`
- Modify: `src/lib/data/nick.ts` and `src/lib/data/bakke.ts`

**Interfaces:**
- Consumes: canonical/legacy live payloads and Task 5 `OccupancySummary`.
- Produces: `Location.fetchedAt` and summary-driven component props, not local total/max/percent calculations.

- [ ] **Step 1: Write failing parser/cache/consumer tests**

```ts
import {resolveInitialSectionKey} from "./CrowdAlertSubscriptionCard";

it("maps canonical FetchedAt onto a configured location", async () => {
    mockLiveCountsEnvelope({rows: [{LocationId: 5761, IsClosed: false, LastCount: 47, LastUpdatedDateAndTime: null, FetchedAt: "2026-08-31T12:00:00Z"}]});

    await expect(fetchFacility(1186)).resolves.toMatchObject({
        locations: expect.arrayContaining([expect.objectContaining({locationId: 5761, fetchedAt: "2026-08-31T12:00:00Z"})]),
    });
});

it("rejects a future facility cache timestamp", () => {
    localStorage.setItem(CACHE_KEY, JSON.stringify({"1186": {version: CACHE_VERSION, cachedAt: Date.now() + 1, payload: fixturePayload}}));

    expect(getFacilityCache(1186)).toBeNull();
});

it("labels a partial summary without calling it complete", () => {
    expect(resolveDashboardWarning({hasAnyError: false, isOffline: false, liveOutageState: "none", liveDataSource: "facility_api", forecastError: null, isScheduledClosedNow: false, isScheduledOpenButDataNotLive: false, occupancyStatus: "partial"}).kind).toBe("partial_live");
});

it("clears a saved alert selection when its summary is insufficient", () => {
    expect(resolveInitialSectionKey(1186, [unavailableOverall, liveWeights])).toBe("weights");
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm run test:run -- src/lib/api/facilityParser.test.ts src/lib/storage/facilityCache.test.ts src/app/warningStatus.test.ts src/facilities/CrowdAlertSubscriptionCard.test.tsx`

Expected: FAIL because `FetchedAt` is absent, future cache entries are fresh, and `partial_live` is unknown.

- [ ] **Step 3: Implement the single summary-driven presentation path**

```ts
interface LiveLocationRow {
    LocationId: number;
    IsClosed: boolean | null;
    LastCount: number | null;
    LastUpdatedDateAndTime: string | null;
    FetchedAt?: string | null;
}

interface LiveCountsPayloadEnvelope {
    ingestion?: {lastSuccessfulFetchAt: string | null; ageSeconds: number | null; status: "healthy" | "stale" | "unavailable"};
    rows?: LiveLocationRow[];
    data?: LiveLocationRow[];
}

export interface AlertSectionOption {
    key: string;
    label: string;
    summary: OccupancySummary;
}
```

Parse `rows` first, then legacy array and `data`; legacy rows receive `fetchedAt: null`. Bump `CACHE_VERSION` from `2` to the manual literal `3` and require `entry.version === 3` plus `entry.cachedAt <= now` in `isFreshEntry`. Keep this Phase 3 cache guard dependency-free: do not add a schema-library dependency or any Phase 6 schema work.

In `App.tsx`, calculate a single render-time `nowTs` and pass that same value to every facility, section, Other-space, per-row, and alert-option `computeOccupancySummary` call. Replace the current `AlertSectionOption` fields `percent`, `total`, and `max` with its `summary` field; change the card’s threshold upper-bound calculation to use `summary.percent` only when status is `live` or `partial`. A saved/default alert selection is valid only for `live` or `partial`; clear an invalid saved selection and choose the first valid option, or no selection when none qualify. Export `resolveInitialSectionKey` from `CrowdAlertSubscriptionCard.tsx` and directly test it in `CrowdAlertSubscriptionCard.test.tsx` so the cleared-selection behavior does not rely on a private helper.

Remove `isDataLikelyStale` from `WarningResolverInput`, every resolver call site, and warning decision logic. Replace it with required `occupancyStatus: OccupancySummary["status"]`; `live`/`partial` mean observed live data is usable, and `insufficient`/`unknown` convey unavailable live occupancy without suppressing a valid forecast. Remove every listed consumer’s `currentCapacity ?? 0`, `maxCapacity ?? 0`, and source-timestamp freshness arithmetic. Hero uses `summary.count / summary.observedCapacity`, and all live freshness derives from `summary.latestFetchedAt`, never source update time. Partial text says `Coverage: N% of open capacity observed`; insufficient/unknown says `Live occupancy unavailable` with no determinate bar. Individual rows use a one-location summary. Add `partial_live` text that preserves forecasts and identifies observed-capacity coverage.

Each heat-map zone derives a summary. Keep fresh confirmed closures distinct; color only live/partial zones; render unknown/insufficient zones neutral. Popovers include `Coverage: N%` for partial and `Live occupancy unavailable` otherwise. Alert options accept `summary: OccupancySummary`, disable unknown/insufficient choices, and label partial choices with coverage.

- [ ] **Step 4: Run Phase 3 checks and create its logical commit**

Run: `npm run test:run -- src/shared/occupancy/computeOccupancySummary.test.ts src/lib/api/facilityParser.test.ts src/lib/storage/facilityCache.test.ts src/app/warningStatus.test.ts src/facilities/CrowdAlertSubscriptionCard.test.tsx && npm run lint && npm run build`

Expected: PASS; stale, missing, invalid, and future rows cannot become zero-percent green.

After this gate, squash the reviewed Tasks 5-6 commits into the single Phase 3 logical commit `feat: show trustworthy occupancy coverage`. Its staged file list is exactly `src/shared/occupancy/computeOccupancySummary.ts`, `src/shared/occupancy/computeOccupancySummary.test.ts`, `src/lib/types/facility.ts`, `src/lib/data/nick.ts`, `src/lib/data/bakke.ts`, `src/lib/api/facilityParser.ts`, `src/lib/api/facilityParser.test.ts`, `src/lib/storage/facilityCache.ts`, `src/lib/storage/facilityCache.test.ts`, `src/app/App.tsx`, `src/app/warningStatus.ts`, `src/app/warningStatus.test.ts`, `src/facilities/OccupancyHero.tsx`, `src/facilities/SectionCommandCenter.tsx`, `src/facilities/SectionSummaryOther.tsx`, `src/facilities/FloorHeatMapCard.tsx`, `src/facilities/CrowdAlertSubscriptionCard.tsx`, and `src/facilities/CrowdAlertSubscriptionCard.test.tsx`.

### Task 7: Build the pure heartbeat-bounded time-weighted actual-hour service

**Files:**
- Create: `server/reclive/actual_hours.py`
- Create: `tests/backend/test_actual_hours.py`

**Interfaces:**
- Consumes: `HistoryState`, `IngestionHeartbeat`, `HourWindow`, location IDs, expected capacity, and threshold.
- Produces: `calculate_actual_hour(location_ids, expected_capacity, window, states, heartbeats, coverage_threshold) -> ActualHourSummary` and `build_chicago_hour_windows(date_key) -> list[HourWindow]`.

- [ ] **Step 1: Write failing step-function and DST tests**

```python
from datetime import datetime, timezone

from reclive.actual_hours import HistoryState, HourWindow, IngestionHeartbeat, build_chicago_hour_windows, calculate_actual_hour


def test_integrates_step_changes_over_one_hour() -> None:
    summary = calculate_actual_hour(
        [5761], 100,
        HourWindow(datetime(2026, 8, 31, 12, tzinfo=timezone.utc), datetime(2026, 8, 31, 13, tzinfo=timezone.utc)),
        [HistoryState(5761, False, 20, 100, datetime(2026, 8, 31, 11, 50, tzinfo=timezone.utc)),
         HistoryState(5761, False, 60, 100, datetime(2026, 8, 31, 12, 30, tzinfo=timezone.utc))],
        [IngestionHeartbeat(datetime(2026, 8, 31, 12, 30, tzinfo=timezone.utc), frozenset({5761})),
         IngestionHeartbeat(datetime(2026, 8, 31, 13, tzinfo=timezone.utc), frozenset({5761}))],
        0.75,
    )

    assert summary.observed_count == 40
    assert summary.actual_count == 40
    assert summary.actual_coverage == 1.0
    assert summary.temporal_coverage == 1.0


def test_low_capacity_coverage_never_scales_observed_count() -> None:
    summary = calculate_actual_hour(
        [5761], 200,
        HourWindow(datetime(2026, 8, 31, 12, tzinfo=timezone.utc), datetime(2026, 8, 31, 13, tzinfo=timezone.utc)),
        [HistoryState(5761, False, 42, 100, datetime(2026, 8, 31, 11, 59, tzinfo=timezone.utc))],
        [IngestionHeartbeat(datetime(2026, 8, 31, 13, tzinfo=timezone.utc), frozenset({5761}))],
        0.75,
    )

    assert summary.observed_count == 42
    assert summary.observed_capacity == 100
    assert summary.expected_capacity == 200
    assert summary.actual_coverage == 0.5
    assert summary.temporal_coverage == 1.0
    assert summary.actual_count is None


def test_capacity_and_temporal_coverage_are_independent() -> None:
    summary = calculate_actual_hour(
        [5761], 200,
        HourWindow(datetime(2026, 8, 31, 12, tzinfo=timezone.utc), datetime(2026, 8, 31, 13, tzinfo=timezone.utc)),
        [HistoryState(5761, False, 42, 100, datetime(2026, 8, 31, 11, 59, tzinfo=timezone.utc))],
        [IngestionHeartbeat(datetime(2026, 8, 31, 12, 57, tzinfo=timezone.utc), frozenset({5761}))],
        0.75,
    )

    assert summary.observed_count == 42
    assert summary.observed_capacity == 100
    assert summary.actual_coverage == 0.5
    assert summary.temporal_coverage == 0.95
    assert summary.actual_count is None


def test_chicago_dst_day_window_counts_are_physical_hours() -> None:
    assert len(build_chicago_hour_windows("2026-03-08")) == 23
    assert len(build_chicago_hour_windows("2026-11-01")) == 25
```

Add constant-state, pre-range seed, no-history, closed-location, below-0.75 temporal coverage, and equal-`fetched_at` ID tie cases. Give `HistoryState` a deterministic event `id`; repository ordering is `(fetched_at, id)`, with the higher ID winning an equal-timestamp transition. Assert half-open intervals `[start, end)`: a state becomes effective exactly at its `fetched_at`, an interval ending at the next event does not double-count either state, and a heartbeat at the boundary confirms through the boundary only. Assert a confirmed closed segment contributes neither count nor observed capacity, but does not reduce the full configured `expected_capacity`; closed time may make an hour unqualified and must never be reported as observed zero occupancy.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/backend/test_actual_hours.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'reclive.actual_hours'`.

- [ ] **Step 3: Implement pure interval integration**

```python
@dataclass(frozen=True)
class ActualHourSummary:
    observed_count: int | None
    observed_capacity: int
    expected_capacity: int
    actual_coverage: float
    temporal_coverage: float
    coverage_threshold: float
    actual_count: int | None


def calculate_actual_hour(
    location_ids: Sequence[int], expected_capacity: int, window: HourWindow,
    states: Sequence[HistoryState], heartbeats: Sequence[IngestionHeartbeat],
    coverage_threshold: float,
) -> ActualHourSummary:
    observed_count_total = 0.0
    observed_capacity_total = 0.0
    confirmed_capacity_seconds = 0.0
    for location_id in location_ids:
        location_known_seconds = 0.0
        location_count_seconds = 0.0
        location_capacity_seconds = 0.0
        segments = known_state_segments(location_id, window, states, heartbeats)
        for segment in segments:
            duration = (segment.end - segment.start).total_seconds()
            if duration <= 0 or segment.is_closed or segment.capacity <= 0:
                continue
            location_known_seconds += duration
            location_count_seconds += segment.count * duration
            location_capacity_seconds += segment.capacity * duration
        if location_known_seconds > 0:
            observed_count_total += location_count_seconds / location_known_seconds
            observed_capacity_total += location_capacity_seconds / location_known_seconds
            confirmed_capacity_seconds += location_capacity_seconds
    hour_seconds = (window.end - window.start).total_seconds()
    observed_count = round(observed_count_total) if observed_capacity_total > 0 else None
    observed_capacity = round(observed_capacity_total)
    actual_coverage = observed_capacity / expected_capacity if expected_capacity > 0 else 0.0
    temporal_denominator = observed_capacity_total * hour_seconds
    temporal_coverage = confirmed_capacity_seconds / temporal_denominator if temporal_denominator > 0 else 0.0
    actual_count = observed_count if observed_count is not None and actual_coverage >= coverage_threshold and temporal_coverage >= coverage_threshold else None
    return ActualHourSummary(observed_count, observed_capacity, expected_capacity, actual_coverage, temporal_coverage, coverage_threshold, actual_count)
```

For every location, seed with its latest event strictly before range start and append later events ordered by `(fetched_at, id)`. `fetched_at` is the only event time. Intervals are half-open `[start, end)`, so a state is effective at its `fetched_at` and no boundary is double-counted. A state is known only from event time through the next successful heartbeat containing that location, capped by the next state change and hour end. Within each location, average count and capacity over only its confirmed non-closed intervals; sum those per-location observed averages without filling unknown time with zero. Capacity coverage is the summed observed-location capacity divided by the full configured expected capacity, including locations that are confirmed closed. A confirmed closed segment contributes neither count nor observed capacity; it can make an hour unqualified but is never observed zero occupancy. Temporal coverage is independently the confirmed capacity-seconds divided by the observed-location capacity-seconds that would exist for the full hour. Thus one 100-capacity location observed for 57 minutes in a 200-capacity facility reports `actualCoverage=0.5` and `temporalCoverage=0.95`, rather than collapsing both dimensions to `0.475`. Return actual count only when both independent measures meet threshold. Never multiply an observed count by a capacity ratio or include unknown seconds as zero.

Build Chicago hours as timezone-aware local boundaries converted to UTC, keeping both fall-back 01:00 offsets and omitting the nonexistent spring-forward hour. Accept only timezone-aware UTC states already filtered by the repository's trust cutover; the service must never parse a naive legacy wall time, guess an offset, or place such a value into either repeated fall-back hour or across the spring-forward gap. Task 2's first-baseline test, this task's 23/25-hour DST test, and Task 8's fall-back legacy-row exclusion test jointly lock the boundary.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/backend/test_actual_hours.py -q`

Expected: PASS; 20 for thirty minutes plus 60 for thirty minutes yields 40, not an event average or extrapolation.

### Task 8: Use historical seed states and run heartbeats in the existing actual-hours API

**Files:**
- Modify: `server/reclive/occupancy_repository.py`
- Modify: `server/forecast_api.py`
- Modify: `tests/backend/test_actual_hours.py`
- Modify: `tests/backend/conftest.py` and `tests/fixtures/reclive_fakes.py`

**Interfaces:**
- Consumes: preserved `GET /api/forecast/facilities/{facility_id}/actual-hours?date=YYYY-MM-DD` and `ACTUAL_HOUR_MIN_COVERAGE=0.75`.
- Produces: per-hour `observedCount`, `observedCapacity`, `expectedCapacity`, `actualCoverage`, `temporalCoverage`, `coverageThreshold`, and `actualCount`.

- [ ] **Step 1: Write failing route tests**

```python
from datetime import datetime, timezone

import pytest


def test_actual_hours_serializes_unscaled_low_coverage(client, repository) -> None:
    repository.actual_hour_summary = {
        "observed_count": 42, "observed_capacity": 100, "expected_capacity": 200,
        "actual_coverage": 0.5, "temporal_coverage": 0.95, "coverage_threshold": 0.75,
        "actual_count": None,
    }

    hour = client.get("/api/forecast/facilities/1186/actual-hours?date=2026-08-31").json()["totalHours"][0]

    assert hour["observedCount"] == 42
    assert hour["expectedCapacity"] == 200
    assert hour["coverageThreshold"] == 0.75
    assert hour["actualCount"] is None


def test_actual_hours_loads_pre_range_seed_and_succeeded_heartbeats(client, sql_recorder) -> None:
    client.get("/api/forecast/facilities/1186/actual-hours?date=2026-08-31")

    assert any("fetched_at <" in query and "ORDER BY fetched_at DESC" in query for query in sql_recorder.queries)
    assert any("ingestion_runs" in query and "observed_location_ids" in query and "status='succeeded'" in query for query in sql_recorder.queries)


def test_actual_hours_excludes_legacy_wall_time_before_first_utc_baseline(client, sql_recorder) -> None:
    sql_recorder.first_succeeded_started_at = datetime(
        2026, 11, 1, 6, 59
    )
    sql_recorder.history_rows = [
        (5761, False, 90, 100, "2026-11-01 01:30:00.000000", 1),
        (5761, False, 40, 100, datetime(2026, 11, 1, 7, 0), 2),
    ]

    response = client.get(
        "/api/forecast/facilities/1186/actual-hours?date=2026-11-01"
    ).json()

    assert sql_recorder.history_cutover == datetime(2026, 11, 1, 6, 59, tzinfo=timezone.utc)
    assert sql_recorder.cutover_binds == [
        datetime(2026, 11, 1, 6, 59),
        datetime(2026, 11, 1, 6, 59),
    ]
    assert sql_recorder.returned_history_rows == sql_recorder.history_rows[1:]
    assert all(state.fetched_at.tzinfo == timezone.utc for state in sql_recorder.parsed_history_states)
    assert all(heartbeat.completed_at.tzinfo == timezone.utc for heartbeat in sql_recorder.parsed_heartbeats)
    assert response["totalHours"][0]["actualCount"] is None


@pytest.mark.parametrize(("date_key", "expected_hours"), [("2026-03-08", 23), ("2026-11-01", 25)])
def test_actual_hours_route_uses_physical_chicago_hour_windows(client, date_key, expected_hours) -> None:
    response = client.get(f"/api/forecast/facilities/1186/actual-hours?date={date_key}")

    assert response.status_code == 200
    hour_starts = [item["hourStart"] for item in response.json()["totalHours"]]
    assert len(hour_starts) == expected_hours
    if date_key == "2026-03-08":
        assert not any(start.startswith("2026-03-08T02:") for start in hour_starts)
    else:
        assert "2026-11-01T01:00:00-05:00" in hour_starts
        assert "2026-11-01T01:00:00-06:00" in hour_starts
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/backend/test_actual_hours.py -q`

Expected: FAIL because current code averages events by hour, scales incomplete capacity, and omits the required fields.

- [ ] **Step 3: Replace hourly event averaging with repository reads plus the service**

```python
def load_actual_hour_inputs(
    self, location_ids: Sequence[int], range_start: datetime, range_end: datetime,
) -> tuple[list[HistoryState], list[IngestionHeartbeat]]:
    placeholders = ",".join(["%s"] * len(location_ids))
    with self.connection.cursor() as cursor:
        cursor.execute(
            "SELECT started_at FROM ingestion_runs WHERE status='succeeded' ORDER BY started_at, id LIMIT 1"
        )
        first_success = cursor.fetchone()
        if first_success is None:
            return [], []
        cutover_started_at = to_aware_utc(first_success[0])
        cutover_bind = as_mysql_utc(cutover_started_at)
        cursor.execute(
            f"SELECT h.location_id, h.is_closed, h.current_capacity, h.max_capacity, h.fetched_at, h.id FROM location_history h JOIN (SELECT id, ROW_NUMBER() OVER (PARTITION BY location_id ORDER BY fetched_at DESC, id DESC) AS row_number FROM location_history WHERE location_id IN ({placeholders}) AND fetched_at >= %s AND fetched_at < %s) AS seed ON seed.id = h.id WHERE seed.row_number = 1",
            (*location_ids, cutover_bind, as_mysql_utc(range_start)),
        )
        seed_rows = cursor.fetchall()
        cursor.execute(
            f"SELECT location_id, is_closed, current_capacity, max_capacity, fetched_at, id FROM location_history WHERE location_id IN ({placeholders}) AND fetched_at >= GREATEST(%s, %s) AND fetched_at < %s ORDER BY location_id, fetched_at, id",
            (*location_ids, cutover_bind, as_mysql_utc(range_start), as_mysql_utc(range_end)),
        )
        change_rows = cursor.fetchall()
        cursor.execute(
            "SELECT completed_at, observed_location_ids FROM ingestion_runs WHERE status='succeeded' AND completed_at >= %s AND completed_at <= %s ORDER BY completed_at",
            (as_mysql_utc(range_start), as_mysql_utc(range_end)),
        )
        heartbeat_rows = cursor.fetchall()
    return (
        parse_history_states(seed_rows + change_rows, timestamp_mapper=to_aware_utc),
        parse_ingestion_heartbeats(heartbeat_rows, timestamp_mapper=to_aware_utc),
    )
```

Load the earliest `succeeded` ingestion run's UTC `started_at` first and map its naive DB `DATETIME(6)` tuple to aware UTC with `to_aware_utc`. Immediately compute `cutover_bind = as_mysql_utc(cutover_started_at)` and use that naive UTC value in both history SQL binds; never bind the aware internal cutoff directly. If none exists, return no actual-hour inputs. Query the latest `location_history` state before range start per requested location only when `fetched_at >= cutover_bind`, breaking ties deterministically by `(fetched_at DESC, id DESC)`; then load all events where `fetched_at >= GREATEST(cutover_bind, range_start) AND fetched_at < range_end` ordered by `(location_id, fetched_at, id)`. Include `is_closed`, `current_capacity`, `max_capacity`, `fetched_at`, and `id`, mapping every post-cutover tuple timestamp through `to_aware_utc` before the service sees it. Bind range values with `as_mysql_utc` only. Never parse or bind a pre-cutover naive legacy value as UTC. Query `succeeded` `ingestion_runs` over that range for `completed_at` and `observed_location_ids`, mapping heartbeat tuples through the same UTC mapper and retaining only integer JSON IDs. The fake asserts the two naive `cutover_bind` values separately from the aware internal history states and heartbeats. A location-hour remains unavailable until both a new UTC baseline and a successful observed-ID heartbeat confirm it; do not shift, estimate, or borrow a pre-cutover state.

The route must build its entire day from `build_chicago_hour_windows(date_key)`, serialize every `HourWindow.start` as `America/Chicago` ISO 8601 including its offset, and derive category/total response items from those physical windows.

Do not derive actual-hour buckets or response length from the forecast hour array. Preserve existing facility/category selection and path, but remove the legacy hourly-event aggregation and scaled-total path. Call `calculate_actual_hour` for every category and facility total hour. Each `categories[].hours[]` and `totalHours[]` item carries its own rounded `coverageThreshold`, `actualCoverage`, and `temporalCoverage`; the client schema requires all three item fields and does not read a response-level threshold. Derive `actualPct` only for non-null `actualCount / expectedCapacity`.

- [ ] **Step 4: Run backend Phase 4 checks**

Run: `pytest tests/backend/test_actual_hours.py tests/backend/test_live_counts_api.py -q && ruff check server/forecast_api.py server/reclive/actual_hours.py server/reclive/occupancy_repository.py`

Expected: PASS; SQL contains pre-range seed and successful observed-ID heartbeat reads.

### Task 9: Qualify frontend actual merges while retaining forecasts

**Files:**
- Modify: `src/lib/types/forecast.ts`
- Modify: `src/lib/api/forecastParser.ts`
- Create: `src/lib/api/forecastParser.test.ts`

**Interfaces:**
- Consumes: actual-hour response fields and timezone-aware `hourStart` values.
- Produces: exported, runtime-validated `mergeActualHoursIntoDays(days, actualPayload): ForecastDay[]`; `fetchForecastDays` uses `Promise.allSettled`.

- [ ] **Step 1: Write failing merge tests**

```ts
import {fetchForecastDays, mergeActualHoursIntoDays} from "./forecastParser";

it("retains forecast when actual is null or below response threshold", () => {
    const merged = mergeActualHoursIntoDays(
        forecastDays,
        actualPayload({actualCount: null, actualCoverage: 0.5, temporalCoverage: 0.95, coverageThreshold: 0.75}),
    );

    expect(merged[0].totalHours?.[0].actualCount).toBeUndefined();
    expect(merged[0].totalHours?.[0].expectedCount).toBe(50);
});

it("merges only exact timezone-aware hour instants", () => {
    const mergedFallBack = mergeActualHoursIntoDays(
        forecastDays,
        actualPayload({hourStart: "2026-11-01T01:00:00-06:00", actualCount: 40, actualCoverage: 1, temporalCoverage: 1, coverageThreshold: 0.75}),
    );

    expect(mergedFallBack[0].totalHours?.find((hour) => hour.hourStart === "2026-11-01T01:00:00-06:00")?.actualCount).toBe(40);
    expect(mergedFallBack[0].totalHours?.find((hour) => hour.hourStart === "2026-11-01T01:00:00-05:00")?.actualCount).toBeUndefined();
});

it("keeps an aborted request rejected even when forecast already fulfilled", async () => {
    await expect(fetchForecastDays(1186, abortedSignal)).rejects.toMatchObject({name: "CanceledError"});
});

it("ignores malformed or nonfinite actual rows before merge", () => {
    expect(() => mergeActualHoursIntoDays(forecastDays, malformedActualPayload)).not.toThrow();
    expect(mergeActualHoursIntoDays(forecastDays, malformedActualPayload)).toEqual(forecastDays);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm run test:run -- src/lib/api/forecastParser.test.ts`

Expected: FAIL because current code merges matching strings without coverage qualification.

- [ ] **Step 3: Implement qualification and concurrent actual request behavior**

```ts
interface ActualHourPayload {
    hourStart: string;
    observedCount: number | null;
    observedCapacity: number;
    expectedCapacity: number;
    actualCoverage: number;
    temporalCoverage: number;
    coverageThreshold: number;
    actualCount: number | null;
    actualPct?: number | null;
}

const isQualifiedActualHour = (
    hour: ActualHourPayload
): hour is ActualHourPayload & {actualCount: number} => (
    Number.isFinite(hour.actualCount)
    && Number.isFinite(hour.actualCoverage)
    && Number.isFinite(hour.temporalCoverage)
    && Number.isFinite(hour.coverageThreshold)
    && hour.actualCoverage >= hour.coverageThreshold
    && hour.temporalCoverage >= hour.coverageThreshold
);
```

Export `mergeActualHoursIntoDays` and validate the actual response shape before it reaches the helper: each candidate item needs a finite `hourStart` epoch, nullable-or-finite `observedCount`, finite nonnegative `observedCapacity`/`expectedCapacity`, finite `actualCoverage`/`temporalCoverage`/`coverageThreshold`, and nullable-or-finite `actualCount`. Invalid actual payloads are ignored as an optional overlay; they never enter merge arithmetic. Use `Date.parse` on forecast and actual hour starts and merge only equal finite epochs. Leave forecast unchanged for malformed hour, null/nonfinite actual, or either coverage below threshold.

Start both forecast and actual Axios requests before awaiting and use `Promise.allSettled`. Reject when the signal is aborted at any point, even if the forecast promise fulfilled or the actual request happened to settle; otherwise fail only when forecast rejects or is malformed, and merge only a fulfilled validated actual response. Extend `ForecastHour` with optional observation/coverage fields only when a qualified actual merges.

- [ ] **Step 4: Run Phase 4 full verification and create its logical commit**

Run: `npm run test:run -- src/lib/api/forecastParser.test.ts && npm run lint && npm run build && pytest tests/backend/test_actual_hours.py -q`

Expected: PASS; a low-coverage actual cannot replace forecast and the two fall-back 01:00 hours remain distinct.

After this gate, squash the reviewed Tasks 7-9 commits into the single Phase 4 logical commit `fix: qualify time-weighted actual occupancy`. Its staged file list is exactly `server/reclive/actual_hours.py`, `server/reclive/occupancy_repository.py`, `server/forecast_api.py`, `tests/backend/conftest.py`, `tests/backend/test_actual_hours.py`, `tests/fixtures/reclive_fakes.py`, `src/lib/types/forecast.ts`, `src/lib/api/forecastParser.ts`, and `src/lib/api/forecastParser.test.ts`.

## Plan Self-Review

**Spec coverage:** Tasks 1-4 cover strict ID/count and dedicated `IsClosed` boolean rejection, dated-over-blank deterministic deduplication, allowlisted bounded error sanitization without raw-detail reflection, exception-sanitizer delegation, fake repository/clock seams, explicit naive-DB-to-aware-UTC mapping, a separately committed `running` run, no pre-success work commit, cleanup-failure precedence, preserved script imports, snapshot heartbeats, explicit evaluator protocol/factory/cleanup, separate evaluator row-list/public envelope contracts, and snapshot-only live API reads. Tasks 5-6 cover required static `fetchedAt: null` layouts, one-App-clock summary types, `latestFetchedAt` freshness, manual cache-v3 future-time rejection without Phase 6 schema-library work, exported/tested valid alert selection, replacement of `isDataLikelyStale` with occupancy status, and hero/section/Other/heat-map/alert/warning migration. Tasks 7-9 cover deterministic `(fetched_at, id)` seed selection, half-open heartbeat-bounded history, conservative closed-state capacity, physical 23/25-hour DST route windows with the missing spring 02:00 and both fall 01:00 offsets, legacy exclusion at the UTC cutoff with SQL binds that are naive while parsed states/heartbeats remain aware, per-item thresholds, exported validated merge identity, and abort-precedence concurrent forecast/actual fetches.

**Placeholder scan:** Each task has exact paths, interfaces, concrete failing tests, expected RED result, implementation details, GREEN command, and phase logical commit. No deferred implementation marker appears.

**Type consistency:** `NormalizedLiveRow.source_updated_at` persists provenance; `Location.fetchedAt` drives frontend freshness; `ActualHourSummary.actual_count` serializes as `actualCount`; frontend checks both returned coverage values against returned `coverageThreshold`.
