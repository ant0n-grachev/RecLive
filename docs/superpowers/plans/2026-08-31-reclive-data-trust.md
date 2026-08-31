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
- Use a centralized ten-minute frontend freshness threshold. Coverage is `live` at >= 0.8, `partial` at >= 0.5 and < 0.8, and `insufficient` below 0.5 with expected capacity; hide an insufficient percentage.
- Preserve `ACTUAL_HOUR_MIN_COVERAGE=0.75`. Do not scale partial observations or put a low-coverage estimate in `actualCount`.
- Use focused backend tests in `tests/backend/`, shared fixtures in `tests/fixtures/`, and colocated frontend tests in `src/**/*.test.ts(x)`.

---

## File Structure

- Create: `server/reclive/ingestion.py` — normalized upstream rows, deterministic validation/deduplication, bounded failure sanitization, and poll orchestration.
- Create: `server/reclive/occupancy_repository.py` — snapshot/history/run transaction operations plus current/historical reads.
- Create: `server/reclive/actual_hours.py` — pure heartbeat-bounded step-function integration.
- Modify: `server/gym_fetch.py` and `server/forecast_api.py`.
- Create: `tests/backend/test_ingestion.py`, `tests/backend/test_live_counts_api.py`, and `tests/backend/test_actual_hours.py`.
- Create: `src/shared/occupancy/computeOccupancySummary.ts` and `src/shared/occupancy/computeOccupancySummary.test.ts`.
- Modify: `src/lib/types/facility.ts`, `src/lib/api/facilityParser.ts`, `src/lib/storage/facilityCache.ts`, `src/app/App.tsx`, `src/app/warningStatus.ts`, `src/facilities/OccupancyHero.tsx`, `src/facilities/SectionCommandCenter.tsx`, `src/facilities/SectionSummaryOther.tsx`, `src/facilities/FloorHeatMapCard.tsx`, and `src/facilities/CrowdAlertSubscriptionCard.tsx`.
- Create: `src/lib/api/facilityParser.test.ts`, `src/lib/storage/facilityCache.test.ts`, and `src/app/warningStatus.test.ts`.
- Modify: `src/lib/types/forecast.ts` and `src/lib/api/forecastParser.ts`; create `src/lib/api/forecastParser.test.ts`.

### Task 1: Define normalized live-row validation and deterministic duplicate selection

**Files:**
- Create: `server/reclive/ingestion.py`
- Create: `tests/backend/test_ingestion.py`

**Interfaces:**
- Consumes: decoded upstream `object` and `Mapping[int, int]` configured capacities.
- Produces: `NormalizedLiveRow`, `ValidationResult`, `validate_and_deduplicate_rows(payload, capacities)`, and `sanitize_ingestion_error(category, detail)`.

- [ ] **Step 1: Write the failing validation tests**

```python
from server.reclive.ingestion import validate_and_deduplicate_rows


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


def test_invalid_rows_do_not_discard_another_valid_row() -> None:
    result = validate_and_deduplicate_rows(
        [None, {"LocationId": "bad", "LastCount": 2}, {"LocationId": 5761, "IsClosed": 0, "LastCount": 2}],
        {5761: 100},
    )

    assert result.invalid_count == 2
    assert [row.location_id for row in result.rows] == [5761]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/backend/test_ingestion.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'server.reclive.ingestion'`.

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

Require a list payload; validate every row independently; accept only configured integer IDs, booleans, nonnegative integer counts, positive configured capacity, and optional timezone-aware source timestamp. Blank timestamp becomes `None`; malformed timestamp makes only its row invalid. Sort retained rows by `location_id`. For each ID retain newest valid source timestamp, or the last valid row on equal or blank timestamps. Implement `sanitize_ingestion_error(category: str, detail: object) -> tuple[str, str]` using categories `network`, `http`, `payload_not_list`, `validation`, `database`, and `transaction`; normalize whitespace and cap message length at 240.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/backend/test_ingestion.py -q`

Expected: PASS with all three validation tests green.

### Task 2: Persist snapshots, state-change history, and ingestion runs atomically

**Files:**
- Read: `server/migrations/0002_snapshot_and_ingestion.sql`
- Create: `server/reclive/occupancy_repository.py`
- Modify: `tests/backend/test_ingestion.py`

**Interfaces:**
- Consumes: `Sequence[NormalizedLiveRow]`, UTC `datetime`, and a PyMySQL connection with `autocommit=False`.
- Produces: `IngestionWriteCounts`, `SnapshotRepository.start_run`, `complete_success`, `complete_failure`, and `persist_successful_poll`.

- [ ] **Step 1: Write failing repository behavior tests**

```python
from datetime import datetime, timezone

from server.reclive.ingestion import NormalizedLiveRow
from server.reclive.occupancy_repository import SnapshotRepository


def test_unchanged_state_advances_snapshot_heartbeat_without_history(fake_db) -> None:
    counts = SnapshotRepository(fake_db).persist_successful_poll(
        7, [NormalizedLiveRow(5761, False, 47, 100, None)],
        datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc),
    )

    assert counts.snapshot_updated == 1
    assert counts.history_inserted == 0
    assert fake_db.snapshot_updates[0]["fetched_at"] == "2026-08-31T12:00:00+00:00"


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
    assert fake_db.history_inserts[0]["fetched_at"] == "2026-11-01T07:00:00+00:00"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/backend/test_ingestion.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'server.reclive.occupancy_repository'`.

- [ ] **Step 3: Write the migration and transactional repository implementation**

```python
@dataclass(frozen=True)
class IngestionWriteCounts:
    history_inserted: int
    snapshot_updated: int


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

Implement `lock_snapshots` with a dynamic validated-ID placeholder query, `upsert_snapshot` with an explicit `INSERT INTO location_snapshot(location_id, is_closed, current_capacity, max_capacity, source_updated_at, fetched_at, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE is_closed=VALUES(is_closed), current_capacity=VALUES(current_capacity), max_capacity=VALUES(max_capacity), source_updated_at=VALUES(source_updated_at), fetched_at=VALUES(fetched_at), updated_at=VALUES(updated_at)` statement, and `insert_history` with identical provenance values in `source_updated_at` and legacy `last_updated`. Compare exactly `(is_closed, current_capacity, max_capacity, source_updated_at)`. While the first new run is still `running`, detect that no earlier `succeeded` run exists and insert one UTC history baseline for every observed location even when its snapshot fingerprint is unchanged; only then mark that run `succeeded` and store sorted integer observed IDs before transaction commit. This run's UTC `started_at` is the durable trust cutover. Do not update, reinterpret, or timezone-shift any pre-existing history value. Convert aware UTC input to naive UTC only at the MySQL `DATETIME(6)` bind boundary.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/backend/test_ingestion.py -q`

Expected: PASS; the first successful new poll inserts one UTC baseline per observed location without changing legacy bytes, later unchanged state advances `fetched_at` without history, and a later count change inserts exactly one event.

### Task 3: Delegate the durable poll lifecycle from the preserved script entry point

**Files:**
- Modify: `server/reclive/ingestion.py`
- Modify: `server/gym_fetch.py`
- Modify: `tests/backend/test_ingestion.py`

**Interfaces:**
- Consumes: `Callable[[], object]` payload loader, `Callable[[], Any]` connector, capacities, and UTC clock.
- Produces: `run_ingestion(fetch_payload, connect, capacities, now) -> IngestionRunResult`; `gym_fetch.main() -> int` remains executable unchanged.

- [ ] **Step 1: Write failing poll lifecycle tests**

```python
from server.reclive.ingestion import run_ingestion


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


def run_ingestion(
    fetch_payload: Callable[[], object], connect: Callable[[], Any],
    capacities: Mapping[int, int], now: Callable[[], datetime],
) -> IngestionRunResult:
    started_at = now()
    work_connection = connect()
    repository = SnapshotRepository(work_connection)
    run_id = repository.start_run(started_at)
    work_connection.commit()
    try:
        validated = validate_and_deduplicate_rows(fetch_payload(), capacities)
        if not validated.rows:
            raise IngestionValidationError("validation", "No valid live rows were received")
        counts = repository.persist_successful_poll(run_id, validated.rows, now())
        repository.complete_success(run_id, now(), validated.received_count, len(validated.rows), counts, [row.location_id for row in validated.rows])
        work_connection.commit()
        return IngestionRunResult("succeeded", validated.received_count, len(validated.rows), counts.history_inserted, counts.snapshot_updated, None)
    except Exception as exc:
        work_connection.rollback()
        category, message = sanitize_ingestion_exception(exc)
        failure_connection = connect()
        try:
            SnapshotRepository(failure_connection).complete_failure(run_id, now(), category, message)
            failure_connection.commit()
        finally:
            failure_connection.close()
        return IngestionRunResult("failed", 0, 0, 0, 0, category)
    finally:
        work_connection.close()
```

Make `fetch_live()` use `requests.get(LIVE_COUNTS_URL, timeout=(5, 20))`, raise for HTTP failure, decode JSON once, and return `object`. Create and commit the `running` run before fetching so a later work rollback cannot erase it. Empty valid input completes as `validation` without touching snapshot rows. After a run exists, rollback failed work and record a sanitized failure through a fresh transaction. If a database connection cannot be opened, return failure without claiming a run was recorded. Replace `print(chicago_now_str(), "ERROR:", e)` with an allowlisted event line containing status, counts, duration milliseconds, and category only.

- [ ] **Step 4: Run focused verification**

Run: `pytest tests/backend/test_ingestion.py -q && ruff check server/gym_fetch.py server/reclive/ingestion.py server/reclive/occupancy_repository.py`

Expected: PASS; assertions and output contain no raw response, exception, URL, or credential data.

### Task 4: Return the snapshot-backed live-count envelope

**Files:**
- Modify: `server/reclive/occupancy_repository.py`
- Modify: `server/forecast_api.py`
- Create: `tests/backend/test_live_counts_api.py`

**Interfaces:**
- Consumes: `SnapshotRepository.fetch_live_snapshot(now: datetime) -> LiveSnapshotRead`.
- Produces: unchanged `GET /api/live-counts` returning `{ingestion, rows}` with `FetchedAt` on each row.

- [ ] **Step 1: Write failing route contract tests**

```python
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
```

Read `location_snapshot` ordered by `location_id` and separately the newest `ingestion_runs` row with `status = 'succeeded'`. Serialize aware UTC via `isoformat()`; calculate nonnegative `ageSeconds`; return `healthy` through ten minutes, `stale` afterwards, and `unavailable` with null success time when no run succeeded. Return 503 only for unavailable database or empty snapshot. Preserve the route and inject the repository via the FastAPI test dependency seam.

- [ ] **Step 4: Run Phase 2 checks and create its logical commit**

Run: `pytest tests/backend/test_ingestion.py tests/backend/test_live_counts_api.py -q && ruff check server/gym_fetch.py server/forecast_api.py server/reclive`

Expected: PASS; no live-count SQL uses `location_history`.

```bash
git add server/reclive/ingestion.py server/reclive/occupancy_repository.py server/gym_fetch.py server/forecast_api.py tests/backend/test_ingestion.py tests/backend/test_live_counts_api.py
git commit -m "feat: trust occupancy ingestion snapshots"
```

### Task 5: Establish the sole shared occupancy-summary contract

**Files:**
- Create: `src/shared/occupancy/computeOccupancySummary.ts`
- Create: `src/shared/occupancy/computeOccupancySummary.test.ts`
- Modify: `src/lib/types/facility.ts`

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

Add `fetchedAt: string | null` to `Location`. A row is fresh only if its parsed timestamp is finite, no later than `nowMs`, and within `freshnessMs`. Fresh closed rows exclude configured positive capacity from expected open capacity. Missing, stale, invalid, or future rows retain expected capacity but add neither count nor observed capacity. Valid fresh open rows require finite nonnegative count and positive finite capacity. Return `unknown` where no location establishes a trustworthy fresh-open or fresh-closed conclusion; return `closed` only when a nonempty expected set is all fresh confirmed closures.

- [ ] **Step 4: Run the test and type check to verify it passes**

Run: `npm run test:run -- src/shared/occupancy/computeOccupancySummary.test.ts && npm run build`

Expected: PASS; no null count is coerced to zero.

### Task 6: Parse FetchedAt, reject future cache values, and migrate all occupancy consumers

**Files:**
- Modify: `src/lib/api/facilityParser.ts`
- Modify: `src/lib/storage/facilityCache.ts`
- Modify: `src/app/App.tsx` and `src/app/warningStatus.ts`
- Modify: `src/facilities/OccupancyHero.tsx`, `src/facilities/SectionCommandCenter.tsx`, `src/facilities/SectionSummaryOther.tsx`, `src/facilities/FloorHeatMapCard.tsx`, and `src/facilities/CrowdAlertSubscriptionCard.tsx`
- Create: `src/lib/api/facilityParser.test.ts`, `src/lib/storage/facilityCache.test.ts`, and `src/app/warningStatus.test.ts`

**Interfaces:**
- Consumes: canonical/legacy live payloads and Task 5 `OccupancySummary`.
- Produces: `Location.fetchedAt` and summary-driven component props, not local total/max/percent calculations.

- [ ] **Step 1: Write failing parser/cache/consumer tests**

```ts
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm run test:run -- src/lib/api/facilityParser.test.ts src/lib/storage/facilityCache.test.ts src/app/warningStatus.test.ts`

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

Parse `rows` first, then legacy array and `data`; legacy rows receive `fetchedAt: null`. Bump `CACHE_VERSION` from `2` to `3` and require `entry.version === 3` plus `entry.cachedAt <= now` in `isFreshEntry`; the Phase 6 API-client `facilityCacheSchema` must require literal `version: z.literal(3)` so old cached entries fail closed.

In `App.tsx`, calculate one facility summary, one summary per configured section, one Other-space summary, and alert-option summaries. Replace the current `AlertSectionOption` fields `percent`, `total`, and `max` with its `summary` field; change the card’s threshold upper-bound calculation to use `summary.percent` only when status is `live` or `partial`. Remove every listed consumer’s `currentCapacity ?? 0`, `maxCapacity ?? 0`, and source-timestamp freshness arithmetic. Hero uses `summary.count / summary.observedCapacity`. Partial text says `Coverage: N% of open capacity observed`; insufficient/unknown says `Live occupancy unavailable` with no determinate bar. Individual rows use a one-location summary. Add `occupancyStatus: OccupancySummary["status"]` to `WarningResolverInput` and `partial_live` text that preserves forecasts and identifies observed-capacity coverage.

Each heat-map zone derives a summary. Keep fresh confirmed closures distinct; color only live/partial zones; render unknown/insufficient zones neutral. Popovers include `Coverage: N%` for partial and `Live occupancy unavailable` otherwise. Alert options accept `summary: OccupancySummary`, disable unknown/insufficient choices, and label partial choices with coverage.

- [ ] **Step 4: Run Phase 3 checks and create its logical commit**

Run: `npm run test:run -- src/shared/occupancy/computeOccupancySummary.test.ts src/lib/api/facilityParser.test.ts src/lib/storage/facilityCache.test.ts src/app/warningStatus.test.ts && npm run lint && npm run build`

Expected: PASS; stale, missing, invalid, and future rows cannot become zero-percent green.

```bash
git add src/shared/occupancy/computeOccupancySummary.ts src/shared/occupancy/computeOccupancySummary.test.ts src/lib/types/facility.ts src/lib/api/facilityParser.ts src/lib/api/facilityParser.test.ts src/lib/storage/facilityCache.ts src/lib/storage/facilityCache.test.ts src/app/App.tsx src/app/warningStatus.ts src/app/warningStatus.test.ts src/facilities/OccupancyHero.tsx src/facilities/SectionCommandCenter.tsx src/facilities/SectionSummaryOther.tsx src/facilities/FloorHeatMapCard.tsx src/facilities/CrowdAlertSubscriptionCard.tsx
git commit -m "feat: show trustworthy occupancy coverage"
```

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

from server.reclive.actual_hours import HistoryState, HourWindow, IngestionHeartbeat, build_chicago_hour_windows, calculate_actual_hour


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

Add constant-state, pre-range seed, no-history, closed-location, and below-0.75 temporal coverage cases.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/backend/test_actual_hours.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'server.reclive.actual_hours'`.

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

For every location, seed with its latest event strictly before range start and append later events ordered by `fetched_at`. `fetched_at` is the only event time. A state is known only from event time through the next successful heartbeat containing that location, capped by next state change and hour end. Within each location, average count and capacity over only its confirmed intervals; sum those per-location observed averages without filling unknown time with zero. Capacity coverage is the summed observed-location capacity divided by full expected capacity. Temporal coverage is independently the confirmed capacity-seconds divided by the observed-location capacity-seconds that would exist for the full hour. Thus one 100-capacity location observed for 57 minutes in a 200-capacity facility reports `actualCoverage=0.5` and `temporalCoverage=0.95`, rather than collapsing both dimensions to `0.475`. Return actual count only when both independent measures meet threshold. Never multiply an observed count by a capacity ratio or include unknown seconds as zero.

Build Chicago hours as timezone-aware local boundaries converted to UTC, keeping both fall-back 01:00 offsets and omitting the nonexistent spring-forward hour. Accept only timezone-aware UTC states already filtered by the repository's trust cutover; the service must never parse a naive legacy wall time, guess an offset, or place such a value into either repeated fall-back hour or across the spring-forward gap. Task 2's first-baseline test, this task's 23/25-hour DST test, and Task 8's fall-back legacy-row exclusion test jointly lock the boundary.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/backend/test_actual_hours.py -q`

Expected: PASS; 20 for thirty minutes plus 60 for thirty minutes yields 40, not an event average or extrapolation.

### Task 8: Use historical seed states and run heartbeats in the existing actual-hours API

**Files:**
- Modify: `server/reclive/occupancy_repository.py`
- Modify: `server/forecast_api.py`
- Modify: `tests/backend/test_actual_hours.py`

**Interfaces:**
- Consumes: preserved `GET /api/forecast/facilities/{facility_id}/actual-hours?date=YYYY-MM-DD` and `ACTUAL_HOUR_MIN_COVERAGE=0.75`.
- Produces: per-hour `observedCount`, `observedCapacity`, `expectedCapacity`, `actualCoverage`, `temporalCoverage`, `coverageThreshold`, and `actualCount`.

- [ ] **Step 1: Write failing route tests**

```python
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
    assert any("ingestion_runs" in query and "observed_location_ids" in query and "status = 'succeeded'" in query for query in sql_recorder.queries)


def test_actual_hours_excludes_legacy_wall_time_before_first_utc_baseline(client, sql_recorder) -> None:
    sql_recorder.first_succeeded_started_at = datetime(
        2026, 11, 1, 6, 59, tzinfo=timezone.utc
    )
    sql_recorder.history_rows = [
        (5761, False, 90, 100, "2026-11-01 01:30:00.000000"),
        (5761, False, 40, 100, datetime(2026, 11, 1, 7, 0)),
    ]

    response = client.get(
        "/api/forecast/facilities/1186/actual-hours?date=2026-11-01"
    ).json()

    assert sql_recorder.history_cutover == datetime(2026, 11, 1, 6, 59)
    assert sql_recorder.returned_history_rows == sql_recorder.history_rows[1:]
    assert response["totalHours"][0]["actualCount"] is None
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
        cutover_started_at = first_success[0]
        cursor.execute(
            f"SELECT h.location_id, h.is_closed, h.current_capacity, h.max_capacity, h.fetched_at FROM location_history h JOIN (SELECT location_id, MAX(fetched_at) AS fetched_at FROM location_history WHERE location_id IN ({placeholders}) AND fetched_at >= %s AND fetched_at < %s GROUP BY location_id) seed ON seed.location_id=h.location_id AND seed.fetched_at=h.fetched_at",
            (*location_ids, cutover_started_at, as_mysql_utc(range_start)),
        )
        seed_rows = cursor.fetchall()
        cursor.execute(
            f"SELECT location_id, is_closed, current_capacity, max_capacity, fetched_at FROM location_history WHERE location_id IN ({placeholders}) AND fetched_at >= GREATEST(%s, %s) AND fetched_at < %s ORDER BY location_id, fetched_at",
            (*location_ids, cutover_started_at, as_mysql_utc(range_start), as_mysql_utc(range_end)),
        )
        change_rows = cursor.fetchall()
        cursor.execute(
            "SELECT completed_at, observed_location_ids FROM ingestion_runs WHERE status='succeeded' AND completed_at >= %s AND completed_at <= %s ORDER BY completed_at",
            (as_mysql_utc(range_start), as_mysql_utc(range_end)),
        )
        heartbeat_rows = cursor.fetchall()
    return parse_history_states(seed_rows + change_rows), parse_ingestion_heartbeats(heartbeat_rows)
```

Load the earliest `succeeded` ingestion run's UTC `started_at` first. If none exists, return no actual-hour inputs. Query the latest `location_history` state before range start per requested location only when `fetched_at >= cutover_started_at`, then all events where `fetched_at >= GREATEST(cutover_started_at, range_start) AND fetched_at < range_end`; include `is_closed`, `current_capacity`, `max_capacity`, and `fetched_at`. Never parse or bind a pre-cutover naive legacy value as UTC. Query `succeeded` `ingestion_runs` over that range for `completed_at` and `observed_location_ids`, retaining only integer JSON IDs. A location-hour remains unavailable until both a new UTC baseline and a successful observed-ID heartbeat confirm it; do not shift, estimate, or borrow a pre-cutover state. Retain existing facility/forecast-day/category selection and route path, but remove `by_location_hour` and `adjusted_total`. Call `calculate_actual_hour` for every category and facility total hour. Each `categories[].hours[]` and `totalHours[]` item carries its own rounded `coverageThreshold`, `actualCoverage`, and `temporalCoverage`; the client schema requires all three item fields and does not read a response-level threshold. Derive `actualPct` only for non-null `actualCount / expectedCapacity`.

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
- Produces: `mergeActualHoursIntoDays(days, actualPayload): ForecastDay[]`; `fetchForecastDays` uses `Promise.allSettled`.

- [ ] **Step 1: Write failing merge tests**

```ts
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

Use `Date.parse` on forecast and actual hour starts and merge only equal finite epochs. Leave forecast unchanged for malformed hour, null/nonfinite actual, or either coverage below threshold. Start forecast and actual Axios requests before awaiting; use `Promise.allSettled`; fail only if forecast rejects or is malformed; merge only a fulfilled valid actual response. Extend `ForecastHour` with optional observation/coverage fields only when a qualified actual merges.

- [ ] **Step 4: Run Phase 4 full verification and create its logical commit**

Run: `npm run test:run -- src/lib/api/forecastParser.test.ts && npm run lint && npm run build && pytest tests/backend/test_actual_hours.py -q`

Expected: PASS; a low-coverage actual cannot replace forecast and the two fall-back 01:00 hours remain distinct.

```bash
git add server/reclive/actual_hours.py server/reclive/occupancy_repository.py server/forecast_api.py tests/backend/test_actual_hours.py src/lib/types/forecast.ts src/lib/api/forecastParser.ts src/lib/api/forecastParser.test.ts
git commit -m "fix: qualify time-weighted actual occupancy"
```

## Plan Self-Review

**Spec coverage:** Tasks 1-4 cover Phase 2 validation, deterministic duplicate selection, transaction/run recording, snapshot heartbeats, state fingerprint history, failed-ingestion preservation, and snapshot-only live API reads. Tasks 5-6 cover Phase 3 summary types, freshness/coverage states, cache future-time rejection, and hero/section/Other/heat-map/alert/warning migration. Tasks 7-9 cover Phase 4 pre-range seed state, `fetched_at` step changes, run heartbeat coverage, time weighting, unscaled partial values, DST, exact merge identity, and concurrent forecast/actual fetches.

**Placeholder scan:** Each task has exact paths, interfaces, concrete failing tests, expected RED result, implementation details, GREEN command, and phase logical commit. No deferred implementation marker appears.

**Type consistency:** `NormalizedLiveRow.source_updated_at` persists provenance; `Location.fetchedAt` drives frontend freshness; `ActualHourSummary.actual_count` serializes as `actualCount`; frontend checks both returned coverage values against returned `coverageThreshold`.
