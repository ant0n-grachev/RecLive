# RecLive Security and Data Trust Design

## Status

This design implements the approved security, reliability, testing, architecture, PWA, and accessibility remediation for RecLive. The repository-history rewrite described in Phase 0 was completed before this document was committed. The implementation branch is `hardening/reclive-security-data-trust`.

## Goals

- Remove the two historical credential values from every published branch and tag without exposing either value.
- Prevent credentials and private upstream URLs from entering frontend bundles, logs, tests, documentation, or future Git history.
- Separate current occupancy state from historical state changes and make ingestion health observable.
- Represent missing, stale, partial, closed, and unknown occupancy honestly and consistently.
- Calculate historical actual-hour occupancy from time-weighted observations without presenting extrapolation as fact.
- Make push-alert creation, management, expiry, rate limiting, and one-time dispatch safe across multiple workers.
- Centralize frontend API configuration, validation, retry behavior, cancellation, and refresh scheduling.
- Preserve valid facility schedules across partial scrape failures.
- Replace the handwritten PWA cache with a build-aware worker while preserving installation and push behavior.
- Add automated frontend, backend, migration, PWA, route, accessibility, and security verification.
- Refactor oversized modules only after their behavior is protected by tests.

## Non-goals and constraints

- Do not rotate, revoke, regenerate, print, log, or otherwise reveal either historical credential.
- Do not change current production database credentials or contact external providers.
- Do not claim that a rewrite removes copies from forks, caches, backup bundles, or old clones.
- Do not merge the feature branch automatically or deploy the application.
- Do not perform an unrelated visual redesign.
- Preserve facility IDs `1186` and `1656`, routes `/nick` and `/bakke`, existing product identity, forecasting, push notifications, and install behavior.
- Do not run full XGBoost training in ordinary CI or rewrite the forecasting algorithm for style.
- Preserve executable compatibility entry points for `server/gym_fetch.py`, `server/forecast_api.py`, `server/forecast_job.py`, and `server/facility_hours_fetch.py`.

## Selected approach

Use staged, contract-first vertical slices. Each behavior change begins with a failing focused test, adds the smallest complete implementation, and ends with relevant package checks. Additive migrations and compatibility adapters allow backend and frontend changes to land together without silently changing public semantics. Refactoring occurs only after phases 1 through 8 establish regression protection.

A big-bang rewrite was rejected because the current application has almost no tests and several large, coupled modules. A minimal patch-in-place approach was rejected because it would retain duplicated occupancy calculations, history-table scans, race-prone push workflows, and configuration drift.

## Phase 0 history rewrite outcome

- The pre-rewrite worktree was clean and freshly matched the advertised remote `main`.
- A complete mode-0600 backup bundle exists outside the repository at `../RecLive-before-history-rewrite.bundle`.
- A fresh sibling mirror was rewritten with `git filter-repo` using exact values extracted internally from the historical commit.
- The exact-value scanner checked every blob reachable from every branch and tag and reported zero matches.
- The rewritten `main` tree ID equals the original current tree ID, proving current application source did not change.
- Harmless placeholders remained present.
- Gitleaks scanned the complete rewritten history without findings; `git fsck --full` passed before and after garbage collection.
- Rewritten branches and tags were force-pushed, advertised remote refs were verified, and a fresh clone proved the old commit unreachable.
- The temporary extractor and replacement file were deleted. The superseded normal clone was deleted after its ignored `.env` was transferred without reading it into the fresh clone.
- The backup bundle intentionally retains recoverable pre-rewrite history and must remain private and untracked.

## System architecture

The target system has five explicit boundaries:

1. **Acquisition:** `gym_fetch` validates the upstream response and records a sanitized ingestion run.
2. **Persistence:** repositories transact against snapshots, change history, ingestion metadata, alert rules, and rate-limit counters.
3. **API:** FastAPI routes expose validated, sanitized response contracts and health state.
4. **Frontend data layer:** one client builds URLs, applies timeout/retry/cancellation rules, and validates every untrusted payload with Zod.
5. **Presentation:** pure occupancy and forecast helpers produce display-ready states consumed consistently by dashboard components.

Database timestamps are stored in UTC with microsecond precision and serialized as timezone-aware ISO 8601 values. Chicago time remains the business timezone for schedules, day grouping, and forecast display.

## Phase 1: test, migration, and CI foundation

### Frontend test system

Add Vitest, React Testing Library, `@testing-library/jest-dom`, MSW, Playwright, `axe-core`, and `vitest-axe`. Use `jsdom` for component/unit tests and Chromium for the smoke suite. Add these scripts:

- `test`: watch-mode Vitest
- `test:run`: deterministic one-shot Vitest
- `test:coverage`: Vitest coverage
- `test:e2e`: Playwright smoke/accessibility suite

Remove `@types/axios` because Axios supplies its own types. Establish shared render helpers, MSW lifecycle setup, localStorage cleanup, deterministic clocks, and service-worker test seams.

### Backend test system

Keep runtime packages in `server/requirements.txt` and pin pytest, pytest-cov, Ruff, HTTPX, time-control utilities, and database test helpers in `server/requirements-dev.txt`. Unit tests import pure helpers without starting XGBoost training. FastAPI tests use dependency-injected repositories and deterministic fixtures.

### Migration runner

Create `server/migrate.py`, `server/reclive/migrations.py`, and ordered MySQL SQL files under `server/migrations/`.

The runner:

- opens a database connection using the same validated private settings as the API;
- acquires a bounded MySQL advisory lock;
- bootstraps `schema_migrations` if absent;
- reads migration files in numeric order;
- computes and records a SHA-256 checksum for each applied migration;
- refuses to continue if an applied filename's checksum changes;
- executes each unapplied file and records it only after all statements succeed;
- uses transactions for DML and acknowledges MySQL DDL auto-commit boundaries;
- always releases the advisory lock and closes the connection.

SQL uses `CREATE TABLE IF NOT EXISTS` where supported. Conditional additions and indexes query `information_schema` and execute dynamic DDL only when absent; no migration assumes `CREATE INDEX IF NOT EXISTS` support. Tests run migrations on a clean MySQL 8.4 database and then run them again unchanged.

Initial ordered migrations are:

- `0001_core_history.sql`: `location_history` compatibility schema and required history indexes.
- `0002_snapshot_and_ingestion.sql`: `location_snapshot`, `ingestion_runs`, and snapshot/run indexes.
- `0003_push_rule_lifecycle.sql`: push-rule identity, expiry, claim, status, and pending indexes.
- `0004_rate_limits.sql`: durable hashed-subject rate-limit counters and cleanup index.

### CI and dependency automation

GitHub Actions runs frontend lint, build/type-check, unit tests, and one Playwright route smoke suite. Backend jobs run Ruff, pytest, and migration tests against a MySQL 8.4 service. Security jobs run Gitleaks with full history and GitHub dependency review where the repository plan supports it. Dependabot tracks npm, pip, and GitHub Actions weekly. Logs never dump environment dictionaries, request bodies, subscription endpoints, or generated secrets.

## Phase 2: snapshot and ingestion architecture

### Tables

`location_snapshot` contains one row per upstream location:

- `location_id` primary key
- `is_closed`
- `current_capacity`
- `max_capacity`
- `source_updated_at`
- `fetched_at`
- `created_at`
- `updated_at`

`ingestion_runs` contains:

- run ID, start and completion timestamps, and status;
- received, valid, history-inserted, and snapshot-updated counts;
- a JSON array of observed valid location IDs for historical coverage reconstruction;
- a bounded sanitized error category and message.

The observed-location array contains integer IDs only, never upstream URLs or response bodies.

### Poll flow

1. Create a running ingestion record.
2. Fetch with a 5-second connect timeout and 20-second read timeout.
3. Require a JSON list and validate each row independently.
4. Normalize IDs, booleans, nonnegative counts, configured maximum capacity, and optional source timestamps.
5. Deduplicate by `location_id`: prefer the row with the newest valid source timestamp; for equal or blank timestamps, the last valid upstream occurrence wins.
6. Begin a database transaction.
7. Lock existing snapshot rows for the valid IDs.
8. Upsert every valid row and always advance `fetched_at`.
9. Append history for a first observation or when `(is_closed, current_capacity, max_capacity, source_updated_at)` changes.
10. Mark the run successful with counts and observed IDs, then commit.

An empty or wholly invalid upstream response is a failed run and cannot clear or replace healthy snapshot rows. A failed fetch or transaction updates the run to a sanitized failure state in a separate bounded transaction when the database remains available. Sanitized stored error messages are capped at 240 characters. Logs contain event names, counts, durations, and error categories, never credential-bearing URLs or database values.

### Live-count API

`/api/live-counts` reads only `location_snapshot` and the latest successful `ingestion_runs` record. Its response is:

```json
{
  "ingestion": {
    "lastSuccessfulFetchAt": "2026-08-31T05:00:00Z",
    "ageSeconds": 42,
    "status": "healthy"
  },
  "rows": []
}
```

Rows preserve the existing external names and add `FetchedAt`. The frontend temporarily accepts the legacy array and `{data: []}` forms during rollout, but backend tests prove the route does not query `location_history`.

## Phase 3: honest occupancy summaries

Create `src/shared/occupancy/computeOccupancySummary.ts` as the only occupancy aggregation implementation.

For every expected configured location:

- A fresh confirmed closure is excluded from expected open capacity.
- A fresh open row contributes only when capacity is positive and finite, count is nonnegative and finite, and `FetchedAt` is valid and not in the future.
- A missing, invalid, future, or stale row contributes no count but its configured capacity remains in expected open capacity.
- `observedCapacity` is the capacity of valid fresh open rows.
- `expectedOpenCapacity` is configured capacity excluding only fresh confirmed closures.
- `coverage` is observed capacity divided by expected open capacity.
- `percent` uses observed count divided by observed capacity; a partial value is explicitly a percentage of observed capacity, not the entire gym.

Status rules:

- `live`: coverage at least 0.8.
- `partial`: coverage from 0.5 through less than 0.8.
- `insufficient`: positive expected capacity with coverage below 0.5; percentage hidden.
- `closed`: expected open capacity is zero because all expected locations are freshly confirmed closed.
- `unknown`: no trustworthy open/closed conclusion can be made.

The default freshness threshold is ten minutes and is centralized in configuration. Facility, section, zone, alert-option, warning, and color logic consume this helper. Missing/stale zones render neutral unknown styling; partial zones expose coverage in visible and accessible text. `facilityCache` validates its runtime schema and rejects future `cachedAt` values.

## Phase 4: time-weighted actual hours

Move actual-hour calculation into a pure backend service. For each requested location and hour:

1. Load the latest history state before the range.
2. Load every subsequent history change through the range.
3. Use `fetched_at` as canonical event time.
4. Treat state changes as a step function.
5. Use successful ingestion-run observed-location heartbeats to confirm how long a state remained observed.
6. Integrate count and capacity over known time intervals.

The response reports `observedCount`, `observedCapacity`, `expectedCapacity`, `actualCoverage`, `temporalCoverage`, and `coverageThreshold`. `actualCount` is populated only when both capacity and temporal coverage meet `ACTUAL_HOUR_MIN_COVERAGE`, whose preserved default is 0.75; otherwise it is `null`. No partial total is scaled to full capacity. Any internal estimate is named `estimatedActualCount` and is never merged into UI fields labelled actual.

Frontend merge requires a finite `actualCount`, coverage at or above the response threshold, and an exact expected hour identity after timezone-aware parsing. Forecast and actual requests run concurrently with `Promise.allSettled`, and forecast data remains visible when actual data is absent or invalid.

DST tests cover 23-hour and 25-hour Chicago days using timezone-aware boundaries rather than assuming 24 buckets.

## Phase 5: push-alert lifecycle

### Validation and privacy

Public write endpoints enforce a 16 KiB request-body limit before JSON parsing. A valid subscription requires an HTTPS endpoint no longer than 2,048 characters, a base64url-decoded 65-byte uncompressed P-256 `p256dh` key, and a 16-byte `auth` secret. Thresholds are integers from 1 through 100. Logs and errors never include a full endpoint or key.

Admin authorization uses `hmac.compare_digest`. Production startup rejects a missing admin token or a token shorter than 32 bytes when admin routes are enabled.

### Rule identity and management

Each rule stores an HMAC-SHA-256 endpoint hash, subscription material required by Web Push, facility, canonical section, threshold, creation/expiry timestamps, status, and claim metadata. A unique key on `(endpoint_hash, facility_id, section_key, threshold)` makes subscription idempotent. The server verifies the full endpoint when resolving a hash match. Production requires a dedicated endpoint-hash key of at least 32 bytes.

Default rule TTL is 24 hours and the maximum accepted TTL is seven days. Each subscription may have at most ten active rules. APIs support idempotent subscribe, list active rules for the current subscription, cancel one, and cancel all. Responses expose rule IDs and safe rule metadata, never raw endpoints or keys. The browser's localStorage remains an optional convenience cache only.

### Durable rate limiting

Public push writes atomically increment MySQL fixed-window counters. The default limit is 20 write requests per subject per ten-minute window. The subject is an HMAC of the normalized subscription endpoint; malformed requests without a usable endpoint fall back to an HMAC of the request client address. Raw client addresses are never stored. Old counters are pruned by a documented maintenance command.

### Evaluator state machine

The evaluator acquires the cross-process MySQL lock before loading rules, schedules, ingestion health, or snapshots. Under the lock it rejects expired rules, closed facilities, stale ingestion, missing sections, coverage below 0.8, stale relevant rows, and percentages above the rule threshold.

An atomic conditional update claims one pending rule before sending. Claimed rules are never reclaimed, favoring at-most-once delivery over duplicates. Successful delivery becomes `sent`; 404/410 becomes `invalid_subscription`; other send failures become terminal `failed`. Notification URLs are `/nick` or `/bakke`.

## Phase 6: API configuration and request behavior

Frontend public variables are only `VITE_API_BASE_URL` and `VITE_SITE_URL`. The upstream `LIVE_COUNTS_URL` is backend-private. `APP_ENV` distinguishes development, test, and production.

Production settings fail closed when required values are absent, equal `change_me`, contain `YOUR_ACCOUNT_API_KEY`, configure wildcard CORS, or enable admin routes without a strong token. Validation reports variable names and reasons only, never values.

One Axios-based client provides base URL handling, timeout, `AbortSignal`, normalized errors, response-schema parsing, transient retry classification, exponential backoff with jitter, and `Retry-After`. It retries only network errors, 408, 429, and 5xx responses. Invalid JSON, schema errors, and 400/401/403/404 responses are not retried. Candidate URLs are normalized and deduplicated before requests.

Default refresh schedules are:

- live counts: 90 seconds;
- forecasts: 15 minutes;
- official schedules: 4 hours.

Polling pauses while the document is hidden, live data refreshes when visibility returns, requests never overlap, and current data remains rendered during refresh. Pull-to-refresh refreshes live data only unless a separate full refresh is explicitly requested.

Zod validates live counts, forecasts, actual hours, schedules, push availability, push-rule responses, and cached localStorage payloads.

## Phase 7: facility-hours ingestion

Pin Beautiful Soup in runtime requirements and keep a real optional fallback:

```python
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
```

Saved fixtures cover both facilities, direct HTML, WordPress JSON, date ranges, weekdays, closures, maintenance notices, structural changes, and anti-bot content. Collection produces a complete candidate payload in memory. A failed facility reuses its prior valid sections, marks them stale, and records a bounded sanitized error. A valid facility can update independently. Empty or invalid facility data never overwrites last-known-good content. Only a fully schema-valid combined payload is atomically published.

Output retains existing `generatedAt`, facilities, sections, and IDs while adding source fetch time, last-success time, stale status, and safe error category. API health marks schedules stale after a configurable maximum age.

## Phase 8: PWA and accessibility

Use `vite-plugin-pwa` with Workbox `injectManifest`. The source worker retains custom push and notification-click handlers. It adds build-versioned precaching, old-cache cleanup, navigation fallback, explicit update messaging, and awaited cache writes. API requests are never cached as app assets. Same-origin images/maps and fonts use bounded runtime strategies.

Notification targets are parsed against the application origin and rejected unless the final origin matches exactly; invalid targets fall back to `/`. Update availability is surfaced to the UI and activation occurs through an explicit user action/message lifecycle.

Production debug query parameters, persisted overrides, and global debug functions are ignored unless `import.meta.env.DEV` or an explicit local-debug build flag is active. Production cannot override time, closures, or prediction visibility.

Accessibility changes include unrestricted viewport zoom, keyboard-operable heat-map zones, visible focus, accessible zone names and state, coverage text independent of color, polite live regions for refresh and alert success, semantic loading/error states, reasonable touch targets, and reduced-motion behavior. Axe covers the primary dashboard and alert dialog.

## Phase 9: behavior-preserving refactor

After phases 1 through 8 are green, move frontend responsibilities into:

- `src/features/dashboard/`
- `src/features/forecast/`
- `src/features/heatmap/`
- `src/features/alerts/`

`App.tsx` becomes orchestration. Forecast display, day controls, actual-hour qualification, floor-map configuration, SVG geometry, and alert form/list components become focused modules. React components are not declared inside other components.

Backend responsibilities move under `server/reclive/` into settings, database, repositories, API routers, ingestion, schedules, push, and forecasting modules. Existing top-level scripts delegate to these modules. Forecast extraction preserves model inputs, output JSON, saved-artifact compatibility, and algorithm behavior.

Remove or rename `precisionPct`. Publish explicit metrics: MAE in people, MAE in capacity percentage points, RMSE, prediction-interval coverage, performance against a simple baseline, and rolling holdout performance by facility.

## Phase 10: documentation, health, and operations

README documents architecture, local setup, MySQL, migrations, environment variables and privacy, ingestion, schedule fetch, forecast generation, API startup, recommended cadences, frontend/backend deployment requirements, PWA/push setup, testing, and collaborator re-clone instructions.

`SECURITY.md` documents private reporting, `.env` use, credential rotation, history-rewrite limitations, and secret-scanning expectations without containing credential material.

Sanitized health responses report API readiness, database reachability, migration state, latest successful ingestion age, forecast age, schedule age, and push readiness. They never expose hosts with credentials, keys, admin tokens, endpoints, or subscription bodies. Structured logs use an event name plus allowlisted scalar fields.

## Error handling and compatibility

- Existing valid UI data remains visible during transient refresh failures.
- Backend failures return stable error categories and appropriate HTTP status codes without exception internals.
- Legacy live payloads are accepted temporarily by the frontend, but malformed untrusted JSON is rejected.
- Forecast-only display remains functional when actual-hour overlay is unavailable.
- A partial schedule fetch preserves prior valid data per facility.
- Push rules are managed server-side even when browser localStorage is empty.
- Existing selected-facility and theme preferences remain compatible.
- Deployment entry points and public routes remain stable.

## Verification strategy

Every behavior change follows red-green-refactor. Focused tests cover all regressions named in the task, including ingestion timestamp edge cases, occupancy coverage, step-function actuals, DST, push races and invalidation, retry classification, schedule fixtures, production debug isolation, service-worker behavior, keyboard interaction, Axe, route smoke tests, and offline cached snapshots.

Before each phase commit, run the narrowest relevant tests plus lint/type checks. Before completion, run the complete required command set, migration tests against clean and already-migrated MySQL 8.4 databases, full-history Gitleaks, production PWA validation, `npm audit --omit=dev`, and `git diff --check`. Results are reported exactly; unavailable external credential-backed checks are listed as unexecuted rather than inferred.

## Commit and release boundaries

- History rewrite is published before any new commit.
- Design and implementation plans are documentation commits after the rewrite.
- Phases 1 through 8 and 10 use one logical commit each when practical.
- Phase 9 uses multiple small behavior-preserving commits because of its size.
- The feature branch is pushed after verification and is not merged or deployed automatically.
- Database migrations must run before application processes that require the new schemas.
- Collaborators must re-clone or carefully rebase onto rewritten history; old clones must not push their original refs.
