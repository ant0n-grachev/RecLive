# RecLive

*Train smarter. Skip the crowd.*

RecLive is a live gym intelligence app for UW students.

It helps people answer one simple question before they walk over:
**"Is it worth going right now?"**

## The Idea

Campus gyms can feel random.
Sometimes they are perfect, sometimes they are packed.
RecLive gives students a fast read on current crowd levels and near-term trends so they can plan better workouts.

## What RecLive Does

- Shows real-time occupancy for Nick and Bakke
- Breaks crowd levels down by key gym areas
- Highlights daily forecast windows (low, medium, peak)
- Sends one-time alerts when occupancy drops below your threshold
- Works as a mobile-first Progressive Web App

## Why It Matters

- Less time wasted traveling to packed gyms
- Better workout consistency
- Better experience for both beginners and regulars

## Product Focus

RecLive is designed to be:

- Fast to read
- Simple to trust
- Useful in seconds

No dashboard overload. Just the info you need to decide when to go.

## Tech Stack

- Frontend: React, TypeScript, Vite, Material UI
- Backend API: FastAPI (Python)
- Data: MySQL + live occupancy feed ingestion
- Forecasting: XGBoost predictions
- Notifications: Web Push (VAPID)
- Platform: Progressive Web App (PWA)

## Architecture and configuration boundaries

The browser is a React/Vite PWA. The FastAPI service reads current snapshots,
forecasts, schedules, and push state from backend-only configuration. MySQL
retains schema migrations, current occupancy snapshots, history, ingestion
runs, alert rules, and rate-limit counters.

Only the public frontend variables `VITE_API_BASE_URL` and `VITE_SITE_URL` may
enter the browser build. `VITE_API_BASE_URL` is an origin or deployment prefix,
without an `/api` suffix; the client appends canonical `/api/...` paths. For a
local frontend-origin build, use
`VITE_API_BASE_URL=http://127.0.0.1:4173` and
`VITE_SITE_URL=http://127.0.0.1:4173`, with a same-origin proxy routing
`/api/...` to FastAPI. A separately exposed backend origin can instead be the
prefix when that is the real deployment topology.

`LIVE_COUNTS_URL`, database settings, `PUSH_VAPID_PUBLIC_KEY`, the VAPID private
key, the endpoint-hash key, and the admin token are backend-only environment
inputs and must never gain a `VITE_` prefix. The public VAPID value is delivered
at runtime by GET `/api/push/public-key` as `{ "publicKey": ... }`; it is not a
third build-time variable. Push subscriptions are browser-generated runtime
material sent to the backend. They must not be bundled, logged, or exposed by
public/readiness responses, while the VAPID private key always remains on the
backend.

## Local setup

```bash
npm ci
python -m pip install -r server/requirements.txt
python -m pip install -r server/requirements-dev.txt
cp .env.example .env
```

Use only local, non-production values in `.env`. In production, inject private
variables through the backend host's secret manager or process manager. Do not
upload `.env` to the frontend host. Production CORS configuration must contain
explicit, comma-separated HTTPS origins; wildcard CORS is rejected.

## Database migrations

Apply checked-in MySQL schema changes before starting an application process that
needs new tables or columns:

```bash
python server/migrate.py
```

The command reads the private `GYM_DB_HOST`, `GYM_DB_PORT`, `GYM_DB_USER`,
`GYM_DB_PASSWORD`, and `GYM_DB_NAME` environment variables. It records every
applied migration filename and SHA-256 checksum in `schema_migrations`. Most
migrations hash their exact SQL bytes. Migration `0003` records an effective,
domain-separated checksum over its exact SQL, push-rule backfill, and push
identity artifacts; never edit any applied migration artifact.

The `0003` push-rule contract migration is an explicit coordinated cutover. To
apply it, first take and verify a recoverable database backup, then drain every
legacy process that reads or writes the raw `push_rules.endpoint` column. Keep
the same `PUSH_ENDPOINT_HASH_KEY` available for the entire attempt and every
recovery run. Then set `PUSH_RULE_SCHEMA_CUTOVER_READY=1`, run the migration
command, and start the Phase 5 application code that consumes the hashed
push-rule schema. Do not enable the gate while a legacy process is still
running, and do not start the legacy application again after the cutover.

During a legacy conversion, `0003` atomically moves the source table behind an
internal cutover name before it snapshots or backfills any row. The public
legacy table name stays absent through validation, so stale INSERT, duplicate
update, UPDATE, and DELETE statements fail closed. If the command stops, leave
the internal table and migration-attempt records in place, keep legacy writers
drained, preserve the exact migration artifacts and hash key, and rerun the same
command; do not rename tables or edit attempt rows by hand. Only after the raw
column is removed and the final contract validates does the migration atomically
restore `push_rules`. The temporary cutover table must then be absent, and stale
endpoint-based statements fail structurally against the final schema.

Use a least-privilege MySQL account that can read and write RecLive tables but
cannot administer unrelated schemas. Apply migrations once before starting a
new application process that needs the schema. The migration runner compares
the exact filenames and effective checksums already recorded; a missing,
additional, reordered, or mismatched artifact stops the operation rather than
silently accepting drift.

## Official facility hours

Refresh the saved Nick and Bakke schedules with:

```bash
python server/facility_hours_fetch.py
```

The command validates the complete two-facility artifact before replacing the
existing file atomically. If one facility cannot be refreshed, only an earlier
valid schedule for that facility may be retained; it is marked stale, the other
valid fresh facility is still published, and the command exits nonzero. Without
a valid earlier schedule, the failed facility is published without invented
hours and the command also exits nonzero.

`SCHEDULE_STALE_AFTER_SECONDS=21600` is the canonical six-hour freshness
setting. At runtime, the legacy `SCHEDULE_MAX_AGE_SECONDS` value is used only
when the canonical setting is absent. Fetch and publication errors use fixed,
sanitized categories and never expose upstream request data, response content,
paths, or credentials.

## Runtime commands and recommended cadences

```bash
python server/gym_fetch.py
python server/facility_hours_fetch.py
python server/forecast_job.py
uvicorn server.forecast_api:app --host 127.0.0.1 --port 8000
curl -fsS http://127.0.0.1:8000/health
```

Recommended cadences are live ingestion at least every 90 seconds, forecast
generation every 15 minutes, facility-hours ingestion every 4 hours, and push
evaluation at its configured interval. These are operating recommendations,
not scheduler configuration supplied by the entry points. Run only one
scheduler for each command. The database-backed evaluator lock protects
cross-process alert dispatch; it does not make duplicate ingestion or forecast
cron entries safe.

The three freshness thresholds are positive seconds:
`INGESTION_STALE_AFTER_SECONDS=600`,
`FORECAST_STALE_AFTER_SECONDS=21600`, and
`SCHEDULE_STALE_AFTER_SECONDS=21600`. A nonpositive value is invalid in every
environment. `SCHEDULE_STALE_AFTER_SECONDS` is canonical;
`SCHEDULE_MAX_AGE_SECONDS` is a backward-compatible alias used only when the
canonical variable is absent.

## Health and operational output

`GET /health` is an unauthenticated, read-only readiness report. Its top-level
keys are exactly `status`, `checkedAt`, and `components`; component names are
exactly `api`, `database`, `migrations`, `ingestion`, `forecast`, `schedules`,
and `push`. Each component is limited to `status`, `observedAt`, `ageSeconds`,
and `detail`, with a status of `ready`, `stale`, `missing`, or `unavailable`.
All timestamps are UTC ISO-8601 values. A safely generated degraded report
returns HTTP 200. Inability to produce any safe report returns HTTP 503 with the
fixed detail `health_unavailable`. The separate `/health/push` route remains
protected.

Health migration readiness compares the migration runner's filenames and
effective checksums, including the frozen helper bytes that contribute to
migration `0003`. Source health retains independent database, artifact, and
schedule evidence: future, unreadable, malformed, stale, or missing sources do
not become ready. Ready push health only verifies local prerequisites; it does
not verify provider acceptance, notification delivery, OS installation, or
device display.

Executable operational output is allowlisted JSON with an aware UTC timestamp,
bounded event/category names, and nonnegative counts where applicable.
Configuration failures report variable names and fixed reasons, never rejected
values. Output does not include environment dictionaries, database values,
upstream bodies, subscriptions, endpoints, artifact paths, exception text, or
tracebacks.

## Forecast reporting diagnostics

The generated forecast's `modelInfo.metrics` and `modelInfo.metricContext`
describe qualified reporting diagnostics:

- `metrics.maePeople` is mean absolute error in people for location rows where
  both the cleaned observed people target and prediction are finite.
- `metrics.rmsePeople` is root mean square error in people over the same valid
  actual-plus-prediction rows.
- `metrics.maeCapacityPercentagePoints` divides absolute people error by that
  row's finite, positive per-location normalization capacity and multiplies by
  100. Its valid population can be smaller than the people-error population.
- `metrics.predictionIntervalCoverage` is a fraction from 0 to 1 for valid
  observed targets within a finite, ordered lower/upper interval; it is not a
  0-to-100 percentage field.
- `metrics.simpleBaselineMaePeople` is mean absolute error in people against a
  per-location raw last-observation persistence baseline frozen at each UTC
  window start. Observation and fetched/availability times must both strictly
  precede the window start; late, absent, invalid, or ambiguous observations
  provide no baseline for that row.
- `metricContext.observationCounts` and each window's `observationCounts` give
  separate samples for every metric; the valid populations are not assumed to
  match.

The overall and per-facility method is `fixed_model_terminal_holdout`, and
`independentBacktest` is `false`. One fitted model is evaluated across
non-overlapping UTC windows anchored at the actual terminal split, normally 24
hours with a possibly shorter final window. This is not rolling-origin
retraining. Retrospective preprocessing, full-history priors, tuning/selection,
and later calibration/champion selection prevent an independent-backtest
claim.

Rows are location observations from freshly trained-and-saved selected
facility-wide `__all__` models for facilities `1186` and `1656`; they are not
facility totals or final served/blended forecasts. The target is the cleaned,
schedule-adjusted bucket mean retained before ratio clipping. Model ratios,
predictions, and intervals are converted to people with each row's normalization
capacity: the maximum capacity encountered in loaded model history, not an
as-of historical capacity.

The `metricContext.timestampAlignment` reporting guard requires every
contributing location's canonical DB observation instant to match its preserved
model instant. Missing, unparseable, mismatched, or mixed alignment suppresses
that selected model's rows and windows instead of publishing uncertain
coverage. This reporting guard does not repair or reinterpret the preserved DB
timezone/source-time integration.

When a valid population is empty, scalar metrics are JSON `null`, its count is
zero, and `rollingHoldoutByFacility` is empty when no facility has qualified
evidence. A missing baseline can leave only the baseline metric/count
unavailable, and one qualifying facility may still contribute when the other
is suppressed. `metricContext.compatibilityAliases` maps public `valMae` and
`valRmse` to the people metrics. Legacy `byFacility`/`byModel` `valMae`,
`valRmse`, `holdoutMae`, and `holdoutRmse` remain weighted occupancy ratios;
guardrail, drift, and blend telemetry retain their algorithm-specific units.

## Push alert limits, lifecycle, and maintenance

Push rules default to 24 hours (`PUSH_RULE_DEFAULT_TTL_SECONDS=86400`) and
cannot exceed seven days (`PUSH_RULE_MAX_TTL_SECONDS=604800`). The API permits
ten active rules per push endpoint (`PUSH_MAX_ACTIVE_RULES_PER_ENDPOINT=10`)
and twenty write attempts per ten-minute window (`PUSH_WRITE_RATE_LIMIT=20`,
`PUSH_WRITE_RATE_WINDOW_SECONDS=600`). Rule management uses server-backed list
and cancel operations.

The write limit is per hashed subject: a usable normalized subscription
endpoint is the subject, and only requests without one fall back to the hashed
immediate client address. Different usable endpoints therefore have independent
counters. This is not a global traffic ceiling, and the rate-limit table stores
no raw endpoint or client address.

Pending rows can become expired or cancelled without being claimed. Normal
claimed sends terminalize as sent, failed, or invalid subscription. A crash or
ambiguous post-claim failure can intentionally leave the row permanently
claimed and never retried. The claim is committed before provider I/O; this
preserves at-most-once provider attempts but does not guarantee terminal state
or delivery. Provider acceptance and device display remain outside RecLive's
control.

Raw push endpoints, subscription keys, client subjects, and provider response
bodies are never returned or logged. Operational output is count-only. Never
dump the environment or include private database settings in logs or support
artifacts.

Prune rate-limit rows older than two 600-second windows with:

```bash
python server/prune_push_rate_limits.py
```

The command reads the same private `GYM_DB_HOST`, `GYM_DB_PORT`, `GYM_DB_USER`,
`GYM_DB_PASSWORD`, and `GYM_DB_NAME` settings as the application. On success it
prints only `pruned_push_rate_limit_windows=<count>`; failures return a fixed,
non-sensitive error without settings, SQL, exceptions, or tracebacks.

## Deployment

Build the frontend with only its two public frontend variables present:

```bash
npm ci
npm run build
```

Serve `dist/` from the frontend host with SPA fallback for `/nick` and `/bakke`.
Deploy the backend with private variables injected by the backend host. Run
`python server/migrate.py` once before starting new backend processes, then
verify both the backend-local route and the externally routed deployment return
the sanitized `/health` contract. This describes the deployment verification
to perform; it is not evidence that a deployment or external route was checked.

Do not deploy `.env`, `server/forecast.json`, model artifacts, database dumps,
push subscriptions, or private backup bundles to the frontend host. The PWA
requires HTTPS in production, a manifest, a service worker, and install icons.

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

The Gitleaks command scans all reachable history while redacting suspected
secret values from output. Do not infer provider, deployment, or notification
success from mocks or local readiness. An attempted check that fails is failed;
a check not run because it is unavailable or unauthorized is unexecuted with
its reason.

## Re-cloning after the history rewrite

The repository history was rewritten before this implementation branch.
Collaborators must re-clone the repository or carefully rebase a clean local
branch onto the rewritten remote history. Rewriting published references does
not erase copies retained in private backups, forks, caches, or old clones.

## Built by

Built by Anton and [Alex](https://github.com/alexgabrichidze).

## Release evidence

Verification does not authorize merging or deployment. Do not merge or deploy
automatically; do not deploy without explicit release authorization. This is an
operator/workflow constraint, not a claim that external host settings were inspected.
Before proposing a release, retain the source commit SHA and record the exact
command and actual result of every attempted check. Mark a failed attempt as
failed; mark a check not run because it is unavailable or unauthorized as
**unexecuted**, with its reason. Follow the ordered checklist in
[`docs/operations/runbook.md`](docs/operations/runbook.md#evidence-record), including
`git diff --check`, frontend build/tests, backend tests, migration tests on clean
and already-migrated MySQL 8.4, Gitleaks, and `npm audit --omit=dev`.
Never infer provider delivery, OS installation, or deployment success from a mock,
upload, workflow configuration, build, or deployment command alone.
