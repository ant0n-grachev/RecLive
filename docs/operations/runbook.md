# RecLive Operator Runbook

This runbook describes actions for a future explicitly authorized operator. It is not evidence that a deployment, provider check, migration, job, server, or notification test was performed. Record the exact command, commit SHA, timestamp, exit status, and bounded result category for each action actually taken. An attempted check that fails is `failed`; a check not run because it is unavailable or unauthorized is `unexecuted` with the reason.

## Preflight

1. Confirm the intended commit and review only the expected changes:

   ```bash
   git rev-parse HEAD
   git status --short
   git diff --check
   ```

2. Run the repository checks in the documented environments. A release decision requires fresh results; an earlier workflow result is not a substitute.

   ```bash
   npm ci
   npm run lint
   npm run build
   npm run test:run
   npm run test:e2e
   ruff check server tests
   python -m pytest -q
   npm audit --omit=dev
   gitleaks git --redact --log-opts="--all"
   ```

3. Confirm through the deployment platform's secret manager or process-manager configuration that required backend variable names are bound. Never display their values or copy `.env` into a build, support artifact, command transcript, or frontend host. Only `VITE_API_BASE_URL` and `VITE_SITE_URL` may enter the public browser build. `VITE_API_BASE_URL` is an origin or deployment prefix without an `/api` suffix.
4. Treat VAPID settings as backend runtime configuration. The public VAPID key is delivered to the browser at runtime by GET `/api/push/public-key`; the private key, endpoint-hash key, admin token, and browser-generated subscription material remain backend-only and must not appear in public artifacts or logs.
5. Confirm that the recoverable database backup, migration cutover plan, process-drain plan, and rollback owner are ready before authorizing a schema change. Preserve the exact checked-in migration artifacts, including the frozen `push_rule_backfill.py` and `push_identity.py` helpers for migration `0003`.
6. Confirm the release workflow requires a deliberate promotion decision. Repository workflow configuration or an upload alone is not evidence that an external host has automatic merge or deployment disabled.

## Deploy

1. Obtain explicit release authorization for the exact commit SHA and target. Record the source branch, target environment, backup evidence, and responsible operator.
2. Build and publish the frontend using only the two public browser variables. Serve the generated `dist/` artifact with SPA fallback for `/nick` and `/bakke`; do not publish `.env`, forecast or model artifacts, database data, push material, or private backup bundles.
3. Inject backend-only settings through the backend host. Do not place backend settings in the frontend build or record their values.
4. With a verified backup and the migration-`0003` cutover safeguards in [Database migrations](database.md) in place, an explicitly authorized operator may run:

   ```bash
   python server/migrate.py
   ```

   This is not a read-only diagnostic: the migration runner can apply missing migrations. Run it once before starting application processes that need the resulting tables or columns. It must stop on missing, additional, reordered, or checksum-mismatched history; never edit `schema_migrations`, migration files, frozen helpers, or cutover-attempt rows by hand.

   The runner identifies MySQL 8.0/8.4 or MariaDB 10.11 before schema changes;
   other engines and versions are refused. MySQL keeps the original SQL and
   checksums. MariaDB executes the existing migrations with
   `utf8mb4_unicode_ci` in place of `utf8mb4_0900_ai_ci`, without editing the
   frozen SQL or Python helpers. These collations have different Unicode
   comparison semantics; MariaDB support is an explicit execution variant,
   not a claim that they are identical. Its versioned checksum includes the
   original effective checksum, translated SQL, and exact hook statement
   mapping. The frozen endpoint-hash finalization uses `ALGORITHM=COPY` on
   MariaDB while retaining `LOCK=EXCLUSIVE`; the existing cutover barrier and
   operator gate remain required. Allow space for the table copy. Attempt recovery and
   read-only migration health use that same checksum. Moving an existing
   migration ledger between engines therefore requires a separate reviewed
   data migration; never rewrite ledger checksums to bypass the mismatch.

   MariaDB migration `0003` also snapshots and hashes
   `mariadb_legacy_timestamps.py`. Preserve this helper after application just
   like the original frozen helpers. It converts the legacy `VARCHAR(64)`
   timezone-aware ISO `created_at` values to UTC `DATETIME(6)` before duplicate
   selection. Every source value is validated before conversion writes; naive
   or malformed timestamps and unexpected timestamp indexes fail closed.
   Conversion runs only behind the existing renamed-table cutover barrier,
   using a marked temporary column and an atomic exclusive-lock swap so
   interruption can be retried with the same helper and attempt checksum.
   Existing `DATETIME` values are left unchanged.
5. Start or restart the backend with the target host's process manager. This compatibility command is available for an authorized local or host-managed process:

   ```bash
   uvicorn server.forecast_api:app --host 127.0.0.1 --port 8000
   ```

6. Register only one scheduler for each required job. The repository entry points do not install or prove scheduler configuration, and the database evaluator lock does not make duplicate ingestion or forecast schedules safe.
7. Query the backend-local read-only endpoint without credential-bearing headers:

   ```bash
   curl -fsS http://127.0.0.1:8000/health
   ```

8. Verify the externally routed sanitized health response plus fresh rendered `/nick` and `/bakke` routes. Record the deployed SHA, backup identifier, migration outcome, frontend artifact identifier, HTTP status, safe health categories, and route results. Do not infer deployment, provider, Web Push delivery, OS installation, or device-display success from local readiness or mocks.

## Health Interpretation

`GET /health` is an unauthenticated, read-only report. Its top-level fields are exactly `status`, `checkedAt`, and `components`. A complete report has exactly `api`, `database`, `migrations`, `ingestion`, `forecast`, `schedules`, and `push`. Each component has only `status`, `observedAt`, `ageSeconds`, and `detail`, with status `ready`, `stale`, `missing`, or `unavailable`; timestamps are UTC ISO-8601 values.

- Overall `ready` requires all seven exact components to be `ready`. Any other safely generated complete report is `degraded` and returns HTTP 200 so operators can inspect its safe categories. Failure to construct a safe report returns HTTP 503 with fixed detail `health_unavailable`.
- `database: unavailable` means the read-only database connection boundary failed. Other evidence remains independently reportable where available.
- Migration health compares the ledger's exact filenames and effective checksums with the checked-in contiguous migration sequence. Migration `0003` uses a domain-separated effective checksum that includes its SQL plus the exact bytes of `push_rule_backfill.py` and `push_identity.py`; row counts or raw SQL hashes alone are insufficient. A readable but unequal ledger is `stale`. Malformed rows, a query failure, or a snapshot failure are `unavailable`. Diagnose with the health result first; the migration runner is mutating, not a checksum-inspection command.
- `ingestion` uses the latest successful ingestion timestamp and `INGESTION_STALE_AFTER_SECONDS`, normally 600 seconds. Missing evidence is not fresh occupancy, and partial or stale coverage must not be represented as complete or invented as zero.
- `forecast` uses its validated artifact timestamp and `FORECAST_STALE_AFTER_SECONDS`, normally 21600 seconds. Missing, unreadable, oversized, malformed, duplicate-key, naive-time, or future-dated evidence must not become ready.
- `schedules` uses the validated status of each supported facility and the oldest `lastSuccessfulAt` among valid source records. `generatedAt` alone does not prove schedule freshness. A stale source remains stale even when the combined artifact was just written. `SCHEDULE_STALE_AFTER_SECONDS`, normally 21600 seconds, is canonical; `SCHEDULE_MAX_AGE_SECONDS` is used only as a backward-compatible alias when the canonical name is absent.
- `push: ready` confirms local prerequisites, not provider acceptance or delivery, end-to-end user receipt, OS installation, or device display. The separate `/health/push` route remains admin-protected.
- `observedAt`, `ageSeconds`, and fixed `detail` categories are bounded diagnostic evidence. A future timestamp is unavailable, not fresh. Never add database identifiers, configuration values, URLs, keys, tokens, endpoints, bodies, exception text, traces, or private logs to an incident record.

## Incident Response

1. Record the deployed SHA, UTC detection time, affected route, HTTP status, component, fixed status/detail category, and whether the evidence came from local or external routing. Preserve safe logs and the last-known-good application artifact without copying private data.
2. For stale or missing ingestion, forecast, or schedule evidence, verify scheduler ownership and the last bounded job event. After explicit authorization, run only the corresponding entry point through the approved scheduler or process manager, then re-check `/health`:

   ```bash
   python server/gym_fetch.py
   python server/forecast_job.py
   python server/facility_hours_fetch.py
   ```

   These commands can contact upstream services, write artifacts or database state, and train forecasts; do not run them as a documentation or read-only health check. A nonzero result is failed, not unexecuted, and must retain only the fixed event/category and bounded counts.
3. For `migrations: stale`, stop rollout. Confirm the intended release contains the expected contiguous migration artifacts and follow the coordinated backup, legacy-process drain, fixed hash-key, and recovery procedure in [Database migrations](database.md). Only an explicitly authorized migration application may invoke `python server/migrate.py`; never repair the ledger directly. For `migrations: unavailable`, first resolve the database/query, malformed-ledger, or local snapshot-read boundary without relabeling it as drift.
4. For `database: unavailable`, inspect host-side service status and sanitized application events. Do not copy connection settings, SQL values, records, or raw errors into incident notes.
5. For `push: unavailable`, validate local configuration bindings and database prerequisites by name only. Provider acceptance and delivery require separately authorized, credential-backed evidence; local readiness does not prove them.
6. For suspected secret exposure, follow `SECURITY.md`: report privately, contain access through the owning provider or secret manager, perform credential rotation out of band, and verify without printing the old or new value. Do not assume a history rewrite removed external copies.
7. If application rollback is authorized, promote the previously verified application artifact through the deployment platform. Do not destructively rewrite a shared checkout. Do not reverse database schema unless a separately reviewed, explicitly tested rollback migration and backup restoration plan authorize it; otherwise use the documented forward recovery for migration `0003`.

## Routine Maintenance

- Recommended operating cadences are live ingestion at least every 90 seconds, forecast generation every 15 minutes, facility-hours ingestion every 4 hours, and push evaluation at its configured interval. These are recommendations, not proof that a scheduler is installed or running.
- Review the read-only `/health` result after scheduler, configuration, migration, or release changes. Investigate stale, missing, unavailable, and future-timestamp evidence without converting it into a success claim.
- After explicit database-maintenance authorization, prune rate-limit rows older than two configured 600-second windows with `python server/prune_push_rate_limits.py`. Record only the bounded count or fixed failure category.
- Review CI results, redacted full-history Gitleaks output, dependency review on pull requests, Dependabot updates, `npm audit --omit=dev`, backup-restoration evidence, and migration immutability weekly. A scanner or workflow result is evidence for the checked scope, not proof of provider or deployment success.
- Keep private pre-rewrite backup bundles untracked and access-controlled. Because the history rewrite does not erase private backups, forks, caches, and old clones, new collaborators must re-clone the rewritten repository (or carefully rebase clean work) and must not republish old references.
- Maintain an evidence ledger that distinguishes passed, failed, and unexecuted checks. Include the exact SHA and bounded results, never secrets, environment values, database contents, raw upstream/provider responses, push subscriptions, private artifact paths, or exception traces.

## Evidence record

Verification does not authorize merging or deployment. Do not merge or deploy
automatically; do not deploy without explicit authorization for the exact target
and source commit SHA. This operator/workflow requirement does not establish that
external host settings were inspected or changed.

For each candidate, retain the source commit SHA, exact commands, actual result,
dependency versions, migration output, immutable artifact checksums, and fresh
health/route evidence when authorized. Record each step separately in this order:

1. Install the final frontend lock with `npm ci`. In an isolated Python environment,
   install both `server/requirements.txt` and `server/requirements-dev.txt`; record
   installed versions and `python -m pip --no-cache-dir check`.
2. Run `env -u NODE_OPTIONS node --test tests/frontend/isolation.test.mjs tests/frontend/vitest-mocker-security.test.mjs`, then
   `npm run lint`. Run the original frontend scripts through the test-only launcher:

   ```bash
   env -u NODE_OPTIONS node tests/frontend/run-isolated.cjs build
   env -u NODE_OPTIONS node tests/frontend/run-isolated.cjs test:run
   env -u NODE_OPTIONS node tests/frontend/run-isolated.cjs test:coverage
   env -u NODE_OPTIONS node tests/frontend/run-isolated.cjs test:e2e
   ```

   The launcher removes inherited public Vite inputs and executable Node preloads,
   sets both public URLs to `http://127.0.0.1:4173`, enables `RECLIVE_TEST_NO_DOTENV=1`,
   and installs the Node content-read tripwire. Exactly `1` disables Vite dotenv
   loading; other present values fail by name, while an unset switch preserves
   normal application configuration. E2E uses `CI=1` to start a fresh guarded server.
   The guard uses directory-entry and symlink metadata to protect dotenv aliases;
   it is a bounded Node tripwire, not an OS sandbox. A successful build proves
   public-value injection only; production-mode parser tests prove validation.
3. Run `ruff check server tests` and `python -m pytest -v` using only the synthetic
   `TEST_MYSQL_*` fixture configuration. Run `python -m pytest tests/backend/test_migrate.py -v`
   explicitly against local MySQL 8.4, retaining clean and already-migrated MySQL 8.4 coverage, plus
   failure-recovery results. Do not run the production migration entry point as a test.
4. Verify the six frozen migration artifacts against their approved checksums.
   Run `gitleaks git --redact --log-opts="--all"`, `npm audit --omit=dev`, a bounded
   all-package audit, and `git diff --check`. Do not paste scanner match text.
5. List every unavailable credential-backed or provider-backed verification as
   **unexecuted**, with its reason. An attempted check with a nonzero result is
   **failed**, even if a later authorized retry succeeds. Record both outcomes.
   Production deployment/health, real notification delivery/clicks, OS installation,
   and model training require their own authorized evidence; local mocks and a
   configured workflow do not establish these outcomes.

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


An attempted check that fails is failed; a check not run because it is unavailable or unauthorized is unexecuted, with its reason.
