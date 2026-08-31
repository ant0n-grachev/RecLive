# RecLive Security and Data-Trust Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Use `superpowers:test-driven-development` for every behavior change and `superpowers:verification-before-completion` before each completion claim.

**Goal:** Execute the approved RecLive hardening and refactor in dependency order, preserve the existing product and compatibility entry points, and finish with a verified branch push only—no merge or deployment.

**Architecture:** The work is split into ten sequential phases. Phase 0 (credential removal and history rewrite) is already complete. Phase 1 establishes deterministic tests, checksummed MySQL migrations, CI, and shared push identity. Phases 2-4 establish ingestion, snapshot, freshness, coverage, and actual-hour truth. Phase 5 hardens push. Phase 6 centralizes configuration, schemas, requests, and polling. Phase 7 hardens official schedules. Phase 8 completes the generated PWA and accessibility boundary. Phase 9 performs behavior-preserving module extraction. Phase 10 adds final health, logging, documentation, and release evidence. Each subsystem plan below contains its RED/GREEN tests, exact interfaces, file ownership, commands, and task-sized commits.

**Branch:** `hardening/reclive-security-data-trust`

**Spec:** `docs/superpowers/specs/2026-08-31-reclive-security-data-trust-design.md`

## Immutable prerequisites

- Phase 0 is complete. Do not rewrite history again, restore an old reference, print the removed credential, or stage the ignored `.env` or private backup bundle.
- Preserve facility IDs `1186` and `1656`, routes `/nick` and `/bakke`, product copy/visual identity, forecasting behavior, push and install behavior, and executable entry points `server/gym_fetch.py`, `server/forecast_api.py`, `server/forecast_job.py`, and `server/facility_hours_fetch.py`.
- Public browser configuration is exactly `VITE_API_BASE_URL` and `VITE_SITE_URL`. Upstream URLs, database settings, admin tokens, endpoint-HMAC keys, VAPID private material, and push subscription data remain backend-only.
- Never log environment mappings, credentials, raw upstream bodies, push endpoints/keys, exception traces, or secret-scanner match text.
- Applied migrations `0001` through `0004` are immutable after Phase 1 records their checksums. Later phases consume them and may not edit them.
- Do not contact credential-backed providers, merge, deploy, force-push, or change production credentials. Provider-backed checks that cannot safely run are recorded as **unexecuted**, never inferred from mocks.
- Preserve unrelated user changes. Before each task, run `git status --short`; stage only the named files and inspect the staged diff before committing.

## Ordered execution map

| Phase | Plan | Starts after | Exit gate |
| --- | --- | --- | --- |
| 1 | [Foundation](2026-08-31-reclive-foundation.md) | Phase 0 evidence remains intact | Deterministic frontend/backend/browser tests, clean/repeat MySQL 8.4 migrations, redacted full-history Gitleaks CI, dependency review |
| 2-4 | [Data trust](2026-08-31-reclive-data-trust.md) | Phase 1 migrations and fixtures are green | Validated ingestion, transactional snapshot/history/run ledger, freshness/coverage states, coverage-qualified actual hours |
| 5 | [Push alerts](2026-08-31-reclive-push-alerts.md) | Phases 2-4 freshness and official shared identity boundaries exist | Bounded parsing, strict subscription validation, durable rate limits, idempotent/owned rules, at-most-once evaluator, management UI |
| 6 | [API client](2026-08-31-reclive-api-client.md) | Phase 5 route/response contracts are fixed | Two public Vite variables, fail-closed production config, strict Zod schemas, one retry boundary, non-overlapping 90s/15m/4h polling |
| 7 | [Facility hours](2026-08-31-reclive-facility-hours.md) | Phase 6 request/schema boundary is green | Saved source fixtures, direct/WP fallback, last-known-good per facility, atomic publication, fail-closed open predicate, truthful stale UI |
| 8 | [PWA and accessibility](2026-08-31-reclive-pwa-accessibility.md) | Final API/schedule contracts are green | Local-only debug gates, inject-manifest service worker, no API runtime cache, one registration path, labelled keyboard dialogs, axe coverage |
| 9 | [Behavior-preserving refactor](2026-08-31-reclive-refactor.md) | Phases 1-8 and characterization baseline are green | Focused frontend/backend modules, thin compatible entry points, unchanged public behavior, explicit forecast metrics |
| 10 | [Operations and health](2026-08-31-reclive-operations-health.md) | Phase 9 settings/app/repository boundaries are green | Final sanitized `/health`, strict allowlisted events, preserved safe env defaults, README/SECURITY/runbook, truthful no-deploy evidence |

The table order is mandatory. Within a phase, follow that plan's task order unless the plan explicitly identifies two read-only or test-only checks as independent. A later phase may modify a file created earlier, but it must consume the earlier public contract and leave all earlier tests green.

## Phase execution protocol

For every numbered task in the active subsystem plan:

1. Read the task's files, consumed interfaces, and produced interfaces. Inspect the current file before editing because earlier tasks may have shifted line numbers.
2. Run the named RED test and verify it fails for the expected missing behavior—not because of a typo, missing unrelated dependency, stale fixture, provider call, or unavailable database.
3. Implement the smallest complete behavior described by the task. Use `apply_patch` for source edits and preserve compatibility wrappers.
4. Run the focused GREEN command, then the phase's cumulative regression command. Do not call a test green if an expected service/browser/provider was not executed.
5. Review `git diff --check`, the unstaged diff, and the staged diff. Confirm no `.env`, database dump, generated model, push identity, test recording, scanner report, or private bundle is staged.
6. Do not commit at ordinary task checkpoints. Task-level `git add`/`git commit` snippets in subsystem plans describe review scope and candidate messages only; this master policy supersedes them. Keep the phase diff reviewable, then make one commit at the phase exit gate shown below. Phase 9 is the sole exception because the brief explicitly requires small behavior-preserving refactor commits.
7. If the same blocker repeats three times, stop changing adjacent code, preserve exact safe error categories/output, and request direction; do not broaden scope.

Use these commit boundaries so the branch remains approximately one logical commit per requested phase:

| Phase | Commit boundary | Commit message |
| --- | --- | --- |
| 1 | Foundation Task 9 complete | `test: establish migrations and CI foundation` |
| 2 | Data-trust Task 4 complete | `fix: make live ingestion snapshot-backed` |
| 3 | Data-trust Task 6 complete | `fix: represent occupancy freshness and coverage` |
| 4 | Data-trust Task 9 complete | `fix: qualify time-weighted actual occupancy` |
| 5 | Push Task 8 complete | `feat(push): complete secure alert lifecycle` |
| 6 | API-client Task 5 complete | `feat: unify API validation and refresh behavior` |
| 7 | Facility-hours Task 7 complete | `feat(schedules): harden official hours ingestion` |
| 8 | PWA/accessibility Task 5 complete | `feat: harden PWA and dashboard accessibility` |
| 9 | Each refactor task after its focused and cumulative gates | Use the small behavior-preserving messages in the refactor plan |
| 10 | Operations Task 7 complete | `docs: complete operational health and safety` |

At every phase boundary, stage the full phase-owned file set, inspect `git diff --cached --check` and `git diff --cached`, commit once with the table message, and record the resulting SHA. Do not squash Phase 0 or rewrite published history again.

## Cross-phase contracts that must remain identical

- **Timestamps:** database `DATETIME(6)` values are UTC at the bind boundary; public timestamps are offset-aware ISO-8601. Chicago time is used only for schedule/forecast business logic and presentation.
- **Freshness:** live data defaults to stale after 600 seconds. Forecast and schedule artifacts default to stale after 21,600 seconds. `SCHEDULE_STALE_AFTER_SECONDS` is canonical; `SCHEDULE_MAX_AGE_SECONDS` is legacy fallback only.
- **Coverage:** live is at least 80%, partial is 50% to below 80%, insufficient is below 50%; a facility is closed only when fresh coverage confirms every expected open location is closed. Actual hours require both returned coverage measures to meet each hour's returned `coverageThreshold`, default `0.75`; no scaling fabricates whole-facility counts.
- **Push:** request body maximum 16 KiB; HTTPS endpoint maximum 2,048 bytes; decoded P-256 key exactly 65 bytes; auth key exactly 16 bytes; threshold 1-100; default TTL 24 hours, maximum seven days; maximum ten active rules per endpoint; write limit 20 per ten-minute fixed window. Terminal claims are never reclaimed.
- **Identity:** `endpoint_hash(endpoint)` and `rate_limit_subject_hash(kind, subject)` load one validated `PUSH_ENDPOINT_HASH_KEY` internally and use distinct HMAC domains. Full canonical endpoint comparison follows every HMAC candidate match.
- **Polling:** live 90 seconds, forecast 15 minutes, schedule four hours; hidden tabs pause and then perform one immediate, non-overlapping refresh when visible.
- **PWA:** the generated worker precaches the app shell and safe static assets, runtime-caches only same-origin non-API images, and never caches `/api/*` or upstream live/forecast responses.
- **Health:** final public body contains only `status`, `checkedAt`, and components `api`, `database`, `migrations`, `ingestion`, `forecast`, `schedules`, and `push`. No path, host, environment value, token, endpoint, or exception text is serializable.

## Cumulative checkpoints

After Phases 1, 4, 6, 8, 9, and 10, run the relevant complete suites rather than only focused tests:

```bash
npm run lint
npm run build
npm run test:run
npm run test:e2e
ruff check server tests
python -m pytest -q
git diff --check
```

Run MySQL migration tests against MySQL 8.4 on both a clean database and an unchanged second pass. Run `gitleaks git --redact --log-opts="--all"` at Phase 1, after any security-sensitive migration/push work, and at the final gate. Run `npm audit --omit=dev` at the final gate. If Chromium, MySQL 8.4, Gitleaks, or any provider-backed check is unavailable, run every independent check and record only the unavailable command as unexecuted with its reason.

## Final branch-only handoff

After Phase 10 is green:

1. Confirm `git status --short` contains no unrelated or sensitive file and `git diff --check` exits zero.
2. Record the branch name, final source SHA, Phase 0 history-rewrite limitation, migration clean/repeat result, frontend/backend/browser results, redacted full-history Gitleaks result, dependency/audit result, and each unexecuted external verification.
3. Push only `hardening/reclive-security-data-trust` to `origin` with a normal non-force push.
4. Verify `git ls-remote --heads origin hardening/reclive-security-data-trust` equals the local SHA.
5. Do not create or merge a pull request and do not deploy. Return the exact remote branch/SHA and concise evidence summary to the user.

## Execution handoff

The plan set is complete when this master plan and all eight subsystem plans pass the plan-only consistency checks and are committed. Source implementation begins only after the user chooses one of the writing-plan skill's supported execution modes:

1. **Subagent-driven in this task (recommended):** dispatch one bounded task at a time, review each task at both spec and code-quality checkpoints, and keep the branch/evidence log in this task.
2. **Inline in this task:** execute the same ordered tasks serially without implementation subagents, retaining every RED/GREEN and checkpoint gate.
