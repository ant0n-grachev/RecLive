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

## Built by

Built by Anton and [Alex](https://github.com/alexgabrichidze).
