from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_release_docs_require_actual_evidence_and_prohibit_automatic_deploy() -> None:
    combined = "\n".join(
        (ROOT / name).read_text(encoding="utf-8")
        for name in ("README.md", "docs/operations/runbook.md")
    ).lower()
    for required in (
        "actual result", "unexecuted", "do not merge", "do not deploy",
        "gitleaks", "git diff --check", "already-migrated mysql 8.4",
        "verification does not authorize", "source commit sha",
    ):
        assert required in combined


def test_linked_docs_preserve_safe_operations_and_existing_runbooks() -> None:
    readme = "\n".join(
        (ROOT / name).read_text(encoding="utf-8")
        for name in ("README.md", "docs/architecture.md",
                     "docs/operations/database.md", "docs/operations/runbook.md")
    )
    normalized = " ".join(readme.split())

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
        "gitleaks git --redact",
        "python server/prune_push_rate_limits.py",
        "coordinated cutover",
        "effective, domain-separated checksum",
        "re-clone",
    ):
        assert required in normalized

    assert "public frontend" in normalized.lower()
    assert "backend-only" in normalized.lower()
    assert "VITE_API_BASE_URL" in normalized
    assert "VITE_SITE_URL" in normalized
    assert "LIVE_COUNTS_URL" in normalized
    assert "origin or deployment prefix" in normalized
    assert "without an `/api` suffix" in normalized
    assert "GET `/api/push/public-key`" in normalized
    assert "browser-generated runtime material" in normalized
    assert "only verifies local prerequisites" in normalized
    assert "attempted check that fails is failed" in normalized
    assert "not run because it is unavailable or unauthorized is unexecuted" in normalized
    assert "--output server/facility_hours.json" not in readme
    assert "printenv" not in readme
    assert "git reset --hard" not in readme


def test_runbook_documents_exact_health_and_sanitized_output_contracts() -> None:
    readme = (ROOT / "docs/operations/runbook.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())

    for required in (
        "`status`, `checkedAt`, and `components`",
        "`api`, `database`, `migrations`, `ingestion`, `forecast`, `schedules`, and `push`",
        "`ready`, `stale`, `missing`, or `unavailable`",
        "HTTP 200",
        "HTTP 503",
        "`health_unavailable`",
        "variable names and fixed reasons",
        "never rejected values",
        "SCHEDULE_MAX_AGE_SECONDS",
    ):
        assert required in normalized


def test_forecasting_docs_qualify_public_forecast_metrics() -> None:
    readme = (ROOT / "docs/forecasting.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())

    for required in (
        "`metrics.maePeople`",
        "`metrics.rmsePeople`",
        "`metrics.maeCapacityPercentagePoints`",
        "`metrics.predictionIntervalCoverage`",
        "`metrics.simpleBaselineMaePeople`",
        "`metricContext.observationCounts`",
        "`fixed_model_terminal_holdout`",
        "`independentBacktest` is `false`",
        "location observations",
        "not facility totals or final served/blended forecasts",
        "cleaned, schedule-adjusted bucket mean retained before ratio clipping",
        "maximum capacity encountered in loaded model history",
        "non-overlapping UTC windows",
        "canonical DB observation instant",
        "JSON `null`",
        "`rollingHoldoutByFacility` is empty",
        "weighted occupancy ratios",
    ):
        assert required in normalized


def test_env_example_uses_safe_placeholders_and_preserves_public_defaults() -> None:
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    values = {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in env_example.splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }
    preserved = {
        "GYM_DB_HOST": "localhost",
        "GYM_DB_PORT": "3306",
        "GYM_DB_USER": "root",
        "GYM_DB_NAME": "gym_data",
        "GYM_DB_TIMEZONE": "America/Chicago",
        "FORECAST_DAY_START_HOUR": "6",
        "FORECAST_DAY_END_HOUR": "23",
        "GYM_RESAMPLE_MINUTES": "15",
        "GYM_WINDOW_RESAMPLE_MINUTES": "30",
        "GYM_WEATHER_URL": "https://api.open-meteo.com/v1/forecast",
        "GYM_WEATHER_ARCHIVE_URL": "https://archive-api.open-meteo.com/v1/archive",
        "GYM_WEATHER_LAT": "43.0731",
        "GYM_WEATHER_LON": "-89.4012",
        "GYM_WEATHER_FORECAST_DAYS": "7",
        "GYM_WEATHER_HISTORY_MAX_DAYS": "180",
        "MODEL_ARTIFACT_DIR": "model_artifacts",
        "MODEL_BASENAME": "forecast_model",
        "SCHEDULE_FILTER_ENABLED": "1",
        "FACILITY_HOURS_JSON_PATH": "facility_hours.json",
        "FORECAST_OUTPUT_INCLUDE_WEATHER": "1",
        "FORECAST_OUTPUT_INCLUDE_INTERVAL_FIELDS": "1",
        "FACILITY_CAPACITIES_JSON_PATH": "shared/facility_capacities.json",
        "FORECAST_JSON_PATH": "forecast.json",
        "FACILITY_SECTION_CONFIG_PATH": "facility_sections.json",
        "FORECAST_API_HOST": "0.0.0.0",
        "FORECAST_API_PORT": "8000",
        "ACTUAL_HOUR_MIN_COVERAGE": "0.75",
        "PUSH_RULES_TABLE": "push_rules",
        "PUSH_EVALUATOR_ENABLED": "true",
        "PUSH_EVALUATOR_INTERVAL_SECONDS": "180",
        "PUSH_EVALUATOR_DB_LOCK_NAME": "reclive_push_eval",
        "PUSH_DEFAULT_NOTIFICATION_URL": "/",
        "PUSH_VAPID_SUBJECT": "mailto:alerts@example.com",
        "PUSH_ADMIN_ROUTES_ENABLED": "false",
        "PUSH_RULE_DEFAULT_TTL_SECONDS": "86400",
        "PUSH_RULE_MAX_TTL_SECONDS": "604800",
        "PUSH_MAX_ACTIVE_RULES_PER_ENDPOINT": "10",
        "PUSH_WRITE_RATE_LIMIT": "20",
        "PUSH_WRITE_RATE_WINDOW_SECONDS": "600",
    }

    assert {name: values.get(name) for name in preserved} == preserved
    assert {name for name in values if name.startswith("VITE_")} == {
        "VITE_API_BASE_URL",
        "VITE_SITE_URL",
    }
    assert values["VITE_API_BASE_URL"] == "change_me"
    assert values["VITE_SITE_URL"] == "change_me"
    assert values["FORECAST_API_ALLOW_ORIGINS"] == "http://127.0.0.1:4173"
    assert values["INGESTION_STALE_AFTER_SECONDS"] == "600"
    assert values["FORECAST_STALE_AFTER_SECONDS"] == "21600"
    assert values["SCHEDULE_STALE_AFTER_SECONDS"] == "21600"
    assert "SCHEDULE_MAX_AGE_SECONDS" not in values
    for name in (
        "LIVE_COUNTS_URL",
        "GYM_DB_PASSWORD",
        "PUSH_VAPID_PUBLIC_KEY",
        "PUSH_VAPID_PRIVATE_KEY",
        "PUSH_ENDPOINT_HASH_KEY",
        "PUSH_ADMIN_TOKEN",
    ):
        assert values[name] == "change_me"


def test_security_policy_and_runbook_cover_required_controls() -> None:
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    runbook = (ROOT / "docs/operations/runbook.md").read_text(encoding="utf-8")
    normalized = " ".join(runbook.split())

    for token in (
        "private",
        ".env",
        "credential rotation",
        "history rewrite",
        "Gitleaks",
        "do not include secrets",
    ):
        assert token.lower() in security.lower()

    for token in (
        "Preflight",
        "Deploy",
        "Health",
        "Incident",
        "Routine maintenance",
        "schema_migrations",
        "curl -fsS",
        "re-clone",
    ):
        assert token.lower() in runbook.lower()

    for required in (
        "exact filenames and effective checksums",
        "push_rule_backfill.py",
        "push_identity.py",
        "readable but unequal ledger",
        "malformed rows, a query failure, or a snapshot failure",
        "oldest `lastSuccessfulAt`",
        "`generatedAt` alone does not prove schedule freshness",
        "`api`, `database`, `migrations`, `ingestion`, `forecast`, `schedules`, and `push`",
        "HTTP 200",
        "HTTP 503",
        "`health_unavailable`",
        "GET `/api/push/public-key`",
        "local prerequisites, not provider acceptance or delivery",
        "can apply missing migrations",
        "an attempted check that fails is `failed`",
        "a check not run because it is unavailable or unauthorized is `unexecuted`",
        "private backups, forks, caches, and old clones",
    ):
        assert required.lower() in normalized.lower()

    assert "curl -fsS http://127.0.0.1:8000/health" in runbook
    assert "git reset --hard" not in runbook
    assert "printenv" not in runbook
    assert "Dependabot updates" in normalized
    assert "migration immutability weekly" in normalized
    assert "agreed maintenance interval" not in normalized
