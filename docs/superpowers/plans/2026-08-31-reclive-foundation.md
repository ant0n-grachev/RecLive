# RecLive Phase 1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish RecLive's repeatable frontend/backend test systems, checksummed MySQL migrations, and security-focused GitHub automation before changing application behavior.

**Architecture:** Browser/unit tests remain separate: colocated Vitest tests exercise TypeScript and React in jsdom, while Playwright runs an isolated Chromium route smoke suite against Vite. Backend tests live under `tests/backend/` and execute the migration CLI against the MySQL 8.4 CI service; production dependencies remain in `server/requirements.txt` and development-only tools remain in `server/requirements-dev.txt`. A small migration package owns ordering, checksums, MySQL advisory locking, and transactional DML execution; each SQL file owns one incremental schema boundary.

**Tech Stack:** React 19, TypeScript 5.9, Vite 7, Vitest, React Testing Library, MSW, Playwright Chromium, axe-core, FastAPI, PyMySQL, pytest, pytest-cov, Ruff, MySQL 8.4, GitHub Actions, Gitleaks, Dependabot.

**Spec:** `docs/superpowers/specs/2026-08-31-reclive-security-data-trust-design.md`

## Global Constraints

- Scope is Phase 1 only: test foundations, the runner and migrations `0001` through `0004`, CI, Gitleaks/dependency review, and Dependabot.
- The master plan's phase-level commit policy is authoritative: task commit snippets describe staging/review scope only; make the single Phase 1 source commit after Task 9.
- Preserve facility IDs `1186` and `1656`, routes `/nick` and `/bakke`, product identity, forecasting, push notifications, install behavior, and all executable compatibility entry points.
- Do not rotate, reveal, print, log, test with, or commit historical or current credentials; logs must not dump environment dictionaries, request bodies, subscription endpoints, or generated secrets.
- Keep production Python packages in `server/requirements.txt`; pin test/lint packages only in `server/requirements-dev.txt`.
- Unit/component tests are colocated at `src/**/*.test.ts` and `src/**/*.test.tsx`; backend tests are under `tests/backend/`; backend fixtures are under `tests/fixtures/`; Playwright tests are under `tests/e2e/`.
- Test components in jsdom and route smoke/accessibility in Chromium. Do not run XGBoost training in ordinary CI.
- Migrations run in numeric filename order, record SHA-256 checksums, reject changed applied files, acquire a bounded MySQL advisory lock, release the lock and close the connection on every path, and record each migration only after its statements complete.
- MySQL DDL can auto-commit. Use explicit transactions for the runner's `schema_migrations` DML; each SQL migration must be independently rerunnable with `CREATE TABLE IF NOT EXISTS` and `information_schema`-guarded dynamic DDL rather than `CREATE INDEX IF NOT EXISTS`.
- Store database timestamps in UTC with microsecond precision. Chicago remains the business timezone outside this foundation phase.
- CI uses Node 22, Python 3.12, and a MySQL 8.4 service. It runs `npm ci`, lint, build/type-check, one-shot unit tests, one Playwright route smoke suite, Ruff, pytest, and migration tests.
- Security CI performs a full-history Gitleaks scan and GitHub dependency review on pull requests. Dependabot checks npm, pip, and GitHub Actions weekly.
- Do not merge, deploy, force-push, alter `.env`, or commit generated credentials, database dumps, model artifacts, test recordings, or the private history-rewrite bundle.

## File Map and Contract Boundaries

| File(s) | Responsibility | Interface produced |
| --- | --- | --- |
| `package.json`, `package-lock.json`, `vite.config.ts`, `tsconfig.app.json` | Deterministic frontend test commands and compiler globals | `npm run test:run`, `npm run test:coverage`, `npm run test:e2e` |
| `src/test/setup.ts`, `src/test/render.tsx`, `src/test/msw/server.ts` | Global jsdom cleanup, deterministic time/storage cleanup, MUI/router render helper, MSW lifecycle | `renderWithApp(ui, options)` and `server` |
| `src/**/*.test.ts(x)` | Colocated unit/component contract tests | Vitest-discovered test files |
| `playwright.config.ts`, `tests/e2e/route-smoke.spec.ts` | Chromium smoke coverage for `/nick` and `/bakke` with mocked API responses | `npm run test:e2e` |
| `server/requirements-dev.txt`, `pytest.ini`, `tests/backend/`, `tests/fixtures/` | Deterministic Python test/lint configuration and reusable non-sensitive fixtures | `pytest -q`, fixture modules imported as `tests.fixtures.*` |
| `server/reclive/migrations.py`, `server/migrate.py` | Ordered migration discovery, checksum verification, MySQL locking, and execution | `run_migrations(settings, migration_dir) -> list[str]`; `python server/migrate.py` |
| `server/migrations/0001_core_history.sql` | Compatible `location_history` baseline, safe `last_updated` to `source_updated_at` backfill, and history indexes | `location_history` fields/indexes used by existing scripts and later services |
| `server/migrations/0002_snapshot_and_ingestion.sql` | Current-state and ingestion-run tables | `location_snapshot`, `ingestion_runs` |
| `server/migrations/0003_push_rule_lifecycle.sql` | Future push-rule lifecycle storage | `push_rules` lifecycle fields and pending-rule indexes |
| `server/migrations/0004_rate_limits.sql` | Durable hash-only fixed-window counters | `push_rate_limits` |
| `.github/workflows/ci.yml`, `.github/workflows/security.yml` | Build, test, migration, Gitleaks, and dependency-review gates | required GitHub checks |
| `.github/dependabot.yml` | Weekly npm, pip, and Actions update configuration | Dependabot update schedules |

---

### Task 1: Establish the colocated Vitest, RTL, and MSW foundation

**Files:**
- Modify: `package.json`
- Modify: `package-lock.json`
- Modify: `vite.config.ts`
- Modify: `tsconfig.app.json`
- Create: `src/test/setup.ts`
- Create: `src/test/render.tsx`
- Create: `src/test/msw/server.ts`
- Create: `src/test/time.ts`
- Test: `src/test/render.test.tsx`

**Interfaces:**
- Consumes: React 19, Material UI, Vite's existing React plugin, and the existing `Root` application entry point.
- Produces: `renderWithApp(ui: ReactElement, options?: { route?: string }): RenderResult`, a configured MSW `server`, and the commands `test`, `test:run`, and `test:coverage`.

- [ ] **Step 1: Write the failing colocated test for the render and cleanup contract**

```tsx
// src/test/render.test.tsx
import {screen} from "@testing-library/react";
import {describe, expect, it} from "vitest";
import {renderWithApp} from "./render";

describe("renderWithApp", () => {
    it("renders with a memory route and clears persisted browser state after each test", () => {
        window.localStorage.setItem("reclive:test", "set");
        renderWithApp(<button type="button">Foundation ready</button>, {route: "/nick"});
        expect(screen.getByRole("button", {name: "Foundation ready"})).toBeVisible();
    });

    it("starts the next test with an empty localStorage", () => {
        expect(window.localStorage.getItem("reclive:test")).toBeNull();
    });
});
```

- [ ] **Step 2: Run the test to verify the missing test system fails**

Run: `npm run test:run -- src/test/render.test.tsx`

Expected: FAIL because `test:run` does not exist in `package.json`.

- [ ] **Step 3: Install exact frontend development dependencies and expose deterministic commands**

```json
// package.json: merge these fields; remove @types/axios from devDependencies
{
  "scripts": {
    "test": "vitest",
    "test:run": "vitest run",
    "test:coverage": "vitest run --coverage",
    "test:e2e": "playwright test"
  },
  "devDependencies": {
    "@playwright/test": "1.54.1",
    "@axe-core/playwright": "4.10.2",
    "@testing-library/jest-dom": "6.6.3",
    "@testing-library/react": "16.3.0",
    "@vitest/coverage-v8": "3.2.4",
    "axe-core": "4.10.3",
    "jsdom": "26.1.0",
    "msw": "2.10.2",
    "vitest": "3.2.4",
    "vitest-axe": "0.1.0"
  }
}
```

Run: `npm install --save-dev @playwright/test@1.54.1 @axe-core/playwright@4.10.2 @testing-library/jest-dom@6.6.3 @testing-library/react@16.3.0 @vitest/coverage-v8@3.2.4 axe-core@4.10.3 jsdom@26.1.0 msw@2.10.2 vitest@3.2.4 vitest-axe@0.1.0 && npm uninstall @types/axios`

This is the only command that changes `package-lock.json`; inspect its diff before continuing and retain no unrelated dependency changes.

- [ ] **Step 4: Configure Vitest and TypeScript globals**

```ts
// vite.config.ts
import {defineConfig} from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
    plugins: [react()],
    test: {
        environment: "jsdom",
        globals: true,
        setupFiles: ["./src/test/setup.ts"],
        clearMocks: true,
        mockReset: true,
        restoreMocks: true,
        coverage: {
            provider: "v8",
            reporter: ["text", "html", "lcov"],
            include: ["src/**/*.{ts,tsx}"],
            exclude: ["src/test/**", "src/main.tsx"],
        },
    },
});
```

```json
// tsconfig.app.json: replace compilerOptions.types
{
  "compilerOptions": {
    "types": ["vite/client", "vitest/globals", "@testing-library/jest-dom"]
  }
}
```

- [ ] **Step 5: Implement shared cleanup, MSW lifecycle, and the app renderer**

```ts
// src/test/msw/server.ts
import {setupServer} from "msw/node";

export const server = setupServer();
```

```ts
// src/test/setup.ts
import "@testing-library/jest-dom/vitest";
import {cleanup} from "@testing-library/react";
import {afterAll, afterEach, beforeAll, vi} from "vitest";
import {server} from "./msw/server";

beforeAll(() => server.listen({onUnhandledRequest: "error"}));

afterEach(() => {
    cleanup();
    server.resetHandlers();
    window.localStorage.clear();
    window.sessionStorage.clear();
    vi.useRealTimers();
});

afterAll(() => server.close());
```

```ts
// src/test/time.ts
import {vi} from "vitest";

export function freezeTime(value = "2026-08-31T12:00:00.000Z"): void {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(value));
}
```

```tsx
// src/test/render.tsx
import type {ReactElement} from "react";
import {render, type RenderOptions, type RenderResult} from "@testing-library/react";
import {CssBaseline, ThemeProvider} from "@mui/material";
import {MemoryRouter} from "react-router-dom";
import {createAppTheme} from "../app/theme";

export function renderWithApp(
    ui: ReactElement,
    {route = "/nick", ...options}: RenderOptions & {route?: string} = {},
): RenderResult {
    return render(
        <ThemeProvider theme={createAppTheme("light")}>
            <CssBaseline />
            <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
        </ThemeProvider>,
        options,
    );
}
```

- [ ] **Step 6: Run the focused test and complete the static checks**

Run: `npm run test:run -- src/test/render.test.tsx && npm run lint && npm run build`

Expected: PASS; both test cases pass, ESLint accepts the test support files, and `tsc -b && vite build` completes.

- [ ] **Step 7: Commit the frontend unit-test foundation**

```bash
git add package.json package-lock.json vite.config.ts tsconfig.app.json src/test
git commit -m "test: establish frontend unit test foundation"
```

### Task 2: Add Playwright Chromium route and accessibility smoke coverage

**Files:**
- Create: `playwright.config.ts`
- Create: `tests/e2e/route-smoke.spec.ts`
- Modify: `package.json`

**Interfaces:**
- Consumes: Task 1's `test:e2e` script, Vite, stable routes `/nick` and `/bakke`.
- Produces: a Chromium-only Playwright project named `chromium` which serves Vite at `http://127.0.0.1:4173` and verifies both public routes with mocked same-origin API results.

- [ ] **Step 1: Write the failing route smoke test**

```ts
// tests/e2e/route-smoke.spec.ts
import {expect, test} from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const liveRows = [{LocationId: 1, IsClosed: false, LastCount: 2, LastUpdatedDateAndTime: "2026-08-31T12:00:00Z"}];
const emptyForecast = {generatedAt: "2026-08-31T12:00:00Z", facilities: []};
const emptySchedule = {generatedAt: "2026-08-31T12:00:00Z", facilities: []};

test.beforeEach(async ({page}) => {
    await page.route("**/api/live-counts", (route) => route.fulfill({json: liveRows}));
    await page.route("**/api/forecast/**", (route) => route.fulfill({json: emptyForecast}));
    await page.route("**/api/facility-hours/**", (route) => route.fulfill({json: emptySchedule}));
});

for (const routePath of ["/nick", "/bakke"]) {
    test(`${routePath} loads a visible dashboard shell without critical axe violations`, async ({page}) => {
        await page.goto(routePath);
        await expect(page.locator("#root")).not.toBeEmpty();
        await expect(page).toHaveURL(new RegExp(`${routePath}$`));
        await expect(page.getByText("Train smarter. Skip the crowd.").first()).toBeVisible();

        const violations = (await new AxeBuilder({page}).withTags(["wcag2a", "wcag2aa"]).analyze()).violations
            .filter((violation) => violation.impact === "critical");
        expect(violations).toEqual([]);
    });
}
```

- [ ] **Step 2: Run the e2e command to verify configuration is absent**

Run: `npm run test:e2e -- tests/e2e/route-smoke.spec.ts`

Expected: FAIL because `playwright.config.ts` and the Chromium browser installation are absent.

- [ ] **Step 3: Configure a single Chromium project and build-backed Vite web server**

```ts
// playwright.config.ts
import {defineConfig, devices} from "@playwright/test";

export default defineConfig({
    testDir: "./tests/e2e",
    fullyParallel: true,
    forbidOnly: Boolean(process.env.CI),
    retries: process.env.CI ? 1 : 0,
    reporter: process.env.CI ? [["github"], ["html", {open: "never"}]] : "list",
    use: {
        baseURL: "http://127.0.0.1:4173",
        trace: "retain-on-failure",
    },
    projects: [{name: "chromium", use: {...devices["Desktop Chrome"]}}],
    webServer: {
        command: "npm run build && npm run preview -- --host 127.0.0.1 --port 4173",
        url: "http://127.0.0.1:4173",
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
    },
});
```

```json
// package.json: retain the Task 1 script verbatim
{
  "scripts": {
    "test:e2e": "playwright test"
  }
}
```

- [ ] **Step 4: Install only Chromium and run the focused smoke suite**

Run: `npx playwright install --with-deps chromium && npm run test:e2e -- tests/e2e/route-smoke.spec.ts`

Expected: PASS; both `/nick` and `/bakke` route tests complete in Chromium and each reports zero critical axe violations.

- [ ] **Step 5: Commit the browser test boundary**

```bash
git add playwright.config.ts tests/e2e/route-smoke.spec.ts package.json package-lock.json
git commit -m "test: add Chromium route smoke coverage"
```

### Task 3: Create the backend test, fixture, and lint foundation

**Files:**
- Create: `server/requirements-dev.txt`
- Create: `pytest.ini`
- Create: `tests/__init__.py`
- Create: `tests/backend/conftest.py`
- Create: `tests/backend/test_test_foundation.py`
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/live_counts.py`

**Interfaces:**
- Consumes: runtime packages pinned in `server/requirements.txt` and a MySQL service configured through `TEST_MYSQL_HOST`, `TEST_MYSQL_PORT`, `TEST_MYSQL_USER`, `TEST_MYSQL_PASSWORD`, and `TEST_MYSQL_DATABASE`; clean-schema setup alone uses separately scoped `TEST_MYSQL_ADMIN_USER` and `TEST_MYSQL_ADMIN_PASSWORD`.
- Produces: `mysql_settings() -> dict[str, object]`, `clean_test_database` fixture, and non-sensitive `LIVE_ROWS` data shared by future backend tests.

- [ ] **Step 1: Write the failing backend fixture test**

```python
# tests/backend/test_test_foundation.py
from datetime import datetime, timezone

from freezegun import freeze_time

from tests.fixtures.live_counts import LIVE_ROWS


@freeze_time("2026-08-31 12:00:00")
def test_fixture_rows_and_frozen_clock_are_deterministic() -> None:
    assert LIVE_ROWS[0]["LocationId"] == 5761
    assert datetime.now(timezone.utc).isoformat() == "2026-08-31T12:00:00+00:00"
```

- [ ] **Step 2: Run the backend test to verify its development dependencies are missing**

Run: `python -m pytest tests/backend/test_test_foundation.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'freezegun'`.

- [ ] **Step 3: Pin development-only Python tooling and define the pytest discovery contract**

```text
# server/requirements-dev.txt
pytest==8.3.4
pytest-cov==6.0.0
ruff==0.9.7
httpx==0.28.1
freezegun==1.5.1
```

```ini
# pytest.ini
[pytest]
testpaths = tests/backend
python_files = test_*.py
addopts = -ra --strict-config --strict-markers
markers =
    mysql: requires the MySQL test service
```

- [ ] **Step 4: Implement the reusable fixture modules and MySQL connection fixture**

```python
# tests/fixtures/live_counts.py
LIVE_ROWS = [
    {
        "LocationId": 5761,
        "IsClosed": False,
        "LastCount": 12,
        "LastUpdatedDateAndTime": "2026-08-31T12:00:00Z",
    },
]
```

```python
# tests/backend/conftest.py
import os
import re
from collections.abc import Iterator

import pymysql
import pytest


def mysql_settings() -> dict[str, object]:
    return {
        "host": os.environ.get("TEST_MYSQL_HOST", "127.0.0.1"),
        "port": int(os.environ.get("TEST_MYSQL_PORT", "3306")),
        "user": os.environ.get("TEST_MYSQL_USER", "reclive"),
        "password": os.environ.get("TEST_MYSQL_PASSWORD", "reclive-ci-password"),
        "database": os.environ.get("TEST_MYSQL_DATABASE", "reclive_test"),
        "charset": "utf8mb4",
        "autocommit": True,
    }


def test_database_name(settings: dict[str, object]) -> str:
    database = str(settings["database"])
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", database):
        raise RuntimeError("TEST_MYSQL_DATABASE must be a MySQL identifier")
    return database


@pytest.fixture()
def clean_test_database() -> Iterator[dict[str, object]]:
    settings = mysql_settings()
    database = test_database_name(settings)
    admin_settings = {
        "host": settings["host"],
        "port": settings["port"],
        "user": os.environ.get("TEST_MYSQL_ADMIN_USER", "root"),
        "password": os.environ.get("TEST_MYSQL_ADMIN_PASSWORD", ""),
        "charset": "utf8mb4",
        "autocommit": True,
    }
    connection = pymysql.connect(**admin_settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS `{database}`")
            cursor.execute(
                f"CREATE DATABASE `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
            )
        yield settings
    finally:
        with connection.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS `{database}`")
        connection.close()
```

```python
# tests/__init__.py
"""RecLive test support package."""
```

```python
# tests/fixtures/__init__.py
"""Reusable deterministic fixtures for RecLive backend tests."""
```

- [ ] **Step 5: Install and verify the backend test foundation**

Run: `python -m pip install -r server/requirements.txt -r server/requirements-dev.txt && python -m pytest tests/backend/test_test_foundation.py -q && ruff check server tests`

Expected: PASS; the deterministic fixture test passes and Ruff reports no violations.

- [ ] **Step 6: Commit the backend test boundary**

```bash
git add server/requirements-dev.txt pytest.ini tests
git commit -m "test: establish backend test and fixture foundation"
```

### Task 4: Implement the checksummed runner and compatible history migration

**Files:**
- Create: `server/reclive/__init__.py`
- Create: `server/reclive/migrations.py`
- Create: `server/migrate.py`
- Create: `server/migrations/0001_core_history.sql`
- Modify: `README.md`
- Create: `tests/backend/test_migrate.py`

**Interfaces:**
- Consumes: Task 3's `clean_test_database` settings fixture, PyMySQL, and `server/migrations/*.sql` numeric filenames.
- Produces: `MigrationSettings.from_environment() -> MigrationSettings`, `run_migrations(settings: MigrationSettings, migration_dir: Path) -> list[str]`, and CLI `python server/migrate.py [--migrations-dir PATH]`.

- [ ] **Step 1: Write the failing MySQL migration test for ordering, idempotency, and checksum refusal**

```python
# tests/backend/test_migrate.py
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pymysql
import pytest

ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = ROOT / "server" / "migrations"


def run_migrate(settings: dict[str, object], migration_dir: Path) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "GYM_DB_HOST": str(settings["host"]),
        "GYM_DB_PORT": str(settings["port"]),
        "GYM_DB_USER": str(settings["user"]),
        "GYM_DB_PASSWORD": str(settings["password"]),
        "GYM_DB_NAME": str(settings["database"]),
        "MIGRATION_LOCK_TIMEOUT_SECONDS": "5",
    }
    hash_key = settings.get("hash_key", "migration-test-key-with-at-least-thirty-two-bytes")
    if hash_key:
        env["PUSH_ENDPOINT_HASH_KEY"] = str(hash_key)
    else:
        env.pop("PUSH_ENDPOINT_HASH_KEY", None)
    return subprocess.run(
        [sys.executable, "server/migrate.py", "--migrations-dir", str(migration_dir)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.mysql
def test_runner_applies_once_then_rejects_an_applied_checksum_change(clean_test_database, tmp_path) -> None:
    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE TABLE location_history (id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, location_id INT NOT NULL, last_updated DATETIME(6) NULL) ENGINE=InnoDB")
            cursor.execute("INSERT INTO location_history (location_id, last_updated) VALUES (1186, '2026-08-31 10:00:00.123456')")
        connection.commit()
    finally:
        connection.close()
    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)
    for later_migration in migration_dir.glob("000[2-9]_*.sql"):
        later_migration.unlink()

    first = run_migrate(clean_test_database, migration_dir)
    assert first.returncode == 0, first.stderr
    second = run_migrate(clean_test_database, migration_dir)
    assert second.returncode == 0, second.stderr

    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT filename FROM schema_migrations ORDER BY filename")
            assert [row[0] for row in cursor.fetchall()] == ["0001_core_history.sql"]
            cursor.execute("SHOW INDEX FROM location_history WHERE Key_name = 'idx_location_history_location_fetched'")
            assert cursor.fetchone() is not None
            cursor.execute("SHOW INDEX FROM location_history WHERE Key_name = 'idx_location_history_location_source_updated'")
            assert cursor.fetchone() is not None
            cursor.execute("SELECT source_updated_at FROM location_history WHERE location_id = 1186")
            assert cursor.fetchone()[0].strftime("%Y-%m-%d %H:%M:%S.%f") == "2026-08-31 10:00:00.123456"
    finally:
        connection.close()

    core_history = migration_dir / "0001_core_history.sql"
    core_history.write_text(core_history.read_text(encoding="utf-8") + "\nSELECT 1;\n", encoding="utf-8")
    changed = run_migrate(clean_test_database, migration_dir)
    assert changed.returncode == 1
    assert "checksum mismatch" in changed.stderr.lower()
```

- [ ] **Step 2: Run the migration test to verify the runner is absent**

Run: `python -m pytest tests/backend/test_migrate.py::test_runner_applies_once_then_rejects_an_applied_checksum_change -q`

Expected: FAIL because `server/migrate.py` and `server/migrations/0001_core_history.sql` do not exist.

- [ ] **Step 3: Implement the migration runner with a bounded advisory lock and checksum contract**

```python
# server/reclive/migrations.py
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pymysql

MIGRATION_NAME = re.compile(r"^\d{4}_[a-z0-9_]+\.sql$")
LOCK_NAME = "reclive_schema_migrations"


@dataclass(frozen=True)
class MigrationSettings:
    host: str
    port: int
    user: str
    password: str
    database: str
    lock_timeout_seconds: int

    @classmethod
    def from_environment(cls) -> "MigrationSettings":
        required = {name: os.environ.get(name, "").strip() for name in ("GYM_DB_HOST", "GYM_DB_PORT", "GYM_DB_USER", "GYM_DB_PASSWORD", "GYM_DB_NAME")}
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
        return cls(
            host=required["GYM_DB_HOST"],
            port=int(required["GYM_DB_PORT"]),
            user=required["GYM_DB_USER"],
            password=required["GYM_DB_PASSWORD"],
            database=required["GYM_DB_NAME"],
            lock_timeout_seconds=max(1, int(os.environ.get("MIGRATION_LOCK_TIMEOUT_SECONDS", "30"))),
        )


def migration_files(migration_dir: Path) -> list[Path]:
    files = sorted(path for path in migration_dir.iterdir() if path.is_file() and MIGRATION_NAME.fullmatch(path.name))
    if not files:
        raise RuntimeError(f"No numeric SQL migrations found in {migration_dir}")
    return files


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def connect(settings: MigrationSettings):
    return pymysql.connect(
        host=settings.host,
        port=settings.port,
        user=settings.user,
        password=settings.password,
        database=settings.database,
        charset="utf8mb4",
        autocommit=False,
        connect_timeout=10,
        read_timeout=20,
        write_timeout=20,
    )


def split_statements(sql: str) -> list[str]:
    without_full_line_comments = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )
    statements = [statement.strip() for statement in without_full_line_comments.split(";")]
    return [statement for statement in statements if statement and not statement.startswith("--")]


def run_post_sql_hook(filename: str, connection, settings: MigrationSettings) -> None:
    if filename != "0003_push_rule_lifecycle.sql":
        return
    from reclive.push_rule_backfill import backfill_push_rule_lifecycle

    backfill_push_rule_lifecycle(connection, settings)


def run_pre_sql_hook(filename: str, connection, settings: MigrationSettings) -> None:
    if filename != "0003_push_rule_lifecycle.sql":
        return
    from reclive.push_rule_backfill import preflight_legacy_push_rules

    preflight_legacy_push_rules(connection, settings)


def run_migrations(settings: MigrationSettings, migration_dir: Path) -> list[str]:
    connection = connect(settings)
    locked = False
    applied: list[str] = []
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT GET_LOCK(%s, %s)", (LOCK_NAME, settings.lock_timeout_seconds))
            locked = bool(cursor.fetchone()[0])
            if not locked:
                raise RuntimeError("Timed out acquiring migration advisory lock")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    filename VARCHAR(255) NOT NULL PRIMARY KEY,
                    checksum CHAR(64) NOT NULL,
                    applied_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
        connection.commit()

        for path in migration_files(migration_dir):
            file_checksum = checksum(path)
            with connection.cursor() as cursor:
                cursor.execute("SELECT checksum FROM schema_migrations WHERE filename = %s", (path.name,))
                recorded = cursor.fetchone()
            if recorded:
                if recorded[0] != file_checksum:
                    raise RuntimeError(f"Migration checksum mismatch for {path.name}")
                continue

            run_pre_sql_hook(path.name, connection, settings)
            statements = split_statements(path.read_text(encoding="utf-8"))
            try:
                with connection.cursor() as cursor:
                    for statement in statements:
                        cursor.execute(statement)
                run_post_sql_hook(path.name, connection, settings)
                with connection.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO schema_migrations (filename, checksum) VALUES (%s, %s)",
                        (path.name, file_checksum),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            applied.append(path.name)
        return applied
    finally:
        if locked:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT RELEASE_LOCK(%s)", (LOCK_NAME,))
            finally:
                connection.commit()
        connection.close()
```

```python
# server/reclive/__init__.py
"""RecLive backend modules."""
```

```python
# server/migrate.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reclive.migrations import MigrationSettings, run_migrations


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply ordered RecLive MySQL migrations")
    parser.add_argument("--migrations-dir", type=Path, default=Path(__file__).with_name("migrations"))
    args = parser.parse_args()
    try:
        applied = run_migrations(MigrationSettings.from_environment(), args.migrations_dir)
    except Exception as exc:
        print(f"migration failed: {exc}", file=sys.stderr)
        return 1
    print(f"migration complete: {len(applied)} applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Author the compatible core-history SQL migration**

```sql
-- server/migrations/0001_core_history.sql
CREATE TABLE IF NOT EXISTS location_history (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    location_id INT NOT NULL,
    is_closed TINYINT(1) NULL,
    current_capacity INT NULL,
    max_capacity INT NULL,
    last_updated DATETIME(6) NULL,
    source_updated_at DATETIME(6) NULL,
    fetched_at DATETIME(6) NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

SET @schema_name = DATABASE();
SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'location_history' AND column_name = 'is_closed') = 0, 'ALTER TABLE location_history ADD COLUMN is_closed TINYINT(1) NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;

SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'location_history' AND column_name = 'current_capacity') = 0, 'ALTER TABLE location_history ADD COLUMN current_capacity INT NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;

SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'location_history' AND column_name = 'max_capacity') = 0, 'ALTER TABLE location_history ADD COLUMN max_capacity INT NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;

SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'location_history' AND column_name = 'last_updated') = 0, 'ALTER TABLE location_history ADD COLUMN last_updated DATETIME(6) NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;

SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'location_history' AND column_name = 'source_updated_at') = 0, 'ALTER TABLE location_history ADD COLUMN source_updated_at DATETIME(6) NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;

UPDATE location_history
SET source_updated_at = last_updated
WHERE source_updated_at IS NULL AND last_updated IS NOT NULL;

SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'location_history' AND column_name = 'fetched_at') = 0, 'ALTER TABLE location_history ADD COLUMN fetched_at DATETIME(6) NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;

SET @statement = IF((SELECT COUNT(*) FROM information_schema.statistics WHERE table_schema = @schema_name AND table_name = 'location_history' AND index_name = 'idx_location_history_location_fetched') = 0, 'CREATE INDEX idx_location_history_location_fetched ON location_history (location_id, fetched_at)', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;

SET @statement = IF((SELECT COUNT(*) FROM information_schema.statistics WHERE table_schema = @schema_name AND table_name = 'location_history' AND index_name = 'idx_location_history_location_source_updated') = 0, 'CREATE INDEX idx_location_history_location_source_updated ON location_history (location_id, source_updated_at)', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;
```

The runner strips only full-line `--` comments and passes every remaining semicolon-terminated MySQL statement to the existing PyMySQL cursor. This keeps `PREPARE`/`EXECUTE`/`DEALLOCATE PREPARE` sequences intact without adding a production SQL-parser dependency.

````markdown
<!-- README.md: append after the Tech Stack section -->
## Database migrations

Apply checked-in MySQL schema changes before starting an application process that needs new tables or columns:

```bash
python server/migrate.py
```

The command reads the private `GYM_DB_HOST`, `GYM_DB_PORT`, `GYM_DB_USER`, `GYM_DB_PASSWORD`, and `GYM_DB_NAME` environment variables. It records every applied SQL filename and SHA-256 checksum in `schema_migrations`; never edit an applied migration file.
````

- [ ] **Step 5: Run migration tests and verify only the expected schema exists**

Run: `python -m pytest tests/backend/test_migrate.py::test_runner_applies_once_then_rejects_an_applied_checksum_change -q`

Expected: PASS; the clean database has one recorded migration and both history indexes, the unchanged second run succeeds, and the changed-copy run exits with a checksum mismatch.

- [ ] **Step 6: Commit the runner and first migration**

```bash
git add server/reclive server/migrate.py server/migrations/0001_core_history.sql README.md tests/backend/test_migrate.py
git commit -m "feat: add checksummed MySQL migration runner"
```

### Task 5: Add snapshot and ingestion-run schema migration

**Files:**
- Create: `server/migrations/0002_snapshot_and_ingestion.sql`
- Modify: `tests/backend/test_migrate.py`

**Interfaces:**
- Consumes: Task 4 runner contract and the existing `location_history` compatibility schema.
- Produces: `location_snapshot` keyed by `location_id` and `ingestion_runs` with count, status, UTC timestamp, observed-ID, and bounded-error columns for Phase 2 ingestion.

- [ ] **Step 1: Extend the migration test with the failing snapshot contract**

```python
# tests/backend/test_migrate.py: append this test
@pytest.mark.mysql
def test_0002_creates_current_snapshot_and_ingestion_metadata(clean_test_database, tmp_path) -> None:
    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)
    result = run_migrate(clean_test_database, migration_dir)
    assert result.returncode == 0, result.stderr

    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SHOW COLUMNS FROM location_snapshot")
            snapshot_columns = {row[0] for row in cursor.fetchall()}
            assert {"location_id", "is_closed", "current_capacity", "max_capacity", "source_updated_at", "fetched_at", "created_at", "updated_at"} <= snapshot_columns
            cursor.execute("SHOW COLUMNS FROM ingestion_runs")
            run_columns = {row[0] for row in cursor.fetchall()}
            assert {"id", "started_at", "completed_at", "status", "received_count", "valid_count", "history_inserted_count", "snapshot_updated_count", "observed_location_ids", "error_category", "error_message"} <= run_columns
            cursor.execute("INSERT INTO ingestion_runs (started_at, status) VALUES (UTC_TIMESTAMP(6), 'running')")
            cursor.execute("SELECT JSON_LENGTH(observed_location_ids) FROM ingestion_runs")
            assert cursor.fetchone()[0] == 0
    finally:
        connection.close()
```

- [ ] **Step 2: Run the focused test to prove migration `0002` is missing**

Run: `python -m pytest tests/backend/test_migrate.py::test_0002_creates_current_snapshot_and_ingestion_metadata -q`

Expected: FAIL because `location_snapshot` does not exist.

- [ ] **Step 3: Create the complete snapshot and ingestion-run migration**

```sql
-- server/migrations/0002_snapshot_and_ingestion.sql
CREATE TABLE IF NOT EXISTS location_snapshot (
    location_id INT NOT NULL,
    is_closed TINYINT(1) NULL,
    current_capacity INT NULL,
    max_capacity INT NULL,
    source_updated_at DATETIME(6) NULL,
    fetched_at DATETIME(6) NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (location_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    started_at DATETIME(6) NOT NULL,
    completed_at DATETIME(6) NULL,
    status VARCHAR(16) NOT NULL,
    received_count INT UNSIGNED NOT NULL DEFAULT 0,
    valid_count INT UNSIGNED NOT NULL DEFAULT 0,
    history_inserted_count INT UNSIGNED NOT NULL DEFAULT 0,
    snapshot_updated_count INT UNSIGNED NOT NULL DEFAULT 0,
    observed_location_ids JSON NOT NULL DEFAULT (JSON_ARRAY()),
    error_category VARCHAR(64) NULL,
    error_message VARCHAR(240) NULL,
    PRIMARY KEY (id),
    CONSTRAINT chk_ingestion_runs_status CHECK (status IN ('running', 'succeeded', 'failed'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

SET @schema_name = DATABASE();
SET @statement = IF((SELECT COUNT(*) FROM information_schema.statistics WHERE table_schema = @schema_name AND table_name = 'location_snapshot' AND index_name = 'idx_location_snapshot_fetched_at') = 0, 'CREATE INDEX idx_location_snapshot_fetched_at ON location_snapshot (fetched_at)', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;

SET @statement = IF((SELECT COUNT(*) FROM information_schema.statistics WHERE table_schema = @schema_name AND table_name = 'ingestion_runs' AND index_name = 'idx_ingestion_runs_status_completed') = 0, 'CREATE INDEX idx_ingestion_runs_status_completed ON ingestion_runs (status, completed_at)', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;
```

- [ ] **Step 4: Run the migration tests for a clean and already-migrated MySQL 8.4 database**

Run: `python -m pytest tests/backend/test_migrate.py -q`

Expected: PASS; Task 4's repeat/checksum test and the snapshot/ingestion table test pass against the clean service, and the second invocation in the first test proves idempotent tracking.

- [ ] **Step 5: Commit the snapshot schema boundary**

```bash
git add server/migrations/0002_snapshot_and_ingestion.sql tests/backend/test_migrate.py
git commit -m "feat: add snapshot and ingestion migrations"
```

### Task 6: Upgrade clean and legacy push-rule schemas without raw-endpoint loss

**Files:**
- Create: `server/migrations/0003_push_rule_lifecycle.sql`
- Create: `server/reclive/push_rule_backfill.py`
- Create: `server/reclive/push_identity.py`
- Modify: `server/reclive/migrations.py`
- Modify: `tests/backend/test_migrate.py`

**Interfaces:**
- Consumes: Task 4's runner and the exact legacy contract inferred from `server/forecast_api.py`: `push_rules(endpoint, subscription_json, facility_id, section_key, threshold, created_at)`.
- Produces: final `push_rules` rows identified by `endpoint_hash BINARY(32)`, generated nullable `active_identity` (`1` only while status is `pending` or `claimed`), `uq_push_rules_identity(endpoint_hash, facility_id, section_key, threshold, active_identity)`, `expires_at`, `status`, `claimed_at`, `sent_at`, `finalized_at`, and `failure_code`; `status` permits `pending`, `claimed`, `sent`, `failed`, `expired`, `invalid_subscription`, and `cancelled`. It also produces the Phase 5 shared identity API: `normalize_push_endpoint(value: str) -> str`, `endpoint_hash(endpoint: str) -> bytes`, and `rate_limit_subject_hash(subject_kind: Literal["endpoint", "client"], subject: str) -> bytes`. Both hashes load and validate the single configured `PUSH_ENDPOINT_HASH_KEY` internally and HMAC domain-separated UTF-8 messages; callers never pass keys or create local hash variants. Multiple terminal rules have a `NULL` active identity, retaining every row and its subscription material, while MySQL permits only one active rule per identity. For deterministic legacy duplicates, the greatest `(created_at, id)` remains pending and each older original row becomes `cancelled` with `failure_code` `migration_duplicate`.

- [ ] **Step 1: Write failing clean-schema, seeded-legacy, and missing-key tests**

```python
# tests/backend/test_migrate.py: append these helpers and tests
import hmac
from hashlib import sha256

from reclive.push_identity import endpoint_hash, rate_limit_subject_hash


def create_legacy_push_rules(settings: dict[str, object]) -> None:
    connection = pymysql.connect(**settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE push_rules (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    endpoint VARCHAR(2048) NOT NULL,
                    subscription_json JSON NOT NULL,
                    facility_id INT NOT NULL,
                    section_key VARCHAR(80) NOT NULL,
                    threshold INT NOT NULL,
                    created_at DATETIME(6) NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cursor.execute("CREATE INDEX idx_legacy_push_rules_endpoint ON push_rules (endpoint(191))")
            rows = [
                ("https://push.example.test/a", '{"endpoint":"https://push.example.test/a","keys":{"p256dh":"x","auth":"y"}}', 1186, "fitness", 40, "2026-08-31 10:00:00.000000"),
                ("https://push.example.test/a", '{"endpoint":"https://push.example.test/a","keys":{"p256dh":"x2","auth":"y2"}}', 1186, "fitness", 40, "2026-08-31 10:01:00.000000"),
                ("https://push.example.test/b", '{"endpoint":"https://push.example.test/b","keys":{"p256dh":"z","auth":"q"}}', 1656, "overall", 55, "2026-08-31 10:02:00.000000"),
                ("https://push.example.test/c", '{"endpoint":"https://push.example.test/c","keys":{"p256dh":"bad","auth":"bad"}}', 1186, "fitness", 101, "2026-08-31 10:03:00.000000"),
                ("https://push.example.test/d", '{"endpoint":"https://push.example.test/other","keys":{"p256dh":"mismatch","auth":"mismatch"}}', 1186, "fitness", 1000, "2026-08-31 10:04:00.000000"),
            ]
            cursor.executemany(
                "INSERT INTO push_rules (endpoint, subscription_json, facility_id, section_key, threshold, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
                rows,
            )
        connection.commit()
    finally:
        connection.close()


def test_0003_uses_canonical_endpoint_hmac_and_a_domain_separated_rate_limit_subject(monkeypatch) -> None:
    key = "migration-test-key-with-at-least-thirty-two-bytes"
    monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", key)
    expected = hmac.new(
        key.encode("utf-8"),
        b"reclive:push:endpoint:v1\x00https://push.example.test/a?x=1",
        sha256,
    ).digest()
    assert endpoint_hash(" HTTPS://PUSH.EXAMPLE.TEST:443/a?x=1 ") == expected
    assert rate_limit_subject_hash("endpoint", "https://push.example.test/a?x=1") != expected


@pytest.mark.mysql
def test_0003_creates_finalized_clean_push_rule_contract(clean_test_database, tmp_path) -> None:
    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)
    result = run_migrate(clean_test_database, migration_dir)
    assert result.returncode == 0, result.stderr
    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SHOW COLUMNS FROM push_rules")
            columns = {row[0]: row for row in cursor.fetchall()}
            assert {"id", "endpoint_hash", "active_identity", "subscription_json", "facility_id", "section_key", "threshold", "created_at", "expires_at", "status", "claimed_at", "sent_at", "finalized_at", "failure_code"} <= set(columns)
            assert columns["endpoint_hash"][2] == "NO"
            assert "endpoint" not in columns
            cursor.execute("SHOW INDEX FROM push_rules WHERE Key_name = 'uq_push_rules_identity'")
            assert cursor.fetchone() is not None
            cursor.execute("SHOW INDEX FROM push_rules WHERE Key_name = 'idx_push_rules_pending'")
            assert cursor.fetchone() is not None
    finally:
        connection.close()


@pytest.mark.mysql
def test_0003_backfills_legacy_rows_cancels_deterministic_duplicates_and_removes_raw_endpoint(clean_test_database, tmp_path) -> None:
    create_legacy_push_rules(clean_test_database)
    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)
    result = run_migrate(clean_test_database, migration_dir)
    assert result.returncode == 0, result.stderr
    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, endpoint_hash, subscription_json, status, finalized_at FROM push_rules ORDER BY id")
            rules = cursor.fetchall()
            assert len(rules) == 5
            expected_hash = hmac.new(b"migration-test-key-with-at-least-thirty-two-bytes", b"reclive:push:endpoint:v1\x00https://push.example.test/a", sha256).digest()
            assert rules[0][0] == 1 and rules[0][1] == expected_hash and rules[0][3] == "cancelled" and rules[0][4] is not None
            assert "https://push.example.test/a" in rules[0][2]
            assert rules[1][0] == 2 and rules[1][3] == "pending" and rules[1][4] is None
            assert rules[3][3] == "cancelled" and rules[4][3] == "cancelled"
            cursor.execute("SHOW COLUMNS FROM push_rules LIKE 'endpoint'")
            assert cursor.fetchone() is None
            cursor.execute("SHOW INDEX FROM push_rules WHERE Key_name = 'idx_legacy_push_rules_endpoint'")
            assert cursor.fetchone() is None
            cursor.execute("UPDATE push_rules SET status = 'sent', sent_at = UTC_TIMESTAMP(6), finalized_at = UTC_TIMESTAMP(6) WHERE id = 2")
            cursor.execute("INSERT INTO push_rules (endpoint_hash, subscription_json, facility_id, section_key, threshold, expires_at, status) VALUES (%s, %s, %s, %s, %s, DATE_ADD(UTC_TIMESTAMP(6), INTERVAL 24 HOUR), 'pending')", (expected_hash, '{"endpoint":"https://push.example.test/a","keys":{"p256dh":"new","auth":"new"}}', 1186, "fitness", 40))
            with pytest.raises(pymysql.err.IntegrityError):
                cursor.execute("INSERT INTO push_rules (endpoint_hash, subscription_json, facility_id, section_key, threshold, expires_at, status) VALUES (%s, %s, %s, %s, %s, DATE_ADD(UTC_TIMESTAMP(6), INTERVAL 24 HOUR), 'pending')", (expected_hash, '{"endpoint":"https://push.example.test/a","keys":{"p256dh":"newer","auth":"newer"}}', 1186, "fitness", 40))
    finally:
        connection.close()


@pytest.mark.mysql
def test_0003_fails_closed_before_recording_when_legacy_rows_exist_without_hash_key(clean_test_database, tmp_path) -> None:
    create_legacy_push_rules(clean_test_database)
    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)
    environment = {**clean_test_database, "hash_key": None}
    result = run_migrate(environment, migration_dir)
    assert result.returncode == 1
    assert "PUSH_ENDPOINT_HASH_KEY" in result.stderr
    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT filename FROM schema_migrations WHERE filename = '0003_push_rule_lifecycle.sql'")
            assert cursor.fetchone() is None
            cursor.execute("SHOW COLUMNS FROM push_rules LIKE 'endpoint'")
            assert cursor.fetchone() is not None
            cursor.execute("SHOW COLUMNS FROM push_rules LIKE 'endpoint_hash'")
            assert cursor.fetchone() is None
    finally:
        connection.close()
```

- [ ] **Step 2: Run the lifecycle tests to observe that the current clean-only migration cannot preserve legacy rules**

Run: `python -m pytest tests/backend/test_migrate.py -k '0003' -q`

Expected: FAIL; the current migration does not preserve/cancel duplicate legacy rows, does not remove the raw endpoint column, lacks `finalized_at` and `cancelled`, and cannot fail closed before recording an unkeyed legacy conversion.

- [ ] **Step 3: Create idempotent SQL that stages both clean and legacy tables for the Python backfill**

```sql
-- server/migrations/0003_push_rule_lifecycle.sql
CREATE TABLE IF NOT EXISTS push_rules (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    endpoint_hash BINARY(32) NULL,
    subscription_json JSON NOT NULL,
    facility_id INT NOT NULL,
    section_key VARCHAR(80) NOT NULL,
    threshold TINYINT UNSIGNED NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    expires_at DATETIME(6) NULL,
    status VARCHAR(32) NULL,
    active_identity TINYINT GENERATED ALWAYS AS (CASE WHEN status IN ('pending', 'claimed') THEN 1 ELSE NULL END) STORED,
    claimed_at DATETIME(6) NULL,
    sent_at DATETIME(6) NULL,
    finalized_at DATETIME(6) NULL,
    failure_code VARCHAR(64) NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

SET @schema_name = DATABASE();
SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'push_rules' AND column_name = 'endpoint_hash') = 0, 'ALTER TABLE push_rules ADD COLUMN endpoint_hash BINARY(32) NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement; EXECUTE migration_statement; DEALLOCATE PREPARE migration_statement;
SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'push_rules' AND column_name = 'expires_at') = 0, 'ALTER TABLE push_rules ADD COLUMN expires_at DATETIME(6) NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement; EXECUTE migration_statement; DEALLOCATE PREPARE migration_statement;
SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'push_rules' AND column_name = 'status') = 0, 'ALTER TABLE push_rules ADD COLUMN status VARCHAR(32) NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement; EXECUTE migration_statement; DEALLOCATE PREPARE migration_statement;
SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'push_rules' AND column_name = 'active_identity') = 0, 'ALTER TABLE push_rules ADD COLUMN active_identity TINYINT GENERATED ALWAYS AS (CASE WHEN status IN (''pending'', ''claimed'') THEN 1 ELSE NULL END) STORED', 'SELECT 1');
PREPARE migration_statement FROM @statement; EXECUTE migration_statement; DEALLOCATE PREPARE migration_statement;
SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'push_rules' AND column_name = 'claimed_at') = 0, 'ALTER TABLE push_rules ADD COLUMN claimed_at DATETIME(6) NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement; EXECUTE migration_statement; DEALLOCATE PREPARE migration_statement;
SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'push_rules' AND column_name = 'sent_at') = 0, 'ALTER TABLE push_rules ADD COLUMN sent_at DATETIME(6) NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement; EXECUTE migration_statement; DEALLOCATE PREPARE migration_statement;
SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'push_rules' AND column_name = 'finalized_at') = 0, 'ALTER TABLE push_rules ADD COLUMN finalized_at DATETIME(6) NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement; EXECUTE migration_statement; DEALLOCATE PREPARE migration_statement;
SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'push_rules' AND column_name = 'failure_code') = 0, 'ALTER TABLE push_rules ADD COLUMN failure_code VARCHAR(64) NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement; EXECUTE migration_statement; DEALLOCATE PREPARE migration_statement;
```

- [ ] **Step 4: Implement preflight, deterministic HMAC backfill, validation, and finalization before recording migration `0003`**

```python
# server/reclive/push_rule_backfill.py
from __future__ import annotations

import re
import json
from reclive.push_identity import configured_push_hash_key, endpoint_hash, normalize_push_endpoint

IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")
LEGACY_COLUMNS = {"endpoint", "subscription_json", "facility_id", "section_key", "threshold", "created_at"}


```

```python
# server/reclive/push_identity.py
from __future__ import annotations

import os
import hmac
from hashlib import sha256
from typing import Literal
from urllib.parse import urlsplit, urlunsplit


HASH_KEY_NAME = "PUSH_ENDPOINT_HASH_KEY"


def configured_push_hash_key() -> bytes:
    key = os.environ.get(HASH_KEY_NAME, "").encode("utf-8")
    if len(key) < 32:
        raise RuntimeError(f"{HASH_KEY_NAME} must be configured with at least 32 bytes")
    return key


def normalize_push_endpoint(value: str) -> str:
    text = value.strip()
    if len(text.encode("utf-8")) > 2048:
        raise ValueError("push endpoint is too long")
    parsed = urlsplit(text)
    if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise ValueError("push endpoint must be HTTPS")
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    port = parsed.port
    authority = host if port in (None, 443) else f"{host}:{port}"
    return urlunsplit(("https", authority, parsed.path or "/", parsed.query, ""))


def _configured_hmac(domain: str, value: str) -> bytes:
    message = f"{domain}\x00{value}".encode("utf-8")
    return hmac.new(configured_push_hash_key(), message, sha256).digest()


def endpoint_hash(endpoint: str) -> bytes:
    return _configured_hmac("reclive:push:endpoint:v1", normalize_push_endpoint(endpoint))


def rate_limit_subject_hash(subject_kind: Literal["endpoint", "client"], subject: str) -> bytes:
    if subject_kind == "endpoint":
        subject = normalize_push_endpoint(subject)
    elif subject_kind != "client" or not subject.strip():
        raise ValueError("invalid rate-limit subject")
    return _configured_hmac(f"reclive:push:rate-limit:{subject_kind}:v1", subject)
```

```python
# server/reclive/push_rule_backfill.py: continue after imports


def table_columns(connection, table: str) -> set[str]:
    with connection.cursor() as cursor:
        cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = DATABASE() AND table_name = %s", (table,))
        return {str(row[0]) for row in cursor.fetchall()}


def require_hash_key() -> bytes:
    return configured_push_hash_key()


def preflight_legacy_push_rules(connection, _settings) -> None:
    columns = table_columns(connection, "push_rules")
    if "endpoint" not in columns:
        return
    if not LEGACY_COLUMNS <= columns:
        raise RuntimeError("push_rules legacy preflight failed: required legacy columns are missing")
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM push_rules")
        has_rows = int(cursor.fetchone()[0]) > 0
    if has_rows:
        require_hash_key()


def backfill_push_rule_lifecycle(connection, _settings) -> None:
    columns = table_columns(connection, "push_rules")
    has_legacy_endpoint = "endpoint" in columns
    if not has_legacy_endpoint:
        finalize_empty_or_resumed_table(connection)
        return
    preflight_legacy_push_rules(connection, _settings)
    with connection.cursor() as cursor:
        cursor.execute("SELECT id, endpoint, subscription_json, facility_id, section_key, threshold, created_at FROM push_rules ORDER BY id")
        legacy_rows = cursor.fetchall()
    if not legacy_rows:
        finalize_empty_or_resumed_table(connection)
        return
    require_hash_key()
    prepared: list[tuple[object, ...]] = []
    for rule_id, endpoint, subscription_json, facility_id, section_key, threshold, created_at in legacy_rows:
        raw_endpoint = str(endpoint or "")
        try:
            normalized_endpoint = normalize_push_endpoint(raw_endpoint)
            decoded_subscription = json.loads(subscription_json) if isinstance(subscription_json, str) else subscription_json
            subscription_endpoint = decoded_subscription.get("endpoint") if isinstance(decoded_subscription, dict) else None
            valid_subscription = isinstance(subscription_endpoint, str) and normalize_push_endpoint(subscription_endpoint) == normalized_endpoint
            valid_threshold = 1 <= int(threshold) <= 100
            failure_code = None if valid_subscription and valid_threshold else ("migration_invalid_threshold" if not valid_threshold else "migration_invalid_subscription")
            digest = endpoint_hash(normalized_endpoint)
        except (ValueError, TypeError, json.JSONDecodeError):
            digest = endpoint_hash(f"https://legacy.invalid/{int(rule_id)}")
            failure_code = "migration_invalid_subscription"
        safe_threshold = int(threshold) if 1 <= int(threshold) <= 100 else 1
        prepared.append((int(rule_id), digest, subscription_json, int(facility_id), str(section_key or "legacy").strip() or "legacy", safe_threshold, created_at, failure_code))
    winners: dict[tuple[bytes, int, str, int], tuple[object, ...]] = {}
    duplicates: list[tuple[object, ...]] = []
    for row in sorted(prepared, key=lambda item: (item[6], item[0]), reverse=True):
        identity = (row[1], row[3], row[4], row[5])
        if row[7] is not None or identity in winners:
            duplicates.append(row)
        else:
            winners[identity] = row
    with connection.cursor() as cursor:
        for rule_id, digest, subscription_json, facility_id, section_key, threshold, _created_at, _failure_code in winners.values():
            cursor.execute(
                "UPDATE push_rules SET endpoint_hash = %s, expires_at = DATE_ADD(UTC_TIMESTAMP(6), INTERVAL 24 HOUR), status = 'pending', claimed_at = NULL, finalized_at = NULL, failure_code = NULL WHERE id = %s",
                (digest, rule_id),
            )
        for rule_id, digest, _subscription_json, _facility_id, _section_key, _threshold, _created_at, failure_code in duplicates:
            cursor.execute(
                "UPDATE push_rules SET endpoint_hash = %s, threshold = %s, expires_at = UTC_TIMESTAMP(6), status = 'cancelled', claimed_at = NULL, sent_at = NULL, finalized_at = UTC_TIMESTAMP(6), failure_code = %s WHERE id = %s",
                (digest, _threshold, failure_code or "migration_duplicate", rule_id),
            )
    finalize_empty_or_resumed_table(connection)


def finalize_empty_or_resumed_table(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM push_rules WHERE endpoint_hash IS NULL OR LENGTH(endpoint_hash) <> 32")
        if int(cursor.fetchone()[0]) != 0:
            raise RuntimeError("push_rules backfill validation failed: endpoint hashes are incomplete")
        cursor.execute("SELECT COUNT(*) FROM (SELECT endpoint_hash, facility_id, section_key, threshold, active_identity FROM push_rules GROUP BY endpoint_hash, facility_id, section_key, threshold, active_identity HAVING active_identity = 1 AND COUNT(*) > 1) AS duplicate_identities")
        if int(cursor.fetchone()[0]) != 0:
            raise RuntimeError("push_rules backfill validation failed: duplicate identities remain")
        cursor.execute("SELECT index_name FROM information_schema.statistics WHERE table_schema = DATABASE() AND table_name = 'push_rules' AND column_name = 'endpoint'")
        endpoint_indexes = {str(row[0]) for row in cursor.fetchall()}
        for name in endpoint_indexes:
            if not IDENTIFIER.fullmatch(name):
                raise RuntimeError("push_rules backfill validation failed: unsafe legacy index name")
            cursor.execute(f"DROP INDEX `{name}` ON push_rules")
        if "endpoint" in table_columns(connection, "push_rules"):
            cursor.execute("ALTER TABLE push_rules DROP COLUMN endpoint")
        cursor.execute("ALTER TABLE push_rules MODIFY endpoint_hash BINARY(32) NOT NULL, MODIFY expires_at DATETIME(6) NOT NULL, MODIFY status VARCHAR(32) NOT NULL")
        ensure_constraint(cursor, "chk_push_rules_threshold", "ALTER TABLE push_rules ADD CONSTRAINT chk_push_rules_threshold CHECK (threshold BETWEEN 1 AND 100)")
        ensure_constraint(cursor, "chk_push_rules_status", "ALTER TABLE push_rules ADD CONSTRAINT chk_push_rules_status CHECK (status IN ('pending', 'claimed', 'sent', 'failed', 'expired', 'invalid_subscription', 'cancelled'))")
        ensure_index(cursor, "uq_push_rules_identity", "CREATE UNIQUE INDEX uq_push_rules_identity ON push_rules (endpoint_hash, facility_id, section_key, threshold, active_identity)")
        ensure_index(cursor, "idx_push_rules_pending", "CREATE INDEX idx_push_rules_pending ON push_rules (status, expires_at, id)")


def ensure_constraint(cursor, name: str, statement: str) -> None:
    cursor.execute("SELECT COUNT(*) FROM information_schema.table_constraints WHERE table_schema = DATABASE() AND table_name = 'push_rules' AND constraint_name = %s", (name,))
    if int(cursor.fetchone()[0]) == 0:
        cursor.execute(statement)


def ensure_index(cursor, name: str, statement: str) -> None:
    cursor.execute("SELECT COUNT(*) FROM information_schema.statistics WHERE table_schema = DATABASE() AND table_name = 'push_rules' AND index_name = %s", (name,))
    if int(cursor.fetchone()[0]) == 0:
        cursor.execute(statement)
```

The concrete `ensure_constraint` and `ensure_index` helpers query `information_schema` before every final DDL statement, so a retry after a crash never errors on an already-created object. The runner calls this hook after SQL and before its `INSERT INTO schema_migrations`; therefore an absent/short hash key, preflight error, duplicate validation failure, or DDL failure leaves `0003_push_rule_lifecycle.sql` unrecorded and rerunnable.

- [ ] **Step 5: Run clean, seeded-legacy, keyless-failure, terminal-recreation, and resumed migration tests**

Run: `python -m pytest tests/backend/test_migrate.py -k '0003 or runner_applies_once' -q`

Expected: PASS; clean MySQL has the final BINARY(32)/active-identity unique schema, seeded legacy rows retain every subscription and deterministically cancel duplicate active identities, terminal rules can be recreated while a second active duplicate is rejected, the raw endpoint column/index disappears only after validation, a missing key aborts before schema recording, and rerunning the migration after a completed conversion changes nothing.

- [ ] **Step 6: Commit the resumable push-rule migration boundary**

```bash
git add server/reclive/migrations.py server/reclive/push_identity.py server/reclive/push_rule_backfill.py server/migrations/0003_push_rule_lifecycle.sql tests/backend/test_migrate.py tests/backend/conftest.py
git commit -m "feat: migrate legacy push rules safely"
```

### Task 7: Add durable rate-limit migration

**Files:**
- Create: `server/migrations/0004_rate_limits.sql`
- Modify: `tests/backend/test_migrate.py`

**Interfaces:**
- Consumes: Task 4's checksum runner and the Phase 5 requirement for nonreversible endpoint/client subjects.
- Produces: `push_rate_limits(subject_hash: BINARY(32), window_started_at: DATETIME(6), request_count: INT UNSIGNED, updated_at: DATETIME(6))` with a bounded cleanup scan index.

- [ ] **Step 1: Write the failing rate-limit table test**

```python
# tests/backend/test_migrate.py: append this test
@pytest.mark.mysql
def test_0004_creates_hashed_rate_limit_counter(clean_test_database, tmp_path) -> None:
    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)
    result = run_migrate(clean_test_database, migration_dir)
    assert result.returncode == 0, result.stderr

    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SHOW COLUMNS FROM push_rate_limits")
            columns = {row[0] for row in cursor.fetchall()}
            assert {"subject_hash", "window_started_at", "request_count", "updated_at"} <= columns
            cursor.execute("SHOW INDEX FROM push_rate_limits WHERE Key_name = 'idx_push_rate_limits_cleanup'")
            assert cursor.fetchone() is not None
    finally:
        connection.close()
```

- [ ] **Step 2: Run the test to prove `0004` is absent**

Run: `python -m pytest tests/backend/test_migrate.py::test_0004_creates_hashed_rate_limit_counter -q`

Expected: FAIL because `push_rate_limits` does not exist.

- [ ] **Step 3: Write the hash-only durable counter migration**

```sql
-- server/migrations/0004_rate_limits.sql
CREATE TABLE IF NOT EXISTS push_rate_limits (
    subject_hash BINARY(32) NOT NULL,
    window_started_at DATETIME(6) NOT NULL,
    request_count INT UNSIGNED NOT NULL DEFAULT 0,
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (subject_hash, window_started_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

SET @schema_name = DATABASE();
SET @statement = IF((SELECT COUNT(*) FROM information_schema.statistics WHERE table_schema = @schema_name AND table_name = 'push_rate_limits' AND index_name = 'idx_push_rate_limits_cleanup') = 0, 'CREATE INDEX idx_push_rate_limits_cleanup ON push_rate_limits (updated_at)', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;
```

- [ ] **Step 4: Run all migration tests against clean and unchanged schemas**

Run: `python -m pytest tests/backend/test_migrate.py -q`

Expected: PASS; all four migration contracts are present, the second execution is unchanged, and a copied applied file with different content is rejected.

- [ ] **Step 5: Commit the rate-limit schema boundary**

```bash
git add server/migrations/0004_rate_limits.sql tests/backend/test_migrate.py
git commit -m "feat: add durable rate limit migration"
```

### Task 8: Install CI, full-history secret scanning, and dependency review

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/security.yml`
- Create: `tests/backend/test_ci_configuration.py`

**Interfaces:**
- Consumes: Tasks 1–7 commands, Node 22, Python 3.12, and MySQL 8.4.
- Produces: required pull-request/push checks named `frontend`, `backend`, `gitleaks`, and `dependency-review`; all MySQL credentials are ephemeral CI configuration and workflow steps never print environment dictionaries.

- [ ] **Step 1: Write a failing contract test for workflow requirements**

```python
# tests/backend/test_ci_configuration.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_ci_workflows_include_phase_one_required_gates() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    security = (ROOT / ".github/workflows/security.yml").read_text(encoding="utf-8")
    for token in ("npm ci", "npm run lint", "npm run build", "npm run test:run", "npm run test:e2e", "ruff check server tests", "python -m pytest -q", "mysql:8.4"):
        assert token in ci
    for token in ("fetch-depth: 0", "zricethezav/gitleaks:v8.28.0", 'gitleaks git --redact --log-opts="--all"', "actions/dependency-review-action"):
        assert token in security
```

- [ ] **Step 2: Run the test to verify workflows do not exist**

Run: `python -m pytest tests/backend/test_ci_configuration.py -q`

Expected: FAIL with `FileNotFoundError` for `.github/workflows/ci.yml`.

- [ ] **Step 3: Create the test workflow with browser and MySQL service boundaries**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, hardening/reclive-security-data-trust]
  pull_request:

permissions:
  contents: read

jobs:
  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: npm
      - run: npm ci
      - run: npm run lint
      - run: npm run build
        env:
          VITE_API_BASE_URL: http://127.0.0.1:8000
          VITE_SITE_URL: http://127.0.0.1:4173
      - run: npm run test:run
      - run: npx playwright install --with-deps chromium
      - run: npm run test:e2e -- tests/e2e/route-smoke.spec.ts
        env:
          VITE_API_BASE_URL: http://127.0.0.1:8000
          VITE_SITE_URL: http://127.0.0.1:4173

  backend:
    runs-on: ubuntu-latest
    services:
      mysql:
        image: mysql:8.4
        env:
          MYSQL_DATABASE: reclive_test
          MYSQL_USER: reclive
          MYSQL_PASSWORD: reclive-ci-password
          MYSQL_ROOT_PASSWORD: root-ci-password
        ports:
          - 3306:3306
        options: >-
          --health-cmd="mysqladmin ping -h 127.0.0.1 -uroot -proot-ci-password"
          --health-interval=10s
          --health-timeout=5s
          --health-retries=10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -r server/requirements.txt -r server/requirements-dev.txt
      - run: ruff check server tests
      - run: python -m pytest -q
        env:
          TEST_MYSQL_HOST: 127.0.0.1
          TEST_MYSQL_PORT: "3306"
          TEST_MYSQL_USER: reclive
          TEST_MYSQL_PASSWORD: reclive-ci-password
          TEST_MYSQL_DATABASE: reclive_test
          TEST_MYSQL_ADMIN_USER: root
          TEST_MYSQL_ADMIN_PASSWORD: root-ci-password
```

- [ ] **Step 4: Create the least-privilege security workflow**

```yaml
# .github/workflows/security.yml
name: Security

on:
  push:
    branches: [main, hardening/reclive-security-data-trust]
  pull_request:

permissions:
  contents: read

jobs:
  gitleaks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 0
      - name: Scan all reachable history with redacted output
        run: docker run --rm -v "$PWD:/repo" zricethezav/gitleaks:v8.28.0 git --redact --log-opts="--all" /repo

  dependency-review:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: read
    steps:
      - uses: actions/checkout@v6
      - uses: actions/dependency-review-action@v4
```

- [ ] **Step 5: Run local static workflow-contract tests and the full backend suite**

Run: `python -m pytest tests/backend/test_ci_configuration.py -q && python -m pytest -q`

Expected: PASS; the text-level contract finds every required command and action, and all backend tests remain green.

- [ ] **Step 6: Commit the CI and security gates**

```bash
git add .github/workflows/ci.yml .github/workflows/security.yml tests/backend/test_ci_configuration.py
git commit -m "ci: add test and security workflows"
```

### Task 9: Configure weekly Dependabot coverage

**Files:**
- Create: `.github/dependabot.yml`
- Modify: `tests/backend/test_ci_configuration.py`

**Interfaces:**
- Consumes: GitHub's Dependabot configuration schema.
- Produces: weekly update checks for the root npm lockfile, `server/requirements*.txt` pip manifests, and GitHub Actions definitions.

- [ ] **Step 1: Extend the workflow configuration test with a failing Dependabot assertion**

```python
# tests/backend/test_ci_configuration.py: append this test
def test_dependabot_tracks_npm_pip_and_github_actions_weekly() -> None:
    dependabot = (ROOT / ".github/dependabot.yml").read_text(encoding="utf-8")
    for token in ('package-ecosystem: "npm"', 'package-ecosystem: "pip"', 'package-ecosystem: "github-actions"', 'interval: "weekly"'):
        assert token in dependabot
```

- [ ] **Step 2: Run the test to verify dependency automation is absent**

Run: `python -m pytest tests/backend/test_ci_configuration.py::test_dependabot_tracks_npm_pip_and_github_actions_weekly -q`

Expected: FAIL with `FileNotFoundError` for `.github/dependabot.yml`.

- [ ] **Step 3: Create exact weekly Dependabot configuration**

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
  - package-ecosystem: "pip"
    directory: "/server"
    schedule:
      interval: "weekly"
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

- [ ] **Step 4: Run the focused test and the complete Phase 1 local verification set**

Run: `python -m pytest tests/backend/test_ci_configuration.py -q && npm run lint && npm run build && npm run test:run && npm run test:e2e && ruff check server tests && python -m pytest -q && gitleaks git --redact --log-opts="--all" && git diff --check`

Expected: PASS; no whitespace errors, all unit/component/browser/backend/migration/configuration tests pass, Gitleaks scans all reachable history with redacted findings, and lint/build commands succeed. If MySQL, Chromium, or Gitleaks is unavailable locally, run all independent checks, record that exact command as unexecuted with its reason, and rely on CI for only that environment-specific check rather than claiming it passed.

- [ ] **Step 5: Commit the dependency automation configuration**

```bash
git add .github/dependabot.yml tests/backend/test_ci_configuration.py
git commit -m "ci: configure weekly dependency updates"
```

## Phase 1 Completion Check

- [ ] Confirm `server/migrations/` contains exactly `0001_core_history.sql`, `0002_snapshot_and_ingestion.sql`, `0003_push_rule_lifecycle.sql`, and `0004_rate_limits.sql`, in lexical order.
- [ ] Confirm `python server/migrate.py` is the documented migration command and has been tested on a clean MySQL 8.4 database and then against the unchanged database.
- [ ] Confirm every applied migration is recorded with its SHA-256 and a changed applied SQL file stops the runner before executing later work.
- [ ] Confirm no test file lies outside `src/**/*.test.ts(x)`, `tests/backend/`, or `tests/e2e/`, and reusable backend fixture data is only in `tests/fixtures/`.
- [ ] Confirm `.github/workflows/security.yml` checks out full history before Gitleaks and that workflows never echo environment variables or subscription payloads.
- [ ] Confirm only Phase 1 files changed, then report exact commands and actual results. Do not merge or deploy.
