import os
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Callable

import pymysql
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reclive.ingestion import validate_and_deduplicate_rows  # noqa: E402
from reclive.migrations import MigrationSettings, run_migrations  # noqa: E402
from tests.fixtures.live_counts import LIVE_ROWS  # noqa: E402
from tests.fixtures.reclive_fakes import (  # noqa: E402
    FakeActualHourRepository,
    FakeConnection,
)


FIXED_PUSH_NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
VALID_PUSH_P256DH = (
    "BGsX0fLhLEJH-Lzm5WOkQPJ3A32BLeszoPShOUXYmMKWT-NC4v4af5uO5-tKfA-"
    "eFivOM1drMV7Oy7ZAaDe_UfU"
)
VALID_PUSH_AUTH = "A" * 22


class PushTestRepository:
    """Small push seam that implements only the migrated rate-limit table."""

    def __init__(self) -> None:
        self.statuses: dict[int, str] = {}
        self.rate_limit_counts: dict[tuple[bytes, datetime], int] = {}
        self.rate_limit_increments = 0
        self.rate_limit_commits = 0
        self.rate_limit_rollbacks = 0
        self.fail_rate_limit_store = False

    def open_connection(self, **kwargs: object) -> object:
        if kwargs != {"autocommit": False}:
            raise AssertionError(f"unexpected push database settings: {kwargs}")
        return PushRateLimitConnection(self)

    def rule_status(self, rule_id: int) -> str:
        return self.statuses[rule_id]

    @property
    def committed_rate_limit_total(self) -> int:
        return sum(self.rate_limit_counts.values())


class PushRateLimitCursor:
    def __init__(self, connection: "PushRateLimitConnection") -> None:
        self.connection = connection
        self._row: tuple[int] | None = None

    def __enter__(self) -> "PushRateLimitCursor":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...]) -> None:
        if self.connection.repository.fail_rate_limit_store:
            raise RuntimeError("database-secret-sentinel")

        normalized = " ".join(sql.split()).lower()
        if normalized.startswith("insert into push_rate_limits"):
            subject_hash, window_started_at, _updated_at = params
            if not isinstance(subject_hash, bytes) or not isinstance(
                window_started_at, datetime
            ):
                raise AssertionError("invalid fixed-window counter parameters")
            key = (subject_hash, window_started_at)
            count = self.connection.repository.rate_limit_counts.get(key, 0) + 1
            self.connection.pending[key] = count
            self.connection.repository.rate_limit_increments += 1
            self._row = None
            return

        if normalized.startswith("select request_count from push_rate_limits"):
            subject_hash, window_started_at = params
            key = (subject_hash, window_started_at)
            count = self.connection.pending.get(
                key,
                self.connection.repository.rate_limit_counts.get(key, 0),
            )
            self._row = (count,)
            return

        raise AssertionError(f"unexpected push SQL: {normalized}")

    def fetchone(self) -> tuple[int] | None:
        return self._row


class PushRateLimitConnection:
    def __init__(self, repository: PushTestRepository) -> None:
        self.repository = repository
        self.pending: dict[tuple[bytes, datetime], int] = {}
        self.closed = False

    def cursor(self) -> PushRateLimitCursor:
        return PushRateLimitCursor(self)

    def commit(self) -> None:
        self.repository.rate_limit_counts.update(self.pending)
        self.pending.clear()
        self.repository.rate_limit_commits += 1

    def rollback(self) -> None:
        self.pending.clear()
        self.repository.rate_limit_rollbacks += 1

    def close(self) -> None:
        self.closed = True


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


@pytest.fixture()
def fake_db() -> FakeConnection:
    return FakeConnection()


@pytest.fixture()
def fixed_utc_clock():
    return datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def valid_subscription() -> dict[str, object]:
    return {
        "endpoint": "https://push.reclive-notify.net/subscription-a",
        "keys": {
            "p256dh": VALID_PUSH_P256DH,
            "auth": VALID_PUSH_AUTH,
        },
    }


@pytest.fixture()
def other_subscription() -> dict[str, object]:
    return {
        "endpoint": "https://push.reclive-notify.net/subscription-b",
        "keys": {
            "p256dh": VALID_PUSH_P256DH,
            "auth": VALID_PUSH_AUTH,
        },
    }


@pytest.fixture(scope="module")
def migrated_push_database() -> Iterator[dict[str, object]]:
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
    previous_cutover = os.environ.get("PUSH_RULE_SCHEMA_CUTOVER_READY")
    previous_hash_key = os.environ.get("PUSH_ENDPOINT_HASH_KEY")
    os.environ["PUSH_RULE_SCHEMA_CUTOVER_READY"] = "1"
    os.environ["PUSH_ENDPOINT_HASH_KEY"] = (
        "push-test-key-with-at-least-thirty-two-bytes"
    )
    connection = pymysql.connect(**admin_settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS `{database}`")
            cursor.execute(
                f"CREATE DATABASE `{database}` CHARACTER SET utf8mb4 "
                "COLLATE utf8mb4_0900_ai_ci"
            )
        run_migrations(
            MigrationSettings(
                host=str(settings["host"]),
                port=int(settings["port"]),
                user=str(settings["user"]),
                password=str(settings["password"]),
                database=database,
                lock_timeout_seconds=5,
            ),
            ROOT / "server" / "migrations",
        )
        yield settings
    finally:
        with connection.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS `{database}`")
        connection.close()
        if previous_cutover is None:
            os.environ.pop("PUSH_RULE_SCHEMA_CUTOVER_READY", None)
        else:
            os.environ["PUSH_RULE_SCHEMA_CUTOVER_READY"] = previous_cutover
        if previous_hash_key is None:
            os.environ.pop("PUSH_ENDPOINT_HASH_KEY", None)
        else:
            os.environ["PUSH_ENDPOINT_HASH_KEY"] = previous_hash_key


@pytest.fixture()
def mysql_push_test_client(
    monkeypatch: pytest.MonkeyPatch,
    migrated_push_database: dict[str, object],
):
    from fastapi.testclient import TestClient

    import forecast_api

    settings = migrated_push_database
    connection = pymysql.connect(**settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM push_rate_limits")
            cursor.execute("DELETE FROM push_rules")
    finally:
        connection.close()

    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv(
        "PUSH_ENDPOINT_HASH_KEY",
        "push-test-key-with-at-least-thirty-two-bytes",
    )
    monkeypatch.setenv("PUSH_EVALUATOR_ENABLED", "false")
    monkeypatch.setenv("PUSH_ADMIN_ROUTES_ENABLED", "false")
    monkeypatch.setenv("GYM_DB_HOST", str(settings["host"]))
    monkeypatch.setenv("GYM_DB_PORT", str(settings["port"]))
    monkeypatch.setenv("GYM_DB_USER", str(settings["user"]))
    monkeypatch.setenv("GYM_DB_PASSWORD", str(settings["password"]))
    monkeypatch.setenv("GYM_DB_NAME", str(settings["database"]))
    monkeypatch.setattr(
        forecast_api,
        "now_utc",
        lambda: FIXED_PUSH_NOW,
        raising=False,
    )

    with TestClient(forecast_api.app) as client:
        yield client


@pytest.fixture()
def push_repository() -> PushTestRepository:
    return PushTestRepository()


@pytest.fixture()
def db_rule_status(
    push_repository: PushTestRepository,
) -> Callable[[int], str]:
    return push_repository.rule_status


@pytest.fixture()
def push_test_client(
    monkeypatch: pytest.MonkeyPatch,
    push_repository: PushTestRepository,
):
    from fastapi.testclient import TestClient

    import forecast_api

    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv(
        "PUSH_ENDPOINT_HASH_KEY",
        "push-test-key-with-at-least-thirty-two-bytes",
    )
    monkeypatch.setenv("PUSH_EVALUATOR_ENABLED", "false")
    monkeypatch.setenv("PUSH_ADMIN_ROUTES_ENABLED", "false")
    monkeypatch.setattr(
        forecast_api,
        "open_db_connection",
        push_repository.open_connection,
    )
    monkeypatch.setattr(forecast_api, "now_utc", lambda: FIXED_PUSH_NOW, raising=False)

    with TestClient(forecast_api.app) as client:
        yield client


@pytest.fixture()
def normalized_live_rows():
    return validate_and_deduplicate_rows(LIVE_ROWS, {5761: 100}).rows


@pytest.fixture()
def actual_hour_repository() -> FakeActualHourRepository:
    return FakeActualHourRepository()


@pytest.fixture()
def actual_hours_forecast_payload() -> dict[str, object]:
    dates = ("2026-08-31", "2026-03-08", "2026-11-01")
    return {
        "facilities": [
            {
                "facilityId": 1186,
                "facilityName": "Nicholas Recreation Center",
                "weeklyForecast": [
                    {
                        "date": date_key,
                        "categories": [
                            {
                                "key": "fitness floors",
                                "title": "Fitness Floors",
                                "maxCapacity": 200,
                                "hours": [
                                    {
                                        "hourStart": (
                                            f"{date_key}T12:00:00-05:00"
                                        )
                                    }
                                ],
                            }
                        ],
                        "totalHours": [],
                    }
                    for date_key in dates
                ],
            }
        ]
    }


@pytest.fixture()
def actual_hours_client(
    monkeypatch: pytest.MonkeyPatch,
    actual_hour_repository: FakeActualHourRepository,
    actual_hours_forecast_payload: dict[str, object],
):
    from fastapi.testclient import TestClient

    import forecast_api

    monkeypatch.setattr(
        forecast_api,
        "load_forecast",
        lambda: actual_hours_forecast_payload,
    )
    monkeypatch.setattr(
        forecast_api,
        "SECTION_IDS",
        {
            1186: {
                "overall": [5761, 5762],
                "fitness floors": [5761, 5762],
            }
        },
    )
    monkeypatch.setattr(forecast_api, "MAX_CAP", {5761: 100, 5762: 100})
    monkeypatch.setattr(forecast_api, "ACTUAL_HOUR_MIN_COVERAGE", 0.75)

    def reject_connection(**kwargs: object) -> object:
        raise AssertionError(f"injected actual-hour route opened a database: {kwargs}")

    monkeypatch.setattr(forecast_api, "open_db_connection", reject_connection)
    dependency = getattr(forecast_api, "get_actual_hour_repository", None)
    if dependency is not None:
        forecast_api.app.dependency_overrides[dependency] = (
            lambda: actual_hour_repository
        )

    try:
        yield TestClient(forecast_api.app)
    finally:
        if dependency is not None:
            forecast_api.app.dependency_overrides.pop(dependency, None)
