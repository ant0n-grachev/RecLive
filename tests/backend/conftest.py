import os
import re
from collections.abc import Iterator
from pathlib import Path
import sys

import pymysql
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reclive.ingestion import validate_and_deduplicate_rows  # noqa: E402
from tests.fixtures.live_counts import LIVE_ROWS  # noqa: E402
from tests.fixtures.reclive_fakes import (  # noqa: E402
    FakeActualHourRepository,
    FakeConnection,
)


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
    from datetime import datetime, timezone

    return datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)


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
