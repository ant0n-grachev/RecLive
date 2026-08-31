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
