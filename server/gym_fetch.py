from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any

import pymysql
import requests

from env_loader import load_project_dotenv
from facility_capacities import load_facility_capacities
from reclive.ingestion import failed_result, finish_ingestion_result, run_ingestion


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Missing required env var: {name}")

    normalized = value.strip()
    if not normalized:
        raise RuntimeError(f"Missing required env var: {name}")
    return normalized


def require_int_env(name: str) -> int:
    raw = require_env(name)
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer for env var {name}") from exc


LIVE_COUNTS_URL: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def db_connect() -> Any:
    return pymysql.connect(
        host=require_env("GYM_DB_HOST"),
        port=require_int_env("GYM_DB_PORT"),
        user=require_env("GYM_DB_USER"),
        password=require_env("GYM_DB_PASSWORD"),
        database=require_env("GYM_DB_NAME"),
        autocommit=False,
        charset="utf8mb4",
        connect_timeout=10,
        read_timeout=20,
        write_timeout=20,
    )


def fetch_live() -> object:
    url = LIVE_COUNTS_URL or require_env("LIVE_COUNTS_URL")
    response = requests.get(url, timeout=(5, 20))
    response.raise_for_status()
    return response.json()


def main() -> int:
    try:
        load_project_dotenv()
        capacities = load_facility_capacities()
    except Exception:
        finish_ingestion_result(
            failed_result("validation"), None, utc_now, print
        )
        return 1

    result = run_ingestion(fetch_live, db_connect, capacities, utc_now)
    return 0 if result.status == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(main())
