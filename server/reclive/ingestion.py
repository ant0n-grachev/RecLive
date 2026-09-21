from __future__ import annotations

import re
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import requests
import pymysql

from server.facility_capacities import load_facility_capacities
from server.reclive.settings import Settings, validate_command_environment
from server.reclive.observability import format_event


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
    except ValueError:
        raise RuntimeError(f"Invalid integer for env var {name}") from None


LIVE_COUNTS_URL: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def db_connect(settings: Settings | None = None) -> Any:
    from server.reclive.db import open_db_connection
    from server.reclive.settings import DatabaseSettings

    database = (
        settings.database
        if settings is not None
        else DatabaseSettings(
            host=require_env("GYM_DB_HOST"),
            port=require_int_env("GYM_DB_PORT"),
            user=require_env("GYM_DB_USER"),
            password=require_env("GYM_DB_PASSWORD"),
            name=require_env("GYM_DB_NAME"),
        )
    )
    for name, value in (
        ("GYM_DB_HOST", database.host),
        ("GYM_DB_PORT", database.port),
        ("GYM_DB_USER", database.user),
        ("GYM_DB_PASSWORD", database.password),
        ("GYM_DB_NAME", database.name),
    ):
        if value is None or not str(value).strip():
            raise RuntimeError(f"Missing required env var: {name}")
    return open_db_connection(database, autocommit=False)


def fetch_live(settings: Settings | None = None) -> object:
    url = LIVE_COUNTS_URL or (
        settings.live_counts_url
        if settings is not None
        else require_env("LIVE_COUNTS_URL")
    )
    if not url:
        raise RuntimeError("Missing required env var: LIVE_COUNTS_URL")
    response = requests.get(url, timeout=(5, 20))
    response.raise_for_status()
    return response.json()


def run_configured_ingestion(
    settings: Settings,
    *,
    fetch_payload: Callable[[], object] | None = None,
    connect: Callable[[], Any] | None = None,
    now: Callable[[], datetime] | None = None,
    repository_factory: Callable[[Any], Any] | None = None,
    event_sink: Callable[[str], None] = print,
) -> IngestionRunResult:
    """Compose the existing transaction owner from captured command inputs."""
    validate_command_environment(
        settings,
        (
            "LIVE_COUNTS_URL",
            "GYM_DB_HOST",
            "GYM_DB_PORT",
            "GYM_DB_USER",
            "GYM_DB_PASSWORD",
            "GYM_DB_NAME",
        ),
    )
    capacities = settings.capacities
    if capacities is None:
        capacities = load_facility_capacities(settings.capacity_config_path)
    return run_ingestion(
        fetch_payload if fetch_payload is not None else lambda: fetch_live(settings),
        connect if connect is not None else lambda: db_connect(settings),
        capacities,
        now if now is not None else utc_now,
        repository_factory=repository_factory,
        event_sink=event_sink,
    )


ALLOWED_ERROR_CATEGORIES = frozenset(
    {"network", "http", "payload_not_list", "validation", "database", "transaction"}
)
SAFE_DETAIL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .,:;()_-]*$")
UNSAFE_DETAIL_PATTERN = re.compile(
    r"(?:https?://|www\.|@|\b(?:api[_ -]?key|authorization|cookie|credential|"
    r"password|secret|token)\b|=)",
    re.IGNORECASE,
)
DECIMAL_INTEGER_PATTERN = re.compile(r"^[0-9]+$")
# Upstream decimal identifiers and occupancy counts fit within a normal 32-bit field.
MAX_DECIMAL_DIGITS = 10


class IngestionValidationError(ValueError):
    """A safe validation error for an incompatible live-feed payload."""


class _NoValidRowsError(IngestionValidationError):
    """Internal signal for a decoded payload with no usable observations."""


@dataclass(frozen=True)
class NormalizedLiveRow:
    location_id: int
    is_closed: bool
    current_capacity: int
    max_capacity: int
    source_updated_at: datetime | None


@dataclass(frozen=True)
class ValidationResult:
    received_count: int
    invalid_count: int
    rows: Sequence[NormalizedLiveRow]


@dataclass(frozen=True)
class IngestionRunResult:
    status: Literal["succeeded", "failed"]
    received_count: int
    valid_count: int
    history_inserted: int
    snapshot_updated: int
    error_category: str | None


def validate_and_deduplicate_rows(
    payload: object, capacities: Mapping[int, int]
) -> ValidationResult:
    if not isinstance(payload, list):
        raise IngestionValidationError(
            "payload_not_list", "Live feed payload is not a list"
        )

    retained: dict[int, tuple[int, NormalizedLiveRow]] = {}
    invalid_count = 0
    for index, raw in enumerate(payload):
        row = parse_live_row(raw, capacities)
        if row is None:
            invalid_count += 1
            continue
        previous = retained.get(row.location_id)
        if previous is None or source_timestamp_sort_key(
            row.source_updated_at, index
        ) >= source_timestamp_sort_key(previous[1].source_updated_at, previous[0]):
            retained[row.location_id] = (index, row)

    return ValidationResult(
        received_count=len(payload),
        invalid_count=invalid_count,
        rows=tuple(
            row
            for _, row in sorted(
                retained.values(), key=lambda item: item[1].location_id
            )
        ),
    )


def parse_live_row(
    raw: object, capacities: Mapping[int, int]
) -> NormalizedLiveRow | None:
    if not isinstance(raw, Mapping):
        return None

    location_id = parse_decimal_integer(raw.get("LocationId"))
    if location_id is None:
        return None
    max_capacity = capacities.get(location_id)
    if not is_positive_integer(max_capacity):
        return None

    is_closed = raw.get("IsClosed")
    if type(is_closed) is not bool:
        return None
    current_capacity = parse_decimal_integer(raw.get("LastCount"))
    if current_capacity is None:
        return None
    source_updated_at = parse_source_timestamp(raw.get("LastUpdatedDateAndTime"))
    if source_updated_at is _INVALID_TIMESTAMP:
        return None

    return NormalizedLiveRow(
        location_id=location_id,
        is_closed=is_closed,
        current_capacity=current_capacity,
        max_capacity=max_capacity,
        source_updated_at=source_updated_at,
    )


def parse_decimal_integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if (
        isinstance(value, str)
        and len(value) <= MAX_DECIMAL_DIGITS
        and DECIMAL_INTEGER_PATTERN.fullmatch(value)
    ):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def is_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


_INVALID_TIMESTAMP = object()
NAIVE_SOURCE_TIMESTAMP_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,9})?"
)


def parse_source_timestamp(value: object) -> datetime | None | object:
    if value is None:
        return None
    if not isinstance(value, str):
        return _INVALID_TIMESTAMP
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return _INVALID_TIMESTAMP
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        # A valid local clock value cannot establish an instant. Preserve the
        # observation with an unknown source time; fetched_at records receipt.
        if NAIVE_SOURCE_TIMESTAMP_PATTERN.fullmatch(value):
            return None
        return _INVALID_TIMESTAMP
    return parsed.astimezone(timezone.utc)


def source_timestamp_sort_key(
    source_updated_at: datetime | None, index: int
) -> tuple[int, datetime, int]:
    if source_updated_at is None:
        return (0, datetime.min.replace(tzinfo=None), index)
    return (1, source_updated_at, index)


def sanitize_ingestion_error(category: str, detail: object) -> tuple[str, str]:
    safe_category = category if category in ALLOWED_ERROR_CATEGORIES else "validation"
    if not isinstance(detail, str):
        return safe_category, "Ingestion failure"

    message = " ".join(detail.split())
    if (
        not message
        or not SAFE_DETAIL_PATTERN.fullmatch(message)
        or UNSAFE_DETAIL_PATTERN.search(message)
    ):
        return safe_category, "Ingestion failure"
    return safe_category, message[:240]


def classify_ingestion_exception(exc: BaseException) -> str:
    if isinstance(exc, _NoValidRowsError):
        return "validation"
    if isinstance(exc, IngestionValidationError):
        return "payload_not_list"
    if isinstance(exc, requests.HTTPError):
        return "http"
    if isinstance(exc, pymysql.MySQLError):
        return "database"
    if isinstance(exc, (requests.RequestException, ConnectionError, TimeoutError)):
        return "network"
    return "network"


def sanitize_ingestion_exception(exc: BaseException) -> tuple[str, str]:
    return sanitize_ingestion_error(classify_ingestion_exception(exc), exc)


def require_aware_utc(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
        or value.utcoffset() != timedelta(0)
    ):
        raise ValueError("timestamps must be aware UTC datetimes")
    return value


def run_ingestion(
    fetch_payload: Callable[[], object],
    connect: Callable[[], Any],
    capacities: Mapping[int, int],
    now: Callable[[], datetime],
    repository_factory: Callable[[Any], Any] | None = None,
    event_sink: Callable[[str], object] = print,
) -> IngestionRunResult:
    run_connection: Any | None = None
    work_connection: Any | None = None
    run_id: int | None = None
    run_committed = False
    started_at: datetime | None = None

    try:
        started_at = require_aware_utc(now())
    except Exception:
        return finish_ingestion_result(
            failed_result("validation"), started_at, now, event_sink
        )

    try:
        factory = (
            repository_factory
            if repository_factory is not None
            else default_repository_factory()
        )
    except Exception:
        return finish_ingestion_result(
            failed_result("database"), started_at, now, event_sink
        )

    initial_stage = "database"
    try:
        run_connection = connect()
        run_repository = factory(run_connection)
        run_id = run_repository.start_run(started_at)
        initial_stage = "transaction"
        run_connection.commit()
        run_committed = True
    except Exception:
        safe_rollback(run_connection)
        safe_close(run_connection)
        return finish_ingestion_result(
            failed_result(initial_stage), started_at, now, event_sink
        )

    if not safe_close(run_connection):
        result = failed_result("transaction")
        record_failed_run(
            connect,
            factory,
            run_id,
            "transaction",
            "Ingestion failure",
            now,
        )
        return finish_ingestion_result(result, started_at, now, event_sink)
    run_connection = None

    received_count = 0
    valid_count = 0
    work_stage = "database"
    try:
        work_connection = connect()
        repository = factory(work_connection)

        work_stage = "fetch"
        payload = fetch_payload()

        work_stage = "validation"
        validated = validate_and_deduplicate_rows(payload, capacities)
        received_count = validated.received_count
        valid_count = len(validated.rows)
        if not validated.rows:
            raise _NoValidRowsError()

        work_stage = "database"
        fetched_at = require_aware_utc(now())
        counts = repository.persist_successful_poll(
            run_id, validated.rows, fetched_at
        )
        repository.complete_success(
            run_id,
            require_aware_utc(now()),
            received_count,
            valid_count,
            counts,
            [row.location_id for row in validated.rows],
        )

        work_stage = "transaction"
        work_connection.commit()
        result = IngestionRunResult(
            status="succeeded",
            received_count=received_count,
            valid_count=valid_count,
            history_inserted=counts.history_inserted,
            snapshot_updated=counts.snapshot_updated,
            error_category=None,
        )
    except Exception as exc:
        safe_rollback(work_connection)
        work_released = safe_close(work_connection)
        work_connection = None
        if isinstance(exc, _NoValidRowsError):
            category, message = sanitize_ingestion_error(
                "validation", "No valid live rows were received"
            )
        elif work_stage in {"fetch", "validation"}:
            category, message = sanitize_ingestion_exception(exc)
        else:
            category, message = sanitize_ingestion_error(work_stage, exc)
        if run_committed and work_released:
            record_failed_run(
                connect, factory, run_id, category, message, now
            )
        result = failed_result(
            category,
            received_count=received_count,
            valid_count=valid_count,
        )
    finally:
        safe_close(work_connection)

    return finish_ingestion_result(result, started_at, now, event_sink)


def default_repository_factory() -> Callable[[Any], Any]:
    from reclive.occupancy_repository import SnapshotRepository

    return SnapshotRepository


def record_failed_run(
    connect: Callable[[], Any],
    repository_factory: Callable[[Any], Any],
    run_id: int | None,
    category: str,
    message: str,
    now: Callable[[], datetime],
) -> None:
    if run_id is None:
        return
    failure_connection: Any | None = None
    try:
        failure_connection = connect()
        repository = repository_factory(failure_connection)
        repository.complete_failure(
            run_id,
            require_aware_utc(now()),
            category,
            message,
        )
        failure_connection.commit()
    except Exception:
        safe_rollback(failure_connection)
    finally:
        safe_close(failure_connection)


def failed_result(
    category: str,
    *,
    received_count: int = 0,
    valid_count: int = 0,
) -> IngestionRunResult:
    safe_category, _ = sanitize_ingestion_error(category, object())
    return IngestionRunResult(
        status="failed",
        received_count=received_count,
        valid_count=valid_count,
        history_inserted=0,
        snapshot_updated=0,
        error_category=safe_category,
    )


def finish_ingestion_result(
    result: IngestionRunResult,
    started_at: datetime | None,
    now: Callable[[], datetime],
    event_sink: Callable[[str], object],
) -> IngestionRunResult:
    _duration_ms = 0
    if started_at is not None:
        try:
            finished_at = require_aware_utc(now())
            _duration_ms = max(
                0, int((finished_at - started_at).total_seconds() * 1000)
            )
        except Exception:
            _duration_ms = 0
    try:
        if result.status == "succeeded":
            line = format_event(
                "ingestion.completed",
                receivedCount=result.received_count,
                historyInsertedCount=result.history_inserted,
                unchangedCount=max(0, result.snapshot_updated - result.history_inserted),
            )
        else:
            category = {
                "network": "network_error", "http": "network_error",
                "payload_not_list": "validation_error", "validation": "validation_error",
                "database": "database_unavailable", "transaction": "database_unavailable",
            }.get(result.error_category, "validation_error")
            line = format_event("ingestion.failed", errorCategory=category)
        event_sink(line)
    except Exception:
        pass
    return result


def safe_rollback(connection: Any | None) -> None:
    if connection is None:
        return
    try:
        connection.rollback()
    except Exception:
        pass


def safe_close(connection: Any | None) -> bool:
    if connection is None:
        return True
    try:
        connection.close()
    except Exception:
        return False
    return True
