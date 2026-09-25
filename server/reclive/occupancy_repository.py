from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from reclive.actual_hours import HistoryState, IngestionHeartbeat
from reclive.ingestion import NormalizedLiveRow


@dataclass(frozen=True)
class IngestionWriteCounts:
    history_inserted: int
    snapshot_updated: int


@dataclass(frozen=True)
class SnapshotRow:
    location_id: int
    is_closed: bool
    current_capacity: int
    max_capacity: int
    source_updated_at: datetime | None
    fetched_at: datetime


@dataclass(frozen=True)
class LiveSnapshotRead:
    last_successful_fetch_at: datetime | None
    rows: list[SnapshotRow]


class SnapshotReadProtocol(Protocol):
    def fetch_live_snapshot_rows(self) -> list[SnapshotRow]: ...


class ActualHourReadProtocol(Protocol):
    def load_actual_hour_inputs(
        self,
        location_ids: Sequence[int],
        range_start: datetime,
        range_end: datetime,
    ) -> tuple[list[HistoryState], list[IngestionHeartbeat]]: ...


RepositoryFactory = Callable[[Any], "SnapshotRepository"]


def as_mysql_utc(value: datetime) -> datetime:
    """Validate an application UTC timestamp and make it a MySQL bind value."""
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
        or value.utcoffset() != timedelta(0)
    ):
        raise ValueError("timestamps must be aware UTC datetimes")
    return value.replace(tzinfo=None)


def to_aware_utc(value: datetime) -> datetime:
    """Interpret the UTC-configured database's naive DATETIME value as UTC."""
    if not isinstance(value, datetime) or value.tzinfo is not None:
        raise ValueError("database timestamps must be naive UTC datetimes")
    return value.replace(tzinfo=timezone.utc)


def state_fingerprint(row: SnapshotRow | NormalizedLiveRow) -> tuple[object, ...]:
    return (
        row.is_closed,
        row.current_capacity,
        row.max_capacity,
        row.source_updated_at,
    )


class SnapshotRepository:
    def __init__(self, connection: Any) -> None:
        get_autocommit = getattr(connection, "get_autocommit", None)
        if not callable(get_autocommit):
            raise TypeError("connection must expose get_autocommit()")
        if get_autocommit():
            raise ValueError("SnapshotRepository requires autocommit=False")
        self.connection = connection

    def start_run(self, started_at: datetime) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO ingestion_runs(started_at, status, observed_location_ids) "
                "VALUES (%s, 'running', JSON_ARRAY())",
                (as_mysql_utc(started_at),),
            )
            return int(cursor.lastrowid)

    def complete_success(
        self,
        run_id: int,
        completed_at: datetime,
        received_count: int,
        valid_count: int,
        counts: IngestionWriteCounts,
        observed_location_ids: Sequence[int],
    ) -> None:
        observed_ids = validated_location_ids(observed_location_ids)
        self.require_run_can_succeed(run_id)
        with self.connection.cursor() as cursor:
            cursor.execute(
                "UPDATE ingestion_runs SET completed_at=%s, status='succeeded', "
                "received_count=%s, valid_count=%s, history_inserted_count=%s, "
                "snapshot_updated_count=%s, observed_location_ids=%s "
                "WHERE id=%s AND status='running'",
                (
                    as_mysql_utc(completed_at),
                    received_count,
                    valid_count,
                    counts.history_inserted,
                    counts.snapshot_updated,
                    json.dumps(sorted(set(observed_ids))),
                    run_id,
                ),
            )
            self.require_running_row(cursor.rowcount)

    def complete_failure(
        self, run_id: int, completed_at: datetime, category: str, message: str
    ) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "UPDATE ingestion_runs SET completed_at=%s, status='failed', "
                "error_category=%s, error_message=%s WHERE id=%s AND status='running'",
                (as_mysql_utc(completed_at), category, message, run_id),
            )
            self.require_running_row(cursor.rowcount)

    def persist_successful_poll(
        self,
        run_id: int,
        rows: Sequence[NormalizedLiveRow],
        fetched_at: datetime,
    ) -> IngestionWriteCounts:
        mysql_fetched_at = as_mysql_utc(fetched_at)
        location_ids = validated_location_ids([row.location_id for row in rows])
        for row in rows:
            if row.source_updated_at is not None:
                as_mysql_utc(row.source_updated_at)

        requires_utc_baseline = self.require_run_can_succeed(run_id)
        previous = self.lock_snapshots(location_ids)
        inserted = 0
        for row in rows:
            self.upsert_snapshot(row, mysql_fetched_at)
            if requires_utc_baseline or previous.get(row.location_id) is None or (
                state_fingerprint(previous[row.location_id]) != state_fingerprint(row)
            ):
                self.insert_history(row, mysql_fetched_at)
                inserted += 1
        return IngestionWriteCounts(history_inserted=inserted, snapshot_updated=len(rows))

    def fetch_live_snapshot_rows(self) -> list[SnapshotRow]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT location_id, is_closed, current_capacity, max_capacity, "
                "source_updated_at, fetched_at FROM location_snapshot ORDER BY location_id"
            )
            return [self.snapshot_row_from_tuple(row) for row in cursor.fetchall()]

    def fetch_live_snapshot(self, now: datetime) -> LiveSnapshotRead:
        as_mysql_utc(now)
        rows = self.fetch_live_snapshot_rows()
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT completed_at FROM ingestion_runs "
                "WHERE status = 'succeeded' "
                "ORDER BY completed_at DESC, id DESC LIMIT 1"
            )
            latest = cursor.fetchone()
        completed_at = None
        if latest is not None and latest[0] is not None:
            completed_at = to_aware_utc(latest[0])
        return LiveSnapshotRead(
            last_successful_fetch_at=completed_at,
            rows=rows,
        )

    def load_actual_hour_inputs(
        self,
        location_ids: Sequence[int],
        range_start: datetime,
        range_end: datetime,
    ) -> tuple[list[HistoryState], list[IngestionHeartbeat]]:
        ids = validated_location_ids(location_ids)
        if not ids:
            return [], []

        range_start_bind = as_mysql_utc(range_start)
        range_end_bind = as_mysql_utc(range_end)
        if range_end <= range_start:
            raise ValueError("actual-hour range must end after it starts")

        placeholders = ", ".join("%s" for _ in ids)
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT started_at FROM ingestion_runs "
                "WHERE status='succeeded' ORDER BY started_at, id LIMIT 1"
            )
            first_success = cursor.fetchone()
            if first_success is None:
                return [], []

            cutover_started_at = to_aware_utc(first_success[0])
            cutover_bind = as_mysql_utc(cutover_started_at)
            cursor.execute(
                "SELECT h.location_id, h.is_closed, h.current_capacity, "
                "h.max_capacity, h.fetched_at, h.id "
                "FROM location_history AS h "
                "INNER JOIN ("
                "SELECT id, ROW_NUMBER() OVER ("
                "PARTITION BY location_id ORDER BY fetched_at DESC, id DESC"
                ") AS seed_rank FROM location_history "
                f"WHERE location_id IN ({placeholders}) "
                "AND fetched_at >= %s AND fetched_at < %s"
                ") AS seed ON seed.id = h.id "
                "WHERE seed.seed_rank = 1 ORDER BY h.location_id",
                (*ids, cutover_bind, range_start_bind),
            )
            seed_rows = cursor.fetchall()
            cursor.execute(
                "SELECT location_id, is_closed, current_capacity, max_capacity, "
                "fetched_at, id FROM location_history "
                f"WHERE location_id IN ({placeholders}) "
                "AND fetched_at >= GREATEST(%s, %s) AND fetched_at < %s "
                "ORDER BY location_id, fetched_at, id",
                (*ids, cutover_bind, range_start_bind, range_end_bind),
            )
            change_rows = cursor.fetchall()
            cursor.execute(
                "SELECT completed_at, observed_location_ids FROM ingestion_runs "
                "WHERE status='succeeded' AND completed_at >= %s "
                "AND completed_at <= %s ORDER BY completed_at, id",
                (range_start_bind, range_end_bind),
            )
            heartbeat_rows = cursor.fetchall()

        return (
            parse_history_states((*seed_rows, *change_rows)),
            parse_ingestion_heartbeats(heartbeat_rows),
        )

    def lock_snapshots(self, location_ids: Sequence[int]) -> dict[int, SnapshotRow]:
        ids = validated_location_ids(location_ids)
        if not ids:
            return {}
        placeholders = ", ".join("%s" for _ in ids)
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT location_id, is_closed, current_capacity, max_capacity, "
                "source_updated_at, fetched_at FROM location_snapshot "
                f"WHERE location_id IN ({placeholders}) FOR UPDATE",
                tuple(ids),
            )
            return {
                row.location_id: row
                for row in (self.snapshot_row_from_tuple(value) for value in cursor.fetchall())
            }

    def require_run_can_succeed(self, run_id: int) -> bool:
        running_started_at, first_success = self.lock_cutover_runs(run_id)
        if first_success is None:
            return True
        if (running_started_at, run_id) < (first_success[1], first_success[0]):
            raise RuntimeError(
                "running ingestion run predates the first succeeded run"
            )
        return False

    def lock_cutover_runs(
        self, run_id: int
    ) -> tuple[datetime, tuple[int, datetime] | None]:
        if type(run_id) is not int or run_id < 1:
            raise ValueError("run ID must be a positive integer")
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, status, started_at FROM ingestion_runs "
                "FORCE INDEX (PRIMARY) ORDER BY id FOR UPDATE"
            )
            rows = cursor.fetchall()
        runs = {
            int(row[0]): (str(row[1]), to_aware_utc(row[2]))
            for row in rows
            if len(row) == 3 and isinstance(row[2], datetime)
        }
        running_run = runs.get(run_id)
        if running_run is None or running_run[0] != "running":
            raise RuntimeError("expected exactly one running ingestion run")
        succeeded_runs = [
            (candidate_run_id, started_at)
            for candidate_run_id, (status, started_at) in runs.items()
            if status == "succeeded"
        ]
        return (
            running_run[1],
            min(succeeded_runs, key=lambda row: (row[1], row[0]), default=None),
        )

    def upsert_snapshot(self, row: NormalizedLiveRow, fetched_at: datetime) -> None:
        source_updated_at = mysql_optional_utc(row.source_updated_at)
        with self.connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO location_snapshot(location_id, is_closed, current_capacity, "
                "max_capacity, source_updated_at, fetched_at, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE "
                "is_closed=VALUES(is_closed), current_capacity=VALUES(current_capacity), "
                "max_capacity=VALUES(max_capacity), "
                "source_updated_at=VALUES(source_updated_at), fetched_at=VALUES(fetched_at), "
                "updated_at=VALUES(updated_at)",
                (
                    row.location_id,
                    row.is_closed,
                    row.current_capacity,
                    row.max_capacity,
                    source_updated_at,
                    fetched_at,
                    fetched_at,
                    fetched_at,
                ),
            )

    def insert_history(self, row: NormalizedLiveRow, fetched_at: datetime) -> None:
        source_updated_at = mysql_optional_utc(row.source_updated_at)
        with self.connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO location_history(location_id, is_closed, current_capacity, "
                "max_capacity, source_updated_at, last_updated, fetched_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    row.location_id,
                    row.is_closed,
                    row.current_capacity,
                    row.max_capacity,
                    source_updated_at,
                    source_updated_at,
                    fetched_at,
                ),
            )

    @staticmethod
    def require_running_row(rowcount: int) -> None:
        if rowcount != 1:
            raise RuntimeError("expected exactly one running ingestion run")

    @staticmethod
    def snapshot_row_from_tuple(values: Sequence[object]) -> SnapshotRow:
        (
            location_id,
            is_closed,
            current_capacity,
            max_capacity,
            source_updated_at,
            fetched_at,
        ) = values
        if not isinstance(fetched_at, datetime):
            raise ValueError("snapshot fetched_at must be a database datetime")
        if source_updated_at is not None and not isinstance(source_updated_at, datetime):
            raise ValueError("snapshot source_updated_at must be a database datetime")
        return SnapshotRow(
            location_id=int(location_id),
            is_closed=bool(is_closed),
            current_capacity=int(current_capacity),
            max_capacity=int(max_capacity),
            source_updated_at=(
                to_aware_utc(source_updated_at)
                if source_updated_at is not None
                else None
            ),
            fetched_at=to_aware_utc(fetched_at),
        )


def mysql_optional_utc(value: datetime | None) -> datetime | None:
    return as_mysql_utc(value) if value is not None else None


def validated_location_ids(location_ids: Sequence[int]) -> tuple[int, ...]:
    ids = tuple(location_ids)
    if any(type(location_id) is not int or location_id < 0 for location_id in ids):
        raise ValueError("location IDs must be non-negative integers")
    return ids


def parse_history_states(rows: Sequence[Sequence[object]]) -> list[HistoryState]:
    states: list[HistoryState] = []
    for values in rows:
        if len(values) != 6:
            continue
        (
            location_id,
            is_closed,
            current_capacity,
            max_capacity,
            fetched_at,
            event_id,
        ) = values
        if (
            not _is_positive_integer(location_id)
            or not _is_database_boolean(is_closed)
            or not _is_non_negative_integer(current_capacity)
            or not _is_non_negative_integer(max_capacity)
            or not isinstance(fetched_at, datetime)
            or fetched_at.tzinfo is not None
            or not _is_positive_integer(event_id)
        ):
            continue
        states.append(
            HistoryState(
                location_id=location_id,
                is_closed=bool(is_closed),
                count=current_capacity,
                capacity=max_capacity,
                fetched_at=to_aware_utc(fetched_at),
                id=event_id,
            )
        )
    return states


def parse_ingestion_heartbeats(
    rows: Sequence[Sequence[object]],
) -> list[IngestionHeartbeat]:
    heartbeats: list[IngestionHeartbeat] = []
    for values in rows:
        if len(values) != 2:
            continue
        completed_at, observed_location_ids = values
        if not isinstance(completed_at, datetime) or completed_at.tzinfo is not None:
            continue
        heartbeats.append(
            IngestionHeartbeat(
                completed_at=to_aware_utc(completed_at),
                observed_location_ids=parse_observed_location_ids(
                    observed_location_ids
                ),
            )
        )
    return heartbeats


def parse_observed_location_ids(value: object) -> frozenset[int]:
    if isinstance(value, (str, bytes, bytearray)):
        try:
            decoded = json.loads(value)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
            return frozenset()
    else:
        decoded = value
    if not isinstance(decoded, list):
        return frozenset()
    return frozenset(
        location_id
        for location_id in decoded
        if _is_positive_integer(location_id)
    )


def _is_positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def _is_non_negative_integer(value: object) -> bool:
    return type(value) is int and value >= 0


def _is_database_boolean(value: object) -> bool:
    return type(value) is bool or (type(value) is int and value in (0, 1))
