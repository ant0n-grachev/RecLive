from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class RecordedQuery:
    statement: str
    parameters: tuple[Any, ...]


class FakeCursor:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.lastrowid = 0
        self.rowcount = 0
        self._rows: list[tuple[Any, ...]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False

    def execute(self, statement: str, parameters: tuple[Any, ...] = ()) -> int:
        bound = tuple(parameters)
        self.connection.queries.append(RecordedQuery(statement, bound))
        self._rows = []
        self.rowcount = 0
        normalized = " ".join(statement.split())

        if normalized.startswith(
            "SELECT location_id, is_closed, current_capacity"
        ) and "WHERE location_id IN" in normalized:
            selected_ids = {int(value) for value in bound}
            self._rows = [
                self.connection.snapshots[location_id]
                for location_id in sorted(selected_ids)
                if location_id in self.connection.snapshots
            ]
        elif normalized.startswith("SELECT id, status, started_at FROM ingestion_runs"):
            self._rows = [
                (run_id, run["status"], run["started_at"])
                for run_id, run in self.connection.run_rows.items()
                if self.connection.succeeded_run_count or run["status"] != "succeeded"
            ]
        elif normalized.startswith("SELECT location_id, is_closed"):
            self._rows = [
                self.connection.snapshots[location_id]
                for location_id in sorted(self.connection.snapshots)
            ]
        elif normalized.startswith("SELECT completed_at FROM ingestion_runs"):
            succeeded = [
                (run_id, run["completed_at"])
                for run_id, run in self.connection.run_rows.items()
                if run["status"] == "succeeded" and run.get("completed_at") is not None
            ]
            if succeeded:
                _, completed_at = max(succeeded, key=lambda row: (row[1], row[0]))
                self._rows = [(completed_at,)]
        elif normalized.startswith("INSERT INTO location_snapshot"):
            (
                location_id,
                is_closed,
                current_capacity,
                max_capacity,
                source_updated_at,
                fetched_at,
                created_at,
                updated_at,
            ) = bound
            self.connection.snapshot_updates.append(
                {
                    "location_id": location_id,
                    "is_closed": is_closed,
                    "current_capacity": current_capacity,
                    "max_capacity": max_capacity,
                    "source_updated_at": source_updated_at,
                    "fetched_at": fetched_at,
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
            )
            self.connection.snapshots[int(location_id)] = (
                location_id,
                is_closed,
                current_capacity,
                max_capacity,
                source_updated_at,
                fetched_at,
            )
            self.rowcount = 1
        elif normalized.startswith("INSERT INTO location_history"):
            (
                location_id,
                is_closed,
                current_capacity,
                max_capacity,
                source_updated_at,
                last_updated,
                fetched_at,
            ) = bound
            self.connection.history_inserts.append(
                {
                    "location_id": location_id,
                    "is_closed": is_closed,
                    "current_capacity": current_capacity,
                    "max_capacity": max_capacity,
                    "source_updated_at": source_updated_at,
                    "last_updated": last_updated,
                    "fetched_at": fetched_at,
                }
            )
            self.rowcount = 1
        elif normalized.startswith("INSERT INTO ingestion_runs"):
            run_id = self.connection.next_run_id
            self.connection.next_run_id += 1
            self.connection.run_rows[run_id] = {
                "status": "running",
                "started_at": bound[0],
            }
            self.lastrowid = run_id
            self.rowcount = 1
        elif normalized.startswith("UPDATE ingestion_runs SET"):
            run_id = int(bound[-1])
            run = self.connection.run_rows.get(run_id)
            if run is not None and run["status"] == "running":
                if "status='succeeded'" in normalized:
                    (
                        completed_at,
                        received_count,
                        valid_count,
                        history_inserted_count,
                        snapshot_updated_count,
                        observed_location_ids,
                        _,
                    ) = bound
                    run.update(
                        {
                            "completed_at": completed_at,
                            "status": "succeeded",
                            "received_count": received_count,
                            "valid_count": valid_count,
                            "history_inserted_count": history_inserted_count,
                            "snapshot_updated_count": snapshot_updated_count,
                            "observed_location_ids": observed_location_ids,
                        }
                    )
                    self.connection.succeeded_run_count += 1
                else:
                    completed_at, category, message, _ = bound
                    run.update(
                        {
                            "completed_at": completed_at,
                            "status": "failed",
                            "error_category": category,
                            "error_message": message,
                        }
                    )
                self.rowcount = 1
        else:
            raise AssertionError(f"Unexpected SQL in fake: {normalized}")
        return self.rowcount

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)


class FakeConnection:
    def __init__(self) -> None:
        self.autocommit_enabled = False
        self.commit_failure: BaseException | None = None
        self.rollback_failure: BaseException | None = None
        self.close_failure: BaseException | None = None
        self.cursor_failure: BaseException | None = None
        self.queries: list[RecordedQuery] = []
        self.transactions: list[str] = []
        self.snapshot_updates: list[dict[str, Any]] = []
        self.history_inserts: list[dict[str, Any]] = []
        self.legacy_history_bytes = b"2026-08-31 12:00:00.000000"
        self.succeeded_run_count = 1
        self.next_run_id = 7
        self.run_rows: dict[int, dict[str, Any]] = {
            6: {"status": "succeeded", "started_at": datetime(2026, 8, 31, 11)},
            7: {"status": "running", "started_at": datetime(2026, 8, 31, 11, 59)},
        }
        self.snapshots: dict[int, tuple[Any, ...]] = {
            5761: (5761, False, 47, 100, None, datetime(2026, 8, 31, 11, 59))
        }

    def get_autocommit(self) -> bool:
        return self.autocommit_enabled

    def cursor(self) -> FakeCursor:
        if self.cursor_failure is not None:
            raise self.cursor_failure
        return FakeCursor(self)

    def commit(self) -> None:
        self.transactions.append("commit")
        if self.commit_failure is not None:
            raise self.commit_failure

    def rollback(self) -> None:
        self.transactions.append("rollback")
        if self.rollback_failure is not None:
            raise self.rollback_failure

    def close(self) -> None:
        self.transactions.append("close")
        if self.close_failure is not None:
            raise self.close_failure


@dataclass(frozen=True)
class FakeIngestionWriteCounts:
    history_inserted: int
    snapshot_updated: int


class LifecycleFakeConnection:
    def __init__(self, factory: LifecycleRepositoryFactory, stage: str) -> None:
        self.factory = factory
        self.stage = stage
        self.events = factory.events_for(stage)
        self.close_attempted = False
        self.locks_may_be_retained = False
        self.pending_failed_run: tuple[str, str] | None = None

    def get_autocommit(self) -> bool:
        return False

    def commit(self) -> None:
        self.factory.trace.append(f"{self.stage}.commit")
        self.events.append("commit")
        self.factory.maybe_raise(f"{self.stage}_commit")
        if self.stage == "run":
            self.factory.run_recorded = True
        elif self.stage == "failure" and self.pending_failed_run is not None:
            self.factory.failed_runs.append(self.pending_failed_run)

    def rollback(self) -> None:
        self.factory.trace.append(f"{self.stage}.rollback")
        self.events.append("rollback")
        point = "rollback" if self.stage == "work" else f"{self.stage}_rollback"
        try:
            self.factory.maybe_raise(point)
        except BaseException:
            self.locks_may_be_retained = True
            raise

    def close(self) -> None:
        self.factory.trace.append(f"{self.stage}.close")
        self.events.append("close")
        self.close_attempted = True
        self.factory.maybe_raise(f"{self.stage}_close")
        self.locks_may_be_retained = False


class LifecycleFakeRepository:
    def __init__(
        self,
        factory: LifecycleRepositoryFactory,
        connection: LifecycleFakeConnection,
    ) -> None:
        self.factory = factory
        self.connection = connection

    def start_run(self, started_at: datetime) -> int:
        del started_at
        self.connection.events.append("start")
        self.factory.maybe_raise("start")
        return self.factory.run_id

    def persist_successful_poll(
        self, run_id: int, rows: object, fetched_at: datetime
    ) -> FakeIngestionWriteCounts:
        del run_id, fetched_at
        self.connection.events.append("persist")
        self.factory.maybe_raise("persist")
        normalized_rows = tuple(rows)  # type: ignore[arg-type]
        self.factory.snapshot_writes.append(normalized_rows)
        return FakeIngestionWriteCounts(
            history_inserted=len(normalized_rows),
            snapshot_updated=len(normalized_rows),
        )

    def complete_success(
        self,
        run_id: int,
        completed_at: datetime,
        received_count: int,
        valid_count: int,
        counts: FakeIngestionWriteCounts,
        observed_location_ids: object,
    ) -> None:
        del (
            run_id,
            completed_at,
            received_count,
            valid_count,
            counts,
            observed_location_ids,
        )
        self.connection.events.append("complete_success")
        self.factory.maybe_raise("complete_success")

    def complete_failure(
        self,
        run_id: int,
        completed_at: datetime,
        category: str,
        message: str,
    ) -> None:
        del run_id, completed_at
        self.factory.trace.append("failure.complete_failure")
        self.connection.events.append("complete_failure")
        self.factory.maybe_raise("complete_failure")
        self.connection.pending_failed_run = (category, message)


class LifecycleRepositoryFactory:
    STAGES = ("run", "work", "failure")

    def __init__(self) -> None:
        self.failures: dict[str, BaseException] = {}
        self.connection_attempts = 0
        self.connections: list[LifecycleFakeConnection] = []
        self.run_events: list[str] = []
        self.work_events: list[str] = []
        self.failure_events: list[str] = []
        self.trace: list[str] = []
        self.event_lines: list[str] = []
        self.failed_runs: list[tuple[str, str]] = []
        self.snapshot_writes: list[tuple[object, ...]] = []
        self.run_id = 41
        self.run_recorded = False

    def raise_on(self, point: str, error: BaseException) -> None:
        self.failures[point] = error

    def maybe_raise(self, point: str) -> None:
        error = self.failures.get(point)
        if error is not None:
            raise error

    def events_for(self, stage: str) -> list[str]:
        return getattr(self, f"{stage}_events")

    def connect(self) -> LifecycleFakeConnection:
        stage_index = self.connection_attempts
        self.connection_attempts += 1
        stage = self.STAGES[min(stage_index, len(self.STAGES) - 1)]
        self.trace.append(f"{stage}.connect")
        if stage == "failure" and any(
            connection.locks_may_be_retained for connection in self.connections
        ):
            self.trace.append("failure.self_block")
            raise RuntimeError("retained work lock would self-block")
        self.maybe_raise(f"{stage}_connect")
        connection = LifecycleFakeConnection(self, stage)
        self.connections.append(connection)
        return connection

    def __call__(
        self, connection: LifecycleFakeConnection
    ) -> LifecycleFakeRepository:
        self.maybe_raise(f"{connection.stage}_repository")
        return LifecycleFakeRepository(self, connection)

    @property
    def failure_transaction_count(self) -> int:
        return sum(connection.stage == "failure" for connection in self.connections)

    @property
    def all_connections_closed(self) -> bool:
        return all(connection.close_attempted for connection in self.connections)
