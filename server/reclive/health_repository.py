from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from server.reclive.facility_schedule import validate_schedule_payload
from server.reclive.migrations import migration_files, snapshot_migration


MAX_HEALTH_ARTIFACT_BYTES = 16 * 1024 * 1024

EvidenceStatus = Literal["ready", "stale", "missing", "unavailable"]

_ARTIFACT_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)
_SAFE_PUSH_STATUSES = frozenset({"ready", "unavailable"})


class Cursor(Protocol):
    def execute(self, sql: str, params: object = None) -> None: ...

    def fetchone(self) -> tuple[object, ...] | None: ...

    def fetchall(self) -> list[tuple[object, ...]]: ...

    def __enter__(self) -> Cursor: ...

    def __exit__(self, *args: object) -> None: ...


class Connection(Protocol):
    def cursor(self) -> Cursor: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class HealthEvidence:
    database: str
    migrations: str
    ingestion_observed_at: datetime | None
    forecast_observed_at: datetime | None
    schedule_observed_at: datetime | None
    push: str
    ingestion_status: EvidenceStatus | None = None
    forecast_status: EvidenceStatus | None = None
    schedule_status: EvidenceStatus | None = None


@dataclass(frozen=True)
class _ObservationEvidence:
    status: EvidenceStatus
    observed_at: datetime | None


def _require_aware(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError("health collection requires an aware timestamp")
    return value.astimezone(timezone.utc)


def _parse_artifact_timestamp(value: object) -> datetime | None:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or _ARTIFACT_TIMESTAMP.fullmatch(value) is None
    ):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        offset = parsed.utcoffset()
        if parsed.tzinfo is None or offset is None:
            return None
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("ambiguous artifact")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(_value: str) -> object:
    raise ValueError("nonstandard JSON constant")


def _read_artifact(path: Path) -> tuple[EvidenceStatus, Mapping[str, object] | None]:
    try:
        with path.open("rb") as handle:
            contents = handle.read(MAX_HEALTH_ARTIFACT_BYTES + 1)
    except FileNotFoundError:
        return "missing", None
    except Exception:
        return "unavailable", None
    if len(contents) > MAX_HEALTH_ARTIFACT_BYTES:
        return "unavailable", None
    try:
        payload = json.loads(
            contents.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except Exception:
        return "unavailable", None
    if not isinstance(payload, Mapping):
        return "unavailable", None
    return "ready", payload


def _forecast_evidence(
    path: Path,
    now: datetime | None = None,
) -> _ObservationEvidence:
    status, payload = _read_artifact(path)
    if status != "ready" or payload is None:
        return _ObservationEvidence(status, None)
    observed_at = _parse_artifact_timestamp(payload.get("generatedAt"))
    if observed_at is None:
        return _ObservationEvidence("unavailable", None)
    if now is not None and observed_at > now:
        return _ObservationEvidence("unavailable", observed_at)
    return _ObservationEvidence("ready", observed_at)


def generated_at(path: Path) -> datetime | None:
    return _forecast_evidence(path).observed_at


def _schedule_evidence(path: Path, now: datetime) -> _ObservationEvidence:
    status, payload = _read_artifact(path)
    if status != "ready" or payload is None:
        return _ObservationEvidence(status, None)

    generated_observed_at = _parse_artifact_timestamp(payload.get("generatedAt"))
    if generated_observed_at is None:
        return _ObservationEvidence("unavailable", None)
    if generated_observed_at > now:
        return _ObservationEvidence("unavailable", generated_observed_at)

    try:
        validated = validate_schedule_payload(payload, now=now)
    except Exception:
        return _ObservationEvidence("unavailable", None)

    facilities = validated.get("facilities")
    if not isinstance(facilities, list):
        return _ObservationEvidence("unavailable", None)

    observations: list[datetime] = []
    statuses: list[str] = []
    for facility in facilities:
        if not isinstance(facility, Mapping):
            return _ObservationEvidence("unavailable", None)
        facility_status = facility.get("status")
        if not isinstance(facility_status, str):
            return _ObservationEvidence("unavailable", None)
        statuses.append(facility_status)
        if facility_status in {"ok", "stale"}:
            source_observed_at = _parse_artifact_timestamp(
                facility.get("lastSuccessfulAt")
            )
            if source_observed_at is None:
                return _ObservationEvidence("unavailable", None)
            if source_observed_at > now:
                return _ObservationEvidence("unavailable", source_observed_at)
            observations.append(source_observed_at)

    oldest_observation = min(observations) if observations else None
    if any(facility_status == "error" for facility_status in statuses):
        return _ObservationEvidence("unavailable", oldest_observation)
    if any(facility_status == "stale" for facility_status in statuses):
        return _ObservationEvidence("stale", oldest_observation)
    if statuses == ["ok", "ok"] and len(observations) == 2:
        return _ObservationEvidence("ready", oldest_observation)
    return _ObservationEvidence("unavailable", oldest_observation)


def healthy_schedule_generated_at(
    path: Path,
    *,
    now: datetime | None = None,
) -> datetime | None:
    reference_now = (
        datetime.now(timezone.utc) if now is None else _require_aware(now)
    )
    evidence = _schedule_evidence(path, reference_now)
    return evidence.observed_at if evidence.status == "ready" else None


def _migration_evidence(connection: Connection, migration_dir: Path) -> str:
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT filename, checksum FROM schema_migrations ORDER BY filename"
            )
            rows = cursor.fetchall()
        recorded: dict[str, str] = {}
        for row in rows:
            if (
                not isinstance(row, (list, tuple))
                or len(row) != 2
                or not isinstance(row[0], str)
                or not isinstance(row[1], str)
                or row[0] in recorded
            ):
                return "unavailable"
            recorded[row[0]] = row[1]

        expected = {
            path.name: snapshot_migration(path).checksum
            for path in migration_files(migration_dir)
        }
    except Exception:
        return "unavailable"
    return "ready" if recorded == expected else "stale"


def _ingestion_evidence(connection: Connection) -> _ObservationEvidence:
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT completed_at FROM ingestion_runs WHERE status = 'succeeded' "
                "ORDER BY completed_at DESC, id DESC LIMIT 1"
            )
            row = cursor.fetchone()
    except Exception:
        return _ObservationEvidence("unavailable", None)
    if row is None:
        return _ObservationEvidence("missing", None)
    if (
        not isinstance(row, (list, tuple))
        or len(row) != 1
        or not isinstance(row[0], datetime)
        or row[0].tzinfo is not None
    ):
        return _ObservationEvidence("unavailable", None)
    return _ObservationEvidence("ready", row[0].replace(tzinfo=timezone.utc))


def _close_connection(connection: Connection) -> None:
    try:
        connection.close()
    except Exception:
        pass


class HealthRepository:
    def __init__(
        self,
        *,
        connect: Callable[[], Connection],
        migration_dir: Path,
        forecast_path: Path,
        schedule_path: Path,
        push_status: Callable[[], str],
    ) -> None:
        self.connect = connect
        self.migration_dir = migration_dir
        self.forecast_path = forecast_path
        self.schedule_path = schedule_path
        self.push_status = push_status

    def collect(self, now: datetime) -> HealthEvidence:
        reference_now = _require_aware(now)

        connection: Connection | None = None
        try:
            connection = self.connect()
            if not callable(getattr(connection, "cursor", None)):
                raise TypeError("invalid health connection")
        except Exception:
            database = "unavailable"
            migrations = "unavailable"
            ingestion = _ObservationEvidence("unavailable", None)
        else:
            database = "ready"
            migrations = _migration_evidence(connection, self.migration_dir)
            ingestion = _ingestion_evidence(connection)
        finally:
            if connection is not None:
                _close_connection(connection)

        forecast = _forecast_evidence(self.forecast_path, reference_now)
        schedules = _schedule_evidence(self.schedule_path, reference_now)

        try:
            candidate_push = self.push_status()
        except Exception:
            push = "unavailable"
        else:
            push = (
                candidate_push
                if isinstance(candidate_push, str)
                and candidate_push in _SAFE_PUSH_STATUSES
                else "unavailable"
            )

        return HealthEvidence(
            database=database,
            migrations=migrations,
            ingestion_observed_at=ingestion.observed_at,
            forecast_observed_at=forecast.observed_at,
            schedule_observed_at=schedules.observed_at,
            push=push,
            ingestion_status=ingestion.status,
            forecast_status=forecast.status,
            schedule_status=schedules.status,
        )
