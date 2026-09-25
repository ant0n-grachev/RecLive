from __future__ import annotations

import sys as _import_sys
import hmac
from datetime import datetime
from typing import Iterator, Optional, Sequence
from fastapi import Header, HTTPException, Request
from server.reclive.actual_hours import HistoryState, IngestionHeartbeat
from server.reclive.ingestion import safe_close
from server.reclive.occupancy_repository import (
    ActualHourReadProtocol,
    SnapshotRepository,
)
from server.reclive.runtime import current_runtime, runtime_scope
from server.reclive import settings as _owner_settings


class OwnedActualHourRepository:
    def __init__(self, runtime=None):
        self.runtime = runtime or current_runtime()

    def load_actual_hour_inputs(
        self, location_ids: Sequence[int], range_start: datetime, range_end: datetime
    ) -> tuple[list[HistoryState], list[IngestionHeartbeat]]:
        connection = None
        try:
            connection = self.runtime.connect(autocommit=False)
            repository = SnapshotRepository(connection)
            return repository.load_actual_hour_inputs(
                location_ids, range_start, range_end
            )
        finally:
            safe_close(connection)


async def bind_runtime(request: Request):
    with runtime_scope(request.app.state.runtime):
        yield request.app.state.runtime


def get_actual_hour_repository(request: Request = None) -> ActualHourReadProtocol:
    return OwnedActualHourRepository(
        request.app.state.runtime if request is not None else current_runtime()
    )


def get_snapshot_repository(request: Request = None) -> Iterator[SnapshotRepository]:
    runtime = request.app.state.runtime if request is not None else current_runtime()
    try:
        connection = runtime.connect(autocommit=False)
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Live occupancy DB is unavailable"
        ) from exc
    try:
        yield SnapshotRepository(connection)
    finally:
        safe_close(connection)


def get_push_rule_repository(request: Request):
    from server.reclive.repositories.push_rules import PushRuleRepository

    connection = None
    try:
        connection = request.app.state.runtime.connect(autocommit=False)
        yield PushRuleRepository(connection, request.app.state.runtime)
    finally:
        safe_close(connection)


def require_admin_token(
    x_reclive_admin_token: Optional[str] = Header(
        default=None, alias="X-RecLive-Admin-Token"
    ),
) -> None:
    if not _owner_settings.push_admin_routes_enabled():
        raise HTTPException(status_code=503, detail="Admin routes are disabled")
    try:
        configured = _owner_settings._validated_admin_token_bytes()
    except RuntimeError:
        raise HTTPException(
            status_code=503, detail="Admin token is not configured"
        ) from None
    try:
        supplied = (x_reclive_admin_token or "").encode("ascii")
    except UnicodeEncodeError:
        raise HTTPException(status_code=401, detail="Admin token is required") from None
    if not hmac.compare_digest(supplied, configured):
        raise HTTPException(status_code=401, detail="Admin token is required")


_import_sys.modules.setdefault(
    "server.reclive.api.dependencies", _import_sys.modules[__name__]
)
_import_sys.modules.setdefault(
    "reclive.api.dependencies", _import_sys.modules[__name__]
)
