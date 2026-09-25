from __future__ import annotations

import sys as _import_sys
from datetime import datetime, timezone
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException
from server.reclive.occupancy_repository import SnapshotRepository
from server.reclive.api.dependencies import get_snapshot_repository
from server.reclive.runtime import current_runtime

router = APIRouter()


@router.get("/api/live-counts")
def live_counts(
    repository: SnapshotRepository = Depends(get_snapshot_repository),
) -> Dict[str, Any]:
    now = current_runtime().clock()
    try:
        snapshot = repository.fetch_live_snapshot(now)
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Failed to query live occupancy snapshot"
        ) from exc
    if not snapshot.rows:
        raise HTTPException(status_code=503, detail="Live occupancy snapshot is empty")
    last_successful_fetch_at = snapshot.last_successful_fetch_at
    if last_successful_fetch_at is None:
        ingestion = {
            "lastSuccessfulFetchAt": None,
            "ageSeconds": None,
            "status": "unavailable",
        }
    else:
        elapsed_seconds = max(0.0, (now - last_successful_fetch_at).total_seconds())
        age_seconds = int(elapsed_seconds)
        ingestion = {
            "lastSuccessfulFetchAt": utc_iso(last_successful_fetch_at),
            "ageSeconds": age_seconds,
            "status": "healthy" if elapsed_seconds <= 600 else "stale",
        }
    try:
        rows = [
            {
                "LocationId": row.location_id,
                "IsClosed": row.is_closed,
                "LastCount": row.current_capacity,
                "LastUpdatedDateAndTime": utc_iso(row.source_updated_at)
                if row.source_updated_at is not None
                else None,
                "FetchedAt": utc_iso(row.fetched_at),
            }
            for row in snapshot.rows
        ]
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=503, detail="Failed to query live occupancy snapshot"
        ) from exc
    return {"ingestion": ingestion, "rows": rows}


def utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


_import_sys.modules.setdefault(
    "server.reclive.api.live_counts", _import_sys.modules[__name__]
)
_import_sys.modules.setdefault("reclive.api.live_counts", _import_sys.modules[__name__])
