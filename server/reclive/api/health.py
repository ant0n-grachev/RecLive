from __future__ import annotations

import sys as _import_sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Protocol

from fastapi import APIRouter, Depends, HTTPException

from server.reclive.api.dependencies import require_admin_token
from server.reclive.api import forecasts as _owner_api_forecasts
from server.reclive import facility_schedule as _owner_facility_schedule
from server.reclive import settings as _owner_settings
from server.reclive import runtime as _owner_runtime
from server.reclive.health import (
    ComponentHealth,
    HealthReport,
    classify_age,
    overall_status,
    utc_iso,
)
from server.reclive.health_repository import HealthEvidence
from server.reclive.repositories import push_rules as _owner_repositories_push_rules
from server.reclive.runtime import current_runtime


class EvidenceRepository(Protocol):
    def collect(self, now: datetime) -> HealthEvidence: ...


def _fixed_component(status: str) -> ComponentHealth:
    safe_status = (
        status
        if isinstance(status, str)
        and status in {"ready", "stale", "missing", "unavailable"}
        else "unavailable"
    )
    return ComponentHealth(status=safe_status, observedAt=None, ageSeconds=None)


def _observed_component(
    status: str | None,
    observed_at: datetime | None,
    checked_at: datetime,
    stale_after_seconds: int,
) -> ComponentHealth:
    classified = classify_age(observed_at, checked_at, stale_after_seconds)
    if classified.detail == "future_timestamp" or status in {None, "ready"}:
        return classified
    if status not in {"stale", "missing", "unavailable"}:
        return _fixed_component("unavailable")
    return ComponentHealth(
        status=status,
        observedAt=classified.observedAt,
        ageSeconds=classified.ageSeconds,
        detail=classified.detail,
    )


def create_health_router(
    *,
    repository: EvidenceRepository,
    ingestion_stale_after_seconds: int,
    forecast_stale_after_seconds: int,
    schedule_stale_after_seconds: int,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> APIRouter:
    health_router = APIRouter()

    @health_router.get("/health")
    def get_health_report() -> dict[str, object]:
        try:
            checked_at = now()
            evidence = repository.collect(checked_at)
            components = {
                "api": _fixed_component("ready"),
                "database": _fixed_component(evidence.database),
                "migrations": _fixed_component(evidence.migrations),
                "ingestion": _observed_component(
                    evidence.ingestion_status,
                    evidence.ingestion_observed_at,
                    checked_at,
                    ingestion_stale_after_seconds,
                ),
                "forecast": _observed_component(
                    evidence.forecast_status,
                    evidence.forecast_observed_at,
                    checked_at,
                    forecast_stale_after_seconds,
                ),
                "schedules": _observed_component(
                    evidence.schedule_status,
                    evidence.schedule_observed_at,
                    checked_at,
                    schedule_stale_after_seconds,
                ),
                "push": _fixed_component(evidence.push),
            }
            report = HealthReport(
                status=overall_status(
                    {name: component.status for name, component in components.items()}
                ),
                checkedAt=utc_iso(checked_at),
                components=components,
            )
            return report.to_dict()
        except Exception:
            raise HTTPException(status_code=503, detail="health_unavailable") from None

    return health_router

router = APIRouter()


def health() -> Dict[str, Any]:
    payload = _owner_api_forecasts.load_forecast()
    try:
        schedule = _owner_facility_schedule.schedule_health(
            _owner_facility_schedule.load_facility_hours(), _owner_runtime.now_utc()
        )
    except HTTPException:
        schedule = {"state": "unavailable", "ageSeconds": None, "facilities": {}}
    return {
        "status": "ok",
        "generatedAt": payload.get("generatedAt"),
        "generatedAgeSeconds": _owner_api_forecasts.generated_age_seconds(payload),
        "facilities": len(payload.get("facilities", [])),
        "modelStatus": payload.get("modelInfo", {}).get("status"),
        "schedule": schedule,
    }


@router.get("/health/push")
def push_health(_admin: None = Depends(require_admin_token)) -> Dict[str, Any]:
    rules_count = _owner_repositories_push_rules.db_rules_count()
    return {
        "status": "ok",
        "rules": rules_count,
        "vapidConfigured": _owner_settings.push_vapid_configured(),
        "evaluatorEnabled": _owner_settings.evaluator_enabled(),
        "evaluatorIntervalSeconds": current_runtime().settings.push.evaluator_interval_seconds,
    }


_import_sys.modules.setdefault(
    "server.reclive.api.health", _import_sys.modules[__name__]
)
_import_sys.modules.setdefault("reclive.api.health", _import_sys.modules[__name__])
