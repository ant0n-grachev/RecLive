from __future__ import annotations

import sys as _import_sys
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException
from server.reclive.runtime import current_runtime
from server.reclive.api.dependencies import require_admin_token
from server.reclive.api import forecasts as _owner_api_forecasts
from server.reclive import facility_schedule as _owner_facility_schedule
from server.reclive.repositories import push_rules as _owner_repositories_push_rules
from server.reclive import runtime as _owner_runtime
from server.reclive import settings as _owner_settings

router = APIRouter()


@router.get("/health")
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
