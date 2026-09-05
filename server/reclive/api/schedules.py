from __future__ import annotations

import sys as _import_sys
from typing import Any, Dict, List
from fastapi import APIRouter
from server.reclive import facility_schedule as _owner_facility_schedule

router = APIRouter()


@router.get("/api/facility-hours")
def facility_hours() -> Dict[str, Any]:
    return _owner_facility_schedule.load_facility_hours()


@router.get("/api/facility-hours/facilities")
def facility_hours_facilities() -> List[Dict[str, Any]]:
    payload = _owner_facility_schedule.load_facility_hours()
    facilities = payload.get("facilities", [])
    items: List[Dict[str, Any]] = []
    for item in facilities:
        if not isinstance(item, dict):
            continue
        sections = item.get("sections", [])
        items.append(
            {
                "facilityId": _owner_facility_schedule._parse_facility_id(
                    item.get("facilityId")
                ),
                "facilityName": str(item.get("facilityName", "")).strip(),
                "status": str(item.get("status", "")).strip(),
                "sections": len(sections) if isinstance(sections, list) else 0,
            }
        )
    return items


@router.get("/api/facility-hours/facilities/{facility_id}")
def facility_hours_for_facility(facility_id: int) -> Dict[str, Any]:
    payload = _owner_facility_schedule.load_facility_hours()
    facility = _owner_facility_schedule.get_facility_hours_entry(payload, facility_id)
    return {
        "generatedAt": payload.get("generatedAt"),
        "sourceSite": payload.get("sourceSite"),
        **facility,
    }


_import_sys.modules.setdefault(
    "server.reclive.api.schedules", _import_sys.modules[__name__]
)
_import_sys.modules.setdefault("reclive.api.schedules", _import_sys.modules[__name__])
