from __future__ import annotations

import sys as _import_sys
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, Request
from server.reclive.api.dependencies import require_admin_token
from server.reclive import push as _owner_push
from server.reclive.repositories import push_rules as _owner_repositories_push_rules
from server.reclive import settings as _owner_settings

router = APIRouter()


@router.get("/api/push/public-key")
def public_key() -> Dict[str, str]:
    return {"publicKey": _owner_settings.get_vapid_public_key()}


@router.get("/api/push/availability")
def push_availability() -> Dict[str, Any]:
    db_available = _owner_repositories_push_rules.push_db_available()
    vapid_configured = _owner_settings.push_vapid_configured()
    identity_configured = _owner_push.push_identity_configured()
    alerts_available = db_available and vapid_configured and identity_configured
    reason: Optional[str] = None
    if not db_available:
        reason = "push_rules_db_unavailable"
    elif not vapid_configured:
        reason = "push_vapid_unconfigured"
    elif not identity_configured:
        reason = "push_identity_unconfigured"
    return {
        "apiAvailable": True,
        "dbAvailable": db_available,
        "alertsAvailable": alerts_available,
        "reason": reason,
        "storeBackend": "db",
    }


@router.post("/api/push/subscribe")
async def subscribe(request: Request) -> Dict[str, Any]:
    payload = await _owner_push._parse_limited_push_write(
        request, _owner_push.PushRuleRequest
    )
    subscription = _owner_push._validated_subscription_from_model(payload.subscription)
    return _owner_push.subscribe_owned_push_rule(
        subscription=subscription,
        facility_id=payload.facility_id,
        section_key=payload.section_key,
        threshold=payload.threshold,
        ttl_seconds=payload.ttl_seconds,
    )


@router.post("/api/push/rules/list")
async def push_rules_list(request: Request) -> Dict[str, Any]:
    payload = await _owner_push.parse_limited_push_body(
        request, _owner_push.PushOwnershipRequest
    )
    subscription = _owner_push._validated_subscription_from_model(payload.subscription)
    return _owner_push.list_owned_push_rules(subscription)


@router.delete("/api/push/rules/{rule_id}")
async def push_rule_cancel(rule_id: str, request: Request) -> Dict[str, Any]:
    payload = await _owner_push._parse_limited_push_write(
        request, _owner_push.PushOwnershipRequest
    )
    parsed_rule_id = _owner_push._parse_push_rule_id(rule_id)
    subscription = _owner_push._validated_subscription_from_model(payload.subscription)
    return _owner_push.cancel_owned_push_rule(subscription, parsed_rule_id)


@router.post("/api/push/rules/cancel-all")
async def push_rules_cancel_all(request: Request) -> Dict[str, Any]:
    payload = await _owner_push._parse_limited_push_write(
        request, _owner_push.PushOwnershipRequest
    )
    subscription = _owner_push._validated_subscription_from_model(payload.subscription)
    return _owner_push.cancel_all_owned_push_rules(subscription)


@router.post("/api/push/dispatch")
async def dispatch(
    request: Request, _admin: None = Depends(require_admin_token)
) -> Dict[str, Any]:
    payload = await _owner_push.parse_limited_push_body(
        request, _owner_push.PushDispatchRequest
    )
    return _owner_push.evaluate_rules_once(
        facility_filter=payload.facilityId, section_filter=payload.sectionKey
    )


@router.post("/api/push/evaluate")
def evaluate(_admin: None = Depends(require_admin_token)) -> Dict[str, Any]:
    return _owner_push.evaluate_rules_once()


_import_sys.modules.setdefault("server.reclive.api.push", _import_sys.modules[__name__])
_import_sys.modules.setdefault("reclive.api.push", _import_sys.modules[__name__])
