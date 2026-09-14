from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Literal


ComponentStatus = Literal["ready", "stale", "missing", "unavailable"]
OverallStatus = Literal["ready", "degraded"]
DetailCategory = Literal["future_timestamp"]

REQUIRED_COMPONENTS = (
    "api",
    "database",
    "migrations",
    "ingestion",
    "forecast",
    "schedules",
    "push",
)

_COMPONENT_STATUSES = frozenset({"ready", "stale", "missing", "unavailable"})
_OVERALL_STATUSES = frozenset({"ready", "degraded"})
_DETAIL_CATEGORIES = frozenset({"future_timestamp"})
_UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{6})?Z$"
)


def _require_aware(value: datetime) -> None:
    try:
        offset = value.utcoffset()
    except Exception:
        raise ValueError("timestamps must be timezone-aware") from None
    if value.tzinfo is None or offset is None:
        raise ValueError("timestamps must be timezone-aware")


def _is_canonical_utc_timestamp(value: str) -> bool:
    if _UTC_TIMESTAMP.fullmatch(value) is None:
        return False
    try:
        datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError:
        return False
    return True


def utc_iso(value: datetime) -> str:
    _require_aware(value)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ComponentHealth:
    status: ComponentStatus
    observedAt: str | None
    ageSeconds: int | None
    detail: DetailCategory | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or self.status not in _COMPONENT_STATUSES:
            raise ValueError("invalid component status")
        if self.observedAt is not None and (
            not isinstance(self.observedAt, str)
            or not _is_canonical_utc_timestamp(self.observedAt)
        ):
            raise ValueError("observedAt must be a canonical UTC timestamp or None")
        if self.ageSeconds is not None and (
            isinstance(self.ageSeconds, bool)
            or not isinstance(self.ageSeconds, int)
            or self.ageSeconds < 0
        ):
            raise ValueError("ageSeconds must be a nonnegative integer or None")
        if self.detail is not None and (
            not isinstance(self.detail, str) or self.detail not in _DETAIL_CATEGORIES
        ):
            raise ValueError("invalid component detail category")
        if self.detail == "future_timestamp" and (
            self.status != "unavailable"
            or self.observedAt is None
            or self.ageSeconds is not None
        ):
            raise ValueError("future_timestamp requires unavailable timestamp evidence")

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "status": self.status,
            "observedAt": self.observedAt,
            "ageSeconds": self.ageSeconds,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class HealthReport:
    status: OverallStatus
    checkedAt: str
    components: Mapping[str, ComponentHealth]

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or self.status not in _OVERALL_STATUSES:
            raise ValueError("invalid overall status")
        if not isinstance(self.checkedAt, str) or not _is_canonical_utc_timestamp(
            self.checkedAt
        ):
            raise ValueError("checkedAt must be a canonical UTC timestamp")
        if not isinstance(self.components, Mapping) or set(self.components) != set(
            REQUIRED_COMPONENTS
        ):
            raise ValueError("health report must contain the exact required components")
        if any(
            not isinstance(self.components[name], ComponentHealth)
            for name in REQUIRED_COMPONENTS
        ):
            raise ValueError("health report entries must be ComponentHealth values")
        component_statuses = {
            name: self.components[name].status for name in REQUIRED_COMPONENTS
        }
        if self.status != overall_status(component_statuses):
            raise ValueError("health report status must match component readiness")
        object.__setattr__(
            self,
            "components",
            MappingProxyType({name: self.components[name] for name in REQUIRED_COMPONENTS}),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "checkedAt": self.checkedAt,
            "components": {
                name: self.components[name].to_dict() for name in REQUIRED_COMPONENTS
            },
        }


def classify_age(
    observed_at: datetime | None,
    now: datetime,
    stale_after_seconds: int | float,
) -> ComponentHealth:
    _require_aware(now)
    if (
        isinstance(stale_after_seconds, bool)
        or not isinstance(stale_after_seconds, (int, float))
        or stale_after_seconds <= 0
        or (isinstance(stale_after_seconds, float) and not math.isfinite(stale_after_seconds))
    ):
        raise ValueError("stale threshold must be a finite positive number")
    if observed_at is None:
        return ComponentHealth(status="missing", observedAt=None, ageSeconds=None)
    _require_aware(observed_at)

    observed_at_utc = observed_at.astimezone(timezone.utc)
    elapsed_seconds = (now.astimezone(timezone.utc) - observed_at_utc).total_seconds()
    if elapsed_seconds < 0:
        return ComponentHealth(
            status="unavailable",
            observedAt=utc_iso(observed_at_utc),
            ageSeconds=None,
            detail="future_timestamp",
        )

    status: ComponentStatus = (
        "ready" if elapsed_seconds <= stale_after_seconds else "stale"
    )
    return ComponentHealth(
        status=status,
        observedAt=utc_iso(observed_at_utc),
        ageSeconds=int(elapsed_seconds),
    )


def overall_status(component_statuses: Mapping[str, str]) -> OverallStatus:
    if set(component_statuses) != set(REQUIRED_COMPONENTS):
        return "degraded"
    if all(component_statuses[name] == "ready" for name in REQUIRED_COMPONENTS):
        return "ready"
    return "degraded"
