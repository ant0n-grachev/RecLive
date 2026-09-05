from __future__ import annotations

import sys as _import_sys
import asyncio
import base64
import binascii
import hmac
import http.client
import ipaddress
import json
import math
import queue
import re
import socket
import ssl
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TypeVar
from urllib.parse import urlsplit
from fastapi import HTTPException, Request
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pywebpush import WebPushException, webpush
from server.reclive.sections import normalize_section_key
from server.reclive.occupancy_repository import (
    RepositoryFactory,
    SnapshotRepository,
    SnapshotRow,
)
from server.reclive.push_identity import (
    endpoint_hash,
    normalize_push_endpoint,
    rate_limit_subject_hash,
)
from server.reclive.runtime import current_runtime, runtime_scope
from pydantic import ValidationInfo
from server.reclive import facility_schedule as _owner_facility_schedule
from server.reclive.repositories import push_rules as _owner_repositories_push_rules
from server.reclive import runtime as _owner_runtime
from server.reclive import sections as _owner_sections
from server.reclive import settings as _owner_settings
from server.reclive.repositories.push_rules import (
    PushRuleRepository as PushRuleRepository,
    push_rules_table_name as push_rules_table_name,
    PushRuleRecord as PushRuleRecord,
    _mysql_utc_datetime as _mysql_utc_datetime,
    _mysql_utc_bind as _mysql_utc_bind,
    _push_rule_from_row as _push_rule_from_row,
    _canonical_subscription_json as _canonical_subscription_json,
    _safe_rollback as _safe_rollback,
    _safe_close_connection as _safe_close_connection,
    _rule_store_unavailable as _rule_store_unavailable,
    _endpoint_lock_name as _endpoint_lock_name,
    _acquire_endpoint_lock as _acquire_endpoint_lock,
    _release_endpoint_lock as _release_endpoint_lock,
    db_select_rule_by_id as db_select_rule_by_id,
    _select_active_identity_rule as _select_active_identity_rule,
    _expire_owned_pending_rules as _expire_owned_pending_rules,
    db_rules_count as db_rules_count,
    db_subscribe_rule as db_subscribe_rule,
    _db_list_owned_rule_records as _db_list_owned_rule_records,
    resolve_owned_rule as resolve_owned_rule,
    db_cancel_owned_rule as db_cancel_owned_rule,
    db_cancel_all_owned_rules as db_cancel_all_owned_rules,
    PushEvaluatorStoreError as PushEvaluatorStoreError,
    db_acquire_evaluator_lock as db_acquire_evaluator_lock,
    db_release_evaluator_lock as db_release_evaluator_lock,
    push_db_available as push_db_available,
    load_evaluator_candidates as load_evaluator_candidates,
    claim_pending_rule as claim_pending_rule,
    finalize_claimed_rule as finalize_claimed_rule,
    _push_http_error as _push_http_error,
    record_public_push_write as record_public_push_write,
)

PUSH_PROVIDER_RESPONSE_MAX_BYTES = 4096
PUSH_TRANSPORT_TIMEOUT_SECONDS = 10
PUSH_TRANSPORT_DEADLINE_SECONDS = 10.0
PUSH_TRANSPORT_WORKER_COUNT = 2
PUSH_TRANSPORT_QUEUE_CAPACITY = 2
_PROTECTED_PUSH_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "cookie",
        "expect",
        "host",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
OCCUPANCY_FRESHNESS = timedelta(minutes=10)
BASE64URL_VALUE = re.compile("^[A-Za-z0-9_-]+={0,2}$")
MYSQL_UNSIGNED_BIGINT_MAX_TEXT = "18446744073709551615"
PushModelT = TypeVar("PushModelT", bound=BaseModel)


class PushRuleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    id: int = Field(gt=0)
    facility_id: Literal[1186, 1656] = Field(alias="facilityId")
    section_key: str = Field(alias="sectionKey", min_length=1, max_length=80)
    threshold: int = Field(ge=1, le=100)
    created_at: datetime = Field(alias="createdAt")
    expires_at: datetime = Field(alias="expiresAt")
    status: Literal["pending"]


def push_rule_response(
    rule: _owner_repositories_push_rules.PushRuleRecord,
) -> Dict[str, Any]:
    created_at = _owner_repositories_push_rules._mysql_utc_datetime(rule.created_at)
    expires_at = _owner_repositories_push_rules._mysql_utc_datetime(rule.expires_at)
    if (
        rule.status != "pending"
        or rule.active_identity is None
        or _owner_sections.canonical_section_key(rule.section_key) != rule.section_key
        or (
            not _owner_sections.location_ids_for_section(
                rule.facility_id, rule.section_key
            )
        )
    ):
        raise ValueError("invalid pending push rule")
    try:
        response = PushRuleResponse(
            id=rule.id,
            facilityId=rule.facility_id,
            sectionKey=rule.section_key,
            threshold=rule.threshold,
            createdAt=created_at,
            expiresAt=expires_at,
            status="pending",
        )
    except ValidationError:
        raise ValueError("invalid pending push rule") from None
    return response.model_dump(mode="json", by_alias=True)


def _decode_stored_subscription(value: object) -> Dict[str, Any] | None:
    try:
        if isinstance(value, bytes):
            decoded: object = json.loads(value.decode("utf-8"))
        elif isinstance(value, str):
            decoded = json.loads(value)
        else:
            decoded = value
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _canonical_stored_endpoint(value: object) -> str | None:
    subscription = _decode_stored_subscription(value)
    if subscription is None or set(subscription) != {"endpoint", "keys"}:
        return None
    try:
        return validate_push_subscription(subscription).endpoint
    except (HTTPException, TypeError, ValueError):
        return None


def _rule_is_owned(
    rule: _owner_repositories_push_rules.PushRuleRecord,
    supplied_endpoint: str,
    supplied_digest: bytes,
) -> bool:
    if not hmac.compare_digest(rule.endpoint_hash, supplied_digest):
        return False
    stored_endpoint = _canonical_stored_endpoint(rule.subscription_json)
    if stored_endpoint is None:
        return False
    return hmac.compare_digest(
        stored_endpoint.encode("utf-8"), supplied_endpoint.encode("utf-8")
    )


class SafePushDispatchError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("push_dispatch_failed")


class PushProviderStatusError(SafePushDispatchError):
    def __init__(self, status_code: int) -> None:
        super().__init__()
        self.status_code = status_code


class _PushAttemptDeadline:
    def __init__(self, timeout_seconds: float) -> None:
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise SafePushDispatchError()
        self._deadline = time.monotonic() + timeout
        self._cancelled = threading.Event()
        self._resource_lock = threading.Lock()
        self._resources: list[Any] = []

    def remaining(self) -> float:
        if self._cancelled.is_set():
            raise SafePushDispatchError()
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise SafePushDispatchError()
        return remaining

    def check(self) -> None:
        self.remaining()

    def register(self, resource: Any) -> Any:
        close_immediately = False
        with self._resource_lock:
            if self._cancelled.is_set():
                close_immediately = True
            else:
                self._resources.append(resource)
        if close_immediately:
            self._close_resource(resource)
            raise SafePushDispatchError()
        self.check()
        return resource

    def cancel(self) -> None:
        self._cancelled.set()
        with self._resource_lock:
            resources = tuple(reversed(self._resources))
            self._resources.clear()
        for resource in resources:
            self._close_resource(resource)

    @staticmethod
    def _close_resource(resource: Any) -> None:
        try:
            resource.close()
        except Exception:
            pass


class _BoundedPushExecutor:
    def __init__(self, worker_count: int, queue_capacity: int) -> None:
        if worker_count < 1 or queue_capacity < 1:
            raise ValueError("invalid push executor bounds")
        self._tasks: queue.Queue[Callable[[], None]] = queue.Queue(
            maxsize=queue_capacity
        )
        self._workers = tuple(
            (
                threading.Thread(
                    target=self._run,
                    name=f"reclive-push-worker-{index + 1}",
                    daemon=True,
                )
                for index in range(worker_count)
            )
        )
        for worker in self._workers:
            worker.start()

    def submit(self, task: Callable[[], None]) -> bool:
        try:
            self._tasks.put_nowait(task)
        except queue.Full:
            return False
        return True

    def _run(self) -> None:
        while True:
            task = self._tasks.get()
            try:
                task()
            except BaseException:
                pass
            finally:
                self._tasks.task_done()


def _get_push_executor() -> _BoundedPushExecutor:
    if current_runtime().executor is None:
        with current_runtime().transport_lock:
            if current_runtime().executor is None:
                current_runtime().executor = _BoundedPushExecutor(
                    PUSH_TRANSPORT_WORKER_COUNT, PUSH_TRANSPORT_QUEUE_CAPACITY
                )
    return current_runtime().executor


@dataclass(frozen=True)
class PinnedPushTarget:
    endpoint: str
    connect_ip: str
    tls_server_hostname: str
    host_header: str
    port: int
    request_target: str


@dataclass(frozen=True)
class SafePushResponse:
    status_code: int
    reason: str = ""
    text: str = ""


def resolve_endpoint_host(host: str, port: int) -> List[str]:
    answers = socket.getaddrinfo(
        host, int(port), socket.AF_UNSPEC, socket.SOCK_STREAM, socket.IPPROTO_TCP
    )
    return [str(answer[4][0]) for answer in answers]


def _is_safe_push_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    if isinstance(address, ipaddress.IPv6Address) and (
        address.scope_id is not None
        or address.ipv4_mapped is not None
        or address.sixtofour is not None
        or (address.teredo is not None)
    ):
        return False
    return bool(
        address.is_global
        and (not address.is_private)
        and (not address.is_loopback)
        and (not address.is_link_local)
        and (not address.is_multicast)
        and (not address.is_unspecified)
        and (not address.is_reserved)
    )


def resolve_public_push_addresses(
    endpoint: str,
    resolver: Any | None = None,
    *,
    attempt: _PushAttemptDeadline | None = None,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    try:
        if attempt is not None:
            attempt.check()
        canonical = normalize_push_endpoint(endpoint)
        parsed = urlsplit(canonical)
        host = parsed.hostname
        port = parsed.port or 443
        if not host:
            raise ValueError("missing push host")
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            raw_answers = (
                resolver or current_runtime().resolver or resolve_endpoint_host
            )(host, port)
        else:
            raw_answers = [literal.compressed]
        if attempt is not None:
            attempt.check()
        if not isinstance(raw_answers, (list, tuple)) or not raw_answers:
            raise ValueError("empty push address set")
        addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
        for raw_answer in raw_answers:
            if not isinstance(raw_answer, str):
                raise ValueError("invalid push address")
            address = ipaddress.ip_address(raw_answer)
            if not _is_safe_push_address(address):
                raise ValueError("unsafe push address")
            addresses.add(address)
        if not addresses:
            raise ValueError("empty push address set")
        return tuple(
            sorted(
                addresses,
                key=lambda address: (
                    0 if isinstance(address, ipaddress.IPv6Address) else 1,
                    int(address),
                ),
            )
        )
    except SafePushDispatchError:
        raise
    except Exception:
        raise SafePushDispatchError() from None


def build_pinned_push_target(
    endpoint: str, *, attempt: _PushAttemptDeadline | None = None
) -> PinnedPushTarget:
    try:
        if attempt is not None:
            attempt.check()
        canonical = normalize_push_endpoint(endpoint)
        parsed = urlsplit(canonical)
        host = parsed.hostname
        if not host:
            raise ValueError("missing push host")
        port = parsed.port or 443
        addresses = resolve_public_push_addresses(canonical, attempt=attempt)
        if attempt is not None:
            attempt.check()
        selected = addresses[0]
        try:
            host_address = ipaddress.ip_address(host)
        except ValueError:
            host_header_name = host
        else:
            host_header_name = (
                f"[{host_address.compressed}]"
                if isinstance(host_address, ipaddress.IPv6Address)
                else host_address.compressed
            )
        host_header = host_header_name if port == 443 else f"{host_header_name}:{port}"
        request_target = parsed.path or "/"
        if parsed.query:
            request_target = f"{request_target}?{parsed.query}"
        request_target.encode("ascii")
        host_header.encode("ascii")
        host.encode("ascii")
        return PinnedPushTarget(
            endpoint=canonical,
            connect_ip=selected.compressed,
            tls_server_hostname=host,
            host_header=host_header,
            port=port,
            request_target=request_target,
        )
    except SafePushDispatchError:
        raise
    except Exception:
        raise SafePushDispatchError() from None


class PinnedPushSession:
    def __init__(
        self, target: PinnedPushTarget, attempt: _PushAttemptDeadline | None = None
    ) -> None:
        self.target = target
        self.attempt = attempt or _PushAttemptDeadline(PUSH_TRANSPORT_DEADLINE_SECONDS)

    def post(
        self,
        url: str,
        *,
        timeout: float = PUSH_TRANSPORT_TIMEOUT_SECONDS,
        data: bytes,
        headers: Mapping[str, Any],
        **kwargs: Any,
    ) -> SafePushResponse:
        raw_socket: Any = None
        tls_socket: Any = None
        response: Any = None
        try:
            self.attempt.check()
            if kwargs or timeout != PUSH_TRANSPORT_TIMEOUT_SECONDS:
                raise ValueError("invalid push request options")
            canonical = normalize_push_endpoint(url)
            if not hmac.compare_digest(
                canonical.encode("utf-8"), self.target.endpoint.encode("utf-8")
            ):
                raise ValueError("push endpoint mismatch")
            if not isinstance(data, bytes) or not isinstance(headers, Mapping):
                raise ValueError("invalid prepared push request")
            serialized_headers: list[tuple[str, str]] = []
            for raw_name, raw_value in headers.items():
                if not isinstance(raw_name, str):
                    raise ValueError("invalid push header")
                name = raw_name.strip()
                value = str(raw_value).strip()
                lowered = name.lower()
                if (
                    not name
                    or re.fullmatch("[!#$%&'*+\\-.^_`|~0-9A-Za-z]+", name) is None
                    or lowered in _PROTECTED_PUSH_HEADERS
                    or lowered.startswith("proxy-")
                    or ("\r" in name)
                    or ("\n" in name)
                    or (":" in name)
                    or ("\r" in value)
                    or ("\n" in value)
                ):
                    raise ValueError("invalid push header")
                name.encode("ascii")
                value.encode("latin-1")
                serialized_headers.append((name, value))
            request_lines = [
                f"POST {self.target.request_target} HTTP/1.1",
                f"Host: {self.target.host_header}",
                "Connection: close",
                f"Content-Length: {len(data)}",
                *(f"{name}: {value}" for name, value in serialized_headers),
                "",
                "",
            ]
            request_head = "\r\n".join(request_lines).encode("latin-1")
            connect_timeout = min(float(timeout), self.attempt.remaining())
            raw_socket = socket.create_connection(
                (self.target.connect_ip, self.target.port), timeout=connect_timeout
            )
            self.attempt.register(raw_socket)
            self.attempt.check()
            context = ssl.create_default_context()
            if not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
                raise ValueError("unsafe TLS context")
            tls_socket = context.wrap_socket(
                raw_socket, server_hostname=self.target.tls_server_hostname
            )
            self.attempt.register(tls_socket)
            self.attempt.check()
            tls_socket.sendall(request_head + data)
            self.attempt.check()
            response = http.client.HTTPResponse(tls_socket)
            self.attempt.register(response)
            response.begin()
            self.attempt.check()
            status = response.status
            if type(status) is not int or not 200 <= status <= 599:
                raise ValueError("invalid push response status")
            response.read(PUSH_PROVIDER_RESPONSE_MAX_BYTES + 1)
            self.attempt.check()
            return SafePushResponse(status_code=status)
        except SafePushDispatchError:
            raise
        except Exception:
            raise SafePushDispatchError() from None
        finally:
            for resource in (response, tls_socket, raw_socket):
                if resource is None:
                    continue
                try:
                    resource.close()
                except Exception:
                    pass


def _send_notification_pinned_attempt(
    subscription: Mapping[str, Any],
    title: str,
    body: str,
    url: str,
    sent_at: str | None,
    attempt: _PushAttemptDeadline,
) -> None:
    attempt.check()
    validated = validate_push_subscription(subscription)
    target = build_pinned_push_target(validated.endpoint, attempt=attempt)
    attempt.check()
    payload = json.dumps(
        {
            "title": str(title)[:80],
            "body": str(body)[:240],
            "url": url,
            "sentAt": sent_at or _owner_runtime.now_iso(),
        },
        separators=(",", ":"),
        ensure_ascii=True,
    )
    attempt.check()
    response = (current_runtime().webpush or webpush)(
        subscription_info=validated.subscription,
        data=payload,
        vapid_private_key=_owner_settings.get_vapid_private_key(),
        vapid_claims=dict(_owner_settings.get_vapid_claims()),
        ttl=120,
        timeout=PUSH_TRANSPORT_TIMEOUT_SECONDS,
        requests_session=PinnedPushSession(target, attempt),
    )
    attempt.check()
    status_code = getattr(response, "status_code", None)
    if type(status_code) is not int or not 200 <= status_code <= 202:
        if type(status_code) is int and status_code in {404, 410}:
            raise PushProviderStatusError(status_code)
        raise SafePushDispatchError()


def send_notification_pinned(
    subscription: Mapping[str, Any],
    title: str,
    body: str,
    url: str,
    sent_at: str | None = None,
) -> None:
    runtime = current_runtime()
    attempt: _PushAttemptDeadline | None = None
    try:
        attempt = _PushAttemptDeadline(PUSH_TRANSPORT_DEADLINE_SECONDS)
        outcome: list[BaseException | None] = []
        finished = threading.Event()

        def scoped_worker() -> None:
            try:
                _send_notification_pinned_attempt(
                    subscription, title, body, url, sent_at, attempt
                )
            except BaseException as exc:
                outcome.append(exc)
            else:
                outcome.append(None)
            finally:
                finished.set()

        def worker() -> None:
            with runtime_scope(runtime):
                scoped_worker()

        if not _get_push_executor().submit(worker):
            attempt.cancel()
            raise SafePushDispatchError()
        if not finished.wait(attempt.remaining()):
            attempt.cancel()
            raise SafePushDispatchError()
        if not outcome:
            raise SafePushDispatchError()
        error = outcome[0]
        if error is None:
            return
        if isinstance(error, PushProviderStatusError):
            raise PushProviderStatusError(error.status_code) from None
        if isinstance(error, WebPushException):
            status_code = getattr(getattr(error, "response", None), "status_code", None)
            if type(status_code) is int and status_code in {404, 410}:
                raise PushProviderStatusError(status_code) from None
        raise SafePushDispatchError() from None
    except PushProviderStatusError:
        raise
    except SafePushDispatchError:
        raise
    except Exception:
        raise SafePushDispatchError() from None
    finally:
        if attempt is not None:
            attempt.cancel()


def push_identity_configured() -> bool:
    try:
        endpoint_hash("https://push.reclive.app/availability-check")
    except Exception:
        return False
    return True


def index_live_rows(rows: Sequence[SnapshotRow]) -> Dict[int, SnapshotRow]:
    output: Dict[int, SnapshotRow] = {}
    for row in rows:
        output[row.location_id] = row
    return output


def is_aware_utc_datetime(value: Any) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and (value.utcoffset() == timedelta(0))
    )


def is_fresh_snapshot(row: SnapshotRow, now: datetime) -> bool:
    fetched_at = row.fetched_at
    if not is_aware_utc_datetime(fetched_at):
        return False
    elapsed = now - fetched_at
    return timedelta(0) <= elapsed <= OCCUPANCY_FRESHNESS


def round_nonnegative_percent(value: float) -> int:
    return max(0, math.floor(value + 0.5))


def compute_section_metrics(
    facility_id: int,
    section_key: str,
    live_index: Dict[int, SnapshotRow],
    now: datetime | None = None,
) -> Optional[Dict[str, Any]]:
    location_ids = _owner_sections.location_ids_for_section(facility_id, section_key)
    if not location_ids:
        return None
    metric_now = now if now is not None else _owner_runtime.now_utc()
    if not is_aware_utc_datetime(metric_now):
        raise ValueError("section metric time must be an aware UTC datetime")
    configured_locations = 0
    closed_locations = 0
    expected_open_capacity = 0
    observed_locations = 0
    observed_capacity = 0
    observed_count = 0
    for location_id in location_ids:
        configured_capacity = current_runtime().capacities.get(location_id)
        if type(configured_capacity) is not int or configured_capacity <= 0:
            continue
        configured_locations += 1
        row = live_index.get(location_id)
        fresh = row is not None and is_fresh_snapshot(row, metric_now)
        if fresh and row is not None and (row.is_closed is True):
            closed_locations += 1
            continue
        expected_open_capacity += configured_capacity
        if (
            fresh
            and row is not None
            and (row.is_closed is False)
            and (type(row.current_capacity) is int)
            and (row.current_capacity >= 0)
        ):
            observed_locations += 1
            observed_capacity += configured_capacity
            observed_count += row.current_capacity
    coverage = (
        observed_capacity / expected_open_capacity if expected_open_capacity > 0 else 0
    )
    all_configured_locations_closed = (
        configured_locations > 0 and closed_locations == configured_locations
    )
    if all_configured_locations_closed:
        status = "closed"
    elif expected_open_capacity <= 0:
        status = "unknown"
    elif coverage >= 0.8:
        status = "live"
    elif coverage >= 0.5:
        status = "partial"
    else:
        status = "insufficient"
    percent = (
        observed_count / observed_capacity * 100
        if status in {"live", "partial"} and observed_capacity > 0
        else None
    )
    return {
        "total": observed_count if observed_locations > 0 else None,
        "max": observed_capacity,
        "expectedOpenCapacity": expected_open_capacity,
        "coverage": coverage,
        "percent": percent,
        "status": status,
    }


def compute_fresh_section_metrics(
    facility_id: int,
    section_key: str,
    snapshots: Mapping[int, SnapshotRow],
    now: datetime,
) -> Optional[Dict[str, Any]]:
    metrics = compute_section_metrics(
        facility_id, section_key, dict(snapshots), now=now
    )
    if metrics is None:
        return None
    coverage = metrics.get("coverage")
    percent = metrics.get("percent")
    if (
        metrics.get("status") != "live"
        or type(coverage) not in {int, float}
        or (not math.isfinite(float(coverage)))
        or (float(coverage) < 0.8)
        or (type(percent) not in {int, float})
        or (not math.isfinite(float(percent)))
    ):
        return None
    return metrics


def facility_notification_url(facility_id: int) -> str:
    if facility_id == 1186:
        return "/nick"
    if facility_id == 1656:
        return "/bakke"
    raise ValueError("unsupported facility")


def _evaluator_store_unavailable() -> HTTPException:
    return HTTPException(status_code=503, detail="push_evaluator_store_unavailable")


def _base_evaluator_result(now: datetime) -> Dict[str, Any]:
    return {
        "status": "ok",
        "rules": 0,
        "sent": 0,
        "failed": 0,
        "skippedThreshold": 0,
        "skippedCooldown": 0,
        "skippedMissingSection": 0,
        "skippedInactive": 0,
        "skippedLocked": 0,
        "evaluatedAt": now.isoformat(),
    }


def _sample_evaluator_now(
    fixed_now: datetime | None, *, not_before: datetime | None = None
) -> datetime:
    sample = fixed_now if fixed_now is not None else _owner_runtime.now_utc()
    if not is_aware_utc_datetime(sample) or (
        not_before is not None and sample < not_before
    ):
        raise ValueError("invalid evaluator time")
    return sample


def _fresh_ingestion_at(value: object, now: datetime) -> bool:
    if not is_aware_utc_datetime(value):
        return False
    age = now - value
    return timedelta(0) <= age <= OCCUPANCY_FRESHNESS


def _rule_schedule_gate(
    rule: _owner_repositories_push_rules.PushRuleRecord,
    schedule_payload: Mapping[str, Any],
    at: datetime,
) -> bool:
    return (
        rule.expires_at > at
        and _owner_sections.canonical_section_key(rule.section_key) == rule.section_key
        and bool(
            _owner_sections.location_ids_for_section(rule.facility_id, rule.section_key)
        )
        and _owner_facility_schedule.official_facility_is_open(
            schedule_payload,
            rule.facility_id,
            at,
            stale_after_seconds=current_runtime().settings.schedule_stale_after_seconds,
        )
    )


def _rule_snapshot_gate(
    rule: _owner_repositories_push_rules.PushRuleRecord,
    schedule_payload: Mapping[str, Any],
    snapshot: Any,
    live_index: Mapping[int, SnapshotRow],
    at: datetime,
) -> tuple[str, int | None]:
    if not _rule_schedule_gate(rule, schedule_payload, at):
        return ("missing", None)
    if not _fresh_ingestion_at(snapshot.last_successful_fetch_at, at):
        return ("missing", None)
    metrics = compute_fresh_section_metrics(
        rule.facility_id, rule.section_key, live_index, at
    )
    if metrics is None:
        return ("missing", None)
    coverage_value = metrics.get("coverage")
    percent_value = metrics.get("percent")
    if (
        metrics.get("status") != "live"
        or type(coverage_value) not in {int, float}
        or (not math.isfinite(float(coverage_value)))
        or (float(coverage_value) < 0.8)
        or (type(percent_value) not in {int, float})
        or (not math.isfinite(float(percent_value)))
    ):
        return ("missing", None)
    percent = round_nonnegative_percent(float(percent_value))
    if percent > rule.threshold:
        return ("threshold", None)
    return ("eligible", percent)


def _provider_failure_state(exc: Exception) -> tuple[str, str]:
    status_code: object = None
    if isinstance(exc, PushProviderStatusError):
        status_code = exc.status_code
    elif isinstance(exc, WebPushException):
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if type(status_code) is int and status_code in {404, 410}:
        return ("invalid_subscription", f"webpush_{status_code}")
    return ("failed", "webpush_failed")


def evaluate_rules_once(
    now: datetime | None = None,
    snapshot_reader: Any | None = None,
    repository_factory: RepositoryFactory = SnapshotRepository,
    *,
    facility_filter: int | None = None,
    section_filter: str | None = None,
) -> Dict[str, Any]:
    try:
        evaluation_now = _sample_evaluator_now(now)
    except Exception:
        raise _evaluator_store_unavailable() from None
    result = _base_evaluator_result(evaluation_now)
    connection = None
    try:
        try:
            connection = _owner_repositories_push_rules.db_acquire_evaluator_lock()
        except Exception:
            raise _evaluator_store_unavailable() from None
        if connection is None:
            result["skippedLocked"] = 1
            return result
        try:
            store = _owner_repositories_push_rules.PushRuleRepository(connection)
            rules = store.load_candidates(evaluation_now)
            result["rules"] = len(rules)
            if not rules:
                connection.commit()
                return result
            schedule_payload = _owner_facility_schedule.load_facility_hours()
            reader = (
                snapshot_reader
                if snapshot_reader is not None
                else repository_factory(connection)
            )
        except Exception:
            _owner_repositories_push_rules._safe_rollback(connection)
            raise _evaluator_store_unavailable() from None
        claimed_any = False
        latest_evaluator_now = evaluation_now
        for rule in rules:
            if facility_filter is not None and rule.facility_id != facility_filter:
                continue
            if section_filter is not None and rule.section_key != section_filter:
                continue
            subscription = _decode_stored_subscription(rule.subscription_json)
            if subscription is None or _canonical_stored_endpoint(subscription) is None:
                result["skippedMissingSection"] += 1
                continue
            try:
                gate_now = _sample_evaluator_now(now, not_before=latest_evaluator_now)
            except Exception:
                result["failed"] += 1
                continue
            latest_evaluator_now = gate_now
            if not _rule_schedule_gate(rule, schedule_payload, gate_now):
                result["skippedMissingSection"] += 1
                continue
            try:
                connection.commit()
                snapshot = reader.fetch_live_snapshot(gate_now)
                live_index = index_live_rows(snapshot.rows)
            except Exception:
                _owner_repositories_push_rules._safe_rollback(connection)
                raise _evaluator_store_unavailable() from None
            try:
                claim_now = _sample_evaluator_now(now, not_before=latest_evaluator_now)
            except Exception:
                result["failed"] += 1
                continue
            latest_evaluator_now = claim_now
            final_status, percent = _rule_snapshot_gate(
                rule, schedule_payload, snapshot, live_index, claim_now
            )
            if final_status == "missing":
                result["skippedMissingSection"] += 1
                continue
            if final_status == "threshold":
                result["skippedThreshold"] += 1
                continue
            if percent is None:
                result["failed"] += 1
                continue
            try:
                claimed = store.claim(rule.id, claim_now)
            except Exception:
                _owner_repositories_push_rules._safe_rollback(connection)
                result["failed"] += 1
                continue
            if not claimed:
                continue
            try:
                connection.commit()
                claimed_any = True
            except Exception:
                _owner_repositories_push_rules._safe_rollback(connection)
                result["failed"] += 1
                continue
            section_label = (
                "entire facility" if rule.section_key == "overall" else rule.section_key
            )
            facility_label = current_runtime().facility_names.get(
                rule.facility_id, "Gym"
            )
            terminal_status = "sent"
            failure_code = None
            try:
                send_notification_pinned(
                    subscription,
                    title="RecLive Alert",
                    body=f"{facility_label} {section_label} is {percent}% full (at or below your {rule.threshold}% alert).",
                    url=facility_notification_url(rule.facility_id),
                    sent_at=claim_now.isoformat(),
                )
            except Exception as exc:
                terminal_status, failure_code = _provider_failure_state(exc)
            try:
                terminal_now = _sample_evaluator_now(
                    now, not_before=latest_evaluator_now
                )
            except Exception:
                result["failed"] += 1
                continue
            latest_evaluator_now = terminal_now
            try:
                if not store.finalize(
                    rule.id, terminal_now, terminal_status, failure_code
                ):
                    raise RuntimeError("push terminal transition lost")
                connection.commit()
            except Exception:
                _owner_repositories_push_rules._safe_rollback(connection)
                result["failed"] += 1
                continue
            if terminal_status == "sent":
                result["sent"] += 1
            else:
                result["failed"] += 1
        if not claimed_any:
            try:
                connection.commit()
            except Exception:
                _owner_repositories_push_rules._safe_rollback(connection)
                raise _evaluator_store_unavailable() from None
        return result
    finally:
        if connection is not None:
            try:
                _owner_repositories_push_rules.db_release_evaluator_lock(connection)
            except Exception:
                raise _evaluator_store_unavailable() from None


async def evaluator_loop() -> None:
    runtime = current_runtime()

    def evaluate_captured():
        with runtime_scope(runtime):
            return evaluate_rules_once()

    while True:
        try:
            if hasattr(asyncio, "to_thread"):
                await asyncio.to_thread(evaluate_captured)
            else:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, evaluate_captured)
        except Exception:
            print("[push-evaluator] error=push_evaluation_failed")
        await asyncio.sleep(
            max(30, current_runtime().settings.push.evaluator_interval_seconds)
        )


class IgnoreExtraModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class StrictPushModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class PushKeysInput(StrictPushModel):
    p256dh: str = Field(min_length=1, max_length=512)
    auth: str = Field(min_length=1, max_length=512)


def _is_safe_expiration_time(value: object) -> bool:
    if value is None:
        return True
    if type(value) is int:
        return value >= 0
    if type(value) is float:
        return math.isfinite(value) and value >= 0
    return False


class PushSubscriptionInput(StrictPushModel):
    endpoint: str = Field(min_length=1)
    keys: PushKeysInput
    expiration_time: int | float | None = Field(default=None, alias="expirationTime")

    @field_validator("expiration_time", mode="before")
    @classmethod
    def validate_expiration_time(cls, value: object) -> object:
        if not _is_safe_expiration_time(value):
            raise ValueError("expirationTime must be finite and nonnegative")
        return value


class PushRuleRequest(StrictPushModel):
    subscription: PushSubscriptionInput
    facility_id: Literal[1186, 1656] = Field(alias="facilityId")
    section_key: str = Field(alias="sectionKey", min_length=1, max_length=80)
    threshold: int = Field(ge=1, le=100)
    ttl_seconds: int | None = Field(default=None, alias="ttlSeconds", ge=1, le=604800)

    @model_validator(mode="after")
    def validate_configured_section(self, info: ValidationInfo) -> "PushRuleRequest":
        policy = info.context if isinstance(info.context, dict) else validation_policy()
        if normalize_section_key(self.section_key) != self.section_key:
            raise ValueError("sectionKey must already be canonical")
        if _owner_sections.canonical_section_key(self.section_key) != self.section_key:
            raise ValueError("sectionKey must already be canonical")
        if not _owner_sections.location_ids_for_section(
            self.facility_id, self.section_key, section_ids=policy["section_ids"]
        ):
            raise ValueError("sectionKey is not configured for the facility")
        if self.ttl_seconds is not None and self.ttl_seconds > policy["max_ttl"]:
            raise ValueError("ttlSeconds exceeds configured maximum")
        return self


class PushOwnershipRequest(StrictPushModel):
    subscription: PushSubscriptionInput


@dataclass(frozen=True)
class ValidatedSubscription:
    endpoint: str
    subscription: Dict[str, Any]


def _content_length_value(request: Request) -> int | None:
    raw_values = [
        value
        for name, value in request.scope.get("headers", [])
        if bytes(name).lower() == b"content-length"
    ]
    if not raw_values:
        return None
    if len(raw_values) != 1:
        raise _owner_repositories_push_rules._push_http_error(
            422, "invalid_push_content_length"
        )
    try:
        text = bytes(raw_values[0]).decode("ascii")
    except UnicodeDecodeError:
        raise _owner_repositories_push_rules._push_http_error(
            422, "invalid_push_content_length"
        ) from None
    if not re.fullmatch("[0-9]+", text):
        raise _owner_repositories_push_rules._push_http_error(
            422, "invalid_push_content_length"
        )
    normalized = text.lstrip("0") or "0"
    maximum = str(current_runtime().settings.push.body_max_bytes)
    if len(normalized) > len(maximum) or (
        len(normalized) == len(maximum) and normalized > maximum
    ):
        raise _owner_repositories_push_rules._push_http_error(
            413, "push_request_too_large"
        )
    return int(normalized, 10)


def _reject_json_constant(_value: str) -> None:
    raise ValueError("nonstandard JSON constant")


async def _read_limited_push_json(request: Request) -> object:
    declared_size = _content_length_value(request)
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > current_runtime().settings.push.body_max_bytes:
            raise _owner_repositories_push_rules._push_http_error(
                413, "push_request_too_large"
            )
        chunks.append(chunk)
    if declared_size is not None and declared_size != size:
        raise _owner_repositories_push_rules._push_http_error(
            422, "invalid_push_content_length"
        )
    try:
        text = b"".join(chunks).decode("utf-8")
        return json.loads(text, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise _owner_repositories_push_rules._push_http_error(
            422, "invalid_push_request"
        ) from None


def _validate_push_model(decoded: object, model_type: type[PushModelT]) -> PushModelT:
    try:
        return model_type.model_validate(decoded, context=validation_policy())
    except (RecursionError, ValidationError):
        raise _owner_repositories_push_rules._push_http_error(
            422, "invalid_push_request"
        ) from None


async def parse_limited_push_body(
    request: Request, model_type: type[PushModelT]
) -> PushModelT:
    decoded = await _read_limited_push_json(request)
    return _validate_push_model(decoded, model_type)


def _decoded_subscription_endpoint(decoded: object) -> str | None:
    if not isinstance(decoded, Mapping):
        return None
    subscription = decoded.get("subscription")
    if isinstance(subscription, Mapping):
        endpoint = subscription.get("endpoint")
    else:
        endpoint = decoded.get("endpoint")
    return endpoint if type(endpoint) is str else None


def rate_limit_public_push_write(request: Request, decoded: object | None) -> None:
    endpoint = _decoded_subscription_endpoint(decoded)
    try:
        if endpoint is not None:
            try:
                subject_hash = rate_limit_subject_hash("endpoint", endpoint)
            except ValueError:
                subject_hash = None
        else:
            subject_hash = None
        if subject_hash is None:
            client_host = (
                request.client.host.strip()
                if request.client is not None and request.client.host.strip()
                else "unavailable"
            )
            subject_hash = rate_limit_subject_hash("client", client_host)
    except Exception:
        raise _owner_repositories_push_rules._push_http_error(
            503, "push_rate_limit_store_unavailable"
        ) from None
    _owner_repositories_push_rules.record_public_push_write(subject_hash)


async def _parse_limited_push_write(
    request: Request, model_type: type[PushModelT]
) -> PushModelT:
    decoded: object | None = None
    parse_error: HTTPException | None = None
    try:
        decoded = await _read_limited_push_json(request)
    except HTTPException as exc:
        parse_error = exc
    rate_limit_public_push_write(request, decoded)
    if parse_error is not None:
        raise parse_error
    return _validate_push_model(decoded, model_type)


def _parse_push_rule_id(raw_rule_id: str) -> int:
    if (
        type(raw_rule_id) is not str
        or re.fullmatch("[1-9][0-9]{0,19}", raw_rule_id) is None
        or (
            len(raw_rule_id) == len(MYSQL_UNSIGNED_BIGINT_MAX_TEXT)
            and raw_rule_id > MYSQL_UNSIGNED_BIGINT_MAX_TEXT
        )
    ):
        raise _owner_repositories_push_rules._push_http_error(
            422, "invalid_push_request"
        )
    return int(raw_rule_id, 10)


def _strict_base64url_decode(value: object) -> bytes:
    if type(value) is not str or not 1 <= len(value) <= 512:
        raise ValueError("invalid base64url value")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError("invalid base64url value") from None
    if BASE64URL_VALUE.fullmatch(value) is None:
        raise ValueError("invalid base64url value")
    unpadded = encoded.rstrip(b"=")
    supplied_padding = len(encoded) - len(unpadded)
    required_padding = -len(unpadded) % 4
    if required_padding > 2 or supplied_padding not in {0, required_padding}:
        raise ValueError("invalid base64url padding")
    try:
        return base64.b64decode(
            unpadded + b"=" * required_padding, altchars=b"-_", validate=True
        )
    except (binascii.Error, ValueError):
        raise ValueError("invalid base64url value") from None


def validate_push_subscription(value: Mapping[str, Any]) -> ValidatedSubscription:
    allowed_fields = {"endpoint", "keys", "expirationTime"}
    if set(value) - allowed_fields or not {"endpoint", "keys"} <= set(value):
        raise _owner_repositories_push_rules._push_http_error(
            422, "invalid_push_subscription"
        )
    expiration_time = value.get("expirationTime")
    if not _is_safe_expiration_time(expiration_time):
        raise _owner_repositories_push_rules._push_http_error(
            422, "invalid_push_subscription"
        )
    endpoint_value = value.get("endpoint")
    keys_value = value.get("keys")
    if type(endpoint_value) is not str or not isinstance(keys_value, Mapping):
        raise _owner_repositories_push_rules._push_http_error(
            422, "invalid_push_subscription"
        )
    if set(keys_value) != {"p256dh", "auth"}:
        raise _owner_repositories_push_rules._push_http_error(
            422, "invalid_push_subscription"
        )
    p256dh_value = keys_value.get("p256dh")
    auth_value = keys_value.get("auth")
    try:
        endpoint = normalize_push_endpoint(endpoint_value)
        p256dh = _strict_base64url_decode(p256dh_value)
        auth = _strict_base64url_decode(auth_value)
    except (TypeError, ValueError):
        raise _owner_repositories_push_rules._push_http_error(
            422, "invalid_push_subscription"
        ) from None
    if len(p256dh) != 65 or p256dh[0] != 4 or len(auth) != 16:
        raise _owner_repositories_push_rules._push_http_error(
            422, "invalid_push_subscription"
        )
    return ValidatedSubscription(
        endpoint=endpoint,
        subscription={
            "endpoint": endpoint,
            "keys": {"p256dh": p256dh_value, "auth": auth_value},
        },
    )


def _validated_subscription_from_model(
    subscription: PushSubscriptionInput,
) -> ValidatedSubscription:
    return validate_push_subscription(subscription.model_dump(by_alias=True))


def subscribe_owned_push_rule(
    subscription: ValidatedSubscription,
    facility_id: int,
    section_key: str,
    threshold: int,
    ttl_seconds: int | None,
) -> Dict[str, Any]:
    created, rule = _owner_repositories_push_rules.db_subscribe_rule(
        subscription,
        facility_id=facility_id,
        section_key=section_key,
        threshold=threshold,
        ttl_seconds=ttl_seconds,
    )
    try:
        response = push_rule_response(rule)
    except ValueError:
        raise _owner_repositories_push_rules._rule_store_unavailable() from None
    return {"status": "ok", "created": created, "rule": response}


def list_owned_push_rules(subscription: ValidatedSubscription) -> Dict[str, Any]:
    rules: List[Dict[str, Any]] = []
    for record in _owner_repositories_push_rules._db_list_owned_rule_records(
        subscription
    ):
        try:
            rules.append(push_rule_response(record))
        except ValueError:
            continue
    return {"status": "ok", "rules": rules}


def cancel_owned_push_rule(
    subscription: ValidatedSubscription, rule_id: int
) -> Dict[str, Any]:
    if type(rule_id) is not int or rule_id <= 0:
        raise _owner_repositories_push_rules._push_http_error(
            404, "push_rule_not_found"
        )
    cancelled = _owner_repositories_push_rules.db_cancel_owned_rule(
        subscription, rule_id
    )
    return {"status": "ok", "cancelled": cancelled}


def cancel_all_owned_push_rules(subscription: ValidatedSubscription) -> Dict[str, Any]:
    cancelled = _owner_repositories_push_rules.db_cancel_all_owned_rules(subscription)
    return {"status": "ok", "cancelled": cancelled}


class PushDispatchRequest(StrictPushModel):
    facilityId: Optional[Literal[1186, 1656]] = None
    sectionKey: Optional[str] = Field(default=None, min_length=1, max_length=80)

    @model_validator(mode="after")
    def validate_filter(self, info: ValidationInfo) -> "PushDispatchRequest":
        policy = info.context if isinstance(info.context, dict) else validation_policy()
        if self.sectionKey is None:
            return self
        if self.facilityId is None:
            raise ValueError("facilityId is required with sectionKey")
        normalized = _owner_sections.canonical_section_key(self.sectionKey)
        if (
            normalized != self.sectionKey
            or not _owner_sections.location_ids_for_section(
                self.facilityId, normalized, section_ids=policy["section_ids"]
            )
        ):
            raise ValueError("sectionKey must be configured")
        return self


def validation_policy():
    runtime = current_runtime()
    return {
        "section_ids": runtime.section_ids,
        "max_ttl": runtime.settings.push.max_rule_ttl_seconds,
    }


_import_sys.modules.setdefault("server.reclive.push", _import_sys.modules[__name__])
_import_sys.modules.setdefault("reclive.push", _import_sys.modules[__name__])
