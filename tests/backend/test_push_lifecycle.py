import asyncio
import copy
import importlib
import inspect
import ipaddress
import json
import os
import runpy
import socket
import ssl
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Coroutine, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pymysql
import pytest
from fastapi import HTTPException
from pywebpush import WebPushException
from starlette.requests import Request

import forecast_api as api
from facility_schedule import (
    get_facility_schedule_open_state,
    official_facility_is_open,
    parse_schedule_date_range,
    parse_schedule_hours_window,
    parse_schedule_weekday_set,
)


VALID_HASH_KEY = "push-test-key-with-at-least-thirty-two-bytes"
VALID_ADMIN_TOKEN = "admin-token-0123456789-ABCDEFGHIJK"
VALID_TEST_P256DH = (
    "BGsX0fLhLEJH-Lzm5WOkQPJ3A32BLeszoPShOUXYmMKWT-NC4v4af5uO5-tKfA-"
    "eFivOM1drMV7Oy7ZAaDe_UfU"
)
VALID_TEST_AUTH = "A" * 22


def subscribe_payload(
    subscription: dict[str, object],
    **overrides: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "subscription": copy.deepcopy(subscription),
        "facilityId": 1186,
        "sectionKey": "overall",
        "threshold": 40,
    }
    payload.update(overrides)
    return payload


def captured_subscribe_response() -> dict[str, object]:
    return {
        "status": "ok",
        "created": True,
        "rule": {
            "id": 1,
            "facilityId": 1186,
            "sectionKey": "overall",
            "threshold": 40,
            "createdAt": "2026-09-01T12:00:00Z",
            "expiresAt": "2026-09-02T12:00:00Z",
            "status": "pending",
        },
    }


def stream_request(
    chunks: list[bytes],
    headers: list[tuple[bytes, bytes]] | None = None,
) -> Request:
    messages = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(chunks) - 1,
        }
        for index, chunk in enumerate(chunks)
    ]

    async def receive() -> dict[str, object]:
        if messages:
            return messages.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/push/subscribe",
            "headers": headers or [],
            "client": ("203.0.113.7", 41234),
        },
        receive,
    )


def run_lifespan_once() -> None:
    async def run() -> None:
        async with api.lifespan(api.app):
            pass

    asyncio.run(run())


@pytest.mark.parametrize("hash_key", [None, "endpoint-key-secret-sentinel"])
def test_production_rejects_missing_or_short_endpoint_hash_key_before_evaluator_start(
    monkeypatch: pytest.MonkeyPatch,
    hash_key: str | None,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    if hash_key is None:
        monkeypatch.delenv("PUSH_ENDPOINT_HASH_KEY", raising=False)
    else:
        monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", hash_key)
    monkeypatch.setenv("PUSH_ADMIN_ROUTES_ENABLED", "false")

    def unexpected_evaluator_check() -> bool:
        raise AssertionError("evaluator was checked before configuration validation")

    monkeypatch.setattr(api, "evaluator_enabled", unexpected_evaluator_check)

    with pytest.raises(RuntimeError, match="PUSH_ENDPOINT_HASH_KEY") as exc_info:
        run_lifespan_once()

    if hash_key is not None:
        assert hash_key not in str(exc_info.value)


def test_production_accepts_valid_endpoint_hash_key_without_network_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", VALID_HASH_KEY)
    monkeypatch.setenv("PUSH_ADMIN_ROUTES_ENABLED", "false")

    def unexpected_resolution(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"configuration attempted network access: {args}, {kwargs}")

    monkeypatch.setattr(socket, "getaddrinfo", unexpected_resolution)

    api.validate_push_configuration()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"PUSH_DEFAULT_RULE_TTL_SECONDS": 0}, "default rule TTL"),
        ({"PUSH_MAX_RULE_TTL_SECONDS": 0}, "maximum rule TTL"),
        ({"PUSH_MAX_ACTIVE_RULES_PER_ENDPOINT": 0}, "active-rule maximum"),
        ({"PUSH_WRITE_RATE_LIMIT": 0}, "write rate limit"),
        ({"PUSH_WRITE_RATE_WINDOW_SECONDS": 0}, "rate-limit window"),
        (
            {
                "PUSH_DEFAULT_RULE_TTL_SECONDS": 101,
                "PUSH_MAX_RULE_TTL_SECONDS": 100,
            },
            "default rule TTL",
        ),
        ({"PUSH_MAX_RULE_TTL_SECONDS": 604_801}, "maximum rule TTL"),
        ({"PUSH_MAX_ACTIVE_RULES_PER_ENDPOINT": 11}, "active-rule maximum"),
        ({"PUSH_WRITE_RATE_LIMIT": 21}, "write rate limit"),
    ],
)
def test_invalid_numeric_configuration_fails_in_development_before_evaluator_start(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, int],
    message: str,
) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    for name, value in overrides.items():
        monkeypatch.setattr(api, name, value)

    def unexpected_evaluator_check() -> bool:
        raise AssertionError("evaluator was checked before configuration validation")

    monkeypatch.setattr(api, "evaluator_enabled", unexpected_evaluator_check)

    with pytest.raises(RuntimeError, match=message):
        run_lifespan_once()


@pytest.mark.parametrize("app_env", ["development", "test", "production"])
@pytest.mark.parametrize("admin_token", ["", "short-secret-sentinel"])
def test_enabled_admin_token_requires_thirty_two_ascii_bytes_in_every_environment(
    monkeypatch: pytest.MonkeyPatch,
    app_env: str,
    admin_token: str,
) -> None:
    monkeypatch.setenv("APP_ENV", app_env)
    monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", VALID_HASH_KEY)
    monkeypatch.setenv("PUSH_ADMIN_ROUTES_ENABLED", "true")
    monkeypatch.setattr(api, "PUSH_ADMIN_TOKEN", admin_token)

    with pytest.raises(RuntimeError, match="PUSH_ADMIN_TOKEN") as exc_info:
        api.validate_push_configuration()

    if admin_token:
        assert admin_token not in str(exc_info.value)


@pytest.mark.parametrize(
    ("admin_token", "message"),
    [
        ("nonascii-secret-é" * 3, "ASCII"),
        ("a" * 513, "512"),
    ],
)
def test_enabled_admin_token_rejects_non_ascii_and_oversized_values_at_startup(
    monkeypatch: pytest.MonkeyPatch,
    admin_token: str,
    message: str,
) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("PUSH_ADMIN_ROUTES_ENABLED", "true")
    monkeypatch.setattr(api, "PUSH_ADMIN_TOKEN", admin_token)

    with pytest.raises(RuntimeError, match=message) as exc_info:
        api.validate_push_configuration()

    assert admin_token not in str(exc_info.value)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "development"),
        (" DeVeLoPmEnT ", "development"),
        ("test", "test"),
        ("PRODUCTION", "production"),
    ],
)
def test_app_environment_parses_only_recognized_values_without_cached_state(
    monkeypatch: pytest.MonkeyPatch,
    raw: str | None,
    expected: str,
) -> None:
    if raw is None:
        monkeypatch.delenv("APP_ENV", raising=False)
    else:
        monkeypatch.setenv("APP_ENV", raw)

    assert api.app_environment() == expected


def test_invalid_app_environment_fails_with_fixed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_value = "prodution-secret-sentinel"
    monkeypatch.setenv("APP_ENV", invalid_value)

    with pytest.raises(RuntimeError, match="APP_ENV") as exc_info:
        api.validate_push_configuration()

    assert invalid_value not in str(exc_info.value)


def test_disabled_admin_routes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PUSH_ADMIN_ROUTES_ENABLED", "false")
    monkeypatch.setattr(api, "PUSH_ADMIN_TOKEN", "configured-admin-token")

    with pytest.raises(HTTPException) as exc_info:
        api.require_admin_token("configured-admin-token")

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Admin routes are disabled"


def test_enabled_admin_route_compares_transport_safe_ascii_bytes_with_fixed_errors(
    monkeypatch: pytest.MonkeyPatch,
    push_test_client: Any,
) -> None:
    comparisons: list[tuple[bytes, bytes]] = []

    def compare_digest(supplied: bytes, expected: bytes) -> bool:
        comparisons.append((supplied, expected))
        return supplied == expected

    monkeypatch.setenv("PUSH_ADMIN_ROUTES_ENABLED", "true")
    monkeypatch.setattr(api, "PUSH_ADMIN_TOKEN", VALID_ADMIN_TOKEN)
    monkeypatch.setattr(api, "db_rules_count", lambda: 0)
    monkeypatch.setattr(api.hmac, "compare_digest", compare_digest)

    accepted = push_test_client.get(
        "/health/push",
        headers={"X-RecLive-Admin-Token": VALID_ADMIN_TOKEN},
    )
    rejected_secret = "wrong-admin-secret-sentinel"
    rejected = push_test_client.get(
        "/health/push",
        headers={"X-RecLive-Admin-Token": rejected_secret},
    )

    assert accepted.status_code == 200
    assert rejected.status_code == 401
    assert rejected.json() == {"detail": "Admin token is required"}
    assert VALID_ADMIN_TOKEN not in rejected.text
    assert rejected_secret not in rejected.text
    assert comparisons == [
        (VALID_ADMIN_TOKEN.encode("ascii"), VALID_ADMIN_TOKEN.encode("ascii")),
        (rejected_secret.encode("ascii"), VALID_ADMIN_TOKEN.encode("ascii")),
    ]


def test_enabled_route_with_empty_configured_token_and_absent_header_is_fixed_503(
    monkeypatch: pytest.MonkeyPatch,
    push_test_client: Any,
) -> None:
    monkeypatch.setenv("PUSH_ADMIN_ROUTES_ENABLED", "true")
    monkeypatch.setattr(api, "PUSH_ADMIN_TOKEN", "")

    def unexpected_compare(*args: object) -> bool:
        raise AssertionError(f"compare_digest called for invalid configuration: {args}")

    monkeypatch.setattr(api.hmac, "compare_digest", unexpected_compare)

    response = push_test_client.get("/health/push")

    assert response.status_code == 503
    assert response.json() == {"detail": "Admin token is not configured"}


def test_disabled_admin_http_route_fails_before_database_access(
    push_test_client: Any,
) -> None:
    response = push_test_client.get(
        "/health/push",
        headers={"X-RecLive-Admin-Token": "anything"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Admin routes are disabled"}


def test_dispatch_admin_gate_runs_before_bounded_body_parser(
    monkeypatch: pytest.MonkeyPatch,
    push_test_client: Any,
) -> None:
    async def unexpected_parse(_request: Request) -> object:
        raise AssertionError("dispatch body parsed before its admin gate")

    monkeypatch.setattr(api, "_read_limited_push_json", unexpected_parse)

    disabled = push_test_client.post(
        "/api/push/dispatch",
        content=b"x" * (16 * 1024 + 1),
    )
    monkeypatch.setenv("PUSH_ADMIN_ROUTES_ENABLED", "true")
    monkeypatch.setattr(api, "PUSH_ADMIN_TOKEN", VALID_ADMIN_TOKEN)
    unauthenticated = push_test_client.post(
        "/api/push/dispatch",
        content=b"x" * (16 * 1024 + 1),
    )

    assert disabled.status_code == 503
    assert disabled.json() == {"detail": "Admin routes are disabled"}
    assert unauthenticated.status_code == 401
    assert unauthenticated.json() == {"detail": "Admin token is required"}


@pytest.mark.parametrize(
    ("content", "status_code", "detail"),
    [
        (b"x" * (16 * 1024 + 1), 413, "push_request_too_large"),
        (b"{", 422, "invalid_push_request"),
    ],
)
def test_authenticated_dispatch_uses_fixed_bounded_body_errors(
    monkeypatch: pytest.MonkeyPatch,
    push_test_client: Any,
    content: bytes,
    status_code: int,
    detail: str,
) -> None:
    monkeypatch.setenv("PUSH_ADMIN_ROUTES_ENABLED", "true")
    monkeypatch.setattr(api, "PUSH_ADMIN_TOKEN", VALID_ADMIN_TOKEN)
    monkeypatch.setattr(
        api,
        "evaluate_rules_once",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid dispatch reached evaluator")
        ),
    )

    response = push_test_client.post(
        "/api/push/dispatch",
        content=content,
        headers={"X-RecLive-Admin-Token": VALID_ADMIN_TOKEN},
    )

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}


def test_authenticated_dispatch_preserves_strict_facility_filter_behavior(
    monkeypatch: pytest.MonkeyPatch,
    push_test_client: Any,
) -> None:
    captured_filters: list[dict[str, object]] = []

    def capture_evaluation(**filters: object) -> dict[str, object]:
        captured_filters.append(filters)
        return {"status": "ok", "rules": 0}

    monkeypatch.setenv("PUSH_ADMIN_ROUTES_ENABLED", "true")
    monkeypatch.setattr(api, "PUSH_ADMIN_TOKEN", VALID_ADMIN_TOKEN)
    monkeypatch.setattr(api, "evaluate_rules_once", capture_evaluation)

    response = push_test_client.post(
        "/api/push/dispatch",
        json={"facilityId": 1186, "sectionKey": "overall"},
        headers={"X-RecLive-Admin-Token": VALID_ADMIN_TOKEN},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "rules": 0}
    assert captured_filters == [
        {"facility_filter": 1186, "section_filter": "overall"}
    ]


def test_push_availability_requires_endpoint_hash_key_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PUSH_ENDPOINT_HASH_KEY", raising=False)
    monkeypatch.setattr(api, "push_db_available", lambda: True)
    monkeypatch.setattr(api, "push_vapid_configured", lambda: True)

    response = api.push_availability()

    assert response["alertsAvailable"] is False
    assert response["reason"] == "push_identity_unconfigured"
    assert "PUSH_ENDPOINT_HASH_KEY" not in str(response)


def test_push_availability_accepts_ready_endpoint_hash_key_without_exposing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unique_hash_key = "a" * 42
    monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", unique_hash_key)
    monkeypatch.setattr(api, "push_db_available", lambda: True)
    monkeypatch.setattr(api, "push_vapid_configured", lambda: True)

    response = api.push_availability()

    assert response["alertsAvailable"] is True
    assert response["reason"] is None
    assert unique_hash_key not in str(response)


def test_lifespan_validates_before_start_and_cancels_its_evaluator_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeTask:
        def done(self) -> bool:
            return False

        def cancel(self) -> None:
            events.append("cancel")

        def __await__(self):
            async def wait() -> None:
                events.append("await")
                raise asyncio.CancelledError

            return wait().__await__()

    fake_task = FakeTask()

    def validate() -> None:
        events.append("validate")

    def enabled() -> bool:
        events.append("enabled")
        return True

    async def evaluator() -> None:
        return None

    def create_task(coroutine: Coroutine[Any, Any, None]) -> FakeTask:
        events.append("create")
        coroutine.close()
        return fake_task

    monkeypatch.setattr(api, "validate_push_configuration", validate)
    monkeypatch.setattr(api, "evaluator_enabled", enabled)
    monkeypatch.setattr(api, "evaluator_loop", evaluator)
    monkeypatch.setattr(api.asyncio, "create_task", create_task)
    monkeypatch.setattr(api, "EVALUATOR_TASK", None)

    run_lifespan_once()

    assert events == ["validate", "enabled", "create", "cancel", "await"]
    assert api.EVALUATOR_TASK is None


def test_lifespan_does_not_cancel_a_task_it_did_not_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class ExistingTask:
        def done(self) -> bool:
            return False

        def cancel(self) -> None:
            events.append("cancel")

    existing_task = ExistingTask()
    monkeypatch.setattr(api, "validate_push_configuration", lambda: None)
    monkeypatch.setattr(api, "evaluator_enabled", lambda: True)
    monkeypatch.setattr(api, "EVALUATOR_TASK", existing_task)

    run_lifespan_once()

    assert events == []
    assert api.EVALUATOR_TASK is existing_task


def test_lifespan_cancels_and_awaits_owned_task_when_body_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeTask:
        def cancel(self) -> None:
            events.append("cancel")

        def __await__(self):
            async def wait() -> None:
                events.append("await")
                raise asyncio.CancelledError

            return wait().__await__()

    fake_task = FakeTask()

    async def evaluator() -> None:
        return None

    def create_task(coroutine: Coroutine[Any, Any, None]) -> FakeTask:
        coroutine.close()
        return fake_task

    monkeypatch.setattr(api, "validate_push_configuration", lambda: None)
    monkeypatch.setattr(api, "evaluator_enabled", lambda: True)
    monkeypatch.setattr(api, "evaluator_loop", evaluator)
    monkeypatch.setattr(api.asyncio, "create_task", create_task)
    monkeypatch.setattr(api, "EVALUATOR_TASK", None)

    async def run() -> None:
        async with api.lifespan(api.app):
            events.append("body")
            raise LookupError("test body failure")

    with pytest.raises(LookupError, match="test body failure"):
        asyncio.run(run())

    assert events == ["body", "cancel", "await"]
    assert api.EVALUATOR_TASK is None


def test_lifespan_preserves_replacement_while_cleaning_up_owned_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class OwnedTask:
        def cancel(self) -> None:
            events.append("owned-cancel")

        def __await__(self):
            async def wait() -> None:
                events.append("owned-await")
                raise asyncio.CancelledError

            return wait().__await__()

    class ReplacementTask:
        def cancel(self) -> None:
            events.append("replacement-cancel")

    owned_task = OwnedTask()
    replacement_task = ReplacementTask()

    async def evaluator() -> None:
        return None

    def create_task(coroutine: Coroutine[Any, Any, None]) -> OwnedTask:
        coroutine.close()
        return owned_task

    monkeypatch.setattr(api, "validate_push_configuration", lambda: None)
    monkeypatch.setattr(api, "evaluator_enabled", lambda: True)
    monkeypatch.setattr(api, "evaluator_loop", evaluator)
    monkeypatch.setattr(api.asyncio, "create_task", create_task)
    monkeypatch.setattr(api, "EVALUATOR_TASK", None)

    async def run() -> None:
        async with api.lifespan(api.app):
            api.EVALUATOR_TASK = replacement_task  # type: ignore[assignment]

    asyncio.run(run())

    assert events == ["owned-cancel", "owned-await"]
    assert api.EVALUATOR_TASK is replacement_task


@pytest.mark.parametrize(
    "subscription",
    [
        {
            "endpoint": "http://push.reclive-notify.net/a",
            "keys": {
                "p256dh": VALID_TEST_P256DH,
                "auth": VALID_TEST_AUTH,
            },
        },
        {
            "endpoint": "https://127.0.0.1/a",
            "keys": {
                "p256dh": VALID_TEST_P256DH,
                "auth": VALID_TEST_AUTH,
            },
        },
        {
            "endpoint": "https://push.example/a",
            "keys": {
                "p256dh": VALID_TEST_P256DH,
                "auth": VALID_TEST_AUTH,
            },
        },
        {
            "endpoint": "https://push.reclive-notify.net/" + "x" * 2049,
            "keys": {
                "p256dh": VALID_TEST_P256DH,
                "auth": VALID_TEST_AUTH,
            },
        },
    ],
)
def test_subscribe_rejects_unsafe_endpoint_with_fixed_private_response(
    push_test_client: Any,
    subscription: dict[str, object],
) -> None:
    response = push_test_client.post(
        "/api/push/subscribe",
        json=subscribe_payload(subscription),
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_push_subscription"}
    assert str(subscription["endpoint"]) not in response.text


@pytest.mark.parametrize(
    ("p256dh", "auth"),
    [
        ("AA AA", VALID_TEST_AUTH),
        ("AA=AA", VALID_TEST_AUTH),
        ("é", VALID_TEST_AUTH),
        ("AA", VALID_TEST_AUTH),
        (
            "AGsX0fLhLEJH-Lzm5WOkQPJ3A32BLeszoPShOUXYmMKWT-NC4v4af5uO5-tKfA-eFivOM1drMV7Oy7ZAaDe_UfU",
            VALID_TEST_AUTH,
        ),
        (VALID_TEST_P256DH, "AA"),
        (VALID_TEST_P256DH, "AA=A"),
    ],
)
def test_subscribe_strictly_rejects_invalid_subscription_keys_without_echo(
    monkeypatch: pytest.MonkeyPatch,
    push_test_client: Any,
    valid_subscription: dict[str, object],
    p256dh: str,
    auth: str,
) -> None:
    invalid = copy.deepcopy(valid_subscription)
    invalid["keys"] = {"p256dh": p256dh, "auth": auth}
    monkeypatch.setattr(
        api,
        "subscribe_owned_push_rule",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"invalid subscription reached storage: {kwargs}")
        ),
        raising=False,
    )

    response = push_test_client.post(
        "/api/push/subscribe",
        json=subscribe_payload(invalid),
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_push_subscription"}
    assert p256dh not in response.text
    assert auth not in response.text


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("expiration_time", ["absent", None, 0, 1234.5])
def test_valid_subscription_accepts_key_padding_and_discards_expiration_time(
    monkeypatch: pytest.MonkeyPatch,
    push_test_client: Any,
    valid_subscription: dict[str, object],
    padded: bool,
    expiration_time: object,
) -> None:
    subscription = copy.deepcopy(valid_subscription)
    keys = dict(subscription["keys"])  # type: ignore[arg-type]
    if padded:
        keys["p256dh"] = f"{keys['p256dh']}="
        keys["auth"] = f"{keys['auth']}=="
    subscription["keys"] = keys
    if expiration_time != "absent":
        subscription["expirationTime"] = expiration_time

    captured: dict[str, object] = {}

    def capture_rule(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return captured_subscribe_response()

    monkeypatch.setattr(
        api,
        "subscribe_owned_push_rule",
        capture_rule,
        raising=False,
    )

    response = push_test_client.post(
        "/api/push/subscribe",
        json=subscribe_payload(subscription),
    )

    assert response.status_code == 200
    captured_subscription = captured["subscription"]
    assert isinstance(captured_subscription, api.ValidatedSubscription)
    assert captured_subscription.subscription == {
        "endpoint": "https://push.reclive-notify.net/subscription-a",
        "keys": keys,
    }
    assert "expirationTime" not in json.dumps(captured_subscription.subscription)


@pytest.mark.parametrize(
    "expiration_time",
    [True, -1, "1234", "Infinity"],
)
def test_subscribe_rejects_unsafe_expiration_time(
    push_test_client: Any,
    valid_subscription: dict[str, object],
    expiration_time: object,
) -> None:
    subscription = copy.deepcopy(valid_subscription)
    subscription["expirationTime"] = expiration_time

    response = push_test_client.post(
        "/api/push/subscribe",
        json=subscribe_payload(subscription),
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_push_request"}


@pytest.mark.parametrize("level", ["top", "subscription", "keys"])
def test_subscribe_rejects_extra_fields_at_every_level_without_echoing_them(
    push_test_client: Any,
    valid_subscription: dict[str, object],
    level: str,
) -> None:
    payload = subscribe_payload(valid_subscription)
    secret = f"{level}-extra-secret-sentinel"
    if level == "top":
        payload["extra"] = secret
    elif level == "subscription":
        subscription = payload["subscription"]
        assert isinstance(subscription, dict)
        subscription["extra"] = secret
    else:
        subscription = payload["subscription"]
        assert isinstance(subscription, dict)
        keys = subscription["keys"]
        assert isinstance(keys, dict)
        keys["extra"] = secret

    response = push_test_client.post("/api/push/subscribe", json=payload)

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_push_request"}
    assert secret not in response.text


@pytest.mark.parametrize(
    "overrides",
    [
        {"facilityId": 9999},
        {"facilityId": "1186"},
        {"facilityId": True},
        {"threshold": 0},
        {"threshold": 101},
        {"threshold": 1.5},
        {"threshold": "40"},
        {"threshold": True},
        {"ttlSeconds": 0},
        {"ttlSeconds": 604_801},
        {"ttlSeconds": 1.5},
        {"ttlSeconds": "60"},
        {"ttlSeconds": True},
        {"sectionKey": "Fitness Floors"},
        {"sectionKey": "fitness  floors"},
        {"sectionKey": " fitness floors"},
        {"sectionKey": "unknown section"},
        {"sectionKey": "x" * 81},
        {"sectionKey": ["overall"]},
    ],
)
def test_subscribe_strictly_rejects_invalid_rule_fields(
    push_test_client: Any,
    valid_subscription: dict[str, object],
    overrides: dict[str, object],
) -> None:
    response = push_test_client.post(
        "/api/push/subscribe",
        json=subscribe_payload(valid_subscription, **overrides),
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_push_request"}


def test_subscribe_accepts_configured_canonical_section_with_single_space(
    monkeypatch: pytest.MonkeyPatch,
    push_test_client: Any,
    valid_subscription: dict[str, object],
) -> None:
    captured: dict[str, object] = {}

    def capture_rule(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return captured_subscribe_response()

    monkeypatch.setattr(
        api,
        "subscribe_owned_push_rule",
        capture_rule,
        raising=False,
    )

    response = push_test_client.post(
        "/api/push/subscribe",
        json=subscribe_payload(
            valid_subscription,
            sectionKey="fitness floors",
            ttlSeconds=60,
            threshold=100,
        ),
    )

    assert response.status_code == 200
    assert captured["section_key"] == "fitness floors"


def test_body_limit_allows_exactly_16_kib_to_reach_json_parsing(
    push_test_client: Any,
) -> None:
    response = push_test_client.post(
        "/api/push/subscribe",
        content=b"{" + b"x" * (16 * 1024 - 1),
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_push_request"}


def test_body_limit_rejects_first_byte_above_16_kib(
    push_test_client: Any,
) -> None:
    response = push_test_client.post(
        "/api/push/subscribe",
        content=b"{" + b"x" * (16 * 1024),
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "push_request_too_large"}


def test_stream_body_limit_stops_on_the_first_oversized_chunk() -> None:
    request = stream_request([b"x" * (16 * 1024), b"y"])

    async def parse() -> None:
        await api.parse_limited_push_body(request, api.PushOwnershipRequest)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(parse())

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == "push_request_too_large"


def test_recursive_json_write_is_fixed_and_rate_limited_once(
    monkeypatch: pytest.MonkeyPatch,
    push_test_client: Any,
    push_repository: Any,
) -> None:
    recursive_json = ("[" * 8_000 + "0" + "]" * 8_000).encode("ascii")
    real_json_loads = json.loads

    def fail_at_parser_depth(value: object, *args: object, **kwargs: object) -> object:
        if value == recursive_json.decode("ascii"):
            raise RecursionError("parser-depth-secret-sentinel")
        return real_json_loads(value, *args, **kwargs)

    monkeypatch.setattr(api.json, "loads", fail_at_parser_depth)

    response = push_test_client.post(
        "/api/push/subscribe",
        content=recursive_json,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_push_request"}
    assert push_repository.rate_limit_increments == 1
    assert push_repository.rate_limit_commits == 1
    assert push_repository.committed_rate_limit_total == 1


def test_recursive_json_rule_list_is_fixed_without_a_write_counter(
    monkeypatch: pytest.MonkeyPatch,
    push_test_client: Any,
    push_repository: Any,
) -> None:
    recursive_json = ("[" * 8_000 + "0" + "]" * 8_000).encode("ascii")
    real_json_loads = json.loads

    def fail_at_parser_depth(value: object, *args: object, **kwargs: object) -> object:
        if value == recursive_json.decode("ascii"):
            raise RecursionError("parser-depth-secret-sentinel")
        return real_json_loads(value, *args, **kwargs)

    monkeypatch.setattr(api.json, "loads", fail_at_parser_depth)

    response = push_test_client.post(
        "/api/push/rules/list",
        content=recursive_json,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_push_request"}
    assert push_repository.rate_limit_increments == 0
    assert push_repository.committed_rate_limit_total == 0


@pytest.mark.parametrize(
    ("content_length", "status_code", "detail"),
    [
        ("-1", 422, "invalid_push_content_length"),
        ("sixteen kib", 422, "invalid_push_content_length"),
        ("16385", 413, "push_request_too_large"),
    ],
)
def test_subscribe_safely_rejects_invalid_or_excessive_content_length(
    push_test_client: Any,
    content_length: str,
    status_code: int,
    detail: str,
) -> None:
    response = push_test_client.post(
        "/api/push/subscribe",
        headers={"content-length": content_length},
        content=b"{}",
    )

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert content_length not in response.text


@pytest.mark.parametrize(
    ("content_length", "expected_status", "expected_detail"),
    [
        ("9" * 5_000, 413, "push_request_too_large"),
        ("0" * 5_000 + "2", 422, "invalid_push_request"),
    ],
)
def test_thousands_digit_content_length_is_bounded_and_rate_limited_once(
    push_test_client: Any,
    push_repository: Any,
    content_length: str,
    expected_status: int,
    expected_detail: str,
) -> None:
    response = push_test_client.post(
        "/api/push/subscribe",
        headers={"content-length": content_length},
        content=b"{}",
    )

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
    assert content_length not in response.text
    assert push_repository.rate_limit_increments == 1
    assert push_repository.rate_limit_commits == 1
    assert push_repository.committed_rate_limit_total == 1


@pytest.mark.parametrize("values", [(b"2", b"2"), (b"2", b"3")])
def test_duplicate_content_length_is_rejected_before_stream_consumption(
    values: tuple[bytes, bytes],
) -> None:
    request = stream_request(
        [b"{}"],
        headers=[
            (b"content-length", values[0]),
            (b"content-length", values[1]),
        ],
    )

    async def parse() -> None:
        await api.parse_limited_push_body(request, api.PushOwnershipRequest)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(parse())

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "invalid_push_content_length"


def test_declared_content_length_must_match_the_consumed_stream() -> None:
    request = stream_request(
        [b"{}"],
        headers=[(b"content-length", b"3")],
    )

    async def parse() -> None:
        await api.parse_limited_push_body(request, api.PushOwnershipRequest)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(parse())

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "invalid_push_content_length"


def test_rule_list_is_bounded_without_consuming_write_counter(
    push_test_client: Any,
    push_repository: Any,
) -> None:
    response = push_test_client.post(
        "/api/push/rules/list",
        content=b"x" * (16 * 1024 + 1),
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "push_request_too_large"}
    assert push_repository.rate_limit_increments == 0
    assert push_repository.committed_rate_limit_total == 0


def test_valid_rule_list_delegates_without_consuming_write_counter(
    monkeypatch: pytest.MonkeyPatch,
    push_test_client: Any,
    push_repository: Any,
    valid_subscription: dict[str, object],
) -> None:
    monkeypatch.setattr(
        api,
        "list_owned_push_rules",
        lambda subscription: {"status": "ok", "rules": []},
    )

    response = push_test_client.post(
        "/api/push/rules/list",
        json={"subscription": valid_subscription},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "rules": []}
    assert push_repository.rate_limit_increments == 0
    assert push_repository.committed_rate_limit_total == 0


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("DELETE", "/api/push/rules/42"),
        ("POST", "/api/push/rules/cancel-all"),
    ],
)
def test_each_management_write_route_limits_malformed_attempt_once(
    push_test_client: Any,
    push_repository: Any,
    method: str,
    path: str,
) -> None:
    response = push_test_client.request(method, path, content=b"{")

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_push_request"}
    assert push_repository.rate_limit_increments == 1
    assert push_repository.rate_limit_commits == 1
    assert push_repository.committed_rate_limit_total == 1


@pytest.mark.parametrize(
    "raw_rule_id",
    [
        "not-a-number",
        "0",
        "-1",
        "+1",
        "1.0",
        "01",
        "18446744073709551616",
        "9" * 1_000,
    ],
)
def test_cancel_one_rejects_noncanonical_unsigned_bigint_after_one_rate_increment(
    monkeypatch: pytest.MonkeyPatch,
    push_test_client: Any,
    push_repository: Any,
    valid_subscription: dict[str, object],
    raw_rule_id: str,
) -> None:
    def unexpected_cancel(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("invalid rule ID reached cancellation")

    monkeypatch.setattr(api, "cancel_owned_push_rule", unexpected_cancel)

    response = push_test_client.request(
        "DELETE",
        f"/api/push/rules/{raw_rule_id}",
        json={"subscription": valid_subscription},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_push_request"}
    assert push_repository.rate_limit_increments == 1
    assert push_repository.rate_limit_commits == 1
    assert push_repository.committed_rate_limit_total == 1


def test_cancel_one_accepts_maximum_unsigned_bigint_after_one_rate_increment(
    monkeypatch: pytest.MonkeyPatch,
    push_test_client: Any,
    push_repository: Any,
    valid_subscription: dict[str, object],
) -> None:
    captured: list[int] = []

    def capture_cancel(
        _subscription: api.ValidatedSubscription,
        rule_id: int,
    ) -> dict[str, object]:
        captured.append(rule_id)
        return {"status": "ok", "cancelled": 1}

    monkeypatch.setattr(api, "cancel_owned_push_rule", capture_cancel)

    response = push_test_client.request(
        "DELETE",
        "/api/push/rules/18446744073709551615",
        json={"subscription": valid_subscription},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "cancelled": 1}
    assert captured == [18_446_744_073_709_551_615]
    assert push_repository.rate_limit_increments == 1
    assert push_repository.rate_limit_commits == 1
    assert push_repository.committed_rate_limit_total == 1


@pytest.mark.parametrize(
    ("request_kwargs", "expected_status", "expected_detail"),
    [
        (
            {"headers": {"content-length": "bad"}, "content": b"{}"},
            422,
            "invalid_push_content_length",
        ),
        (
            {"content": b"x" * (16 * 1024 + 1)},
            413,
            "push_request_too_large",
        ),
        ({"content": b"\xff"}, 422, "invalid_push_request"),
        ({"content": b"{"}, 422, "invalid_push_request"),
        ({"json": {}}, 422, "invalid_push_request"),
        (
            {
                "json": {
                    "subscription": {
                        "endpoint": "https://push.reclive-notify.net/a",
                        "keys": {"p256dh": "AA", "auth": "AA"},
                    },
                    "facilityId": 1186,
                    "sectionKey": "overall",
                    "threshold": 40,
                }
            },
            422,
            "invalid_push_subscription",
        ),
    ],
)
def test_every_malformed_push_write_commits_exactly_one_rate_increment(
    push_test_client: Any,
    push_repository: Any,
    request_kwargs: dict[str, object],
    expected_status: int,
    expected_detail: str,
) -> None:
    response = push_test_client.post("/api/push/subscribe", **request_kwargs)

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
    assert push_repository.rate_limit_increments == 1
    assert push_repository.rate_limit_commits == 1
    assert push_repository.committed_rate_limit_total == 1


def test_twenty_writes_are_allowed_and_twenty_first_is_committed_then_limited(
    monkeypatch: pytest.MonkeyPatch,
    push_test_client: Any,
    push_repository: Any,
    valid_subscription: dict[str, object],
) -> None:
    monkeypatch.setattr(
        api,
        "cancel_all_owned_push_rules",
        lambda subscription: {"status": "ok", "cancelled": 0},
    )
    payload = {"subscription": valid_subscription}

    for _ in range(20):
        response = push_test_client.post(
            "/api/push/rules/cancel-all",
            json=payload,
        )
        assert response.status_code == 200

    limited = push_test_client.post(
        "/api/push/rules/cancel-all",
        json=payload,
    )

    assert limited.status_code == 429
    assert limited.json() == {"detail": "push_write_rate_limited"}
    assert push_repository.rate_limit_increments == 21
    assert push_repository.rate_limit_commits == 21
    assert push_repository.committed_rate_limit_total == 21


def test_rate_increment_commits_before_later_subscription_validation_failure(
    push_test_client: Any,
    push_repository: Any,
    valid_subscription: dict[str, object],
) -> None:
    invalid = copy.deepcopy(valid_subscription)
    invalid["keys"] = {"p256dh": "AA", "auth": "AA"}

    response = push_test_client.post(
        "/api/push/subscribe",
        json=subscribe_payload(invalid),
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_push_subscription"}
    assert push_repository.rate_limit_commits == 1
    assert push_repository.committed_rate_limit_total == 1


def test_huge_finite_integer_expiration_time_is_accepted_then_discarded(
    monkeypatch: pytest.MonkeyPatch,
    push_test_client: Any,
    push_repository: Any,
    valid_subscription: dict[str, object],
) -> None:
    huge_expiration_time = 10**400
    subscription = copy.deepcopy(valid_subscription)
    subscription["expirationTime"] = huge_expiration_time
    model = api.PushSubscriptionInput.model_validate(subscription)
    assert model.expiration_time == huge_expiration_time

    captured: dict[str, object] = {}

    def capture_rule(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return captured_subscribe_response()

    monkeypatch.setattr(
        api,
        "subscribe_owned_push_rule",
        capture_rule,
        raising=False,
    )

    response = push_test_client.post(
        "/api/push/subscribe",
        json=subscribe_payload(subscription),
    )

    assert response.status_code == 200
    captured_subscription = captured["subscription"]
    assert isinstance(captured_subscription, api.ValidatedSubscription)
    assert captured_subscription.subscription == valid_subscription
    assert push_repository.rate_limit_commits == 1
    assert push_repository.committed_rate_limit_total == 1


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_json_numbers_are_rejected_and_rate_limited(
    push_test_client: Any,
    push_repository: Any,
    valid_subscription: dict[str, object],
    constant: str,
) -> None:
    payload = subscribe_payload(valid_subscription)
    subscription = payload["subscription"]
    assert isinstance(subscription, dict)
    subscription["expirationTime"] = None
    raw = json.dumps(payload).replace("null", constant)

    response = push_test_client.post(
        "/api/push/subscribe",
        content=raw.encode("ascii"),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_push_request"}
    assert push_repository.rate_limit_commits == 1
    assert push_repository.committed_rate_limit_total == 1


def test_rate_limit_uses_endpoint_identity_when_safe_and_client_fallback_otherwise(
    push_test_client: Any,
    push_repository: Any,
    valid_subscription: dict[str, object],
) -> None:
    invalid_schema = subscribe_payload(valid_subscription, extra="rejected")

    endpoint_scoped = push_test_client.post(
        "/api/push/subscribe",
        json=invalid_schema,
    )
    client_scoped = push_test_client.post(
        "/api/push/subscribe",
        content=b"{",
    )

    assert endpoint_scoped.status_code == 422
    assert client_scoped.status_code == 422
    assert len(push_repository.rate_limit_counts) == 2
    assert sorted(push_repository.rate_limit_counts.values()) == [1, 1]


def test_management_write_uses_the_same_hashed_endpoint_subject(
    monkeypatch: pytest.MonkeyPatch,
    push_test_client: Any,
    push_repository: Any,
    valid_subscription: dict[str, object],
) -> None:
    monkeypatch.setattr(
        api,
        "cancel_all_owned_push_rules",
        lambda subscription: {"status": "ok", "cancelled": 0},
    )

    management = push_test_client.post(
        "/api/push/rules/cancel-all",
        json={"subscription": valid_subscription},
    )
    invalid_schema = push_test_client.post(
        "/api/push/subscribe",
        json=subscribe_payload(valid_subscription, extra="rejected"),
    )

    assert management.status_code == 200
    assert invalid_schema.status_code == 422
    assert list(push_repository.rate_limit_counts.values()) == [2]


def test_rate_limit_floors_aware_utc_time_to_configured_window(
    monkeypatch: pytest.MonkeyPatch,
    push_test_client: Any,
    push_repository: Any,
) -> None:
    monkeypatch.setattr(
        api,
        "now_utc",
        lambda: datetime(2026, 9, 1, 12, 7, 59, tzinfo=timezone.utc),
    )

    response = push_test_client.post("/api/push/subscribe", content=b"{")

    assert response.status_code == 422
    [(subject_hash, window_started_at)] = push_repository.rate_limit_counts
    assert isinstance(subject_hash, bytes)
    assert window_started_at == datetime(2026, 9, 1, 12, 0)


def test_rate_limit_store_failure_is_sanitized_and_rolls_back(
    push_test_client: Any,
    push_repository: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    push_repository.fail_rate_limit_store = True

    response = push_test_client.post("/api/push/subscribe", content=b"{")

    assert response.status_code == 503
    assert response.json() == {"detail": "push_rate_limit_store_unavailable"}
    assert "database-secret-sentinel" not in response.text
    assert "database-secret-sentinel" not in capsys.readouterr().out
    assert push_repository.rate_limit_rollbacks == 1
    assert push_repository.rate_limit_commits == 0


def test_rate_limit_clock_failure_is_sanitized_before_database_access(
    monkeypatch: pytest.MonkeyPatch,
    push_test_client: Any,
    push_repository: Any,
) -> None:
    def broken_clock() -> datetime:
        raise RuntimeError("clock-secret-sentinel")

    monkeypatch.setattr(api, "now_utc", broken_clock)

    response = push_test_client.post("/api/push/subscribe", content=b"{")

    assert response.status_code == 503
    assert response.json() == {"detail": "push_rate_limit_store_unavailable"}
    assert "clock-secret-sentinel" not in response.text
    assert push_repository.rate_limit_increments == 0


def test_rate_limit_counter_never_persists_raw_endpoint_or_client_material(
    push_test_client: Any,
    push_repository: Any,
    valid_subscription: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    endpoint = str(valid_subscription["endpoint"])
    invalid_payload = subscribe_payload(valid_subscription, extra="schema-secret")

    response = push_test_client.post("/api/push/subscribe", json=invalid_payload)

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_push_request"}
    persisted = repr(push_repository.rate_limit_counts)
    output = capsys.readouterr()
    assert endpoint not in persisted
    assert "testclient" not in persisted
    assert endpoint not in response.text
    assert endpoint not in output.out
    assert endpoint not in output.err


def mysql_fetch_all(
    settings: dict[str, object],
    sql: str,
    params: tuple[object, ...] = (),
) -> tuple[tuple[object, ...], ...]:
    connection = pymysql.connect(**settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return tuple(cursor.fetchall())
    finally:
        connection.close()


def mysql_execute(
    settings: dict[str, object],
    sql: str,
    params: tuple[object, ...] = (),
) -> int:
    connection = pymysql.connect(**settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return int(cursor.lastrowid or cursor.rowcount or 0)
    finally:
        connection.close()


class TrackedMySQLConnection:
    def __init__(self, connection: Any) -> None:
        self.connection = connection
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self) -> Any:
        return self.connection.cursor()

    def commit(self) -> None:
        self.commits += 1
        self.connection.commit()

    def rollback(self) -> None:
        self.rollbacks += 1
        self.connection.rollback()

    def close(self) -> None:
        self.closed = True
        self.connection.close()


def insert_mysql_push_rule(
    settings: dict[str, object],
    subscription: dict[str, object],
    *,
    threshold: int,
    status: str = "pending",
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
    digest: bytes | None = None,
    stored_subscription: object | None = None,
) -> int:
    endpoint = str(subscription["endpoint"])
    created = created_at or datetime(2026, 9, 1, 11, 0, tzinfo=timezone.utc)
    expires = expires_at or datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    terminal = status in {
        "sent",
        "failed",
        "expired",
        "invalid_subscription",
        "cancelled",
    }
    return mysql_execute(
        settings,
        """
        INSERT INTO push_rules
            (endpoint_hash, subscription_json, facility_id, section_key,
             threshold, created_at, expires_at, status, claimed_at, finalized_at)
        VALUES (%s, %s, 1186, 'overall', %s, %s, %s, %s, %s, %s)
        """,
        (
            digest or api.endpoint_hash(endpoint),
            json.dumps(
                subscription if stored_subscription is None else stored_subscription,
                separators=(",", ":"),
            ),
            threshold,
            created.astimezone(timezone.utc).replace(tzinfo=None),
            expires.astimezone(timezone.utc).replace(tzinfo=None),
            status,
            created.astimezone(timezone.utc).replace(tzinfo=None)
            if status == "claimed"
            else None,
            created.astimezone(timezone.utc).replace(tzinfo=None)
            if terminal
            else None,
        ),
    )


@pytest.mark.mysql
def test_duplicate_subscribe_returns_same_safe_rule_without_extending_expiry(
    monkeypatch: pytest.MonkeyPatch,
    mysql_push_test_client: Any,
    migrated_push_database: dict[str, object],
    valid_subscription: dict[str, object],
) -> None:
    payload = subscribe_payload(valid_subscription, ttlSeconds=3600)

    first = mysql_push_test_client.post("/api/push/subscribe", json=payload)
    monkeypatch.setattr(
        api,
        "now_utc",
        lambda: datetime(2026, 9, 1, 12, 30, tzinfo=timezone.utc),
    )
    second = mysql_push_test_client.post("/api/push/subscribe", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    first_body = first.json()
    assert first_body["created"] is True
    assert second.json() == {
        "status": "ok",
        "created": False,
        "rule": first_body["rule"],
    }
    rule = first_body["rule"]
    assert set(rule) == {
        "id",
        "facilityId",
        "sectionKey",
        "threshold",
        "createdAt",
        "expiresAt",
        "status",
    }
    assert rule["createdAt"].endswith("Z")
    assert rule["expiresAt"] == "2026-09-01T13:00:00Z"
    assert str(valid_subscription["endpoint"]) not in second.text
    assert VALID_TEST_P256DH not in second.text
    [(expires_at, count)] = mysql_fetch_all(
        migrated_push_database,
        "SELECT expires_at, COUNT(*) FROM push_rules GROUP BY expires_at",
    )
    assert expires_at == datetime(2026, 9, 1, 13, 0)
    assert count == 1


@pytest.mark.mysql
def test_terminal_rule_allows_recreation_without_losing_audit_history(
    mysql_push_test_client: Any,
    migrated_push_database: dict[str, object],
    valid_subscription: dict[str, object],
) -> None:
    payload = subscribe_payload(valid_subscription)
    first = mysql_push_test_client.post("/api/push/subscribe", json=payload).json()

    cancelled = mysql_push_test_client.request(
        "DELETE",
        f"/api/push/rules/{first['rule']['id']}",
        json={"subscription": valid_subscription},
    )
    recreated = mysql_push_test_client.post("/api/push/subscribe", json=payload)

    assert cancelled.status_code == 200
    assert cancelled.json() == {"status": "ok", "cancelled": 1}
    assert recreated.status_code == 200
    assert recreated.json()["created"] is True
    assert recreated.json()["rule"]["id"] != first["rule"]["id"]
    assert mysql_fetch_all(
        migrated_push_database,
        "SELECT status, finalized_at IS NOT NULL FROM push_rules ORDER BY id",
    ) == (("cancelled", 1), ("pending", 0))


@pytest.mark.mysql
def test_pending_and_claimed_rules_both_count_toward_active_rule_limit(
    mysql_push_test_client: Any,
    migrated_push_database: dict[str, object],
    valid_subscription: dict[str, object],
) -> None:
    created_ids: list[int] = []
    for threshold in range(1, 11):
        response = mysql_push_test_client.post(
            "/api/push/subscribe",
            json=subscribe_payload(valid_subscription, threshold=threshold),
        )
        assert response.status_code == 200
        created_ids.append(response.json()["rule"]["id"])
    mysql_execute(
        migrated_push_database,
        "UPDATE push_rules SET status = 'claimed', claimed_at = %s WHERE id = %s",
        (datetime(2026, 9, 1, 12, 0), created_ids[0]),
    )

    duplicate_claimed = mysql_push_test_client.post(
        "/api/push/subscribe",
        json=subscribe_payload(valid_subscription, threshold=1),
    )
    eleventh = mysql_push_test_client.post(
        "/api/push/subscribe",
        json=subscribe_payload(valid_subscription, threshold=11),
    )

    assert duplicate_claimed.status_code == 409
    assert duplicate_claimed.json() == {"detail": "push_rule_in_progress"}
    assert eleventh.status_code == 409
    assert eleventh.json() == {"detail": "push_rule_limit_reached"}
    assert mysql_fetch_all(
        migrated_push_database,
        "SELECT COUNT(*) FROM push_rules WHERE status IN ('pending', 'claimed')",
    ) == ((10,),)


@pytest.mark.mysql
def test_forced_endpoint_hash_collision_never_grants_ownership(
    monkeypatch: pytest.MonkeyPatch,
    mysql_push_test_client: Any,
    valid_subscription: dict[str, object],
    other_subscription: dict[str, object],
) -> None:
    monkeypatch.setattr(api, "endpoint_hash", lambda _endpoint: b"x" * 32)
    created = mysql_push_test_client.post(
        "/api/push/subscribe",
        json=subscribe_payload(valid_subscription),
    )
    assert created.status_code == 200
    rule_id = created.json()["rule"]["id"]

    listed = mysql_push_test_client.post(
        "/api/push/rules/list",
        json={"subscription": other_subscription},
    )
    cancelled = mysql_push_test_client.request(
        "DELETE",
        f"/api/push/rules/{rule_id}",
        json={"subscription": other_subscription},
    )
    conflicting = mysql_push_test_client.post(
        "/api/push/subscribe",
        json=subscribe_payload(other_subscription),
    )

    assert listed.status_code == 200
    assert listed.json() == {"status": "ok", "rules": []}
    assert cancelled.status_code == 404
    assert cancelled.json() == {"detail": "push_rule_not_found"}
    assert conflicting.status_code == 409
    assert conflicting.json() == {"detail": "push_identity_conflict"}
    combined = listed.text + cancelled.text + conflicting.text
    assert str(valid_subscription["endpoint"]) not in combined
    assert str(other_subscription["endpoint"]) not in combined


@pytest.mark.mysql
def test_list_filters_unowned_expired_claimed_terminal_and_malformed_rows(
    mysql_push_test_client: Any,
    migrated_push_database: dict[str, object],
    valid_subscription: dict[str, object],
    other_subscription: dict[str, object],
) -> None:
    digest = api.endpoint_hash(str(valid_subscription["endpoint"]))
    first = insert_mysql_push_rule(
        migrated_push_database,
        valid_subscription,
        threshold=10,
        created_at=datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc),
    )
    second = insert_mysql_push_rule(
        migrated_push_database,
        valid_subscription,
        threshold=20,
        created_at=datetime(2026, 9, 1, 11, 0, tzinfo=timezone.utc),
    )
    insert_mysql_push_rule(
        migrated_push_database,
        other_subscription,
        threshold=30,
        digest=digest,
    )
    insert_mysql_push_rule(
        migrated_push_database,
        valid_subscription,
        threshold=40,
        expires_at=datetime(2026, 9, 1, 11, 59, tzinfo=timezone.utc),
    )
    insert_mysql_push_rule(
        migrated_push_database,
        valid_subscription,
        threshold=50,
        status="claimed",
    )
    insert_mysql_push_rule(
        migrated_push_database,
        valid_subscription,
        threshold=60,
        status="cancelled",
    )
    insert_mysql_push_rule(
        migrated_push_database,
        valid_subscription,
        threshold=70,
        stored_subscription={},
    )
    insert_mysql_push_rule(
        migrated_push_database,
        valid_subscription,
        threshold=80,
        stored_subscription={"endpoint": valid_subscription["endpoint"]},
    )
    insert_mysql_push_rule(
        migrated_push_database,
        valid_subscription,
        threshold=90,
        stored_subscription={**valid_subscription, "extra": "stored-secret"},
    )

    response = mysql_push_test_client.post(
        "/api/push/rules/list",
        json={"subscription": valid_subscription},
    )

    assert response.status_code == 200
    rules = response.json()["rules"]
    assert [rule["id"] for rule in rules] == [first, second]
    assert all(rule["status"] == "pending" for rule in rules)
    assert all(set(rule) == {
        "id",
        "facilityId",
        "sectionKey",
        "threshold",
        "createdAt",
        "expiresAt",
        "status",
    } for rule in rules)
    assert str(valid_subscription["endpoint"]) not in response.text
    assert VALID_TEST_AUTH not in response.text


@pytest.mark.mysql
def test_cancel_one_and_all_only_terminalize_owned_pending_rules(
    mysql_push_test_client: Any,
    migrated_push_database: dict[str, object],
    valid_subscription: dict[str, object],
    other_subscription: dict[str, object],
) -> None:
    digest = api.endpoint_hash(str(valid_subscription["endpoint"]))
    first = insert_mysql_push_rule(
        migrated_push_database, valid_subscription, threshold=10
    )
    second = insert_mysql_push_rule(
        migrated_push_database, valid_subscription, threshold=20
    )
    claimed = insert_mysql_push_rule(
        migrated_push_database,
        valid_subscription,
        threshold=30,
        status="claimed",
    )
    foreign = insert_mysql_push_rule(
        migrated_push_database,
        other_subscription,
        threshold=40,
        digest=digest,
    )

    first_cancel = mysql_push_test_client.request(
        "DELETE",
        f"/api/push/rules/{first}",
        json={"subscription": valid_subscription},
    )
    repeated = mysql_push_test_client.request(
        "DELETE",
        f"/api/push/rules/{first}",
        json={"subscription": valid_subscription},
    )
    foreign_cancel = mysql_push_test_client.request(
        "DELETE",
        f"/api/push/rules/{foreign}",
        json={"subscription": valid_subscription},
    )
    all_cancel = mysql_push_test_client.post(
        "/api/push/rules/cancel-all",
        json={"subscription": valid_subscription},
    )
    repeated_all = mysql_push_test_client.post(
        "/api/push/rules/cancel-all",
        json={"subscription": valid_subscription},
    )

    assert first_cancel.json() == {"status": "ok", "cancelled": 1}
    assert repeated.status_code == 404
    assert repeated.json() == {"detail": "push_rule_not_found"}
    assert foreign_cancel.status_code == 404
    assert all_cancel.json() == {"status": "ok", "cancelled": 1}
    assert repeated_all.json() == {"status": "ok", "cancelled": 0}
    assert mysql_fetch_all(
        migrated_push_database,
        "SELECT id, status, finalized_at IS NOT NULL FROM push_rules ORDER BY id",
    ) == (
        (first, "cancelled", 1),
        (second, "cancelled", 1),
        (claimed, "claimed", 0),
        (foreign, "pending", 0),
    )


@pytest.mark.mysql
def test_failed_cancel_rolls_back_expiry_cleanup_but_keeps_one_rate_increment(
    monkeypatch: pytest.MonkeyPatch,
    mysql_push_test_client: Any,
    migrated_push_database: dict[str, object],
    valid_subscription: dict[str, object],
) -> None:
    expired = insert_mysql_push_rule(
        migrated_push_database,
        valid_subscription,
        threshold=10,
        expires_at=datetime(2026, 9, 1, 11, 59, tzinfo=timezone.utc),
    )
    tracked: list[TrackedMySQLConnection] = []

    def open_tracked_connection(
        *,
        autocommit: bool = True,
    ) -> TrackedMySQLConnection:
        connection = pymysql.connect(
            **{**migrated_push_database, "autocommit": autocommit}
        )
        wrapped = TrackedMySQLConnection(connection)
        tracked.append(wrapped)
        return wrapped

    monkeypatch.setattr(api, "open_db_connection", open_tracked_connection)

    response = mysql_push_test_client.request(
        "DELETE",
        f"/api/push/rules/{expired + 10_000}",
        json={"subscription": valid_subscription},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "push_rule_not_found"}
    assert len(tracked) == 2
    rate_connection, lifecycle_connection = tracked
    assert (rate_connection.commits, rate_connection.rollbacks) == (1, 0)
    assert (lifecycle_connection.commits, lifecycle_connection.rollbacks) == (0, 1)
    assert rate_connection.closed is True
    assert lifecycle_connection.closed is True
    assert mysql_fetch_all(
        migrated_push_database,
        "SELECT request_count FROM push_rate_limits",
    ) == ((1,),)
    assert mysql_fetch_all(
        migrated_push_database,
        "SELECT status, finalized_at FROM push_rules WHERE id = %s",
        (expired,),
    ) == (("pending", None),)


def test_safe_rule_response_assumes_naive_mysql_utc_and_rejects_non_utc_values(
) -> None:
    record = api.PushRuleRecord(
        id=7,
        endpoint_hash=b"x" * 32,
        subscription_json="{}",
        facility_id=1186,
        section_key="overall",
        threshold=40,
        created_at=datetime(2026, 9, 1, 12, 0),
        expires_at=datetime(2026, 9, 2, 12, 0),
        status="pending",
        active_identity=1,
    )
    response = api.push_rule_response(record)
    assert response["createdAt"] == "2026-09-01T12:00:00Z"
    assert response["expiresAt"] == "2026-09-02T12:00:00Z"

    unsafe = api.PushRuleRecord(
        **{
            **record.__dict__,
            "created_at": datetime(
                2026,
                9,
                1,
                13,
                0,
                tzinfo=timezone(timedelta(hours=1)),
            ),
        }
    )
    with pytest.raises(ValueError, match="UTC"):
        api.push_rule_response(unsafe)


class EndpointLockCursor:
    def __init__(self, connection: "EndpointLockConnection") -> None:
        self.connection = connection
        self.row: tuple[object, ...] | None = None

    def __enter__(self) -> "EndpointLockCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, _params: tuple[object, ...]) -> None:
        if "GET_LOCK" not in sql:
            raise AssertionError("lifecycle continued after endpoint lock failure")
        if self.connection.failure:
            raise RuntimeError("database-lock-secret-sentinel")
        self.row = (0,)

    def fetchone(self) -> tuple[object, ...] | None:
        return self.row


class EndpointLockConnection:
    def __init__(self, *, failure: bool) -> None:
        self.failure = failure
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> EndpointLockCursor:
        return EndpointLockCursor(self)

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


class DuplicateRecoveryCursor:
    def __init__(self, connection: "DuplicateRecoveryConnection") -> None:
        self.connection = connection
        self.row: tuple[object, ...] | None = None
        self.rows: tuple[tuple[object, ...], ...] = ()
        self.rowcount = 0
        self.lastrowid = 0

    def __enter__(self) -> "DuplicateRecoveryCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, _params: tuple[object, ...]) -> None:
        normalized = " ".join(sql.split()).lower()
        self.row = None
        self.rows = ()
        if normalized.startswith("select get_lock"):
            self.row = (1,)
            return
        if normalized.startswith("select release_lock"):
            self.connection.released += 1
            self.row = (1,)
            return
        if normalized.startswith("select") and "expires_at <=" in normalized:
            self.rows = ()
            return
        if normalized.startswith("select count(*)"):
            self.row = (0,)
            return
        if normalized.startswith("select") and "active_identity is not null" in normalized:
            self.connection.identity_selects += 1
            if self.connection.identity_selects == 1:
                self.row = None
            else:
                self.row = self.connection.existing_row
            return
        if normalized.startswith("insert into push_rules"):
            raise pymysql.err.IntegrityError(
                1062,
                "duplicate-key-secret-sentinel",
            )
        raise AssertionError(f"unexpected duplicate recovery SQL: {normalized}")

    def fetchone(self) -> tuple[object, ...] | None:
        return self.row

    def fetchall(self) -> tuple[tuple[object, ...], ...]:
        return self.rows


class DuplicateRecoveryConnection:
    def __init__(
        self,
        subscription: dict[str, object],
        digest: bytes,
    ) -> None:
        self.existing_row = (
            91,
            digest,
            json.dumps(subscription, separators=(",", ":")),
            1186,
            "overall",
            40,
            datetime(2026, 9, 1, 12, 0),
            datetime(2026, 9, 2, 12, 0),
            "pending",
            1,
        )
        self.identity_selects = 0
        self.commits = 0
        self.rollbacks = 0
        self.released = 0
        self.closed = False

    def cursor(self) -> DuplicateRecoveryCursor:
        return DuplicateRecoveryCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("failure", "expected_detail"),
    [
        (False, "push_rule_store_busy"),
        (True, "push_rule_store_unavailable"),
    ],
)
def test_endpoint_lock_contention_and_database_failure_are_distinct_and_private(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    valid_subscription: dict[str, object],
    failure: bool,
    expected_detail: str,
) -> None:
    connection = EndpointLockConnection(failure=failure)
    monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", VALID_HASH_KEY)
    monkeypatch.setattr(api, "open_db_connection", lambda **_kwargs: connection)
    subscription = api.validate_push_subscription(valid_subscription)

    with pytest.raises(HTTPException) as exc_info:
        api.db_subscribe_rule(
            subscription,
            facility_id=1186,
            section_key="overall",
            threshold=40,
            ttl_seconds=None,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == expected_detail
    assert connection.rolled_back is True
    assert connection.closed is True
    captured = capsys.readouterr()
    combined = str(exc_info.value.detail) + captured.out + captured.err
    assert "database-lock-secret-sentinel" not in combined
    assert str(valid_subscription["endpoint"]) not in combined


def test_duplicate_key_recovery_reselects_and_returns_existing_pending_rule(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    valid_subscription: dict[str, object],
) -> None:
    monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", VALID_HASH_KEY)
    monkeypatch.setattr(
        api,
        "now_utc",
        lambda: datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc),
    )
    digest = api.endpoint_hash(str(valid_subscription["endpoint"]))
    connection = DuplicateRecoveryConnection(valid_subscription, digest)
    monkeypatch.setattr(api, "open_db_connection", lambda **_kwargs: connection)
    subscription = api.validate_push_subscription(valid_subscription)

    created, record = api.db_subscribe_rule(
        subscription,
        facility_id=1186,
        section_key="overall",
        threshold=40,
        ttl_seconds=None,
    )

    assert created is False
    assert record.id == 91
    assert connection.identity_selects == 2
    assert connection.commits == 1
    assert connection.rollbacks == 0
    assert connection.released == 1
    assert connection.closed is True
    captured = capsys.readouterr()
    assert "duplicate-key-secret-sentinel" not in captured.out + captured.err


@pytest.mark.mysql
def test_concurrent_tenth_and_eleventh_rules_leave_exactly_ten_and_release_lock(
    monkeypatch: pytest.MonkeyPatch,
    mysql_push_test_client: Any,
    migrated_push_database: dict[str, object],
    valid_subscription: dict[str, object],
) -> None:
    del mysql_push_test_client
    monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", VALID_HASH_KEY)
    subscription = api.validate_push_subscription(valid_subscription)
    for threshold in range(1, 10):
        created, _record = api.db_subscribe_rule(
            subscription,
            facility_id=1186,
            section_key="overall",
            threshold=threshold,
            ttl_seconds=None,
        )
        assert created is True

    actual_acquire_endpoint_lock = api._acquire_endpoint_lock
    first_lock_acquired = threading.Event()
    second_lock_entered = threading.Event()
    lock_call_guard = threading.Lock()
    lock_call_count = 0

    def coordinated_acquire_endpoint_lock(cursor: Any, lock_name: str) -> None:
        nonlocal lock_call_count
        with lock_call_guard:
            call_index = lock_call_count
            lock_call_count += 1
        if call_index == 0:
            actual_acquire_endpoint_lock(cursor, lock_name)
            first_lock_acquired.set()
            if not second_lock_entered.wait(5):
                pytest.fail("second subscription did not enter advisory-lock acquisition")
            return
        if call_index == 1:
            if not first_lock_acquired.wait(5):
                pytest.fail("first subscription did not acquire the advisory lock")
            second_lock_entered.set()
        actual_acquire_endpoint_lock(cursor, lock_name)

    monkeypatch.setattr(
        api,
        "_acquire_endpoint_lock",
        coordinated_acquire_endpoint_lock,
    )

    def submit(threshold: int) -> int:
        try:
            api.db_subscribe_rule(
                subscription,
                facility_id=1186,
                section_key="overall",
                threshold=threshold,
                ttl_seconds=None,
            )
        except HTTPException as exc:
            return exc.status_code
        return 200

    with ThreadPoolExecutor(max_workers=2) as executor:
        statuses = list(executor.map(submit, (10, 11)))

    assert sorted(statuses) == [200, 409]
    assert lock_call_count == 2
    assert first_lock_acquired.is_set()
    assert second_lock_entered.is_set()
    assert mysql_fetch_all(
        migrated_push_database,
        "SELECT COUNT(*) FROM push_rules WHERE status IN ('pending', 'claimed')",
    ) == ((10,),)
    digest = api.endpoint_hash(str(valid_subscription["endpoint"]))
    lock_name = f"reclive:push:{digest.hex()[:48]}"
    assert mysql_fetch_all(
        migrated_push_database,
        "SELECT IS_FREE_LOCK(%s)",
        (lock_name,),
    ) == ((1,),)


@pytest.mark.mysql
def test_migrated_schema_routes_work_and_legacy_raw_endpoint_routes_are_absent(
    mysql_push_test_client: Any,
    migrated_push_database: dict[str, object],
    valid_subscription: dict[str, object],
) -> None:
    created = mysql_push_test_client.post(
        "/api/push/subscribe",
        json=subscribe_payload(valid_subscription),
    )
    listed = mysql_push_test_client.post(
        "/api/push/rules/list",
        json={"subscription": valid_subscription},
    )
    legacy_exists = mysql_push_test_client.post(
        "/api/push/rules/exists",
        json=subscribe_payload(valid_subscription),
    )
    legacy_unsubscribe = mysql_push_test_client.post(
        "/api/push/unsubscribe",
        json={"endpoint": valid_subscription["endpoint"]},
    )

    assert created.status_code == 200
    assert listed.status_code == 200
    assert legacy_exists.status_code in {404, 405}
    assert legacy_unsubscribe.status_code == 404
    assert mysql_fetch_all(
        migrated_push_database,
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_schema = DATABASE() AND table_name = 'push_rules' "
        "AND column_name = 'endpoint'",
    ) == ((0,),)
    source = inspect.getsource(api)
    assert '@app.post("/api/push/rules/exists")' not in source
    assert '@app.post("/api/push/unsubscribe")' not in source
    assert not hasattr(api, "db_rule_exists")
    assert not hasattr(api, "db_delete_rules_by_endpoint")
    assert not hasattr(api, "db_upsert_rule")
    assert "WHERE endpoint = %s" not in source
    assert "(endpoint, subscription_json" not in source


CHICAGO = ZoneInfo("America/Chicago")
TASK4_NOW = datetime(2026, 9, 1, 17, 0, tzinfo=timezone.utc)


def task4_schedule_payload(
    rows: list[dict[str, str]],
    *,
    generated_at: str = "2026-09-01T16:59:00Z",
    successful_at: str = "2026-09-01T16:59:00Z",
    status: str = "ok",
    stale: bool = False,
    section_title: str = "Building Hours",
) -> dict[str, object]:
    return {
        "generatedAt": generated_at,
        "facilities": [
            {
                "facilityId": 1186,
                "status": status,
                "stale": stale,
                "lastSuccessfulAt": successful_at,
                "sections": [
                    {
                        "title": section_title,
                        "rows": rows,
                        "note": "No scheduled maintenance closures",
                    }
                ],
            }
        ],
    }


def test_schedule_parser_contract_handles_dates_weekdays_closed_and_overnight() -> None:
    assert parse_schedule_date_range("Aug 31 - Sep 4", 2026) == (
        datetime(2026, 8, 31).date(),
        datetime(2026, 9, 4).date(),
        5,
    )
    assert parse_schedule_date_range("Dec 31 - Jan 2", 2026) == (
        datetime(2026, 12, 31).date(),
        datetime(2027, 1, 2).date(),
        3,
    )
    assert parse_schedule_weekday_set("Mon-Fri") == {0, 1, 2, 3, 4}
    assert parse_schedule_weekday_set("Tues, Thurs") == {1, 3}
    assert parse_schedule_weekday_set("Weekends") == {5, 6}
    assert parse_schedule_hours_window("Closed") == (0, 0, True)
    assert parse_schedule_hours_window("Open 24 Hours") == (0, 1440, False)
    assert parse_schedule_hours_window("10:00 pm - 2:00 am") == (
        1320,
        1560,
        False,
    )
    assert parse_schedule_hours_window("midnight - midnight") is None
    assert parse_schedule_hours_window("24:00 - 1:00") is None


def test_schedule_date_parser_accepts_compact_ranges_and_rejects_trailing_text() -> None:
    assert parse_schedule_date_range("Aug 31-Sep 4", 2026) == (
        datetime(2026, 8, 31).date(),
        datetime(2026, 9, 4).date(),
        5,
    )
    assert parse_schedule_date_range("Dec 31-Jan 2", 2026) == (
        datetime(2026, 12, 31).date(),
        datetime(2027, 1, 2).date(),
        3,
    )
    assert parse_schedule_date_range("2026-08-31", 2026) == (
        datetime(2026, 8, 31).date(),
        datetime(2026, 8, 31).date(),
        1,
    )
    assert parse_schedule_date_range("2026-08-31-2026-09-04", 2026) == (
        datetime(2026, 8, 31).date(),
        datetime(2026, 9, 4).date(),
        5,
    )
    assert parse_schedule_date_range("Aug 31-Sep 4 Pool", 2026) is None


@pytest.mark.parametrize(
    "label",
    ["Pool Mon-Fri", "Basketball Courts Mon-Fri", "Climbing Mon-Fri"],
)
def test_schedule_weekday_parser_rejects_area_prefixes(label: str) -> None:
    assert parse_schedule_weekday_set(label) is None
    assert get_facility_schedule_open_state(
        [
            {
                "title": "Building Hours",
                "rows": [{"label": label, "hours": "Open 24 Hours"}],
            }
        ],
        datetime(2026, 9, 1, 12, 0, tzinfo=CHICAGO),
    ) is None


def test_schedule_date_override_is_more_specific_and_narrower_range_wins() -> None:
    sections = [
        {
            "title": "Building Hours",
            "rows": [
                {"label": "Daily", "hours": "6:00 am - 10:00 pm"},
                {"label": "Sep 1 - Sep 5", "hours": "Closed"},
                {"label": "Sep 1", "hours": "11:00 am - 1:00 pm"},
            ],
        }
    ]
    at = datetime(2026, 9, 1, 12, 0, tzinfo=CHICAGO)

    assert get_facility_schedule_open_state(sections, at) is True


def test_schedule_stable_source_order_breaks_exact_precedence_tie() -> None:
    sections = [
        {
            "title": "Building Hours",
            "rows": [
                {"label": "Sep 1", "hours": "Closed"},
                {"label": "Sep 1", "hours": "Open 24 Hours"},
            ],
        }
    ]

    assert (
        get_facility_schedule_open_state(
            sections,
            datetime(2026, 9, 1, 12, 0, tzinfo=CHICAGO),
        )
        is False
    )


@pytest.mark.parametrize(
    ("scoped_first", "expected"),
    [(True, False), (False, True)],
)
def test_schedule_title_date_and_row_date_share_specificity_and_source_tie(
    scoped_first: bool,
    expected: bool,
) -> None:
    scoped = {
        "title": "Building Hours: Sep 1",
        "rows": [{"label": "Daily", "hours": "Closed"}],
    }
    row_scoped = {
        "title": "Building Hours",
        "rows": [{"label": "Sep 1", "hours": "Open 24 Hours"}],
    }
    sections = [scoped, row_scoped] if scoped_first else [row_scoped, scoped]

    assert get_facility_schedule_open_state(
        sections,
        datetime(2026, 9, 1, 12, 0, tzinfo=CHICAGO),
    ) is expected


def test_schedule_title_date_is_a_hard_scope_for_row_dates() -> None:
    sections = [
        {
            "title": "Building Hours: Sep 1",
            "rows": [{"label": "Sep 2", "hours": "Open 24 Hours"}],
        }
    ]

    assert get_facility_schedule_open_state(
        sections,
        datetime(2026, 9, 2, 12, 0, tzinfo=CHICAGO),
    ) is None


def test_schedule_effective_title_row_intersection_outranks_broader_date_row() -> None:
    sections = [
        {
            "title": "Building Hours",
            "rows": [{"label": "Sep 1 - Sep 3", "hours": "Open 24 Hours"}],
        },
        {
            "title": "Building Hours: Sep 2",
            "rows": [{"label": "Sep 1 - Sep 5", "hours": "Closed"}],
        },
    ]

    assert get_facility_schedule_open_state(
        sections,
        datetime(2026, 9, 2, 12, 0, tzinfo=CHICAGO),
    ) is False


def test_schedule_narrow_title_recurring_row_outranks_broad_unscoped_date_row() -> None:
    sections = [
        {
            "title": "Building Hours",
            "rows": [{"label": "Sep 1 - Sep 5", "hours": "Open 24 Hours"}],
        },
        {
            "title": "Building Hours: Sep 1",
            "rows": [{"label": "Daily", "hours": "Closed"}],
        },
    ]

    assert get_facility_schedule_open_state(
        sections,
        datetime(2026, 9, 1, 12, 0, tzinfo=CHICAGO),
    ) is False


def test_schedule_overnight_spillover_and_explicit_current_date_closure() -> None:
    sections = [
        {
            "title": "Building Hours",
            "rows": [
                {"label": "Mon", "hours": "10:00 pm - 2:00 am"},
                {"label": "Tue", "hours": "8:00 am - 8:00 pm"},
            ],
        }
    ]
    tuesday_one_am = datetime(2026, 9, 1, 1, 0, tzinfo=CHICAGO)
    assert get_facility_schedule_open_state(sections, tuesday_one_am) is True

    sections[0]["rows"].append({"label": "Sep 1", "hours": "Closed"})
    assert get_facility_schedule_open_state(sections, tuesday_one_am) is False


def test_schedule_recurring_closed_does_not_override_overnight_spillover() -> None:
    sections = [
        {
            "title": "Building Hours",
            "rows": [
                {"label": "Mon", "hours": "10:00 pm - 2:00 am"},
                {"label": "Tue", "hours": "Closed"},
            ],
        }
    ]
    tuesday_one_am = datetime(2026, 9, 1, 1, 0, tzinfo=CHICAGO)

    assert get_facility_schedule_open_state(sections, tuesday_one_am) is True

    sections[0]["rows"].append({"label": "Sep 1", "hours": "Closed"})
    assert get_facility_schedule_open_state(sections, tuesday_one_am) is False


def test_schedule_does_not_treat_current_days_future_overnight_as_open() -> None:
    sections = [
        {
            "title": "Building Hours",
            "rows": [{"label": "Tue", "hours": "10:00 pm - 2:00 am"}],
        }
    ]

    assert (
        get_facility_schedule_open_state(
            sections,
            datetime(2026, 9, 1, 1, 0, tzinfo=CHICAGO),
        )
        is False
    )


def test_schedule_intervals_are_half_open_at_exact_close() -> None:
    sections = [
        {
            "title": "Building Hours",
            "rows": [{"label": "Daily", "hours": "6:00 am - 10:00 pm"}],
        }
    ]
    assert get_facility_schedule_open_state(
        sections,
        datetime(2026, 9, 1, 6, 0, tzinfo=CHICAGO),
    ) is True
    assert get_facility_schedule_open_state(
        sections,
        datetime(2026, 9, 1, 22, 0, tzinfo=CHICAGO),
    ) is False


def test_schedule_cross_year_range_matches_january_side() -> None:
    sections = [
        {
            "title": "Building Hours",
            "rows": [
                {"label": "Daily", "hours": "6:00 am - 10:00 pm"},
                {"label": "Dec 31 - Jan 2", "hours": "Open 24 Hours"},
            ],
        }
    ]
    assert get_facility_schedule_open_state(
        sections,
        datetime(2027, 1, 1, 2, 0, tzinfo=CHICAGO),
    ) is True


def test_schedule_malformed_specific_override_is_ambiguous_not_recurring_open() -> None:
    sections = [
        {
            "title": "Building Hours",
            "rows": [
                {"label": "Daily", "hours": "Open 24 Hours"},
                {"label": "Sep 1", "hours": "hours unavailable"},
            ],
        }
    ]
    assert get_facility_schedule_open_state(
        sections,
        datetime(2026, 9, 1, 12, 0, tzinfo=CHICAGO),
    ) is None


def test_schedule_malformed_date_ranged_section_is_ambiguous() -> None:
    sections = [
        {
            "title": "Building Hours: Sep 32 - Sep 35",
            "rows": [{"label": "Daily", "hours": "Open 24 Hours"}],
        },
        {
            "title": "Building Hours",
            "rows": [{"label": "Daily", "hours": "Open 24 Hours"}],
        },
    ]

    assert get_facility_schedule_open_state(
        sections,
        datetime(2026, 9, 1, 12, 0, tzinfo=CHICAGO),
    ) is None


def test_schedule_malformed_date_like_row_is_ambiguous() -> None:
    rows = [
        {"label": "Daily", "hours": "Open 24 Hours"},
        {"label": "Sep 99", "hours": "Closed"},
    ]
    sections = [
        {
            "title": "Building Hours",
            "rows": rows,
        }
    ]

    assert get_facility_schedule_open_state(
        sections,
        datetime(2026, 9, 1, 12, 0, tzinfo=CHICAGO),
    ) is None
    assert official_facility_is_open(
        task4_schedule_payload(rows),
        1186,
        TASK4_NOW,
    ) is False


@pytest.mark.parametrize("ambiguous_first", [False, True])
@pytest.mark.parametrize("ambiguous_label", ["Today", "Sep. 1", "September 1st"])
def test_unparseable_building_label_fails_closed_in_any_row_order(
    ambiguous_first: bool,
    ambiguous_label: str,
) -> None:
    open_row = {"label": "Daily", "hours": "Open 24 Hours"}
    ambiguous_row = {"label": ambiguous_label, "hours": "Closed"}
    rows = (
        [ambiguous_row, open_row]
        if ambiguous_first
        else [open_row, ambiguous_row]
    )
    sections = [{"title": "Building Hours", "rows": rows}]

    assert get_facility_schedule_open_state(
        sections,
        datetime(2026, 9, 1, 12, 0, tzinfo=CHICAGO),
    ) is None
    assert official_facility_is_open(
        task4_schedule_payload(rows),
        1186,
        TASK4_NOW,
    ) is False


def test_date_ranged_building_section_outranks_ordinary_recurring_row() -> None:
    sections = [
        {
            "title": "Building Hours",
            "rows": [{"label": "Daily", "hours": "Closed"}],
        },
        {
            "title": "Building Hours: Sep 1 - Sep 2",
            "rows": [{"label": "Daily", "hours": "Open 24 Hours"}],
        },
    ]
    assert get_facility_schedule_open_state(
        sections,
        datetime(2026, 9, 1, 12, 0, tzinfo=CHICAGO),
    ) is True


def test_maintenance_section_is_veto_only_and_notes_are_not_rows() -> None:
    maintenance = [
        {
            "title": "Maintenance Closures",
            "rows": [{"label": "Daily", "hours": "Open 24 Hours"}],
            "note": "No scheduled maintenance closures",
        }
    ]
    assert get_facility_schedule_open_state(
        maintenance,
        datetime(2026, 9, 1, 12, 0, tzinfo=CHICAGO),
    ) is None


@pytest.mark.parametrize("maintenance_first", [False, True])
def test_maintenance_closure_vetoes_building_open_in_any_source_order(
    maintenance_first: bool,
) -> None:
    building = {
        "title": "Building Hours",
        "rows": [{"label": "Daily", "hours": "Open 24 Hours"}],
    }
    maintenance = {
        "title": "Maintenance Closures",
        "rows": [{"label": "Daily", "hours": "Closed"}],
    }
    sections = (
        [maintenance, building]
        if maintenance_first
        else [building, maintenance]
    )

    assert get_facility_schedule_open_state(
        sections,
        datetime(2026, 9, 1, 12, 0, tzinfo=CHICAGO),
    ) is False


def test_malformed_matching_maintenance_row_fails_closed_despite_building_open() -> None:
    sections = [
        {
            "title": "Building Hours",
            "rows": [{"label": "Sep 1", "hours": "Open 24 Hours"}],
        },
        {
            "title": "Maintenance Closures",
            "rows": [{"label": "Sep 1", "hours": "hours unavailable"}],
        },
    ]

    assert get_facility_schedule_open_state(
        sections,
        datetime(2026, 9, 1, 12, 0, tzinfo=CHICAGO),
    ) is None


@pytest.mark.parametrize("maintenance_first", [False, True])
@pytest.mark.parametrize("malformed_label", ["Today", "Sep. 1", "September 1st"])
def test_unparseable_maintenance_label_fails_closed_in_any_source_order(
    maintenance_first: bool,
    malformed_label: str,
) -> None:
    building = {
        "title": "Building Hours",
        "rows": [{"label": "Daily", "hours": "Open 24 Hours"}],
    }
    maintenance = {
        "title": "Maintenance Closures",
        "rows": [{"label": malformed_label, "hours": "Closed"}],
    }
    sections = (
        [maintenance, building]
        if maintenance_first
        else [building, maintenance]
    )

    assert get_facility_schedule_open_state(
        sections,
        datetime(2026, 9, 1, 12, 0, tzinfo=CHICAGO),
    ) is None


def test_schedule_uses_same_wall_rule_for_both_fall_back_folds() -> None:
    sections = [
        {
            "title": "Building Hours",
            "rows": [{"label": "Sun", "hours": "12:00 am - 3:00 am"}],
        }
    ]
    first_fold = datetime(2026, 11, 1, 1, 30, tzinfo=CHICAGO, fold=0)
    second_fold = datetime(2026, 11, 1, 1, 30, tzinfo=CHICAGO, fold=1)
    assert get_facility_schedule_open_state(sections, first_fold) is True
    assert get_facility_schedule_open_state(sections, second_fold) is True


@pytest.mark.parametrize(
    "section_title",
    [
        "Pool Hours",
        "Basketball Courts",
        "Running Track",
        "Fitness Center",
        "Basketball Hours of Operation",
        "Swimming Facility Hours",
        "Ice Skating Facility Schedule",
        "Building Hours - Pool",
        "Building Hours: Basketball Court",
        "Facility Hours (Swimming)",
        "Building Hours - Sep 1 Pool",
    ],
)
def test_area_specific_schedule_rows_cannot_open_whole_facility(
    section_title: str,
) -> None:
    sections = [
        {
            "title": section_title,
            "rows": [{"label": "Daily", "hours": "Open 24 Hours"}],
        }
    ]
    assert (
        get_facility_schedule_open_state(
            sections,
            datetime(2026, 9, 1, 12, 0, tzinfo=CHICAGO),
        )
        is None
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "stale",
        "facility_error",
        "old_generated",
        "future_generated",
        "old_success",
        "future_success",
        "naive_generated",
        "missing_success",
    ],
)
def test_official_schedule_predicate_fails_closed_for_untrusted_freshness(
    mutation: str,
) -> None:
    payload = task4_schedule_payload(
        [{"label": "Daily", "hours": "Open 24 Hours"}]
    )
    facility = payload["facilities"][0]  # type: ignore[index]
    assert isinstance(facility, dict)
    if mutation == "stale":
        facility["stale"] = True
    elif mutation == "facility_error":
        facility["status"] = "error"
    elif mutation == "old_generated":
        payload["generatedAt"] = "2026-09-01T10:59:59Z"
    elif mutation == "future_generated":
        payload["generatedAt"] = "2026-09-01T17:00:01Z"
    elif mutation == "old_success":
        facility["lastSuccessfulAt"] = "2026-09-01T10:59:59Z"
    elif mutation == "future_success":
        facility["lastSuccessfulAt"] = "2026-09-01T17:00:01Z"
    elif mutation == "naive_generated":
        payload["generatedAt"] = "2026-09-01T16:59:00"
    else:
        facility.pop("lastSuccessfulAt")

    assert official_facility_is_open(payload, 1186, TASK4_NOW) is False


def test_official_schedule_accepts_exact_freshness_boundary_and_dst_instants() -> None:
    payload = task4_schedule_payload(
        [{"label": "Daily", "hours": "Open 24 Hours"}],
        generated_at="2026-09-01T11:00:00Z",
        successful_at="2026-09-01T11:00:00Z",
    )
    assert official_facility_is_open(payload, 1186, TASK4_NOW) is True

    spring_payload = task4_schedule_payload(
        [{"label": "Sun", "hours": "12:00 am - 4:00 am"}],
        generated_at="2026-03-08T07:59:00Z",
        successful_at="2026-03-08T07:59:00Z",
    )
    assert official_facility_is_open(
        spring_payload,
        1186,
        datetime(2026, 3, 8, 8, 0, tzinfo=timezone.utc),
    ) is True


def test_official_schedule_rejects_naive_clock_unsupported_facility_and_bad_limit() -> None:
    payload = task4_schedule_payload(
        [{"label": "Daily", "hours": "Open 24 Hours"}]
    )
    assert official_facility_is_open(payload, 1186, datetime(2026, 9, 1, 12)) is False
    assert official_facility_is_open(payload, 9999, TASK4_NOW) is False
    assert official_facility_is_open(
        payload,
        1186,
        TASK4_NOW,
        stale_after_seconds=0,
    ) is False


def test_official_schedule_rejects_malformed_collection_members() -> None:
    payload = task4_schedule_payload(
        [{"label": "Daily", "hours": "Open 24 Hours"}]
    )
    facilities = payload["facilities"]
    assert isinstance(facilities, list)
    facilities.append("malformed")
    assert official_facility_is_open(payload, 1186, TASK4_NOW) is False

    payload = task4_schedule_payload(
        [{"label": "Daily", "hours": "Open 24 Hours"}]
    )
    facility = payload["facilities"][0]  # type: ignore[index]
    assert isinstance(facility, dict)
    sections = facility["sections"]
    assert isinstance(sections, list)
    sections.append({"title": "Building Hours", "rows": ["malformed"]})
    assert official_facility_is_open(payload, 1186, TASK4_NOW) is False


@pytest.mark.parametrize(
    "answers",
    [
        [],
        ["10.0.0.8"],
        ["8.8.8.8", "127.0.0.1"],
        ["2606:4700:4700::1111", "fc00::8"],
        ["169.254.10.1"],
        ["224.0.0.1"],
        ["0.0.0.0"],
        ["240.0.0.1"],
        ["fe80::1"],
        ["fe80::1%en0"],
        ["::ffff:8.8.8.8"],
        ["2002:0808:0808::"],
        ["2001:0000:4136:e378:8000:63bf:3fff:fdd2"],
        ["not-an-ip"],
        [123],
    ],
)
def test_push_dns_rejects_empty_mixed_and_unsafe_answer_sets(
    monkeypatch: pytest.MonkeyPatch,
    answers: list[object],
) -> None:
    calls: list[tuple[str, int]] = []

    def resolve(host: str, port: int) -> list[object]:
        calls.append((host, port))
        return answers

    monkeypatch.setattr(api, "resolve_endpoint_host", resolve, raising=False)
    with pytest.raises(api.SafePushDispatchError):
        api.resolve_public_push_addresses(
            "https://push.reclive-notify.net/subscription-a"
        )
    assert calls == [("push.reclive-notify.net", 443)]


def test_push_dns_deduplicates_and_sorts_ipv6_before_ipv4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api,
        "resolve_endpoint_host",
        lambda host, port: [
            "8.8.8.8",
            "2606:4700:4700::1111",
            "8.8.8.8",
        ],
        raising=False,
    )

    assert api.resolve_public_push_addresses(
        "https://push.reclive-notify.net/subscription-a"
    ) == (
        ipaddress.ip_address("2606:4700:4700::1111"),
        ipaddress.ip_address("8.8.8.8"),
    )


def test_pinned_target_preserves_hostname_authority_port_path_and_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api,
        "resolve_endpoint_host",
        lambda host, port: ["8.8.8.8"],
        raising=False,
    )
    target = api.build_pinned_push_target(
        "https://push.reclive-notify.net:8443/a/b?token=one%20two"
    )

    assert target.connect_ip == "8.8.8.8"
    assert target.tls_server_hostname == "push.reclive-notify.net"
    assert target.host_header == "push.reclive-notify.net:8443"
    assert target.port == 8443
    assert target.request_target == "/a/b?token=one%20two"


def test_pinned_target_formats_ipv6_literal_host_without_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api,
        "resolve_endpoint_host",
        lambda *_: pytest.fail("literal address must not be resolved"),
        raising=False,
    )
    target = api.build_pinned_push_target(
        "https://[2606:4700:4700::1111]:8443/a"
    )
    assert target.connect_ip == "2606:4700:4700::1111"
    assert target.tls_server_hostname == "2606:4700:4700::1111"
    assert target.host_header == "[2606:4700:4700::1111]:8443"


def test_send_notification_uses_public_webpush_session_seam_once(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        api,
        "build_pinned_push_target",
        lambda endpoint, **kwargs: api.PinnedPushTarget(
            endpoint=endpoint,
            connect_ip="8.8.8.8",
            tls_server_hostname="push.reclive-notify.net",
            host_header="push.reclive-notify.net",
            port=443,
            request_target="/subscription-a",
        ),
        raising=False,
    )
    monkeypatch.setattr(api, "get_vapid_private_key", lambda: "private")
    monkeypatch.setattr(
        api,
        "get_vapid_claims",
        lambda: {"sub": "mailto:test@reclive.app"},
    )

    def fake_webpush(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        session = kwargs["requests_session"]
        assert isinstance(session, api.PinnedPushSession)
        return SimpleNamespace(status_code=201, reason="", text="")

    monkeypatch.setattr(api, "webpush", fake_webpush)
    api.send_notification_pinned(
        valid_subscription,
        title="RecLive Alert",
        body="Nick is 20% full.",
        url="/nick",
        sent_at="2026-09-01T17:00:00+00:00",
    )

    assert len(calls) == 1
    assert calls[0]["timeout"] == 10
    assert calls[0]["ttl"] == 120
    assert isinstance(calls[0]["requests_session"], api.PinnedPushSession)


def test_real_pywebpush_prepares_encrypted_payload_through_public_session_only(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
) -> None:
    target = api.PinnedPushTarget(
        endpoint="https://push.reclive-notify.net/subscription-a",
        connect_ip="8.8.8.8",
        tls_server_hostname="push.reclive-notify.net",
        host_header="push.reclive-notify.net",
        port=443,
        request_target="/subscription-a",
    )
    calls: list[tuple[str, dict[str, object]]] = []

    class RecordingSession:
        def post(self, url: str, **kwargs: object) -> api.SafePushResponse:
            calls.append((url, kwargs))
            return api.SafePushResponse(status_code=201)

    monkeypatch.setattr(
        api,
        "build_pinned_push_target",
        lambda endpoint, **kwargs: target,
    )
    monkeypatch.setattr(
        api,
        "PinnedPushSession",
        lambda candidate, attempt=None: RecordingSession(),
    )
    monkeypatch.setattr(api, "get_vapid_private_key", lambda: "")
    monkeypatch.setattr(api, "get_vapid_claims", lambda: {})

    api.send_notification_pinned(
        valid_subscription,
        title="RecLive Alert",
        body="Provider plaintext sentinel",
        url="/nick",
        sent_at="2026-09-01T17:00:00+00:00",
    )

    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url == target.endpoint
    assert kwargs["timeout"] == 10
    encrypted = kwargs["data"]
    assert isinstance(encrypted, bytes)
    assert b"Provider plaintext sentinel" not in encrypted
    headers = kwargs["headers"]
    assert isinstance(headers, Mapping)
    assert headers["Content-Encoding"] == "aes128gcm"


class _FakeRawSocket:
    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self.events.append("raw-close")


class _FakeTLSSocket:
    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.sent = b""
        self.closed = False

    def sendall(self, data: bytes) -> None:
        self.sent += data
        self.events.append(("send", data))

    def close(self) -> None:
        self.closed = True
        self.events.append("tls-close")


def test_pinned_session_uses_numeric_socket_sni_host_and_exact_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    raw = _FakeRawSocket(events)
    tls = _FakeTLSSocket(events)

    class FakeContext:
        check_hostname = True
        verify_mode = ssl.CERT_REQUIRED

        def wrap_socket(
            self,
            candidate: object,
            *,
            server_hostname: str,
        ) -> _FakeTLSSocket:
            assert candidate is raw
            events.append(("sni", server_hostname))
            return tls

    class FakeHTTPResponse:
        status = 302

        def __init__(self, candidate: object) -> None:
            assert candidate is tls

        def begin(self) -> None:
            events.append("response-begin")

        def read(self, amount: int) -> bytes:
            events.append(("read", amount))
            return b"provider-secret-sentinel"

        def close(self) -> None:
            events.append("response-close")

    monkeypatch.setattr(
        api.socket,
        "create_connection",
        lambda address, timeout: (
            events.append(("connect", address, timeout)) or raw
        ),
    )
    monkeypatch.setattr(api.ssl, "create_default_context", lambda: FakeContext())
    monkeypatch.setattr(api.http.client, "HTTPResponse", FakeHTTPResponse)
    target = api.PinnedPushTarget(
        endpoint="https://push.reclive-notify.net:8443/a?b=1",
        connect_ip="8.8.8.8",
        tls_server_hostname="push.reclive-notify.net",
        host_header="push.reclive-notify.net:8443",
        port=8443,
        request_target="/a?b=1",
    )

    response = api.PinnedPushSession(target).post(
        target.endpoint,
        timeout=10,
        data=b"encrypted",
        headers={"Content-Encoding": "aes128gcm", "TTL": "120"},
    )

    assert response.status_code == 302
    assert response.reason == ""
    assert response.text == ""
    connect_events = [event for event in events if isinstance(event, tuple) and event[0] == "connect"]
    assert len(connect_events) == 1
    assert connect_events[0][1] == ("8.8.8.8", 8443)
    assert 0 < connect_events[0][2] <= 10
    assert ("sni", "push.reclive-notify.net") in events
    assert ("read", api.PUSH_PROVIDER_RESPONSE_MAX_BYTES + 1) in events
    request = tls.sent.split(b"\r\n\r\n", 1)[0]
    assert request.startswith(b"POST /a?b=1 HTTP/1.1\r\n")
    assert b"Host: push.reclive-notify.net:8443\r\n" in request
    assert b"Connection: close\r\n" in request
    assert b"Location:" not in request
    assert raw.closed is True
    assert tls.closed is True
    assert "response-close" in events


def test_pinned_session_rejects_endpoint_mismatch_and_header_injection_before_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = api.PinnedPushTarget(
        endpoint="https://push.reclive-notify.net/a",
        connect_ip="8.8.8.8",
        tls_server_hostname="push.reclive-notify.net",
        host_header="push.reclive-notify.net",
        port=443,
        request_target="/a",
    )
    socket_calls: list[int] = []

    def reject_socket(*args: object, **kwargs: object) -> None:
        socket_calls.append(1)
        raise RuntimeError("invalid request opened socket")

    monkeypatch.setattr(api.socket, "create_connection", reject_socket)
    session = api.PinnedPushSession(target)
    with pytest.raises(api.SafePushDispatchError):
        session.post("https://push.reclive-notify.net/b", data=b"x", headers={})
    with pytest.raises(api.SafePushDispatchError):
        session.post(
            target.endpoint,
            data=b"x",
            headers={"X-Bad": "one\r\ntwo"},
        )
    with pytest.raises(api.SafePushDispatchError):
        session.post(
            target.endpoint,
            data=b"x",
            headers={"Bad Header Name": "value"},
        )
    assert socket_calls == []


def test_evaluator_deadline_releases_lock_and_never_sends_after_slow_dns(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
) -> None:
    original_sender = api.send_notification_pinned
    events, _connection, reader = configure_task4_evaluator(
        monkeypatch,
        valid_subscription,
        sender=original_sender,
    )
    state = {"status": "pending"}
    rule = task4_rule(valid_subscription)
    monkeypatch.setattr(
        api,
        "load_evaluator_candidates",
        lambda connection, now: [rule] if state["status"] == "pending" else [],
    )

    def claim(connection: object, rule_id: int, now: datetime) -> bool:
        events.append("claim")
        if state["status"] != "pending":
            return False
        state["status"] = "claimed"
        return True

    def finalize(
        connection: object,
        rule_id: int,
        now: datetime,
        status: str,
        failure_code: str | None = None,
    ) -> bool:
        events.append(f"terminal:{status}:{failure_code or '-'}")
        if state["status"] != "claimed":
            return False
        state["status"] = status
        return True

    monkeypatch.setattr(api, "claim_pending_rule", claim)
    monkeypatch.setattr(api, "finalize_claimed_rule", finalize)
    monkeypatch.setattr(
        api,
        "PUSH_TRANSPORT_DEADLINE_SECONDS",
        0.04,
        raising=False,
    )
    resolver_finished = threading.Event()

    def delayed_resolver(_host: str, _port: int) -> list[str]:
        time.sleep(0.2)
        resolver_finished.set()
        return ["8.8.8.8"]

    monkeypatch.setattr(api, "resolve_endpoint_host", delayed_resolver)
    webpush_calls: list[int] = []
    monkeypatch.setattr(
        api,
        "webpush",
        lambda **kwargs: webpush_calls.append(1)
        or api.SafePushResponse(status_code=201),
    )
    monkeypatch.setattr(api, "get_vapid_private_key", lambda: "")
    monkeypatch.setattr(api, "get_vapid_claims", lambda: {})

    started = time.monotonic()
    first = api.evaluate_rules_once(now=TASK4_NOW, snapshot_reader=reader)
    elapsed = time.monotonic() - started
    second = api.evaluate_rules_once(now=TASK4_NOW, snapshot_reader=reader)

    assert elapsed < 0.15
    assert first["failed"] == 1
    assert second["rules"] == 0
    assert state["status"] == "failed"
    assert events.count("claim") == 1
    assert events.count("release") == 2
    assert resolver_finished.wait(0.5)
    assert webpush_calls == []


def test_repeated_stuck_dns_attempts_use_only_bounded_daemon_workers(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
) -> None:
    monkeypatch.setattr(
        api,
        "PUSH_TRANSPORT_DEADLINE_SECONDS",
        0.02,
        raising=False,
    )
    release_resolver = threading.Event()
    resolver_calls: list[int] = []
    webpush_calls: list[int] = []

    def stuck_resolver(_host: str, _port: int) -> list[str]:
        resolver_calls.append(1)
        release_resolver.wait()
        return ["8.8.8.8"]

    monkeypatch.setattr(api, "resolve_endpoint_host", stuck_resolver)
    monkeypatch.setattr(
        api,
        "webpush",
        lambda **kwargs: webpush_calls.append(1)
        or api.SafePushResponse(status_code=201),
    )
    monkeypatch.setattr(api, "get_vapid_private_key", lambda: "")
    monkeypatch.setattr(api, "get_vapid_claims", lambda: {})
    baseline_threads = set(threading.enumerate())
    executor = api._BoundedPushExecutor(worker_count=2, queue_capacity=2)
    monkeypatch.setattr(api, "_push_executor", executor)

    started = time.monotonic()
    try:
        for _ in range(8):
            with pytest.raises(api.SafePushDispatchError):
                api.send_notification_pinned(
                    valid_subscription,
                    title="RecLive Alert",
                    body="Ready",
                    url="/nick",
                )
        push_workers = [
            thread
            for thread in threading.enumerate()
            if thread not in baseline_threads
            and thread.name.startswith("reclive-push")
        ]
        assert len(push_workers) <= 2
        assert all(thread.daemon for thread in push_workers)
        assert len(resolver_calls) <= 2
        assert time.monotonic() - started < 0.5
    finally:
        release_resolver.set()

    queue_drained = threading.Event()

    def wait_for_queue() -> None:
        executor._tasks.join()
        queue_drained.set()

    threading.Thread(target=wait_for_queue, daemon=True).start()
    assert queue_drained.wait(0.5)
    assert webpush_calls == []


def test_send_deadline_aborts_and_closes_slow_provider_response(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
) -> None:
    monkeypatch.setattr(
        api,
        "PUSH_TRANSPORT_DEADLINE_SECONDS",
        0.04,
        raising=False,
    )
    target = api.PinnedPushTarget(
        endpoint="https://push.reclive-notify.net/subscription-a",
        connect_ip="8.8.8.8",
        tls_server_hostname="push.reclive-notify.net",
        host_header="push.reclive-notify.net",
        port=443,
        request_target="/subscription-a",
    )
    monkeypatch.setattr(
        api,
        "build_pinned_push_target",
        lambda endpoint, **kwargs: target,
    )
    events: list[str] = []
    raw = _FakeRawSocket(events)
    tls = _FakeTLSSocket(events)
    response_instances: list[object] = []
    response_read_finished = threading.Event()

    class FakeContext:
        check_hostname = True
        verify_mode = ssl.CERT_REQUIRED

        def wrap_socket(
            self,
            candidate: object,
            *,
            server_hostname: str,
        ) -> _FakeTLSSocket:
            assert candidate is raw
            assert server_hostname == target.tls_server_hostname
            return tls

    class SlowHTTPResponse:
        status = 201

        def __init__(self, candidate: object) -> None:
            assert candidate is tls
            self.closed = False
            response_instances.append(self)

        def begin(self) -> None:
            return None

        def read(self, _amount: int) -> bytes:
            time.sleep(0.2)
            response_read_finished.set()
            return b""

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(api.socket, "create_connection", lambda *_args, **_kwargs: raw)
    monkeypatch.setattr(api.ssl, "create_default_context", lambda: FakeContext())
    monkeypatch.setattr(api.http.client, "HTTPResponse", SlowHTTPResponse)
    monkeypatch.setattr(api, "get_vapid_private_key", lambda: "")
    monkeypatch.setattr(api, "get_vapid_claims", lambda: {})

    started = time.monotonic()
    with pytest.raises(api.SafePushDispatchError):
        api.send_notification_pinned(
            valid_subscription,
            title="RecLive Alert",
            body="Ready",
            url="/nick",
        )
    elapsed = time.monotonic() - started

    assert elapsed < 0.15
    assert raw.closed is True
    assert tls.closed is True
    assert response_instances
    assert all(response.closed for response in response_instances)
    assert response_read_finished.wait(0.5)


def test_facility_notification_routes_are_exact_and_unsupported_fails_closed() -> None:
    assert api.facility_notification_url(1186) == "/nick"
    assert api.facility_notification_url(1656) == "/bakke"
    with pytest.raises(ValueError):
        api.facility_notification_url(9999)


def task4_rule(
    valid_subscription: dict[str, object],
    *,
    rule_id: int = 41,
    facility_id: int = 1186,
    section_key: str = "overall",
    threshold: int = 40,
    expires_at: datetime = datetime(2026, 9, 2, 17, 0, tzinfo=timezone.utc),
) -> api.PushRuleRecord:
    return api.PushRuleRecord(
        id=rule_id,
        endpoint_hash=b"h" * 32,
        subscription_json=json.dumps(valid_subscription),
        facility_id=facility_id,
        section_key=section_key,
        threshold=threshold,
        created_at=datetime(2026, 9, 1, 16, 0, tzinfo=timezone.utc),
        expires_at=expires_at,
        status="pending",
        active_identity=1,
    )


class Task4EvaluatorConnection:
    def __init__(
        self,
        events: list[str],
        *,
        fail_terminal_commit: bool = False,
    ) -> None:
        self.events = events
        self.commit_count = 0
        self.fail_terminal_commit = fail_terminal_commit

    def commit(self) -> None:
        self.commit_count += 1
        self.events.append(f"commit:{self.commit_count}")
        if self.fail_terminal_commit and self.commit_count == 3:
            raise RuntimeError("terminal-database-secret")

    def rollback(self) -> None:
        self.events.append("rollback")


class Task4SnapshotReader:
    def __init__(
        self,
        events: list[str],
        *,
        ingestion_at: datetime | None = datetime(
            2026, 9, 1, 16, 59, tzinfo=timezone.utc
        ),
    ) -> None:
        self.events = events
        self.ingestion_at = ingestion_at
        self.fetch_times: list[datetime] = []

    def fetch_live_snapshot(self, now: datetime) -> SimpleNamespace:
        assert now.tzinfo is not None and now.utcoffset() == timedelta(0)
        self.fetch_times.append(now)
        self.events.append("snapshot")
        return SimpleNamespace(
            last_successful_fetch_at=self.ingestion_at,
            rows=[],
        )


def configure_task4_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
    *,
    connection: Task4EvaluatorConnection | None = None,
    ingestion_at: datetime | None = datetime(
        2026, 9, 1, 16, 59, tzinfo=timezone.utc
    ),
    schedule_open: bool = True,
    metrics: dict[str, object] | None = None,
    claim: bool = True,
    sender: object | None = None,
) -> tuple[list[str], Task4EvaluatorConnection, Task4SnapshotReader]:
    events: list[str] = connection.events if connection is not None else []
    conn = connection or Task4EvaluatorConnection(events)
    reader = Task4SnapshotReader(events, ingestion_at=ingestion_at)
    rule = task4_rule(valid_subscription)

    monkeypatch.setattr(
        api,
        "db_acquire_evaluator_lock",
        lambda: events.append("lock") or conn,
    )
    monkeypatch.setattr(
        api,
        "load_evaluator_candidates",
        lambda candidate_conn, now: (
            events.append("rules") or [rule]
        ),
        raising=False,
    )
    monkeypatch.setattr(
        api,
        "load_facility_hours",
        lambda: events.append("schedule")
        or task4_schedule_payload(
            [{"label": "Daily", "hours": "Open 24 Hours"}]
        ),
    )
    monkeypatch.setattr(
        api,
        "official_facility_is_open",
        lambda payload, facility_id, at, *, stale_after_seconds: (
            schedule_open
            and stale_after_seconds == api.SCHEDULE_STALE_AFTER_SECONDS
        ),
        raising=False,
    )
    monkeypatch.setattr(
        api,
        "compute_fresh_section_metrics",
        lambda facility_id, section_key, snapshots, now: metrics
        if metrics is not None
        else {
            "status": "live",
            "coverage": 0.8,
            "percent": 20.0,
        },
        raising=False,
    )
    monkeypatch.setattr(
        api,
        "claim_pending_rule",
        lambda candidate_conn, rule_id, now: events.append("claim") or claim,
        raising=False,
    )
    monkeypatch.setattr(
        api,
        "finalize_claimed_rule",
        lambda candidate_conn, rule_id, now, status, failure_code=None: events.append(
            f"terminal:{status}:{failure_code or '-'}"
        )
        or True,
        raising=False,
    )
    monkeypatch.setattr(
        api,
        "db_release_evaluator_lock",
        lambda candidate_conn: events.append("release"),
    )
    if sender is None:
        monkeypatch.setattr(
            api,
            "send_notification_pinned",
            lambda *args, **kwargs: events.append("send"),
            raising=False,
        )
    else:
        monkeypatch.setattr(api, "send_notification_pinned", sender, raising=False)
    return events, conn, reader


def test_evaluator_event_order_commits_claim_before_dns_or_send_and_terminalizes(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
) -> None:
    events, _conn, reader = configure_task4_evaluator(
        monkeypatch,
        valid_subscription,
    )

    result = api.evaluate_rules_once(now=TASK4_NOW, snapshot_reader=reader)

    assert result["sent"] == 1
    assert events == [
        "lock",
        "rules",
        "schedule",
        "commit:1",
        "snapshot",
        "claim",
        "commit:2",
        "send",
        "terminal:sent:-",
        "commit:3",
        "release",
    ]


@pytest.mark.parametrize(
    ("facility_id", "location_id", "expected_url"),
    [(1186, 5763, "/nick"), (1656, 8694, "/bakke")],
)
@pytest.mark.mysql
def test_real_mysql_subscribe_to_evaluator_reaches_only_fixed_facility_route(
    monkeypatch: pytest.MonkeyPatch,
    mysql_push_test_client: Any,
    migrated_push_database: dict[str, object],
    valid_subscription: dict[str, object],
    facility_id: int,
    location_id: int,
    expected_url: str,
) -> None:
    for rejected_threshold in (101, True, 40.5):
        rejected = mysql_push_test_client.post(
            "/api/push/subscribe",
            json=subscribe_payload(
                valid_subscription,
                facilityId=facility_id,
                sectionKey="running track",
                threshold=rejected_threshold,
            ),
        )
        assert rejected.status_code == 422
        assert rejected.json() == {"detail": "invalid_push_request"}

    caller_override = mysql_push_test_client.post(
        "/api/push/subscribe",
        json=subscribe_payload(
            valid_subscription,
            facilityId=facility_id,
            sectionKey="running track",
            threshold=100,
            url="/caller-controlled",
        ),
    )
    assert caller_override.status_code == 422
    assert caller_override.json() == {"detail": "invalid_push_request"}

    accepted = mysql_push_test_client.post(
        "/api/push/subscribe",
        json=subscribe_payload(
            valid_subscription,
            facilityId=facility_id,
            sectionKey="running track",
            threshold=100,
        ),
    )
    assert accepted.status_code == 200
    accepted_payload = accepted.json()
    assert accepted_payload["status"] == "ok"
    assert accepted_payload["created"] is True
    assert accepted_payload["rule"]["facilityId"] == facility_id
    assert accepted_payload["rule"]["sectionKey"] == "running track"
    assert accepted_payload["rule"]["threshold"] == 100
    rule_id = accepted_payload["rule"]["id"]
    assert mysql_fetch_all(
        migrated_push_database,
        "SELECT id, facility_id, section_key, threshold, status "
        "FROM push_rules WHERE id = %s",
        (rule_id,),
    ) == ((rule_id, facility_id, "running track", 100, "pending"),)

    mysql_execute(migrated_push_database, "DELETE FROM location_snapshot")
    mysql_execute(migrated_push_database, "DELETE FROM ingestion_runs")
    mysql_execute(
        migrated_push_database,
        "INSERT INTO ingestion_runs "
        "(started_at, completed_at, status, observed_location_ids) "
        "VALUES (%s, %s, 'succeeded', JSON_ARRAY(%s))",
        (
            datetime(2026, 9, 1, 16, 58),
            datetime(2026, 9, 1, 16, 59),
            location_id,
        ),
    )
    mysql_execute(
        migrated_push_database,
        "INSERT INTO location_snapshot "
        "(location_id, is_closed, current_capacity, max_capacity, fetched_at) "
        "VALUES (%s, 0, 20, %s, %s)",
        (
            location_id,
            api.MAX_CAP[location_id],
            datetime(2026, 9, 1, 16, 59),
        ),
    )
    monkeypatch.setattr(
        api,
        "load_facility_hours",
        lambda: {
            "generatedAt": "2026-09-01T16:59:00Z",
            "facilities": [
                {
                    "facilityId": facility_id,
                    "status": "ok",
                    "stale": False,
                    "lastSuccessfulAt": "2026-09-01T16:59:00Z",
                    "sections": [
                        {
                            "title": "Building Hours",
                            "rows": [{"label": "Daily", "hours": "Open 24 Hours"}],
                        }
                    ],
                }
            ],
        },
    )
    sent_urls: list[object] = []

    def capture_pinned_send(
        _subscription: Mapping[str, object],
        **kwargs: object,
    ) -> None:
        sent_urls.append(kwargs.get("url"))

    monkeypatch.setattr(api, "send_notification_pinned", capture_pinned_send)

    result = api.evaluate_rules_once(now=TASK4_NOW)

    assert result == {
        "status": "ok",
        "rules": 1,
        "sent": 1,
        "failed": 0,
        "skippedThreshold": 0,
        "skippedCooldown": 0,
        "skippedMissingSection": 0,
        "skippedInactive": 0,
        "skippedLocked": 0,
        "evaluatedAt": "2026-09-01T17:00:00+00:00",
    }
    assert sent_urls == [expected_url]
    assert mysql_fetch_all(
        migrated_push_database,
        "SELECT facility_id, section_key, threshold, status, sent_at IS NOT NULL, "
        "finalized_at IS NOT NULL, failure_code FROM push_rules WHERE id = %s",
        (rule_id,),
    ) == ((facility_id, "running track", 100, "sent", 1, 1, None),)


def test_evaluator_commits_before_each_candidate_snapshot_without_releasing_lock(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
) -> None:
    events: list[str] = []

    class BoundaryConnection:
        def commit(self) -> None:
            events.append("commit")

        def rollback(self) -> None:
            events.append("rollback")

    class VersionedReader:
        calls = 0

        def fetch_live_snapshot(self, at: datetime) -> SimpleNamespace:
            self.calls += 1
            events.append(f"snapshot:{self.calls}")
            return SimpleNamespace(
                last_successful_fetch_at=TASK4_NOW - timedelta(seconds=60),
                rows=[{"version": self.calls}],
            )

    connection = BoundaryConnection()
    rules = [
        task4_rule(valid_subscription, rule_id=41),
        task4_rule(valid_subscription, rule_id=42),
    ]
    reader = VersionedReader()
    monkeypatch.setattr(
        api,
        "db_acquire_evaluator_lock",
        lambda: events.append("lock") or connection,
    )
    monkeypatch.setattr(
        api,
        "load_evaluator_candidates",
        lambda candidate_conn, at: events.append("rules") or rules,
    )
    monkeypatch.setattr(api, "load_facility_hours", lambda: events.append("schedule") or {})
    monkeypatch.setattr(
        api,
        "official_facility_is_open",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(api, "index_live_rows", lambda rows: rows[0])
    monkeypatch.setattr(
        api,
        "compute_fresh_section_metrics",
        lambda facility_id, section_key, snapshots, at: {
            "status": "live",
            "coverage": 0.8,
            "percent": 90.0 if snapshots["version"] == 1 else 20.0,
        },
    )
    monkeypatch.setattr(
        api,
        "claim_pending_rule",
        lambda candidate_conn, rule_id, at: events.append(f"claim:{rule_id}") or True,
    )
    monkeypatch.setattr(
        api,
        "send_notification_pinned",
        lambda *args, **kwargs: events.append("send"),
    )
    monkeypatch.setattr(
        api,
        "finalize_claimed_rule",
        lambda *args, **kwargs: events.append("terminal") or True,
    )
    monkeypatch.setattr(
        api,
        "db_release_evaluator_lock",
        lambda candidate_conn: events.append("release"),
    )

    result = api.evaluate_rules_once(now=TASK4_NOW, snapshot_reader=reader)

    assert result["sent"] == 1
    assert events == [
        "lock",
        "rules",
        "schedule",
        "commit",
        "snapshot:1",
        "commit",
        "snapshot:2",
        "claim:42",
        "commit",
        "send",
        "terminal",
        "commit",
        "release",
    ]


@pytest.mark.parametrize(
    "boundary",
    ["expiry", "ingestion", "schedule", "snapshot", "threshold"],
)
def test_evaluator_resamples_all_claim_gates_after_first_send(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
    boundary: str,
) -> None:
    advanced_now = TASK4_NOW + timedelta(seconds=2)

    class MutableClock:
        value = TASK4_NOW

        def __call__(self) -> datetime:
            return self.value

    clock = MutableClock()
    events: list[str] = []
    connection = Task4EvaluatorConnection(events)
    ingestion_at = (
        TASK4_NOW - timedelta(seconds=599)
        if boundary == "ingestion"
        else TASK4_NOW - timedelta(seconds=60)
    )
    reader = Task4SnapshotReader(events, ingestion_at=ingestion_at)
    first = task4_rule(valid_subscription, rule_id=41)
    second = task4_rule(
        valid_subscription,
        rule_id=42,
        threshold=41,
        expires_at=(
            TASK4_NOW + timedelta(seconds=1)
            if boundary == "expiry"
            else TASK4_NOW + timedelta(days=1)
        ),
    )
    claim_times: list[tuple[int, datetime]] = []
    terminal_times: list[tuple[int, datetime]] = []
    schedule_times: list[datetime] = []
    metric_times: list[datetime] = []
    send_calls: list[int] = []

    monkeypatch.setattr(api, "now_utc", clock)
    monkeypatch.setattr(api, "db_acquire_evaluator_lock", lambda: connection)
    monkeypatch.setattr(
        api,
        "load_evaluator_candidates",
        lambda candidate_conn, at: [first, second],
    )
    monkeypatch.setattr(
        api,
        "load_facility_hours",
        lambda: task4_schedule_payload(
            [{"label": "Daily", "hours": "Open 24 Hours"}]
        ),
    )

    def schedule_open(
        payload: object,
        facility_id: int,
        at: datetime,
        *,
        stale_after_seconds: int,
    ) -> bool:
        assert stale_after_seconds == api.SCHEDULE_STALE_AFTER_SECONDS
        schedule_times.append(at)
        return boundary != "schedule" or at < advanced_now

    def metrics(
        facility_id: int,
        section_key: str,
        snapshots: object,
        at: datetime,
    ) -> dict[str, object] | None:
        metric_times.append(at)
        if boundary == "snapshot" and at >= advanced_now:
            return None
        return {
            "status": "live",
            "coverage": 0.8,
            "percent": 50.0 if boundary == "threshold" and at >= advanced_now else 20.0,
        }

    def claim(
        candidate_conn: object,
        rule_id: int,
        at: datetime,
    ) -> bool:
        claim_times.append((rule_id, at))
        return True

    def finalize(
        candidate_conn: object,
        rule_id: int,
        at: datetime,
        status: str,
        failure_code: str | None = None,
    ) -> bool:
        terminal_times.append((rule_id, at))
        return True

    def send(*args: object, **kwargs: object) -> None:
        send_calls.append(1)
        clock.value = advanced_now

    monkeypatch.setattr(api, "official_facility_is_open", schedule_open)
    monkeypatch.setattr(api, "compute_fresh_section_metrics", metrics)
    monkeypatch.setattr(api, "claim_pending_rule", claim)
    monkeypatch.setattr(api, "finalize_claimed_rule", finalize)
    monkeypatch.setattr(api, "send_notification_pinned", send)
    monkeypatch.setattr(
        api,
        "db_release_evaluator_lock",
        lambda candidate_conn: events.append("release"),
    )

    result = api.evaluate_rules_once(snapshot_reader=reader)

    assert result["sent"] == 1
    assert len(send_calls) == 1
    assert claim_times == [(41, TASK4_NOW)]
    assert terminal_times == [(41, advanced_now)]
    if boundary != "expiry":
        assert schedule_times[-1] == advanced_now
    if boundary not in {"expiry", "ingestion", "schedule"}:
        assert metric_times[-1] == advanced_now
    assert events[-1] == "release"


@pytest.mark.parametrize(
    "boundary",
    ["expiry", "ingestion", "schedule", "snapshot", "threshold"],
)
def test_evaluator_resamples_all_claim_gates_after_snapshot_read(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
    boundary: str,
) -> None:
    advanced_now = TASK4_NOW + timedelta(seconds=2)

    class MutableClock:
        value = TASK4_NOW

        def __call__(self) -> datetime:
            return self.value

    clock = MutableClock()
    events: list[str] = []
    connection = Task4EvaluatorConnection(events)
    rule = task4_rule(
        valid_subscription,
        expires_at=(
            TASK4_NOW + timedelta(seconds=1)
            if boundary == "expiry"
            else TASK4_NOW + timedelta(days=1)
        ),
    )
    schedule_times: list[datetime] = []
    metric_times: list[datetime] = []
    claim_times: list[datetime] = []
    send_calls: list[int] = []

    class AdvancingSnapshotReader:
        def __init__(self) -> None:
            self.fetch_times: list[datetime] = []

        def fetch_live_snapshot(self, at: datetime) -> SimpleNamespace:
            self.fetch_times.append(at)
            clock.value = advanced_now
            return SimpleNamespace(
                last_successful_fetch_at=(
                    TASK4_NOW - timedelta(seconds=599)
                    if boundary == "ingestion"
                    else TASK4_NOW - timedelta(seconds=60)
                ),
                rows=[],
            )

    reader = AdvancingSnapshotReader()
    monkeypatch.setattr(api, "now_utc", clock)
    monkeypatch.setattr(api, "db_acquire_evaluator_lock", lambda: connection)
    monkeypatch.setattr(
        api,
        "load_evaluator_candidates",
        lambda candidate_conn, at: [rule],
    )
    monkeypatch.setattr(api, "load_facility_hours", lambda: {})

    def schedule_open(
        payload: object,
        facility_id: int,
        at: datetime,
        *,
        stale_after_seconds: int,
    ) -> bool:
        assert stale_after_seconds == api.SCHEDULE_STALE_AFTER_SECONDS
        schedule_times.append(at)
        return boundary != "schedule" or at < advanced_now

    def metrics(
        facility_id: int,
        section_key: str,
        snapshots: object,
        at: datetime,
    ) -> dict[str, object] | None:
        metric_times.append(at)
        if boundary == "snapshot" and at >= advanced_now:
            return None
        return {
            "status": "live",
            "coverage": 0.8,
            "percent": 50.0 if boundary == "threshold" and at >= advanced_now else 20.0,
        }

    monkeypatch.setattr(api, "official_facility_is_open", schedule_open)
    monkeypatch.setattr(api, "compute_fresh_section_metrics", metrics)
    monkeypatch.setattr(
        api,
        "claim_pending_rule",
        lambda candidate_conn, rule_id, at: claim_times.append(at) or True,
    )
    monkeypatch.setattr(
        api,
        "send_notification_pinned",
        lambda *args, **kwargs: send_calls.append(1),
    )
    monkeypatch.setattr(api, "finalize_claimed_rule", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        api,
        "db_release_evaluator_lock",
        lambda candidate_conn: events.append("release"),
    )

    result = api.evaluate_rules_once(snapshot_reader=reader)

    assert result["sent"] == 0
    assert claim_times == []
    assert send_calls == []
    assert reader.fetch_times == [TASK4_NOW]
    if boundary != "expiry":
        assert schedule_times[-1] == advanced_now
    if boundary not in {"expiry", "ingestion", "schedule"}:
        assert metric_times[-1] == advanced_now
    assert events[-1] == "release"


def test_evaluator_rereads_snapshot_before_each_claim_gate(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
) -> None:
    advanced_now = TASK4_NOW + timedelta(seconds=1)

    class MutableClock:
        value = TASK4_NOW

        def __call__(self) -> datetime:
            return self.value

    class AdvancingSnapshotReader:
        def __init__(self) -> None:
            self.read_times: list[datetime] = []

        def fetch_live_snapshot(self, at: datetime) -> SimpleNamespace:
            self.read_times.append(at)
            return SimpleNamespace(
                last_successful_fetch_at=at,
                rows=[{"version": len(self.read_times)}],
            )

    clock = MutableClock()
    reader = AdvancingSnapshotReader()
    events: list[str] = []
    connection = Task4EvaluatorConnection(events)
    rules = [
        task4_rule(valid_subscription, rule_id=41),
        task4_rule(valid_subscription, rule_id=42, threshold=41),
    ]
    claim_ids: list[int] = []
    monkeypatch.setattr(api, "now_utc", clock)
    monkeypatch.setattr(api, "db_acquire_evaluator_lock", lambda: connection)
    monkeypatch.setattr(
        api,
        "load_evaluator_candidates",
        lambda candidate_conn, at: rules,
    )
    monkeypatch.setattr(api, "load_facility_hours", lambda: {})
    monkeypatch.setattr(
        api,
        "official_facility_is_open",
        lambda payload, facility_id, at, *, stale_after_seconds: (
            stale_after_seconds == api.SCHEDULE_STALE_AFTER_SECONDS
        ),
    )
    monkeypatch.setattr(api, "index_live_rows", lambda rows: rows[0])
    monkeypatch.setattr(
        api,
        "compute_fresh_section_metrics",
        lambda facility_id, section_key, snapshots, at: {
            "status": "live",
            "coverage": 0.8,
            "percent": 20.0 if snapshots["version"] == 1 else 50.0,
        },
    )
    monkeypatch.setattr(
        api,
        "claim_pending_rule",
        lambda candidate_conn, rule_id, at: claim_ids.append(rule_id) or True,
    )
    monkeypatch.setattr(
        api,
        "finalize_claimed_rule",
        lambda *args, **kwargs: True,
    )

    def send(*args: object, **kwargs: object) -> None:
        events.append("send")
        clock.value = advanced_now

    monkeypatch.setattr(api, "send_notification_pinned", send)
    monkeypatch.setattr(
        api,
        "db_release_evaluator_lock",
        lambda candidate_conn: events.append("release"),
    )

    result = api.evaluate_rules_once(snapshot_reader=reader)

    assert result["sent"] == 1
    assert result["skippedThreshold"] == 1
    assert reader.read_times == [TASK4_NOW, advanced_now]
    assert claim_ids == [41]
    assert events.count("send") == 1
    assert events[-1] == "release"


@pytest.mark.parametrize("backward_stage", ["claim", "terminal"])
def test_evaluator_rejects_backward_live_clock_without_unsafe_transition(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
    backward_stage: str,
) -> None:
    claim_now = TASK4_NOW + timedelta(seconds=1)
    samples = iter(
        [TASK4_NOW, claim_now, TASK4_NOW]
        if backward_stage == "claim"
        else [TASK4_NOW, TASK4_NOW, claim_now, TASK4_NOW]
    )
    events, _connection, reader = configure_task4_evaluator(
        monkeypatch,
        valid_subscription,
    )
    state = {"status": "pending"}
    claim_times: list[datetime] = []
    terminal_times: list[datetime] = []
    rule = task4_rule(valid_subscription)
    monkeypatch.setattr(api, "now_utc", lambda: next(samples))
    monkeypatch.setattr(
        api,
        "load_evaluator_candidates",
        lambda connection, at: [rule],
    )

    def claim(connection: object, rule_id: int, at: datetime) -> bool:
        claim_times.append(at)
        state["status"] = "claimed"
        return True

    def finalize(
        connection: object,
        rule_id: int,
        at: datetime,
        status: str,
        failure_code: str | None = None,
    ) -> bool:
        terminal_times.append(at)
        state["status"] = status
        return True

    monkeypatch.setattr(api, "claim_pending_rule", claim)
    monkeypatch.setattr(api, "finalize_claimed_rule", finalize)

    result = api.evaluate_rules_once(snapshot_reader=reader)

    assert result["failed"] == 1
    if backward_stage == "claim":
        assert state["status"] == "pending"
        assert claim_times == []
        assert "send" not in events
    else:
        assert state["status"] == "claimed"
        assert claim_times == [claim_now]
        assert events.count("send") == 1
    assert terminal_times == []
    assert events[-1] == "release"


def test_evaluator_rejects_second_claim_clock_older_than_prior_terminal(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
) -> None:
    first_claim_at = TASK4_NOW + timedelta(seconds=1)
    first_terminal_at = TASK4_NOW + timedelta(seconds=3)
    samples = iter(
        [
            TASK4_NOW,
            TASK4_NOW,
            first_claim_at,
            first_terminal_at,
            TASK4_NOW + timedelta(seconds=2),
        ]
    )
    events, _connection, reader = configure_task4_evaluator(
        monkeypatch,
        valid_subscription,
    )
    rules = [
        task4_rule(valid_subscription, rule_id=41),
        task4_rule(valid_subscription, rule_id=42, threshold=41),
    ]
    claim_times: list[tuple[int, datetime]] = []
    terminal_times: list[tuple[int, datetime]] = []
    monkeypatch.setattr(api, "now_utc", lambda: next(samples))
    monkeypatch.setattr(
        api,
        "load_evaluator_candidates",
        lambda connection, at: rules,
    )
    monkeypatch.setattr(
        api,
        "claim_pending_rule",
        lambda connection, rule_id, at: claim_times.append((rule_id, at)) or True,
    )
    monkeypatch.setattr(
        api,
        "finalize_claimed_rule",
        lambda connection, rule_id, at, status, failure_code=None: terminal_times.append(
            (rule_id, at)
        )
        or True,
    )

    result = api.evaluate_rules_once(snapshot_reader=reader)

    assert result["sent"] == 1
    assert result["failed"] == 1
    assert claim_times == [(41, first_claim_at)]
    assert terminal_times == [(41, first_terminal_at)]
    assert events.count("send") == 1
    assert events[-1] == "release"


def test_evaluator_lock_contention_performs_no_rule_schedule_snapshot_or_count_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "db_acquire_evaluator_lock", lambda: None)
    for name in (
        "load_evaluator_candidates",
        "load_facility_hours",
        "db_rules_count",
    ):
        monkeypatch.setattr(
            api,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail(
                f"lock contention reached {_name}"
            ),
            raising=False,
        )

    result = api.evaluate_rules_once(now=TASK4_NOW)

    assert result["skippedLocked"] == 1
    assert result["rules"] == 0


@pytest.mark.mysql
def test_real_mysql_evaluator_lock_contention_skips_every_state_read_and_releases(
    monkeypatch: pytest.MonkeyPatch,
    mysql_push_test_client: Any,
    migrated_push_database: dict[str, object],
) -> None:
    del mysql_push_test_client
    forbidden_reads: list[str] = []

    def forbid_read(name: str) -> None:
        forbidden_reads.append(name)
        pytest.fail(f"evaluator lock contention reached {name}")

    monkeypatch.setattr(
        api,
        "load_evaluator_candidates",
        lambda *args, **kwargs: forbid_read("rules"),
    )
    monkeypatch.setattr(
        api,
        "load_facility_hours",
        lambda *args, **kwargs: forbid_read("schedule"),
    )
    monkeypatch.setattr(
        api,
        "db_rules_count",
        lambda *args, **kwargs: forbid_read("count"),
    )

    class ForbiddenSnapshotReader:
        def fetch_live_snapshot(self, at: datetime) -> object:
            del at
            return forbid_read("snapshot")

    lock_owner_settings = {**migrated_push_database, "autocommit": True}
    lock_owner = pymysql.connect(**lock_owner_settings)
    acquired = False
    release_result: tuple[object, ...] | None = None
    try:
        with lock_owner.cursor() as cursor:
            cursor.execute(
                "SELECT GET_LOCK(%s, 0)",
                (api.PUSH_EVALUATOR_DB_LOCK_NAME,),
            )
            acquired = cursor.fetchone() == (1,)
        assert acquired is True

        result = api.evaluate_rules_once(
            now=TASK4_NOW,
            snapshot_reader=ForbiddenSnapshotReader(),
        )
    finally:
        if acquired:
            with lock_owner.cursor() as cursor:
                cursor.execute(
                    "SELECT RELEASE_LOCK(%s)",
                    (api.PUSH_EVALUATOR_DB_LOCK_NAME,),
                )
                release_result = cursor.fetchone()
        lock_owner.close()

    assert result == {
        "status": "ok",
        "rules": 0,
        "sent": 0,
        "failed": 0,
        "skippedThreshold": 0,
        "skippedCooldown": 0,
        "skippedMissingSection": 0,
        "skippedInactive": 0,
        "skippedLocked": 1,
        "evaluatedAt": "2026-09-01T17:00:00+00:00",
    }
    assert forbidden_reads == []
    assert release_result == (1,)
    assert mysql_fetch_all(
        migrated_push_database,
        "SELECT IS_FREE_LOCK(%s)",
        (api.PUSH_EVALUATOR_DB_LOCK_NAME,),
    ) == ((1,),)


def test_evaluator_database_lock_failure_is_distinct_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api,
        "db_acquire_evaluator_lock",
        lambda: (_ for _ in ()).throw(
            api.PushEvaluatorStoreError("database-secret-sentinel")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        api.evaluate_rules_once(now=TASK4_NOW)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "push_evaluator_store_unavailable"
    assert "database-secret-sentinel" not in str(exc_info.value)


@pytest.mark.parametrize("operation", ["acquire", "release"])
def test_evaluator_lock_rejects_malformed_multi_column_results(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    class LockCursor:
        def __enter__(self) -> "LockCursor":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, _sql: str, _params: tuple[object, ...]) -> None:
            return None

        def fetchone(self) -> tuple[int, int]:
            return (1, 1)

    class LockConnection:
        def __init__(self) -> None:
            self.close_count = 0

        def cursor(self) -> LockCursor:
            return LockCursor()

        def close(self) -> None:
            self.close_count += 1

    connection = LockConnection()
    if operation == "acquire":
        monkeypatch.setattr(
            api,
            "open_db_connection",
            lambda *, autocommit: connection,
        )
        invoke = api.db_acquire_evaluator_lock
    else:
        def invoke() -> None:
            api.db_release_evaluator_lock(connection)

    with pytest.raises(api.PushEvaluatorStoreError):
        invoke()
    assert connection.close_count == 1


def test_evaluator_with_no_candidates_commits_expiry_without_schedule_or_snapshot_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    connection = Task4EvaluatorConnection(events)
    monkeypatch.setattr(api, "db_acquire_evaluator_lock", lambda: connection)
    monkeypatch.setattr(api, "load_evaluator_candidates", lambda conn, now: [])
    monkeypatch.setattr(
        api,
        "load_facility_hours",
        lambda: pytest.fail("empty evaluator loaded schedule"),
    )
    monkeypatch.setattr(
        api,
        "db_release_evaluator_lock",
        lambda conn: events.append("release"),
    )

    result = api.evaluate_rules_once(now=TASK4_NOW)

    assert result["rules"] == 0
    assert result["sent"] == 0
    assert events == ["commit:1", "release"]


def test_evaluator_lock_release_failure_is_fixed_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = Task4EvaluatorConnection([])
    monkeypatch.setattr(api, "db_acquire_evaluator_lock", lambda: connection)
    monkeypatch.setattr(api, "load_evaluator_candidates", lambda conn, now: [])
    monkeypatch.setattr(api, "load_facility_hours", lambda: {})
    monkeypatch.setattr(
        api,
        "db_release_evaluator_lock",
        lambda conn: (_ for _ in ()).throw(
            api.PushEvaluatorStoreError("release-secret-sentinel")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        api.evaluate_rules_once(now=TASK4_NOW)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "push_evaluator_store_unavailable"
    assert "release-secret-sentinel" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("ingestion_at", "schedule_open", "metrics", "expected_threshold"),
    [
        (datetime(2026, 9, 1, 16, 50, tzinfo=timezone.utc), True, None, 0),
        (datetime(2026, 9, 1, 16, 49, 59, 999999, tzinfo=timezone.utc), True, None, 0),
        (datetime(2026, 9, 1, 17, 0, 0, 1, tzinfo=timezone.utc), True, None, 0),
        (datetime(2026, 9, 1, 16, 59), True, None, 0),
        (None, True, None, 0),
        (datetime(2026, 9, 1, 16, 59, tzinfo=timezone.utc), False, None, 0),
        (
            datetime(2026, 9, 1, 16, 59, tzinfo=timezone.utc),
            True,
            {"status": "partial", "coverage": 0.79, "percent": 1.0},
            0,
        ),
        (
            datetime(2026, 9, 1, 16, 59, tzinfo=timezone.utc),
            True,
            {"status": "live", "coverage": 0.8, "percent": 41.0},
            1,
        ),
    ],
    ids=[
        "exact-600-seconds-qualifies",
        "just-stale-ingestion",
        "future-ingestion",
        "naive-ingestion",
        "missing-ingestion",
        "closed-schedule",
        "coverage-below-boundary",
        "above-threshold",
    ],
)
def test_evaluator_enforces_ingestion_schedule_coverage_and_threshold_gates(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
    ingestion_at: datetime | None,
    schedule_open: bool,
    metrics: dict[str, object] | None,
    expected_threshold: int,
) -> None:
    events, _conn, reader = configure_task4_evaluator(
        monkeypatch,
        valid_subscription,
        ingestion_at=ingestion_at,
        schedule_open=schedule_open,
        metrics=metrics,
    )

    result = api.evaluate_rules_once(now=TASK4_NOW, snapshot_reader=reader)

    if ingestion_at == datetime(2026, 9, 1, 16, 50, tzinfo=timezone.utc):
        assert result["sent"] == 1
        assert "send" in events
    else:
        assert result["sent"] == 0
        assert "claim" not in events
        assert "send" not in events
    assert result.get("skippedThreshold", 0) == expected_threshold


def test_evaluator_conditional_claim_loss_never_sends(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
) -> None:
    events, _conn, reader = configure_task4_evaluator(
        monkeypatch,
        valid_subscription,
        claim=False,
    )

    result = api.evaluate_rules_once(now=TASK4_NOW, snapshot_reader=reader)

    assert result["sent"] == 0
    assert "claim" in events
    assert "send" not in events
    assert not any(item.startswith("terminal:") for item in events)


def test_evaluator_malformed_stored_subscription_never_claims_sends_or_leaks(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    events, _conn, reader = configure_task4_evaluator(
        monkeypatch,
        valid_subscription,
    )
    secret = "stored-subscription-secret-sentinel"
    malformed = task4_rule({"endpoint": secret})
    monkeypatch.setattr(
        api,
        "load_evaluator_candidates",
        lambda connection, now: [malformed],
    )

    result = api.evaluate_rules_once(now=TASK4_NOW, snapshot_reader=reader)

    assert result["sent"] == 0
    assert "claim" not in events
    assert "send" not in events
    output = capsys.readouterr()
    assert secret not in output.out + output.err + repr(result)


def test_process_abort_after_committed_claim_never_finalizes_or_retries(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
) -> None:
    def abort_after_claim(*args: object, **kwargs: object) -> None:
        events.append("send-abort")
        raise SystemExit("process-like-abort")

    events: list[str] = []
    connection = Task4EvaluatorConnection(events)
    events, _conn, reader = configure_task4_evaluator(
        monkeypatch,
        valid_subscription,
        connection=connection,
        sender=abort_after_claim,
    )

    with pytest.raises(SystemExit, match="process-like-abort"):
        api.evaluate_rules_once(now=TASK4_NOW, snapshot_reader=reader)

    assert events.index("commit:2") < events.index("send-abort")
    assert not any(item.startswith("terminal:") for item in events)
    assert "release" in events


def test_terminal_commit_failure_leaves_claimed_and_never_retries_send(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
) -> None:
    events: list[str] = []
    connection = Task4EvaluatorConnection(events, fail_terminal_commit=True)
    events, _conn, reader = configure_task4_evaluator(
        monkeypatch,
        valid_subscription,
        connection=connection,
    )

    result = api.evaluate_rules_once(now=TASK4_NOW, snapshot_reader=reader)

    assert result["sent"] == 0
    assert result["failed"] == 1
    assert events.count("send") == 1
    assert events.index("commit:2") < events.index("send")
    assert "rollback" in events


@pytest.mark.parametrize(
    ("status_code", "expected_status", "expected_code"),
    [
        (404, "invalid_subscription", "webpush_404"),
        (410, "invalid_subscription", "webpush_410"),
        (302, "failed", "webpush_failed"),
        (500, "failed", "webpush_failed"),
    ],
)
def test_evaluator_maps_provider_status_to_fixed_terminal_audit_state(
    monkeypatch: pytest.MonkeyPatch,
    valid_subscription: dict[str, object],
    status_code: int,
    expected_status: str,
    expected_code: str,
) -> None:
    def fail_send(*args: object, **kwargs: object) -> None:
        raise WebPushException(
            "provider-secret-sentinel",
            response=SimpleNamespace(status_code=status_code),
        )

    events, _conn, reader = configure_task4_evaluator(
        monkeypatch,
        valid_subscription,
        sender=fail_send,
    )

    result = api.evaluate_rules_once(now=TASK4_NOW, snapshot_reader=reader)

    assert result["sent"] == 0
    assert result["failed"] == 1
    assert f"terminal:{expected_status}:{expected_code}" in events
    assert not any("provider-secret-sentinel" in event for event in events)


def test_evaluator_loop_emits_only_fixed_allowlisted_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def stop_after_error(_seconds: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(
        api,
        "evaluate_rules_once",
        lambda: (_ for _ in ()).throw(RuntimeError("evaluator-secret-sentinel")),
    )
    monkeypatch.setattr(api.asyncio, "sleep", stop_after_error)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(api.evaluator_loop())

    output = capsys.readouterr()
    assert output.out.strip() == "[push-evaluator] error=push_evaluation_failed"
    assert "evaluator-secret-sentinel" not in output.out + output.err


def test_admin_dispatch_request_forbids_copy_and_url_overrides() -> None:
    assert set(api.PushDispatchRequest.model_fields) == {"facilityId", "sectionKey"}
    with pytest.raises(Exception):
        api.PushDispatchRequest.model_validate(
            {"facilityId": 1186, "title": "evil", "body": "evil", "url": "https://evil"}
        )


def test_evaluator_lifecycle_has_no_rule_delete_or_legacy_unpinned_store_path() -> None:
    source = inspect.getsource(api)
    assert "DELETE FROM {table_name}" not in source
    assert not hasattr(api, "db_delete_rule_by_id")
    assert not hasattr(api, "load_store_from_db")
    assert "def send_notification(" not in source


@pytest.mark.mysql
def test_candidate_load_expires_only_pending_and_never_reclaims_claimed(
    mysql_push_test_client: Any,
    migrated_push_database: dict[str, object],
    valid_subscription: dict[str, object],
) -> None:
    del mysql_push_test_client
    expired_pending = insert_mysql_push_rule(
        migrated_push_database,
        valid_subscription,
        threshold=10,
        expires_at=TASK4_NOW - timedelta(seconds=1),
    )
    expired_claimed = insert_mysql_push_rule(
        migrated_push_database,
        valid_subscription,
        threshold=20,
        status="claimed",
        expires_at=TASK4_NOW - timedelta(seconds=1),
    )
    live_pending = insert_mysql_push_rule(
        migrated_push_database,
        valid_subscription,
        threshold=30,
        expires_at=TASK4_NOW + timedelta(seconds=1),
    )
    connection = api.open_db_connection(autocommit=False)
    try:
        candidates = api.load_evaluator_candidates(connection, TASK4_NOW)
        connection.commit()
    finally:
        connection.close()

    assert [rule.id for rule in candidates] == [live_pending]
    assert mysql_fetch_all(
        migrated_push_database,
        "SELECT id, status, finalized_at IS NOT NULL FROM push_rules ORDER BY id",
    ) == (
        (expired_pending, "expired", 1),
        (expired_claimed, "claimed", 0),
        (live_pending, "pending", 0),
    )


@pytest.mark.mysql
def test_evaluator_refreshes_repeatable_read_view_before_later_candidate(
    monkeypatch: pytest.MonkeyPatch,
    mysql_push_test_client: Any,
    migrated_push_database: dict[str, object],
    valid_subscription: dict[str, object],
) -> None:
    del mysql_push_test_client
    mysql_execute(migrated_push_database, "DELETE FROM location_snapshot")
    mysql_execute(migrated_push_database, "DELETE FROM ingestion_runs")
    first_rule_id = insert_mysql_push_rule(
        migrated_push_database,
        valid_subscription,
        threshold=10,
        expires_at=TASK4_NOW + timedelta(hours=1),
    )
    second_rule_id = insert_mysql_push_rule(
        migrated_push_database,
        valid_subscription,
        threshold=40,
        expires_at=TASK4_NOW + timedelta(hours=1),
    )
    mysql_execute(
        migrated_push_database,
        "INSERT INTO ingestion_runs "
        "(started_at, completed_at, status, observed_location_ids) "
        "VALUES (%s, %s, 'succeeded', JSON_ARRAY(99001))",
        (datetime(2026, 9, 1, 16, 57), datetime(2026, 9, 1, 16, 58)),
    )
    mysql_execute(
        migrated_push_database,
        "INSERT INTO location_snapshot "
        "(location_id, is_closed, current_capacity, max_capacity, fetched_at) "
        "VALUES (99001, 0, 90, 100, %s)",
        (datetime(2026, 9, 1, 16, 58),),
    )
    monkeypatch.setattr(api, "location_ids_for_section", lambda *_: [99001])
    monkeypatch.setattr(api, "MAX_CAP", {99001: 100})
    monkeypatch.setattr(
        api,
        "load_facility_hours",
        lambda: task4_schedule_payload(
            [{"label": "Daily", "hours": "Open 24 Hours"}]
        ),
    )
    sends: list[int] = []
    monkeypatch.setattr(
        api,
        "send_notification_pinned",
        lambda *args, **kwargs: sends.append(1),
    )
    first_snapshot_read = threading.Event()
    writer_committed = threading.Event()
    lock_owner_ids: list[int] = []

    class CoordinatedRepository:
        def __init__(self, connection: object) -> None:
            self.inner = api.SnapshotRepository(connection)
            self.calls = 0

        def fetch_live_snapshot(self, at: datetime) -> object:
            snapshot = self.inner.fetch_live_snapshot(at)
            self.calls += 1
            if self.calls == 1:
                first_snapshot_read.set()
                if not writer_committed.wait(5):
                    raise RuntimeError("concurrent snapshot writer did not commit")
            return snapshot

    def commit_new_snapshot() -> None:
        if not first_snapshot_read.wait(5):
            raise RuntimeError("evaluator did not read its first snapshot")
        writer_settings = {**migrated_push_database, "autocommit": False}
        writer_connection = pymysql.connect(**writer_settings)
        try:
            with writer_connection.cursor() as cursor:
                cursor.execute(
                    "SELECT IS_USED_LOCK(%s)",
                    (api.PUSH_EVALUATOR_DB_LOCK_NAME,),
                )
                [(lock_owner,)] = [cursor.fetchone()]
                if type(lock_owner) is not int:
                    raise RuntimeError("evaluator advisory lock was not held")
                lock_owner_ids.append(lock_owner)
                cursor.execute(
                    "UPDATE location_snapshot SET current_capacity = 20, fetched_at = %s "
                    "WHERE location_id = 99001",
                    (datetime(2026, 9, 1, 16, 59),),
                )
                cursor.execute(
                    "INSERT INTO ingestion_runs "
                    "(started_at, completed_at, status, observed_location_ids) "
                    "VALUES (%s, %s, 'succeeded', JSON_ARRAY(99001))",
                    (
                        datetime(2026, 9, 1, 16, 58, 30),
                        datetime(2026, 9, 1, 16, 59),
                    ),
                )
            writer_connection.commit()
        finally:
            writer_connection.close()
            writer_committed.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        writer = executor.submit(commit_new_snapshot)
        first_result = api.evaluate_rules_once(
            now=TASK4_NOW,
            repository_factory=CoordinatedRepository,
        )
        writer.result(timeout=5)

    assert first_result["sent"] == 1
    assert sends == [1]
    assert len(lock_owner_ids) == 1
    second_result = api.evaluate_rules_once(now=TASK4_NOW)
    assert second_result["sent"] == 0
    assert sends == [1]
    assert mysql_fetch_all(
        migrated_push_database,
        "SELECT id, status FROM push_rules ORDER BY id",
    ) == (
        (first_rule_id, "pending"),
        (second_rule_id, "sent"),
    )


@pytest.mark.mysql
def test_two_real_mysql_evaluators_send_qualifying_rule_at_most_once(
    monkeypatch: pytest.MonkeyPatch,
    mysql_push_test_client: Any,
    migrated_push_database: dict[str, object],
    valid_subscription: dict[str, object],
) -> None:
    del mysql_push_test_client
    mysql_execute(migrated_push_database, "DELETE FROM location_snapshot")
    mysql_execute(migrated_push_database, "DELETE FROM ingestion_runs")
    rule_id = insert_mysql_push_rule(
        migrated_push_database,
        valid_subscription,
        threshold=40,
        expires_at=TASK4_NOW + timedelta(hours=1),
    )
    mysql_execute(
        migrated_push_database,
        "INSERT INTO ingestion_runs "
        "(started_at, completed_at, status, observed_location_ids) "
        "VALUES (%s, %s, 'succeeded', JSON_ARRAY(99001))",
        (
            datetime(2026, 9, 1, 16, 58),
            datetime(2026, 9, 1, 16, 59),
        ),
    )
    mysql_execute(
        migrated_push_database,
        "INSERT INTO location_snapshot "
        "(location_id, is_closed, current_capacity, max_capacity, fetched_at) "
        "VALUES (99001, 0, 20, 100, %s)",
        (datetime(2026, 9, 1, 16, 59),),
    )
    monkeypatch.setattr(api, "location_ids_for_section", lambda *_: [99001])
    monkeypatch.setattr(api, "MAX_CAP", {99001: 100})
    monkeypatch.setattr(
        api,
        "load_facility_hours",
        lambda: task4_schedule_payload(
            [{"label": "Daily", "hours": "Open 24 Hours"}]
        ),
    )
    sends: list[int] = []
    send_lock = __import__("threading").Lock()

    def send_once(*args: object, **kwargs: object) -> None:
        with send_lock:
            sends.append(1)

    monkeypatch.setattr(api, "send_notification_pinned", send_once)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda _: api.evaluate_rules_once(now=TASK4_NOW), range(2))
        )

    assert sum(result["sent"] for result in results) == 1
    assert sends == [1]
    assert mysql_fetch_all(
        migrated_push_database,
        "SELECT status, sent_at IS NOT NULL, finalized_at IS NOT NULL, failure_code "
        "FROM push_rules WHERE id = %s",
        (rule_id,),
    ) == (("sent", 1, 1, None),)
    assert mysql_fetch_all(
        migrated_push_database,
        "SELECT IS_FREE_LOCK(%s)",
        (api.PUSH_EVALUATOR_DB_LOCK_NAME,),
    ) == ((1,),)


TASK5_NOW = datetime(2026, 9, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)
TASK5_DELETE_SQL = "DELETE FROM push_rate_limits WHERE updated_at < %s"
TASK5_SAFE_ERROR = "push_rate_limit_prune_failed"


class PruneCursor:
    def __init__(
        self,
        connection: "PruneConnection",
        *,
        rowcount: object,
        fail_at: str | None = None,
    ) -> None:
        self.connection = connection
        self.rowcount = rowcount
        self.fail_at = fail_at

    def __enter__(self) -> "PruneCursor":
        self.connection.events.append("cursor_enter")
        if self.fail_at == "cursor_enter":
            raise RuntimeError("prune-cursor-enter-secret-sentinel")
        return self

    def __exit__(self, *_args: object) -> None:
        self.connection.events.append("cursor_exit")
        if self.fail_at == "cursor_exit":
            raise RuntimeError("prune-cursor-exit-secret-sentinel")

    def execute(self, sql: str, params: tuple[object, ...]) -> None:
        self.connection.events.append("execute")
        self.connection.statements.append((sql, params))
        if self.fail_at == "execute":
            raise RuntimeError("prune-execute-secret-sentinel")


class PruneConnection:
    def __init__(
        self,
        *,
        rowcount: object = 1,
        fail_at: str | None = None,
        rollback_fails: bool = False,
    ) -> None:
        self.rowcount = rowcount
        self.fail_at = fail_at
        self.rollback_fails = rollback_fails
        self.events: list[str] = []
        self.statements: list[tuple[str, tuple[object, ...]]] = []
        self.commits = 0
        self.rollbacks = 0
        self.closes = 0

    def cursor(self) -> PruneCursor:
        self.events.append("cursor")
        if self.fail_at == "cursor":
            raise RuntimeError("prune-cursor-secret-sentinel")
        return PruneCursor(
            self,
            rowcount=self.rowcount,
            fail_at=self.fail_at,
        )

    def commit(self) -> None:
        self.events.append("commit")
        self.commits += 1
        if self.fail_at == "commit":
            raise RuntimeError("prune-commit-secret-sentinel")

    def rollback(self) -> None:
        self.events.append("rollback")
        self.rollbacks += 1
        if self.rollback_fails:
            raise RuntimeError("prune-rollback-secret-sentinel")

    def close(self) -> None:
        self.events.append("close")
        self.closes += 1
        if self.fail_at == "close":
            raise RuntimeError("prune-close-secret-sentinel")


class ExplodingOutput:
    def __init__(self, sentinel: str) -> None:
        self.sentinel = sentinel
        self.writes: list[str] = []

    def write(self, value: str) -> int:
        self.writes.append(value)
        raise RuntimeError(self.sentinel)

    def flush(self) -> None:
        return None


def load_prune_module() -> Any:
    return importlib.import_module("prune_push_rate_limits")


def configure_prune_runtime(
    monkeypatch: pytest.MonkeyPatch,
    prune: Any,
    connection: PruneConnection,
    *,
    window_seconds: object = 600,
    open_fails: bool = False,
) -> list[dict[str, object]]:
    open_calls: list[dict[str, object]] = []

    def open_connection(**kwargs: object) -> PruneConnection:
        open_calls.append(kwargs)
        if open_fails:
            raise RuntimeError("prune-open-secret-sentinel")
        return connection

    monkeypatch.setattr(
        prune,
        "_load_runtime_dependencies",
        lambda: (open_connection, window_seconds),
    )
    return open_calls


@pytest.mark.parametrize(
    ("now", "expected_cutoff"),
    [
        (
            TASK5_NOW,
            datetime(2026, 9, 1, 11, 40, 0, 123456),
        ),
        (
            datetime(9999, 12, 31, 23, 59, tzinfo=timezone.utc),
            datetime(9999, 12, 31, 23, 39),
        ),
    ],
)
def test_prune_uses_one_strict_parameterized_delete_and_commits(
    monkeypatch: pytest.MonkeyPatch,
    now: datetime,
    expected_cutoff: datetime,
) -> None:
    prune = load_prune_module()
    connection = PruneConnection(rowcount=2)
    open_calls = configure_prune_runtime(monkeypatch, prune, connection)

    assert prune.prune_push_rate_limits(now=now) == 2

    assert open_calls == [{"autocommit": False}]
    assert connection.statements == [
        (TASK5_DELETE_SQL, (expected_cutoff,)),
    ]
    assert connection.events == [
        "cursor",
        "cursor_enter",
        "execute",
        "cursor_exit",
        "commit",
        "close",
    ]
    assert (connection.commits, connection.rollbacks, connection.closes) == (1, 0, 1)


def test_prune_samples_default_utc_clock_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prune = load_prune_module()
    connection = PruneConnection(rowcount=0)
    configure_prune_runtime(monkeypatch, prune, connection)
    samples: list[datetime] = []

    def sample_once() -> datetime:
        samples.append(TASK5_NOW)
        return TASK5_NOW

    monkeypatch.setattr(prune, "_utc_now", sample_once)

    assert prune.prune_push_rate_limits() == 0
    assert samples == [TASK5_NOW]


def test_prune_sanitizes_clock_failure_before_dependency_or_database_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prune = load_prune_module()
    connection = PruneConnection()
    dependency_loads: list[int] = []

    def fail_clock() -> datetime:
        raise RuntimeError("prune-clock-secret-sentinel")

    def load_dependencies() -> tuple[object, int]:
        dependency_loads.append(1)
        return (lambda **_kwargs: connection), 600

    monkeypatch.setattr(prune, "_utc_now", fail_clock)
    monkeypatch.setattr(prune, "_load_runtime_dependencies", load_dependencies)

    with pytest.raises(prune.PushRateLimitPruneError) as error:
        prune.prune_push_rate_limits()

    assert str(error.value) == TASK5_SAFE_ERROR
    assert "secret-sentinel" not in str(error.value) + repr(error.value)
    assert dependency_loads == []
    assert connection.events == []


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 9, 1, 12, 0),
        datetime(
            2026,
            9,
            1,
            13,
            0,
            tzinfo=timezone(timedelta(hours=1)),
        ),
    ],
)
def test_prune_rejects_naive_and_non_utc_clock_before_database_access(
    monkeypatch: pytest.MonkeyPatch,
    now: datetime,
) -> None:
    prune = load_prune_module()
    connection = PruneConnection()
    open_calls = configure_prune_runtime(monkeypatch, prune, connection)

    with pytest.raises(prune.PushRateLimitPruneError) as error:
        prune.prune_push_rate_limits(now=now)

    assert str(error.value) == TASK5_SAFE_ERROR
    assert open_calls == []
    assert connection.events == []


@pytest.mark.parametrize("window_seconds", [0, -1, True, 600.0, "600"])
def test_prune_rejects_nonpositive_or_non_plain_integer_windows(
    monkeypatch: pytest.MonkeyPatch,
    window_seconds: object,
) -> None:
    prune = load_prune_module()
    connection = PruneConnection()
    open_calls = configure_prune_runtime(
        monkeypatch,
        prune,
        connection,
        window_seconds=window_seconds,
    )

    with pytest.raises(prune.PushRateLimitPruneError) as error:
        prune.prune_push_rate_limits(now=TASK5_NOW)

    assert str(error.value) == TASK5_SAFE_ERROR
    assert open_calls == []
    assert connection.events == []


@pytest.mark.parametrize(
    ("now", "window_seconds"),
    [
        (datetime.min.replace(tzinfo=timezone.utc), 600),
        (TASK5_NOW, 10**30),
    ],
)
def test_prune_sanitizes_cutoff_subtraction_overflow_before_database_access(
    monkeypatch: pytest.MonkeyPatch,
    now: datetime,
    window_seconds: int,
) -> None:
    prune = load_prune_module()
    connection = PruneConnection()
    open_calls = configure_prune_runtime(
        monkeypatch,
        prune,
        connection,
        window_seconds=window_seconds,
    )

    with pytest.raises(prune.PushRateLimitPruneError) as error:
        prune.prune_push_rate_limits(now=now)

    assert str(error.value) == TASK5_SAFE_ERROR
    assert open_calls == []
    assert connection.events == []


@pytest.mark.parametrize("rowcount", [None, True, "1", -1])
def test_prune_rejects_ambiguous_rowcount_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    rowcount: object,
) -> None:
    prune = load_prune_module()
    connection = PruneConnection(rowcount=rowcount)
    configure_prune_runtime(monkeypatch, prune, connection)

    with pytest.raises(prune.PushRateLimitPruneError) as error:
        prune.prune_push_rate_limits(now=TASK5_NOW)

    assert str(error.value) == TASK5_SAFE_ERROR
    assert (connection.commits, connection.rollbacks, connection.closes) == (0, 1, 1)
    assert connection.events[-2:] == ["rollback", "close"]


@pytest.mark.parametrize(
    "fail_at",
    ["open", "cursor", "cursor_enter", "execute", "cursor_exit", "commit"],
)
def test_prune_database_failures_use_fixed_error_and_close_once(
    monkeypatch: pytest.MonkeyPatch,
    fail_at: str,
) -> None:
    prune = load_prune_module()
    connection = PruneConnection(fail_at=None if fail_at == "open" else fail_at)
    configure_prune_runtime(
        monkeypatch,
        prune,
        connection,
        open_fails=fail_at == "open",
    )

    with pytest.raises(prune.PushRateLimitPruneError) as error:
        prune.prune_push_rate_limits(now=TASK5_NOW)

    rendered = str(error.value) + repr(error.value)
    assert rendered == TASK5_SAFE_ERROR + repr(error.value)
    assert "secret-sentinel" not in rendered
    if fail_at == "open":
        assert (connection.rollbacks, connection.closes) == (0, 0)
    else:
        assert (connection.rollbacks, connection.closes) == (1, 1)


def test_prune_suppresses_rollback_failure_and_reports_close_failure_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prune = load_prune_module()
    rollback_failure = PruneConnection(fail_at="execute", rollback_fails=True)
    configure_prune_runtime(monkeypatch, prune, rollback_failure)

    with pytest.raises(prune.PushRateLimitPruneError) as rollback_error:
        prune.prune_push_rate_limits(now=TASK5_NOW)

    assert str(rollback_error.value) == TASK5_SAFE_ERROR
    assert (rollback_failure.rollbacks, rollback_failure.closes) == (1, 1)

    close_failure = PruneConnection(fail_at="close")
    configure_prune_runtime(monkeypatch, prune, close_failure)

    with pytest.raises(prune.PushRateLimitPruneError) as close_error:
        prune.prune_push_rate_limits(now=TASK5_NOW)

    assert str(close_error.value) == TASK5_SAFE_ERROR
    assert (close_failure.commits, close_failure.rollbacks, close_failure.closes) == (
        1,
        0,
        1,
    )


def test_prune_cli_has_exact_count_only_success_and_fixed_failure_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prune = load_prune_module()
    monkeypatch.setattr(prune, "prune_push_rate_limits", lambda: 7)

    assert prune.main() == 0
    success = capsys.readouterr()
    assert success.out == "pruned_push_rate_limit_windows=7\n"
    assert success.err == ""

    def fail() -> int:
        raise RuntimeError("cli-config-secret-sentinel")

    monkeypatch.setattr(prune, "prune_push_rate_limits", fail)

    assert prune.main() == 1
    failure = capsys.readouterr()
    assert failure.out == ""
    assert failure.err == TASK5_SAFE_ERROR + "\n"
    assert "secret-sentinel" not in failure.out + failure.err


def test_prune_main_sanitizes_success_stdout_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prune = load_prune_module()
    monkeypatch.setattr(prune, "prune_push_rate_limits", lambda: 7)
    stdout = ExplodingOutput("stdout-write-secret-sentinel")

    with monkeypatch.context() as stream_patch:
        stream_patch.setattr(prune.sys, "stdout", stdout)
        result = prune.main()

    assert result == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == TASK5_SAFE_ERROR + "\n"
    assert "secret-sentinel" not in output.out + output.err


def test_prune_direct_main_suppresses_stdout_and_stderr_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ModuleType("forecast_api")
    connection = PruneConnection(rowcount=3)
    runtime.PUSH_WRITE_RATE_WINDOW_SECONDS = 600
    runtime.open_db_connection = lambda **_kwargs: connection
    monkeypatch.setitem(sys.modules, "forecast_api", runtime)
    script = Path(__file__).resolve().parents[2] / "server/prune_push_rate_limits.py"
    stdout = ExplodingOutput("stdout-write-secret-sentinel")
    stderr = ExplodingOutput("stderr-write-secret-sentinel")

    with monkeypatch.context() as stream_patch:
        stream_patch.setattr(sys, "stdout", stdout)
        stream_patch.setattr(sys, "stderr", stderr)
        with pytest.raises(SystemExit) as result:
            runpy.run_path(str(script), run_name="__main__")

    assert result.value.code == 1
    attempted_output = "".join(stdout.writes + stderr.writes)
    assert TASK5_SAFE_ERROR in attempted_output
    assert "secret-sentinel" not in attempted_output


def test_prune_direct_script_executes_with_exact_success_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = ModuleType("forecast_api")
    connection = PruneConnection(rowcount=3)
    runtime.PUSH_WRITE_RATE_WINDOW_SECONDS = 600
    runtime.open_db_connection = lambda **_kwargs: connection
    monkeypatch.setitem(sys.modules, "forecast_api", runtime)
    script = Path(__file__).resolve().parents[2] / "server/prune_push_rate_limits.py"

    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(script), run_name="__main__")

    assert result.value.code == 0
    output = capsys.readouterr()
    assert output.out == "pruned_push_rate_limit_windows=3\n"
    assert output.err == ""


def test_prune_direct_script_sanitizes_runtime_import_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class ExplodingRuntime(ModuleType):
        def __getattr__(self, _name: str) -> object:
            raise RuntimeError("runtime-import-secret-sentinel")

    monkeypatch.setitem(sys.modules, "forecast_api", ExplodingRuntime("forecast_api"))
    script = Path(__file__).resolve().parents[2] / "server/prune_push_rate_limits.py"

    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(script), run_name="__main__")

    assert result.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == TASK5_SAFE_ERROR + "\n"
    assert "secret-sentinel" not in output.out + output.err


def test_prune_package_import_resolves_script_runtime_without_path_pollution(
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    code = """
import sys
from pathlib import Path

project_root = Path.cwd()
server_path = str(project_root / "server")
assert server_path not in sys.path
import server.prune_push_rate_limits as prune
before = list(sys.path)
open_connection, window_seconds = prune._load_runtime_dependencies()
import forecast_api as runtime
assert open_connection is runtime.open_db_connection
assert window_seconds == runtime.PUSH_WRITE_RATE_WINDOW_SECONDS
assert sys.path == before
assert server_path not in sys.path
print("package_runtime_resolved")
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PUSH_WRITE_RATE_WINDOW_SECONDS"] = "600"

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == "package_runtime_resolved\n"
    assert result.stderr == ""


def test_prune_package_module_reaches_database_and_sanitizes_failure(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    marker = tmp_path / "database-attempted"
    (tmp_path / "sitecustomize.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "import pymysql\n"
        "def fail_connect(*_args, **_kwargs):\n"
        "    Path(os.environ['PRUNE_TEST_DB_MARKER']).write_text('attempted')\n"
        "    raise RuntimeError('package-db-secret-sentinel')\n"
        "pymysql.connect = fail_connect\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(project_root))
    )
    environment["PRUNE_TEST_DB_MARKER"] = str(marker)
    environment["PUSH_WRITE_RATE_WINDOW_SECONDS"] = "600"
    environment.update(
        {
            "GYM_DB_HOST": "127.0.0.1",
            "GYM_DB_PORT": "3306",
            "GYM_DB_USER": "prune-test-user",
            "GYM_DB_PASSWORD": "prune-test-password",
            "GYM_DB_NAME": "prune_test_database",
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "server.prune_push_rate_limits"],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 1
    assert marker.read_text() == "attempted"
    assert result.stdout == ""
    assert result.stderr == TASK5_SAFE_ERROR + "\n"
    assert "secret-sentinel" not in result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr


def test_readme_documents_push_limits_lifecycle_privacy_and_pruning() -> None:
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
    normalized_readme = " ".join(readme.split())
    required_text = (
        "python server/prune_push_rate_limits.py",
        "two 600-second windows",
        "pruned_push_rate_limit_windows=<count>",
        "PUSH_RULE_DEFAULT_TTL_SECONDS=86400",
        "PUSH_RULE_MAX_TTL_SECONDS=604800",
        "PUSH_MAX_ACTIVE_RULES_PER_ENDPOINT=10",
        "PUSH_WRITE_RATE_LIMIT=20",
        "PUSH_WRITE_RATE_WINDOW_SECONDS=600",
        "server-backed list and cancel operations",
        "Normal claimed sends terminalize as sent, failed, or invalid subscription.",
        "A crash or ambiguous post-claim failure can intentionally leave the row "
        "permanently claimed and never retried.",
        "claim is committed before provider I/O",
        "at-most-once provider attempts",
        "does not guarantee terminal state or delivery",
        "Raw push endpoints, subscription keys, client subjects, and provider "
        "response bodies are never returned or logged.",
        "Never dump the environment",
    )
    for expected in required_text:
        assert expected in normalized_readme
    assert "GYM_DB_PASSWORD=" not in readme


@pytest.mark.mysql
def test_prune_real_mysql_keeps_exact_cutoff_and_newer_windows(
    monkeypatch: pytest.MonkeyPatch,
    migrated_push_database: dict[str, object],
) -> None:
    prune = load_prune_module()
    mysql_execute(migrated_push_database, "DELETE FROM push_rate_limits")
    cutoff = datetime(2026, 9, 1, 11, 40)
    rows = (
        (b"a" * 32, datetime(2026, 9, 1, 11, 0), cutoff - timedelta(microseconds=1)),
        (b"b" * 32, datetime(2026, 9, 1, 11, 10), cutoff),
        (b"c" * 32, datetime(2026, 9, 1, 11, 20), cutoff + timedelta(microseconds=1)),
        (b"d" * 32, datetime(2026, 9, 1, 12, 10), cutoff + timedelta(hours=1)),
    )
    for subject_hash, window_started_at, updated_at in rows:
        mysql_execute(
            migrated_push_database,
            "INSERT INTO push_rate_limits "
            "(subject_hash, window_started_at, request_count, updated_at) "
            "VALUES (%s, %s, 1, %s)",
            (subject_hash, window_started_at, updated_at),
        )

    def open_mysql_connection(*, autocommit: bool = True) -> Any:
        return pymysql.connect(
            **{**migrated_push_database, "autocommit": autocommit}
        )

    monkeypatch.setattr(
        prune,
        "_load_runtime_dependencies",
        lambda: (open_mysql_connection, 600),
    )

    assert prune.prune_push_rate_limits(
        now=datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    ) == 1
    assert mysql_fetch_all(
        migrated_push_database,
        "SELECT subject_hash, updated_at FROM push_rate_limits ORDER BY subject_hash",
    ) == (
        (b"b" * 32, cutoff),
        (b"c" * 32, cutoff + timedelta(microseconds=1)),
        (b"d" * 32, cutoff + timedelta(hours=1)),
    )
