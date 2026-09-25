"""Application ownership contracts for the Phase 9 API extraction."""

from dataclasses import replace
import asyncio
import json
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from server.reclive.api.app import create_app
from server.reclive.api.dependencies import get_snapshot_repository
from server.reclive.settings import Settings
from server.reclive.runtime import Runtime, current_runtime, runtime_scope
from server.reclive import push


def production_settings():
    from server.reclive.settings import build_settings_from_environment

    return replace(
        build_settings_from_environment(
            {
                "APP_ENV": "production",
                "GYM_DB_HOST": "fixture-db",
                "GYM_DB_PORT": "3306",
                "GYM_DB_USER": "fixture-user",
                "GYM_DB_PASSWORD": "fixture-password",
                "GYM_DB_NAME": "fixture-db",
                "FORECAST_API_ALLOW_ORIGINS": "https://fixture.invalid",
                "FORECAST_JSON_PATH": "/tmp/fixture-forecast.json",
                "FACILITY_HOURS_JSON_PATH": "/tmp/fixture-hours.json",
                "PUSH_EVALUATOR_ENABLED": "false",
            }
        ),
        capacities={},
        facility_names={},
        section_ids={},
    )


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("cors_origins", ("*",), "FORECAST_API_ALLOW_ORIGINS"),
        ("cors_origins", (), "FORECAST_API_ALLOW_ORIGINS"),
        ("database.password", "", "GYM_DB_PASSWORD"),
        ("database.password", None, "GYM_DB_PASSWORD"),
        ("database.host", "change_me", "GYM_DB_HOST"),
        ("database.user", None, "GYM_DB_USER"),
        ("database.name", " ", "GYM_DB_NAME"),
        ("database.port", 0, "GYM_DB_PORT"),
        ("database.port", 65536, "GYM_DB_PORT"),
        ("database.port", None, "GYM_DB_PORT"),
        ("database.port", "not-a-port", "GYM_DB_PORT"),
        ("forecast_json_path", "", "FORECAST_JSON_PATH"),
        ("facility_hours_json_path", "", "FACILITY_HOURS_JSON_PATH"),
    ],
)
def test_production_lifespan_validates_effective_settings(
    field, value, expected, monkeypatch
):
    monkeypatch.setenv(
        "PUSH_ENDPOINT_HASH_KEY", "fixture-hash-key-with-at-least-32-bytes"
    )
    settings = production_settings()
    if field.startswith("database."):
        settings = replace(
            settings,
            database=replace(settings.database, **{field.split(".")[1]: value}),
        )
    else:
        settings = replace(settings, **{field: value})
    app = create_app(settings)

    async def run():
        async with app.router.lifespan_context(app):
            pytest.fail("unsafe effective settings reached startup")

    with pytest.raises(RuntimeError, match=expected):
        asyncio.run(run())


def test_production_lifespan_uses_effective_environment_not_captured_values(
    monkeypatch,
):
    monkeypatch.setenv(
        "PUSH_ENDPOINT_HASH_KEY", "fixture-hash-key-with-at-least-32-bytes"
    )
    captured = {"APP_ENV": "test", "GYM_DB_PORT": "invalid"}
    good = replace(production_settings(), environment_values=captured)
    app = create_app(good)
    captured["APP_ENV"] = "invalid"

    async def run():
        async with app.router.lifespan_context(app):
            assert app.state.runtime.task is None
        bad = create_app(replace(good, database=replace(good.database, port=0)))
        with pytest.raises(RuntimeError, match="GYM_DB_PORT"):
            async with bad.router.lifespan_context(bad):
                pytest.fail("captured test mode bypassed production validation")

    asyncio.run(run())
    assert good.environment_values["APP_ENV"] == "test"


def test_live_counts_and_forecast_age_use_each_apps_single_clock_sample(tmp_path):
    path = tmp_path / "clock.json"
    generated = datetime(2026, 8, 31, 12, tzinfo=timezone.utc)
    path.write_text(
        json.dumps({"generatedAt": generated.isoformat(), "facilities": []})
    )
    settings = replace(
        Settings.for_test(forecast_json_path=str(path)),
        facility_hours_json_path=str(tmp_path / "missing-facility-hours.json"),
    )
    first = create_app(settings)
    second = create_app(settings)
    samples = []

    class Repository:
        def fetch_live_snapshot(self, now):
            samples.append(now)
            return SimpleNamespace(
                last_successful_fetch_at=generated,
                rows=[
                    SimpleNamespace(
                        location_id=5761,
                        is_closed=False,
                        current_capacity=5,
                        source_updated_at=None,
                        fetched_at=generated,
                    )
                ],
            )

    for app, seconds in ((first, 10), (second, 20)):
        app.state.runtime.clock = lambda delta=seconds: generated + timedelta(
            seconds=delta
        )
        app.dependency_overrides[get_snapshot_repository] = Repository
    for app, seconds in ((first, 10), (second, 20), (first, 10)):
        client = TestClient(app)
        assert (
            client.get("/api/live-counts").json()["ingestion"]["ageSeconds"] == seconds
        )
        assert (
            client.get("/health").json()["components"]["forecast"]["ageSeconds"]
            == seconds
        )
    assert samples == [generated + timedelta(seconds=delta) for delta in (10, 20, 10)]


@pytest.mark.parametrize("first_prefix", ["server.reclive.", "reclive."])
def test_bare_and_package_imports_share_models_dependencies_and_default_app(
    first_prefix, tmp_path
):
    root = Path(__file__).resolve().parents[2]
    code = f"""
import importlib, sys
sys.path[:0] = [{str(root)!r}, {str(root / "server")!r}]
import env_loader
env_loader._DOTENV_STATE.loaded = True
first = {first_prefix!r}
second = 'reclive.' if first == 'server.reclive.' else 'server.reclive.'
for name in ('settings', 'runtime', 'push', 'api.dependencies', 'api.health', 'api.app', 'repositories.push_rules'):
    assert importlib.import_module(first + name) is importlib.import_module(second + name), name
import server.reclive, reclive, server.env_loader
assert server.reclive is reclive
assert server.env_loader is env_loader
bare = importlib.import_module('forecast_api')
package = importlib.import_module('server.forecast_api')
assert bare is package
assert server.forecast_api is bare
from server.reclive import push
from server.reclive.api import dependencies, health as health_owner
assert bare.PushRuleRequest is push.PushRuleRequest
assert bare.PushRuleRecord is push.PushRuleRecord
assert bare.get_snapshot_repository is dependencies.get_snapshot_repository
assert bare.health is health_owner.health
assert bare.push_health is health_owner.push_health
assert bare.app is package.app
print('canonical-imports-ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={"APP_ENV": "test", "PUSH_EVALUATOR_ENABLED": "false"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "canonical-imports-ok"


def test_explicit_database_connectors_keep_captured_options_and_close(monkeypatch):
    from server.reclive import db
    from starlette.requests import Request

    calls = []

    class Connection:
        def get_autocommit(self):
            return False

        def close(self):
            calls.append("close")

    def connect(**kwargs):
        calls.append(kwargs)
        return Connection()

    monkeypatch.setattr(db.pymysql, "connect", connect)
    base = Settings.for_test()
    first = create_app(
        replace(base, database=replace(base.database, host="first-db", port=3307))
    )
    second = create_app(
        replace(base, database=replace(base.database, host="second-db", port=3308))
    )
    for app in (first, second, first):
        generator = get_snapshot_repository(Request({"type": "http", "app": app}))
        next(generator)
        generator.close()
    assert [call["host"] for call in calls if isinstance(call, dict)] == [
        "first-db",
        "second-db",
        "first-db",
    ]
    assert [call["port"] for call in calls if isinstance(call, dict)] == [
        3307,
        3308,
        3307,
    ]
    assert calls[1::2] == ["close", "close", "close"]
    assert all(
        call["autocommit"] is False
        and call["charset"] == "utf8mb4"
        and (call["connect_timeout"], call["read_timeout"], call["write_timeout"])
        == (10, 20, 20)
        for call in calls
        if isinstance(call, dict)
    )


def test_snapshot_connection_failure_is_sanitized_and_actual_owner_is_lazy():
    from server.reclive.api.dependencies import OwnedActualHourRepository
    from starlette.requests import Request

    app = create_app(Settings.for_test())
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("private connection detail")

    app.state.runtime.connect = fail
    OwnedActualHourRepository(app.state.runtime)
    assert calls == []
    with pytest.raises(HTTPException) as failure:
        next(get_snapshot_repository(Request({"type": "http", "app": app})))
    assert failure.value.status_code == 503
    assert failure.value.detail == "Live occupancy DB is unavailable"
    assert calls == [{"autocommit": False}]


def test_push_route_body_validation_is_owned_by_each_app(monkeypatch):
    base = Settings.for_test()
    first = create_app(
        replace(
            base,
            section_ids={1186: {"first": [1]}},
            push=replace(
                base.push, default_rule_ttl_seconds=10, max_rule_ttl_seconds=20
            ),
        )
    )
    second = create_app(
        replace(
            base,
            section_ids={1186: {"second": [2]}},
            push=replace(
                base.push, default_rule_ttl_seconds=10, max_rule_ttl_seconds=40
            ),
        )
    )
    monkeypatch.setattr(
        push, "rate_limit_public_push_write", lambda request, decoded: None
    )
    monkeypatch.setattr(
        push,
        "subscribe_owned_push_rule",
        lambda **values: {"section": values["section_key"]},
    )
    payload = {
        "subscription": {
            "endpoint": "https://push.reclive-notify.net/push",
            "keys": {"p256dh": "B" + "A" * 86, "auth": "A" * 22},
        },
        "facilityId": 1186,
        "sectionKey": "first",
        "threshold": 20,
        "ttlSeconds": 10,
    }
    a, b = TestClient(first), TestClient(second)
    assert a.post("/api/push/subscribe", json=payload).json() == {"section": "first"}
    assert b.post("/api/push/subscribe", json=payload).status_code == 422
    assert b.post(
        "/api/push/subscribe",
        json={**payload, "sectionKey": "second", "ttlSeconds": 30},
    ).json() == {"section": "second"}
    assert a.post("/api/push/subscribe", json={**payload, "ttlSeconds": 30}).json() == {
        "detail": "invalid_push_request"
    }
    assert a.post("/api/push/subscribe", json=payload).status_code == 200


def test_factory_import_and_test_settings_do_not_initialize_production():
    root = Path(__file__).resolve().parents[2]
    code = """
import os
os.environ['APP_ENV'] = 'production'
os.environ['FORECAST_API_PORT'] = 'invalid-if-parsed'
import server.env_loader as env_loader
def forbidden(*args, **kwargs):
    raise AssertionError('production initialization during factory import')
env_loader.load_project_dotenv = forbidden
from server.reclive.api.app import create_app
from server.reclive.settings import Settings
from server.reclive import sections, runtime
sections.ensure_runtime_facility_configuration = forbidden
settings = Settings.for_test()
app = create_app(settings)
assert app.state.settings is settings
assert runtime.legacy_runtime is None
assert settings.environment == 'test'
print('inert-factory-ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "inert-factory-ok"


def test_direct_api_script_from_another_cwd_keeps_uvicorn_target_and_paths(tmp_path):
    root = Path(__file__).resolve().parents[2]
    code = f"""
import sys, runpy, importlib
sys.path[:0] = [{str(root)!r}, {str(root / "server")!r}]
import env_loader
env_loader._DOTENV_STATE.loaded = True
import uvicorn
calls = []
def run(target, **options):
    calls.append((target, options))
    api = importlib.import_module(target.split(':')[0])
    assert api.FORECAST_JSON_PATH == {str(root / "server/forecast.json")!r}
    from server.reclive.api.app import get_default_app
    assert api.app is get_default_app()
uvicorn.run = run
runpy.run_path({str(root / "server/forecast_api.py")!r}, run_name='__main__')
assert calls == [('forecast_api:app', {{'host': '127.0.0.1', 'port': 8123, 'reload': False}})]
print('direct-api-ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={
            "APP_ENV": "test",
            "FORECAST_API_HOST": "127.0.0.1",
            "FORECAST_API_PORT": "8123",
            "PUSH_EVALUATOR_ENABLED": "false",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "direct-api-ok"


def test_async_and_threadpool_scopes_interleave_and_restore_after_error():
    first = create_app(replace(Settings.for_test(), port=8001))
    second = create_app(replace(Settings.for_test(), port=8002))
    outside = Runtime(replace(Settings.for_test(), port=8999))

    async def run():
        entered = {8001: asyncio.Event(), 8002: asyncio.Event()}

        async def probe():
            marker = current_runtime().settings.port
            entered[marker].set()
            await entered[8002 if marker == 8001 else 8001].wait()
            threaded = await run_in_threadpool(lambda: current_runtime().settings.port)
            return {
                "before": marker,
                "after": current_runtime().settings.port,
                "threaded": threaded,
            }

        async def failure():
            raise LookupError(str(current_runtime().settings.port))

        for app in (first, second):
            app.add_api_route("/scope-probe", probe)
            app.add_api_route("/scope-failure", failure)
        with runtime_scope(outside):
            async with (
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=first), base_url="http://first"
                ) as a,
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=second), base_url="http://second"
                ) as b,
            ):
                ra, rb = await asyncio.wait_for(
                    asyncio.gather(a.get("/scope-probe"), b.get("/scope-probe")), 5
                )
                assert ra.json() == {"before": 8001, "after": 8001, "threaded": 8001}
                assert rb.json() == {"before": 8002, "after": 8002, "threaded": 8002}
                with pytest.raises(LookupError, match="8001"):
                    await a.get("/scope-failure")
                assert current_runtime() is outside

    asyncio.run(run())


def test_single_model_identity_obeys_opposite_section_and_ttl_policies():
    original = Settings.for_test()
    first = create_app(
        replace(
            original,
            section_ids={1186: {"first": [1]}},
            push=replace(
                original.push, default_rule_ttl_seconds=10, max_rule_ttl_seconds=20
            ),
        )
    )
    second = create_app(
        replace(
            original,
            section_ids={1186: {"second": [2]}},
            push=replace(
                original.push, default_rule_ttl_seconds=10, max_rule_ttl_seconds=40
            ),
        )
    )
    payload = {
        "subscription": {
            "endpoint": "https://example.com/push",
            "keys": {"p256dh": "a", "auth": "b"},
        },
        "facilityId": 1186,
        "sectionKey": "first",
        "threshold": 20,
        "ttlSeconds": 30,
    }
    with runtime_scope(first.state.runtime):
        first_policy = push.validation_policy()
        with pytest.raises(ValidationError):
            push.PushRuleRequest.model_validate(payload, context=first_policy)
    with runtime_scope(second.state.runtime):
        second_policy = push.validation_policy()
        model = push.PushRuleRequest.model_validate(
            {**payload, "sectionKey": "second"}, context=second_policy
        )
        assert type(model) is push.PushRuleRequest
        with pytest.raises(ValidationError):
            push.PushRuleRequest.model_validate(
                {**payload, "ttlSeconds": 10}, context=second_policy
            )
        # Explicit context takes priority over the ambient other application.
        assert (
            push.PushRuleRequest.model_validate(
                {**payload, "ttlSeconds": 10}, context=first_policy
            ).section_key
            == "first"
        )
    assert (
        push.PushRuleRequest.model_json_schema()["properties"]["ttlSeconds"]["anyOf"][
            0
        ]["maximum"]
        == 604800
    )


def test_explicit_lifespans_own_tasks_and_capture_runtime(monkeypatch):
    base = Settings.for_test()
    enabled = replace(
        base.push,
        evaluator_enabled=True,
        vapid_public_key="public",
        vapid_private_key="private",
        vapid_subject="mailto:test@example.com",
    )
    first = create_app(replace(base, port=8101, push=enabled))
    second = create_app(replace(base, port=8102, push=enabled))
    observed = []

    async def evaluator():
        observed.append(("start", current_runtime().settings.port))
        try:
            await asyncio.Event().wait()
        finally:
            observed.append(("cancel", current_runtime().settings.port))

    monkeypatch.setattr(push, "evaluator_loop", evaluator)

    async def run():
        async with first.router.lifespan_context(first):
            await asyncio.sleep(0)
            first_task = first.state.runtime.task
            async with second.router.lifespan_context(second):
                await asyncio.sleep(0)
                assert first_task is not second.state.runtime.task
            assert first_task is first.state.runtime.task
            assert first_task.done() is False
        assert first.state.runtime.task is None
        assert second.state.runtime.task is None

    asyncio.run(run())
    assert observed == [
        ("start", 8101),
        ("start", 8102),
        ("cancel", 8102),
        ("cancel", 8101),
    ]


def test_queued_transport_uses_captured_app_resolver_vapid_and_executor():
    base = Settings.for_test()
    first = create_app(
        replace(
            base,
            push=replace(
                base.push,
                vapid_private_key="first",
                vapid_subject="mailto:first@example.com",
            ),
        )
    )
    second = create_app(
        replace(
            base,
            push=replace(
                base.push,
                vapid_private_key="second",
                vapid_subject="mailto:second@example.com",
            ),
        )
    )
    barrier = threading.Barrier(2)
    observed = []
    subscription = {
        "endpoint": "https://push.reclive-notify.net/push",
        "keys": {"p256dh": "B" + "A" * 86, "auth": "A" * 22},
    }

    for app, name, ip in ((first, "first", "8.8.8.8"), (second, "second", "1.1.1.1")):
        runtime = app.state.runtime
        runtime.resolver = lambda host, port, answer=ip: [answer]

        def send(
            *, vapid_private_key, requests_session, expected=name, address=ip, **kwargs
        ):
            barrier.wait(timeout=5)
            observed.append(
                (
                    expected,
                    vapid_private_key,
                    current_runtime().settings.push.vapid_private_key,
                    requests_session.target.connect_ip,
                )
            )
            assert requests_session.target.connect_ip == address
            return SimpleNamespace(status_code=201)

        runtime.webpush = send

    def submit(runtime):
        with runtime_scope(runtime):
            push.send_notification_pinned(subscription, "title", "body", "/nick")

    async def run():
        await asyncio.gather(
            asyncio.to_thread(submit, first.state.runtime),
            asyncio.to_thread(submit, second.state.runtime),
        )

    asyncio.run(run())
    assert sorted(observed) == [
        ("first", "first", "first", "8.8.8.8"),
        ("second", "second", "second", "1.1.1.1"),
    ]
    assert first.state.runtime.executor is not second.state.runtime.executor


def test_explicit_apps_own_artifacts_and_cors(tmp_path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps({"facilities": [], "marker": "first"}))
    second_path.write_text(json.dumps({"facilities": [], "marker": "second"}))
    first = create_app(
        replace(
            Settings.for_test(forecast_json_path=str(first_path)),
            cors_origins=("https://first.example",),
        )
    )
    second = create_app(
        replace(
            Settings.for_test(forecast_json_path=str(second_path)),
            cors_origins=("https://second.example",),
        )
    )
    assert first.state.runtime is not second.state.runtime
    assert first.state.runtime.transport_lock is not second.state.runtime.transport_lock
    for app, marker, origin in (
        (first, "first", "https://first.example"),
        (second, "second", "https://second.example"),
        (first, "first", "https://first.example"),
    ):
        client = TestClient(app)
        response = client.get("/api/forecast", headers={"Origin": origin})
        assert response.json()["marker"] == marker
        assert response.headers["access-control-allow-origin"] == origin


def test_snapshot_dependency_is_canonical_and_closes_its_connection():
    from starlette.requests import Request

    calls = []

    class Connection:
        def get_autocommit(self):
            return False

        def close(self):
            calls.append("close")

    def connect(*, autocommit=True):
        calls.append(("connect", autocommit))
        return Connection()

    app = create_app(Settings.for_test())
    app.state.runtime.connect = connect
    request = Request({"type": "http", "app": app})
    dependency = get_snapshot_repository(request)
    next(dependency)
    assert calls == [("connect", False)]
    dependency.close()
    assert calls == [("connect", False), "close"]
    route = next(route for route in app.routes if route.path == "/api/live-counts")
    assert get_snapshot_repository in {
        item.call for item in route.dependant.dependencies
    }


@pytest.mark.parametrize("raw_environment", ["", " \t "])
def test_captured_api_settings_reject_blank_environment_before_startup_work(
    raw_environment, monkeypatch
):
    from server.reclive import db, sections
    from server.reclive.settings import build_settings_from_environment

    calls = []

    def unexpected_work(*args, **kwargs):
        calls.append("work")
        raise AssertionError("blank environment reached startup work")

    monkeypatch.setattr(
        sections, "ensure_runtime_facility_configuration", unexpected_work
    )
    monkeypatch.setattr(db, "open_db_connection", unexpected_work)
    monkeypatch.setattr(push, "evaluator_loop", unexpected_work)
    settings = build_settings_from_environment({"APP_ENV": raw_environment})
    app = create_app(settings)

    async def run():
        async with app.router.lifespan_context(app):
            pytest.fail("blank environment reached completed startup")

    with pytest.raises(RuntimeError) as error:
        asyncio.run(run())
    assert str(error.value) == "APP_ENV must be development, test, or production"
    assert calls == []
    assert app.state.runtime.task is None
    assert app.state.runtime.configuration_loaded is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, "development"),
        ("development", "development"),
        (" DeVelopMent ", "development"),
        (" TEST ", "test"),
        (" production ", "production"),
    ],
)
def test_api_environment_capture_keeps_missing_default_and_valid_modes(raw, expected):
    from server.reclive.settings import app_environment, build_settings_from_environment

    values = {"SCHEDULE_MAX_AGE_SECONDS": "120", "FORECAST_API_PORT": " 8001 "}
    if raw is not None:
        values["APP_ENV"] = raw
    settings = build_settings_from_environment(values)
    assert settings.environment == expected
    assert settings.schedule_stale_after_seconds == 120
    assert settings.port == 8001
    with runtime_scope(Runtime(settings)):
        assert app_environment() == expected
