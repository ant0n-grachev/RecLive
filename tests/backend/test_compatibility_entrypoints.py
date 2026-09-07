import importlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from dataclasses import replace
from datetime import datetime, timezone

import pytest


@pytest.mark.parametrize("first", ["forecast_job", "server.forecast_job"])
def test_forecasting_imports_share_owners_and_script_paths(first, tmp_path):
    root = Path(__file__).resolve().parents[2]
    code = textwrap.dedent(f"""
        import importlib, sys, os
        sys.path[:0] = [{str(root)!r}, {str(root / 'server')!r}]
        import env_loader
        os.environ.update(env_loader.DEFAULT_ENV)
        env_loader._DOTENV_STATE.loaded = True
        importlib.import_module({first!r})
        import forecast_job
        import server.forecast_job as package
        from server.reclive.forecasting import config, features, job
        assert forecast_job is package
        assert forecast_job.build_forecast is job.build_forecast
        assert forecast_job.main is job.main
        assert forecast_job.configured_direct_horizon_hours is features.configured_direct_horizon_hours
        assert config.SCRIPT_DIR == {str(root / 'server')!r}
        assert config.resolve_path('fixture.json') == {str(root / 'server/fixture.json')!r}
        print('forecast-owner-ok')
    """)
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "forecast-owner-ok\n"


@pytest.mark.parametrize("status, expected", [("succeeded", 0), ("failed", 1)])
def test_gym_fetch_main_delegates_to_configured_ingestion(
    monkeypatch, status, expected
):
    from server import gym_fetch
    from server.reclive.ingestion import IngestionRunResult

    monkeypatch.setattr(gym_fetch, "load_project_dotenv", lambda: None)
    monkeypatch.setenv("APP_ENV", "test")
    calls = []

    def run(settings):
        calls.append(settings)
        return IngestionRunResult(status, 0, 0, 0, 0, None)

    assert hasattr(gym_fetch, "run_configured_ingestion")
    monkeypatch.setattr(gym_fetch, "run_configured_ingestion", run)
    assert gym_fetch.main() == expected
    assert len(calls) == 1
    assert calls[0].environment == "test"


def test_schedule_main_delegates_with_script_relative_output(monkeypatch, capsys):
    import sys
    from pathlib import Path
    from server import facility_hours_fetch as command

    monkeypatch.setattr(command, "load_project_dotenv", lambda: None)
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setattr(
        sys, "argv", ["facility_hours_fetch.py", "--output", "server/result.json"]
    )
    calls = []

    def run(settings):
        calls.append(settings)
        return 1

    assert hasattr(command, "run_facility_hours_fetch")
    monkeypatch.setattr(command, "run_facility_hours_fetch", run)
    monkeypatch.setattr(
        command,
        "load_previous_schedule",
        lambda path: {
            "facilities": [{"status": "ok"}, {"status": "stale"}],
        },
    )
    assert command.main() == 1
    assert len(calls) == 1
    assert calls[0].facility_hours_json_path == str(
        Path(__file__).resolve().parents[2] / "server/result.json"
    )
    assert (
        capsys.readouterr().out
        == "facility_hours_fetch: published ok=1 stale=1 error=0 total=2\n"
    )


def test_collector_and_normalizer_have_service_owners():
    from server import facility_hours_fetch, forecast_shared
    from server.reclive import facility_schedule, sections

    assert (
        facility_hours_fetch.validate_schedule_payload
        is facility_schedule.validate_schedule_payload
    )
    assert (
        facility_schedule.validate_schedule_payload.__module__
        == "server.reclive.facility_schedule"
    )
    assert (
        facility_hours_fetch.collect_facility_candidate
        is facility_schedule.collect_facility_candidate
    )
    assert forecast_shared.normalize_section_key is sections.normalize_section_key
    assert sections.normalize_section_key.__module__ == "server.reclive.sections"
    assert forecast_shared.normalize_section_key(" Fitness_Floors ") == "fitness_floors"
    assert (
        forecast_shared.normalize_section_key(" Fitness   Floors ") == "fitness floors"
    )


def test_legacy_backend_entrypoints_export_their_existing_callables() -> None:
    assert callable(importlib.import_module("server.gym_fetch").main)
    assert callable(importlib.import_module("server.forecast_api").app)
    assert callable(importlib.import_module("server.forecast_job").build_forecast)
    assert callable(importlib.import_module("server.facility_hours_fetch").main)


def test_bare_backend_entrypoints_remain_importable() -> None:
    assert callable(importlib.import_module("gym_fetch").main)
    assert callable(importlib.import_module("forecast_api").app)
    assert callable(importlib.import_module("forecast_job").build_forecast)
    assert callable(importlib.import_module("facility_hours_fetch").main)


@pytest.mark.parametrize("first", ["bare", "package"])
def test_command_import_orders_share_callables_without_api_runtime(first, tmp_path):
    root = Path(__file__).resolve().parents[2]
    code = textwrap.dedent(f"""
        import importlib
        import sys
        sys.path[:0] = [{str(root)!r}, {str(root / "server")!r}]
        names = ['gym_fetch', 'facility_hours_fetch', 'forecast_shared']
        prefixes = ['', 'server.'] if {first!r} == 'bare' else ['server.', '']
        for prefix in prefixes:
            for name in names:
                importlib.import_module(prefix + name)
        import server.gym_fetch as gym
        import server.facility_hours_fetch as hours
        import server.forecast_shared as shared
        import server
        from server.reclive import ingestion, facility_schedule, sections, runtime
        for name in names:
            assert sys.modules[name] is sys.modules['server.' + name]
            assert getattr(server, name) is sys.modules[name]
        assert gym.run_configured_ingestion is ingestion.run_configured_ingestion
        assert gym.run_ingestion is ingestion.run_ingestion
        assert hours.run_facility_hours_fetch is facility_schedule.run_facility_hours_fetch
        assert hours.FacilityCandidate is facility_schedule.FacilityCandidate
        assert hours.validate_schedule_payload is facility_schedule.validate_schedule_payload
        assert shared.normalize_section_key is sections.normalize_section_key
        assert runtime.legacy_runtime is None
        assert 'server.reclive.api.app' not in sys.modules
        assert 'forecast_job' not in sys.modules
        print('identity-ok')
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "identity-ok\n"
    assert result.stderr == ""


@pytest.mark.parametrize("command", ["gym_fetch", "facility_hours_fetch"])
def test_direct_collectors_use_configured_service_from_another_cwd(command, tmp_path):
    root = Path(__file__).resolve().parents[2]
    code = textwrap.dedent(f"""
        import runpy
        import sys
        sys.path.insert(0, {str(root / "server")!r})
        import reclive
        import env_loader
        env_loader._DOTENV_STATE.loaded = True
        from server.reclive import ingestion, facility_schedule, runtime
        calls = []
        def ingest(settings):
            assert settings.environment == 'test'
            calls.append('called')
            return ingestion.IngestionRunResult('succeeded', 0, 0, 0, 0, None)
        def schedule(settings):
            assert settings.facility_hours_json_path == {str(root / "server/result.json")!r}
            calls.append('called')
            return 1
        ingestion.run_configured_ingestion = ingest
        facility_schedule.run_facility_hours_fetch = schedule
        facility_schedule.load_previous_schedule = lambda path: {{'facilities': [{{'status': 'ok'}}, {{'status': 'stale'}}]}}
        sys.argv = [{command!r}]
        if {command!r} == 'facility_hours_fetch':
            sys.argv += ['--output', 'server/result.json']
        try:
            runpy.run_path({str(root / "server" / (command + ".py"))!r}, run_name='__main__')
        except SystemExit as error:
            assert error.code == (1 if {command!r} == 'facility_hours_fetch' else 0)
        else:
            raise AssertionError('missing CLI exit')
        assert calls == ['called']
        assert runtime.legacy_runtime is None
        assert 'server.reclive.api.app' not in sys.modules
        print('direct-ok')
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "APP_ENV": "test"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    expected = (
        "facility_hours_fetch: published ok=1 stale=1 error=0 total=2\n"
        if command == "facility_hours_fetch"
        else ""
    )
    assert result.stdout == expected + "direct-ok\n"
    assert result.stderr == ""


def test_command_settings_capture_inputs_without_parsing_api_options(
    monkeypatch, tmp_path
):
    from server.reclive.settings import Settings

    values = {
        "APP_ENV": "test",
        "FORECAST_API_PORT": "invalid-api-option",
        "PUSH_EVALUATOR_ENABLED": "invalid-api-option",
        "SCHEDULE_STALE_AFTER_SECONDS": "invalid-api-option",
        "LIVE_COUNTS_URL": "https://captured.invalid/live",
        "RECWELL_SITE_BASE": "https://captured.invalid",
        "FACILITY_HOURS_JSON_PATH": "server/captured.json",
        "FACILITY_CAPACITIES_JSON_PATH": str(tmp_path / "capacities.json"),
    }
    settings = Settings.for_commands(values)
    values["LIVE_COUNTS_URL"] = "https://changed.invalid/live"
    monkeypatch.setenv("LIVE_COUNTS_URL", "https://ambient.invalid/live")
    assert settings.live_counts_url == "https://captured.invalid/live"
    assert settings.recwell_site_base == "https://captured.invalid"
    assert settings.capacity_config_path == str(tmp_path / "capacities.json")
    assert settings.facility_hours_json_path.endswith("/server/captured.json")
    with pytest.raises(TypeError):
        settings.environment_values["APP_ENV"] = "production"


@pytest.mark.parametrize("fails", [False, True])
def test_configured_ingestion_retains_transactions_and_optional_seams(fails):
    from server.reclive import ingestion
    from server.reclive.settings import Settings
    from tests.fixtures.live_counts import LIVE_ROWS
    from tests.fixtures.reclive_fakes import LifecycleRepositoryFactory

    factory = LifecycleRepositoryFactory()
    settings = replace(Settings.for_test(), capacities={5761: 100})

    def fetch():
        if fails:
            raise RuntimeError("private-source-marker")
        return LIVE_ROWS

    result = ingestion.run_configured_ingestion(
        settings,
        fetch_payload=fetch,
        connect=factory.connect,
        now=lambda: datetime(2026, 8, 31, 12, tzinfo=timezone.utc),
        repository_factory=factory,
        event_sink=factory.event_lines.append,
    )
    assert result.status == ("failed" if fails else "succeeded")
    assert factory.run_events == ["start", "commit", "close"]
    assert factory.work_events == (
        ["rollback", "close"]
        if fails
        else ["persist", "complete_success", "commit", "close"]
    )
    assert len(factory.connections) == (3 if fails else 2)
    assert factory.all_connections_closed
    assert len(factory.event_lines) == 1
    assert "private-source-marker" not in factory.event_lines[0]
    if fails:
        assert factory.failure_events == ["complete_failure", "commit", "close"]


@pytest.mark.parametrize("field", ["host", "user", "password", "name", "port"])
def test_configured_ingestion_validates_effective_production_before_io(
    field, monkeypatch
):
    from server.reclive import ingestion
    from server.reclive.settings import Settings

    settings = replace(
        Settings.for_test(),
        environment="production",
        live_counts_url="https://fixture.invalid/live",
        capacities=None,
    )
    settings = replace(settings, database=replace(settings.database, **{field: None}))
    calls = []
    monkeypatch.setattr(
        ingestion, "load_facility_capacities", lambda *args: calls.append("capacity")
    )
    with pytest.raises(RuntimeError, match="Unsafe production configuration: GYM_DB_"):
        ingestion.run_configured_ingestion(settings)
    assert calls == []


def test_configured_schedule_uses_captured_urls_output_and_partial_result(
    tmp_path, monkeypatch
):
    from server.reclive import facility_schedule as schedule
    from server.reclive.settings import Settings

    target = tmp_path / "hours.json"
    settings = replace(
        Settings.for_test(),
        facility_hours_json_path=str(target),
        recwell_site_base="https://captured.invalid",
        recwell_nick_url="https://captured.invalid/nick/",
        recwell_bakke_url="https://captured.invalid/bakke/",
    )
    urls = []

    def fail_direct(url):
        urls.append(url)
        raise schedule.ScheduleFetchError("upstream_http")

    def fail_wp(site_base, slug):
        assert site_base == "https://captured.invalid"
        assert slug in {"nick", "bakke"}
        raise schedule.ScheduleFetchError("wp_payload_invalid")

    monkeypatch.setenv("RECWELL_SITE_BASE", "https://ambient.invalid")
    monkeypatch.setenv("RECWELL_NICK_URL", "https://ambient.invalid/nick/")
    monkeypatch.setattr(schedule, "fetch_direct_html", fail_direct)
    monkeypatch.setattr(schedule, "fetch_wp_json_html", fail_wp)
    assert schedule.run_facility_hours_fetch(settings) == 1
    payload = schedule.load_previous_schedule(str(target))
    assert payload is not None
    assert urls == ["https://captured.invalid/nick/", "https://captured.invalid/bakke/"]
    assert payload["sourceSite"] == "https://captured.invalid"
    assert payload["okCount"] == 0
    assert [row["status"] for row in payload["facilities"]] == ["error", "error"]


def test_configured_ingestion_resolves_captured_capacity_and_connection_options(
    tmp_path, monkeypatch
):
    from server.reclive import ingestion
    from server.reclive.settings import Settings
    from tests.fixtures.live_counts import LIVE_ROWS
    from tests.fixtures.reclive_fakes import LifecycleRepositoryFactory

    capacity_path = tmp_path / "capacity.json"
    capacity_path.write_text('{"5761": 100}', encoding="utf-8")
    settings = replace(
        Settings.for_test(),
        capacities=None,
        capacity_config_path=str(capacity_path),
        live_counts_url="https://captured.invalid/live",
    )
    factory = LifecycleRepositoryFactory()
    connects = []
    requests = []

    class Response:
        def raise_for_status(self):
            requests.append("raise")

        def json(self):
            requests.append("json")
            return LIVE_ROWS

    def get(url, *, timeout):
        requests.append((url, timeout))
        return Response()

    def connect(**options):
        connects.append(options)
        return factory.connect()

    monkeypatch.setenv("FACILITY_CAPACITIES_JSON_PATH", str(tmp_path / "absent.json"))
    monkeypatch.setenv("LIVE_COUNTS_URL", "https://ambient.invalid/live")
    monkeypatch.setenv("GYM_DB_HOST", "ambient.invalid")
    monkeypatch.setattr(ingestion.requests, "get", get)
    monkeypatch.setattr(ingestion.pymysql, "connect", connect)
    result = ingestion.run_configured_ingestion(
        settings, repository_factory=factory, event_sink=factory.event_lines.append
    )
    assert result.status == "succeeded"
    assert requests == [("https://captured.invalid/live", (5, 20)), "raise", "json"]
    assert len(connects) == 2
    assert (
        connects
        == [
            {
                "host": "127.0.0.1",
                "port": 3306,
                "user": "reclive",
                "password": "reclive-ci-password",
                "database": "reclive_test",
                "autocommit": False,
                "charset": "utf8mb4",
                "connect_timeout": 10,
                "read_timeout": 20,
                "write_timeout": 20,
            }
        ]
        * 2
    )
    assert factory.snapshot_writes[0][0].max_capacity == 100


def test_configured_fetch_preserves_explicit_legacy_url_override(monkeypatch):
    from server.reclive import ingestion
    from server.reclive.settings import Settings

    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return []

    def get(url, *, timeout):
        calls.append((url, timeout))
        return Response()

    monkeypatch.setattr(ingestion, "LIVE_COUNTS_URL", "https://override.invalid/live")
    monkeypatch.setattr(ingestion.requests, "get", get)
    assert (
        ingestion.fetch_live(
            replace(
                Settings.for_test(), live_counts_url="https://captured.invalid/live"
            )
        )
        == []
    )
    assert calls == [("https://override.invalid/live", (5, 20))]


def test_configured_db_requires_missing_fields_in_development(monkeypatch):
    from server.reclive import ingestion
    from server.reclive.settings import Settings

    settings = Settings.for_commands({"APP_ENV": "development", "GYM_DB_PORT": "3306"})
    calls = []
    monkeypatch.setattr(
        ingestion.pymysql, "connect", lambda **kwargs: calls.append(kwargs)
    )
    with pytest.raises(RuntimeError, match="Missing required env var: GYM_DB_HOST"):
        ingestion.db_connect(settings)
    assert calls == []


@pytest.mark.parametrize("raw_environment", ["", " \t "])
def test_initialized_dotenv_gym_command_rejects_blank_environment_before_io(
    raw_environment, monkeypatch, capsys
):
    from server import env_loader, gym_fetch
    from server.reclive import ingestion

    calls = []

    def unexpected_io(*args, **kwargs):
        calls.append("io")
        raise AssertionError("blank environment reached ingestion work")

    monkeypatch.setattr(env_loader._DOTENV_STATE, "loaded", True)
    monkeypatch.setenv("APP_ENV", raw_environment)
    monkeypatch.setattr(ingestion, "load_facility_capacities", unexpected_io)
    monkeypatch.setattr(ingestion, "db_connect", unexpected_io)
    monkeypatch.setattr(ingestion, "fetch_live", unexpected_io)

    assert gym_fetch.main() == 1
    assert calls == []
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        "status=failed received=0 valid=0 history=0 snapshot=0 "
        "durationMs=0 category=validation\n"
    )


@pytest.mark.parametrize("raw_environment", ["", " \t "])
@pytest.mark.parametrize("service_name", ["ingestion", "schedule"])
def test_captured_command_settings_reject_blank_environment_before_io(
    raw_environment, service_name, monkeypatch
):
    from server.reclive import facility_schedule, ingestion
    from server.reclive.settings import Settings

    calls = []

    def unexpected_io(*args, **kwargs):
        calls.append("io")
        raise AssertionError("blank environment reached collector work")

    settings = Settings.for_commands({"APP_ENV": raw_environment})
    monkeypatch.setattr(ingestion, "load_facility_capacities", unexpected_io)
    monkeypatch.setattr(ingestion, "db_connect", unexpected_io)
    monkeypatch.setattr(ingestion, "fetch_live", unexpected_io)
    monkeypatch.setattr(facility_schedule, "load_previous_schedule", unexpected_io)
    monkeypatch.setattr(facility_schedule, "collect_facility_candidate", unexpected_io)
    monkeypatch.setattr(facility_schedule, "atomic_write_json", unexpected_io)
    run = (
        ingestion.run_configured_ingestion
        if service_name == "ingestion"
        else facility_schedule.run_facility_hours_fetch
    )
    with pytest.raises(RuntimeError) as error:
        run(settings)
    assert str(error.value) == "Unsafe environment configuration: APP_ENV"
    assert calls == []


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
def test_command_environment_capture_keeps_missing_default_and_valid_modes(
    raw, expected
):
    from server.reclive.settings import Settings, validate_command_environment

    values = {} if raw is None else {"APP_ENV": raw}
    settings = Settings.for_commands(values)
    assert settings.environment == expected
    validate_command_environment(settings, ())
