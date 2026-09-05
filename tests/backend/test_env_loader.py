from __future__ import annotations
from server.reclive.api import lifespan_compat as _seam_api_lifespan_compat
from server.reclive import runtime as _seam_runtime
from server.reclive import sections as _seam_sections
from server.reclive import settings as _seam_settings


import asyncio
import importlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
import traceback
from types import SimpleNamespace

import pytest

from server import env_loader
from server.env_loader import validate_production_environment


@pytest.fixture(autouse=True)
def reset_project_dotenv_loader_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_loader._DOTENV_STATE, "loaded", False)


def safe_values() -> dict[str, str]:
    return {
        "APP_ENV": "production",
        "LIVE_COUNTS_URL": "https://upstream.invalid/live",
        "FORECAST_API_ALLOW_ORIGINS": "https://reclive.example",
        "PUSH_ADMIN_TOKEN": "a" * 32,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LIVE_COUNTS_URL", "change_me"),
        ("LIVE_COUNTS_URL", "YOUR_ACCOUNT_API_KEY"),
    ],
)
def test_production_rejects_placeholder_without_echoing_value(
    name: str,
    value: str,
) -> None:
    values = safe_values()
    values[name] = value

    with pytest.raises(RuntimeError) as error:
        validate_production_environment(
            values,
            required_names=("LIVE_COUNTS_URL",),
            cors_name="FORECAST_API_ALLOW_ORIGINS",
            admin_enabled=True,
        )

    assert name in str(error.value)
    assert value not in str(error.value)


def test_production_rejects_missing_required_value_by_name_only() -> None:
    values = safe_values()
    values.pop("LIVE_COUNTS_URL")

    with pytest.raises(RuntimeError) as error:
        validate_production_environment(
            values,
            required_names=("LIVE_COUNTS_URL",),
            cors_name=None,
            admin_enabled=False,
        )

    assert str(error.value) == "Unsafe production configuration: LIVE_COUNTS_URL"


@pytest.mark.parametrize("origins", ["", "*", "https://site.invalid, *"])
def test_production_rejects_missing_or_wildcard_cors(origins: str) -> None:
    values = safe_values() | {"FORECAST_API_ALLOW_ORIGINS": origins}

    with pytest.raises(RuntimeError, match="FORECAST_API_ALLOW_ORIGINS") as error:
        validate_production_environment(
            values,
            required_names=("LIVE_COUNTS_URL",),
            cors_name="FORECAST_API_ALLOW_ORIGINS",
            admin_enabled=True,
        )

    if origins:
        assert origins not in str(error.value)


def test_production_rejects_short_admin_token_when_routes_are_enabled() -> None:
    token = "short"
    values = safe_values() | {"PUSH_ADMIN_TOKEN": token}

    with pytest.raises(RuntimeError, match="PUSH_ADMIN_TOKEN") as error:
        validate_production_environment(
            values,
            required_names=("LIVE_COUNTS_URL",),
            cors_name="FORECAST_API_ALLOW_ORIGINS",
            admin_enabled=True,
        )

    assert token not in str(error.value)


@pytest.mark.parametrize("environment", ["development", "test"])
def test_nonproduction_environment_keeps_explicit_development_defaults(
    environment: str,
) -> None:
    validate_production_environment(
        {"APP_ENV": environment},
        required_names=("LIVE_COUNTS_URL",),
        cors_name="FORECAST_API_ALLOW_ORIGINS",
        admin_enabled=True,
    )


def test_unknown_environment_is_rejected_without_echoing_its_value() -> None:
    environment = "private-environment-marker"

    with pytest.raises(RuntimeError) as error:
        validate_production_environment(
            {"APP_ENV": environment},
            required_names=("LIVE_COUNTS_URL",),
            cors_name=None,
            admin_enabled=False,
        )

    assert str(error.value) == "Unsafe environment configuration: APP_ENV"
    assert environment not in str(error.value)


@pytest.mark.parametrize(
    "port",
    ["private-port-marker", "0", "65536", "3306.0"],
)
def test_production_rejects_invalid_database_port_by_name_only(port: str) -> None:
    values = safe_values() | {"GYM_DB_PORT": port}

    with pytest.raises(RuntimeError) as error:
        validate_production_environment(
            values,
            required_names=("GYM_DB_PORT",),
            cors_name=None,
            admin_enabled=False,
        )

    assert str(error.value) == "Unsafe production configuration: GYM_DB_PORT"
    assert port not in str(error.value)
    assert port not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize("port", ["1", "65535"])
def test_production_accepts_database_port_boundaries(port: str) -> None:
    validate_production_environment(
        safe_values() | {"GYM_DB_PORT": port},
        required_names=("GYM_DB_PORT",),
        cors_name=None,
        admin_enabled=False,
    )


def test_project_dotenv_loader_reads_the_first_existing_file_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values: dict[str, str] = {"APP_ENV": "development"}
    loaded_paths: list[str] = []
    monkeypatch.setattr(env_loader.os, "environ", values)
    monkeypatch.setattr(env_loader.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(
        env_loader,
        "load_dotenv",
        lambda path, *, override: loaded_paths.append(path),
    )

    env_loader.load_project_dotenv()
    env_loader.load_project_dotenv()

    assert loaded_paths == [env_loader.os.path.join(env_loader.SCRIPT_DIR, ".env")]


def test_project_dotenv_loader_retries_after_a_failed_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values: dict[str, str] = {"APP_ENV": "test"}
    load_attempts: list[str] = []
    monkeypatch.setattr(env_loader.os, "environ", values)
    monkeypatch.setattr(env_loader.os.path, "exists", lambda _path: True)

    def fail_once(path: str, *, override: bool) -> None:
        assert override is False
        load_attempts.append(path)
        if len(load_attempts) == 1:
            raise OSError("fixture load failure")

    monkeypatch.setattr(env_loader, "load_dotenv", fail_once)

    with pytest.raises(OSError, match="fixture load failure"):
        env_loader.load_project_dotenv()
    assert env_loader._DOTENV_STATE.loaded is False

    env_loader.load_project_dotenv()

    expected_path = env_loader.os.path.join(env_loader.SCRIPT_DIR, ".env")
    assert load_attempts == [expected_path, expected_path]
    assert env_loader._DOTENV_STATE.loaded is True


def test_project_dotenv_loader_detects_production_from_dotenv_before_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values: dict[str, str] = {}
    loaded_paths: list[str] = []
    monkeypatch.setattr(env_loader.os, "environ", values)
    monkeypatch.setattr(env_loader.os.path, "exists", lambda _path: True)

    def load_production(path: str, *, override: bool) -> None:
        assert override is False
        loaded_paths.append(path)
        values["APP_ENV"] = "production"

    monkeypatch.setattr(env_loader, "load_dotenv", load_production)

    env_loader.load_project_dotenv()

    assert loaded_paths == [env_loader.os.path.join(env_loader.SCRIPT_DIR, ".env")]
    assert values == {"APP_ENV": "production"}


def test_project_dotenv_loader_rejects_unknown_environment_before_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = "private-environment-marker"
    values: dict[str, str] = {"APP_ENV": environment}
    path_checks: list[str] = []
    monkeypatch.setattr(env_loader.os, "environ", values)

    def record_path_check(path: str) -> bool:
        path_checks.append(path)
        return False

    monkeypatch.setattr(env_loader.os.path, "exists", record_path_check)

    with pytest.raises(RuntimeError) as error:
        env_loader.load_project_dotenv()

    assert str(error.value) == "Unsafe environment configuration: APP_ENV"
    assert environment not in str(error.value)
    assert values == {"APP_ENV": environment}
    assert path_checks == []


def test_project_dotenv_state_is_shared_across_supported_import_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    top_level_env_loader = importlib.import_module("env_loader")
    modules = {id(module): module for module in (env_loader, top_level_env_loader)}
    states = {
        id(module._DOTENV_STATE): module._DOTENV_STATE
        for module in modules.values()
    }
    values: dict[str, str] = {"APP_ENV": "test"}
    loaded_paths: list[str] = []
    monkeypatch.setattr(env_loader.os, "environ", values)
    monkeypatch.setattr(env_loader.os.path, "exists", lambda _path: True)

    for state in states.values():
        monkeypatch.setattr(state, "loaded", False)
        monkeypatch.setattr(state, "lock", threading.Lock())

    for module in modules.values():
        monkeypatch.setattr(
            module,
            "load_dotenv",
            lambda path, *, override: loaded_paths.append(path),
        )

    env_loader.load_project_dotenv()
    top_level_env_loader.load_project_dotenv()

    assert loaded_paths == [env_loader.os.path.join(env_loader.SCRIPT_DIR, ".env")]


def test_concurrent_project_dotenv_calls_load_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values: dict[str, str] = {"APP_ENV": "test"}
    state_guard = threading.Lock()
    first_load_started = threading.Event()
    second_attempted = threading.Event()
    loaded_paths: list[str] = []
    errors: list[BaseException] = []

    monkeypatch.setattr(env_loader.os, "environ", values)
    monkeypatch.setattr(env_loader.os.path, "exists", lambda _path: True)

    class CoordinatedLock:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._attempt_guard = threading.Lock()
            self._attempts = 0

        def __enter__(self) -> CoordinatedLock:
            with self._attempt_guard:
                self._attempts += 1
                if self._attempts == 2:
                    second_attempted.set()
            self._lock.acquire()
            return self

        def __exit__(self, *_args: object) -> None:
            self._lock.release()

    def coordinated_load(path: str, *, override: bool) -> None:
        assert override is False
        with state_guard:
            loaded_paths.append(path)
            call_number = len(loaded_paths)
        if call_number == 1:
            first_load_started.set()
            if not second_attempted.wait(timeout=5):
                raise AssertionError("second dotenv call did not reach synchronization")
        else:
            second_attempted.set()

    monkeypatch.setattr(env_loader._DOTENV_STATE, "lock", CoordinatedLock())
    monkeypatch.setattr(env_loader, "load_dotenv", coordinated_load)

    def run_loader() -> None:
        try:
            env_loader.load_project_dotenv()
        except BaseException as exc:
            with state_guard:
                errors.append(exc)

    first = threading.Thread(target=run_loader, name="dotenv-first")
    second = threading.Thread(target=run_loader, name="dotenv-second")
    first.start()
    assert first_load_started.wait(timeout=5)
    second.start()
    first.join(timeout=10)
    second.join(timeout=10)

    assert first.is_alive() is False
    assert second.is_alive() is False
    assert errors == []
    assert loaded_paths == [env_loader.os.path.join(env_loader.SCRIPT_DIR, ".env")]


def test_direct_forecast_api_bootstrap_loads_dotenv_once(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("APP_ENV=test\n", encoding="utf-8")
    smoke = textwrap.dedent(
        f"""
        import importlib
        import os
        import runpy
        import sys
        import uvicorn

        sys.path.insert(0, {str(root / "server")!r})
        os.environ["APP_ENV"] = "test"

        import env_loader

        env_loader.SCRIPT_DIR = {str(tmp_path)!r}
        env_loader.PROJECT_ROOT = {str(tmp_path)!r}
        env_loader._DOTENV_STATE.loaded = False
        loaded_paths = []
        env_loader.load_dotenv = lambda path, *, override: loaded_paths.append(path)

        def import_forecast_api(app_path, **_kwargs):
            importlib.import_module(app_path.split(":", 1)[0])

        uvicorn.run = import_forecast_api
        runpy.run_path({str(root / "server" / "forecast_api.py")!r}, run_name="__main__")

        if loaded_paths != [{str(dotenv_path)!r}]:
            raise AssertionError(f"dotenv loads: {{loaded_paths!r}}")
        print("dotenv-once-ok")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", smoke],
        cwd=root,
        env={**os.environ, "APP_ENV": "test"},
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0
    assert completed.stdout == "dotenv-once-ok\n"
    assert completed.stderr == ""


def test_gym_fetch_validates_before_loading_capacity_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gym_fetch

    data_calls: list[str] = []
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("LIVE_COUNTS_URL", raising=False)
    monkeypatch.setattr(gym_fetch, "load_project_dotenv", lambda: None)
    monkeypatch.setattr(
        gym_fetch,
        "load_facility_capacities",
        lambda: data_calls.append("capacities") or {},
    )
    monkeypatch.setattr(
        gym_fetch,
        "run_ingestion",
        lambda *_args: data_calls.append("ingestion")
        or SimpleNamespace(status="succeeded"),
    )
    monkeypatch.setattr(gym_fetch, "finish_ingestion_result", lambda *_args: None)

    assert gym_fetch.main() == 1
    assert data_calls == []


def test_gym_fetch_rejects_unknown_environment_before_loading_capacity_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gym_fetch

    data_calls: list[str] = []
    monkeypatch.setenv("APP_ENV", "private-environment-marker")
    monkeypatch.setattr(gym_fetch, "load_project_dotenv", lambda: None)
    monkeypatch.setattr(
        gym_fetch,
        "load_facility_capacities",
        lambda: data_calls.append("capacities") or {},
    )
    monkeypatch.setattr(gym_fetch, "finish_ingestion_result", lambda *_args: None)

    assert gym_fetch.main() == 1
    assert data_calls == []


def test_forecast_job_validates_before_building_or_writing_forecast(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import forecast_job

    data_calls: list[str] = []
    for name in (
        "GYM_DB_HOST",
        "GYM_DB_PORT",
        "GYM_DB_USER",
        "GYM_DB_PASSWORD",
        "GYM_DB_NAME",
        "MODEL_ARTIFACT_DIR",
        "FORECAST_JSON_PATH",
    ):
        monkeypatch.setenv(name, f"safe-{name.lower()}")
    monkeypatch.setenv("GYM_DB_PORT", "3306")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("MODEL_BASENAME", raising=False)
    monkeypatch.setattr(
        forecast_job,
        "build_forecast",
        lambda: data_calls.append("build") or {"facilities": [], "modelInfo": {}},
    )
    monkeypatch.setattr(
        forecast_job,
        "write_forecast",
        lambda _payload: data_calls.append("write"),
    )

    assert forecast_job.main() == 1
    assert data_calls == []
    output = capsys.readouterr().out
    assert "MODEL_BASENAME" in output
    assert "safe-" not in output


def test_forecast_job_rejects_invalid_database_port_without_value_or_trace(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import forecast_job

    port = "private-port-marker"
    data_calls: list[str] = []
    production_values = {
        "APP_ENV": "production",
        "GYM_DB_HOST": "db.reclive.example",
        "GYM_DB_PORT": port,
        "GYM_DB_USER": "reclive",
        "GYM_DB_PASSWORD": "private-password-marker",
        "GYM_DB_NAME": "reclive",
        "MODEL_ARTIFACT_DIR": "model_artifacts",
        "MODEL_BASENAME": "forecast_model",
        "FORECAST_JSON_PATH": "forecast.json",
    }
    for name, value in production_values.items():
        monkeypatch.setenv(name, value)

    def build_with_port() -> int:
        data_calls.append("build")
        return forecast_job.require_int_env("GYM_DB_PORT")

    monkeypatch.setattr(
        forecast_job,
        "build_forecast",
        build_with_port,
    )
    monkeypatch.setattr(
        forecast_job,
        "write_forecast",
        lambda _payload: data_calls.append("write"),
    )

    assert forecast_job.main() == 1
    assert data_calls == []
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "GYM_DB_PORT" in output
    assert port not in output
    assert "private-password-marker" not in output
    assert "Traceback" not in output
    assert captured.err == ""


def test_forecast_job_integer_environment_error_is_name_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import forecast_job

    value = "private-integer-marker"
    monkeypatch.setenv("GYM_DB_PORT", value)

    with pytest.raises(RuntimeError) as error:
        forecast_job.require_int_env("GYM_DB_PORT")

    assert str(error.value) == "Invalid integer for env var: GYM_DB_PORT"
    assert value not in "".join(traceback.format_exception(error.value))


def test_forecast_job_uses_fixed_safe_output_for_unexpected_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import forecast_job

    production_values = {
        "APP_ENV": "production",
        "GYM_DB_HOST": "db.reclive.example",
        "GYM_DB_PORT": "3306",
        "GYM_DB_USER": "reclive",
        "GYM_DB_PASSWORD": "private-password-marker",
        "GYM_DB_NAME": "reclive",
        "MODEL_ARTIFACT_DIR": "model_artifacts",
        "MODEL_BASENAME": "forecast_model",
        "FORECAST_JSON_PATH": "forecast.json",
    }
    for name, value in production_values.items():
        monkeypatch.setenv(name, value)

    def fail_forecast() -> dict[str, object]:
        raise RuntimeError("private-upstream-marker")

    monkeypatch.setattr(forecast_job, "build_forecast", fail_forecast)

    assert forecast_job.main() == 1
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "Forecast generation failed" in output
    assert "private-upstream-marker" not in output
    assert "private-password-marker" not in output
    assert "Traceback" not in output
    assert captured.err == ""


def test_direct_forecast_job_unknown_environment_is_name_only_without_trace() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = "private-environment-marker"

    completed = subprocess.run(
        [sys.executable, str(root / "server" / "forecast_job.py")],
        cwd=root,
        env={**os.environ, "APP_ENV": environment},
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 1
    assert completed.stdout == (
        "forecast_job: ERROR: Unsafe environment configuration: APP_ENV\n"
    )
    assert completed.stderr == ""
    assert environment not in completed.stdout + completed.stderr
    assert "Traceback" not in completed.stdout + completed.stderr


def test_facility_hours_fetch_validates_before_collecting_or_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "bs4",
        SimpleNamespace(BeautifulSoup=object),
    )
    facility_hours_fetch = importlib.import_module("facility_hours_fetch")

    data_calls: list[str] = []
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("FACILITY_HOURS_JSON_PATH", raising=False)
    monkeypatch.setattr(facility_hours_fetch, "load_project_dotenv", lambda: None, raising=False)
    monkeypatch.setattr(sys, "argv", ["facility_hours_fetch.py"])
    monkeypatch.setattr(
        facility_hours_fetch,
        "collect_facility_hours",
        lambda **_kwargs: data_calls.append("collect") or {"status": "ok"},
    )
    monkeypatch.setattr(
        facility_hours_fetch,
        "write_json",
        lambda *_args: data_calls.append("write"),
    )

    with pytest.raises(RuntimeError, match="FACILITY_HOURS_JSON_PATH") as error:
        facility_hours_fetch.main()

    assert data_calls == []
    assert str(error.value) == (
        "Unsafe production configuration: FACILITY_HOURS_JSON_PATH"
    )


def test_forecast_api_validates_runtime_before_starting_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import forecast_api

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", "k" * 32)
    monkeypatch.setenv("PUSH_ADMIN_ROUTES_ENABLED", "false")
    monkeypatch.delenv("GYM_DB_HOST", raising=False)
    monkeypatch.setattr(
        _seam_settings,
        "evaluator_enabled",
        lambda: (_ for _ in ()).throw(
            AssertionError("evaluator-started-before-runtime-validation")
        ),
    )

    async def run() -> None:
        async with forecast_api.lifespan(forecast_api.app):
            pass

    with pytest.raises(RuntimeError, match="GYM_DB_HOST") as error:
        asyncio.run(run())

    assert "evaluator-started" not in str(error.value)


def test_forecast_api_production_import_defers_config_reads_until_validation() -> None:
    root = Path(__file__).resolve().parents[2]
    smoke = textwrap.dedent(
        f"""
        import asyncio
        import os
        import sys

        sys.path.insert(0, {str(root / "server")!r})

        import env_loader
        import facility_capacities

        env_loader.load_project_dotenv = lambda: None
        reads = []

        def capacity_read():
            reads.append("capacity")
            raise AssertionError("capacity-read-before-validation")

        facility_capacities.load_facility_capacities = capacity_read

        os.environ["APP_ENV"] = "production"
        os.environ["PUSH_ENDPOINT_HASH_KEY"] = "k" * 32
        os.environ["PUSH_ADMIN_ROUTES_ENABLED"] = "false"
        os.environ.pop("GYM_DB_HOST", None)

        import forecast_api as api

        from server.reclive import sections
        sections.load_facility_capacities = capacity_read

        def section_read():
            reads.append("sections")
            raise AssertionError("section-read-before-validation")

        sections.load_facility_sections = section_read

        async def run():
            async with api.lifespan(api.app):
                pass

        try:
            asyncio.run(run())
        except RuntimeError as error:
            if str(error) != "Unsafe production configuration: GYM_DB_HOST":
                raise
        else:
            raise AssertionError("unsafe production configuration was accepted")

        if reads:
            raise AssertionError(f"configuration reads occurred: {{reads!r}}")
        print("import-order-ok")
        """
    )
    environment = {
        **os.environ,
        "APP_ENV": "production",
        "PUSH_ENDPOINT_HASH_KEY": "k" * 32,
        "PUSH_ADMIN_ROUTES_ENABLED": "false",
    }
    environment.pop("GYM_DB_HOST", None)

    completed = subprocess.run(
        [sys.executable, "-c", smoke],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0
    assert completed.stdout == "import-order-ok\n"
    assert completed.stderr == ""


def test_concurrent_production_lifespans_load_facility_configuration_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import forecast_api

    production_values = {
        "APP_ENV": "production",
        "PUSH_ENDPOINT_HASH_KEY": "k" * 32,
        "PUSH_ADMIN_ROUTES_ENABLED": "false",
        "GYM_DB_HOST": "db.reclive.example",
        "GYM_DB_PORT": "3306",
        "GYM_DB_USER": "reclive",
        "GYM_DB_PASSWORD": "safe-test-password",
        "GYM_DB_NAME": "reclive",
        "FORECAST_JSON_PATH": "/tmp/reclive-forecast.json",
        "FACILITY_HOURS_JSON_PATH": "/tmp/reclive-facility-hours.json",
        "FORECAST_API_ALLOW_ORIGINS": "https://reclive.example",
    }
    for name, value in production_values.items():
        monkeypatch.setenv(name, value)

    state_guard = threading.Lock()
    validation_barrier = threading.Barrier(2)
    second_configuration_attempted = threading.Event()
    duplicate_loader_entered = threading.Event()
    push_validated: set[str] = set()
    generic_validated: set[str] = set()
    evaluator_checks: list[str] = []
    capacity_calls = 0
    section_calls = 0
    errors: list[BaseException] = []

    actual_push_validation = forecast_api.validate_push_configuration
    actual_generic_validation = forecast_api.validate_production_environment

    def validate_push() -> None:
        actual_push_validation()
        with state_guard:
            push_validated.add(threading.current_thread().name)

    def validate_generic(*args: object, **kwargs: object) -> None:
        thread_name = threading.current_thread().name
        with state_guard:
            assert thread_name in push_validated
        actual_generic_validation(*args, **kwargs)  # type: ignore[arg-type]
        with state_guard:
            generic_validated.add(thread_name)
        validation_barrier.wait(timeout=5)

    class CoordinatedLock:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._attempt_guard = threading.Lock()
            self._attempts = 0

        def __enter__(self) -> CoordinatedLock:
            with self._attempt_guard:
                self._attempts += 1
                if self._attempts == 2:
                    second_configuration_attempted.set()
            self._lock.acquire()
            return self

        def __exit__(self, *_args: object) -> None:
            self._lock.release()

    def load_capacities() -> dict[int, int]:
        nonlocal capacity_calls
        thread_name = threading.current_thread().name
        with state_guard:
            assert thread_name in generic_validated
            capacity_calls += 1
            call_number = capacity_calls
        if call_number == 1:
            if not second_configuration_attempted.wait(timeout=5):
                raise AssertionError("second startup did not attempt configuration")
        else:
            duplicate_loader_entered.set()
            second_configuration_attempted.set()
        return {1186: 100}

    def load_sections() -> tuple[dict[int, str], dict[int, dict[str, list[int]]]]:
        nonlocal section_calls
        with state_guard:
            section_calls += 1
        return {1186: "Nick"}, {1186: {"overall": [1]}}

    def evaluator_enabled() -> bool:
        assert _seam_runtime.current_runtime().configuration_loaded is True
        with state_guard:
            evaluator_checks.append(threading.current_thread().name)
        return False

    monkeypatch.setattr(_seam_settings, "validate_push_configuration", validate_push)
    monkeypatch.setattr(
        _seam_api_lifespan_compat,
        "validate_production_environment",
        validate_generic,
    )
    monkeypatch.setattr(_seam_sections, "load_facility_capacities", load_capacities)
    monkeypatch.setattr(_seam_sections, "load_facility_sections", load_sections)
    monkeypatch.setattr(_seam_settings, "evaluator_enabled", evaluator_enabled)
    monkeypatch.setattr(_seam_runtime.current_runtime(), 'configuration_loaded', False)
    monkeypatch.setattr(_seam_runtime.current_runtime(), 'capacities', {})
    monkeypatch.setattr(_seam_runtime.current_runtime(), 'facility_names', {})
    monkeypatch.setattr(_seam_runtime.current_runtime(), 'section_ids', {})
    monkeypatch.setattr(
        _seam_runtime.current_runtime(),
        'configuration_lock',
        CoordinatedLock(),
        raising=False,
    )

    def run_lifespan() -> None:
        async def run() -> None:
            async with forecast_api.lifespan(forecast_api.app):
                pass

        try:
            asyncio.run(run())
        except BaseException as exc:
            with state_guard:
                errors.append(exc)

    threads = [
        threading.Thread(target=run_lifespan, name=f"startup-{index}")
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert duplicate_loader_entered.is_set() is False
    assert capacity_calls == 1
    assert section_calls == 1
    assert _seam_runtime.current_runtime().capacities == {1186: 100}
    assert _seam_runtime.current_runtime().facility_names == {1186: "Nick"}
    assert _seam_runtime.current_runtime().section_ids == {1186: {"overall": [1]}}
    assert sorted(evaluator_checks) == ["startup-0", "startup-1"]


@pytest.mark.parametrize(
    "name",
    ["FORECAST_API_PORT", "PUSH_ADMIN_ROUTES_ENABLED", "ACTUAL_HOUR_MIN_COVERAGE"],
)
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("malformed_kind", ["text", "mixed-number"])
def test_api_configuration_errors_hide_rejected_input_and_conversion_chain(
    name, legacy, malformed_kind, monkeypatch
):
    rejected = (
        "private-" + "configuration-probe"
        if malformed_kind == "text"
        else "12.5-invalid"
    )
    if legacy:
        monkeypatch.setenv(name, rejected)
        runtime = _seam_runtime.Runtime(_seam_settings.Settings.for_test(), legacy=True)
        action = {
            "FORECAST_API_PORT": lambda: _seam_settings.int_with_default(name, 8000),
            "PUSH_ADMIN_ROUTES_ENABLED": lambda: _seam_settings.bool_with_default(
                name, False
            ),
            "ACTUAL_HOUR_MIN_COVERAGE": lambda: _seam_settings.build_settings_from_environment(
                {name: rejected}, legacy=True
            ),
        }[name]
    else:
        runtime = _seam_runtime.Runtime(_seam_settings.Settings.for_test())

        def action():
            return _seam_settings.build_settings_from_environment({name: rejected})

    with _seam_runtime.runtime_scope(runtime):
        try:
            action()
        except Exception as error:
            safe = (
                isinstance(error, RuntimeError)
                and name in str(error)
                and rejected not in str(error)
                and rejected not in "".join(traceback.format_exception(error))
                and error.__cause__ is None
            )
        else:
            safe = False
    assert safe, "configuration failure must identify only the field/category"


def test_api_configuration_valid_parsing_controls(monkeypatch):
    values = {
        "FORECAST_API_PORT": " 8123 ",
        "PUSH_ADMIN_ROUTES_ENABLED": " YES ",
        "ACTUAL_HOUR_MIN_COVERAGE": "0.8",
    }
    settings = _seam_settings.build_settings_from_environment(values)
    assert (
        settings.port,
        settings.push.admin_routes_enabled,
        settings.actual_hour_min_coverage,
    ) == (8123, True, 0.8)
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    with _seam_runtime.runtime_scope(_seam_runtime.Runtime(settings, legacy=True)):
        assert _seam_settings.int_with_default("FORECAST_API_PORT", 8000) == 8123
        assert (
            _seam_settings.bool_with_default("PUSH_ADMIN_ROUTES_ENABLED", False) is True
        )
