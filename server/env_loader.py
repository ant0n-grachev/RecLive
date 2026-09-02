import os
import sys
import threading
from collections.abc import Mapping, Sequence
from types import ModuleType

from dotenv import load_dotenv


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DEFAULT_ENV = {
    "LIVE_COUNTS_URL": "https://goboardapi.azurewebsites.net/api/FacilityCount/GetCountsByAccount?AccountAPIKey=YOUR_ACCOUNT_API_KEY",
    "GYM_DB_HOST": "localhost",
    "GYM_DB_PORT": "3306",
    "GYM_DB_USER": "root",
    "GYM_DB_PASSWORD": "change_me",
    "GYM_DB_NAME": "gym_data",
    "GYM_DB_TIMEZONE": "America/Chicago",
    "FORECAST_DAY_START_HOUR": "6",
    "FORECAST_DAY_END_HOUR": "23",
    "GYM_WEATHER_URL": "https://api.open-meteo.com/v1/forecast",
    "GYM_WEATHER_ARCHIVE_URL": "https://archive-api.open-meteo.com/v1/archive",
    "GYM_WEATHER_LAT": "43.0731",
    "GYM_WEATHER_LON": "-89.4012",
    "GYM_WEATHER_FORECAST_DAYS": "7",
    "GYM_WEATHER_HISTORY_MAX_DAYS": "180",
    "MODEL_ARTIFACT_DIR": "model_artifacts",
    "MODEL_BASENAME": "forecast_model",
    "FORECAST_JSON_PATH": "forecast.json",
    "FACILITY_HOURS_JSON_PATH": "facility_hours.json",
    "FACILITY_CAPACITIES_JSON_PATH": "shared/facility_capacities.json",
}
APP_ENVIRONMENTS = frozenset({"development", "test", "production"})
_DOTENV_STATE_KEY = "_reclive_dotenv_state"
_dotenv_state_candidate = ModuleType(_DOTENV_STATE_KEY)
_dotenv_state_candidate.loaded = False
_dotenv_state_candidate.lock = threading.Lock()
_DOTENV_STATE = sys.modules.setdefault(_DOTENV_STATE_KEY, _dotenv_state_candidate)


class EnvironmentConfigurationError(RuntimeError):
    pass


def _app_environment(values: Mapping[str, str]) -> str:
    environment = str(values.get("APP_ENV", "development")).strip().lower()
    if environment not in APP_ENVIRONMENTS:
        raise EnvironmentConfigurationError(
            "Unsafe environment configuration: APP_ENV"
        )
    return environment


def _unsafe_production_configuration(name: str) -> EnvironmentConfigurationError:
    return EnvironmentConfigurationError(f"Unsafe production configuration: {name}")


def validate_production_environment(
    values: Mapping[str, str],
    *,
    required_names: Sequence[str],
    cors_name: str | None,
    admin_enabled: bool,
) -> None:
    if _app_environment(values) != "production":
        return

    for name in required_names:
        value = str(values.get(name, "")).strip()
        if (
            not value
            or value == "change_me"
            or "YOUR_ACCOUNT_API_KEY" in value
        ):
            raise _unsafe_production_configuration(name)
        if name == "GYM_DB_PORT":
            try:
                port = int(value)
            except ValueError:
                raise _unsafe_production_configuration(name) from None
            if not 1 <= port <= 65_535:
                raise _unsafe_production_configuration(name)

    if cors_name:
        origins = [
            item.strip()
            for item in str(values.get(cors_name, "")).split(",")
            if item.strip()
        ]
        if not origins or "*" in origins:
            raise _unsafe_production_configuration(cors_name)

    token = str(values.get("PUSH_ADMIN_TOKEN", ""))
    if admin_enabled and len(token.encode("utf-8")) < 32:
        raise _unsafe_production_configuration("PUSH_ADMIN_TOKEN")


def load_project_dotenv() -> None:
    if _DOTENV_STATE.loaded:
        return

    with _DOTENV_STATE.lock:
        if _DOTENV_STATE.loaded:
            return

        if "APP_ENV" in os.environ:
            _app_environment(os.environ)

        for path in (
            os.path.join(SCRIPT_DIR, ".env"),
            os.path.join(PROJECT_ROOT, ".env"),
        ):
            if os.path.exists(path):
                load_dotenv(path, override=False)
                break

        if _app_environment(os.environ) != "production":
            for key, value in DEFAULT_ENV.items():
                os.environ.setdefault(key, value)

        _DOTENV_STATE.loaded = True
