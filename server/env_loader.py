import os

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
}


def load_project_dotenv() -> None:
    for path in (
        os.path.join(SCRIPT_DIR, ".env"),
        os.path.join(PROJECT_ROOT, ".env"),
    ):
        if os.path.exists(path):
            load_dotenv(path, override=False)
    for key, value in DEFAULT_ENV.items():
        os.environ.setdefault(key, value)
