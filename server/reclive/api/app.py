"""FastAPI composition and the one lazy legacy application."""

import sys as _import_sys

import threading
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.env_loader import load_project_dotenv
from server.reclive import push as push_service, runtime as runtime_owner, sections
from server.reclive import settings as settings_owner
from server.reclive.health_repository import HealthRepository
from server.reclive.repositories import push_rules as push_rules_owner
from server.reclive.runtime import Runtime, runtime_scope
from server.reclive.settings import Settings, app_environment
from . import forecasts, health, live_counts, push, schedules
from .dependencies import bind_runtime
from .lifespan_compat import build_lifespan


def create_app(settings: Settings | None = None) -> FastAPI:
    effective_settings = (
        settings if settings is not None else Settings.from_environment()
    )
    runtime = Runtime(effective_settings)
    runtime.rule_ownership = lambda *args: push_service._rule_is_owned(*args)
    app = FastAPI(
        title="RecLive Forecast API",
        version="1.1.0",
        lifespan=build_lifespan(effective_settings),
        dependencies=[Depends(bind_runtime)],
    )
    app.state.settings = effective_settings
    app.state.runtime = runtime
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(effective_settings.cors_origins),
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(
        health.create_health_router(
            repository=HealthRepository(
                connect=lambda: runtime.connect(autocommit=False),
                migration_dir=Path(__file__).resolve().parents[2] / "migrations",
                forecast_path=Path(runtime.settings.forecast_json_path),
                schedule_path=Path(runtime.settings.facility_hours_json_path),
                push_status=lambda: (
                    "ready"
                    if push_rules_owner.push_db_available()
                    and settings_owner.push_vapid_configured()
                    and push_service.push_identity_configured()
                    else "unavailable"
                ),
            ),
            now=lambda: runtime.clock(),
            ingestion_stale_after_seconds=runtime.settings.ingestion_stale_after_seconds,
            forecast_stale_after_seconds=runtime.settings.forecast_stale_after_seconds,
            schedule_stale_after_seconds=runtime.settings.schedule_stale_after_seconds,
        )
    )
    for router in (
        health.router,
        forecasts.router,
        live_counts.router,
        schedules.router,
        push.router,
    ):
        app.include_router(router)
    return app


_default_app: FastAPI | None = None
_default_lock = threading.Lock()


def get_default_app() -> FastAPI:
    global _default_app
    if _default_app is None:
        with _default_lock:
            if _default_app is None:
                load_project_dotenv()
                app = create_app(Settings.from_environment(legacy=True))
                runtime = app.state.runtime
                runtime.legacy = True
                runtime_owner.legacy_runtime = runtime
                with runtime_scope(runtime):
                    if app_environment() != "production":
                        sections.ensure_runtime_facility_configuration()
                _default_app = app
    return _default_app


def __getattr__(name):
    if name == "app":
        return get_default_app()
    raise AttributeError(name)


_import_sys.modules.setdefault("server.reclive.api.app", _import_sys.modules[__name__])
_import_sys.modules.setdefault("reclive.api.app", _import_sys.modules[__name__])
