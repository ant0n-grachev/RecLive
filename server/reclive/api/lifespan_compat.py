from __future__ import annotations

import sys as _import_sys
import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from fastapi import FastAPI
from server.env_loader import validate_production_environment
from server.reclive.runtime import current_runtime, runtime_scope
from server.reclive import push as _owner_push
from server.reclive import sections as _owner_sections
from server.reclive import settings as _owner_settings


@asynccontextmanager
async def _scoped_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    _owner_settings.validate_push_configuration()
    runtime = current_runtime()
    settings = runtime.settings
    # Explicit apps consume these effective fields, not their original env capture.
    values = os.environ if runtime.legacy else {
        "APP_ENV": settings.environment,
        "GYM_DB_HOST": settings.database.host or "",
        "GYM_DB_PORT": settings.database.port or "",
        "GYM_DB_USER": settings.database.user or "",
        "GYM_DB_PASSWORD": settings.database.password or "",
        "GYM_DB_NAME": settings.database.name or "",
        "FORECAST_JSON_PATH": settings.forecast_json_path or "",
        "FACILITY_HOURS_JSON_PATH": settings.facility_hours_json_path or "",
        "FORECAST_API_ALLOW_ORIGINS": ",".join(settings.cors_origins),
        "PUSH_ADMIN_TOKEN": settings.push.admin_token,
    }
    validate_production_environment(
        values,
        required_names=(
            "GYM_DB_HOST",
            "GYM_DB_PORT",
            "GYM_DB_USER",
            "GYM_DB_PASSWORD",
            "GYM_DB_NAME",
            "FORECAST_JSON_PATH",
            "FACILITY_HOURS_JSON_PATH",
        ),
        cors_name="FORECAST_API_ALLOW_ORIGINS",
        admin_enabled=_owner_settings.push_admin_routes_enabled(),
    )
    _owner_sections.ensure_runtime_facility_configuration()
    started_task: Optional[asyncio.Task] = None
    if _owner_settings.evaluator_enabled() and (
        current_runtime().task is None or current_runtime().task.done()
    ):
        started_task = asyncio.create_task(_owner_push.evaluator_loop())
        current_runtime().task = started_task
    try:
        yield
    finally:
        if started_task is not None:
            started_task.cancel()
            try:
                await started_task
            except asyncio.CancelledError:
                pass
            if current_runtime().task is started_task:
                current_runtime().task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    with runtime_scope(app.state.runtime):
        async with _scoped_lifespan(app):
            yield


def build_lifespan(settings):
    return lifespan


_import_sys.modules.setdefault(
    "server.reclive.api.lifespan_compat", _import_sys.modules[__name__]
)
_import_sys.modules.setdefault(
    "reclive.api.lifespan_compat", _import_sys.modules[__name__]
)
