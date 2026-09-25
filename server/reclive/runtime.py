"""Application-owned state and the explicit scope used by API dependencies.

The only process default is installed by legacy bootstrap. A request scope never
replaces it. Thread work must capture and enter its originating runtime scope.
"""

import sys as _import_sys

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Any, Callable

from .settings import Settings


@dataclass
class Runtime:
    settings: Settings
    legacy: bool = False
    capacities: dict = field(default_factory=dict)
    facility_names: dict = field(default_factory=dict)
    section_ids: dict = field(default_factory=dict)
    configuration_loaded: bool = False
    configuration_lock: Any = field(default_factory=threading.Lock)
    task: Any = None
    executor: Any = None
    transport_lock: Any = field(default_factory=threading.Lock)
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))
    rule_ownership: Callable | None = None
    webpush: Callable | None = None
    resolver: Callable | None = None

    def __post_init__(self):
        if self.settings.capacities is not None:
            self.capacities = dict(self.settings.capacities)
            self.facility_names = dict(self.settings.facility_names or {})
            self.section_ids = {
                key: {section: list(ids) for section, ids in sections.items()}
                for key, sections in (self.settings.section_ids or {}).items()
            }
            self.configuration_loaded = True

    def connect(self, *, autocommit=True):
        from . import db

        if self.legacy:
            return db.open_db_connection(autocommit=autocommit)
        return db.open_db_connection(
            self.settings.database,
            autocommit=autocommit,
        )


_active_runtime: ContextVar[Runtime | None] = ContextVar(
    "reclive_runtime", default=None
)
legacy_runtime: Runtime | None = None


def current_runtime() -> Runtime:
    runtime = _active_runtime.get()
    if runtime is None:
        runtime = legacy_runtime
    if runtime is None:
        raise RuntimeError("RecLive runtime has not been initialized")
    return runtime


@contextmanager
def runtime_scope(runtime: Runtime):
    token = _active_runtime.set(runtime)
    try:
        yield runtime
    finally:
        _active_runtime.reset(token)


def now_utc() -> datetime:
    return current_runtime().clock()


def now_iso() -> str:
    return now_utc().isoformat()


_import_sys.modules.setdefault("server.reclive.runtime", _import_sys.modules[__name__])
_import_sys.modules.setdefault("reclive.runtime", _import_sys.modules[__name__])
