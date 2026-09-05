"""Persistence owners."""

import sys as _import_sys

_import_sys.modules.setdefault(
    "server.reclive.repositories", _import_sys.modules[__name__]
)
_import_sys.modules.setdefault("reclive.repositories", _import_sys.modules[__name__])
