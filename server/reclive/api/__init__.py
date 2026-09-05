"""Inert API package; production bootstrap is explicit in app.get_default_app."""

import sys as _import_sys

_import_sys.modules.setdefault("server.reclive.api", _import_sys.modules[__name__])
_import_sys.modules.setdefault("reclive.api", _import_sys.modules[__name__])
