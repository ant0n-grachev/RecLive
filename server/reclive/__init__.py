"""RecLive backend modules, with one identity for historical import paths."""

import importlib
import sys
from pathlib import Path

# Direct server/*.py commands expose server/ on sys.path, not its parent.
# Load the real parent namespace once and restore the caller's search path.
if "server" not in sys.modules:
    _root = str(Path(__file__).resolve().parents[2])
    sys.path.insert(0, _root)
    try:
        importlib.import_module("server")
    finally:
        sys.path.remove(_root)

sys.modules.setdefault("reclive", sys.modules[__name__])
sys.modules.setdefault("server.reclive", sys.modules[__name__])
setattr(sys.modules["server"], "reclive", sys.modules[__name__])

for _name in ("env_loader", "forecast_shared", "facility_capacities"):
    _canonical = "server." + _name
    _module = sys.modules.get(_canonical) or sys.modules.get(_name)
    if _module is None:
        _module = importlib.import_module(_canonical)
    sys.modules[_canonical] = _module
    sys.modules[_name] = _module
    setattr(sys.modules["server"], _name, _module)

# These pre-existing lower owners are inert. Keep their bare imports canonical
# without modifying the frozen migration helpers or focused algorithms.
for _name in ("actual_hours", "ingestion", "occupancy_repository", "push_identity", "push_rule_backfill"):
    _canonical = "server.reclive." + _name
    _bare = "reclive." + _name
    _module = sys.modules.get(_canonical) or sys.modules.get(_bare)
    if _module is None:
        _module = importlib.import_module(_canonical)
    sys.modules[_canonical] = _module
    sys.modules[_bare] = _module
    setattr(sys.modules[__name__], _name, _module)
