"""Forecasting modules; metrics can load without model initialization."""

import sys

sys.modules.setdefault("server.reclive.forecasting", sys.modules[__name__])
sys.modules.setdefault("reclive.forecasting", sys.modules[__name__])
