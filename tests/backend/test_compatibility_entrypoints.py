import importlib


def test_legacy_backend_entrypoints_export_their_existing_callables() -> None:
    assert callable(importlib.import_module("server.gym_fetch").main)
    assert callable(importlib.import_module("server.forecast_api").app)
    assert callable(importlib.import_module("server.forecast_job").build_forecast)
    assert callable(importlib.import_module("server.facility_hours_fetch").main)


def test_bare_backend_entrypoints_remain_importable() -> None:
    assert callable(importlib.import_module("gym_fetch").main)
    assert callable(importlib.import_module("forecast_api").app)
    assert callable(importlib.import_module("forecast_job").build_forecast)
    assert callable(importlib.import_module("facility_hours_fetch").main)
