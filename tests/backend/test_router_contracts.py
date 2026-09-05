from copy import deepcopy
import json
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from server import forecast_api


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "refactor"
    / "forecast-contract.json"
)

EXPECTED_CUSTOM_ROUTES = {
    ("GET", "/health"),
    ("GET", "/health/push"),
    ("GET", "/api/forecast"),
    ("GET", "/api/forecast/facilities"),
    ("GET", "/api/forecast/facilities/{facility_id}"),
    ("GET", "/api/forecast/facilities/{facility_id}/actual-hours"),
    ("GET", "/api/live-counts"),
    ("GET", "/api/facility-hours"),
    ("GET", "/api/facility-hours/facilities"),
    ("GET", "/api/facility-hours/facilities/{facility_id}"),
    ("GET", "/api/push/public-key"),
    ("GET", "/api/push/availability"),
    ("POST", "/api/push/subscribe"),
    ("POST", "/api/push/rules/list"),
    ("DELETE", "/api/push/rules/{rule_id}"),
    ("POST", "/api/push/rules/cancel-all"),
    ("POST", "/api/push/dispatch"),
    ("POST", "/api/push/evaluate"),
}


@pytest.fixture()
def forecast_payload() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def api_client(
    monkeypatch: pytest.MonkeyPatch,
    forecast_payload: dict[str, object],
) -> TestClient:
    monkeypatch.setattr(
        forecast_api,
        "load_forecast",
        lambda: deepcopy(forecast_payload),
    )
    return TestClient(forecast_api.app)


def test_router_preserves_all_custom_method_path_pairs() -> None:
    actual_routes = {
        (method, route.path)
        for route in forecast_api.app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
        if route.path == "/health" or route.path.startswith(("/health/", "/api/"))
    }

    assert actual_routes == EXPECTED_CUSTOM_ROUTES


def test_forecast_root_preserves_sanitized_fixture(
    api_client: TestClient,
    forecast_payload: dict[str, object],
) -> None:
    response = api_client.get("/api/forecast")

    assert response.status_code == 200
    assert response.json() == forecast_payload


def test_forecast_facility_list_preserves_sorted_view_model(
    api_client: TestClient,
) -> None:
    response = api_client.get("/api/forecast/facilities")

    assert response.status_code == 200
    assert sorted(response.json(), key=lambda item: item["facilityId"]) == [
        {
            "facilityId": 1186,
            "facilityName": "Nicholas Recreation Center",
            "days": 1,
        },
        {
            "facilityId": 1656,
            "facilityName": "Bakke Recreation & Wellbeing Center",
            "days": 1,
        },
    ]


def test_full_facility_route_retains_interval_and_qualified_actual_fields(
    api_client: TestClient,
    forecast_payload: dict[str, object],
) -> None:
    response = api_client.get("/api/forecast/facilities/1186")

    assert response.status_code == 200
    facilities = forecast_payload["facilities"]
    assert isinstance(facilities, list)
    nick = next(
        facility
        for facility in facilities
        if isinstance(facility, dict) and facility.get("facilityId") == 1186
    )
    assert response.json() == {
        **nick,
        "forecastDayStartHour": 6,
        "forecastDayEndHour": 23,
    }


def test_forecast_router_preserves_compact_contract(api_client: TestClient) -> None:
    response = api_client.get("/api/forecast/facilities/1186?compact=1")

    assert response.status_code == 200
    assert response.json() == {
        "facilityId": 1186,
        "facilityName": "Nicholas Recreation Center",
        "occupancyThresholds": {"lowMax": 34, "peakMin": 70},
        "sectionOccupancyThresholds": {
            "fitness floors": {"lowMax": 30, "peakMin": 68}
        },
        "locationOccupancyThresholds": {
            "5761": {"lowMax": 28, "peakMin": 65}
        },
        "weeklyForecast": [
            {
                "dayName": "Monday",
                "date": "2026-08-31",
                "categories": [
                    {
                        "key": "fitness floors",
                        "title": "Fitness Floors",
                        "maxCapacity": 200,
                        "hours": [
                            {
                                "hour": 9,
                                "hourStart": "2026-08-31T09:00:00-05:00",
                                "expectedCount": 60,
                                "expectedPct": 0.3,
                                "actualCount": 58,
                                "actualPct": 0.29,
                                "actualSampleCount": 12,
                                "actualCoverage": 1.0,
                                "spikeAdjusted": False,
                            },
                            {
                                "hour": 10,
                                "hourStart": "2026-08-31T10:00:00-05:00",
                                "expectedCount": 72,
                                "expectedPct": 0.36,
                                "actualCount": 70,
                                "actualPct": 0.35,
                                "actualSampleCount": 11,
                                "actualCoverage": 0.92,
                                "spikeAdjusted": True,
                            },
                        ],
                    }
                ],
                "totalHours": [
                    {
                        "hour": 9,
                        "hourStart": "2026-08-31T09:00:00-05:00",
                        "expectedCount": 210,
                        "expectedPct": 0.42,
                        "actualCount": 205,
                        "actualPct": 0.41,
                        "actualSampleCount": 24,
                        "actualCoverage": 1.0,
                    },
                    {
                        "hour": 10,
                        "hourStart": "2026-08-31T10:00:00-05:00",
                        "expectedCount": 240,
                        "expectedPct": 0.48,
                        "actualCount": 232,
                        "actualPct": 0.464,
                        "actualSampleCount": 22,
                        "actualCoverage": 0.91,
                    },
                ],
                "avoidWindows": [],
                "bestWindows": [
                    {
                        "start": "2026-08-31T09:00:00-05:00",
                        "end": "2026-08-31T10:00:00-05:00",
                        "startHour": 9,
                        "endHour": 10,
                        "windowHours": 1,
                        "expectedTotal": 210,
                        "expectedAvg": 210,
                    }
                ],
                "crowdBands": [
                    {
                        "start": "2026-08-31T09:00:00-05:00",
                        "end": "2026-08-31T11:00:00-05:00",
                        "level": "medium",
                    }
                ],
            }
        ],
        "forecastDayStartHour": 6,
        "forecastDayEndHour": 23,
    }


def test_date_route_preserves_full_and_compact_bakke_views(
    api_client: TestClient,
    forecast_payload: dict[str, object],
) -> None:
    full_response = api_client.get(
        "/api/forecast/facilities/1656?date=2026-08-31"
    )
    compact_response = api_client.get(
        "/api/forecast/facilities/1656?date=2026-08-31&compact=1"
    )

    assert full_response.status_code == 200
    assert compact_response.status_code == 200
    facilities = forecast_payload["facilities"]
    assert isinstance(facilities, list)
    bakke = next(
        facility
        for facility in facilities
        if isinstance(facility, dict) and facility.get("facilityId") == 1656
    )
    weekly_forecast = bakke["weeklyForecast"]
    assert isinstance(weekly_forecast, list)
    assert full_response.json() == {
        "facilityId": 1656,
        "facilityName": "Bakke Recreation & Wellbeing Center",
        "forecastDayStartHour": 6,
        "forecastDayEndHour": 23,
        "occupancyThresholds": {"lowMax": 32, "peakMin": 68},
        "sectionOccupancyThresholds": {
            "fitness floors": {"lowMax": 29, "peakMin": 66}
        },
        "locationOccupancyThresholds": {
            "8717": {"lowMax": 27, "peakMin": 64}
        },
        "day": weekly_forecast[0],
    }
    assert compact_response.json() == {
        "facilityId": 1656,
        "facilityName": "Bakke Recreation & Wellbeing Center",
        "forecastDayStartHour": 6,
        "forecastDayEndHour": 23,
        "occupancyThresholds": {"lowMax": 32, "peakMin": 68},
        "sectionOccupancyThresholds": {
            "fitness floors": {"lowMax": 29, "peakMin": 66}
        },
        "locationOccupancyThresholds": {
            "8717": {"lowMax": 27, "peakMin": 64}
        },
        "day": {
            "dayName": "Monday",
            "date": "2026-08-31",
            "categories": [
                {
                    "key": "fitness floors",
                    "title": "Fitness Floors",
                    "maxCapacity": 240,
                    "hours": [
                        {
                            "hour": 9,
                            "hourStart": "2026-08-31T09:00:00-05:00",
                            "expectedCount": 66,
                            "expectedPct": 0.275,
                            "actualCount": 64,
                            "actualPct": 0.2667,
                            "actualSampleCount": 10,
                            "actualCoverage": 1.0,
                            "spikeAdjusted": False,
                        },
                        {
                            "hour": 10,
                            "hourStart": "2026-08-31T10:00:00-05:00",
                            "expectedCount": 78,
                            "expectedPct": 0.325,
                            "actualCount": 75,
                            "actualPct": 0.3125,
                            "actualSampleCount": 9,
                            "actualCoverage": 0.9,
                            "spikeAdjusted": True,
                        },
                    ],
                }
            ],
            "totalHours": [
                {
                    "hour": 9,
                    "hourStart": "2026-08-31T09:00:00-05:00",
                    "expectedCount": 260,
                    "expectedPct": 0.4,
                    "actualCount": 252,
                    "actualPct": 0.3877,
                    "actualSampleCount": 21,
                    "actualCoverage": 1.0,
                },
                {
                    "hour": 10,
                    "hourStart": "2026-08-31T10:00:00-05:00",
                    "expectedCount": 292,
                    "expectedPct": 0.4492,
                    "actualCount": 286,
                    "actualPct": 0.44,
                    "actualSampleCount": 19,
                    "actualCoverage": 0.9,
                },
            ],
            "avoidWindows": [
                {
                    "start": "2026-08-31T10:00:00-05:00",
                    "end": "2026-08-31T11:00:00-05:00",
                    "startHour": 10,
                    "endHour": 11,
                    "windowHours": 1,
                    "expectedTotal": 292,
                    "expectedAvg": 292,
                }
            ],
            "bestWindows": [],
            "crowdBands": [
                {
                    "start": "2026-08-31T09:00:00-05:00",
                    "end": "2026-08-31T10:00:00-05:00",
                    "level": "medium",
                },
                {
                    "start": "2026-08-31T10:00:00-05:00",
                    "end": "2026-08-31T11:00:00-05:00",
                    "level": "peak",
                },
            ],
        },
    }


@pytest.mark.parametrize(
    ("path", "detail"),
    [
        ("/api/forecast/facilities/9999", "Facility not found"),
        (
            "/api/forecast/facilities/1186?date=2026-09-01",
            "Date not found for facility",
        ),
    ],
)
def test_forecast_router_preserves_not_found_errors(
    api_client: TestClient,
    path: str,
    detail: str,
) -> None:
    response = api_client.get(path)

    assert response.status_code == 404
    assert response.json() == {"detail": detail}
