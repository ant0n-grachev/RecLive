from server.reclive.api.forecasts import compact_day_payload


def test_compact_forecast_discards_embedded_actuals_in_both_paths():
    hour = {"hourStart": "2026-08-31T09:00:00-05:00", "expectedCount": 50,
            "actualCount": 777, "actualPct": 0.9, "actualCoverage": 0.1, "actualSampleCount": 1}
    day = compact_day_payload({"date": "2026-08-31", "dayName": "Monday", "totalHours": [hour],
                               "categories": [{"key": "fitness floors", "title": "Fitness Floors", "hours": [hour]}]})
    for output in [day["totalHours"][0], day["categories"][0]["hours"][0]]:
        assert output == {"hourStart": "2026-08-31T09:00:00-05:00", "expectedCount": 50}
