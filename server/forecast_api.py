import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytz
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

FORECAST_JSON_PATH = os.getenv(
    "FORECAST_JSON_PATH",
    os.path.join(os.path.dirname(__file__), "forecast.json"),
)

API_HOST = os.getenv("FORECAST_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("FORECAST_API_PORT", "8000"))
CHICAGO_TZ = pytz.timezone("America/Chicago")


def parse_allowed_origins() -> List[str]:
    raw = os.getenv("FORECAST_API_ALLOW_ORIGINS", "*").strip()
    if not raw:
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_forecast() -> Dict[str, Any]:
    if not os.path.exists(FORECAST_JSON_PATH):
        raise HTTPException(status_code=503, detail="Forecast not generated yet")

    try:
        with open(FORECAST_JSON_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Forecast file corrupted") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to read forecast file") from exc


def generated_age_seconds(payload: Dict[str, Any]) -> Optional[int]:
    generated_at = payload.get("generatedAt")
    if not generated_at:
        return None
    try:
        ts = datetime.fromisoformat(str(generated_at))
        if ts.tzinfo is None:
            ts = CHICAGO_TZ.localize(ts)
        else:
            ts = ts.astimezone(CHICAGO_TZ)
        now = datetime.now(CHICAGO_TZ)
        return int((now - ts).total_seconds())
    except Exception:
        return None


app = FastAPI(title="RecLive Forecast API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_allowed_origins(),
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    payload = load_forecast()
    return {
        "status": "ok",
        "generatedAt": payload.get("generatedAt"),
        "generatedAgeSeconds": generated_age_seconds(payload),
        "facilities": len(payload.get("facilities", [])),
        "modelStatus": payload.get("modelInfo", {}).get("status"),
    }


@app.get("/api/forecast")
def forecast() -> Dict[str, Any]:
    return load_forecast()


@app.get("/api/forecast/facilities")
def facilities() -> List[Dict[str, Any]]:
    payload = load_forecast()
    items = []
    for facility in payload.get("facilities", []):
        items.append(
            {
                "facilityId": facility.get("facilityId"),
                "facilityName": facility.get("facilityName"),
                "days": len(facility.get("weeklyForecast", [])),
            }
        )
    return items


@app.get("/api/forecast/facilities/{facility_id}")
def facility_forecast(
    facility_id: int,
    date: Optional[str] = Query(None, description="YYYY-MM-DD"),
) -> Dict[str, Any]:
    payload = load_forecast()
    facilities_data = payload.get("facilities", [])
    facility = next((row for row in facilities_data if row.get("facilityId") == facility_id), None)
    if not facility:
        raise HTTPException(status_code=404, detail="Facility not found")

    if not date:
        return facility

    day = next((row for row in facility.get("weeklyForecast", []) if row.get("date") == date), None)
    if not day:
        raise HTTPException(status_code=404, detail="Date not found for facility")

    return {
        "facilityId": facility.get("facilityId"),
        "facilityName": facility.get("facilityName"),
        "day": day,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("forecast_api:app", host=API_HOST, port=API_PORT, reload=False)
