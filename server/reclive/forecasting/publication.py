"""Retain compact, immutable evidence of forecasts after successful publication."""

import gzip
import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4


ARCHIVE_RETENTION_DAYS = 90
_ARCHIVE_NAME = re.compile(r"^forecast-(\d{8}T\d{12}Z)-[0-9a-f]{32}\.json\.gz$")


def archive_published_forecast(payload, forecast_path, published_at=None):
    """Record final facility predictions, never retrospective actuals or model internals.

    The caller must have replaced the served forecast before calling this function.
    Publication time is intentionally separate from the start of model computation.
    """
    published_at = published_at or datetime.now(timezone.utc)
    if published_at.tzinfo is None or published_at.utcoffset() is None:
        raise ValueError("publication time must have a timezone")
    published_at = published_at.astimezone(timezone.utc)
    facilities = []
    for facility in payload.get("facilities", []):
        days = []
        for day in facility.get("weeklyForecast", []):
            days.append({
                "date": day.get("date"),
                "totalHours": [
                    {key: hour[key] for key in ("hourStart", "expectedCount", "expectedPct") if key in hour}
                    for hour in day.get("totalHours", [])
                ],
            })
        facilities.append({
            "facilityId": facility.get("facilityId"),
            "occupancyThresholds": facility.get("occupancyThresholds"),
            "weeklyForecast": days,
        })
    archive = {
        "schemaVersion": 1,
        "generatedAt": payload.get("generatedAt"),
        "publishedAt": published_at.isoformat(),
        "timezone": payload.get("timezone"),
        "facilities": facilities,
    }
    encoded = gzip.compress(json.dumps(archive, allow_nan=False, separators=(",", ":")).encode("utf-8"), mtime=0)
    directory = Path(forecast_path).parent / "forecast-history"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    timestamp = published_at.strftime("%Y%m%dT%H%M%S%fZ")
    destination = directory / f"forecast-{timestamp}-{uuid4().hex}.json.gz"
    pending = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".pending-", delete=False) as handle:
            pending = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, destination)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)

    # Only this writer's named archives are eligible; retain unrelated files and links.
    cutoff = published_at - timedelta(days=ARCHIVE_RETENTION_DAYS)
    for candidate in directory.iterdir():
        match = _ARCHIVE_NAME.fullmatch(candidate.name)
        if match is None or candidate.is_symlink() or not candidate.is_file():
            continue
        try:
            issued_at = datetime.strptime(match[1], "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if issued_at < cutoff:
            candidate.unlink()
    return destination
