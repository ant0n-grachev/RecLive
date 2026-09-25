"""Independent publication evidence must precede the attendance being scored."""

import gzip
import json
from datetime import datetime, timedelta, timezone

import pytest

from server.reclive.forecasting.verification import evaluate_hour, load_predictions


UTC = timezone.utc
TARGET = datetime.fromisoformat("2026-09-26T12:00:00-05:00")
NOW = TARGET + timedelta(hours=2)


def archive(tmp_path, published, count=100, *, name="a", start=TARGET, percentages=None):
    payload = {
        "schemaVersion": 1,
        "generatedAt": (published - timedelta(minutes=4)).isoformat(),
        "publishedAt": published.isoformat(),
        "timezone": "America/Chicago",
        "facilities": [{
            "facilityId": 1186,
            "occupancyThresholds": {"lowMax": 11, "peakMin": 26},
            "weeklyForecast": [{
                "date": start.date().isoformat(),
                "totalHours": [{
                    "hourStart": (start + timedelta(minutes=minute)).isoformat(),
                    "expectedCount": count if isinstance(count, int) else count[index],
                    "expectedPct": percentages[index] if percentages else count / 1000 if isinstance(count, int) else .1,
                } for index, minute in enumerate((0, 15, 30, 45))],
            }],
        }],
    }
    path = tmp_path / f"forecast-{name}.json.gz"
    save(path, payload)
    return path, payload


def save(path, payload):
    path.write_bytes(gzip.compress(json.dumps(payload).encode()))


def actual(count=90, *, start=TARGET, **changes):
    result = {
        "hourStart": start.isoformat(), "actualCount": count,
        "actualPct": count / 1000 if count is not None else None,
        "observedCount": count, "observedCapacity": 1000, "expectedCapacity": 1000,
        "actualCoverage": 1.0, "temporalCoverage": 1.0, "coverageThreshold": .75,
    }
    return {**result, **changes}


def evaluate(tmp_path, evidence=None, *, start=TARGET, now=NOW):
    predictions, _ = load_predictions(tmp_path, now)
    return evaluate_hour(predictions, 1186, start, evidence, now)


def test_no_archives_reports_no_forecast_without_fabricated_scores(tmp_path):
    predictions, invalid = load_predictions(tmp_path, NOW)
    assert predictions == {}
    assert invalid == 0
    records = evaluate(tmp_path, actual())
    assert [row["leadHours"] for row in records] == [1, 24]
    assert all(row["status"] == "no_forecast" for row in records)
    assert all(row["predictedCount"] is None and row["errorPeople"] is None for row in records)
    assert all(row["actualCount"] == 90 for row in records)


def test_latest_eligible_publication_at_each_lead_excludes_revisions_and_stale_forecasts(tmp_path):
    archive(tmp_path, TARGET - timedelta(hours=25), 800, name="stale")
    archive(tmp_path, TARGET - timedelta(hours=24, minutes=45), 110, name="day-old")
    archive(tmp_path, TARGET - timedelta(hours=24, minutes=5), 120, name="day")
    archive(tmp_path, TARGET - timedelta(hours=2), 700, name="hour-stale")
    chosen, _ = archive(tmp_path, TARGET - timedelta(hours=1, minutes=5), 100, name="hour")
    archive(tmp_path, TARGET - timedelta(minutes=59), 900, name="too-recent")
    archive(tmp_path, TARGET + timedelta(minutes=10), 90, name="lookahead")
    rows = evaluate(tmp_path, actual())
    assert [row["predictedCount"] for row in rows] == [100, 120]
    assert [row["errorPeople"] for row in rows] == [10, 30]
    assert rows[0]["archive"] == chosen.name
    assert rows[0]["actualLeadHours"] == pytest.approx(65 / 60)
    assert rows[0]["status"] == "scored"


def test_exact_cutoff_is_eligible_but_full_extra_hour_is_stale(tmp_path):
    archive(tmp_path, TARGET - timedelta(hours=1), 100)
    assert evaluate(tmp_path, actual())[0]["predictedCount"] == 100
    archive(tmp_path, TARGET - timedelta(hours=25), 800, name="stale-day")
    assert evaluate(tmp_path, actual())[1]["status"] == "no_forecast"


def test_retries_select_deterministically_without_duplicate_rows(tmp_path):
    published = TARGET - timedelta(hours=1)
    archive(tmp_path, published, 100, name="a")
    archive(tmp_path, published, 100, name="b")
    first = evaluate(tmp_path, actual())
    second = evaluate(tmp_path, actual())
    assert first == second
    assert len(first) == 2
    assert first[0]["archive"] == "forecast-b.json.gz"


@pytest.mark.parametrize("evidence", [
    None,
    actual(None),
    actual(90, temporalCoverage=.74),
    actual(90, actualCoverage=.74),
    actual(90, actualCoverage=None),
    actual(90, coverageThreshold=None),
    actual(90, expectedCapacity=0),
    actual(90, hourStart=(TARGET + timedelta(hours=1)).isoformat()),
])
def test_unqualified_attendance_never_becomes_zero_or_a_score(tmp_path, evidence):
    archive(tmp_path, TARGET - timedelta(hours=1))
    row = evaluate(tmp_path, evidence)[0]
    assert row["status"] == "insufficient_actual"
    assert row["actualCount"] is None
    assert row["errorPeople"] is None
    assert row["absoluteError"] is None
    assert row["crowdMatch"] is None


def test_true_zero_attendance_is_scoreable(tmp_path):
    archive(tmp_path, TARGET - timedelta(hours=1), 2)
    row = evaluate(tmp_path, actual(0))[0]
    assert row["status"] == "scored"
    assert row["actualCount"] == 0
    assert row["absoluteError"] == 2
    assert row["actualCrowd"] == "low"


def test_incomplete_hour_has_no_records(tmp_path):
    archive(tmp_path, TARGET - timedelta(hours=1))
    assert evaluate(tmp_path, actual(), now=TARGET + timedelta(minutes=59, seconds=59)) == []
    assert len(evaluate(tmp_path, actual(), now=TARGET + timedelta(hours=1))) == 2


def test_frontend_rounding_and_thresholds_are_preserved(tmp_path):
    archive(tmp_path, TARGET - timedelta(hours=1), [100, 100, 101, 101], percentages=[.254, .255, .256, .255])
    row = evaluate(tmp_path, actual(90, actualPct=.2549))[0]
    assert row["predictedCount"] == 101  # JS rounds 100.5 up.
    assert row["expectedPct"] == pytest.approx(.255)
    assert row["predictedCrowd"] == "medium"  # Chart classifies the unrounded 25.5%.
    assert row["actualCrowd"] == "medium"
    assert row["crowdMatch"] is True


@pytest.mark.parametrize("percent, expected", [(.11, "low"), (.114, "medium"), (.115, "medium"), (.26, "peak")])
def test_crowd_boundaries(tmp_path, percent, expected):
    archive(tmp_path, TARGET - timedelta(hours=1), percentages=[percent] * 4)
    assert evaluate(tmp_path, actual())[0]["predictedCrowd"] == expected


def test_unknown_thresholds_preserve_headcount_score_without_crowd_score(tmp_path):
    path, payload = archive(tmp_path, TARGET - timedelta(hours=1))
    payload["facilities"][0]["occupancyThresholds"] = None
    save(path, payload)
    row = evaluate(tmp_path, actual())[0]
    assert row["status"] == "scored"
    assert row["absoluteError"] == 10
    assert row["predictedCrowd"] is None
    assert row["actualCrowd"] is None
    assert row["crowdMatch"] is None


def test_missing_percentage_preserves_headcount_but_not_category(tmp_path):
    path, payload = archive(tmp_path, TARGET - timedelta(hours=1))
    payload["facilities"][0]["weeklyForecast"][0]["totalHours"][0].pop("expectedPct")
    save(path, payload)
    row = evaluate(tmp_path, actual())[0]
    assert row["status"] == "scored"
    assert row["expectedPct"] is None
    assert row["crowdMatch"] is None


def test_incomplete_or_ambiguous_forecast_hour_is_not_averaged(tmp_path):
    path, payload = archive(tmp_path, TARGET - timedelta(hours=1))
    points = payload["facilities"][0]["weeklyForecast"][0]["totalHours"]
    points.pop()
    save(path, payload)
    assert evaluate(tmp_path, actual())[0]["status"] == "no_forecast"
    points.append({**points[0], "expectedCount": 900})
    save(path, payload)
    assert evaluate(tmp_path, actual())[0]["status"] == "no_forecast"


def test_dst_repeated_hours_remain_distinct(tmp_path):
    first = datetime.fromisoformat("2026-11-01T01:00:00-05:00")
    second = datetime.fromisoformat("2026-11-01T01:00:00-06:00")
    archive(tmp_path, first - timedelta(hours=1), 100, name="early", start=first)
    archive(tmp_path, second - timedelta(hours=1), 200, name="late", start=second)
    now = second + timedelta(hours=2)
    early = evaluate(tmp_path, actual(start=first), start=first, now=now)[0]
    late = evaluate(tmp_path, actual(start=second), start=second, now=now)[0]
    assert early["predictedCount"] == 100
    assert late["predictedCount"] == 200
    assert early["hourStart"] != late["hourStart"]


@pytest.mark.parametrize("field, value", [
    ("publishedAt", "2026-09-26T11:00:00"),
    ("generatedAt", "nonsense"),
    ("generatedAt", (TARGET + timedelta(days=1)).isoformat()),
    ("publishedAt", (NOW + timedelta(hours=1)).isoformat()),
    ("schemaVersion", 999),
    ("timezone", "bad/timezone"),
])
def test_invalid_metadata_is_counted_and_excluded(tmp_path, field, value):
    path, payload = archive(tmp_path, TARGET - timedelta(hours=1))
    payload[field] = value
    save(path, payload)
    predictions, invalid = load_predictions(tmp_path, NOW)
    assert predictions == {}
    assert invalid == 1


@pytest.mark.parametrize("timestamp", ["2026-09-26T12:00:00", "broken", "2026-09-26T12:07:00-05:00"])
def test_invalid_or_ambiguous_target_timestamp_is_excluded(tmp_path, timestamp):
    path, payload = archive(tmp_path, TARGET - timedelta(hours=1))
    payload["facilities"][0]["weeklyForecast"][0]["totalHours"][0]["hourStart"] = timestamp
    save(path, payload)
    predictions, invalid = load_predictions(tmp_path, NOW)
    assert predictions == {}
    assert invalid == 1


def test_corrupt_archives_do_not_hide_valid_evidence(tmp_path):
    (tmp_path / "forecast-corrupt.json.gz").write_bytes(b"not-gzip")
    archive(tmp_path, TARGET - timedelta(hours=1))
    predictions, invalid = load_predictions(tmp_path, NOW)
    assert invalid == 1
    assert evaluate_hour(predictions, 1186, TARGET, actual(), NOW)[0]["status"] == "scored"


def test_naive_caller_times_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_predictions(tmp_path, NOW.replace(tzinfo=None))
    with pytest.raises(ValueError):
        evaluate_hour({}, 1186, TARGET.replace(tzinfo=None), actual(), NOW)


def test_conflicting_publications_at_same_instant_are_not_arbitrarily_scored(tmp_path):
    archive(tmp_path, TARGET - timedelta(hours=1), 100, name="a")
    archive(tmp_path, TARGET - timedelta(hours=1), 900, name="b")
    assert evaluate(tmp_path, actual())[0]["status"] == "no_forecast"


def test_minimum_target_hour_keeps_only_relevant_evidence(tmp_path):
    archive(tmp_path, TARGET - timedelta(hours=1))
    archive(tmp_path, TARGET - timedelta(hours=2), name="previous", start=TARGET - timedelta(hours=1))
    predictions, invalid = load_predictions(tmp_path, NOW, min_hour=TARGET)
    assert invalid == 0
    assert set(predictions) == {(1186, TARGET.astimezone(UTC))}


def test_owned_old_archive_can_be_skipped_without_reading_it(tmp_path):
    stale = TARGET - timedelta(hours=25)
    name = f"forecast-{stale.astimezone(UTC):%Y%m%dT%H%M%S%fZ}-{'a' * 32}.json.gz"
    (tmp_path / name).write_bytes(b"expired unreadable archive")
    assert load_predictions(tmp_path, NOW, min_hour=TARGET) == ({}, 0)


def test_owned_filename_must_agree_with_publication_time(tmp_path):
    path, _ = archive(tmp_path, TARGET - timedelta(hours=1))
    wrong_time = (TARGET - timedelta(hours=2)).astimezone(UTC)
    path.rename(tmp_path / f"forecast-{wrong_time:%Y%m%dT%H%M%S%fZ}-{'a' * 32}.json.gz")
    assert load_predictions(tmp_path, NOW) == ({}, 1)


def test_symlinks_are_ignored(tmp_path):
    elsewhere = tmp_path / "outside"
    elsewhere.mkdir()
    path, _ = archive(elsewhere, TARGET - timedelta(hours=1))
    (tmp_path / "forecast-symlink.json.gz").symlink_to(path)
    assert load_predictions(tmp_path, NOW) == ({}, 0)


@pytest.mark.parametrize("value", [None, -1, True, "100", float("nan"), float("inf"), 10 ** 500])
def test_invalid_prediction_counts_are_not_coerced(tmp_path, value):
    path, payload = archive(tmp_path, TARGET - timedelta(hours=1))
    payload["facilities"][0]["weeklyForecast"][0]["totalHours"][0]["expectedCount"] = value
    save(path, payload)
    assert load_predictions(tmp_path, NOW) == ({}, 1)


def test_impossible_dst_local_hour_is_rejected(tmp_path):
    start = datetime.fromisoformat("2026-03-08T02:00:00-06:00")
    archive(tmp_path, start - timedelta(hours=1), start=start)
    assert load_predictions(tmp_path, NOW) == ({}, 1)


def test_damaged_deflate_stream_does_not_block_other_archives(tmp_path):
    archive(tmp_path, TARGET - timedelta(hours=1))
    (tmp_path / 'forecast-corrupt.json.gz').write_bytes(
        bytes.fromhex('1f8b0800000000000003') + b'\xff' * 20)
    predictions, invalid = load_predictions(tmp_path, NOW)
    assert invalid == 1
    assert evaluate_hour(predictions, 1186, TARGET, actual(), NOW)[0]['status'] == 'scored'
