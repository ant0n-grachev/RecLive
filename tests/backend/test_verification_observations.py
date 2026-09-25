from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from server.reclive import db, facility_schedule
from server.reclive.forecasting.verification_observations import (
    _hour_status,
    _schedule_sections,
    collect_actual_hours,
)
from server.reclive.actual_hours import CHICAGO, build_chicago_hour_windows
from server.reclive.settings import Settings
from tests.fixtures.health_payloads import healthy_schedule_payload
from tests.fixtures.reclive_fakes import ActualHourSqlConnection, ActualHourSqlCursor


UTC = timezone.utc
NOW = datetime(2026, 8, 31, 13, 7, tzinfo=UTC)


class ReadOnlyCursor(ActualHourSqlCursor):
    def execute(self, statement, parameters=()):
        if statement.startswith(("SET ", "START TRANSACTION")):
            self.connection.controls.append(statement)
            return 0
        assert self.connection.controls[-1] == "START TRANSACTION READ ONLY"
        assert statement.startswith("SELECT ")
        return super().execute(statement, parameters)


class ReadOnlyConnection(ActualHourSqlConnection):
    def __init__(self):
        super().__init__()
        self.controls = []
        self.rollbacks = 0
        self.history_rows = [
            (5761, False, 20, 100, datetime(2026, 8, 31, 11), 1),
            (5761, False, 60, 100, datetime(2026, 8, 31, 11, 30), 2),
        ]
        self.runs = [
            {
                "id": 1,
                "status": "succeeded",
                "started_at": datetime(2026, 8, 31, 11),
                "completed_at": datetime(2026, 8, 31, 11, 30),
                "observed_location_ids": "[5761]",
            },
            {
                "id": 2,
                "status": "succeeded",
                "started_at": datetime(2026, 8, 31, 13, 2),
                "completed_at": datetime(2026, 8, 31, 13, 3),
                "observed_location_ids": "[5761]",
            },
        ]

    def cursor(self):
        return ReadOnlyCursor(self)

    def rollback(self):
        self.rollbacks += 1


@pytest.fixture
def setup_observations(tmp_path, monkeypatch):
    schedule_path = tmp_path / "hours.json"
    schedule = healthy_schedule_payload()
    schedule_path.write_text(json.dumps(schedule))
    settings = replace(
        Settings.for_test(),
        facility_hours_json_path=str(schedule_path),
        forecast_json_path=str(tmp_path / "does-not-exist.json"),
        capacities={5761: 100},
        facility_names={1186: "Nick"},
        section_ids={1186: {"overall": (5761,)}},
    )
    connection = ReadOnlyConnection()
    calls = []

    def connect(database, *, autocommit):
        assert database is settings.database
        assert autocommit is False
        calls.append(True)
        return connection

    monkeypatch.setattr(db, "open_db_connection", connect)
    return settings, schedule, schedule_path, connection, calls


def row_at(rows, hour):
    return next(row for row in rows if row["hourStart"][11:13] == hour)


def test_integrates_historical_counts_without_current_forecast_and_rolls_back(
    setup_observations,
):
    settings, _, _, connection, calls = setup_observations
    rows = collect_actual_hours(settings, ["2026-08-31", "2026-08-31"], NOW)
    assert len(rows) == 8
    row = row_at(rows, "06")
    assert row["facilityId"] == 1186
    assert row["observationStatus"] == "ready"
    assert row["actualCount"] == row["observedCount"] == 40
    assert row["actualPct"] == 0.4
    assert row["actualCoverage"] == row["temporalCoverage"] == 1.0
    assert row["expectedCapacity"] == 100
    assert row["coverageThreshold"] == 0.75
    assert row_at(rows, "07")["actualCount"] == 60
    assert row_at(rows, "05")["observationStatus"] == "closed"
    assert len(calls) == connection.rollbacks == connection.close_count == 1
    assert connection.controls[-1] == "START TRANSACTION READ ONLY"


def test_low_temporal_coverage_is_not_scored_as_zero(setup_observations):
    settings, _, _, connection, _ = setup_observations
    connection.runs = connection.runs[:1]
    row = row_at(collect_actual_hours(settings, ["2026-08-31"], NOW), "06")
    assert row["actualCount"] is None
    assert row["observedCount"] == 20
    assert row["temporalCoverage"] == 0.5
    assert row["observationStatus"] == "insufficient_coverage"


def test_partial_facility_coverage_is_not_a_complete_headcount(setup_observations):
    settings, _, _, connection, _ = setup_observations
    settings = replace(
        settings,
        capacities={5761: 100, 5762: 100},
        section_ids={1186: {"overall": (5761, 5762)}},
    )
    row = row_at(collect_actual_hours(settings, ["2026-08-31"], NOW), "06")
    assert row["observedCount"] == 40
    assert row["actualCount"] is None
    assert row["actualCoverage"] == 0.5
    assert row["temporalCoverage"] == 1.0
    assert row["expectedCapacity"] == 200
    assert row["observationStatus"] == "insufficient_coverage"
    assert connection.rollbacks == connection.close_count == 1


def test_historical_capacity_above_configuration_clamps_coverage_not_people(
    setup_observations,
):
    settings, _, _, connection, _ = setup_observations
    connection.history_rows = [(5761, False, 110, 120, datetime(2026, 8, 31, 11), 1)]
    row = row_at(collect_actual_hours(settings, ["2026-08-31"], NOW), "06")
    assert row["actualCoverage"] == 1.0
    assert row["observedCapacity"] == 120
    assert row["actualCount"] == 110
    assert row["actualPct"] == 1.0
    assert row["observationStatus"] == "ready"


@pytest.mark.parametrize(
    "facility_id,date_key,hour,status",
    [
        (1186, "2026-09-24", 0, "closed"),
        (1186, "2026-09-24", 6, "ready"),
        (1186, "2026-09-24", 12, "ready"),
        (1186, "2026-09-24", 23, "ready"),
        (1656, "2026-09-25", 22, "closed"),
        (1186, "2026-09-26", 7, "closed"),
        (1186, "2026-09-26", 8, "ready"),
        (1656, "2026-09-27", 23, "ready"),
        (1186, "2026-11-25", 18, "partial_open"),
        (1186, "2026-11-26", 12, "closed"),
        (1656, "2026-11-26", 12, "closed"),
        (1186, "2026-11-28", 12, "closed"),
        (1656, "2026-11-28", 12, "ready"),
        (1186, "2026-12-13", 12, "ready"),
        (1186, "2026-12-13", 22, "closed"),
        (1186, "2027-01-05", 12, "closed"),
        (1656, "2026-05-13", 12, "closed"),
        (1656, "2026-05-18", 12, "ready"),
    ],
)
def test_full_public_production_schedule(facility_id, date_key, hour, status):
    path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "verification_facility_hours.json"
    )
    settings = replace(Settings.for_test(), facility_hours_json_path=str(path))
    schedules = _schedule_sections(settings, datetime(2026, 9, 25, 2, tzinfo=UTC))
    window = next(
        window
        for window in build_chicago_hour_windows(date_key)
        if window.start.astimezone(CHICAGO).hour == hour
    )
    assert _hour_status(schedules.get(facility_id), window) == status


@pytest.mark.parametrize(
    "title", ["Pool", "Sub Zero Ice Rink", "Unrelated Event", "Seasonal Notice"]
)
def test_unrelated_section_cannot_close_facility(setup_observations, title):
    settings, schedule, path, _, _ = setup_observations
    schedule["facilities"][0]["sections"].append(
        {
            "title": title,
            "rows": [{"label": "August 31", "hours": "CLOSED"}],
            "note": None,
        }
    )
    path.write_text(json.dumps(schedule))
    row = row_at(collect_actual_hours(settings, ["2026-08-31"], NOW), "06")
    assert row["observationStatus"] == "ready"
    assert row["actualCount"] == 40


def test_unknown_section_alone_is_not_promoted_to_building_hours(setup_observations):
    settings, schedule, path, _, calls = setup_observations
    schedule["facilities"][0]["sections"][0]["title"] = "Unrelated Event"
    path.write_text(json.dumps(schedule))
    rows = collect_actual_hours(settings, ["2026-08-31"], NOW)
    assert {row["observationStatus"] for row in rows} == {"schedule_unavailable"}
    assert calls == []


def _minute_reference_status(schedule, window):
    states = {
        facility_schedule.get_facility_schedule_open_state(
            schedule, window.start + timedelta(minutes=minute)
        )
        for minute in range(60)
    }
    if None in states:
        return "schedule_unavailable"
    if states == {True}:
        return "ready"
    return "closed" if states == {False} else "partial_open"


@pytest.mark.parametrize(
    "date_key", ["2026-09-24", "2026-11-25", "2026-11-28", "2026-12-13", "2027-01-05"]
)
def test_checkpoint_schedule_equals_all_minutes_for_full_production_fixture(date_key):
    path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "verification_facility_hours.json"
    )
    settings = replace(Settings.for_test(), facility_hours_json_path=str(path))
    schedules = _schedule_sections(settings, datetime(2026, 9, 25, 2, tzinfo=UTC))
    for schedule in schedules.values():
        for window in build_chicago_hour_windows(date_key):
            assert _hour_status(schedule, window) == _minute_reference_status(
                schedule, window
            )


def test_short_open_interval_inside_hour_is_not_missed_and_uses_only_boundaries(
    monkeypatch,
):
    schedule = [
        {
            "title": "Building Hours",
            "rows": [{"label": "Daily", "hours": "10:15 am - 10:45 am"}],
        }
    ]
    window = next(
        w
        for w in build_chicago_hour_windows("2026-09-24")
        if w.start.astimezone(CHICAGO).hour == 10
    )
    original = facility_schedule.get_facility_schedule_open_state
    calls = []

    def record(schedule, at):
        calls.append(at)
        return original(schedule, at)

    monkeypatch.setattr(facility_schedule, "get_facility_schedule_open_state", record)
    assert _hour_status(schedule, window) == "partial_open"
    assert [at.astimezone(CHICAGO).minute for at in calls] == [0, 15, 45]


@pytest.mark.parametrize("date_key", ["2026-03-08", "2026-11-01", "2026-09-24"])
def test_checkpoint_schedule_equals_all_minutes_for_overnight_and_dst(date_key):
    schedule = [
        {
            "title": "Building Hours",
            "rows": [{"label": "Daily", "hours": "10:15 pm - 1:45 am"}],
        }
    ]
    for window in build_chicago_hour_windows(date_key):
        assert _hour_status(schedule, window) == _minute_reference_status(
            schedule, window
        )


def test_observed_zero_remains_valid_and_missing_data_is_distinct(setup_observations):
    settings, _, _, connection, _ = setup_observations
    connection.history_rows = [(5761, False, 0, 100, datetime(2026, 8, 31, 11), 1)]
    row = row_at(collect_actual_hours(settings, ["2026-08-31"], NOW), "06")
    assert row["actualCount"] == 0
    assert row["observationStatus"] == "ready"
    connection.history_rows = []
    row = row_at(collect_actual_hours(settings, ["2026-08-31"], NOW), "06")
    assert row["actualCount"] is None
    assert row["observationStatus"] == "missing_data"


@pytest.mark.parametrize(
    "hours,partial_hour", [("6:30 am - 10:00 pm", "06"), ("6:00 am - 7:30 am", "07")]
)
def test_partial_opening_or_closing_hour_is_explicitly_excluded(
    setup_observations, hours, partial_hour
):
    settings, schedule, path, _, _ = setup_observations
    schedule["facilities"][0]["sections"][0]["rows"][0]["hours"] = hours
    path.write_text(json.dumps(schedule))
    row = row_at(collect_actual_hours(settings, ["2026-08-31"], NOW), partial_hour)
    assert row["observationStatus"] == "partial_open"
    assert row["actualCount"] is None
    assert row["skipReason"]


@pytest.mark.parametrize("mode", ["missing", "malformed", "stale"])
def test_unknown_schedule_is_explicit_and_does_not_query_database(
    setup_observations, mode
):
    settings, _, path, _, calls = setup_observations
    if mode == "missing":
        path.unlink()
    elif mode == "malformed":
        path.write_text("{}")
    else:
        settings = replace(settings, schedule_stale_after_seconds=60)
    rows = collect_actual_hours(settings, ["2026-08-31"], NOW)
    assert rows and {row["observationStatus"] for row in rows} == {
        "schedule_unavailable"
    }
    assert all(row["actualCount"] is None and row["skipReason"] for row in rows)
    assert calls == []


def test_database_failure_rolls_back_and_closes(setup_observations):
    settings, _, _, connection, _ = setup_observations
    connection.query_failure = RuntimeError("database unavailable")
    with pytest.raises(RuntimeError, match="database unavailable"):
        collect_actual_hours(settings, ["2026-08-31"], NOW)
    assert connection.rollbacks == connection.close_count == 1


def test_failure_to_start_read_only_transaction_never_queries_data(
    setup_observations, monkeypatch
):
    settings, _, _, connection, _ = setup_observations
    original_execute = ReadOnlyCursor.execute

    def execute(cursor, statement, parameters=()):
        if statement == "START TRANSACTION READ ONLY":
            raise RuntimeError("read-only transaction unavailable")
        return original_execute(cursor, statement, parameters)

    monkeypatch.setattr(ReadOnlyCursor, "execute", execute)
    with pytest.raises(RuntimeError, match="read-only transaction unavailable"):
        collect_actual_hours(settings, ["2026-08-31"], NOW)
    assert connection.queries == []
    assert connection.rollbacks == connection.close_count == 1


@pytest.mark.parametrize("date_key,expected", [("2026-03-08", 23), ("2026-11-01", 25)])
def test_dst_day_windows_preserve_distinct_offset_keys(
    setup_observations, date_key, expected
):
    settings, _, path, _, calls = setup_observations
    path.unlink()
    rows = collect_actual_hours(
        settings, [date_key], datetime(2026, 11, 2, 12, tzinfo=UTC)
    )
    assert len(rows) == expected
    assert len({row["hourStart"] for row in rows}) == expected
    assert calls == []


def test_rejects_naive_now_and_bad_date_before_database_io(setup_observations):
    settings, _, _, _, calls = setup_observations
    with pytest.raises(ValueError, match="aware"):
        collect_actual_hours(settings, ["2026-08-31"], NOW.replace(tzinfo=None))
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        collect_actual_hours(settings, ["2026-08-00"], NOW)
    assert calls == []


def test_future_days_and_empty_dates_need_no_database_io(setup_observations):
    settings, _, _, _, calls = setup_observations
    assert collect_actual_hours(settings, [], NOW) == []
    assert collect_actual_hours(settings, ["2026-09-01"], NOW) == []
    assert calls == []
