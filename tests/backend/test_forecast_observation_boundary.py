"""Synthetic history boundary checks; never connect or train."""
from datetime import datetime, timedelta, timezone

import pytest
import pytz

from server.reclive.forecasting import config, data, features


class HistoryConnection:
    def __init__(self, rows, cutover):
        self.rows, self.cutover, self.queries = rows, cutover, []

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params=()):
        self.queries.append((sql, params))

    def fetchone(self):
        return (self.cutover,) if self.cutover else None

    def fetchall(self):
        return self.rows


@pytest.fixture(autouse=True)
def isolated_history(monkeypatch):
    monkeypatch.setattr(config, "DB_TZ", pytz.timezone("America/Chicago"))
    monkeypatch.setattr(config, "TZ", pytz.timezone("America/Chicago"))
    monkeypatch.setattr(config, "HISTORY_DAYS", 0)
    monkeypatch.setattr(config, "RESAMPLE_MINUTES", 15)
    monkeypatch.setattr(features, "drop_impossible_jumps", lambda rows, **kwargs: (rows, 0))
    monkeypatch.setattr(features, "drop_flatline_plateaus", lambda rows, **kwargs: (rows, 0, 0))


@pytest.mark.parametrize("month,local_hour", [(1, 6), (7, 7)])
@pytest.mark.parametrize("source", ["unchanged", "missing", "aware"])
def test_trusted_fetch_is_observation_identity_and_reporting_availability(month, local_hour, source):
    fetched = datetime(2026, month, 15, 12)
    observed = None if source == "missing" else fetched - timedelta(hours=1)
    if source == "aware":
        fetched = fetched.replace(tzinfo=timezone.utc)
        observed = observed.replace(tzinfo=timezone.utc)
    conn = HistoryConnection([(11, observed, fetched, 20, False, 100),
                              (11, observed, fetched + timedelta(minutes=15), 40, False, 100)],
                             datetime(2026, month, 1))
    location = data.load_history(conn)[0][11]
    assert [(t.hour, t.minute) for t in location["raw_times"]] == [(local_hour, 0), (local_hour, 15)]
    assert location["raw_values"] == [20, 40]
    assert location["reporting_time_aligned"] is True
    for row, minute in zip(location["reporting_raw_baseline"], [0, 15]):
        assert row.observed_at == datetime(2026, month, 15, 12, minute, tzinfo=timezone.utc)
        assert row.available_at == row.observed_at


@pytest.mark.parametrize("cutover", [None, datetime(2026, 2, 1)])
def test_legacy_history_keeps_source_first_database_timezone(cutover):
    conn = HistoryConnection([(11, datetime(2026, 1, 15, 6), datetime(2026, 1, 15, 7), 20, False, 100)], cutover)
    location = data.load_history(conn)[0][11]
    assert location["raw_times"][0].hour == 6
    assert location["reporting_time_aligned"] is False


def test_history_cutoff_binds_utc_for_trusted_and_local_for_legacy(monkeypatch):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 15, 12, tzinfo=timezone.utc).astimezone(tz)
    monkeypatch.setattr(data, "datetime", FixedDatetime)
    monkeypatch.setattr(config, "HISTORY_DAYS", 1)
    conn = HistoryConnection([], datetime(2026, 1, 1))
    data.load_history(conn)
    boundary_sql, _ = conn.queries[0]
    assert "ORDER BY started_at, id LIMIT 1" in boundary_sql
    sql, params = conn.queries[-1]
    assert "fetched_at < %s" in sql and "fetched_at >= %s" in sql
    assert datetime(2026, 1, 14, 12) in params
    assert datetime(2026, 1, 14, 6) in params


@pytest.mark.parametrize("month,day,expected_legacy", [
    (3, 9, datetime(2026, 3, 7, 7)),
    (11, 2, datetime(2026, 10, 31, 6)),
])
@pytest.mark.parametrize("cutover", [None, datetime(2026, 1, 1)])
def test_legacy_cutoff_keeps_wall_time_subtraction_across_dst(monkeypatch, month, day, expected_legacy, cutover):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, month, day, 12, tzinfo=timezone.utc).astimezone(tz)
    monkeypatch.setattr(data, "datetime", FixedDatetime)
    monkeypatch.setattr(config, "HISTORY_DAYS", 2)
    conn = HistoryConnection([], cutover)
    data.load_history(conn)
    params = conn.queries[-1][1]
    assert expected_legacy in params
