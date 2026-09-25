from datetime import datetime, timezone

from freezegun import freeze_time

from tests.fixtures.live_counts import LIVE_ROWS


@freeze_time("2026-08-31 12:00:00")
def test_fixture_rows_and_frozen_clock_are_deterministic() -> None:
    assert LIVE_ROWS[0]["LocationId"] == 5761
    assert datetime.now(timezone.utc).isoformat() == "2026-08-31T12:00:00+00:00"
