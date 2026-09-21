from server.reclive.api import forecasts as _seam_api_forecasts
from server.reclive import db as _seam_db
from server.reclive import runtime as _seam_runtime

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient
import pymysql
import pytest

import forecast_api
from reclive.actual_hours import (
    HistoryState,
    HourWindow,
    IngestionHeartbeat,
    build_chicago_hour_windows,
    calculate_actual_hour,
)
from reclive.database_dialect import detect_database_dialect
from reclive.migrations import execution_snapshot, snapshot_migration, split_statements
from reclive.occupancy_repository import SnapshotRepository
from tests.fixtures.reclive_fakes import ActualHourSqlConnection


UTC = timezone.utc
CHICAGO = ZoneInfo("America/Chicago")


def utc_time(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 8, 31, hour, minute, tzinfo=UTC)


def hour_window() -> HourWindow:
    return HourWindow(utc_time(12), utc_time(13))


def state(
    *,
    event_id: int,
    count: int,
    fetched_at: datetime,
    location_id: int = 5761,
    capacity: int = 100,
    is_closed: bool = False,
) -> HistoryState:
    return HistoryState(
        location_id=location_id,
        is_closed=is_closed,
        count=count,
        capacity=capacity,
        fetched_at=fetched_at,
        id=event_id,
    )


def heartbeat(
    completed_at: datetime,
    location_ids: frozenset[int] = frozenset({5761}),
) -> IngestionHeartbeat:
    return IngestionHeartbeat(
        completed_at=completed_at,
        observed_location_ids=location_ids,
    )


def test_integrates_step_changes_over_one_hour() -> None:
    summary = calculate_actual_hour(
        [5761],
        100,
        hour_window(),
        [
            state(event_id=1, count=20, fetched_at=utc_time(11, 50)),
            state(event_id=2, count=60, fetched_at=utc_time(12, 30)),
        ],
        [heartbeat(utc_time(12, 30)), heartbeat(utc_time(13))],
        0.75,
    )

    assert summary.observed_count == 40
    assert summary.observed_capacity == 100
    assert summary.expected_capacity == 100
    assert summary.actual_coverage == 1.0
    assert summary.temporal_coverage == 1.0
    assert summary.coverage_threshold == 0.75
    assert summary.actual_count == 40


def test_latest_pre_range_state_seeds_a_constant_hour() -> None:
    summary = calculate_actual_hour(
        [5761],
        100,
        hour_window(),
        [
            state(event_id=1, count=10, fetched_at=utc_time(10)),
            state(event_id=2, count=42, fetched_at=utc_time(11, 59)),
        ],
        [heartbeat(utc_time(13))],
        0.75,
    )

    assert summary.observed_count == 42
    assert summary.observed_capacity == 100
    assert summary.actual_coverage == 1.0
    assert summary.temporal_coverage == 1.0
    assert summary.actual_count == 42


def test_no_history_is_unknown_even_with_a_heartbeat() -> None:
    summary = calculate_actual_hour(
        [5761],
        100,
        hour_window(),
        [],
        [heartbeat(utc_time(13))],
        0.75,
    )

    assert summary.observed_count is None
    assert summary.observed_capacity == 0
    assert summary.actual_coverage == 0.0
    assert summary.temporal_coverage == 0.0
    assert summary.actual_count is None


def test_history_without_a_successful_heartbeat_is_unknown() -> None:
    summary = calculate_actual_hour(
        [5761],
        100,
        hour_window(),
        [state(event_id=1, count=42, fetched_at=utc_time(11, 59))],
        [],
        0.75,
    )

    assert summary.observed_count is None
    assert summary.observed_capacity == 0
    assert summary.actual_coverage == 0.0
    assert summary.temporal_coverage == 0.0
    assert summary.actual_count is None


def test_heartbeat_for_another_location_does_not_extend_trust() -> None:
    summary = calculate_actual_hour(
        [5761],
        100,
        hour_window(),
        [state(event_id=1, count=42, fetched_at=utc_time(11, 59))],
        [heartbeat(utc_time(13), frozenset({9999}))],
        0.75,
    )

    assert summary.observed_count is None
    assert summary.observed_capacity == 0
    assert summary.temporal_coverage == 0.0
    assert summary.actual_count is None


def test_later_heartbeat_extends_an_unchanged_state_past_event_time() -> None:
    summary = calculate_actual_hour(
        [5761],
        100,
        hour_window(),
        [state(event_id=1, count=42, fetched_at=utc_time(12))],
        [heartbeat(utc_time(12)), heartbeat(utc_time(12, 45))],
        0.75,
    )

    assert summary.observed_count == 42
    assert summary.observed_capacity == 100
    assert summary.actual_coverage == 1.0
    assert summary.temporal_coverage == 0.75
    assert summary.actual_count == 42


def test_boundary_heartbeat_confirms_only_the_state_ending_at_boundary() -> None:
    summary = calculate_actual_hour(
        [5761],
        100,
        hour_window(),
        [
            state(event_id=1, count=20, fetched_at=utc_time(11, 50)),
            state(event_id=2, count=60, fetched_at=utc_time(12, 30)),
        ],
        [heartbeat(utc_time(12, 30))],
        0.75,
    )

    assert summary.observed_count == 20
    assert summary.observed_capacity == 100
    assert summary.actual_coverage == 1.0
    assert summary.temporal_coverage == 0.5
    assert summary.actual_count is None


def test_heartbeat_after_next_event_does_not_extend_the_previous_state() -> None:
    summary = calculate_actual_hour(
        [5761],
        100,
        hour_window(),
        [
            state(event_id=1, count=10, fetched_at=utc_time(11, 50)),
            state(event_id=2, count=70, fetched_at=utc_time(12, 30)),
        ],
        [heartbeat(utc_time(12, 20)), heartbeat(utc_time(13))],
        0.75,
    )

    assert summary.observed_count == 46
    assert summary.observed_capacity == 100
    assert summary.actual_coverage == 1.0
    assert summary.temporal_coverage == pytest.approx(5 / 6)
    assert summary.actual_count == 46


def test_state_at_window_start_replaces_the_seed_for_the_full_hour() -> None:
    summary = calculate_actual_hour(
        [5761],
        100,
        hour_window(),
        [
            state(event_id=1, count=10, fetched_at=utc_time(11, 59)),
            state(event_id=2, count=30, fetched_at=utc_time(12)),
        ],
        [heartbeat(utc_time(13))],
        0.75,
    )

    assert summary.observed_count == 30
    assert summary.temporal_coverage == 1.0
    assert summary.actual_count == 30


def test_state_at_window_end_does_not_change_the_half_open_hour() -> None:
    summary = calculate_actual_hour(
        [5761],
        100,
        hour_window(),
        [
            state(event_id=1, count=10, fetched_at=utc_time(11, 59)),
            state(event_id=2, count=90, fetched_at=utc_time(13)),
        ],
        [heartbeat(utc_time(13))],
        0.75,
    )

    assert summary.observed_count == 10
    assert summary.temporal_coverage == 1.0
    assert summary.actual_count == 10


def test_higher_id_wins_an_equal_timestamp_regardless_of_input_order() -> None:
    summary = calculate_actual_hour(
        [5761],
        100,
        hour_window(),
        [
            state(event_id=20, count=60, fetched_at=utc_time(12)),
            state(event_id=10, count=20, fetched_at=utc_time(12)),
        ],
        [heartbeat(utc_time(13))],
        0.75,
    )

    assert summary.observed_count == 60
    assert summary.temporal_coverage == 1.0
    assert summary.actual_count == 60


def test_duplicate_history_event_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="history event IDs must be unique"):
        calculate_actual_hour(
            [5761],
            100,
            hour_window(),
            [
                state(event_id=1, count=10, fetched_at=utc_time(11, 59)),
                state(event_id=1, count=20, fetched_at=utc_time(12, 30)),
            ],
            [heartbeat(utc_time(13))],
            0.75,
        )


def test_low_capacity_coverage_never_scales_observed_count() -> None:
    summary = calculate_actual_hour(
        [5761],
        200,
        hour_window(),
        [state(event_id=1, count=42, fetched_at=utc_time(11, 59))],
        [heartbeat(utc_time(13))],
        0.75,
    )

    assert summary.observed_count == 42
    assert summary.observed_capacity == 100
    assert summary.expected_capacity == 200
    assert summary.actual_coverage == 0.5
    assert summary.temporal_coverage == 1.0
    assert summary.actual_count is None


def test_capacity_and_temporal_coverage_are_independent() -> None:
    summary = calculate_actual_hour(
        [5761],
        200,
        hour_window(),
        [state(event_id=1, count=42, fetched_at=utc_time(11, 59))],
        [heartbeat(utc_time(12, 57))],
        0.75,
    )

    assert summary.observed_count == 42
    assert summary.observed_capacity == 100
    assert summary.actual_coverage == 0.5
    assert summary.temporal_coverage == pytest.approx(0.95)
    assert summary.actual_count is None


def test_temporal_coverage_below_threshold_withholds_actual_count() -> None:
    summary = calculate_actual_hour(
        [5761],
        100,
        hour_window(),
        [state(event_id=1, count=42, fetched_at=utc_time(12))],
        [heartbeat(utc_time(12, 44))],
        0.75,
    )

    assert summary.observed_count == 42
    assert summary.actual_coverage == 1.0
    assert summary.temporal_coverage == pytest.approx(11 / 15)
    assert summary.actual_count is None


def test_both_coverages_qualify_at_the_inclusive_threshold() -> None:
    summary = calculate_actual_hour(
        [5761],
        100,
        hour_window(),
        [state(event_id=1, count=42, fetched_at=utc_time(12))],
        [heartbeat(utc_time(12, 45))],
        0.75,
    )

    assert summary.actual_coverage == 1.0
    assert summary.temporal_coverage == 0.75
    assert summary.actual_count == 42


def test_exact_temporal_threshold_qualifies_despite_float_noise() -> None:
    transition = utc_time(12) + timedelta(seconds=1234, microseconds=1)
    summary = calculate_actual_hour(
        [5761],
        4,
        hour_window(),
        [
            state(
                event_id=1,
                count=1,
                capacity=1,
                fetched_at=utc_time(12),
            ),
            state(
                event_id=2,
                count=7,
                capacity=7,
                fetched_at=transition,
            ),
        ],
        [heartbeat(transition), heartbeat(utc_time(12, 45))],
        0.75,
    )

    assert summary.observed_count == 4
    assert summary.observed_capacity == 4
    assert summary.actual_coverage == 1.0
    assert summary.temporal_coverage == pytest.approx(0.75)
    assert summary.actual_count == 4


def test_temporal_threshold_tolerance_rejects_one_microsecond_short() -> None:
    transition = utc_time(12) + timedelta(seconds=1234, microseconds=1)
    summary = calculate_actual_hour(
        [5761],
        4,
        hour_window(),
        [
            state(
                event_id=1,
                count=1,
                capacity=1,
                fetched_at=utc_time(12),
            ),
            state(
                event_id=2,
                count=7,
                capacity=7,
                fetched_at=transition,
            ),
        ],
        [
            heartbeat(transition),
            heartbeat(utc_time(12, 45) - timedelta(microseconds=1)),
        ],
        0.75,
    )

    assert summary.temporal_coverage < 0.75
    assert summary.actual_count is None


def test_per_location_averages_are_summed_without_zero_filling_unknown_time() -> None:
    summary = calculate_actual_hour(
        [5761, 5762],
        200,
        hour_window(),
        [
            state(event_id=1, count=20, fetched_at=utc_time(11, 59)),
            state(
                event_id=2,
                location_id=5762,
                count=60,
                fetched_at=utc_time(12),
            ),
        ],
        [
            heartbeat(utc_time(13), frozenset({5761})),
            heartbeat(utc_time(12, 30), frozenset({5762})),
        ],
        0.75,
    )

    assert summary.observed_count == 80
    assert summary.observed_capacity == 200
    assert summary.actual_coverage == 1.0
    assert summary.temporal_coverage == 0.75
    assert summary.actual_count == 80


def test_capacity_step_changes_are_time_weighted_without_ratio_scaling() -> None:
    summary = calculate_actual_hour(
        [5761],
        200,
        hour_window(),
        [
            state(
                event_id=1,
                count=40,
                capacity=100,
                fetched_at=utc_time(11, 59),
            ),
            state(
                event_id=2,
                count=80,
                capacity=200,
                fetched_at=utc_time(12, 30),
            ),
        ],
        [heartbeat(utc_time(12, 30)), heartbeat(utc_time(13))],
        0.75,
    )

    assert summary.observed_count == 60
    assert summary.observed_capacity == 150
    assert summary.actual_coverage == 0.75
    assert summary.temporal_coverage == 1.0
    assert summary.actual_count == 60


def test_confirmed_closed_hour_is_unknown_not_observed_zero() -> None:
    summary = calculate_actual_hour(
        [5761],
        100,
        hour_window(),
        [
            state(
                event_id=1,
                count=0,
                fetched_at=utc_time(11, 59),
                is_closed=True,
            )
        ],
        [heartbeat(utc_time(13))],
        0.75,
    )

    assert summary.observed_count is None
    assert summary.observed_capacity == 0
    assert summary.expected_capacity == 100
    assert summary.actual_coverage == 0.0
    assert summary.temporal_coverage == 0.0
    assert summary.actual_count is None


def test_closed_segment_does_not_dilute_open_count_or_expected_capacity() -> None:
    summary = calculate_actual_hour(
        [5761],
        100,
        hour_window(),
        [
            state(event_id=1, count=42, fetched_at=utc_time(11, 59)),
            state(
                event_id=2,
                count=0,
                fetched_at=utc_time(12, 30),
                is_closed=True,
            ),
        ],
        [heartbeat(utc_time(12, 30)), heartbeat(utc_time(13))],
        0.75,
    )

    assert summary.observed_count == 42
    assert summary.observed_capacity == 100
    assert summary.expected_capacity == 100
    assert summary.actual_coverage == 1.0
    assert summary.temporal_coverage == 0.5
    assert summary.actual_count is None


def test_closed_location_never_reduces_full_facility_expected_capacity() -> None:
    summary = calculate_actual_hour(
        [5761, 5762],
        200,
        hour_window(),
        [
            state(event_id=1, count=42, fetched_at=utc_time(11, 59)),
            state(
                event_id=2,
                location_id=5762,
                count=0,
                fetched_at=utc_time(11, 59),
                is_closed=True,
            ),
        ],
        [heartbeat(utc_time(13), frozenset({5761, 5762}))],
        0.75,
    )

    assert summary.observed_count == 42
    assert summary.observed_capacity == 100
    assert summary.expected_capacity == 200
    assert summary.actual_coverage == 0.5
    assert summary.temporal_coverage == 1.0
    assert summary.actual_count is None


def test_zero_capacity_state_contributes_no_observation() -> None:
    summary = calculate_actual_hour(
        [5761],
        100,
        hour_window(),
        [
            state(
                event_id=1,
                count=0,
                capacity=0,
                fetched_at=utc_time(11, 59),
            )
        ],
        [heartbeat(utc_time(13))],
        0.75,
    )

    assert summary.observed_count is None
    assert summary.observed_capacity == 0
    assert summary.actual_coverage == 0.0
    assert summary.temporal_coverage == 0.0
    assert summary.actual_count is None


def test_empty_location_set_returns_a_conservative_unknown_summary() -> None:
    summary = calculate_actual_hour(
        [],
        100,
        hour_window(),
        [],
        [],
        0.75,
    )

    assert summary.observed_count is None
    assert summary.observed_capacity == 0
    assert summary.expected_capacity == 100
    assert summary.actual_coverage == 0.0
    assert summary.temporal_coverage == 0.0
    assert summary.actual_count is None


def test_duplicate_requested_location_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="location IDs must be unique"):
        calculate_actual_hour(
            [5761, 5761],
            100,
            hour_window(),
            [],
            [],
            0.75,
        )


@pytest.mark.parametrize("location_ids", [[True], [-1], [5761.0]])
def test_requested_location_ids_reject_bool_negative_and_non_integer_values(
    location_ids: list[object],
) -> None:
    with pytest.raises(ValueError, match="non-negative integers"):
        calculate_actual_hour(
            location_ids,  # type: ignore[arg-type]
            100,
            hour_window(),
            [],
            [],
            0.75,
        )


@pytest.mark.parametrize("expected_capacity", [True, -1, 100.0])
def test_expected_capacity_requires_a_non_negative_integer(
    expected_capacity: object,
) -> None:
    with pytest.raises(ValueError, match="expected capacity"):
        calculate_actual_hour(
            [],
            expected_capacity,  # type: ignore[arg-type]
            hour_window(),
            [],
            [],
            0.75,
        )


def test_zero_expected_capacity_never_qualifies_even_at_zero_threshold() -> None:
    summary = calculate_actual_hour(
        [5761],
        0,
        hour_window(),
        [state(event_id=1, count=42, fetched_at=utc_time(11, 59))],
        [heartbeat(utc_time(13))],
        0.0,
    )

    assert summary.observed_count == 42
    assert summary.observed_capacity == 100
    assert summary.actual_coverage == 0.0
    assert summary.temporal_coverage == 1.0
    assert summary.actual_count is None


@pytest.mark.parametrize(
    "coverage_threshold",
    [True, -0.01, 1.01, float("nan"), float("inf")],
)
def test_coverage_threshold_must_be_a_finite_unit_interval_number(
    coverage_threshold: object,
) -> None:
    with pytest.raises(ValueError, match="coverage threshold"):
        calculate_actual_hour(
            [],
            100,
            hour_window(),
            [],
            [],
            coverage_threshold,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("location_id", True, "location ID"),
        ("location_id", -1, "location ID"),
        ("is_closed", 1, "is_closed"),
        ("count", True, "count"),
        ("count", -1, "count"),
        ("capacity", True, "capacity"),
        ("capacity", -1, "capacity"),
        ("id", True, "event ID"),
        ("id", -1, "event ID"),
    ],
)
def test_history_state_rejects_invalid_scalar_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "location_id": 5761,
        "is_closed": False,
        "count": 42,
        "capacity": 100,
        "fetched_at": utc_time(12),
        "id": 1,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        HistoryState(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("observed_location_ids", [frozenset({True}), frozenset({-1})])
def test_heartbeat_rejects_invalid_observed_location_ids(
    observed_location_ids: frozenset[object],
) -> None:
    with pytest.raises(ValueError, match="observed location IDs"):
        IngestionHeartbeat(
            completed_at=utc_time(12),
            observed_location_ids=observed_location_ids,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_time",
    [
        datetime(2026, 8, 31, 12),
        datetime(
            2026,
            8,
            31,
            7,
            tzinfo=timezone(timedelta(hours=-5)),
        ),
    ],
)
def test_history_state_rejects_naive_and_non_utc_event_times(
    invalid_time: datetime,
) -> None:
    with pytest.raises(ValueError, match="aware UTC"):
        state(event_id=1, count=42, fetched_at=invalid_time)


@pytest.mark.parametrize(
    "invalid_time",
    [
        datetime(2026, 8, 31, 12),
        datetime(
            2026,
            8,
            31,
            7,
            tzinfo=timezone(timedelta(hours=-5)),
        ),
    ],
)
def test_heartbeat_rejects_naive_and_non_utc_times(
    invalid_time: datetime,
) -> None:
    with pytest.raises(ValueError, match="aware UTC"):
        heartbeat(invalid_time)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (datetime(2026, 8, 31, 12), utc_time(13)),
        (
            datetime(
                2026,
                8,
                31,
                7,
                tzinfo=timezone(timedelta(hours=-5)),
            ),
            utc_time(13),
        ),
        (utc_time(12), datetime(2026, 8, 31, 13)),
        (
            utc_time(12),
            datetime(
                2026,
                8,
                31,
                8,
                tzinfo=timezone(timedelta(hours=-5)),
            ),
        ),
    ],
)
def test_hour_window_rejects_naive_and_non_utc_boundaries(
    start: datetime,
    end: datetime,
) -> None:
    with pytest.raises(ValueError, match="aware UTC"):
        HourWindow(start, end)


@pytest.mark.parametrize(
    ("start", "end"),
    [(utc_time(12), utc_time(12)), (utc_time(13), utc_time(12))],
)
def test_hour_window_requires_strictly_increasing_boundaries(
    start: datetime,
    end: datetime,
) -> None:
    with pytest.raises(ValueError, match="end after start"):
        HourWindow(start, end)


def test_chicago_spring_day_has_23_contiguous_physical_hours() -> None:
    windows = build_chicago_hour_windows("2026-03-08")
    local_starts = [window.start.astimezone(CHICAGO) for window in windows]

    assert len(windows) == 23
    assert all(window.end - window.start == timedelta(hours=1) for window in windows)
    assert all(
        previous.end == following.start
        for previous, following in zip(windows, windows[1:])
    )
    assert [value.hour for value in local_starts[:4]] == [0, 1, 3, 4]
    assert all(value.hour != 2 for value in local_starts)
    assert local_starts[1].utcoffset() == timedelta(hours=-6)
    assert local_starts[2].utcoffset() == timedelta(hours=-5)


def test_chicago_fall_day_has_both_repeated_01_offsets() -> None:
    windows = build_chicago_hour_windows("2026-11-01")
    local_starts = [window.start.astimezone(CHICAGO) for window in windows]
    repeated_ones = [value for value in local_starts if value.hour == 1]

    assert len(windows) == 25
    assert all(window.end - window.start == timedelta(hours=1) for window in windows)
    assert all(
        previous.end == following.start
        for previous, following in zip(windows, windows[1:])
    )
    assert len(repeated_ones) == 2
    assert [value.utcoffset() for value in repeated_ones] == [
        timedelta(hours=-5),
        timedelta(hours=-6),
    ]


def test_chicago_ordinary_day_has_24_physical_hours() -> None:
    windows = build_chicago_hour_windows("2026-08-31")

    assert len(windows) == 24
    assert windows[0].start == datetime(2026, 8, 31, 5, tzinfo=UTC)
    assert windows[-1].end == datetime(2026, 9, 1, 5, tzinfo=UTC)


@pytest.mark.parametrize(
    "date_key",
    ["", "2026-02-30", "2026-3-08", "2026-03-08T00:00:00", 20260308],
)
def test_chicago_windows_reject_invalid_or_noncanonical_dates(
    date_key: object,
) -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        build_chicago_hour_windows(date_key)  # type: ignore[arg-type]


def load_repository_actual_hour_inputs(
    connection: ActualHourSqlConnection,
    location_ids: list[int],
    range_start: datetime,
    range_end: datetime,
):
    repository = SnapshotRepository(connection)
    loader = getattr(repository, "load_actual_hour_inputs", None)
    assert callable(loader), "actual-hour repository loader dependency is missing"
    return loader(location_ids, range_start, range_end)


def normalized_statements(connection: ActualHourSqlConnection) -> list[str]:
    return [" ".join(query.statement.split()) for query in connection.queries]


@pytest.mark.mysql
def test_mysql_repository_loads_only_deterministic_postcutover_actual_inputs(
    clean_test_database: dict[str, object],
) -> None:
    migrations = Path(__file__).resolve().parents[2] / "server" / "migrations"
    schema_connection = pymysql.connect(**clean_test_database)
    try:
        dialect = detect_database_dialect(schema_connection)
        with schema_connection.cursor() as cursor:
            for migration_name in (
                "0001_core_history.sql",
                "0002_snapshot_and_ingestion.sql",
            ):
                migration = execution_snapshot(
                    snapshot_migration(migrations / migration_name), dialect
                ).sql_bytes.decode("utf-8")
                for statement in split_statements(migration):
                    cursor.execute(statement)
    finally:
        schema_connection.close()

    connection = pymysql.connect(
        **{**clean_test_database, "autocommit": False}
    )
    try:
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO ingestion_runs("
                "id, started_at, completed_at, status, observed_location_ids"
                ") VALUES (%s, %s, %s, %s, %s)",
                # Higher ID 2 starts first; ID ordering would choose the wrong cutoff.
                [
                    (
                        1,
                        datetime(2026, 11, 1, 7),
                        datetime(2026, 11, 1, 8, 10),
                        "succeeded",
                        "[5761]",
                    ),
                    (
                        2,
                        datetime(2026, 11, 1, 6, 59),
                        datetime(2026, 11, 1, 8, 20),
                        "succeeded",
                        "[5762]",
                    ),
                    (
                        3,
                        datetime(2026, 11, 1, 6),
                        datetime(2026, 11, 1, 8, 30),
                        "failed",
                        "[5761, 5762]",
                    ),
                    (
                        4,
                        datetime(2026, 11, 1, 7, 5),
                        datetime(2026, 11, 1, 10),
                        "succeeded",
                        '[5761, true, -1, 0, "5762", 5764]',
                    ),
                    (
                        5,
                        datetime(2026, 11, 1, 7, 6),
                        datetime(2026, 11, 1, 10, 0, 0, 1),
                        "succeeded",
                        "[9999]",
                    ),
                    (
                        6,
                        datetime(2026, 11, 1, 7, 10),
                        datetime(2026, 11, 1, 7, 59, 59, 999999),
                        "succeeded",
                        "[9001]",
                    ),
                    (
                        7,
                        datetime(2026, 11, 1, 7, 11),
                        datetime(2026, 11, 1, 8),
                        "succeeded",
                        "[9002]",
                    ),
                ],
            )
            cursor.executemany(
                "INSERT INTO location_history("
                "id, location_id, is_closed, current_capacity, "
                "max_capacity, fetched_at"
                ") VALUES (%s, %s, %s, %s, %s, %s)",
                # Requested 5763 has only legacy/pre-cutover rows and must vanish.
                [
                    (1, 5763, False, 90, 100, datetime(2026, 11, 1, 1, 30)),
                    (2, 5763, False, 80, 100, datetime(2026, 11, 1, 6, 58)),
                    (10, 5761, False, 40, 100, datetime(2026, 11, 1, 7, 45)),
                    (20, 5761, False, 42, 100, datetime(2026, 11, 1, 7, 45)),
                    (
                        21,
                        5762,
                        False,
                        21,
                        100,
                        datetime(2026, 11, 1, 6, 59, 30),
                    ),
                    (25, 9999, False, 99, 100, datetime(2026, 11, 1, 7, 59)),
                    (30, 5762, False, 31, 100, datetime(2026, 11, 1, 8, 15)),
                    (40, 5761, False, 60, 100, datetime(2026, 11, 1, 8, 30)),
                    (41, 5761, False, 61, 100, datetime(2026, 11, 1, 8, 30)),
                    (50, 5761, False, 100, 100, datetime(2026, 11, 1, 10)),
                ],
            )
        connection.commit()

        states, heartbeats = SnapshotRepository(
            connection
        ).load_actual_hour_inputs(
            [5763, 5762, 5761],
            datetime(2026, 11, 1, 8, tzinfo=UTC),
            datetime(2026, 11, 1, 10, tzinfo=UTC),
        )

        assert [
            (
                item.location_id,
                item.count,
                item.fetched_at,
                item.id,
            )
            for item in states
        ] == [
            (5761, 42, datetime(2026, 11, 1, 7, 45, tzinfo=UTC), 20),
            (5762, 21, datetime(2026, 11, 1, 6, 59, 30, tzinfo=UTC), 21),
            (5761, 60, datetime(2026, 11, 1, 8, 30, tzinfo=UTC), 40),
            (5761, 61, datetime(2026, 11, 1, 8, 30, tzinfo=UTC), 41),
            (5762, 31, datetime(2026, 11, 1, 8, 15, tzinfo=UTC), 30),
        ]
        assert all(item.fetched_at.tzinfo is UTC for item in states)
        assert [item.completed_at for item in heartbeats] == [
            datetime(2026, 11, 1, 8, tzinfo=UTC),
            datetime(2026, 11, 1, 8, 10, tzinfo=UTC),
            datetime(2026, 11, 1, 8, 20, tzinfo=UTC),
            datetime(2026, 11, 1, 10, tzinfo=UTC),
        ]
        assert [item.observed_location_ids for item in heartbeats] == [
            frozenset({9002}),
            frozenset({5761}),
            frozenset({5762}),
            frozenset({5761, 5764}),
        ]
        assert all(item.completed_at.tzinfo is UTC for item in heartbeats)
    finally:
        connection.close()


def test_repository_returns_empty_without_querying_for_empty_location_ids() -> None:
    connection = ActualHourSqlConnection()

    result = load_repository_actual_hour_inputs(
        connection,
        [],
        datetime(2026, 8, 31, 12, tzinfo=UTC),
        datetime(2026, 8, 31, 13, tzinfo=UTC),
    )

    assert result == ([], [])
    assert connection.queries == []


def test_repository_returns_empty_after_stable_first_success_read_when_absent() -> None:
    connection = ActualHourSqlConnection()
    connection.runs = [
        {
            "id": 1,
            "status": "failed",
            "started_at": datetime(2026, 8, 31, 11),
            "completed_at": datetime(2026, 8, 31, 11, 1),
            "observed_location_ids": "[5761]",
        }
    ]

    result = load_repository_actual_hour_inputs(
        connection,
        [5761],
        datetime(2026, 8, 31, 12, tzinfo=UTC),
        datetime(2026, 8, 31, 13, tzinfo=UTC),
    )

    assert result == ([], [])
    assert len(connection.queries) == 1
    assert (
        "WHERE status='succeeded' ORDER BY started_at, id LIMIT 1"
        in normalized_statements(connection)[0]
    )


def test_repository_excludes_legacy_wall_time_and_maps_postcutover_tuples_to_utc() -> None:
    connection = ActualHourSqlConnection()
    connection.runs = [
        {
            "id": 8,
            "status": "succeeded",
            "started_at": datetime(2026, 11, 1, 6, 59),
            "completed_at": datetime(2026, 11, 1, 9),
            "observed_location_ids": "[5761]",
        }
    ]
    legacy_row = (5761, False, 90, 100, "2026-11-01 01:30:00.000000", 1)
    precutover_row = (5761, False, 80, 100, datetime(2026, 11, 1, 6, 58), 2)
    trusted_row = (5761, False, 40, 100, datetime(2026, 11, 1, 7), 3)
    connection.history_rows = [legacy_row, precutover_row, trusted_row]

    states, heartbeats = load_repository_actual_hour_inputs(
        connection,
        [5761],
        datetime(2026, 11, 1, 8, tzinfo=UTC),
        datetime(2026, 11, 1, 10, tzinfo=UTC),
    )

    assert connection.cutover_binds == [
        datetime(2026, 11, 1, 6, 59),
        datetime(2026, 11, 1, 6, 59),
    ]
    assert all(
        isinstance(value, datetime) and value.tzinfo is None
        for value in connection.cutover_binds
    )
    assert connection.returned_history_rows == [trusted_row]
    assert [item.id for item in states] == [3]
    assert all(item.fetched_at.tzinfo is UTC for item in states)
    assert all(item.completed_at.tzinfo is UTC for item in heartbeats)


def test_repository_uses_higher_id_for_equal_time_seed_and_orders_changes() -> None:
    connection = ActualHourSqlConnection()
    connection.history_rows = [
        (5761, False, 10, 100, datetime(2026, 8, 31, 11, 59), 10),
        (5761, False, 60, 100, datetime(2026, 8, 31, 11, 59), 20),
        (5762, False, 30, 100, datetime(2026, 8, 31, 12, 30), 40),
        (5761, False, 70, 100, datetime(2026, 8, 31, 12, 30), 30),
    ]

    states, _ = load_repository_actual_hour_inputs(
        connection,
        [5762, 5761],
        datetime(2026, 8, 31, 12, tzinfo=UTC),
        datetime(2026, 8, 31, 13, tzinfo=UTC),
    )

    assert [(item.location_id, item.count, item.id) for item in states] == [
        (5761, 60, 20),
        (5761, 70, 30),
        (5762, 30, 40),
    ]
    statements = normalized_statements(connection)
    assert any(
        "PARTITION BY location_id ORDER BY fetched_at DESC, id DESC" in statement
        and "fetched_at >= %s"
        and "fetched_at < %s" in statement
        for statement in statements
    )
    assert any(
        "fetched_at >= GREATEST(%s, %s)"
        in statement
        and "ORDER BY location_id, fetched_at, id" in statement
        for statement in statements
    )
    seed_parameters = connection.queries[1].parameters
    change_parameters = connection.queries[2].parameters
    assert seed_parameters == (
        5762,
        5761,
        datetime(2026, 8, 31, 11),
        datetime(2026, 8, 31, 12),
    )
    assert change_parameters == (
        5762,
        5761,
        datetime(2026, 8, 31, 11),
        datetime(2026, 8, 31, 12),
        datetime(2026, 8, 31, 13),
    )
    assert all(
        parameter.tzinfo is None
        for query in connection.queries[1:]
        for parameter in query.parameters
        if isinstance(parameter, datetime)
    )


def test_repository_reads_only_succeeded_heartbeats_through_inclusive_end() -> None:
    connection = ActualHourSqlConnection()
    connection.runs = [
        {
            "id": 1,
            "status": "succeeded",
            "started_at": datetime(2026, 8, 31, 11),
            "completed_at": datetime(2026, 8, 31, 12),
            "observed_location_ids": "[5761]",
        },
        {
            "id": 2,
            "status": "failed",
            "started_at": datetime(2026, 8, 31, 11, 1),
            "completed_at": datetime(2026, 8, 31, 12, 30),
            "observed_location_ids": "[5762]",
        },
        {
            "id": 3,
            "status": "succeeded",
            "started_at": datetime(2026, 8, 31, 11, 2),
            "completed_at": datetime(2026, 8, 31, 13),
            "observed_location_ids": "[5761, 5762]",
        },
        {
            "id": 4,
            "status": "succeeded",
            "started_at": datetime(2026, 8, 31, 11, 3),
            "completed_at": datetime(2026, 8, 31, 13, 0, 0, 1),
            "observed_location_ids": "[5761]",
        },
    ]

    _, heartbeats = load_repository_actual_hour_inputs(
        connection,
        [5761, 5762],
        datetime(2026, 8, 31, 12, tzinfo=UTC),
        datetime(2026, 8, 31, 13, tzinfo=UTC),
    )

    assert [item.completed_at for item in heartbeats] == [
        datetime(2026, 8, 31, 12, tzinfo=UTC),
        datetime(2026, 8, 31, 13, tzinfo=UTC),
    ]
    assert [item.observed_location_ids for item in heartbeats] == [
        frozenset({5761}),
        frozenset({5761, 5762}),
    ]
    heartbeat_statement = normalized_statements(connection)[-1]
    assert "WHERE status='succeeded'" in heartbeat_statement
    assert "completed_at >= %s AND completed_at <= %s" in heartbeat_statement
    assert connection.queries[-1].parameters == (
        datetime(2026, 8, 31, 12),
        datetime(2026, 8, 31, 13),
    )


def test_repository_parses_heartbeat_json_ids_conservatively() -> None:
    connection = ActualHourSqlConnection()
    connection.runs = [
        {
            "id": 1,
            "status": "succeeded",
            "started_at": datetime(2026, 8, 31, 11),
            "completed_at": datetime(2026, 8, 31, 12, 10),
            "observed_location_ids": "{",
        },
        {
            "id": 2,
            "status": "succeeded",
            "started_at": datetime(2026, 8, 31, 11, 1),
            "completed_at": datetime(2026, 8, 31, 12, 20),
            "observed_location_ids": '{"5761": true}',
        },
        {
            "id": 3,
            "status": "succeeded",
            "started_at": datetime(2026, 8, 31, 11, 2),
            "completed_at": datetime(2026, 8, 31, 12, 30),
            "observed_location_ids": '[5761, true, -1, 0, "5762", 5763.0, 5764]',
        },
    ]

    _, heartbeats = load_repository_actual_hour_inputs(
        connection,
        [5761, 5764],
        datetime(2026, 8, 31, 12, tzinfo=UTC),
        datetime(2026, 8, 31, 13, tzinfo=UTC),
    )

    assert [item.observed_location_ids for item in heartbeats] == [
        frozenset(),
        frozenset(),
        frozenset({5761, 5764}),
    ]


def seed_full_day_actuals(actual_hour_repository: object, date_key: str) -> None:
    windows = build_chicago_hour_windows(date_key)
    actual_hour_repository.states = [  # type: ignore[attr-defined]
        HistoryState(
            location_id=5761,
            is_closed=False,
            count=42,
            capacity=100,
            fetched_at=windows[0].start - timedelta(minutes=1),
            id=1,
        )
    ]
    actual_hour_repository.heartbeats = [  # type: ignore[attr-defined]
        IngestionHeartbeat(
            completed_at=windows[-1].end,
            observed_location_ids=frozenset({5761}),
        )
    ]


def test_actual_hours_serializes_unscaled_low_coverage_for_total_and_category(
    actual_hours_client: TestClient,
    actual_hour_repository: object,
) -> None:
    seed_full_day_actuals(actual_hour_repository, "2026-08-31")

    response = actual_hours_client.get(
        "/api/forecast/facilities/1186/actual-hours?date=2026-08-31"
    )

    assert response.status_code == 200
    payload = response.json()
    assert "coverageThreshold" not in payload
    assert len(payload["totalHours"]) == 24
    assert len(payload["categories"]) == 1
    assert len(payload["categories"][0]["hours"]) == 24
    for item in (payload["totalHours"][0], payload["categories"][0]["hours"][0]):
        assert item == {
            "hourStart": "2026-08-31T00:00:00-05:00",
            "observedCount": 42,
            "observedCapacity": 100,
            "expectedCapacity": 200,
            "actualCoverage": 0.5,
            "temporalCoverage": 1.0,
            "coverageThreshold": 0.75,
            "actualCount": None,
        }
        assert "actualPct" not in item


def test_actual_hours_qualified_items_include_actual_pct_and_per_item_thresholds(
    actual_hours_client: TestClient,
    actual_hour_repository: object,
) -> None:
    windows = build_chicago_hour_windows("2026-08-31")
    actual_hour_repository.states = [  # type: ignore[attr-defined]
        HistoryState(5761, False, 42, 100, windows[0].start, 1),
        HistoryState(5762, False, 8, 100, windows[0].start, 2),
    ]
    actual_hour_repository.heartbeats = [  # type: ignore[attr-defined]
        IngestionHeartbeat(windows[-1].end, frozenset({5761, 5762}))
    ]

    payload = actual_hours_client.get(
        "/api/forecast/facilities/1186/actual-hours?date=2026-08-31"
    ).json()

    assert payload["categories"][0]["key"] == "fitness floors"
    assert payload["categories"][0]["title"] == "Fitness Floors"
    for item in (payload["totalHours"][0], payload["categories"][0]["hours"][0]):
        assert item["observedCount"] == 50
        assert item["observedCapacity"] == 200
        assert item["expectedCapacity"] == 200
        assert item["actualCoverage"] == 1.0
        assert item["temporalCoverage"] == 1.0
        assert item["coverageThreshold"] == 0.75
        assert item["actualCount"] == 50
        assert item["actualPct"] == 0.25


@pytest.mark.parametrize(
    ("date_key", "expected_hours", "expected_start", "expected_end"),
    [
        (
            "2026-03-08",
            23,
            datetime(2026, 3, 8, 6, tzinfo=UTC),
            datetime(2026, 3, 9, 5, tzinfo=UTC),
        ),
        (
            "2026-11-01",
            25,
            datetime(2026, 11, 1, 5, tzinfo=UTC),
            datetime(2026, 11, 2, 6, tzinfo=UTC),
        ),
    ],
)
def test_actual_hours_route_uses_physical_windows_not_forecast_array_length(
    actual_hours_client: TestClient,
    actual_hour_repository: object,
    date_key: str,
    expected_hours: int,
    expected_start: datetime,
    expected_end: datetime,
) -> None:
    response = actual_hours_client.get(
        f"/api/forecast/facilities/1186/actual-hours?date={date_key}"
    )

    assert response.status_code == 200
    payload = response.json()
    hour_starts = [item["hourStart"] for item in payload["totalHours"]]
    assert len(hour_starts) == expected_hours
    assert len(payload["categories"][0]["hours"]) == expected_hours
    assert actual_hour_repository.calls == [  # type: ignore[attr-defined]
        ((5761, 5762), expected_start, expected_end)
    ]
    if date_key == "2026-03-08":
        assert not any(start.startswith("2026-03-08T02:") for start in hour_starts)
    else:
        assert "2026-11-01T01:00:00-05:00" in hour_starts
        assert "2026-11-01T01:00:00-06:00" in hour_starts


def test_actual_hours_loads_requested_union_once_without_opening_database(
    actual_hours_client: TestClient,
    actual_hour_repository: object,
) -> None:
    response = actual_hours_client.get(
        "/api/forecast/facilities/1186/actual-hours?date=2026-08-31"
    )

    assert response.status_code == 200
    assert actual_hour_repository.calls == [  # type: ignore[attr-defined]
        (
            (5761, 5762),
            datetime(2026, 8, 31, 5, tzinfo=UTC),
            datetime(2026, 9, 1, 5, tzinfo=UTC),
        )
    ]


def replace_actual_hour_day_categories(
    forecast_payload: dict[str, object],
    date_key: str,
    categories: object,
) -> None:
    facilities = forecast_payload.get("facilities")
    assert isinstance(facilities, list)
    assert facilities and isinstance(facilities[0], dict)
    weekly_forecast = facilities[0].get("weeklyForecast")
    assert isinstance(weekly_forecast, list)
    day = next(
        item
        for item in weekly_forecast
        if isinstance(item, dict) and item.get("date") == date_key
    )
    day["categories"] = categories


@pytest.mark.parametrize(
    (
        "date_key",
        "categories_value",
        "expected_hours",
        "expected_start",
        "expected_end",
        "expected_first_hour",
    ),
    [
        (
            "2026-08-31",
            [],
            24,
            datetime(2026, 8, 31, 5, tzinfo=UTC),
            datetime(2026, 9, 1, 5, tzinfo=UTC),
            "2026-08-31T00:00:00-05:00",
        ),
        (
            "2026-03-08",
            {"unexpected": "object"},
            23,
            datetime(2026, 3, 8, 6, tzinfo=UTC),
            datetime(2026, 3, 9, 5, tzinfo=UTC),
            "2026-03-08T00:00:00-06:00",
        ),
        (
            "2026-11-01",
            [
                {
                    "key": "unmapped section",
                    "title": "Unmapped Section",
                    "maxCapacity": 200,
                    "hours": [],
                }
            ],
            25,
            datetime(2026, 11, 1, 5, tzinfo=UTC),
            datetime(2026, 11, 2, 6, tzinfo=UTC),
            "2026-11-01T00:00:00-05:00",
        ),
    ],
)
def test_actual_hours_keeps_physical_totals_without_mapped_categories(
    actual_hours_client: TestClient,
    actual_hour_repository: object,
    actual_hours_forecast_payload: dict[str, object],
    date_key: str,
    categories_value: object,
    expected_hours: int,
    expected_start: datetime,
    expected_end: datetime,
    expected_first_hour: str,
) -> None:
    replace_actual_hour_day_categories(
        actual_hours_forecast_payload,
        date_key,
        categories_value,
    )

    response = actual_hours_client.get(
        f"/api/forecast/facilities/1186/actual-hours?date={date_key}"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["categories"] == []
    assert len(payload["totalHours"]) == expected_hours
    assert payload["totalHours"][0] == {
        "hourStart": expected_first_hour,
        "observedCount": None,
        "observedCapacity": 0,
        "expectedCapacity": 200,
        "actualCoverage": 0.0,
        "temporalCoverage": 0.0,
        "coverageThreshold": 0.75,
        "actualCount": None,
    }
    assert actual_hour_repository.calls == [  # type: ignore[attr-defined]
        ((5761, 5762), expected_start, expected_end)
    ]


@pytest.mark.usefixtures("actual_hours_client")
@pytest.mark.parametrize(
    ("date_key", "expected_status", "expected_detail"),
    [
        ("0001-01-01", 404, "Date not found for facility"),
        ("9999-12-31", 400, "date must be in YYYY-MM-DD format"),
    ],
)
def test_actual_hours_boundary_dates_never_escape_as_500(
    actual_hour_repository: object,
    date_key: str,
    expected_status: int,
    expected_detail: str,
) -> None:
    response = TestClient(
        forecast_api.app,
        raise_server_exceptions=False,
    ).get(f"/api/forecast/facilities/1186/actual-hours?date={date_key}")

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
    assert actual_hour_repository.calls == []  # type: ignore[attr-defined]


def test_actual_hours_preserves_facility_and_date_404s_without_repository_read(
    actual_hours_client: TestClient,
    actual_hour_repository: object,
) -> None:
    missing_facility = actual_hours_client.get(
        "/api/forecast/facilities/9999/actual-hours?date=2026-08-31"
    )
    missing_date = actual_hours_client.get(
        "/api/forecast/facilities/1186/actual-hours?date=2026-09-01"
    )

    assert missing_facility.status_code == 404
    assert missing_facility.json() == {"detail": "Facility not found"}
    assert missing_date.status_code == 404
    assert missing_date.json() == {"detail": "Date not found for facility"}
    assert actual_hour_repository.calls == []  # type: ignore[attr-defined]


def test_actual_hours_preserves_invalid_date_error(
    actual_hours_client: TestClient,
    actual_hour_repository: object,
) -> None:
    response = actual_hours_client.get(
        "/api/forecast/facilities/1186/actual-hours?date=not-a-date"
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "date must be in YYYY-MM-DD format"}
    assert actual_hour_repository.calls == []  # type: ignore[attr-defined]


def test_actual_hours_repository_failure_is_sanitized(
    actual_hours_client: TestClient,
    actual_hour_repository: object,
) -> None:
    actual_hour_repository.error = RuntimeError(  # type: ignore[attr-defined]
        "private database detail"
    )

    response = actual_hours_client.get(
        "/api/forecast/facilities/1186/actual-hours?date=2026-08-31"
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Actual-hour DB query failed"}
    assert "private database detail" not in response.text
    assert len(actual_hour_repository.calls) == 1  # type: ignore[attr-defined]


def configure_owned_actual_hour_route(
    monkeypatch: pytest.MonkeyPatch,
    actual_hours_forecast_payload: dict[str, object],
    connection: ActualHourSqlConnection,
) -> object:
    dependency = getattr(forecast_api, "get_actual_hour_repository", None)
    assert callable(dependency), "actual-hour repository dependency is missing"
    forecast_api.app.dependency_overrides.pop(dependency, None)
    monkeypatch.setattr(
        _seam_api_forecasts,
        "load_forecast",
        lambda: actual_hours_forecast_payload,
    )
    monkeypatch.setattr(
        _seam_runtime.current_runtime(),
        'section_ids',
        {
            1186: {
                "overall": [5761, 5762],
                "fitness floors": [5761, 5762],
            }
        },
    )
    monkeypatch.setattr(_seam_runtime.current_runtime(), 'capacities', {5761: 100, 5762: 100})

    def open_connection(*, autocommit: bool = True) -> ActualHourSqlConnection:
        assert autocommit is False
        return connection

    monkeypatch.setattr(_seam_db, "open_db_connection", open_connection)
    return dependency


def test_actual_hour_owned_dependency_preserves_404s_before_opening_database(
    monkeypatch: pytest.MonkeyPatch,
    actual_hours_forecast_payload: dict[str, object],
) -> None:
    dependency = getattr(forecast_api, "get_actual_hour_repository", None)
    assert callable(dependency), "actual-hour repository dependency is missing"
    forecast_api.app.dependency_overrides.pop(dependency, None)
    monkeypatch.setattr(
        _seam_api_forecasts,
        "load_forecast",
        lambda: actual_hours_forecast_payload,
    )
    connection_attempts = 0

    def reject_connection(**kwargs: object) -> object:
        nonlocal connection_attempts
        connection_attempts += 1
        raise AssertionError(f"404 path opened a database: {kwargs}")

    monkeypatch.setattr(_seam_db, "open_db_connection", reject_connection)
    client = TestClient(forecast_api.app)

    missing_facility = client.get(
        "/api/forecast/facilities/9999/actual-hours?date=2026-08-31"
    )
    missing_date = client.get(
        "/api/forecast/facilities/1186/actual-hours?date=2026-09-01"
    )

    assert missing_facility.status_code == 404
    assert missing_facility.json() == {"detail": "Facility not found"}
    assert missing_date.status_code == 404
    assert missing_date.json() == {"detail": "Date not found for facility"}
    assert connection_attempts == 0


def test_actual_hour_owned_dependency_sanitizes_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
    actual_hours_forecast_payload: dict[str, object],
) -> None:
    dependency = getattr(forecast_api, "get_actual_hour_repository", None)
    assert callable(dependency), "actual-hour repository dependency is missing"
    forecast_api.app.dependency_overrides.pop(dependency, None)
    monkeypatch.setattr(
        _seam_api_forecasts,
        "load_forecast",
        lambda: actual_hours_forecast_payload,
    )

    def reject_connection(**kwargs: object) -> object:
        raise RuntimeError(f"private connection detail: {kwargs}")

    monkeypatch.setattr(_seam_db, "open_db_connection", reject_connection)

    response = TestClient(forecast_api.app).get(
        "/api/forecast/facilities/1186/actual-hours?date=2026-08-31"
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Actual-hour DB query failed"}
    assert "private connection detail" not in response.text


def test_actual_hour_dependency_closes_owned_connection_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    actual_hours_forecast_payload: dict[str, object],
) -> None:
    connection = ActualHourSqlConnection()
    connection.runs = []
    configure_owned_actual_hour_route(
        monkeypatch,
        actual_hours_forecast_payload,
        connection,
    )

    response = TestClient(forecast_api.app).get(
        "/api/forecast/facilities/1186/actual-hours?date=2026-08-31"
    )

    assert response.status_code == 200
    assert connection.close_count == 1


def test_actual_hour_dependency_closes_after_sanitized_query_failure(
    monkeypatch: pytest.MonkeyPatch,
    actual_hours_forecast_payload: dict[str, object],
) -> None:
    connection = ActualHourSqlConnection()
    connection.query_failure = RuntimeError("private SQL detail")
    configure_owned_actual_hour_route(
        monkeypatch,
        actual_hours_forecast_payload,
        connection,
    )

    response = TestClient(forecast_api.app).get(
        "/api/forecast/facilities/1186/actual-hours?date=2026-08-31"
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Actual-hour DB query failed"}
    assert "private SQL detail" not in response.text
    assert connection.close_count == 1
