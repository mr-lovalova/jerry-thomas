from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from jerrythomas.utils.time import (
    ceil_time_to_cadence,
    count_cadence_buckets,
    parse_cadence,
    parse_timecode,
    round_time_to_cadence,
)


@pytest.mark.parametrize(
    ("time", "cadence", "expected"),
    [
        ("2024-01-01T05:00:00+00:00", "1d", "2024-01-02T00:00:00+00:00"),
        ("2024-01-01T00:00:00+00:00", "1d", "2024-01-01T00:00:00+00:00"),
        ("2024-01-01T00:00:00.000001+00:00", "1d", "2024-01-02T00:00:00+00:00"),
        ("2024-01-01T03:10:00+02:00", "90m", "2024-01-01T03:30:00+02:00"),
        ("1969-12-31T23:59:00+00:00", "1h", "1970-01-01T00:00:00+00:00"),
        ("2024-01-01T00:30:00", "1h", "2024-01-01T01:00:00"),
    ],
)
def test_ceil_assigns_the_first_tick_at_or_after_time(
    time: str, cadence: str, expected: str
) -> None:
    assert ceil_time_to_cadence(
        datetime.fromisoformat(time), parse_cadence(cadence)
    ) == datetime.fromisoformat(expected)


@pytest.mark.parametrize(
    ("time", "expected"),
    [
        ("2024-03-10T01:30:00-05:00", "2024-03-10T07:00:00+00:00"),
        ("2024-11-03T01:30:00-04:00", "2024-11-03T06:00:00+00:00"),
        ("2024-11-03T01:30:00-05:00", "2024-11-03T07:00:00+00:00"),
    ],
)
def test_ceil_uses_elapsed_time_across_dst(time: str, expected: str) -> None:
    local = datetime.fromisoformat(time).astimezone(ZoneInfo("America/New_York"))
    result = ceil_time_to_cadence(local, timedelta(hours=1))
    assert result.astimezone(timezone.utc) == datetime.fromisoformat(expected)


@pytest.mark.parametrize(
    ("cadence", "expected_hour", "expected_minute"), [("75m", 6, 15), ("5h", 10, 0)]
)
def test_ceil_does_not_confuse_repeated_local_clock_times(
    cadence: str,
    expected_hour: int,
    expected_minute: int,
) -> None:
    local = datetime(2024, 11, 3, 1, tzinfo=ZoneInfo("America/New_York"), fold=1)
    result = ceil_time_to_cadence(local, parse_cadence(cadence))
    assert result.astimezone(timezone.utc) == datetime(
        2024,
        11,
        3,
        expected_hour,
        expected_minute,
        tzinfo=timezone.utc,
    )


def test_exact_rejects_a_different_instant_with_the_same_local_clock_time() -> None:
    local = datetime(2024, 11, 3, 1, tzinfo=ZoneInfo("America/New_York"), fold=1)
    with pytest.raises(ValueError, match="not aligned"):
        round_time_to_cadence(local, parse_cadence("75m"), "exact")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("30s", timedelta(seconds=30)),
        ("10m", timedelta(minutes=10)),
        ("10min", timedelta(minutes=10)),
        ("-1h", timedelta(hours=-1)),
        ("+2d", timedelta(days=2)),
    ],
)
def test_parse_timecode(value: str, expected: timedelta) -> None:
    assert parse_timecode(value) == expected


@pytest.mark.parametrize("value", ["", "1", "1w", "1.5h"])
def test_parse_timecode_rejects_unsupported_values(value: str) -> None:
    with pytest.raises(ValueError, match="Unsupported timecode"):
        parse_timecode(value)


def test_parse_cadence_accepts_positive_dataset_units() -> None:
    assert parse_cadence("90min") == timedelta(minutes=90)


@pytest.mark.parametrize("value", ["0m", "-1h", "1s"])
def test_parse_cadence_rejects_values_outside_dataset_contract(value: str) -> None:
    with pytest.raises(ValueError, match="Unsupported cadence"):
        parse_cadence(value)


@pytest.mark.parametrize(
    ("cadence", "expected"),
    [("1h", 7), ("2h", 4), ("15m", 25)],
)
def test_count_cadence_buckets_uses_inclusive_floored_bounds(
    cadence: str,
    expected: int,
) -> None:
    start = datetime(2024, 1, 1, 4, 5, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 10, 10, tzinfo=timezone.utc)

    assert count_cadence_buckets(start, end, parse_cadence(cadence)) == expected
