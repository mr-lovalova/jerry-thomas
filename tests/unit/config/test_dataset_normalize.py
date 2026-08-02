from datetime import datetime, timedelta, timezone

import pytest

from jerrythomas.utils.time import floor_time_to_cadence, parse_cadence


def test_floor_time_uses_continuous_grid_for_multi_hour_minutes() -> None:
    timestamp = datetime(2025, 1, 2, 1, 40, tzinfo=timezone.utc)

    floored = floor_time_to_cadence(timestamp, parse_cadence("90m"))

    assert floored == datetime(2025, 1, 2, 1, 30, tzinfo=timezone.utc)


def test_floor_time_uses_continuous_grid_across_days() -> None:
    timestamp = datetime(1970, 1, 2, 12, 0, tzinfo=timezone.utc)

    floored = floor_time_to_cadence(timestamp, parse_cadence("48h"))

    assert floored == datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc)


def test_floor_time_uses_utc_grid_for_offset_timestamp() -> None:
    offset = timezone(timedelta(hours=2))
    timestamp = datetime(2025, 1, 1, 4, 10, tzinfo=offset)

    floored = floor_time_to_cadence(timestamp, parse_cadence("3h"))

    assert floored == datetime(2025, 1, 1, 2, 0, tzinfo=offset)


def test_floor_time_anchors_multi_day_cadence_to_unix_epoch() -> None:
    timestamp = datetime(2025, 1, 8, 12, 0, tzinfo=timezone.utc)

    floored = floor_time_to_cadence(timestamp, parse_cadence("7d"))

    assert floored == datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("cadence", "expected"),
    [
        ("5m", datetime(2025, 1, 2, 1, 15, tzinfo=timezone.utc)),
        ("15m", datetime(2025, 1, 2, 1, 15, tzinfo=timezone.utc)),
        ("1h", datetime(2025, 1, 2, 1, 0, tzinfo=timezone.utc)),
        ("2h", datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc)),
        ("1d", datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc)),
    ],
)
def test_floor_time_preserves_conventional_cadence_behavior(
    cadence: str,
    expected: datetime,
) -> None:
    timestamp = datetime(2025, 1, 2, 1, 17, tzinfo=timezone.utc)

    assert floor_time_to_cadence(timestamp, parse_cadence(cadence)) == expected
