from datetime import datetime, timedelta, timezone

import pytest

from jerrythomas.utils.time import (
    count_cadence_buckets,
    parse_cadence,
    parse_timecode,
)


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
