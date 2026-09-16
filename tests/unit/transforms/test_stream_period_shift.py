from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from jerrythomas.config.transforms import LagConfig, LeadConfig
from jerrythomas.transforms.stream.lag import LagTransform
from jerrythomas.transforms.stream.lead import LeadTransform
from jerrythomas.transforms.time import RoundTimeTransform, ShiftTimeTransform
from tests.unit.transforms.helpers import make_time_record


def _record(value: float | None, hour: int, ticker: str):
    record = make_time_record(value, hour)
    record.ticker = ticker
    return record


def test_shift_time_moves_record_timestamp() -> None:
    record = make_time_record(10.0, 2)
    original_time = record.time
    transform = ShiftTimeTransform(by="-1h")

    [shifted] = list(transform.apply(iter([record])))

    assert shifted.time == original_time - timedelta(hours=1)
    assert record.time == original_time
    assert shifted.value == 10.0


@pytest.mark.parametrize(("direction", "expected_hour"), [("floor", 3), ("ceil", 6)])
def test_round_time_uses_a_continuous_utc_grid(
    direction: str, expected_hour: int
) -> None:
    record = make_time_record(10.0, 5)

    [rounded] = RoundTimeTransform(cadence="3h", direction=direction).apply(
        iter([record])
    )

    assert rounded.time.hour == expected_hour
    assert record.time.hour == 5


@pytest.mark.parametrize(("direction", "expected_hour"), [("floor", 0), ("ceil", 3)])
def test_round_time_preserves_records_sharing_a_bucket_and_their_provenance(
    direction: str, expected_hour: int
) -> None:
    first = _record(10.0, 1, "A")
    second = _record(20.0, 2, "A")
    second._establishes_domain = False

    output = list(
        RoundTimeTransform(cadence="3h", direction=direction).apply(
            iter([first, second])
        )
    )

    assert [(r.time.hour, r.ticker, r.value) for r in output] == [
        (expected_hour, "A", 10.0),
        (expected_hour, "A", 20.0),
    ]
    assert [r._establishes_domain for r in output] == [True, False]
    assert [first.time.hour, second.time.hour] == [1, 2]


@pytest.mark.parametrize("direction", ["floor", "ceil"])
def test_round_time_keeps_exact_grid_timestamps(direction: str) -> None:
    record = make_time_record(10.0, 3)
    [rounded] = RoundTimeTransform(cadence="90m", direction=direction).apply(
        iter([record])
    )
    assert rounded.time == record.time
    assert rounded.value == record.value


def test_round_time_floor_uses_epoch_alignment_before_1970() -> None:
    record = make_time_record(1.0, 0)
    record.time = datetime(1969, 12, 31, 23, 59, tzinfo=timezone.utc)
    [output] = RoundTimeTransform(cadence="90m", direction="floor").apply(
        iter([record])
    )
    assert output.time == datetime(1969, 12, 31, 22, 30, tzinfo=timezone.utc)


def test_stream_lag_copies_previous_partition_value() -> None:
    stream = iter(
        [
            _record(10.0, 0, "AAPL"),
            _record(11.0, 1, "AAPL"),
            _record(12.0, 2, "AAPL"),
            _record(20.0, 0, "MSFT"),
            _record(21.0, 1, "MSFT"),
        ]
    )
    transform = LagTransform(
        field="value",
        to="value_lag_1",
        periods=1,
        partition_fields=("ticker",),
    )

    out = list(transform.apply(stream))

    assert [getattr(record, "value_lag_1") for record in out] == [
        None,
        10.0,
        11.0,
        None,
        20.0,
    ]


def test_stream_lead_copies_future_partition_value() -> None:
    stream = iter(
        [
            _record(10.0, 0, "AAPL"),
            _record(11.0, 1, "AAPL"),
            _record(12.0, 2, "AAPL"),
            _record(20.0, 0, "MSFT"),
            _record(21.0, 1, "MSFT"),
        ]
    )
    transform = LeadTransform(
        field="value",
        to="value_lead_1",
        periods=1,
        partition_fields=("ticker",),
    )

    out = list(transform.apply(stream))

    assert [getattr(record, "value_lead_1") for record in out] == [
        11.0,
        12.0,
        None,
        21.0,
        None,
    ]


def test_stream_lead_accepts_normalized_partition_fields() -> None:
    stream = iter(
        [
            _record(10.0, 0, "AAPL"),
            _record(11.0, 1, "AAPL"),
        ]
    )

    out = list(
        LeadTransform(
            field="value",
            to="value_lead_1",
            periods=1,
            partition_fields=("ticker",),
        ).apply(stream)
    )

    assert [getattr(record, "value_lead_1") for record in out] == [11.0, None]


@pytest.mark.parametrize("config_type", [LagConfig, LeadConfig])
@pytest.mark.parametrize("periods", [0, True, 1.5])
def test_period_shift_requires_positive_integer_periods(config_type, periods) -> None:
    with pytest.raises(ValidationError, match="periods"):
        config_type(field="value", periods=periods)
