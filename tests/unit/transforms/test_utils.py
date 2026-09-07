from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from jerrythomas.domain.record import TemporalRecord, public_record_fields
from jerrythomas.transforms.utils import (
    adjacent_partitions,
    clone_record,
    finite_number,
    finite_number_or_none,
    get_field,
    partition_key,
    record_establishes_domain,
    set_record_domain_anchor,
)
from tests.unit.transforms.helpers import make_time_record


def test_clone_record_preserves_custom_constructor_and_dynamic_fields() -> None:
    constructor_calls = 0

    @dataclass(init=False)
    class PluginRecord(TemporalRecord):
        value: float

        def __init__(self, time: datetime, value: float) -> None:
            nonlocal constructor_calls
            constructor_calls += 1
            super().__init__(time)
            self.value = value

    time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    record = PluginRecord(time, 1.0)
    record.history = [3.0, 4.0]
    set_record_domain_anchor(record, False)

    cloned = clone_record(record, value=2.0)

    assert type(cloned) is PluginRecord
    assert cloned is not record
    assert constructor_calls == 1
    assert record.value == 1.0
    assert cloned.value == 2.0
    assert cloned.history is record.history
    assert not record_establishes_domain(cloned)
    assert public_record_fields(cloned) == {
        "time": time,
        "value": 2.0,
        "history": [3.0, 4.0],
    }
    assert "_establishes_domain" not in repr(cloned)


def test_clone_record_normalizes_time_updates_without_mutating_input() -> None:
    original_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    record = TemporalRecord(original_time)
    set_record_domain_anchor(record, False)
    updated_time = datetime(2024, 1, 2, tzinfo=timezone(timedelta(hours=2)))

    cloned = clone_record(record, time=updated_time)

    assert record.time == original_time
    assert cloned.time == datetime(2024, 1, 1, 22, tzinfo=timezone.utc)
    assert cloned.time.tzinfo is timezone.utc
    assert not record_establishes_domain(cloned)
    with pytest.raises(ValueError, match="time must be timezone-aware"):
        clone_record(record, time=updated_time.replace(tzinfo=None))
    assert record.time == original_time


def test_adjacent_partitions_preserves_stream_group_boundaries() -> None:
    records = [
        make_time_record(1.0, 0),
        make_time_record(2.0, 1),
        make_time_record(3.0, 2),
        make_time_record(4.0, 3),
    ]
    for record, security_id in zip(records, ("AAPL", "AAPL", "MSFT", "AAPL")):
        record.security_id = security_id

    partitions = [
        (key, [record.value for record in partition])
        for key, partition in adjacent_partitions(iter(records), ("security_id",))
    ]

    assert partitions == [
        (("AAPL",), [1.0, 2.0]),
        (("MSFT",), [3.0]),
        (("AAPL",), [4.0]),
    ]


def test_partition_key_requires_declared_object_field() -> None:
    record = make_time_record(1.0, 0)

    with pytest.raises(KeyError, match="Partition field 'security_id'"):
        partition_key(record, ("security_id",))


def test_partition_key_preserves_present_none_value() -> None:
    record = make_time_record(1.0, 0)
    record.security_id = None

    assert partition_key(record, ("security_id",)) == (None,)


def test_partition_key_reads_multiple_fields_in_declared_order() -> None:
    record = make_time_record(1.0, 0)
    record.security_id = "AAPL"
    record.venue = "XNAS"

    assert partition_key(record, ("security_id", "venue")) == ("AAPL", "XNAS")


def test_get_field_distinguishes_absent_from_missing() -> None:
    record = make_time_record(None, 0)

    assert get_field(record, "value") is None
    with pytest.raises(KeyError, match="missing"):
        get_field(record, "missing")


@pytest.mark.parametrize("value", [True, "1.0", object()])
def test_finite_number_rejects_non_numeric_values(value: object) -> None:
    with pytest.raises(TypeError, match="numeric values"):
        finite_number(value, "value")


@pytest.mark.parametrize("value", [None, float("nan")])
def test_finite_number_or_none_normalizes_missing_values(value: object) -> None:
    assert finite_number_or_none(value, "value") is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, 1.0), (1.5, 1.5), (Decimal("1.25"), 1.25)],
)
def test_finite_number_or_none_converts_finite_values(
    value: object,
    expected: float,
) -> None:
    assert finite_number_or_none(value, "value") == expected


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, TypeError),
        ("1.0", TypeError),
        (b"1.0", TypeError),
        (object(), TypeError),
        (float("inf"), ValueError),
        (float("-inf"), ValueError),
        (Decimal("NaN"), ValueError),
        (Decimal("Infinity"), ValueError),
        (Decimal("-Infinity"), ValueError),
    ],
)
def test_finite_number_or_none_rejects_invalid_values(
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match="Field 'value'"):
        finite_number_or_none(value, "value")
