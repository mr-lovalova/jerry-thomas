from decimal import Decimal

import pytest

from datapipeline.transforms.utils import (
    adjacent_partitions,
    finite_number,
    finite_number_or_none,
    get_field,
    partition_key,
)
from tests.unit.transforms.helpers import make_time_record


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
