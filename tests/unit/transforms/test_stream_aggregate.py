from itertools import permutations

import pytest

from jerrythomas.transforms.stream.aggregate import AggregateSumTransform
from jerrythomas.transforms.utils import (
    record_establishes_domain,
    set_record_domain_anchor,
)
from tests.unit.transforms.helpers import make_time_record


def _record(value: object, hour: int, partition: str):
    record = make_time_record(None, hour)
    record.value = value
    record.partition = partition
    return record


def test_aggregate_sum_emits_one_record_per_canonical_key() -> None:
    first = _record(1.5, 0, "A")
    first.note = "first"
    second = _record(2.5, 0, "A")
    second.note = "second"
    third = _record(3, 1, "A")

    output = list(
        AggregateSumTransform(
            field="value",
            partition_fields=("partition",),
            count_to="events",
        ).apply(iter([first, second, third]))
    )

    assert [(record.value, record.events) for record in output] == [
        (4.0, 2),
        (3.0, 1),
    ]
    assert output[0].note == "first"
    assert output[0] is not first
    assert first.value == 1.5
    assert not hasattr(first, "events")


def test_aggregate_sum_keeps_partitions_separate() -> None:
    records = [
        _record(1, 0, "A"),
        _record(2, 0, "A"),
        _record(4, 0, "B"),
    ]

    output = AggregateSumTransform(
        field="value",
        partition_fields=("partition",),
    ).apply(iter(records))

    assert [(record.partition, record.value) for record in output] == [
        ("A", 3.0),
        ("B", 4.0),
    ]


@pytest.mark.parametrize("value", [None, True, float("nan"), "1", float("inf")])
def test_aggregate_sum_rejects_nonfinite_or_nonnumeric_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="numeric|finite"):
        list(
            AggregateSumTransform(
                field="value",
                partition_fields=("partition",),
            ).apply(iter([_record(value, 0, "A")]))
        )


def test_aggregate_sum_rejects_overflow() -> None:
    with pytest.raises(OverflowError, match="exceeds"):
        list(
            AggregateSumTransform(
                field="value",
                partition_fields=("partition",),
            ).apply(
                iter(
                    [
                        _record(1e308, 0, "A"),
                        _record(1e308, 0, "A"),
                    ]
                )
            )
        )


def test_aggregate_sum_preserves_exact_integer_values() -> None:
    large = 9_007_199_254_740_993

    [aggregate] = AggregateSumTransform(
        field="value",
        partition_fields=("partition",),
    ).apply(iter([_record(large, 0, "A"), _record(1, 0, "A")]))

    assert aggregate.value == large + 1
    assert type(aggregate.value) is int


def test_aggregate_sum_is_independent_of_equal_key_record_order() -> None:
    values = [1e100, 1e84, -1e100, -1e84, 1.0]

    for ordered_values in permutations(values):
        [aggregate] = AggregateSumTransform(
            field="value",
            partition_fields=("partition",),
        ).apply(iter(_record(value, 0, "A") for value in ordered_values))

        assert aggregate.value == 1.0


def test_aggregate_sum_rejects_lossy_mixed_numeric_types_in_every_order() -> None:
    values = [2**53, 1, 0.0, -float(2**53)]

    for ordered_values in permutations(values):
        with pytest.raises(ValueError, match="without precision loss"):
            list(
                AggregateSumTransform(
                    field="value",
                    partition_fields=("partition",),
                ).apply(iter(_record(value, 0, "A") for value in ordered_values))
            )


def test_aggregate_sum_avoids_intermediate_floating_point_overflow() -> None:
    values = [1e308, 1e308, -1e308]

    for ordered_values in permutations(values):
        [aggregate] = AggregateSumTransform(
            field="value",
            partition_fields=("partition",),
        ).apply(iter(_record(value, 0, "A") for value in ordered_values))

        assert aggregate.value == 1e308


def test_aggregate_sum_requires_distinct_output_fields() -> None:
    with pytest.raises(ValueError, match="count_to must differ from field"):
        AggregateSumTransform(
            field="value",
            partition_fields=(),
            count_to="value",
        )


def test_aggregate_sum_preserves_group_domain_provenance() -> None:
    first = _record(1, 0, "A")
    second = _record(2, 0, "A")
    set_record_domain_anchor(first, False)
    set_record_domain_anchor(second, True)

    [aggregate] = AggregateSumTransform(
        field="value",
        partition_fields=("partition",),
    ).apply(iter([first, second]))

    assert record_establishes_domain(aggregate) is True


def test_aggregate_sum_emits_before_consuming_the_next_group() -> None:
    consumed: list[int] = []

    def records():
        for hour in range(3):
            consumed.append(hour)
            yield _record(hour + 1, hour, "A")

    output = AggregateSumTransform(
        field="value",
        partition_fields=("partition",),
    ).apply(records())

    assert next(output).value == 1.0
    assert consumed == [0, 1]
