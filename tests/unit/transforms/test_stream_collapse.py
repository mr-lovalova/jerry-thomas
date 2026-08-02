from typing import Literal

import pytest

from jerrythomas.transforms.stream.collapse import CollapseTransform
from tests.unit.transforms.helpers import make_time_record


def _record(value: float, hour: int, partition: str):
    record = make_time_record(value, hour)
    record.partition = partition
    return record


@pytest.mark.parametrize(
    ("keep", "expected"),
    [
        ("first", 1.0),
        ("last", 3.0),
    ],
)
def test_collapse_keeps_one_record_per_partition_and_timestamp(
    keep: Literal["first", "last"],
    expected: float,
) -> None:
    records = [_record(1.0, 0, "A"), _record(3.0, 0, "A")]
    transform = CollapseTransform(
        partition_fields=("partition",),
        keep=keep,
    )

    output = list(transform.apply(iter(records)))

    assert [record.value for record in output] == [expected]


def test_collapse_keeps_partitions_separate() -> None:
    records = [_record(1.0, 0, "A"), _record(2.0, 0, "B")]

    output = CollapseTransform(
        partition_fields=("partition",),
        keep="last",
    ).apply(iter(records))

    assert [record.value for record in output] == [1.0, 2.0]


def test_collapse_emits_before_consuming_the_stream() -> None:
    consumed: list[int] = []

    def records():
        for hour in range(3):
            consumed.append(hour)
            yield _record(float(hour), hour, "A")

    output = CollapseTransform(
        partition_fields=("partition",),
        keep="last",
    ).apply(records())

    assert next(output).value == 0.0
    assert consumed == [0, 1]
