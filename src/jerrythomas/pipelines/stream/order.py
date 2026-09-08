from collections.abc import Iterator
from functools import partial
from typing import Any

from jerrythomas.domain.stream import (
    canonical_record_order,
    require_consistent_partition_types,
)
from jerrythomas.execution.pipeline import Stage
from jerrythomas.pipelines.sort import SortProgress, batch_sort
from jerrythomas.transforms.utils import partition_key


def build_record_order_stage(
    partition_by: tuple[str, ...],
    presorted: bool,
    buffer_bytes: int,
) -> Stage:
    if presorted:
        return Stage(
            name="ensure_record_order",
            apply=partial(validate_record_order, partition_by),
        )

    progress = SortProgress()
    return Stage(
        name="ensure_record_order",
        apply=partial(sort_records, partition_by, buffer_bytes, progress),
        progress=progress.snapshot,
    )


def validate_record_order(
    partition_by: tuple[str, ...],
    records: Iterator[Any],
) -> Iterator[Any]:
    required_order = list(canonical_record_order(partition_by))
    expected_types: dict[str, type] = {}
    previous_key = None
    for position, record in enumerate(records, start=1):
        partition = partition_key(record, partition_by)
        require_consistent_partition_types(
            partition_by,
            partition,
            expected_types,
            position,
        )
        current_key = partition, record.time
        if previous_key is not None and not previous_key <= current_key:
            raise ValueError(
                f"Record {position} violates presorted order {required_order!r}: "
                f"key {current_key!r} follows {previous_key!r}."
            )
        previous_key = current_key
        yield record


def sort_records(
    partition_by: tuple[str, ...],
    buffer_bytes: int,
    progress: SortProgress,
    records: Iterator[Any],
) -> Iterator[Any]:
    expected_types: dict[str, type] = {}

    def record_order_key(record: Any) -> tuple[Any, Any]:
        return partition_key(record, partition_by), record.time

    def validated_records() -> Iterator[Any]:
        for position, record in enumerate(records, start=1):
            require_consistent_partition_types(
                partition_by,
                partition_key(record, partition_by),
                expected_types,
                position,
            )
            yield record

    yield from batch_sort(
        validated_records(),
        buffer_bytes=buffer_bytes,
        key=record_order_key,
        progress=progress,
    )
