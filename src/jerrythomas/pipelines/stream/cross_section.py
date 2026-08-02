from collections.abc import Iterator, Sequence
from functools import partial
from itertools import groupby
from typing import Any

from jerrythomas.config.cross_section import (
    CrossSectionOperation,
    OlsResidualConfig,
    RankScoreConfig,
)
from jerrythomas.domain.record import TemporalRecord
from jerrythomas.execution.pipeline import Stage
from jerrythomas.pipelines.sort import SortProgress, batch_sort
from jerrythomas.pipelines.stream.order import build_record_order_stage
from jerrythomas.transforms.cross_section import (
    OlsResidualTransform,
    RankScoreTransform,
)
from jerrythomas.transforms.utils import partition_key


_CrossSectionTransform = RankScoreTransform | OlsResidualTransform


def build_cross_section_stages(
    operations: tuple[CrossSectionOperation, ...],
    partition_by: tuple[str, ...],
    buffer_bytes: int,
) -> tuple[Stage, ...]:
    progress = SortProgress()
    transforms = tuple(_build_transform(operation) for operation in operations)
    return (
        Stage(
            name="order_cross_sections",
            apply=partial(
                _order_cross_sections,
                partition_by,
                buffer_bytes,
                progress,
            ),
            progress=progress.snapshot,
        ),
        Stage(
            name="apply_cross_section",
            apply=partial(
                _apply_cross_sections,
                partition_by,
                transforms,
            ),
        ),
        build_record_order_stage(
            partition_by,
            presorted=False,
            buffer_bytes=buffer_bytes,
        ),
    )


def _build_transform(operation: CrossSectionOperation) -> _CrossSectionTransform:
    if isinstance(operation, RankScoreConfig):
        return RankScoreTransform(
            operation.field,
            operation.to,
            operation.min_samples,
        )
    if isinstance(operation, OlsResidualConfig):
        return OlsResidualTransform(
            operation.y,
            operation.x,
            operation.to,
            operation.min_samples,
        )
    raise TypeError(f"Unsupported cross-section operation: {type(operation).__name__}")


def _order_cross_sections(
    partition_by: tuple[str, ...],
    buffer_bytes: int,
    progress: SortProgress,
    records: Iterator[Any],
) -> Iterator[Any]:
    yield from batch_sort(
        records,
        buffer_bytes=buffer_bytes,
        key=lambda record: (record.time, partition_key(record, partition_by)),
        progress=progress,
    )


def _apply_cross_sections(
    partition_by: tuple[str, ...],
    transforms: tuple[_CrossSectionTransform, ...],
    records: Iterator[TemporalRecord],
) -> Iterator[TemporalRecord]:
    for _, timestamp_records in groupby(records, key=lambda record: record.time):
        cross_section = tuple(timestamp_records)
        _require_unique_partitions(cross_section, partition_by)
        for transform in transforms:
            cross_section = transform.apply(cross_section)
        yield from cross_section


def _require_unique_partitions(
    records: Sequence[TemporalRecord],
    partition_by: tuple[str, ...],
) -> None:
    previous = None
    for record in records:
        current = partition_key(record, partition_by)
        if previous is not None and current == previous:
            raise ValueError(
                "Cross-section input contains duplicate partition "
                f"{current!r} at {record.time.isoformat()}."
            )
        previous = current
