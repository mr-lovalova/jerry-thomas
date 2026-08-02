from collections import deque
from collections.abc import Iterator
from math import isfinite

from jerrythomas.domain.record import TemporalRecord
from jerrythomas.transforms.rolling_window import RollingSum
from jerrythomas.transforms.utils import (
    adjacent_partitions,
    clone_record_with_field,
    finite_number_or_none,
    get_field,
)


class ForwardSumTransform:
    """Sum the next fixed number of records within each partition."""

    def __init__(
        self,
        field: str,
        window: int,
        partition_fields: tuple[str, ...],
        to: str,
    ) -> None:
        self.field = field
        self.to = to
        self.window = window
        self.partition_fields = partition_fields

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        for _, records in adjacent_partitions(stream, self.partition_fields):
            yield from self._sum_partition(records)

    def _sum_partition(
        self,
        records: Iterator[TemporalRecord],
    ) -> Iterator[TemporalRecord]:
        pending: deque[TemporalRecord] = deque()
        total = RollingSum(self.window)
        for record in records:
            pending.append(record)
            total.append(
                finite_number_or_none(get_field(record, self.field), self.field)
            )
            if len(pending) <= self.window:
                continue

            value = total.result() if total.sample_count == self.window else None
            if value is not None and not isfinite(value):
                raise OverflowError(
                    f"Forward sum field {self.field!r} exceeds the supported "
                    "floating-point range"
                )
            yield clone_record_with_field(pending.popleft(), self.to, value)

        while pending:
            yield clone_record_with_field(pending.popleft(), self.to, None)
