from collections import deque
from collections.abc import Iterator

from datapipeline.domain.record import TemporalRecord
from datapipeline.transforms.utils import (
    adjacent_partitions,
    clone_record_with_field,
    get_field,
)


class LagTransform:
    def __init__(
        self,
        field: str,
        periods: int,
        partition_fields: tuple[str, ...],
        to: str | None = None,
    ) -> None:
        self.field = field
        self.to = field if to is None else to
        self.periods = periods
        self.partition_fields = partition_fields

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        for _, records in adjacent_partitions(stream, self.partition_fields):
            previous: deque[object] = deque(maxlen=self.periods)
            for record in records:
                value = previous[0] if len(previous) == self.periods else None
                previous.append(get_field(record, self.field))
                yield clone_record_with_field(record, self.to, value)
