from collections import deque
from collections.abc import Iterator

from datapipeline.domain.record import TemporalRecord
from datapipeline.transforms.utils import (
    adjacent_partitions,
    clone_record_with_field,
    get_field,
)


class LeadTransform:
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
            yield from self._lead(records)

    def _lead(self, records: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        future: deque[TemporalRecord] = deque()
        for _ in range(self.periods):
            record = next(records, None)
            if record is None:
                break
            get_field(record, self.field)
            future.append(record)

        for record in records:
            value = get_field(record, self.field)
            lead_record = future.popleft()
            future.append(record)
            yield clone_record_with_field(
                lead_record,
                self.to,
                value,
            )

        while future:
            yield clone_record_with_field(future.popleft(), self.to, None)
