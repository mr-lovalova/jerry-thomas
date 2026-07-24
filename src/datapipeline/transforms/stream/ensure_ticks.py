from collections.abc import Iterator
from datetime import datetime
from itertools import chain, groupby

from datapipeline.artifacts.ticks import TickGrid
from datapipeline.domain.record import TemporalRecord
from datapipeline.transforms.utils import (
    clone_record,
    partition_key,
    set_record_domain_anchor,
)
from datapipeline.utils.time import parse_timecode


class EnsureCadenceTransform:
    """Insert missing records at a fixed interval within each partition."""

    def __init__(
        self,
        cadence: str,
        partition_fields: tuple[str, ...],
    ) -> None:
        self.partition_fields = partition_fields
        self.step = parse_timecode(cadence)

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        for _, records in groupby(
            stream,
            key=lambda record: partition_key(record, self.partition_fields),
        ):
            previous: TemporalRecord | None = None
            for record in records:
                if previous is not None:
                    expected = previous.time + self.step
                    while expected < record.time:
                        yield _placeholder_record(
                            previous,
                            expected,
                            self.partition_fields,
                        )
                        expected += self.step
                yield record
                previous = record


class EnsureTicksTransform:
    """Reindex each input partition against a resolved tick grid."""

    def __init__(
        self,
        ticks: TickGrid,
        partition_fields: tuple[str, ...],
    ) -> None:
        self.partition_fields = partition_fields
        if ticks.grid_by != self.partition_fields:
            raise ValueError(
                f"Tick grid fields {list(ticks.grid_by)!r} must match stream "
                f"partition_by {list(self.partition_fields)!r}."
            )
        self.ticks = ticks

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        partition_fields = self.partition_fields

        for key, records in groupby(
            stream,
            key=lambda record: partition_key(record, self.partition_fields),
        ):
            source = iter(records)
            first = next(source, None)
            if first is None:
                continue

            ticks = self.ticks.ticks_for(key)
            tick_index = 0
            template = first
            for record in chain((first,), source):
                while tick_index < len(ticks) and ticks[tick_index] < record.time:
                    yield _placeholder_record(
                        record,
                        ticks[tick_index],
                        partition_fields,
                    )
                    tick_index += 1
                if tick_index < len(ticks) and ticks[tick_index] == record.time:
                    tick_index += 1
                yield record
                template = record

            while tick_index < len(ticks):
                yield _placeholder_record(
                    template,
                    ticks[tick_index],
                    partition_fields,
                )
                tick_index += 1


def _placeholder_record(
    record: TemporalRecord,
    time: datetime,
    partition_fields: tuple[str, ...],
) -> TemporalRecord:
    keep = {"time", *partition_fields}
    updates = {
        key: None for key in vars(record) if not key.startswith("_") and key not in keep
    }
    placeholder = clone_record(record, time=time, **updates)
    set_record_domain_anchor(placeholder, False)
    return placeholder
