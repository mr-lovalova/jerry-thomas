from collections.abc import Iterator
from datetime import datetime
from itertools import chain

from datapipeline.artifacts.schedule import Schedule
from datapipeline.domain.record import TemporalRecord
from datapipeline.transforms.utils import (
    adjacent_partitions,
    clone_record,
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
        for _, records in adjacent_partitions(stream, self.partition_fields):
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


class EnsureScheduleTransform:
    """Insert missing records at every scheduled time in each partition."""

    def __init__(
        self,
        schedule: Schedule,
        partition_fields: tuple[str, ...],
    ) -> None:
        self.partition_fields = partition_fields
        if schedule.partition_by != self.partition_fields:
            raise ValueError(
                f"Schedule partition fields {list(schedule.partition_by)!r} "
                "must match stream "
                f"partition_by {list(self.partition_fields)!r}."
            )
        self.schedule = schedule

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        partition_fields = self.partition_fields

        for key, records in adjacent_partitions(stream, partition_fields):
            source = iter(records)
            first = next(source, None)
            if first is None:
                continue

            times = self.schedule.times_for(key)
            time_index = 0
            template = first
            for record in chain((first,), source):
                while time_index < len(times) and times[time_index] < record.time:
                    yield _placeholder_record(
                        record,
                        times[time_index],
                        partition_fields,
                    )
                    time_index += 1
                if time_index < len(times) and times[time_index] == record.time:
                    time_index += 1
                yield record
                template = record

            while time_index < len(times):
                yield _placeholder_record(
                    template,
                    times[time_index],
                    partition_fields,
                )
                time_index += 1


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
