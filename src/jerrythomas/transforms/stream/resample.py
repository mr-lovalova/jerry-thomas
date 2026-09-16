from collections.abc import Iterator
from datetime import datetime, timezone
from itertools import chain
from typing import Any
from zoneinfo import ZoneInfo

from jerrythomas.config.resample import (
    FixedPeriodConfig,
    ResampleAggregation,
    ResampleConfig,
)
from jerrythomas.domain.record import TemporalRecord
from jerrythomas.transforms.numeric import ExactNumericSum, finite_aggregate_number
from jerrythomas.transforms.utils import (
    adjacent_partitions,
    get_field,
    is_missing,
    record_establishes_domain,
    set_record_domain_anchor,
)
from jerrythomas.utils.periods import CalendarMonth, FixedPeriod
from jerrythomas.utils.time import parse_timecode


class _Aggregation:
    def __init__(self, config: ResampleAggregation) -> None:
        self.config = config
        self.count = 0
        self.seen = False
        self.value: Any = None
        self.total = ExactNumericSum(config.field)

    def add(self, record: TemporalRecord) -> None:
        value = get_field(record, self.config.field)
        value = None if is_missing(value) else value
        statistic = self.config.statistic
        if statistic == "last" or (statistic == "first" and not self.seen):
            self.value = value
        self.seen = True
        if value is None:
            return
        self.count += 1
        if statistic in {"sum", "mean"}:
            self.total.add(value)
        elif statistic in {"min", "max"}:
            number = finite_aggregate_number(value, self.config.field)
            if self.value is None:
                self.value = number
            else:
                select = min if statistic == "min" else max
                self.value = select(self.value, number)

    def result(self) -> Any:
        statistic = self.config.statistic
        if statistic == "count":
            return self.count
        if statistic in {"sum", "mean"}:
            if not self.count:
                return None
            return self.total.result(self.count if statistic == "mean" else 1)
        return self.value


class ResampleTransform:
    """Aggregate complete [start, end) periods, publishing at their end."""

    def __init__(
        self, config: ResampleConfig, partition_fields: tuple[str, ...]
    ) -> None:
        self.aggregations = config.aggregations
        self.partition_fields = partition_fields
        self.start = config.start.astimezone(timezone.utc) if config.start else None
        self.end = config.end.astimezone(timezone.utc) if config.end else None
        self.period: FixedPeriod | CalendarMonth = (
            FixedPeriod(parse_timecode(config.period.every))
            if isinstance(config.period, FixedPeriodConfig)
            else CalendarMonth(ZoneInfo(config.period.timezone))
        )

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        for key, records in adjacent_partitions(stream, self.partition_fields):
            yield from self._resample_partition(key, records)

    def _resample_partition(
        self, key: tuple[Any, ...], records: Iterator[TemporalRecord]
    ) -> Iterator[TemporalRecord]:
        first = next(records, None)
        if first is None:
            return
        coverage_start = self.start if self.start is not None else first.time
        start = self.period.floor(coverage_start)
        if start < coverage_start:
            start = self.period.next(start)
        end = self.period.next(start)
        aggregates = self._new_aggregations()
        establishes_domain = False
        previous: datetime | None = None

        for record in chain((first,), records):
            if previous is not None and record.time < previous:
                raise ValueError(
                    "resample requires time-ordered records within each partition"
                )
            previous = record.time
            watermark = (
                min(record.time, self.end) if self.end is not None else record.time
            )
            while end <= watermark:
                yield self._record(key, end, aggregates, establishes_domain)
                start, end = end, self.period.next(end)
                aggregates = self._new_aggregations()
                establishes_domain = False
            if self.end is not None and record.time >= self.end:
                return
            if record.time < start:
                continue
            for aggregation in aggregates.values():
                aggregation.add(record)
            establishes_domain |= record_establishes_domain(record)

        # EOF alone is not a coverage boundary. Only a declared end can finish
        # the last period or establish trailing empty periods.
        if self.end is not None:
            while end <= self.end:
                yield self._record(key, end, aggregates, establishes_domain)
                end = self.period.next(end)
                aggregates = self._new_aggregations()
                establishes_domain = False

    def _new_aggregations(self) -> dict[str, _Aggregation]:
        return {
            name: _Aggregation(config) for name, config in self.aggregations.items()
        }

    def _record(
        self,
        key: tuple[Any, ...],
        end: datetime,
        aggregates: dict[str, _Aggregation],
        establishes_domain: bool,
    ) -> TemporalRecord:
        record = TemporalRecord(time=end)
        for field, value in zip(self.partition_fields, key):
            setattr(record, field, value)
        for field, aggregation in aggregates.items():
            setattr(record, field, aggregation.result())
        set_record_domain_anchor(record, establishes_domain)
        return record
