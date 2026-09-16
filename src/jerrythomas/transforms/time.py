from collections.abc import Iterator
from datetime import timezone
from typing import Literal

from jerrythomas.domain.record import TemporalRecord
from jerrythomas.transforms.utils import clone_record
from jerrythomas.utils.time import (
    parse_cadence,
    parse_timecode,
    round_time_to_cadence,
)


class RoundTimeTransform:
    """Round record timestamps in an explicit direction on a fixed UTC grid."""

    def __init__(self, cadence: str, direction: Literal["floor", "ceil"]) -> None:
        self.step = parse_cadence(cadence)
        self.direction = direction

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        for record in stream:
            time = round_time_to_cadence(
                record.time, self.step, self.direction
            ).astimezone(timezone.utc)
            yield clone_record(record, time=time)


class ShiftTimeTransform:
    """Shift record timestamps by a fixed duration."""

    def __init__(self, by: str) -> None:
        self.delta = parse_timecode(by)

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        for record in stream:
            yield clone_record(record, time=record.time + self.delta)
