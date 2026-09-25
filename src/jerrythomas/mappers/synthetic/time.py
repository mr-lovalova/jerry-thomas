from dataclasses import dataclass
from datetime import datetime
from math import sin, pi
from typing import Iterator

from jerrythomas.domain.record import TemporalRecord


@dataclass
class TimeEncodedRecord(TemporalRecord):
    value: float


def encode(stream: Iterator[TemporalRecord], mode: str) -> Iterator[TemporalRecord]:
    for rec in stream:
        t: datetime = rec.time
        if mode == "hour_sin":
            val = sin(2 * pi * t.hour / 24)
        elif mode == "weekday_sin":
            val = sin(2 * pi * t.weekday() / 7)
        elif mode == "linear":
            val = t.timestamp()
        else:
            raise ValueError(f"Unsupported core.encode_time mode: {mode}")
        yield TimeEncodedRecord(time=rec.time, value=val)
