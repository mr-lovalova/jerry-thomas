from collections.abc import Iterator
from typing import Any

from datapipeline.config.interpolation import coalesce_missing_interpolation
from datapipeline.sources.loader import GeneratorLoader
from datapipeline.utils.time import parse_datetime, parse_timecode


class TimeTicksGenerator:
    def __init__(self, start: str, end: str, frequency: str | None = "1h"):
        self.start = parse_datetime(start)
        self.end = parse_datetime(end)
        self.frequency = parse_timecode(frequency or "1h")
        if self.frequency.total_seconds() <= 0:
            raise ValueError("frequency must be positive")
        if self.end < self.start:
            raise ValueError("end must not precede start")

    def generate(self) -> Iterator[dict[str, Any]]:
        current = self.start
        while current <= self.end:
            yield {"time": current}
            current += self.frequency


def make_time_loader(
    start: str,
    end: str,
    frequency: str | None = "1h",
) -> GeneratorLoader:
    """Build a bounded synthetic time source."""
    start_value = coalesce_missing_interpolation(start)
    end_value = coalesce_missing_interpolation(end)
    frequency_value = coalesce_missing_interpolation(frequency, default="1h")

    if start_value is None or end_value is None:
        raise ValueError(
            "synthetic time loader requires non-null start and end; "
            "set explicit project.globals.start_time/end_time or override source.loader.args."
        )
    return GeneratorLoader(
        TimeTicksGenerator(start_value, end_value, frequency_value),
        progress_unit="ticks",
    )
