from collections.abc import Iterator

from datapipeline.domain.record import TemporalRecord
from datapipeline.transforms.rolling_slope import RollingSlope
from datapipeline.transforms.utils import (
    adjacent_partitions,
    clone_record_with_field,
    finite_number,
    get_field,
    is_missing,
)


class RollingSlopeTransform:
    """Compute the rolling least-squares slope of y on x."""

    def __init__(
        self,
        x: str,
        y: str,
        window: int,
        partition_fields: tuple[str, ...],
        to: str,
    ) -> None:
        self.x = x
        self.y = y
        self.window = window
        self.partition_fields = partition_fields
        self.to = to

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        for _, records in adjacent_partitions(stream, self.partition_fields):
            rolling_slope = RollingSlope(self.window)

            for record in records:
                x = self._number(get_field(record, self.x), self.x)
                y = self._number(get_field(record, self.y), self.y)
                if x is None or y is None:
                    rolling_slope.clear()
                    slope = None
                else:
                    rolling_slope.append(x, y)
                    slope = rolling_slope.result() if rolling_slope.full else None
                yield clone_record_with_field(record, self.to, slope)

    @staticmethod
    def _number(value: object, field: str) -> float | None:
        if is_missing(value):
            return None
        return finite_number(value, field)
