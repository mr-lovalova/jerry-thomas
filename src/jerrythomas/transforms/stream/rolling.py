from collections.abc import Iterator
from math import isfinite

from jerrythomas.domain.record import TemporalRecord
from jerrythomas.transforms.rolling_window import (
    RollingMaximum,
    RollingMean,
    RollingMedian,
    RollingMinimum,
    RollingPopulationStandardDeviation,
    RollingSampleStandardDeviation,
    RollingWindow,
)
from jerrythomas.transforms.utils import (
    adjacent_partitions,
    clone_record_with_field,
    finite_number_or_none,
    get_field,
)


_STATISTICS: dict[str, type[RollingWindow]] = {
    "mean": RollingMean,
    "median": RollingMedian,
    "stdev": RollingSampleStandardDeviation,
    "pstdev": RollingPopulationStandardDeviation,
    "max": RollingMaximum,
    "min": RollingMinimum,
}


class RollingTransform:
    """Compute a rolling statistic over record field values."""

    def __init__(
        self,
        field: str,
        window: int,
        partition_fields: tuple[str, ...],
        to: str | None = None,
        min_samples: int | None = None,
        statistic: str = "mean",
    ) -> None:
        self.field = field
        self.to = field if to is None else to
        self.partition_fields = partition_fields
        self._window_type = _STATISTICS[statistic]
        self.window = window
        self.min_samples = window if min_samples is None else min_samples

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        for _, records in adjacent_partitions(stream, self.partition_fields):
            rolling_window = self._window_type(self.window)

            for record in records:
                value = finite_number_or_none(get_field(record, self.field), self.field)
                rolling_window.append(value)
                if rolling_window.sample_count >= self.min_samples:
                    rolled = rolling_window.result()
                    if not isfinite(rolled):
                        raise OverflowError(
                            f"Rolling field {self.field!r} exceeds the supported "
                            "floating-point range"
                        )
                else:
                    rolled = None
                yield clone_record_with_field(record, self.to, rolled)
