from collections.abc import Iterator
from itertools import groupby
from math import isfinite

from datapipeline.domain.record import TemporalRecord
from datapipeline.transforms.rolling_window import (
    RollingMean,
    RollingMedian,
    RollingWindow,
)
from datapipeline.transforms.utils import (
    clone_record_with_field,
    finite_number,
    get_field,
    is_missing,
    partition_key,
)


_STATISTICS: dict[str, type[RollingWindow]] = {
    "mean": RollingMean,
    "median": RollingMedian,
}


class StatisticalFillTransform:
    """Fill missing values from the preceding fixed-size tick window."""

    def __init__(
        self,
        field: str,
        window: int,
        statistic: str,
        partition_fields: tuple[str, ...],
        to: str | None = None,
        min_samples: int = 1,
    ) -> None:
        self.field = field
        self.to = field if to is None else to
        self.partition_fields = partition_fields
        self._window_type = _STATISTICS[statistic]
        self.window = window
        self.min_samples = min_samples

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        for _, records in groupby(
            stream,
            key=lambda record: partition_key(record, self.partition_fields),
        ):
            history = self._window_type(self.window)
            for record in records:
                raw_value = get_field(record, self.field)
                if is_missing(raw_value):
                    value = (
                        history.result()
                        if history.sample_count >= self.min_samples
                        else None
                    )
                    history.append(None)
                else:
                    numeric_value = finite_number(raw_value, self.field)
                    history.append(numeric_value)
                    value = numeric_value
                if value is not None and not isfinite(value):
                    raise OverflowError(
                        f"Fill field {self.field!r} exceeds the supported "
                        "floating-point range"
                    )
                yield clone_record_with_field(record, self.to, value)


class ForwardFillTransform:
    """Carry the most recent nonmissing field value within each partition."""

    def __init__(
        self,
        field: str,
        partition_fields: tuple[str, ...],
        to: str | None = None,
    ) -> None:
        self.field = field
        self.to = field if to is None else to
        self.partition_fields = partition_fields

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        for _, records in groupby(
            stream,
            key=lambda record: partition_key(record, self.partition_fields),
        ):
            last_value = None
            for record in records:
                value = get_field(record, self.field)
                if not is_missing(value):
                    last_value = value
                yield clone_record_with_field(record, self.to, last_value)
