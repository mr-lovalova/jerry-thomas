from collections.abc import Iterator
from math import log, log1p

from datapipeline.domain.record import TemporalRecord
from datapipeline.transforms.utils import (
    clone_record_with_field,
    finite_number_or_none,
    get_field,
)


class LogTransform:
    """Write the natural logarithm of one record field."""

    def __init__(self, field: str, to: str) -> None:
        self.field = field
        self.to = to

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        for record in stream:
            number = finite_number_or_none(get_field(record, self.field), self.field)
            if number is None:
                result = None
            else:
                if number <= 0:
                    raise ValueError(
                        f"Field {self.field!r} must be greater than zero for log"
                    )
                result = log(number)
            yield clone_record_with_field(record, self.to, result)


class Log1pTransform:
    """Write log(1 + x) accurately for one record field."""

    def __init__(self, field: str, to: str) -> None:
        self.field = field
        self.to = to

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        for record in stream:
            number = finite_number_or_none(get_field(record, self.field), self.field)
            if number is None:
                result = None
            else:
                if number <= -1:
                    raise ValueError(
                        f"Field {self.field!r} must be greater than -1 for log1p"
                    )
                result = log1p(number)
            yield clone_record_with_field(record, self.to, result)
