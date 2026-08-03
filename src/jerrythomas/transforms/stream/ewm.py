from collections.abc import Iterator
from math import isfinite

from jerrythomas.domain.record import TemporalRecord
from jerrythomas.transforms.utils import (
    adjacent_partitions,
    clone_record_with_field,
    finite_number_or_none,
    get_field,
)


class EwmMeanTransform:
    """Compute a recursive, unadjusted exponentially weighted mean."""

    def __init__(
        self,
        field: str,
        alpha: float,
        partition_fields: tuple[str, ...],
        to: str | None = None,
        min_samples: int = 1,
    ) -> None:
        if isinstance(alpha, bool):
            raise TypeError("ewm_mean alpha must be numeric")
        if not isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("ewm_mean alpha must be greater than 0 and at most 1")
        if isinstance(min_samples, bool) or not isinstance(min_samples, int):
            raise TypeError("ewm_mean min_samples must be an integer")
        if min_samples <= 0:
            raise ValueError("ewm_mean min_samples must be positive")
        self.field = field
        self.alpha = float(alpha)
        self.to = field if to is None else to
        self.partition_fields = partition_fields
        self.min_samples = min_samples

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        for _, records in adjacent_partitions(stream, self.partition_fields):
            mean: float | None = None
            sample_count = 0
            for record in records:
                value = finite_number_or_none(get_field(record, self.field), self.field)
                if value is not None:
                    sample_count += 1
                    if mean is None or self.alpha == 1.0:
                        mean = value
                    else:
                        delta = value - mean
                        if isfinite(delta):
                            mean += self.alpha * delta
                        else:
                            mean = (1.0 - self.alpha) * mean + self.alpha * value
                    if not isfinite(mean):
                        raise OverflowError(
                            f"EWM mean field {self.field!r} exceeds the supported "
                            "floating-point range"
                        )
                output = mean if sample_count >= self.min_samples else None
                yield clone_record_with_field(record, self.to, output)
