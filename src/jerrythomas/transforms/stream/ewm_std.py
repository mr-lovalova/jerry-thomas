from collections.abc import Iterator
from math import isfinite, sqrt

from jerrythomas.domain.record import TemporalRecord
from jerrythomas.transforms.utils import (
    adjacent_partitions,
    clone_record_with_field,
    finite_number_or_none,
    get_field,
)


class EwmStdTransform:
    """Compute a recursive exponentially weighted standard deviation.

    Weights decay geometrically by ``1 - alpha`` per real observation; the
    newest observation carries weight 1. Track central squared deviations and
    the mean's offset from the newest observation to preserve small spreads
    at large levels. Missing values freeze the state and do not advance decay.
    Normalize the squared deviations by total weight when reading variance.
    """

    def __init__(
        self,
        field: str,
        alpha: float,
        partition_fields: tuple[str, ...],
        to: str | None = None,
        min_samples: int = 1,
    ) -> None:
        if isinstance(alpha, bool):
            raise TypeError("ewm_std alpha must be numeric")
        if not isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("ewm_std alpha must be greater than 0 and at most 1")
        if isinstance(min_samples, bool) or not isinstance(min_samples, int):
            raise TypeError("ewm_std min_samples must be an integer")
        if min_samples <= 0:
            raise ValueError("ewm_std min_samples must be positive")
        self.field = field
        self.alpha = float(alpha)
        self.to = field if to is None else to
        self.partition_fields = partition_fields
        self.min_samples = min_samples

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        for _, records in adjacent_partitions(stream, self.partition_fields):
            yield from _roll_partition(
                records,
                self.alpha,
                self.field,
                self.to,
                self.min_samples,
            )


def _roll_partition(
    records: Iterator[TemporalRecord],
    alpha: float,
    field: str,
    to: str,
    min_samples: int,
) -> Iterator[TemporalRecord]:
    beta = 1.0 - alpha
    sample_count = 0
    origin = 0.0
    weight = 0.0
    mean_offset = 0.0
    squared_deviations = 0.0

    for record in records:
        value = finite_number_or_none(get_field(record, field), field)
        if value is not None:
            sample_count += 1
            if sample_count == 1 or alpha == 1.0:
                origin = value
                weight = 1.0
                mean_offset = 0.0
                squared_deviations = 0.0
            else:
                previous_weight = beta * weight
                weight = previous_weight + 1.0
                previous_fraction = previous_weight / weight
                delta = (value - origin) - mean_offset
                squared_deviations = (
                    beta * squared_deviations + previous_fraction * delta * delta
                )
                mean_offset = -previous_fraction * delta
                origin = value

        if sample_count >= min_samples and sample_count > 1 and alpha != 1.0:
            variance = squared_deviations / weight
            if not isfinite(variance):
                raise OverflowError(
                    f"EWM std field {field!r} exceeds the supported "
                    "floating-point range"
                )
            output = sqrt(max(variance, 0.0))
        else:
            output = None
        yield clone_record_with_field(record, to, output)
