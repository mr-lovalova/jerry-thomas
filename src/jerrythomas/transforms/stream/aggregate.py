from collections.abc import Iterator
from fractions import Fraction
from itertools import chain, groupby
from math import isfinite
from typing import Any

from jerrythomas.domain.record import TemporalRecord
from jerrythomas.transforms.utils import (
    clone_record,
    get_field,
    partition_key,
    record_establishes_domain,
    set_record_domain_anchor,
)


class _ExactFloatSum:
    """Sum finite binary floats exactly, rounding once at the end."""

    def __init__(self) -> None:
        self._numerator = 0
        self._denominator = 1

    def add(self, value: float) -> None:
        numerator, denominator = value.as_integer_ratio()
        if denominator > self._denominator:
            self._numerator *= denominator // self._denominator
            self._denominator = denominator
        else:
            numerator *= self._denominator // denominator
        self._numerator += numerator

    def result(self) -> float:
        return float(Fraction(self._numerator, self._denominator))


class AggregateSumTransform:
    """Sum one field for each adjacent canonical record key."""

    def __init__(
        self,
        field: str,
        partition_fields: tuple[str, ...],
        count_to: str | None = None,
    ) -> None:
        if count_to == field:
            raise ValueError("aggregate_sum count_to must differ from field")
        self.field = field
        self.partition_fields = partition_fields
        self.count_to = count_to

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        groups = groupby(
            stream,
            key=lambda record: (
                partition_key(record, self.partition_fields),
                record.time,
            ),
        )
        for _, records in groups:
            first = next(records)
            integer_total = 0
            floating_total: _ExactFloatSum | None = None
            count = 0
            establishes_domain = False
            for record in chain((first,), records):
                value = _aggregate_number(get_field(record, self.field), self.field)
                if type(value) is int:
                    integer_total += value
                else:
                    if floating_total is None:
                        floating_total = _ExactFloatSum()
                    floating_total.add(value)
                count += 1
                establishes_domain = (
                    record_establishes_domain(record) or establishes_domain
                )

            aggregate_total: int | float
            if floating_total is None:
                aggregate_total = integer_total
            else:
                floating_total.add(_exact_float(integer_total, self.field))
                try:
                    aggregate_total = floating_total.result()
                except OverflowError as exc:
                    raise OverflowError(
                        f"Aggregate sum field {self.field!r} exceeds the supported "
                        "floating-point range"
                    ) from exc

            updates: dict[str, object] = {self.field: aggregate_total}
            if self.count_to is not None:
                updates[self.count_to] = count
            aggregate = clone_record(first, **updates)
            set_record_domain_anchor(aggregate, establishes_domain)
            yield aggregate


def _aggregate_number(value: Any, field: str) -> int | float:
    if type(value) not in {int, float}:
        raise TypeError(f"Field {field!r} must contain numeric values")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"Field {field!r} must contain finite numeric values")
    return value


def _exact_float(value: int, field: str) -> float:
    try:
        converted = float(value)
    except OverflowError as exc:
        raise OverflowError(
            f"Aggregate sum field {field!r} exceeds the supported floating-point range"
        ) from exc
    if converted != value:
        raise ValueError(
            f"Aggregate sum field {field!r} cannot combine integer and "
            "floating-point values without precision loss"
        )
    return converted
