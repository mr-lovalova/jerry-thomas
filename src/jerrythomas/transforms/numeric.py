from fractions import Fraction
from math import isfinite
from typing import Any


class CompensatedSum:
    """Accumulate floating-point values with Neumaier compensation."""

    def __init__(self) -> None:
        self.total = 0.0
        self.correction = 0.0

    def add(self, value: float) -> None:
        updated = self.total + value
        if abs(self.total) >= abs(value):
            self.correction += (self.total - updated) + value
        else:
            self.correction += (value - updated) + self.total
        self.total = updated

    def reset(self) -> None:
        self.total = 0.0
        self.correction = 0.0

    def result(self) -> float:
        return self.total + self.correction


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

    def fraction(self) -> Fraction:
        return Fraction(self._numerator, self._denominator)


class ExactNumericSum:
    """Preserve integer sums and round finite float sums only at completion."""

    def __init__(self, field: str) -> None:
        self.field = field
        self.integer_total = 0
        self.floating_total: _ExactFloatSum | None = None

    def add(self, value: object) -> None:
        number = finite_aggregate_number(value, self.field)
        if type(number) is int:
            self.integer_total += number
        else:
            if self.floating_total is None:
                self.floating_total = _ExactFloatSum()
            self.floating_total.add(number)

    def result(self, divisor: int = 1) -> int | float:
        if self.floating_total is None and divisor == 1:
            return self.integer_total
        try:
            if self.floating_total is None:
                return float(Fraction(self.integer_total, divisor))
            total = self.floating_total.fraction() + Fraction(
                _exact_float(self.integer_total, self.field)
            )
            return float(total / divisor)
        except OverflowError as exc:
            raise OverflowError(
                f"Aggregate sum field {self.field!r} exceeds the supported floating-point range"
            ) from exc


def finite_aggregate_number(value: Any, field: str) -> int | float:
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
