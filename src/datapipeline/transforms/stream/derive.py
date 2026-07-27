from collections.abc import Iterator
from math import isfinite
from typing import Literal

from datapipeline.domain.record import TemporalRecord
from datapipeline.transforms.utils import (
    clone_record_with_field,
    finite_number_or_none,
    get_field,
)

Operator = Literal["add", "sub", "mul", "div"]


class DeriveTransform:
    """Derive one record field from a binary arithmetic operation."""

    def __init__(
        self,
        left: str,
        operator: Operator,
        to: str,
        right_field: str | None = None,
        right_value: int | float | None = None,
    ) -> None:
        self.left = left
        self.operator = operator
        self.to = to
        self.right_field = right_field
        self.right_value = None if right_value is None else float(right_value)

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        for record in stream:
            left = finite_number_or_none(get_field(record, self.left), self.left)
            if self.right_field is not None:
                right = finite_number_or_none(
                    get_field(record, self.right_field),
                    self.right_field,
                )
            else:
                right = self.right_value
            value = self._derive(left, right)
            if value is not None and not isfinite(value):
                raise OverflowError(
                    f"Derived field {self.to!r} exceeds the supported "
                    "floating-point range"
                )
            yield clone_record_with_field(record, self.to, value)

    def _derive(self, left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        if self.operator == "add":
            return left + right
        if self.operator == "sub":
            return left - right
        if self.operator == "mul":
            return left * right
        if right == 0:
            raise ZeroDivisionError(f"Cannot divide by zero in field {self.to!r}")
        return left / right
