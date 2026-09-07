import operator as comparison
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any, Callable

from jerrythomas.transforms.utils import get_field
from jerrythomas.utils.time import parse_datetime


_COMPARISONS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": comparison.eq,
    "ne": comparison.ne,
    "lt": comparison.lt,
    "le": comparison.le,
    "gt": comparison.gt,
    "ge": comparison.ge,
}
_MEMBERSHIP = ("in", "not_in")


class WhereTransform:
    """Keep records whose field satisfies one explicit comparison."""

    def __init__(self, field: str, operator: str, comparand: Any) -> None:
        if field == "time":
            if operator in _MEMBERSHIP:
                comparand = tuple(_as_utc_datetime(value) for value in comparand)
            else:
                comparand = _as_utc_datetime(comparand)

        self.field = field
        self.operator = operator
        self.comparand = comparand

    def apply(self, stream: Iterator[Any]) -> Iterator[Any]:
        if self.operator in _MEMBERSHIP:
            keep_matches = self.operator == "in"
            for record in stream:
                value = get_field(record, self.field)
                if self.field == "time":
                    value = _record_time(value)
                if (value in self.comparand) == keep_matches:
                    yield record
            return

        predicate = _COMPARISONS[self.operator]
        for record in stream:
            value = get_field(record, self.field)
            if self.field == "time":
                value = _record_time(value)
            try:
                selected = predicate(value, self.comparand)
            except TypeError as exc:
                raise TypeError(
                    f"Cannot apply where operator {self.operator!r} to field "
                    f"{self.field!r}: {type(value).__name__} and "
                    f"{type(self.comparand).__name__}"
                ) from exc
            if selected:
                yield record


def _record_time(value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(
            "Where field 'time' must contain a datetime value, "
            f"got {type(value).__name__}"
        )
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _as_utc_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return parse_datetime(value).astimezone(timezone.utc)
