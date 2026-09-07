import copy
import math
from collections.abc import Iterator
from itertools import groupby
from typing import Any, TypeVar

from jerrythomas.domain.record import TemporalRecord


TRecord = TypeVar("TRecord", bound=TemporalRecord)


def is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def finite_number(value: Any, field: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise TypeError(f"Field {field!r} must contain numeric values")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Field {field!r} must contain numeric values") from exc
    if not math.isfinite(number):
        raise ValueError(f"Field {field!r} must contain finite numeric values")
    return number


def finite_number_or_none(value: Any, field: str) -> float | None:
    if is_missing(value):
        return None
    return finite_number(value, field)


def get_field(record: object, field: str) -> Any:
    try:
        return getattr(record, field)
    except AttributeError as exc:
        raise KeyError(
            f"Record field '{field}' not found on {type(record).__name__}"
        ) from exc


def partition_key(
    record: object,
    partition_by: tuple[str, ...],
) -> tuple[Any, ...]:
    values: list[Any] = []
    for field in partition_by:
        try:
            value = getattr(record, field)
        except AttributeError as exc:
            raise KeyError(
                f"Partition field '{field}' not found on {type(record).__name__}"
            ) from exc
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"Partition field {field!r} must contain finite floats")
        values.append(value)
    return tuple(values)


def adjacent_partitions(
    records: Iterator[TRecord],
    partition_by: tuple[str, ...],
) -> Iterator[tuple[tuple[Any, ...], Iterator[TRecord]]]:
    return groupby(
        records,
        key=lambda record: partition_key(record, partition_by),
    )


def record_establishes_domain(record: object) -> bool:
    try:
        value = getattr(record, "_establishes_domain")
    except AttributeError as exc:
        raise RuntimeError(
            f"{type(record).__name__} lost its sample-domain provenance."
        ) from exc
    if type(value) is not bool:
        raise TypeError("Record sample-domain provenance must be a boolean.")
    return value


def set_record_domain_anchor(record: object, establishes_domain: bool) -> None:
    setattr(record, "_establishes_domain", establishes_domain)


def clone_record(record: TRecord, **updates: Any) -> TRecord:
    """Shallow-copy a record, preserving dynamic fields and domain provenance."""
    cloned = copy.copy(record)
    for key, value in updates.items():
        setattr(cloned, key, value)
    if "time" in updates:
        TemporalRecord.__post_init__(cloned)
    return cloned


def clone_record_with_field(record: TRecord, field: str, value: Any) -> TRecord:
    """Return a shallow clone of record with a specific field updated."""
    return clone_record(record, **{field: value})
