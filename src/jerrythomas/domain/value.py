import math
from collections.abc import Mapping
from typing import Any


_SERIES_SCALAR_TYPES = {type(None), bool, int, float, str}


def normalize_data_value(value: Any) -> Any:
    """Use None for missing floats and reject infinity in data values."""

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return None
        raise ValueError("Data values must not contain infinity.")
    if isinstance(value, (list, tuple)):
        return _normalize_sequence(value)
    if isinstance(value, Mapping):
        items = iter(value.items())
        for key, item in items:
            normalized = normalize_data_value(item)
            if normalized is item:
                continue
            normalized_mapping = dict(value.items())
            normalized_mapping[key] = normalized
            for remaining_key, remaining_item in items:
                normalized_mapping[remaining_key] = normalize_data_value(remaining_item)
            return normalized_mapping
        return value
    return value


def normalize_series_value(value: Any) -> Any:
    """Normalize a value admitted by the flat series artifact contract."""

    value = normalize_data_value(value)
    validate_series_value(value)
    return value


def validate_series_value(value: Any) -> None:
    """Require a canonical scalar or non-empty flat list of scalars."""

    values: list[Any] | tuple[Any, ...]
    if type(value) is list:
        if not value:
            raise ValueError("Series list values must not be empty.")
        values = value
    else:
        values = (value,)

    for item in values:
        if type(item) not in _SERIES_SCALAR_TYPES:
            raise TypeError(
                "Series values must be a scalar or a flat list of scalar values; "
                f"got {type(item).__name__}."
            )
        if type(item) is float and not math.isfinite(item):
            raise ValueError("Series values must contain only finite floats or None.")


def _normalize_sequence(
    value: list[Any] | tuple[Any, ...],
) -> list[Any] | tuple[Any, ...]:
    for index, item in enumerate(value):
        normalized = normalize_data_value(item)
        if normalized is item:
            continue
        normalized_items = list(value)
        normalized_items[index] = normalized
        for remaining in range(index + 1, len(value)):
            normalized_items[remaining] = normalize_data_value(value[remaining])
        return tuple(normalized_items) if isinstance(value, tuple) else normalized_items
    return value
