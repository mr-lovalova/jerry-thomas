import math
from collections.abc import Mapping
from typing import Any


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
