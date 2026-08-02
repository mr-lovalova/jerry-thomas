from collections.abc import Mapping, Sequence, Set
from typing import Any

from jerrythomas.transforms.utils import is_missing


def select_expected_ids(
    expected_ids: Sequence[str],
    ids: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    expected = tuple(expected_ids)
    if not expected:
        raise ValueError("expected_ids must not be empty")
    if any(
        not isinstance(identifier, str) or not identifier for identifier in expected
    ):
        raise ValueError("expected_ids must contain non-empty strings")
    expected_set = set(expected)
    if len(expected_set) != len(expected):
        raise ValueError("expected_ids must not contain duplicates")

    selected = expected if ids is None else tuple(ids)
    if not selected:
        raise ValueError("ids must not be empty")
    if any(
        not isinstance(identifier, str) or not identifier for identifier in selected
    ):
        raise ValueError("ids must contain non-empty strings")
    if len(set(selected)) != len(selected):
        raise ValueError("ids must not contain duplicates")

    unknown = [identifier for identifier in selected if identifier not in expected_set]
    if unknown:
        raise ValueError(f"Unknown vector ids: {unknown!r}")
    return expected, selected


def cell_coverage(value: Any) -> float:
    if isinstance(value, list):
        if not value:
            return 0.0
        present = sum(not is_missing(item) for item in value)
        return present / len(value)
    return 0.0 if is_missing(value) else 1.0


def reject_unknown_values(
    values: Mapping[str, Any],
    expected_ids: Set[str],
) -> None:
    unknown = [identifier for identifier in values if identifier not in expected_ids]
    if unknown:
        raise ValueError(f"Vector contains unexpected ids: {unknown!r}")
