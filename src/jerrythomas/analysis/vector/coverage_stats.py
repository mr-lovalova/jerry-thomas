from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jerrythomas.artifacts.models import (
    CoverageBaseStats,
    CoverageColumnStats,
    CoverageStatsSection,
    ListCoverageColumnStats,
    ListVectorMetadataEntry,
    ScalarCoverageColumnStats,
    VectorMetadataEntry,
)
from jerrythomas.transforms.utils import is_missing


@dataclass(slots=True)
class _Counts:
    present_samples: int = 0
    non_null_samples: int = 0
    observed_elements: int = 0


class CoverageStatsAccumulator:
    """Collect bounded availability counters for one vector section."""

    def __init__(self, entries: Sequence[VectorMetadataEntry]) -> None:
        self._entries = {entry.id: entry for entry in entries}
        self._bases = {entry.base_id: _Counts() for entry in entries}
        self._columns = {entry.id: _Counts() for entry in entries}

    def update(self, values: Mapping[str, Any]) -> None:
        present_bases: set[str] = set()
        non_null_bases: set[str] = set()

        for identifier, value in values.items():
            entry = self._entries.get(identifier)
            if entry is None:
                raise ValueError(
                    f"Vector contains ID {identifier!r} missing from metadata. "
                    "Rebuild vector metadata."
                )

            present_bases.add(entry.base_id)
            counts = self._columns[identifier]
            counts.present_samples += 1

            if isinstance(entry, ListVectorMetadataEntry):
                observed = _observed_list_elements(entry, value)
                counts.observed_elements += observed
                if observed:
                    counts.non_null_samples += 1
                    non_null_bases.add(entry.base_id)
            else:
                if isinstance(value, list):
                    raise ValueError(
                        f"Scalar vector {identifier!r} contains a list value."
                    )
                if not is_missing(value):
                    counts.non_null_samples += 1
                    non_null_bases.add(entry.base_id)

        for base_id in present_bases:
            self._bases[base_id].present_samples += 1
        for base_id in non_null_bases:
            self._bases[base_id].non_null_samples += 1

    def finish(self) -> CoverageStatsSection:
        bases = tuple(
            CoverageBaseStats(
                id=identifier,
                present_samples=counts.present_samples,
                non_null_samples=counts.non_null_samples,
            )
            for identifier, counts in sorted(self._bases.items())
        )
        columns: list[CoverageColumnStats] = []
        for identifier, entry in sorted(self._entries.items()):
            counts = self._columns[identifier]
            if isinstance(entry, ListVectorMetadataEntry):
                columns.append(
                    ListCoverageColumnStats(
                        id=identifier,
                        base_id=entry.base_id,
                        present_samples=counts.present_samples,
                        non_null_samples=counts.non_null_samples,
                        kind="list",
                        length=entry.length,
                        observed_elements=counts.observed_elements,
                    )
                )
            else:
                columns.append(
                    ScalarCoverageColumnStats(
                        id=identifier,
                        base_id=entry.base_id,
                        present_samples=counts.present_samples,
                        non_null_samples=counts.non_null_samples,
                        kind="scalar",
                    )
                )
        return CoverageStatsSection(bases=bases, columns=tuple(columns))


def _observed_list_elements(
    entry: ListVectorMetadataEntry,
    value: object,
) -> int:
    if is_missing(value):
        return 0
    if not isinstance(value, list):
        raise ValueError(f"List vector {entry.id!r} contains a scalar value.")
    expected = entry.length
    if len(value) != expected:
        raise ValueError(
            f"List vector {entry.id!r} has length {len(value)}; expected {expected}."
        )
    return sum(not is_missing(element) for element in value)
