from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from datapipeline.artifacts.models import (
    ListVectorMetadataEntry,
    ScalarVectorMetadataEntry,
    VectorMetadataEntry,
)
from datapipeline.transforms.utils import is_missing


def _type_name(value: object) -> str:
    if value is None:
        return "null"
    return type(value).__name__


@dataclass
class VectorMetadataStats:
    id: str
    base_id: str
    kind: Literal["scalar", "list"] | None = None
    list_length: int | None = None
    present_count: int = 0
    null_count: int = 0
    scalar_types: set[str] = field(default_factory=set)
    element_types: set[str] = field(default_factory=set)
    first_observed: datetime | None = None
    last_observed: datetime | None = None
    observed_elements: int = 0

    def observe(self, value: object, observed_at: datetime | None) -> None:
        if observed_at is not None:
            self.first_observed = (
                observed_at
                if self.first_observed is None
                else min(self.first_observed, observed_at)
            )
            self.last_observed = (
                observed_at
                if self.last_observed is None
                else max(self.last_observed, observed_at)
            )

        self.present_count += 1
        if is_missing(value):
            self.null_count += 1
            return

        if isinstance(value, list):
            if self.kind == "scalar":
                raise ValueError(
                    f"Vector {self.id!r} contains both scalar and list values."
                )
            if not value:
                raise ValueError(
                    f"Vector {self.id!r} contains an empty list; "
                    "list vectors require a positive fixed length."
                )
            length = len(value)
            if self.list_length is not None and self.list_length != length:
                raise ValueError(
                    f"Vector {self.id!r} contains list values with different "
                    f"lengths: {self.list_length} and {length}."
                )
            self.kind = "list"
            self.list_length = length
            self.observed_elements += sum(
                1 for element in value if not is_missing(element)
            )
            self.element_types.update(_type_name(element) for element in value)
            return

        if self.kind == "list":
            raise ValueError(
                f"Vector {self.id!r} contains both list and scalar values."
            )
        self.kind = "scalar"
        self.scalar_types.add(_type_name(value))


def metadata_entries_from_stats(
    entries: Sequence[VectorMetadataStats],
) -> tuple[VectorMetadataEntry, ...]:
    metadata_entries: list[VectorMetadataEntry] = []
    for entry in entries:
        kind = entry.kind or "scalar"
        if kind == "list":
            length = entry.list_length
            if length is None:
                raise ValueError(
                    f"List metadata entry {entry.id!r} has no fixed sequence length."
                )
            metadata_entries.append(
                ListVectorMetadataEntry(
                    id=entry.id,
                    base_id=entry.base_id,
                    kind="list",
                    present_count=entry.present_count,
                    null_count=entry.null_count,
                    first_observed=entry.first_observed,
                    last_observed=entry.last_observed,
                    element_types=tuple(sorted(entry.element_types)),
                    length=length,
                    observed_elements=entry.observed_elements,
                )
            )
            continue
        if kind != "scalar":
            raise ValueError(f"Unsupported vector metadata kind {kind!r}.")
        metadata_entries.append(
            ScalarVectorMetadataEntry(
                id=entry.id,
                base_id=entry.base_id,
                kind="scalar",
                present_count=entry.present_count,
                null_count=entry.null_count,
                first_observed=entry.first_observed,
                last_observed=entry.last_observed,
                value_types=tuple(sorted(entry.scalar_types)),
            )
        )
    return tuple(metadata_entries)
