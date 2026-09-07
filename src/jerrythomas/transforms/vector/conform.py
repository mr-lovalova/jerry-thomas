from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from jerrythomas.artifacts.models import VectorMetadataEntry
from jerrythomas.domain.sample import Sample
from jerrythomas.domain.vector import Vector
from jerrythomas.transforms.utils import is_missing


class _VectorConformer:
    def __init__(self, entries: Sequence[VectorMetadataEntry]) -> None:
        self.entries = tuple(entries)
        if not self.entries:
            raise ValueError("metadata entries must not be empty")
        identifiers = [entry.id for entry in self.entries]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("metadata entry ids must be unique")
        self.expected_id_set = frozenset(identifiers)

    def apply(self, values: Mapping[str, Any]) -> dict[str, Any]:
        extras = [
            identifier
            for identifier in values
            if identifier not in self.expected_id_set
        ]
        if extras:
            raise ValueError(f"Vector contains unexpected ids: {extras!r}")

        conformed: dict[str, Any] = {}
        for entry in self.entries:
            value = values.get(entry.id)
            if is_missing(value):
                value = None if entry.kind == "scalar" else [None] * entry.length
            elif entry.kind == "scalar":
                if isinstance(value, list):
                    raise TypeError(f"Vector id {entry.id!r} must contain a scalar.")
            else:
                if not isinstance(value, list):
                    raise TypeError(f"Vector id {entry.id!r} must contain a list.")
                if len(value) != entry.length:
                    raise ValueError(
                        f"Vector id {entry.id!r} requires {entry.length} values; "
                        f"got {len(value)}."
                    )
            conformed[entry.id] = value
        return conformed


class ConformFeaturesTransform:
    """Conform feature vectors to their metadata contract."""

    def __init__(self, entries: Sequence[VectorMetadataEntry]) -> None:
        self._conformer = _VectorConformer(entries)

    def apply(self, stream: Iterator[Sample]) -> Iterator[Sample]:
        for sample in stream:
            values = self._conformer.apply(sample.features.values)
            yield sample.with_features(Vector(values=values))


class ConformTargetsTransform:
    """Conform target vectors to their metadata contract."""

    def __init__(self, entries: Sequence[VectorMetadataEntry]) -> None:
        self._conformer = _VectorConformer(entries)

    def apply(self, stream: Iterator[Sample]) -> Iterator[Sample]:
        for sample in stream:
            current = sample.targets.values if sample.targets is not None else {}
            values = self._conformer.apply(current)
            yield sample.with_targets(Vector(values=values))
