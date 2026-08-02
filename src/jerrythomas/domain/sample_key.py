from collections.abc import Sequence
from math import isfinite
from typing import Literal


SampleKeyValueType = Literal["string", "boolean", "integer", "float"]


def sample_key_value_type(field: str, value: object) -> SampleKeyValueType:
    if value is None:
        raise ValueError(f"Sample key field {field!r} must not be null.")
    if type(value) is str:
        return "string"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"Sample key field {field!r} must contain finite floats.")
        return "float"
    raise TypeError(
        f"Sample key field {field!r} must contain a string, integer, float, "
        f"or boolean; got {type(value).__name__}."
    )


class SampleKeyContract:
    def __init__(
        self,
        fields: Sequence[str],
        expected_types: Sequence[SampleKeyValueType] | None = None,
    ) -> None:
        self.fields = tuple(fields)
        if expected_types is None:
            self._types: list[SampleKeyValueType | None] = [None] * len(self.fields)
        else:
            if len(expected_types) != len(self.fields):
                raise ValueError(
                    "Sample key type count must match the configured sample keys."
                )
            self._types = list(expected_types)

    @property
    def types(self) -> tuple[SampleKeyValueType, ...]:
        missing = [
            field
            for field, value_type in zip(self.fields, self._types)
            if value_type is None
        ]
        if missing:
            raise ValueError(
                "Sample key fields produced no values: " + ", ".join(missing)
            )
        return tuple(value_type for value_type in self._types if value_type is not None)

    def validate(self, values: Sequence[object]) -> None:
        if len(values) != len(self.fields):
            raise ValueError(
                f"Sample key has {len(values)} values; expected {len(self.fields)}."
            )
        for index, (field, value) in enumerate(zip(self.fields, values)):
            value_type = sample_key_value_type(field, value)
            expected = self._types[index]
            if expected is None:
                self._types[index] = value_type
                continue
            if value_type != expected:
                raise TypeError(
                    f"Sample key field {field!r} changed type from {expected} "
                    f"to {value_type}."
                )
