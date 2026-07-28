from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal, Sequence, cast

from datapipeline.artifacts.models import (
    ListVectorMetadataEntry,
    VectorMetadataEntry,
)
from datapipeline.domain.sample import Sample
from datapipeline.domain.sample_key import SampleKeyValueType
from datapipeline.domain.value import normalize_data_value
from datapipeline.domain.series_id import base_id


TableValueType = Literal[
    "null",
    "boolean",
    "integer",
    "float",
    "string",
    "datetime",
    "date",
]

_MIN_INT64 = -(2**63)
_MAX_INT64 = 2**63 - 1
_PYTHON_VALUE_TYPES: dict[TableValueType, type] = {
    "boolean": bool,
    "integer": int,
    "float": float,
    "string": str,
    "datetime": datetime,
    "date": date,
}


@dataclass(frozen=True)
class TableColumn:
    name: str
    value_type: TableValueType
    nullable: bool


@dataclass(frozen=True)
class _VectorField:
    id: str
    is_list: bool
    columns: tuple[TableColumn, ...]


class DatasetTable:
    """Stable tabular projection for assembled dataset samples."""

    def __init__(
        self,
        sample_keys: Sequence[str],
        sample_key_types: Sequence[SampleKeyValueType],
        feature_entries: Sequence[VectorMetadataEntry],
        target_entries: Sequence[VectorMetadataEntry],
        scaled_feature_ids: Sequence[str] = (),
        scaled_target_ids: Sequence[str] = (),
    ) -> None:
        self.sample_keys = tuple(sample_keys)
        self.sample_key_types = tuple(sample_key_types)
        if len(self.sample_keys) != len(self.sample_key_types):
            raise ValueError("Sample key type count must match sample keys.")

        self._key_columns = (
            TableColumn("sample.time", "datetime", False),
            *(
                TableColumn(
                    f"sample.{field}",
                    _sample_key_table_type(value_type),
                    False,
                )
                for field, value_type in zip(
                    self.sample_keys,
                    self.sample_key_types,
                )
            ),
        )
        self._feature_fields = _vector_fields(
            "features",
            feature_entries,
            frozenset(scaled_feature_ids),
        )
        self._target_fields = _vector_fields(
            "targets",
            target_entries,
            frozenset(scaled_target_ids),
        )
        self.columns = (
            *self._key_columns,
            *(column for field in self._feature_fields for column in field.columns),
            *(column for field in self._target_fields for column in field.columns),
        )
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            duplicate = next(name for name in names if names.count(name) > 1)
            raise ValueError(f"Dataset table column {duplicate!r} is produced twice.")

        self._feature_ids = frozenset(entry.id for entry in feature_entries)
        self._target_ids = frozenset(entry.id for entry in target_entries)

    def project(self, sample: Sample) -> dict[str, Any]:
        key = sample.key
        if type(key) is not tuple:
            raise TypeError("Dataset sample key must be a tuple.")
        expected_key_size = len(self.sample_keys) + 1
        if len(key) != expected_key_size:
            raise ValueError(
                f"Dataset sample key has {len(key)} values; "
                f"expected {expected_key_size}."
            )

        sample_time = key[0]
        if not isinstance(sample_time, datetime) or sample_time.utcoffset() is None:
            raise ValueError("Dataset sample time must be timezone-aware.")

        row: dict[str, Any] = {"sample.time": sample_time}
        for column, value in zip(
            self._key_columns[1:],
            key[1:],
        ):
            row[column.name] = _table_value(column, value)

        feature_values = sample.features.values
        _project_vectors(
            row,
            "feature",
            feature_values,
            self._feature_ids,
            self._feature_fields,
        )

        target_values = {} if sample.targets is None else sample.targets.values
        _project_vectors(
            row,
            "target",
            target_values,
            self._target_ids,
            self._target_fields,
        )
        return row


def _vector_fields(
    namespace: str,
    entries: Sequence[VectorMetadataEntry],
    scaled_ids: frozenset[str],
) -> tuple[_VectorField, ...]:
    fields: list[_VectorField] = []
    for entry in entries:
        if isinstance(entry, ListVectorMetadataEntry):
            value_type = (
                "float"
                if base_id(entry.id) in scaled_ids
                else _metadata_table_type(entry.element_types)
            )
            fields.append(
                _VectorField(
                    entry.id,
                    True,
                    tuple(
                        TableColumn(
                            f"{namespace}.{entry.id}.{index}",
                            value_type,
                            True,
                        )
                        for index in range(entry.length)
                    ),
                )
            )
            continue
        fields.append(
            _VectorField(
                entry.id,
                False,
                (
                    TableColumn(
                        f"{namespace}.{entry.id}",
                        (
                            "float"
                            if base_id(entry.id) in scaled_ids
                            else _metadata_table_type(entry.value_types)
                        ),
                        True,
                    ),
                ),
            )
        )
    return tuple(fields)


def _metadata_table_type(type_names: Sequence[str]) -> TableValueType:
    concrete = frozenset(type_names) - {"null"}
    if not concrete:
        return "null"
    if concrete == {"int", "float"}:
        return "float"
    if len(concrete) != 1:
        raise ValueError(
            "Dataset table columns require one concrete value type; got "
            + ", ".join(sorted(concrete))
            + "."
        )
    type_name = next(iter(concrete))
    supported: dict[str, TableValueType] = {
        "bool": "boolean",
        "int": "integer",
        "float": "float",
        "str": "string",
        "datetime": "datetime",
        "date": "date",
    }
    try:
        return supported[type_name]
    except KeyError as exc:
        raise ValueError(
            f"Dataset table does not support value type {type_name!r}."
        ) from exc


def _sample_key_table_type(value_type: SampleKeyValueType) -> TableValueType:
    table_types: dict[SampleKeyValueType, TableValueType] = {
        "boolean": "boolean",
        "integer": "integer",
        "float": "float",
        "string": "string",
    }
    return table_types[value_type]


def _project_vectors(
    row: dict[str, Any],
    kind: str,
    values: dict[str, Any],
    expected: frozenset[str],
    fields: Sequence[_VectorField],
) -> None:
    unexpected = sorted(set(values) - expected)
    if unexpected:
        raise ValueError(
            f"Dataset sample contains unexpected {kind} ids: {unexpected!r}."
        )

    for field in fields:
        if field.id not in values:
            for column in field.columns:
                row[column.name] = None
            continue

        value = values[field.id]
        if not field.is_list:
            if isinstance(value, list):
                raise TypeError(
                    f"Dataset vector {field.id!r} must contain a scalar value."
                )
            column = field.columns[0]
            row[column.name] = _table_value(column, value)
            continue

        if not isinstance(value, list):
            raise TypeError(f"Dataset vector {field.id!r} must contain a list.")
        if len(value) != len(field.columns):
            raise ValueError(
                f"Dataset vector {field.id!r} requires {len(field.columns)} values; "
                f"got {len(value)}."
            )
        for column, element in zip(field.columns, value):
            row[column.name] = _table_value(column, element)


def _table_value(column: TableColumn, value: Any) -> Any:
    value = normalize_data_value(value)
    if value is None:
        if column.nullable:
            return None
        raise ValueError(f"Dataset table column {column.name!r} must not be null.")

    if column.value_type == "null":
        raise TypeError(f"Dataset table column {column.name!r} accepts only null.")
    if column.value_type == "float" and type(value) is int:
        try:
            converted = float(value)
        except OverflowError as exc:
            raise ValueError(
                f"Dataset table column {column.name!r} exceeds float64 range."
            ) from exc
        if int(converted) != value:
            raise ValueError(
                f"Dataset table column {column.name!r} cannot represent "
                f"integer {value!r} exactly as float64."
            )
        return converted
    expected = _PYTHON_VALUE_TYPES[column.value_type]
    if type(value) is not expected:
        raise TypeError(
            f"Dataset table column {column.name!r} requires "
            f"{column.value_type}; got {type(value).__name__}."
        )
    if isinstance(value, datetime) and value.utcoffset() is None:
        raise ValueError(
            f"Dataset table column {column.name!r} requires a timezone-aware datetime."
        )
    if column.value_type == "integer":
        integer = cast(int, value)
        if not _MIN_INT64 <= integer <= _MAX_INT64:
            raise ValueError(
                f"Dataset table column {column.name!r} requires a signed 64-bit integer."
            )
    return value
