from __future__ import annotations

import math
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from jerrythomas.artifacts.hydration import hydrate_runtime_artifacts_for_pipeline
from jerrythomas.artifacts.models import (
    UnsplitMetadataLayout,
    VectorMetadataEntry,
    VectorSchema,
)
from jerrythomas.artifacts.registry import VECTOR_METADATA_SPEC
from jerrythomas.artifacts.specs import dataset_requires_scaler
from jerrythomas.config.dataset.split import (
    resolve_fold_output,
    split_output_ids,
)
from jerrythomas.domain.sample import Sample
from jerrythomas.domain.vector import Vector
from jerrythomas.pipelines.dataset.pipeline import (
    FoldOutputPlan,
    resolve_fold_output_plans,
    run_dataset_pipeline,
    run_fold_dataset_pipeline,
    run_scaled_dataset_pipeline,
)
from jerrythomas.pipelines.sample.keys import (
    RectangularKeyPlan,
    require_metadata_key_plan,
)
from jerrythomas.runtime import Runtime
from jerrythomas.services.project_definition import load_project_definition
from jerrythomas.services.runtime_compiler import compile_runtime

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    _FloatArray = NDArray[np.floating[Any]]
else:
    _FloatArray = Any


@dataclass(eq=False, slots=True)
class ModelBatch:
    """One bounded, metadata-ordered numerical batch."""

    keys: tuple[tuple[Any, ...], ...]
    features: _FloatArray
    targets: _FloatArray | None
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SampleSource:
    runtime: Runtime
    fold_output: FoldOutputPlan | None
    schema: VectorSchema
    key_plan: RectangularKeyPlan | None

    @classmethod
    def from_project(
        cls,
        project_yaml: str | Path,
        output_id: str | None,
    ) -> _SampleSource:
        definition = load_project_definition(Path(project_yaml))
        split = definition.dataset.split
        if split is None:
            if output_id is not None:
                raise ValueError(
                    "output_id is only valid when dataset.split is configured."
                )
        else:
            available = split_output_ids(split)
            if output_id is None:
                raise ValueError(
                    "Dataset defines split outputs; pass output_id with one of: "
                    + ", ".join(available)
                )
            try:
                resolve_fold_output(split, output_id)
            except KeyError as exc:
                raise ValueError(
                    f"Dataset output {output_id!r} is not defined; choose one of: "
                    + ", ".join(available)
                ) from exc

        runtime = compile_runtime(definition)
        hydrate_runtime_artifacts_for_pipeline(runtime, definition)
        metadata = runtime.artifacts.load(VECTOR_METADATA_SPEC)
        if split is None:
            if not isinstance(metadata.layout, UnsplitMetadataLayout):
                raise RuntimeError(
                    "Unsplit dataset requires unsplit metadata. "
                    "Rebuild build/metadata.json."
                )
            key_plan = require_metadata_key_plan(
                metadata.catalog.window,
                metadata.catalog.sample,
                definition.dataset.sample.cadence,
                definition.dataset.sample.keys,
            )
            return cls(
                runtime=runtime,
                fold_output=None,
                schema=metadata.catalog,
                key_plan=key_plan,
            )

        assert output_id is not None
        fold_output = resolve_fold_output_plans(runtime, (output_id,))[0]
        return cls(
            runtime=runtime,
            fold_output=fold_output,
            schema=fold_output.schema,
            key_plan=None,
        )

    def iter_samples(self, limit: int | None) -> Generator[Sample, None, None]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        dataset = self.runtime.dataset
        if not dataset.features:
            raise ValueError(
                "Dataset does not define any features. Configure at least one feature "
                "stream before iterating samples."
            )

        samples: Iterator[Sample]
        if self.fold_output is not None:
            samples = run_fold_dataset_pipeline(
                self.runtime,
                self.fold_output,
            )
        else:
            run = (
                run_scaled_dataset_pipeline
                if dataset_requires_scaler(dataset)
                else run_dataset_pipeline
            )
            samples = run(
                self.runtime,
                self.schema,
                self.key_plan,
            )

        try:
            if limit is None:
                yield from samples
            else:
                yield from islice(samples, limit)
        finally:
            close = getattr(samples, "close", None)
            if callable(close):
                close()


def iter_samples(
    project_yaml: str | Path,
    *,
    output_id: str | None = None,
    limit: int | None = None,
) -> Iterator[Sample]:
    """Stream final samples from current Jerry artifacts."""

    source = _SampleSource.from_project(project_yaml, output_id)
    return source.iter_samples(limit)


def iter_model_batches(
    project_yaml: str | Path,
    *,
    output_id: str | None = None,
    batch_size: int = 4096,
    limit: int | None = None,
    dtype: Literal["float32", "float64"] = "float32",
) -> Iterator[ModelBatch]:
    """Stream bounded numerical batches from the final dataset pipeline."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by runtime users
        raise RuntimeError(
            "NumPy is required for model batches; install jerry-thomas[ml]."
        ) from exc

    source = _SampleSource.from_project(project_yaml, output_id)
    feature_entries = source.schema.features
    target_entries = source.schema.targets
    feature_columns = _columns(feature_entries)
    target_columns = _columns(target_entries)
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Postprocessed features produce duplicate model columns.")
    if len(target_columns) != len(set(target_columns)):
        raise ValueError("Postprocessed targets produce duplicate model columns.")

    pending: list[Sample] = []
    samples = source.iter_samples(limit)
    try:
        for sample in samples:
            pending.append(sample)
            if len(pending) == batch_size:
                yield _model_batch(
                    np,
                    pending,
                    feature_entries,
                    target_entries,
                    feature_columns,
                    target_columns,
                    dtype,
                )
                pending = []
    finally:
        samples.close()

    if pending:
        yield _model_batch(
            np,
            pending,
            feature_entries,
            target_entries,
            feature_columns,
            target_columns,
            dtype,
        )


def _columns(entries: Sequence[VectorMetadataEntry]) -> tuple[str, ...]:
    columns: list[str] = []
    for entry in entries:
        if entry.kind == "list":
            columns.extend(f"{entry.id}[{index}]" for index in range(entry.length))
        else:
            columns.append(entry.id)
    return tuple(columns)


def _model_row(
    vector: Vector,
    entries: Sequence[VectorMetadataEntry],
) -> list[Any]:
    row: list[Any] = []
    for entry in entries:
        value = vector.values[entry.id]
        if entry.kind == "list":
            row.extend(value)
        else:
            row.append(value)
    return row


def _model_batch(
    np: Any,
    samples: Sequence[Sample],
    feature_entries: Sequence[VectorMetadataEntry],
    target_entries: Sequence[VectorMetadataEntry],
    feature_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    dtype: Literal["float32", "float64"],
) -> ModelBatch:
    keys = tuple(tuple(sample.key) for sample in samples)
    feature_rows = [_model_row(sample.features, feature_entries) for sample in samples]
    target_rows: list[list[Any]] | None = None
    if target_entries:
        target_rows = []
        for sample, key in zip(samples, keys):
            if sample.targets is None:
                raise RuntimeError(
                    f"Sample {key!r} has no targets, but target columns are declared."
                )
            target_rows.append(_model_row(sample.targets, target_entries))

    features = _numeric_array(
        np,
        feature_rows,
        keys,
        feature_columns,
        dtype,
    )
    targets = (
        None
        if target_rows is None
        else _numeric_array(np, target_rows, keys, target_columns, dtype)
    )

    return ModelBatch(
        keys=keys,
        features=features,
        targets=targets,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )


def _numeric_array(
    np: Any,
    rows: Sequence[Sequence[Any]],
    keys: Sequence[tuple[Any, ...]],
    columns: Sequence[str],
    dtype: str,
) -> _FloatArray:
    expected_shape = (len(rows), len(columns))
    try:
        values = np.asarray(rows)
    except ValueError as exc:
        _validate_model_rows(rows, keys, columns)
        raise ValueError("Model batch rows do not match the declared columns.") from exc

    wrong_shape = values.ndim != 2 or values.shape != expected_shape
    value_types = {type(value) for row in rows for value in row}
    contains_invalid_type = any(
        value_type is bool or not issubclass(value_type, Real)
        for value_type in value_types
    )
    if wrong_shape or contains_invalid_type or values.dtype.kind not in "iuf":
        _validate_model_rows(rows, keys, columns)
    if wrong_shape:
        raise ValueError("Model batch rows do not match the declared columns.")

    with np.errstate(over="ignore", invalid="ignore"):
        converted = values.astype(dtype, copy=False)

    invalid = np.argwhere(~np.isfinite(converted))
    if invalid.size:
        row_index, column_index = invalid[0]
        _validate_model_value(
            rows[row_index][column_index],
            columns[column_index],
            keys[row_index],
        )
        raise ValueError(
            f"Model column {columns[column_index]!r} at sample "
            f"{keys[row_index]!r} cannot be represented as {dtype}."
        )
    return converted


def _validate_model_rows(
    rows: Sequence[Sequence[Any]],
    keys: Sequence[tuple[Any, ...]],
    columns: Sequence[str],
) -> None:
    for row, key in zip(rows, keys):
        for column, value in zip(columns, row):
            _validate_model_value(value, column, key)


def _validate_model_value(
    value: Any,
    column: str,
    key: tuple[Any, ...],
) -> None:
    if value is None:
        raise ValueError(
            f"Model column {column!r} is missing at sample {key!r}. "
            "Fill or filter missing values before model batching."
        )
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"Model column {column!r} at sample {key!r} must be numeric; "
            f"got {type(value).__name__}."
        )
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise ValueError(
            f"Model column {column!r} at sample {key!r} is outside the "
            "supported floating-point range."
        ) from exc
    if not finite:
        raise ValueError(f"Model column {column!r} at sample {key!r} must be finite.")
