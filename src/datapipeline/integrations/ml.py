from __future__ import annotations

import math
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from datapipeline.artifacts.hydration import hydrate_runtime_artifacts_for_pipeline
from datapipeline.artifacts.models import (
    UnsplitMetadataLayout,
    VectorMetadataEntry,
    VectorSchema,
)
from datapipeline.artifacts.registry import VECTOR_METADATA_SPEC
from datapipeline.artifacts.specs import dataset_requires_scaler
from datapipeline.config.dataset.split import (
    resolve_fold_output,
    split_output_ids,
)
from datapipeline.domain.sample import Sample
from datapipeline.domain.vector import Vector
from datapipeline.execution.context import PipelineContext
from datapipeline.pipelines.dataset.pipeline import (
    FoldOutputPlan,
    resolve_fold_output_plans,
    run_dataset_pipeline,
    run_fold_dataset_pipeline,
    run_scaled_dataset_pipeline,
)
from datapipeline.pipelines.dataset.postprocess import build_postprocess_plan
from datapipeline.pipelines.sample.keys import (
    RectangularKeyPlan,
    require_metadata_key_plan,
)
from datapipeline.runtime import Runtime
from datapipeline.services.project_definition import load_project_definition
from datapipeline.services.runtime_compiler import compile_runtime

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
        context = PipelineContext(runtime)
        metadata = context.require_artifact(VECTOR_METADATA_SPEC)
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
        fold_output = resolve_fold_output_plans(context, (output_id,))[0]
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

        context = PipelineContext(self.runtime)
        if self.fold_output is not None:
            samples = run_fold_dataset_pipeline(
                context,
                dataset.features,
                dataset.sample.cadence,
                self.fold_output,
                target_configs=dataset.targets,
                sample_keys=dataset.sample.keys,
            )
        else:
            run = (
                run_scaled_dataset_pipeline
                if dataset_requires_scaler(dataset)
                else run_dataset_pipeline
            )
            samples = run(
                context,
                dataset.features,
                dataset.sample.cadence,
                self.schema,
                self.key_plan,
                target_configs=dataset.targets,
                sample_keys=dataset.sample.keys,
            )

        try:
            if limit is None:
                yield from samples
            else:
                yield from islice(samples, limit)
        finally:
            samples.close()


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
    plan = build_postprocess_plan(
        source.runtime.dataset.postprocess,
        source.schema,
    )
    feature_columns = _columns(plan.feature_entries)
    target_columns = _columns(plan.target_entries)
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
                    plan.feature_entries,
                    plan.target_entries,
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
            plan.feature_entries,
            plan.target_entries,
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
    columns: Sequence[str],
    key: tuple[Any, ...],
) -> list[Real]:
    row: list[Real] = []
    column_index = 0
    for entry in entries:
        value = vector.values[entry.id]
        values = value if entry.kind == "list" else (value,)
        for item in values:
            column = columns[column_index]
            column_index += 1
            if item is None:
                raise ValueError(
                    f"Model column {column!r} is missing at sample {key!r}. "
                    "Fill or filter missing values before model batching."
                )
            if isinstance(item, bool) or not isinstance(item, Real):
                raise TypeError(
                    f"Model column {column!r} at sample {key!r} must be numeric; "
                    f"got {type(item).__name__}."
                )
            try:
                finite = math.isfinite(item)
            except OverflowError as exc:
                raise ValueError(
                    f"Model column {column!r} at sample {key!r} is outside the "
                    "supported floating-point range."
                ) from exc
            if not finite:
                raise ValueError(
                    f"Model column {column!r} at sample {key!r} must be finite."
                )
            row.append(item)
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
    feature_rows = [
        _model_row(sample.features, feature_entries, feature_columns, key)
        for sample, key in zip(samples, keys)
    ]
    target_rows: list[list[Real]] | None = None
    if target_entries:
        target_rows = []
        for sample, key in zip(samples, keys):
            if sample.targets is None:
                raise RuntimeError(
                    f"Sample {key!r} has no targets, but target columns are declared."
                )
            target_rows.append(
                _model_row(sample.targets, target_entries, target_columns, key)
            )

    with np.errstate(over="ignore", invalid="ignore"):
        features = np.asarray(feature_rows, dtype=dtype)
        targets = None if target_rows is None else np.asarray(target_rows, dtype=dtype)

    _require_finite_array(np, features, keys, feature_columns, dtype)
    if targets is not None:
        _require_finite_array(np, targets, keys, target_columns, dtype)

    return ModelBatch(
        keys=keys,
        features=features,
        targets=targets,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )


def _require_finite_array(
    np: Any,
    values: _FloatArray,
    keys: Sequence[tuple[Any, ...]],
    columns: Sequence[str],
    dtype: str,
) -> None:
    invalid = np.argwhere(~np.isfinite(values))
    if invalid.size:
        row_index, column_index = invalid[0]
        raise ValueError(
            f"Model column {columns[column_index]!r} at sample "
            f"{keys[row_index]!r} cannot be represented as {dtype}."
        )
