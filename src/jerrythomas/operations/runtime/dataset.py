import logging
import time
from collections.abc import Iterator, Sequence
from itertools import islice
from typing import TypeVar

from jerrythomas.artifacts.models import (
    UnsplitMetadataLayout,
    VectorMetadataEntry,
    VectorSchema,
)
from jerrythomas.artifacts.registry import VECTOR_METADATA_SPEC
from jerrythomas.artifacts.series import load_series_manifest
from jerrythomas.artifacts.specs import SERIES, dataset_requires_scaler
from jerrythomas.config.preview import PreviewStage
from jerrythomas.config.profiles.output import Format
from jerrythomas.domain.sample import Sample
from jerrythomas.io.dataset_table import DatasetTable
from jerrythomas.operations.persistence import (
    DatasetTableOutput,
    RoutedDatasetTableOutput,
    RoutedRuntimeOutput,
    RuntimeOutput,
    RuntimeOutputBatch,
    RuntimeOutputItem,
)
from jerrythomas.pipelines.dataset.pipeline import (
    resolve_fold_output_plans,
    run_dataset_pipeline,
    run_fold_outputs_pipeline,
    run_sample_pipeline,
    run_scaled_dataset_pipeline,
)
from jerrythomas.pipelines.dataset.preview import preview_output_plan
from jerrythomas.pipelines.series.pipeline import run_series_pipeline
from jerrythomas.pipelines.sample.keys import require_metadata_key_plan
from jerrythomas.pipelines.stream.pipeline import run_stream_preview_pipeline
from jerrythomas.runtime import Runtime

logger = logging.getLogger(__name__)
T = TypeVar("T")


def limit_items(items: Iterator[T], limit: int | None) -> Iterator[T]:
    selected = items if limit is None else islice(items, limit)
    try:
        for item in selected:
            yield item
    finally:
        _close_iterator(items)


def throttle_items(
    items: Iterator[T],
    throttle_ms: float | None,
) -> Iterator[T]:
    try:
        if not throttle_ms or throttle_ms <= 0:
            for item in items:
                yield item
            return
        delay = throttle_ms / 1000.0
        for item in items:
            yield item
            time.sleep(delay)
    finally:
        _close_iterator(items)


def _close_iterator(items: Iterator[object]) -> None:
    closer = getattr(items, "close", None)
    if callable(closer):
        closer()


def _runtime_output(
    stream: Iterator[object],
    limit: int | None,
) -> RuntimeOutput:
    return RuntimeOutput(rows=limit_items(stream, limit))


def _sample_output(
    stream: Iterator[Sample],
    limit: int | None,
    throttle_ms: float | None,
) -> RuntimeOutput:
    return _runtime_output(throttle_items(stream, throttle_ms), limit)


def _parquet_sample_output(
    stream: Iterator[Sample],
    limit: int | None,
    throttle_ms: float | None,
    table: DatasetTable,
) -> DatasetTableOutput:
    return DatasetTableOutput(
        rows=limit_items(throttle_items(stream, throttle_ms), limit),
        table=table,
    )


def _serve_preview(
    runtime: Runtime,
    limit: int | None,
    output_format: Format,
    throttle_ms: float | None,
    preview: PreviewStage,
) -> RuntimeOutputItem | RuntimeOutputBatch:
    dataset = runtime.dataset
    if output_format == "parquet" and preview not in {"samples", "postprocess"}:
        raise ValueError(
            "Parquet preview supports only the 'samples' and 'postprocess' stages."
        )
    if preview in {"samples", "postprocess"}:
        metadata = runtime.artifacts.load(VECTOR_METADATA_SPEC)
        key_plan = require_metadata_key_plan(
            metadata.catalog.window,
            metadata.catalog.sample,
            dataset.sample.cadence,
            dataset.sample.keys,
        )
        if preview == "samples":
            sample_stream = run_sample_pipeline(
                runtime,
                metadata.catalog,
                key_plan,
            )
        else:
            sample_stream = run_dataset_pipeline(
                runtime,
                metadata.catalog,
                key_plan,
            )
        if output_format == "parquet":
            table = _dataset_table(
                runtime,
                metadata.catalog.features,
                metadata.catalog.targets,
            )
            return _parquet_sample_output(
                sample_stream,
                limit,
                throttle_ms,
                table,
            )
        return _sample_output(sample_stream, limit, throttle_ms)

    outputs: dict[str, RuntimeOutput] = {}
    for output_id, cfg in preview_output_plan(dataset.series, preview):
        stream: Iterator[object]
        match preview:
            case "input" | "canonical" | "records":
                stream = run_stream_preview_pipeline(
                    runtime,
                    output_id,
                    preview,
                )
            case "series":
                stream = run_series_pipeline(runtime, cfg)
            case _:
                raise ValueError(f"Unsupported preview stage: {preview!r}")
        outputs[output_id] = _runtime_output(stream, limit)
    return RuntimeOutputBatch(outputs=outputs)


def _serve_dataset(
    runtime: Runtime,
    limit: int | None,
    output_format: Format,
    throttle_ms: float | None,
) -> RuntimeOutput | DatasetTableOutput:
    dataset = runtime.dataset
    metadata = runtime.artifacts.load(VECTOR_METADATA_SPEC)
    if not isinstance(metadata.layout, UnsplitMetadataLayout):
        raise RuntimeError(
            "Unsplit dataset requires unsplit metadata. Rebuild build/metadata.json."
        )
    key_plan = require_metadata_key_plan(
        metadata.catalog.window,
        metadata.catalog.sample,
        dataset.sample.cadence,
        dataset.sample.keys,
    )
    run = (
        run_scaled_dataset_pipeline
        if dataset_requires_scaler(dataset)
        else run_dataset_pipeline
    )
    samples = run(
        runtime,
        metadata.catalog,
        key_plan,
    )
    if output_format == "parquet":
        return _parquet_sample_output(
            samples,
            limit,
            throttle_ms,
            _served_dataset_table(
                runtime,
                metadata.catalog,
            ),
        )
    return _sample_output(samples, limit, throttle_ms)


def _serve_fold_outputs(
    runtime: Runtime,
    output_ids: tuple[str, ...],
    limit: int | None,
    output_format: Format,
    throttle_ms: float | None,
) -> RoutedRuntimeOutput | RoutedDatasetTableOutput:
    dataset = runtime.dataset
    split_cfg = dataset.split
    if split_cfg is None:
        raise ValueError("Fold outputs require dataset split configuration.")

    plans = resolve_fold_output_plans(runtime, output_ids)
    samples = run_fold_outputs_pipeline(
        runtime,
        plans,
    )
    rows = throttle_items(samples, throttle_ms)
    tables = (
        {
            output_id: _served_dataset_table(
                runtime,
                plan.schema,
            )
            for plan in plans
            for output_id in plan.outputs
        }
        if output_format == "parquet"
        else None
    )
    output = (
        RoutedDatasetTableOutput(
            rows=rows,
            tables=tables,
            limit_per_output=limit,
        )
        if tables is not None
        else RoutedRuntimeOutput(
            rows=rows,
            limit_per_output=limit,
        )
    )
    return output


def _served_dataset_table(
    runtime: Runtime,
    schema: VectorSchema,
) -> DatasetTable:
    dataset = runtime.dataset
    return _dataset_table(
        runtime,
        schema.features,
        schema.targets,
        scaled_feature_ids=tuple(cfg.id for cfg in dataset.features if cfg.scale),
        scaled_target_ids=tuple(cfg.id for cfg in dataset.targets if cfg.scale),
    )


def _dataset_table(
    runtime: Runtime,
    feature_entries: Sequence[VectorMetadataEntry],
    target_entries: Sequence[VectorMetadataEntry],
    scaled_feature_ids: tuple[str, ...] = (),
    scaled_target_ids: tuple[str, ...] = (),
) -> DatasetTable:
    sample_keys = runtime.dataset.sample.keys
    manifest = load_series_manifest(runtime.artifacts.resolve_path(SERIES))
    if manifest.sample_keys != tuple(sample_keys):
        raise RuntimeError(
            "Series sample keys do not match the dataset table contract."
        )
    return DatasetTable(
        sample_keys,
        manifest.sample_key_types,
        feature_entries,
        target_entries,
        scaled_feature_ids,
        scaled_target_ids,
    )


def run_dataset_operation(
    runtime: Runtime,
    output_ids: tuple[str, ...],
    limit: int | None,
    output_format: Format,
    throttle_ms: float | None,
    preview: PreviewStage | None,
) -> RuntimeOutputItem | RuntimeOutputBatch | None:
    dataset = runtime.dataset

    if not dataset.series:
        logger.warning("(no features configured; nothing to serve)")
        return None

    if preview is not None:
        return _serve_preview(
            runtime,
            limit,
            output_format,
            throttle_ms,
            preview,
        )

    if dataset.split is not None:
        if not output_ids:
            raise ValueError("A split dataset requires at least one fold output.")
        return _serve_fold_outputs(
            runtime,
            output_ids,
            limit,
            output_format,
            throttle_ms,
        )

    return _serve_dataset(
        runtime,
        limit,
        output_format,
        throttle_ms,
    )
