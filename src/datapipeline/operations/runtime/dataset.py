import logging
import time
from collections.abc import Iterator, Sequence
from itertools import islice
from typing import TypeVar

from datapipeline.artifacts.models import (
    UnsplitMetadataLayout,
    VectorMetadataEntry,
    VectorSchema,
)
from datapipeline.artifacts.registry import VECTOR_METADATA_SPEC
from datapipeline.artifacts.series import load_series_manifest
from datapipeline.artifacts.specs import SERIES, dataset_requires_scaler
from datapipeline.config.dataset.series import SeriesConfig
from datapipeline.config.preview import PreviewStage
from datapipeline.domain.sample import Sample
from datapipeline.io.dataset_table import DatasetTable
from datapipeline.io.output import OutputTarget, output_destination_key
from datapipeline.operations.persistence import (
    DatasetTableOutput,
    RoutedDatasetTableOutput,
    RoutedRuntimeOutput,
    RuntimeOutput,
    RuntimeOutputBatch,
    RuntimeOutputItem,
)
from datapipeline.pipelines.dataset.pipeline import (
    resolve_fold_output_plans,
    run_dataset_pipeline,
    run_fold_outputs_pipeline,
    run_sample_pipeline,
    run_scaled_dataset_pipeline,
)
from datapipeline.pipelines.series.pipeline import run_series_pipeline
from datapipeline.pipelines.sample.keys import require_metadata_key_plan
from datapipeline.pipelines.stream.pipeline import run_stream_preview_pipeline
from datapipeline.runtime import Runtime

logger = logging.getLogger(__name__)
T = TypeVar("T")

_RECORD_PREVIEWS = {"input", "canonical", "records"}


def limit_items(items: Iterator[T], limit: int | None) -> Iterator[T]:
    if limit is None:
        yield from items
    else:
        yield from islice(items, limit)


def throttle_items(
    items: Iterator[T],
    throttle_ms: float | None,
) -> Iterator[T]:
    if not throttle_ms or throttle_ms <= 0:
        yield from items
        return
    delay = throttle_ms / 1000.0
    for item in items:
        yield item
        time.sleep(delay)


def _close_iterator(items: Iterator[object]) -> None:
    closer = getattr(items, "close", None)
    if callable(closer):
        closer()


def _managed_items(stream: Iterator[T]) -> Iterator[T]:
    try:
        yield from stream
    finally:
        _close_iterator(stream)


def _runtime_output(
    stream: Iterator[object],
    target: OutputTarget,
    limit: int | None,
) -> RuntimeOutput:
    return RuntimeOutput(
        rows=limit_items(_managed_items(stream), limit),
        target=target,
    )


def _sample_output(
    stream: Iterator[Sample],
    target: OutputTarget,
    limit: int | None,
    throttle_ms: float | None,
) -> RuntimeOutput:
    return _runtime_output(throttle_items(stream, throttle_ms), target, limit)


def _parquet_sample_output(
    stream: Iterator[Sample],
    target: OutputTarget,
    limit: int | None,
    throttle_ms: float | None,
    table: DatasetTable,
) -> DatasetTableOutput:
    return DatasetTableOutput(
        rows=limit_items(
            _managed_items(throttle_items(stream, throttle_ms)),
            limit,
        ),
        table=table,
        target=target,
    )


def _preview_plan(
    preview_cfgs: Sequence[SeriesConfig],
    preview: PreviewStage,
) -> list[tuple[str, SeriesConfig]]:
    if preview not in _RECORD_PREVIEWS:
        return [(cfg.id, cfg) for cfg in preview_cfgs]

    seen: set[str] = set()
    plan: list[tuple[str, SeriesConfig]] = []
    for cfg in preview_cfgs:
        stream_id = cfg.stream
        if stream_id in seen:
            continue
        seen.add(stream_id)
        plan.append((stream_id, cfg))
    return plan


def _serve_preview(
    runtime: Runtime,
    limit: int | None,
    target: OutputTarget,
    throttle_ms: float | None,
    preview: PreviewStage,
) -> RuntimeOutputItem | RuntimeOutputBatch:
    dataset = runtime.dataset
    if target.format == "parquet" and preview not in {"samples", "postprocess"}:
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
        if target.format == "parquet":
            table = _dataset_table(
                runtime,
                metadata.catalog.features,
                metadata.catalog.targets,
            )
            return _parquet_sample_output(
                sample_stream,
                target,
                limit,
                throttle_ms,
                table,
            )
        return _sample_output(sample_stream, target, limit, throttle_ms)

    outputs: list[RuntimeOutput] = []
    preview_plan = _preview_plan(dataset.series, preview)
    resolved_outputs: list[tuple[str, SeriesConfig, OutputTarget]] = []
    destinations: dict[str, str] = {}
    for output_id, cfg in preview_plan:
        output_target = target.for_output(output_id)
        destination = output_target.destination
        if destination is not None:
            collision_key = output_destination_key(destination)
            if collision_key in destinations:
                first_id = destinations[collision_key]
                raise ValueError(
                    f"Preview outputs {first_id!r} and {output_id!r} resolve to "
                    f"the same destination: {destination}"
                )
            destinations[collision_key] = output_id
        resolved_outputs.append((output_id, cfg, output_target))

    for output_id, cfg, output_target in resolved_outputs:
        stream: Iterator[object]
        match preview:
            case "input" | "canonical" | "records":
                stream = run_stream_preview_pipeline(
                    runtime,
                    output_id,
                    preview,
                )
            case "series":
                stream = run_series_pipeline(
                    runtime,
                    cfg,
                    sample_keys=dataset.sample.keys,
                    group_by_cadence=dataset.sample.cadence,
                )
            case _:
                raise ValueError(f"Unsupported preview stage: {preview!r}")
        outputs.append(_runtime_output(stream, output_target, limit))
    return RuntimeOutputBatch(outputs=tuple(outputs))


def _serve_dataset(
    runtime: Runtime,
    limit: int | None,
    target: OutputTarget,
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
    if target.format == "parquet":
        return _parquet_sample_output(
            samples,
            target,
            limit,
            throttle_ms,
            _served_dataset_table(
                runtime,
                metadata.catalog,
            ),
        )
    return _sample_output(samples, target, limit, throttle_ms)


def _serve_fold_outputs(
    runtime: Runtime,
    output_ids: tuple[str, ...],
    limit: int | None,
    target: OutputTarget,
    throttle_ms: float | None,
) -> RoutedRuntimeOutput | RoutedDatasetTableOutput:
    dataset = runtime.dataset
    if target.transport != "fs":
        raise ValueError("Fold outputs require fs output.")
    split_cfg = dataset.split
    if split_cfg is None:
        raise ValueError("Fold outputs require dataset split configuration.")

    plans = resolve_fold_output_plans(runtime, output_ids)
    samples = run_fold_outputs_pipeline(
        runtime,
        plans,
    )
    rows = throttle_items(_managed_items(samples), throttle_ms)
    output_targets = {
        output_id: target.for_output(output_id) for output_id in output_ids
    }
    tables = (
        {
            output_id: _served_dataset_table(
                runtime,
                plan.schema,
            )
            for plan in plans
            for output_id in plan.outputs
        }
        if target.format == "parquet"
        else None
    )
    output = (
        RoutedDatasetTableOutput(
            rows=rows,
            tables=tables,
            targets=output_targets,
            limit_per_output=limit,
        )
        if tables is not None
        else RoutedRuntimeOutput(
            rows=rows,
            targets=output_targets,
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
    target: OutputTarget,
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
            target,
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
            target,
            throttle_ms,
        )

    return _serve_dataset(
        runtime,
        limit,
        target,
        throttle_ms,
    )
