import logging
import time
from collections.abc import Iterator, Sequence
from itertools import islice
from typing import TypeVar

from datapipeline.artifacts.models import VectorMetadataEntry
from datapipeline.artifacts.registry import VECTOR_METADATA_SPEC
from datapipeline.artifacts.series import load_series_manifest
from datapipeline.artifacts.specs import SERIES, dataset_requires_scaler
from datapipeline.config.dataset.series import SeriesConfig
from datapipeline.config.dataset.split import (
    DatasetFold,
    SplitConfig,
    resolve_fold_output,
)
from datapipeline.config.preview import PreviewStage
from datapipeline.domain.sample import Sample
from datapipeline.execution.context import PipelineContext
from datapipeline.execution.runner import run_pipeline
from datapipeline.io.dataset_table import DatasetTable
from datapipeline.io.output import OutputTarget, output_destination_key
from datapipeline.operations.persistence import (
    DatasetTableOutput,
    RoutedDatasetTableOutput,
    RoutedRuntimeOutput,
    RuntimeOutput,
    RuntimeOutputBatch,
)
from datapipeline.pipelines.dataset.postprocess import build_postprocess_plan
from datapipeline.pipelines.dataset.pipeline import (
    FoldOutputPlan,
    build_dataset_pipeline,
    run_dataset_pipeline,
    run_fold_outputs_pipeline,
    run_scaled_dataset_pipeline,
)
from datapipeline.pipelines.series.pipeline import run_series_pipeline
from datapipeline.pipelines.stream.pipeline import build_stream_pipeline
from datapipeline.runtime import (
    CombinedRuntimeStream,
    DerivedRuntimeStream,
    Runtime,
    require_runtime_stream,
)
from datapipeline.utils.window import resolve_window_bounds

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


def _record_preview_stream(
    context: PipelineContext,
    stream_id: str,
    preview: PreviewStage,
) -> Iterator[object]:
    pipeline = build_stream_pipeline(context, stream_id)
    stream = require_runtime_stream(context.runtime, stream_id)
    if preview == "input":
        if isinstance(stream, DerivedRuntimeStream):
            upstream = build_stream_pipeline(context, stream.input_stream)
            return run_pipeline(
                context,
                pipeline.through_stage_count(len(upstream.stages)),
            )
        return run_pipeline(context, pipeline.input_only())
    if preview == "canonical":
        if isinstance(stream, DerivedRuntimeStream):
            upstream = build_stream_pipeline(context, stream.input_stream)
            return run_pipeline(
                context,
                pipeline.through_stage_count(len(upstream.stages)),
            )
        if isinstance(stream, CombinedRuntimeStream):
            node_name = "combine_records"
        else:
            node_name = "map_records"
        return run_pipeline(context, pipeline.through_stage_named(node_name))
    if preview == "records":
        return run_pipeline(context, pipeline)
    raise ValueError(f"Preview stage {preview!r} does not produce records")


def _serve_preview(
    *,
    context: PipelineContext,
    runtime: Runtime,
    feature_cfgs: list[SeriesConfig],
    target_cfgs: Sequence[SeriesConfig],
    cadence: str,
    sample_keys: list[str],
    limit: int | None,
    target: OutputTarget,
    throttle_ms: float | None,
    preview: PreviewStage,
) -> RuntimeOutputBatch:
    if target.format == "parquet" and preview not in {"samples", "postprocess"}:
        raise ValueError(
            "Parquet preview supports only the 'samples' and 'postprocess' stages."
        )
    if preview in {"samples", "postprocess"}:
        runtime.window_bounds = resolve_window_bounds(runtime, True)
        dataset_pipeline = build_dataset_pipeline(
            context,
            feature_cfgs,
            cadence,
            target_configs=target_cfgs,
            rectangular=True,
            sample_keys=sample_keys,
        )
        selected_pipeline = (
            dataset_pipeline.input_only() if preview == "samples" else dataset_pipeline
        )
        sample_stream = run_pipeline(context, selected_pipeline)
        if target.format == "parquet":
            table = (
                _assembled_dataset_table(context, sample_keys)
                if preview == "samples"
                else _postprocessed_dataset_table(context, sample_keys)
            )
            return RuntimeOutputBatch(
                outputs=(
                    _parquet_sample_output(
                        sample_stream,
                        target,
                        limit,
                        throttle_ms,
                        table,
                    ),
                ),
            )
        return RuntimeOutputBatch(
            outputs=(_sample_output(sample_stream, target, limit, throttle_ms),),
        )

    outputs: list[RuntimeOutput] = []
    preview_plan = _preview_plan((*feature_cfgs, *target_cfgs), preview)
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
        if preview in _RECORD_PREVIEWS:
            stream = _record_preview_stream(
                context,
                str(output_id),
                preview,
            )
        elif preview == "series":
            stream = run_series_pipeline(
                context,
                cfg,
                sample_keys=sample_keys,
                group_by_cadence=cadence,
            )
        else:
            raise ValueError(f"Unsupported preview stage: {preview!r}")
        outputs.append(_runtime_output(stream, output_target, limit))
    return RuntimeOutputBatch(outputs=tuple(outputs))


def _serve_dataset(
    *,
    context: PipelineContext,
    runtime: Runtime,
    feature_cfgs: list[SeriesConfig],
    target_cfgs: Sequence[SeriesConfig],
    cadence: str,
    sample_keys: list[str],
    limit: int | None,
    target: OutputTarget,
    throttle_ms: float | None,
) -> RuntimeOutputBatch:
    runtime.window_bounds = resolve_window_bounds(runtime, True)
    run = (
        run_scaled_dataset_pipeline
        if dataset_requires_scaler(runtime.dataset)
        else run_dataset_pipeline
    )
    samples = run(
        context,
        feature_cfgs,
        cadence,
        target_configs=target_cfgs,
        rectangular=True,
        sample_keys=sample_keys,
    )
    if target.format == "parquet":
        return RuntimeOutputBatch(
            outputs=(
                _parquet_sample_output(
                    samples,
                    target,
                    limit,
                    throttle_ms,
                    _served_dataset_table(
                        context,
                        sample_keys,
                        feature_cfgs,
                        target_cfgs,
                    ),
                ),
            ),
        )
    return RuntimeOutputBatch(
        outputs=(_sample_output(samples, target, limit, throttle_ms),),
    )


def _serve_fold_outputs(
    *,
    context: PipelineContext,
    runtime: Runtime,
    feature_cfgs: list[SeriesConfig],
    target_cfgs: Sequence[SeriesConfig],
    cadence: str,
    sample_keys: list[str],
    output_ids: tuple[str, ...],
    limit: int | None,
    target: OutputTarget,
    throttle_ms: float | None,
) -> RuntimeOutputBatch:
    if target.transport != "fs":
        raise ValueError("Fold outputs require fs output.")
    split_cfg = runtime.dataset.split
    if split_cfg is None:
        raise ValueError("Fold outputs require dataset split configuration.")

    runtime.window_bounds = resolve_window_bounds(runtime, True)
    selected_folds = _selected_fold_outputs(split_cfg, output_ids)
    table = (
        _served_dataset_table(
            context,
            sample_keys,
            feature_cfgs,
            target_cfgs,
        )
        if target.format == "parquet"
        else None
    )
    plans = tuple(
        FoldOutputPlan(
            fold=fold,
            output_by_label={
                label: output_id
                for output_id, labels in labels_by_output.items()
                for label in labels
            },
        )
        for fold, labels_by_output in selected_folds
    )
    samples = run_fold_outputs_pipeline(
        context,
        feature_cfgs,
        cadence,
        plans,
        target_configs=target_cfgs,
        rectangular=True,
        sample_keys=sample_keys,
    )
    rows = throttle_items(_managed_items(samples), throttle_ms)
    output_targets = {
        output_id: target.for_output(output_id) for output_id in output_ids
    }
    output = (
        RoutedDatasetTableOutput(
            rows=rows,
            table=table,
            targets=output_targets,
            limit_per_output=limit,
        )
        if table is not None
        else RoutedRuntimeOutput(
            rows=rows,
            targets=output_targets,
            limit_per_output=limit,
        )
    )
    return RuntimeOutputBatch(outputs=(output,))


def _assembled_dataset_table(
    context: PipelineContext,
    sample_keys: list[str],
) -> DatasetTable:
    metadata = context.require_artifact(VECTOR_METADATA_SPEC)
    return _dataset_table(
        context,
        sample_keys,
        metadata.features,
        metadata.targets,
    )


def _postprocessed_dataset_table(
    context: PipelineContext,
    sample_keys: list[str],
) -> DatasetTable:
    plan = build_postprocess_plan(context)
    return _dataset_table(
        context,
        sample_keys,
        plan.feature_entries,
        plan.target_entries,
    )


def _served_dataset_table(
    context: PipelineContext,
    sample_keys: list[str],
    feature_cfgs: list[SeriesConfig],
    target_cfgs: Sequence[SeriesConfig],
) -> DatasetTable:
    plan = build_postprocess_plan(context)
    return _dataset_table(
        context,
        sample_keys,
        plan.feature_entries,
        plan.target_entries,
        scaled_feature_ids=tuple(cfg.id for cfg in feature_cfgs if cfg.scale),
        scaled_target_ids=tuple(cfg.id for cfg in target_cfgs if cfg.scale),
    )


def _dataset_table(
    context: PipelineContext,
    sample_keys: list[str],
    feature_entries: Sequence[VectorMetadataEntry],
    target_entries: Sequence[VectorMetadataEntry],
    scaled_feature_ids: tuple[str, ...] = (),
    scaled_target_ids: tuple[str, ...] = (),
) -> DatasetTable:
    manifest = load_series_manifest(context.resolve_artifact_path(SERIES))
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


def _selected_fold_outputs(
    split: SplitConfig,
    output_ids: tuple[str, ...],
) -> list[tuple[DatasetFold, dict[str, tuple[str, ...]]]]:
    selected: dict[
        str,
        tuple[DatasetFold, dict[str, tuple[str, ...]]],
    ] = {}
    for output_id in output_ids:
        fold, labels = resolve_fold_output(split, output_id)
        entry = selected.get(fold.id)
        if entry is None:
            outputs: dict[str, tuple[str, ...]] = {}
            selected[fold.id] = fold, outputs
        else:
            _, outputs = entry
        outputs[output_id] = labels
    return list(selected.values())


def run_dataset_operation(
    runtime: Runtime,
    limit: int | None,
    target: OutputTarget,
    throttle_ms: float | None,
    preview: PreviewStage | None,
) -> RuntimeOutputBatch | None:
    dataset = runtime.dataset

    context = PipelineContext(runtime)
    feature_cfgs = list(dataset.features)
    target_cfgs = list(dataset.targets)
    if not feature_cfgs and not target_cfgs:
        logger.warning("(no features configured; nothing to serve)")
        return None
    cadence = dataset.sample.cadence

    output_ids = runtime.output_ids
    if preview is not None:
        return _serve_preview(
            context=context,
            runtime=runtime,
            feature_cfgs=feature_cfgs,
            target_cfgs=target_cfgs,
            cadence=cadence,
            sample_keys=dataset.sample.keys,
            limit=limit,
            target=target,
            throttle_ms=throttle_ms,
            preview=preview,
        )

    if dataset.split is not None:
        if not output_ids:
            raise ValueError("A split dataset requires at least one fold output.")
        return _serve_fold_outputs(
            context=context,
            runtime=runtime,
            feature_cfgs=feature_cfgs,
            target_cfgs=target_cfgs,
            cadence=cadence,
            sample_keys=dataset.sample.keys,
            output_ids=output_ids,
            limit=limit,
            target=target,
            throttle_ms=throttle_ms,
        )

    return _serve_dataset(
        context=context,
        runtime=runtime,
        feature_cfgs=feature_cfgs,
        target_cfgs=target_cfgs,
        cadence=cadence,
        sample_keys=dataset.sample.keys,
        limit=limit,
        target=target,
        throttle_ms=throttle_ms,
    )
