from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any

from jerrythomas.alignment.as_of import as_of_stream
from jerrythomas.alignment.broadcast import broadcast_stream
from jerrythomas.alignment.broadcast_as_of import broadcast_as_of_stream
from jerrythomas.alignment.engine import align_streams
from jerrythomas.config.preview import RecordPreviewStage
from jerrythomas.execution.observability import ignore_execution_event
from jerrythomas.execution.pipeline import Input, Pipeline, Stage
from jerrythomas.execution.runner import run_pipeline
from jerrythomas.pipelines.stream.cross_section import build_cross_section_stages
from jerrythomas.pipelines.stream.order import build_record_order_stage
from jerrythomas.pipelines.stream.stages import (
    build_preprocess_stages,
    build_transform_stages,
)
from jerrythomas.runtime import (
    AlignedRuntimeStream,
    AsOfRuntimeStream,
    BroadcastAsOfRuntimeStream,
    BroadcastRuntimeStream,
    CombinedRuntimeStream,
    CrossSectionRuntimeStream,
    DerivedRuntimeStream,
    RecordStage,
    Runtime,
    SourceRuntimeStream,
    require_runtime_stream,
)
from jerrythomas.sources.observability import source_progress, source_summary
from jerrythomas.transforms.utils import set_record_domain_anchor


def run_stream_pipeline(
    runtime: Runtime,
    stream_id: str,
) -> Iterator[Any]:
    return run_pipeline(runtime, build_stream_pipeline(runtime, stream_id))


def run_stream_preview_pipeline(
    runtime: Runtime,
    stream_id: str,
    preview: RecordPreviewStage,
) -> Iterator[Any]:
    pipeline = build_stream_pipeline(runtime, stream_id)
    stream = require_runtime_stream(runtime, stream_id)
    if preview in {"input", "canonical"} and isinstance(
        stream,
        CrossSectionRuntimeStream,
    ):
        upstream = build_stream_pipeline(runtime, stream.input_stream)
        pipeline = (
            pipeline.through_stage_count(len(upstream.stages))
            if preview == "input"
            else pipeline.through_stage_named("ensure_record_order")
        )
    elif preview in {"input", "canonical"} and isinstance(
        stream,
        DerivedRuntimeStream,
    ):
        upstream = build_stream_pipeline(runtime, stream.input_stream)
        pipeline = pipeline.through_stage_count(len(upstream.stages))
    elif preview == "input":
        pipeline = pipeline.input_only()
    elif preview == "canonical":
        stage_name = (
            "combine_records"
            if isinstance(stream, CombinedRuntimeStream)
            else "map_records"
        )
        pipeline = pipeline.through_stage_named(stage_name)
    return run_pipeline(runtime, pipeline)


def build_stream_pipeline(
    runtime: Runtime,
    stream_id: str,
) -> Pipeline:
    stream = require_runtime_stream(runtime, stream_id)
    if isinstance(stream, SourceRuntimeStream):
        return Pipeline(
            name=f"stream:{stream_id}",
            input=Input(
                name="open_source",
                open=stream.source.stream,
                progress=source_progress(stream.source),
            ),
            stages=_source_stages(runtime, stream),
            summary=source_summary(stream.source),
        )
    if isinstance(stream, DerivedRuntimeStream):
        upstream = build_stream_pipeline(runtime, stream.input_stream)
        return upstream.continue_as(
            f"stream:{stream_id}",
            build_transform_stages(
                runtime,
                stream.transforms,
                stream.partition_by,
            ),
        )
    if isinstance(stream, CrossSectionRuntimeStream):
        upstream = build_stream_pipeline(runtime, stream.input_stream)
        return upstream.continue_as(
            f"stream:{stream_id}",
            (
                *build_cross_section_stages(
                    stream.cross_section,
                    stream.partition_by,
                    runtime.execution.sort_buffer_bytes,
                ),
                *build_transform_stages(
                    runtime,
                    stream.transforms,
                    stream.partition_by,
                ),
            ),
        )
    if isinstance(stream, BroadcastRuntimeStream):
        return Pipeline(
            name=f"stream:{stream_id}",
            input=Input(
                name="broadcast_inputs",
                open=partial(
                    _broadcast_inputs,
                    runtime,
                    stream.input_stream,
                    stream.broadcast_stream,
                    stream.partition_by,
                ),
            ),
            stages=_combined_stages(runtime, stream),
            summary=(
                f"primary={stream.input_stream},broadcast={stream.broadcast_stream}"
            ),
        )
    if isinstance(stream, AsOfRuntimeStream):
        return Pipeline(
            name=f"stream:{stream_id}",
            input=Input(
                name="as_of_inputs",
                open=partial(
                    _as_of_inputs,
                    runtime,
                    stream.input_stream,
                    stream.lookup_stream,
                    stream.partition_by,
                    stream.max_age,
                    stream.require_match,
                    stream.direction,
                ),
            ),
            stages=_combined_stages(runtime, stream),
            summary=f"primary={stream.input_stream},as_of={stream.lookup_stream}",
        )
    if isinstance(stream, BroadcastAsOfRuntimeStream):
        return Pipeline(
            name=f"stream:{stream_id}",
            input=Input(
                name="broadcast_as_of_inputs",
                open=partial(
                    _broadcast_as_of_inputs,
                    runtime,
                    stream.input_stream,
                    stream.lookup_stream,
                    stream.partition_by,
                    stream.max_age,
                    stream.require_match,
                    stream.direction,
                ),
            ),
            stages=_combined_stages(runtime, stream),
            summary=(
                f"primary={stream.input_stream},broadcast_as_of={stream.lookup_stream}"
            ),
        )
    if isinstance(stream, AlignedRuntimeStream):
        return Pipeline(
            name=f"stream:{stream_id}",
            input=Input(
                name="align_inputs",
                open=partial(
                    _align_inputs,
                    runtime,
                    stream.inputs,
                    stream.partition_by,
                ),
            ),
            stages=_combined_stages(runtime, stream),
            summary="inputs=" + ",".join(stream.inputs),
        )
    raise TypeError(f"Unsupported runtime stream: {type(stream).__name__}")


def _source_stages(
    runtime: Runtime,
    stream: SourceRuntimeStream,
) -> tuple[Stage, ...]:
    return (
        Stage(
            name="map_records",
            apply=partial(_map_records, stream.mapper),
        ),
        *build_preprocess_stages(stream.preprocess),
        build_record_order_stage(
            stream.partition_by,
            stream.presorted,
            runtime.execution.sort_buffer_bytes,
        ),
        *build_transform_stages(
            runtime,
            stream.transforms,
            stream.partition_by,
        ),
    )


def _combined_stages(
    runtime: Runtime,
    stream: CombinedRuntimeStream,
) -> tuple[Stage, ...]:
    return (
        Stage(name="combine_records", apply=stream.combine),
        *build_transform_stages(
            runtime,
            stream.transforms,
            stream.partition_by,
        ),
    )


def _map_records(mapper: RecordStage, records: Iterator[Any]) -> Iterator[Any]:
    mapped: Iterable[Any] = ()
    iterator: Iterator[Any] = iter(())
    processing_failed = False
    try:
        mapped = mapper(records)
        if mapped is None:
            raise TypeError("Mapper returned None; return an iterable.")
        iterator = iter(mapped)
        for position, record in enumerate(iterator, start=1):
            record_time = getattr(record, "time", None)
            if not isinstance(record_time, datetime):
                raise TypeError(
                    f"Mapped record {position} time must be a datetime; "
                    f"got {type(record_time).__name__}."
                )
            if record_time.tzinfo is not timezone.utc:
                if record_time.tzinfo is None or record_time.utcoffset() is None:
                    raise ValueError(
                        f"Mapped record {position} time must be timezone-aware."
                    )
                record.time = record_time.astimezone(timezone.utc)
            set_record_domain_anchor(record, True)
            yield record
    except GeneratorExit:
        raise
    except BaseException:
        processing_failed = True
        raise
    finally:
        try:
            if iterator is not records:
                closer = getattr(iterator, "close", None)
                if callable(closer):
                    closer()
            if iterator is not mapped and mapped is not records:
                closer = getattr(mapped, "close", None)
                if callable(closer):
                    closer()
        except BaseException:
            if not processing_failed:
                raise


def _align_inputs(
    runtime: Runtime,
    input_streams: tuple[str, ...],
    partition_by: tuple[str, ...],
) -> Iterator[tuple[Any, ...]]:
    inputs = [
        (
            stream_id,
            _run_internal_stream(runtime, stream_id),
        )
        for stream_id in input_streams
    ]
    return align_streams(inputs, partition_by=partition_by)


def _broadcast_inputs(
    runtime: Runtime,
    input_stream: str,
    broadcast_input: str,
    partition_by: tuple[str, ...],
) -> Iterator[tuple[Any, Any]]:
    primary = _run_internal_stream(runtime, input_stream)
    broadcast = _run_internal_stream(runtime, broadcast_input)
    return broadcast_stream(primary, broadcast, partition_by)


def _as_of_inputs(
    runtime: Runtime,
    input_stream: str,
    lookup_stream: str,
    partition_by: tuple[str, ...],
    max_age: timedelta | None,
    require_match: bool,
    direction: str,
) -> Iterator[tuple[Any, Any | None]]:
    primary = _run_internal_stream(runtime, input_stream)
    lookup = _run_internal_stream(runtime, lookup_stream)
    return as_of_stream(
        primary,
        lookup,
        partition_by,
        max_age=max_age,
        require_match=require_match,
        direction=direction,
    )


def _broadcast_as_of_inputs(
    runtime: Runtime,
    input_stream: str,
    lookup_stream: str,
    partition_by: tuple[str, ...],
    max_age: timedelta | None,
    require_match: bool,
    direction: str,
) -> Iterator[tuple[Any, Any | None]]:
    primary = _run_internal_stream(runtime, input_stream)
    lookup = _run_internal_stream(runtime, lookup_stream)
    return broadcast_as_of_stream(
        primary,
        lookup,
        partition_by,
        max_age=max_age,
        require_match=require_match,
        direction=direction,
    )


def _run_internal_stream(
    runtime: Runtime,
    stream_id: str,
) -> Iterator[Any]:
    return run_pipeline(
        runtime,
        build_stream_pipeline(runtime, stream_id),
        observer=ignore_execution_event,
    )
