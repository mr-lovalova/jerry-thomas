from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from datapipeline.config.tasks.schedule import ScheduleTask
from datapipeline.domain.value import normalize_data_value
from datapipeline.execution.context import PipelineContext
from datapipeline.execution.observability import OperationProgressTracker
from datapipeline.execution.settings import resolve_heartbeat_interval_seconds
from datapipeline.io.normalization import json_text
from datapipeline.io.sinks.files import AtomicTextFileSink
from datapipeline.operations.persistence import ArtifactOutput
from datapipeline.pipelines.stream.pipeline import run_stream_pipeline
from datapipeline.pipelines.sort import batch_sort
from datapipeline.runtime import Runtime, require_runtime_stream
from datapipeline.transforms.utils import get_field


def _close_iterator(iterator: Any) -> None:
    closer = getattr(iterator, "close", None)
    if callable(closer):
        closer()


def _to_iso(ts: datetime) -> str:
    text = ts.isoformat()
    if text.endswith("+00:00"):
        return text[:-6] + "Z"
    return text


def _schedule_row(record, partition_by: list[str]) -> tuple:
    values = tuple(
        normalize_data_value(get_field(record, field)) for field in partition_by
    )
    for field, value in zip(partition_by, values):
        if value is None:
            raise ValueError(
                f"Schedule stream row is missing partition_by field '{field}'."
            )
    return (record.time, *values)


def _json_schedule_row(row: tuple, partition_by: list[str]) -> dict:
    payload = {"time": _to_iso(row[0])}
    for field, value in zip(partition_by, row[1:]):
        payload[field] = value
    return payload


def _schedule_sort_key(row: tuple) -> tuple:
    return (*row[1:], row[0])


def _unique_schedule_rows(rows) -> Iterator[tuple]:
    previous = None
    previous_key = None
    for position, row in enumerate(rows, start=1):
        current_key = _schedule_sort_key(row)
        if previous_key is not None and not previous_key <= current_key:
            raise ValueError(
                f"Schedule row {position} violates canonical order: "
                f"key {current_key!r} follows {previous_key!r}."
            )
        if row != previous:
            yield row
        previous = row
        previous_key = current_key


def build_schedule_artifact(
    runtime: Runtime,
    task_cfg: ScheduleTask,
) -> ArtifactOutput:
    heartbeat_interval = resolve_heartbeat_interval_seconds(
        runtime.heartbeat_interval_seconds
    )
    context = PipelineContext(runtime)
    runtime_stream = require_runtime_stream(runtime, task_cfg.stream)
    stream = run_stream_pipeline(context, task_cfg.stream)
    project_progress = OperationProgressTracker(
        "project_schedule",
        "records",
        heartbeat_interval,
    )
    schedule_rows = _project_schedule_rows(
        stream,
        task_cfg.partition_by,
        project_progress,
    )
    if tuple(task_cfg.partition_by) == runtime_stream.partition_by:
        ordered_rows = schedule_rows
    else:
        ordered_rows = batch_sort(
            schedule_rows,
            buffer_bytes=runtime.execution.sort_buffer_bytes,
            key=_schedule_sort_key,
        )
    rows = 0
    try:
        relative_path = Path(task_cfg.output)
        destination = (runtime.artifacts_root / relative_path).resolve()
        write_progress = OperationProgressTracker(
            "write_artifact",
            "rows",
            heartbeat_interval,
        )
        sink = AtomicTextFileSink(destination)
        try:
            for row in _unique_schedule_rows(ordered_rows):
                rows += 1
                sink.write_text(
                    json_text(_json_schedule_row(row, task_cfg.partition_by))
                )
                sink.write_text("\n")
                write_progress.advance()
            sink.close()
        except BaseException:
            sink.abort()
            raise
    finally:
        _close_iterator(ordered_rows)
        _close_iterator(stream)

    return ArtifactOutput(
        relative_path=str(relative_path),
        meta={
            "rows": rows,
            "stream": task_cfg.stream,
            "partition_by": list(task_cfg.partition_by),
        },
    )


def _project_schedule_rows(
    stream,
    partition_by: list[str],
    progress: OperationProgressTracker,
) -> Iterator[tuple]:
    for record in stream:
        yield _schedule_row(record, partition_by)
        progress.advance()
