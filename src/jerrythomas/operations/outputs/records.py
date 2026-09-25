from collections.abc import Iterator
from itertools import islice
from typing import TypeVar

from jerrythomas.config.tasks.stream import StreamTask
from jerrythomas.operations.persistence import RuntimeOutput
from jerrythomas.operations.outputs.execution import OutputOptions
from jerrythomas.pipelines.stream.pipeline import run_stream_pipeline
from jerrythomas.runtime import Runtime

T = TypeVar("T")


def limit_items(items: Iterator[T], limit: int | None) -> Iterator[T]:
    selected = items if limit is None else islice(items, limit)
    try:
        for item in selected:
            yield item
    finally:
        closer = getattr(items, "close", None)
        if callable(closer):
            closer()


def run_records_operation(
    runtime: Runtime,
    task: StreamTask,
    options: OutputOptions,
) -> RuntimeOutput:
    return RuntimeOutput(
        rows=limit_items(run_stream_pipeline(runtime, task.stream), options.limit)
    )
