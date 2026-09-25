import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeAlias

from jerrythomas.domain.sample import Sample
from jerrythomas.execution.observability import (
    OperationProgressTracker,
    emit_file_result,
    emit_rows_written,
)
from jerrythomas.execution.settings import resolve_heartbeat_interval_seconds
from jerrythomas.io.factory import writer_factory
from jerrythomas.io.factory import dataset_writer_factory
from jerrythomas.io.dataset_table import DatasetTable
from jerrythomas.io.normalization import json_text, raw_payload
from jerrythomas.io.output import OutputTarget, validate_output_destinations
from jerrythomas.io.protocols import Writer
from jerrythomas.io.sinks.files import AtomicTextFileSink
from jerrythomas.io.writers.parquet import DEFAULT_ROW_GROUP_ROWS


@dataclass(frozen=True)
class WrittenOutput:
    """A file committed by a writer, with its actual record count."""

    path: Path
    output_id: str | None
    row_count: int | None


@dataclass(frozen=True)
class RuntimeOutput:
    rows: Iterable[Any] | None = None
    payload: Mapping[str, Any] | None = None
    render_html: Callable[[], str] | None = None

    def __post_init__(self) -> None:
        if self.rows is None and self.payload is None and self.render_html is None:
            raise ValueError("RuntimeOutput requires rows, payload, or render_html.")
        if self.rows is not None and self.payload is not None:
            raise ValueError("RuntimeOutput cannot define both rows and payload.")


@dataclass(frozen=True)
class RoutedRuntimeOutput:
    rows: Iterable[tuple[str, Any]]
    limit_per_output: int | None = None


@dataclass(frozen=True)
class DatasetTableOutput:
    rows: Iterable[Sample]
    table: DatasetTable


@dataclass(frozen=True)
class RoutedDatasetTableOutput:
    rows: Iterable[tuple[str, Sample]]
    tables: Mapping[str, DatasetTable]
    limit_per_output: int | None = None


RoutedOutput: TypeAlias = RoutedRuntimeOutput | RoutedDatasetTableOutput

RuntimeOutputItem = RuntimeOutput | RoutedOutput | DatasetTableOutput


@dataclass(frozen=True)
class RuntimeOutputBatch:
    outputs: Mapping[str, RuntimeOutput]


def _close_runtime_rows(rows: Iterable[Any]) -> None:
    close = getattr(rows, "close", None)
    if callable(close):
        close()


@contextmanager
def _runtime_rows(
    rows: Iterable[Any],
    logger: logging.Logger,
) -> Iterator[Iterator[Any]]:
    iterator: Iterator[Any] | None = None
    failed = False
    try:
        iterator = iter(rows)
        yield iterator
    except BaseException:
        failed = True
        raise
    finally:
        try:
            try:
                if iterator is not None:
                    _close_runtime_rows(iterator)
            finally:
                if iterator is not rows:
                    _close_runtime_rows(rows)
        except BaseException:
            if not failed:
                raise
            logger.debug("Failed to close runtime output rows", exc_info=True)


def _payload_rows(result: RuntimeOutput, target: OutputTarget) -> Iterator[Any]:
    payload = result.payload
    if payload is None:
        raise ValueError("Runtime output does not support a non-HTML representation.")
    if target.format == "txt":
        return iter((json_text(raw_payload(payload), indent=2),))
    return iter((dict(payload),))


def _write_html_output(
    result: RuntimeOutput,
    target: OutputTarget,
    logger: logging.Logger,
    overwrite: bool,
) -> None:
    if result.render_html is None:
        raise ValueError("html output is not supported for this operation.")
    destination = target.destination
    if destination is None:
        raise ValueError("html output requires fs destination.")

    sink = AtomicTextFileSink(
        destination,
        encoding=target.encoding or "utf-8",
        overwrite=overwrite,
    )
    try:
        sink.write_text(result.render_html())
        sink.close()
    except BaseException:
        try:
            sink.abort()
        except BaseException:
            logger.debug("Failed to abort html output writer", exc_info=True)
        raise
    emit_file_result("Output", destination)


def persist_runtime_output(
    result: RuntimeOutput,
    target: OutputTarget,
    heartbeat_interval_seconds: float | None,
    logger: logging.Logger,
    overwrite: bool = True,
) -> int | None:
    """Write one runtime output, closing its rows before committing the file."""
    row_count = 0
    supplied_rows = result.rows
    owned_rows = supplied_rows if supplied_rows is not None else ()
    writer = None
    try:
        with _runtime_rows(owned_rows, logger) as rows:
            if target.format != "html":
                if supplied_rows is None:
                    rows = _payload_rows(result, target)
                writer = writer_factory(target, overwrite=overwrite)
                progress = OperationProgressTracker(
                    "write_output",
                    "rows",
                    resolve_heartbeat_interval_seconds(heartbeat_interval_seconds),
                )
                for row in rows:
                    writer.write(row)
                    row_count += 1
                    progress.advance()

        if target.format == "html":
            _write_html_output(result, target, logger, overwrite)
            return None

        assert writer is not None
        writer.close()
    except BaseException:
        if writer is not None:
            try:
                writer.abort()
            except BaseException:
                logger.debug("Failed to abort runtime output writer", exc_info=True)
        raise

    if target.destination is not None:
        emit_file_result("Output", target.destination)
    elif target.transport == "stdout":
        logger.info("Output: stdout")
    return row_count


def _persist_dataset_table_output(
    result: DatasetTableOutput,
    target: OutputTarget,
    heartbeat_interval_seconds: float | None,
    logger: logging.Logger,
) -> int:
    row_count = 0
    writer = None
    try:
        with _runtime_rows(result.rows, logger) as rows:
            writer = dataset_writer_factory(target, result.table)
            progress = OperationProgressTracker(
                "write_output",
                "rows",
                resolve_heartbeat_interval_seconds(heartbeat_interval_seconds),
            )
            for row in rows:
                writer.write(row)
                row_count += 1
                progress.advance()
        writer.close()
    except BaseException:
        if writer is not None:
            try:
                writer.abort()
            except BaseException:
                logger.debug("Failed to abort dataset table writer", exc_info=True)
        raise

    if target.destination is not None:
        emit_file_result("Output", target.destination)
    return row_count


def _planned_routed_targets(
    result: RoutedOutput,
    target: OutputTarget,
    output_ids: Sequence[str],
) -> list[tuple[str, OutputTarget, Path]]:
    if not output_ids:
        raise ValueError("Routed runtime output requires planned output IDs.")
    if target.transport != "fs" or target.destination is None:
        raise ValueError("Routed runtime output requires fs destination.")

    if isinstance(result, RoutedDatasetTableOutput):
        missing_table_ids = sorted(set(output_ids) - set(result.tables))
        if missing_table_ids:
            raise ValueError(
                "Routed dataset table output is missing tables for output IDs: "
                f"{missing_table_ids}."
            )
        unmatched_table_ids = sorted(set(result.tables) - set(output_ids))
        if unmatched_table_ids:
            raise ValueError(
                "Routed dataset table output has tables without planned output IDs: "
                f"{unmatched_table_ids}."
            )

    planned: list[tuple[str, OutputTarget, Path]] = []
    for output_id in output_ids:
        output_target = target.for_output(output_id)
        assert output_target.destination is not None
        planned.append((output_id, output_target, output_target.destination))
    return planned


def _write_routed_rows(
    rows: Iterator[tuple[str, Any]],
    writers: Mapping[str, Writer],
    progress: OperationProgressTracker,
    limit_per_output: int | None,
) -> dict[str, int]:
    counts = dict.fromkeys(writers, 0)
    for output_id, row in rows:
        writer = writers.get(output_id)
        if writer is None:
            raise ValueError(f"Routed row has unknown output ID {output_id!r}.")
        if limit_per_output is not None and counts[output_id] >= limit_per_output:
            if all(count >= limit_per_output for count in counts.values()):
                break
            continue
        writer.write(row)
        counts[output_id] += 1
        progress.advance()
        if limit_per_output is not None and all(
            count >= limit_per_output for count in counts.values()
        ):
            break
    return counts


def _persist_routed_output(
    result: RoutedOutput,
    target: OutputTarget,
    output_ids: Sequence[str],
    heartbeat_interval_seconds: float | None,
    logger: logging.Logger,
) -> tuple[WrittenOutput, ...]:
    writers: dict[str, Writer] = {}
    try:
        with _runtime_rows(result.rows, logger) as rows:
            planned = _planned_routed_targets(result, target, output_ids)
            validate_output_destinations([item[1] for item in planned])
            parquet_row_group_rows = max(
                1,
                DEFAULT_ROW_GROUP_ROWS // len(planned),
            )
            for output_id, target, _destination in planned:
                writers[output_id] = (
                    dataset_writer_factory(
                        target,
                        result.tables[output_id],
                        row_group_rows=parquet_row_group_rows,
                    )
                    if isinstance(result, RoutedDatasetTableOutput)
                    else writer_factory(target)
                )

            progress = OperationProgressTracker(
                "write_output",
                "rows",
                resolve_heartbeat_interval_seconds(heartbeat_interval_seconds),
            )
            counts = _write_routed_rows(
                rows,
                writers,
                progress,
                result.limit_per_output,
            )

        for writer in writers.values():
            writer.close()
    except BaseException:
        for writer in writers.values():
            try:
                writer.abort()
            except BaseException:
                logger.debug(
                    "Failed to abort routed runtime output writer",
                    exc_info=True,
                )
        raise

    for output_id, _target, destination in planned:
        emit_rows_written(output_id, counts[output_id])
        emit_file_result(output_id, destination)
    return tuple(
        WrittenOutput(destination, output_id, counts[output_id])
        for output_id, _target, destination in planned
    )


def _close_pending_runtime_outputs(
    outputs: Iterable[RuntimeOutputItem],
    logger: logging.Logger,
) -> None:
    for output in outputs:
        if output.rows is None:
            continue
        try:
            _close_runtime_rows(output.rows)
        except BaseException:
            logger.debug(
                "Failed to close pending runtime output rows",
                exc_info=True,
            )


def _persist_runtime_outputs(
    outputs: Sequence[tuple[str, RuntimeOutput, OutputTarget]],
    heartbeat_interval_seconds: float | None,
    logger: logging.Logger,
) -> tuple[WrittenOutput, ...]:
    attempted = 0
    completed: list[WrittenOutput] = []
    try:
        validate_output_destinations(
            [target for _output_id, _output, target in outputs]
        )
        for output_id, output, target in outputs:
            attempted += 1
            row_count = persist_runtime_output(
                output,
                target,
                heartbeat_interval_seconds,
                logger,
            )
            if target.destination is not None:
                completed.append(
                    WrittenOutput(target.destination, output_id, row_count)
                )
    except BaseException:
        _close_pending_runtime_outputs(
            (output for _output_id, output, _target in outputs[attempted:]),
            logger,
        )
        raise
    return tuple(completed)


def persist_runtime_result(
    result: object,
    target: OutputTarget,
    output_ids: Sequence[str] = (),
    heartbeat_interval_seconds: float | None = None,
    *,
    logger: logging.Logger,
) -> tuple[WrittenOutput, ...]:
    """Persist an operation result and return its completed filesystem outputs."""
    row_count: int | None
    if result is None:
        return ()
    if isinstance(result, RuntimeOutputBatch):
        if set(result.outputs) != set(output_ids) or len(result.outputs) != len(
            output_ids
        ):
            _close_pending_runtime_outputs(result.outputs.values(), logger)
            raise ValueError(
                "Runtime output batch IDs do not match the planned output IDs: "
                f"expected {tuple(output_ids)!r}, got {tuple(result.outputs)!r}."
            )
        return _persist_runtime_outputs(
            [
                (output_id, result.outputs[output_id], target.for_output(output_id))
                for output_id in output_ids
            ],
            heartbeat_interval_seconds,
            logger,
        )
    if isinstance(result, RoutedOutput):
        return _persist_routed_output(
            result,
            target,
            output_ids,
            heartbeat_interval_seconds,
            logger,
        )
    if output_ids and isinstance(result, RuntimeOutputItem):
        _close_pending_runtime_outputs((result,), logger)
        raise ValueError("Single runtime output cannot use planned output IDs.")
    if isinstance(result, DatasetTableOutput):
        row_count = _persist_dataset_table_output(
            result,
            target,
            heartbeat_interval_seconds,
            logger,
        )
    elif isinstance(result, RuntimeOutput):
        row_count = persist_runtime_output(
            result,
            target,
            heartbeat_interval_seconds,
            logger,
        )
    else:
        raise TypeError("Output operation returned an unsupported output type.")
    if target.destination is None:
        return ()
    return (WrittenOutput(target.destination, None, row_count),)
