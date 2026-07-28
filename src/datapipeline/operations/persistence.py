import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeAlias

from datapipeline.artifacts.state import ArtifactFileFingerprint
from datapipeline.config.tasks.base import ArtifactTask
from datapipeline.domain.sample import Sample
from datapipeline.execution.observability import (
    OperationProgressTracker,
    emit_file_result,
    emit_rows_written,
)
from datapipeline.execution.settings import resolve_heartbeat_interval_seconds
from datapipeline.io.factory import writer_factory
from datapipeline.io.factory import dataset_writer_factory
from datapipeline.io.dataset_table import DatasetTable
from datapipeline.io.normalization import json_text, raw_payload
from datapipeline.io.output import OutputTarget, output_destination_key
from datapipeline.io.protocols import Writer
from datapipeline.io.sinks.files import AtomicTextFileSink
from datapipeline.io.writers.parquet import DEFAULT_ROW_GROUP_ROWS


@dataclass(frozen=True, kw_only=True)
class ArtifactOutput:
    companion_paths: tuple[str, ...] = ()
    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeOutput:
    rows: Iterable[Any] | None = None
    payload: Mapping[str, Any] | None = None
    render_html: Callable[[], str] | None = None
    target: OutputTarget | None = None

    def __post_init__(self) -> None:
        if self.rows is None and self.payload is None and self.render_html is None:
            raise ValueError("RuntimeOutput requires rows, payload, or render_html.")
        if self.rows is not None and self.payload is not None:
            raise ValueError("RuntimeOutput cannot define both rows and payload.")


@dataclass(frozen=True)
class RoutedRuntimeOutput:
    rows: Iterable[tuple[str, Any]]
    targets: Mapping[str, OutputTarget]
    limit_per_output: int | None = None


@dataclass(frozen=True)
class DatasetTableOutput:
    rows: Iterable[Sample]
    table: DatasetTable
    target: OutputTarget | None = None


@dataclass(frozen=True)
class RoutedDatasetTableOutput:
    rows: Iterable[tuple[str, Sample]]
    tables: Mapping[str, DatasetTable]
    targets: Mapping[str, OutputTarget]
    limit_per_output: int | None = None

    def __post_init__(self) -> None:
        missing_table_ids = sorted(set(self.targets) - set(self.tables))
        if missing_table_ids:
            raise ValueError(
                "Routed dataset table output is missing tables for output IDs: "
                f"{missing_table_ids}."
            )

        unmatched_table_ids = sorted(set(self.tables) - set(self.targets))
        if unmatched_table_ids:
            raise ValueError(
                "Routed dataset table output has tables without targets for output "
                f"IDs: {unmatched_table_ids}."
            )


RoutedOutput: TypeAlias = RoutedRuntimeOutput | RoutedDatasetTableOutput

RuntimeOutputItem = RuntimeOutput | RoutedOutput | DatasetTableOutput


@dataclass(frozen=True)
class RuntimeOutputBatch:
    outputs: Sequence[RuntimeOutputItem]


def fingerprint_artifact_output(
    output: ArtifactOutput,
    *,
    task: ArtifactTask,
    artifacts_root: Path,
) -> tuple[ArtifactFileFingerprint, ...]:
    relative_paths = (task.output, *output.companion_paths)
    normalized_paths = tuple(Path(relative_path) for relative_path in relative_paths)
    path_keys = {output_destination_key(path) for path in normalized_paths}
    if len(normalized_paths) != len(path_keys):
        raise ValueError(f"Artifact '{task.id}' output paths must be unique.")

    artifacts_root = artifacts_root.resolve()
    files: list[ArtifactFileFingerprint] = []
    for relative_path, normalized_path in zip(relative_paths, normalized_paths):
        if normalized_path.is_absolute() or ".." in normalized_path.parts:
            raise ValueError(
                f"Artifact '{task.id}' output path '{relative_path}' must be "
                "relative to the artifacts root."
            )
        full_path = (artifacts_root / normalized_path).resolve()
        try:
            full_path.relative_to(artifacts_root)
        except ValueError as exc:
            raise ValueError(
                f"Artifact '{task.id}' output must stay under {artifacts_root}."
            ) from exc
        if not full_path.is_file():
            raise RuntimeError(
                f"Artifact '{task.id}' did not create its declared output: {full_path}."
            )
        files.append(ArtifactFileFingerprint.from_path(str(normalized_path), full_path))

    return tuple(files)


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
) -> None:
    if result.render_html is None:
        raise ValueError("html output is not supported for this operation.")
    destination = target.destination
    if destination is None:
        raise ValueError("html output requires fs destination.")

    sink = AtomicTextFileSink(
        destination,
        encoding=target.encoding or "utf-8",
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


def _persist_runtime_output(
    result: RuntimeOutput,
    *,
    target: OutputTarget | None,
    heartbeat_interval_seconds: float | None,
    logger: logging.Logger,
) -> None:
    supplied_rows = result.rows
    owned_rows = supplied_rows if supplied_rows is not None else ()
    writer = None
    try:
        with _runtime_rows(owned_rows, logger) as rows:
            effective_target = result.target or target
            if effective_target is None:
                raise ValueError("Runtime operation requires profile output target.")

            if effective_target.format != "html":
                if supplied_rows is None:
                    rows = _payload_rows(result, effective_target)
                writer = writer_factory(effective_target)
                progress = OperationProgressTracker(
                    "write_output",
                    "rows",
                    resolve_heartbeat_interval_seconds(heartbeat_interval_seconds),
                )
                for row in rows:
                    writer.write(row)
                    progress.advance()

        if effective_target.format == "html":
            _write_html_output(result, effective_target, logger)
            return

        assert writer is not None
        writer.close()
    except BaseException:
        if writer is not None:
            try:
                writer.abort()
            except BaseException:
                logger.debug("Failed to abort runtime output writer", exc_info=True)
        raise

    if effective_target.destination is not None:
        emit_file_result("Output", effective_target.destination)
    elif effective_target.transport == "stdout":
        logger.info("Output: stdout")


def _persist_dataset_table_output(
    result: DatasetTableOutput,
    *,
    target: OutputTarget | None,
    heartbeat_interval_seconds: float | None,
    logger: logging.Logger,
) -> None:
    writer = None
    try:
        with _runtime_rows(result.rows, logger) as rows:
            effective_target = result.target or target
            if effective_target is None:
                raise ValueError("Runtime operation requires profile output target.")
            writer = dataset_writer_factory(effective_target, result.table)
            progress = OperationProgressTracker(
                "write_output",
                "rows",
                resolve_heartbeat_interval_seconds(heartbeat_interval_seconds),
            )
            for row in rows:
                writer.write(row)
                progress.advance()
        writer.close()
    except BaseException:
        if writer is not None:
            try:
                writer.abort()
            except BaseException:
                logger.debug("Failed to abort dataset table writer", exc_info=True)
        raise

    if effective_target.destination is not None:
        emit_file_result("Output", effective_target.destination)


def _planned_routed_targets(
    result: RoutedOutput,
) -> list[tuple[str, OutputTarget, Path]]:
    if not result.targets:
        raise ValueError("Routed runtime output requires at least one target.")

    planned: list[tuple[str, OutputTarget, Path]] = []
    destination_owners: dict[str, str] = {}
    for output_id, target in result.targets.items():
        destination = target.destination
        if target.transport != "fs" or destination is None:
            raise ValueError("Routed runtime output requires fs destinations.")
        destination_key = output_destination_key(destination)
        previous = destination_owners.get(destination_key)
        if previous is not None:
            raise ValueError(
                f"Routed outputs {previous!r} and {output_id!r} resolve to the same "
                f"destination: {destination}"
            )
        destination_owners[destination_key] = output_id
        planned.append((output_id, target, destination))
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
    *,
    heartbeat_interval_seconds: float | None,
    logger: logging.Logger,
) -> None:
    writers: dict[str, Writer] = {}
    try:
        with _runtime_rows(result.rows, logger) as rows:
            planned = _planned_routed_targets(result)
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


def _close_pending_runtime_outputs(
    outputs: Sequence[RuntimeOutputItem],
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
    outputs: Sequence[RuntimeOutputItem],
    target: OutputTarget | None,
    heartbeat_interval_seconds: float | None,
    logger: logging.Logger,
) -> None:
    attempted = 0
    try:
        for output in outputs:
            attempted += 1
            if isinstance(output, RoutedOutput):
                _persist_routed_output(
                    output,
                    heartbeat_interval_seconds=heartbeat_interval_seconds,
                    logger=logger,
                )
            elif isinstance(output, DatasetTableOutput):
                _persist_dataset_table_output(
                    output,
                    target=target,
                    heartbeat_interval_seconds=heartbeat_interval_seconds,
                    logger=logger,
                )
            else:
                _persist_runtime_output(
                    output,
                    target=target,
                    heartbeat_interval_seconds=heartbeat_interval_seconds,
                    logger=logger,
                )
    except BaseException:
        _close_pending_runtime_outputs(outputs[attempted:], logger)
        raise


def persist_runtime_result(
    result: object,
    *,
    target: OutputTarget | None,
    heartbeat_interval_seconds: float | None = None,
    logger: logging.Logger,
) -> None:
    if result is None:
        return
    if isinstance(result, RuntimeOutputBatch):
        outputs = result.outputs
    elif isinstance(result, RuntimeOutputItem):
        outputs = (result,)
    else:
        raise TypeError("Runtime operation returned an unsupported output type.")
    _persist_runtime_outputs(
        outputs,
        target,
        heartbeat_interval_seconds,
        logger,
    )
