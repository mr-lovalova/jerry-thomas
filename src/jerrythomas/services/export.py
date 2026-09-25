import logging
from pathlib import Path

from jerrythomas.config.tasks.stream import StreamTask
from jerrythomas.io.compression import Compression
from jerrythomas.operations.persistence import (
    RuntimeOutput,
    WrittenOutput,
    persist_runtime_output,
)
from jerrythomas.io.output import OutputTarget
from jerrythomas.io.runs import export_receipt_path
from jerrythomas.io.recipes import recipe_path
from jerrythomas.operations.outputs.execution import OutputOptions, run_output_operation
from jerrythomas.runtime import Runtime
from jerrythomas.services.execution_lock import output_lock_path


def resolve_export_output(output: Path) -> OutputTarget:
    compression = _export_compression(output)
    return OutputTarget(
        transport="fs",
        format="jsonl",
        view="raw",
        encoding="utf-8",
        destination=output.resolve(),
        compression=compression,
    )


def export_stream(
    runtime: Runtime,
    task: StreamTask,
    output: OutputTarget,
    overwrite: bool = False,
) -> WrittenOutput:
    output_path = output.destination
    if (
        output.transport != "fs"
        or output.format != "jsonl"
        or output.view != "raw"
        or output_path is None
    ):
        raise ValueError("export requires a raw JSONL filesystem output")
    artifacts_root = runtime.artifacts_root.resolve()
    if output_path.is_relative_to(artifacts_root):
        raise ValueError(
            f"export output must be outside the managed artifacts root: {output_path}"
        )
    _check_data_destination(output_path, overwrite)

    result = run_output_operation(
        runtime, task, OutputOptions(output_format=output.format)
    )
    if not isinstance(result, RuntimeOutput) or result.rows is None:
        raise TypeError("Records operation must return RuntimeOutput with rows.")
    row_count = persist_runtime_output(
        result,
        output,
        runtime.heartbeat_interval_seconds,
        logging.getLogger(__name__),
        overwrite=overwrite,
    )
    assert row_count is not None
    return WrittenOutput(output_path.resolve(), None, row_count)


def check_export_destination(
    path: Path,
    overwrite: bool,
) -> None:
    _check_data_destination(path, overwrite)
    lock = output_lock_path(path)
    if lock.is_symlink() or (lock.exists() and not lock.is_file()):
        raise ValueError(f"Export lock must be a regular file: {lock}")
    receipt = export_receipt_path(path)
    for sidecar in (receipt, recipe_path(receipt)):
        if sidecar.is_symlink() or (sidecar.exists() and not sidecar.is_file()):
            raise ValueError(f"Export receipt/recipe must be a regular file: {sidecar}")
        if not overwrite and sidecar.exists():
            raise FileExistsError(
                f"{sidecar} already exists; pass --overwrite to replace it"
            )


def _check_data_destination(path: Path, overwrite: bool) -> None:
    if not overwrite and path.exists():
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")


def _export_compression(path: Path) -> Compression | None:
    if path.name.endswith(".jsonl.gz"):
        return "gzip"
    if path.suffix == ".jsonl":
        return None
    raise ValueError("export output must use a .jsonl or .jsonl.gz path")
