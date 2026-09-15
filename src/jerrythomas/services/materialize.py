import logging
from pathlib import Path

from jerrythomas.io.compression import Compression
from jerrythomas.operations.persistence import (
    RuntimeOutput,
    WrittenOutput,
    persist_runtime_output,
)
from jerrythomas.io.output import OutputTarget
from jerrythomas.io.runs import materialize_receipt_path
from jerrythomas.pipelines.stream.pipeline import run_stream_pipeline
from jerrythomas.runtime import Runtime
from jerrythomas.services.execution_lock import output_lock_path


def resolve_materialize_output(output: Path) -> OutputTarget:
    compression = _materialize_compression(output)
    return OutputTarget(
        transport="fs",
        format="jsonl",
        view="raw",
        encoding="utf-8",
        destination=output.resolve(),
        compression=compression,
    )


def materialize_stream(
    runtime: Runtime,
    stream_id: str,
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
        raise ValueError("materialize requires a raw JSONL filesystem output")
    artifacts_root = runtime.artifacts_root.resolve()
    if output_path.is_relative_to(artifacts_root):
        raise ValueError(
            f"materialize output must be outside the managed artifacts root: {output_path}"
        )
    _check_data_destination(output_path, overwrite)

    row_count = persist_runtime_output(
        RuntimeOutput(rows=run_stream_pipeline(runtime, stream_id)),
        output,
        runtime.heartbeat_interval_seconds,
        logging.getLogger(__name__),
        overwrite=overwrite,
    )
    assert row_count is not None
    return WrittenOutput(output_path.resolve(), None, row_count)


def check_materialize_destination(
    path: Path,
    overwrite: bool,
) -> None:
    _check_data_destination(path, overwrite)
    lock = output_lock_path(path)
    if lock.is_symlink() or (lock.exists() and not lock.is_file()):
        raise ValueError(f"Materialize lock must be a regular file: {lock}")
    receipt = materialize_receipt_path(path)
    if receipt.is_symlink() or (receipt.exists() and not receipt.is_file()):
        raise ValueError(f"Materialize receipt must be a regular file: {receipt}")
    if not overwrite and receipt.exists():
        raise FileExistsError(
            f"{receipt} already exists; pass --overwrite to replace it"
        )


def _check_data_destination(path: Path, overwrite: bool) -> None:
    if not overwrite and path.exists():
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")


def _materialize_compression(path: Path) -> Compression | None:
    if path.name.endswith(".jsonl.gz"):
        return "gzip"
    if path.suffix == ".jsonl":
        return None
    raise ValueError("materialize output must use a .jsonl or .jsonl.gz path")
