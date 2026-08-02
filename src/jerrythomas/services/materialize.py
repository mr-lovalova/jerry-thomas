from pathlib import Path

from jerrythomas.io.compression import Compression
from jerrythomas.io.factory import writer_factory
from jerrythomas.io.output import OutputTarget
from jerrythomas.pipelines.stream.pipeline import run_stream_pipeline
from jerrythomas.runtime import Runtime


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
) -> Path:
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
    check_materialize_destination(output_path, overwrite)

    rows = run_stream_pipeline(runtime, stream_id)
    try:
        writer = writer_factory(output, overwrite=overwrite)
        try:
            for row in rows:
                writer.write(row)
            writer.close()
        except BaseException:
            writer.abort()
            raise
    finally:
        close = getattr(rows, "close", None)
        if callable(close):
            close()
    return output_path.resolve()


def check_materialize_destination(
    path: Path,
    overwrite: bool,
) -> None:
    if not overwrite and path.exists():
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")


def _materialize_compression(path: Path) -> Compression | None:
    if path.name.endswith(".jsonl.gz"):
        return "gzip"
    if path.suffix == ".jsonl":
        return None
    raise ValueError("materialize output must use a .jsonl or .jsonl.gz path")
