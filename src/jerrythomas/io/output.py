from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from unicodedata import normalize

from jerrythomas.config.profiles.output import (
    Format,
    ServeOutputConfig,
    Transport,
    View,
)
from jerrythomas.io.compression import Compression
from jerrythomas.io.runs import RunPaths
from jerrythomas.services.path_policy import (
    resolve_relative_to_base,
    sanitize_path_segment,
    workspace_cwd,
)


def _format_suffix(fmt: Format) -> str:
    suffix_map = {
        "jsonl": ".jsonl",
        "csv": ".csv",
        "parquet": ".parquet",
        "pickle": ".pkl",
        "txt": ".txt",
        "html": ".html",
    }
    return suffix_map[fmt]


def _output_suffix(fmt: Format, compression: Compression | None) -> str:
    suffix = _format_suffix(fmt)
    return f"{suffix}.gz" if compression == "gzip" else suffix


def _default_view_for_format(fmt: Format) -> View:
    if fmt in {"csv", "parquet"}:
        return "flat"
    return "raw"


def _resolve_view(fmt: Format, configured_view: View | None) -> View:
    return configured_view or _default_view_for_format(fmt)


def _default_filename_for_format(
    fmt: Format,
    compression: Compression | None,
) -> str:
    return f"dataset{_output_suffix(fmt, compression)}"


@dataclass(frozen=True)
class OutputTarget:
    """Resolved writer target describing how and where to emit records."""

    transport: Transport
    format: Format
    view: View
    encoding: str | None
    destination: Path | None
    compression: Compression | None = None
    run: RunPaths | None = None

    def for_output(self, output_id: str) -> "OutputTarget":
        if self.transport != "fs" or self.destination is None:
            return self
        safe_output_id = sanitize_path_segment(output_id)
        dest = self.destination
        suffix = _output_suffix(self.format, self.compression)
        stem = dest.name.removesuffix(suffix)
        new_name = f"{stem}.{safe_output_id}{suffix}"
        new_path = dest.with_name(new_name)
        return replace(self, destination=new_path)


def output_destination_key(path: Path) -> str:
    """Return the portable identity used to reject colliding output paths."""
    return normalize("NFC", str(path)).casefold()


def output_paths_overlap(first: Path, second: Path) -> bool:
    """Return whether two filesystem destinations cannot both be files."""
    first_path = Path(output_destination_key(first.resolve()))
    second_path = Path(output_destination_key(second.resolve()))
    return first_path.is_relative_to(second_path) or second_path.is_relative_to(
        first_path
    )


def validate_output_destinations(
    outputs: Sequence[OutputTarget],
) -> None:
    data_paths: list[tuple[Path, str]] = []

    for output in outputs:
        if output.transport == "stdout":
            continue
        if output.destination is None:
            raise ValueError("Filesystem data output has no destination.")
        destination = output.destination.resolve()
        destination_key = output_destination_key(destination)
        for previous, previous_key in data_paths:
            if destination_key == previous_key:
                raise ValueError(
                    f"Data outputs resolve to the same path '{destination}'."
                )
            if output_paths_overlap(destination, previous):
                raise ValueError(
                    f"Data output paths overlap: '{previous}' and '{destination}'."
                )
        data_paths.append((destination, destination_key))


class OutputResolutionError(ValueError):
    """Raised when CLI/config output options cannot be resolved."""


def resolve_output_directory(
    config: ServeOutputConfig | None,
    *,
    base_path: Path | None = None,
) -> Path | None:
    """Resolve the filesystem output directory when the target uses fs transport."""
    if config is None or config.transport != "fs" or config.directory is None:
        return None
    base = base_path or workspace_cwd()
    return resolve_relative_to_base(config.directory, base)


def resolve_output_target(
    cli_output: ServeOutputConfig | None,
    config_output: ServeOutputConfig | None,
    default: ServeOutputConfig | None = None,
    base_path: Path | None = None,
    profile_name: str | None = None,
    run_paths: RunPaths | None = None,
) -> OutputTarget:
    """
    Resolve the effective output target using CLI override, run config, or default.
    """

    base_path = base_path or workspace_cwd()

    config = cli_output or config_output or default
    if config is None:
        config = ServeOutputConfig(transport="stdout", format="jsonl")

    if config.transport == "stdout":
        return OutputTarget(
            transport="stdout",
            format=config.format,
            view=_resolve_view(config.format, config.view),
            encoding=None,
            destination=None,
            compression=None,
            run=None,
        )

    if config.directory is None:
        raise OutputResolutionError("fs output requires a directory")
    directory = resolve_relative_to_base(
        config.directory,
        base_path,
    )
    if run_paths is not None:
        base_dest_dir = run_paths.dataset_dir
    else:
        base_dest_dir = directory
        if profile_name:
            base_dest_dir = base_dest_dir / sanitize_path_segment(profile_name)
    suffix = _output_suffix(config.format, config.compression)
    format_suffix = _format_suffix(config.format)
    compressed_suffix = f"{format_suffix}.gz"
    filename_stem = config.filename or (
        sanitize_path_segment(profile_name) if profile_name else None
    )
    if filename_stem:
        for extension in (compressed_suffix, format_suffix):
            if filename_stem.endswith(extension):
                raise OutputResolutionError(
                    f"filename must omit the '{extension}' extension"
                )
        filename = f"{filename_stem}{suffix}"
    else:
        filename = _default_filename_for_format(config.format, config.compression)
    dest_path = (base_dest_dir / filename).resolve()

    return OutputTarget(
        transport="fs",
        format=config.format,
        view=_resolve_view(config.format, config.view),
        encoding=config.encoding,
        destination=dest_path,
        compression=config.compression,
        run=run_paths,
    )
