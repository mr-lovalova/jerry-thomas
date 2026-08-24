from pathlib import Path

from jerrythomas.services.path_policy import resolve_workspace_path


def build_cli_output_config(
    transport: str | None,
    fmt: str | None,
    directory: str | None,
    output_encoding: str | None = None,
    output_compression: str | None = None,
    workspace_root: Path | None = None,
    view: str | None = None,
) -> dict[str, object] | None:
    """Collect explicitly provided --output-* leaves for per-profile merging.

    Each flag addresses one leaf of the profile ``output`` block; unset flags
    inherit from the selected profile. Combination validation happens against
    the merged configuration (see merge_output_overrides).
    """
    overrides: dict[str, object] = {}
    if transport is not None:
        overrides["transport"] = transport.lower()
    if fmt is not None:
        overrides["format"] = fmt.lower()
    if view is not None:
        overrides["view"] = view
    if output_encoding is not None:
        overrides["encoding"] = output_encoding
    if output_compression is not None:
        overrides["compression"] = output_compression
    if directory is not None:
        overrides["directory"] = resolve_workspace_path(directory, workspace_root)
    return overrides or None
