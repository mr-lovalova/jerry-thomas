import re
from pathlib import Path

from jerrythomas.services.project import load_project
from jerrythomas.services.scaffold.locking import ScaffoldLock, acquire_scaffold_lock
from jerrythomas.services.scaffold.paths import (
    default_project_yaml_path,
    ensure_project_scaffold,
    pkg_root,
)
from jerrythomas.services.scaffold.templates import render
from jerrythomas.services.scaffold.utils import write_new_file
from jerrythomas.services.streams.loader import declared_source_ids

DEFAULT_SYNTHETIC_LOADER_EP = "core.synthetic.ticks"
DEFAULT_TEMPORAL_RECORD_PARSER_EP = "core.temporal_record"


def default_loader_config(
    transport: str,
    source_format: str | None,
) -> dict[str, object]:
    if transport == "synthetic":
        if source_format is not None:
            raise ValueError("Synthetic sources do not use a source format")
        return {
            "entrypoint": DEFAULT_SYNTHETIC_LOADER_EP,
            "args": {
                "start": "<ISO8601>",
                "end": "<ISO8601>",
                "frequency": "1h",
            },
        }
    if source_format not in {None, "csv", "json", "jsonl", "parquet"}:
        raise ValueError(f"Unsupported source format: {source_format!r}")

    if transport == "fs":
        reader: dict[str, object] = {
            "format": source_format or "<FORMAT (csv|json|jsonl|parquet)>",
        }
        if source_format != "parquet":
            reader["encoding"] = "utf-8"
        if source_format == "csv":
            reader["delimiter"] = ","
        return {
            "transport": "fs",
            "path": "<PATH OR GLOB>",
            "reader": reader,
        }
    if transport == "http":
        if source_format == "parquet":
            raise ValueError("HTTP sources do not support parquet format")
        reader = {
            "format": source_format or "<FORMAT (json|jsonl|csv)>",
            "encoding": "utf-8",
        }
        if source_format == "csv":
            reader["delimiter"] = ","
        return {
            "transport": "http",
            "url": "<https://api.example.com/data.json>",
            "headers": {},
            "params": {},
            "reader": reader,
        }
    raise ValueError(f"Unsupported source transport: {transport!r}")


def validate_source_id(source_id: str) -> None:
    if (
        "." not in source_id
        or re.fullmatch(
            r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*",
            source_id,
        )
        is None
    ):
        raise ValueError(
            "source_id must contain at least two dot-separated segments using only "
            "letters, numbers, underscores, or hyphens"
        )


def source_freshness(loader: dict[str, object]) -> str | None:
    """Return the freshness declaration scaffolded sources must carry."""
    if loader.get("transport") == "fs":
        return None
    return "opaque"


def create_source_yaml(
    *,
    source_id: str,
    loader: dict[str, object],
    parser_ep: str,
    parser_args: dict[str, object] | None = None,
    root: Path | None,
    project_yaml: Path | None = None,
    scaffold_lock: ScaffoldLock | None = None,
) -> Path:
    root_dir, _, _ = pkg_root(root)
    proj_yaml = (
        project_yaml.resolve()
        if project_yaml is not None
        else default_project_yaml_path(root_dir)
    )
    with acquire_scaffold_lock(proj_yaml.parent, scaffold_lock) as project_lock:
        validate_source_id(source_id)
        parser_args = parser_args or {}
        if proj_yaml.exists():
            project = load_project(proj_yaml)
            existing_path = project.source_dirs[0] / f"{source_id}.yaml"
            if existing_path.exists():
                raise FileExistsError(f"{existing_path} already exists")
        project = ensure_project_scaffold(proj_yaml, project_lock)
        if source_id in declared_source_ids(project):
            raise FileExistsError(f"Source id '{source_id}' already exists")
        src_cfg_path = project.source_dirs[0] / f"{source_id}.yaml"
        write_new_file(
            src_cfg_path,
            render(
                "source.yaml.j2",
                id=source_id,
                parser_ep=parser_ep,
                parser_args=parser_args,
                loader=loader,
                freshness=source_freshness(loader),
            ),
        )
        return src_cfg_path.resolve()
