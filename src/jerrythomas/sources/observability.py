import os
from pathlib import Path
from urllib.parse import urlparse

from jerrythomas.domain.stream import RecordStream
from jerrythomas.execution.events import ProgressResource, ProgressSnapshot
from jerrythomas.execution.pipeline import ProgressReader
from jerrythomas.sources.adapters.fs import FsFileTransport, FsGlobTransport
from jerrythomas.sources.adapters.http import HttpTransport
from jerrythomas.sources.loader import DataLoader
from jerrythomas.sources.parquet_loader import ParquetLoader
from jerrythomas.sources.ports import SourceTransport
from jerrythomas.sources.source import Source


def source_progress(
    stream_source: RecordStream[object],
) -> ProgressReader | None:
    if not isinstance(stream_source, Source):
        return None

    loader = stream_source.loader
    transport = loader.transport if isinstance(loader, DataLoader) else None
    resources_by_id: dict[str, ProgressResource] = {}
    resource: ProgressResource | None = None
    files: tuple[str, ...] | None = None
    if isinstance(loader, ParquetLoader):
        if loader.is_glob:
            files = loader.files
    elif isinstance(transport, FsGlobTransport):
        files = transport.files
    if files is not None:
        total = len(files)
        root = _glob_root(files)
        resources_by_id = {
            path: ProgressResource(
                index,
                total,
                f'"{_relative_label(path, root)}"',
            )
            for index, path in enumerate(files, start=1)
        }
        resource = resources_by_id[files[0]]
    elif isinstance(loader, ParquetLoader):
        name = Path(loader.path).name or loader.path
        resource = ProgressResource(1, 1, f'"{name}"')
    elif isinstance(transport, FsFileTransport):
        name = Path(transport.path).name or transport.path
        resource = ProgressResource(1, 1, f'"{name}"')
    elif isinstance(transport, HttpTransport):
        host = urlparse(transport.url).netloc or "http"
        resource = ProgressResource(1, 1, f"@{host}")
    initial_resource = resource

    def read_progress(completed: int) -> ProgressSnapshot:
        nonlocal resource
        current_resource_id = loader.current_resource_uri
        if completed == 0 and current_resource_id is None:
            resource = initial_resource
        elif current_resource_id is not None and resources_by_id:
            resource = resources_by_id.get(current_resource_id)
        return ProgressSnapshot(
            completed=completed,
            unit=loader.progress_unit,
            resource=resource,
        )

    return read_progress


def source_summary(stream_source: RecordStream[object]) -> str | None:
    if not isinstance(stream_source, Source):
        return None
    loader = stream_source.loader
    if isinstance(loader, ParquetLoader):
        files = loader.files
        if not loader.is_glob:
            return f"transport=fs.file file={Path(loader.path).name or loader.path}"
        total = len(files)
        root = _glob_root(files)
        if total == 1:
            return f"transport=fs.glob count=1 file={_relative_label(files[0], root)}"
        return (
            f"transport=fs.glob count={total} "
            f"first={_relative_label(files[0], root)} "
            f"last={_relative_label(files[-1], root)}"
        )
    if not isinstance(loader, DataLoader):
        return None
    return _transport_source_summary(loader.transport)


def _transport_source_summary(transport: SourceTransport) -> str | None:
    if isinstance(transport, FsFileTransport):
        path = transport.path
        parts = ["transport=fs.file"]
        if transport.compression is not None:
            parts.append(f"compression={transport.compression}")
        parts.append(f"file={Path(path).name or str(path)}")
        return " ".join(parts)

    if isinstance(transport, FsGlobTransport):
        files = transport.files
        total = len(files)
        parts = ["transport=fs.glob"]
        if transport.compression is not None:
            parts.append(f"compression={transport.compression}")
        parts.append(f"count={total}")
        root = _glob_root(files)
        if total == 1:
            parts.append(f"file={_relative_label(files[0], root)}")
        else:
            parts.append(f"first={_relative_label(files[0], root)}")
            parts.append(f"last={_relative_label(files[-1], root)}")
        return " ".join(parts)

    if isinstance(transport, HttpTransport):
        url = transport.url
        parsed_url = urlparse(url)
        host = parsed_url.netloc or "http"
        resource = Path(parsed_url.path or "").name
        summary = f"transport=http.fetch host={host}"
        return f"{summary} resource={resource}" if resource else summary

    return None


def _glob_root(files: tuple[str, ...]) -> Path:
    if len(files) == 1:
        return Path(files[0]).parent
    return Path(os.path.commonpath(files))


def _relative_label(path: str, root: Path) -> str:
    try:
        relative = Path(path).relative_to(root)
    except ValueError:
        return Path(path).name or path
    return relative.as_posix() or relative.name or path
