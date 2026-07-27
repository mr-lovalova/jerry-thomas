import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from datapipeline.config.preview import PreviewStage
from datapipeline.io.json_file import write_json_object

RunStatus = Literal["running", "success", "failed"]


@dataclass(frozen=True)
class RunPaths:
    """Resolved filesystem paths for a single run rooted at a serve directory.

    The serve directory is typically the user-configured `directory` for the
    filesystem transport (e.g. `data/processed/...`).

    Layout:

        serve_root/
          runs/
            <run_id>/
              dataset/        # main output for this run
              run.json        # metadata for this run
          latest/             # symlink pointing at the current live run
    """

    serve_root: Path
    run_id: str
    run_root: Path
    dataset_dir: Path
    metadata_path: Path


@dataclass
class RunMetadata:
    """Metadata describing a single run."""

    run_id: str
    started_at: str
    finished_at: str | None = None
    status: RunStatus | None = None
    notes: str | None = None
    preview: PreviewStage | None = None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id() -> str:
    """Create a filesystem-safe, sortable run identifier."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%fZ")


def get_run_paths(serve_root: Path, run_id: str | None = None) -> RunPaths:
    """Build RunPaths for a run rooted at the given serve directory."""
    if run_id is None:
        run_id = make_run_id()

    serve_root = serve_root.resolve()
    runs_root = serve_root / "runs"
    run_root = runs_root / run_id
    dataset_dir = run_root / "dataset"
    metadata_path = run_root / "run.json"

    return RunPaths(
        serve_root=serve_root,
        run_id=run_id,
        run_root=run_root,
        dataset_dir=dataset_dir,
        metadata_path=metadata_path,
    )


def _write_run_metadata(meta: RunMetadata, path: Path) -> None:
    write_json_object(path, asdict(meta))


def _load_run_metadata(path: Path) -> RunMetadata:
    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return RunMetadata(**data)


def start_run(
    paths: RunPaths,
    *,
    preview: PreviewStage | None = None,
) -> RunMetadata:
    """Initialise a previously planned run."""

    paths.dataset_dir.mkdir(parents=True, exist_ok=True)

    meta = RunMetadata(
        run_id=paths.run_id,
        started_at=_now_utc_iso(),
        finished_at=None,
        status="running",
        notes=None,
        preview=preview,
    )
    _write_run_metadata(meta, paths.metadata_path)
    return meta


def finish_run(
    paths: RunPaths,
    status: Literal["success", "failed"],
    notes: str | None = None,
) -> RunMetadata:
    """Mark an existing run as finished with the given status."""
    meta = _load_run_metadata(paths.metadata_path)

    meta.finished_at = _now_utc_iso()
    meta.status = status
    if notes is not None:
        meta.notes = notes

    _write_run_metadata(meta, paths.metadata_path)
    return meta


def finish_run_success(paths: RunPaths, notes: str | None = None) -> RunMetadata:
    """Convenience wrapper to mark a run as successful."""
    return finish_run(paths, status="success", notes=notes)


def finish_run_failed(paths: RunPaths, notes: str | None = None) -> RunMetadata:
    """Convenience wrapper to mark a run as failed."""
    return finish_run(paths, status="failed", notes=notes)


def set_latest_run(paths: RunPaths) -> None:
    """Point ``latest`` at a completed run without copying its output."""
    latest_root = paths.serve_root / "latest"
    pending_root = paths.serve_root / f".latest-{paths.run_id}"
    paths.serve_root.mkdir(parents=True, exist_ok=True)

    if latest_root.exists() and not latest_root.is_symlink():
        raise FileExistsError(f"{latest_root} exists and is not a symbolic link")

    pending_root.symlink_to(paths.run_root, target_is_directory=True)
    try:
        pending_root.replace(latest_root)
    finally:
        if pending_root.is_symlink():
            pending_root.unlink()
