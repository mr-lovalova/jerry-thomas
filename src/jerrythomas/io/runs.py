from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jerrythomas.config.dataset.split import FoldRole, SplitConfig, resolve_fold_output
from jerrythomas.config.preview import PreviewStage
from jerrythomas.config.profiles.output import Format, View
from jerrythomas.io.compression import Compression
from jerrythomas.io.json_file import write_json_object
from jerrythomas.io.recipes import (
    RecipeReference,
    RunRecipe,
    load_recipe,
    recipe_path,
    recipe_reference,
    save_recipe,
)

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


class RunFoldOutput(BaseModel):
    """The configured fold role represented by an output."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str
    role: FoldRole
    labels: tuple[str, ...] = Field(min_length=1)


class RunOutput(BaseModel):
    """A completed output, addressed relative to the saved run directory."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    profile: str
    operation: str
    stream: str | None = None
    output_id: str | None
    path: str
    format: Format
    view: View | None
    encoding: str | None
    compression: Compression | None
    row_count: int | None = Field(ge=0)
    fold: RunFoldOutput | None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    size_bytes: int | None = Field(default=None, ge=0)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in value
            or PureWindowsPath(value).drive
            or "\x00" in value
            or path.as_posix() != value
        ):
            raise ValueError("output path must be a relative POSIX file path")
        return value


class RunMetadata(BaseModel):
    """Versioned manifest describing a single run and its completed outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[4]
    dataset_id: str | None
    dataset_version: str | None
    command: Literal["serve", "export"]
    run_id: str
    started_at: str
    finished_at: str | None = None
    status: RunStatus
    notes: str | None = None
    preview: PreviewStage | None = None
    split: SplitConfig | None
    outputs: tuple[RunOutput, ...]
    recipe: RecipeReference

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 4:
            raise ValueError("unsupported run manifest schema_version; expected 4")
        return value

    @model_validator(mode="after")
    def validate_outputs(self) -> Self:
        if (self.dataset_id is None) != (self.dataset_version is None):
            raise ValueError("dataset_id and dataset_version must be supplied together")
        if self.dataset_id is not None:
            assert self.dataset_version is not None
            if not self.dataset_id.strip() or not self.dataset_version.strip():
                raise ValueError("dataset identity must not be empty")
        if self.command == "export":
            if (
                self.preview is not None
                or self.split is not None
                or self.dataset_id is not None
            ):
                raise ValueError(
                    "export receipts must not declare dataset, preview, or split"
                )
            if self.status == "success" and len(self.outputs) != 1:
                raise ValueError("successful export receipts require one output")
            for output in self.outputs:
                if (
                    not output.stream
                    or output.output_id is not None
                    or output.fold is not None
                    or PurePosixPath(output.path).name != output.path
                ):
                    raise ValueError(
                        "export receipts require one adjacent stream output"
                    )
        identities = [(output.profile, output.output_id) for output in self.outputs]
        if len(identities) != len(set(identities)):
            raise ValueError("run outputs must have unique profile/output_id pairs")
        paths = [output.path for output in self.outputs]
        if len(paths) != len(set(paths)):
            raise ValueError("run outputs must have unique paths")
        for output in self.outputs:
            if self.preview is not None:
                if output.fold is not None:
                    raise ValueError("preview outputs must not declare a fold")
                continue
            if self.split is None:
                if output.fold is not None:
                    raise ValueError("fold outputs require a saved split configuration")
                continue
            if output.output_id is None:
                if output.fold is not None:
                    raise ValueError("fold outputs require an output_id")
                continue
            try:
                fold, role, labels = resolve_fold_output(self.split, output.output_id)
            except KeyError as exc:
                raise ValueError(
                    f"output_id {output.output_id!r} is not defined by the saved split"
                ) from exc
            if output.fold != RunFoldOutput(id=fold.id, role=role, labels=labels):
                raise ValueError(
                    f"fold metadata for {output.output_id!r} does not match the saved split"
                )
        return self


@dataclass(frozen=True)
class SavedRun:
    """A manifest loaded independently of the current project configuration."""

    metadata_path: Path
    metadata: RunMetadata

    @property
    def directory(self) -> Path:
        return self.metadata_path.parent

    @property
    def outputs(self) -> tuple[Path, ...]:
        return tuple(self.directory / output.path for output in self.metadata.outputs)

    @property
    def recipe_path(self) -> Path:
        return self.directory / self.metadata.recipe.path

    def load_recipe(self) -> RunRecipe:
        """Read and verify the saved recipe without reopening project configuration."""
        if self.metadata.command == "export":
            if _load_run_metadata(self.metadata_path) != self.metadata:
                raise ValueError("export receipt changed; reload the saved run")
        return load_recipe(self.directory, self.metadata.recipe)

    def output(self, profile: str, output_id: str | None = None) -> RunOutput:
        """Select an exact profile/output ID; None selects an unsplit output."""
        for output in self.metadata.outputs:
            if output.profile == profile and output.output_id == output_id:
                return output
        raise KeyError(
            f"Run has no output for profile {profile!r}, output_id {output_id!r}"
        )

    def output_path(self, profile: str, output_id: str | None = None) -> Path:
        """Resolve a selected file without opening any output data."""
        if self.metadata.command == "export":
            if _load_run_metadata(self.metadata_path) != self.metadata:
                raise ValueError("export receipt changed; reload the saved run")
        output = self.output(profile, output_id)
        path = (self.directory / output.path).resolve(strict=True)
        if not path.is_relative_to(self.directory):
            raise ValueError("saved output resolves outside the run directory")
        if not path.is_file():
            raise ValueError(f"Saved output is not a file: {path}")
        return path


def load_run(path: str | Path) -> SavedRun:
    """Load a successful saved run, pinning a latest symlink to its current run.

    Accepts a run directory or an explicit receipt file. Only metadata is read.
    Output existence is checked when output_path is called.
    Unversioned manifests and unsupported schema versions are rejected.
    """
    path = Path(path).resolve(strict=True)
    metadata_path = path / "run.json" if path.is_dir() else path
    metadata = _load_run_metadata(metadata_path)
    if metadata.status != "success" or metadata.finished_at is None:
        raise ValueError("saved run must be successfully finished")
    return SavedRun(metadata_path, metadata)


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


def _write_run_metadata(
    meta: RunMetadata, path: Path, *, overwrite: bool = True
) -> None:
    write_json_object(path, meta.model_dump(mode="json"), overwrite=overwrite)


def _load_run_metadata(path: Path) -> RunMetadata:
    return RunMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def start_run(
    paths: RunPaths | Path,
    *,
    recipe: RunRecipe,
    run_id: str | None = None,
    command: Literal["serve", "export"] = "serve",
    overwrite: bool = True,
    preview: PreviewStage | None = None,
    split: SplitConfig | None = None,
) -> RunMetadata:
    """Initialise a previously planned run."""

    if isinstance(paths, RunPaths):
        paths.dataset_dir.mkdir(parents=True, exist_ok=True)
        run_id = paths.run_id
        metadata_path = paths.metadata_path
    else:
        metadata_path = paths

    meta = RunMetadata(
        schema_version=4,
        dataset_id=recipe.dataset_id,
        dataset_version=recipe.dataset_version,
        command=command,
        run_id=run_id or make_run_id(),
        started_at=_now_utc_iso(),
        finished_at=None,
        status="running",
        notes=None,
        preview=preview,
        split=split.model_copy(deep=True) if split is not None else None,
        outputs=(),
        recipe=recipe_reference(recipe_path(metadata_path), recipe),
    )
    if recipe.command != command:
        raise ValueError("Recipe command does not match the run")
    # Invalidate any previous success before replacing its recipe or data.
    _write_run_metadata(meta, metadata_path, overwrite=overwrite)
    save_recipe(recipe_path(metadata_path), recipe, overwrite=overwrite)
    return meta


def finish_run(
    paths: RunPaths | Path,
    status: Literal["success", "failed"],
    notes: str | None = None,
    outputs: tuple[RunOutput, ...] = (),
) -> RunMetadata:
    """Mark an existing run as finished with the given status."""
    metadata_path = paths.metadata_path if isinstance(paths, RunPaths) else paths
    meta = _load_run_metadata(metadata_path)

    if status == "success":
        load_recipe(metadata_path.parent, meta.recipe)

    meta = RunMetadata(
        schema_version=meta.schema_version,
        dataset_id=meta.dataset_id,
        dataset_version=meta.dataset_version,
        command=meta.command,
        run_id=meta.run_id,
        started_at=meta.started_at,
        finished_at=_now_utc_iso(),
        status=status,
        notes=notes if notes is not None else meta.notes,
        preview=meta.preview,
        split=meta.split,
        outputs=outputs,
        recipe=meta.recipe,
    )

    _write_run_metadata(meta, metadata_path)
    return meta


def finish_run_success(
    paths: RunPaths | Path,
    notes: str | None = None,
    outputs: tuple[RunOutput, ...] = (),
) -> RunMetadata:
    """Convenience wrapper to mark a run as successful."""
    return finish_run(paths, status="success", notes=notes, outputs=outputs)


def finish_run_failed(paths: RunPaths | Path, notes: str | None = None) -> RunMetadata:
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


def export_receipt_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.run.json")
