from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from jerrythomas.artifacts.state import ArtifactFileFingerprint
from jerrythomas.config.tasks.base import ArtifactTask
from jerrythomas.io.output import output_destination_key


@dataclass(frozen=True, kw_only=True)
class ArtifactOutput:
    companion_paths: tuple[str, ...] = ()
    meta: Mapping[str, object] = field(default_factory=dict)


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
