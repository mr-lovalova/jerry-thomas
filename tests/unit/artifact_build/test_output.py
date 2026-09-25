from pathlib import Path

import pytest

from jerrythomas.artifacts.output import (
    ArtifactOutput,
    fingerprint_artifact_output,
)
from jerrythomas.config.tasks.base import ArtifactTask


def test_fingerprint_artifact_output_requires_declared_file(tmp_path: Path) -> None:
    task = ArtifactTask(
        id="snapshot",
        entrypoint="plugin.snapshot",
        path="snapshot.json",
    )

    with pytest.raises(RuntimeError, match="did not create its declared output"):
        fingerprint_artifact_output(
            ArtifactOutput(),
            task=task,
            artifacts_root=tmp_path,
        )


def test_fingerprint_artifact_output_snapshots_declared_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "snapshot.json"
    output.write_text("{}", encoding="utf-8")
    task = ArtifactTask(
        id="snapshot",
        entrypoint="plugin.snapshot",
        path="snapshot.json",
    )

    result = ArtifactOutput(meta={"rows": 1})
    files = fingerprint_artifact_output(
        result,
        task=task,
        artifacts_root=tmp_path,
    )

    assert tuple(file.relative_path for file in files) == ("snapshot.json",)
    assert result.meta == {"rows": 1}


def test_fingerprint_artifact_output_validates_companion_files(
    tmp_path: Path,
) -> None:
    output = tmp_path / "manifest.json"
    companion = tmp_path / "manifest.shards/000000.jsonl.gz"
    output.write_text("{}", encoding="utf-8")
    companion.parent.mkdir()
    companion.write_bytes(b"shard")
    task = ArtifactTask(
        id="series",
        entrypoint="plugin.series",
        path="manifest.json",
    )

    files = fingerprint_artifact_output(
        ArtifactOutput(
            companion_paths=("manifest.shards/000000.jsonl.gz",),
        ),
        task=task,
        artifacts_root=tmp_path,
    )

    assert tuple(file.relative_path for file in files) == (
        "manifest.json",
        "manifest.shards/000000.jsonl.gz",
    )


def test_fingerprint_artifact_output_rejects_escaping_companion(
    tmp_path: Path,
) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    task = ArtifactTask(
        id="series",
        entrypoint="plugin.series",
        path="manifest.json",
    )

    with pytest.raises(ValueError, match="must be relative"):
        fingerprint_artifact_output(
            ArtifactOutput(
                companion_paths=("../outside.jsonl.gz",),
            ),
            task=task,
            artifacts_root=tmp_path,
        )


def test_fingerprint_artifact_output_rejects_case_colliding_companion(
    tmp_path: Path,
) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    task = ArtifactTask(
        id="series",
        entrypoint="plugin.series",
        path="manifest.json",
    )

    with pytest.raises(ValueError, match="output paths must be unique"):
        fingerprint_artifact_output(
            ArtifactOutput(
                companion_paths=("MANIFEST.json",),
            ),
            task=task,
            artifacts_root=tmp_path,
        )
