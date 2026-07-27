import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from datapipeline.build.state import (
    ArtifactFileFingerprint,
    ArtifactInfo,
    BuildState,
    load_build_state,
    save_build_state,
)


def test_build_state_derives_primary_path_from_first_file(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    artifact_path = artifacts_root / "snapshot.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("{}", encoding="utf-8")
    fingerprint = ArtifactFileFingerprint.from_path("snapshot.json", artifact_path)
    state = BuildState()
    state.register("snapshot", artifact_hash="hash", files=(fingerprint,))

    save_build_state(state, artifacts_root)

    payload = json.loads(
        (artifacts_root / "_system/build/state.json").read_text(encoding="utf-8")
    )
    assert "relative_path" not in payload["artifacts"]["snapshot"]
    loaded = load_build_state(artifacts_root)
    assert loaded is not None
    assert loaded.artifacts["snapshot"].relative_path == "snapshot.json"


def test_artifact_info_requires_at_least_one_file() -> None:
    with pytest.raises(ValidationError, match="artifact files must not be empty"):
        ArtifactInfo(artifact_hash="hash", files=())


def test_artifact_info_rejects_duplicate_files() -> None:
    fingerprint = ArtifactFileFingerprint(
        relative_path="snapshot.json",
        size=1,
        mtime_ns=1,
        ctime_ns=1,
    )

    with pytest.raises(ValidationError, match="artifact file paths must be unique"):
        ArtifactInfo(
            artifact_hash="hash",
            files=(fingerprint, fingerprint),
        )


def test_build_state_rejects_symlink_escape(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    build_dir = artifacts_root / "_system/build"
    build_dir.mkdir(parents=True)
    outside = tmp_path / "outside-state.json"
    outside.write_text("keep", encoding="utf-8")
    (build_dir / "state.json").symlink_to(outside)

    with pytest.raises(ValueError, match="must stay under artifacts root"):
        save_build_state(BuildState(), artifacts_root)

    assert outside.read_text(encoding="utf-8") == "keep"
