"""Persisted artifact build state."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jerrythomas.io.json_file import write_json_object
from jerrythomas.services.path_policy import resolve_artifact_output_path

BUILD_STATE_VERSION = 8
_BUILD_STATE_PATH = Path("_system/build/state.json")


class ArtifactFileFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str
    size: int = Field(strict=True, ge=0)
    mtime_ns: int = Field(strict=True, ge=0)
    ctime_ns: int = Field(strict=True, ge=0)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = Path(value)
        if not value or path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact file path must be relative")
        return str(path)

    @classmethod
    def from_path(cls, relative_path: str, path: Path) -> Self:
        stat = path.stat()
        return cls(
            relative_path=relative_path,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            ctime_ns=stat.st_ctime_ns,
        )


class ArtifactInfo(BaseModel):
    """Metadata describing a materialized artifact."""

    model_config = ConfigDict(extra="forbid")

    artifact_hash: str
    files: tuple[ArtifactFileFingerprint, ...]
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def relative_path(self) -> str:
        return self.files[0].relative_path

    @model_validator(mode="after")
    def validate_files(self) -> Self:
        if not self.files:
            raise ValueError("artifact files must not be empty")
        paths = [file.relative_path for file in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("artifact file paths must be unique")
        return self


class BuildState(BaseModel):
    """Minimal persisted state for caching build outputs."""

    model_config = ConfigDict(extra="forbid")

    version: int = BUILD_STATE_VERSION
    artifacts: dict[str, ArtifactInfo] = Field(default_factory=dict)

    def register(
        self,
        key: str,
        artifact_hash: str,
        files: tuple[ArtifactFileFingerprint, ...],
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        self.artifacts[key] = ArtifactInfo(
            artifact_hash=artifact_hash,
            files=files,
            meta=dict(meta or {}),
        )


def load_build_state(artifacts_root: Path) -> BuildState | None:
    path = resolve_artifact_output_path(_BUILD_STATE_PATH, artifacts_root)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("version") != BUILD_STATE_VERSION:
        return None
    return BuildState.model_validate(data)


def save_build_state(state: BuildState, artifacts_root: Path) -> None:
    path = resolve_artifact_output_path(_BUILD_STATE_PATH, artifacts_root)
    write_json_object(path, state.model_dump())
