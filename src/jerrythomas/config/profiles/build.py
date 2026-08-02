from typing import Literal

from pydantic import Field, field_validator

from .base import OperationProfile

ArtifactMode = Literal["AUTO", "FORCE", "OFF"]
ARTIFACT_MODES: tuple[ArtifactMode, ...] = ("AUTO", "FORCE", "OFF")


def normalize_artifact_mode(value: object) -> ArtifactMode | None:
    if value is None:
        return None
    name = str(value).strip().upper()
    if name == "AUTO":
        return "AUTO"
    if name == "FORCE":
        return "FORCE"
    if name == "OFF":
        return "OFF"
    raise ValueError(f"artifact mode must be one of {', '.join(ARTIFACT_MODES)}")


class BuildProfile(OperationProfile):
    cmd: Literal["build"]
    mode: ArtifactMode | None = Field(default=None)

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: object) -> ArtifactMode | None:
        return normalize_artifact_mode(value)
