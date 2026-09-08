from typing import Literal

from pydantic import TypeAdapter, model_validator
from pydantic_core import PydanticCustomError

from .base import OperationProfile

ArtifactMode = Literal["auto", "rebuild", "require_current"]
ARTIFACT_MODES: tuple[ArtifactMode, ...] = ("auto", "rebuild", "require_current")
ARTIFACT_MODE_ADAPTER: TypeAdapter[ArtifactMode] = TypeAdapter(ArtifactMode)


class BuildProfile(OperationProfile):
    cmd: Literal["build"]
    artifact_mode: ArtifactMode | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_mode(cls, value: object) -> object:
        if isinstance(value, dict) and "mode" in value:
            raise PydanticCustomError(
                "removed_build_mode",
                "Build mode was renamed in v11; use artifact_mode with "
                "auto, rebuild, or require_current",
            )
        return value
