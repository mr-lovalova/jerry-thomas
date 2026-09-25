from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jerrythomas.config.execution import ExecutionConfig
from jerrythomas.config.observability import ObservabilityConfig
from jerrythomas.config.preview import PreviewStage

from .build import ArtifactMode
from .output import ServeOutputConfig
from .serve import normalize_include_outputs


class ProfileDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cmd: str
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    observability: ObservabilityConfig | None = None


class ServeProfileDefaults(ProfileDefaults):
    cmd: Literal["serve"]
    output: ServeOutputConfig | None = None
    artifact_mode: ArtifactMode | None = Field(default=None)
    include_outputs: list[str] | None = Field(default=None, min_length=1)
    limit: int | None = Field(default=None, ge=1)
    preview: PreviewStage | None = None
    throttle_ms: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)

    @field_validator("include_outputs", mode="before")
    @classmethod
    def _normalize_include_outputs(cls, value: object) -> list[str] | None:
        return normalize_include_outputs(value)


class BuildProfileDefaults(ProfileDefaults):
    cmd: Literal["build"]
    artifact_mode: ArtifactMode | None = Field(default=None)

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


class InspectProfileDefaults(ProfileDefaults):
    cmd: Literal["inspect"]
    output: ServeOutputConfig | None = None
    artifact_mode: ArtifactMode | None = Field(default=None)


class ExportProfileDefaults(ProfileDefaults):
    cmd: Literal["export"]
    artifact_mode: ArtifactMode | None = Field(default=None)
    overwrite: StrictBool | None = None
