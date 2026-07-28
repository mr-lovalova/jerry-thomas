from datetime import datetime
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from datapipeline.config.constraints import NonEmptyString as ProjectPath


class ProjectPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    streams: ProjectPath | list[ProjectPath]
    sources: ProjectPath | list[ProjectPath]
    dataset: ProjectPath
    artifacts: ProjectPath
    operations: ProjectPath | None = None
    profiles: ProjectPath = "./profiles"

    @field_validator("streams", "sources")
    @classmethod
    def require_config_roots(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, list) and not value:
            raise ValueError("project path lists must not be empty")
        return value


class ProjectGlobals(BaseModel):
    model_config = ConfigDict(extra="allow")
    start_time: datetime | None = None
    end_time: datetime | None = None

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        if self.start_time is not None and (
            self.start_time.tzinfo is None or self.start_time.utcoffset() is None
        ):
            raise ValueError("globals.start_time must be timezone-aware")
        if self.end_time is not None and (
            self.end_time.tzinfo is None or self.end_time.utcoffset() is None
        ):
            raise ValueError("globals.end_time must be timezone-aware")
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.start_time > self.end_time
        ):
            raise ValueError("globals.start_time must not be after globals.end_time")
        return self


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[4]
    artifact_revision: int = Field(strict=True, gt=0)
    name: str | None = None
    variant: str | None = None
    paths: ProjectPaths
    globals: ProjectGlobals = Field(default_factory=ProjectGlobals)
