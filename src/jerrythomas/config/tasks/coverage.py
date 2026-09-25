from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat

from .base import RuntimeTask


class CoverageOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold: StrictFloat = Field(default=0.95, ge=0.0, le=1.0)


class CoverageTask(RuntimeTask):
    entrypoint: Literal["core.coverage_report"] = Field(default="core.coverage_report")
    options: CoverageOptions = Field(default_factory=CoverageOptions)
