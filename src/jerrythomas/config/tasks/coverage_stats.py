from typing import Literal

from pydantic import Field

from .base import ArtifactTask


class CoverageStatsTask(ArtifactTask):
    id: str = Field(default="coverage_stats")
    entrypoint: str = Field(default="core.coverage_statistics")
    output: str = Field(default="build/coverage_stats.json", validation_alias="path")
    stage: Literal["assembled", "postprocessed"] = "postprocessed"
