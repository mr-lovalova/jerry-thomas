from typing import Literal

from pydantic import Field

from .base import ArtifactTask


class CoverageStatsTask(ArtifactTask):
    id: Literal["coverage_stats"] = Field(default="coverage_stats")
    entrypoint: str = Field(default="core.artifact.coverage_stats")
    output: str = Field(default="build/coverage_stats.json")
    stage: Literal["assembled", "postprocessed"] = "postprocessed"
