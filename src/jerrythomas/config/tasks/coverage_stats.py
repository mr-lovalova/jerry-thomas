from typing import ClassVar, Literal

from pydantic import Field

from .base import DatasetArtifactTask


class CoverageStatsTask(DatasetArtifactTask):
    default_filename: ClassVar[str] = "coverage_stats.json"

    id: str = Field(default="coverage_stats")
    entrypoint: str = Field(default="core.coverage_statistics")
    stage: Literal["assembled", "postprocessed"] = "postprocessed"
