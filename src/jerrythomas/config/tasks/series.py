from typing import Literal

from pydantic import Field

from .base import ArtifactTask


class SeriesTask(ArtifactTask):
    id: Literal["series"] = Field(default="series")
    entrypoint: str = Field(default="core.artifact.series")
    output: str = Field(default="build/series/manifest.json")
