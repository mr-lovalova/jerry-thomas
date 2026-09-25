from pydantic import Field

from .base import ArtifactTask


class SeriesTask(ArtifactTask):
    id: str = Field(default="series")
    entrypoint: str = Field(default="core.artifact.series")
    output: str = Field(default="build/series/manifest.json", validation_alias="path")
