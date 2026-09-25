from typing import ClassVar

from pydantic import Field

from .base import DatasetArtifactTask


class SeriesTask(DatasetArtifactTask):
    default_filename: ClassVar[str] = "series/manifest.json"

    id: str = Field(default="series")
    entrypoint: str = Field(default="core.dataset_series")
