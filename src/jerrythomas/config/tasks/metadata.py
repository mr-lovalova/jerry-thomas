from typing import ClassVar

from pydantic import Field

from .base import DatasetArtifactTask


class MetadataTask(DatasetArtifactTask):
    default_filename: ClassVar[str] = "metadata.json"

    id: str = Field(default="metadata")
    entrypoint: str = Field(default="core.dataset_metadata")
