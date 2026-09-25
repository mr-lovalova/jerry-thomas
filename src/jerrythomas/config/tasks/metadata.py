from pydantic import Field

from .base import ArtifactTask


class MetadataTask(ArtifactTask):
    id: str = Field(default="metadata")
    entrypoint: str = Field(default="core.dataset_metadata")
    output: str = Field(default="build/metadata.json", validation_alias="path")
