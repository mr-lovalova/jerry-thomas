from typing import Literal

from pydantic import Field

from .base import ArtifactTask


class MetadataTask(ArtifactTask):
    id: Literal["metadata"] = Field(default="metadata")
    entrypoint: str = Field(default="core.artifact.metadata")
    output: str = Field(default="build/metadata.json")
