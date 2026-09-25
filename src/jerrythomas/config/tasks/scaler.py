from pydantic import Field

from .base import ArtifactTask


class ScalerTask(ArtifactTask):
    id: str = Field(default="scaler")
    entrypoint: str = Field(default="core.scaler")
    output: str = Field(default="build/scaler.json", validation_alias="path")
