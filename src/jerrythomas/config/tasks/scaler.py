from typing import ClassVar

from pydantic import Field

from .base import DatasetArtifactTask


class ScalerTask(DatasetArtifactTask):
    default_filename: ClassVar[str] = "scaler.json"

    id: str = Field(default="scaler")
    entrypoint: str = Field(default="core.scaler")
