from typing import Literal

from pydantic import Field

from .base import ArtifactTask


class ScalerTask(ArtifactTask):
    id: Literal["scaler"] = Field(default="scaler")
    entrypoint: str = Field(default="core.artifact.scaler")
    output: str = Field(default="build/scaler.json")
    with_mean: bool = Field(default=True, strict=True)
    with_std: bool = Field(default=True, strict=True)
    epsilon: float = Field(
        default=1e-12,
        gt=0,
        allow_inf_nan=False,
        strict=True,
    )
