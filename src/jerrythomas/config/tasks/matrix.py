from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from .base import RuntimeTask


class MatrixOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal["assembled", "postprocessed"] = "postprocessed"
    max_cells: StrictInt = Field(default=1_000_000, gt=0)


class MatrixTask(RuntimeTask):
    entrypoint: Literal["core.availability_matrix"] = Field(
        default="core.availability_matrix"
    )
    options: MatrixOptions = Field(default_factory=MatrixOptions)
