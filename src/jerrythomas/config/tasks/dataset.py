from typing import Literal

from pydantic import Field

from .base import OutputTask


class DatasetTask(OutputTask):
    entrypoint: Literal["core.dataset"] = Field(default="core.dataset")
