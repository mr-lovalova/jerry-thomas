from typing import Literal

from pydantic import Field

from .base import RuntimeTask


class DatasetTask(RuntimeTask):
    entrypoint: Literal["core.dataset"] = Field(default="core.dataset")
