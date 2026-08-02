from typing import Literal

from pydantic import Field

from .base import RuntimeTask


class DatasetTask(RuntimeTask):
    entrypoint: Literal["core.runtime.dataset"] = Field(default="core.runtime.dataset")
