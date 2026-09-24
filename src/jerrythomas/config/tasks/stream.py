from typing import Literal, Self

from pydantic import Field, model_validator

from jerrythomas.config.constraints import NonEmptyString

from .base import RuntimeTask


class StreamTask(RuntimeTask):
    entrypoint: Literal["core.runtime.stream"] = Field(default="core.runtime.stream")
    stream: NonEmptyString

    @model_validator(mode="after")
    def reject_dataset(self) -> Self:
        if self.dataset is not None:
            raise ValueError("Stream operations must not bind a dataset")
        return self
