from pathlib import Path
from typing import Literal

from pydantic import StrictBool, field_validator

from .base import OperationProfile


class ExportProfile(OperationProfile):
    cmd: Literal["export"]
    output: Path
    overwrite: StrictBool = False

    @field_validator("output", mode="before")
    @classmethod
    def _normalize_output(cls, value: object) -> Path:
        output = str(value).strip() if value is not None else ""
        if not output:
            raise ValueError("output must be set")
        if not output.endswith((".jsonl", ".jsonl.gz")):
            raise ValueError("output must use a .jsonl or .jsonl.gz path")
        return Path(output)
