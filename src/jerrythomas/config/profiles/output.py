import codecs
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jerrythomas.config.options import OUTPUT_STDOUT_FORMATS, OUTPUT_VIEWS
from jerrythomas.io.compression import Compression

Transport = Literal["fs", "stdout"]
Format = Literal["csv", "jsonl", "parquet", "pickle", "txt", "html"]
View = Literal["flat", "raw"]


class ServeOutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transport: Transport
    format: Format
    view: View | None = Field(default=None)
    directory: Path | None = Field(default=None)
    filename: str | None = Field(default=None)
    encoding: str | None = Field(default=None)
    compression: Compression | None = None

    @field_validator("filename", mode="before")
    @classmethod
    def _normalize_filename(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if any(sep in text for sep in ("/", "\\")):
            raise ValueError("filename must not contain path separators")
        return text

    @field_validator("view", mode="before")
    @classmethod
    def _normalize_view(cls, value):
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text:
            return None
        if text not in set(OUTPUT_VIEWS):
            raise ValueError(
                f"view must be one of {', '.join(repr(x) for x in OUTPUT_VIEWS)}"
            )
        return text

    @field_validator("encoding", mode="before")
    @classmethod
    def _normalize_encoding(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @model_validator(mode="after")
    def _validate(self):
        if self.transport == "stdout":
            if self.directory is not None:
                raise ValueError("stdout cannot define a directory")
            if self.filename is not None:
                raise ValueError("stdout outputs do not support filenames")
            if self.encoding is not None:
                raise ValueError("stdout outputs do not support encoding")
            if self.compression is not None:
                raise ValueError("stdout outputs do not support compression")
            if self.format not in set(OUTPUT_STDOUT_FORMATS):
                raise ValueError(
                    f"stdout output supports {', '.join(repr(x) for x in OUTPUT_STDOUT_FORMATS)} formats"
                )
        elif self.directory is None:
            raise ValueError("fs outputs require a directory")
        if self.format == "csv" and self.view not in {None, "flat"}:
            raise ValueError("csv output supports only view='flat'")
        if self.format == "parquet" and self.view not in {None, "flat"}:
            raise ValueError("parquet output supports only view='flat'")
        if self.format == "pickle" and self.view not in {None, "raw"}:
            raise ValueError("pickle output supports only view='raw'")
        if self.compression is not None and self.format not in {"jsonl", "csv"}:
            raise ValueError("gzip compression supports only jsonl and csv output")
        if self.format in {"txt", "html"} and self.view is not None:
            raise ValueError(f"{self.format} output does not support view")
        if self.transport == "fs":
            if self.format in {"parquet", "pickle", "html"}:
                if self.encoding is not None:
                    raise ValueError(f"{self.format} output does not support encoding")
            elif self.format in {"jsonl", "csv", "txt"}:
                if self.encoding is None:
                    self.encoding = "utf-8"
                try:
                    codecs.lookup(self.encoding)
                except LookupError as exc:
                    raise ValueError("encoding must name a registered codec") from exc
        return self
