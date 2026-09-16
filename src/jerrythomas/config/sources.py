import codecs
from typing import Annotated, Any, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from jerrythomas.config.constraints import DottedIdentifier, NonEmptyString
from jerrythomas.config.interpolation import PluginArgs
from jerrythomas.io.compression import Compression


_EntryPoint = NonEmptyString
_SourceId = DottedIdentifier

_CsvDelimiter = Annotated[
    str,
    StringConstraints(min_length=1, max_length=1, pattern=r'[^\r\n"]'),
]

_CsvErrorPrefix = Annotated[
    str,
    StringConstraints(min_length=1, pattern=r".*\S.*"),
]


class EntryPointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entrypoint: _EntryPoint
    args: PluginArgs = Field(default_factory=dict)


class _TextReaderBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoding: NonEmptyString = "utf-8"

    @field_validator("encoding")
    @classmethod
    def validate_encoding(cls, encoding: str) -> str:
        try:
            decoded = codecs.getincrementaldecoder(encoding)().decode(b"", final=True)
        except (LookupError, TypeError, ValueError) as exc:
            raise ValueError(f"unsupported text encoding {encoding!r}") from exc
        if not isinstance(decoded, str):
            raise ValueError(f"unsupported text encoding {encoding!r}")
        return encoding


class CsvReaderConfig(_TextReaderBase):
    format: Literal["csv"]
    delimiter: _CsvDelimiter = ";"
    error_prefixes: tuple[_CsvErrorPrefix, ...] = ()


class JsonReaderConfig(_TextReaderBase):
    format: Literal["json"]
    array_field: NonEmptyString | None = None


class JsonLinesReaderConfig(_TextReaderBase):
    format: Literal["jsonl"]


class ParquetReaderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["parquet"]


TextReaderConfig: TypeAlias = Annotated[
    CsvReaderConfig | JsonReaderConfig | JsonLinesReaderConfig,
    Field(discriminator="format"),
]

SourceReaderConfig: TypeAlias = Annotated[
    CsvReaderConfig | JsonReaderConfig | JsonLinesReaderConfig | ParquetReaderConfig,
    Field(discriminator="format"),
]


class FsLoaderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transport: Literal["fs"]
    path: NonEmptyString
    reader: SourceReaderConfig
    compression: Compression | None = None

    @model_validator(mode="after")
    def validate_compression(self) -> Self:
        if isinstance(self.reader, ParquetReaderConfig):
            if self.compression is not None:
                raise ValueError("parquet input does not support external compression")
            return self
        if self.compression == "gzip" and self.reader.format not in {
            "csv",
            "jsonl",
        }:
            raise ValueError(
                "gzip compression is supported only for csv and jsonl formats"
            )
        return self


class HttpLoaderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transport: Literal["http"]
    url: NonEmptyString
    reader: TextReaderConfig
    headers: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: (
        Annotated[
            float,
            Field(strict=True, gt=0, allow_inf_nan=False),
        ]
        | None
    ) = None


BuiltInLoaderConfig: TypeAlias = Annotated[
    FsLoaderConfig | HttpLoaderConfig,
    Field(discriminator="transport"),
]


class SourceInputsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: tuple[NonEmptyString, ...] = Field(min_length=1)

    @field_validator("files")
    @classmethod
    def normalize_files(cls, files: tuple[str, ...]) -> tuple[str, ...]:
        if len(files) != len(set(files)):
            raise ValueError("source input files must not contain duplicates")
        return tuple(sorted(files))


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: _SourceId
    parser: EntryPointConfig
    loader: BuiltInLoaderConfig | EntryPointConfig
    inputs: SourceInputsConfig | None = None
    freshness: Literal["tracked", "opaque"] = "tracked"

    @model_validator(mode="after")
    def validate_input_freshness(self) -> Self:
        if isinstance(self.loader, FsLoaderConfig):
            if self.freshness == "opaque":
                raise ValueError(
                    f"Source '{self.id}' reads the filesystem, which Jerry "
                    "always tracks for artifact freshness; remove "
                    "'freshness: opaque'."
                )
            return self
        if self.inputs is not None or self.freshness == "opaque":
            return self
        kind = (
            "an HTTP loader"
            if isinstance(self.loader, HttpLoaderConfig)
            else "a custom loader"
        )
        raise ValueError(
            f"Source '{self.id}' uses {kind}, whose inputs Jerry cannot "
            "see. Declare them under 'inputs.files' so artifacts track "
            "freshness, or set 'freshness: opaque' to acknowledge that "
            "changes to these inputs will not invalidate artifacts."
        )
