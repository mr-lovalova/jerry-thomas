from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jerrythomas.config.constraints import DottedIdentifier, NonEmptyString
from jerrythomas.config.cross_section import CrossSectionOperation
from jerrythomas.config.sources import EntryPointConfig, SourceConfig
from jerrythomas.config.transforms import PreprocessConfig, TransformConfig
from jerrythomas.utils.time import parse_timecode


_StreamId = DottedIdentifier
_FieldName = NonEmptyString
_Timecode = NonEmptyString


class SourceRefConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: _StreamId


class StreamRefConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stream: _StreamId


class BroadcastJoin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["broadcast"]
    with_: _StreamId = Field(alias="with")

    def partner_stream_ids(self) -> tuple[str, ...]:
        return (self.with_,)


class _LookupJoin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_age: _Timecode | None = None
    require_match: bool = Field(default=True, strict=True)
    direction: Literal["backward", "forward"] = "backward"

    @field_validator("max_age")
    @classmethod
    def validate_max_age(cls, max_age: str | None) -> str | None:
        if max_age is not None and parse_timecode(max_age).total_seconds() < 0:
            raise ValueError("max_age must be non-negative")
        return max_age


class AsOfJoin(_LookupJoin):
    kind: Literal["as_of"]
    lookup: _StreamId

    def partner_stream_ids(self) -> tuple[str, ...]:
        return (self.lookup,)


class BroadcastAsOfJoin(_LookupJoin):
    kind: Literal["broadcast_as_of"]
    lookup: _StreamId

    def partner_stream_ids(self) -> tuple[str, ...]:
        return (self.lookup,)


class AlignJoin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["align"]
    streams: tuple[_StreamId, ...] = Field(min_length=1)

    @field_validator("streams")
    @classmethod
    def validate_streams(cls, streams: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(streams)) != len(streams):
            raise ValueError("align join must not contain duplicate stream ids")
        return streams

    def partner_stream_ids(self) -> tuple[str, ...]:
        return self.streams


CombinedJoin: TypeAlias = Annotated[
    Annotated[BroadcastJoin, Tag("broadcast")]
    | Annotated[AsOfJoin, Tag("as_of")]
    | Annotated[BroadcastAsOfJoin, Tag("broadcast_as_of")]
    | Annotated[AlignJoin, Tag("align")],
    Discriminator(
        "kind",
        custom_error_type="join_config_type",
        custom_error_message=(
            "Stream 'join.kind' must be one of broadcast, as_of, "
            "broadcast_as_of, or align"
        ),
    ),
]


class _StreamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: _StreamId
    transforms: list[TransformConfig] = Field(default_factory=list)


class SourceStreamConfig(_StreamConfig):
    from_: SourceRefConfig = Field(alias="from")
    map: EntryPointConfig
    preprocess: list[PreprocessConfig] = Field(default_factory=list)
    partition_by: tuple[_FieldName, ...] = ()
    presorted: bool = Field(default=False, strict=True)

    @model_validator(mode="before")
    @classmethod
    def reject_ordered_by(cls, value: object) -> object:
        if isinstance(value, dict) and "ordered_by" in value:
            raise PydanticCustomError(
                "removed_ordered_by",
                "ordered_by was removed in v11; use presorted: true to assert "
                "records are ordered by partition_by fields, then time",
            )
        return value

    @field_validator("partition_by")
    @classmethod
    def validate_partition_by(cls, fields: tuple[str, ...]) -> tuple[str, ...]:
        if len(fields) != len(set(fields)):
            raise ValueError("partition_by must not contain duplicate fields")
        if "time" in fields:
            raise ValueError("partition_by must not contain the reserved field 'time'")
        return fields

    def input_streams(self) -> tuple[str, ...]:
        return ()


class DerivedStreamConfig(_StreamConfig):
    from_: StreamRefConfig = Field(alias="from")
    transforms: list[TransformConfig] = Field(min_length=1)

    def input_streams(self) -> tuple[str, ...]:
        return (self.from_.stream,)


class CrossSectionStreamConfig(_StreamConfig):
    from_: StreamRefConfig = Field(alias="from")
    cross_section: list[CrossSectionOperation] = Field(min_length=1)

    def input_streams(self) -> tuple[str, ...]:
        return (self.from_.stream,)


class CombinedStreamConfig(_StreamConfig):
    from_: StreamRefConfig = Field(alias="from")
    join: CombinedJoin
    combine: EntryPointConfig

    @model_validator(mode="after")
    def validate_distinct_inputs(self) -> "CombinedStreamConfig":
        primary = self.from_.stream
        for partner in self.join.partner_stream_ids():
            if partner == primary:
                raise ValueError(
                    f"join input '{partner}' must differ from the primary "
                    f"stream '{primary}'"
                )
        return self

    def input_streams(self) -> tuple[str, ...]:
        return (self.from_.stream, *self.join.partner_stream_ids())


def _stream_config_tag(value: object) -> str | None:
    if isinstance(value, _StreamConfig):
        value = value.model_dump(by_alias=True)
    if not isinstance(value, dict):
        return None
    if "join" in value:
        return "combined"
    from_ = value.get("from")
    if not isinstance(from_, dict):
        return None
    if "source" in from_:
        return "source"
    if "cross_section" in value:
        return "cross_section"
    return "derived"


StreamConfig: TypeAlias = Annotated[
    Annotated[SourceStreamConfig, Tag("source")]
    | Annotated[DerivedStreamConfig, Tag("derived")]
    | Annotated[CrossSectionStreamConfig, Tag("cross_section")]
    | Annotated[CombinedStreamConfig, Tag("combined")],
    Discriminator(
        _stream_config_tag,
        custom_error_type="stream_config_type",
        custom_error_message=(
            "Stream 'from' must use source, stream+transforms, "
            "stream+cross_section, or stream+join"
        ),
    ),
]


class StreamsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: dict[str, SourceConfig] = Field(default_factory=dict)
    streams: dict[str, StreamConfig] = Field(default_factory=dict)

    @field_validator("sources")
    @classmethod
    def validate_source_keys(
        cls,
        sources: dict[str, SourceConfig],
    ) -> dict[str, SourceConfig]:
        for source_id, source in sources.items():
            if source_id != source.id:
                raise ValueError(
                    f"Source registry key {source_id!r} does not match "
                    f"source id {source.id!r}"
                )
        return sources

    @field_validator("streams")
    @classmethod
    def validate_stream_keys(
        cls,
        streams: dict[str, StreamConfig],
    ) -> dict[str, StreamConfig]:
        for stream_id, stream in streams.items():
            if stream_id != stream.id:
                raise ValueError(
                    f"Stream registry key {stream_id!r} does not match "
                    f"stream id {stream.id!r}"
                )
        return streams
