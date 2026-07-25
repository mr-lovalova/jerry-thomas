from datetime import timedelta
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from datapipeline.domain.series_id import SERIES_ID_SEPARATOR
from datapipeline.utils.time import parse_timecode


NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class SequenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    size: int = Field(gt=0, strict=True)
    stride: int = Field(default=1, gt=0, strict=True)


class SeriesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: NonEmptyString
    stream: NonEmptyString
    field: NonEmptyString
    scale: bool = Field(default=False, strict=True)
    sequence: SequenceConfig | None = None
    collect: int | None = Field(default=None, gt=0, strict=True)

    @field_validator("id")
    @classmethod
    def validate_id(cls, series_id: str) -> str:
        if SERIES_ID_SEPARATOR in series_id:
            raise ValueError(
                f"series id must not contain reserved separator {SERIES_ID_SEPARATOR!r}"
            )
        return series_id

    @model_validator(mode="after")
    def validate_shaping(self) -> Self:
        if self.sequence is not None and self.collect is not None:
            raise ValueError("sequence and collect are mutually exclusive")
        return self


class TargetSeriesConfig(SeriesConfig):
    horizon: NonEmptyString

    @field_validator("horizon")
    @classmethod
    def validate_horizon(cls, horizon: str) -> str:
        if parse_timecode(horizon) < timedelta():
            raise ValueError("target horizon must not be negative")
        return horizon

    @property
    def horizon_duration(self) -> timedelta:
        return parse_timecode(self.horizon)
