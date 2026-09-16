from datetime import datetime, timedelta
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jerrythomas.config.constraints import NonEmptyString
from jerrythomas.utils.time import parse_datetime, parse_timecode


class _ResampleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FixedPeriodConfig(_ResampleModel):
    kind: Literal["fixed"]
    every: NonEmptyString

    @field_validator("every")
    @classmethod
    def validate_every(cls, value: str) -> str:
        if parse_timecode(value) <= timedelta():
            raise ValueError("resample period must be positive")
        return value


class CalendarPeriodConfig(_ResampleModel):
    kind: Literal["calendar"]
    unit: Literal["month"]
    timezone: NonEmptyString

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unknown calendar timezone: {value}") from exc
        return value


class ResampleAggregation(_ResampleModel):
    field: NonEmptyString
    statistic: Literal["first", "last", "sum", "mean", "min", "max", "count"]


class ResampleConfig(_ResampleModel):
    operation: Literal["resample"] = "resample"
    period: Annotated[
        FixedPeriodConfig | CalendarPeriodConfig, Field(discriminator="kind")
    ]
    aggregations: dict[NonEmptyString, ResampleAggregation] = Field(min_length=1)
    start: datetime | None = None
    end: datetime | None = None

    @field_validator("start", "end", mode="before")
    @classmethod
    def parse_bound(cls, value: object) -> object:
        return parse_datetime(value) if isinstance(value, str) else value

    @field_validator("start", "end")
    @classmethod
    def validate_bound(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("resample coverage bounds must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_resample(self) -> Self:
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("resample start must be before end")
        if any(name == "time" or name.startswith("_") for name in self.aggregations):
            raise ValueError(
                "resample outputs must not overwrite time or private fields"
            )
        return self
