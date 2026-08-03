from datetime import datetime
from math import isfinite
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from jerrythomas.config.constraints import NonEmptyString
from jerrythomas.config.interpolation import is_missing_interpolation
from jerrythomas.utils.time import parse_cadence, parse_datetime, parse_timecode


PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class _TransformConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class WhereConfig(_TransformConfig):
    operation: Literal["where"] = "where"

    field: NonEmptyString
    operator: Literal["eq", "ne", "lt", "le", "gt", "ge", "in", "not_in"]
    comparand: Any

    @model_validator(mode="after")
    def validate_comparand(self) -> "WhereConfig":
        if is_missing_interpolation(self.comparand):
            raise ValueError("where comparand must resolve to a value")
        if self.operator in {"in", "not_in"}:
            if not isinstance(self.comparand, (list, tuple)):
                raise ValueError(
                    f"where operator {self.operator!r} requires a list or tuple"
                )
            values = self.comparand
        else:
            values = (self.comparand,)
        if self.field == "time":
            for value in values:
                if isinstance(value, datetime):
                    continue
                if not isinstance(value, str):
                    raise ValueError("where time comparands must be datetimes")
                parse_datetime(value)
        return self


class FloorTimeConfig(_TransformConfig):
    operation: Literal["floor_time"] = "floor_time"

    cadence: NonEmptyString

    @field_validator("cadence")
    @classmethod
    def validate_cadence(cls, cadence: str) -> str:
        parse_cadence(cadence)
        return cadence


class ShiftTimeConfig(_TransformConfig):
    operation: Literal["shift_time"] = "shift_time"

    by: NonEmptyString

    @field_validator("by")
    @classmethod
    def validate_shift(cls, by: str) -> str:
        if parse_timecode(by).total_seconds() == 0:
            raise ValueError("shift_time by must be non-zero")
        return by


class DedupeConfig(_TransformConfig):
    operation: Literal["dedupe"] = "dedupe"


class LagConfig(_TransformConfig):
    operation: Literal["lag"] = "lag"

    field: NonEmptyString
    periods: PositiveInt
    to: NonEmptyString | None = None


class LeadConfig(_TransformConfig):
    operation: Literal["lead"] = "lead"

    field: NonEmptyString
    periods: PositiveInt
    to: NonEmptyString | None = None


class ForwardSumConfig(_TransformConfig):
    operation: Literal["forward_sum"] = "forward_sum"

    field: NonEmptyString
    window: PositiveInt
    to: NonEmptyString


class EnsureCadenceConfig(_TransformConfig):
    operation: Literal["ensure_cadence"] = "ensure_cadence"

    cadence: NonEmptyString

    @field_validator("cadence")
    @classmethod
    def validate_cadence(cls, cadence: str) -> str:
        if parse_timecode(cadence).total_seconds() <= 0:
            raise ValueError("ensure_cadence cadence must be positive")
        return cadence


class EnsureScheduleConfig(_TransformConfig):
    operation: Literal["ensure_schedule"] = "ensure_schedule"

    schedule: NonEmptyString


class FillConfig(_TransformConfig):
    operation: Literal["fill"] = "fill"

    field: NonEmptyString
    window: PositiveInt
    statistic: Literal["mean", "median"]
    to: NonEmptyString | None = None
    min_samples: PositiveInt = 1

    @model_validator(mode="after")
    def validate_samples(self) -> "FillConfig":
        if self.min_samples > self.window:
            raise ValueError("fill min_samples cannot exceed window")
        return self


class ForwardFillConfig(_TransformConfig):
    operation: Literal["forward_fill"] = "forward_fill"

    field: NonEmptyString
    to: NonEmptyString | None = None


class CollapseConfig(_TransformConfig):
    operation: Literal["collapse"] = "collapse"

    keep: Literal["first", "last"]


class EwmMeanConfig(_TransformConfig):
    operation: Literal["ewm_mean"] = "ewm_mean"

    field: NonEmptyString
    alpha: Annotated[
        float,
        Field(strict=True, gt=0.0, le=1.0, allow_inf_nan=False),
    ]
    to: NonEmptyString | None = None
    min_samples: PositiveInt = 1


class RollingConfig(_TransformConfig):
    operation: Literal["rolling"] = "rolling"

    field: NonEmptyString
    window: PositiveInt
    to: NonEmptyString | None = None
    min_samples: PositiveInt | None = None
    statistic: Literal["mean", "median", "stdev", "pstdev", "max", "min"] = "mean"

    @model_validator(mode="after")
    def validate_samples(self) -> "RollingConfig":
        min_samples = self.window if self.min_samples is None else self.min_samples
        if min_samples > self.window:
            raise ValueError("rolling min_samples cannot exceed window")
        if self.statistic == "stdev" and min_samples < 2:
            raise ValueError(
                "rolling min_samples must be at least 2 for statistic='stdev'"
            )
        return self


class RollingQuantileConfig(_TransformConfig):
    operation: Literal["rolling_quantile"] = "rolling_quantile"

    field: NonEmptyString
    window: PositiveInt
    quantile: Annotated[
        float,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ]
    to: NonEmptyString | None = None
    min_samples: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_samples(self) -> "RollingQuantileConfig":
        min_samples = self.window if self.min_samples is None else self.min_samples
        if min_samples > self.window:
            raise ValueError("rolling_quantile min_samples cannot exceed window")
        return self


class RollingSlopeConfig(_TransformConfig):
    operation: Literal["rolling_slope"] = "rolling_slope"

    x: NonEmptyString
    y: NonEmptyString
    window: Annotated[int, Field(strict=True, ge=2)]
    to: NonEmptyString


class RollingOlsConfig(_TransformConfig):
    operation: Literal["rolling_ols"] = "rolling_ols"

    y: NonEmptyString
    x: tuple[NonEmptyString, ...] = Field(min_length=2)
    window: PositiveInt
    coefficient: NonEmptyString
    to: NonEmptyString

    @field_validator("x", mode="before")
    @classmethod
    def normalize_predictors(cls, fields: object) -> object:
        return tuple(fields) if isinstance(fields, list) else fields

    @model_validator(mode="after")
    def validate_model(self) -> "RollingOlsConfig":
        if len(self.x) != len(set(self.x)):
            raise ValueError("rolling_ols x must not contain duplicate fields")
        if self.y in self.x:
            raise ValueError("rolling_ols x must not contain the dependent field y")
        if self.coefficient not in self.x:
            raise ValueError("rolling_ols coefficient must name a field from x")
        if self.window < len(self.x) + 1:
            raise ValueError(
                "rolling_ols window must contain at least one more record "
                "than the number of predictors"
            )
        return self


class LogConfig(_TransformConfig):
    operation: Literal["log"] = "log"

    field: NonEmptyString
    to: NonEmptyString


class Log1pConfig(_TransformConfig):
    operation: Literal["log1p"] = "log1p"

    field: NonEmptyString
    to: NonEmptyString


class DeriveConfig(_TransformConfig):
    operation: Literal["derive"] = "derive"

    left: NonEmptyString
    operator: Literal["add", "sub", "mul", "div"]
    to: NonEmptyString
    right_field: NonEmptyString | None = None
    right_value: int | float | None = None

    @model_validator(mode="after")
    def validate_right_operand(self) -> "DeriveConfig":
        has_field = self.right_field is not None
        has_value = self.right_value is not None
        if has_field == has_value:
            raise ValueError(
                "derive requires exactly one of right_field or right_value"
            )
        if isinstance(self.right_value, float) and not isfinite(self.right_value):
            raise ValueError("derive right_value must be finite")
        return self


PreprocessConfig = Annotated[
    WhereConfig | FloorTimeConfig | ShiftTimeConfig,
    Field(discriminator="operation"),
]
TransformConfig = Annotated[
    WhereConfig
    | DedupeConfig
    | LagConfig
    | LeadConfig
    | ForwardSumConfig
    | EnsureCadenceConfig
    | EnsureScheduleConfig
    | FillConfig
    | ForwardFillConfig
    | CollapseConfig
    | EwmMeanConfig
    | RollingConfig
    | RollingQuantileConfig
    | RollingSlopeConfig
    | RollingOlsConfig
    | LogConfig
    | Log1pConfig
    | DeriveConfig,
    Field(discriminator="operation"),
]
