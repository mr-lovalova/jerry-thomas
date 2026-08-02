from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from jerrythomas.config.constraints import NonEmptyString


class _CrossSectionOperationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RankScoreConfig(_CrossSectionOperationConfig):
    operation: Literal["rank_score"] = "rank_score"

    field: NonEmptyString
    to: NonEmptyString
    min_samples: Annotated[int, Field(strict=True, ge=2)]


class OlsResidualConfig(_CrossSectionOperationConfig):
    operation: Literal["ols_residual"] = "ols_residual"

    y: NonEmptyString
    x: tuple[NonEmptyString, ...] = Field(min_length=1)
    to: NonEmptyString
    min_samples: Annotated[int, Field(strict=True, gt=0)]

    @field_validator("x", mode="before")
    @classmethod
    def normalize_predictors(cls, fields: object) -> object:
        return tuple(fields) if isinstance(fields, list) else fields

    @model_validator(mode="after")
    def validate_model(self) -> "OlsResidualConfig":
        if len(self.x) != len(set(self.x)):
            raise ValueError("ols_residual x must not contain duplicate fields")
        if self.y in self.x:
            raise ValueError("ols_residual x must not contain the dependent field y")
        if self.min_samples < len(self.x) + 1:
            raise ValueError(
                "ols_residual min_samples must contain at least one more record "
                "than the number of predictors"
            )
        return self


CrossSectionOperation = Annotated[
    RankScoreConfig | OlsResidualConfig,
    Field(discriminator="operation"),
]
