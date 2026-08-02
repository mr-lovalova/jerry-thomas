from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from jerrythomas.config.constraints import NonEmptyString


class _PostprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CoverageConfig(_PostprocessConfig):
    threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    ids: list[NonEmptyString] | None = None

    @field_validator("ids")
    @classmethod
    def validate_ids(cls, ids: list[str] | None) -> list[str] | None:
        if ids is None:
            return None
        if not ids:
            raise ValueError("ids must not be empty")
        if len(ids) != len(set(ids)):
            raise ValueError("ids must not contain duplicates")
        return ids


class PostprocessConfig(_PostprocessConfig):
    """Row filters applied after feature and target vectors are assembled."""

    features: CoverageConfig | None = None
    targets: CoverageConfig | None = None
