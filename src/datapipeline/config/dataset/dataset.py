from datetime import timedelta
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from datapipeline.config.constraints import NonEmptyString
from datapipeline.config.dataset.series import SeriesConfig, TargetSeriesConfig
from datapipeline.config.dataset.postprocess import PostprocessConfig
from datapipeline.config.dataset.split import HashSplitConfig, SplitConfig
from datapipeline.domain.sample import WindowMode
from datapipeline.utils.time import CADENCE_PATTERN


class SampleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    cadence: str = Field(..., pattern=CADENCE_PATTERN)
    keys: list[NonEmptyString] = Field(default_factory=list)
    window_mode: WindowMode = "intersection"

    @field_validator("keys")
    @classmethod
    def validate_keys(cls, keys: list[str]) -> list[str]:
        if any(not key.strip() for key in keys):
            raise ValueError("sample keys must not be empty")
        if len(keys) != len(set(keys)):
            raise ValueError("sample keys must not contain duplicates")
        return keys


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sample: SampleConfig
    features: list[SeriesConfig] = Field(default_factory=list)
    targets: list[TargetSeriesConfig] = Field(default_factory=list)
    split: SplitConfig | None = None
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)

    @property
    def series(self) -> tuple[SeriesConfig, ...]:
        return (*self.features, *self.targets)

    @property
    def max_target_horizon(self) -> timedelta:
        return max(
            (target.horizon_duration for target in self.targets),
            default=timedelta(),
        )

    @model_validator(mode="after")
    def validate_series(self) -> Self:
        if self.targets and not self.features:
            raise ValueError("datasets with targets must define at least one feature")
        if not self.targets and self.postprocess.targets is not None:
            raise ValueError("postprocess.targets requires at least one dataset target")
        seen: set[str] = set()
        for config in self.series:
            if config.id in seen:
                raise ValueError(
                    f"dataset series id {config.id!r} must be unique across "
                    "features and targets"
                )
            seen.add(config.id)

        future_target_fields = {
            (target.stream, target.field)
            for target in self.targets
            if target.horizon_duration > timedelta()
        }
        for feature in self.features:
            if (feature.stream, feature.field) in future_target_fields:
                raise ValueError(
                    f"feature {feature.id!r} cannot select a field declared as a "
                    "future target"
                )
        return self

    @model_validator(mode="after")
    def validate_hash_split_temporal_dependencies(self) -> Self:
        if not isinstance(self.split, HashSplitConfig):
            return self
        sequenced = [config.id for config in self.series if config.sequence is not None]
        if sequenced:
            raise ValueError(
                "hash splits cannot be used with sequenced features or targets: "
                + ", ".join(sequenced)
            )
        future_targets = [
            target.id
            for target in self.targets
            if target.horizon_duration > timedelta()
        ]
        if future_targets:
            raise ValueError(
                "hash splits cannot be used with positive target horizons: "
                + ", ".join(future_targets)
            )
        return self
