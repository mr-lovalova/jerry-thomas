import math
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from datapipeline.utils.time import parse_datetime


Ratio = Annotated[float, Field(gt=0.0, le=1.0)]
FoldRole = Literal["train", "validation", "test"]
FOLD_ROLES: tuple[FoldRole, ...] = ("train", "validation", "test")


class DatasetFold(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str
    train: list[str] = Field(min_length=1)
    validation: list[str] = Field(default_factory=list)
    test: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def validate_id(cls, fold_id: str) -> str:
        if not fold_id.strip():
            raise ValueError("dataset fold id must not be empty")
        if fold_id != fold_id.strip():
            raise ValueError("dataset fold id must not contain outer whitespace")
        return fold_id

    @field_validator("train", "validation", "test")
    @classmethod
    def validate_role_labels(
        cls,
        labels: list[str],
        info: ValidationInfo,
    ) -> list[str]:
        for label in labels:
            if not label.strip():
                raise ValueError(
                    f"dataset fold {info.field_name} labels must not be empty"
                )
            if label != label.strip():
                raise ValueError(
                    f"dataset fold {info.field_name} labels must not contain "
                    "outer whitespace"
                )
        if len(labels) != len(set(labels)):
            raise ValueError(
                f"dataset fold {info.field_name} labels must not contain duplicates"
            )
        return labels

    @model_validator(mode="after")
    def validate_disjoint_roles(self) -> Self:
        train = set(self.train)
        validation = set(self.validation)
        test = set(self.test)
        shared = (train & validation) | (train & test) | (validation & test)
        if shared:
            raise ValueError(
                "dataset fold labels must belong to only one role: "
                + ", ".join(sorted(shared))
            )
        return self

    @property
    def role_labels(self) -> tuple[tuple[FoldRole, tuple[str, ...]], ...]:
        return (
            ("train", tuple(self.train)),
            ("validation", tuple(self.validation)),
            ("test", tuple(self.test)),
        )


class TimeInterval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str
    until: str | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, interval_id: str) -> str:
        if not interval_id.strip():
            raise ValueError("time interval id must not be empty")
        if interval_id != interval_id.strip():
            raise ValueError("time interval id must not contain outer whitespace")
        return interval_id

    @field_validator("until")
    @classmethod
    def validate_until(cls, until: str | None) -> str | None:
        if until is None:
            return None
        if not until.strip():
            raise ValueError("time interval until must not be empty")
        if until != until.strip():
            raise ValueError("time interval until must not contain outer whitespace")
        parse_datetime(until)
        return until


def _validate_fold_ids(folds: list[DatasetFold]) -> list[DatasetFold]:
    fold_ids = [fold.id for fold in folds]
    if len(fold_ids) != len(set(fold_ids)):
        raise ValueError("dataset fold ids must be unique")
    return folds


class HashSplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    mode: Literal["hash"] = "hash"
    ratios: dict[str, Ratio]
    folds: list[DatasetFold] = Field(min_length=1)
    seed: int = 42

    @field_validator("ratios")
    @classmethod
    def validate_ratios(cls, ratios: dict[str, float]) -> dict[str, float]:
        if not ratios:
            raise ValueError("hash split ratios must not be empty")
        for label in ratios:
            if not label.strip():
                raise ValueError("hash split labels must not be empty")
            if label != label.strip():
                raise ValueError("hash split labels must not contain outer whitespace")
        total = sum(ratios.values())
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"hash split ratios must sum to 1.0 (got {total})")
        return dict(sorted(ratios.items()))

    @field_validator("folds")
    @classmethod
    def validate_fold_ids(cls, folds: list[DatasetFold]) -> list[DatasetFold]:
        return _validate_fold_ids(folds)

    @model_validator(mode="after")
    def validate_fold_labels(self) -> Self:
        defined = set(self.ratios)
        for fold in self.folds:
            unknown = set((*fold.train, *fold.validation, *fold.test)) - defined
            if unknown:
                raise ValueError(
                    f"dataset fold {fold.id!r} references unknown hash split labels: "
                    + ", ".join(sorted(unknown))
                )
        return self


class TimeSplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    mode: Literal["time"] = "time"
    intervals: list[TimeInterval] = Field(min_length=1)
    folds: list[DatasetFold] = Field(min_length=1)

    @field_validator("intervals")
    @classmethod
    def validate_intervals(
        cls,
        intervals: list[TimeInterval],
    ) -> list[TimeInterval]:
        interval_ids = [interval.id for interval in intervals]
        if len(interval_ids) != len(set(interval_ids)):
            raise ValueError("time interval ids must be unique")

        boundaries = []
        for interval in intervals[:-1]:
            if interval.until is None:
                raise ValueError(
                    "every time interval except the final interval requires until"
                )
            boundaries.append(parse_datetime(interval.until))
        if intervals[-1].until is not None:
            raise ValueError("the final time interval must omit until")
        if any(
            previous >= current for previous, current in zip(boundaries, boundaries[1:])
        ):
            raise ValueError("time interval boundaries must be strictly increasing")
        return intervals

    @field_validator("folds")
    @classmethod
    def validate_fold_ids(cls, folds: list[DatasetFold]) -> list[DatasetFold]:
        return _validate_fold_ids(folds)

    @model_validator(mode="after")
    def validate_folds(self) -> Self:
        interval_positions = {
            interval.id: position for position, interval in enumerate(self.intervals)
        }
        for fold in self.folds:
            unknown = (
                set((*fold.train, *fold.validation, *fold.test))
                - interval_positions.keys()
            )
            if unknown:
                raise ValueError(
                    f"dataset fold {fold.id!r} references unknown time intervals: "
                    + ", ".join(sorted(unknown))
                )

            previous_positions: list[int] | None = None
            previous_role: FoldRole | None = None
            for role, labels in fold.role_labels:
                positions = [interval_positions[interval_id] for interval_id in labels]
                if not positions:
                    continue
                if previous_positions is not None and max(previous_positions) >= min(
                    positions
                ):
                    raise ValueError(
                        f"dataset fold {fold.id!r} requires {previous_role} intervals "
                        f"before {role} intervals"
                    )
                previous_positions = positions
                previous_role = role
        return self


SplitConfig = Annotated[
    HashSplitConfig | TimeSplitConfig,
    Field(discriminator="mode"),
]


def fold_output_id(fold_id: str, role: FoldRole) -> str:
    return f"{fold_id}.{role}"


def split_output_ids(config: SplitConfig) -> tuple[str, ...]:
    return tuple(
        fold_output_id(fold.id, role)
        for fold in config.folds
        for role, labels in fold.role_labels
        if labels
    )


def resolve_fold_output(
    config: SplitConfig,
    output_id: str,
) -> tuple[DatasetFold, FoldRole, tuple[str, ...]]:
    for fold in config.folds:
        for role, labels in fold.role_labels:
            if labels and output_id == fold_output_id(fold.id, role):
                return fold, role, labels
    raise KeyError(f"dataset fold output {output_id!r} is not defined")
