from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from jerrythomas.config.dataset.series import ScalingConfig
from jerrythomas.io.json_file import write_json_object


SCALER_ARTIFACT_VERSION: Final = 5


class ScalerStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    mean: float = Field(allow_inf_nan=False)
    std: float = Field(gt=0, allow_inf_nan=False)
    count: int = Field(gt=0, strict=True)


class PositionalScalerStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    positions: tuple[ScalerStatistics, ...] = Field(min_length=1)

    @property
    def count(self) -> int:
        return sum(position.count for position in self.positions)


class FittedScaler(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    settings: ScalingConfig
    statistics: ScalerStatistics | PositionalScalerStatistics


class StandardScalerArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["standard_scaler"] = "standard_scaler"
    version: Literal[5] = SCALER_ARTIFACT_VERSION
    observations: int = Field(gt=0, strict=True)
    scalers: dict[str, FittedScaler] = Field(min_length=1)

    @field_validator("scalers")
    @classmethod
    def _validate_vector_ids(
        cls,
        scalers: dict[str, FittedScaler],
    ) -> dict[str, FittedScaler]:
        for vector_id in scalers:
            if not vector_id.strip():
                raise ValueError("scaler vector ids must not be empty")
            if vector_id != vector_id.strip():
                raise ValueError("scaler vector ids must not contain outer whitespace")
        return scalers

    @model_validator(mode="after")
    def _validate_observation_count(self) -> Self:
        observed = sum(scaler.statistics.count for scaler in self.scalers.values())
        if self.observations != observed:
            raise ValueError(
                "scaler observations must equal the sum of series statistic counts"
            )
        return self


class FoldedScalerArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["folded_scaler"] = "folded_scaler"
    version: Literal[5] = SCALER_ARTIFACT_VERSION
    folds: dict[str, StandardScalerArtifact] = Field(min_length=1)

    @field_validator("folds")
    @classmethod
    def _validate_fold_ids(
        cls,
        folds: dict[str, StandardScalerArtifact],
    ) -> dict[str, StandardScalerArtifact]:
        for fold_id in folds:
            if not fold_id.strip():
                raise ValueError("scaler fold ids must not be empty")
            if fold_id != fold_id.strip():
                raise ValueError("scaler fold ids must not contain outer whitespace")
        return folds

    def for_fold(self, fold_id: str) -> StandardScalerArtifact:
        try:
            return self.folds[fold_id]
        except KeyError as exc:
            raise KeyError(f"Scaler artifact has no fold {fold_id!r}.") from exc


ScalerArtifact = Annotated[
    StandardScalerArtifact | FoldedScalerArtifact,
    Field(discriminator="kind"),
]
_SCALER_ARTIFACT: TypeAdapter[ScalerArtifact] = TypeAdapter(ScalerArtifact)


def load_scaler_artifact(path: Path) -> ScalerArtifact:
    return _SCALER_ARTIFACT.validate_json(path.read_text(encoding="utf-8"))


def save_scaler_artifact(path: Path, artifact: ScalerArtifact) -> None:
    write_json_object(path, artifact.model_dump(mode="json"))
