from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

from jerrythomas.config.dataset.dataset import DatasetConfig
from jerrythomas.config.dataset.split import TimeSplitConfig

SCALER_STATISTICS = "scaler"
VECTOR_METADATA = "metadata"
COVERAGE_STATS = "coverage_stats"
SERIES = "series"


@dataclass(frozen=True)
class ArtifactDefinition:
    key: str
    dependencies: tuple[str, ...] = ()
    required_if: Callable[[DatasetConfig], bool] | None = None

    def is_required_for(self, dataset: DatasetConfig) -> bool:
        if self.required_if is None:
            return True
        return self.required_if(dataset)


def dataset_requires_scaler(dataset: DatasetConfig) -> bool:
    return any(config.scale is not None for config in dataset.series)


def scaler_fit_inputs(dataset: DatasetConfig) -> dict[str, object]:
    """Dataset settings that determine fitted moments; scaling policies excluded."""
    inputs: dict[str, object] = {
        "sample": dataset.sample.model_dump(mode="json", exclude={"window_mode"}),
        "split": (
            dataset.split.model_dump(mode="json") if dataset.split is not None else None
        ),
        "scaled_series": [
            config.model_dump(
                mode="json", exclude={"collect", "horizon", "sequence", "scale"}
            )
            for config in dataset.series
            if config.scale is not None
        ],
    }
    target_horizon = dataset.max_target_horizon
    if isinstance(dataset.split, TimeSplitConfig) and target_horizon > timedelta():
        inputs["target_horizon_seconds"] = int(target_horizon.total_seconds())
    return inputs


ARTIFACT_DEFINITIONS: tuple[ArtifactDefinition, ...] = (
    ArtifactDefinition(
        key=SERIES,
    ),
    # The series pass stores scaler fits, so the scaler rarely reads streams itself.
    ArtifactDefinition(
        key=SCALER_STATISTICS,
        dependencies=(SERIES,),
        required_if=dataset_requires_scaler,
    ),
    ArtifactDefinition(
        key=VECTOR_METADATA,
        dependencies=(SERIES,),
    ),
    ArtifactDefinition(
        key=COVERAGE_STATS,
        dependencies=(VECTOR_METADATA,),
    ),
)
