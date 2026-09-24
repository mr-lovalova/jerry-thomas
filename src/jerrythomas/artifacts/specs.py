from collections.abc import Callable
from dataclasses import dataclass

from jerrythomas.config.dataset.dataset import DatasetConfig

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


ARTIFACT_DEFINITIONS: tuple[ArtifactDefinition, ...] = (
    ArtifactDefinition(
        key=SCALER_STATISTICS,
        required_if=dataset_requires_scaler,
    ),
    ArtifactDefinition(
        key=SERIES,
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
