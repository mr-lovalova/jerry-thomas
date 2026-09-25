from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Collection, Mapping

from jerrythomas.config.dataset.dataset import DatasetConfig
from jerrythomas.config.project import ProjectConfig
from jerrythomas.config.streams import StreamsConfig
from jerrythomas.config.tasks.base import OutputTask
from jerrythomas.services.config_refs import (
    interpolate_config_vars,
    resolve_config_refs,
)

if TYPE_CHECKING:
    from jerrythomas.artifacts.planning import ArtifactGraph

# Resolved only once an operation binds a dataset; left as ${...} until then.
DATASET_VARIABLES = frozenset({"dataset_id", "dataset_version"})


@dataclass(frozen=True, slots=True)
class ProjectManifest:
    path: Path
    config: ProjectConfig
    variables: Mapping[str, Any]
    environment: Mapping[str, str]
    stream_dirs: tuple[Path, ...]
    source_dirs: tuple[Path, ...]
    dataset_dirs: tuple[Path, ...]
    artifacts_root: Path
    operations_dir: Path | None
    profiles_dir: Path

    def resolve_config(
        self,
        value: Any,
        *,
        variables: Mapping[str, Any] | None = None,
        deferred: Collection[str] = (),
    ) -> Any:
        value = resolve_config_refs(
            value,
            project_yaml=self.path,
            env=self.environment,
        )
        return interpolate_config_vars(
            value, {**self.variables, **(variables or {})}, deferred
        )


@dataclass(frozen=True, slots=True)
class ArtifactHashes:
    values: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def for_artifact(self, key: str) -> str:
        return self.values[key]


@dataclass(frozen=True, slots=True)
class ProjectDefinition:
    project: ProjectManifest
    datasets: Mapping[str, DatasetConfig]
    streams: StreamsConfig
    artifact_graph: ArtifactGraph
    output_operations: tuple[OutputTask, ...]
    artifact_hashes: ArtifactHashes

    def __post_init__(self) -> None:
        object.__setattr__(self, "datasets", MappingProxyType(dict(self.datasets)))

    def require_dataset(self, dataset_id: str) -> DatasetConfig:
        try:
            return self.datasets[dataset_id]
        except KeyError as exc:
            available = ", ".join(sorted(self.datasets)) or "(none)"
            raise ValueError(
                f"Unknown dataset '{dataset_id}'. Available datasets: {available}"
            ) from exc
