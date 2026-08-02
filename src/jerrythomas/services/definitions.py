from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from jerrythomas.config.dataset.dataset import DatasetConfig
from jerrythomas.config.project import ProjectConfig
from jerrythomas.config.streams import StreamsConfig
from jerrythomas.config.tasks.base import RuntimeTask
from jerrythomas.services.config_refs import (
    interpolate_config_vars,
    resolve_config_refs,
)

if TYPE_CHECKING:
    from jerrythomas.artifacts.planning import ArtifactGraph


@dataclass(frozen=True, slots=True)
class ProjectManifest:
    path: Path
    config: ProjectConfig
    variables: Mapping[str, Any]
    environment: Mapping[str, str]
    stream_dirs: tuple[Path, ...]
    source_dirs: tuple[Path, ...]
    dataset_path: Path
    artifacts_root: Path
    operations_dir: Path | None
    profiles_dir: Path

    def resolve_config(self, value: Any) -> Any:
        value = resolve_config_refs(
            value,
            project_yaml=self.path,
            env=self.environment,
        )
        return interpolate_config_vars(value, self.variables)


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
    dataset: DatasetConfig
    streams: StreamsConfig
    artifact_graph: ArtifactGraph
    runtime_operations: tuple[RuntimeTask, ...]
    artifact_hashes: ArtifactHashes
