from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from jerrythomas.artifacts.registry import ArtifactRecord, ArtifactRegistry
from jerrythomas.config.cross_section import CrossSectionOperation
from jerrythomas.config.dataset.dataset import DatasetConfig
from jerrythomas.config.execution import ExecutionConfig
from jerrythomas.config.streams import StreamsConfig
from jerrythomas.config.transforms import PreprocessConfig, TransformConfig
from jerrythomas.domain.stream import RecordStream

RecordStage = Callable[[Iterator[Any]], Iterable[Any]]


@dataclass(frozen=True)
class SourceRuntimeStream:
    source: RecordStream[Any]
    mapper: RecordStage
    preprocess: tuple[PreprocessConfig, ...]
    partition_by: tuple[str, ...]
    presorted: bool
    transforms: tuple[TransformConfig, ...]


@dataclass(frozen=True)
class DerivedRuntimeStream:
    input_stream: str
    partition_by: tuple[str, ...]
    transforms: tuple[TransformConfig, ...]


@dataclass(frozen=True)
class CrossSectionRuntimeStream:
    input_stream: str
    partition_by: tuple[str, ...]
    cross_section: tuple[CrossSectionOperation, ...]
    transforms: tuple[TransformConfig, ...]


@dataclass(frozen=True)
class BroadcastRuntimeStream:
    input_stream: str
    broadcast_stream: str
    combine: RecordStage
    partition_by: tuple[str, ...]
    transforms: tuple[TransformConfig, ...]


@dataclass(frozen=True)
class AsOfRuntimeStream:
    input_stream: str
    lookup_stream: str
    combine: RecordStage
    partition_by: tuple[str, ...]
    max_age: timedelta | None
    require_match: bool
    transforms: tuple[TransformConfig, ...]
    direction: str = "backward"


@dataclass(frozen=True)
class BroadcastAsOfRuntimeStream:
    input_stream: str
    lookup_stream: str
    combine: RecordStage
    partition_by: tuple[str, ...]
    max_age: timedelta | None
    require_match: bool
    transforms: tuple[TransformConfig, ...]
    direction: str = "backward"


@dataclass(frozen=True)
class AlignedRuntimeStream:
    inputs: tuple[str, ...]
    combine: RecordStage
    partition_by: tuple[str, ...]
    transforms: tuple[TransformConfig, ...]


CombinedRuntimeStream = (
    BroadcastRuntimeStream
    | AsOfRuntimeStream
    | BroadcastAsOfRuntimeStream
    | AlignedRuntimeStream
)

RuntimeStream = (
    SourceRuntimeStream
    | DerivedRuntimeStream
    | CrossSectionRuntimeStream
    | CombinedRuntimeStream
)


@dataclass(frozen=True)
class RuntimeSnapshot:
    """Resolved configuration and artifact references for an isolated worker."""

    project_yaml: Path
    artifacts_root: Path
    dataset: DatasetConfig | None
    execution: ExecutionConfig
    streams: StreamsConfig
    artifact_registry_root: Path
    artifact_registrations: dict[str, ArtifactRecord]
    heartbeat_interval_seconds: float | None
    observe_node_events: bool
    dataset_id: str | None = None
    artifact_aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class Runtime:
    """Holds the active project state and prepared streams."""

    project_yaml: Path
    artifacts_root: Path
    dataset: DatasetConfig | None = None
    dataset_id: str | None = None
    artifact_aliases: dict[str, str] = field(default_factory=dict)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    streams: dict[str, RuntimeStream] = field(default_factory=dict)
    heartbeat_interval_seconds: float | None = None
    observe_node_events: bool = True
    artifacts: ArtifactRegistry = field(init=False)
    _stream_configs: StreamsConfig | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.artifacts = ArtifactRegistry(
            self.artifacts_root, aliases=self.artifact_aliases
        )

    def require_dataset(self) -> DatasetConfig:
        if self.dataset is None:
            raise ValueError("This operation requires a selected dataset.")
        return self.dataset


def require_runtime_stream(runtime: Runtime, stream_id: str) -> RuntimeStream:
    try:
        return runtime.streams[stream_id]
    except KeyError as exc:
        available = ", ".join(sorted(runtime.streams)) or "(none)"
        raise KeyError(
            f"Unknown stream '{stream_id}'. Check the stream catalog. "
            f"Available streams: {available}"
        ) from exc
