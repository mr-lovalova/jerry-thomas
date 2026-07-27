from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datapipeline.artifacts.registry import ArtifactRegistry
from datapipeline.config.dataset.dataset import DatasetConfig
from datapipeline.config.execution import ExecutionConfig
from datapipeline.config.transforms import PreprocessConfig, TransformConfig
from datapipeline.domain.stream import RecordStream

if TYPE_CHECKING:
    from datapipeline.execution.observer import PipelineObserver

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


@dataclass(frozen=True)
class BroadcastAsOfRuntimeStream:
    input_stream: str
    lookup_stream: str
    combine: RecordStage
    partition_by: tuple[str, ...]
    max_age: timedelta | None
    require_match: bool
    transforms: tuple[TransformConfig, ...]


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

RuntimeStream = SourceRuntimeStream | DerivedRuntimeStream | CombinedRuntimeStream


@dataclass
class Runtime:
    """Holds the active project state and prepared streams."""

    project_yaml: Path
    artifacts_root: Path
    dataset: DatasetConfig
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    streams: dict[str, RuntimeStream] = field(default_factory=dict)
    heartbeat_interval_seconds: float | None = None
    pipeline_observer: PipelineObserver | None = None
    observe_node_events: bool = True
    artifacts: ArtifactRegistry = field(init=False)

    def __post_init__(self) -> None:
        self.artifacts = ArtifactRegistry(self.artifacts_root)


def require_runtime_stream(runtime: Runtime, stream_id: str) -> RuntimeStream:
    try:
        return runtime.streams[stream_id]
    except KeyError as exc:
        available = ", ".join(sorted(runtime.streams)) or "(none)"
        raise KeyError(
            f"Unknown stream '{stream_id}'. Check dataset.yaml and stream ids. "
            f"Available streams: {available}"
        ) from exc
