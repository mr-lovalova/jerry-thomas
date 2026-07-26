from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from datapipeline.artifacts.registry import (
    ArtifactRegistry,
    ArtifactSpec,
    ArtifactValue,
)
from datapipeline.runtime import Runtime

if TYPE_CHECKING:
    from datapipeline.execution.observer import PipelineObserver


@dataclass
class PipelineContext:
    """Lightweight runtime context shared across pipeline stages."""

    runtime: Runtime
    pipeline_observer: PipelineObserver | None = None
    heartbeat_interval_seconds: float | None = None
    observe_node_events: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.pipeline_observer is None:
            self.pipeline_observer = self.runtime.pipeline_observer
        self.observe_node_events = self.runtime.observe_node_events
        if self.heartbeat_interval_seconds is None:
            self.heartbeat_interval_seconds = self.runtime.heartbeat_interval_seconds

    @property
    def artifacts(self) -> ArtifactRegistry:
        return self.runtime.artifacts

    def artifact_metadata(self, key: str) -> Mapping[str, Any]:
        return self.artifacts.require(key).meta

    def resolve_artifact_path(self, key: str) -> Path:
        return self.artifacts.resolve_path(key)

    def require_artifact(self, spec: ArtifactSpec[ArtifactValue]) -> ArtifactValue:
        return self.artifacts.load(spec)
