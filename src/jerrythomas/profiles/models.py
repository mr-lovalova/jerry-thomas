from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

from jerrythomas.artifacts.settings import BuildSettings
from jerrythomas.config.execution import ExecutionConfig
from jerrythomas.config.preview import PreviewStage
from jerrythomas.config.profiles.output import Format, View
from jerrythomas.io.compression import Compression
from jerrythomas.config.tasks.base import ArtifactTask, RuntimeTask
from jerrythomas.execution.settings import (
    ObservabilitySettings,
)
from jerrythomas.io.output import OutputTarget
from jerrythomas.io.runs import RunPaths
from jerrythomas.runtime import Runtime
from jerrythomas.services.definitions import ProjectDefinition


@dataclass(frozen=True)
class ServeRunPlan:
    paths: RunPaths
    preview: PreviewStage | None


@dataclass(frozen=True)
class ServeRunResult:
    """A successfully published serve run and its completed output files."""

    paths: RunPaths
    preview: PreviewStage | None
    outputs: tuple[Path, ...]


@dataclass(frozen=True)
class MaterializedOutput:
    """A completed stream export at its configured destination."""

    profile: str
    stream: str
    path: Path
    format: Format
    view: View
    encoding: str | None
    compression: Compression | None
    row_count: int


@dataclass(frozen=True)
class MaterializeRunResult:
    """One successful materialize invocation, including all selected profiles."""

    run_id: str
    started_at: str
    finished_at: str
    outputs: tuple[MaterializedOutput, ...]


ProfileRunResult = ServeRunResult | MaterializeRunResult


@dataclass(frozen=True)
class BuildJob:
    task: ArtifactTask
    settings: BuildSettings


@dataclass(frozen=True)
class RuntimeJob:
    name: str
    task: RuntimeTask
    runtime: Runtime
    output: OutputTarget
    observability: ObservabilitySettings
    limit: int | None
    throttle_ms: float | None
    preview: PreviewStage | None
    output_ids: tuple[str, ...]

    @property
    def configured_outputs(self) -> tuple[OutputTarget, ...]:
        if self.output_ids:
            return tuple(
                self.output.for_output(output_id) for output_id in self.output_ids
            )
        return (self.output,)


@dataclass(frozen=True)
class MaterializeJob:
    name: str
    stream: str
    output: OutputTarget
    overwrite: bool
    observability: ObservabilitySettings


@dataclass(frozen=True, kw_only=True)
class BuildRunRequest:
    definition: ProjectDefinition
    jobs: Sequence[BuildJob]
    execution: ExecutionConfig
    command: Literal["build"] = field(default="build", init=False)


@dataclass(frozen=True, kw_only=True)
class RuntimeRunRequest:
    command: Literal["serve", "inspect"]
    definition: ProjectDefinition
    jobs: Sequence[RuntimeJob]
    execution: ExecutionConfig
    artifact_settings: BuildSettings
    serve_run_plans: tuple[ServeRunPlan, ...] = ()


@dataclass(frozen=True, kw_only=True)
class MaterializeRunRequest:
    definition: ProjectDefinition
    jobs: Sequence[MaterializeJob]
    execution: ExecutionConfig
    artifact_settings: BuildSettings
    runtime: Runtime
    command: Literal["materialize"] = field(default="materialize", init=False)


ProfileRunRequest = BuildRunRequest | RuntimeRunRequest | MaterializeRunRequest
