from dataclasses import dataclass, field
from typing import Literal, Sequence

from jerrythomas.artifacts.settings import BuildSettings
from jerrythomas.config.execution import ExecutionConfig
from jerrythomas.config.preview import PreviewStage
from jerrythomas.config.tasks.base import ArtifactTask, OutputTask
from jerrythomas.config.tasks.stream import StreamTask
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
    dataset_id: str | None


@dataclass(frozen=True)
class BuildJob:
    task: ArtifactTask
    settings: BuildSettings


@dataclass(frozen=True)
class RuntimeJob:
    name: str
    task: OutputTask
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
class ExportJob:
    name: str
    task: StreamTask
    output: OutputTarget
    overwrite: bool
    observability: ObservabilitySettings

    @property
    def stream(self) -> str:
        return self.task.stream


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
class ExportRunRequest:
    definition: ProjectDefinition
    jobs: Sequence[ExportJob]
    execution: ExecutionConfig
    artifact_settings: BuildSettings
    runtime: Runtime
    command: Literal["export"] = field(default="export", init=False)


ProfileRunRequest = BuildRunRequest | RuntimeRunRequest | ExportRunRequest
