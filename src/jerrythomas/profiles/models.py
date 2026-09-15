from dataclasses import dataclass, field
from typing import Literal, Sequence

from jerrythomas.artifacts.settings import BuildSettings
from jerrythomas.config.execution import ExecutionConfig
from jerrythomas.config.preview import PreviewStage
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
