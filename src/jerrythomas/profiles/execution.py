import json
import logging
from dataclasses import dataclass
from typing import Literal

from jerrythomas.artifacts.errors import ArtifactResolutionError
from jerrythomas.artifacts.hydration import hydrate_runtime_artifacts_for_pipeline
from jerrythomas.artifacts.validation import validate_artifact_plan
from jerrythomas.config.tasks.base import ArtifactTask
from jerrythomas.config.tasks.coverage import CoverageTask
from jerrythomas.config.tasks.dataset import DatasetTask
from jerrythomas.execution.observability import (
    emit_execution_message,
    operation_scope,
)
from jerrythomas.operations.persistence import (
    WrittenOutput,
    persist_runtime_result,
)
from jerrythomas.operations.outputs.execution import OutputOptions, run_output_operation
from jerrythomas.services.definitions import ProjectDefinition

from .models import ExportJob, RuntimeJob

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeJobPlan:
    job: RuntimeJob
    required_artifacts: tuple[str, ...]


@dataclass(frozen=True)
class ExportJobPlan:
    job: ExportJob
    required_artifacts: tuple[str, ...]


def plan_export_job(
    job: ExportJob,
    definition: ProjectDefinition,
) -> ExportJobPlan:
    graph = definition.artifact_graph
    return ExportJobPlan(
        job=job,
        required_artifacts=graph.runtime_dependency_closure(job.task, preview=None),
    )


def validate_build_job(
    task: ArtifactTask,
    definition: ProjectDefinition,
) -> None:
    graph = definition.artifact_graph
    roots = {task.id}
    artifact_keys = set(graph.dependency_closure(roots))
    validate_artifact_plan(definition.streams, graph, artifact_keys)


def plan_runtime_job(
    job: RuntimeJob,
    definition: ProjectDefinition,
) -> RuntimeJobPlan:
    graph = definition.artifact_graph
    if job.output.format == "parquet" and not isinstance(job.task, DatasetTask):
        raise ValueError("Parquet output is supported only by the dataset operation.")
    if job.output.format == "parquet" and job.preview not in {
        None,
        "samples",
        "postprocess",
    }:
        raise ValueError(
            "Parquet dataset previews support only 'samples' and 'postprocess'."
        )
    if isinstance(job.task, CoverageTask) and job.limit is not None:
        raise ValueError("The coverage operation does not support a record limit.")
    if not isinstance(job.task, DatasetTask) and job.preview is not None:
        raise ValueError("Only the dataset operation supports preview.")
    if not isinstance(job.task, DatasetTask) and job.throttle_ms is not None:
        raise ValueError("Only the dataset operation supports throttle_ms.")
    if not isinstance(job.task, DatasetTask) and job.output_ids:
        raise ValueError("Only the dataset operation supports routed outputs.")
    required_artifacts = graph.runtime_dependency_closure(
        job.task,
        preview=job.preview,
    )
    validate_artifact_plan(definition.streams, graph, set(required_artifacts))
    return RuntimeJobPlan(job=job, required_artifacts=required_artifacts)


def execute_runtime_job(
    command: Literal["serve", "inspect"],
    definition: ProjectDefinition,
    plan: RuntimeJobPlan,
) -> tuple[WrittenOutput, ...]:
    job = plan.job
    current_artifacts = set(
        hydrate_runtime_artifacts_for_pipeline(
            job.runtime,
            definition,
        )
    )
    unavailable = [
        key for key in plan.required_artifacts if key not in current_artifacts
    ]
    if unavailable:
        raise ArtifactResolutionError(
            f"Output operation '{job.task.id}' requires missing or stale artifacts: "
            f"{', '.join(unavailable)}."
        )

    with operation_scope(f"{command}:{job.name}"):
        emit_execution_message(
            "Config:\n"
            + json.dumps(
                {
                    "operation": job.task.model_dump(
                        mode="json",
                        exclude={"kind"},
                        exclude_none=True,
                    ),
                    "limit": job.limit,
                    "preview": job.preview,
                    "throttle_ms": job.throttle_ms,
                    "output_ids": list(job.output_ids),
                    "output": {
                        "transport": job.output.transport,
                        "format": job.output.format,
                        "view": job.output.view,
                        "encoding": job.output.encoding,
                        "compression": job.output.compression,
                        "destination": (
                            str(job.output.destination)
                            if job.output.destination is not None
                            else None
                        ),
                    },
                    "execution": job.runtime.execution.model_dump(mode="json"),
                    "observability": job.observability.effective_config(),
                },
                indent=2,
            ),
            level=logging.DEBUG,
        )
        result = run_output_operation(
            job.runtime,
            job.task,
            OutputOptions(
                limit=job.limit,
                output_format=job.output.format,
                throttle_ms=job.throttle_ms,
                preview=job.preview,
                output_ids=job.output_ids,
            ),
        )
        return persist_runtime_result(
            result,
            job.output,
            job.output_ids,
            job.runtime.heartbeat_interval_seconds,
            logger=logger,
        )
