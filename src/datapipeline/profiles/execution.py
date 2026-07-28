import json
import logging
from dataclasses import dataclass
from typing import Literal

from datapipeline.artifacts.errors import ArtifactResolutionError
from datapipeline.artifacts.hydration import hydrate_runtime_artifacts_for_pipeline
from datapipeline.artifacts.planning import ArtifactGraph
from datapipeline.artifacts.validation import validate_artifact_plan
from datapipeline.config.tasks.base import ArtifactTask, PluginRuntimeTask
from datapipeline.config.tasks.coverage import CoverageTask
from datapipeline.config.tasks.dataset import DatasetTask
from datapipeline.config.tasks.matrix import MatrixTask
from datapipeline.execution.observability import (
    emit_execution_message,
    operation_scope,
)
from datapipeline.operations.persistence import persist_runtime_result
from datapipeline.operations.runtime.coverage import run_coverage_operation
from datapipeline.operations.runtime.dataset import run_dataset_operation
from datapipeline.operations.runtime.matrix import run_matrix_operation
from datapipeline.plugins import RUNTIME_OPERATIONS_EP, load_entrypoint
from datapipeline.services.definitions import ProjectDefinition

from .models import RuntimeJob

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeJobPlan:
    job: RuntimeJob
    required_artifacts: tuple[str, ...]


def validate_build_job(
    task: ArtifactTask,
    graph: ArtifactGraph,
    definition: ProjectDefinition,
) -> None:
    roots = {task.id}
    artifact_keys = set(graph.dependency_closure(roots, definition.dataset))
    validate_artifact_plan(definition.streams, graph, artifact_keys)


def plan_runtime_job(
    job: RuntimeJob,
    graph: ArtifactGraph,
    definition: ProjectDefinition,
) -> RuntimeJobPlan:
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
        dataset=definition.dataset,
    )
    validate_artifact_plan(definition.streams, graph, set(required_artifacts))
    return RuntimeJobPlan(job=job, required_artifacts=required_artifacts)


def run_runtime_operation(job: RuntimeJob) -> object:
    task = job.task
    if isinstance(task, DatasetTask):
        return run_dataset_operation(
            runtime=job.runtime,
            output_ids=job.output_ids,
            limit=job.limit,
            target=job.output,
            throttle_ms=job.throttle_ms,
            preview=job.preview,
        )
    if isinstance(task, MatrixTask):
        return run_matrix_operation(job.runtime, task, job.limit)
    if isinstance(task, CoverageTask):
        return run_coverage_operation(job.runtime, task)
    if not isinstance(task, PluginRuntimeTask):
        raise TypeError(f"Unsupported runtime task: {type(task).__name__}")

    plugin = load_entrypoint(RUNTIME_OPERATIONS_EP, task.entrypoint)
    return plugin(job.runtime, task, job.limit)


def execute_runtime_job(
    command: Literal["serve", "inspect"],
    definition: ProjectDefinition,
    graph: ArtifactGraph,
    plan: RuntimeJobPlan,
) -> None:
    job = plan.job
    current_artifacts = set(
        hydrate_runtime_artifacts_for_pipeline(
            job.runtime,
            definition,
            graph=graph,
        )
    )
    unavailable = [
        key for key in plan.required_artifacts if key not in current_artifacts
    ]
    if unavailable:
        raise ArtifactResolutionError(
            f"Runtime operation '{job.task.id}' requires missing or stale artifacts: "
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
        result = run_runtime_operation(job)
        persist_runtime_result(
            result,
            target=job.output,
            heartbeat_interval_seconds=job.runtime.heartbeat_interval_seconds,
            logger=logger,
        )
