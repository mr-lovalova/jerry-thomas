import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jerrythomas.artifacts.errors import ArtifactResolutionError
from jerrythomas.artifacts.hydration import hydrate_runtime_artifacts_for_pipeline
from jerrythomas.artifacts.validation import validate_artifact_plan
from jerrythomas.config.tasks.base import ArtifactTask, PluginRuntimeTask
from jerrythomas.config.tasks.coverage import CoverageTask
from jerrythomas.config.tasks.dataset import DatasetTask
from jerrythomas.config.tasks.matrix import MatrixTask
from jerrythomas.execution.observability import (
    emit_execution_message,
    operation_scope,
)
from jerrythomas.operations.persistence import RuntimeOutput, persist_runtime_result
from jerrythomas.operations.runtime.coverage import run_coverage_operation
from jerrythomas.operations.runtime.dataset import run_dataset_operation
from jerrythomas.operations.runtime.matrix import run_matrix_operation
from jerrythomas.plugins import RUNTIME_OPERATIONS_EP, load_entrypoint
from jerrythomas.services.definitions import ProjectDefinition

from .models import RuntimeJob

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeJobPlan:
    job: RuntimeJob
    required_artifacts: tuple[str, ...]


def validate_build_job(
    task: ArtifactTask,
    definition: ProjectDefinition,
) -> None:
    graph = definition.artifact_graph
    roots = {task.id}
    artifact_keys = set(graph.dependency_closure(roots, definition.dataset))
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
            output_format=job.output.format,
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
    result = plugin(job.runtime, task, job.limit)
    if result is not None and not isinstance(result, RuntimeOutput):
        raise TypeError("Custom runtime operation must return RuntimeOutput or None.")
    return result


def execute_runtime_job(
    command: Literal["serve", "inspect"],
    definition: ProjectDefinition,
    plan: RuntimeJobPlan,
) -> tuple[Path, ...]:
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
        return persist_runtime_result(
            result,
            job.output,
            job.output_ids,
            job.runtime.heartbeat_interval_seconds,
            logger=logger,
        )
