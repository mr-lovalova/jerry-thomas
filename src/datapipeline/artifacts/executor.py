import json
import logging
from dataclasses import dataclass
from typing import Literal

from datapipeline.artifacts.errors import ArtifactResolutionError
from datapipeline.artifacts.fingerprints import calculate_artifact_hashes
from datapipeline.artifacts.hydration import hydrate_runtime_artifacts
from datapipeline.artifacts.planning import ArtifactGraph
from datapipeline.artifacts.settings import BuildSettings
from datapipeline.artifacts.validation import validate_artifact_plan
from datapipeline.build.state import (
    BuildState,
    load_build_state,
    save_build_state,
)
from datapipeline.config.profiles.build import ArtifactMode
from datapipeline.config.tasks.base import ArtifactTask
from datapipeline.execution.observability import (
    emit_execution_message,
    emit_file_result,
    operation_scope,
)
from datapipeline.operations.persistence import (
    ArtifactOutput,
    fingerprint_artifact_output,
)
from datapipeline.plugins import BUILD_OPERATIONS_EP, load_entrypoint
from datapipeline.runtime import Runtime
from datapipeline.services.definitions import ArtifactHashes, ProjectDefinition
from datapipeline.services.path_policy import resolve_artifact_output_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactBuildJob:
    task: ArtifactTask
    invalidated_artifacts: tuple[str, ...]


@dataclass(frozen=True)
class SkippedBuild:
    reason: Literal[
        "already_resolved",
        "mode_off",
        "no_artifacts_selected",
        "not_required",
        "up_to_date",
    ]
    artifacts: tuple[str, ...]


@dataclass(frozen=True)
class BuildPlan:
    reason: Literal["force", "missing", "stale"]
    artifacts: tuple[str, ...]
    jobs: tuple[ArtifactBuildJob, ...]
    artifact_hashes: ArtifactHashes
    previous_state: BuildState | None
    graph: ArtifactGraph


ArtifactPlan = BuildPlan | SkippedBuild


def _report_artifact_plan(
    plan: ArtifactPlan,
    mode: ArtifactMode,
    requested_artifacts: set[str],
) -> None:
    if isinstance(plan, BuildPlan):
        jobs = [job.task.id for job in plan.jobs]
        scheduled = set(jobs)
        current = [artifact for artifact in plan.artifacts if artifact not in scheduled]
    else:
        jobs = []
        current = list(plan.artifacts)

    emit_execution_message(
        "Artifact plan:\n"
        + json.dumps(
            {
                "action": "run" if isinstance(plan, BuildPlan) else "skip",
                "reason": plan.reason,
                "mode": mode,
                "requested": sorted(requested_artifacts),
                "required": list(plan.artifacts),
                "jobs": jobs,
                "current": current,
            },
            indent=2,
        ),
        level=logging.DEBUG,
    )


def _plan_build(
    *,
    definition: ProjectDefinition,
    graph: ArtifactGraph,
    required_artifacts: set[str],
    mode: ArtifactMode,
    resolved_artifacts: set[str] | None = None,
) -> ArtifactPlan:
    try:
        selected_roots = set(required_artifacts)
        selected_keys = set(graph.dependency_closure(selected_roots))
    except ValueError as exc:
        raise ArtifactResolutionError(str(exc)) from exc

    if not selected_keys:
        return SkippedBuild(reason="no_artifacts_selected", artifacts=())

    artifact_hashes = definition.artifact_hashes

    dataset = None
    if graph.requires_dataset(selected_keys):
        dataset = definition.dataset
        selected_keys = set(graph.dependency_closure(selected_roots, dataset))
    if not selected_keys:
        return SkippedBuild(reason="not_required", artifacts=())

    expanded_artifacts = graph.topological_order(selected_keys)
    try:
        validate_artifact_plan(definition.streams, graph, selected_keys)
    except ValueError as exc:
        raise ArtifactResolutionError(str(exc)) from exc

    previous_state = load_build_state(definition.project.artifacts_root)
    resolved = resolved_artifacts if resolved_artifacts is not None else set()
    freshness = graph.freshness(
        keys=selected_keys | resolved,
        state=previous_state,
        artifact_hashes=artifact_hashes,
        artifacts_root=definition.project.artifacts_root,
    )
    stale_resolved = freshness.outdated & resolved
    if stale_resolved:
        artifacts = ", ".join(graph.topological_order(stale_resolved))
        raise RuntimeError(
            "Artifacts resolved by an earlier build profile became stale before "
            f"the command completed: {artifacts}. Rerun the build."
        )
    selected_outdated = freshness.outdated & selected_keys
    if mode == "OFF":
        if selected_outdated:
            artifacts = ", ".join(graph.topological_order(selected_outdated))
            raise ArtifactResolutionError(
                "Artifact mode is OFF, but required artifacts are missing or stale: "
                f"{artifacts}."
            )
        return SkippedBuild(
            reason="mode_off",
            artifacts=expanded_artifacts,
        )

    force = mode == "FORCE"
    if not force and not selected_outdated:
        return SkippedBuild(
            reason="up_to_date",
            artifacts=expanded_artifacts,
        )

    build_keys = set(selected_keys) - resolved if force else set(selected_outdated)
    if not build_keys:
        return SkippedBuild(
            reason="already_resolved",
            artifacts=expanded_artifacts,
        )
    all_active_keys = (
        set(
            graph.dependency_closure(
                (definition.key for definition in graph.definitions),
                dataset,
            )
        )
        if dataset is not None
        else {definition.key for definition in graph.definitions}
    )
    jobs = tuple(
        ArtifactBuildJob(
            task=graph.tasks_by_id[key],
            invalidated_artifacts=graph.topological_order(
                {key}
                | graph.dependents_of(
                    {key},
                    active_keys=all_active_keys,
                )
            ),
        )
        for key in graph.topological_order(build_keys)
    )
    return BuildPlan(
        reason=(
            "force"
            if force
            else ("missing" if freshness.missing & selected_keys else "stale")
        ),
        artifacts=expanded_artifacts,
        jobs=jobs,
        artifact_hashes=artifact_hashes,
        previous_state=previous_state,
        graph=graph,
    )


def _execute_build_jobs(
    definition: ProjectDefinition,
    runtime: Runtime,
    plan: BuildPlan,
    settings: BuildSettings,
) -> BuildState:
    for job in plan.jobs:
        resolve_artifact_output_path(job.task.output, runtime.artifacts_root)

    current_state = (
        plan.previous_state.model_copy(deep=True)
        if plan.previous_state is not None
        else BuildState()
    )
    for job in plan.jobs:
        artifact_hash = plan.artifact_hashes.for_artifact(job.task.id)
        with operation_scope(f"build:{job.task.id}"):
            _require_stable_artifact_inputs(
                definition,
                job.task.id,
                artifact_hash,
            )
            emit_execution_message(
                "Config:\n"
                + json.dumps(
                    {
                        "operation": job.task.model_dump(
                            mode="json",
                            exclude={"kind", "id"},
                            exclude_none=True,
                        ),
                        "mode": settings.mode,
                        "execution": runtime.execution.model_dump(mode="json"),
                        "observability": settings.observability.effective_config(),
                    },
                    indent=2,
                ),
                level=logging.DEBUG,
            )
            for key in job.invalidated_artifacts:
                current_state.artifacts.pop(key, None)
            hydrate_runtime_artifacts(
                runtime=runtime,
                graph=plan.graph,
                state=current_state,
                artifact_hashes=plan.artifact_hashes,
                artifact_keys=plan.artifacts,
            )

            runner = load_entrypoint(BUILD_OPERATIONS_EP, job.task.entrypoint)
            operation_result = runner(
                runtime=runtime,
                task_cfg=job.task,
            )
            if not isinstance(operation_result, ArtifactOutput):
                raise TypeError("Build operation must return ArtifactOutput.")
            output = operation_result
            files = fingerprint_artifact_output(
                output,
                task=job.task,
                artifacts_root=runtime.artifacts_root,
            )
            _require_stable_artifact_inputs(
                definition,
                job.task.id,
                artifact_hash,
            )
            current_state.register(
                job.task.id,
                artifact_hash=artifact_hash,
                files=files,
                meta=output.meta,
            )
            save_build_state(current_state, runtime.artifacts_root)
            runtime.artifacts.register(
                job.task.id,
                relative_path=job.task.output,
                meta=output.meta,
            )
            label = job.task.id.replace("_", " ").capitalize()
            path = (runtime.artifacts_root / job.task.output).resolve()
            emit_file_result(label, path)
    return current_state


def _require_stable_artifact_inputs(
    definition: ProjectDefinition,
    artifact_id: str,
    expected_hash: str,
) -> None:
    current_hashes = calculate_artifact_hashes(
        definition.project,
        definition.dataset,
        definition.streams,
        definition.artifact_graph,
    )
    if current_hashes.for_artifact(artifact_id) != expected_hash:
        raise RuntimeError(
            f"Source files changed while building artifact '{artifact_id}'. "
            "Rerun the command with stable inputs."
        )


def run_build_if_needed(
    definition: ProjectDefinition,
    *,
    required_artifacts: set[str],
    settings: BuildSettings,
    runtime: Runtime,
    resolved_artifacts: set[str] | None = None,
) -> bool:
    """Execute artifact-producing operations when selected artifacts are missing or stale."""
    mode = settings.mode
    graph = definition.artifact_graph

    plan = _plan_build(
        definition=definition,
        graph=graph,
        required_artifacts=required_artifacts,
        mode=mode,
        resolved_artifacts=resolved_artifacts,
    )
    _report_artifact_plan(
        plan,
        mode=mode,
        requested_artifacts=required_artifacts,
    )
    if isinstance(plan, SkippedBuild):
        if plan.artifacts:
            hydrate_runtime_artifacts(
                runtime=runtime,
                graph=graph,
                state=load_build_state(definition.project.artifacts_root),
                artifact_hashes=definition.artifact_hashes,
                artifact_keys=plan.artifacts,
            )
        if resolved_artifacts is not None:
            resolved_artifacts.update(plan.artifacts)
        return False

    runtime.heartbeat_interval_seconds = (
        settings.observability.heartbeat_interval_seconds
    )
    _execute_build_jobs(
        definition,
        runtime=runtime,
        plan=plan,
        settings=settings,
    )
    if resolved_artifacts is not None:
        resolved_artifacts.update(plan.artifacts)
    return True
