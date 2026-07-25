import logging
import time

from datapipeline.artifacts.errors import ArtifactResolutionError
from datapipeline.artifacts.executor import run_build_if_needed
from datapipeline.artifacts.planning import (
    ArtifactGraph,
    build_artifact_graph,
    required_tick_artifacts,
)
from datapipeline.artifacts.series import prune_series_cache
from datapipeline.cli.visuals.execution import route_execution_event
from datapipeline.cli.visuals.rich.progress import visual_summary
from datapipeline.config.tasks.series import SeriesTask
from datapipeline.execution.events import RunStatus
from datapipeline.execution.observability import CommandFinished
from datapipeline.io.runs import (
    finish_run_failed,
    finish_run_success,
    set_latest_run,
    start_run,
)
from datapipeline.profiles.executor import execution_scope
from datapipeline.profiles.materialize import (
    execute_materialize_job,
    preflight_materialize_jobs,
)
from datapipeline.services.execution_lock import (
    ProjectExecutionBusyError,
    project_execution_lock,
)
from datapipeline.services.runtime_compiler import compile_runtime

from .execution import (
    RuntimeJobPlan,
    execute_runtime_job,
    plan_runtime_job,
    validate_build_job,
)
from .models import (
    BuildJob,
    BuildRunRequest,
    MaterializeRunRequest,
    ProfileRunRequest,
    RuntimeRunRequest,
    ServeRunPlan,
)

logger = logging.getLogger(__name__)


def _command_uses_visuals(request: ProfileRunRequest) -> bool:
    if isinstance(request, BuildRunRequest):
        return any(job.settings.observability.visuals == "on" for job in request.jobs)
    return request.artifact_settings.observability.visuals == "on" or any(
        job.observability.visuals == "on" for job in request.jobs
    )


def run_profiles(request: ProfileRunRequest) -> None:
    started_at = time.perf_counter()
    status: RunStatus = "error"
    try:
        with project_execution_lock(request.definition.project.artifacts_root):
            if isinstance(request, BuildRunRequest):
                _run_build_profiles(request)
            elif isinstance(request, RuntimeRunRequest):
                _run_runtime_profiles(request)
            elif isinstance(request, MaterializeRunRequest):
                _run_materialize_profiles(request)
            else:
                raise TypeError(
                    f"Unsupported profile request: {type(request).__name__}"
                )
            _prune_series_caches(request)
    except (ArtifactResolutionError, ProjectExecutionBusyError) as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc
    else:
        status = "success"
    finally:
        with visual_summary(logger.getEffectiveLevel(), _command_uses_visuals(request)):
            route_execution_event(
                CommandFinished(
                    command=request.command,
                    status=status,
                    elapsed_seconds=time.perf_counter() - started_at,
                ),
                logger,
            )


def _prune_series_caches(request: ProfileRunRequest) -> None:
    root = request.definition.project.artifacts_root
    for task in request.definition.artifact_operations:
        if isinstance(task, SeriesTask):
            prune_series_cache(root / task.output, root)


def _run_build_profiles(request: BuildRunRequest) -> None:
    jobs = list(request.jobs)
    if not jobs:
        return
    try:
        graph = build_artifact_graph(
            request.definition.artifact_operations,
            request.definition.dataset,
            request.definition.streams,
        )
        _validate_build_order(jobs, graph)
        for job in jobs:
            validate_build_job(job.task, graph, request.definition)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc

    resolved_artifacts: set[str] = set()
    for job in jobs:
        runtime = compile_runtime(request.definition)
        runtime.execution = request.execution
        with execution_scope(runtime, job.settings.observability):
            run_build_if_needed(
                request.definition,
                graph=graph,
                required_artifacts={job.task.id},
                settings=job.settings,
                runtime=runtime,
                resolved_artifacts=resolved_artifacts,
            )


def _run_runtime_profiles(request: RuntimeRunRequest) -> None:
    jobs = list(request.jobs)
    if not jobs:
        return
    try:
        graph = build_artifact_graph(
            request.definition.artifact_operations,
            request.definition.dataset,
            request.definition.streams,
        )
        plans = [plan_runtime_job(job, graph, request.definition) for job in jobs]
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc

    started_runs: list[ServeRunPlan] = []
    succeeded = False
    try:
        _prepare_runtime_artifacts(request, graph, plans)
        for run_plan in request.serve_run_plans:
            start_run(run_plan.paths, preview=run_plan.preview)
            started_runs.append(run_plan)

        for plan in plans:
            job = plan.job
            job.runtime.execution = request.execution
            job.runtime.heartbeat_interval_seconds = (
                job.observability.heartbeat_interval_seconds
            )
            job.runtime.output_ids = job.output_ids
            with execution_scope(job.runtime, job.observability):
                execute_runtime_job(
                    request.command,
                    request.definition,
                    graph,
                    plan,
                )
        succeeded = True
    finally:
        _finalize_serve_runs(started_runs, succeeded)


def _run_materialize_profiles(request: MaterializeRunRequest) -> None:
    jobs = list(request.jobs)
    if not jobs:
        return
    request.runtime.execution = request.execution
    try:
        preflight_materialize_jobs(request.runtime, jobs)
        graph = build_artifact_graph(
            request.definition.artifact_operations,
            request.definition.dataset,
            request.definition.streams,
        )
        required_artifacts = set(
            required_tick_artifacts(
                (job.stream for job in jobs),
                request.definition.streams,
                graph.tasks_by_id,
            )
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc

    _prepare_materialize_artifacts(request, graph, required_artifacts)
    for job in jobs:
        request.runtime.heartbeat_interval_seconds = (
            job.observability.heartbeat_interval_seconds
        )
        with execution_scope(request.runtime, job.observability):
            execute_materialize_job(job, request.runtime)


def _validate_build_order(jobs: list[BuildJob], graph: ArtifactGraph) -> None:
    operations = [job.task.id for job in jobs]
    if len(operations) != len(set(operations)):
        raise ValueError("Build profiles must reference unique artifact operations.")

    positions = {operation: index for index, operation in enumerate(operations)}
    for operation, position in positions.items():
        for dependency in graph.dependency_closure({operation}):
            dependency_position = positions.get(dependency)
            if dependency_position is not None and dependency_position > position:
                raise ValueError(
                    f"Build profile operation '{dependency}' must be ordered before "
                    f"dependent operation '{operation}'."
                )


def _finalize_serve_runs(plans: list[ServeRunPlan], succeeded: bool) -> None:
    if not succeeded:
        for plan in plans:
            try:
                finish_run_failed(plan.paths)
            except Exception:
                logger.exception(
                    "Failed to finalize serve run '%s' after command failure.",
                    plan.paths.run_id,
                )
        return

    first_error: Exception | None = None
    for plan in plans:
        try:
            finish_run_success(plan.paths)
            if plan.preview is None:
                set_latest_run(plan.paths)
        except Exception as exc:
            if first_error is None:
                first_error = exc
            else:
                logger.exception(
                    "Failed to finalize serve run '%s' after an earlier "
                    "finalization failure.",
                    plan.paths.run_id,
                )
    if first_error is not None:
        raise first_error


def _prepare_runtime_artifacts(
    request: RuntimeRunRequest,
    graph: ArtifactGraph,
    plans: list[RuntimeJobPlan],
) -> None:
    required_artifacts = {
        artifact for plan in plans for artifact in plan.required_artifacts
    }
    if not required_artifacts:
        return

    settings = request.artifact_settings
    runtime = compile_runtime(request.definition)
    runtime.execution = request.execution
    with execution_scope(runtime, settings.observability):
        run_build_if_needed(
            request.definition,
            graph=graph,
            required_artifacts=required_artifacts,
            settings=settings,
            runtime=runtime,
        )


def _prepare_materialize_artifacts(
    request: MaterializeRunRequest,
    graph: ArtifactGraph,
    required_artifacts: set[str],
) -> None:
    if not required_artifacts:
        return
    with execution_scope(
        request.runtime,
        request.artifact_settings.observability,
    ):
        run_build_if_needed(
            request.definition,
            graph=graph,
            required_artifacts=required_artifacts,
            settings=request.artifact_settings,
            runtime=request.runtime,
        )
