import logging

from jerrythomas.artifacts.errors import ArtifactResolutionError
from jerrythomas.artifacts.executor import run_build_if_needed
from jerrythomas.artifacts.planning import ArtifactGraph
from jerrythomas.artifacts.series import prune_series_cache
from jerrythomas.config.dataset.split import resolve_fold_output
from jerrythomas.config.tasks.series import SeriesTask
from jerrythomas.io.runs import (
    RunFoldOutput,
    RunOutput,
    RunPaths,
    finish_run_failed,
    finish_run_success,
    set_latest_run,
    start_run,
    make_run_id,
    SavedRun,
    load_run,
)
from jerrythomas.profiles.errors import ProfileCommandError
from jerrythomas.profiles.executor import execution_scope
from jerrythomas.profiles.export import (
    execute_export_job,
    preflight_export_jobs,
)
from jerrythomas.runtime import Runtime
from jerrythomas.services.execution_lock import (
    ProjectExecutionBusyError,
    project_execution_lock,
)
from jerrythomas.services.runtime_compiler import compile_runtime
from jerrythomas.profiles.recipes import (
    capture_recipe,
    capture_artifacts,
    validate_recipe_inputs,
    output_identity,
)

from .execution import (
    RuntimeJobPlan,
    execute_runtime_job,
    plan_export_job,
    plan_runtime_job,
    validate_build_job,
)
from .models import (
    BuildJob,
    BuildRunRequest,
    ExportRunRequest,
    ProfileRunRequest,
    RuntimeRunRequest,
    ServeRunPlan,
)

logger = logging.getLogger(__name__)


def run_profiles(request: ProfileRunRequest) -> tuple[SavedRun, ...]:
    """Return completed serve runs or a completed export invocation.

    Build, inspect, and unmanaged runtime outputs return an empty tuple.
    Execution or publication failures raise; no results are returned.
    """
    results: tuple[SavedRun, ...]
    try:
        with project_execution_lock(request.definition.project.artifacts_root):
            if isinstance(request, BuildRunRequest):
                _run_build_profiles(request)
                results = ()
            elif isinstance(request, RuntimeRunRequest):
                results = _run_runtime_profiles(request)
            elif isinstance(request, ExportRunRequest):
                results = _run_export_profiles(request)
            else:
                raise TypeError(
                    f"Unsupported profile request: {type(request).__name__}"
                )
            try:
                _prune_series_caches(request)
            except OSError as exc:
                logger.warning(
                    "Series cache cleanup skipped under '%s': %s",
                    request.definition.project.artifacts_root,
                    exc,
                )
            return results
    except (ArtifactResolutionError, ProjectExecutionBusyError) as exc:
        error = ProfileCommandError(str(exc))
        for note in getattr(exc, "__notes__", ()):
            error.add_note(note)
        raise error from exc


def _prune_series_caches(request: ProfileRunRequest) -> None:
    root = request.definition.project.artifacts_root
    for task in request.definition.artifact_graph.tasks_by_id.values():
        if isinstance(task, SeriesTask):
            prune_series_cache(root / task.path, root)


def _run_build_profiles(request: BuildRunRequest) -> None:
    jobs = list(request.jobs)
    if not jobs:
        return
    graph = request.definition.artifact_graph
    try:
        _validate_build_order(jobs, graph)
        for job in jobs:
            validate_build_job(job.task, request.definition)
    except ValueError as exc:
        raise ProfileCommandError(str(exc)) from exc

    resolved_artifacts: set[str] = set()
    for job in jobs:
        runtime = compile_runtime(request.definition, job.task.dataset)
        runtime.execution = request.execution
        with execution_scope(runtime, job.settings.observability):
            run_build_if_needed(
                request.definition,
                required_artifacts={job.task.id},
                settings=job.settings,
                runtime=runtime,
                resolved_artifacts=resolved_artifacts,
            )


def _run_runtime_profiles(request: RuntimeRunRequest) -> tuple[SavedRun, ...]:
    jobs = list(request.jobs)
    if not jobs:
        return ()
    try:
        plans = [plan_runtime_job(job, request.definition) for job in jobs]
    except ValueError as exc:
        raise ProfileCommandError(str(exc)) from exc

    started_runs: list[ServeRunPlan] = []
    run_outputs: dict[RunPaths, list[RunOutput]] = {
        plan.paths: [] for plan in request.serve_run_plans
    }
    recipes = {
        run.paths: capture_recipe(
            request.definition,
            "serve",
            [plan.job for plan in plans if plan.job.output.run == run.paths],
            request.execution,
            tuple(
                dict.fromkeys(
                    key
                    for plan in plans
                    if plan.job.output.run == run.paths
                    for key in plan.required_artifacts
                )
            ),
        )
        for run in request.serve_run_plans
    }
    try:
        _prepare_runtime_artifacts(request, plans)
        for run_plan in request.serve_run_plans:
            recipes[run_plan.paths] = capture_artifacts(
                recipes[run_plan.paths], request.definition
            )
            start_run(
                run_plan.paths,
                recipe=recipes[run_plan.paths],
                preview=run_plan.preview,
                split=(
                    request.definition.require_dataset(run_plan.dataset_id).split
                    if run_plan.dataset_id is not None
                    else None
                ),
            )
            started_runs.append(run_plan)

        for plan in plans:
            job = plan.job
            job.runtime.execution = request.execution
            job.runtime.heartbeat_interval_seconds = (
                job.observability.heartbeat_interval_seconds
            )
            with execution_scope(job.runtime, job.observability):
                written_outputs = execute_runtime_job(
                    request.command,
                    request.definition,
                    plan,
                )
            if job.output.run is not None:
                split = (
                    job.runtime.dataset.split
                    if job.runtime.dataset is not None
                    else None
                )
                for written in written_outputs:
                    fold_output = None
                    if (
                        job.preview is None
                        and split is not None
                        and written.output_id is not None
                    ):
                        fold, role, labels = resolve_fold_output(
                            split, written.output_id
                        )
                        fold_output = RunFoldOutput(
                            id=fold.id, role=role, labels=labels
                        )
                    run_outputs[job.output.run].append(
                        RunOutput(
                            profile=job.name,
                            operation=job.task.id,
                            stream=getattr(job.task, "stream", None),
                            output_id=written.output_id,
                            path=written.path.relative_to(
                                job.output.run.run_root
                            ).as_posix(),
                            format=job.output.format,
                            view=job.output.view
                            if job.output.format not in {"txt", "html"}
                            else None,
                            encoding=job.output.encoding,
                            compression=job.output.compression,
                            row_count=written.row_count,
                            fold=fold_output,
                            **output_identity(written.path),
                        )
                    )
        for recipe in recipes.values():
            validate_recipe_inputs(recipe, request.definition)
    except BaseException as exc:
        _mark_serve_runs_failed(started_runs, exc)
        raise
    else:
        _publish_serve_runs(started_runs, run_outputs)
    return tuple(load_run(plan.paths.metadata_path) for plan in started_runs)


def _run_export_profiles(
    request: ExportRunRequest,
) -> tuple[SavedRun, ...]:
    jobs = list(request.jobs)
    if not jobs:
        return ()
    run_id = make_run_id()
    request.runtime.execution = request.execution
    try:
        preflight_export_jobs(request.runtime, jobs)
        plans = [plan_export_job(job, request.definition) for job in jobs]
    except (OSError, ValueError) as exc:
        raise ProfileCommandError(str(exc)) from exc

    required_artifacts = {
        artifact for plan in plans for artifact in plan.required_artifacts
    }
    _build_prerequisites(request, required_artifacts, request.runtime)
    results: list[SavedRun] = []
    for plan in plans:
        job = plan.job
        request.runtime.heartbeat_interval_seconds = (
            job.observability.heartbeat_interval_seconds
        )
        with execution_scope(request.runtime, job.observability):
            recipe = capture_recipe(
                request.definition,
                "export",
                [job],
                request.execution,
                plan.required_artifacts,
            )
            recipe = capture_artifacts(recipe, request.definition)
            results.append(
                execute_export_job(
                    job,
                    request.runtime,
                    recipe=recipe,
                    definition=request.definition,
                    run_id=run_id,
                )
            )
    return tuple(results)


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


def _mark_serve_runs_failed(
    plans: list[ServeRunPlan],
    command_error: BaseException,
) -> None:
    for plan in plans:
        try:
            finish_run_failed(plan.paths)
        except Exception as exc:
            command_error.add_note(
                f"Failed to finalize serve run '{plan.paths.run_id}': {exc}"
            )


def _publish_serve_runs(
    plans: list[ServeRunPlan],
    run_outputs: dict[RunPaths, list[RunOutput]],
) -> None:
    first_error: Exception | None = None
    for plan in plans:
        try:
            finish_run_success(plan.paths, outputs=tuple(run_outputs[plan.paths]))
            if plan.preview is None:
                set_latest_run(plan.paths)
        except Exception as exc:
            if first_error is None:
                first_error = exc
            else:
                first_error.add_note(
                    f"Also failed to finalize serve run '{plan.paths.run_id}': {exc}"
                )
    if first_error is not None:
        raise first_error


def _prepare_runtime_artifacts(
    request: RuntimeRunRequest,
    plans: list[RuntimeJobPlan],
) -> None:
    required_artifacts = {
        artifact for plan in plans for artifact in plan.required_artifacts
    }
    if not required_artifacts:
        return

    runtime = compile_runtime(request.definition)
    runtime.execution = request.execution
    _build_prerequisites(request, required_artifacts, runtime)


def _build_prerequisites(
    request: RuntimeRunRequest | ExportRunRequest,
    required_artifacts: set[str],
    runtime: Runtime,
) -> None:
    if not required_artifacts:
        return
    settings = request.artifact_settings
    with execution_scope(runtime, settings.observability):
        run_build_if_needed(
            request.definition,
            required_artifacts=required_artifacts,
            settings=settings,
            runtime=runtime,
        )
