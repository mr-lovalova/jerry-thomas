import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

from pydantic import ValidationError

from datapipeline.artifacts.settings import BuildSettings
from datapipeline.config.preview import PreviewStage
from datapipeline.config.profiles.base import Profile, ProfileCommand
from datapipeline.config.profiles.build import (
    BuildProfile,
    normalize_artifact_mode,
)
from datapipeline.config.profiles.defaults import (
    BuildProfileDefaults,
    InspectProfileDefaults,
    MaterializeProfileDefaults,
    ProfileDefaults,
    ServeProfileDefaults,
)
from datapipeline.config.profiles.inspect import InspectProfile
from datapipeline.config.profiles.materialize import MaterializeProfile
from datapipeline.config.profiles.output import ServeOutputConfig
from datapipeline.config.profiles.serve import ServeProfile
from datapipeline.execution.settings import (
    LogOutputTarget,
    resolve_execution_log_outputs,
    resolve_observability_settings,
)
from datapipeline.io.output import OutputResolutionError
from datapipeline.io.runs import RunPaths
from datapipeline.profiles.loader import (
    apply_profile_defaults,
    profile_specs_with_defaults,
)
from datapipeline.profiles.materialize import resolve_materialize_jobs
from datapipeline.profiles.models import (
    BuildJob,
    BuildRunRequest,
    MaterializeRunRequest,
    RuntimeJob,
    RuntimeRunRequest,
    ServeRunPlan,
)
from datapipeline.profiles.runtime_profiles import (
    resolve_inspect_profiles,
    resolve_serve_profiles,
)
from datapipeline.services.definitions import ProjectDefinition, ProjectManifest
from datapipeline.services.path_policy import sanitize_path_segment
from datapipeline.services.project_definition import load_project_definition
from datapipeline.services.runtime_compiler import compile_runtime

logger = logging.getLogger(__name__)


def _validation_error_without_inputs(exc: ValidationError) -> str:
    count = exc.error_count()
    heading = f"{count} validation error{'s' if count != 1 else ''} for {exc.title}"
    details = []
    for error in exc.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in error["loc"]) or exc.title
        message = "Invalid configuration value" if error.get("ctx") else error["msg"]
        details.append(f"{location}\n  {message} [type={error['type']}]")
    return "\n".join((heading, *details))


def _execution_root(artifacts_root: Path) -> Path:
    execution_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    return (artifacts_root / "_system" / "executions" / execution_id).resolve()


def _load_definition(project: str) -> ProjectDefinition:
    try:
        return load_project_definition(Path(project))
    except ValidationError as exc:
        logger.error(
            "Failed to load project definition: %s",
            _validation_error_without_inputs(exc),
        )
        raise SystemExit(2) from exc
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.error("Failed to load project definition: %s", exc)
        raise SystemExit(2) from exc


def _select_profiles(
    project: ProjectManifest,
    command: ProfileCommand,
    profile_name: str | None,
) -> tuple[list[Profile], ProfileDefaults]:
    try:
        profiles, defaults = profile_specs_with_defaults(
            project,
            cmd=command,
        )
    except ValidationError as exc:
        logger.error(
            "Failed to load %s profiles: %s",
            command,
            _validation_error_without_inputs(exc),
        )
        raise SystemExit(2) from exc
    except (OSError, TypeError, ValueError) as exc:
        logger.error("Failed to load %s profiles: %s", command, exc)
        raise SystemExit(2) from exc
    if not profiles:
        logger.error("Project does not define %s profiles.", command)
        raise SystemExit(2)
    if profile_name is None:
        selected = [profile for profile in profiles if profile.enabled]
    else:
        normalized_name = profile_name.strip()
        if not normalized_name:
            logger.error("%s profile name must not be empty.", command.capitalize())
            raise SystemExit(2)
        selected = [profile for profile in profiles if profile.name == normalized_name]
        if not selected:
            logger.error("Unknown %s profile '%s'", command, normalized_name)
            raise SystemExit(2)
    try:
        return (
            [apply_profile_defaults(profile, defaults) for profile in selected],
            defaults,
        )
    except ValidationError as exc:
        logger.error("%s", _validation_error_without_inputs(exc))
        raise SystemExit(2) from exc
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc


def _serve_run_plans(
    jobs: Sequence[RuntimeJob],
) -> tuple[ServeRunPlan, ...]:
    plans_by_run: dict[RunPaths, ServeRunPlan] = {}
    for job in jobs:
        if job.output.run is None:
            continue
        if job.output.run not in plans_by_run:
            plans_by_run[job.output.run] = ServeRunPlan(
                paths=job.output.run,
                preview=job.preview,
            )
    return tuple(plans_by_run.values())


def build_build_run_request(
    project: str,
    profile_name: str | None = None,
    force: bool = False,
    cli_log_level: str | None = None,
    cli_log_outputs: Sequence[LogOutputTarget] | None = None,
    base_log_level: str = "INFO",
    cli_visuals: str | None = None,
    cli_heartbeat_interval_seconds: float | None = None,
) -> BuildRunRequest | None:
    definition = _load_definition(project)
    project_path = definition.project.path
    loaded_profiles, defaults = _select_profiles(
        definition.project,
        "build",
        profile_name,
    )
    if not loaded_profiles:
        return None
    if not isinstance(defaults, BuildProfileDefaults):
        raise TypeError("Build profile loading returned the wrong defaults type")

    artifact_tasks_by_id = {task.id: task for task in definition.artifact_operations}
    runtime_task_ids = {task.id for task in definition.runtime_operations}
    build_profiles: list[BuildProfile] = []
    for profile in loaded_profiles:
        if not isinstance(profile, BuildProfile):
            raise TypeError("Build profile loading returned the wrong profile type")
        task = artifact_tasks_by_id.get(profile.operation)
        if task is None:
            if profile.operation in runtime_task_ids:
                logger.error(
                    "Build profile '%s' must reference an artifact operation; '%s' "
                    "is a runtime operation.",
                    profile.name,
                    profile.operation,
                )
                raise SystemExit(2)
            logger.error(
                "Build profile '%s' references unknown operation '%s'.",
                profile.name,
                profile.operation,
            )
            raise SystemExit(2)
        build_profiles.append(profile)

    execution_dir = _execution_root(definition.project.artifacts_root)
    build_jobs: list[BuildJob] = []
    for profile in build_profiles:
        try:
            observability = resolve_observability_settings(
                project_path,
                profile.observability,
                cli_visuals=cli_visuals,
                cli_heartbeat_interval_seconds=cli_heartbeat_interval_seconds,
                cli_log_level=cli_log_level,
                cli_log_outputs=cli_log_outputs,
                base_log_level=base_log_level,
            )
        except ValueError as exc:
            logger.error("Invalid build configuration: %s", exc)
            raise SystemExit(2) from exc
        build_jobs.append(
            BuildJob(
                task=artifact_tasks_by_id[profile.operation].model_copy(deep=True),
                settings=BuildSettings(
                    mode="FORCE" if force else profile.mode or "AUTO",
                    observability=replace(
                        observability,
                        log_output=resolve_execution_log_outputs(
                            observability.log_output,
                            execution_dir,
                            default_path=(
                                Path("logs")
                                / f"build.{sanitize_path_segment(profile.name)}.log"
                            ),
                        ),
                    ),
                ),
            )
        )

    return BuildRunRequest(
        definition=definition,
        jobs=build_jobs,
        execution=defaults.execution,
    )


def build_runtime_run_request(
    command: Literal["serve", "inspect"],
    project: str,
    profile_name: str | None = None,
    artifact_mode: str | None = None,
    limit: int | None = None,
    preview: PreviewStage | None = None,
    cli_output: ServeOutputConfig | None = None,
    cli_log_level: str | None = None,
    cli_log_outputs: Sequence[LogOutputTarget] | None = None,
    base_log_level: str = "INFO",
    cli_visuals: str | None = None,
    cli_heartbeat_interval_seconds: float | None = None,
) -> RuntimeRunRequest | None:
    definition = _load_definition(project)
    project_path = definition.project.path
    loaded_profiles, defaults = _select_profiles(
        definition.project,
        command,
        profile_name,
    )
    if not loaded_profiles:
        return None

    if command == "serve":
        if not isinstance(defaults, ServeProfileDefaults):
            raise TypeError("Serve profile loading returned the wrong defaults type")
        serve_profiles = [
            profile for profile in loaded_profiles if isinstance(profile, ServeProfile)
        ]
        if len(serve_profiles) != len(loaded_profiles):
            raise TypeError("Serve profile loading returned the wrong profile type")
        runtime_profiles: Sequence[ServeProfile | InspectProfile] = serve_profiles
    else:
        if not isinstance(defaults, InspectProfileDefaults):
            raise TypeError("Inspect profile loading returned the wrong defaults type")
        inspect_profiles = [
            profile
            for profile in loaded_profiles
            if isinstance(profile, InspectProfile)
        ]
        if len(inspect_profiles) != len(loaded_profiles):
            raise TypeError("Inspect profile loading returned the wrong profile type")
        runtime_profiles = inspect_profiles

    runtime_tasks_by_id = {task.id: task for task in definition.runtime_operations}
    artifact_task_ids = {task.id for task in definition.artifact_operations}
    for profile in runtime_profiles:
        task = runtime_tasks_by_id.get(profile.operation)
        if task is None:
            if profile.operation in artifact_task_ids:
                logger.error(
                    "%s profile '%s' must reference a runtime operation; '%s' is "
                    "an artifact operation.",
                    command.capitalize(),
                    profile.name,
                    profile.operation,
                )
                raise SystemExit(2)
            logger.error(
                "%s profile '%s' references unknown operation '%s'.",
                command.capitalize(),
                profile.name,
                profile.operation,
            )
            raise SystemExit(2)

    try:
        resolved_artifact_mode = (
            normalize_artifact_mode(artifact_mode) or defaults.artifact_mode or "AUTO"
        )
    except ValueError as exc:
        logger.error("Invalid artifact mode: %s", exc)
        raise SystemExit(2) from exc

    execution_dir = _execution_root(definition.project.artifacts_root)
    try:
        artifact_observability = resolve_observability_settings(
            project_path,
            defaults.observability,
            cli_visuals=cli_visuals,
            cli_heartbeat_interval_seconds=cli_heartbeat_interval_seconds,
            cli_log_level=cli_log_level,
            cli_log_outputs=cli_log_outputs,
            base_log_level=base_log_level,
        )
    except ValueError as exc:
        logger.error("Invalid prerequisite observability: %s", exc)
        raise SystemExit(2) from exc
    artifact_settings = BuildSettings(
        mode=resolved_artifact_mode,
        observability=replace(
            artifact_observability,
            log_output=resolve_execution_log_outputs(
                artifact_observability.log_output,
                execution_dir,
                default_path=Path("logs") / f"{command}.artifacts.log",
            ),
        ),
    )

    try:
        if command == "serve":
            resolved_profiles = resolve_serve_profiles(
                definition,
                serve_profiles,
                preview,
                limit,
                cli_output,
                cli_log_level=cli_log_level,
                cli_log_outputs=cli_log_outputs,
                base_log_level=base_log_level,
                cli_visuals=cli_visuals,
                cli_heartbeat_interval_seconds=cli_heartbeat_interval_seconds,
            )
        else:
            if preview is not None:
                raise ValueError("Inspect profiles do not support previews.")
            resolved_profiles = resolve_inspect_profiles(
                definition,
                inspect_profiles,
                limit,
                cli_output,
                cli_log_level=cli_log_level,
                cli_log_outputs=cli_log_outputs,
                base_log_level=base_log_level,
                cli_visuals=cli_visuals,
                cli_heartbeat_interval_seconds=cli_heartbeat_interval_seconds,
            )
        jobs = [
            RuntimeJob(
                name=profile.name,
                task=runtime_tasks_by_id[profile.operation_id].model_copy(deep=True),
                runtime=compile_runtime(definition),
                output=profile.output,
                observability=replace(
                    profile.observability,
                    log_output=resolve_execution_log_outputs(
                        profile.observability.log_output,
                        execution_dir,
                        default_path=(
                            Path("logs")
                            / f"{command}.{sanitize_path_segment(profile.name)}.log"
                        ),
                    ),
                ),
                limit=profile.limit,
                throttle_ms=profile.throttle_ms,
                preview=profile.preview,
                output_ids=profile.output_ids,
            )
            for profile in resolved_profiles
        ]
    except OutputResolutionError as exc:
        logger.error("Invalid output configuration: %s", exc)
        raise SystemExit(2) from exc
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc

    return RuntimeRunRequest(
        command=command,
        definition=definition,
        jobs=jobs,
        execution=defaults.execution,
        artifact_settings=artifact_settings,
        serve_run_plans=(_serve_run_plans(jobs) if command == "serve" else ()),
    )


def build_materialize_run_request(
    project: str,
    profile_name: str | None,
    overwrite: bool | None,
    output: Path | None,
    artifact_mode: str | None,
    cli_log_level: str | None,
    cli_log_outputs: Sequence[LogOutputTarget],
    base_log_level: str,
    cli_visuals: str | None,
    cli_heartbeat_interval_seconds: float | None,
) -> MaterializeRunRequest | None:
    definition = _load_definition(project)
    project_path = definition.project.path
    loaded_profiles, defaults = _select_profiles(
        definition.project,
        "materialize",
        profile_name,
    )
    materialize_profiles = [
        profile
        for profile in loaded_profiles
        if isinstance(profile, MaterializeProfile)
    ]
    if len(materialize_profiles) != len(loaded_profiles):
        raise TypeError("Materialize profile loading returned the wrong profile type")
    if not isinstance(defaults, MaterializeProfileDefaults):
        raise TypeError("Materialize profile loading returned the wrong defaults type")
    if not materialize_profiles:
        return None

    try:
        execution_dir = _execution_root(definition.project.artifacts_root)
        jobs = resolve_materialize_jobs(
            profiles=materialize_profiles,
            project_path=project_path,
            execution_dir=execution_dir,
            overwrite=overwrite,
            cli_output=output,
            cli_visuals=cli_visuals,
            cli_heartbeat_interval_seconds=cli_heartbeat_interval_seconds,
            cli_log_level=cli_log_level,
            cli_log_outputs=cli_log_outputs,
            base_log_level=base_log_level,
        )
        resolved_artifact_mode = (
            normalize_artifact_mode(artifact_mode) or defaults.artifact_mode or "AUTO"
        )
        artifact_observability = resolve_observability_settings(
            project_path,
            defaults.observability,
            cli_visuals=cli_visuals,
            cli_heartbeat_interval_seconds=cli_heartbeat_interval_seconds,
            cli_log_level=cli_log_level,
            cli_log_outputs=cli_log_outputs,
            base_log_level=base_log_level,
        )
        runtime = compile_runtime(definition)
    except (OSError, TypeError, ValueError) as exc:
        logger.error("Invalid materialize configuration: %s", exc)
        raise SystemExit(2) from exc

    artifact_settings = BuildSettings(
        mode=resolved_artifact_mode,
        observability=replace(
            artifact_observability,
            log_output=resolve_execution_log_outputs(
                artifact_observability.log_output,
                execution_dir,
                default_path=Path("logs") / "materialize.artifacts.log",
            ),
        ),
    )
    return MaterializeRunRequest(
        definition=definition,
        jobs=jobs,
        execution=defaults.execution,
        artifact_settings=artifact_settings,
        runtime=runtime,
    )
