from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import ValidationError

from jerrythomas.artifacts.settings import BuildSettings
from jerrythomas.config.preview import PreviewStage
from jerrythomas.config.observability import ObservabilityConfig
from jerrythomas.config.profiles.base import Profile, ProfileCommand
from jerrythomas.config.profiles.build import (
    ARTIFACT_MODE_ADAPTER,
    ArtifactMode,
    BuildProfile,
)
from jerrythomas.config.profiles.defaults import (
    BuildProfileDefaults,
    InspectProfileDefaults,
    MaterializeProfileDefaults,
    ProfileDefaults,
    ServeProfileDefaults,
)
from jerrythomas.config.profiles.inspect import InspectProfile
from jerrythomas.config.profiles.materialize import MaterializeProfile
from jerrythomas.config.profiles.output import ServeOutputConfig
from jerrythomas.config.profiles.serve import ServeProfile
from jerrythomas.execution.settings import (
    CommandObservability,
    resolve_execution_log_outputs,
    resolve_observability_settings,
)
from jerrythomas.io.output import OutputResolutionError
from jerrythomas.io.runs import RunPaths
from jerrythomas.profiles.destinations import validate_command_destinations
from jerrythomas.profiles.errors import ProfileCommandError
from jerrythomas.profiles.loader import (
    apply_profile_defaults,
    bind_profile,
    profile_specs_with_defaults,
)
from jerrythomas.profiles.materialize import (
    resolve_materialize_jobs,
    materialize_reserved_paths,
)
from jerrythomas.profiles.models import (
    BuildJob,
    BuildRunRequest,
    MaterializeRunRequest,
    RuntimeJob,
    RuntimeRunRequest,
    ServeRunPlan,
)
from jerrythomas.profiles.runtime_profiles import (
    resolve_inspect_profiles,
    resolve_serve_profiles,
)
from jerrythomas.services.definitions import ProjectDefinition
from jerrythomas.services.path_policy import sanitize_path_segment
from jerrythomas.services.project_definition import load_project_definition
from jerrythomas.services.runtime_compiler import compile_runtime


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
        raise ProfileCommandError(
            "Failed to load project definition: "
            f"{_validation_error_without_inputs(exc)}"
        ) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise ProfileCommandError(f"Failed to load project definition: {exc}") from exc


def _select_profiles(
    definition: ProjectDefinition,
    command: ProfileCommand,
    profile_name: str | None,
) -> tuple[list[Profile], ProfileDefaults]:
    try:
        profiles, defaults = profile_specs_with_defaults(
            definition.project,
            cmd=command,
        )
    except ValidationError as exc:
        raise ProfileCommandError(
            f"Failed to load {command} profiles: "
            f"{_validation_error_without_inputs(exc)}"
        ) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise ProfileCommandError(f"Failed to load {command} profiles: {exc}") from exc
    if not profiles:
        raise ProfileCommandError(f"Project does not define {command} profiles.")
    if profile_name is None:
        selected = [profile for profile in profiles if profile.enabled]
    else:
        normalized_name = profile_name.strip()
        if not normalized_name:
            raise ProfileCommandError(
                f"{command.capitalize()} profile name must not be empty."
            )
        selected = [profile for profile in profiles if profile.name == normalized_name]
        if not selected:
            raise ProfileCommandError(f"Unknown {command} profile '{normalized_name}'")
    try:
        return (
            [
                bind_profile(apply_profile_defaults(profile, defaults), definition)
                for profile in selected
            ],
            defaults,
        )
    except ValidationError as exc:
        raise ProfileCommandError(_validation_error_without_inputs(exc)) from exc
    except ValueError as exc:
        raise ProfileCommandError(str(exc)) from exc


def _artifact_mode(configured_mode: str | None, cli_mode: str | None) -> ArtifactMode:
    try:
        return ARTIFACT_MODE_ADAPTER.validate_python(
            cli_mode if cli_mode is not None else configured_mode or "auto"
        )
    except ValueError as exc:
        raise ProfileCommandError(f"Invalid artifact mode: {exc}") from exc


def _prerequisite_settings(
    definition: ProjectDefinition,
    command: str,
    configured_mode: str | None,
    cli_mode: str | None,
    observability: ObservabilityConfig | None,
    command_observability: CommandObservability,
    execution_dir: Path,
) -> BuildSettings:
    mode = _artifact_mode(configured_mode, cli_mode)
    try:
        if observability is not None:
            logging = observability.logging
            if logging is not None and any(
                token in (output.path or "")
                for output in logging.outputs or ()
                for token in ("${dataset_id}", "${dataset_version}")
            ):
                raise ValueError(
                    "Shared prerequisite logs have no selected dataset. Put "
                    "dataset-specific log paths in concrete profiles instead of "
                    f"{command}.defaults.yaml."
                )
            observability = ObservabilityConfig.model_validate(
                definition.project.resolve_config(observability.model_dump(mode="json"))
            )
        settings = resolve_observability_settings(
            definition.project.path, observability, command_observability
        )
    except ValueError as exc:
        raise ProfileCommandError(f"Invalid prerequisite observability: {exc}") from exc
    return BuildSettings(
        mode=mode,
        observability=replace(
            settings,
            log_output=resolve_execution_log_outputs(
                settings.log_output,
                execution_dir,
                default_path=Path("logs") / f"{command}.artifacts.log",
            ),
        ),
    )


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
                dataset_id=job.runtime.dataset_id,
            )
    return tuple(plans_by_run.values())


def build_build_run_request(
    project: str,
    profile_name: str | None = None,
    artifact_mode: str | None = None,
    command_observability: CommandObservability = CommandObservability(),
) -> BuildRunRequest | None:
    definition = _load_definition(project)
    project_path = definition.project.path
    loaded_profiles, defaults = _select_profiles(
        definition,
        "build",
        profile_name,
    )
    if not loaded_profiles:
        return None
    if not isinstance(defaults, BuildProfileDefaults):
        raise TypeError("Build profile loading returned the wrong defaults type")

    artifact_tasks_by_id = definition.artifact_graph.tasks_by_id
    runtime_task_ids = {task.id for task in definition.runtime_operations}
    build_profiles: list[BuildProfile] = []
    for profile in loaded_profiles:
        if not isinstance(profile, BuildProfile):
            raise TypeError("Build profile loading returned the wrong profile type")
        task = artifact_tasks_by_id.get(profile.operation)
        if task is None:
            if profile.operation in runtime_task_ids:
                raise ProfileCommandError(
                    f"Build profile '{profile.name}' must reference an artifact "
                    f"operation; '{profile.operation}' is a runtime operation."
                )
            raise ProfileCommandError(
                f"Build profile '{profile.name}' references unknown operation "
                f"'{profile.operation}'."
            )
        build_profiles.append(profile)

    execution_dir = _execution_root(definition.project.artifacts_root)
    build_jobs: list[BuildJob] = []
    for profile in build_profiles:
        try:
            observability = resolve_observability_settings(
                project_path,
                profile.observability,
                command_observability,
            )
        except ValueError as exc:
            raise ProfileCommandError(f"Invalid build configuration: {exc}") from exc
        build_jobs.append(
            BuildJob(
                task=artifact_tasks_by_id[profile.operation].model_copy(deep=True),
                settings=BuildSettings(
                    mode=_artifact_mode(profile.artifact_mode, artifact_mode),
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

    try:
        validate_command_destinations(
            definition.project.artifacts_root,
            (),
            [job.settings.observability.log_output for job in build_jobs],
            (),
        )
    except ValueError as exc:
        raise ProfileCommandError(f"Invalid build configuration: {exc}") from exc

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
    cli_output: ServeOutputConfig | Mapping[str, object] | None = None,
    command_observability: CommandObservability = CommandObservability(),
) -> RuntimeRunRequest | None:
    definition = _load_definition(project)
    loaded_profiles, defaults = _select_profiles(
        definition,
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
    artifact_task_ids = set(definition.artifact_graph.tasks_by_id)
    for profile in runtime_profiles:
        task = runtime_tasks_by_id.get(profile.operation)
        if task is None:
            if profile.operation in artifact_task_ids:
                raise ProfileCommandError(
                    f"{command.capitalize()} profile '{profile.name}' must reference "
                    f"a runtime operation; '{profile.operation}' is an artifact "
                    "operation."
                )
            raise ProfileCommandError(
                f"{command.capitalize()} profile '{profile.name}' references unknown "
                f"operation '{profile.operation}'."
            )

    execution_dir = _execution_root(definition.project.artifacts_root)
    artifact_settings = _prerequisite_settings(
        definition,
        command,
        defaults.artifact_mode,
        artifact_mode,
        defaults.observability,
        command_observability,
        execution_dir,
    )

    try:
        if command == "serve":
            resolved_profiles = resolve_serve_profiles(
                definition,
                serve_profiles,
                preview,
                limit,
                cli_output,
                command_observability,
            )
        else:
            if preview is not None:
                raise ValueError("Inspect profiles do not support previews.")
            resolved_profiles = resolve_inspect_profiles(
                definition,
                inspect_profiles,
                limit,
                cli_output,
                command_observability,
            )
        jobs = [
            RuntimeJob(
                name=profile.name,
                task=runtime_tasks_by_id[profile.operation_id].model_copy(deep=True),
                runtime=compile_runtime(
                    definition, runtime_tasks_by_id[profile.operation_id].dataset
                ),
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
        outputs = [output for job in jobs for output in job.configured_outputs]
        log_outputs = (
            artifact_settings.observability.log_output,
            *(job.observability.log_output for job in jobs),
        )
        serve_run_plans = _serve_run_plans(jobs) if command == "serve" else ()
        validate_command_destinations(
            definition.project.artifacts_root,
            outputs,
            log_outputs,
            [plan.paths for plan in serve_run_plans],
        )
    except OutputResolutionError as exc:
        raise ProfileCommandError(f"Invalid output configuration: {exc}") from exc
    except ValueError as exc:
        raise ProfileCommandError(str(exc)) from exc

    return RuntimeRunRequest(
        command=command,
        definition=definition,
        jobs=jobs,
        execution=defaults.execution,
        artifact_settings=artifact_settings,
        serve_run_plans=serve_run_plans,
    )


def build_materialize_run_request(
    project: str,
    profile_name: str | None,
    overwrite: bool | None,
    output: Path | None,
    artifact_mode: str | None,
    command_observability: CommandObservability = CommandObservability(),
) -> MaterializeRunRequest | None:
    definition = _load_definition(project)
    loaded_profiles, defaults = _select_profiles(
        definition,
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
            definition=definition,
            execution_dir=execution_dir,
            overwrite=overwrite,
            cli_output=output,
            command_observability=command_observability,
        )
        artifact_settings = _prerequisite_settings(
            definition,
            "materialize",
            defaults.artifact_mode,
            artifact_mode,
            defaults.observability,
            command_observability,
            execution_dir,
        )
        log_outputs = (
            artifact_settings.observability.log_output,
            *(job.observability.log_output for job in jobs),
        )
        validate_command_destinations(
            definition.project.artifacts_root,
            [job.output for job in jobs],
            log_outputs,
            (),
            reserved_paths=materialize_reserved_paths(jobs),
        )
        runtime = compile_runtime(definition)
    except (OSError, TypeError, ValueError) as exc:
        raise ProfileCommandError(f"Invalid materialize configuration: {exc}") from exc
    return MaterializeRunRequest(
        definition=definition,
        jobs=jobs,
        execution=defaults.execution,
        artifact_settings=artifact_settings,
        runtime=runtime,
    )
