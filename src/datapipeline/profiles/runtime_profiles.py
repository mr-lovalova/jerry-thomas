from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from datapipeline.config.dataset.split import split_output_ids
from datapipeline.config.preview import PreviewStage
from datapipeline.config.profiles.inspect import InspectProfile
from datapipeline.config.profiles.output import ServeOutputConfig
from datapipeline.config.profiles.serve import ServeProfile
from datapipeline.config.tasks.base import RuntimeTask
from datapipeline.config.tasks.dataset import DatasetTask
from datapipeline.execution.settings import (
    CommandObservability,
    ObservabilitySettings,
    resolve_observability_settings,
)
from datapipeline.io.output import (
    OutputTarget,
    output_destination_key,
    resolve_output_directory,
    resolve_output_target,
)
from datapipeline.io.runs import RunPaths, get_run_paths
from datapipeline.services.definitions import ProjectDefinition


@dataclass(frozen=True)
class ResolvedRuntimeProfile:
    name: str
    operation_id: str
    preview: PreviewStage | None
    limit: int | None
    throttle_ms: float | None
    observability: ObservabilitySettings
    output: OutputTarget
    output_ids: tuple[str, ...]


def _resolve_serve_output_ids(
    definition: ProjectDefinition,
    profile: ServeProfile,
    preview: PreviewStage | None,
    operation: RuntimeTask | None,
) -> tuple[str, ...]:
    include_outputs = tuple(profile.include_outputs or ())
    if include_outputs and preview is not None:
        raise ValueError(
            f"Serve profile '{profile.name}' cannot combine preview with "
            "include_outputs."
        )

    split = definition.dataset.split
    dataset_output_ids = split_output_ids(split) if split is not None else ()
    if include_outputs:
        if split is None:
            raise ValueError(
                f"Serve profile '{profile.name}' defines include_outputs but "
                "dataset split is not configured."
            )
        unknown_ids = [
            output_id
            for output_id in include_outputs
            if output_id not in dataset_output_ids
        ]
        if unknown_ids:
            unknown = ", ".join(repr(output_id) for output_id in unknown_ids)
            raise ValueError(
                f"Serve profile '{profile.name}' includes outputs not published "
                f"by the dataset: {unknown}"
            )

    if isinstance(operation, DatasetTask) and preview is None and not include_outputs:
        return dataset_output_ids
    return include_outputs


def _validate_output_collisions(profiles: Sequence[ResolvedRuntimeProfile]) -> None:
    output_owners: dict[str, str] = {}
    for profile in profiles:
        if profile.output.destination is None:
            continue
        destinations = (
            [
                (output_id, profile.output.for_output(output_id).destination)
                for output_id in profile.output_ids
            ]
            if profile.output_ids
            else [(None, profile.output.destination)]
        )
        for output_id, destination in destinations:
            assert destination is not None
            owner = f"profile '{profile.name}'"
            if output_id is not None:
                owner = f"{owner} output {output_id!r}"
            destination_key = output_destination_key(destination)
            previous = output_owners.get(destination_key)
            if previous is not None:
                raise ValueError(
                    f"Runtime outputs for {previous} and {owner} resolve to the "
                    f"same path '{destination}'."
                )
            output_owners[destination_key] = owner


def resolve_serve_profiles(
    definition: ProjectDefinition,
    profiles: Sequence[ServeProfile],
    preview: PreviewStage | None,
    limit: int | None,
    cli_output: ServeOutputConfig | None,
    command_observability: CommandObservability = CommandObservability(),
) -> list[ResolvedRuntimeProfile]:
    project_path = definition.project.path
    runtime_operations = {
        operation.id: operation for operation in definition.runtime_operations
    }
    shared_runs: dict[Path, RunPaths] = {}
    previews_by_root: dict[Path, PreviewStage | None] = {}

    resolved: list[ResolvedRuntimeProfile] = []
    for profile in profiles:
        resolved_preview = preview if preview is not None else profile.preview
        resolved_limit = limit if limit is not None else profile.limit
        output_ids = _resolve_serve_output_ids(
            definition,
            profile,
            resolved_preview,
            runtime_operations.get(profile.operation),
        )
        serve_root = resolve_output_directory(
            cli_output or profile.output,
            base_path=project_path.parent,
        )
        run_paths = None
        if serve_root is not None:
            if (
                serve_root in previews_by_root
                and previews_by_root[serve_root] != resolved_preview
            ):
                raise ValueError(
                    "Serve profiles sharing output directory "
                    f"'{serve_root}' must use the same preview."
                )
            previews_by_root[serve_root] = resolved_preview
            run_paths = shared_runs.get(serve_root)
            if run_paths is None:
                run_paths = get_run_paths(serve_root)
                shared_runs[serve_root] = run_paths

        target = resolve_output_target(
            cli_output=cli_output,
            config_output=profile.output,
            default=None,
            base_path=project_path.parent,
            profile_name=profile.name,
            run_paths=run_paths,
        )
        if output_ids and target.transport != "fs":
            raise ValueError(
                f"Serve profile '{profile.name}' requires fs output for routed "
                "dataset outputs."
            )

        observability = resolve_observability_settings(
            project_path,
            profile.observability,
            command_observability,
        )
        resolved.append(
            ResolvedRuntimeProfile(
                name=profile.name,
                operation_id=profile.operation,
                preview=resolved_preview,
                limit=resolved_limit,
                throttle_ms=profile.throttle_ms,
                observability=observability,
                output=target,
                output_ids=output_ids,
            )
        )

    _validate_output_collisions(resolved)
    return resolved


def resolve_inspect_profiles(
    definition: ProjectDefinition,
    profiles: Sequence[InspectProfile],
    limit: int | None,
    cli_output: ServeOutputConfig | None,
    command_observability: CommandObservability = CommandObservability(),
) -> list[ResolvedRuntimeProfile]:
    project_path = definition.project.path
    resolved: list[ResolvedRuntimeProfile] = []
    for profile in profiles:
        target = resolve_output_target(
            cli_output=cli_output,
            config_output=profile.output,
            default=None,
            base_path=project_path.parent,
            profile_name=profile.name,
        )
        observability = resolve_observability_settings(
            project_path,
            profile.observability,
            command_observability,
        )
        resolved.append(
            ResolvedRuntimeProfile(
                name=profile.name,
                operation_id=profile.operation,
                preview=None,
                limit=limit,
                throttle_ms=None,
                observability=observability,
                output=target,
                output_ids=(),
            )
        )

    _validate_output_collisions(resolved)
    return resolved
