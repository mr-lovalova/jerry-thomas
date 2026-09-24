from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from jerrythomas.config.dataset.split import split_output_ids
from jerrythomas.config.preview import PreviewStage
from jerrythomas.config.profiles.inspect import InspectProfile
from jerrythomas.config.profiles.output import (
    ServeOutputConfig,
    merge_output_overrides,
)
from jerrythomas.config.profiles.serve import ServeProfile
from jerrythomas.config.tasks.base import RuntimeTask
from jerrythomas.config.tasks.dataset import DatasetTask
from jerrythomas.execution.settings import (
    CommandObservability,
    ObservabilitySettings,
    resolve_observability_settings,
)
from jerrythomas.io.output import (
    OutputTarget,
    resolve_output_directory,
    resolve_output_target,
)
from jerrythomas.io.runs import RunPaths, get_run_paths
from jerrythomas.pipelines.dataset.preview import preview_output_plan
from jerrythomas.services.definitions import ProjectDefinition


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
    operation: RuntimeTask,
) -> tuple[str, ...]:
    include_outputs = tuple(profile.include_outputs or ())
    if include_outputs and preview is not None:
        raise ValueError(
            f"Serve profile '{profile.name}' cannot combine preview with "
            "include_outputs."
        )

    dataset = (
        definition.require_dataset(operation.dataset)
        if operation.dataset is not None
        else None
    )
    split = dataset.split if dataset is not None else None
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
    if isinstance(operation, DatasetTask) and preview is not None:
        assert dataset is not None
        return tuple(
            output_id
            for output_id, _config in preview_output_plan(
                dataset.series,
                preview,
            )
        )
    return include_outputs


def _effective_cli_output(
    profile_output: ServeOutputConfig | None,
    cli_output: ServeOutputConfig | Mapping[str, object] | None,
) -> ServeOutputConfig | None:
    """Apply CLI leaves onto a profile output; complete configs pass through."""
    if cli_output is None or isinstance(cli_output, ServeOutputConfig):
        return profile_output if cli_output is None else cli_output
    return merge_output_overrides(profile_output, cli_output)


def resolve_serve_profiles(
    definition: ProjectDefinition,
    profiles: Sequence[ServeProfile],
    preview: PreviewStage | None,
    limit: int | None,
    cli_output: ServeOutputConfig | Mapping[str, object] | None,
    command_observability: CommandObservability = CommandObservability(),
) -> list[ResolvedRuntimeProfile]:
    project_path = definition.project.path
    runtime_operations = {
        operation.id: operation for operation in definition.runtime_operations
    }
    shared_runs: dict[Path, RunPaths] = {}
    identities_by_root: dict[Path, tuple[str | None, PreviewStage | None]] = {}

    resolved: list[ResolvedRuntimeProfile] = []
    for profile in profiles:
        operation = runtime_operations[profile.operation]
        resolved_preview = preview if preview is not None else profile.preview
        resolved_limit = limit if limit is not None else profile.limit
        output_ids = _resolve_serve_output_ids(
            definition,
            profile,
            resolved_preview,
            operation,
        )
        effective_output = _effective_cli_output(profile.output, cli_output)
        serve_root = resolve_output_directory(
            effective_output,
            base_path=project_path.parent,
        )
        run_paths = None
        if serve_root is not None:
            identity = (operation.dataset, resolved_preview)
            if serve_root in identities_by_root:
                previous_dataset, previous_preview = identities_by_root[serve_root]
                if previous_dataset != operation.dataset:
                    raise ValueError(
                        "Serve profiles sharing output directory "
                        f"'{serve_root}' must use the same dataset."
                    )
                if previous_preview != resolved_preview:
                    raise ValueError(
                        "Serve profiles sharing output directory "
                        f"'{serve_root}' must use the same preview."
                    )
            identities_by_root[serve_root] = identity
            run_paths = shared_runs.get(serve_root)
            if run_paths is None:
                run_paths = get_run_paths(serve_root)
                shared_runs[serve_root] = run_paths

        target = resolve_output_target(
            cli_output=effective_output,
            config_output=profile.output,
            default=None,
            base_path=project_path.parent,
            profile_name=profile.name,
            run_paths=run_paths,
        )
        if output_ids and resolved_preview is None and target.transport != "fs":
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

    return resolved


def resolve_inspect_profiles(
    definition: ProjectDefinition,
    profiles: Sequence[InspectProfile],
    limit: int | None,
    cli_output: ServeOutputConfig | Mapping[str, object] | None,
    command_observability: CommandObservability = CommandObservability(),
) -> list[ResolvedRuntimeProfile]:
    project_path = definition.project.path
    resolved: list[ResolvedRuntimeProfile] = []
    for profile in profiles:
        target = resolve_output_target(
            cli_output=_effective_cli_output(profile.output, cli_output),
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

    return resolved
