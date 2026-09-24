import logging
from pathlib import Path

from jerrythomas.cli.prompts import (
    choose_name,
    pick_from_list,
    pick_from_menu,
    pick_multiple_from_list,
    prompt_required,
)
from jerrythomas.cli.workspace import WorkspaceContext, resolve_scaffold_project_yaml
from jerrythomas.services.project import load_project
from jerrythomas.services.scaffold.discovery import (
    list_combiners,
    list_mappers,
    list_sources,
    list_streams,
)
from jerrythomas.services.scaffold.layout import (
    default_stream_id_for_source,
    source_id_parts,
)
from jerrythomas.services.scaffold.paths import default_project_yaml_path, pkg_root
from jerrythomas.services.scaffold.stream_yaml import (
    write_aligned_stream,
    write_broadcast_stream,
    write_source_stream,
)
from jerrythomas.services.streams.loader import load_streams
from jerrythomas.services.streams.validation import stream_partition_by

logger = logging.getLogger(__name__)


def _select_source_mapper(root: Path | None) -> str:
    mappers = list_mappers(root=root)
    options: list[tuple[str, str]] = []
    if mappers:
        options.append(("existing", "Select existing mapper (default)"))
    options.append(("identity", "Identity mapper"))
    options.append(("custom", "Custom mapper"))

    choice = pick_from_menu("Mapper:", options)
    if choice == "existing":
        entrypoint = pick_from_menu(
            "Select mapper entrypoint:",
            [(key, key) for key in sorted(mappers)],
        )
        return entrypoint
    if choice == "identity":
        return "identity"
    if choice == "custom":
        return prompt_required("Mapper entrypoint")
    raise ValueError(f"Unknown mapper choice: {choice}")


def _select_combiner(root: Path | None) -> str:
    combiners = list_combiners(root=root)
    if not combiners:
        return prompt_required("Combine entrypoint")

    choice = pick_from_menu(
        "Combiner:",
        [
            ("existing", "Select existing combiner"),
            ("custom", "Custom combiner"),
        ],
    )
    if choice == "existing":
        return pick_from_menu(
            "Select combiner entrypoint:",
            [(key, key) for key in sorted(combiners)],
        )
    if choice == "custom":
        return prompt_required("Combine entrypoint")
    raise ValueError(f"Unknown combiner choice: {choice}")


def handle(
    *,
    plugin_root: Path | None = None,
    use_identity: bool = False,
    workspace: WorkspaceContext | None = None,
    project: str | None = None,
) -> None:
    project_yaml = resolve_scaffold_project_yaml(project, workspace)
    if project_yaml is None:
        root_dir, _package_name, _pyproject = pkg_root(plugin_root)
        project_yaml = default_project_yaml_path(root_dir)
    stream_type = pick_from_menu(
        "Stream type:",
        [
            ("source", "Source-backed (source → ordered stream)"),
            ("aligned", "Aligned (streams → ordered stream)"),
            (
                "broadcast",
                "Broadcast (partitioned stream + global stream → ordered stream)",
            ),
        ],
    )
    if use_identity and stream_type != "source":
        raise SystemExit("--identity is only supported for source-backed streams.")
    if stream_type == "aligned":
        _scaffold_aligned_stream(
            plugin_root=plugin_root,
            project_yaml=project_yaml,
        )
        return
    if stream_type == "broadcast":
        _scaffold_broadcast_stream(
            plugin_root=plugin_root,
            project_yaml=project_yaml,
        )
        return

    source_options = list_sources(project_yaml)
    if not source_options:
        raise SystemExit("No sources found. Create one first (jerry source create ...)")

    src_key = pick_from_list("Select source:", source_options)
    _provider, dataset, _source_variant = source_id_parts(src_key)
    if not dataset:
        raise SystemExit(
            "Source alias must be 'provider.dataset[.variant]' (from source file's id)"
        )

    if use_identity:
        mapper_entrypoint = "identity"
    else:
        mapper_entrypoint = _select_source_mapper(plugin_root)

    stream_id = choose_name(
        "Stream id",
        default_stream_id_for_source(dataset, src_key),
    )

    try:
        path = write_source_stream(
            project_yaml=project_yaml,
            stream_id=stream_id,
            source=src_key,
            mapper_entrypoint=mapper_entrypoint,
        )
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    logger.info("Stream: %s", path)


def _scaffold_aligned_stream(
    plugin_root: Path | None,
    project_yaml: Path,
) -> None:
    streams = list_streams(project_yaml)
    if len(streams) < 2:
        raise SystemExit("Aligned streams require at least two input streams.")
    input_streams = pick_multiple_from_list(
        "Select at least two input streams (comma-separated numbers):",
        streams,
    )
    if len(set(input_streams)) < 2:
        raise SystemExit("Aligned streams require at least two distinct input streams.")

    stream_id = choose_name("Stream id")
    combine_entrypoint = _select_combiner(plugin_root)

    try:
        path = write_aligned_stream(
            project_yaml=project_yaml,
            stream_id=stream_id,
            input_streams=input_streams,
            combine_entrypoint=combine_entrypoint,
        )
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    logger.info("Stream: %s", path)


def _scaffold_broadcast_stream(
    plugin_root: Path | None,
    project_yaml: Path,
) -> None:
    streams = load_streams(load_project(project_yaml)).streams
    primary_streams: list[str] = []
    broadcast_streams: list[str] = []
    for stream_id in sorted(streams):
        if stream_partition_by(streams, stream_id):
            primary_streams.append(stream_id)
        else:
            broadcast_streams.append(stream_id)
    if not primary_streams:
        raise SystemExit(
            "No partitioned streams found. Broadcast streams require a primary "
            "stream with non-empty partition_by."
        )
    if not broadcast_streams:
        raise SystemExit(
            "No global streams found. Broadcast streams require an input stream "
            "with empty partition_by."
        )

    primary_stream = pick_from_list(
        "Select primary stream (partitioned):",
        primary_streams,
    )
    broadcast_stream = pick_from_list(
        "Select broadcast stream (global):",
        broadcast_streams,
    )
    stream_id = choose_name("Stream id")
    combine_entrypoint = _select_combiner(plugin_root)

    try:
        path = write_broadcast_stream(
            project_yaml=project_yaml,
            stream_id=stream_id,
            primary_stream=primary_stream,
            broadcast_stream=broadcast_stream,
            combine_entrypoint=combine_entrypoint,
        )
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    logger.info("Stream: %s", path)
