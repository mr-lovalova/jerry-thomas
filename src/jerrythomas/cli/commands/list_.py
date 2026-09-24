from pathlib import Path

from jerrythomas.config.profiles.base import OperationProfile
from jerrythomas.cli.workspace import WorkspaceContext, resolve_default_project_yaml
from jerrythomas.services.project import load_project
from jerrythomas.services.project_definition import load_project_definition
from jerrythomas.profiles.loader import (
    PROFILE_KINDS,
    profile_specs_with_defaults,
    apply_profile_defaults,
)
from jerrythomas.services.scaffold.discovery import (
    list_combiners,
    list_domains,
    list_dtos,
    list_loaders,
    list_mappers,
    list_parsers,
)
from jerrythomas.services.scaffold.paths import default_project_yaml_path, pkg_root
from jerrythomas.services.streams.loader import load_streams


def handle(
    subcmd: str,
    *,
    plugin_root: Path | None = None,
    project: str | None = None,
    workspace: WorkspaceContext | None = None,
) -> None:
    if subcmd in {"datasets", "streams", "profiles", "sources"}:
        project_path = (
            Path(project)
            if project is not None
            else resolve_default_project_yaml(workspace)
        )
        if project_path is None:
            root_dir, _, _ = pkg_root(plugin_root)
            project_path = default_project_yaml_path(root_dir)
        try:
            if subcmd in {"sources", "streams"}:
                streams = load_streams(load_project(project_path))
                entries = streams.sources if subcmd == "sources" else streams.streams
                for entry in sorted(entries):
                    print(entry)
            else:
                definition = load_project_definition(project_path)
                if subcmd == "datasets":
                    for dataset_id, dataset in sorted(definition.datasets.items()):
                        print(f"{dataset_id}\tversion={dataset.version}")
                else:
                    operations = {
                        **definition.artifact_graph.tasks_by_id,
                        **{task.id: task for task in definition.runtime_operations},
                    }
                    for command in PROFILE_KINDS:
                        profiles, defaults = profile_specs_with_defaults(
                            definition.project, command
                        )
                        for profile in profiles:
                            profile = apply_profile_defaults(profile, defaults)
                            assert isinstance(profile, OperationProfile)
                            task = operations.get(profile.operation)
                            binding = ""
                            if task is not None:
                                bound_dataset = task.dataset
                                stream_id = getattr(task, "stream", None)
                                if bound_dataset is not None:
                                    binding = f"\tdataset={bound_dataset}"
                                elif stream_id is not None:
                                    binding = f"\tstream={stream_id}"
                            enabled = str(profile.enabled).lower()
                            print(
                                f"{command}.{profile.name}\toperation={profile.operation}{binding}\tenabled={enabled}"
                            )
        except (OSError, TypeError, ValueError) as exc:
            raise SystemExit(str(exc)) from None
    elif subcmd == "domains":
        for k in list_domains(root=plugin_root):
            print(k)
    elif subcmd == "parsers":
        for k in sorted(list_parsers(root=plugin_root).keys()):
            print(k)
    elif subcmd == "mappers":
        for k in sorted(list_mappers(root=plugin_root).keys()):
            print(k)
    elif subcmd == "combiners":
        for k in sorted(list_combiners(root=plugin_root).keys()):
            print(k)
    elif subcmd == "loaders":
        for k in sorted(list_loaders(root=plugin_root).keys()):
            print(k)
    elif subcmd == "dtos":
        for k in sorted(list_dtos(root=plugin_root).keys()):
            print(k)
