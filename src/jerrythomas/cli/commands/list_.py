from pathlib import Path

from jerrythomas.cli.workspace import WorkspaceContext, resolve_default_project_yaml
from jerrythomas.services.project import load_project
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
    workspace: WorkspaceContext | None = None,
) -> None:
    if subcmd == "sources":
        proj_path = resolve_default_project_yaml(workspace)
        if proj_path is None:
            root_dir, _, _ = pkg_root(plugin_root)
            proj_path = default_project_yaml_path(root_dir)
        try:
            streams = load_streams(load_project(proj_path))
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from None
        aliases = sorted(streams.sources)
        for alias in aliases:
            print(alias)
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
