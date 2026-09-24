from dataclasses import dataclass
from pathlib import Path

from jerrythomas.config.workspace import WorkspaceConfig
from jerrythomas.services.path_policy import resolve_workspace_path, workspace_cwd
from jerrythomas.io.yaml import load_yaml


@dataclass(frozen=True)
class WorkspaceContext:
    file_path: Path
    config: WorkspaceConfig

    @property
    def root(self) -> Path:
        return self.file_path.parent

    def resolve_path(self, raw_path: str | Path) -> Path:
        return resolve_workspace_path(raw_path, self.root)

    def resolve_project_alias(self, alias: str) -> Path | None:
        raw = self.config.projects.get(alias)
        if raw is None:
            return None
        candidate = self.resolve_path(raw)
        if candidate.is_dir():
            candidate = candidate / "project.yaml"
        return candidate.resolve()

    def resolve_plugin_root(self) -> Path | None:
        if self.config.plugin_root is None:
            return None
        return self.resolve_path(self.config.plugin_root)


def load_workspace_context(start_dir: Path | None = None) -> WorkspaceContext | None:
    directory = (start_dir or workspace_cwd()).resolve()
    for path in (directory, *directory.parents):
        candidate = path / "jerry.yaml"
        if candidate.is_file():
            config = WorkspaceConfig.model_validate(load_yaml(candidate))
            return WorkspaceContext(file_path=candidate, config=config)
    return None


def resolve_default_project_yaml(workspace: WorkspaceContext | None) -> Path | None:
    if workspace is None or workspace.config.default_project is None:
        return None
    alias = workspace.config.default_project
    resolved = workspace.resolve_project_alias(alias)
    if resolved is None:
        raise SystemExit(
            f"Unknown default_project '{alias}'. Define it under projects: in jerry.yaml."
        )
    return resolved
