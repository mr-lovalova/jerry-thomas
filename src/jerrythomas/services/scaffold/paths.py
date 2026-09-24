import sys
from pathlib import Path

from jerrythomas.config.project import PROJECT_SCHEMA_VERSION
from jerrythomas.services.definitions import ProjectManifest
from jerrythomas.services.path_policy import workspace_cwd
from jerrythomas.services.project import load_project
from jerrythomas.services.scaffold.locking import ScaffoldLock, acquire_scaffold_lock
from jerrythomas.services.scaffold.utils import write_new_file

_DEFAULT_DOTENV_EXAMPLE = (
    "# Copy this file to .env next to project.yaml for local project secrets.\n"
    "RAW_ROOT=\n"
)


def pkg_root(start: Path | None = None) -> tuple[Path, str, Path]:
    here = start or workspace_cwd()
    for d in [here, *here.parents]:
        pyproject = d / "pyproject.toml"
        if pyproject.exists():
            pkg_name = d.name
            src_dir = d / "src"
            if src_dir.exists():
                candidates = [
                    p
                    for p in src_dir.iterdir()
                    if p.is_dir() and (p / "__init__.py").exists()
                ]
                if len(candidates) == 1:
                    pkg_name = candidates[0].name
            return d, pkg_name, pyproject
    print(
        "[error] pyproject.toml not found (searched current and parent dirs)",
        file=sys.stderr,
    )
    raise SystemExit(1)


def resolve_base_pkg_dir(root_dir: Path, pkg_name: str) -> Path:
    preferred = root_dir / "src" / pkg_name
    if preferred.exists():
        return preferred
    src_dir = root_dir / "src"
    if src_dir.exists():
        candidates = [
            p for p in src_dir.iterdir() if p.is_dir() and (p / "__init__.py").exists()
        ]
        if len(candidates) == 1:
            return candidates[0]
    return preferred


def ensure_base_pkg_dir(root_dir: Path, pkg_name: str) -> Path:
    path = resolve_base_pkg_dir(root_dir, pkg_name)
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").touch(exist_ok=True)
    return path


def default_project_yaml_path(plugin_root: Path) -> Path:
    """Return the project path created by the standard plugin scaffold."""
    return plugin_root / "your-project" / "project.yaml"


def ensure_project_scaffold(
    project_yaml: Path,
    scaffold_lock: ScaffoldLock | None = None,
) -> ProjectManifest:
    """Create a minimal project and its configured directories when absent."""
    with acquire_scaffold_lock(project_yaml.parent, scaffold_lock):
        if not project_yaml.exists():
            project_yaml.parent.mkdir(parents=True, exist_ok=True)
            write_new_file(
                project_yaml,
                f"schema_version: {PROJECT_SCHEMA_VERSION}\n"
                "artifact_revision: 1\n"
                "name: default\n"
                "paths:\n"
                "  streams: ./streams\n"
                "  sources: ./sources\n"
                "  datasets: ./datasets\n"
                "  artifacts: ../artifacts/default\n"
                "  profiles: ./profiles\n"
                "globals:\n"
                "  start_time: 2021-01-01T00:00:00Z\n"
                "  end_time: 2021-12-31T23:00:00Z\n",
            )

        project = load_project(project_yaml)
        for streams in project.stream_dirs:
            streams.mkdir(parents=True, exist_ok=True)
        for sources in project.source_dirs:
            sources.mkdir(parents=True, exist_ok=True)
        for datasets in project.dataset_dirs:
            datasets.mkdir(parents=True, exist_ok=True)
        if project.operations_dir is not None:
            project.operations_dir.mkdir(parents=True, exist_ok=True)
        project.profiles_dir.mkdir(parents=True, exist_ok=True)

        dotenv_example = project_yaml.parent / ".env.example"
        if not dotenv_example.exists():
            write_new_file(dotenv_example, _DEFAULT_DOTENV_EXAMPLE)
        return project
