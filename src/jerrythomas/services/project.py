from pathlib import Path
from types import MappingProxyType

from jerrythomas.config.interpolation import coalesce_missing_interpolation
from jerrythomas.config.project import PROJECT_SCHEMA_VERSION, ProjectConfig
from jerrythomas.services.config_refs import (
    interpolate_config_vars,
    merged_project_env,
    project_vars_from_data,
    resolve_config_refs,
)
from jerrythomas.services.definitions import ProjectManifest
from jerrythomas.services.path_policy import resolve_project_path
from jerrythomas.io.yaml import read_yaml_document


def _config_roots(project_yaml: Path, value: str | list[str]) -> tuple[Path, ...]:
    paths = value if isinstance(value, list) else [value]
    return tuple(resolve_project_path(project_yaml, path) for path in paths)


def load_project(project_yaml: Path) -> ProjectManifest:
    path = project_yaml.resolve()
    document = read_yaml_document(path)
    if "schema_version" not in document.data:
        raise ValueError(
            f"Project config requires schema_version: {PROJECT_SCHEMA_VERSION}."
        )
    schema_version = document.data["schema_version"]
    if type(schema_version) is not int:
        raise TypeError(
            "Project schema_version must be the integer "
            f"{PROJECT_SCHEMA_VERSION}, got {schema_version!r}."
        )
    if schema_version != PROJECT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported project schema version {schema_version!r}; "
            f"expected {PROJECT_SCHEMA_VERSION}."
        )
    environment = merged_project_env(path)
    data = resolve_config_refs(document.data, project_yaml=path, env=environment)
    variables = project_vars_from_data(data)
    raw_globals = data.get("globals")
    if isinstance(raw_globals, dict):
        for bound in ("start_time", "end_time"):
            if isinstance(raw_globals.get(bound), str):
                raw_globals[bound] = coalesce_missing_interpolation(variables[bound])
    raw_paths = data.get("paths")
    if isinstance(raw_paths, dict):
        data["paths"] = interpolate_config_vars(raw_paths, variables)
    config = ProjectConfig.model_validate(data)

    operations_dir = (
        resolve_project_path(path, config.paths.operations)
        if config.paths.operations is not None
        else None
    )
    return ProjectManifest(
        path=path,
        config=config,
        variables=MappingProxyType(variables),
        environment=MappingProxyType(environment),
        stream_dirs=_config_roots(path, config.paths.streams),
        source_dirs=_config_roots(path, config.paths.sources),
        dataset_path=resolve_project_path(path, config.paths.dataset),
        artifacts_root=resolve_project_path(path, config.paths.artifacts),
        operations_dir=operations_dir,
        profiles_dir=resolve_project_path(path, config.paths.profiles),
    )
