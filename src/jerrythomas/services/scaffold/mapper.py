from pathlib import Path

from jerrythomas.plugins import MAPPERS_EP
from jerrythomas.services.scaffold.entrypoints import (
    read_entry_points,
    register_entry_point,
)
from jerrythomas.services.scaffold.layout import (
    DIR_MAPPERS,
    TPL_MAPPER_SOURCE,
    domain_record_class,
    ep_key_from_name,
    to_snake,
)
from jerrythomas.services.scaffold.locking import ScaffoldLock, acquire_scaffold_lock
from jerrythomas.services.scaffold.paths import (
    ensure_base_pkg_dir,
    pkg_root,
    resolve_base_pkg_dir,
)
from jerrythomas.services.scaffold.templates import render
from jerrythomas.services.scaffold.utils import (
    ensure_pkg_dir,
    is_python_identifier,
    rollback_new_scaffold_paths,
    write_new_file,
)


def mapper_scaffold_paths(name: str, root: Path | None) -> tuple[Path, ...]:
    root_dir, pkg_name, _ = pkg_root(root)
    base = resolve_base_pkg_dir(root_dir, pkg_name)
    mappers_dir = base / DIR_MAPPERS
    return (
        base / "__init__.py",
        mappers_dir / "__init__.py",
        mappers_dir / f"{to_snake(name)}.py",
    )


def validate_mapper_creation(name: str, root: Path | None) -> None:
    if not is_python_identifier(name):
        raise ValueError("Mapper name must be a valid Python identifier")
    if not is_python_identifier(to_snake(name)):
        raise ValueError(
            "Mapper name must produce a valid Python identifier after "
            "snake_case normalization"
        )
    _, _, pyproject = pkg_root(root)
    entrypoint = ep_key_from_name(name)
    if entrypoint in read_entry_points(pyproject, MAPPERS_EP):
        raise FileExistsError(f"Mapper entry point '{entrypoint}' already exists")
    path = mapper_scaffold_paths(name, root)[-1]
    if path.exists():
        raise FileExistsError(f"{path} already exists")


def create_mapper(
    *,
    name: str,
    input_class: str,
    input_module: str,
    domain: str,
    root: Path | None,
    scaffold_lock: ScaffoldLock | None = None,
) -> str:
    root_dir, pkg_name, pyproject = pkg_root(root)
    with acquire_scaffold_lock(pyproject.parent, scaffold_lock) as lock:
        validate_mapper_creation(name, root)
        created_paths = mapper_scaffold_paths(name, root)
        base = created_paths[0].parent
        path = created_paths[-1]
        module_name = path.stem
        content = render(
            TPL_MAPPER_SOURCE,
            FUNCTION_NAME=module_name,
            INPUT_CLASS=input_class,
            INPUT_IMPORT=input_module,
            DOMAIN_MODULE=f"{base.name}.domains.{domain}.model",
            DOMAIN_RECORD=domain_record_class(domain),
        )
        with rollback_new_scaffold_paths(created_paths):
            ensure_base_pkg_dir(root_dir, pkg_name)
            ensure_pkg_dir(base, DIR_MAPPERS)
            write_new_file(path, content)
            ep_key = ep_key_from_name(name)
            register_entry_point(
                pyproject,
                MAPPERS_EP,
                ep_key,
                f"{base.name}.mappers.{module_name}:{module_name}",
                scaffold_lock=lock,
            )
            return ep_key
