from pathlib import Path

from jerrythomas.plugins import LOADERS_EP
from jerrythomas.services.scaffold.entrypoints import (
    read_entry_points,
    register_entry_point,
)
from jerrythomas.services.scaffold.layout import (
    DIR_LOADERS,
    TPL_LOADER_BASIC,
    ep_key_from_name,
    loader_class_name,
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


def loader_scaffold_paths(name: str, root: Path | None) -> tuple[Path, ...]:
    root_dir, pkg_name, _ = pkg_root(root)
    base = resolve_base_pkg_dir(root_dir, pkg_name)
    loaders_dir = base / DIR_LOADERS
    return (
        base / "__init__.py",
        loaders_dir / "__init__.py",
        loaders_dir / f"{to_snake(name)}.py",
    )


def validate_loader_creation(name: str, root: Path | None) -> None:
    if not is_python_identifier(name):
        raise ValueError("Loader name must be a valid Python identifier")
    _, _, pyproject = pkg_root(root)
    entrypoint = ep_key_from_name(name)
    if entrypoint in read_entry_points(pyproject, LOADERS_EP):
        raise FileExistsError(f"Loader entry point '{entrypoint}' already exists")
    path = loader_scaffold_paths(name, root)[-1]
    if path.exists():
        raise FileExistsError(f"{path} already exists")


def create_loader(
    name: str,
    root: Path | None,
    scaffold_lock: ScaffoldLock | None = None,
) -> str:
    root_dir, pkg_name, pyproject = pkg_root(root)
    with acquire_scaffold_lock(pyproject.parent, scaffold_lock) as lock:
        validate_loader_creation(name, root)
        created_paths = loader_scaffold_paths(name, root)
        base = created_paths[0].parent
        path = created_paths[-1]
        module_name = path.stem
        class_name = loader_class_name(name)
        content = render(TPL_LOADER_BASIC, CLASS_NAME=class_name)
        with rollback_new_scaffold_paths(created_paths):
            ensure_base_pkg_dir(root_dir, pkg_name)
            ensure_pkg_dir(base, DIR_LOADERS)
            write_new_file(path, content)
            ep_key = ep_key_from_name(name)
            register_entry_point(
                pyproject,
                LOADERS_EP,
                ep_key,
                f"{base.name}.loaders.{module_name}:{class_name}",
                scaffold_lock=lock,
            )
            return ep_key
