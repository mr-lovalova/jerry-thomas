from pathlib import Path

from jerrythomas.services.scaffold.layout import DIR_DTOS, TPL_DTO, to_snake
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


def dto_scaffold_paths(name: str, root: Path | None) -> tuple[Path, ...]:
    root_dir, pkg_name, _ = pkg_root(root)
    base = resolve_base_pkg_dir(root_dir, pkg_name)
    dtos_dir = base / DIR_DTOS
    return (
        base / "__init__.py",
        dtos_dir / "__init__.py",
        dtos_dir / f"{to_snake(name)}.py",
    )


def validate_dto_creation(name: str, root: Path | None) -> None:
    if not is_python_identifier(name):
        raise ValueError("DTO name must be a valid Python identifier")
    path = dto_scaffold_paths(name, root)[-1]
    if path.exists():
        raise FileExistsError(f"{path} already exists")


def create_dto(
    name: str,
    root: Path | None,
    scaffold_lock: ScaffoldLock | None = None,
) -> Path:
    root_dir, pkg_name, pyproject = pkg_root(root)
    with acquire_scaffold_lock(pyproject.parent, scaffold_lock):
        validate_dto_creation(name, root)
        created_paths = dto_scaffold_paths(name, root)
        with rollback_new_scaffold_paths(created_paths):
            base = ensure_base_pkg_dir(root_dir, pkg_name)
            dtos_dir = ensure_pkg_dir(base, DIR_DTOS)
            path = dtos_dir / f"{to_snake(name)}.py"
            write_new_file(
                path,
                render(
                    TPL_DTO,
                    CLASS_NAME=name,
                    DOMAIN=name,
                ),
            )
            return path
