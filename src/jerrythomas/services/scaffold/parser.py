from pathlib import Path

from jerrythomas.plugins import PARSERS_EP
from jerrythomas.services.scaffold.entrypoints import (
    read_entry_points,
    register_entry_point,
)
from jerrythomas.services.scaffold.layout import (
    DIR_PARSERS,
    TPL_PARSER,
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


def parser_scaffold_paths(name: str, root: Path | None) -> tuple[Path, ...]:
    root_dir, pkg_name, _ = pkg_root(root)
    base = resolve_base_pkg_dir(root_dir, pkg_name)
    parsers_dir = base / DIR_PARSERS
    return (
        base / "__init__.py",
        parsers_dir / "__init__.py",
        parsers_dir / f"{to_snake(name)}.py",
    )


def validate_parser_creation(name: str, root: Path | None) -> None:
    if not is_python_identifier(name):
        raise ValueError("Parser name must be a valid Python identifier")
    _, _, pyproject = pkg_root(root)
    entrypoint = ep_key_from_name(name)
    if entrypoint in read_entry_points(pyproject, PARSERS_EP):
        raise FileExistsError(f"Parser entry point '{entrypoint}' already exists")
    path = parser_scaffold_paths(name, root)[-1]
    if path.exists():
        raise FileExistsError(f"{path} already exists")


def create_parser(
    *,
    name: str,
    dto_class: str,
    dto_module: str,
    root: Path | None,
    scaffold_lock: ScaffoldLock | None = None,
) -> str:
    root_dir, pkg_name, pyproject = pkg_root(root)
    with acquire_scaffold_lock(pyproject.parent, scaffold_lock) as lock:
        validate_parser_creation(name, root)
        created_paths = parser_scaffold_paths(name, root)
        base = created_paths[0].parent
        path = created_paths[-1]
        module_name = path.stem
        content = render(
            TPL_PARSER,
            CLASS_NAME=name,
            DTO_CLASS=dto_class,
            DTO_IMPORT=dto_module,
        )
        with rollback_new_scaffold_paths(created_paths):
            ensure_base_pkg_dir(root_dir, pkg_name)
            ensure_pkg_dir(base, DIR_PARSERS)
            write_new_file(path, content)
            ep_key = ep_key_from_name(name)
            register_entry_point(
                pyproject,
                PARSERS_EP,
                ep_key,
                f"{base.name}.parsers.{module_name}:{name}",
                scaffold_lock=lock,
            )
            return ep_key
