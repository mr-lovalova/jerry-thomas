import ast
from pathlib import Path

from jerrythomas.plugins import COMBINERS_EP, LOADERS_EP, MAPPERS_EP, PARSERS_EP
from jerrythomas.services.project import load_project
from jerrythomas.services.scaffold.entrypoints import read_entry_points
from jerrythomas.services.scaffold.paths import (
    find_pyproject,
    pkg_root,
    resolve_base_pkg_dir,
)
from jerrythomas.services.streams.loader import load_streams


def list_dtos(root: Path | None = None) -> dict[str, str]:
    """Return mapping of DTO class name -> module path."""
    root_dir, pkg_name, _ = pkg_root(root)
    base = resolve_base_pkg_dir(root_dir, pkg_name)
    dtos_dir = base / "dtos"
    if not dtos_dir.exists():
        return {}

    package_name = base.name
    found: dict[str, str] = {}
    for path in sorted(dtos_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        module = f"{package_name}.dtos.{path.stem}"
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and _is_dataclass(node):
                found[node.name] = module
    return found


def _is_dataclass(node: ast.ClassDef) -> bool:
    for deco in node.decorator_list:
        if isinstance(deco, ast.Name) and deco.id == "dataclass":
            return True
        if isinstance(deco, ast.Attribute) and deco.attr == "dataclass":
            return True
    return False


def _list_entry_points(root: Path | None, group: str) -> dict[str, str]:
    pyproject = find_pyproject(root)
    return read_entry_points(pyproject, group) if pyproject is not None else {}


def list_parsers(root: Path | None = None) -> dict[str, str]:
    return _list_entry_points(root, PARSERS_EP)


def list_loaders(root: Path | None = None) -> dict[str, str]:
    return _list_entry_points(root, LOADERS_EP)


def list_mappers(root: Path | None = None) -> dict[str, str]:
    return _list_entry_points(root, MAPPERS_EP)


def list_combiners(root: Path | None = None) -> dict[str, str]:
    return _list_entry_points(root, COMBINERS_EP)


def list_domains(root: Path | None = None) -> list[str]:
    root_dir, pkg_name, _ = pkg_root(root)
    base = resolve_base_pkg_dir(root_dir, pkg_name)
    dom_dir = base / "domains"
    if not dom_dir.exists():
        return []
    return sorted(
        p.name for p in dom_dir.iterdir() if p.is_dir() and (p / "model.py").exists()
    )


def list_sources(project_yaml: Path) -> list[str]:
    streams = load_streams(load_project(project_yaml))
    return sorted(streams.sources)


def list_streams(project_yaml: Path) -> list[str]:
    streams = load_streams(load_project(project_yaml))
    return sorted(streams.streams)
