from pathlib import Path

import tomlkit
from tomlkit.items import AbstractTable, InlineTable
from tomlkit.toml_document import TOMLDocument

from jerrythomas.io.sinks.files import AtomicTextFileSink
from jerrythomas.services.scaffold.locking import (
    ScaffoldLock,
    acquire_scaffold_lock,
)


def _write_document(pyproject: Path, document: TOMLDocument, original: str) -> None:
    text = tomlkit.dumps(document)
    if "\r\n" in original and original.count("\n") == original.count("\r\n"):
        text = text.replace("\r\n", "\n").replace("\n", "\r\n")

    sink = AtomicTextFileSink(pyproject.resolve())
    try:
        sink.write_text(text)
        sink.close()
    except BaseException:
        sink.abort()
        raise


def _register_entry_point(
    pyproject: Path,
    group: str,
    key: str,
    target: str,
) -> bool:
    original = pyproject.read_bytes().decode("utf-8")
    document = tomlkit.parse(original)

    project = document.get("project")
    if not isinstance(project, AbstractTable):
        raise ValueError("pyproject.toml must define a [project] table")

    entry_points = project.get("entry-points")
    if entry_points is not None and not isinstance(entry_points, AbstractTable):
        raise ValueError("project.entry-points must be a TOML table")

    entries = entry_points.get(group) if entry_points is not None else None
    if entries is not None and not isinstance(entries, AbstractTable):
        raise ValueError(f"project.entry-points.{group} must be a TOML table")

    if (
        entries is None
        and not isinstance(project, InlineTable)
        and not isinstance(entry_points, InlineTable)
    ):
        # Append new tables so comments on the next existing table stay attached.
        entries = tomlkit.table()
        entries[key] = target
        entry_point_update = tomlkit.table(is_super_table=True)
        entry_point_update.append(group, entries)
        project_update = tomlkit.table(is_super_table=True)
        project_update.append("entry-points", entry_point_update)
        document.append("project", project_update)
        _write_document(pyproject, document, original)
        return True

    if entry_points is None:
        entry_points = tomlkit.inline_table()
        project["entry-points"] = entry_points

    if entries is None:
        entries = (
            tomlkit.inline_table()
            if isinstance(entry_points, InlineTable)
            else tomlkit.table()
        )
        entry_points[group] = entries

    for entry_key, entry_target in entries.items():
        if not isinstance(entry_target, str):
            raise ValueError(f"Entry point '{entry_key}' in {group} must be a string")

    if entries.get(key) == target:
        return False

    if isinstance(entries, InlineTable) and key not in entries:
        # TOMLKit cannot append a separator to an inline table parsed from disk.
        updated_entries = tomlkit.inline_table()
        for entry_key, entry_target in entries.items():
            updated_entries.append(entry_key, entry_target)
        updated_entries.append(key, target)
        entry_points[group] = updated_entries
    else:
        entries[key] = target
    _write_document(pyproject, document, original)
    return True


def register_entry_point(
    pyproject: Path,
    group: str,
    key: str,
    target: str,
    scaffold_lock: ScaffoldLock | None = None,
) -> bool:
    resolved = pyproject.resolve()
    with acquire_scaffold_lock(resolved.parent, scaffold_lock):
        return _register_entry_point(resolved, group, key, target)


def read_entry_points(pyproject: Path, group: str) -> dict[str, str]:
    document = tomlkit.parse(pyproject.read_bytes().decode("utf-8"))

    project = document.get("project")
    if project is None:
        raise ValueError("pyproject.toml must define a [project] table")
    if not isinstance(project, AbstractTable):
        raise ValueError("pyproject.toml project must be a TOML table")

    entry_points = project.get("entry-points")
    if entry_points is None:
        return {}
    if not isinstance(entry_points, AbstractTable):
        raise ValueError("project.entry-points must be a TOML table")

    entries = entry_points.get(group)
    if entries is None:
        return {}
    if not isinstance(entries, AbstractTable):
        raise ValueError(f"project.entry-points.{group} must be a TOML table")

    result: dict[str, str] = {}
    for key, target in entries.items():
        if not isinstance(target, str):
            raise ValueError(f"Entry point '{key}' in {group} must be a string")
        result[key] = target
    return result
