import errno
import keyword
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path


def ensure_pkg_dir(base: Path, name: str) -> Path:
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").touch(exist_ok=True)
    return path


def is_python_identifier(name: str) -> bool:
    return bool(name and name.isidentifier() and not keyword.iskeyword(name))


def write_new_file(path: Path, text: str) -> None:
    file = path.open("x", encoding="utf-8")
    try:
        with file:
            file.write(text)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _absent_scaffold_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    """Capture requested paths and missing parents that this scaffold may create."""

    absent: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        current = path
        while not current.exists() and not current.is_symlink():
            if current in seen:
                break
            absent.append(current)
            seen.add(current)
            current = current.parent
    return tuple(absent)


def _remove_scaffold_paths(paths: Iterable[Path]) -> None:
    """Remove captured paths only when files are new or directories are empty."""

    for path in sorted(set(paths), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise


@contextmanager
def rollback_new_scaffold_paths(paths: Iterable[Path]) -> Iterator[None]:
    """Remove files and empty directories first created inside this context."""

    rollback_paths = _absent_scaffold_paths(paths)
    try:
        yield
    except BaseException:
        _remove_scaffold_paths(rollback_paths)
        raise
