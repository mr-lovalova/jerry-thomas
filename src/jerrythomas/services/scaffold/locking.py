from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from jerrythomas.services.execution_lock import exclusive_file_lock

_LOCK_FILE = ".jerry-scaffold.lock"


class ScaffoldLock:
    def __init__(self, path: Path):
        self.path = path
        self._active = True

    def _release(self) -> None:
        self._active = False

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("Scaffold lock is no longer active")


@contextmanager
def acquire_scaffold_lock(
    directory: Path,
    held_lock: ScaffoldLock | None = None,
) -> Iterator[ScaffoldLock]:
    """Serialize scaffold mutations owned by one directory."""

    path = directory.resolve() / _LOCK_FILE
    if held_lock is not None and held_lock.path == path:
        held_lock._require_active()
        yield held_lock
        return

    with exclusive_file_lock(path):
        lock = ScaffoldLock(path)
        try:
            yield lock
        finally:
            lock._release()
