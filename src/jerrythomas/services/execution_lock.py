import errno
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

if sys.platform == "win32":
    import msvcrt

    def try_acquire_file_lock(lock_file: BinaryIO) -> bool:
        lock_file.seek(0)
        if not lock_file.read(1):
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        try:
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            return False
        return True

    def release_file_lock(lock_file: BinaryIO) -> None:
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def try_acquire_file_lock(lock_file: BinaryIO) -> bool:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            return False
        return True

    def release_file_lock(lock_file: BinaryIO) -> None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class ProjectExecutionBusyError(RuntimeError):
    pass


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """Wait for exclusive access to a stable lock file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock_file:
        while not try_acquire_file_lock(lock_file):
            time.sleep(0.01)
        try:
            yield
        finally:
            release_file_lock(lock_file)


@contextmanager
def project_execution_lock(artifacts_root: Path) -> Iterator[None]:
    """Prevent overlapping commands from mutating one artifact workspace."""

    path = artifacts_root.resolve() / "_system" / "execution.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock_file:
        if not try_acquire_file_lock(lock_file):
            raise ProjectExecutionBusyError(
                f"Another Jerry command is using artifacts root '{artifacts_root}'."
            )
        try:
            yield
        finally:
            release_file_lock(lock_file)
