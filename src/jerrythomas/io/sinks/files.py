import codecs
import gzip
import io
import os
import stat
import tempfile
from pathlib import Path
from typing import BinaryIO, TextIO

from jerrythomas.io.compression import Compression


def _commit_temp_file(temp: Path, dest: Path, overwrite: bool) -> None:
    if overwrite:
        os.replace(temp, dest)
        return
    try:
        os.link(temp, dest)
    except FileExistsError as exc:
        raise FileExistsError(f"{dest} already exists") from exc
    finally:
        temp.unlink(missing_ok=True)


def _temporary_file(dest: Path) -> tuple[int, Path]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=dest.parent)
    temp = Path(temp_path)
    try:
        mode = stat.S_IMODE(dest.stat().st_mode)
        os.fchmod(fd, mode)
    except FileNotFoundError:
        pass
    except BaseException:
        os.close(fd)
        temp.unlink(missing_ok=True)
        raise
    return fd, temp


class AtomicTextFileSink:
    def __init__(
        self,
        dest: Path,
        encoding: str = "utf-8",
        overwrite: bool = True,
        newline: str | None = None,
        compression: Compression | None = None,
    ):
        self._dest = dest
        self._overwrite = overwrite
        codecs.lookup(encoding)
        if compression not in {None, "gzip"}:
            raise ValueError(f"Unsupported compression {compression!r}")
        fd, self._tmp = _temporary_file(dest)
        self._raw: BinaryIO | None = None
        self._fh: TextIO
        if compression == "gzip":
            self._raw = os.fdopen(fd, "wb")
            compressed = gzip.GzipFile(
                filename="",
                fileobj=self._raw,
                mode="wb",
                mtime=0,
            )
            self._fh = io.TextIOWrapper(
                compressed,
                encoding=encoding,
                newline=newline,
            )
        else:
            self._fh = os.fdopen(fd, "w", encoding=encoding, newline=newline)

    @property
    def fh(self):
        return self._fh

    def write_text(self, s: str) -> None:
        self._fh.write(s)

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            if self._raw is not None:
                self._raw.close()
        _commit_temp_file(self._tmp, self._dest, self._overwrite)

    def abort(self) -> None:
        try:
            self._fh.close()
        finally:
            if self._raw is not None:
                self._raw.close()
            self._tmp.unlink(missing_ok=True)


class AtomicBinaryFileSink:
    def __init__(self, dest: Path, overwrite: bool = True):
        self._dest = dest
        self._overwrite = overwrite
        fd, self._tmp = _temporary_file(dest)
        self._fh = os.fdopen(fd, "wb")

    @property
    def fh(self):
        return self._fh

    def write_bytes(self, b: bytes) -> None:
        self._fh.write(b)

    def close(self) -> None:
        self._fh.close()
        _commit_temp_file(self._tmp, self._dest, self._overwrite)

    def abort(self) -> None:
        self._fh.close()
        self._tmp.unlink(missing_ok=True)


class GzipBinarySink:
    def __init__(
        self,
        dest: Path,
        compression_level: int,
        overwrite: bool = True,
    ):
        self._dest = dest
        self._overwrite = overwrite
        fd, self._tmp = _temporary_file(dest)
        self._raw = os.fdopen(fd, "wb")
        self._fh = gzip.GzipFile(
            filename="",
            fileobj=self._raw,
            mode="wb",
            compresslevel=compression_level,
            mtime=0,
        )

    def write_bytes(self, b: bytes) -> None:
        self._fh.write(b)

    def close(self) -> None:
        self._fh.close()
        self._raw.close()
        _commit_temp_file(self._tmp, self._dest, self._overwrite)

    def abort(self) -> None:
        self._fh.close()
        self._raw.close()
        self._tmp.unlink(missing_ok=True)
