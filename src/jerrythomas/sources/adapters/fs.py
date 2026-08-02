import gzip
from collections.abc import Iterator
from typing import BinaryIO

from jerrythomas.io.compression import Compression
from jerrythomas.sources.ports import SourceResource, SourceTransport


def _read_chunks(
    path: str,
    chunk_size: int,
    compression: Compression | None,
) -> Iterator[bytes]:
    stream: BinaryIO | gzip.GzipFile
    if compression is None:
        stream = open(path, "rb")
    elif compression == "gzip":
        stream = gzip.open(path, "rb")
    else:
        raise ValueError(f"unsupported fs compression: {compression}")
    try:
        while chunk := stream.read(chunk_size):
            yield chunk
    finally:
        stream.close()


class FsFileTransport(SourceTransport):
    def __init__(
        self,
        path: str,
        chunk_size: int = 65536,
        compression: Compression | None = None,
    ):
        self.path = path
        self.chunk_size = chunk_size
        self.compression = compression

    def resources(self) -> Iterator[SourceResource]:
        yield SourceResource(
            uri=self.path,
            stream=_read_chunks(self.path, self.chunk_size, self.compression),
        )


class FsGlobTransport(SourceTransport):
    def __init__(
        self,
        pattern: str,
        chunk_size: int = 65536,
        compression: Compression | None = None,
    ):
        import glob as _glob

        self.pattern = pattern
        self.chunk_size = chunk_size
        self.compression = compression
        self._files = tuple(sorted(_glob.glob(pattern)))
        if not self._files:
            raise FileNotFoundError(f"Source glob matched no files: {pattern}")

    @property
    def files(self) -> tuple[str, ...]:
        return self._files

    def resources(self) -> Iterator[SourceResource]:
        for p in self._files:
            yield SourceResource(
                uri=p,
                stream=_read_chunks(p, self.chunk_size, self.compression),
            )
