import glob
from collections.abc import Iterator
from typing import Any

from jerrythomas.sources.loader import BaseDataLoader, SourceProgressUnit


DEFAULT_BATCH_ROWS = 65_536


class ParquetLoader(BaseDataLoader):
    """Read local Parquet files as bounded batches of row mappings."""

    def __init__(self, path: str, batch_rows: int = DEFAULT_BATCH_ROWS) -> None:
        if batch_rows <= 0:
            raise ValueError("Parquet batch size must be positive.")
        self.path = path
        self.batch_rows = batch_rows
        self.is_glob = glob.has_magic(path)
        self._files = tuple(sorted(glob.glob(path))) if self.is_glob else (path,)
        if not self._files:
            raise FileNotFoundError(f"Source glob matched no files: {path}")
        self._current_resource_uri: str | None = None

    @property
    def files(self) -> tuple[str, ...]:
        return self._files

    @property
    def current_resource_uri(self) -> str | None:
        return self._current_resource_uri

    @property
    def progress_unit(self) -> SourceProgressUnit:
        return "rows"

    def load(self) -> Iterator[Any]:
        try:
            import pyarrow.parquet as parquet  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "Parquet input requires pyarrow; install jerry-thomas[parquet]."
            ) from exc

        try:
            for path in self._files:
                self._current_resource_uri = path
                with parquet.ParquetFile(path) as source:
                    names = source.schema_arrow.names
                    if len(names) != len(set(names)):
                        duplicates = sorted(
                            name for name in set(names) if names.count(name) > 1
                        )
                        raise ValueError(
                            "Parquet source has duplicate columns: "
                            + ", ".join(repr(name) for name in duplicates)
                            + "."
                        )
                    for batch in source.iter_batches(batch_size=self.batch_rows):
                        yield from batch.to_pylist()
        finally:
            self._current_resource_uri = None
