import builtins
from datetime import datetime, timezone

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from jerrythomas.config.sources import FsLoaderConfig, ParquetReaderConfig
from jerrythomas.sources.factory import build_builtin_loader
from jerrythomas.sources.parquet_loader import ParquetLoader


def _write_rows(path, rows) -> None:
    parquet.write_table(arrow.Table.from_pylist(rows), path)


def test_parquet_loader_preserves_native_values(tmp_path) -> None:
    path = tmp_path / "rows.parquet"
    timestamp = datetime(2024, 1, 2, tzinfo=timezone.utc)
    _write_rows(
        path,
        [
            {"time": timestamp, "value": 1.5, "history": [1, None]},
            {"time": timestamp, "value": None, "history": [2, 3]},
        ],
    )

    loader = build_builtin_loader(
        FsLoaderConfig(
            transport="fs",
            path=str(path),
            reader=ParquetReaderConfig(format="parquet"),
        )
    )

    assert isinstance(loader, ParquetLoader)
    assert list(loader.load()) == [
        {"time": timestamp, "value": 1.5, "history": [1, None]},
        {"time": timestamp, "value": None, "history": [2, 3]},
    ]
    assert loader.current_resource_uri is None
    assert loader.progress_unit == "rows"


def test_parquet_loader_reads_globs_in_sorted_order(tmp_path) -> None:
    second = tmp_path / "02.parquet"
    first = tmp_path / "01.parquet"
    _write_rows(second, [{"value": 2}])
    _write_rows(first, [{"value": 1}])
    loader = ParquetLoader(str(tmp_path / "*.parquet"))

    rows = loader.load()

    assert next(rows) == {"value": 1}
    assert loader.current_resource_uri == str(first)
    assert next(rows) == {"value": 2}
    assert loader.current_resource_uri == str(second)
    assert list(rows) == []
    assert loader.current_resource_uri is None


def test_parquet_loader_uses_bounded_batches_and_closes_partial_reads(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "rows.parquet"
    _write_rows(path, [{"value": value} for value in range(5)])
    actual_parquet_file = parquet.ParquetFile
    batch_sizes: list[int] = []
    closed = False

    class TrackedParquetFile:
        def __init__(self, source_path) -> None:
            self.source = actual_parquet_file(source_path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            nonlocal closed
            closed = True
            self.source.close()

        @property
        def schema_arrow(self):
            return self.source.schema_arrow

        def iter_batches(self, batch_size):
            batch_sizes.append(batch_size)
            yield from self.source.iter_batches(batch_size=batch_size)

    monkeypatch.setattr(parquet, "ParquetFile", TrackedParquetFile)
    rows = ParquetLoader(str(path), batch_rows=2).load()

    assert next(rows) == {"value": 0}
    rows.close()

    assert batch_sizes == [2]
    assert closed


def test_parquet_loader_rejects_duplicate_columns(tmp_path) -> None:
    path = tmp_path / "duplicate.parquet"
    table = arrow.Table.from_arrays(
        [arrow.array([1]), arrow.array([2])],
        names=["value", "value"],
    )
    parquet.write_table(table, path)

    with pytest.raises(ValueError, match="duplicate columns.*'value'"):
        list(ParquetLoader(str(path)).load())


def test_parquet_loader_requires_a_glob_match(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Source glob matched no files"):
        ParquetLoader(str(tmp_path / "*.parquet"))


def test_parquet_loader_reports_missing_dependency(monkeypatch) -> None:
    real_import = builtins.__import__

    def import_without_pyarrow(name, *args, **kwargs):
        if name.startswith("pyarrow"):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pyarrow)

    with pytest.raises(RuntimeError, match=r"install jerry-thomas\[parquet\]"):
        list(ParquetLoader("rows.parquet").load())
