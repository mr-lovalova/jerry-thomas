import builtins
from datetime import datetime, timezone

import pyarrow.parquet as parquet
import pytest

from jerrythomas.artifacts.models import (
    ListVectorMetadataEntry,
    ScalarVectorMetadataEntry,
)
from jerrythomas.domain.sample import Sample
from jerrythomas.domain.vector import Vector
from jerrythomas.io.dataset_table import DatasetTable
from jerrythomas.io.writers.parquet import ParquetFileWriter


def _table() -> DatasetTable:
    return DatasetTable(
        sample_keys=("ticker",),
        sample_key_types=("string",),
        feature_entries=(
            ScalarVectorMetadataEntry(
                id="price",
                base_id="price",
                kind="scalar",
                present_count=5,
                null_count=0,
                value_types=("float",),
            ),
            ListVectorMetadataEntry(
                id="history",
                base_id="history",
                kind="list",
                present_count=5,
                null_count=0,
                element_types=("float", "null"),
                length=2,
                observed_elements=9,
            ),
        ),
        target_entries=(),
    )


def _sample(day: int) -> Sample:
    return Sample(
        key=(datetime(2024, 1, day, tzinfo=timezone.utc), f"T{day}"),
        features=Vector(
            values={
                "price": float(day),
                "history": [None if day == 1 else float(day - 1), float(day)],
            }
        ),
    )


def test_parquet_writer_writes_bounded_row_groups_and_stable_schema(tmp_path) -> None:
    destination = tmp_path / "samples.parquet"
    writer = ParquetFileWriter(
        destination,
        _table(),
        row_group_rows=2,
    )

    for day in range(1, 6):
        writer.write(_sample(day))
    writer.close()

    source = parquet.ParquetFile(destination)
    assert source.metadata.num_rows == 5
    assert source.metadata.num_row_groups == 3
    assert source.metadata.row_group(0).column(0).compression == "ZSTD"
    result = source.read()
    assert result.column_names == [
        "sample.time",
        "sample.ticker",
        "features.price",
        "features.history.0",
        "features.history.1",
    ]
    assert result.column("sample.ticker").to_pylist() == [
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
    ]
    assert result.column("features.history.0").to_pylist() == [
        None,
        1.0,
        2.0,
        3.0,
        4.0,
    ]


def test_parquet_writer_writes_empty_table_with_declared_schema(tmp_path) -> None:
    destination = tmp_path / "empty.parquet"
    writer = ParquetFileWriter(destination, _table())

    writer.close()

    result = parquet.read_table(destination)
    assert result.num_rows == 0
    assert result.column_names == [
        "sample.time",
        "sample.ticker",
        "features.price",
        "features.history.0",
        "features.history.1",
    ]


def test_parquet_writer_abort_preserves_existing_destination(tmp_path) -> None:
    destination = tmp_path / "samples.parquet"
    destination.write_bytes(b"previous")
    writer = ParquetFileWriter(destination, _table())

    with pytest.raises(TypeError, match="features.price.*requires float"):
        writer.write(
            Sample(
                key=(datetime(2024, 1, 1, tzinfo=timezone.utc), "AAPL"),
                features=Vector(values={"price": "wrong", "history": [1.0, 2.0]}),
            )
        )
    writer.abort()

    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [destination]


def test_parquet_writer_abort_after_flushing_preserves_destination(tmp_path) -> None:
    destination = tmp_path / "samples.parquet"
    destination.write_bytes(b"previous")
    writer = ParquetFileWriter(
        destination,
        _table(),
        row_group_rows=1,
    )
    writer.write(_sample(1))

    with pytest.raises(TypeError, match="features.price.*requires float"):
        writer.write(
            Sample(
                key=(datetime(2024, 1, 2, tzinfo=timezone.utc), "AAPL"),
                features=Vector(values={"price": "wrong", "history": [1.0, 2.0]}),
            )
        )
    writer.abort()

    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [destination]


def test_parquet_writer_respects_no_overwrite(tmp_path) -> None:
    destination = tmp_path / "samples.parquet"
    destination.write_bytes(b"previous")
    writer = ParquetFileWriter(destination, _table(), overwrite=False)
    writer.write(_sample(1))

    with pytest.raises(FileExistsError, match="already exists"):
        writer.close()

    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [destination]


def test_parquet_writer_requires_sample_rows(tmp_path) -> None:
    writer = ParquetFileWriter(tmp_path / "samples.parquet", _table())

    with pytest.raises(TypeError, match="requires Sample rows"):
        writer.write({"sample.time": "wrong"})
    writer.abort()


def test_parquet_writer_reports_missing_dependency_before_creating_temp(
    tmp_path,
    monkeypatch,
) -> None:
    real_import = builtins.__import__

    def import_without_pyarrow(name, *args, **kwargs):
        if name.startswith("pyarrow"):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pyarrow)

    with pytest.raises(RuntimeError, match=r"install jerry-thomas\[parquet\]"):
        ParquetFileWriter(tmp_path / "samples.parquet", _table())

    assert list(tmp_path.iterdir()) == []
