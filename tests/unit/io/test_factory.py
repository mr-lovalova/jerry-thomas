import csv
import gzip
import json
import pickle
from dataclasses import dataclass

import pytest

from jerrythomas.artifacts.models import ScalarVectorMetadataEntry
from jerrythomas.io.dataset_table import DatasetTable
from jerrythomas.io.factory import dataset_writer_factory, writer_factory
from jerrythomas.io.output import OutputTarget
from jerrythomas.io.sinks.files import AtomicTextFileSink
from jerrythomas.io.sinks.stdout import StdoutTextSink
from jerrythomas.io.writers.base import LineWriter
from jerrythomas.io.writers.parquet import ParquetFileWriter


def _stdout_target(format_: str = "jsonl") -> OutputTarget:
    return OutputTarget(
        transport="stdout",
        format=format_,
        view="raw",
        encoding=None,
        destination=None,
    )


def test_writer_factory_stdout_jsonl_uses_plain_stdout_sink() -> None:
    writer = writer_factory(_stdout_target("jsonl"))
    assert isinstance(writer, LineWriter)
    assert isinstance(writer.sink, StdoutTextSink)


def test_writer_factory_stdout_txt_uses_plain_stdout_sink() -> None:
    writer = writer_factory(_stdout_target("txt"))
    assert isinstance(writer, LineWriter)
    assert isinstance(writer.sink, StdoutTextSink)


def test_writer_factory_fs_txt_uses_line_writer(tmp_path) -> None:
    writer = writer_factory(
        OutputTarget(
            transport="fs",
            format="txt",
            view="flat",
            encoding="utf-8",
            destination=(tmp_path / "out.txt").resolve(),
        ),
    )
    assert isinstance(writer, LineWriter)
    assert isinstance(writer.sink, AtomicTextFileSink)
    writer.close()


def test_writer_factory_fs_jsonl_applies_flat_view(tmp_path) -> None:
    destination = tmp_path / "out.jsonl"
    writer = writer_factory(
        OutputTarget(
            transport="fs",
            format="jsonl",
            view="flat",
            encoding="utf-8",
            destination=destination,
        )
    )

    writer.write({"features": {"x": 1.0}})
    writer.close()

    assert json.loads(destination.read_text(encoding="utf-8")) == {"features.x": 1.0}


def test_writer_factory_writes_compressed_jsonl(tmp_path) -> None:
    destination = tmp_path / "out.jsonl.gz"
    writer = writer_factory(
        OutputTarget(
            transport="fs",
            format="jsonl",
            view="raw",
            encoding="utf-8",
            destination=destination,
            compression="gzip",
        )
    )

    writer.write({"value": 1})
    writer.close()

    with gzip.open(destination, "rt", encoding="utf-8") as stream:
        assert json.loads(stream.read()) == {"value": 1}


def test_writer_factory_writes_compressed_csv(tmp_path) -> None:
    destination = tmp_path / "out.csv.gz"
    writer = writer_factory(
        OutputTarget(
            transport="fs",
            format="csv",
            view="flat",
            encoding="utf-8",
            destination=destination,
            compression="gzip",
        )
    )

    writer.write({"symbol": "AAPL", "price": 10.0})
    writer.close()

    with gzip.open(destination, "rt", encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == [{"symbol": "AAPL", "price": "10.0"}]


def test_writer_factory_rejects_compression_for_unsupported_target(tmp_path) -> None:
    with pytest.raises(
        ValueError,
        match="gzip compression supports only fs jsonl and csv output",
    ):
        writer_factory(
            OutputTarget(
                transport="fs",
                format="pickle",
                view="raw",
                encoding=None,
                destination=tmp_path / "out.pkl.gz",
                compression="gzip",
            )
        )


def test_writer_factory_fs_pickle_writes_raw_payload(tmp_path) -> None:
    @dataclass
    class Payload:
        value: int

    destination = tmp_path / "out.pkl"
    writer = writer_factory(
        OutputTarget(
            transport="fs",
            format="pickle",
            view="raw",
            encoding=None,
            destination=destination,
        )
    )

    writer.write(Payload(1))
    writer.write(Payload(2))
    writer.close()

    with destination.open("rb") as fh:
        assert pickle.load(fh) == {"value": 1}
        assert pickle.load(fh) == {"value": 2}


def test_dataset_writer_factory_builds_parquet_writer(tmp_path) -> None:
    table = DatasetTable(
        (),
        (),
        (
            ScalarVectorMetadataEntry(
                id="price",
                base_id="price",
                kind="scalar",
                present_count=1,
                null_count=0,
                value_types=("float",),
            ),
        ),
        (),
    )

    writer = dataset_writer_factory(
        OutputTarget(
            transport="fs",
            format="parquet",
            view="flat",
            encoding=None,
            destination=tmp_path / "samples.parquet",
        ),
        table,
    )

    assert isinstance(writer, ParquetFileWriter)
    writer.abort()


def test_plain_writer_factory_rejects_parquet_without_dataset_schema(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="Unsupported fs format 'parquet'"):
        writer_factory(
            OutputTarget(
                transport="fs",
                format="parquet",
                view="flat",
                encoding=None,
                destination=tmp_path / "samples.parquet",
            )
        )


def test_dataset_writer_factory_rejects_non_parquet_target(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires parquet format"):
        dataset_writer_factory(
            OutputTarget(
                transport="fs",
                format="jsonl",
                view="flat",
                encoding="utf-8",
                destination=tmp_path / "samples.jsonl",
            ),
            DatasetTable((), (), (), ()),
        )


@pytest.mark.parametrize(
    ("format_", "view", "message"),
    [
        ("csv", "raw", "csv output supports only view='flat'"),
        ("parquet", "raw", "Unsupported fs format 'parquet'"),
        ("pickle", "flat", "pickle output supports only view='raw'"),
    ],
)
def test_writer_factory_rejects_unsupported_view(
    tmp_path,
    format_: str,
    view: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        writer_factory(
            OutputTarget(
                transport="fs",
                format=format_,
                view=view,
                encoding=None,
                destination=tmp_path / f"out.{format_}",
            )
        )
