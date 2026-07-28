from datapipeline.io.protocols import Writer
from datapipeline.io.dataset_table import DatasetTable
from datapipeline.io.output import OutputTarget
from datapipeline.io.serializers import (
    json_line_serializer,
    text_line_serializer,
)
from datapipeline.io.sinks.files import AtomicTextFileSink
from datapipeline.io.sinks.stdout import StdoutTextSink
from datapipeline.io.writers.base import LineWriter
from datapipeline.io.writers.csv_writer import CsvFileWriter
from datapipeline.io.writers.pickle_writer import PickleFileWriter
from datapipeline.io.writers.parquet import DEFAULT_ROW_GROUP_ROWS, ParquetFileWriter


def writer_factory(target: OutputTarget, overwrite: bool = True) -> Writer:
    if target.compression is not None and (
        target.transport != "fs" or target.format not in {"jsonl", "csv"}
    ):
        raise ValueError("gzip compression supports only fs jsonl and csv output")

    if target.transport == "stdout":
        if target.format == "jsonl":
            return LineWriter(StdoutTextSink(), json_line_serializer(target.view))
        if target.format == "txt":
            return LineWriter(StdoutTextSink(), text_line_serializer())
        raise ValueError(f"Unsupported stdout format '{target.format}'")

    destination = target.destination
    if destination is None:
        raise ValueError("fs output requires a destination path")
    text_encoding = target.encoding or "utf-8"

    if target.format == "jsonl":
        return LineWriter(
            AtomicTextFileSink(
                destination,
                encoding=text_encoding,
                overwrite=overwrite,
                compression=target.compression,
            ),
            json_line_serializer(target.view),
        )
    if target.format == "csv":
        if target.view != "flat":
            raise ValueError("csv output supports only view='flat'")
        return CsvFileWriter(
            destination,
            encoding=text_encoding,
            overwrite=overwrite,
            compression=target.compression,
        )
    if target.format == "pickle":
        if target.view != "raw":
            raise ValueError("pickle output supports only view='raw'")
        return PickleFileWriter(destination, overwrite=overwrite)
    if target.format == "txt":
        return LineWriter(
            AtomicTextFileSink(
                destination,
                encoding=text_encoding,
                overwrite=overwrite,
            ),
            text_line_serializer(),
        )

    raise ValueError(f"Unsupported fs format '{target.format}'")


def dataset_writer_factory(
    target: OutputTarget,
    table: DatasetTable,
    overwrite: bool = True,
    row_group_rows: int = DEFAULT_ROW_GROUP_ROWS,
) -> Writer:
    if target.format != "parquet":
        raise ValueError("Dataset table output requires parquet format.")
    destination = target.destination
    if target.transport != "fs" or destination is None:
        raise ValueError("parquet output requires an fs destination")
    if target.view != "flat":
        raise ValueError("parquet output supports only view='flat'")
    if target.encoding is not None or target.compression is not None:
        raise ValueError(
            "parquet output does not support text encoding or gzip compression"
        )
    return ParquetFileWriter(
        destination,
        table,
        overwrite=overwrite,
        row_group_rows=row_group_rows,
    )
