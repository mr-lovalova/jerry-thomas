from pathlib import Path
from typing import Any

from datapipeline.domain.sample import Sample
from datapipeline.io.dataset_table import DatasetTable, TableColumn
from datapipeline.io.sinks.files import AtomicBinaryFileSink


DEFAULT_ROW_GROUP_ROWS = 4096


class ParquetFileWriter:
    def __init__(
        self,
        destination: Path,
        table: DatasetTable,
        overwrite: bool = True,
        row_group_rows: int = DEFAULT_ROW_GROUP_ROWS,
    ) -> None:
        if row_group_rows <= 0:
            raise ValueError("Parquet row group size must be positive.")
        try:
            import pyarrow as arrow  # type: ignore[import-untyped]
            import pyarrow.parquet as parquet  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "Parquet output requires pyarrow; install jerry-thomas[parquet]."
            ) from exc

        self._arrow = arrow
        self._table = table
        self._row_group_rows = row_group_rows
        self._rows: list[dict[str, Any]] = []
        self._schema = _arrow_schema(arrow, table.columns)
        self._sink = AtomicBinaryFileSink(destination, overwrite=overwrite)
        try:
            self._writer = parquet.ParquetWriter(
                self._sink.fh,
                self._schema,
                compression="zstd",
            )
        except BaseException:
            self._sink.abort()
            raise

    def write(self, item: object) -> None:
        if not isinstance(item, Sample):
            raise TypeError("Parquet dataset output requires Sample rows.")
        self._rows.append(self._table.project(item))
        if len(self._rows) >= self._row_group_rows:
            self._flush()

    def close(self) -> None:
        self._flush()
        writer = self._writer
        self._writer = None
        if writer is not None:
            writer.close()
        self._sink.close()

    def abort(self) -> None:
        self._rows.clear()
        writer = self._writer
        self._writer = None
        try:
            if writer is not None:
                writer.close()
        finally:
            self._sink.abort()

    def _flush(self) -> None:
        if not self._rows:
            return
        batch = self._arrow.Table.from_pylist(self._rows, schema=self._schema)
        writer = self._writer
        if writer is None:
            raise RuntimeError("Parquet writer is closed.")
        writer.write_table(batch, row_group_size=len(self._rows))
        self._rows.clear()


def _arrow_schema(arrow: Any, columns: tuple[TableColumn, ...]) -> Any:
    arrow_types = {
        "null": arrow.null(),
        "boolean": arrow.bool_(),
        "integer": arrow.int64(),
        "float": arrow.float64(),
        "string": arrow.string(),
        "datetime": arrow.timestamp("us", tz="UTC"),
        "date": arrow.date32(),
    }
    return arrow.schema(
        [
            arrow.field(
                column.name,
                arrow_types[column.value_type],
                nullable=column.nullable,
            )
            for column in columns
        ]
    )
