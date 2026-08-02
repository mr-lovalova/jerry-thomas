import csv
from pathlib import Path

from jerrythomas.io.compression import Compression
from jerrythomas.io.normalization import flat_payload
from jerrythomas.io.sinks.files import AtomicTextFileSink


class CsvFileWriter:
    def __init__(
        self,
        dest: Path,
        encoding: str = "utf-8",
        overwrite: bool = True,
        compression: Compression | None = None,
    ) -> None:
        self.sink = AtomicTextFileSink(
            dest,
            encoding=encoding,
            overwrite=overwrite,
            newline="",
            compression=compression,
        )
        self.writer = csv.writer(self.sink.fh)
        self._header: tuple[str, ...] | None = None
        self._header_fields: frozenset[str] = frozenset()

    def write(self, item: object) -> None:
        row = flat_payload(item)
        if self._header is None:
            self._header = tuple(row)
            self._header_fields = frozenset(self._header)
            self.writer.writerow(self._header)
        else:
            unexpected = [field for field in row if field not in self._header_fields]
            if unexpected:
                raise ValueError(
                    "CSV row contains fields not present in header: "
                    + ", ".join(unexpected)
                )
        self.writer.writerow(row.get(field, "") for field in self._header)

    def close(self) -> None:
        self.sink.close()

    def abort(self) -> None:
        self.sink.abort()
