import pyarrow as arrow
import pyarrow.parquet as parquet

from jerrythomas.sources.adapters.fs import FsFileTransport, FsGlobTransport
from jerrythomas.sources.adapters.http import HttpTransport
from jerrythomas.sources.loader import DataLoader
from jerrythomas.sources.decoders import CsvDecoder, JsonLinesDecoder
from jerrythomas.sources.source import Source
from jerrythomas.sources.observability import (
    source_progress,
    source_summary,
)
from jerrythomas.sources.synthetic.time.loader import make_time_loader
from jerrythomas.sources.parquet_loader import ParquetLoader
from jerrythomas.parsers.identity import IdentityParser


class _SourceLookalike:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader

    def stream(self):
        return iter(())


def test_source_summary_describes_http_transport() -> None:
    loader = DataLoader(
        HttpTransport("https://example.test/data/demo.jsonl"),
        JsonLinesDecoder(),
    )
    source = Source(loader, IdentityParser())

    assert (
        source_summary(source)
        == "transport=http.fetch host=example.test resource=demo.jsonl"
    )
    progress = source_progress(source)
    assert progress is not None
    snapshot = progress(7)
    assert snapshot.completed == 7
    assert snapshot.unit == "items"
    assert snapshot.resource is not None
    assert (
        snapshot.resource.index,
        snapshot.resource.total,
        snapshot.resource.label,
    ) == (1, 1, "@example.test")


def test_glob_progress_tracks_current_file_and_resource_sequence(
    tmp_path,
) -> None:
    (tmp_path / "APPL.jsonl").write_text('{"n":1}\n', encoding="utf-8")
    (tmp_path / "MSFT.jsonl").write_text('{"n":2}\n', encoding="utf-8")
    loader = DataLoader(
        transport=FsGlobTransport(str(tmp_path / "*.jsonl")),
        decoder=JsonLinesDecoder(),
    )
    source = Source(loader, IdentityParser())

    progress = source_progress(source)
    assert progress is not None

    initial = progress(0)
    assert initial.completed == 0
    assert initial.unit == "items"
    assert initial.resource is not None
    assert (
        initial.resource.index,
        initial.resource.total,
        initial.resource.label,
    ) == (1, 2, '"APPL.jsonl"')

    rows = source.stream()
    assert next(rows) == {"n": 1}
    assert loader.current_resource_uri == str(tmp_path / "APPL.jsonl")
    first = progress(1)
    assert first.resource == initial.resource

    assert next(rows) == {"n": 2}
    assert loader.current_resource_uri == str(tmp_path / "MSFT.jsonl")
    second = progress(2)
    assert second.resource is not None
    assert (
        second.resource.index,
        second.resource.total,
        second.resource.label,
    ) == (2, 2, '"MSFT.jsonl"')
    rows.close()
    assert progress(0).resource == initial.resource


def test_single_file_glob_uses_the_file_name_in_progress(tmp_path) -> None:
    path = tmp_path / "only.jsonl"
    path.write_text('{"n":1}\n', encoding="utf-8")
    loader = DataLoader(
        transport=FsGlobTransport(str(tmp_path / "*.jsonl")),
        decoder=JsonLinesDecoder(),
    )

    source = Source(loader, IdentityParser())

    assert source_summary(source) == "transport=fs.glob count=1 file=only.jsonl"
    progress = source_progress(source)
    assert progress is not None
    assert progress(0).resource is not None
    assert progress(0).resource.label == '"only.jsonl"'


def test_file_loader_uses_file_name_and_csv_rows() -> None:
    loader = DataLoader(FsFileTransport("/tmp/demo.csv"), CsvDecoder())
    source = Source(loader, IdentityParser())

    assert source_summary(source) == "transport=fs.file file=demo.csv"
    progress = source_progress(source)
    assert progress is not None
    snapshot = progress(0)
    assert snapshot.unit == "rows"
    assert snapshot.resource is not None
    assert (
        snapshot.resource.index,
        snapshot.resource.total,
        snapshot.resource.label,
    ) == (1, 1, '"demo.csv"')


def test_parquet_glob_uses_file_progress_and_row_units(tmp_path) -> None:
    first = tmp_path / "01.parquet"
    second = tmp_path / "02.parquet"
    parquet.write_table(arrow.Table.from_pylist([{"n": 1}]), first)
    parquet.write_table(arrow.Table.from_pylist([{"n": 2}]), second)
    loader = ParquetLoader(str(tmp_path / "*.parquet"))
    source = Source(loader, IdentityParser())

    assert source_summary(source) == (
        "transport=fs.glob count=2 first=01.parquet last=02.parquet"
    )
    progress = source_progress(source)
    assert progress is not None
    initial = progress(0)
    assert initial.unit == "rows"
    assert initial.resource is not None
    assert (
        initial.resource.index,
        initial.resource.total,
        initial.resource.label,
    ) == (1, 2, '"01.parquet"')

    rows = source.stream()
    assert next(rows) == {"n": 1}
    assert progress(1).resource == initial.resource
    assert next(rows) == {"n": 2}
    second_progress = progress(2)
    assert second_progress.resource is not None
    assert (
        second_progress.resource.index,
        second_progress.resource.total,
        second_progress.resource.label,
    ) == (2, 2, '"02.parquet"')
    rows.close()


def test_parquet_file_uses_file_summary(tmp_path) -> None:
    path = tmp_path / "rows.parquet"
    parquet.write_table(arrow.Table.from_pylist([{"n": 1}]), path)
    source = Source(ParquetLoader(str(path)), IdentityParser())

    assert source_summary(source) == "transport=fs.file file=rows.parquet"
    progress = source_progress(source)
    assert progress is not None
    assert progress(0).resource is not None
    assert progress(0).resource.label == '"rows.parquet"'


def test_compressed_fs_source_summaries_include_compression(tmp_path) -> None:
    first = tmp_path / "01.jsonl.gz"
    second = tmp_path / "02.jsonl.gz"
    first.write_bytes(b"")
    second.write_bytes(b"")

    file_source = Source(
        DataLoader(
            FsFileTransport(str(first), compression="gzip"),
            JsonLinesDecoder(),
        ),
        IdentityParser(),
    )
    glob_source = Source(
        DataLoader(
            FsGlobTransport(str(tmp_path / "*.jsonl.gz"), compression="gzip"),
            JsonLinesDecoder(),
        ),
        IdentityParser(),
    )

    assert (
        source_summary(file_source)
        == "transport=fs.file compression=gzip file=01.jsonl.gz"
    )
    assert (
        source_summary(glob_source) == "transport=fs.glob compression=gzip count=2 "
        "first=01.jsonl.gz last=02.jsonl.gz"
    )


def test_synthetic_loader_uses_tick_progress_without_a_resource() -> None:
    loader = make_time_loader(
        "2024-01-01T00:00:00Z",
        "2024-01-01T01:00:00Z",
        "1h",
    )
    source = Source(loader, IdentityParser())

    assert source_summary(source) is None
    progress = source_progress(source)
    assert progress is not None
    snapshot = progress(2)
    assert snapshot.completed == 2
    assert snapshot.unit == "ticks"
    assert snapshot.resource is None


def test_source_summary_does_not_infer_a_source_from_a_loader_attribute() -> None:
    loader = DataLoader(
        HttpTransport("https://example.test/data/demo.jsonl"),
        JsonLinesDecoder(),
    )

    lookalike = _SourceLookalike(loader)

    assert source_summary(lookalike) is None
    assert source_progress(lookalike) is None
