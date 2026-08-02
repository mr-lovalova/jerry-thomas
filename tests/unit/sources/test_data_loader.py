import gzip
from urllib.parse import parse_qsl, urlparse

import pytest

import jerrythomas.sources.adapters.fs as fs_adapter
import jerrythomas.sources.adapters.http as http_adapter
from jerrythomas.config.sources import (
    CsvReaderConfig,
    FsLoaderConfig,
    HttpLoaderConfig,
    JsonReaderConfig,
    JsonLinesReaderConfig,
)
from jerrythomas.sources.adapters.fs import FsFileTransport, FsGlobTransport
from jerrythomas.sources.adapters.http import HttpTransport
from jerrythomas.sources.loader import DataLoader
from jerrythomas.sources.decoders import JsonDecoder, JsonLinesDecoder
from jerrythomas.sources.factory import build_builtin_loader
from jerrythomas.sources.ports import SourceResource, SourceTransport


def _write_gzip(path, text: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        stream.write(text)


def test_fs_path_selects_file_or_glob_transport(tmp_path) -> None:
    (tmp_path / "rows.jsonl").write_text("", encoding="utf-8")
    reader = JsonLinesReaderConfig(format="jsonl")
    file_loader = build_builtin_loader(
        FsLoaderConfig(
            transport="fs",
            path=str(tmp_path / "rows.jsonl"),
            reader=reader,
        )
    )
    glob_loader = build_builtin_loader(
        FsLoaderConfig(
            transport="fs",
            path=str(tmp_path / "*.jsonl"),
            reader=reader,
        )
    )

    assert isinstance(file_loader.transport, FsFileTransport)
    assert isinstance(glob_loader.transport, FsGlobTransport)


def test_fs_glob_transport_exposes_sorted_immutable_files(tmp_path) -> None:
    second = tmp_path / "02.jsonl"
    first = tmp_path / "01.jsonl"
    second.write_text("", encoding="utf-8")
    first.write_text("", encoding="utf-8")

    transport = FsGlobTransport(str(tmp_path / "*.jsonl"))

    assert transport.files == (str(first), str(second))


def test_fs_file_loader_decompresses_gzip_explicitly(tmp_path) -> None:
    path = tmp_path / "rows.data"
    _write_gzip(path, '{"value":1}\n{"value":2}\n')

    loader = build_builtin_loader(
        FsLoaderConfig(
            transport="fs",
            path=str(path),
            reader=JsonLinesReaderConfig(format="jsonl"),
            compression="gzip",
        )
    )

    assert list(loader.load()) == [{"value": 1}, {"value": 2}]
    assert loader.transport.compression == "gzip"


def test_fs_glob_loader_decompresses_each_gzip_file(tmp_path) -> None:
    _write_gzip(tmp_path / "01.data", '{"value":1}\n')
    _write_gzip(tmp_path / "02.data", '{"value":2}\n')

    loader = build_builtin_loader(
        FsLoaderConfig(
            transport="fs",
            path=str(tmp_path / "*.data"),
            reader=JsonLinesReaderConfig(format="jsonl"),
            compression="gzip",
        )
    )

    assert list(loader.load()) == [{"value": 1}, {"value": 2}]
    assert loader.transport.compression == "gzip"


def test_fs_csv_loader_decompresses_gzip(tmp_path) -> None:
    path = tmp_path / "rows.data"
    _write_gzip(path, "value;label\n1;one\n")

    loader = build_builtin_loader(
        FsLoaderConfig(
            transport="fs",
            path=str(path),
            reader=CsvReaderConfig(format="csv"),
            compression="gzip",
        )
    )

    assert list(loader.load()) == [{"value": "1", "label": "one"}]


def test_fs_loader_does_not_infer_compression_from_suffix(tmp_path) -> None:
    path = tmp_path / "rows.jsonl.gz"
    _write_gzip(path, '{"value":1}\n')

    loader = build_builtin_loader(
        FsLoaderConfig(
            transport="fs",
            path=str(path),
            reader=JsonLinesReaderConfig(format="jsonl"),
        )
    )

    with pytest.raises(UnicodeDecodeError):
        list(loader.load())


def test_fs_gzip_loader_closes_file_after_partial_read(tmp_path, monkeypatch) -> None:
    path = tmp_path / "rows.data"
    _write_gzip(path, '{"value":1}\n{"value":2}\n')

    class TrackedStream:
        def __init__(self):
            self.stream = gzip.open(path, "rb")
            self.closed = False

        def read(self, size):
            return self.stream.read(size)

        def close(self):
            self.closed = True
            self.stream.close()

    tracked = TrackedStream()
    monkeypatch.setattr(fs_adapter.gzip, "open", lambda *args, **kwargs: tracked)
    rows = build_builtin_loader(
        FsLoaderConfig(
            transport="fs",
            path=str(path),
            reader=JsonLinesReaderConfig(format="jsonl"),
            compression="gzip",
        )
    ).load()

    assert next(rows) == {"value": 1}
    rows.close()

    assert tracked.closed


def test_fs_glob_requires_at_least_one_file(tmp_path) -> None:
    pattern = str(tmp_path / "*.jsonl")

    with pytest.raises(FileNotFoundError, match="Source glob matched no files"):
        build_builtin_loader(
            FsLoaderConfig(
                transport="fs",
                path=pattern,
                reader=JsonLinesReaderConfig(format="jsonl"),
            )
        )


def test_http_loader_config_builds_transport_and_decoder() -> None:
    loader = build_builtin_loader(
        HttpLoaderConfig(
            transport="http",
            url="https://example.test/rows",
            headers={"Authorization": "token"},
            params={"page": 2},
            timeout_seconds=10,
            reader=JsonReaderConfig(
                format="json",
                array_field="results",
            ),
        )
    )

    assert isinstance(loader.transport, HttpTransport)
    assert loader.transport.headers == {"Authorization": "token"}
    assert loader.transport.params == {"page": 2}
    assert loader.transport.timeout_seconds == 10
    assert isinstance(loader.decoder, JsonDecoder)
    assert loader.decoder.array_field == "results"


def test_http_transport_merges_existing_and_configured_query_parameters() -> None:
    transport = HttpTransport(
        "https://example.test/rows?existing=1",
        params={"symbol": ["AAPL", "MSFT"], "empty": ""},
    )

    query = parse_qsl(urlparse(transport._build_url()).query, keep_blank_values=True)

    assert query == [
        ("existing", "1"),
        ("symbol", "AAPL"),
        ("symbol", "MSFT"),
        ("empty", ""),
    ]


def test_http_transport_does_not_drop_parameters_when_encoding_fails(
    monkeypatch,
) -> None:
    class InvalidParameter:
        def __str__(self) -> str:
            raise RuntimeError("invalid parameter")

    opened = False

    def open_url(request, timeout):
        nonlocal opened
        opened = True
        raise AssertionError("request must not be opened")

    monkeypatch.setattr(http_adapter, "urlopen", open_url)
    transport = HttpTransport(
        "https://example.test/rows",
        params={"symbol": InvalidParameter()},
    )

    with pytest.raises(RuntimeError, match="invalid parameter"):
        list(transport.resources())

    assert not opened


def test_data_loader_tracks_current_resource_uri(tmp_path):
    appl = tmp_path / "APPL.jsonl"
    msft = tmp_path / "MSFT.jsonl"
    appl.write_text('{"symbol":"AAPL"}\n', encoding="utf-8")
    msft.write_text('{"symbol":"MSFT"}\n', encoding="utf-8")

    loader = DataLoader(
        transport=FsGlobTransport(str(tmp_path / "*.jsonl")),
        decoder=JsonLinesDecoder(),
    )

    rows = loader.load()

    assert next(rows) == {"symbol": "AAPL"}
    assert loader.current_resource_uri == str(appl)
    assert next(rows) == {"symbol": "MSFT"}
    assert loader.current_resource_uri == str(msft)
    assert list(rows) == []
    assert loader.current_resource_uri is None


def test_data_loader_closes_resource_after_partial_read() -> None:
    closed = False

    def chunks():
        nonlocal closed
        try:
            yield b'{"value": 1}\n'
            yield b'{"value": 2}\n'
        finally:
            closed = True

    class Transport(SourceTransport):
        def resources(self):
            yield SourceResource(uri="rows", stream=chunks())

    rows = DataLoader(Transport(), JsonLinesDecoder()).load()

    assert next(rows) == {"value": 1}
    rows.close()
    assert closed
