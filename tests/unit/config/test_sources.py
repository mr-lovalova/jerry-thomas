import pytest

from jerrythomas.config.sources import (
    FsLoaderConfig,
    HttpLoaderConfig,
    JsonReaderConfig,
    JsonLinesReaderConfig,
    SourceConfig,
)


def source_config(**overrides: object) -> SourceConfig:
    values: dict[str, object] = {
        "id": "sample.source",
        "parser": {"entrypoint": "parse"},
        "loader": {"entrypoint": "load"},
    }
    values.update(overrides)
    return SourceConfig.model_validate(values)


def test_source_input_files_have_canonical_order() -> None:
    source = source_config(inputs={"files": ["data/b.jsonl", "data/a.jsonl"]})

    assert source.inputs is not None
    assert source.inputs.files == ("data/a.jsonl", "data/b.jsonl")


def test_source_input_files_reject_duplicates() -> None:
    with pytest.raises(ValueError, match="must not contain duplicates"):
        source_config(inputs={"files": ["data/a.jsonl", "data/a.jsonl"]})


@pytest.mark.parametrize(
    "source_id",
    ["bad/id", "bad id", "bad@id", "bad:id", ".bad", "bad.", "bad..id"],
)
def test_source_rejects_noncanonical_id(source_id: str) -> None:
    with pytest.raises(ValueError, match="String should match pattern"):
        source_config(id=source_id)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cadence", "1d"),
        ("fallback", "hidden.magic"),
    ],
)
def test_source_rejects_unknown_top_level_field(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        source_config(**{field: value})


def test_source_rejects_unknown_parser_field() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        source_config(parser={"entrypoint": "parse", "fallback": "hidden.magic"})


def test_source_rejects_unknown_loader_field() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        source_config(loader={"entrypoint": "load", "fallback": "hidden.magic"})


def test_built_in_source_rejects_unknown_transport_argument() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        source_config(
            loader={
                "transport": "fs",
                "path": "prices.csv",
                "delimeter": ",",
                "reader": {"format": "csv"},
            }
        )


def test_reader_rejects_options_for_another_format() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        source_config(
            loader={
                "transport": "fs",
                "path": "prices.jsonl",
                "reader": {"format": "jsonl", "delimiter": ","},
            }
        )


@pytest.mark.parametrize("delimiter", ["\n", "\r", '"'])
def test_csv_reader_rejects_invalid_delimiter(delimiter: str) -> None:
    with pytest.raises(ValueError, match="string_pattern_mismatch"):
        source_config(
            loader={
                "transport": "fs",
                "path": "prices.csv",
                "reader": {"format": "csv", "delimiter": delimiter},
            }
        )


@pytest.mark.parametrize(
    "encoding",
    ["not-a-codec", "base64_codec", "rot_13", "undefined", "uu_codec"],
)
def test_text_reader_rejects_non_text_encoding(encoding: str) -> None:
    with pytest.raises(ValueError, match="unsupported text encoding"):
        source_config(
            loader={
                "transport": "fs",
                "path": "prices.jsonl",
                "reader": {"format": "jsonl", "encoding": encoding},
            }
        )


@pytest.mark.parametrize("prefix", ["", "   "])
def test_csv_reader_rejects_blank_error_prefix(prefix: str) -> None:
    with pytest.raises(ValueError):
        source_config(
            loader={
                "transport": "fs",
                "path": "prices.csv",
                "reader": {
                    "format": "csv",
                    "error_prefixes": [prefix],
                },
            }
        )


def test_csv_reader_preserves_error_prefix_whitespace() -> None:
    source = source_config(
        loader={
            "transport": "fs",
            "path": "prices.csv",
            "reader": {
                "format": "csv",
                "error_prefixes": ["ERROR "],
            },
        }
    )

    assert source.loader.reader.error_prefixes == ("ERROR ",)


@pytest.mark.parametrize("format_", ["csv", "jsonl"])
def test_fs_source_accepts_gzip_compression(format_: str) -> None:
    source = source_config(
        loader={
            "transport": "fs",
            "path": "records.gz",
            "compression": "gzip",
            "reader": {"format": format_},
        }
    )

    assert source.loader.compression == "gzip"


def test_fs_source_rejects_gzip_for_json() -> None:
    with pytest.raises(
        ValueError,
        match="gzip compression is supported only for csv and jsonl formats",
    ):
        source_config(
            loader={
                "transport": "fs",
                "path": "records.gz",
                "compression": "gzip",
                "reader": {"format": "json"},
            }
        )


def test_fs_source_accepts_parquet_without_text_options() -> None:
    source = source_config(
        loader={
            "transport": "fs",
            "path": "records/*.parquet",
            "reader": {"format": "parquet"},
        }
    )

    assert source.loader.model_dump(exclude_unset=True) == {
        "transport": "fs",
        "path": "records/*.parquet",
        "reader": {"format": "parquet"},
    }


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("encoding", "utf-8"),
        ("delimiter", ","),
        ("error_prefixes", ["#"]),
        ("array_field", "rows"),
    ],
)
def test_parquet_reader_rejects_text_options(option: str, value: object) -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        source_config(
            loader={
                "transport": "fs",
                "path": "records.parquet",
                "reader": {"format": "parquet", option: value},
            }
        )


def test_fs_source_rejects_compressed_parquet() -> None:
    with pytest.raises(
        ValueError,
        match="parquet input does not support external compression",
    ):
        source_config(
            loader={
                "transport": "fs",
                "path": "records.parquet",
                "compression": "gzip",
                "reader": {"format": "parquet"},
            }
        )


def test_http_source_rejects_parquet() -> None:
    with pytest.raises(ValueError, match="csv.*json.*jsonl"):
        source_config(
            loader={
                "transport": "http",
                "url": "https://example.test/records.parquet",
                "reader": {"format": "parquet"},
            }
        )


@pytest.mark.parametrize("format_", ["csv", "json", "jsonl", "parquet"])
def test_core_source_config_round_trips(format_: str) -> None:
    source = source_config(
        loader={
            "transport": "fs",
            "path": f"records.{format_}",
            "reader": {"format": format_},
        }
    )

    assert SourceConfig.model_validate(source.model_dump()) == source


def test_http_source_config_round_trips() -> None:
    source = source_config(
        loader={
            "transport": "http",
            "url": "https://example.test/records.json",
            "reader": {"format": "json", "array_field": "rows"},
        }
    )

    assert SourceConfig.model_validate(source.model_dump()) == source


def test_loader_config_models_remain_directly_constructible() -> None:
    fs = FsLoaderConfig(
        transport="fs",
        path="records.jsonl",
        reader=JsonLinesReaderConfig(format="jsonl"),
    )
    http = HttpLoaderConfig(
        transport="http",
        url="https://example.test/records.json",
        reader=JsonReaderConfig(format="json"),
    )

    assert fs.reader.format == "jsonl"
    assert http.reader.format == "json"


def test_built_in_source_rejects_pickle_format() -> None:
    with pytest.raises(ValueError, match="expected tags.*parquet"):
        source_config(
            loader={
                "transport": "fs",
                "path": "records.pkl",
                "reader": {"format": "pickle"},
            }
        )


def test_http_source_rejects_compression() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        source_config(
            loader={
                "transport": "http",
                "url": "https://example.test/records.jsonl.gz",
                "compression": "gzip",
                "reader": {"format": "jsonl"},
            }
        )


def test_custom_source_loader_keeps_plugin_arguments_explicitly_open() -> None:
    source = source_config(
        loader={
            "entrypoint": "custom.loader",
            "args": {"vendor_option": "value"},
        }
    )

    assert source.loader.args == {"vendor_option": "value"}


@pytest.mark.parametrize("value", ["10", True, float("inf")])
def test_http_source_rejects_invalid_timeout(value: object) -> None:
    with pytest.raises(ValueError):
        source_config(
            loader={
                "transport": "http",
                "url": "https://example.test/prices.jsonl",
                "timeout_seconds": value,
                "reader": {"format": "jsonl"},
            }
        )
