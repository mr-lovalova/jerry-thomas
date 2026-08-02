from jerrythomas.services.scaffold.layout import (
    default_stream_id_for_source,
    source_id_parts,
)


def test_existing_source_variant_is_preserved_in_default_stream_id() -> None:
    assert source_id_parts("alpaca.ohlcv.benchmark") == (
        "alpaca",
        "ohlcv",
        "benchmark",
    )

    assert default_stream_id_for_source("equity", "alpaca.ohlcv.benchmark") == (
        "equity.ohlcv.benchmark"
    )


def test_existing_source_without_variant_keeps_existing_default_stream_id() -> None:
    assert default_stream_id_for_source("equity", "alpaca.ohlcv") == "equity.ohlcv"
