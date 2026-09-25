import pytest

from jerrythomas.config.sources import SourceConfig
from jerrythomas.config.streams import (
    CombinedStreamConfig,
    CrossSectionStreamConfig,
    DerivedStreamConfig,
    SourceStreamConfig,
    StreamConfig,
)
from jerrythomas.services.streams.validation import (
    stream_partition_by,
    validate_stream_configs,
)


def _source(source_id: str = "source.alias") -> SourceConfig:
    return SourceConfig.model_validate(
        {
            "id": source_id,
            "parser": {"entrypoint": "core.identity"},
            "loader": {"entrypoint": "load"},
            "freshness": "opaque",
        }
    )


def _source_stream(
    stream_id: str,
    partition_by: list[str] | None = None,
    presorted: bool = False,
    transforms: list[dict[str, object]] | None = None,
) -> SourceStreamConfig:
    return SourceStreamConfig.model_validate(
        {
            "id": stream_id,
            "from": {"source": "source.alias"},
            "map": {"entrypoint": "core.identity"},
            "partition_by": [] if partition_by is None else partition_by,
            "presorted": presorted,
            "transforms": [] if transforms is None else transforms,
        }
    )


def _derived(
    stream_id: str,
    upstream: str,
    transforms: list[dict[str, object]] | None = None,
) -> DerivedStreamConfig:
    return DerivedStreamConfig.model_validate(
        {
            "id": stream_id,
            "from": {"stream": upstream},
            "transforms": (
                [{"operation": "dedupe"}] if transforms is None else transforms
            ),
        }
    )


def _cross_section(
    stream_id: str,
    upstream: str,
    operations: list[dict[str, object]] | None = None,
    transforms: list[dict[str, object]] | None = None,
) -> CrossSectionStreamConfig:
    return CrossSectionStreamConfig.model_validate(
        {
            "id": stream_id,
            "from": {"stream": upstream},
            "cross_section": (
                [
                    {
                        "operation": "rank_score",
                        "field": "value",
                        "to": "rank",
                        "min_samples": 2,
                    }
                ]
                if operations is None
                else operations
            ),
            "transforms": [] if transforms is None else transforms,
        }
    )


def _aligned(stream_id: str, inputs: list[str]) -> CombinedStreamConfig:
    primary, *rest = inputs
    return CombinedStreamConfig.model_validate(
        {
            "id": stream_id,
            "from": {"stream": primary},
            "join": {"kind": "align", "streams": rest},
            "combine": {"entrypoint": "calculate"},
        }
    )


def _broadcast(
    stream_id: str,
    primary: str,
    broadcast: str,
    transforms: list[dict[str, object]] | None = None,
) -> CombinedStreamConfig:
    return CombinedStreamConfig.model_validate(
        {
            "id": stream_id,
            "from": {"stream": primary},
            "join": {"kind": "broadcast", "with": broadcast},
            "combine": {"entrypoint": "attach_reference"},
            "transforms": [] if transforms is None else transforms,
        }
    )


def _as_of(
    stream_id: str,
    primary: str,
    lookup: str,
) -> CombinedStreamConfig:
    return CombinedStreamConfig.model_validate(
        {
            "id": stream_id,
            "from": {"stream": primary},
            "join": {"kind": "as_of", "lookup": lookup},
            "combine": {"entrypoint": "attach_lookup"},
        }
    )


def _broadcast_as_of(
    stream_id: str,
    primary: str,
    lookup: str,
) -> CombinedStreamConfig:
    return CombinedStreamConfig.model_validate(
        {
            "id": stream_id,
            "from": {"stream": primary},
            "join": {"kind": "broadcast_as_of", "lookup": lookup},
            "combine": {"entrypoint": "attach_lookup"},
        }
    )


def test_validation_rejects_unknown_source() -> None:
    streams: dict[str, StreamConfig] = {"prices": _source_stream("prices")}

    with pytest.raises(ValueError, match="references unknown source 'source.alias'"):
        validate_stream_configs({}, streams)


def test_validation_rejects_unknown_stream() -> None:
    streams: dict[str, StreamConfig] = {"returns": _derived("returns", "prices")}

    with pytest.raises(ValueError, match="references unknown stream"):
        validate_stream_configs({}, streams)


def test_validation_checks_both_broadcast_inputs() -> None:
    streams: dict[str, StreamConfig] = {
        "primary": _source_stream("primary", partition_by=["station"]),
        "enriched": _broadcast("enriched", "primary", "missing.reference"),
    }

    with pytest.raises(
        ValueError,
        match=r"references unknown stream\(s\): \['missing.reference'\]",
    ):
        validate_stream_configs({"source.alias": _source()}, streams)


def test_validation_rejects_dependency_cycle() -> None:
    streams: dict[str, StreamConfig] = {
        "first": _derived("first", "second"),
        "second": _derived("second", "first"),
    }

    with pytest.raises(ValueError, match="first -> second -> first"):
        validate_stream_configs({}, streams)


def test_validation_detects_cycle_through_broadcast_input() -> None:
    streams: dict[str, StreamConfig] = {
        "primary": _source_stream("primary", partition_by=["station"]),
        "enriched": _broadcast("enriched", "primary", "reference"),
        "reference": _derived("reference", "enriched"),
    }

    with pytest.raises(ValueError, match="enriched -> reference -> enriched"):
        validate_stream_configs({"source.alias": _source()}, streams)


def test_derived_partition_inheritance_is_transitive() -> None:
    streams: dict[str, StreamConfig] = {
        "prices": _source_stream("prices", partition_by=["ticker"]),
        "daily": _derived("daily", "prices"),
        "returns": _derived("returns", "daily"),
    }

    validate_stream_configs({"source.alias": _source()}, streams)

    assert stream_partition_by(streams, "returns") == ("ticker",)


def test_cross_section_inherits_transitive_partition() -> None:
    streams: dict[str, StreamConfig] = {
        "prices": _source_stream("prices", partition_by=["ticker"]),
        "daily": _derived("daily", "prices"),
        "ranked": _cross_section("ranked", "daily"),
    }

    validate_stream_configs({"source.alias": _source()}, streams)

    assert stream_partition_by(streams, "ranked") == ("ticker",)


def test_validation_rejects_unpartitioned_cross_section_input() -> None:
    streams: dict[str, StreamConfig] = {
        "global": _source_stream("global"),
        "ranked": _cross_section("ranked", "global"),
    }

    with pytest.raises(
        ValueError,
        match=(
            "Cross-section stream 'ranked' input 'global' must have a non-empty "
            "partition_by"
        ),
    ):
        validate_stream_configs({"source.alias": _source()}, streams)


def test_broadcast_inherits_transitive_primary_partition() -> None:
    streams: dict[str, StreamConfig] = {
        "measurements": _source_stream("measurements", partition_by=["station"]),
        "primary": _derived("primary", "measurements"),
        "global": _source_stream("global"),
        "reference": _derived("reference", "global"),
        "enriched": _broadcast("enriched", "primary", "reference"),
    }

    validate_stream_configs({"source.alias": _source()}, streams)

    assert stream_partition_by(streams, "enriched") == ("station",)


def test_as_of_inherits_matching_transitive_partition() -> None:
    streams: dict[str, StreamConfig] = {
        "prices": _source_stream("prices", partition_by=["ticker"]),
        "primary": _derived("primary", "prices"),
        "reports": _source_stream("reports", partition_by=["ticker"]),
        "lookup": _derived("lookup", "reports"),
        "enriched": _as_of("enriched", "primary", "lookup"),
    }

    validate_stream_configs({"source.alias": _source()}, streams)

    assert stream_partition_by(streams, "enriched") == ("ticker",)


def test_as_of_accepts_two_global_inputs() -> None:
    streams: dict[str, StreamConfig] = {
        "primary": _source_stream("primary"),
        "lookup": _source_stream("lookup"),
        "enriched": _as_of("enriched", "primary", "lookup"),
    }

    validate_stream_configs({"source.alias": _source()}, streams)

    assert stream_partition_by(streams, "enriched") == ()


def test_validation_rejects_as_of_partition_mismatch() -> None:
    streams: dict[str, StreamConfig] = {
        "prices": _source_stream("prices", partition_by=["ticker"]),
        "reports": _source_stream("reports", partition_by=["company_id"]),
        "enriched": _as_of("enriched", "prices", "reports"),
    }

    with pytest.raises(
        ValueError,
        match=r"partition_by \['company_id'\]; expected \['ticker'\]",
    ):
        validate_stream_configs({"source.alias": _source()}, streams)


def test_broadcast_as_of_inherits_transitive_primary_partition() -> None:
    streams: dict[str, StreamConfig] = {
        "measurements": _source_stream("measurements", partition_by=["station"]),
        "primary": _derived("primary", "measurements"),
        "global": _source_stream("global"),
        "lookup": _derived("lookup", "global"),
        "enriched": _broadcast_as_of("enriched", "primary", "lookup"),
    }

    validate_stream_configs({"source.alias": _source()}, streams)

    assert stream_partition_by(streams, "enriched") == ("station",)


def test_validation_rejects_unpartitioned_broadcast_as_of_primary() -> None:
    streams: dict[str, StreamConfig] = {
        "primary": _source_stream("primary"),
        "lookup": _source_stream("lookup"),
        "enriched": _broadcast_as_of("enriched", "primary", "lookup"),
    }

    with pytest.raises(
        ValueError,
        match=(
            "Broadcast as-of stream 'enriched' primary input 'primary' "
            "must have a non-empty partition_by"
        ),
    ):
        validate_stream_configs({"source.alias": _source()}, streams)


def test_validation_rejects_partitioned_broadcast_as_of_lookup() -> None:
    streams: dict[str, StreamConfig] = {
        "primary": _source_stream("primary", partition_by=["station"]),
        "lookup": _source_stream("lookup", partition_by=["region"]),
        "enriched": _broadcast_as_of("enriched", "primary", "lookup"),
    }

    with pytest.raises(
        ValueError,
        match=(
            r"Stream 'enriched' join input 'lookup' must have an "
            r"empty partition_by; got \['region'\]"
        ),
    ):
        validate_stream_configs({"source.alias": _source()}, streams)


def test_validation_rejects_unpartitioned_broadcast_primary() -> None:
    streams: dict[str, StreamConfig] = {
        "measurements": _source_stream("measurements"),
        "reference": _source_stream("reference"),
        "enriched": _broadcast("enriched", "measurements", "reference"),
    }

    with pytest.raises(
        ValueError,
        match=(
            "Broadcast stream 'enriched' primary input 'measurements' "
            "must have a non-empty partition_by"
        ),
    ):
        validate_stream_configs({"source.alias": _source()}, streams)


def test_validation_rejects_partitioned_broadcast_input() -> None:
    streams: dict[str, StreamConfig] = {
        "measurements": _source_stream("measurements", partition_by=["station"]),
        "reference": _source_stream("reference", partition_by=["region"]),
        "enriched": _broadcast("enriched", "measurements", "reference"),
    }

    with pytest.raises(
        ValueError,
        match=(
            r"Stream 'enriched' join input 'reference' must have an "
            r"empty partition_by; got \['region'\]"
        ),
    ):
        validate_stream_configs({"source.alias": _source()}, streams)


def test_aligned_partition_inheritance_is_transitive() -> None:
    streams: dict[str, StreamConfig] = {
        "a": _source_stream("a", partition_by=["ticker"]),
        "b": _source_stream("b", partition_by=["ticker"]),
        "c": _source_stream("c", partition_by=["ticker"]),
        "first": _aligned("first", ["a", "b"]),
        "second": _aligned("second", ["first", "c"]),
    }

    validate_stream_configs({"source.alias": _source()}, streams)

    assert stream_partition_by(streams, "second") == ("ticker",)


def test_validation_rejects_aligned_partition_mismatch() -> None:
    streams: dict[str, StreamConfig] = {
        "a": _source_stream("a", partition_by=["station"]),
        "b": _source_stream("b", partition_by=["ticker"]),
        "aligned": _aligned("aligned", ["a", "b"]),
    }

    with pytest.raises(
        ValueError,
        match=r"partition_by \['ticker'\]; expected \['station'\]",
    ):
        validate_stream_configs({"source.alias": _source()}, streams)


def test_validation_accepts_declared_canonical_order() -> None:
    streams: dict[str, StreamConfig] = {
        "prices": _source_stream(
            "prices",
            partition_by=["ticker"],
            presorted=True,
        )
    }

    validate_stream_configs({"source.alias": _source()}, streams)


@pytest.mark.parametrize(
    "operation",
    [
        {"operation": "lag", "field": "ticker", "periods": 1},
        {
            "operation": "lead",
            "field": "close",
            "periods": 1,
            "to": "ticker",
        },
        {
            "operation": "fill",
            "field": "close",
            "window": 2,
            "statistic": "mean",
            "to": "ticker",
        },
        {"operation": "forward_fill", "field": "close", "to": "ticker"},
        {"operation": "fill_missing", "field": "close", "value": 0, "to": "ticker"},
        {"operation": "aggregate_sum", "field": "ticker"},
        {
            "operation": "resample",
            "period": {"kind": "calendar", "unit": "month", "timezone": "UTC"},
            "aggregations": {"ticker": {"field": "value", "statistic": "last"}},
        },
        {
            "operation": "aggregate_sum",
            "field": "value",
            "count_to": "ticker",
        },
        {"operation": "rolling", "field": "close", "window": 2, "to": "ticker"},
        {
            "operation": "rolling_quantile",
            "field": "close",
            "window": 2,
            "quantile": 0.9,
            "to": "ticker",
        },
        {
            "operation": "ewm_mean",
            "field": "close",
            "alpha": 0.1,
            "to": "ticker",
        },
        {
            "operation": "rolling_slope",
            "x": "market_return",
            "y": "stock_return",
            "window": 2,
            "to": "ticker",
        },
        {
            "operation": "rolling_ols",
            "y": "stock_return",
            "x": ["market_return", "credit_return"],
            "window": 3,
            "coefficient": "credit_return",
            "to": "ticker",
        },
        {
            "operation": "forward_sum",
            "field": "return",
            "window": 2,
            "to": "ticker",
        },
        {"operation": "log", "field": "price", "to": "ticker"},
        {"operation": "log1p", "field": "return", "to": "ticker"},
        {
            "operation": "derive",
            "left": "close",
            "operator": "mul",
            "right_value": 2,
            "to": "ticker",
        },
    ],
)
def test_transforms_cannot_write_partition_fields(
    operation: dict[str, object],
) -> None:
    streams: dict[str, StreamConfig] = {
        "prices": _source_stream("prices", partition_by=["ticker"]),
        "derived": _derived("derived", "prices", transforms=[operation]),
    }

    with pytest.raises(
        ValueError,
        match="cannot write canonical order field 'ticker'",
    ):
        validate_stream_configs({"source.alias": _source()}, streams)


def test_source_stream_transforms_use_the_same_order_invariant() -> None:
    streams: dict[str, StreamConfig] = {
        "prices": _source_stream(
            "prices",
            partition_by=["ticker"],
            transforms=[
                {
                    "operation": "derive",
                    "left": "close",
                    "operator": "mul",
                    "right_value": 2,
                    "to": "time",
                }
            ],
        )
    }

    with pytest.raises(
        ValueError,
        match="cannot write canonical order field 'time'",
    ):
        validate_stream_configs({"source.alias": _source()}, streams)


@pytest.mark.parametrize(
    ("operation", "field"),
    [
        (
            {
                "operation": "rank_score",
                "field": "signal",
                "to": "ticker",
                "min_samples": 2,
            },
            "ticker",
        ),
        (
            {
                "operation": "ols_residual",
                "y": "signal",
                "x": ["control"],
                "to": "time",
                "min_samples": 2,
            },
            "time",
        ),
    ],
)
def test_cross_section_operations_cannot_write_canonical_fields(
    operation: dict[str, object],
    field: str,
) -> None:
    streams: dict[str, StreamConfig] = {
        "prices": _source_stream("prices", partition_by=["ticker"]),
        "cross-sectional": _cross_section(
            "cross-sectional",
            "prices",
            operations=[operation],
        ),
    }

    with pytest.raises(
        ValueError,
        match=f"cannot write canonical order field '{field}'",
    ):
        validate_stream_configs({"source.alias": _source()}, streams)


def test_cross_section_ordinary_transforms_use_inherited_partition() -> None:
    streams: dict[str, StreamConfig] = {
        "prices": _source_stream("prices", partition_by=["ticker"]),
        "cross-sectional": _cross_section(
            "cross-sectional",
            "prices",
            transforms=[
                {
                    "operation": "derive",
                    "left": "rank",
                    "operator": "mul",
                    "right_value": 2,
                    "to": "ticker",
                }
            ],
        ),
    }

    with pytest.raises(
        ValueError,
        match="cannot write canonical order field 'ticker'",
    ):
        validate_stream_configs({"source.alias": _source()}, streams)


def test_aligned_transforms_use_the_inherited_partition() -> None:
    aligned = CombinedStreamConfig.model_validate(
        {
            "id": "market_cap",
            "from": {"stream": "prices"},
            "join": {"kind": "align", "streams": ["shares"]},
            "combine": {"entrypoint": "calculate"},
            "transforms": [
                {
                    "operation": "derive",
                    "left": "price",
                    "operator": "mul",
                    "right_field": "shares",
                    "to": "ticker",
                }
            ],
        }
    )
    streams: dict[str, StreamConfig] = {
        "prices": _source_stream("prices", partition_by=["ticker"]),
        "shares": _source_stream("shares", partition_by=["ticker"]),
        "market_cap": aligned,
    }

    with pytest.raises(
        ValueError,
        match="cannot write canonical order field 'ticker'",
    ):
        validate_stream_configs({"source.alias": _source()}, streams)
