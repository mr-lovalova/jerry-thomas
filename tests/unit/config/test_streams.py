import pytest
from pydantic import TypeAdapter, ValidationError

from jerrythomas.config.streams import (
    AsOfJoin,
    BroadcastAsOfJoin,
    BroadcastJoin,
    CombinedStreamConfig,
    CrossSectionStreamConfig,
    DerivedStreamConfig,
    SourceStreamConfig,
    StreamConfig,
    StreamsConfig,
)


def test_source_stream_has_explicit_pipeline_phases() -> None:
    stream = SourceStreamConfig.model_validate(
        {
            "id": "prices",
            "from": {"source": "vendor.prices"},
            "map": {"entrypoint": "map_price"},
            "preprocess": [
                {
                    "operation": "where",
                    "field": "time",
                    "operator": "ge",
                    "comparand": "2024-01-01T00:00:00Z",
                }
            ],
            "partition_by": ["ticker"],
            "ordered_by": ["ticker", "time"],
            "transforms": [
                {
                    "operation": "rolling",
                    "field": "close",
                    "window": 20,
                }
            ],
        }
    )

    assert stream.from_.source == "vendor.prices"
    assert stream.partition_by == ("ticker",)
    assert stream.ordered_by == ("ticker", "time")
    assert len(stream.preprocess) == 1
    assert len(stream.transforms) == 1


def test_source_stream_requires_mapper() -> None:
    with pytest.raises(ValueError, match="map"):
        SourceStreamConfig.model_validate(
            {"id": "prices", "from": {"source": "vendor.prices"}}
        )


def test_source_stream_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        SourceStreamConfig.model_validate(
            {
                "id": "prices",
                "from": {"source": "vendor.prices"},
                "map": {"entrypoint": "map_price"},
                "unexpected": True,
            }
        )


def test_derived_stream_inherits_its_input_contract() -> None:
    stream = DerivedStreamConfig.model_validate(
        {
            "id": "returns",
            "from": {"stream": " prices "},
            "transforms": [{"operation": "lag", "field": "close", "periods": 1}],
        }
    )

    assert stream.input_streams() == ("prices",)
    assert len(stream.transforms) == 1


def test_derived_stream_requires_a_transform() -> None:
    with pytest.raises(ValueError, match="at least 1 item"):
        DerivedStreamConfig.model_validate(
            {"id": "returns", "from": {"stream": "prices"}, "transforms": []}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("map", {"entrypoint": "identity"}),
        ("preprocess", []),
        ("partition_by", ["ticker"]),
        ("ordered_by", ["ticker", "time"]),
    ],
)
def test_derived_stream_rejects_source_fields(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        DerivedStreamConfig.model_validate(
            {"id": "returns", "from": {"stream": "prices"}, field: value}
        )


def test_aligned_stream_has_combiner_and_transforms() -> None:
    stream = CombinedStreamConfig.model_validate(
        {
            "id": "market_cap",
            "from": {"stream": " prices "},
            "join": {"kind": "align", "streams": [" shares "]},
            "combine": {"entrypoint": "market_cap"},
            "transforms": [{"operation": "dedupe"}],
        }
    )

    assert stream.input_streams() == ("prices", "shares")
    assert len(stream.transforms) == 1


def test_broadcast_stream_has_primary_broadcast_combiner_and_transforms() -> None:
    stream = CombinedStreamConfig.model_validate(
        {
            "id": "enriched",
            "from": {"stream": " partitioned.measurements "},
            "join": {"kind": "broadcast", "with": " global.reference "},
            "combine": {"entrypoint": "attach_reference"},
            "transforms": [{"operation": "dedupe"}],
        }
    )

    assert isinstance(stream.join, BroadcastJoin)
    assert stream.from_.stream == "partitioned.measurements"
    assert stream.join.with_ == "global.reference"
    assert stream.input_streams() == (
        "partitioned.measurements",
        "global.reference",
    )
    assert stream.combine.entrypoint == "attach_reference"
    assert len(stream.transforms) == 1


def test_as_of_stream_has_primary_lookup_and_match_policy() -> None:
    stream = CombinedStreamConfig.model_validate(
        {
            "id": "priced",
            "from": {"stream": " prices "},
            "join": {
                "kind": "as_of",
                "lookup": " fundamentals.reported ",
                "max_age": " 180d ",
                "require_match": False,
            },
            "combine": {"entrypoint": "attach_fundamentals"},
        }
    )

    assert stream.input_streams() == ("prices", "fundamentals.reported")
    assert isinstance(stream.join, AsOfJoin)
    assert stream.join.max_age == "180d"
    assert stream.join.require_match is False


def test_as_of_stream_direction_defaults_to_backward_and_validates() -> None:
    def _data(**join_extra):
        return {
            "id": "priced",
            "from": {"stream": "prices"},
            "join": {"kind": "as_of", "lookup": "reports", **join_extra},
            "combine": {"entrypoint": "attach"},
        }

    defaults = CombinedStreamConfig.model_validate(_data())
    assert defaults.join.direction == "backward"

    forward = CombinedStreamConfig.model_validate(_data(direction="forward"))
    assert forward.join.direction == "forward"

    with pytest.raises(ValidationError, match="direction"):
        CombinedStreamConfig.model_validate(_data(direction="sideways"))


def test_broadcast_as_of_stream_has_primary_and_global_lookup() -> None:
    stream = CombinedStreamConfig.model_validate(
        {
            "id": "factor_adjusted",
            "from": {"stream": " returns "},
            "join": {"kind": "broadcast_as_of", "lookup": " factors.published "},
            "combine": {"entrypoint": "attach_factor"},
        }
    )

    assert stream.input_streams() == ("returns", "factors.published")
    assert isinstance(stream.join, BroadcastAsOfJoin)
    assert stream.join.max_age is None
    assert stream.join.require_match is True


@pytest.mark.parametrize("join_kind", ["as_of", "broadcast_as_of"])
@pytest.mark.parametrize("max_age", ["-1d", "forever"])
def test_as_of_streams_require_a_non_negative_max_age(
    join_kind: str,
    max_age: str,
) -> None:
    with pytest.raises(ValueError, match="max_age|Unsupported timecode"):
        CombinedStreamConfig.model_validate(
            {
                "id": "joined",
                "from": {"stream": "primary"},
                "join": {"kind": join_kind, "lookup": "lookup", "max_age": max_age},
                "combine": {"entrypoint": "combine"},
            }
        )


@pytest.mark.parametrize("join_kind", ["as_of", "broadcast_as_of"])
def test_as_of_streams_accept_zero_max_age_for_exact_matching(join_kind: str) -> None:
    stream = CombinedStreamConfig.model_validate(
        {
            "id": "joined",
            "from": {"stream": "primary"},
            "join": {
                "kind": join_kind,
                "lookup": "lookup",
                "max_age": "0s",
                "require_match": False,
            },
            "combine": {"entrypoint": "combine"},
        }
    )

    assert stream.join.max_age == "0s"


@pytest.mark.parametrize("join_kind", ["as_of", "broadcast_as_of", "broadcast"])
def test_combined_streams_require_distinct_inputs(join_kind: str) -> None:
    join = (
        {"kind": join_kind, "with": "same"}
        if join_kind == "broadcast"
        else {"kind": join_kind, "lookup": "same"}
    )
    with pytest.raises(ValueError, match="must differ from the primary"):
        CombinedStreamConfig.model_validate(
            {
                "id": "joined",
                "from": {"stream": "same"},
                "join": join,
                "combine": {"entrypoint": "combine"},
            }
        )


@pytest.mark.parametrize("join_kind", ["as_of", "broadcast_as_of"])
@pytest.mark.parametrize("require_match", ["false", 0, 1])
def test_as_of_streams_require_an_explicit_boolean_match_policy(
    join_kind: str,
    require_match: object,
) -> None:
    with pytest.raises(ValueError, match="valid boolean"):
        CombinedStreamConfig.model_validate(
            {
                "id": "joined",
                "from": {"stream": "primary"},
                "join": {
                    "kind": join_kind,
                    "lookup": "lookup",
                    "require_match": require_match,
                },
                "combine": {"entrypoint": "combine"},
            }
        )


@pytest.mark.parametrize("field", ["map", "preprocess", "partition_by", "ordered_by"])
def test_broadcast_stream_rejects_other_stream_contracts(field: str) -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        CombinedStreamConfig.model_validate(
            {
                "id": "enriched",
                "from": {"stream": "measurements"},
                "join": {"kind": "broadcast", "with": "reference"},
                "combine": {"entrypoint": "attach_reference"},
                field: {"entrypoint": "identity"} if field == "map" else [],
            }
        )


def test_aligned_stream_requires_two_inputs() -> None:
    with pytest.raises(ValueError, match="at least 1 item"):
        CombinedStreamConfig.model_validate(
            {
                "id": "market_cap",
                "from": {"stream": "prices"},
                "join": {"kind": "align", "streams": []},
                "combine": {"entrypoint": "market_cap"},
            }
        )


def test_aligned_stream_rejects_duplicate_inputs() -> None:
    with pytest.raises(ValueError, match="must not contain duplicate"):
        CombinedStreamConfig.model_validate(
            {
                "id": "market_cap",
                "from": {"stream": "primary"},
                "join": {"kind": "align", "streams": ["shares", "shares"]},
                "combine": {"entrypoint": "market_cap"},
            }
        )

    with pytest.raises(ValueError, match="must differ from the primary"):
        CombinedStreamConfig.model_validate(
            {
                "id": "market_cap",
                "from": {"stream": "prices"},
                "join": {"kind": "align", "streams": ["prices"]},
                "combine": {"entrypoint": "market_cap"},
            }
        )


@pytest.mark.parametrize("field", ["map", "partition_by", "ordered_by"])
def test_aligned_stream_rejects_other_stream_contracts(field: str) -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        CombinedStreamConfig.model_validate(
            {
                "id": "market_cap",
                "from": {"stream": "prices"},
                "join": {"kind": "align", "streams": ["shares"]},
                "combine": {"entrypoint": "market_cap"},
                field: {"entrypoint": "identity"} if field == "map" else [],
            }
        )


@pytest.mark.parametrize(
    "partition_by", ["ticker", [""], ["ticker", "ticker"], ["time"]]
)
def test_source_stream_rejects_invalid_partition_fields(partition_by: object) -> None:
    with pytest.raises(ValueError, match="partition_by|tuple|at least 1 character"):
        SourceStreamConfig.model_validate(
            {
                "id": "prices",
                "from": {"source": "vendor.prices"},
                "map": {"entrypoint": "map_price"},
                "partition_by": partition_by,
            }
        )


def test_stream_catalog_selects_and_round_trips_all_concrete_stream_types() -> None:
    catalog = StreamsConfig.model_validate(
        {
            "streams": {
                "prices": {
                    "id": "prices",
                    "from": {"source": "vendor.prices"},
                    "map": {"entrypoint": "map_price"},
                },
                "returns": {
                    "id": "returns",
                    "from": {"stream": "prices"},
                    "transforms": [{"operation": "dedupe"}],
                },
                "ranked": {
                    "id": "ranked",
                    "from": {"stream": "returns"},
                    "cross_section": [
                        {
                            "operation": "rank_score",
                            "field": "value",
                            "to": "rank",
                            "min_samples": 2,
                        }
                    ],
                },
                "market_cap": {
                    "id": "market_cap",
                    "from": {"stream": "prices"},
                    "join": {"kind": "align", "streams": ["returns"]},
                    "combine": {"entrypoint": "market_cap"},
                },
                "enriched": {
                    "id": "enriched",
                    "from": {"stream": "prices"},
                    "join": {"kind": "broadcast", "with": "reference"},
                    "combine": {"entrypoint": "attach_reference"},
                },
                "reported": {
                    "id": "reported",
                    "from": {"stream": "prices"},
                    "join": {"kind": "as_of", "lookup": "fundamentals"},
                    "combine": {"entrypoint": "attach_fundamentals"},
                },
                "factor_adjusted": {
                    "id": "factor_adjusted",
                    "from": {"stream": "prices"},
                    "join": {"kind": "broadcast_as_of", "lookup": "factors"},
                    "combine": {"entrypoint": "attach_factors"},
                },
            }
        }
    )

    assert isinstance(catalog.streams["prices"], SourceStreamConfig)
    assert isinstance(catalog.streams["returns"], DerivedStreamConfig)
    assert isinstance(catalog.streams["ranked"], CrossSectionStreamConfig)
    assert isinstance(catalog.streams["market_cap"], CombinedStreamConfig)
    assert isinstance(catalog.streams["enriched"], CombinedStreamConfig)
    assert isinstance(catalog.streams["reported"], CombinedStreamConfig)
    assert isinstance(catalog.streams["factor_adjusted"], CombinedStreamConfig)
    assert StreamsConfig.model_validate(catalog.model_dump(by_alias=True)) == catalog


@pytest.mark.parametrize(
    ("config", "error_location"),
    [
        (
            {"id": "prices", "from": {"source": "vendor.prices"}},
            ("source", "map"),
        ),
        (
            {"id": "returns", "from": {"stream": "prices"}},
            ("derived", "transforms"),
        ),
        (
            {
                "id": "aligned",
                "from": {"stream": "prices"},
                "join": {"kind": "align", "streams": ["returns"]},
            },
            ("combined", "combine"),
        ),
    ],
)
def test_stream_union_reports_only_the_selected_contract(
    config: dict[str, object],
    error_location: tuple[str, str],
) -> None:
    with pytest.raises(ValidationError) as error:
        TypeAdapter(StreamConfig).validate_python(config)

    assert error.value.error_count() == 1
    assert error.value.errors()[0]["loc"] == error_location


def test_combined_stream_rejects_unknown_join_kind_without_guessing() -> None:
    with pytest.raises(ValidationError, match="Input tag 'merge'"):
        TypeAdapter(StreamConfig).validate_python(
            {
                "id": "enriched",
                "from": {"stream": "prices"},
                "join": {"kind": "merge", "with": "other"},
                "combine": {"entrypoint": "combine"},
            }
        )


def test_combined_stream_rejects_ambiguous_extra_join_keys() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        TypeAdapter(StreamConfig).validate_python(
            {
                "id": "enriched",
                "from": {"stream": "prices"},
                "join": {"kind": "broadcast", "with": "other", "lookup": "more"},
                "combine": {"entrypoint": "combine"},
            }
        )


def test_stream_catalog_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        StreamsConfig.model_validate({"strams": {}})


def test_stream_catalog_rejects_source_registry_key_mismatch() -> None:
    with pytest.raises(ValueError, match="Source registry key 'alias'.*'demo.source'"):
        StreamsConfig.model_validate(
            {
                "sources": {
                    "alias": {
                        "id": "demo.source",
                        "parser": {"entrypoint": "parse"},
                        "loader": {"entrypoint": "load"},
                        "freshness": "opaque",
                    }
                }
            }
        )


def test_stream_catalog_rejects_stream_registry_key_mismatch() -> None:
    with pytest.raises(ValueError, match="Stream registry key 'alias'.*'prices'"):
        StreamsConfig.model_validate(
            {
                "streams": {
                    "alias": {
                        "id": "prices",
                        "from": {"source": "vendor.prices"},
                        "map": {"entrypoint": "map_price"},
                    }
                }
            }
        )
