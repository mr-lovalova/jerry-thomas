import pytest

import jerrythomas.config.transforms as transform_config
from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.dataset.series import SeriesConfig, SequenceConfig
from jerrythomas.config.dataset.split import DatasetFold, HashSplitConfig
from jerrythomas.config.streams import (
    CombinedStreamConfig,
    CrossSectionStreamConfig,
    DerivedStreamConfig,
    SourceStreamConfig,
    StreamsConfig,
)
from jerrythomas.services.dataset import validate_dataset_streams


def _streams(
    partition_by: list[str],
    transforms: list[transform_config.TransformConfig] | None = None,
) -> StreamsConfig:
    stream = SourceStreamConfig.model_validate(
        {
            "id": "prices",
            "from": {"source": "raw"},
            "map": {"entrypoint": "identity"},
            "partition_by": partition_by,
            "transforms": [] if transforms is None else transforms,
        }
    )
    return StreamsConfig(streams={"prices": stream})


def _as_of_streams(max_age: str | None) -> StreamsConfig:
    prices = _streams(["ticker"]).streams["prices"]
    fundamentals = SourceStreamConfig.model_validate(
        {
            "id": "fundamentals",
            "from": {"source": "raw.fundamentals"},
            "map": {"entrypoint": "identity"},
            "partition_by": ["ticker"],
        }
    )
    joined = CombinedStreamConfig.model_validate(
        {
            "id": "joined",
            "from": {"stream": "prices"},
            "join": {
                "kind": "as_of",
                "lookup": "fundamentals",
                "max_age": max_age,
                "require_match": False,
            },
            "combine": {"entrypoint": "combine"},
        }
    )
    return StreamsConfig(
        streams={"prices": prices, "fundamentals": fundamentals, "joined": joined}
    )


def _dataset(
    sample_keys: list[str] | None = None,
    stream: str = "prices",
    sequence: SequenceConfig | None = None,
    split: HashSplitConfig | None = None,
) -> DatasetConfig:
    feature = SeriesConfig(
        stream=stream,
        id="close",
        field="close",
        sequence=sequence,
    )
    return DatasetConfig(
        sample=SampleConfig(cadence="1d", keys=sample_keys or []),
        features=[feature],
        split=split,
    )


def _hash_split() -> HashSplitConfig:
    return HashSplitConfig(
        ratios={"train": 1.0},
        folds=[DatasetFold(id="default", train=["train"])],
    )


def test_dataset_feature_must_reference_known_stream() -> None:
    dataset = _dataset(stream="missing")

    with pytest.raises(ValueError, match="references unknown stream 'missing'"):
        validate_dataset_streams(dataset, StreamsConfig())


def test_dataset_accepts_long_identity() -> None:
    streams = _streams(["exchange", "security_id"])
    dataset = _dataset(sample_keys=["exchange", "security_id"])

    validate_dataset_streams(dataset, streams)


def test_dataset_accepts_wide_identity() -> None:
    validate_dataset_streams(_dataset(), _streams(["security_id"]))


def test_dataset_accepts_hybrid_identity() -> None:
    streams = _streams(["exchange", "security_id"])
    dataset = _dataset(sample_keys=["exchange"])

    validate_dataset_streams(dataset, streams)


def test_dataset_rejects_sample_key_outside_stream_partition() -> None:
    streams = _streams(["security_id"])
    dataset = _dataset(sample_keys=["exchange"])

    with pytest.raises(
        ValueError,
        match=r"sample\.keys \['exchange'\].*partition_by \['security_id'\]",
    ):
        validate_dataset_streams(dataset, streams)


def test_dataset_accepts_unpartitioned_scalar_feature() -> None:
    validate_dataset_streams(_dataset(), _streams([]))


def test_dataset_accepts_hybrid_sequence_identity() -> None:
    streams = _streams(["exchange", "security_id"])
    dataset = _dataset(
        sample_keys=["exchange"],
        sequence=SequenceConfig(size=2),
    )

    validate_dataset_streams(dataset, streams)


def test_dataset_rejects_sequences_with_hash_split() -> None:
    feature = SeriesConfig(
        stream="prices",
        id="close",
        field="close",
        sequence=SequenceConfig(size=2),
    )

    with pytest.raises(ValueError, match="hash splits cannot be used.*close"):
        DatasetConfig(
            sample=SampleConfig(cadence="1d"),
            features=[feature],
            split=HashSplitConfig(
                ratios={"train": 1.0},
                folds=[DatasetFold(id="default", train=["train"])],
            ),
        )


def test_hash_split_rejects_selected_cross_section_dependency() -> None:
    prices = _streams(["ticker"]).streams["prices"]
    ranked = CrossSectionStreamConfig.model_validate(
        {
            "id": "ranked",
            "from": {"stream": "prices"},
            "cross_section": [
                {
                    "operation": "rank_score",
                    "field": "close",
                    "to": "close_rank",
                    "min_samples": 2,
                }
            ],
        }
    )
    selected = DerivedStreamConfig.model_validate(
        {
            "id": "selected",
            "from": {"stream": "ranked"},
            "transforms": [{"operation": "dedupe"}],
        }
    )
    streams = StreamsConfig(
        streams={"prices": prices, "ranked": ranked, "selected": selected}
    )

    with pytest.raises(
        ValueError,
        match="hash splits cannot be used with cross-sectional streams: ranked",
    ):
        validate_dataset_streams(
            _dataset(stream="selected", split=_hash_split()),
            streams,
        )


def test_hash_split_allows_unselected_cross_section_stream() -> None:
    prices = _streams(["ticker"]).streams["prices"]
    ranked = CrossSectionStreamConfig.model_validate(
        {
            "id": "ranked",
            "from": {"stream": "prices"},
            "cross_section": [
                {
                    "operation": "rank_score",
                    "field": "close",
                    "to": "close_rank",
                    "min_samples": 2,
                }
            ],
        }
    )
    streams = StreamsConfig(streams={"prices": prices, "ranked": ranked})

    validate_dataset_streams(
        _dataset(stream="prices", split=_hash_split()),
        streams,
    )


@pytest.mark.parametrize(
    "transform",
    [
        transform_config.LagConfig(field="close", periods=1),
        transform_config.LeadConfig(field="close", periods=1),
        transform_config.ForwardSumConfig(field="close", window=2, to="forward_close"),
        transform_config.EnsureCadenceConfig(cadence="1d"),
        transform_config.EnsureScheduleConfig(schedule="schedule"),
        transform_config.FillConfig(field="close", window=2, statistic="mean"),
        transform_config.ForwardFillConfig(field="close"),
        transform_config.EwmMeanConfig(field="close", alpha=0.5),
        transform_config.RollingConfig(field="close", window=2),
        transform_config.RollingQuantileConfig(field="close", window=2, quantile=0.5),
        transform_config.RollingSlopeConfig(
            x="market", y="close", window=2, to="slope"
        ),
        transform_config.RollingOlsConfig(
            y="close",
            x=("market", "credit"),
            window=3,
            coefficient="credit",
            to="credit_beta",
        ),
    ],
    ids=lambda transform: transform.operation,
)
def test_hash_split_rejects_cross_timestamp_transform_dependency(
    transform: transform_config.TransformConfig,
) -> None:
    prices = _streams(["ticker"], [transform]).streams["prices"]
    selected = DerivedStreamConfig.model_validate(
        {
            "id": "selected",
            "from": {"stream": "prices"},
            "transforms": [{"operation": "dedupe"}],
        }
    )
    streams = StreamsConfig(streams={"prices": prices, "selected": selected})

    with pytest.raises(
        ValueError,
        match="hash splits cannot be used with cross-timestamp streams: prices",
    ):
        validate_dataset_streams(
            _dataset(stream="selected", split=_hash_split()),
            streams,
        )


@pytest.mark.parametrize("max_age", [None, "1d"])
def test_hash_split_rejects_temporal_as_of_dependency(
    max_age: str | None,
) -> None:
    with pytest.raises(
        ValueError,
        match="hash splits cannot be used with cross-timestamp streams: joined",
    ):
        validate_dataset_streams(
            _dataset(stream="joined", split=_hash_split()),
            _as_of_streams(max_age),
        )


def test_hash_split_allows_exact_optional_as_of() -> None:
    validate_dataset_streams(
        _dataset(stream="joined", split=_hash_split()),
        _as_of_streams("0s"),
    )


def test_hash_split_rejects_broadcast_as_of_dependency() -> None:
    prices = _streams(["ticker"]).streams["prices"]
    market = SourceStreamConfig.model_validate(
        {
            "id": "market",
            "from": {"source": "raw.market"},
            "map": {"entrypoint": "identity"},
        }
    )
    joined = CombinedStreamConfig.model_validate(
        {
            "id": "joined",
            "from": {"stream": "prices"},
            "join": {"kind": "broadcast_as_of", "lookup": "market", "max_age": "1d"},
            "combine": {"entrypoint": "combine"},
        }
    )
    streams = StreamsConfig(
        streams={"prices": prices, "market": market, "joined": joined}
    )

    with pytest.raises(
        ValueError,
        match="hash splits cannot be used with cross-timestamp streams: joined",
    ):
        validate_dataset_streams(
            _dataset(stream="joined", split=_hash_split()),
            streams,
        )
