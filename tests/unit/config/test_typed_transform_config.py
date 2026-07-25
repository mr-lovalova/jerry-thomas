import pytest
from pydantic import ValidationError

from datapipeline.config.dataset.series import SeriesConfig, SequenceConfig
from datapipeline.config.streams import DerivedStreamConfig, SourceStreamConfig
from datapipeline.config.transforms import (
    CollapseConfig,
    DedupeConfig,
    EnsureCadenceConfig,
    EnsureTicksConfig,
    FillConfig,
    ForwardFillConfig,
    ForwardSumConfig,
    Log1pConfig,
    LogConfig,
    RollingConfig,
    RollingSlopeConfig,
    ShiftTimeConfig,
)


def _source_stream(**values: object) -> SourceStreamConfig:
    return SourceStreamConfig.model_validate(
        {
            "id": "prices.raw",
            "from": {"source": "prices"},
            "map": {"entrypoint": "identity"},
            **values,
        }
    )


def _stream(**values: object) -> DerivedStreamConfig:
    return DerivedStreamConfig.model_validate(
        {
            "id": "prices.daily",
            "from": {"stream": "prices.raw"},
            **values,
        }
    )


def test_streams_parse_builtins_into_typed_configs() -> None:
    source_stream = _source_stream(preprocess=[{"operation": "shift_time", "by": "1d"}])
    stream = _stream(
        transforms=[
            {"operation": "dedupe"},
            {
                "operation": "rolling",
                "field": "close",
                "window": 20,
                "statistic": "mean",
            },
            {
                "operation": "fill",
                "field": "close",
                "window": 5,
                "statistic": "median",
            },
            {"operation": "forward_fill", "field": "close", "to": "close_asof"},
            {"operation": "collapse", "keep": "last"},
            {"operation": "ensure_ticks", "artifact": "model_grid"},
        ]
    )

    assert source_stream.preprocess == [ShiftTimeConfig(by="1d")]
    assert stream.transforms == [
        DedupeConfig(),
        RollingConfig(field="close", window=20, statistic="mean"),
        FillConfig(field="close", window=5, statistic="median"),
        ForwardFillConfig(field="close", to="close_asof"),
        CollapseConfig(keep="last"),
        EnsureTicksConfig(artifact="model_grid"),
    ]
    assert stream.model_dump()["transforms"] == [
        {"operation": "dedupe"},
        {
            "operation": "rolling",
            "field": "close",
            "window": 20,
            "to": None,
            "min_samples": None,
            "statistic": "mean",
        },
        {
            "operation": "fill",
            "field": "close",
            "window": 5,
            "statistic": "median",
            "to": None,
            "min_samples": 1,
        },
        {
            "operation": "forward_fill",
            "field": "close",
            "to": "close_asof",
        },
        {"operation": "collapse", "keep": "last"},
        {"operation": "ensure_ticks", "artifact": "model_grid"},
    ]


@pytest.mark.parametrize(
    "clause",
    [
        {"operation": "rolling", "field": "close", "windwo": 20},
        {"operation": "rolling", "field": "close", "window": 2.5},
        {
            "operation": "rolling_slope",
            "x": "market_return",
            "y": "stock_return",
            "window": 20,
        },
        {
            "operation": "rolling_slope",
            "x": "market_return",
            "y": "stock_return",
            "window": 20,
            "to": "beta",
            "min_samples": 10,
        },
        {
            "operation": "forward_sum",
            "field": "return",
            "window": 21,
        },
        {
            "operation": "forward_sum",
            "field": "return",
            "window": 21,
            "to": "future_return",
            "min_samples": 1,
        },
        {"operation": "log", "field": "price"},
        {
            "operation": "log1p",
            "field": "return",
            "to": "log_return",
            "base": 10,
        },
        {
            "operation": "fill",
            "field": "close",
            "window": 5,
            "statistic": "typo",
        },
        {"operation": "fill", "field": "close", "method": "forward"},
        {"operation": "collapse", "keep": "mean"},
        {"operation": "collapse", "keep": "last", "field": "close"},
        {"operation": "floor_time", "cadence": "1h"},
        {"operation": "granularity", "field": "close", "mode": "last"},
        {"operation": "unknown"},
    ],
)
def test_stream_config_rejects_invalid_builtin_parameters(clause: object) -> None:
    with pytest.raises(ValidationError):
        _stream(transforms=[clause])


def test_stream_config_rejects_a_record_only_transform_model() -> None:
    with pytest.raises(ValidationError, match="shift_time"):
        _stream(transforms=[ShiftTimeConfig(by="1h")])


def test_stream_parses_strict_rolling_slope_config() -> None:
    stream = _stream(
        transforms=[
            {
                "operation": "rolling_slope",
                "x": "market_return",
                "y": "stock_return",
                "window": 252,
                "to": "beta",
            }
        ]
    )

    assert stream.transforms == [
        RollingSlopeConfig(
            x="market_return",
            y="stock_return",
            window=252,
            to="beta",
        )
    ]


def test_stream_parses_strict_forward_sum_config() -> None:
    stream = _stream(
        transforms=[
            {
                "operation": "forward_sum",
                "field": "return",
                "window": 21,
                "to": "future_return_21",
            }
        ]
    )

    assert stream.transforms == [
        ForwardSumConfig(
            field="return",
            window=21,
            to="future_return_21",
        )
    ]


def test_stream_parses_explicit_logarithm_configs() -> None:
    stream = _stream(
        transforms=[
            {"operation": "log", "field": "price", "to": "log_price"},
            {"operation": "log1p", "field": "return", "to": "log_return"},
        ]
    )

    assert stream.transforms == [
        LogConfig(field="price", to="log_price"),
        Log1pConfig(field="return", to="log_return"),
    ]


@pytest.mark.parametrize("window", [0, True, 1.5])
def test_forward_sum_requires_a_positive_integer_window(window: object) -> None:
    with pytest.raises(ValidationError, match="window"):
        ForwardSumConfig(
            field="return",
            window=window,
            to="future_return",
        )


@pytest.mark.parametrize("window", [1, True, 2.5])
def test_rolling_slope_requires_at_least_two_records(window: object) -> None:
    with pytest.raises(ValidationError, match="window"):
        RollingSlopeConfig(
            x="market_return",
            y="stock_return",
            window=window,
            to="beta",
        )


@pytest.mark.parametrize("cadence", [None, "", "ticks", "0m", "-1h"])
def test_ensure_cadence_requires_a_positive_duration(cadence: object) -> None:
    with pytest.raises(ValidationError):
        _stream(transforms=[{"operation": "ensure_cadence", "cadence": cadence}])

    assert EnsureCadenceConfig(cadence="1h").cadence == "1h"


@pytest.mark.parametrize(
    "sequence",
    [
        {"size": True},
        {"size": 1.5},
        {"size": "2"},
        {"size": 2, "strdie": 1},
    ],
)
def test_sequence_config_is_strict(sequence: object) -> None:
    with pytest.raises(ValidationError):
        SeriesConfig.model_validate(
            {
                "id": "close",
                "stream": "prices.daily",
                "field": "close",
                "sequence": sequence,
            }
        )


@pytest.mark.parametrize(
    "collect",
    [
        True,
        0,
        -1,
        1.5,
        "2",
        {"size": 2},
    ],
)
def test_collect_is_a_strict_positive_integer(collect: object) -> None:
    with pytest.raises(ValidationError):
        SeriesConfig.model_validate(
            {
                "id": "close",
                "stream": "prices.hourly",
                "field": "close",
                "collect": collect,
            }
        )


def test_series_config_rejects_sequence_and_collect_together() -> None:
    with pytest.raises(
        ValidationError,
        match="sequence and collect are mutually exclusive",
    ):
        SeriesConfig.model_validate(
            {
                "id": "close",
                "stream": "prices.hourly",
                "field": "close",
                "sequence": {"size": 4},
                "collect": 4,
            }
        )


def test_series_config_uses_explicit_shaping_fields() -> None:
    config = SeriesConfig.model_validate(
        {
            "id": "close",
            "stream": "prices.daily",
            "field": "close",
            "scale": True,
            "sequence": {"size": 20, "stride": 5},
        }
    )

    assert config.scale is True
    assert config.sequence == SequenceConfig(size=20, stride=5)

    collected = SeriesConfig.model_validate(
        {
            "id": "intraday_close",
            "stream": "prices.hourly",
            "field": "close",
            "collect": 4,
        }
    )

    assert collected.collect == 4
