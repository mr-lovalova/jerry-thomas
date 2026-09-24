import pytest
from pydantic import ValidationError

from jerrythomas.config.dataset.series import (
    ScalingConfig,
    SeriesConfig,
    SequenceConfig,
)
from jerrythomas.config.streams import DerivedStreamConfig, SourceStreamConfig
from jerrythomas.config.transforms import (
    AggregateSumConfig,
    CollapseConfig,
    DedupeConfig,
    EwmMeanConfig,
    EnsureCadenceConfig,
    EnsureScheduleConfig,
    FillConfig,
    FillMissingConfig,
    ForwardFillConfig,
    ForwardSumConfig,
    Log1pConfig,
    LogConfig,
    RoundTimeConfig,
    RollingConfig,
    RollingQuantileConfig,
    RollingOlsConfig,
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


@pytest.mark.parametrize("direction", ["floor", "ceil"])
def test_round_time_requires_explicit_direction_in_preprocess(direction: str) -> None:
    config = {"operation": "round_time", "cadence": "1h", "direction": direction}
    source = _source_stream(preprocess=[config])
    assert source.preprocess == [RoundTimeConfig(cadence="1h", direction=direction)]
    with pytest.raises(ValidationError):
        _stream(transforms=[config])


@pytest.mark.parametrize(
    "clause",
    [
        {"operation": "floor_time", "cadence": "1h"},
        {"operation": "ceil_time", "cadence": "1h"},
        {"operation": "round_time", "cadence": "1h"},
        {"operation": "round_time", "cadence": "1h", "direction": "nearest"},
        {"operation": "round_time", "cadence": "1h", "direction": None},
        {"operation": "round_time", "cadence": "0h", "direction": "floor"},
        {"operation": "round_time", "cadence": "-1h", "direction": "ceil"},
        {
            "operation": "round_time",
            "cadence": "1h",
            "direction": "ceil",
            "keep": "last",
        },
    ],
)
def test_preprocess_rejects_old_or_ambiguous_time_rounding(clause: dict) -> None:
    with pytest.raises(ValidationError):
        _source_stream(preprocess=[clause])


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
                "operation": "rolling_quantile",
                "field": "close",
                "window": 20,
                "quantile": 0.9,
                "to": "close_q90",
            },
            {
                "operation": "ewm_mean",
                "field": "close",
                "alpha": 0.1,
                "to": "close_ewm",
            },
            {
                "operation": "fill",
                "field": "close",
                "window": 5,
                "statistic": "median",
            },
            {"operation": "forward_fill", "field": "close", "to": "close_asof"},
            {"operation": "collapse", "keep": "last"},
            {"operation": "ensure_schedule", "schedule": "schedule"},
        ]
    )

    assert source_stream.preprocess == [ShiftTimeConfig(by="1d")]
    assert stream.transforms == [
        DedupeConfig(),
        RollingConfig(field="close", window=20, statistic="mean"),
        RollingQuantileConfig(
            field="close",
            window=20,
            quantile=0.9,
            to="close_q90",
        ),
        EwmMeanConfig(field="close", alpha=0.1, to="close_ewm"),
        FillConfig(field="close", window=5, statistic="median"),
        ForwardFillConfig(field="close", to="close_asof"),
        CollapseConfig(keep="last"),
        EnsureScheduleConfig(schedule="schedule"),
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
            "operation": "rolling_quantile",
            "field": "close",
            "window": 20,
            "quantile": 0.9,
            "to": "close_q90",
            "min_samples": None,
        },
        {
            "operation": "ewm_mean",
            "field": "close",
            "alpha": 0.1,
            "to": "close_ewm",
            "min_samples": 1,
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
        {"operation": "ensure_schedule", "schedule": "schedule"},
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
            "operation": "rolling_ols",
            "y": "stock_return",
            "x": ["market_return", "credit_return"],
            "window": 252,
            "coefficient": "credit_return",
        },
        {
            "operation": "rolling_ols",
            "y": "stock_return",
            "x": ["market_return", "credit_return"],
            "window": 252,
            "coefficient": "credit_return",
            "to": "credit_beta",
            "gap": 21,
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


def test_stream_parses_aggregate_and_literal_fill_configs() -> None:
    stream = _stream(
        transforms=[
            {
                "operation": "aggregate_sum",
                "field": "signed_amount",
                "count_to": "event_count",
            },
            {
                "operation": "fill_missing",
                "field": "event_count",
                "value": 0,
                "to": "event_count_filled",
            },
        ]
    )

    assert stream.transforms == [
        AggregateSumConfig(field="signed_amount", count_to="event_count"),
        FillMissingConfig(
            field="event_count",
            value=0,
            to="event_count_filled",
        ),
    ]


def test_aggregate_sum_requires_distinct_output_fields() -> None:
    with pytest.raises(ValidationError, match="count_to must differ from field"):
        AggregateSumConfig(field="events", count_to="events")


@pytest.mark.parametrize("value", [None, [], {}, float("nan"), float("inf")])
def test_fill_missing_requires_a_finite_scalar_literal(value: object) -> None:
    with pytest.raises(ValidationError, match="value|finite"):
        FillMissingConfig(field="value", value=value)  # type: ignore[arg-type]


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


def test_stream_parses_strict_rolling_ols_config() -> None:
    stream = _stream(
        transforms=[
            {
                "operation": "rolling_ols",
                "y": "stock_return",
                "x": ["market_return", "credit_return", "bond_return"],
                "window": 252,
                "coefficient": "credit_return",
                "to": "credit_beta",
            }
        ]
    )

    assert stream.transforms == [
        RollingOlsConfig(
            y="stock_return",
            x=("market_return", "credit_return", "bond_return"),
            window=252,
            coefficient="credit_return",
            to="credit_beta",
        )
    ]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {
                "y": "stock_return",
                "x": ["market_return"],
                "window": 252,
                "coefficient": "market_return",
                "to": "beta",
            },
            "at least 2",
        ),
        (
            {
                "y": "stock_return",
                "x": ["market_return", "market_return"],
                "window": 252,
                "coefficient": "market_return",
                "to": "beta",
            },
            "duplicate",
        ),
        (
            {
                "y": "stock_return",
                "x": ["market_return", "credit_return"],
                "window": 252,
                "coefficient": "bond_return",
                "to": "beta",
            },
            "coefficient",
        ),
        (
            {
                "y": "stock_return",
                "x": ["market_return", "stock_return"],
                "window": 252,
                "coefficient": "market_return",
                "to": "beta",
            },
            "dependent field",
        ),
        (
            {
                "y": "stock_return",
                "x": ["market_return", "credit_return"],
                "window": 2,
                "coefficient": "credit_return",
                "to": "beta",
            },
            "window",
        ),
    ],
)
def test_rolling_ols_rejects_invalid_contracts(
    values: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        RollingOlsConfig.model_validate(values)


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


@pytest.mark.parametrize("cadence", [None, "", "schedule", "0m", "-1h"])
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

    assert config.scale == ScalingConfig()
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
