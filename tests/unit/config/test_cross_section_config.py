import pytest
from pydantic import ValidationError

from jerrythomas.config.cross_section import OlsResidualConfig, RankScoreConfig
from jerrythomas.config.streams import CrossSectionStreamConfig
from jerrythomas.config.transforms import DedupeConfig


def _stream(**values: object) -> CrossSectionStreamConfig:
    return CrossSectionStreamConfig.model_validate(
        {
            "id": "signals.cross-sectional",
            "from": {"stream": "signals.temporal"},
            **values,
        }
    )


def test_cross_section_stream_parses_typed_operations_and_transforms() -> None:
    stream = _stream(
        cross_section=[
            {
                "operation": "rank_score",
                "field": "signal",
                "to": "signal_rank",
                "min_samples": 30,
            },
            {
                "operation": "ols_residual",
                "y": "signal_rank",
                "x": ["liquidity_rank", "volatility_rank"],
                "to": "signal_residual",
                "min_samples": 30,
            },
        ],
        transforms=[{"operation": "dedupe"}],
    )

    assert stream.cross_section == [
        RankScoreConfig(
            field="signal",
            to="signal_rank",
            min_samples=30,
        ),
        OlsResidualConfig(
            y="signal_rank",
            x=("liquidity_rank", "volatility_rank"),
            to="signal_residual",
            min_samples=30,
        ),
    ]
    assert stream.transforms == [DedupeConfig()]
    assert stream.input_streams() == ("signals.temporal",)


def test_cross_section_stream_defaults_to_no_ordinary_transforms() -> None:
    stream = _stream(
        cross_section=[
            {
                "operation": "rank_score",
                "field": "signal",
                "to": "signal_rank",
                "min_samples": 2,
            }
        ]
    )

    assert stream.transforms == []


@pytest.mark.parametrize(
    "cross_section",
    [
        None,
        [],
        [{"operation": "unknown"}],
    ],
)
def test_cross_section_stream_requires_supported_operations(
    cross_section: object,
) -> None:
    values = {} if cross_section is None else {"cross_section": cross_section}

    with pytest.raises(ValidationError):
        _stream(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"field": "signal", "to": "rank"},
        {"field": "signal", "to": "rank", "min_samples": 1},
        {"field": "signal", "to": "rank", "min_samples": True},
        {"field": "signal", "to": "rank", "min_samples": 2.0},
        {
            "field": "signal",
            "to": "rank",
            "min_samples": 2,
            "method": "dense",
        },
    ],
)
def test_rank_score_rejects_invalid_contracts(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RankScoreConfig.model_validate(values)


def test_rank_score_config_is_frozen() -> None:
    config = RankScoreConfig(field="signal", to="rank", min_samples=2)

    with pytest.raises(ValidationError, match="frozen"):
        config.min_samples = 3


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {"y": "signal", "x": [], "to": "residual", "min_samples": 2},
            "at least 1",
        ),
        (
            {
                "y": "signal",
                "x": ["control", "control"],
                "to": "residual",
                "min_samples": 3,
            },
            "duplicate",
        ),
        (
            {
                "y": "signal",
                "x": ["signal"],
                "to": "residual",
                "min_samples": 2,
            },
            "dependent field",
        ),
        (
            {
                "y": "signal",
                "x": ["control_a", "control_b"],
                "to": "residual",
                "min_samples": 2,
            },
            "min_samples",
        ),
        (
            {"y": "signal", "x": ["control"], "to": "residual"},
            "min_samples",
        ),
    ],
)
def test_ols_residual_rejects_invalid_contracts(
    values: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        OlsResidualConfig.model_validate(values)


def test_ols_residual_config_is_frozen() -> None:
    config = OlsResidualConfig(
        y="signal",
        x=("control",),
        to="residual",
        min_samples=2,
    )

    with pytest.raises(ValidationError, match="frozen"):
        config.to = "other"
