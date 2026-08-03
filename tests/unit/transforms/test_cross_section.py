import builtins
from datetime import datetime, timezone

import pytest

from jerrythomas.domain.record import TemporalRecord
from jerrythomas.transforms.cross_section import (
    OlsResidualTransform,
    RankScoreTransform,
)


def _record(value: object, **fields: object) -> TemporalRecord:
    record = TemporalRecord(time=datetime(2024, 1, 1, tzinfo=timezone.utc))
    record.value = value
    for field, field_value in fields.items():
        setattr(record, field, field_value)
    return record


def test_rank_score_uses_normalized_average_ranks_without_mutating_inputs() -> None:
    records = [_record(value) for value in (30.0, 10.0, 20.0, 20.0)]
    transform = RankScoreTransform("value", "score", min_samples=2)

    output = transform.apply(records)

    assert [record.score for record in output] == pytest.approx([0.5, -0.5, 0.0, 0.0])
    assert all(not hasattr(record, "score") for record in records)
    assert all(result is not source for result, source in zip(output, records))


@pytest.mark.parametrize("min_samples", [0, 1])
def test_rank_score_requires_two_samples(min_samples: int) -> None:
    with pytest.raises(ValueError, match="min_samples must be at least 2"):
        RankScoreTransform("value", "score", min_samples)


def test_rank_score_preserves_missing_values() -> None:
    records = [_record(None), _record(1.0), _record(float("nan")), _record(3.0)]

    output = RankScoreTransform("value", "score", min_samples=2).apply(records)

    assert [record.score for record in output] == [None, -0.5, None, 0.5]


def test_rank_score_maps_an_all_equal_cross_section_to_zero() -> None:
    records = [_record(4.0), _record(4.0), _record(4.0)]

    output = RankScoreTransform("value", "score", min_samples=2).apply(records)

    assert [record.score for record in output] == [0.0, 0.0, 0.0]


def test_rank_score_preserves_large_integer_order() -> None:
    records = [_record(2**53), _record(2**53 + 1)]

    output = RankScoreTransform("value", "score", min_samples=2).apply(records)

    assert [record.score for record in output] == [-0.5, 0.5]


def test_rank_score_emits_only_missing_values_below_min_samples() -> None:
    records = [_record(None), _record(2.0), _record(3.0)]

    output = RankScoreTransform("value", "score", min_samples=3).apply(records)

    assert [record.score for record in output] == [None, None, None]


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (True, TypeError, "numeric values"),
        ("1", TypeError, "numeric values"),
        (float("inf"), ValueError, "finite numeric values"),
    ],
)
def test_rank_score_rejects_invalid_numeric_values(
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    transform = RankScoreTransform("value", "score", min_samples=2)

    with pytest.raises(error, match=message):
        transform.apply([_record(value), _record(1.0)])


def test_rank_score_requires_its_configured_field() -> None:
    record = _record(1.0)
    del record.value

    with pytest.raises(KeyError, match="value"):
        RankScoreTransform("value", "score", min_samples=2).apply([record])


def test_ols_residual_computes_complete_case_residuals_without_mutating_inputs() -> (
    None
):
    records = [
        _record(0.0, market=1.0, size=0.0, stock=4.0),
        _record(0.0, market=2.0, size=1.0, stock=8.5),
        _record(0.0, market=4.0, size=-1.0, stock=10.0),
        _record(0.0, market=5.0, size=2.0, stock=18.5),
        _record(0.0, market=7.0, size=1.0, stock=20.0),
    ]
    transform = OlsResidualTransform(
        y="stock",
        x=("market", "size"),
        to="residual",
        min_samples=4,
    )

    output = transform.apply(records)

    assert sum(record.residual for record in output) == pytest.approx(0.0, abs=1e-12)
    assert any(abs(record.residual) > 1e-6 for record in output)
    assert all(not hasattr(record, "residual") for record in records)
    assert all(result is not source for result, source in zip(output, records))


def test_ols_residual_matches_an_exact_linear_model_with_an_intercept() -> None:
    records = [
        _record(
            0.0,
            market=market,
            size=size,
            stock=7.0 + 2.5 * market - 1.25 * size,
        )
        for market, size in [(1.0, 2.0), (2.0, -1.0), (4.0, 3.0), (7.0, 0.5)]
    ]

    output = OlsResidualTransform(
        "stock",
        ("market", "size"),
        "residual",
        min_samples=3,
    ).apply(records)

    assert [record.residual for record in output] == pytest.approx(
        [0.0, 0.0, 0.0, 0.0],
        abs=1e-12,
    )


def test_ols_residual_uses_complete_cases_and_marks_incomplete_rows_missing() -> None:
    records = [
        _record(0.0, market=1.0, size=0.0, stock=4.0),
        _record(0.0, market=None, size=1.0, stock=8.0),
        _record(0.0, market=3.0, size=1.0, stock=float("nan")),
        _record(0.0, market=4.0, size=-1.0, stock=12.0),
        _record(0.0, market=5.0, size=2.0, stock=17.0),
    ]

    output = OlsResidualTransform(
        "stock",
        ("market", "size"),
        "residual",
        min_samples=3,
    ).apply(records)

    assert output[0].residual == pytest.approx(0.0, abs=1e-12)
    assert output[1].residual is None
    assert output[2].residual is None
    assert output[3].residual == pytest.approx(0.0, abs=1e-12)
    assert output[4].residual == pytest.approx(0.0, abs=1e-12)


def test_ols_residual_emits_only_missing_values_below_min_samples() -> None:
    records = [
        _record(0.0, market=1.0, stock=2.0),
        _record(0.0, market=2.0, stock=None),
        _record(0.0, market=3.0, stock=6.0),
    ]

    output = OlsResidualTransform(
        "stock",
        ("market",),
        "residual",
        min_samples=3,
    ).apply(records)

    assert [record.residual for record in output] == [None, None, None]


def test_ols_residual_rejects_rank_deficient_designs() -> None:
    records = [
        _record(0.0, first=value, second=2.0 * value, outcome=3.0 * value)
        for value in (1.0, 2.0, 3.0, 4.0)
    ]

    with pytest.raises(ValueError, match="rank deficient"):
        OlsResidualTransform(
            "outcome",
            ("first", "second"),
            "residual",
            min_samples=3,
        ).apply(records)


@pytest.mark.parametrize("field", ["market", "stock"])
@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (True, TypeError, "numeric values"),
        (float("inf"), ValueError, "finite numeric values"),
    ],
)
def test_ols_residual_rejects_invalid_numeric_values(
    field: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    record = _record(0.0, market=1.0, stock=2.0)
    setattr(record, field, value)

    with pytest.raises(error, match=message):
        OlsResidualTransform(
            "stock",
            ("market",),
            "residual",
            min_samples=2,
        ).apply([record])


def test_ols_residual_requires_every_configured_field() -> None:
    record = _record(0.0, market=1.0, stock=2.0)
    del record.market

    with pytest.raises(KeyError, match="market"):
        OlsResidualTransform(
            "stock",
            ("market",),
            "residual",
            min_samples=2,
        ).apply([record])


def test_ols_residual_reports_floating_point_overflow() -> None:
    records = [
        _record(0.0, market=market, stock=stock)
        for market, stock in [(1e308, 1.0), (1e308, 2.0), (-1e308, 3.0)]
    ]

    with pytest.raises(OverflowError, match="floating-point range"):
        OlsResidualTransform(
            "stock",
            ("market",),
            "residual",
            min_samples=3,
        ).apply(records)


def test_ols_residual_loads_numpy_only_when_a_group_can_be_fitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def import_without_numpy(name: str, *args: object, **kwargs: object) -> object:
        if name == "numpy":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_numpy)
    transform = OlsResidualTransform(
        "stock",
        ("market",),
        "residual",
        min_samples=2,
    )

    insufficient = transform.apply([_record(0.0, market=1.0, stock=2.0)])
    assert insufficient[0].residual is None
    with pytest.raises(RuntimeError, match=r"jerry-thomas\[numerical\]"):
        transform.apply(
            [
                _record(0.0, market=1.0, stock=2.0),
                _record(0.0, market=2.0, stock=4.0),
            ]
        )
