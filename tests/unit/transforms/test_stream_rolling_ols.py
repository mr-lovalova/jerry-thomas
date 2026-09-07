import builtins
from datetime import datetime, timedelta, timezone
from random import Random

import numpy as np
import pytest

from jerrythomas.domain.record import TemporalRecord
from jerrythomas.transforms.stream.lag import LagTransform
from jerrythomas.transforms.stream.rolling_ols import RollingOlsTransform


def _record(
    market: object,
    credit: object,
    stock: object,
    position: int,
    partition: str = "A",
) -> TemporalRecord:
    record = TemporalRecord(
        time=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=position),
    )
    record.market = market
    record.credit = credit
    record.stock = stock
    record.partition = partition
    return record


def _coefficients(
    rows: list[tuple[object, object, object]],
    window: int,
) -> list[float | None]:
    records = [
        _record(market, credit, stock, position)
        for position, (market, credit, stock) in enumerate(rows)
    ]
    transform = RollingOlsTransform(
        y="stock",
        x=("market", "credit"),
        window=window,
        coefficient="credit",
        partition_fields=("partition",),
        to="credit_beta",
    )
    return [record.credit_beta for record in transform.apply(iter(records))]


def test_rolling_ols_emits_selected_coefficient_without_mutating_input() -> None:
    records = [
        _record(market, credit, 3.0 + 2.0 * market - 4.0 * credit, position)
        for position, (market, credit) in enumerate(
            [(1.0, 2.0), (2.0, -1.0), (4.0, 3.0), (7.0, 0.5)]
        )
    ]
    transform = RollingOlsTransform(
        y="stock",
        x=("market", "credit"),
        window=3,
        coefficient="credit",
        partition_fields=("partition",),
        to="credit_beta",
    )

    outputs = list(transform.apply(iter(records)))

    assert [record.credit_beta for record in outputs[:2]] == [None, None]
    assert [record.credit_beta for record in outputs[2:]] == pytest.approx([-4.0, -4.0])
    assert all(not hasattr(record, "credit_beta") for record in records)


def test_rolling_ols_matches_direct_least_squares_windows() -> None:
    random = Random(42)
    rows = []
    for _ in range(200):
        market = random.uniform(-2.0, 2.0)
        credit = random.uniform(-1.0, 3.0)
        noise = random.uniform(-0.05, 0.05)
        stock = 0.75 + 1.5 * market - 0.8 * credit + noise
        rows.append((market, credit, stock))

    window = 17
    actual = _coefficients(rows, window)

    for position, coefficient in enumerate(actual):
        if position < window - 1:
            assert coefficient is None
            continue
        current = np.asarray(
            rows[position - window + 1 : position + 1],
            dtype=np.float64,
        )
        design = np.column_stack((np.ones(window), current[:, :2]))
        expected = np.linalg.lstsq(design, current[:, 2], rcond=None)[0][2]
        assert coefficient == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_rolling_ols_resets_after_any_missing_value() -> None:
    actual = _coefficients(
        [
            (1.0, 1.0, 2.0),
            (2.0, 4.0, 3.0),
            (3.0, None, 4.0),
            (4.0, 2.0, 5.0),
            (5.0, 5.0, float("nan")),
            (6.0, 1.0, 7.0),
            (7.0, 4.0, 8.0),
            (8.0, 2.0, 9.0),
        ],
        window=3,
    )

    assert actual[:-1] == [None] * 7
    assert actual[-1] is not None


def test_rolling_ols_resets_between_partitions() -> None:
    records = [
        _record(1.0, 1.0, 2.0, 0, "A"),
        _record(2.0, 4.0, 3.0, 1, "A"),
        _record(3.0, 2.0, 5.0, 2, "A"),
        _record(10.0, 3.0, -1.0, 3, "B"),
        _record(20.0, 7.0, -2.0, 4, "B"),
        _record(15.0, 9.0, -3.0, 5, "B"),
    ]
    transform = RollingOlsTransform(
        y="stock",
        x=("market", "credit"),
        window=3,
        coefficient="credit",
        partition_fields=("partition",),
        to="credit_beta",
    )

    outputs = list(transform.apply(iter(records)))

    assert [record.credit_beta is None for record in outputs] == [
        True,
        True,
        False,
        True,
        True,
        False,
    ]


def test_rolling_ols_is_stable_for_large_offsets() -> None:
    rows = [
        (
            1e12 + market,
            -2e12 + credit,
            5e11 + 2.5 * market - 1.75 * credit,
        )
        for market, credit in [
            (0.0, 0.0),
            (0.25, 1.0),
            (1.0, -0.5),
            (2.5, 3.0),
            (4.0, 0.25),
            (7.0, -2.0),
        ]
    ]

    assert _coefficients(rows, window=5)[-1] == pytest.approx(
        -1.75,
        rel=1e-7,
        abs=1e-7,
    )


def test_rolling_ols_reports_floating_point_overflow() -> None:
    with pytest.raises(OverflowError, match="floating-point range"):
        _coefficients(
            [
                (1e308, 1.0, 1.0),
                (1e308, 2.0, 2.0),
                (-1e308, 4.0, 3.0),
            ],
            window=3,
        )


def test_rolling_ols_preserves_coefficients_with_a_rounded_predictor_mean() -> None:
    # stock = 1 + 3 * market + (credit - 1e16), with an exact coefficient of 1.
    rows = [
        (0.0, 1e16, 1.0),
        (1.0, 1e16, 4.0),
        (0.0, 1e16 + 2.0, 3.0),
    ]

    assert _coefficients(rows, window=3)[-1] == pytest.approx(1.0, rel=1e-12, abs=1e-12)


def test_rolling_ols_preserves_finite_fits_across_large_ranges() -> None:
    rows = [
        (-1e308, 0.0, -1.0),
        (1e308, 0.0, 1.0),
        (0.0, -1e308, -2.0),
        (0.0, 1e308, 2.0),
    ]

    assert _coefficients(rows, window=4)[-1] == pytest.approx(
        2e-308, rel=1e-12, abs=0.0
    )


def test_rolling_ols_preserves_finite_fits_across_large_response_ranges() -> None:
    rows = [
        (-1.0, 0.0, -1.7e308),
        (1.0, 0.0, 1.7e308),
        (0.0, -1.0, -1.7e308),
        (0.0, 1.0, 1.7e308),
    ]

    assert _coefficients(rows, window=4)[-1] == pytest.approx(1.7e308, rel=1e-12)


def test_rolling_ols_preserves_coefficients_at_large_response_levels() -> None:
    # stock = 1e16 + 2 * market + 2 * credit.
    rows = [(0.0, 0.0, 1e16), (1.0, 0.0, 1e16 + 2), (0.0, 1.0, 1e16 + 2)]

    assert _coefficients(rows, window=3)[-1] == pytest.approx(2.0, rel=1e-12)


def test_rolling_ols_rejects_rank_deficient_windows() -> None:
    with pytest.raises(ValueError, match="rank deficient"):
        _coefficients(
            [
                (1.0, 2.0, 3.0),
                (2.0, 4.0, 5.0),
                (3.0, 6.0, 7.0),
            ],
            window=3,
        )


@pytest.mark.parametrize("field", ["market", "credit", "stock"])
@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (True, TypeError, "numeric values"),
        (float("inf"), ValueError, "finite numeric values"),
    ],
)
def test_rolling_ols_rejects_invalid_values(
    field: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    record = _record(1.0, 2.0, 3.0, 0)
    setattr(record, field, value)
    transform = RollingOlsTransform(
        y="stock",
        x=("market", "credit"),
        window=3,
        coefficient="credit",
        partition_fields=("partition",),
        to="credit_beta",
    )

    with pytest.raises(error, match=message):
        list(transform.apply(iter([record])))


def test_rolling_ols_requires_every_configured_field() -> None:
    record = _record(1.0, 2.0, 3.0, 0)
    del record.credit
    transform = RollingOlsTransform(
        y="stock",
        x=("market", "credit"),
        window=3,
        coefficient="credit",
        partition_fields=("partition",),
        to="credit_beta",
    )

    with pytest.raises(KeyError, match="credit"):
        list(transform.apply(iter([record])))


def test_lag_expresses_the_availability_gap_after_rolling_ols() -> None:
    records = [
        _record(market, credit, 1.0 + 2.0 * market - 3.0 * credit, position)
        for position, (market, credit) in enumerate(
            [(1.0, 2.0), (2.0, -1.0), (4.0, 3.0), (7.0, 0.5), (9.0, 4.0)]
        )
    ]
    ols = RollingOlsTransform(
        y="stock",
        x=("market", "credit"),
        window=3,
        coefficient="credit",
        partition_fields=("partition",),
        to="credit_beta_raw",
    ).apply(iter(records))
    lagged = LagTransform(
        field="credit_beta_raw",
        periods=2,
        partition_fields=("partition",),
        to="credit_beta",
    ).apply(ols)

    outputs = list(lagged)

    assert [record.credit_beta for record in outputs[:4]] == [None] * 4
    assert outputs[4].credit_beta == pytest.approx(-3.0)


def test_rolling_ols_reports_its_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def import_without_numpy(name: str, *args: object, **kwargs: object) -> object:
        if name == "numpy":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_numpy)

    with pytest.raises(
        RuntimeError,
        match=r"install jerry-thomas\[numerical\]",
    ):
        list(
            RollingOlsTransform(
                y="stock",
                x=("market", "credit"),
                window=3,
                coefficient="credit",
                partition_fields=("partition",),
                to="credit_beta",
            ).apply(iter(()))
        )
