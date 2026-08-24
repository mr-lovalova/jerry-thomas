from collections import deque
from collections.abc import Sequence
from random import Random
from statistics import mean, median, pstdev, stdev
from typing import Any, Callable

import pytest
from pydantic import ValidationError

from jerrythomas.config.transforms import RollingConfig, RollingQuantileConfig
from jerrythomas.transforms.rolling_window import (
    RollingMean,
    RollingQuantile,
    RollingSampleStandardDeviation,
)
from jerrythomas.transforms.stream.rolling import (
    RollingQuantileTransform,
    RollingTransform,
)
from tests.unit.transforms.helpers import make_time_record


def _rolling_values(
    values: Sequence[float | None],
    statistic: str,
    window: int,
    min_samples: int | None = None,
) -> list[float | None]:
    transform = RollingTransform(
        field="value",
        to="rolled",
        window=window,
        partition_fields=(),
        statistic=statistic,
        min_samples=min_samples,
    )
    records = (make_time_record(value, 0) for value in values)
    return [record.rolled for record in transform.apply(records)]


def _rolling_quantile_values(
    values: Sequence[float | None],
    quantile: float,
    window: int,
    min_samples: int | None = None,
) -> list[float | None]:
    transform = RollingQuantileTransform(
        field="value",
        to="rolled",
        window=window,
        quantile=quantile,
        partition_fields=(),
        min_samples=min_samples,
    )
    records = (make_time_record(value, 0) for value in values)
    return [record.rolled for record in transform.apply(records)]


def test_rolling_pstdev_matches_statistics_pstdev() -> None:
    values = [0.01, 0.02, -0.01, 0.03]
    actual = _rolling_values(
        values,
        statistic="pstdev",
        window=3,
        min_samples=3,
    )

    assert actual[:2] == [None, None]
    assert actual[2:] == pytest.approx([pstdev(values[:3]), pstdev(values[1:])])


@pytest.mark.parametrize("window", [-1, 0])
def test_rolling_window_requires_positive_size(window: int) -> None:
    with pytest.raises(ValueError, match="window must be positive"):
        RollingMean(window)


def test_rolling_pstdev_of_one_sample_is_zero() -> None:
    assert _rolling_values(
        [1e12],
        statistic="pstdev",
        window=3,
        min_samples=1,
    ) == [0.0]


def test_rolling_quantile_uses_linear_interpolation() -> None:
    assert _rolling_quantile_values(
        [0.0, 10.0, 20.0, 30.0],
        quantile=0.25,
        window=4,
        min_samples=1,
    ) == [0.0, 2.5, 5.0, 7.5]


@pytest.mark.parametrize(
    ("quantile", "expected"),
    [
        (0.0, [None, None, 1.0, 3.0]),
        (1.0, [None, None, 5.0, 5.0]),
    ],
)
def test_rolling_quantile_endpoints_follow_expiring_window(
    quantile: float,
    expected: list[float | None],
) -> None:
    assert (
        _rolling_quantile_values(
            [1.0, 5.0, 3.0, 4.0],
            quantile=quantile,
            window=3,
        )
        == expected
    )


def test_rolling_quantile_expires_one_duplicate_at_a_time() -> None:
    assert _rolling_quantile_values(
        [5.0, 5.0, 1.0, 1.0],
        quantile=0.5,
        window=3,
    ) == [None, None, 5.0, 1.0]


def test_rolling_quantile_excludes_missing_values_from_interpolation() -> None:
    assert _rolling_quantile_values(
        [0.0, None, 20.0, 40.0],
        quantile=0.25,
        window=3,
        min_samples=2,
    ) == [None, None, 5.0, 25.0]


def test_rolling_quantile_interpolates_opposite_extremes_without_overflow() -> None:
    assert _rolling_quantile_values(
        [-1e308, 1e308],
        quantile=0.5,
        window=2,
    ) == [None, 0.0]


def test_rolling_quantile_config_round_trips() -> None:
    config = RollingQuantileConfig(
        field="value",
        window=20,
        min_samples=10,
        quantile=0.9,
    )

    assert RollingQuantileConfig.model_validate(config.model_dump()) == config


def test_rolling_quantile_requires_probability() -> None:
    with pytest.raises(ValidationError, match="quantile"):
        RollingQuantileConfig(field="value", window=3)


def test_rolling_quantile_rejects_impossible_minimum_sample_count() -> None:
    with pytest.raises(ValidationError, match="min_samples cannot exceed window"):
        RollingQuantileConfig(
            field="value",
            window=3,
            min_samples=4,
            quantile=0.5,
        )


def test_regular_rolling_rejects_quantile_parameter() -> None:
    with pytest.raises(ValidationError, match="quantile"):
        RollingConfig(field="value", window=3, quantile=0.5)


@pytest.mark.parametrize(
    "quantile",
    [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"],
)
def test_rolling_quantile_requires_finite_probability(quantile: object) -> None:
    with pytest.raises(ValidationError, match="quantile"):
        RollingQuantileConfig(
            field="value",
            window=3,
            quantile=quantile,
        )


@pytest.mark.parametrize("quantile", [-0.1, 1.1, float("nan")])
def test_rolling_quantile_algorithm_rejects_invalid_probability(
    quantile: float,
) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        RollingQuantile(window=3, quantile=quantile)


def test_rolling_quantile_requires_a_sample() -> None:
    with pytest.raises(ValueError, match="requires at least one sample"):
        RollingQuantile(window=3, quantile=0.5).result()


def test_rolling_median_preserves_subnormal_midpoint() -> None:
    assert _rolling_values(
        [-5e-324, 1.5e-323],
        statistic="median",
        window=2,
    ) == [None, 5e-324]


def test_rolling_stdev_matches_statistics_stdev() -> None:
    values = [1.0, 2.0, 4.0]
    actual = _rolling_values(
        values,
        statistic="stdev",
        window=3,
        min_samples=3,
    )

    assert actual[:2] == [None, None]
    assert actual[2] == pytest.approx(stdev(values))


@pytest.mark.parametrize("values", [[], [1.0]])
def test_rolling_sample_standard_deviation_requires_two_samples(
    values: list[float],
) -> None:
    deviation = RollingSampleStandardDeviation(window=2)
    for value in values:
        deviation.append(value)

    with pytest.raises(ValueError, match="requires at least two samples"):
        deviation.result()


def test_rolling_stdev_respects_missing_values_and_min_samples() -> None:
    actual = _rolling_values(
        [1.0, None, 3.0],
        statistic="stdev",
        window=3,
        min_samples=2,
    )

    assert actual[:2] == [None, None]
    assert actual[2] == pytest.approx(stdev([1.0, 3.0]))


@pytest.mark.parametrize(
    ("statistic", "expected"),
    [("max", [None, None, 12.0, 12.0]), ("min", [None, None, 8.0, 8.0])],
)
def test_rolling_extrema_match_window(
    statistic: str,
    expected: list[float | None],
) -> None:
    assert (
        _rolling_values(
            [10.0, 8.0, 12.0, 9.0],
            statistic=statistic,
            window=3,
            min_samples=3,
        )
        == expected
    )


def test_rolling_rejects_unknown_statistic_with_supported_names() -> None:
    with pytest.raises(ValidationError, match="statistic"):
        RollingConfig(
            field="value",
            window=3,
            statistic="variance",
        )


def test_rolling_stdev_requires_two_samples() -> None:
    with pytest.raises(ValidationError, match="min_samples must be at least 2"):
        RollingConfig(
            field="value",
            window=3,
            statistic="stdev",
            min_samples=1,
        )


def test_rolling_rejects_impossible_minimum_sample_count() -> None:
    with pytest.raises(ValidationError, match="min_samples cannot exceed window"):
        RollingConfig(
            field="value",
            window=3,
            min_samples=4,
        )


@pytest.mark.parametrize("window", [True, 3.5])
def test_rolling_requires_an_integer_window(window: Any) -> None:
    with pytest.raises(ValidationError, match="window"):
        RollingConfig(field="value", window=window)


@pytest.mark.parametrize("min_samples", [True, 1.5])
def test_rolling_requires_an_integer_minimum_sample_count(min_samples: Any) -> None:
    with pytest.raises(ValidationError, match="min_samples"):
        RollingConfig(
            field="value",
            window=3,
            min_samples=min_samples,
        )


def test_rolling_missing_ticks_expire_valid_values() -> None:
    assert _rolling_values(
        [1.0, 2.0, None, None, 3.0],
        statistic="mean",
        window=3,
        min_samples=2,
    ) == [None, 1.5, 1.5, None, None]


def test_rolling_default_minimum_requires_a_full_valid_window() -> None:
    assert _rolling_values(
        [1.0, None, 3.0, 4.0],
        statistic="mean",
        window=3,
    ) == [None, None, None, None]


def test_rolling_resets_state_between_partitions() -> None:
    records = [
        make_time_record(1.0, 0),
        make_time_record(3.0, 0),
        make_time_record(10.0, 0),
        make_time_record(30.0, 0),
    ]
    for record, partition in zip(records, ["A", "A", "B", "B"]):
        setattr(record, "partition", partition)
    transform = RollingTransform(
        field="value",
        window=2,
        min_samples=2,
        partition_fields=("partition",),
    )

    assert [record.value for record in transform.apply(iter(records))] == [
        None,
        2.0,
        None,
        20.0,
    ]


@pytest.mark.parametrize(
    ("statistic", "expected"),
    [
        ("max", [None, None, 5.0, 5.0, 1.0]),
        ("min", [None, None, 1.0, 1.0, 1.0]),
    ],
)
def test_rolling_extrema_preserve_duplicates(
    statistic: str,
    expected: list[float | None],
) -> None:
    assert (
        _rolling_values(
            [5.0, 5.0, 1.0, 1.0, 1.0],
            statistic=statistic,
            window=3,
            min_samples=3,
        )
        == expected
    )


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_rolling_rejects_infinite_values(value: float) -> None:
    transform = RollingTransform(field="value", window=1, partition_fields=())

    with pytest.raises(ValueError, match="finite numeric values"):
        list(transform.apply(iter([make_time_record(value, 0)])))


def test_rolling_rejects_non_numeric_values() -> None:
    transform = RollingTransform(
        field="value",
        window=1,
        partition_fields=(),
    )

    with pytest.raises(TypeError, match="numeric values"):
        list(transform.apply(iter([make_time_record("1.0", 0)])))


def test_rolling_requires_configured_field() -> None:
    transform = RollingTransform(
        field="missing",
        window=1,
        partition_fields=(),
    )

    with pytest.raises(KeyError, match="missing"):
        list(transform.apply(iter([make_time_record(1.0, 0)])))


@pytest.mark.parametrize(
    ("statistic", "values"),
    [
        ("mean", [1e308, 1e308]),
        ("stdev", [1e200, -1e200]),
        ("pstdev", [1e200, -1e200]),
    ],
)
def test_rolling_reports_floating_point_overflow(
    statistic: str,
    values: list[float],
) -> None:
    with pytest.raises(OverflowError, match="supported floating-point range"):
        _rolling_values(values, statistic, window=2, min_samples=2)


def test_rolling_sum_matches_naive_sum() -> None:
    values = [1.5, 2.0, -0.5, 4.0]
    actual = _rolling_values(
        values,
        statistic="sum",
        window=3,
        min_samples=3,
    )

    assert actual[:2] == [None, None]
    assert actual[2] == pytest.approx(3.0)
    assert actual[3] == pytest.approx(5.5)


def test_rolling_sum_over_partial_windows_sums_available_samples() -> None:
    assert _rolling_values(
        [1.0, 2.0, 4.0],
        statistic="sum",
        window=3,
        min_samples=1,
    ) == [1.0, 3.0, 7.0]


def test_rolling_sum_excludes_missing_values() -> None:
    # Missing values occupy window slots and do not count toward min_samples.
    assert _rolling_values(
        [1.0, None, 2.0],
        statistic="sum",
        window=2,
        min_samples=1,
    ) == [1.0, 1.0, 2.0]

    assert _rolling_values(
        [1.0, None, 2.0],
        statistic="sum",
        window=2,
        min_samples=2,
    ) == [None, None, None]


def test_rolling_sum_survives_cancellation() -> None:
    values = [1e308, -1e308, 1e308]
    actual = _rolling_values(
        values,
        statistic="sum",
        window=3,
        min_samples=3,
    )

    assert actual[:2] == [None, None]
    assert actual[2] == pytest.approx(1e308)


def test_rolling_mean_preserves_cancellation() -> None:
    assert _rolling_values(
        [0.0, -1_000_000_000_001.0, 1_000_000_000_000.0],
        statistic="mean",
        window=3,
        min_samples=3,
    )[-1] == pytest.approx(-1 / 3, rel=0.0, abs=1e-15)


def test_rolling_mean_handles_opposite_large_values() -> None:
    assert _rolling_values(
        [1e308, -1e308],
        statistic="mean",
        window=2,
        min_samples=2,
    ) == [None, 0.0]


def test_rolling_treats_nan_as_missing() -> None:
    values = _rolling_values(
        [1.0, float("nan"), 3.0],
        statistic="mean",
        window=3,
        min_samples=2,
    )

    assert values[:2] == [None, None]
    assert values[2] == 2.0


@pytest.mark.parametrize(
    ("statistic", "calculate"),
    [
        ("mean", mean),
        ("median", median),
        ("stdev", stdev),
        ("pstdev", pstdev),
        ("max", max),
        ("min", min),
    ],
)
def test_rolling_matches_reference_windows(
    statistic: str,
    calculate: Callable[[list[float]], float],
) -> None:
    random = Random(42)
    values = [
        None if random.random() < 0.2 else random.uniform(-100.0, 100.0)
        for _ in range(500)
    ]
    actual = _rolling_values(values, statistic, window=17, min_samples=5)
    ticks: deque[float | None] = deque(maxlen=17)

    for value, rolled in zip(values, actual):
        ticks.append(value)
        valid = [tick for tick in ticks if tick is not None]
        if len(valid) < 5:
            assert rolled is None
        else:
            assert rolled == pytest.approx(
                float(calculate(valid)),
                rel=1e-12,
                abs=1e-12,
            )


@pytest.mark.parametrize(
    ("statistic", "calculate"), [("stdev", stdev), ("pstdev", pstdev)]
)
def test_rolling_deviation_remains_stable_for_large_values(
    statistic: str,
    calculate: Callable[[list[float]], float],
) -> None:
    random = Random(7)
    values = [1e12 + random.uniform(-3.0, 3.0) for _ in range(10_000)]
    actual = _rolling_values(values, statistic, window=252, min_samples=252)

    assert actual[-1] == pytest.approx(
        float(calculate(values[-252:])),
        rel=1e-12,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("statistic", "calculate"), [("stdev", stdev), ("pstdev", pstdev)]
)
def test_rolling_deviation_rebases_when_an_outlier_expires(
    statistic: str,
    calculate: Callable[[list[float]], float],
) -> None:
    random = Random(11)
    values = [0.0] + [1e12 + random.uniform(-3.0, 3.0) for _ in range(300)]
    actual = _rolling_values(values, statistic, window=252, min_samples=252)

    for position in [252, 300]:
        assert actual[position] == pytest.approx(
            float(calculate(values[position - 251 : position + 1])),
            rel=1e-12,
            abs=1e-12,
        )


@pytest.mark.parametrize(
    ("statistic", "calculate"), [("stdev", stdev), ("pstdev", pstdev)]
)
def test_rolling_deviation_preserves_small_variance_after_outlier_expires(
    statistic: str,
    calculate: Callable[[list[float]], float],
) -> None:
    values = [
        0.0,
        0.0,
        -999_999_999_999.0,
        -999_999_999_997.0,
        -999_999_999_999.0,
    ]
    actual = _rolling_values(values, statistic, window=3, min_samples=3)

    assert actual[-1] == pytest.approx(
        float(calculate(values[-3:])),
        rel=1e-12,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("statistic", "calculate"), [("stdev", stdev), ("pstdev", pstdev)]
)
def test_rolling_deviation_preserves_sub_ulp_terms_across_rebase(
    statistic: str,
    calculate: Callable[[list[float]], float],
) -> None:
    values = [1_000_000_000_003.0, 1_000_000_000_000.0, -1.0, 0.0, 0.0, 0.0, 3.0]
    actual = _rolling_values(values, statistic, window=4, min_samples=4)

    for position in [5, 6]:
        assert actual[position] == pytest.approx(
            float(calculate(values[position - 3 : position + 1])),
            rel=1e-12,
            abs=1e-12,
        )
