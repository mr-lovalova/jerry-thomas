from datetime import datetime, timedelta, timezone
from random import Random
from statistics import linear_regression

import pytest

from jerrythomas.domain.record import TemporalRecord
from jerrythomas.transforms.stream.lag import LagTransform
from jerrythomas.transforms.stream.rolling_slope import RollingSlopeTransform


def _record(
    x: object,
    y: object,
    position: int,
    partition: str = "A",
) -> TemporalRecord:
    record = TemporalRecord(
        time=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=position),
    )
    record.x = x
    record.y = y
    record.partition = partition
    return record


def _slopes(
    points: list[tuple[object, object]],
    window: int,
) -> list[float | None]:
    records = [_record(x, y, position) for position, (x, y) in enumerate(points)]
    transform = RollingSlopeTransform(
        x="x",
        y="y",
        window=window,
        partition_fields=("partition",),
        to="slope",
    )
    return [record.slope for record in transform.apply(iter(records))]


def test_rolling_slope_computes_y_on_x_without_mutating_input() -> None:
    records = [_record(x, 2 * x + 3, x) for x in range(4)]
    transform = RollingSlopeTransform(
        x="x",
        y="y",
        window=3,
        partition_fields=("partition",),
        to="slope",
    )

    outputs = list(transform.apply(iter(records)))

    assert [record.slope for record in outputs] == [None, None, 2.0, 2.0]
    assert all(not hasattr(record, "slope") for record in records)


def test_rolling_slope_matches_reference_windows() -> None:
    random = Random(42)
    points = [
        (x := random.uniform(-100.0, 100.0), 1.75 * x + random.uniform(-2.0, 2.0))
        for _ in range(500)
    ]
    window = 17
    actual = _slopes(points, window)

    for position, slope in enumerate(actual):
        if position < window - 1:
            assert slope is None
            continue
        current = points[position - window + 1 : position + 1]
        expected = linear_regression(
            [point[0] for point in current],
            [point[1] for point in current],
        ).slope
        assert slope == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_rolling_slope_resets_after_missing_x_or_y() -> None:
    actual = _slopes(
        [
            (1.0, 2.0),
            (2.0, 4.0),
            (None, 6.0),
            (4.0, 8.0),
            (5.0, float("nan")),
            (6.0, 12.0),
            (7.0, 14.0),
            (8.0, 16.0),
        ],
        window=3,
    )

    assert actual[:-1] == [None] * 7
    assert actual[-1] == 2.0


def test_rolling_slope_resets_between_partitions() -> None:
    records = [
        _record(1.0, 2.0, 0, "A"),
        _record(2.0, 4.0, 1, "A"),
        _record(10.0, -30.0, 2, "B"),
        _record(20.0, -60.0, 3, "B"),
    ]
    transform = RollingSlopeTransform(
        x="x",
        y="y",
        window=2,
        partition_fields=("partition",),
        to="slope",
    )

    assert [record.slope for record in transform.apply(iter(records))] == [
        None,
        2.0,
        None,
        -3.0,
    ]


def test_rolling_slope_is_stable_for_large_offsets() -> None:
    points = [
        (1e12 + offset, -2e12 + 2.5 * offset)
        for offset in [0.0, 0.25, 1.0, 2.5, 4.0, 7.0]
    ]

    assert _slopes(points, window=4)[-1] == pytest.approx(
        2.5,
        rel=1e-12,
        abs=1e-12,
    )


def test_rolling_slope_rebases_when_an_outlier_expires() -> None:
    x_values = [0.0, 1e12, 1e12 + 1.0, 1e12 + 2.0, 1e12 + 3.0]
    points = [(x, 3.0 * x - 5e11) for x in x_values]
    actual = _slopes(points, window=4)

    assert actual[-2:] == pytest.approx([3.0, 3.0], rel=1e-12, abs=1e-12)


def test_rolling_slope_avoids_overflow_in_finite_mean_correction() -> None:
    points = [(0.0, 0.0), *[(5e153, 5e153)] * 3]

    assert _slopes(points, window=4)[-1] == 1.0


def test_rolling_slope_rejects_zero_x_variance() -> None:
    with pytest.raises(ZeroDivisionError, match="x has zero variance"):
        _slopes([(1.0, 1.0), (1.0, 2.0)], window=2)


def test_lagging_both_inputs_excludes_the_current_record() -> None:
    records = [_record(1.0, 2.0, 0), _record(2.0, 4.0, 1), _record(100.0, -999.0, 2)]
    lagged_x = LagTransform("x", 1, ("partition",), "lagged_x").apply(iter(records))
    lagged_xy = LagTransform("y", 1, ("partition",), "lagged_y").apply(lagged_x)
    transform = RollingSlopeTransform(
        x="lagged_x",
        y="lagged_y",
        window=2,
        partition_fields=("partition",),
        to="slope",
    )

    output = list(transform.apply(lagged_xy))

    assert [record.slope for record in output] == [None, None, 2.0]


@pytest.mark.parametrize("field", ["x", "y"])
@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (True, TypeError, "numeric values"),
        (float("inf"), ValueError, "finite numeric values"),
    ],
)
def test_rolling_slope_rejects_invalid_values(
    field: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    record = _record(1.0, 2.0, 0)
    setattr(record, field, value)
    transform = RollingSlopeTransform(
        x="x",
        y="y",
        window=2,
        partition_fields=("partition",),
        to="slope",
    )

    with pytest.raises(error, match=message):
        list(transform.apply(iter([record])))


def test_rolling_slope_requires_both_fields() -> None:
    record = _record(1.0, 2.0, 0)
    del record.y
    transform = RollingSlopeTransform(
        x="x",
        y="y",
        window=2,
        partition_fields=("partition",),
        to="slope",
    )

    with pytest.raises(KeyError, match="y"):
        list(transform.apply(iter([record])))


def test_rolling_slope_reports_floating_point_overflow() -> None:
    with pytest.raises(OverflowError, match="floating-point range"):
        _slopes([(-1e308, 1.0), (1e308, 2.0)], window=2)
