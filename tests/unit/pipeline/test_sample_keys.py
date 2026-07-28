from datetime import datetime, timedelta, timezone

import pytest

from datapipeline.pipelines.sample.keys import (
    RectangularKeyPlan,
    merge_rectangular_key_plans,
)


def _ts(hour: int, minute: int = 0) -> datetime:
    return datetime(2024, 1, 1, hour, minute, tzinfo=timezone.utc)


def test_rectangular_key_plan_contains_repeated_and_disjoint_windows() -> None:
    plan = RectangularKeyPlan(
        start=_ts(0),
        end=_ts(7),
        step=timedelta(hours=1),
        windows=(
            (("AAPL",), _ts(0), _ts(2)),
            (("AAPL",), _ts(1), _ts(2)),
            (("AAPL",), _ts(6), _ts(7)),
            (("MSFT",), _ts(3), _ts(4)),
        ),
    )

    assert plan.contains((_ts(1), "AAPL"))
    assert plan.contains((_ts(6), "AAPL"))
    assert plan.contains((_ts(3), "MSFT"))
    assert not plan.contains((_ts(3), "AAPL"))
    assert not plan.contains((_ts(1), "MSFT"))
    assert not plan.contains((_ts(1, 30), "AAPL"))
    assert not plan.contains((_ts(1),))


def test_merge_rectangular_key_plans_coalesces_and_sorts_union() -> None:
    first = RectangularKeyPlan(
        start=_ts(0),
        end=_ts(7),
        step=timedelta(hours=1),
        windows=(
            (("AAPL",), _ts(0), _ts(2)),
            (("AAPL",), _ts(6), _ts(7)),
            (("MSFT",), _ts(1), _ts(2)),
        ),
    )
    second = RectangularKeyPlan(
        start=_ts(1),
        end=_ts(6),
        step=timedelta(hours=1),
        windows=(
            (("AAPL",), _ts(1), _ts(3)),
            (("AAPL",), _ts(2), _ts(3)),
            (("AAPL",), _ts(6), _ts(6)),
            (("MSFT",), _ts(2), _ts(4)),
        ),
    )

    merged = merge_rectangular_key_plans((first, second))

    assert merged.start == _ts(0)
    assert merged.end == _ts(7)
    assert merged.windows == (
        (("AAPL",), _ts(0), _ts(3)),
        (("AAPL",), _ts(6), _ts(7)),
        (("MSFT",), _ts(1), _ts(4)),
    )
    assert tuple(merged.keys()) == (
        (_ts(0), "AAPL"),
        (_ts(1), "AAPL"),
        (_ts(1), "MSFT"),
        (_ts(2), "AAPL"),
        (_ts(2), "MSFT"),
        (_ts(3), "AAPL"),
        (_ts(3), "MSFT"),
        (_ts(4), "MSFT"),
        (_ts(6), "AAPL"),
        (_ts(7), "AAPL"),
    )
    assert merged.total == 10


def test_merge_rectangular_key_plans_clips_windows_to_each_plan() -> None:
    first = RectangularKeyPlan(
        start=_ts(1),
        end=_ts(2),
        step=timedelta(hours=1),
        windows=((("AAPL",), _ts(0), _ts(3)),),
    )
    second = RectangularKeyPlan(
        start=_ts(4),
        end=_ts(5),
        step=timedelta(hours=1),
        windows=((("AAPL",), _ts(3), _ts(6)),),
    )

    merged = merge_rectangular_key_plans((first, second))

    assert tuple(merged.keys()) == (
        (_ts(1), "AAPL"),
        (_ts(2), "AAPL"),
        (_ts(4), "AAPL"),
        (_ts(5), "AAPL"),
    )


def test_merge_rectangular_key_plans_rejects_different_cadences() -> None:
    hourly = RectangularKeyPlan(
        start=_ts(0),
        end=_ts(2),
        step=timedelta(hours=1),
        windows=(((), _ts(0), _ts(2)),),
    )
    half_hourly = RectangularKeyPlan(
        start=_ts(0),
        end=_ts(2),
        step=timedelta(minutes=30),
        windows=(((), _ts(0), _ts(2)),),
    )

    with pytest.raises(ValueError, match="different cadence"):
        merge_rectangular_key_plans((hourly, half_hourly))


def test_merge_rectangular_key_plans_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty sequence"):
        merge_rectangular_key_plans(())


def test_merge_rectangular_key_plans_rejects_different_lattice_origins() -> None:
    on_the_hour = RectangularKeyPlan(
        start=_ts(0),
        end=_ts(2),
        step=timedelta(hours=1),
        windows=(((), _ts(0), _ts(2)),),
    )
    on_the_half_hour = RectangularKeyPlan(
        start=_ts(0, 30),
        end=_ts(2, 30),
        step=timedelta(hours=1),
        windows=(((), _ts(0, 30), _ts(2, 30)),),
    )

    with pytest.raises(ValueError, match="incompatible time lattices"):
        merge_rectangular_key_plans((on_the_hour, on_the_half_hour))
