from math import nan

import pytest
from pydantic import ValidationError

from jerrythomas.config.transforms import EwmMeanConfig
from jerrythomas.transforms.stream.ewm import EwmMeanTransform
from tests.unit.transforms.helpers import make_time_record


def _ewm_values(
    values: list[float | None],
    alpha: float,
    min_samples: int = 1,
) -> list[float | None]:
    transform = EwmMeanTransform(
        field="value",
        alpha=alpha,
        partition_fields=(),
        min_samples=min_samples,
    )
    records = (make_time_record(value, 0) for value in values)
    return [record.value for record in transform.apply(records)]


def test_ewm_mean_uses_recursive_unadjusted_weights() -> None:
    assert _ewm_values([10.0, 20.0, 30.0], alpha=0.5) == [10.0, 15.0, 22.5]


def test_ewm_mean_respects_minimum_samples_and_ignores_missing_values() -> None:
    assert _ewm_values(
        [10.0, 14.0, None, 22.0],
        alpha=0.25,
        min_samples=2,
    ) == [None, 11.0, 11.0, 13.75]


def test_ewm_mean_treats_nan_as_missing() -> None:
    assert _ewm_values([10.0, nan, 20.0], alpha=0.5) == [10.0, 10.0, 15.0]


def test_ewm_mean_alpha_one_is_identity() -> None:
    assert _ewm_values([1.0, -2.0, 3.0], alpha=1.0) == [1.0, -2.0, 3.0]


def test_ewm_mean_handles_opposite_extremes_without_overflow() -> None:
    assert _ewm_values([-1e308, 1e308], alpha=0.5) == [-1e308, 0.0]


def test_ewm_mean_resets_state_between_partitions() -> None:
    records = [
        make_time_record(10.0, 0),
        make_time_record(20.0, 0),
        make_time_record(100.0, 0),
        make_time_record(200.0, 0),
    ]
    for record, partition in zip(records, ["A", "A", "B", "B"], strict=True):
        record.partition = partition
    transform = EwmMeanTransform(
        field="value",
        alpha=0.5,
        partition_fields=("partition",),
    )

    assert [record.value for record in transform.apply(iter(records))] == [
        10.0,
        15.0,
        100.0,
        150.0,
    ]


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1, nan, float("inf")])
def test_ewm_mean_algorithm_requires_valid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be greater than 0 and at most 1"):
        EwmMeanTransform("value", alpha, ())


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1, nan, float("inf"), True, "0.5"])
def test_ewm_mean_config_requires_finite_alpha(alpha: object) -> None:
    with pytest.raises(ValidationError, match="alpha"):
        EwmMeanConfig(field="value", alpha=alpha)


def test_ewm_mean_config_round_trips() -> None:
    config = EwmMeanConfig(
        field="value",
        alpha=0.2,
        min_samples=5,
        to="smoothed",
    )

    assert EwmMeanConfig.model_validate(config.model_dump()) == config


@pytest.mark.parametrize("value", ["1.0", float("inf")])
def test_ewm_mean_rejects_invalid_values(value: object) -> None:
    transform = EwmMeanTransform("value", 0.5, ())

    with pytest.raises((TypeError, ValueError), match="numeric|finite"):
        list(transform.apply(iter([make_time_record(value, 0)])))
