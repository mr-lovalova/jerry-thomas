from datetime import UTC, datetime
from fractions import Fraction
from math import sqrt

import pytest
from pydantic import ValidationError

from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.dataset.series import SeriesConfig
from jerrythomas.config.dataset.split import DatasetFold, HashSplitConfig
from jerrythomas.config.streams import SourceStreamConfig, StreamsConfig
from jerrythomas.config.transforms import EwmStdConfig
from jerrythomas.services.dataset import validate_dataset_streams
from jerrythomas.transforms.stream.ewm_std import EwmStdTransform
from tests.unit.transforms.helpers import make_time_record


def naive_ewm_std(
    values: list[float | None],
    alpha: float,
    min_samples: int,
) -> list[float | None]:
    """Recompute the full weighted history at every step; O(n^2) on purpose."""
    beta = 1.0 - alpha
    history: list[tuple[float, float]] = []
    output: list[float | None] = []
    for value in values:
        if value is not None:
            history = [(beta * weight, sample) for weight, sample in history]
            history.append((1.0, value))
        total_weight = sum(weight for weight, _ in history)
        if len(history) >= min_samples and len(history) > 1:
            mean = sum(weight * sample for weight, sample in history) / total_weight
            variance = (
                sum(weight * (sample - mean) ** 2 for weight, sample in history)
                / total_weight
            )
            output.append(sqrt(variance))
        else:
            output.append(None)
    return output


def _actual(values: list[float | None], alpha: float, min_samples: int = 2) -> list:
    transform = EwmStdTransform(
        field="value",
        to="rolled",
        alpha=alpha,
        partition_fields=(),
        min_samples=min_samples,
    )
    records = (make_time_record(value, 0) for value in values)
    return [record.rolled for record in transform.apply(records)]


def _assert_matches_reference(
    values: list[float | None],
    alpha: float,
    min_samples: int = 2,
    tolerance: float = 1e-9,
) -> None:
    actual = _actual(values, alpha, min_samples)
    expected = naive_ewm_std(values, alpha, min_samples)
    for got, want in zip(actual, expected, strict=True):
        if want is None:
            assert got is None
        else:
            assert got == pytest.approx(want, rel=tolerance, abs=tolerance)


@pytest.mark.parametrize("alpha", [0.05, 0.3, 0.9])
@pytest.mark.parametrize("seed", [7, 42])
def test_ewm_std_matches_brute_force_reference(alpha: float, seed: int) -> None:
    from random import Random

    rng = Random(seed)
    values: list[float | None] = []
    level = 10.0
    for index in range(120):
        level += rng.uniform(-1.5, 1.5)
        values.append(level + rng.gauss(0.0, 2.0))
        if index % 17 == 5:
            values[-1] = None

    _assert_matches_reference(values, alpha)


def test_ewm_std_of_constant_series_is_zero_after_two_samples() -> None:
    assert _actual([5.0, 5.0, 5.0, 5.0], alpha=0.3) == [None, 0.0, 0.0, 0.0]


def test_ewm_std_survives_large_level_offsets() -> None:
    # sigma = 0.001 around a level of 1e9: the classic cancellation trap.
    values = [1e9 + (0.001 if index % 2 else -0.001) for index in range(40)]
    actual = _actual(values, alpha=0.2)

    reference = naive_ewm_std(values, 0.2, 2)
    assert actual[-1] == pytest.approx(reference[-1], rel=1e-6)
    assert 0.0005 < actual[-1] < 0.002


def test_ewm_std_matches_reference_across_large_level_changes() -> None:
    from random import Random

    rng = Random(11)
    values = [rng.uniform(-1e6, 1e6) + rng.gauss(0.0, 1.0) for _ in range(60)]

    _assert_matches_reference(values, alpha=0.15)


@pytest.mark.parametrize("sample_count", [100, 2100])
def test_ewm_std_preserves_spread_after_a_level_change(sample_count: int) -> None:
    values = [0.0] + [1e8 + index % 2 for index in range(sample_count)]
    weights = [Fraction(1, 2) ** age for age in reversed(range(len(values)))]
    weight_sum = sum(weights)
    mean = (
        sum(weight * Fraction(value) for weight, value in zip(weights, values))
        / weight_sum
    )
    variance = (
        sum(
            weight * (Fraction(value) - mean) ** 2
            for weight, value in zip(weights, values)
        )
        / weight_sum
    )

    assert _actual(values, alpha=0.5)[-1] == pytest.approx(sqrt(variance), rel=1e-12)


def test_ewm_std_freezes_state_across_missing_values() -> None:
    values: list[float | None] = [1.0, 3.0, None, None, 5.0]
    _assert_matches_reference(values, alpha=0.5)

    # The gap must not decay the stored history.
    frozen = naive_ewm_std([1.0, 3.0], alpha=0.5, min_samples=2)[-1]
    assert _actual([1.0, 3.0, None], alpha=0.5)[-1] == pytest.approx(frozen)


def test_ewm_std_alpha_one_yields_none_because_one_sample_has_no_spread() -> None:
    # alpha=1 keeps only the newest observation; dispersion is undefined there,
    # mirroring ewm_mean's degenerate shortcut rather than inventing semantics.
    assert _actual([1.0, 3.0, 8.0], alpha=1.0) == [None, None, None]


def test_ewm_std_resets_state_per_partition() -> None:
    transform = EwmStdTransform(
        field="value",
        to="rolled",
        alpha=0.5,
        partition_fields=("partition",),
        min_samples=2,
    )

    def record(partition: str, day: int, value: float):
        item = make_time_record(value, 0)
        item.partition = partition
        item.time = datetime(2025, 1, day, tzinfo=UTC)
        return item

    rows = list(
        transform.apply(
            iter(
                [
                    record("A", 1, 1.0),
                    record("A", 2, 3.0),
                    record("B", 1, 100.0),
                    record("B", 2, 100.0),
                ]
            )
        )
    )

    # A: values 1,3 at alpha=.5 -> weighted var = 8/9 -> std = sqrt(8)/3
    assert [row.rolled for row in rows][:2] == [None, pytest.approx((8 / 9) ** 0.5)]
    assert [row.rolled for row in rows][2:] == [None, 0.0]


def test_ewm_std_min_samples_gates_output() -> None:
    transform = EwmStdTransform(
        field="value",
        to="rolled",
        alpha=0.5,
        partition_fields=(),
        min_samples=4,
    )
    records = (make_time_record(float(index), 0) for index in range(6))

    outputs = [record.rolled for record in transform.apply(records)]

    assert outputs[:3] == [None, None, None]
    assert all(value is not None for value in outputs[3:])


def test_ewm_std_requires_two_observations_when_min_samples_is_one() -> None:
    assert _actual([None, 1.0, None, 3.0], alpha=0.5, min_samples=1) == [
        None,
        None,
        None,
        pytest.approx(sqrt(8 / 9)),
    ]


def test_ewm_std_config_rejects_invalid_alpha() -> None:
    with pytest.raises(ValidationError, match="alpha"):
        EwmStdConfig(field="value", alpha=0.0)
    with pytest.raises(ValidationError, match="alpha"):
        EwmStdConfig(field="value", alpha=1.5)


def test_hash_splits_reject_ewm_std() -> None:
    stream = SourceStreamConfig.model_validate(
        {
            "id": "prices",
            "from": {"source": "raw"},
            "map": {"entrypoint": "core.identity"},
            "partition_by": ["ticker"],
            "transforms": [{"operation": "ewm_std", "field": "close", "alpha": 0.3}],
        }
    )
    dataset = DatasetConfig(
        sample=SampleConfig(rounding="ceil", cadence="1d", keys=[]),
        features=[SeriesConfig(stream="prices", id="close", field="close")],
        split=HashSplitConfig(
            ratios={"train": 1.0},
            folds=[DatasetFold(id="default", train=["train"])],
        ),
    )

    with pytest.raises(
        ValueError,
        match="hash splits cannot be used with cross-timestamp streams: prices",
    ):
        validate_dataset_streams(dataset, StreamsConfig(streams={"prices": stream}))
