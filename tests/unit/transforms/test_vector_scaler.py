from collections.abc import Iterator

import pytest

from datapipeline.artifacts.scaler import (
    PositionalScalerStatistics,
    ScalerStatistics,
    StandardScalerArtifact,
)
from datapipeline.domain.sample import Sample
from datapipeline.domain.vector import Vector
from datapipeline.transforms.vector.scaler import SampleScaler


def _artifact(
    with_mean: bool = True,
    with_std: bool = True,
) -> StandardScalerArtifact:
    return StandardScalerArtifact(
        with_mean=with_mean,
        with_std=with_std,
        epsilon=1e-12,
        observations=5,
        statistics={
            "price": ScalerStatistics(mean=10.0, std=2.0, count=2),
            "price__@ticker:AAPL": ScalerStatistics(
                mean=20.0,
                std=4.0,
                count=2,
            ),
            "return": ScalerStatistics(mean=1.0, std=0.5, count=1),
        },
    )


def _positional_artifact() -> StandardScalerArtifact:
    return StandardScalerArtifact(
        with_mean=True,
        with_std=True,
        epsilon=1e-12,
        observations=4,
        statistics={
            "embedding": PositionalScalerStatistics(
                positions=(
                    ScalerStatistics(mean=2.0, std=1.0, count=2),
                    ScalerStatistics(mean=150.0, std=50.0, count=2),
                )
            )
        },
    )


def _sample(
    features: dict[str, object],
    targets: dict[str, object] | None = None,
) -> Sample:
    return Sample(
        key=("sample",),
        features=Vector(features),
        targets=None if targets is None else Vector(targets),
    )


def test_sample_scaler_scales_feature_and_target_vectors() -> None:
    sample = _sample(
        {
            "price": 12.0,
            "price__@ticker:AAPL": [16.0, None, 24.0],
            "volume": "unchanged",
        },
        {"return": 1.5},
    )
    scaler = SampleScaler(
        _artifact(),
        scaled_feature_ids={"price"},
        scaled_target_ids={"return"},
    )

    scaled = scaler.scale(sample)

    assert scaled.features.values == {
        "price": 1.0,
        "price__@ticker:AAPL": [-1.0, None, 1.0],
        "volume": "unchanged",
    }
    assert scaled.targets is not None
    assert scaled.targets.values == {"return": 1.0}
    assert sample.features.values["price"] == 12.0
    assert sample.targets is not None
    assert sample.targets.values["return"] == 1.5


def test_sample_scaler_respects_disabled_options() -> None:
    sample = _sample({"price": [10.0, 12.0]})
    scaler = SampleScaler(
        _artifact(with_mean=False, with_std=False),
        scaled_feature_ids={"price"},
        scaled_target_ids=(),
    )

    assert scaler.scale(sample).features.values["price"] == [10.0, 12.0]


def test_sample_scaler_scales_intrinsic_lists_by_position() -> None:
    scaler = SampleScaler(
        _positional_artifact(),
        scaled_feature_ids={"embedding"},
        scaled_target_ids=(),
    )

    scaled = scaler.scale(_sample({"embedding": [1.0, 200.0]}))

    assert scaled.features.values["embedding"] == [-1.0, 1.0]


def test_sample_scaler_preserves_missing_intrinsic_list_positions() -> None:
    scaler = SampleScaler(
        _positional_artifact(),
        scaled_feature_ids={"embedding"},
        scaled_target_ids=(),
    )

    scaled = scaler.scale(_sample({"embedding": [None, 200.0]}))

    assert scaled.features.values["embedding"] == [None, 1.0]


def test_sample_scaler_requires_positional_list_width() -> None:
    scaler = SampleScaler(
        _positional_artifact(),
        scaled_feature_ids={"embedding"},
        scaled_target_ids=(),
    )

    with pytest.raises(
        ValueError,
        match=r"'embedding' requires 2 values.*got 1",
    ):
        scaler.scale(_sample({"embedding": [1.0]}))


def test_sample_scaler_rejects_scalar_for_positional_statistics() -> None:
    scaler = SampleScaler(
        _positional_artifact(),
        scaled_feature_ids={"embedding"},
        scaled_target_ids=(),
    )

    with pytest.raises(TypeError, match=r"'embedding' requires a list value"):
        scaler.scale(_sample({"embedding": 1.0}))


@pytest.mark.parametrize(
    ("with_mean", "with_std", "expected"),
    [
        (True, False, 2.0),
        (False, True, 6.0),
    ],
)
def test_sample_scaler_applies_each_scaling_option_independently(
    with_mean: bool,
    with_std: bool,
    expected: float,
) -> None:
    sample = _sample({"price": 12.0})
    scaler = SampleScaler(
        _artifact(with_mean=with_mean, with_std=with_std),
        scaled_feature_ids={"price"},
        scaled_target_ids=(),
    )

    assert scaler.scale(sample).features.values["price"] == expected


def test_sample_scaler_preserves_scalar_none_without_statistics() -> None:
    sample = _sample({"missing": None})
    scaler = SampleScaler(
        _artifact(),
        scaled_feature_ids={"missing"},
        scaled_target_ids=(),
    )

    assert scaler.scale(sample).features.values == sample.features.values


@pytest.mark.parametrize("value", [21.0, [None, None]])
def test_sample_scaler_requires_exact_partition_statistics(value: object) -> None:
    sample = _sample({"price__@ticker:MSFT": value})
    scaler = SampleScaler(
        _artifact(),
        scaled_feature_ids={"price"},
        scaled_target_ids=(),
    )

    with pytest.raises(
        KeyError,
        match="Missing scaler statistics for vector 'price__@ticker:MSFT'",
    ):
        scaler.scale(sample)


@pytest.mark.parametrize("value", [True, "1.0", [1.0, "2.0"]])
def test_sample_scaler_rejects_non_numeric_scaled_values(value: object) -> None:
    sample = _sample({"price": value})
    scaler = SampleScaler(
        _artifact(),
        scaled_feature_ids={"price"},
        scaled_target_ids=(),
    )

    with pytest.raises(TypeError, match="numeric or None"):
        scaler.scale(sample)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_sample_scaler_rejects_non_finite_scaled_values(value: float) -> None:
    sample = _sample({"price": value})
    scaler = SampleScaler(
        _artifact(),
        scaled_feature_ids={"price"},
        scaled_target_ids=(),
    )

    with pytest.raises(ValueError, match="must be finite"):
        scaler.scale(sample)


def test_sample_scaler_applies_to_an_iterator() -> None:
    samples: Iterator[Sample] = iter(
        [_sample({"price": 10.0}), _sample({"price": 12.0})]
    )
    scaler = SampleScaler(
        _artifact(),
        scaled_feature_ids={"price"},
        scaled_target_ids=(),
    )

    scaled = list(scaler.apply(samples))

    assert [sample.features.values["price"] for sample in scaled] == [0.0, 1.0]
