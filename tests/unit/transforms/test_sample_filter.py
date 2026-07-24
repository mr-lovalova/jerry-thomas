import pytest

from datapipeline.transforms.vector.sample_filter import (
    FilterFeatureSamplesTransform,
    FilterTargetSamplesTransform,
)
from tests.unit.transforms.helpers import make_vector


def test_feature_filter_uses_scalar_and_sequence_cell_coverage() -> None:
    samples = [
        make_vector(0, {"scalar": 1.0, "sequence": [1.0, None, None]}),
        make_vector(1, {"scalar": 1.0, "sequence": [1.0, 2.0, 3.0]}),
    ]
    transform = FilterFeatureSamplesTransform(
        ["scalar", "sequence"],
        threshold=0.7,
    )

    output = list(transform.apply(iter(samples)))

    assert output == [samples[1]]


def test_feature_filter_honors_explicit_ids() -> None:
    sample = make_vector(0, {"required": 1.0, "ignored": None})
    transform = FilterFeatureSamplesTransform(
        ["required", "ignored"],
        threshold=1.0,
        ids=["required"],
    )

    assert list(transform.apply(iter([sample]))) == [sample]


def test_feature_filter_rejects_unknown_ids() -> None:
    with pytest.raises(ValueError, match="Unknown vector ids"):
        FilterFeatureSamplesTransform(["a"], threshold=1.0, ids=["typo"])


def test_feature_filter_counts_categorical_values_as_present() -> None:
    sample = make_vector(0, {"category": "healthcare"})
    transform = FilterFeatureSamplesTransform(["category"], threshold=1.0)

    assert list(transform.apply(iter([sample]))) == [sample]


def test_target_filter_counts_absent_target_vector_as_missing() -> None:
    sample = make_vector(0, {"feature": 1.0})
    transform = FilterTargetSamplesTransform(["target"], threshold=0.1)

    assert list(transform.apply(iter([sample]))) == []
