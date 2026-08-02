import pytest

from jerrythomas.artifacts.models import (
    ListVectorMetadataEntry,
    ScalarVectorMetadataEntry,
    VectorMetadataEntry,
)
from jerrythomas.domain.sample import Sample
from jerrythomas.domain.vector import Vector
from jerrythomas.transforms.vector.conform import (
    ConformFeaturesTransform,
    ConformTargetsTransform,
)
from tests.unit.transforms.helpers import make_vector


def _scalar(identifier: str) -> VectorMetadataEntry:
    return ScalarVectorMetadataEntry(
        id=identifier,
        base_id=identifier,
        kind="scalar",
        present_count=0,
        null_count=0,
    )


def _sequence(identifier: str, size: int) -> VectorMetadataEntry:
    return ListVectorMetadataEntry(
        id=identifier,
        base_id=identifier,
        kind="list",
        present_count=0,
        null_count=0,
        length=size,
        observed_elements=0,
    )


def test_conform_features_fills_missing_values_and_orders_schema() -> None:
    transform = ConformFeaturesTransform(
        [_scalar("first"), _sequence("history", 2), _scalar("last")]
    )

    output = list(transform.apply(iter([make_vector(0, {"last": 3.0})])))

    assert output[0].features.values == {
        "first": None,
        "history": [None, None],
        "last": 3.0,
    }


def test_conform_features_reuses_validated_sequence() -> None:
    history = [1.0, 2.0]
    sample = make_vector(0, {"history": history})

    [output] = ConformFeaturesTransform([_sequence("history", 2)]).apply(iter([sample]))

    assert output.features.values["history"] is history


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_conform_features_expands_null_sequence(missing: object) -> None:
    sample = make_vector(0, {"history": missing})

    [output] = ConformFeaturesTransform([_sequence("history", 2)]).apply(iter([sample]))

    assert output.features.values == {"history": [None, None]}


def test_conform_features_canonicalizes_null_scalar() -> None:
    sample = make_vector(0, {"value": float("nan")})

    [output] = ConformFeaturesTransform([_scalar("value")]).apply(iter([sample]))

    assert output.features.values == {"value": None}


def test_conform_features_rejects_unknown_ids() -> None:
    transform = ConformFeaturesTransform([_scalar("known")])

    with pytest.raises(ValueError, match="unexpected ids"):
        list(transform.apply(iter([make_vector(0, {"known": 1.0, "extra": 2.0})])))


@pytest.mark.parametrize(
    "value",
    [[1.0], [1.0, 2.0, 3.0], 1.0],
)
def test_conform_features_rejects_wrong_sequence_shape(value: object) -> None:
    transform = ConformFeaturesTransform([_sequence("history", 2)])

    with pytest.raises((TypeError, ValueError)):
        list(transform.apply(iter([make_vector(0, {"history": value})])))


@pytest.mark.parametrize("value", ["category", float("inf")])
def test_conform_features_preserves_scalar_values(value: object) -> None:
    transform = ConformFeaturesTransform([_scalar("value")])

    [output] = transform.apply(iter([make_vector(0, {"value": value})]))

    assert output.features.values == {"value": value}


def test_conform_features_rejects_list_for_scalar() -> None:
    transform = ConformFeaturesTransform([_scalar("value")])

    with pytest.raises(TypeError, match="must contain a scalar"):
        list(transform.apply(iter([make_vector(0, {"value": [1.0]})])))


def test_conform_targets_creates_missing_target_vector() -> None:
    sample = make_vector(0, {"feature": 1.0})
    transform = ConformTargetsTransform([_scalar("target")])

    output = list(transform.apply(iter([sample])))

    assert output[0].features.values == {"feature": 1.0}
    assert output[0].targets is not None
    assert output[0].targets.values == {"target": None}


def test_conform_targets_preserves_feature_vector() -> None:
    sample = Sample(
        key=(0,),
        features=Vector(values={"feature": 1.0}),
        targets=Vector(values={"target": 2.0}),
    )
    transform = ConformTargetsTransform([_scalar("target")])

    output = list(transform.apply(iter([sample])))

    assert output[0].features is sample.features
    assert output[0].targets is not None
    assert output[0].targets.values == {"target": 2.0}


def test_conform_targets_fills_and_orders_scalar_and_sequence_entries() -> None:
    sample = Sample(
        key=(0,),
        features=Vector(values={"feature": 1.0}),
        targets=Vector(values={"last": 3.0, "history": [1.0, 2.0]}),
    )
    transform = ConformTargetsTransform(
        [_scalar("first"), _sequence("history", 2), _scalar("last")]
    )

    [output] = transform.apply(iter([sample]))

    assert output.targets is not None
    assert list(output.targets.values) == ["first", "history", "last"]
    assert output.targets.values == {
        "first": None,
        "history": [1.0, 2.0],
        "last": 3.0,
    }
