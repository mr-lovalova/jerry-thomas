import math

import pytest

from datapipeline.domain.value import normalize_data_value


@pytest.mark.parametrize(
    "value",
    [
        [1.0, [2.0, None]],
        (1.0, (2.0, None)),
        {"first": 1.0, "nested": {"second": None}},
    ],
)
def test_normalize_data_value_reuses_unchanged_containers(value: object) -> None:
    assert normalize_data_value(value) is value


def test_normalize_data_value_copies_only_changed_containers() -> None:
    unchanged = [1.0]
    value = {"unchanged": unchanged, "changed": [2.0, float("nan")]}

    normalized = normalize_data_value(value)

    assert normalized == {"unchanged": [1.0], "changed": [2.0, None]}
    assert normalized is not value
    assert normalized["unchanged"] is unchanged
    assert normalized["changed"] is not value["changed"]
    assert math.isnan(value["changed"][1])


def test_normalize_data_value_preserves_changed_tuple_shape() -> None:
    value = (1.0, float("nan"))

    assert normalize_data_value(value) == (1.0, None)


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_normalize_data_value_rejects_nested_infinity(value: float) -> None:
    nested = [2.0, value]
    original = [1.0, nested]

    with pytest.raises(ValueError, match="must not contain infinity"):
        normalize_data_value(original)

    assert original[1] is nested
    assert nested[1] is value
