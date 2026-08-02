import pytest
from pydantic import ValidationError

from jerrythomas.config.transforms import DeriveConfig
from jerrythomas.transforms.stream.derive import DeriveTransform
from tests.unit.transforms.helpers import make_time_record


def test_derive_divides_two_fields() -> None:
    record = make_time_record(0.0, 0)
    record.close_lag_21 = 110.0
    record.close_lag_189 = 100.0
    transform = DeriveTransform(
        left="close_lag_21",
        operator="div",
        right_field="close_lag_189",
        to="close_ratio_21_189",
    )

    [out] = list(transform.apply(iter([record])))

    assert out.close_ratio_21_189 == 1.1
    assert not hasattr(record, "close_ratio_21_189")


def test_derive_accepts_right_value() -> None:
    record = make_time_record(0.0, 0)
    record.close_ratio_21_189 = 1.1
    transform = DeriveTransform(
        left="close_ratio_21_189",
        operator="sub",
        right_value=1.0,
        to="momentum_189_21",
    )

    [out] = list(transform.apply(iter([record])))

    assert out.momentum_189_21 == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("add", 8.0),
        ("sub", 2.0),
        ("mul", 15.0),
        ("div", 5.0 / 3.0),
    ],
)
def test_derive_supports_basic_operators(operator: str, expected: float) -> None:
    record = make_time_record(0.0, 0)
    record.left = 5.0
    record.right = 3.0
    transform = DeriveTransform(
        left="left",
        operator=operator,
        right_field="right",
        to="derived",
    )

    [out] = list(transform.apply(iter([record])))

    assert out.derived == expected


@pytest.mark.parametrize("right", [None, float("nan")])
def test_derive_writes_none_for_missing_inputs(right) -> None:
    record = make_time_record(0.0, 0)
    record.left = 5.0
    record.right = right
    transform = DeriveTransform(
        left="left",
        operator="add",
        right_field="right",
        to="derived",
    )

    [out] = list(transform.apply(iter([record])))

    assert out.derived is None


def test_derive_rejects_non_numeric_inputs() -> None:
    record = make_time_record(0.0, 0)
    record.left = 5.0
    record.right = "not-a-number"
    transform = DeriveTransform(
        left="left",
        operator="add",
        right_field="right",
        to="derived",
    )

    with pytest.raises(TypeError, match="Field 'right'.*numeric"):
        list(transform.apply(iter([record])))


def test_derive_rejects_divide_by_zero() -> None:
    record = make_time_record(0.0, 0)
    record.left = 5.0
    record.right = 0.0
    transform = DeriveTransform(
        left="left",
        operator="div",
        right_field="right",
        to="derived",
    )

    with pytest.raises(ZeroDivisionError, match="divide by zero"):
        list(transform.apply(iter([record])))


def test_derive_requires_configured_fields() -> None:
    record = make_time_record(0.0, 0)
    record.right = 1.0
    transform = DeriveTransform(
        left="missing",
        operator="add",
        right_field="right",
        to="derived",
    )

    with pytest.raises(KeyError, match="missing"):
        list(transform.apply(iter([record])))


def test_derive_operations_compose_explicitly() -> None:
    record = make_time_record(0.0, 0)
    record.close_lag_21 = 110.0
    record.close_lag_189 = 100.0

    ratio = DeriveTransform(
        left="close_lag_21",
        operator="div",
        right_field="close_lag_189",
        to="close_ratio_21_189",
    ).apply(iter([record]))
    [out] = DeriveTransform(
        left="close_ratio_21_189",
        operator="sub",
        right_value=1.0,
        to="momentum_189_21",
    ).apply(ratio)

    assert out.close_ratio_21_189 == 1.1
    assert out.momentum_189_21 == pytest.approx(0.1)


def test_derive_requires_exactly_one_right_operand() -> None:
    message = "requires exactly one of right_field or right_value"

    with pytest.raises(ValidationError, match=message):
        DeriveConfig(left="left", operator="add", to="derived")

    with pytest.raises(ValidationError, match=message):
        DeriveConfig(
            left="left",
            operator="add",
            right_field="right",
            right_value=1.0,
            to="derived",
        )


@pytest.mark.parametrize(
    "config",
    [
        DeriveConfig(
            left="left",
            operator="add",
            right_field="right",
            to="derived",
        ),
        DeriveConfig(
            left="left",
            operator="add",
            right_value=0,
            to="derived",
        ),
    ],
)
def test_derive_config_round_trips(config: DeriveConfig) -> None:
    assert DeriveConfig.model_validate(config.model_dump()) == config


def test_derive_rejects_unknown_operator() -> None:
    with pytest.raises(ValidationError, match="operator"):
        DeriveConfig(
            left="left",
            operator="pow",
            right_value=2.0,
            to="derived",
        )


@pytest.mark.parametrize("right_value", [True, "1", None, float("inf")])
def test_derive_literal_must_be_a_finite_number(right_value: object) -> None:
    with pytest.raises(ValidationError, match="right_value"):
        DeriveConfig.model_validate(
            {
                "left": "left",
                "operator": "add",
                "right_value": right_value,
                "to": "derived",
            }
        )
