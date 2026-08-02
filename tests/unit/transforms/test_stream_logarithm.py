import math

import pytest

from jerrythomas.transforms.stream.derive import DeriveTransform
from jerrythomas.transforms.stream.logarithm import Log1pTransform, LogTransform
from tests.unit.transforms.helpers import make_time_record


def test_log_computes_natural_log_without_mutating_input() -> None:
    record = make_time_record(math.e**2, 0)
    setattr(record, "label", "kept")

    [output] = LogTransform("value", "logged").apply(iter([record]))

    assert getattr(output, "logged") == pytest.approx(2.0)
    assert getattr(output, "label") == "kept"
    assert not hasattr(record, "logged")


def test_log_can_replace_a_field_on_the_cloned_record() -> None:
    record = make_time_record(math.e, 0)

    [output] = LogTransform("value", "value").apply(iter([record]))

    assert getattr(output, "value") == pytest.approx(1.0)
    assert getattr(record, "value") == pytest.approx(math.e)


@pytest.mark.parametrize("value", [1e-20, -1e-20])
def test_log1p_preserves_tiny_values(value: float) -> None:
    record = make_time_record(value, 0)

    [output] = Log1pTransform("value", "logged").apply(iter([record]))

    assert getattr(output, "logged") == math.log1p(value)
    assert getattr(output, "logged") != 0.0


def test_log_accepts_the_smallest_positive_float() -> None:
    value = math.nextafter(0.0, 1.0)
    record = make_time_record(value, 0)

    [output] = LogTransform("value", "logged").apply(iter([record]))

    assert getattr(output, "logged") == math.log(value)


def test_log1p_accepts_the_float_immediately_above_minus_one() -> None:
    value = math.nextafter(-1.0, 0.0)
    record = make_time_record(value, 0)

    [output] = Log1pTransform("value", "logged").apply(iter([record]))

    assert getattr(output, "logged") == math.log1p(value)


@pytest.mark.parametrize("value", [0.0, -0.0, -1.0])
def test_log_rejects_nonpositive_values(value: float) -> None:
    record = make_time_record(value, 0)

    with pytest.raises(ValueError, match="greater than zero for log"):
        list(LogTransform("value", "logged").apply(iter([record])))


@pytest.mark.parametrize("value", [-1.0, -2.0])
def test_log1p_rejects_values_at_or_below_minus_one(value: float) -> None:
    record = make_time_record(value, 0)

    with pytest.raises(ValueError, match="greater than -1 for log1p"):
        list(Log1pTransform("value", "logged").apply(iter([record])))


@pytest.mark.parametrize("value", [None, float("nan")])
def test_logarithms_preserve_missing_values(value: float | None) -> None:
    for transform in (
        LogTransform("value", "logged"),
        Log1pTransform("value", "logged"),
    ):
        record = make_time_record(value, 0)

        [output] = transform.apply(iter([record]))

        assert getattr(output, "logged") is None


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, TypeError),
        ("1", TypeError),
        (float("inf"), ValueError),
        (float("-inf"), ValueError),
    ],
)
@pytest.mark.parametrize("transform_type", [LogTransform, Log1pTransform])
def test_logarithms_require_finite_numeric_values(
    value: object,
    error: type[Exception],
    transform_type: type[LogTransform] | type[Log1pTransform],
) -> None:
    record = make_time_record(0.0, 0)
    setattr(record, "value", value)

    with pytest.raises(error, match="Field 'value'"):
        list(transform_type("value", "logged").apply(iter([record])))


@pytest.mark.parametrize("transform_type", [LogTransform, Log1pTransform])
def test_logarithms_require_the_configured_field(
    transform_type: type[LogTransform] | type[Log1pTransform],
) -> None:
    record = make_time_record(1.0, 0)

    with pytest.raises(KeyError, match="missing"):
        list(transform_type("missing", "logged").apply(iter([record])))


def test_log1p_composes_with_derive_for_log_returns() -> None:
    record = make_time_record(0.0, 0)
    setattr(record, "current", 101.0)
    setattr(record, "previous", 100.0)

    ratio = DeriveTransform(
        left="current",
        operator="div",
        right_field="previous",
        to="ratio",
    ).apply(iter([record]))
    simple_return = DeriveTransform(
        left="ratio",
        operator="sub",
        right_value=1.0,
        to="simple_return",
    ).apply(ratio)
    [output] = Log1pTransform("simple_return", "log_return").apply(simple_return)

    assert getattr(output, "log_return") == pytest.approx(math.log(1.01))
