from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from datapipeline.config.interpolation import MissingInterpolation
from datapipeline.config.transforms import WhereConfig
from datapipeline.transforms.where import WhereTransform
from tests.unit.transforms.helpers import make_time_record


def test_where_rejects_missing_interpolation_comparand():
    with pytest.raises(ValidationError, match="comparand must resolve"):
        WhereConfig(
            field="time",
            operator="ge",
            comparand=MissingInterpolation("start_time"),
        )


def test_time_where_rejects_invalid_datetime_comparand():
    with pytest.raises(ValidationError, match="Invalid ISO-8601 datetime"):
        WhereConfig(
            field="time",
            operator="ge",
            comparand="not-a-date",
        )


def test_time_where_rejects_invalid_record_time():
    record = make_time_record(1.0, 0)
    record.time = "not-a-date"
    stream = iter([record])
    selected = WhereTransform(
        field="time",
        operator="ge",
        comparand="2024-01-01T00:00:00Z",
    ).apply(stream)

    try:
        list(selected)
    except TypeError as exc:
        message = str(exc)
    else:
        raise AssertionError("time where should reject invalid record timestamps")

    assert "field 'time'" in message
    assert "datetime value" in message


def test_time_where_compares_valid_datetimes():
    before = make_time_record(1.0, 0)
    before.time = datetime(2023, 12, 31, tzinfo=timezone.utc)
    at = make_time_record(2.0, 0)
    at.time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    selected = WhereTransform(
        field="time",
        operator="ge",
        comparand="2024-01-01T00:00:00Z",
    ).apply(iter([before, at]))

    assert list(selected) == [at]


@pytest.mark.parametrize(
    ("operator", "comparand", "expected"),
    [
        ("eq", 2, [2]),
        ("ne", 2, [1, 3]),
        ("lt", 2, [1]),
        ("le", 2, [1, 2]),
        ("gt", 2, [3]),
        ("ge", 2, [2, 3]),
        ("in", [1, 3], [1, 3]),
        ("not_in", [1, 3], [2]),
    ],
)
def test_where_supports_explicit_builtin_operators(
    operator: str,
    comparand: object,
    expected: list[int],
) -> None:
    records = [make_time_record(value, 0) for value in [1, 2, 3]]

    selected = WhereTransform(
        field="value",
        operator=operator,
        comparand=comparand,
    ).apply(iter(records))

    assert [record.value for record in selected] == expected


def test_where_rejects_operator_aliases() -> None:
    with pytest.raises(ValidationError, match="operator"):
        WhereConfig(field="value", operator="nin", comparand=[1])


def test_where_requires_configured_field() -> None:
    selected = WhereTransform(
        field="missing",
        operator="eq",
        comparand=1,
    ).apply(iter([make_time_record(1.0, 0)]))

    with pytest.raises(KeyError, match="missing"):
        list(selected)


def test_where_reports_incompatible_comparison_types() -> None:
    selected = WhereTransform(
        field="value",
        operator="gt",
        comparand="1",
    ).apply(iter([make_time_record(1.0, 0)]))

    with pytest.raises(TypeError, match="operator 'gt'.*field 'value'"):
        list(selected)


def test_membership_requires_a_sequence_comparand() -> None:
    with pytest.raises(ValidationError, match="requires a list or tuple"):
        WhereConfig(field="value", operator="in", comparand=1)
