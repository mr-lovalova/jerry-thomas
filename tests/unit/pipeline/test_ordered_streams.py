from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from jerrythomas.pipelines.stream.order import (
    build_record_order_stage,
    sort_records,
    validate_record_order,
)
from jerrythomas.pipelines.sort import SortProgress


@dataclass
class _Record:
    time: datetime
    security_id: object


def _ts(day: int) -> datetime:
    return datetime(2024, 1, day, tzinfo=timezone.utc)


def test_validate_record_order_preserves_records() -> None:
    records = [
        _Record(time=_ts(1), security_id="AAPL"),
        _Record(time=_ts(1), security_id="AAPL"),
        _Record(time=_ts(2), security_id="MSFT"),
    ]

    ordered = list(
        validate_record_order(
            ("security_id",),
            iter(records),
        )
    )

    assert ordered == records
    assert all(actual is expected for actual, expected in zip(ordered, records))


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([(2, "AAPL"), (1, "AAPL")], id="time-regresses"),
        pytest.param([(1, "AAPL"), (1, "MSFT"), (2, "AAPL")], id="partition-reappears"),
        pytest.param(
            [(1, float("nan")), (2, float("nan"))], id="unordered-partition-key"
        ),
    ],
)
def test_validate_record_order_rejects_false_presorted_declaration(rows) -> None:
    records = [
        _Record(time=_ts(day), security_id=security_id) for day, security_id in rows
    ]
    with pytest.raises(ValueError, match="violates presorted order|finite floats"):
        list(
            validate_record_order(
                ("security_id",),
                iter(records),
            )
        )


def test_validate_record_order_error_uses_config_list_syntax() -> None:
    records = [
        _Record(time=_ts(2), security_id="AAPL"),
        _Record(time=_ts(1), security_id="AAPL"),
    ]

    with pytest.raises(
        ValueError,
        match=r"presorted order \['security_id', 'time'\]",
    ):
        list(
            validate_record_order(
                ("security_id",),
                iter(records),
            )
        )


def test_sort_records_orders_by_partition_then_time() -> None:
    records = [
        _Record(time=_ts(2), security_id="MSFT"),
        _Record(time=_ts(1), security_id="AAPL"),
    ]

    ordered = list(
        sort_records(
            ("security_id",),
            1_000_000,
            SortProgress(),
            iter(records),
        )
    )

    assert [(rec.security_id, rec.time.day) for rec in ordered] == [
        ("AAPL", 1),
        ("MSFT", 2),
    ]


@pytest.mark.parametrize(
    "values",
    [
        [False, 0],
        [1, 1.0],
    ],
)
def test_presorted_records_reject_mixed_exact_partition_types(values) -> None:
    records = [
        _Record(time=_ts(position), security_id=value)
        for position, value in enumerate(values, start=1)
    ]

    with pytest.raises(
        TypeError,
        match="partition field 'security_id'.*one exact type",
    ):
        list(validate_record_order(("security_id",), iter(records)))


@pytest.mark.parametrize("values", [[False, 0], [1, 1.0]])
def test_sorted_records_reject_mixed_exact_partition_types(values) -> None:
    records = [
        _Record(time=_ts(position), security_id=value)
        for position, value in enumerate(values, start=1)
    ]

    with pytest.raises(
        TypeError,
        match="partition field 'security_id'.*one exact type",
    ):
        list(
            sort_records(
                ("security_id",),
                1_000_000,
                SortProgress(),
                iter(records),
            )
        )


def test_only_real_sort_nodes_expose_sort_progress() -> None:
    validate = build_record_order_stage(("security_id",), True, 1_000_000)
    sort = build_record_order_stage(("security_id",), False, 1_000_000)

    assert validate.progress is None
    assert sort.progress is not None
