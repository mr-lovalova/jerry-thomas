from datetime import datetime

import pytest
from pydantic import ValidationError

from jerrythomas.config.resample import ResampleConfig
from jerrythomas.domain.record import TemporalRecord
from jerrythomas.transforms.stream.resample import ResampleTransform
from jerrythomas.transforms.utils import (
    record_establishes_domain,
    set_record_domain_anchor,
)
from jerrythomas.utils.time import parse_datetime


def _record(time: str, value: object, ticker: str = "A") -> TemporalRecord:
    record = TemporalRecord(time=parse_datetime(time))
    record.value = value
    record.ticker = ticker
    record.unrelated = "must not survive aggregation"
    return record


def _config(**overrides: object) -> ResampleConfig:
    return ResampleConfig.model_validate(
        {
            "period": {"kind": "calendar", "unit": "month", "timezone": "UTC"},
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-04-01T00:00:00Z",
            "aggregations": {
                statistic: {"field": "value", "statistic": statistic}
                for statistic in ("first", "last", "sum", "mean", "min", "max", "count")
            },
            **overrides,
        }
    )


def _run(records: list[TemporalRecord], **overrides: object) -> list[TemporalRecord]:
    return list(
        ResampleTransform(_config(**overrides), ("ticker",)).apply(iter(records))
    )


def test_months_preserve_gaps_and_publish_at_exclusive_end() -> None:
    records = [
        _record("2023-12-31", 999),
        _record("2024-01-01", 2),
        _record("2024-01-31T23:59:59Z", 4),
        _record("2024-03-01", 8),
        _record("2024-04-01", 999),
    ]
    output = _run(records)
    assert [r.time for r in output] == list(
        map(parse_datetime, ["2024-02-01", "2024-03-01", "2024-04-01"])
    )
    assert [
        (r.first, r.last, r.sum, r.mean, r.min, r.max, r.count) for r in output
    ] == [
        (2, 4, 6, 3.0, 2, 4, 2),
        (None, None, None, None, None, None, 0),
        (8, 8, 8, 8.0, 8, 8, 1),
    ]
    assert [record_establishes_domain(r) for r in output] == [True, False, True]
    assert all(r.ticker == "A" and not hasattr(r, "unrelated") for r in output)
    assert records[1].value == 2 and not hasattr(records[1], "sum")


def test_inferred_coverage_drops_partial_first_and_last_month() -> None:
    output = _run(
        [
            _record("2024-01-15", 1),
            _record("2024-02-15", 2),
            _record("2024-03-15", 3),
        ],
        start=None,
        end=None,
    )
    assert [(r.time, r.last) for r in output] == [(parse_datetime("2024-03-01"), 2)]


def test_declared_partial_bounds_do_not_complete_edge_months() -> None:
    output = _run(
        [_record("2024-01-20", 1), _record("2024-02-20", 2), _record("2024-03-05", 3)],
        start="2024-01-15",
        end="2024-03-15",
    )
    assert [(r.time, r.last) for r in output] == [(parse_datetime("2024-03-01"), 2)]


def test_declared_coverage_emits_leading_and_trailing_empty_months() -> None:
    output = _run([_record("2024-02-15", 2)])
    assert [r.last for r in output] == [None, 2, None]
    assert _run([]) == []


def test_no_inferred_final_bucket_at_eof_even_when_first_is_aligned() -> None:
    assert _run([_record("2024-01-01", 1)], start=None, end=None) == []


def test_partitions_have_independent_buckets_and_coverage() -> None:
    output = _run(
        [
            _record("2024-01-01", 1, "A"),
            _record("2024-02-01", 2, "A"),
            _record("2024-02-01", 3, "B"),
            _record("2024-03-01", 4, "B"),
        ],
        start=None,
        end=None,
    )
    assert [(r.ticker, r.time, r.last) for r in output] == [
        ("A", parse_datetime("2024-02-01"), 1),
        ("B", parse_datetime("2024-03-01"), 3),
    ]


def test_calendar_boundaries_follow_local_month_and_dst() -> None:
    output = _run(
        [
            _record("2024-03-01T04:59:59Z", 99),
            _record("2024-03-01T05:00:00Z", 1),
            _record("2024-04-01T03:59:59Z", 2),
            _record("2024-04-01T04:00:00Z", 3),
        ],
        period={"kind": "calendar", "unit": "month", "timezone": "America/New_York"},
        start="2024-03-01T00:00:00-05:00",
        end="2024-05-01T00:00:00-04:00",
    )
    assert [(r.time, r.sum) for r in output] == [
        (parse_datetime("2024-04-01T04:00:00Z"), 3),
        (parse_datetime("2024-05-01T04:00:00Z"), 3),
    ]


def test_calendar_month_rolls_over_year() -> None:
    output = _run([_record("2023-12-31", 1)], start="2023-12-01", end="2024-02-01")
    assert [r.time for r in output] == [
        parse_datetime("2024-01-01"),
        parse_datetime("2024-02-01"),
    ]


def test_fixed_periods_use_utc_epoch_alignment_and_same_completion_rules() -> None:
    output = _run(
        [
            _record("2024-01-01T00:01:00Z", 1),
            _record("2024-01-01T00:10:00Z", 2),
            _record("2024-01-01T00:19:59Z", 3),
        ],
        period={"kind": "fixed", "every": "10m"},
        start="2024-01-01T01:00:00+01:00",
        end="2024-01-01T00:30:00Z",
    )
    assert [(r.time.minute, r.sum) for r in output] == [(10, 1), (20, 5), (30, None)]


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_first_and_last_preserve_missing_endpoints(missing: object) -> None:
    output = _run(
        [
            _record("2024-01-01", missing),
            _record("2024-01-02", 5),
            _record("2024-01-03", missing),
        ]
    )[0]
    assert output.first is None and output.last is None
    assert (output.sum, output.mean, output.count) == (5, 5.0, 1)


def test_placeholder_inputs_do_not_establish_sample_domain() -> None:
    placeholder = _record("2024-01-01", None)
    set_record_domain_anchor(placeholder, False)
    assert not record_establishes_domain(_run([placeholder])[0])


def test_resample_preserves_large_integer_sums_and_cancellation() -> None:
    assert (
        _run([_record("2024-01-01", 2**60 + 1), _record("2024-01-02", 2)])[0].sum
        == 2**60 + 3
    )
    assert _run([_record("2024-01-01", n) for n in (1e16, 1.0, -1e16)])[0].sum == 1.0


def test_mean_does_not_overflow_when_sum_would() -> None:
    output = _run(
        [_record("2024-01-01", 1e308), _record("2024-01-02", 1e308)],
        aggregations={"value": {"field": "value", "statistic": "mean"}},
    )
    assert output[0].value == 1e308


@pytest.mark.parametrize("statistic", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("value", [True, "1", float("inf")])
def test_numeric_aggregations_reject_invalid_values(
    statistic: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError), match="numeric"):
        _run(
            [_record("2024-01-01", value)],
            aggregations={"value": {"field": "value", "statistic": statistic}},
        )


def test_first_last_and_count_support_nonnumeric_fields() -> None:
    output = _run(
        [_record("2024-01-01", "a"), _record("2024-01-02", "b")],
        aggregations={
            s: {"field": "value", "statistic": s} for s in ("first", "last", "count")
        },
    )
    assert (output[0].first, output[0].last, output[0].count) == ("a", "b", 2)


def test_resample_rejects_time_reversal_and_missing_fields() -> None:
    with pytest.raises(ValueError, match="time-ordered"):
        _run([_record("2024-01-02", 1), _record("2024-01-01", 2)])
    with pytest.raises(KeyError, match="absent"):
        _run(
            [_record("2024-01-01", 1)],
            aggregations={"x": {"field": "absent", "statistic": "last"}},
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"period": {"kind": "fixed", "every": "0d"}},
        {"period": {"kind": "fixed", "every": "1mo"}},
        {"period": {"kind": "calendar", "unit": "month"}},
        {"period": {"kind": "calendar", "unit": "month", "timezone": "Not/AZone"}},
        {"period": {"kind": "calendar", "unit": "day", "timezone": "UTC"}},
        {"start": "2024-05-01"},
        {"start": datetime(2024, 1, 1)},
        {"aggregations": {}},
        {"aggregations": {"time": {"field": "value", "statistic": "last"}}},
        {
            "aggregations": {
                "_establishes_domain": {"field": "value", "statistic": "last"}
            }
        },
        {"aggregations": {"x": {"field": "value", "statistic": "median"}}},
        {"closed": "right"},
    ],
)
def test_invalid_resample_configuration_is_rejected(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _config(**overrides)
