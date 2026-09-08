from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from jerrythomas.alignment.as_of import as_of_stream
from jerrythomas.domain.record import TemporalRecord


@dataclass
class _Record(TemporalRecord):
    id_: object
    value: str


def _record(id_: object, hour: int, value: str) -> _Record:
    return _Record(
        time=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=hour),
        id_=id_,
        value=value,
    )


class _TrackedIterator(Iterator[TemporalRecord]):
    def __init__(
        self,
        records: Sequence[TemporalRecord],
        close_error: Exception | None = None,
    ) -> None:
        self._records = iter(records)
        self._close_error = close_error
        self.close_calls = 0

    def __next__(self) -> TemporalRecord:
        return next(self._records)

    def close(self) -> None:
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error


def test_as_of_stream_uses_latest_eligible_lookup_in_the_same_partition() -> None:
    primary = [
        _record("A", 2, "primary-A2"),
        _record("A", 4, "primary-A4"),
        _record("A", 5, "primary-A5"),
        _record("B", 2, "primary-B2"),
        _record("B", 4, "primary-B4"),
    ]
    lookup = [
        _record("A", 1, "lookup-A1"),
        _record("A", 4, "lookup-A4"),
        _record("A", 6, "unused-A6"),
        _record("B", 1, "lookup-B1"),
        _record("B", 3, "lookup-B3"),
    ]

    rows = list(
        as_of_stream(
            iter(primary),
            iter(lookup),
            partition_by=("id_",),
        )
    )

    assert [
        (primary_record.value, lookup_record.value if lookup_record else None)
        for primary_record, lookup_record in rows
    ] == [
        ("primary-A2", "lookup-A1"),
        ("primary-A4", "lookup-A4"),
        ("primary-A5", "lookup-A4"),
        ("primary-B2", "lookup-B1"),
        ("primary-B4", "lookup-B3"),
    ]
    assert rows[1][1] is rows[2][1]


def test_as_of_stream_does_not_use_a_future_lookup() -> None:
    rows = list(
        as_of_stream(
            iter([_record("A", 1, "primary")]),
            iter([_record("A", 2, "future")]),
            partition_by=("id_",),
            require_match=False,
        )
    )

    assert [(primary.value, lookup) for primary, lookup in rows] == [("primary", None)]


def test_as_of_stream_does_not_reuse_a_lookup_across_partitions() -> None:
    rows = list(
        as_of_stream(
            iter(
                [
                    _record("A", 2, "primary-A"),
                    _record("B", 2, "primary-B"),
                ]
            ),
            iter([_record("A", 1, "lookup-A")]),
            partition_by=("id_",),
            require_match=False,
        )
    )

    assert [
        (primary.value, lookup.value if lookup else None) for primary, lookup in rows
    ] == [
        ("primary-A", "lookup-A"),
        ("primary-B", None),
    ]


def test_as_of_stream_requires_a_match_by_default() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "As-of lookup has no eligible record for primary "
            r"partition=\('A',\), time=2025-01-01T01:00:00\+00:00"
        ),
    ):
        list(
            as_of_stream(
                iter([_record("A", 1, "primary")]),
                iter([_record("A", 2, "future")]),
                partition_by=("id_",),
            )
        )


def test_as_of_stream_treats_max_age_as_inclusive() -> None:
    rows = list(
        as_of_stream(
            iter(
                [
                    _record("A", 3, "inclusive"),
                    _record("A", 4, "stale"),
                ]
            ),
            iter([_record("A", 1, "lookup")]),
            partition_by=("id_",),
            max_age=timedelta(hours=2),
            require_match=False,
        )
    )

    assert rows[0][1] is not None
    assert rows[0][1].value == "lookup"
    assert rows[1][1] is None


def test_as_of_stream_zero_max_age_matches_only_exact_keys() -> None:
    rows = list(
        as_of_stream(
            iter(
                [
                    _record("A", 1, "primary-A1"),
                    _record("A", 2, "primary-A2"),
                    _record("A", 3, "primary-A3"),
                ]
            ),
            iter(
                [
                    _record("A", 0, "stale-A0"),
                    _record("A", 2, "exact-A2"),
                ]
            ),
            partition_by=("id_",),
            max_age=timedelta(0),
            require_match=False,
        )
    )

    assert [lookup.value if lookup is not None else None for _, lookup in rows] == [
        None,
        "exact-A2",
        None,
    ]


def test_as_of_stream_zero_max_age_can_require_an_exact_match() -> None:
    with pytest.raises(ValueError, match="has no eligible record"):
        list(
            as_of_stream(
                iter([_record("A", 2, "primary")]),
                iter([_record("A", 1, "stale")]),
                partition_by=("id_",),
                max_age=timedelta(0),
            )
        )


def test_as_of_stream_rejects_negative_max_age() -> None:
    with pytest.raises(ValueError, match="max_age must be a non-negative timedelta"):
        list(
            as_of_stream(
                iter(()),
                iter(()),
                partition_by=("id_",),
                max_age=timedelta(microseconds=-1),
            )
        )


def test_as_of_stream_rejects_non_timedelta_max_age() -> None:
    with pytest.raises(TypeError, match="max_age must be a timedelta or None"):
        list(
            as_of_stream(
                iter(()),
                iter(()),
                partition_by=("id_",),
                max_age=60,  # type: ignore[arg-type]
            )
        )


def test_as_of_stream_rejects_non_boolean_require_match() -> None:
    with pytest.raises(TypeError, match="require_match must be a boolean"):
        list(
            as_of_stream(
                iter(()),
                iter(()),
                partition_by=("id_",),
                require_match=1,  # type: ignore[arg-type]
            )
        )


def test_as_of_stream_rejects_duplicate_primary_keys() -> None:
    with pytest.raises(ValueError) as error:
        list(
            as_of_stream(
                iter(
                    [
                        _record("A", 1, "first"),
                        _record("A", 1, "duplicate"),
                    ]
                ),
                iter([_record("A", 1, "lookup")]),
                partition_by=("id_",),
            )
        )

    assert "As-of primary has duplicate canonical key" in str(error.value)
    assert "partition=('A',)" in str(error.value)
    assert "2025-01-01T01:00:00+00:00" in str(error.value)


@pytest.mark.parametrize(
    "primary",
    [
        [
            _record("A", 2, "second"),
            _record("A", 1, "first"),
        ],
        [
            _record("B", 1, "B"),
            _record("A", 1, "A"),
        ],
    ],
)
def test_as_of_stream_rejects_unordered_primary(
    primary: list[_Record],
) -> None:
    with pytest.raises(ValueError, match="As-of primary is not ordered"):
        list(
            as_of_stream(
                iter(primary),
                iter([_record("A", 0, "lookup-A"), _record("B", 0, "lookup-B")]),
                partition_by=("id_",),
                require_match=False,
            )
        )


def test_as_of_stream_rejects_duplicate_lookup_keys() -> None:
    with pytest.raises(ValueError) as error:
        list(
            as_of_stream(
                iter([_record("A", 2, "primary")]),
                iter(
                    [
                        _record("A", 1, "first"),
                        _record("A", 1, "duplicate"),
                    ]
                ),
                partition_by=("id_",),
            )
        )

    assert "As-of lookup has duplicate canonical key" in str(error.value)
    assert "partition=('A',)" in str(error.value)
    assert "2025-01-01T01:00:00+00:00" in str(error.value)


@pytest.mark.parametrize(
    "lookup",
    [
        [
            _record("A", 2, "second"),
            _record("A", 1, "first"),
        ],
        [
            _record("B", 1, "B"),
            _record("A", 1, "A"),
        ],
    ],
)
def test_as_of_stream_rejects_unordered_lookup(lookup: list[_Record]) -> None:
    with pytest.raises(ValueError, match="As-of lookup is not ordered"):
        list(
            as_of_stream(
                iter([_record("B", 3, "primary")]),
                iter(lookup),
                partition_by=("id_",),
            )
        )


@pytest.mark.parametrize("missing_from", ["primary", "lookup"])
def test_as_of_stream_requires_every_partition_field(missing_from: str) -> None:
    primary = [_record("A", 1, "primary")]
    lookup = [_record("A", 1, "lookup")]
    if missing_from == "primary":
        del primary[0].id_
    else:
        del lookup[0].id_

    with pytest.raises(KeyError, match="Partition field 'id_' not found"):
        list(
            as_of_stream(
                iter(primary),
                iter(lookup),
                partition_by=("id_",),
            )
        )


def test_as_of_stream_requires_matching_partition_types() -> None:
    with pytest.raises(
        TypeError,
        match=("As-of lookup partition field 'id_' uses bool; primary uses int"),
    ):
        list(
            as_of_stream(
                iter([_record(1, 1, "primary")]),
                iter([_record(True, 1, "lookup")]),
                partition_by=("id_",),
            )
        )


def test_as_of_stream_consumes_lookup_with_one_record_of_lookahead() -> None:
    consumed: list[int] = []

    def lookup_records():
        for hour in (1, 3, 4):
            consumed.append(hour)
            yield _record("A", hour, f"lookup-{hour}")

    rows = as_of_stream(
        iter([_record("A", 2, "primary")]),
        lookup_records(),
        partition_by=("id_",),
    )

    primary, lookup = next(rows)

    assert primary.value == "primary"
    assert lookup is not None
    assert lookup.value == "lookup-1"
    assert consumed == [1, 3]

    rows.close()

    assert consumed == [1, 3]


def test_as_of_stream_rejects_unordered_lookup_hidden_after_lookahead() -> None:
    with pytest.raises(ValueError, match="As-of lookup is not ordered"):
        list(
            as_of_stream(
                iter([_record("A", 3, "primary")]),
                iter(
                    [
                        _record("A", 1, "lookup-1"),
                        _record("A", 4, "future-lookahead"),
                        _record("A", 2, "hidden-eligible"),
                    ]
                ),
                partition_by=("id_",),
            )
        )


def test_as_of_stream_rejects_duplicate_lookup_hidden_after_lookahead() -> None:
    with pytest.raises(ValueError, match="duplicate canonical key"):
        list(
            as_of_stream(
                iter([_record("A", 3, "primary")]),
                iter(
                    [
                        _record("A", 1, "lookup-1"),
                        _record("A", 4, "future-lookahead"),
                        _record("A", 4, "hidden-duplicate"),
                    ]
                ),
                partition_by=("id_",),
            )
        )


def test_as_of_stream_validates_lookup_when_primary_is_empty() -> None:
    with pytest.raises(ValueError, match="As-of lookup is not ordered"):
        list(
            as_of_stream(
                iter(()),
                iter(
                    [
                        _record("A", 2, "second"),
                        _record("A", 1, "first"),
                    ]
                ),
                partition_by=("id_",),
            )
        )


def test_as_of_stream_does_not_drain_lookup_after_a_match_error() -> None:
    consumed: list[int] = []

    def lookup_records():
        for hour in (2, 3):
            consumed.append(hour)
            yield _record("A", hour, f"lookup-{hour}")

    with pytest.raises(ValueError, match="has no eligible record"):
        list(
            as_of_stream(
                iter([_record("A", 1, "primary")]),
                lookup_records(),
                partition_by=("id_",),
            )
        )

    assert consumed == [2]


def test_as_of_stream_closes_both_inputs_after_success() -> None:
    primary = _TrackedIterator([_record("A", 1, "primary")])
    lookup = _TrackedIterator([_record("A", 1, "lookup")])

    assert len(list(as_of_stream(primary, lookup, ("id_",)))) == 1
    assert primary.close_calls == 1
    assert lookup.close_calls == 1


def test_as_of_stream_closes_both_inputs_when_consumer_stops() -> None:
    primary = _TrackedIterator(
        [
            _record("A", 1, "primary-1"),
            _record("A", 2, "primary-2"),
        ]
    )
    lookup = _TrackedIterator(
        [
            _record("A", 1, "lookup-1"),
            _record("A", 2, "lookup-2"),
        ]
    )
    rows = as_of_stream(primary, lookup, ("id_",))

    next(rows)
    rows.close()

    assert primary.close_calls == 1
    assert lookup.close_calls == 1


def test_as_of_stream_preserves_processing_error_when_cleanup_fails() -> None:
    primary = _TrackedIterator(
        [_record("A", 1, "primary")],
        close_error=RuntimeError("primary close failed"),
    )
    lookup = _TrackedIterator(
        [_record("A", 2, "future")],
        close_error=RuntimeError("lookup close failed"),
    )

    with pytest.raises(ValueError, match="has no eligible record"):
        list(as_of_stream(primary, lookup, ("id_",)))

    assert primary.close_calls == 1
    assert lookup.close_calls == 1


def test_as_of_stream_forward_pairs_with_next_lookup_in_partition() -> None:
    primary = [
        _record("A", 2, "primary-A2"),
        _record("A", 4, "primary-A4"),
        _record("A", 5, "primary-A5"),
        _record("B", 2, "primary-B2"),
    ]
    lookup = [
        _record("A", 5, "lookup-A5"),
        _record("A", 7, "unused-A7"),
        _record("B", 3, "lookup-B3"),
        _record("B", 6, "lookup-B6"),
    ]

    rows = list(
        as_of_stream(
            iter(primary),
            iter(lookup),
            partition_by=("id_",),
            direction="forward",
        )
    )

    assert [
        (primary_record.value, lookup_record.value if lookup_record else None)
        for primary_record, lookup_record in rows
    ] == [
        ("primary-A2", "lookup-A5"),
        ("primary-A4", "lookup-A5"),
        ("primary-A5", "lookup-A5"),
        ("primary-B2", "lookup-B3"),
    ]


def test_as_of_stream_forward_requires_match_by_default() -> None:
    with pytest.raises(ValueError, match="has no eligible record"):
        list(
            as_of_stream(
                iter([_record("A", 8, "primary")]),
                iter([_record("A", 1, "past"), _record("A", 5, "last")]),
                partition_by=("id_",),
                direction="forward",
            )
        )


def test_as_of_stream_forward_drops_unmatched_past_last_lookup() -> None:
    rows = list(
        as_of_stream(
            iter([_record("A", 2, "early"), _record("A", 8, "late")]),
            iter([_record("A", 5, "last")]),
            partition_by=("id_",),
            direction="forward",
            require_match=False,
        )
    )

    assert [
        (primary.value, lookup.value if lookup else None) for primary, lookup in rows
    ] == [("early", "last"), ("late", None)]


def test_as_of_stream_forward_treats_max_age_as_inclusive_ahead() -> None:
    within = list(
        as_of_stream(
            iter([_record("A", 3, "primary")]),
            iter([_record("A", 6, "lookup")]),
            partition_by=("id_",),
            max_age=timedelta(hours=3),
            direction="forward",
        )
    )
    assert [
        (primary.value, lookup.value if lookup else None) for primary, lookup in within
    ] == [("primary", "lookup")]

    dropped = list(
        as_of_stream(
            iter([_record("A", 4, "primary")]),
            iter([_record("A", 6, "lookup")]),
            partition_by=("id_",),
            max_age=timedelta(hours=1),
            direction="forward",
            require_match=False,
        )
    )
    assert [
        (primary.value, lookup.value if lookup else None) for primary, lookup in dropped
    ] == [("primary", None)]


def test_as_of_stream_forward_rejects_invalid_direction() -> None:
    with pytest.raises(ValueError, match="direction must be"):
        list(
            as_of_stream(
                iter([_record("A", 2, "primary")]),
                iter([_record("A", 3, "lookup")]),
                partition_by=("id_",),
                direction="sideways",
            )
        )
