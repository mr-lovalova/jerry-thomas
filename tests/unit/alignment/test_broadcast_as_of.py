from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from jerrythomas.alignment.broadcast_as_of import broadcast_as_of_stream
from jerrythomas.domain.record import TemporalRecord


@dataclass
class _PrimaryRecord(TemporalRecord):
    id_: str
    value: str


@dataclass
class _LookupRecord(TemporalRecord):
    value: str


def _time(day: int, hour: int = 0) -> datetime:
    return datetime(2025, 1, day, hour, tzinfo=UTC)


def _primary(
    id_: str,
    day: int,
    value: str,
    hour: int = 0,
) -> _PrimaryRecord:
    return _PrimaryRecord(time=_time(day, hour), id_=id_, value=value)


def _lookup(day: int, value: str, hour: int = 0) -> _LookupRecord:
    return _LookupRecord(time=_time(day, hour), value=value)


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


def test_broadcast_as_of_uses_latest_lookup_across_primary_partitions() -> None:
    primary = [
        _primary("A", 2, "A-2"),
        _primary("A", 3, "A-3"),
        _primary("A", 4, "A-4"),
        _primary("B", 1, "B-1"),
        _primary("B", 2, "B-2"),
        _primary("B", 4, "B-4"),
    ]
    lookup = [_lookup(1, "day-1"), _lookup(3, "day-3")]

    rows = list(
        broadcast_as_of_stream(
            iter(primary),
            iter(lookup),
            partition_by=("id_",),
        )
    )

    assert [
        (left.value, right.value if right is not None else None) for left, right in rows
    ] == [
        ("A-2", "day-1"),
        ("A-3", "day-3"),
        ("A-4", "day-3"),
        ("B-1", "day-1"),
        ("B-2", "day-1"),
        ("B-4", "day-3"),
    ]
    assert rows[0][1] is rows[4][1]
    assert rows[2][1] is rows[5][1]


def test_broadcast_as_of_indexes_lookup_before_consuming_primary() -> None:
    lookup_exhausted = False

    def lookup_records():
        nonlocal lookup_exhausted
        try:
            yield _lookup(1, "lookup")
        finally:
            lookup_exhausted = True

    def primary_records():
        assert lookup_exhausted
        yield _primary("A", 1, "primary")

    rows = list(
        broadcast_as_of_stream(
            primary_records(),
            lookup_records(),
            partition_by=("id_",),
        )
    )

    assert len(rows) == 1


def test_broadcast_as_of_exact_match_is_eligible_at_max_age_boundary() -> None:
    rows = list(
        broadcast_as_of_stream(
            iter(
                [
                    _primary("A", 1, "exact", hour=2),
                    _primary("A", 1, "boundary", hour=4),
                ]
            ),
            iter([_lookup(1, "lookup", hour=2)]),
            partition_by=("id_",),
            max_age=timedelta(hours=2),
        )
    )

    assert [
        (left.value, right.value if right is not None else None) for left, right in rows
    ] == [("exact", "lookup"), ("boundary", "lookup")]


def test_broadcast_as_of_rejects_lookup_older_than_max_age() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Broadcast as-of lookup record is too old for primary "
            r"partition=\('A',\), time=2025-01-01T05:00:00\+00:00"
        ),
    ):
        list(
            broadcast_as_of_stream(
                iter([_primary("A", 1, "primary", hour=5)]),
                iter([_lookup(1, "lookup", hour=2)]),
                partition_by=("id_",),
                max_age=timedelta(hours=2),
            )
        )


def test_broadcast_as_of_yields_none_when_lookup_is_too_old_and_optional() -> None:
    rows = list(
        broadcast_as_of_stream(
            iter([_primary("A", 1, "primary", hour=5)]),
            iter([_lookup(1, "lookup", hour=2)]),
            partition_by=("id_",),
            max_age=timedelta(hours=2),
            require_match=False,
        )
    )

    assert rows[0][0].value == "primary"
    assert rows[0][1] is None


def test_broadcast_as_of_rejects_primary_before_first_lookup() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Broadcast as-of lookup has no record at or before primary "
            r"partition=\('A',\), time=2025-01-01T00:00:00\+00:00"
        ),
    ):
        list(
            broadcast_as_of_stream(
                iter([_primary("A", 1, "primary")]),
                iter([_lookup(2, "future")]),
                partition_by=("id_",),
            )
        )


def test_broadcast_as_of_yields_none_before_first_lookup_when_optional() -> None:
    rows = list(
        broadcast_as_of_stream(
            iter(
                [
                    _primary("A", 1, "before"),
                    _primary("A", 2, "exact"),
                ]
            ),
            iter([_lookup(2, "lookup")]),
            partition_by=("id_",),
            require_match=False,
        )
    )

    assert rows[0][1] is None
    assert rows[1][1] is not None
    assert rows[1][1].value == "lookup"


@pytest.mark.parametrize("max_age", [timedelta(0), timedelta(microseconds=-1)])
def test_broadcast_as_of_requires_positive_max_age(max_age: timedelta) -> None:
    with pytest.raises(ValueError, match="max_age must be a positive timedelta"):
        list(
            broadcast_as_of_stream(
                iter(()),
                iter(()),
                partition_by=("id_",),
                max_age=max_age,
            )
        )


def test_broadcast_as_of_requires_timedelta_max_age() -> None:
    with pytest.raises(TypeError, match="max_age must be a timedelta or None"):
        list(
            broadcast_as_of_stream(
                iter(()),
                iter(()),
                partition_by=("id_",),
                max_age=1,  # type: ignore[arg-type]
            )
        )


def test_broadcast_as_of_requires_boolean_require_match() -> None:
    with pytest.raises(TypeError, match="require_match must be a boolean"):
        list(
            broadcast_as_of_stream(
                iter(()),
                iter(()),
                partition_by=("id_",),
                require_match=1,  # type: ignore[arg-type]
            )
        )


def test_broadcast_as_of_rejects_duplicate_lookup_times() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"Broadcast as-of lookup has duplicate time "
            r"2025-01-01T00:00:00\+00:00"
        ),
    ):
        list(
            broadcast_as_of_stream(
                iter([_primary("A", 1, "primary")]),
                iter([_lookup(1, "first"), _lookup(1, "duplicate")]),
                partition_by=("id_",),
            )
        )


def test_broadcast_as_of_rejects_unordered_lookup_times() -> None:
    with pytest.raises(ValueError, match="Broadcast as-of lookup is not ordered"):
        list(
            broadcast_as_of_stream(
                iter([_primary("A", 1, "primary")]),
                iter([_lookup(2, "second"), _lookup(1, "first")]),
                partition_by=("id_",),
            )
        )


def test_broadcast_as_of_validates_lookup_when_primary_is_empty() -> None:
    with pytest.raises(ValueError, match="Broadcast as-of lookup is not ordered"):
        list(
            broadcast_as_of_stream(
                iter(()),
                iter([_lookup(2, "second"), _lookup(1, "first")]),
                partition_by=("id_",),
            )
        )


def test_broadcast_as_of_rejects_duplicate_primary_canonical_keys() -> None:
    with pytest.raises(ValueError) as error:
        list(
            broadcast_as_of_stream(
                iter(
                    [
                        _primary("A", 1, "first"),
                        _primary("A", 1, "duplicate"),
                    ]
                ),
                iter([_lookup(1, "lookup")]),
                partition_by=("id_",),
            )
        )

    assert "duplicate canonical key" in str(error.value)
    assert "partition=('A',)" in str(error.value)
    assert "2025-01-01T00:00:00+00:00" in str(error.value)


@pytest.mark.parametrize(
    "primary",
    [
        [
            _primary("A", 2, "second"),
            _primary("A", 1, "first"),
        ],
        [
            _primary("B", 1, "B"),
            _primary("A", 1, "A"),
        ],
    ],
)
def test_broadcast_as_of_rejects_unordered_primary_canonical_keys(
    primary: list[_PrimaryRecord],
) -> None:
    with pytest.raises(ValueError, match="Broadcast as-of primary is not ordered"):
        list(
            broadcast_as_of_stream(
                iter(primary),
                iter([_lookup(1, "first"), _lookup(2, "second")]),
                partition_by=("id_",),
            )
        )


def test_broadcast_as_of_requires_every_primary_partition_field() -> None:
    with pytest.raises(KeyError, match="Partition field 'region' not found"):
        list(
            broadcast_as_of_stream(
                iter([_primary("A", 1, "primary")]),
                iter([_lookup(1, "lookup")]),
                partition_by=("id_", "region"),
            )
        )


def test_broadcast_as_of_closes_both_inputs_once_after_success() -> None:
    primary = _TrackedIterator([_primary("A", 1, "primary")])
    lookup = _TrackedIterator([_lookup(1, "lookup")])

    assert (
        len(
            list(
                broadcast_as_of_stream(
                    primary,
                    lookup,
                    partition_by=("id_",),
                )
            )
        )
        == 1
    )
    assert primary.close_calls == 1
    assert lookup.close_calls == 1


def test_broadcast_as_of_preserves_processing_error_when_cleanup_fails() -> None:
    primary = _TrackedIterator(
        [_primary("A", 1, "primary")],
        close_error=RuntimeError("primary close failed"),
    )
    lookup = _TrackedIterator(
        [_lookup(2, "future")],
        close_error=RuntimeError("lookup close failed"),
    )

    with pytest.raises(ValueError, match="has no record at or before primary"):
        list(
            broadcast_as_of_stream(
                primary,
                lookup,
                partition_by=("id_",),
            )
        )

    assert primary.close_calls == 1
    assert lookup.close_calls == 1


def test_broadcast_as_of_closes_both_inputs_when_consumer_stops() -> None:
    primary = _TrackedIterator(
        [
            _primary("A", 1, "first"),
            _primary("A", 2, "second"),
        ]
    )
    lookup = _TrackedIterator([_lookup(1, "first"), _lookup(2, "second")])
    rows = broadcast_as_of_stream(
        primary,
        lookup,
        partition_by=("id_",),
    )

    next(rows)
    rows.close()

    assert primary.close_calls == 1
    assert lookup.close_calls == 1
