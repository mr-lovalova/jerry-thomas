from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from jerrythomas.alignment.broadcast import broadcast_stream
from jerrythomas.domain.record import TemporalRecord


@dataclass
class _PrimaryRecord(TemporalRecord):
    id_: str
    value: str


@dataclass
class _BroadcastRecord(TemporalRecord):
    value: str


def _time(day: int) -> datetime:
    return datetime(2025, 1, day, tzinfo=UTC)


def _primary(id_: str, day: int, value: str) -> _PrimaryRecord:
    return _PrimaryRecord(time=_time(day), id_=id_, value=value)


def _broadcast(day: int, value: str) -> _BroadcastRecord:
    return _BroadcastRecord(time=_time(day), value=value)


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


def test_broadcast_stream_pairs_input_with_every_partition_at_exact_time() -> None:
    primary = [
        _primary("A", 1, "A-1"),
        _primary("A", 2, "A-2"),
        _primary("B", 1, "B-1"),
        _primary("B", 2, "B-2"),
    ]
    broadcast = [_broadcast(1, "day-1"), _broadcast(2, "day-2")]

    rows = list(broadcast_stream(iter(primary), iter(broadcast), ("id_",)))

    assert [(left.value, right.value) for left, right in rows] == [
        ("A-1", "day-1"),
        ("A-2", "day-2"),
        ("B-1", "day-1"),
        ("B-2", "day-2"),
    ]
    assert rows[0][1] is rows[2][1]
    assert rows[1][1] is rows[3][1]


def test_broadcast_stream_indexes_broadcast_input_before_primary() -> None:
    broadcast_exhausted = False

    def broadcast_records():
        nonlocal broadcast_exhausted
        try:
            yield _broadcast(1, "broadcast")
        finally:
            broadcast_exhausted = True

    def primary_records():
        assert broadcast_exhausted
        yield _primary("A", 1, "primary")

    rows = list(broadcast_stream(primary_records(), broadcast_records(), ("id_",)))

    assert len(rows) == 1


def test_broadcast_stream_ignores_unused_broadcast_times() -> None:
    rows = list(
        broadcast_stream(
            iter([_primary("A", 2, "primary")]),
            iter(
                [
                    _broadcast(1, "unused-before"),
                    _broadcast(2, "used"),
                    _broadcast(3, "unused-after"),
                ]
            ),
            ("id_",),
        )
    )

    assert [(left.value, right.value) for left, right in rows] == [("primary", "used")]


def test_broadcast_stream_does_not_fill_a_missing_broadcast_time() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Broadcast input has no record for primary "
            r"partition=\('A',\), time=2025-01-02T00:00:00\+00:00"
        ),
    ):
        list(
            broadcast_stream(
                iter([_primary("A", 2, "primary")]),
                iter([_broadcast(1, "previous")]),
                ("id_",),
            )
        )


def test_broadcast_stream_rejects_duplicate_broadcast_times() -> None:
    with pytest.raises(
        ValueError,
        match=r"Broadcast input has duplicate time 2025-01-01T00:00:00\+00:00",
    ):
        list(
            broadcast_stream(
                iter([_primary("A", 1, "primary")]),
                iter([_broadcast(1, "first"), _broadcast(1, "duplicate")]),
                ("id_",),
            )
        )


def test_broadcast_stream_rejects_unordered_broadcast_times() -> None:
    with pytest.raises(ValueError, match="Broadcast input is not ordered"):
        list(
            broadcast_stream(
                iter([_primary("A", 1, "primary")]),
                iter([_broadcast(2, "second"), _broadcast(1, "first")]),
                ("id_",),
            )
        )


def test_broadcast_stream_validates_broadcast_when_primary_is_empty() -> None:
    with pytest.raises(ValueError, match="Broadcast input is not ordered"):
        list(
            broadcast_stream(
                iter(()),
                iter([_broadcast(2, "second"), _broadcast(1, "first")]),
                ("id_",),
            )
        )


def test_broadcast_stream_rejects_duplicate_primary_canonical_keys() -> None:
    with pytest.raises(ValueError) as error:
        list(
            broadcast_stream(
                iter(
                    [
                        _primary("A", 1, "first"),
                        _primary("A", 1, "duplicate"),
                    ]
                ),
                iter([_broadcast(1, "broadcast")]),
                ("id_",),
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
def test_broadcast_stream_rejects_unordered_primary_canonical_keys(
    primary: list[_PrimaryRecord],
) -> None:
    with pytest.raises(ValueError, match="Broadcast primary is not ordered"):
        list(
            broadcast_stream(
                iter(primary),
                iter([_broadcast(1, "first"), _broadcast(2, "second")]),
                ("id_",),
            )
        )


def test_broadcast_stream_requires_every_primary_partition_field() -> None:
    with pytest.raises(KeyError, match="Partition field 'region' not found"):
        list(
            broadcast_stream(
                iter([_primary("A", 1, "primary")]),
                iter([_broadcast(1, "broadcast")]),
                ("id_", "region"),
            )
        )


def test_broadcast_stream_closes_both_inputs_once_after_success() -> None:
    primary = _TrackedIterator([_primary("A", 1, "primary")])
    broadcast = _TrackedIterator([_broadcast(1, "broadcast")])

    assert len(list(broadcast_stream(primary, broadcast, ("id_",)))) == 1
    assert primary.close_calls == 1
    assert broadcast.close_calls == 1


def test_broadcast_stream_closes_both_inputs_once_after_error() -> None:
    primary = _TrackedIterator([_primary("A", 1, "primary")])
    broadcast = _TrackedIterator([_broadcast(2, "second"), _broadcast(1, "first")])

    with pytest.raises(ValueError, match="Broadcast input is not ordered"):
        list(broadcast_stream(primary, broadcast, ("id_",)))

    assert primary.close_calls == 1
    assert broadcast.close_calls == 1


def test_broadcast_stream_preserves_processing_error_when_cleanup_fails() -> None:
    primary = _TrackedIterator(
        [_primary("A", 2, "primary")],
        close_error=RuntimeError("primary close failed"),
    )
    broadcast = _TrackedIterator(
        [_broadcast(1, "broadcast")],
        close_error=RuntimeError("broadcast close failed"),
    )

    with pytest.raises(ValueError, match="has no record for primary"):
        list(broadcast_stream(primary, broadcast, ("id_",)))

    assert primary.close_calls == 1
    assert broadcast.close_calls == 1


def test_broadcast_stream_closes_both_inputs_once_when_consumer_stops() -> None:
    primary = _TrackedIterator([_primary("A", 1, "first"), _primary("A", 2, "second")])
    broadcast = _TrackedIterator([_broadcast(1, "first"), _broadcast(2, "second")])
    rows = broadcast_stream(primary, broadcast, ("id_",))

    next(rows)
    rows.close()

    assert primary.close_calls == 1
    assert broadcast.close_calls == 1
