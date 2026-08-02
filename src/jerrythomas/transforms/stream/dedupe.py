from collections.abc import Iterator

from jerrythomas.domain.record import TemporalRecord


class DedupeTransform:
    """Drop consecutive identical records (timestamp + payload)."""

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        last: TemporalRecord | None = None
        for record in stream:
            if last is not None and record == last:
                continue
            last = record
            yield record
