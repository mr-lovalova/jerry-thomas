from collections.abc import Iterator
from typing import Protocol, TypeVar


TRecord = TypeVar("TRecord", covariant=True)


def canonical_record_order(
    partition_by: tuple[str, ...],
) -> tuple[str, ...]:
    return (*partition_by, "time")


class RecordStream(Protocol[TRecord]):
    def stream(self) -> Iterator[TRecord]:
        pass
