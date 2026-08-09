from collections.abc import Iterator, Sequence
from typing import Protocol, TypeVar


TRecord = TypeVar("TRecord", covariant=True)


def canonical_record_order(
    partition_by: tuple[str, ...],
) -> tuple[str, ...]:
    return (*partition_by, "time")


def require_consistent_partition_types(
    partition_by: Sequence[str],
    values: Sequence[object],
    expected_types: dict[str, type],
    position: int,
) -> None:
    """Require each partition field to retain one exact value type."""
    for field, value in zip(partition_by, values, strict=True):
        value_type = type(value)
        expected_type = expected_types.setdefault(field, value_type)
        if value_type is not expected_type:
            raise TypeError(
                f"Record {position} changes partition field {field!r} from "
                f"{expected_type.__name__} to {value_type.__name__}; partition "
                "fields must use one exact type."
            )


class RecordStream(Protocol[TRecord]):
    def stream(self) -> Iterator[TRecord]:
        pass
