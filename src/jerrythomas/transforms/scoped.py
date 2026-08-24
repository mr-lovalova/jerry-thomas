from collections.abc import Iterator, Mapping
from typing import Any

from jerrythomas.transforms.utils import adjacent_partitions


class PartitionScopedTransform:
    """Base class for stateful plugin stream transforms.

    Subclasses implement ``process_partition`` and receive each partition's
    records in canonical ``(partition, time)`` order. State resets between
    partitions, including when one partition's records reappear non-adjacently
    after another partition, matching built-in stateful transforms such as
    ``lag``. Yield zero or more output records per input record; yielding
    nothing drops records.
    """

    def __init__(
        self,
        args: Mapping[str, Any],
        partition_by: tuple[str, ...],
    ) -> None:
        self.args = dict(args)
        self.partition_by = tuple(partition_by)

    def process_partition(self, records: Iterator[Any]) -> Iterator[Any]:
        raise NotImplementedError(
            f"{type(self).__name__} must implement process_partition"
        )

    def apply(self, stream: Iterator[Any]) -> Iterator[Any]:
        for _, records in adjacent_partitions(stream, self.partition_by):
            yield from self.process_partition(records)
