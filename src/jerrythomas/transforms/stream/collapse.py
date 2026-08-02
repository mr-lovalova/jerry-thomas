from collections.abc import Iterator
from typing import Literal

from jerrythomas.domain.record import TemporalRecord
from jerrythomas.transforms.utils import partition_key


class CollapseTransform:
    """Keep one adjacent record for each partition and timestamp."""

    def __init__(
        self,
        partition_fields: tuple[str, ...],
        keep: Literal["first", "last"],
    ) -> None:
        self.partition_fields = partition_fields
        self.keep = keep

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        selected: TemporalRecord | None = None
        selected_key: tuple[tuple, object] | None = None

        for record in stream:
            key = (partition_key(record, self.partition_fields), record.time)
            if selected is None:
                selected = record
                selected_key = key
                continue
            if key != selected_key:
                yield selected
                selected = record
                selected_key = key
            elif self.keep == "last":
                selected = record

        if selected is not None:
            yield selected
