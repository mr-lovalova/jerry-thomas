from collections.abc import Iterator
from itertools import groupby

from jerrythomas.domain.record import TemporalRecord, public_record_fields
from jerrythomas.transforms.utils import (
    clone_record,
    partition_key,
    record_establishes_domain,
    set_record_domain_anchor,
)


class DedupeTransform:
    """Drop exact duplicates within each canonical record key."""

    def __init__(self, partition_fields: tuple[str, ...]) -> None:
        self.partition_fields = partition_fields

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        groups = groupby(
            stream,
            key=lambda record: (
                partition_key(record, self.partition_fields),
                record.time,
            ),
        )
        for _, records in groups:
            unique: list[tuple[dict[str, object], TemporalRecord]] = []
            for record in records:
                fields = public_record_fields(record)
                establishes_domain = record_establishes_domain(record)
                for index, (existing_fields, retained) in enumerate(unique):
                    if fields == existing_fields:
                        if establishes_domain and not record_establishes_domain(
                            retained
                        ):
                            retained = clone_record(retained)
                            set_record_domain_anchor(retained, True)
                            unique[index] = (existing_fields, retained)
                        break
                else:
                    unique.append((fields, record))
            for _, record in unique:
                yield record
