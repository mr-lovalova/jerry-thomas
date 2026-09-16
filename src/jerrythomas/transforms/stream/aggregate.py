from collections.abc import Iterator
from itertools import chain, groupby

from jerrythomas.domain.record import TemporalRecord
from jerrythomas.transforms.numeric import ExactNumericSum
from jerrythomas.transforms.utils import (
    clone_record,
    get_field,
    partition_key,
    record_establishes_domain,
    set_record_domain_anchor,
)


class AggregateSumTransform:
    """Sum one field for each adjacent canonical record key."""

    def __init__(
        self,
        field: str,
        partition_fields: tuple[str, ...],
        count_to: str | None = None,
    ) -> None:
        if count_to == field:
            raise ValueError("aggregate_sum count_to must differ from field")
        self.field = field
        self.partition_fields = partition_fields
        self.count_to = count_to

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        groups = groupby(
            stream,
            key=lambda record: (
                partition_key(record, self.partition_fields),
                record.time,
            ),
        )
        for _, records in groups:
            first = next(records)
            total = ExactNumericSum(self.field)
            count = 0
            establishes_domain = False
            for record in chain((first,), records):
                total.add(get_field(record, self.field))
                count += 1
                establishes_domain = (
                    record_establishes_domain(record) or establishes_domain
                )

            updates: dict[str, object] = {self.field: total.result()}
            if self.count_to is not None:
                updates[self.count_to] = count
            aggregate = clone_record(first, **updates)
            set_record_domain_anchor(aggregate, establishes_domain)
            yield aggregate
