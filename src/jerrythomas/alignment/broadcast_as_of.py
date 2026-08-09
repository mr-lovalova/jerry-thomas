from bisect import bisect_right
from collections.abc import Generator, Iterator
from datetime import datetime, timedelta

from jerrythomas.alignment._lifecycle import closing_alignment_inputs
from jerrythomas.domain.record import TemporalRecord
from jerrythomas.transforms.utils import partition_key


_CanonicalKey = tuple[tuple[object, ...], datetime]


def broadcast_as_of_stream(
    primary: Iterator[TemporalRecord],
    lookup: Iterator[TemporalRecord],
    partition_by: tuple[str, ...],
    max_age: timedelta | None = None,
    require_match: bool = True,
) -> Generator[tuple[TemporalRecord, TemporalRecord | None], None, None]:
    """Attach the latest eligible global lookup record to each primary record.

    Lookup records at the primary time are eligible. When max_age is set, a
    lookup exactly max_age before the primary is also eligible.

    The lookup is fully indexed before primary records are consumed. This uses
    O(number of lookup records) memory so the same global history can be reused
    when time restarts for each primary partition.
    """
    lookup_times: list[datetime] = []
    lookup_records: list[TemporalRecord] = []

    with closing_alignment_inputs((primary, lookup)):
        if max_age is not None and not isinstance(max_age, timedelta):
            raise TypeError("Broadcast as-of max_age must be a timedelta or None")
        if max_age is not None and max_age < timedelta(0):
            raise ValueError("Broadcast as-of max_age must be a non-negative timedelta")
        if type(require_match) is not bool:
            raise TypeError("Broadcast as-of require_match must be a boolean")

        previous_lookup_time: datetime | None = None
        for lookup_record in lookup:
            lookup_time = lookup_record.time
            if lookup_time == previous_lookup_time:
                raise ValueError(
                    "Broadcast as-of lookup has duplicate time "
                    f"{lookup_time.isoformat()}"
                )
            if previous_lookup_time is not None and lookup_time < previous_lookup_time:
                raise ValueError(
                    "Broadcast as-of lookup is not ordered: "
                    f"time {lookup_time.isoformat()} follows "
                    f"{previous_lookup_time.isoformat()}"
                )
            lookup_times.append(lookup_time)
            lookup_records.append(lookup_record)
            previous_lookup_time = lookup_time

        previous_primary_key: _CanonicalKey | None = None
        for primary_record in primary:
            partition = partition_key(primary_record, partition_by)
            key = partition, primary_record.time
            if key == previous_primary_key:
                raise ValueError(
                    "Broadcast as-of primary has duplicate canonical key "
                    f"partition={partition!r}, "
                    f"time={primary_record.time.isoformat()}"
                )
            if previous_primary_key is not None and key < previous_primary_key:
                raise ValueError(
                    "Broadcast as-of primary is not ordered: "
                    f"key {key!r} follows {previous_primary_key!r}"
                )
            previous_primary_key = key

            lookup_index = bisect_right(lookup_times, primary_record.time) - 1
            if lookup_index < 0:
                if require_match:
                    raise ValueError(
                        "Broadcast as-of lookup has no record at or before primary "
                        f"partition={partition!r}, "
                        f"time={primary_record.time.isoformat()}"
                    )
                yield primary_record, None
                continue

            lookup_record = lookup_records[lookup_index]
            age = primary_record.time - lookup_record.time
            if max_age is not None and age > max_age:
                if require_match:
                    raise ValueError(
                        "Broadcast as-of lookup record is too old for primary "
                        f"partition={partition!r}, "
                        f"time={primary_record.time.isoformat()}: "
                        f"lookup_time={lookup_record.time.isoformat()}, "
                        f"max_age={max_age}"
                    )
                yield primary_record, None
                continue

            yield primary_record, lookup_record
