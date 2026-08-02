from collections.abc import Generator, Iterator
from datetime import datetime

from jerrythomas.alignment._lifecycle import closing_alignment_inputs
from jerrythomas.domain.record import TemporalRecord
from jerrythomas.transforms.utils import partition_key


_CanonicalKey = tuple[tuple[object, ...], datetime]


def broadcast_stream(
    primary: Iterator[TemporalRecord],
    broadcast: Iterator[TemporalRecord],
    partition_by: tuple[str, ...],
) -> Generator[tuple[TemporalRecord, TemporalRecord], None, None]:
    """Pair each partitioned primary record with broadcast data at the same time."""
    broadcast_by_time: dict[datetime, TemporalRecord] = {}

    with closing_alignment_inputs((primary, broadcast)):
        previous_broadcast_time: datetime | None = None
        for broadcast_record in broadcast:
            broadcast_time = broadcast_record.time
            if broadcast_time == previous_broadcast_time:
                raise ValueError(
                    f"Broadcast input has duplicate time {broadcast_time.isoformat()}"
                )
            if (
                previous_broadcast_time is not None
                and broadcast_time < previous_broadcast_time
            ):
                raise ValueError(
                    "Broadcast input is not ordered: "
                    f"time {broadcast_time.isoformat()} follows "
                    f"{previous_broadcast_time.isoformat()}"
                )
            broadcast_by_time[broadcast_time] = broadcast_record
            previous_broadcast_time = broadcast_time

        previous_primary_key: _CanonicalKey | None = None
        for primary_record in primary:
            partition = partition_key(primary_record, partition_by)
            key = partition, primary_record.time
            if key == previous_primary_key:
                raise ValueError(
                    "Broadcast primary has duplicate canonical key "
                    f"partition={partition!r}, time={primary_record.time.isoformat()}"
                )
            if previous_primary_key is not None and key < previous_primary_key:
                raise ValueError(
                    "Broadcast primary is not ordered: "
                    f"key {key!r} follows {previous_primary_key!r}"
                )
            previous_primary_key = key

            try:
                broadcast_record = broadcast_by_time[primary_record.time]
            except KeyError as exc:
                raise ValueError(
                    "Broadcast input has no record for primary "
                    f"partition={partition!r}, time={primary_record.time.isoformat()}"
                ) from exc

            yield primary_record, broadcast_record
