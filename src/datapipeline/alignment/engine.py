from collections.abc import Generator, Iterator, Sequence
from datetime import datetime

from datapipeline.alignment._lifecycle import closing_alignment_inputs
from datapipeline.domain.record import TemporalRecord
from datapipeline.transforms.utils import partition_key


_CanonicalKey = tuple[tuple[object, ...], datetime]


def align_streams(
    inputs: Sequence[tuple[str, Iterator[TemporalRecord]]],
    partition_by: tuple[str, ...],
) -> Generator[tuple[TemporalRecord, ...], None, None]:
    stream_ids = [stream_id for stream_id, _ in inputs]
    streams = [records for _, records in inputs]
    current_records: list[TemporalRecord] = []
    current_keys: list[_CanonicalKey] = []
    with closing_alignment_inputs(streams):
        if len(streams) < 2:
            raise ValueError("Alignment requires at least two input streams")

        for records in streams:
            try:
                first = next(records)
            except StopIteration:
                return
            current_records.append(first)
            current_keys.append(_canonical_key(first, partition_by))

        expected_partition = current_keys[0][0]
        for index, (partition, _) in enumerate(current_keys[1:], start=1):
            for field, expected, value in zip(
                partition_by,
                expected_partition,
                partition,
                strict=True,
            ):
                if type(value) is not type(expected):
                    raise TypeError(
                        f"Alignment input {stream_ids[index]!r} partition field "
                        f"{field!r} uses {type(value).__name__}; input "
                        f"{stream_ids[0]!r} uses {type(expected).__name__}."
                    )

        def advance(index: int) -> bool:
            try:
                record = next(streams[index])
            except StopIteration:
                return False

            previous = current_keys[index]
            key = _canonical_key(record, partition_by)
            if key == previous:
                partition, time = key
                raise ValueError(
                    f"Alignment input {stream_ids[index]!r} has duplicate canonical key "
                    f"partition={partition!r}, time={time.isoformat()}"
                )
            if key < previous:
                raise ValueError(
                    f"Alignment input {stream_ids[index]!r} is not ordered: "
                    f"key {key!r} follows {previous!r}"
                )
            current_records[index] = record
            current_keys[index] = key
            return True

        while True:
            target = max(current_keys)
            for index in range(len(streams)):
                while current_keys[index] < target:
                    if not advance(index):
                        return
                    if current_keys[index] > target:
                        target = current_keys[index]

            if any(key != target for key in current_keys):
                continue

            yield tuple(current_records)
            for index in range(len(streams)):
                if not advance(index):
                    return


def _canonical_key(
    record: TemporalRecord,
    partition_by: tuple[str, ...],
) -> _CanonicalKey:
    return partition_key(record, partition_by), record.time
