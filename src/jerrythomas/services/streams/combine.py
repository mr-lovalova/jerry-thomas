from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
from typing import Any

from jerrythomas.config.streams import (
    AlignedStreamConfig,
    AsOfStreamConfig,
    BroadcastAsOfStreamConfig,
    BroadcastStreamConfig,
)
from jerrythomas.config.interpolation import normalize_interpolated_args
from jerrythomas.plugins import COMBINERS_EP, load_entrypoint
from jerrythomas.transforms.utils import (
    partition_key,
    record_establishes_domain,
    set_record_domain_anchor,
)


def build_combine_stage(
    config: (
        AlignedStreamConfig
        | BroadcastStreamConfig
        | AsOfStreamConfig
        | BroadcastAsOfStreamConfig
    ),
    partition_by: tuple[str, ...],
) -> Callable[[Iterator[tuple[Any, ...]]], Iterable[Any]]:
    combine = load_entrypoint(COMBINERS_EP, config.combine.entrypoint)
    args = normalize_interpolated_args(config.combine.args)

    def combine_records(rows: Iterator[tuple[Any, ...]]) -> Iterator[Any]:
        for records in rows:
            expected_time = records[0].time
            expected_partition = partition_key(records[0], partition_by)
            record = combine(*records, **args)
            if record is None:
                continue

            try:
                actual_time = record.time
                actual_partition = partition_key(record, partition_by)
            except (AttributeError, KeyError) as exc:
                raise ValueError(
                    f"Stream '{config.id}' combine output must contain "
                    "time and partition fields."
                ) from exc

            if not isinstance(actual_time, datetime):
                raise TypeError(
                    f"Stream '{config.id}' combine output time must be a "
                    f"datetime; got {type(actual_time).__name__}."
                )
            if actual_time.tzinfo is None or actual_time.utcoffset() is None:
                raise ValueError(
                    f"Stream '{config.id}' combine output time must be timezone-aware."
                )
            if actual_time.astimezone(timezone.utc) != expected_time:
                raise ValueError(
                    f"Stream '{config.id}' combine must preserve input time: "
                    f"expected {expected_time!r}, got {actual_time!r}."
                )
            record.time = expected_time
            for field, expected, actual in zip(
                partition_by,
                expected_partition,
                actual_partition,
                strict=True,
            ):
                if type(actual) is not type(expected) or actual != expected:
                    raise ValueError(
                        f"Stream '{config.id}' combine must preserve partition "
                        f"field {field!r}: expected {expected!r}, got {actual!r}."
                    )
            establishes_domain = (
                any(record_establishes_domain(item) for item in records)
                if isinstance(config, AlignedStreamConfig)
                else record_establishes_domain(records[0])
            )
            set_record_domain_anchor(record, establishes_domain)
            yield record

    return combine_records
