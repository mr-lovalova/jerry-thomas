from collections.abc import Callable, Iterable, Iterator
from typing import Any

from datapipeline.config.streams import (
    AlignedStreamConfig,
    AsOfStreamConfig,
    BroadcastAsOfStreamConfig,
    BroadcastStreamConfig,
)
from datapipeline.plugins import COMBINERS_EP
from datapipeline.transforms.utils import (
    partition_key,
    record_establishes_domain,
    set_record_domain_anchor,
)
from datapipeline.utils.load import load_ep
from datapipeline.utils.placeholders import normalize_args


def build_combine_stage(
    config: (
        AlignedStreamConfig
        | BroadcastStreamConfig
        | AsOfStreamConfig
        | BroadcastAsOfStreamConfig
    ),
    partition_by: tuple[str, ...],
) -> Callable[[Iterator[tuple[Any, ...]]], Iterable[Any]]:
    combine = load_ep(COMBINERS_EP, config.combine.entrypoint)
    args = normalize_args(config.combine.args)

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

            if (
                type(actual_time) is not type(expected_time)
                or actual_time != expected_time
            ):
                raise ValueError(
                    f"Stream '{config.id}' combine must preserve input time: "
                    f"expected {expected_time!r}, got {actual_time!r}."
                )
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
