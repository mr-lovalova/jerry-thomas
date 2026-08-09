from collections.abc import Iterable

from jerrythomas.config.sources import SourceConfig
from jerrythomas.config.streams import (
    AlignedStreamConfig,
    AsOfStreamConfig,
    BroadcastAsOfStreamConfig,
    BroadcastStreamConfig,
    CrossSectionStreamConfig,
    DerivedStreamConfig,
    SourceStreamConfig,
    StreamConfig,
)
from jerrythomas.config.transforms import (
    AggregateSumConfig,
    DeriveConfig,
    EwmMeanConfig,
    FillConfig,
    FillMissingConfig,
    ForwardFillConfig,
    ForwardSumConfig,
    LagConfig,
    LeadConfig,
    Log1pConfig,
    LogConfig,
    RollingConfig,
    RollingQuantileConfig,
    RollingOlsConfig,
    RollingSlopeConfig,
)
from jerrythomas.domain.stream import canonical_record_order


def validate_stream_configs(
    sources: dict[str, SourceConfig],
    streams: dict[str, StreamConfig],
) -> None:
    for stream_id, stream in streams.items():
        if isinstance(stream, SourceStreamConfig):
            if stream.from_.source not in sources:
                raise ValueError(
                    f"Stream '{stream_id}' references unknown source "
                    f"'{stream.from_.source}'"
                )
            continue

        missing = [
            input_stream
            for input_stream in stream.input_streams()
            if input_stream not in streams
        ]
        if missing:
            raise ValueError(
                f"Stream '{stream_id}' references unknown stream(s): {missing}"
            )

    _validate_stream_cycles(streams)

    for stream_id, stream in streams.items():
        partition_by = stream_partition_by(streams, stream_id)
        if isinstance(stream, SourceStreamConfig) and stream.ordered_by is not None:
            canonical_order = canonical_record_order(partition_by)
            if stream.ordered_by != canonical_order:
                raise ValueError(
                    f"Stream '{stream_id}' ordered_by must be "
                    f"{list(canonical_order)!r}; got {list(stream.ordered_by)!r}"
                )

        canonical_fields = {"time", *partition_by}
        for operation in stream.transforms:
            if isinstance(operation, AggregateSumConfig):
                output_fields = (
                    (operation.field,)
                    if operation.count_to is None
                    else (operation.field, operation.count_to)
                )
            elif isinstance(
                operation,
                (
                    LagConfig,
                    LeadConfig,
                    EwmMeanConfig,
                    FillConfig,
                    FillMissingConfig,
                    ForwardFillConfig,
                    RollingConfig,
                    RollingQuantileConfig,
                ),
            ):
                output_fields = (
                    operation.field if operation.to is None else operation.to,
                )
            elif isinstance(
                operation,
                (
                    DeriveConfig,
                    ForwardSumConfig,
                    LogConfig,
                    Log1pConfig,
                    RollingOlsConfig,
                    RollingSlopeConfig,
                ),
            ):
                output_fields = (operation.to,)
            else:
                continue
            for output_field in output_fields:
                if output_field in canonical_fields:
                    raise ValueError(
                        f"Stream '{stream_id}' transform '{operation.operation}' "
                        f"cannot write canonical order field '{output_field}'"
                    )

        if isinstance(stream, CrossSectionStreamConfig):
            for cross_section_operation in stream.cross_section:
                if cross_section_operation.to in canonical_fields:
                    raise ValueError(
                        f"Stream '{stream_id}' cross-section operation "
                        f"'{cross_section_operation.operation}' cannot write canonical "
                        f"order field '{cross_section_operation.to}'"
                    )


def stream_partition_by(
    streams: dict[str, StreamConfig],
    stream_id: str,
) -> tuple[str, ...]:
    stream = streams[stream_id]
    if isinstance(stream, SourceStreamConfig):
        return stream.partition_by
    if isinstance(stream, DerivedStreamConfig):
        return stream_partition_by(streams, stream.from_.stream)
    if isinstance(stream, CrossSectionStreamConfig):
        partition_by = stream_partition_by(streams, stream.from_.stream)
        if not partition_by:
            raise ValueError(
                f"Cross-section stream '{stream_id}' input "
                f"'{stream.from_.stream}' must have a non-empty partition_by"
            )
        return partition_by
    if isinstance(stream, AsOfStreamConfig):
        primary_partition = stream_partition_by(streams, stream.from_.stream)
        lookup_partition = stream_partition_by(streams, stream.from_.as_of)
        if lookup_partition != primary_partition:
            raise ValueError(
                f"As-of stream '{stream_id}' lookup input '{stream.from_.as_of}' has "
                f"partition_by {list(lookup_partition)!r}; expected "
                f"{list(primary_partition)!r}"
            )
        return primary_partition
    if isinstance(stream, BroadcastStreamConfig):
        primary_partition = stream_partition_by(streams, stream.from_.stream)
        if not primary_partition:
            raise ValueError(
                f"Broadcast stream '{stream_id}' primary input "
                f"'{stream.from_.stream}' must have a non-empty partition_by"
            )

        broadcast_partition = stream_partition_by(
            streams,
            stream.from_.broadcast,
        )
        if broadcast_partition:
            raise ValueError(
                f"Broadcast stream '{stream_id}' broadcast input "
                f"'{stream.from_.broadcast}' must have an empty partition_by; "
                f"got {list(broadcast_partition)!r}"
            )
        return primary_partition
    if isinstance(stream, BroadcastAsOfStreamConfig):
        primary_partition = stream_partition_by(streams, stream.from_.stream)
        if not primary_partition:
            raise ValueError(
                f"Broadcast as-of stream '{stream_id}' primary input "
                f"'{stream.from_.stream}' must have a non-empty partition_by"
            )

        lookup_partition = stream_partition_by(
            streams,
            stream.from_.broadcast_as_of,
        )
        if lookup_partition:
            raise ValueError(
                f"Broadcast as-of stream '{stream_id}' lookup input "
                f"'{stream.from_.broadcast_as_of}' must have an empty partition_by; "
                f"got {list(lookup_partition)!r}"
            )
        return primary_partition

    if isinstance(stream, AlignedStreamConfig):
        input_partitions = [
            stream_partition_by(streams, input_stream)
            for input_stream in stream.input_streams()
        ]
        expected = input_partitions[0]
        for input_stream, partition_by in zip(
            stream.input_streams()[1:],
            input_partitions[1:],
            strict=True,
        ):
            if partition_by != expected:
                raise ValueError(
                    f"Aligned stream '{stream_id}' input '{input_stream}' has "
                    f"partition_by {list(partition_by)!r}; expected {list(expected)!r}"
                )
        return expected

    raise TypeError(f"Unsupported stream config: {type(stream).__name__}")


def stream_dependency_closure(
    streams: dict[str, StreamConfig],
    root_stream_ids: Iterable[str],
) -> frozenset[str]:
    visited: set[str] = set()

    def visit(stream_id: str) -> None:
        if stream_id in visited:
            return
        try:
            stream = streams[stream_id]
        except KeyError as exc:
            raise ValueError(
                f"Unknown stream '{stream_id}' in stream dependency graph."
            ) from exc
        visited.add(stream_id)
        for input_stream_id in stream.input_streams():
            visit(input_stream_id)

    for root_stream_id in root_stream_ids:
        visit(root_stream_id)
    return frozenset(visited)


def _validate_stream_cycles(streams: dict[str, StreamConfig]) -> None:
    visited: set[str] = set()
    visiting: list[str] = []

    def visit(stream_id: str) -> None:
        if stream_id in visited:
            return
        if stream_id in visiting:
            start = visiting.index(stream_id)
            cycle = [*visiting[start:], stream_id]
            raise ValueError("Stream dependency cycle: " + " -> ".join(cycle))

        visiting.append(stream_id)
        for input_stream in streams[stream_id].input_streams():
            visit(input_stream)
        visiting.pop()
        visited.add(stream_id)

    for stream_id in streams:
        visit(stream_id)
