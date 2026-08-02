from pathlib import Path

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
from jerrythomas.runtime import (
    AlignedRuntimeStream,
    AsOfRuntimeStream,
    BroadcastAsOfRuntimeStream,
    BroadcastRuntimeStream,
    CrossSectionRuntimeStream,
    DerivedRuntimeStream,
    Runtime,
    RuntimeStream,
    SourceRuntimeStream,
)
from jerrythomas.services.definitions import ProjectDefinition
from jerrythomas.services.streams.combine import build_combine_stage
from jerrythomas.services.streams.source import build_mapper, build_source
from jerrythomas.services.streams.validation import stream_partition_by
from jerrythomas.utils.time import parse_timecode


def _compile_source_stream(
    config: SourceStreamConfig,
    source_config: SourceConfig,
    project_yaml: Path,
) -> SourceRuntimeStream:
    return SourceRuntimeStream(
        source=build_source(source_config, project_yaml),
        mapper=build_mapper(config.map),
        preprocess=tuple(config.preprocess),
        partition_by=config.partition_by,
        presorted=config.ordered_by is not None,
        transforms=tuple(config.transforms),
    )


def _compile_derived_stream(
    config: DerivedStreamConfig,
    stream_configs: dict[str, StreamConfig],
) -> DerivedRuntimeStream:
    return DerivedRuntimeStream(
        input_stream=config.from_.stream,
        partition_by=stream_partition_by(stream_configs, config.id),
        transforms=tuple(config.transforms),
    )


def _compile_cross_section_stream(
    config: CrossSectionStreamConfig,
    stream_configs: dict[str, StreamConfig],
) -> CrossSectionRuntimeStream:
    return CrossSectionRuntimeStream(
        input_stream=config.from_.stream,
        partition_by=stream_partition_by(stream_configs, config.id),
        cross_section=tuple(config.cross_section),
        transforms=tuple(config.transforms),
    )


def _compile_broadcast_stream(
    config: BroadcastStreamConfig,
    stream_configs: dict[str, StreamConfig],
) -> BroadcastRuntimeStream:
    partition_by = stream_partition_by(stream_configs, config.id)
    return BroadcastRuntimeStream(
        input_stream=config.from_.stream,
        broadcast_stream=config.from_.broadcast,
        combine=build_combine_stage(config, partition_by),
        partition_by=partition_by,
        transforms=tuple(config.transforms),
    )


def _compile_as_of_stream(
    config: AsOfStreamConfig,
    stream_configs: dict[str, StreamConfig],
) -> AsOfRuntimeStream:
    partition_by = stream_partition_by(stream_configs, config.id)
    return AsOfRuntimeStream(
        input_stream=config.from_.stream,
        lookup_stream=config.from_.as_of,
        combine=build_combine_stage(config, partition_by),
        partition_by=partition_by,
        max_age=None if config.max_age is None else parse_timecode(config.max_age),
        require_match=config.require_match,
        transforms=tuple(config.transforms),
    )


def _compile_broadcast_as_of_stream(
    config: BroadcastAsOfStreamConfig,
    stream_configs: dict[str, StreamConfig],
) -> BroadcastAsOfRuntimeStream:
    partition_by = stream_partition_by(stream_configs, config.id)
    return BroadcastAsOfRuntimeStream(
        input_stream=config.from_.stream,
        lookup_stream=config.from_.broadcast_as_of,
        combine=build_combine_stage(config, partition_by),
        partition_by=partition_by,
        max_age=None if config.max_age is None else parse_timecode(config.max_age),
        require_match=config.require_match,
        transforms=tuple(config.transforms),
    )


def _compile_aligned_stream(
    config: AlignedStreamConfig,
    stream_configs: dict[str, StreamConfig],
) -> AlignedRuntimeStream:
    partition_by = stream_partition_by(stream_configs, config.id)
    return AlignedRuntimeStream(
        inputs=config.from_.align,
        combine=build_combine_stage(config, partition_by),
        partition_by=partition_by,
        transforms=tuple(config.transforms),
    )


def compile_runtime(definition: ProjectDefinition) -> Runtime:
    stream_configs = definition.streams.streams
    runtime_streams: dict[str, RuntimeStream] = {}
    for stream_id, config in stream_configs.items():
        if isinstance(config, SourceStreamConfig):
            runtime_streams[stream_id] = _compile_source_stream(
                config,
                definition.streams.sources[config.from_.source],
                definition.project.path,
            )
        elif isinstance(config, DerivedStreamConfig):
            runtime_streams[stream_id] = _compile_derived_stream(
                config,
                stream_configs,
            )
        elif isinstance(config, CrossSectionStreamConfig):
            runtime_streams[stream_id] = _compile_cross_section_stream(
                config,
                stream_configs,
            )
        elif isinstance(config, BroadcastStreamConfig):
            runtime_streams[stream_id] = _compile_broadcast_stream(
                config,
                stream_configs,
            )
        elif isinstance(config, AsOfStreamConfig):
            runtime_streams[stream_id] = _compile_as_of_stream(
                config,
                stream_configs,
            )
        elif isinstance(config, BroadcastAsOfStreamConfig):
            runtime_streams[stream_id] = _compile_broadcast_as_of_stream(
                config,
                stream_configs,
            )
        elif isinstance(config, AlignedStreamConfig):
            runtime_streams[stream_id] = _compile_aligned_stream(
                config,
                stream_configs,
            )
        else:
            raise TypeError(f"Unsupported stream config: {type(config).__name__}")

    return Runtime(
        project_yaml=definition.project.path,
        artifacts_root=definition.project.artifacts_root,
        dataset=definition.dataset.model_copy(deep=True),
        streams=runtime_streams,
    )
