from pathlib import Path

from jerrythomas.config.sources import SourceConfig
from jerrythomas.config.streams import (
    AlignJoin,
    AsOfJoin,
    BroadcastAsOfJoin,
    BroadcastJoin,
    CombinedStreamConfig,
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
    CombinedRuntimeStream,
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


def _compile_combined_stream(
    config: CombinedStreamConfig,
    stream_configs: dict[str, StreamConfig],
) -> CombinedRuntimeStream:
    partition_by = stream_partition_by(stream_configs, config.id)
    combine = build_combine_stage(config, partition_by)
    transforms = tuple(config.transforms)
    primary = config.from_.stream
    join = config.join

    if isinstance(join, AsOfJoin):
        return AsOfRuntimeStream(
            input_stream=primary,
            lookup_stream=join.lookup,
            combine=combine,
            partition_by=partition_by,
            max_age=None if join.max_age is None else parse_timecode(join.max_age),
            require_match=join.require_match,
            direction=join.direction,
            transforms=transforms,
        )
    if isinstance(join, BroadcastAsOfJoin):
        return BroadcastAsOfRuntimeStream(
            input_stream=primary,
            lookup_stream=join.lookup,
            combine=combine,
            partition_by=partition_by,
            max_age=None if join.max_age is None else parse_timecode(join.max_age),
            require_match=join.require_match,
            direction=join.direction,
            transforms=transforms,
        )
    if isinstance(join, BroadcastJoin):
        return BroadcastRuntimeStream(
            input_stream=primary,
            broadcast_stream=join.with_,
            combine=combine,
            partition_by=partition_by,
            transforms=transforms,
        )
    assert isinstance(join, AlignJoin)
    return AlignedRuntimeStream(
        inputs=(primary, *join.streams),
        combine=combine,
        partition_by=partition_by,
        transforms=transforms,
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
        elif isinstance(config, CombinedStreamConfig):
            runtime_streams[stream_id] = _compile_combined_stream(
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
