from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

from jerrythomas.artifacts.registry import ArtifactRegistry
from jerrythomas.config.dataset.dataset import DatasetConfig
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
    StreamsConfig,
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
    RuntimeSnapshot,
    RuntimeStream,
    SourceRuntimeStream,
)
from jerrythomas.services.definitions import ProjectDefinition
from jerrythomas.services.streams.combine import build_combine_stage
from jerrythomas.services.streams.source import build_mapper, build_source
from jerrythomas.services.streams.validation import (
    stream_dependency_closure,
    stream_partition_by,
)
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
        presorted=config.presorted,
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


def _compile_runtime(
    project_yaml: Path,
    artifacts_root: Path,
    dataset: DatasetConfig | None,
    configuration: StreamsConfig,
    *,
    dataset_id: str | None = None,
    artifact_aliases: dict[str, str] | None = None,
) -> Runtime:
    blueprint = configuration.model_copy(deep=True)
    streams = blueprint.model_copy(deep=True)
    stream_configs = streams.streams
    runtime_streams: dict[str, RuntimeStream] = {}
    for stream_id, config in stream_configs.items():
        if isinstance(config, SourceStreamConfig):
            runtime_streams[stream_id] = _compile_source_stream(
                config,
                streams.sources[config.from_.source],
                project_yaml,
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
        project_yaml=project_yaml,
        artifacts_root=artifacts_root,
        dataset=dataset.model_copy(deep=True) if dataset is not None else None,
        dataset_id=dataset_id,
        artifact_aliases=dict(artifact_aliases or {}),
        streams=runtime_streams,
        _stream_configs=blueprint,
    )


def compile_runtime(
    definition: ProjectDefinition, dataset_id: str | None = None
) -> Runtime:
    return _compile_runtime(
        definition.project.path,
        definition.project.artifacts_root,
        definition.require_dataset(dataset_id) if dataset_id is not None else None,
        definition.streams,
        dataset_id=dataset_id,
        artifact_aliases=(
            dict(definition.artifact_graph.dataset_artifact_keys(dataset_id))
            if dataset_id is not None
            else {}
        ),
    )


def snapshot_runtime(runtime: Runtime, stream_ids: Iterable[str]) -> RuntimeSnapshot:
    """Capture selected streams without carrying live loaders into another process."""
    if runtime._stream_configs is None:
        raise ValueError(
            "Parallel stream execution requires a runtime created by compile_runtime; "
            "use compile_runtime or execution.workers=1 for manually assembled runtimes."
        )
    streams = runtime._stream_configs.model_copy(deep=True)
    selected = stream_dependency_closure(streams.streams, stream_ids)
    streams.streams = {
        stream_id: config
        for stream_id, config in streams.streams.items()
        if stream_id in selected
    }
    source_ids = {
        config.from_.source
        for config in streams.streams.values()
        if isinstance(config, SourceStreamConfig)
    }
    streams.sources = {
        source_id: config
        for source_id, config in streams.sources.items()
        if source_id in source_ids
    }
    return RuntimeSnapshot(
        project_yaml=runtime.project_yaml,
        artifacts_root=runtime.artifacts_root,
        dataset=(
            runtime.dataset.model_copy(deep=True)
            if runtime.dataset is not None
            else None
        ),
        dataset_id=runtime.dataset_id,
        artifact_aliases=dict(runtime.artifact_aliases),
        execution=runtime.execution.model_copy(deep=True),
        streams=streams,
        artifact_registry_root=runtime.artifacts.root,
        artifact_registrations=runtime.artifacts.registrations(),
        heartbeat_interval_seconds=runtime.heartbeat_interval_seconds,
        observe_node_events=runtime.observe_node_events,
    )


def compile_runtime_snapshot(snapshot: RuntimeSnapshot) -> Runtime:
    """Rebuild worker-local state from resolved values, without reading project files."""
    runtime = _compile_runtime(
        snapshot.project_yaml,
        snapshot.artifacts_root,
        snapshot.dataset,
        snapshot.streams,
        dataset_id=snapshot.dataset_id,
        artifact_aliases=snapshot.artifact_aliases,
    )
    runtime.execution = snapshot.execution.model_copy(update={"workers": 1}, deep=True)
    runtime.heartbeat_interval_seconds = snapshot.heartbeat_interval_seconds
    runtime.observe_node_events = snapshot.observe_node_events
    runtime.artifacts = ArtifactRegistry(
        snapshot.artifact_registry_root, aliases=runtime.artifact_aliases
    )
    for key, record in snapshot.artifact_registrations.items():
        runtime.artifacts.register(
            key, record.relative_path, deepcopy(dict(record.meta))
        )
    return runtime
