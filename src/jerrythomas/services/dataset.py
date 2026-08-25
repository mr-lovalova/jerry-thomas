import jerrythomas.config.transforms as transform_config
from jerrythomas.config.dataset.dataset import DatasetConfig
from jerrythomas.config.dataset.split import HashSplitConfig
from jerrythomas.config.streams import (
    AsOfJoin,
    BroadcastAsOfJoin,
    CombinedStreamConfig,
    CrossSectionStreamConfig,
    StreamsConfig,
)
from jerrythomas.io.yaml import YamlDocument
from jerrythomas.services.definitions import ProjectManifest
from jerrythomas.services.streams.validation import (
    stream_dependency_closure,
    stream_partition_by,
)
from jerrythomas.utils.time import parse_timecode


_CROSS_TIMESTAMP_TRANSFORMS = (
    transform_config.CustomTransformConfig,
    transform_config.EwmMeanConfig,
    transform_config.EwmStdConfig,
    transform_config.EnsureCadenceConfig,
    transform_config.EnsureScheduleConfig,
    transform_config.FillConfig,
    transform_config.ForwardFillConfig,
    transform_config.ForwardSumConfig,
    transform_config.LagConfig,
    transform_config.LeadConfig,
    transform_config.RollingConfig,
    transform_config.RollingOlsConfig,
    transform_config.RollingQuantileConfig,
    transform_config.RollingSlopeConfig,
)


def dataset_from_document(
    project: ProjectManifest,
    document: YamlDocument,
) -> DatasetConfig:
    return DatasetConfig.model_validate(project.resolve_config(document.data))


def validate_dataset_streams(
    dataset: DatasetConfig,
    streams: StreamsConfig,
) -> None:
    for config in dataset.series:
        if config.stream not in streams.streams:
            raise ValueError(
                f"Dataset series '{config.id}' references unknown stream "
                f"'{config.stream}'."
            )
        partition_by = stream_partition_by(streams.streams, config.stream)
        missing_sample_keys = [
            field for field in dataset.sample.keys if field not in partition_by
        ]
        if missing_sample_keys:
            raise ValueError(
                f"Dataset series '{config.id}' uses sample.keys "
                f"{missing_sample_keys!r} that are not part of stream "
                f"'{config.stream}' partition_by {list(partition_by)!r}."
            )

    if isinstance(dataset.split, HashSplitConfig):
        selected_streams = stream_dependency_closure(
            streams.streams,
            (config.stream for config in dataset.series),
        )
        cross_sections = sorted(
            stream_id
            for stream_id in selected_streams
            if isinstance(streams.streams[stream_id], CrossSectionStreamConfig)
        )
        if cross_sections:
            raise ValueError(
                "hash splits cannot be used with cross-sectional streams: "
                + ", ".join(cross_sections)
            )

        cross_timestamp_streams: list[str] = []
        for stream_id in selected_streams:
            stream = streams.streams[stream_id]
            uses_cross_timestamp_transform = any(
                isinstance(operation, _CROSS_TIMESTAMP_TRANSFORMS)
                for operation in stream.transforms
            )
            uses_temporal_as_of = (
                isinstance(stream, CombinedStreamConfig)
                and isinstance(stream.join, (AsOfJoin, BroadcastAsOfJoin))
                and (
                    stream.join.max_age is None
                    or parse_timecode(stream.join.max_age).total_seconds() > 0
                )
            )
            if uses_cross_timestamp_transform or uses_temporal_as_of:
                cross_timestamp_streams.append(stream_id)

        if cross_timestamp_streams:
            raise ValueError(
                "hash splits cannot be used with cross-timestamp streams: "
                + ", ".join(sorted(cross_timestamp_streams))
            )
