from datapipeline.config.dataset.dataset import DatasetConfig
from datapipeline.config.dataset.split import HashSplitConfig
from datapipeline.config.streams import CrossSectionStreamConfig, StreamsConfig
from datapipeline.io.yaml import YamlDocument
from datapipeline.services.definitions import ProjectManifest
from datapipeline.services.streams.validation import (
    stream_dependency_closure,
    stream_partition_by,
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
