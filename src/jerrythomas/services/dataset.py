import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from pydantic_core import InitErrorDetails

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
from jerrythomas.io.yaml import YamlDocument, read_yaml_document
from jerrythomas.services.config_inventory import pipeline_yaml_files
from jerrythomas.services.definitions import ProjectManifest
from jerrythomas.services.path_policy import resolve_project_path
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
    transform_config.ResampleConfig,
    transform_config.RollingConfig,
    transform_config.RollingOlsConfig,
    transform_config.RollingQuantileConfig,
    transform_config.RollingSlopeConfig,
)


def dataset_from_document(
    project: ProjectManifest,
    document: YamlDocument,
    *,
    section_cache: dict[Path, dict[str, Any]] | None = None,
) -> DatasetConfig:
    data = project.resolve_config(document.data)
    if not isinstance(data, dict):
        raise ValueError(f"Dataset '{document.path}' must be a mapping")
    if section_cache is None:
        section_cache = {}
    references: dict[str, Path] = {}
    for section in ("sample", "split", "postprocess"):
        value = data.get(section)
        if not isinstance(value, dict) or "file" not in value:
            continue
        filename = value["file"]
        origin = f"Dataset '{document.path}', section '{section}'"
        context = f"{origin}, file {filename!r}"
        if set(value) != {"file"}:
            raise ValueError(f"{context}: file references cannot have overrides")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"{context}: file must be a non-empty path string")
        try:
            path = resolve_project_path(project.path, filename)
            context = f"{origin}, file '{path}'"
            if path not in section_cache:
                content = project.resolve_config(read_yaml_document(path).data)
                if "file" in content:
                    raise ValueError(
                        "shared section files cannot reference another file"
                    )
                section_cache[path] = content
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"{context}: {exc}") from exc
        data[section] = section_cache[path]
        references[section] = path
    try:
        return DatasetConfig.model_validate(data)
    except ValidationError as exc:
        if not references:
            raise
        origins = ", ".join(
            f"{section}: {path}" for section, path in references.items()
        )
        raise ValidationError.from_exception_data(
            f"Dataset '{document.path}' ({origins})",
            [
                InitErrorDetails(
                    type=error["type"],
                    loc=error["loc"],
                    input=error["input"],
                    ctx=error.get("ctx", {}),
                )
                for error in exc.errors(include_url=False)
            ],
        ) from exc


def load_datasets(project: ProjectManifest) -> dict[str, DatasetConfig]:
    datasets: dict[str, DatasetConfig] = {}
    section_cache: dict[Path, dict[str, Any]] = {}
    for root in project.dataset_dirs:
        try:
            paths = pipeline_yaml_files(root)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise FileNotFoundError(f"datasets directory not found: {root}") from exc
        for path in paths:
            dataset_id = path.stem
            if re.fullmatch(r"[a-z0-9_-]+(?:\.[a-z0-9_-]+)*", dataset_id) is None:
                raise ValueError(
                    f"Dataset filename '{path.name}' must use a lowercase dataset ID."
                )
            if dataset_id in datasets:
                raise ValueError(f"Duplicate dataset ID '{dataset_id}' at {path}")
            document = read_yaml_document(path)
            if "id" in document.data:
                raise ValueError(
                    f"{path} must not define id; the filename supplies '{dataset_id}'."
                )
            if "version" not in document.data:
                raise ValueError(f"Dataset '{dataset_id}' must declare version")
            datasets[dataset_id] = dataset_from_document(
                project, document, section_cache=section_cache
            )
    return datasets


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
