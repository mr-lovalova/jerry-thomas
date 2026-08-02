from pathlib import Path

from pydantic import ConfigDict, TypeAdapter

from datapipeline.config.sources import SourceConfig
from datapipeline.config.streams import StreamConfig, StreamsConfig
from datapipeline.services.config_inventory import pipeline_yaml_files
from datapipeline.services.definitions import ProjectManifest
from datapipeline.services.streams.validation import validate_stream_configs
from datapipeline.io.yaml import YamlDocument, read_yaml_document


_STREAM_CONFIG_ADAPTER: TypeAdapter[StreamConfig] = TypeAdapter(
    StreamConfig,
    config=ConfigDict(title="StreamConfig"),
)


def _source_documents(project: ProjectManifest) -> tuple[YamlDocument, ...]:
    documents: list[YamlDocument] = []
    for root in project.source_dirs:
        try:
            paths = pipeline_yaml_files(root)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise FileNotFoundError(f"sources dir not found: {root}") from exc
        documents.extend(read_yaml_document(path) for path in paths)
    return tuple(documents)


def _stream_documents(project: ProjectManifest) -> tuple[YamlDocument, ...]:
    documents: list[YamlDocument] = []
    for root in project.stream_dirs:
        try:
            paths = pipeline_yaml_files(root)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise FileNotFoundError(f"streams dir not found: {root}") from exc
        documents.extend(read_yaml_document(path) for path in paths)
    return tuple(documents)


def _source_from_document(
    project: ProjectManifest,
    document: YamlDocument,
) -> SourceConfig:
    data = project.resolve_config(document.data)
    if not data.get("id"):
        raise ValueError(f"Missing 'id' in source file: {document.path}")
    return SourceConfig.model_validate(data)


def _stream_from_document(
    project: ProjectManifest,
    document: YamlDocument,
) -> StreamConfig:
    data = project.resolve_config(document.data)
    if not data.get("id"):
        raise ValueError(f"Missing 'id' in stream file: {document.path}")
    return _STREAM_CONFIG_ADAPTER.validate_python(data)


def _declared_ids(
    project: ProjectManifest,
    documents: tuple[YamlDocument, ...],
) -> frozenset[str]:
    ids: dict[str, Path] = {}
    for document in documents:
        raw_id = document.data.get("id")
        if raw_id is None:
            raise ValueError(f"Missing 'id' in config file: {document.path}")
        resolved_id = project.resolve_config(raw_id)
        if not isinstance(resolved_id, str) or not resolved_id.strip():
            raise ValueError(f"Invalid 'id' in config file: {document.path}")
        resolved_id = resolved_id.strip()
        previous = ids.get(resolved_id)
        if previous is not None:
            raise ValueError(
                f"Duplicate ids in config files: {previous} and {document.path}"
            )
        ids[resolved_id] = document.path
    return frozenset(ids)


def declared_source_ids(project: ProjectManifest) -> frozenset[str]:
    return _declared_ids(project, _source_documents(project))


def declared_stream_ids(project: ProjectManifest) -> frozenset[str]:
    return _declared_ids(project, _stream_documents(project))


def _streams_from_documents(
    project: ProjectManifest,
    source_documents: tuple[YamlDocument, ...],
    stream_documents: tuple[YamlDocument, ...],
) -> StreamsConfig:
    sources: dict[str, SourceConfig] = {}
    source_paths: dict[str, Path] = {}
    for document in source_documents:
        source = _source_from_document(project, document)
        if source.id in sources:
            raise ValueError(
                f"Duplicate source id '{source.id}' in source files: "
                f"{source_paths[source.id]} and {document.path}"
            )
        sources[source.id] = source
        source_paths[source.id] = document.path

    streams: dict[str, StreamConfig] = {}
    stream_paths: dict[str, Path] = {}
    for document in stream_documents:
        stream = _stream_from_document(project, document)
        if stream.id in streams:
            raise ValueError(
                f"Duplicate stream id '{stream.id}' in stream files: "
                f"{stream_paths[stream.id]} and {document.path}"
            )
        streams[stream.id] = stream
        stream_paths[stream.id] = document.path

    config = StreamsConfig(sources=sources, streams=streams)
    validate_stream_configs(config.sources, config.streams)
    return config


def load_streams(project: ProjectManifest) -> StreamsConfig:
    return _streams_from_documents(
        project,
        _source_documents(project),
        _stream_documents(project),
    )
