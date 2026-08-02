import glob
import hashlib
import json
import stat
from collections.abc import Iterable, Mapping
from datetime import timedelta
from pathlib import Path

from datapipeline.artifacts.models import VECTOR_METADATA_VERSION
from datapipeline.artifacts.planning import ArtifactGraph
from datapipeline.artifacts.scaler import SCALER_ARTIFACT_VERSION
from datapipeline.artifacts.series import SERIES_MANIFEST_VERSION
from datapipeline.artifacts.specs import dataset_requires_scaler
from datapipeline.config.dataset.dataset import DatasetConfig
from datapipeline.config.dataset.split import TimeSplitConfig
from datapipeline.config.sources import (
    FsLoaderConfig,
    SourceConfig,
)
from datapipeline.config.streams import SourceStreamConfig, StreamsConfig
from datapipeline.config.tasks.base import ArtifactTask
from datapipeline.config.tasks.coverage_stats import CoverageStatsTask
from datapipeline.config.tasks.metadata import MetadataTask
from datapipeline.config.tasks.scaler import ScalerTask
from datapipeline.config.tasks.series import SeriesTask
from datapipeline.config.tasks.schedule import ScheduleTask
from datapipeline.services.definitions import ArtifactHashes, ProjectManifest
from datapipeline.services.streams.validation import stream_dependency_closure

# Increment when Jerry's core artifact semantics change without a config change.
ARTIFACT_CACHE_VERSION = 9


def _normalized_label(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir))
    except ValueError:
        return str(path.resolve())


def _source_label(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def _hash_source_file(hasher, path: Path, base_dir: Path) -> None:
    try:
        metadata = path.stat()
    except FileNotFoundError:
        state = "missing"
    else:
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Source input is not a regular file: {path}")
        state = f"file:{metadata.st_size}:{metadata.st_mtime_ns}:{metadata.st_ctime_ns}"
    snapshot = (
        f"{_source_label(path, base_dir)}\0"
        f"{_normalized_label(path, base_dir)}\0{state}\0"
    )
    hasher.update(snapshot.encode("utf-8"))


def _hash_source_pattern(
    hasher,
    pattern: Path,
    expands_glob: bool,
    base_dir: Path,
) -> None:
    hasher.update(f"[source]{_source_label(pattern, base_dir)}\0".encode("utf-8"))
    if expands_glob:
        matches = [Path(match) for match in sorted(glob.glob(str(pattern)))]
        if not matches:
            hasher.update(b"[no-matches]")
            return
    else:
        matches = [pattern]
    for path in matches:
        _hash_source_file(hasher, path, base_dir)


def _project_source_path(project: ProjectManifest, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project.path.parent / path


def _source_input_patterns(
    source: SourceConfig,
    project: ProjectManifest,
) -> Iterable[tuple[Path, bool]]:
    loader = source.loader
    if isinstance(loader, FsLoaderConfig):
        yield (
            _project_source_path(project, loader.path),
            glob.has_magic(loader.path),
        )
    if source.inputs is not None:
        for raw_path in source.inputs.files:
            yield _project_source_path(project, raw_path), glob.has_magic(raw_path)


def _hash_source_inputs(
    hasher,
    project: ProjectManifest,
    sources: Mapping[str, SourceConfig],
    base_dir: Path,
) -> None:
    for source_id in sorted(sources):
        source = sources[source_id]
        for path, expands_glob in _source_input_patterns(source, project):
            _hash_source_pattern(hasher, path, expands_glob, base_dir)


def _stream_config_closure(
    root_stream_ids: Iterable[str],
    streams: StreamsConfig,
) -> tuple[dict[str, object], set[str]]:
    stream_ids = stream_dependency_closure(streams.streams, root_stream_ids)
    source_ids: set[str] = set()
    for stream_id in stream_ids:
        stream = streams.streams[stream_id]
        if isinstance(stream, SourceStreamConfig):
            source_ids.add(stream.from_.source)

    config: dict[str, object] = {
        "sources": {
            source_id: streams.sources[source_id].model_dump(mode="json")
            for source_id in sorted(source_ids)
        },
        "streams": {
            stream_id: streams.streams[stream_id].model_dump(mode="json")
            for stream_id in sorted(stream_ids)
        },
    }
    return config, source_ids


def _artifact_inputs(
    task: ArtifactTask,
    dataset: DatasetConfig,
    streams: StreamsConfig,
) -> tuple[dict[str, object], set[str]]:
    if isinstance(task, ScheduleTask):
        stream_config, source_ids = _stream_config_closure((task.stream,), streams)
        return {"streams": stream_config}, source_ids

    if isinstance(task, ScalerTask):
        scaled = tuple(config for config in dataset.series if config.scale)
        stream_config, source_ids = _stream_config_closure(
            (config.stream for config in scaled),
            streams,
        )
        dataset_inputs: dict[str, object] = {
            "sample": dataset.sample.model_dump(mode="json"),
            "split": (
                dataset.split.model_dump(mode="json")
                if dataset.split is not None
                else None
            ),
            "scaled_series": [
                config.model_dump(
                    mode="json",
                    exclude={"collect", "horizon", "sequence"},
                )
                for config in scaled
            ],
        }
        target_horizon = dataset.max_target_horizon
        if isinstance(dataset.split, TimeSplitConfig) and target_horizon > timedelta():
            dataset_inputs["target_horizon_seconds"] = int(
                target_horizon.total_seconds()
            )
        return (
            {
                "scaler_format_version": SCALER_ARTIFACT_VERSION,
                "dataset": dataset_inputs,
                "streams": stream_config,
            },
            source_ids,
        )

    if isinstance(task, SeriesTask):
        input_configs = dataset.series
        stream_config, source_ids = _stream_config_closure(
            (config.stream for config in input_configs),
            streams,
        )
        return (
            {
                "series_format_version": SERIES_MANIFEST_VERSION,
                "dataset": {
                    "sample": dataset.sample.model_dump(mode="json"),
                    "features": [
                        config.model_dump(mode="json", exclude={"scale"})
                        for config in dataset.features
                    ],
                    "targets": [
                        config.model_dump(
                            mode="json",
                            exclude={"horizon", "scale"},
                        )
                        for config in dataset.targets
                    ],
                },
                "streams": stream_config,
            },
            source_ids,
        )

    if isinstance(task, CoverageStatsTask) and task.stage == "postprocessed":
        return {"postprocess": dataset.postprocess.model_dump(mode="json")}, set()

    if isinstance(task, MetadataTask):
        inputs: dict[str, object] = {"metadata_format_version": VECTOR_METADATA_VERSION}
        if dataset.split is not None:
            inputs["split"] = dataset.split.model_dump(mode="json")
            inputs["target_horizon_seconds"] = int(
                dataset.max_target_horizon.total_seconds()
            )
        return inputs, set()

    if isinstance(task, CoverageStatsTask):
        return {}, set()

    # Plugin artifacts receive the full Runtime and declare no input contract.
    # Their only safe cache boundary is the complete dataset and stream catalog.
    return (
        {
            "dataset": dataset.model_dump(mode="json"),
            "streams": streams.model_dump(mode="json"),
        },
        set(streams.sources),
    )


def _artifact_digest(
    project: ProjectManifest,
    task: ArtifactTask,
    dependencies: Mapping[str, str],
    inputs: Mapping[str, object],
    source_snapshots: Mapping[str, str],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"[artifact-cache]{ARTIFACT_CACHE_VERSION}\0".encode("utf-8"))
    payload = {
        "artifact_revision": project.config.artifact_revision,
        "task": task.model_dump(mode="json"),
        "dependencies": dict(dependencies),
        "inputs": dict(inputs),
        "source_snapshots": dict(source_snapshots),
    }
    hasher.update(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return hasher.hexdigest()


def calculate_artifact_hashes(
    project: ProjectManifest,
    dataset: DatasetConfig,
    streams: StreamsConfig,
    graph: ArtifactGraph,
) -> ArtifactHashes:
    if dataset_requires_scaler(dataset) and not any(
        isinstance(task, ScalerTask) for task in graph.tasks_by_id.values()
    ):
        raise ValueError("Required artifact operation 'scaler' is not declared.")

    base_dir = project.path.parent
    active_keys = graph.dependency_closure(
        graph.declared_artifact_keys(),
        dataset,
    )
    graph.validate_producers(active_keys)
    hashes: dict[str, str] = {}
    source_snapshot_cache: dict[str, str] = {}

    def source_snapshot(source_id: str) -> str:
        cached = source_snapshot_cache.get(source_id)
        if cached is not None:
            return cached
        hasher = hashlib.sha256()
        _hash_source_inputs(
            hasher,
            project,
            {source_id: streams.sources[source_id]},
            base_dir,
        )
        snapshot = hasher.hexdigest()
        source_snapshot_cache[source_id] = snapshot
        return snapshot

    for key in graph.topological_order(graph.declared_artifact_keys()):
        task = graph.tasks_by_id[key]
        dependency_hashes = {
            dependency: hashes[dependency]
            for dependency in graph.definition(key).dependencies
            if graph.definition(dependency).is_required_for(dataset)
        }
        inputs, source_ids = _artifact_inputs(task, dataset, streams)
        snapshots = {
            source_id: source_snapshot(source_id) for source_id in sorted(source_ids)
        }
        hashes[key] = _artifact_digest(
            project,
            task,
            dependency_hashes,
            inputs,
            snapshots,
        )
    return ArtifactHashes(hashes)
