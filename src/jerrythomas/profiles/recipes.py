"""Capture recipes from the same resolved definitions and jobs used for execution."""

import glob
import platform
import subprocess
from pathlib import Path
from typing import Any, Literal, Sequence

from jerrythomas.artifacts.fingerprints import artifact_inputs, source_input_patterns
from jerrythomas.artifacts.state import ArtifactFileFingerprint, load_build_state
from jerrythomas.config.execution import ExecutionConfig
from jerrythomas.config.sources import FsLoaderConfig
from jerrythomas.config.streams import SourceStreamConfig
from jerrythomas.config.tasks.base import PluginRuntimeTask
from jerrythomas.config.tasks.stream import StreamTask
from jerrythomas.io.recipes import RunRecipe
from jerrythomas.io.runs import materialize_receipt_path
from jerrythomas.plugins import plugin_distributions
from jerrythomas.profiles.models import MaterializeJob, RuntimeJob
from jerrythomas.services.definitions import ProjectDefinition
from jerrythomas.services.path_policy import resolve_relative_fs_loader_path
from jerrythomas.services.streams.validation import stream_dependency_closure


def capture_recipe(
    definition: ProjectDefinition,
    command: Literal["serve", "materialize"],
    jobs: Sequence[RuntimeJob] | Sequence[MaterializeJob],
    execution: ExecutionConfig,
    required_artifacts: Sequence[str],
) -> RunRecipe:
    graph = definition.artifact_graph
    streams = definition.streams
    dataset_ids = {job.task.dataset for job in jobs if job.task.dataset is not None}
    if len(dataset_ids) > 1:
        raise ValueError("A saved run must belong to one dataset")
    dataset_id = next(iter(dataset_ids), None)
    dataset = definition.require_dataset(dataset_id) if dataset_id is not None else None
    captured_ids = dataset_ids | {
        task.dataset
        for key in required_artifacts
        if (task := graph.tasks_by_id[key]).dataset is not None
    }
    datasets = {key: definition.require_dataset(key) for key in sorted(captured_ids)}
    roots = {job.task.stream for job in jobs if isinstance(job.task, StreamTask)}
    for selected in datasets.values():
        roots.update(config.stream for config in selected.series)
    limitations = [
        "Inputs and executable code are identified, not archived.",
        "Local source files use filesystem fingerprints, not content hashes.",
        "Package versions and Git state do not prove which code was loaded.",
    ]
    if any(
        isinstance(job, RuntimeJob) and isinstance(job.task, PluginRuntimeTask)
        for job in jobs
    ):
        roots.update(streams.streams)
        limitations.append("Plugin runtime operations can access the entire catalog.")
    for key in required_artifacts:
        task = graph.tasks_by_id[key]
        bound_dataset = (
            definition.require_dataset(task.dataset)
            if task.dataset is not None
            else None
        )
        inputs, _ = artifact_inputs(task, bound_dataset, streams)
        catalog = inputs.get("streams", {})
        if isinstance(catalog, dict):
            roots.update(catalog.get("streams", {}))

    stream_ids = stream_dependency_closure(streams.streams, roots)
    source_ids = {
        stream.from_.source
        for key in stream_ids
        if isinstance(stream := streams.streams[key], SourceStreamConfig)
    }
    sources = {
        key: streams.sources[key].model_dump(mode="json", by_alias=True)
        for key in sorted(source_ids)
    }
    for key, source in sources.items():
        config = streams.sources[key]
        if isinstance(config.loader, FsLoaderConfig):
            source["loader"]["path"] = resolve_relative_fs_loader_path(
                config.loader.path, definition.project.path.parent
            )
        if config.inputs is not None:
            source["inputs"]["files"] = [
                str(
                    path
                    if path.is_absolute()
                    else definition.project.path.parent / path
                )
                for path in map(Path, config.inputs.files)
            ]
    configuration = {
        "artifact_revision": definition.project.config.artifact_revision,
        "execution": execution.model_dump(mode="json"),
        "datasets": {
            key: value.model_dump(mode="json") for key, value in datasets.items()
        },
        "sources": sources,
        "streams": {
            key: streams.streams[key].model_dump(mode="json", by_alias=True)
            for key in sorted(stream_ids)
        },
        "artifacts": {
            key: graph.tasks_by_id[key].model_dump(mode="json", by_alias=True)
            for key in required_artifacts
        },
        "jobs": [_job_configuration(job) for job in jobs],
    }
    return RunRecipe(
        command=command,
        project=str(definition.project.path),
        dataset_id=dataset_id,
        dataset_version=dataset.version if dataset is not None else None,
        configuration=configuration,
        implementation=_implementation(definition.project.path.parent, configuration),
        inputs={key: _source_snapshot(definition, key) for key in sorted(source_ids)},
        limitations=tuple(limitations),
    )


def capture_artifacts(recipe: RunRecipe, definition: ProjectDefinition) -> RunRecipe:
    """Record the artifacts selected after prerequisite resolution, not their history."""
    keys = recipe.configuration["artifacts"]
    state = load_build_state(definition.project.artifacts_root) if keys else None
    artifacts = {}
    for key in keys:
        info = state.artifacts.get(key) if state is not None else None
        artifacts[key] = {
            "expected_config_hash": definition.artifact_hashes.for_artifact(key),
            "root": str(definition.project.artifacts_root),
            "identity": info.model_dump(mode="json", exclude={"meta"})
            if info
            else None,
            "producer_recipe": None,
        }
    return recipe.model_copy(update={"artifacts": artifacts})


def validate_recipe_inputs(recipe: RunRecipe, definition: ProjectDefinition) -> None:
    for key, before in recipe.inputs.items():
        if _source_snapshot(definition, key) != before:
            raise RuntimeError(f"Source inputs changed during execution: {key}")


def output_identity(path: Path) -> dict[str, Any]:
    fingerprint = ArtifactFileFingerprint.from_path(path.name, path)
    return {"sha256": fingerprint.sha256, "size_bytes": fingerprint.size}


def _job_configuration(job: RuntimeJob | MaterializeJob) -> dict[str, Any]:
    output = job.output
    config: dict[str, Any] = {
        "profile": job.name,
        "output": {
            "transport": output.transport,
            "format": output.format,
            "view": output.view,
            "encoding": output.encoding,
            "compression": output.compression,
            "destination": str(output.destination) if output.destination else None,
        },
    }
    if isinstance(job, MaterializeJob):
        config.update(
            operation=job.task.model_dump(mode="json", by_alias=True),
            stream=job.stream,
            overwrite=job.overwrite,
        )
    else:
        config.update(
            operation=job.task.model_dump(mode="json", by_alias=True),
            limit=job.limit,
            preview=job.preview,
            throttle_ms=job.throttle_ms,
            output_ids=list(job.output_ids),
        )
    return config


def _source_snapshot(definition: ProjectDefinition, source_id: str) -> dict[str, Any]:
    source = definition.streams.sources[source_id]
    patterns = {}
    for pattern, expands_glob in source_input_patterns(source, definition.project):
        matches = sorted(glob.glob(str(pattern))) if expands_glob else [str(pattern)]
        patterns[str(pattern)] = [_file_snapshot(Path(path)) for path in matches]
    return {"kind": "filesystem" if patterns else "opaque", "patterns": patterns}


def _file_snapshot(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "resolved_path": str(path.resolve())}
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {**result, "missing": True}
    result.update(
        size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns
    )
    receipt = materialize_receipt_path(path)
    if receipt.is_file():
        result["upstream_receipt"] = {
            "path": str(receipt),
            "sha256": ArtifactFileFingerprint.content_digest(receipt),
            "relationship": "adjacent_receipt_not_content_verified",
        }
    return result


def _git_identity(path: Path) -> dict[str, Any] | None:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return result.stdout.strip()

    try:
        root = git("rev-parse", "--show-toplevel")
        return {
            "root": root,
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
        }
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None


def _implementation(project_dir: Path, configuration: dict[str, Any]) -> dict[str, Any]:
    # Names are only unique within their entry-point group. Never inspect
    # arbitrary plugin arguments, which may also contain an "entrypoint" key.
    selected: set[tuple[str, str]] = set()

    def include(group: str, value: dict[str, Any]) -> None:
        if "entrypoint" in value:
            selected.add((f"jerrythomas.{group}", value["entrypoint"]))

    for source in configuration["sources"].values():
        include("parsers", source["parser"])
        include("loaders", source["loader"])
    for stream in configuration["streams"].values():
        include("mappers", stream.get("map", {}))
        include("combiners", stream.get("combine", {}))
        for transform in stream["transforms"]:
            include("transforms", transform)
    for task in configuration["artifacts"].values():
        include("operations.build", task)
    for job in configuration["jobs"]:
        include("operations.runtime", job.get("operation", {}))

    packages = {}
    repositories = {str(project_dir): _git_identity(project_dir)}
    entrypoints = []
    for dist in plugin_distributions(selected):
        packages[dist.name] = dist.version
        for ep in dist.entrypoints:
            entrypoints.append(
                {
                    "group": ep.group,
                    "name": ep.name,
                    "value": ep.value,
                    "distribution": dist.name,
                }
            )
        if dist.editable_path is not None:
            path = str(dist.editable_path)
            if path not in repositories:
                repositories[path] = _git_identity(dist.editable_path)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "entrypoints": sorted(entrypoints, key=lambda ep: (ep["group"], ep["name"])),
        "unidentified_entrypoints": [
            {"group": group, "name": name}
            for group, name in sorted(
                selected - {(ep["group"], ep["name"]) for ep in entrypoints}
            )
        ],
        "repositories": repositories,
    }
