from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from stat import S_ISREG
from types import MappingProxyType

from jerrythomas.artifacts.specs import (
    ARTIFACT_DEFINITIONS,
    SCALER_STATISTICS,
    SERIES,
    VECTOR_METADATA,
    COVERAGE_STATS,
    ArtifactDefinition,
    dataset_requires_scaler,
)
from jerrythomas.artifacts.series import series_cache_root
from jerrythomas.artifacts.state import ArtifactFileFingerprint, BuildState
from jerrythomas.config.dataset.dataset import DatasetConfig
from jerrythomas.config.preview import PREVIEW_STAGES, PreviewStage
from jerrythomas.config.streams import StreamsConfig
from jerrythomas.config.tasks.base import ArtifactTask, RuntimeTask
from jerrythomas.config.tasks.coverage import CoverageTask
from jerrythomas.config.tasks.dataset import DatasetTask
from jerrythomas.config.tasks.matrix import MatrixTask
from jerrythomas.config.tasks.schedule import ScheduleTask
from jerrythomas.config.tasks.series import SeriesTask
from jerrythomas.config.tasks.stream import StreamTask
from jerrythomas.config.transforms import EnsureScheduleConfig
from jerrythomas.io.output import output_destination_key
from jerrythomas.services.definitions import ArtifactHashes
from jerrythomas.services.operations import artifact_kind
from jerrythomas.services.streams.validation import stream_dependency_closure


@dataclass(frozen=True)
class ArtifactFreshness:
    missing: frozenset[str]
    stale: frozenset[str]
    outdated: frozenset[str]


@dataclass(frozen=True)
class ArtifactGraph:
    definitions: tuple[ArtifactDefinition, ...]
    tasks_by_id: Mapping[str, ArtifactTask]
    datasets: Mapping[str, DatasetConfig] = field(default_factory=dict)
    streams: StreamsConfig = field(default_factory=StreamsConfig)
    _definitions_by_key: Mapping[str, ArtifactDefinition] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        definitions_by_key: dict[str, ArtifactDefinition] = {}
        for definition in self.definitions:
            if definition.key in definitions_by_key:
                raise ValueError(f"Duplicate artifact key '{definition.key}'.")
            definitions_by_key[definition.key] = definition

        for definition in self.definitions:
            for dependency in definition.dependencies:
                if dependency not in definitions_by_key:
                    raise ValueError(
                        f"Artifact '{definition.key}' has unknown dependency '{dependency}'."
                    )

        for task_id, task in self.tasks_by_id.items():
            if task_id != task.id:
                raise ValueError(
                    f"Artifact operation mapping key '{task_id}' does not match operation id "
                    f"'{task.id}'."
                )
            if task_id not in definitions_by_key:
                raise ValueError(
                    f"Artifact operation '{task_id}' has no matching artifact definition."
                )

        object.__setattr__(
            self,
            "tasks_by_id",
            MappingProxyType(dict(self.tasks_by_id)),
        )
        object.__setattr__(
            self,
            "_definitions_by_key",
            MappingProxyType(definitions_by_key),
        )
        object.__setattr__(self, "datasets", MappingProxyType(dict(self.datasets)))
        self._validate_acyclic()

    def _validate_acyclic(self) -> None:
        visited: set[str] = set()
        path: list[str] = []

        def visit(key: str) -> None:
            if key in path:
                start = path.index(key)
                cycle = [*path[start:], key]
                raise ValueError("Artifact dependency cycle: " + " -> ".join(cycle))
            if key in visited:
                return
            path.append(key)
            for dependency in self.definition(key).dependencies:
                visit(dependency)
            path.pop()
            visited.add(key)

        for definition in self.definitions:
            visit(definition.key)

    def definition(self, key: str) -> ArtifactDefinition:
        try:
            return self._definitions_by_key[key]
        except KeyError as exc:
            raise ValueError(f"Unknown artifact '{key}'.") from exc

    def declared_artifact_keys(self) -> set[str]:
        return set(self.tasks_by_id)

    def dataset_artifact_keys(self, dataset_id: str | None) -> dict[str, str]:
        return {
            kind: task.id
            for task in self.tasks_by_id.values()
            if task.dataset == dataset_id and (kind := artifact_kind(task)) is not None
        }

    def is_active(self, key: str) -> bool:
        definition = self.definition(key)
        if definition.required_if is None:
            return True
        task = self.tasks_by_id.get(key)
        if task is None or task.dataset is None:
            return True
        return definition.is_required_for(self.datasets[task.dataset])

    def runtime_requirements(
        self,
        task: RuntimeTask,
        *,
        preview: PreviewStage | None,
    ) -> set[str]:
        declared = set(task.requires)
        if isinstance(task, StreamTask):
            return declared | set(
                required_schedule_artifacts(
                    (task.stream,), self.streams, self.tasks_by_id
                )
            )
        if not isinstance(task, (DatasetTask, CoverageTask, MatrixTask)):
            return declared
        if task.dataset is None or task.dataset not in self.datasets:
            raise ValueError(f"Runtime operation '{task.id}' requires a named dataset.")
        if isinstance(task, DatasetTask):
            if preview is not None and preview not in PREVIEW_STAGES:
                raise ValueError(f"preview must be one of: {', '.join(PREVIEW_STAGES)}")
            dataset = self.datasets[task.dataset]
            if not dataset.series:
                return declared
            if preview in {"input", "canonical", "records", "series"}:
                return declared | set(
                    required_schedule_artifacts(
                        (config.stream for config in dataset.series),
                        self.streams,
                        self.tasks_by_id,
                    )
                )
        keys = self.dataset_artifact_keys(task.dataset)
        if isinstance(task, DatasetTask):
            return declared | {keys[VECTOR_METADATA]}
        if isinstance(task, CoverageTask):
            return declared | {keys[COVERAGE_STATS]}
        return declared | {keys[VECTOR_METADATA]}

    def runtime_dependency_closure(
        self,
        task: RuntimeTask,
        *,
        preview: PreviewStage | None,
    ) -> tuple[str, ...]:
        roots = self.runtime_requirements(task, preview=preview)
        if isinstance(task, DatasetTask):
            assert task.dataset is not None
            dataset = self.datasets[task.dataset]
            if dataset.series and preview is None and dataset_requires_scaler(dataset):
                roots.add(self.dataset_artifact_keys(task.dataset)[SCALER_STATISTICS])
        inactive = {key for key in task.requires if not self.is_active(key)}
        if inactive:
            raise ValueError(
                f"Runtime operation '{task.id}' explicitly requires artifact(s) "
                f"that are inactive for this dataset: {', '.join(sorted(inactive))}."
            )
        return self.dependency_closure(roots)

    def dependency_closure(self, roots: Iterable[str]) -> tuple[str, ...]:
        root_keys = set(roots)
        for key in root_keys:
            self.definition(key)
        selected: set[str] = set()

        def include(key: str) -> None:
            if key in selected or not self.is_active(key):
                return
            selected.add(key)
            for dependency in self.definition(key).dependencies:
                include(dependency)

        for definition in self.definitions:
            if definition.key in root_keys:
                include(definition.key)
        return self.topological_order(selected)

    def topological_order(self, keys: Iterable[str]) -> tuple[str, ...]:
        selected = set(keys)
        for key in selected:
            self.definition(key)

        ordered: list[str] = []
        visited: set[str] = set()

        def visit(key: str) -> None:
            if key in visited:
                return
            for dependency in self.definition(key).dependencies:
                if dependency in selected:
                    visit(dependency)
            visited.add(key)
            ordered.append(key)

        for definition in self.definitions:
            if definition.key in selected:
                visit(definition.key)
        return tuple(ordered)

    def validate_producers(self, keys: Iterable[str]) -> None:
        for key in self.topological_order(keys):
            if key not in self.tasks_by_id:
                raise ValueError(
                    f"Required artifact operation '{key}' is not declared."
                )

    def dependents_of(
        self,
        keys: Iterable[str],
        *,
        active_keys: Iterable[str] | None = None,
    ) -> set[str]:
        roots = set(keys)
        active = (
            set(active_keys)
            if active_keys is not None
            else set(self._definitions_by_key)
        )
        dependents: set[str] = set()
        pending = list(roots)
        while pending:
            dependency = pending.pop()
            for definition in self.definitions:
                if definition.key not in active or definition.key in roots:
                    continue
                if dependency not in definition.dependencies:
                    continue
                if definition.key in dependents:
                    continue
                dependents.add(definition.key)
                pending.append(definition.key)
        return dependents

    def freshness(
        self,
        *,
        keys: Iterable[str],
        state: BuildState | None,
        artifact_hashes: ArtifactHashes,
        artifacts_root: Path,
    ) -> ArtifactFreshness:
        selected = set(keys)
        missing: set[str] = set()
        stale: set[str] = set()
        root = artifacts_root.resolve()

        for key in selected:
            if state is None:
                missing.add(key)
                continue
            info = state.artifacts.get(key)
            if info is None:
                missing.add(key)
                continue
            producer = self.tasks_by_id.get(key)
            if producer is not None and Path(info.relative_path) != Path(
                producer.output
            ):
                stale.add(key)
                continue
            if info.artifact_hash != artifact_hashes.for_artifact(key):
                stale.add(key)
                continue
            for fingerprint in info.files:
                artifact_path = (root / fingerprint.relative_path).resolve()
                try:
                    artifact_path.relative_to(root)
                except ValueError:
                    stale.add(key)
                    break
                try:
                    file_stat = artifact_path.stat()
                except FileNotFoundError:
                    missing.add(key)
                    break
                if not S_ISREG(file_stat.st_mode):
                    missing.add(key)
                    break
                if (
                    file_stat.st_size != fingerprint.size
                    or file_stat.st_mtime_ns != fingerprint.mtime_ns
                ):
                    stale.add(key)
                    break
                if file_stat.st_ctime_ns != fingerprint.ctime_ns:
                    # Metadata changes also update ctime; verify bytes before
                    # invalidating an otherwise unchanged artifact.
                    try:
                        verified = ArtifactFileFingerprint.from_path(
                            fingerprint.relative_path, artifact_path
                        )
                    except FileNotFoundError:
                        missing.add(key)
                        break
                    if (
                        verified.size != fingerprint.size
                        or verified.mtime_ns != fingerprint.mtime_ns
                        or verified.sha256 != fingerprint.sha256
                    ):
                        stale.add(key)
                        break

        outdated = missing | stale
        for key in self.topological_order(selected):
            dependencies = self.definition(key).dependencies
            if any(
                dependency in selected and dependency in outdated
                for dependency in dependencies
            ):
                outdated.add(key)

        return ArtifactFreshness(
            missing=frozenset(missing),
            stale=frozenset(stale),
            outdated=frozenset(outdated),
        )


def stream_schedule_artifacts(
    stream_id: str,
    streams: StreamsConfig,
) -> set[str]:
    artifacts: set[str] = set()
    for current_stream_id in stream_dependency_closure(
        streams.streams,
        (stream_id,),
    ):
        stream = streams.streams[current_stream_id]
        for operation in stream.transforms:
            if isinstance(operation, EnsureScheduleConfig):
                artifacts.add(operation.schedule)
    return artifacts


def required_schedule_artifacts(
    stream_ids: Iterable[str],
    streams: StreamsConfig,
    tasks_by_id: Mapping[str, ArtifactTask],
) -> tuple[str, ...]:
    artifact_ids = {
        artifact_id
        for stream_id in stream_ids
        for artifact_id in stream_schedule_artifacts(stream_id, streams)
    }
    for artifact_id in sorted(artifact_ids):
        task = tasks_by_id.get(artifact_id)
        if task is None:
            raise ValueError(
                f"Schedule artifact '{artifact_id}' requires a declared schedule "
                "operation "
                "with the same id."
            )
        if not isinstance(task, ScheduleTask):
            raise ValueError(
                f"Schedule artifact '{artifact_id}' references operation entrypoint "
                f"'{task.entrypoint}', not a schedule operation."
            )
    return tuple(sorted(artifact_ids))


def build_artifact_graph(
    task_configs: Iterable[ArtifactTask],
    datasets: Mapping[str, DatasetConfig] | None = None,
    streams: StreamsConfig | None = None,
) -> ArtifactGraph:
    catalog = dict(datasets or {})
    stream_configs = streams if streams is not None else StreamsConfig()
    tasks = tuple(task.model_copy(deep=True) for task in task_configs)
    tasks_by_id: dict[str, ArtifactTask] = {}
    producers: dict[tuple[str | None, str], str] = {}
    for task in tasks:
        if task.id in tasks_by_id:
            raise ValueError(f"Duplicate artifact operation id '{task.id}'.")
        if task.dataset is not None and task.dataset not in catalog:
            raise ValueError(
                f"Operation '{task.id}' references unknown dataset '{task.dataset}'."
            )
        tasks_by_id[task.id] = task
        if (kind := artifact_kind(task)) is not None:
            identity = (task.dataset, kind)
            if identity in producers:
                raise ValueError(
                    f"Multiple producers for dataset artifact {identity!r}."
                )
            producers[identity] = task.id
    _validate_artifact_output_paths(tasks)

    def key_for(dataset_id: str | None, kind: str) -> str:
        return producers.get(
            (dataset_id, kind),
            f"dataset.{dataset_id}.{kind}" if dataset_id is not None else kind,
        )

    definitions: list[ArtifactDefinition] = []
    contexts = dict.fromkeys([*catalog, *(dataset_id for dataset_id, _ in producers)])
    for dataset_id in contexts:
        dataset = catalog.get(dataset_id) if dataset_id is not None else None
        for blueprint in ARTIFACT_DEFINITIONS:
            dependencies = tuple(
                key_for(dataset_id, key) for key in blueprint.dependencies
            )
            if dataset is not None and blueprint.key in {SERIES, SCALER_STATISTICS}:
                selected_streams = {
                    config.stream
                    for config in dataset.series
                    if blueprint.key == SERIES or config.scale
                }
                dependencies += required_schedule_artifacts(
                    selected_streams, stream_configs, tasks_by_id
                )
            definitions.append(
                ArtifactDefinition(
                    key=key_for(dataset_id, blueprint.key),
                    dependencies=dependencies,
                    required_if=blueprint.required_if,
                )
            )
    known_keys = {definition.key for definition in definitions}
    definitions.extend(
        ArtifactDefinition(key=task.id) for task in tasks if task.id not in known_keys
    )
    return ArtifactGraph(tuple(definitions), tasks_by_id, catalog, stream_configs)


def _validate_artifact_output_paths(tasks: tuple[ArtifactTask, ...]) -> None:
    outputs = [
        (task, Path(output_destination_key(Path(task.output)))) for task in tasks
    ]
    for index, (task, output) in enumerate(outputs):
        for previous_task, previous_output in outputs[:index]:
            if output == previous_output:
                raise ValueError(
                    f"Artifact operations '{previous_task.id}' and '{task.id}' write "
                    f"the same output '{task.output}'."
                )
            if output.is_relative_to(previous_output) or previous_output.is_relative_to(
                output
            ):
                raise ValueError(
                    f"Artifact operations '{previous_task.id}' and '{task.id}' declare "
                    f"nested output paths '{previous_task.output}' and '{task.output}'."
                )

    for series_task in (task for task in tasks if isinstance(task, SeriesTask)):
        cache_root = Path(
            output_destination_key(series_cache_root(Path(series_task.output)))
        )
        for task, output in outputs:
            if task is series_task:
                continue
            if output.is_relative_to(cache_root):
                raise ValueError(
                    f"Artifact operation '{task.id}' writes inside series cache directory "
                    f"'{series_cache_root(Path(series_task.output))}' owned by artifact "
                    f"operation '{series_task.id}'."
                )
