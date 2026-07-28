from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from stat import S_ISREG
from types import MappingProxyType

from datapipeline.artifacts.specs import (
    ARTIFACT_DEFINITIONS,
    SCALER_STATISTICS,
    SERIES,
    VECTOR_METADATA,
    COVERAGE_STATS,
    ArtifactDefinition,
    dataset_requires_scaler,
)
from datapipeline.artifacts.state import BuildState
from datapipeline.config.dataset.dataset import DatasetConfig
from datapipeline.config.preview import PREVIEW_STAGES, PreviewStage
from datapipeline.config.streams import StreamsConfig
from datapipeline.config.tasks.base import ArtifactTask, RuntimeTask
from datapipeline.config.tasks.coverage import CoverageTask
from datapipeline.config.tasks.dataset import DatasetTask
from datapipeline.config.tasks.matrix import MatrixTask
from datapipeline.config.tasks.schedule import ScheduleTask
from datapipeline.config.transforms import EnsureScheduleConfig
from datapipeline.io.output import output_destination_key
from datapipeline.services.definitions import ArtifactHashes


@dataclass(frozen=True)
class ArtifactFreshness:
    missing: frozenset[str]
    stale: frozenset[str]
    outdated: frozenset[str]


@dataclass(frozen=True)
class ArtifactGraph:
    definitions: tuple[ArtifactDefinition, ...]
    tasks_by_id: Mapping[str, ArtifactTask]
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
        self._validate_acyclic()

    @classmethod
    def from_tasks(
        cls,
        task_configs: Iterable[ArtifactTask],
        dataset: DatasetConfig | None = None,
        streams: StreamsConfig | None = None,
    ) -> "ArtifactGraph":
        if (dataset is None) != (streams is None):
            raise ValueError("dataset and streams must be provided together")

        tasks = tuple(task.model_copy(deep=True) for task in task_configs)
        tasks_by_id: dict[str, ArtifactTask] = {}
        for task in tasks:
            if task.id in tasks_by_id:
                raise ValueError(f"Duplicate artifact operation id '{task.id}'.")
            tasks_by_id[task.id] = task

        tasks_by_output: dict[str, str] = {}
        for task in tasks:
            output = Path(task.output)
            destination_key = output_destination_key(output)
            previous_task_id = tasks_by_output.get(destination_key)
            if previous_task_id is not None:
                raise ValueError(
                    f"Artifact operations '{previous_task_id}' and '{task.id}' write "
                    f"the same output '{task.output}'."
                )
            tasks_by_output[destination_key] = task.id

        definitions = list(ARTIFACT_DEFINITIONS)
        built_in_keys = {definition.key for definition in definitions}
        if dataset is not None and streams is not None:
            scaler_streams = {
                config.stream for config in dataset.series if config.scale
            }
            input_streams = {config.stream for config in dataset.series}
            scaler_schedules = required_schedule_artifacts(
                scaler_streams,
                streams,
                tasks_by_id,
            )
            input_schedules = required_schedule_artifacts(
                input_streams,
                streams,
                tasks_by_id,
            )
            definitions = [
                replace(
                    definition,
                    dependencies=(
                        *definition.dependencies,
                        *(
                            scaler_schedules
                            if definition.key == SCALER_STATISTICS
                            else ()
                        ),
                        *(input_schedules if definition.key == SERIES else ()),
                    ),
                )
                if definition.key in {SCALER_STATISTICS, SERIES}
                else definition
                for definition in definitions
            ]

        for task in tasks:
            if task.id in built_in_keys:
                continue
            definitions.append(ArtifactDefinition(key=task.id))

        return cls(tuple(definitions), tasks_by_id)

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

    def runtime_requirements(
        self,
        task: RuntimeTask,
        *,
        preview: PreviewStage | None,
    ) -> set[str]:
        declared = set(task.requires)
        dataset_schedule_artifacts = {
            dependency
            for dependency in self.definition(SERIES).dependencies
            if isinstance(self.tasks_by_id.get(dependency), ScheduleTask)
        }
        if isinstance(task, DatasetTask):
            if preview is None:
                return declared | {VECTOR_METADATA}
            if preview not in PREVIEW_STAGES:
                expected = ", ".join(PREVIEW_STAGES)
                raise ValueError(f"preview must be one of: {expected}")
            if preview in {"input", "canonical", "records", "series"}:
                return declared | dataset_schedule_artifacts
            return declared | {VECTOR_METADATA}
        if isinstance(task, CoverageTask):
            return declared | {COVERAGE_STATS}
        if isinstance(task, MatrixTask):
            return declared | {VECTOR_METADATA}
        return declared

    def runtime_dependency_closure(
        self,
        task: RuntimeTask,
        *,
        preview: PreviewStage | None,
        dataset: DatasetConfig | None,
    ) -> tuple[str, ...]:
        roots = self.runtime_requirements(task, preview=preview)
        if (
            isinstance(task, DatasetTask)
            and dataset is not None
            and not dataset.features
            and not dataset.targets
        ):
            roots = set(task.requires)
        elif (
            isinstance(task, DatasetTask)
            and preview is None
            and dataset is not None
            and dataset_requires_scaler(dataset)
        ):
            roots.add(SCALER_STATISTICS)
        keys = set(self.dependency_closure(roots))
        if self.requires_dataset(keys):
            if dataset is None:
                raise ValueError(
                    f"Runtime operation '{task.id}' requires dataset configuration to "
                    "resolve artifact dependencies."
                )
            inactive_declared = {
                key
                for key in task.requires
                if not self.definition(key).is_required_for(dataset)
            }
            if inactive_declared:
                artifacts = ", ".join(self.topological_order(inactive_declared))
                raise ValueError(
                    f"Runtime operation '{task.id}' explicitly requires artifact(s) "
                    f"that are inactive for this dataset: {artifacts}."
                )
            keys = set(self.dependency_closure(roots, dataset))
        return self.topological_order(keys)

    def dependency_closure(
        self,
        roots: Iterable[str],
        dataset: DatasetConfig | None = None,
    ) -> tuple[str, ...]:
        root_keys = set(roots)
        for key in root_keys:
            self.definition(key)

        selected: set[str] = set()

        def include(key: str) -> None:
            if key in selected:
                return
            definition = self.definition(key)
            if dataset is not None and not definition.is_required_for(dataset):
                return
            selected.add(key)
            for dependency in definition.dependencies:
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

    def requires_dataset(self, keys: Iterable[str]) -> bool:
        return any(self.definition(key).requires_dataset() for key in keys)

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
                    or file_stat.st_ctime_ns != fingerprint.ctime_ns
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
    visited: set[str] = set()

    def visit(current_stream_id: str) -> None:
        if current_stream_id in visited:
            return
        visited.add(current_stream_id)
        try:
            stream = streams.streams[current_stream_id]
        except KeyError as exc:
            raise ValueError(
                f"Unknown stream '{current_stream_id}' in artifact dependency graph."
            ) from exc
        for operation in stream.transforms:
            if isinstance(operation, EnsureScheduleConfig):
                artifacts.add(operation.schedule)
        for input_stream_id in stream.input_streams():
            visit(input_stream_id)

    visit(stream_id)
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
    dataset: DatasetConfig | None = None,
    streams: StreamsConfig | None = None,
) -> ArtifactGraph:
    return ArtifactGraph.from_tasks(task_configs, dataset, streams)
