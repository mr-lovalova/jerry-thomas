from collections.abc import Iterable

from datapipeline.artifacts.planning import ArtifactGraph, build_artifact_graph
from datapipeline.artifacts.validation import nested_schedule_dependencies
from datapipeline.build.state import BuildState, load_build_state
from datapipeline.runtime import Runtime
from datapipeline.services.definitions import ArtifactHashes, ProjectDefinition


def hydrate_runtime_artifacts(
    *,
    runtime: Runtime,
    graph: ArtifactGraph,
    state: BuildState | None,
    artifact_hashes: ArtifactHashes,
    artifact_keys: Iterable[str],
) -> tuple[str, ...]:
    keys = set(artifact_keys)
    runtime.artifacts.clear()
    if state is None or not keys:
        return ()

    freshness = graph.freshness(
        keys=keys,
        state=state,
        artifact_hashes=artifact_hashes,
        artifacts_root=runtime.artifacts.root,
    )
    keys_without_producers = {key for key in keys if key not in graph.tasks_by_id}
    unavailable_keys = keys_without_producers | graph.dependents_of(
        keys_without_producers,
        active_keys=keys,
    )
    current_keys = keys - set(freshness.outdated) - unavailable_keys
    ordered_current = graph.topological_order(current_keys)
    for key in ordered_current:
        info = state.artifacts[key]
        runtime.artifacts.register(
            key,
            relative_path=info.relative_path,
            meta=info.meta,
        )
    return ordered_current


def hydrate_runtime_artifacts_for_pipeline(
    runtime: Runtime,
    definition: ProjectDefinition,
    *,
    graph: ArtifactGraph | None = None,
) -> tuple[str, ...]:
    state = load_build_state(definition.project.artifacts_root)
    if state is None:
        runtime.artifacts.clear()
        return ()

    if graph is None:
        graph = build_artifact_graph(
            definition.artifact_operations,
            definition.dataset,
            definition.streams,
        )

    artifact_roots = graph.declared_artifact_keys()
    artifact_keys = set(graph.dependency_closure(artifact_roots, definition.dataset))
    if not artifact_keys:
        runtime.artifacts.clear()
        return ()
    nested_schedules = {
        dependency.task.id
        for dependency in nested_schedule_dependencies(
            definition.streams,
            graph,
            artifact_keys,
        )
    }
    artifact_keys -= nested_schedules | graph.dependents_of(
        nested_schedules,
        active_keys=artifact_keys,
    )
    return hydrate_runtime_artifacts(
        runtime=runtime,
        graph=graph,
        state=state,
        artifact_hashes=definition.artifact_hashes,
        artifact_keys=artifact_keys,
    )
