from collections.abc import Iterable

from jerrythomas.artifacts.planning import ArtifactGraph
from jerrythomas.artifacts.validation import nested_schedule_dependencies
from jerrythomas.artifacts.state import BuildState, load_build_state
from jerrythomas.runtime import Runtime
from jerrythomas.services.definitions import ArtifactHashes, ProjectDefinition


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
) -> tuple[str, ...]:
    state = load_build_state(definition.project.artifacts_root)
    if state is None:
        runtime.artifacts.clear()
        return ()

    graph = definition.artifact_graph
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
