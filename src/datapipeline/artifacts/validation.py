from dataclasses import dataclass
from datapipeline.artifacts.planning import ArtifactGraph, stream_tick_artifacts
from datapipeline.config.streams import StreamsConfig
from datapipeline.config.tasks.ticks import TicksTask


@dataclass(frozen=True)
class NestedTickDependency:
    task: TicksTask
    tick_artifacts: frozenset[str]


def validate_artifact_plan(
    streams: StreamsConfig,
    graph: ArtifactGraph,
    artifact_keys: set[str],
) -> None:
    graph.validate_producers(artifact_keys)
    nested_dependencies = nested_tick_dependencies(
        streams,
        graph,
        artifact_keys,
    )
    if nested_dependencies:
        dependency = nested_dependencies[0]
        artifacts = ", ".join(sorted(dependency.tick_artifacts))
        raise ValueError(
            f"Tick artifact '{dependency.task.id}' source stream "
            f"'{dependency.task.stream}' uses tick artifact(s): {artifacts}. "
            "Nested tick artifact "
            "dependencies are not supported."
        )


def nested_tick_dependencies(
    streams: StreamsConfig,
    graph: ArtifactGraph,
    artifact_keys: set[str],
) -> tuple[NestedTickDependency, ...]:
    tick_tasks: list[TicksTask] = []
    for key in graph.topological_order(artifact_keys):
        task = graph.tasks_by_id.get(key)
        if isinstance(task, TicksTask):
            tick_tasks.append(task)
    if not tick_tasks:
        return ()

    nested: list[NestedTickDependency] = []
    for task in tick_tasks:
        tick_artifacts = stream_tick_artifacts(task.stream, streams)
        if tick_artifacts:
            nested.append(
                NestedTickDependency(
                    task=task,
                    tick_artifacts=frozenset(tick_artifacts),
                )
            )
    return tuple(nested)
