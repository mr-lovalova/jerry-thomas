from dataclasses import dataclass
from datapipeline.artifacts.planning import ArtifactGraph, stream_schedule_artifacts
from datapipeline.config.streams import StreamsConfig
from datapipeline.config.tasks.schedule import ScheduleTask


@dataclass(frozen=True)
class NestedScheduleDependency:
    task: ScheduleTask
    schedule_artifacts: frozenset[str]


def validate_artifact_plan(
    streams: StreamsConfig,
    graph: ArtifactGraph,
    artifact_keys: set[str],
) -> None:
    graph.validate_producers(artifact_keys)
    nested_dependencies = nested_schedule_dependencies(
        streams,
        graph,
        artifact_keys,
    )
    if nested_dependencies:
        dependency = nested_dependencies[0]
        artifacts = ", ".join(sorted(dependency.schedule_artifacts))
        raise ValueError(
            f"Schedule artifact '{dependency.task.id}' source stream "
            f"'{dependency.task.stream}' uses schedule artifact(s): {artifacts}. "
            "Nested schedule artifact "
            "dependencies are not supported."
        )


def nested_schedule_dependencies(
    streams: StreamsConfig,
    graph: ArtifactGraph,
    artifact_keys: set[str],
) -> tuple[NestedScheduleDependency, ...]:
    schedule_tasks: list[ScheduleTask] = []
    for key in graph.topological_order(artifact_keys):
        task = graph.tasks_by_id.get(key)
        if isinstance(task, ScheduleTask):
            schedule_tasks.append(task)
    if not schedule_tasks:
        return ()

    nested: list[NestedScheduleDependency] = []
    for task in schedule_tasks:
        schedule_artifacts = stream_schedule_artifacts(task.stream, streams)
        if schedule_artifacts:
            nested.append(
                NestedScheduleDependency(
                    task=task,
                    schedule_artifacts=frozenset(schedule_artifacts),
                )
            )
    return tuple(nested)
