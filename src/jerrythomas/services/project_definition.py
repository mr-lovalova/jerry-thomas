from pathlib import Path

from jerrythomas.artifacts.fingerprints import calculate_artifact_hashes
from jerrythomas.artifacts.planning import build_artifact_graph
from jerrythomas.config.tasks.base import ArtifactTask, RuntimeTask
from jerrythomas.config.tasks.stream import StreamTask
from jerrythomas.services.dataset import load_datasets, validate_dataset_streams
from jerrythomas.services.definitions import ProjectDefinition
from jerrythomas.services.operations import (
    operation_documents,
    operations_from_documents,
)
from jerrythomas.services.project import load_project
from jerrythomas.services.streams.loader import load_streams


def load_project_definition(project_yaml: Path) -> ProjectDefinition:
    project = load_project(project_yaml)
    datasets = load_datasets(project)
    streams = load_streams(project)
    for dataset_id, dataset in datasets.items():
        try:
            validate_dataset_streams(dataset, streams)
        except ValueError as exc:
            raise ValueError(f"Dataset '{dataset_id}': {exc}") from exc
    operations = operations_from_documents(
        project, operation_documents(project), datasets
    )
    for operation in operations:
        if (
            isinstance(operation, StreamTask)
            and operation.stream not in streams.streams
        ):
            raise ValueError(
                f"Operation '{operation.id}' references unknown stream '{operation.stream}'."
            )
    artifact_operations = tuple(
        operation for operation in operations if isinstance(operation, ArtifactTask)
    )
    runtime_operations = tuple(
        operation for operation in operations if isinstance(operation, RuntimeTask)
    )
    artifact_graph = build_artifact_graph(artifact_operations, datasets, streams)
    artifact_hashes = calculate_artifact_hashes(
        project, datasets, streams, artifact_graph
    )
    return ProjectDefinition(
        project=project,
        datasets=datasets,
        streams=streams,
        artifact_graph=artifact_graph,
        runtime_operations=runtime_operations,
        artifact_hashes=artifact_hashes,
    )
