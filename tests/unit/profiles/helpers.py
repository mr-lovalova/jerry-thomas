from collections.abc import Sequence
from pathlib import Path

from jerrythomas.artifacts.planning import build_artifact_graph
from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.project import ProjectConfig
from jerrythomas.config.streams import StreamsConfig
from jerrythomas.config.tasks.base import ArtifactTask, OutputTask
from jerrythomas.config.tasks.coverage import CoverageTask
from jerrythomas.config.tasks.dataset import DatasetTask
from jerrythomas.config.tasks.matrix import MatrixTask
from jerrythomas.services.operations import artifact_kind
from jerrythomas.services.definitions import (
    ArtifactHashes,
    ProjectDefinition,
    ProjectManifest,
)


def project_definition(
    project_path: Path,
    *,
    dataset: DatasetConfig | None = None,
    streams: StreamsConfig | None = None,
    artifact_operations: Sequence[ArtifactTask] = (),
    output_operations: Sequence[OutputTask] = (),
    artifact_hash: str = "artifact-hash",
) -> ProjectDefinition:
    project_path = project_path.resolve()
    root = project_path.parent
    resolved_dataset = (
        dataset
        if dataset is not None
        else DatasetConfig(sample=SampleConfig(rounding="ceil", cadence="1h"))
    )
    resolved_streams = streams if streams is not None else StreamsConfig()
    artifact_graph = build_artifact_graph(
        [
            operation.model_copy(update={"dataset": "default"})
            if operation.dataset is None and artifact_kind(operation) is not None
            else operation
            for operation in artifact_operations
        ],
        {"default": resolved_dataset},
        resolved_streams,
    )
    project = ProjectManifest(
        path=project_path,
        config=ProjectConfig.model_validate(
            {
                "schema_version": 7,
                "artifact_revision": 1,
                "paths": {
                    "streams": "streams",
                    "sources": "sources",
                    "datasets": "datasets",
                    "artifacts": "artifacts",
                    "operations": "operations",
                    "profiles": "profiles",
                },
            }
        ),
        variables={},
        environment={},
        stream_dirs=(root / "streams",),
        source_dirs=(root / "sources",),
        dataset_dirs=(root / "datasets",),
        artifacts_root=root / "artifacts",
        operations_dir=root / "operations",
        profiles_dir=root / "profiles",
    )
    return ProjectDefinition(
        project=project,
        datasets={"default": resolved_dataset},
        streams=resolved_streams,
        artifact_graph=artifact_graph,
        output_operations=tuple(
            operation.model_copy(update={"dataset": "default"})
            if operation.dataset is None
            and isinstance(operation, (DatasetTask, CoverageTask, MatrixTask))
            else operation
            for operation in output_operations
        ),
        artifact_hashes=ArtifactHashes(
            {operation_id: artifact_hash for operation_id in artifact_graph.tasks_by_id}
        ),
    )
