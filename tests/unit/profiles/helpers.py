from collections.abc import Sequence
from pathlib import Path

from datapipeline.artifacts.planning import build_artifact_graph
from datapipeline.config.dataset.dataset import DatasetConfig, SampleConfig
from datapipeline.config.project import ProjectConfig
from datapipeline.config.streams import StreamsConfig
from datapipeline.config.tasks.base import ArtifactTask, RuntimeTask
from datapipeline.services.definitions import (
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
    runtime_operations: Sequence[RuntimeTask] = (),
    artifact_hash: str = "artifact-hash",
) -> ProjectDefinition:
    project_path = project_path.resolve()
    root = project_path.parent
    resolved_dataset = (
        dataset
        if dataset is not None
        else DatasetConfig(sample=SampleConfig(cadence="1h"))
    )
    resolved_streams = streams if streams is not None else StreamsConfig()
    artifact_graph = build_artifact_graph(
        artifact_operations,
        resolved_dataset,
        resolved_streams,
    )
    project = ProjectManifest(
        path=project_path,
        config=ProjectConfig.model_validate(
            {
                "schema_version": 4,
                "artifact_revision": 1,
                "paths": {
                    "streams": "streams",
                    "sources": "sources",
                    "dataset": "dataset.yaml",
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
        dataset_path=root / "dataset.yaml",
        artifacts_root=root / "artifacts",
        operations_dir=root / "operations",
        profiles_dir=root / "profiles",
    )
    return ProjectDefinition(
        project=project,
        dataset=resolved_dataset,
        streams=resolved_streams,
        artifact_graph=artifact_graph,
        runtime_operations=tuple(runtime_operations),
        artifact_hashes=ArtifactHashes(
            {operation_id: artifact_hash for operation_id in artifact_graph.tasks_by_id}
        ),
    )
