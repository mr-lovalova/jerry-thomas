from typing import Any

from datapipeline.config.tasks.base import ArtifactTask, RuntimeTask, Task
from datapipeline.config.tasks.coverage import CoverageTask
from datapipeline.config.tasks.coverage_stats import CoverageStatsTask
from datapipeline.config.tasks.dataset import DatasetTask
from datapipeline.config.tasks.matrix import MatrixTask
from datapipeline.config.tasks.metadata import MetadataTask
from datapipeline.config.tasks.scaler import ScalerTask
from datapipeline.config.tasks.schedule import ScheduleTask
from datapipeline.config.tasks.series import SeriesTask
from datapipeline.services.config_inventory import pipeline_yaml_files
from datapipeline.services.definitions import ProjectManifest
from datapipeline.utils.load import YamlDocument, read_yaml_document

CORE_OPERATION_MODELS: dict[str, type[Task]] = {
    "scaler": ScalerTask,
    "series": SeriesTask,
    "metadata": MetadataTask,
    "coverage_stats": CoverageStatsTask,
    "dataset": DatasetTask,
    "coverage": CoverageTask,
    "matrix": MatrixTask,
}
CORE_RUNTIME_MODELS: dict[str, type[RuntimeTask[Any]]] = {
    "core.runtime.dataset": DatasetTask,
    "core.runtime.coverage": CoverageTask,
    "core.runtime.matrix": MatrixTask,
}
CORE_ARTIFACT_IDS_BY_ENTRYPOINT = {
    "core.artifact.scaler": "scaler",
    "core.artifact.series": "series",
    "core.artifact.metadata": "metadata",
    "core.artifact.coverage_stats": "coverage_stats",
}


def _core_operations() -> list[Task]:
    return [
        model.model_validate({"id": operation_id})
        for operation_id, model in CORE_OPERATION_MODELS.items()
    ]


def _custom_operation(operation_id: str, entry: dict[str, object]) -> Task:
    kind = entry.get("kind")
    entrypoint = entry.get("entrypoint")
    if isinstance(entrypoint, str) and entrypoint != entrypoint.strip():
        raise ValueError(
            f"Custom operation '{operation_id}' entrypoint must not contain "
            "outer whitespace."
        )
    if kind == "runtime":
        model: type[RuntimeTask[Any]] = RuntimeTask
        if isinstance(entrypoint, str):
            core_model = CORE_RUNTIME_MODELS.get(entrypoint)
            if core_model is not None:
                model = core_model
            elif entrypoint.startswith("core.runtime."):
                supported = ", ".join(sorted(CORE_RUNTIME_MODELS))
                raise ValueError(
                    f"Unsupported core runtime entrypoint '{entrypoint}'. "
                    f"Supported entrypoints: {supported}."
                )
        return model.model_validate({"id": operation_id, **entry})
    if kind != "artifact":
        raise ValueError(
            f"Custom operation '{operation_id}' must set kind to artifact or runtime."
        )
    if entrypoint == "core.artifact.schedule":
        return ScheduleTask.model_validate({"id": operation_id, **entry})
    core_operation_id = (
        CORE_ARTIFACT_IDS_BY_ENTRYPOINT.get(entrypoint)
        if isinstance(entrypoint, str)
        else None
    )
    if core_operation_id is not None:
        raise ValueError(
            f"Artifact entrypoint '{entrypoint}' is reserved for core operation "
            f"'{core_operation_id}'. Override {core_operation_id}.yaml instead."
        )
    return ArtifactTask.model_validate({"id": operation_id, **entry})


def _operation_from_document(
    project: ProjectManifest,
    document: YamlDocument,
) -> Task:
    path = document.path
    operation_id = path.stem
    if operation_id != operation_id.strip().lower():
        raise ValueError(
            f"Operation filename '{path.name}' must use a lowercase operation ID."
        )
    entry = project.resolve_config(document.data)
    if "id" in entry:
        raise ValueError(
            f"{path} must not define id; the filename supplies '{operation_id}'."
        )

    model = CORE_OPERATION_MODELS.get(operation_id)
    if model is not None:
        if "kind" in entry or "entrypoint" in entry:
            raise ValueError(
                f"Core operation '{operation_id}' cannot replace its kind or entrypoint."
            )
        return model.model_validate({"id": operation_id, **entry})
    return _custom_operation(operation_id, entry)


def operation_documents(project: ProjectManifest) -> tuple[YamlDocument, ...]:
    root = project.operations_dir
    if root is None:
        return ()
    try:
        paths = pipeline_yaml_files(root)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise FileNotFoundError(f"operations directory not found: {root}") from exc
    return tuple(read_yaml_document(path) for path in paths)


def operations_from_documents(
    project: ProjectManifest,
    documents: tuple[YamlDocument, ...],
) -> list[Task]:
    paths_by_id: dict[str, list[str]] = {}
    for document in documents:
        paths_by_id.setdefault(document.path.stem, []).append(str(document.path))
    duplicates = {
        operation_id: paths
        for operation_id, paths in paths_by_id.items()
        if len(paths) > 1
    }
    if duplicates:
        details = "; ".join(
            f"{operation_id} ({', '.join(paths)})"
            for operation_id, paths in sorted(duplicates.items())
        )
        raise ValueError(f"Duplicate operation ids are not allowed: {details}")

    overrides = [_operation_from_document(project, document) for document in documents]
    overrides_by_id = {operation.id: operation for operation in overrides}
    specs = [
        overrides_by_id.pop(operation.id, operation) for operation in _core_operations()
    ]
    specs.extend(overrides_by_id.values())
    return specs
