from collections.abc import Mapping

from jerrythomas.config.dataset.dataset import DatasetConfig
from jerrythomas.config.tasks.base import DatasetArtifactTask, Task
from jerrythomas.config.tasks.coverage import CoverageTask
from jerrythomas.config.tasks.coverage_stats import CoverageStatsTask
from jerrythomas.config.tasks.dataset import DatasetTask
from jerrythomas.config.tasks.matrix import MatrixTask
from jerrythomas.config.tasks.metadata import MetadataTask
from jerrythomas.config.tasks.scaler import ScalerTask
from jerrythomas.config.tasks.schedule import ScheduleTask
from jerrythomas.config.tasks.series import SeriesTask
from jerrythomas.config.tasks.registry import operation_model
from jerrythomas.services.config_inventory import pipeline_yaml_files
from jerrythomas.services.definitions import DATASET_VARIABLES, ProjectManifest
from jerrythomas.io.yaml import YamlDocument, read_yaml_document


DATASET_ARTIFACT_MODELS: dict[str, type[DatasetArtifactTask]] = {
    "scaler": ScalerTask,
    "series": SeriesTask,
    "metadata": MetadataTask,
    "coverage_stats": CoverageStatsTask,
}


def artifact_kind(task: Task) -> str | None:
    return next(
        (
            kind
            for kind, model in DATASET_ARTIFACT_MODELS.items()
            if isinstance(task, model)
        ),
        None,
    )


def _operation_from_document(
    project: ProjectManifest,
    document: YamlDocument,
    datasets: Mapping[str, DatasetConfig],
) -> Task:
    path = document.path
    operation_id = path.stem
    if operation_id != operation_id.strip().lower():
        raise ValueError(
            f"Operation filename '{path.name}' must use a lowercase operation ID."
        )
    entry = project.resolve_config(document.data, deferred=DATASET_VARIABLES)
    dataset_id = entry.get("dataset")
    selected_dataset = datasets.get(dataset_id) if isinstance(dataset_id, str) else None
    entry = project.resolve_config(
        entry,
        variables=(
            {"dataset_id": dataset_id, "dataset_version": selected_dataset.version}
            if selected_dataset is not None
            else None
        ),
    )
    if "id" in entry:
        raise ValueError(
            f"{path} must not define id; the filename supplies '{operation_id}'."
        )
    kind = entry.get("kind")
    if kind not in ("output", "artifact"):
        raise ValueError(
            f"Operation '{operation_id}' must set kind to output or artifact."
        )
    entrypoint = entry.get("entrypoint")
    if (
        not isinstance(entrypoint, str)
        or not entrypoint
        or entrypoint != entrypoint.strip()
    ):
        raise ValueError(
            f"Operation '{operation_id}' must declare a nonempty entrypoint "
            "without outer whitespace."
        )
    try:
        model = operation_model(kind, entrypoint)
    except ValueError as exc:
        raise ValueError(f"Operation '{operation_id}': {exc}") from exc
    task = model.model_validate({"id": operation_id, **entry})
    if task.dataset is None and (
        artifact_kind(task) is not None
        or isinstance(task, (DatasetTask, CoverageTask, MatrixTask))
    ):
        raise ValueError(f"Operation '{operation_id}' must bind a dataset.")
    if task.dataset is not None:
        if task.dataset not in datasets:
            raise ValueError(
                f"Operation '{operation_id}' references unknown dataset '{task.dataset}'."
            )
        if isinstance(task, ScheduleTask):
            raise ValueError("Schedule operations must not bind a dataset.")
    return task


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
    datasets: Mapping[str, DatasetConfig],
) -> list[Task]:
    """Declared operations plus a default producer for each dataset artifact.

    Duplicate producers of one dataset artifact are reported by the artifact graph.
    """
    operations: list[Task] = []
    by_id: dict[str, Task] = {}
    declared: set[tuple[str | None, str]] = set()
    for document in documents:
        task = _operation_from_document(project, document, datasets)
        if task.id in by_id:
            raise ValueError(f"Duplicate operation ID '{task.id}' at {document.path}")
        by_id[task.id] = task
        operations.append(task)
        kind = artifact_kind(task)
        if kind is not None:
            declared.add((task.dataset, kind))

    defaults: list[Task] = []
    for dataset_id in datasets:
        for kind, model in DATASET_ARTIFACT_MODELS.items():
            if (dataset_id, kind) in declared:
                continue
            operation_id = f"dataset.{dataset_id}.{kind}"
            if operation_id in by_id:
                raise ValueError(
                    f"Operation ID '{operation_id}' is reserved for the default "
                    f"{kind} producer of dataset '{dataset_id}'."
                )
            defaults.append(
                model.model_validate({"id": operation_id, "dataset": dataset_id})
            )
    return [*defaults, *operations]
