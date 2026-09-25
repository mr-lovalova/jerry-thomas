from typing import Literal

from .base import ArtifactTask, PluginOutputTask, Task
from .coverage import CoverageTask
from .coverage_stats import CoverageStatsTask
from .dataset import DatasetTask
from .matrix import MatrixTask
from .metadata import MetadataTask
from .scaler import ScalerTask
from .schedule import ScheduleTask
from .series import SeriesTask
from .stream import StreamTask


CORE_OPERATION_MODELS: dict[str, type[Task]] = {
    "core.records": StreamTask,
    "core.dataset": DatasetTask,
    "core.availability_matrix": MatrixTask,
    "core.coverage_report": CoverageTask,
    "core.dataset_series": SeriesTask,
    "core.scaler": ScalerTask,
    "core.dataset_metadata": MetadataTask,
    "core.coverage_statistics": CoverageStatsTask,
    "core.schedule": ScheduleTask,
}

OperationKind = Literal["output", "artifact"]


def operation_model(kind: OperationKind, entrypoint: str) -> type[Task]:
    """Return the task model for an operation's kind and entrypoint.

    ``core.*`` entrypoints name built-ins, whose kind is fixed; anything else is a
    plugin operation of the declared kind.
    """
    model = CORE_OPERATION_MODELS.get(entrypoint)
    if model is None:
        if entrypoint.startswith("core."):
            raise ValueError(f"Unknown built-in entrypoint '{entrypoint}'.")
        return PluginOutputTask if kind == "output" else ArtifactTask
    builtin_kind = "artifact" if issubclass(model, ArtifactTask) else "output"
    if kind != builtin_kind:
        raise ValueError(
            f"Entrypoint '{entrypoint}' is a built-in {builtin_kind} operation; "
            f"set kind: {builtin_kind}."
        )
    return model
