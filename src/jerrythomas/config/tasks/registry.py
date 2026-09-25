from .base import Task
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
