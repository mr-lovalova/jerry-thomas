from collections import Counter
from collections.abc import Iterator, Sequence
from functools import partial
from typing import Any

from datapipeline.artifacts.registry import ArtifactSpec
from datapipeline.artifacts.schedule import (
    Schedule,
    read_schedule,
    schedule_partition_by_from_metadata,
)
from datapipeline.config.transforms import (
    CollapseConfig,
    DedupeConfig,
    DeriveConfig,
    EnsureCadenceConfig,
    EnsureScheduleConfig,
    FillConfig,
    FloorTimeConfig,
    ForwardFillConfig,
    ForwardSumConfig,
    LagConfig,
    LeadConfig,
    Log1pConfig,
    LogConfig,
    PreprocessConfig,
    RollingConfig,
    RollingSlopeConfig,
    ShiftTimeConfig,
    TransformConfig,
    WhereConfig,
)
from datapipeline.execution.pipeline import Stage, StageOp
from datapipeline.runtime import Runtime
from datapipeline.transforms.stream.dedupe import DedupeTransform
from datapipeline.transforms.stream.derive import DeriveTransform
from datapipeline.transforms.stream.fill import (
    ForwardFillTransform,
    StatisticalFillTransform,
)
from datapipeline.transforms.stream.forward_sum import ForwardSumTransform
from datapipeline.transforms.stream.collapse import CollapseTransform
from datapipeline.transforms.stream.lag import LagTransform
from datapipeline.transforms.stream.lead import LeadTransform
from datapipeline.transforms.stream.logarithm import Log1pTransform, LogTransform
from datapipeline.transforms.stream.rolling import RollingTransform
from datapipeline.transforms.stream.rolling_slope import RollingSlopeTransform
from datapipeline.transforms.stream.time_completion import (
    EnsureCadenceTransform,
    EnsureScheduleTransform,
)
from datapipeline.transforms.time import FloorTimeTransform, ShiftTimeTransform
from datapipeline.transforms.where import WhereTransform


def build_preprocess_stages(
    operations: Sequence[PreprocessConfig],
) -> tuple[Stage, ...]:
    configured = tuple(operations)
    totals = Counter(operation.operation for operation in configured)
    occurrences: Counter[str] = Counter()
    stages: list[Stage] = []
    for operation in configured:
        occurrences[operation.operation] += 1
        stage_name = f"preprocess_{operation.operation}"
        if totals[operation.operation] > 1:
            stage_name += f"_{occurrences[operation.operation]}"

        if isinstance(operation, WhereConfig):
            stage_op: StageOp = WhereTransform(
                operation.field,
                operation.operator,
                operation.comparand,
            ).apply
        elif isinstance(operation, FloorTimeConfig):
            stage_op = FloorTimeTransform(operation.cadence).apply
        elif isinstance(operation, ShiftTimeConfig):
            stage_op = ShiftTimeTransform(operation.by).apply
        else:
            raise TypeError(
                f"Unsupported preprocess config: {type(operation).__name__}"
            )

        stages.append(
            Stage(
                name=stage_name,
                apply=stage_op,
            )
        )
    return tuple(stages)


def build_transform_stages(
    runtime: Runtime,
    operations: Sequence[TransformConfig],
    partition_by: tuple[str, ...],
) -> tuple[Stage, ...]:
    configured = tuple(operations)
    totals = Counter(operation.operation for operation in configured)
    occurrences: Counter[str] = Counter()
    stages: list[Stage] = []
    for operation in configured:
        occurrences[operation.operation] += 1
        stage_name: str = operation.operation
        if totals[operation.operation] > 1:
            stage_name = f"{stage_name}_{occurrences[operation.operation]}"

        stage_op: StageOp
        if isinstance(operation, WhereConfig):
            stage_op = WhereTransform(
                operation.field,
                operation.operator,
                operation.comparand,
            ).apply
        elif isinstance(operation, DedupeConfig):
            stage_op = DedupeTransform().apply
        elif isinstance(operation, LagConfig):
            stage_op = LagTransform(
                operation.field,
                operation.periods,
                partition_by,
                operation.to,
            ).apply
        elif isinstance(operation, LeadConfig):
            stage_op = LeadTransform(
                operation.field,
                operation.periods,
                partition_by,
                operation.to,
            ).apply
        elif isinstance(operation, ForwardSumConfig):
            stage_op = ForwardSumTransform(
                operation.field,
                operation.window,
                partition_by,
                operation.to,
            ).apply
        elif isinstance(operation, EnsureCadenceConfig):
            stage_op = EnsureCadenceTransform(
                operation.cadence,
                partition_by,
            ).apply
        elif isinstance(operation, EnsureScheduleConfig):
            stage_op = partial(
                apply_schedule,
                runtime,
                operation,
                partition_by,
            )
        elif isinstance(operation, FillConfig):
            stage_op = StatisticalFillTransform(
                operation.field,
                operation.window,
                operation.statistic,
                partition_by,
                operation.to,
                operation.min_samples,
            ).apply
        elif isinstance(operation, ForwardFillConfig):
            stage_op = ForwardFillTransform(
                operation.field,
                partition_by,
                operation.to,
            ).apply
        elif isinstance(operation, CollapseConfig):
            stage_op = CollapseTransform(
                partition_by,
                operation.keep,
            ).apply
        elif isinstance(operation, RollingConfig):
            stage_op = RollingTransform(
                operation.field,
                operation.window,
                partition_by,
                operation.to,
                operation.min_samples,
                operation.statistic,
            ).apply
        elif isinstance(operation, RollingSlopeConfig):
            stage_op = RollingSlopeTransform(
                operation.x,
                operation.y,
                operation.window,
                partition_by,
                operation.to,
            ).apply
        elif isinstance(operation, LogConfig):
            stage_op = LogTransform(operation.field, operation.to).apply
        elif isinstance(operation, Log1pConfig):
            stage_op = Log1pTransform(operation.field, operation.to).apply
        elif isinstance(operation, DeriveConfig):
            stage_op = DeriveTransform(
                operation.left,
                operation.operator,
                operation.to,
                right_field=operation.right_field,
                right_value=operation.right_value,
            ).apply
        else:
            raise TypeError(f"Unsupported transform config: {type(operation).__name__}")

        stages.append(
            Stage(
                name=stage_name,
                apply=stage_op,
            )
        )
    return tuple(stages)


def apply_schedule(
    runtime: Runtime,
    operation: EnsureScheduleConfig,
    partition_fields: tuple[str, ...],
    records: Iterator[Any],
) -> Iterator[Any]:
    partition_by = schedule_partition_by_from_metadata(
        operation.schedule,
        runtime.artifacts.require(operation.schedule).meta,
    )
    schedule = runtime.artifacts.load(
        ArtifactSpec[Schedule](
            key=operation.schedule,
            loader=partial(read_schedule, partition_by=partition_by),
        )
    )
    return EnsureScheduleTransform(
        schedule,
        partition_fields,
    ).apply(records)
