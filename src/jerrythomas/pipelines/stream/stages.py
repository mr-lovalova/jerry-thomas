from collections import Counter
from collections.abc import Iterator, Sequence
from functools import partial
from typing import Any

from jerrythomas.artifacts.registry import ArtifactSpec
from jerrythomas.artifacts.schedule import (
    Schedule,
    read_schedule,
    schedule_partition_by_from_metadata,
)
from jerrythomas.config.interpolation import normalize_interpolated_args
from jerrythomas.config.transforms import (
    AggregateSumConfig,
    CollapseConfig,
    CustomTransformConfig,
    DedupeConfig,
    DeriveConfig,
    EwmMeanConfig,
    EnsureCadenceConfig,
    EnsureScheduleConfig,
    FillConfig,
    FillMissingConfig,
    FloorTimeConfig,
    ForwardFillConfig,
    ForwardSumConfig,
    LagConfig,
    LeadConfig,
    Log1pConfig,
    LogConfig,
    PreprocessConfig,
    RollingConfig,
    RollingQuantileConfig,
    RollingOlsConfig,
    RollingSlopeConfig,
    ShiftTimeConfig,
    TransformConfig,
    WhereConfig,
)
from jerrythomas.execution.pipeline import Stage, StageOp
from jerrythomas.plugins import TRANSFORMS_EP, load_entrypoint
from jerrythomas.runtime import Runtime
from jerrythomas.transforms.stream.aggregate import AggregateSumTransform
from jerrythomas.transforms.stream.collapse import CollapseTransform
from jerrythomas.transforms.stream.dedupe import DedupeTransform
from jerrythomas.transforms.stream.derive import DeriveTransform
from jerrythomas.transforms.stream.ewm import EwmMeanTransform
from jerrythomas.transforms.stream.fill import (
    FillMissingTransform,
    ForwardFillTransform,
    StatisticalFillTransform,
)
from jerrythomas.transforms.stream.forward_sum import ForwardSumTransform
from jerrythomas.transforms.stream.lag import LagTransform
from jerrythomas.transforms.stream.lead import LeadTransform
from jerrythomas.transforms.stream.logarithm import Log1pTransform, LogTransform
from jerrythomas.transforms.stream.rolling import (
    RollingQuantileTransform,
    RollingTransform,
)
from jerrythomas.transforms.stream.rolling_ols import RollingOlsTransform
from jerrythomas.transforms.stream.rolling_slope import RollingSlopeTransform
from jerrythomas.transforms.stream.time_completion import (
    EnsureCadenceTransform,
    EnsureScheduleTransform,
)
from jerrythomas.transforms.time import FloorTimeTransform, ShiftTimeTransform
from jerrythomas.transforms.where import WhereTransform


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
            stage_op = DedupeTransform(partition_by).apply
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
        elif isinstance(operation, AggregateSumConfig):
            stage_op = AggregateSumTransform(
                operation.field,
                partition_by,
                operation.count_to,
            ).apply
        elif isinstance(operation, FillMissingConfig):
            stage_op = FillMissingTransform(
                operation.field,
                operation.value,
                operation.to,
            ).apply
        elif isinstance(operation, EwmMeanConfig):
            stage_op = EwmMeanTransform(
                operation.field,
                operation.alpha,
                partition_by,
                operation.to,
                operation.min_samples,
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
        elif isinstance(operation, RollingQuantileConfig):
            stage_op = RollingQuantileTransform(
                operation.field,
                operation.window,
                operation.quantile,
                partition_by,
                operation.to,
                operation.min_samples,
            ).apply
        elif isinstance(operation, RollingSlopeConfig):
            stage_op = RollingSlopeTransform(
                operation.x,
                operation.y,
                operation.window,
                partition_by,
                operation.to,
            ).apply
        elif isinstance(operation, RollingOlsConfig):
            stage_op = RollingOlsTransform(
                operation.y,
                operation.x,
                operation.window,
                operation.coefficient,
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
        elif isinstance(operation, CustomTransformConfig):
            factory = load_entrypoint(TRANSFORMS_EP, operation.entrypoint)
            instance = factory(
                normalize_interpolated_args(operation.args),
                partition_by,
            )
            stage_op_candidate = getattr(instance, "apply", None)
            if not callable(stage_op_candidate):
                raise TypeError(
                    f"Custom transform '{operation.entrypoint}' must return "
                    "an object with an apply(records) method; got "
                    f"{type(instance).__name__}."
                )
            stage_op = stage_op_candidate
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
