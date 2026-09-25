"""Project one stream's records into series rows keyed by sample time and entity."""

from collections import defaultdict
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from jerrythomas.config.dataset.series import SeriesConfig
from jerrythomas.domain.sample_key import SampleKeyContract
from jerrythomas.domain.series import SeriesSequence
from jerrythomas.execution.pipeline import Pipeline, Stage
from jerrythomas.execution.runner import run_pipeline
from jerrythomas.operations.artifacts.scaler_fit import StreamScalerFitter
from jerrythomas.pipelines.series.projector import SeriesProjector
from jerrythomas.pipelines.series.stages import SeriesSequencer
from jerrythomas.pipelines.stream.pipeline import build_stream_pipeline
from jerrythomas.runtime import Runtime, require_runtime_stream
from jerrythomas.transforms.utils import record_establishes_domain
from jerrythomas.utils.time import round_time_to_cadence


@dataclass(frozen=True)
class StreamPlan:
    stream_id: str
    features: tuple[SeriesConfig, ...]
    targets: tuple[SeriesConfig, ...]


@dataclass(frozen=True)
class ProjectedScalar:
    id: str
    value: Any
    establishes_domain: bool


@dataclass(frozen=True)
class ProjectedSequence:
    id: str
    values: list[Any]
    establishes_domain: bool


ProjectedValue = ProjectedScalar | ProjectedSequence


@dataclass(frozen=True)
class ProjectedRow:
    key: tuple[Any, ...]
    time: datetime
    features: tuple[ProjectedValue, ...]
    targets: tuple[ProjectedValue, ...]


def projected_row_key(row: ProjectedRow) -> tuple[tuple[Any, ...], datetime]:
    return row.key, row.time


def stream_plans(
    features: Sequence[SeriesConfig],
    targets: Sequence[SeriesConfig],
) -> tuple[StreamPlan, ...]:
    feature_configs: dict[str, list[SeriesConfig]] = defaultdict(list)
    target_configs: dict[str, list[SeriesConfig]] = defaultdict(list)
    for config in features:
        feature_configs[config.stream].append(config)
    for config in targets:
        target_configs[config.stream].append(config)

    return tuple(
        StreamPlan(
            stream_id=stream_id,
            features=tuple(feature_configs[stream_id]),
            targets=tuple(target_configs[stream_id]),
        )
        for stream_id in dict.fromkeys((*feature_configs, *target_configs))
    )


def project_stream(
    runtime: Runtime,
    plan: StreamPlan,
    sample_keys: SampleKeyContract,
    cadence: timedelta,
    scaler_fitter: StreamScalerFitter | None = None,
) -> Generator[ProjectedRow, None, None]:
    """Yield projected rows in stream order.

    When ``scaler_fitter`` is given, it observes each record's raw scaled values
    before sequencing, exactly as a separate scaler pass over the stream would.
    A value the scaler cannot fit abandons the fitter instead of failing the series.
    """
    stream = require_runtime_stream(runtime, plan.stream_id)
    rounding = runtime.require_dataset().sample.rounding
    configs = (*plan.features, *plan.targets)
    feature_ids = {config.id for config in plan.features}
    scaled_positions = tuple(
        index for index, config in enumerate(configs) if config.scale is not None
    )
    fitter = scaler_fitter if scaled_positions else None
    projector = SeriesProjector(stream.partition_by, sample_keys, configs)
    sequencers = {
        config.id: SeriesSequencer(config.sequence)
        for config in configs
        if config.sequence is not None
    }

    def project(records: Iterator[Any]) -> Iterator[ProjectedRow]:
        for record in records:
            projected = tuple(projector.project(record))
            sample_time = round_time_to_cadence(record.time, cadence, rounding)
            if fitter is not None:
                scaled = tuple(projected[index] for index in scaled_positions)
                fitter.observe_or_abandon((sample_time, *scaled[0].entity_key), scaled)

            features: list[ProjectedValue] = []
            targets: list[ProjectedValue] = []
            row_key: tuple[Any, ...] | None = None
            row_time: datetime | None = None
            for config, series_record in zip(configs, projected, strict=True):
                sequencer = sequencers.get(config.id)
                result = (
                    series_record
                    if sequencer is None
                    else sequencer.append(series_record)
                )
                if result is None:
                    continue

                key = (sample_time, *result.entity_key)
                if row_key is None:
                    row_key = key
                    row_time = result.time
                elif key != row_key or result.time != row_time:
                    raise RuntimeError(
                        f"Stream '{plan.stream_id}' projected one record into different "
                        "sample keys."
                    )

                value: ProjectedValue
                if isinstance(result, SeriesSequence):
                    value = ProjectedSequence(
                        result.id,
                        result.values,
                        record_establishes_domain(result),
                    )
                else:
                    value = ProjectedScalar(
                        result.id,
                        result.value,
                        record_establishes_domain(result),
                    )
                if config.id in feature_ids:
                    features.append(value)
                else:
                    targets.append(value)

            if row_key is not None and row_time is not None:
                yield ProjectedRow(
                    key=row_key,
                    time=row_time,
                    features=tuple(features),
                    targets=tuple(targets),
                )

    record_pipeline = build_stream_pipeline(runtime, plan.stream_id)
    pipeline = Pipeline(
        name=f"series:{plan.stream_id}",
        input=record_pipeline.input,
        stages=(
            *record_pipeline.stages,
            Stage(name="project_series", apply=project),
        ),
        summary=record_pipeline.summary,
    )
    projected = run_pipeline(runtime, pipeline)
    try:
        yield from projected
    finally:
        close = getattr(projected, "close", None)
        if callable(close):
            close()
