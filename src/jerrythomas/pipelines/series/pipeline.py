from collections.abc import Iterator
from functools import partial

from jerrythomas.config.dataset.series import SeriesConfig
from jerrythomas.domain.sample_key import SampleKeyContract
from jerrythomas.domain.series import SeriesRecord, SeriesSequence
from jerrythomas.execution.pipeline import Pipeline, Stage
from jerrythomas.execution.runner import run_pipeline
from jerrythomas.pipelines.series.stages import (
    order_series,
    project_series,
    sequence_series,
)
from jerrythomas.pipelines.series.projector import SeriesProjector
from jerrythomas.pipelines.sort import SortProgress
from jerrythomas.pipelines.stream.pipeline import build_stream_pipeline
from jerrythomas.runtime import Runtime, require_runtime_stream


def run_series_pipeline(
    runtime: Runtime,
    cfg: SeriesConfig,
) -> Iterator[SeriesRecord | SeriesSequence]:
    return run_pipeline(
        runtime,
        build_series_pipeline(runtime, cfg),
    )


def build_series_pipeline(
    runtime: Runtime,
    cfg: SeriesConfig,
) -> Pipeline:
    record_pipeline = build_stream_pipeline(runtime, cfg.stream)
    return record_pipeline.continue_as(
        f"series:{cfg.id}",
        build_series_stages(runtime, cfg),
    )


def build_series_stages(
    runtime: Runtime,
    config: SeriesConfig,
) -> tuple[Stage, ...]:
    stream = require_runtime_stream(runtime, config.stream)
    sample = runtime.require_dataset().sample
    sample_keys = tuple(sample.keys)
    projector = SeriesProjector(
        stream.partition_by,
        SampleKeyContract(sample_keys),
        (config,),
    )
    stages = [
        Stage(
            name="project_series",
            apply=partial(project_series, projector, sample),
        ),
    ]
    if config.sequence is not None:
        stages.append(
            Stage(
                name="sequence_series",
                apply=partial(sequence_series, config.sequence),
            )
        )
    if stream.partition_by or sample_keys:
        sort_progress = SortProgress()
        stages.append(
            Stage(
                name="order_series",
                apply=partial(
                    order_series,
                    runtime.execution.sort_buffer_bytes,
                    sample.cadence,
                    sample.rounding,
                    sample_keys,
                    sort_progress,
                ),
                progress=sort_progress.snapshot,
            ),
        )
    return tuple(stages)
