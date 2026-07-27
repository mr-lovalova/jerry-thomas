from collections.abc import Iterator
from functools import partial

from datapipeline.config.dataset.series import SeriesConfig
from datapipeline.domain.sample_key import SampleKeyContract
from datapipeline.domain.series import SeriesRecord, SeriesSequence
from datapipeline.execution.pipeline import Pipeline, Stage
from datapipeline.execution.runner import run_pipeline
from datapipeline.pipelines.series.stages import (
    order_series,
    project_series,
    sequence_series,
)
from datapipeline.pipelines.series.projector import SeriesProjector
from datapipeline.pipelines.sort import SortProgress
from datapipeline.pipelines.stream.pipeline import build_stream_pipeline
from datapipeline.runtime import Runtime, require_runtime_stream


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
    sample = runtime.dataset.sample
    sample_keys = tuple(sample.keys)
    projector = SeriesProjector(
        stream.partition_by,
        SampleKeyContract(sample_keys),
        (config,),
    )
    stages = [
        Stage(
            name="project_series",
            apply=partial(project_series, projector),
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
                    sample_keys,
                    sort_progress,
                ),
                progress=sort_progress.snapshot,
            ),
        )
    return tuple(stages)
