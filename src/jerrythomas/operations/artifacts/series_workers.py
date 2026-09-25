"""Prepare every stream's sorted series runs, inline or in worker processes."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path

from jerrythomas.domain.sample_key import SampleKeyContract, SampleKeyValueType
from jerrythomas.execution.events import ProgressSnapshot
from jerrythomas.execution.pipeline import Input, Pipeline
from jerrythomas.execution.runner import run_pipeline
from jerrythomas.operations.artifacts.scaler_fit import (
    ScalerFits,
    StreamScalerFitter,
    merge_fits,
)
from jerrythomas.operations.artifacts.series_projection import (
    ProjectedRow,
    StreamPlan,
    project_stream,
    projected_row_key,
)
from jerrythomas.pipelines.sort import (
    SortedRuns,
    SortProgress,
    merge_sort_runs,
    write_sort_runs,
)
from jerrythomas.runtime import Runtime
from jerrythomas.services.stream_workers import StreamWorkerProgress, run_stream_jobs
from jerrythomas.services.temp_cleanup import sort_spill_directory


class SeriesWorkerProgress(StreamWorkerProgress):
    def __init__(self) -> None:
        super().__init__()
        self._sort_progress: SortProgress | None = None

    def snapshot(self, output_items: int) -> ProgressSnapshot:
        if self._sort_progress is not None:
            return self._sort_progress.snapshot(output_items)
        return super().snapshot(output_items)

    def start_merging(self) -> SortProgress:
        self._sort_progress = SortProgress()
        return self._sort_progress


@dataclass(frozen=True)
class _PreparedStream:
    runs: SortedRuns
    key_types: tuple[SampleKeyValueType | None, ...]
    # None without scaler fitting, or when the stream's fitting was abandoned.
    scaler_fits: ScalerFits | None


def _prepare_stream(
    runtime: Runtime,
    plan: StreamPlan,
    job_dir: Path,
    *,
    cadence: timedelta,
    fit_scaler: bool,
) -> _PreparedStream:
    progress = SortProgress()

    def prepare() -> Iterator[_PreparedStream]:
        dataset = runtime.require_dataset()
        sample_keys = SampleKeyContract(dataset.sample.keys)
        fitter = StreamScalerFitter(dataset) if fit_scaler else None
        rows = project_stream(runtime, plan, sample_keys, cadence, fitter)
        try:
            runs = write_sort_runs(
                rows,
                runtime.execution.sort_buffer_bytes,
                projected_row_key,
                job_dir,
                progress,
            )
        finally:
            rows.close()
        yield _PreparedStream(
            runs,
            sample_keys.inferred_types,
            None if fitter is None or fitter.abandoned else fitter.fits(),
        )

    (result,) = run_pipeline(
        runtime,
        Pipeline(
            name=f"prepare:{plan.stream_id}",
            input=Input(name="order_series", open=prepare, progress=progress.snapshot),
        ),
    )
    return result


def order_streams(
    runtime: Runtime,
    plans: Sequence[StreamPlan],
    sample_keys: SampleKeyContract,
    cadence: timedelta,
    progress: SeriesWorkerProgress,
    on_scaler_fits: Callable[[ScalerFits], None] | None = None,
) -> Iterator[ProjectedRow]:
    """Yield every stream's projected rows ordered by sample key and time.

    With ``on_scaler_fits``, each stream pass also fits the dataset's scaled series;
    the merged fits are handed over once every stream is prepared, before any row.
    """
    # This parent-owned lock protects every child spill, including after a crash.
    with sort_spill_directory() as temp_root:
        prepared = run_stream_jobs(
            runtime,
            plans,
            partial(
                _prepare_stream,
                cadence=cadence,
                fit_scaler=on_scaler_fits is not None,
            ),
            temp_root,
            progress,
        )
        paths: list[Path] = []
        rows = 0
        for result in prepared:
            sample_keys.merge_types(result.key_types)
            paths.extend(result.runs.paths)
            rows += result.runs.rows
        stream_fits = [result.scaler_fits for result in prepared]
        # A stream without fits hit a value the scaler cannot fit; the scaler
        # then refits from streams and reports it.
        if on_scaler_fits is not None and all(fits is not None for fits in stream_fits):
            on_scaler_fits(
                merge_fits(
                    runtime.require_dataset(),
                    (fits for fits in stream_fits if fits is not None),
                )
            )
        yield from merge_sort_runs(
            SortedRuns(tuple(paths), rows),
            projected_row_key,
            temp_root,
            progress.start_merging(),
        )
