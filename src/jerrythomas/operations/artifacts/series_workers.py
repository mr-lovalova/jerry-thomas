from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from jerrythomas.domain.sample_key import SampleKeyContract, SampleKeyValueType
from jerrythomas.execution.events import ProgressSnapshot
from jerrythomas.execution.pipeline import Input, Pipeline
from jerrythomas.execution.runner import run_pipeline
from jerrythomas.pipelines.sort import (
    SortedRuns,
    SortProgress,
    merge_sort_runs,
    write_sort_runs,
)
from jerrythomas.runtime import Runtime
from jerrythomas.services.stream_workers import StreamWorkerProgress, run_stream_jobs
from jerrythomas.services.temp_cleanup import sort_spill_directory

if TYPE_CHECKING:
    from jerrythomas.operations.artifacts.series import _ProjectedRow, _StreamPlan


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


def _prepare_stream(
    runtime: Runtime,
    plan: _StreamPlan,
    temp_root: Path,
    *,
    cadence: timedelta,
) -> _PreparedStream:
    from jerrythomas.operations.artifacts.series import (
        _project_stream,
        _projected_row_key,
    )

    progress = SortProgress()

    def prepare() -> Iterator[_PreparedStream]:
        sample_keys = SampleKeyContract(runtime.require_dataset().sample.keys)
        rows = _project_stream(runtime, plan, sample_keys, cadence)
        try:
            runs = write_sort_runs(
                rows,
                runtime.execution.sort_buffer_bytes,
                _projected_row_key,
                temp_root,
                progress,
            )
        finally:
            rows.close()
        yield _PreparedStream(runs, sample_keys.inferred_types)

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
    plans: Sequence[_StreamPlan],
    sample_keys: SampleKeyContract,
    cadence: timedelta,
    progress: SeriesWorkerProgress,
) -> Iterator[_ProjectedRow]:
    from jerrythomas.operations.artifacts.series import _projected_row_key

    # This parent-owned lock protects every child spill, including after a crash.
    with sort_spill_directory() as temp_root:
        completed = run_stream_jobs(
            runtime,
            plans,
            partial(_prepare_stream, cadence=cadence),
            temp_root,
            progress,
        )
        paths: list[Path] = []
        rows = 0
        for result in completed:
            sample_keys.merge_types(result.key_types)
            paths.extend(result.runs.paths)
            rows += result.runs.rows
        yield from merge_sort_runs(
            SortedRuns(tuple(paths), rows),
            _projected_row_key,
            temp_root,
            progress.start_merging(),
        )
