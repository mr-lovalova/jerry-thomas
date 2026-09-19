from __future__ import annotations

import logging
import multiprocessing
import signal
import tempfile
import threading
import traceback
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import TYPE_CHECKING

from jerrythomas.domain.sample_key import SampleKeyContract, SampleKeyValueType
from jerrythomas.execution.events import ProgressSnapshot
from jerrythomas.execution.observability import (
    ExecutionEvent,
    ExecutionMessage,
    emit_execution_message,
    execution_observer,
)
from jerrythomas.pipelines.sort import (
    SortedRuns,
    SortProgress,
    merge_sort_runs,
    write_sort_runs,
)
from jerrythomas.runtime import Runtime, RuntimeSnapshot
from jerrythomas.services.runtime_compiler import (
    compile_runtime_snapshot,
    snapshot_runtime,
)
from jerrythomas.services.temp_cleanup import sort_spill_directory

if TYPE_CHECKING:
    from jerrythomas.operations.artifacts.series import _ProjectedRow, _StreamPlan


_KeyTypes = tuple[SampleKeyValueType | None, ...]


class StreamWorkerError(RuntimeError):
    """A stream process failed before its output could be consumed."""


class StreamWorkerProgress:
    def __init__(self) -> None:
        self._detail = "preparing streams"
        self._sort_progress: SortProgress | None = None

    def update(self, completed: int, total: int, active: int) -> None:
        workers = "worker" if active == 1 else "workers"
        self._detail = f"{completed}/{total} streams complete, {active} {workers}"

    def snapshot(self, output_items: int) -> ProgressSnapshot:
        if self._sort_progress is not None:
            return self._sort_progress.snapshot(output_items)
        return ProgressSnapshot(
            completed=output_items, phase="preparing", detail=self._detail
        )

    def start_merging(self) -> SortProgress:
        self._sort_progress = SortProgress()
        return self._sort_progress


@dataclass(frozen=True)
class _Complete:
    runs: SortedRuns
    key_types: _KeyTypes


@dataclass(frozen=True)
class _Failure:
    error: str
    traceback: str


@dataclass(frozen=True)
class _Log:
    name: str
    level: int
    message: str


@dataclass
class _Worker:
    index: int
    stream_id: str
    process: BaseProcess
    control: Connection
    complete: _Complete | None = None
    control_closed: bool = False


class _WorkerLogHandler(logging.Handler):
    def __init__(self, connection: Connection, lock: threading.Lock) -> None:
        super().__init__()
        self.connection = connection
        self.send_lock = lock

    def emit(self, record: logging.LogRecord) -> None:
        with self.send_lock:
            self.connection.send(_Log(record.name, record.levelno, self.format(record)))


def _prepare_stream(
    runtime: Runtime,
    plan: _StreamPlan,
    cadence: timedelta,
    temp_root: Path,
) -> _Complete:
    from jerrythomas.operations.artifacts.series import (
        _project_stream,
        _projected_row_key,
    )

    sample_keys = SampleKeyContract(runtime.dataset.sample.keys)
    rows = _project_stream(runtime, plan, sample_keys, cadence)
    try:
        runs = write_sort_runs(
            rows,
            runtime.execution.sort_buffer_bytes,
            _projected_row_key,
            temp_root,
        )
    finally:
        rows.close()
    return _Complete(runs, sample_keys.inferred_types)


def _run_stream(
    snapshot: RuntimeSnapshot,
    plan: _StreamPlan,
    cadence: timedelta,
    temp_root: Path,
    control: Connection,
    log_level: int,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    tempfile.tempdir = str(temp_root)
    send_lock = threading.Lock()
    root_logger = logging.getLogger()
    root_logger.handlers = [_WorkerLogHandler(control, send_lock)]
    root_logger.setLevel(log_level)

    def observe(event: ExecutionEvent) -> None:
        # Concurrent pipeline events cannot share the parent's ordered UI stack.
        if isinstance(event, ExecutionMessage):
            with send_lock:
                control.send(event)

    try:
        with execution_observer(observe):
            runtime = compile_runtime_snapshot(snapshot)
            runtime.heartbeat_interval_seconds = 0
            runtime.observe_node_events = False
            result = _prepare_stream(runtime, plan, cadence, temp_root)
        with send_lock:
            control.send(result)
    except BaseException as error:
        failure = _Failure(f"{type(error).__name__}: {error}", traceback.format_exc())
        with send_lock:
            control.send(failure)
    finally:
        control.close()


def order_streams(
    runtime: Runtime,
    plans: Sequence[_StreamPlan],
    sample_keys: SampleKeyContract,
    cadence: timedelta,
    progress: StreamWorkerProgress,
) -> Iterator[_ProjectedRow]:
    from jerrythomas.operations.artifacts.series import _projected_row_key

    active: list[_Worker] = []
    pending = iter(enumerate(plans))
    completed: dict[int, _Complete] = {}

    # This parent-owned lock protects every child spill, including after a crash.
    with sort_spill_directory() as temp_root:
        try:
            if runtime.execution.workers == 1 or len(plans) <= 1:
                for index, plan in pending:
                    progress.update(index, len(plans), 1)
                    worker_root = temp_root / str(index)
                    worker_root.mkdir()
                    completed[index] = _prepare_stream(
                        runtime, plan, cadence, worker_root
                    )
            else:
                context = multiprocessing.get_context("spawn")

                def start_next() -> None:
                    entry = next(pending, None)
                    if entry is None:
                        return
                    index, plan = entry
                    snapshot = snapshot_runtime(runtime, (plan.stream_id,))
                    worker_root = temp_root / str(index)
                    worker_root.mkdir()
                    control_read, control_write = context.Pipe(duplex=False)
                    process = context.Process(
                        target=_run_stream,
                        name=f"jerry-stream:{plan.stream_id}",
                        args=(
                            snapshot,
                            plan,
                            cadence,
                            worker_root,
                            control_write,
                            logging.getLogger("jerrythomas").getEffectiveLevel(),
                        ),
                    )
                    active.append(_Worker(index, plan.stream_id, process, control_read))
                    try:
                        process.start()
                    finally:
                        # Keeping these open here would hide EOF after a child crash.
                        control_write.close()

                for _ in range(min(runtime.execution.workers, len(plans))):
                    start_next()
                while active:
                    progress.update(len(completed), len(plans), len(active))
                    wait(
                        [w.control for w in active if not w.control_closed], timeout=0.1
                    )
                    _check_workers(active)
                    for worker in tuple(active):
                        if worker.complete is None:
                            continue
                        worker.process.join(timeout=5)
                        if worker.process.is_alive() or worker.process.exitcode != 0:
                            raise StreamWorkerError(
                                f"Stream '{worker.stream_id}' did not exit successfully."
                            )
                        completed[worker.index] = worker.complete
                        # Remove before close so cancellation cannot revisit a closed process.
                        active.remove(worker)
                        _close_worker(worker)
                        start_next()
            progress.update(len(plans), len(plans), 0)

            paths: list[Path] = []
            rows = 0
            # Completion order must not change equal-key precedence or type errors.
            for index in range(len(plans)):
                result = completed[index]
                sample_keys.merge_types(result.key_types)
                paths.extend(result.runs.paths)
                rows += result.runs.rows
            yield from merge_sort_runs(
                SortedRuns(tuple(paths), rows),
                _projected_row_key,
                temp_root,
                progress.start_merging(),
            )
        finally:
            _stop_workers(active)


def _check_workers(workers: Sequence[_Worker]) -> None:
    for worker in workers:
        exitcode = worker.process.exitcode
        # A finished process can still have a buffered completion or error.
        while not worker.control_closed and worker.control.poll():
            try:
                message = worker.control.recv()
            except EOFError:
                worker.control_closed = True
                break
            if isinstance(message, _Failure):
                error = StreamWorkerError(
                    f"Stream '{worker.stream_id}' failed: {message.error}"
                )
                error.add_note(message.traceback)
                raise error
            if isinstance(message, _Complete):
                worker.complete = message
            elif isinstance(message, _Log):
                logging.getLogger(message.name).log(message.level, message.message)
            elif isinstance(message, ExecutionMessage):
                if not emit_execution_message(message.message, message.log_level):
                    logging.getLogger(__name__).log(message.log_level, message.message)
        if exitcode is not None and worker.complete is None:
            raise StreamWorkerError(
                f"Stream '{worker.stream_id}' exited without completing "
                f"(exit code {exitcode})."
            )


def _close_worker(worker: _Worker) -> None:
    worker.control.close()
    worker.process.close()


def _stop_workers(workers: Sequence[_Worker]) -> None:
    for worker in workers:
        if worker.process.pid is not None and worker.process.is_alive():
            worker.process.terminate()
    for worker in workers:
        if worker.process.pid is not None:
            worker.process.join(timeout=5)
            if worker.process.is_alive():
                worker.process.kill()
                worker.process.join()
        _close_worker(worker)
