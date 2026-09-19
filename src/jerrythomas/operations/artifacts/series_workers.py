from __future__ import annotations

import logging
import multiprocessing
import pickle
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
from jerrythomas.runtime import Runtime, RuntimeSnapshot
from jerrythomas.services.runtime_compiler import (
    compile_runtime_snapshot,
    snapshot_runtime,
)
from jerrythomas.services.temp_cleanup import sort_spill_directory

if TYPE_CHECKING:
    from jerrythomas.operations.artifacts.series import _ProjectedRow, _StreamPlan


_BATCH_BYTES = 1024 * 1024
_KeyTypes = tuple[SampleKeyValueType | None, ...]


class StreamWorkerError(RuntimeError):
    """A stream process failed before its output could be consumed."""


class StreamWorkerProgress:
    def __init__(self) -> None:
        self._detail = "starting workers"

    def update(self, completed: int, total: int, active: int) -> None:
        self._detail = f"{completed}/{total} streams complete, {active} workers"

    def snapshot(self, output_items: int) -> ProgressSnapshot:
        return ProgressSnapshot(
            completed=output_items, phase="projecting", detail=self._detail
        )


@dataclass(frozen=True)
class _Batch:
    rows: tuple[bytes, ...]
    key_types: _KeyTypes


@dataclass(frozen=True)
class _Complete:
    rows: int
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
    stream_id: str
    process: BaseProcess
    data: Connection
    control: Connection
    complete: _Complete | None = None
    data_closed: bool = False
    control_closed: bool = False


class _WorkerLogHandler(logging.Handler):
    def __init__(self, connection: Connection, lock: threading.Lock) -> None:
        super().__init__()
        self.connection = connection
        self.send_lock = lock

    def emit(self, record: logging.LogRecord) -> None:
        with self.send_lock:
            self.connection.send(_Log(record.name, record.levelno, self.format(record)))


def _run_stream(
    snapshot: RuntimeSnapshot,
    plan: _StreamPlan,
    cadence: timedelta,
    temp_root: Path,
    data: Connection,
    control: Connection,
    log_level: int,
) -> None:
    from jerrythomas.operations.artifacts.series import _project_stream

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
            sample_keys = SampleKeyContract(runtime.dataset.sample.keys)
            rows = _project_stream(runtime, plan, sample_keys, cadence)
            try:
                count = _send_rows(rows, sample_keys, data)
            finally:
                rows.close()
        with send_lock:
            control.send(_Complete(count, sample_keys.inferred_types))
    except BaseException as error:
        failure = _Failure(f"{type(error).__name__}: {error}", traceback.format_exc())
        with send_lock:
            control.send(failure)
    finally:
        data.close()
        control.close()


def _send_rows(
    rows: Iterator[_ProjectedRow], sample_keys: SampleKeyContract, data: Connection
) -> int:
    batch: list[bytes] = []
    size = 0
    count = 0
    for row in rows:
        # Snapshot before advancing plugins that may reuse mutable record values.
        payload = pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL)
        batch.append(payload)
        size += len(payload)
        count += 1
        if size >= _BATCH_BYTES:
            data.send(_Batch(tuple(batch), sample_keys.inferred_types))
            batch.clear()
            size = 0
    if batch:
        data.send(_Batch(tuple(batch), sample_keys.inferred_types))
    return count


def project_streams_parallel(
    runtime: Runtime,
    plans: Sequence[_StreamPlan],
    sample_keys: SampleKeyContract,
    cadence: timedelta,
    progress: StreamWorkerProgress,
) -> Iterator[_ProjectedRow]:
    context = multiprocessing.get_context("spawn")
    active: list[_Worker] = []
    pending = iter(enumerate(plans))

    # This parent-owned lock protects every child spill, including after a crash.
    with sort_spill_directory() as temp_root:
        try:

            def start_next() -> None:
                index, plan = next(pending)
                snapshot = snapshot_runtime(runtime, (plan.stream_id,))
                worker_root = temp_root / str(index)
                worker_root.mkdir()
                data_read, data_write = context.Pipe(duplex=False)
                control_read, control_write = context.Pipe(duplex=False)
                process = context.Process(
                    target=_run_stream,
                    name=f"jerry-stream:{plan.stream_id}",
                    args=(
                        snapshot,
                        plan,
                        cadence,
                        worker_root,
                        data_write,
                        control_write,
                        logging.getLogger("jerrythomas").getEffectiveLevel(),
                    ),
                )
                active.append(_Worker(plan.stream_id, process, data_read, control_read))
                try:
                    process.start()
                finally:
                    # Keeping these open here would hide EOF after a child crash.
                    data_write.close()
                    control_write.close()

            for _ in range(min(runtime.execution.workers, len(plans))):
                start_next()
            for completed in range(len(plans)):
                progress.update(completed, len(plans), len(active))
                current = active[0]
                count = 0
                while True:
                    _check_workers(active)
                    if current.data_closed and current.complete is not None:
                        break
                    readers = [w.control for w in active if not w.control_closed]
                    if not current.data_closed:
                        readers.append(current.data)
                    ready = wait(readers, timeout=0.1)
                    _check_workers(active)
                    if not current.data_closed and current.data in ready:
                        try:
                            batch = current.data.recv()
                        except EOFError:
                            current.data_closed = True
                            continue
                        sample_keys.merge_types(batch.key_types)
                        for payload in batch.rows:
                            count += 1
                            yield pickle.loads(payload)

                sample_keys.merge_types(current.complete.key_types)
                if count != current.complete.rows:
                    raise StreamWorkerError(
                        f"Stream '{current.stream_id}' returned incomplete output."
                    )
                current.process.join(timeout=5)
                if current.process.is_alive() or current.process.exitcode != 0:
                    raise StreamWorkerError(
                        f"Stream '{current.stream_id}' did not exit successfully."
                    )
                active.pop(0)
                _close_worker(current)
                if completed + len(active) + 1 < len(plans):
                    start_next()
            progress.update(len(plans), len(plans), 0)
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
    worker.data.close()
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
