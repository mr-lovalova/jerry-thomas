from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import tempfile
import threading
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Generic, Protocol, TypeVar

from jerrythomas.execution.events import PipelineEvent, ProgressSnapshot
from jerrythomas.execution.observability import (
    ExecutionEvent,
    ExecutionMessage,
    ExecutionScope,
    ScopedExecutionEvent,
    current_execution_observer,
    execution_observer,
    ignore_execution_event,
)
from jerrythomas.runtime import Runtime, RuntimeSnapshot
from jerrythomas.services.runtime_compiler import (
    compile_runtime_snapshot,
    snapshot_runtime,
)

logger = logging.getLogger(__name__)

# How long a process may take to exit after its stream completed before it is stopped.
_EXIT_TIMEOUT_SECONDS = 5.0


class StreamJob(Protocol):
    @property
    def stream_id(self) -> str: ...


_P = TypeVar("_P", bound=StreamJob)
_R = TypeVar("_R")


class StreamWorkerError(RuntimeError):
    """A stream process failed before its output could be consumed."""


class StreamWorkerProgress:
    def __init__(self) -> None:
        self._detail = "preparing streams"

    def update(self, completed: int, total: int, active: int) -> None:
        workers = "worker" if active == 1 else "workers"
        self._detail = f"{completed}/{total} streams complete, {active} {workers}"

    def snapshot(self, output_items: int) -> ProgressSnapshot:
        return ProgressSnapshot(
            completed=output_items, phase="preparing", detail=self._detail
        )


@dataclass(frozen=True)
class _Complete(Generic[_R]):
    result: _R


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
class _Worker(Generic[_R]):
    index: int
    slot: int
    stream_id: str
    scope: ExecutionScope
    process: BaseProcess
    control: Connection
    complete: _Complete[_R] | None = None
    control_closed: bool = False


class _WorkerLogHandler(logging.Handler):
    def __init__(self, connection: Connection, lock: threading.Lock) -> None:
        super().__init__()
        self.connection = connection
        self.send_lock = lock

    def emit(self, record: logging.LogRecord) -> None:
        # Like any handler, a bad log call must be reported, never fail the stream.
        try:
            message = _Log(record.name, record.levelno, self.format(record))
            with self.send_lock:
                self.connection.send(message)
        except Exception:
            self.handleError(record)


def _exit_with_parent() -> None:
    """Stop this worker when its parent dies, for example after SIGTERM or SIGKILL."""
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        parent.join()
        os._exit(1)

    threading.Thread(target=watch, name="jerry-parent-watch", daemon=True).start()


def _run_stream(
    snapshot: RuntimeSnapshot,
    plan: _P,
    prepare: Callable[[Runtime, _P, Path], _R],
    temp_root: Path,
    control: Connection,
    log_level: int,
    observe_events: bool,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _exit_with_parent()
    tempfile.tempdir = str(temp_root)
    send_lock = threading.Lock()
    root_logger = logging.getLogger()
    handler = _WorkerLogHandler(control, send_lock)
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    def observe(event: ExecutionEvent) -> None:
        if isinstance(event, ExecutionMessage) or (
            observe_events and isinstance(event, PipelineEvent)
        ):
            with send_lock:
                control.send(event)

    try:
        with execution_observer(observe):
            runtime = compile_runtime_snapshot(snapshot)
            if not observe_events:
                runtime.heartbeat_interval_seconds = 0
                runtime.observe_node_events = False
            result = prepare(runtime, plan, temp_root)
        with send_lock:
            control.send(_Complete(result))
    except BaseException as error:
        failure = _Failure(f"{type(error).__name__}: {error}", traceback.format_exc())
        with send_lock:
            control.send(failure)
    finally:
        root_logger.removeHandler(handler)
        control.close()


def run_stream_jobs(
    runtime: Runtime,
    plans: Sequence[_P],
    prepare: Callable[[Runtime, _P, Path], _R],
    temp_root: Path,
    progress: StreamWorkerProgress,
) -> list[_R]:
    """Return results in plan order; the caller owns temp_root until consumption.

    Spawned jobs require picklable plans, preparation callbacks, and results.
    """
    observer = current_execution_observer()
    active: list[_Worker[_R]] = []
    pending = iter(enumerate(plans))
    completed: dict[int, _R] = {}

    try:
        if runtime.execution.workers == 1 or len(plans) <= 1:
            for index, plan in pending:
                progress.update(index, len(plans), 1)
                worker_root = temp_root / str(index)
                worker_root.mkdir()
                completed[index] = prepare(runtime, plan, worker_root)
        else:
            context = multiprocessing.get_context("spawn")

            def start_next(slot: int) -> None:
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
                        prepare,
                        worker_root,
                        control_write,
                        logging.getLogger("jerrythomas").getEffectiveLevel(),
                        observer is not None and observer is not ignore_execution_event,
                    ),
                )
                active.append(
                    _Worker(
                        index=index,
                        slot=slot,
                        stream_id=plan.stream_id,
                        scope=ExecutionScope(
                            id=str(worker_root), label=f"Worker {slot + 1}"
                        ),
                        process=process,
                        control=control_read,
                    )
                )
                try:
                    process.start()
                finally:
                    # Keeping these open here would hide EOF after a child crash.
                    control_write.close()

            for slot in range(min(runtime.execution.workers, len(plans))):
                start_next(slot)
            while active:
                progress.update(len(completed), len(plans), len(active))
                wait([w.control for w in active if not w.control_closed], timeout=0.1)
                _check_workers(active)
                for worker in tuple(active):
                    if worker.complete is None:
                        continue
                    _join_completed(worker)
                    completed[worker.index] = worker.complete.result
                    # Remove before close so cancellation cannot revisit a closed process.
                    active.remove(worker)
                    _close_worker(worker)
                    start_next(worker.slot)
        progress.update(len(plans), len(plans), 0)
        # Completion order must not change result precedence or validation errors.
        return [completed[index] for index in range(len(plans))]
    finally:
        _stop_workers(active)


def _check_workers(workers: Sequence[_Worker[_R]]) -> None:
    observer = current_execution_observer()
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
                logging.getLogger(message.name).log(
                    message.level, f"[{worker.scope.label}] {message.message}"
                )
            elif isinstance(message, PipelineEvent | ExecutionMessage):
                if observer is not None:
                    observer(ScopedExecutionEvent(scope=worker.scope, event=message))
                elif isinstance(message, ExecutionMessage):
                    logging.getLogger(__name__).log(
                        message.log_level,
                        f"[{worker.scope.label}] {message.message}",
                    )
        if exitcode is not None and worker.complete is None:
            raise StreamWorkerError(
                f"Stream '{worker.stream_id}' exited without completing "
                f"(exit code {exitcode})."
            )


def _join_completed(worker: _Worker[_R]) -> None:
    """Wait for a process whose result arrived; a lingering one is stopped, not failed."""
    worker.process.join(timeout=_EXIT_TIMEOUT_SECONDS)
    if worker.process.is_alive():
        logger.warning(
            "Stream '%s' finished but its process did not exit within %.0fs; "
            "stopping it.",
            worker.stream_id,
            _EXIT_TIMEOUT_SECONDS,
        )
        worker.process.terminate()
        worker.process.join(timeout=_EXIT_TIMEOUT_SECONDS)
        if worker.process.is_alive():
            worker.process.kill()
            worker.process.join()
    elif worker.process.exitcode != 0:
        raise StreamWorkerError(
            f"Stream '{worker.stream_id}' did not exit successfully."
        )


def _close_worker(worker: _Worker[_R]) -> None:
    worker.control.close()
    worker.process.close()


def _stop_workers(workers: Sequence[_Worker[_R]]) -> None:
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
