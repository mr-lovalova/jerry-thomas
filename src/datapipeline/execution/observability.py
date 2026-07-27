import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from datapipeline.execution.events import PipelineEvent, RunStatus


OperationStatus = Literal["success", "error"]


@dataclass(frozen=True)
class OperationStarted:
    name: str


@dataclass(frozen=True)
class OperationProgress:
    name: str
    step: str
    reported_at_seconds: float
    completed: int
    unit: str


@dataclass(frozen=True)
class OperationFinished:
    name: str
    status: OperationStatus
    elapsed_seconds: float
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class CommandFinished:
    command: str
    status: RunStatus
    elapsed_seconds: float


@dataclass(frozen=True)
class FileResult:
    label: str
    path: Path


@dataclass(frozen=True)
class RowsWritten:
    output_id: str
    row_count: int


@dataclass(frozen=True, kw_only=True)
class ExecutionMessage:
    message: str
    log_level: int = logging.INFO


ExecutionEvent = (
    PipelineEvent
    | ExecutionMessage
    | CommandFinished
    | FileResult
    | RowsWritten
    | OperationStarted
    | OperationProgress
    | OperationFinished
)
ExecutionObserver = Callable[[ExecutionEvent], None]


def ignore_execution_event(event: ExecutionEvent) -> None:
    pass


@dataclass(frozen=True)
class _OperationContext:
    name: str
    started_at: float


_CURRENT_EXECUTION_OBSERVER: ContextVar[ExecutionObserver | None] = ContextVar(
    "datapipeline_current_execution_observer",
    default=None,
)
_CURRENT_OPERATION: ContextVar[_OperationContext | None] = ContextVar(
    "datapipeline_current_operation",
    default=None,
)


def current_execution_observer() -> ExecutionObserver | None:
    return _CURRENT_EXECUTION_OBSERVER.get()


@contextmanager
def execution_observer(observer: ExecutionObserver):
    token = _CURRENT_EXECUTION_OBSERVER.set(observer)
    try:
        yield
    finally:
        _CURRENT_EXECUTION_OBSERVER.reset(token)


@contextmanager
def operation_scope(name: str):
    observer = current_execution_observer()
    context = _OperationContext(name, time.perf_counter())
    if observer is not None:
        observer(OperationStarted(name))
    token = _CURRENT_OPERATION.set(context)
    try:
        yield
    except BaseException as exc:
        if observer is not None:
            try:
                observer(
                    OperationFinished(
                        name=name,
                        status="error",
                        elapsed_seconds=time.perf_counter() - context.started_at,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
            except BaseException as observer_error:
                exc.add_note(
                    f"Reporting the operation failure also failed: {observer_error}"
                )
        raise
    else:
        if observer is not None:
            observer(
                OperationFinished(
                    name=name,
                    status="success",
                    elapsed_seconds=time.perf_counter() - context.started_at,
                )
            )
    finally:
        _CURRENT_OPERATION.reset(token)


def emit_file_result(
    label: str,
    path: Path,
) -> bool:
    observer = current_execution_observer()
    if observer is None:
        return False
    observer(FileResult(label, path))
    return True


def emit_rows_written(output_id: str, row_count: int) -> bool:
    observer = current_execution_observer()
    if observer is None:
        return False
    observer(RowsWritten(output_id, row_count))
    return True


def emit_execution_message(
    message: str,
    level: int = logging.INFO,
) -> bool:
    observer = current_execution_observer()
    if observer is None:
        return False
    observer(ExecutionMessage(message=message, log_level=int(level)))
    return True


def emit_operation_progress(
    step: str,
    completed: int,
    unit: str,
) -> bool:
    context = _CURRENT_OPERATION.get()
    observer = current_execution_observer()
    if context is None or observer is None:
        return False
    observer(
        OperationProgress(
            name=context.name,
            step=step,
            reported_at_seconds=time.perf_counter() - context.started_at,
            completed=completed,
            unit=unit,
        )
    )
    return True


class OperationProgressTracker:
    def __init__(self, step: str, unit: str, interval_seconds: float) -> None:
        interval = float(interval_seconds)
        if interval < 0:
            raise ValueError("interval_seconds must be non-negative")
        self._step = step
        self._unit = unit
        self._interval_seconds = interval
        self._last_emit_at = time.perf_counter() if interval > 0 else 0.0
        self._completed = 0

    def advance(self, count: int = 1) -> None:
        if self._interval_seconds == 0:
            return
        self._completed += int(count)
        now = time.perf_counter()
        if now - self._last_emit_at < self._interval_seconds:
            return
        self._last_emit_at = now
        emit_operation_progress(self._step, self._completed, self._unit)
