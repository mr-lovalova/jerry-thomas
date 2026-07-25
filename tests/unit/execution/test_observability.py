import logging
from pathlib import Path

import pytest

import datapipeline.execution.observability as observability
from datapipeline.execution.observability import (
    ExecutionEvent,
    ExecutionMessage,
    FileResult,
    OperationFinished,
    OperationProgress,
    OperationProgressTracker,
    OperationStarted,
    RowsWritten,
    current_execution_observer,
    emit_execution_message,
    emit_file_result,
    emit_operation_progress,
    emit_rows_written,
    execution_observer,
    operation_scope,
)


def test_observer_routes_operation_lifecycle_results_and_progress(monkeypatch) -> None:
    times = iter((3.0, 4.0, 4.25))
    monkeypatch.setattr(observability.time, "perf_counter", lambda: next(times))
    events: list[ExecutionEvent] = []
    observer = events.append

    assert current_execution_observer() is None
    assert emit_execution_message("outside") is False
    assert emit_file_result("Output", Path("/tmp/out.jsonl")) is False
    assert emit_rows_written("train", 3) is False
    assert emit_operation_progress("outside", 1, "rows") is False

    with execution_observer(observer):
        assert current_execution_observer() is observer
        assert emit_execution_message("details", logging.DEBUG)
        assert emit_rows_written("train", 3)
        assert emit_file_result(
            "Model grid",
            Path("/tmp/model_grid.jsonl"),
        )
        with operation_scope("build:model_grid"):
            assert emit_operation_progress("write", 3, "rows")

    assert current_execution_observer() is None
    assert events == [
        ExecutionMessage(message="details", log_level=logging.DEBUG),
        RowsWritten("train", 3),
        FileResult("Model grid", Path("/tmp/model_grid.jsonl")),
        OperationStarted("build:model_grid"),
        OperationProgress(
            name="build:model_grid",
            step="write",
            reported_at_seconds=1,
            completed=3,
            unit="rows",
        ),
        OperationFinished(
            "build:model_grid",
            "success",
            elapsed_seconds=1.25,
        ),
    ]


def test_operation_scope_emits_failure_and_restores_progress_context(
    monkeypatch,
) -> None:
    times = iter((5.0, 5.5))
    monkeypatch.setattr(observability.time, "perf_counter", lambda: next(times))
    events: list[ExecutionEvent] = []

    with execution_observer(events.append):
        with pytest.raises(ValueError, match="bad input"):
            with operation_scope("serve:test"):
                raise ValueError("  bad input  ")

        assert emit_operation_progress("after", 1, "rows") is False

    assert events == [
        OperationStarted("serve:test"),
        OperationFinished(
            "serve:test",
            "error",
            elapsed_seconds=0.5,
            error_type="ValueError",
            error_message="  bad input  ",
        ),
    ]


def test_operation_scope_preserves_body_failure_when_finish_observer_fails() -> None:
    events: list[ExecutionEvent] = []

    def observer(event: ExecutionEvent) -> None:
        events.append(event)
        if isinstance(event, OperationFinished):
            raise RuntimeError("observer failed")

    with execution_observer(observer):
        with pytest.raises(ValueError, match="operation failed") as raised:
            with operation_scope("serve:test"):
                raise ValueError("operation failed")

    assert raised.value.__notes__ == [
        "Reporting the operation failure also failed: observer failed"
    ]
    assert isinstance(events[0], OperationStarted)
    assert isinstance(events[1], OperationFinished)
    assert events[1].status == "error"


def test_operation_scope_surfaces_finish_observer_failure_after_success() -> None:
    def observer(event: ExecutionEvent) -> None:
        if isinstance(event, OperationFinished):
            raise RuntimeError("observer failed")

    with execution_observer(observer):
        with pytest.raises(RuntimeError, match="observer failed"):
            with operation_scope("serve:test"):
                pass


def test_operation_progress_tracker_preserves_interval_counts_and_unit(
    monkeypatch,
) -> None:
    times = iter((0.0, 0.5, 1.0, 1.4, 2.1))
    monkeypatch.setattr(observability.time, "perf_counter", lambda: next(times))
    emitted: list[tuple[str, int, str]] = []

    def capture_progress(
        step: str,
        completed: int,
        unit: str,
    ) -> bool:
        emitted.append((step, completed, unit))
        return True

    monkeypatch.setattr(observability, "emit_operation_progress", capture_progress)
    progress = OperationProgressTracker("write", "rows", interval_seconds=1)

    progress.advance(2)
    progress.advance()
    progress.advance(4)
    progress.advance()

    assert emitted == [
        ("write", 3, "rows"),
        ("write", 8, "rows"),
    ]


def test_operation_progress_tracker_rejects_negative_interval() -> None:
    with pytest.raises(ValueError, match="interval_seconds must be non-negative"):
        OperationProgressTracker("write", "rows", interval_seconds=-0.1)
