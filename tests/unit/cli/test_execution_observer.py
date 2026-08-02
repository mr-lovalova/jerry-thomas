import logging
from pathlib import Path

import pytest

import jerrythomas.execution.observability as observability
from jerrythomas.cli.visuals.execution import (
    ExecutionEventFormatter,
    make_execution_observer,
)
from jerrythomas.cli.visuals.execution_context import (
    reset_current_execution_event_handler,
    set_current_execution_event_handler,
)
from jerrythomas.execution.events import (
    NodeFinished,
    NodeProgress,
    NodeStarted,
    PipelineFinished,
    PipelineProgress,
    PipelineStarted,
    PipelineSummary,
    ProgressSnapshot,
    format_elapsed,
)
from jerrythomas.execution.observability import (
    CommandFinished,
    ExecutionMessage,
    FileResult,
    OperationFinished,
    OperationProgress,
    OperationStarted,
    RowsWritten,
    emit_execution_message,
    emit_file_result,
    emit_operation_progress,
    execution_observer,
    operation_scope,
)


class _CaptureHandler:
    def __init__(self) -> None:
        self.events = []

    def __call__(self, event) -> None:
        self.events.append(event)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0ms"),
        (0.2396, "240ms"),
        (0.9996, "1.0s"),
        (12.34, "12.3s"),
        (59.96, "1m00.0s"),
        (239.853907, "3m59.9s"),
        (1_383.786434, "23m03.8s"),
        (3_723.4, "1h02m03.4s"),
    ],
)
def test_elapsed_time_formatting(seconds, expected) -> None:
    assert format_elapsed(seconds) == expected


@pytest.mark.parametrize("seconds", [-0.1, float("inf"), float("nan")])
def test_elapsed_time_formatting_rejects_invalid_values(seconds) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        format_elapsed(seconds)


@pytest.mark.parametrize(
    ("event", "level", "message"),
    [
        (
            ExecutionMessage(message="plain", log_level=logging.WARNING),
            logging.WARNING,
            "plain",
        ),
        (
            FileResult("train_0", Path("/tmp/dataset.train_0.jsonl")),
            logging.INFO,
            "train_0: /tmp/dataset.train_0.jsonl",
        ),
        (
            RowsWritten("train_0", 1250),
            logging.INFO,
            "train_0 rows: 1,250",
        ),
        (
            CommandFinished("serve", "success", 2.5),
            logging.INFO,
            "Command serve finished status=success elapsed=2.5s",
        ),
        (
            CommandFinished("serve", "error", 2.5),
            logging.ERROR,
            "Command serve finished status=error elapsed=2.5s",
        ),
        (
            PipelineStarted(pipeline_name="pipeline"),
            logging.INFO,
            "[pipeline] started",
        ),
        (
            PipelineSummary(
                pipeline_name="pipeline",
                summary="transport=fs.file file=prices.jsonl",
            ),
            logging.INFO,
            "[pipeline] transport=fs.file file=prices.jsonl",
        ),
        (
            PipelineProgress(
                pipeline_name="pipeline",
                output_items=10,
                elapsed_seconds=60,
            ),
            logging.INFO,
            "[pipeline] running elapsed=1m00.0s items=10",
        ),
        (
            PipelineFinished(
                pipeline_name="pipeline",
                status="success",
                output_items=3,
                elapsed_seconds=0.5,
            ),
            logging.INFO,
            "[pipeline] finished status=success items=3 elapsed=500ms",
        ),
        (
            NodeStarted(pipeline_name="pipeline", node_name="load", node_index=0),
            logging.DEBUG,
            "[pipeline/load] started",
        ),
        (
            NodeProgress(
                pipeline_name="pipeline",
                node_name="load",
                node_index=0,
                progress=ProgressSnapshot(completed=0),
                elapsed_seconds=0,
            ),
            logging.DEBUG,
            "[pipeline/load] running elapsed=0ms items=0",
        ),
        (
            NodeFinished(
                pipeline_name="pipeline",
                node_name="load",
                node_index=0,
                status="success",
                output_items=3,
                elapsed_seconds=0.25,
            ),
            logging.DEBUG,
            "[pipeline/load] finished status=success out=3 elapsed=250ms",
        ),
        (
            OperationProgress(
                name="build:schema",
                step="write",
                reported_at_seconds=1.9,
                completed=3,
                unit="rows",
            ),
            logging.INFO,
            "Operation build:schema · write · running reported_at=1.9s rows=3",
        ),
        (
            OperationStarted("serve:dataset"),
            logging.INFO,
            "Operation serve:dataset started",
        ),
        (
            OperationFinished("serve:dataset", "success", 0.25),
            logging.INFO,
            "Operation serve:dataset finished status=success elapsed=250ms",
        ),
        (
            OperationFinished(
                "serve:dataset",
                "error",
                0.5,
                error_type="ValueError",
                error_message="bad\ninput",
            ),
            logging.ERROR,
            "Operation serve:dataset finished status=error "
            "error=ValueError: bad\\ninput elapsed=500ms",
        ),
    ],
)
def test_typed_execution_event_formatting(event, level, message) -> None:
    assert ExecutionEventFormatter.level(event) == level
    assert ExecutionEventFormatter.message(event) == message


def test_failed_terminal_events_are_errors() -> None:
    event = PipelineFinished(
        pipeline_name="pipeline",
        status="error",
        output_items=0,
        elapsed_seconds=1,
        error_type="ValueError",
    )

    assert ExecutionEventFormatter.level(event) == logging.ERROR


def test_observer_logs_root_lifecycle_and_summary_at_info(caplog) -> None:
    logger = logging.getLogger("jerrythomas.cli.visuals.execution.test.root")
    observer = make_execution_observer(logger)

    with caplog.at_level(logging.INFO, logger=logger.name):
        observer(PipelineStarted(pipeline_name="stream:prices"))
        observer(
            PipelineSummary(
                pipeline_name="stream:prices",
                summary="transport=fs.file file=prices",
            )
        )
        observer(
            PipelineFinished(
                pipeline_name="stream:prices",
                output_items=3,
                elapsed_seconds=0.02,
                status="success",
            )
        )

    assert [record.getMessage() for record in caplog.records] == [
        "[stream:prices] started",
        "[stream:prices] transport=fs.file file=prices",
        "[stream:prices] finished status=success items=3 elapsed=20ms",
    ]


def test_observer_logs_stages_at_debug(caplog) -> None:
    logger = logging.getLogger("jerrythomas.cli.visuals.execution.test.stages")
    observer = make_execution_observer(logger)

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        observer(
            NodeStarted(
                pipeline_name="pipeline",
                node_name="load",
                node_index=0,
            )
        )
        observer(
            NodeFinished(
                pipeline_name="pipeline",
                node_name="load",
                node_index=0,
                output_items=2,
                elapsed_seconds=0.01,
                status="success",
            )
        )

    assert [record.getMessage() for record in caplog.records] == [
        "[pipeline/load] started",
        "[pipeline/load] finished status=success out=2 elapsed=10ms",
    ]


def test_observer_logs_pipeline_heartbeat_at_info(caplog) -> None:
    logger = logging.getLogger("jerrythomas.cli.visuals.execution.test.progress")
    observer = make_execution_observer(logger)

    with caplog.at_level(logging.INFO, logger=logger.name):
        observer(
            NodeProgress(
                pipeline_name="series:close",
                node_name="ensure_record_order",
                node_index=2,
                progress=ProgressSnapshot(completed=20),
                elapsed_seconds=60,
                heartbeat=True,
            )
        )
        observer(
            PipelineProgress(
                pipeline_name="series:close",
                output_items=15,
                elapsed_seconds=60,
            )
        )

    assert len(caplog.records) == 1
    assert caplog.records[0].getMessage() == (
        "[series:close] running elapsed=1m00.0s items=15"
    )
    assert getattr(caplog.records[0], "dp_event_kind", None) == "execution"


def test_observer_logs_only_heartbeat_node_progress_at_debug(caplog) -> None:
    logger = logging.getLogger("jerrythomas.cli.visuals.execution.test.node-progress")
    observer = make_execution_observer(logger)

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        observer(
            NodeProgress(
                pipeline_name="series:close",
                node_name="ensure_record_order",
                node_index=2,
                progress=ProgressSnapshot(completed=10),
                elapsed_seconds=1,
            )
        )
        observer(
            NodeProgress(
                pipeline_name="series:close",
                node_name="ensure_record_order",
                node_index=2,
                progress=ProgressSnapshot(completed=20),
                elapsed_seconds=60,
                heartbeat=True,
            )
        )

    assert [record.getMessage() for record in caplog.records] == [
        "[series:close/ensure_record_order] running elapsed=1m00.0s items=20"
    ]


def test_observer_includes_error_details_on_failure(caplog) -> None:
    logger = logging.getLogger("jerrythomas.cli.visuals.execution.test.error")
    observer = make_execution_observer(logger)

    with caplog.at_level(logging.INFO, logger=logger.name):
        observer(
            PipelineFinished(
                pipeline_name="dataset",
                output_items=0,
                elapsed_seconds=0.5,
                status="error",
                error_type="ValueError",
                error_message="No entry point 'target_mapper'",
            )
        )

    assert (
        caplog.records[0]
        .getMessage()
        .startswith(
            "[dataset] finished status=error "
            "error=ValueError: No entry point 'target_mapper'"
        )
    )


def test_execution_observer_routes_pipeline_events_to_logger_and_handler(
    caplog,
) -> None:
    logger = logging.getLogger("jerrythomas.cli.visuals.execution.test.context")
    capture = _CaptureHandler()
    token = set_current_execution_event_handler(capture)
    try:
        observer = make_execution_observer(logger=logger)
        with caplog.at_level(logging.INFO, logger=logger.name):
            observer(PipelineStarted(pipeline_name="dataset"))
            observer(
                PipelineFinished(
                    pipeline_name="dataset",
                    output_items=1,
                    elapsed_seconds=0.5,
                    status="success",
                )
            )
    finally:
        reset_current_execution_event_handler(token)

    assert [type(event) for event in capture.events] == [
        PipelineStarted,
        PipelineFinished,
    ]
    assert [record.getMessage() for record in caplog.records] == [
        "[dataset] started",
        "[dataset] finished status=success items=1 elapsed=500ms",
    ]


def test_context_handler_is_resolved_when_each_event_is_emitted(caplog) -> None:
    logger = logging.getLogger("jerrythomas.cli.visuals.execution.test.reset")
    capture = _CaptureHandler()
    token = set_current_execution_event_handler(capture)
    try:
        observer = make_execution_observer(logger=logger)
        observer(PipelineStarted(pipeline_name="dataset"))
    finally:
        reset_current_execution_event_handler(token)

    with caplog.at_level(logging.INFO, logger=logger.name):
        observer(
            PipelineFinished(
                pipeline_name="dataset",
                output_items=1,
                elapsed_seconds=0.5,
                status="success",
            )
        )

    assert [type(event) for event in capture.events] == [PipelineStarted]
    assert caplog.records[-1].getMessage().startswith("[dataset] finished")


def test_execution_message_uses_execution_observer_and_logger(caplog) -> None:
    capture = _CaptureHandler()
    logger = logging.getLogger("jerrythomas.cli.visuals.execution.test.message")
    token = set_current_execution_event_handler(capture)
    try:
        with caplog.at_level(logging.INFO, logger=logger.name):
            observer = make_execution_observer(logger)
            with execution_observer(observer):
                assert emit_execution_message("Saved 2 items")
    finally:
        reset_current_execution_event_handler(token)

    assert len(capture.events) == 1
    assert capture.events[0] == ExecutionMessage(message="Saved 2 items")
    assert caplog.records[-1].getMessage() == "Saved 2 items"


def test_operation_scope_emits_flat_lifecycle_result_and_progress(
    caplog,
    monkeypatch,
) -> None:
    times = iter((0.0, 1.0, 1.25))
    monkeypatch.setattr(observability.time, "perf_counter", lambda: next(times))
    capture = _CaptureHandler()
    logger = logging.getLogger("jerrythomas.cli.visuals.execution.test.operation")
    token = set_current_execution_event_handler(capture)
    try:
        with caplog.at_level(logging.INFO, logger=logger.name):
            observer = make_execution_observer(logger)
            with execution_observer(observer), operation_scope("build:schedule"):
                assert emit_file_result("Schedule", Path("/tmp/schedule.jsonl"))
                assert emit_operation_progress(
                    "write_artifact",
                    3,
                    "rows",
                )
    finally:
        reset_current_execution_event_handler(token)

    assert [type(event) for event in capture.events] == [
        OperationStarted,
        FileResult,
        OperationProgress,
        OperationFinished,
    ]
    assert capture.events[1].path == Path("/tmp/schedule.jsonl")
    assert capture.events[2].step == "write_artifact"
    messages = [record.getMessage() for record in caplog.records]
    assert "Operation build:schedule started" in messages
    assert "Schedule: /tmp/schedule.jsonl" in messages
    assert (
        "Operation build:schedule · write_artifact · running "
        "reported_at=1.0s rows=3" in messages
    )
    assert messages[-1].startswith("Operation build:schedule finished status=success")
