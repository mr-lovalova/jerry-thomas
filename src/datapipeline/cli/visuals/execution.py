import logging
from collections.abc import Callable
from functools import partial

from datapipeline.cli.visuals.execution_context import (
    current_execution_event_handler,
)
from datapipeline.execution.events import (
    NodeFinished,
    NodeProgress,
    NodeStarted,
    PipelineEvent,
    PipelineFinished,
    PipelineProgress,
    PipelineStarted,
    PipelineSummary,
    format_elapsed,
    format_node_progress,
)
from datapipeline.execution.observability import (
    CommandFinished,
    ExecutionEvent,
    ExecutionMessage,
    FileResult,
    OperationFinished,
    OperationProgress,
    OperationStarted,
    RowsWritten,
)


ExecutionLogEvent = ExecutionEvent | PipelineEvent


class ExecutionEventFormatter:
    @staticmethod
    def error_suffix(
        event: PipelineFinished | NodeFinished | OperationFinished,
    ) -> str:
        if event.status != "error" or event.error_type is None:
            return ""
        suffix = f" error={event.error_type}"
        if event.error_message:
            message = event.error_message.replace("\n", "\\n")
            suffix = f"{suffix}: {message}"
        return suffix

    @staticmethod
    def progress_message(event: NodeProgress) -> str:
        return format_node_progress(event.progress, event.elapsed_seconds)

    @staticmethod
    def level(event: ExecutionLogEvent) -> int:
        if isinstance(event, ExecutionMessage):
            return int(event.log_level)
        if isinstance(
            event,
            CommandFinished | PipelineFinished | NodeFinished | OperationFinished,
        ):
            if event.status == "error":
                return logging.ERROR
        if isinstance(
            event,
            CommandFinished
            | FileResult
            | RowsWritten
            | PipelineStarted
            | PipelineSummary
            | PipelineProgress
            | PipelineFinished
            | OperationStarted
            | OperationFinished
            | OperationProgress,
        ):
            return logging.INFO
        if isinstance(event, NodeStarted | NodeProgress | NodeFinished):
            return logging.DEBUG
        raise TypeError(f"Unsupported execution event: {type(event).__name__}")

    @classmethod
    def message(cls, event: ExecutionLogEvent) -> str:
        if isinstance(event, FileResult):
            return f"{event.label}: {event.path}"
        if isinstance(event, RowsWritten):
            return f"{event.output_id} rows: {event.row_count:,}"
        if isinstance(event, CommandFinished):
            return (
                f"Command {event.command} finished status={event.status} "
                f"elapsed={format_elapsed(event.elapsed_seconds)}"
            )
        if isinstance(event, OperationStarted):
            return f"Operation {event.name} started"
        if isinstance(event, OperationFinished):
            error_suffix = cls.error_suffix(event)
            return (
                f"Operation {event.name} finished status={event.status}"
                f"{error_suffix} elapsed={format_elapsed(event.elapsed_seconds)}"
            )
        if isinstance(event, ExecutionMessage):
            return event.message
        if isinstance(event, PipelineSummary):
            return f"[{event.pipeline_name}] {event.summary}"
        if isinstance(event, PipelineStarted):
            return f"[{event.pipeline_name}] started"
        if isinstance(event, PipelineProgress):
            return (
                f"[{event.pipeline_name}] running "
                f"elapsed={format_elapsed(event.elapsed_seconds)} "
                f"items={event.output_items}"
            )
        if isinstance(event, PipelineFinished):
            error_suffix = cls.error_suffix(event)
            return (
                f"[{event.pipeline_name}] finished "
                f"status={event.status}{error_suffix} items={event.output_items} "
                f"elapsed={format_elapsed(event.elapsed_seconds)}"
            )
        if isinstance(event, NodeStarted):
            label = f"{event.pipeline_name}/{event.node_name}"
            return f"[{label}] started"
        if isinstance(event, NodeProgress):
            label = f"{event.pipeline_name}/{event.node_name}"
            return f"[{label}] {cls.progress_message(event)}"
        if isinstance(event, OperationProgress):
            return (
                f"Operation {event.name} · {event.step} · running "
                f"reported_at={format_elapsed(event.reported_at_seconds)} "
                f"{event.unit}={event.completed}"
            )
        if isinstance(event, NodeFinished):
            error_suffix = cls.error_suffix(event)
            label = f"{event.pipeline_name}/{event.node_name}"
            return (
                f"[{label}] finished "
                f"status={event.status}{error_suffix} out={event.output_items} "
                f"elapsed={format_elapsed(event.elapsed_seconds)}"
            )
        raise TypeError(f"Unsupported execution event: {type(event).__name__}")


def route_execution_event(
    event: ExecutionLogEvent,
    logger: logging.Logger | None = None,
) -> None:
    if not isinstance(event, NodeProgress) or event.heartbeat:
        active_logger = logger or logging.getLogger(__name__)
        level = ExecutionEventFormatter.level(event)
        if active_logger.isEnabledFor(level):
            active_logger.log(
                level,
                ExecutionEventFormatter.message(event),
                extra={"dp_event_kind": "execution"},
            )

    handler = current_execution_event_handler()
    if handler is not None:
        handler(event)


def make_execution_observer(
    logger: logging.Logger | None = None,
) -> Callable[[ExecutionLogEvent], None]:
    return partial(
        route_execution_event,
        logger=logger or logging.getLogger(__name__),
    )
