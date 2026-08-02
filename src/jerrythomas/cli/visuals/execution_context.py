from collections.abc import Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jerrythomas.execution.observability import ExecutionEvent

ExecutionEventHandler = Callable[["ExecutionEvent"], None]

_CURRENT_EXECUTION_EVENT_HANDLER: ContextVar[ExecutionEventHandler | None] = ContextVar(
    "jerrythomas_visual_current_execution_event_handler",
    default=None,
)
_CURRENT_TERMINAL_LOG_HANDLER: ContextVar[ExecutionEventHandler | None] = ContextVar(
    "jerrythomas_visual_current_terminal_log_handler",
    default=None,
)


def set_current_execution_event_handler(handler: ExecutionEventHandler | None):
    return _CURRENT_EXECUTION_EVENT_HANDLER.set(handler)


def reset_current_execution_event_handler(token) -> None:
    _CURRENT_EXECUTION_EVENT_HANDLER.reset(token)


def current_execution_event_handler() -> ExecutionEventHandler | None:
    return _CURRENT_EXECUTION_EVENT_HANDLER.get()


def set_current_terminal_log_handler(handler: ExecutionEventHandler | None):
    return _CURRENT_TERMINAL_LOG_HANDLER.set(handler)


def reset_current_terminal_log_handler(token) -> None:
    _CURRENT_TERMINAL_LOG_HANDLER.reset(token)


def current_terminal_log_handler() -> ExecutionEventHandler | None:
    return _CURRENT_TERMINAL_LOG_HANDLER.get()
