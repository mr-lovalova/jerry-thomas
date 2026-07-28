import logging
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from datapipeline.cli.visuals.execution_context import current_execution_event_handler
from datapipeline.cli.visuals.execution_context import current_terminal_log_handler
from datapipeline.execution.observability import ExecutionMessage
from datapipeline.execution.settings import LogOutputSettings, LogOutputTarget


def parse_log_output_specs(
    specs: Sequence[str] | None,
    resolve_global_path: Callable[[str], Path],
) -> list[LogOutputTarget]:
    if not specs:
        return []

    outputs: list[LogOutputTarget] = []
    for raw_spec in specs:
        spec = raw_spec.strip()
        if not spec:
            raise ValueError("--log-output value cannot be empty")

        lower = spec.lower()
        if lower in {"stderr", "stdout"}:
            outputs.append(LogOutputTarget(transport=lower))
            continue
        if lower in {"execution", "fs:execution", "fs=execution"}:
            outputs.append(LogOutputTarget(transport="fs", scope="execution"))
            continue
        if lower.startswith("execution:") or lower.startswith("execution="):
            path = spec[10:].strip()
            if not path:
                raise ValueError(
                    "--log-output execution target requires a relative path"
                )
            outputs.append(
                LogOutputTarget(
                    transport="fs",
                    destination=Path(path),
                    scope="execution",
                )
            )
            continue

        if lower.startswith("fs:"):
            path = spec.split(":", 1)[1].strip()
        elif lower.startswith("fs="):
            path = spec.split("=", 1)[1].strip()
        else:
            raise ValueError(
                "invalid --log-output value. Use 'stderr', 'stdout', "
                "'fs:<path>', or 'execution[:<relative-path>]'"
            )
        if not path:
            raise ValueError("--log-output fs target requires a path")
        outputs.append(
            LogOutputTarget(
                transport="fs",
                destination=resolve_global_path(path),
            )
        )
    return outputs


class _TerminalExecutionEventDedupeFilter(logging.Filter):
    """Suppress execution-event log records while visuals are actively rendering."""

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "dp_event_kind", None) is None:
            return True
        return current_execution_event_handler() is None


class _TerminalVisualProxyHandler(logging.StreamHandler):
    """Route plain logger records through active Rich visuals when available."""

    def emit(self, record: logging.LogRecord) -> None:
        # Execution-event records are already routed through the active visuals.
        if getattr(record, "dp_event_kind", None) is None:
            handler = current_terminal_log_handler()
            if handler is not None:
                handler(
                    ExecutionMessage(
                        message=self.format(record),
                        log_level=int(record.levelno),
                    )
                )
                return
        super().emit(record)


def configure_root_logging(level: int, output: LogOutputSettings) -> None:
    """Configure root logging handlers for the resolved outputs."""
    handlers: list[logging.Handler] = []
    seen: set[tuple[str, str | None, str]] = set()
    for target in output.outputs:
        transport = target.transport.lower()
        destination = target.destination
        key = (
            transport,
            str(destination) if destination is not None else None,
            str(target.scope).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        handler: logging.Handler
        if transport == "stderr":
            handler = _TerminalVisualProxyHandler(sys.stderr)
            handler.addFilter(_TerminalExecutionEventDedupeFilter())
        elif transport == "stdout":
            handler = _TerminalVisualProxyHandler(sys.stdout)
            handler.addFilter(_TerminalExecutionEventDedupeFilter())
        elif transport == "fs":
            if destination is None:
                raise ValueError("log transport 'fs' requires a destination path")
            path = Path(destination).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(path, encoding="utf-8", delay=True)
        else:
            raise ValueError(f"Unsupported log transport: {target.transport!r}")
        handlers.append(handler)

    logging.basicConfig(
        level=int(level),
        format="%(message)s",
        handlers=handlers,
        force=True,
    )


@contextmanager
def root_logging_scope(
    level: int,
    output: LogOutputSettings,
) -> Iterator[None]:
    """Temporarily replace root logging without closing the outer handlers."""

    root = logging.getLogger()
    previous_level = root.level
    previous_handlers = tuple(root.handlers)
    for handler in previous_handlers:
        root.removeHandler(handler)

    try:
        configure_root_logging(level, output)
        yield
    finally:
        temporary_handlers = tuple(root.handlers)
        for handler in temporary_handlers:
            root.removeHandler(handler)

        try:
            for handler in temporary_handlers:
                handler.close()
        finally:
            root.setLevel(previous_level)
            for handler in previous_handlers:
                root.addHandler(handler)
