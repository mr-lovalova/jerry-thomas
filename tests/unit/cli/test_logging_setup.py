import io
import logging
import sys
from pathlib import Path

import pytest

import datapipeline.execution.observability as observability
from datapipeline.cli.logging_setup import (
    configure_root_logging,
    parse_log_output_specs,
    root_logging_scope,
)
from datapipeline.cli.visuals.execution import (
    make_execution_observer,
    route_execution_event,
)
from datapipeline.cli.visuals.execution_context import (
    reset_current_execution_event_handler,
    reset_current_terminal_log_handler,
    set_current_execution_event_handler,
    set_current_terminal_log_handler,
)
from datapipeline.cli.visuals.rich.progress import visual_summary
from datapipeline.execution.observability import (
    CommandFinished,
    ExecutionMessage,
    emit_file_result,
    emit_operation_progress,
    execution_observer,
    operation_scope,
)
from datapipeline.execution.settings import LogOutputSettings, LogOutputTarget


def _flush_root_handlers() -> None:
    root = logging.getLogger()
    for handler in root.handlers:
        handler.flush()


def _emit_materialize_outputs(logger: logging.Logger) -> None:
    with execution_observer(make_execution_observer(logger)):
        with operation_scope("materialize:adv.20"):
            emit_file_result("Output", Path("/tmp/adv.20.jsonl"))


def test_operation_and_output_events_render_as_flat_plain_logs_without_visuals(
    monkeypatch,
):
    monkeypatch.setattr(observability.time, "perf_counter", lambda: 10.0)
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    configure_root_logging(
        level=logging.INFO,
        output=LogOutputSettings(outputs=(LogOutputTarget(transport="stderr"),)),
    )

    logger = logging.getLogger("datapipeline.tests.logging_setup.plain_result")
    _emit_materialize_outputs(logger)
    _flush_root_handlers()

    assert stream.getvalue().splitlines() == [
        "Operation materialize:adv.20 started",
        "Output: /tmp/adv.20.jsonl",
        "Operation materialize:adv.20 finished status=success elapsed=0ms",
    ]


def test_operation_and_output_logs_do_not_depend_on_visual_handler(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(observability.time, "perf_counter", lambda: 10.0)
    without_visuals = tmp_path / "without-visuals.log"
    configure_root_logging(
        level=logging.INFO,
        output=LogOutputSettings(
            outputs=(LogOutputTarget(transport="fs", destination=without_visuals),)
        ),
    )
    logger = logging.getLogger("datapipeline.tests.logging_setup.file_result")
    _emit_materialize_outputs(logger)
    _flush_root_handlers()

    with_visuals = tmp_path / "with-visuals.log"
    configure_root_logging(
        level=logging.INFO,
        output=LogOutputSettings(
            outputs=(LogOutputTarget(transport="fs", destination=with_visuals),)
        ),
    )

    class _CaptureHandler:
        def __init__(self) -> None:
            self.events = []

        def __call__(self, event) -> None:
            self.events.append(event)

    handler = _CaptureHandler()
    token = set_current_execution_event_handler(handler)
    try:
        _emit_materialize_outputs(logger)
    finally:
        reset_current_execution_event_handler(token)
        _flush_root_handlers()

    plain_content = without_visuals.read_text(encoding="utf-8")
    visual_content = with_visuals.read_text(encoding="utf-8")
    assert visual_content == plain_content
    for expected in (
        "Operation materialize:adv.20 started",
        "Output: /tmp/adv.20.jsonl",
        "Operation materialize:adv.20 finished status=success elapsed=0ms",
    ):
        assert visual_content.count(expected) == 1
    assert len(handler.events) == 3


def test_operation_heartbeat_stays_in_file_during_visuals(
    monkeypatch,
    tmp_path,
) -> None:
    times = iter((0.0, 180.0, 180.0))
    monkeypatch.setattr(observability.time, "perf_counter", lambda: next(times))
    log_path = tmp_path / "heartbeat.log"
    configure_root_logging(
        level=logging.INFO,
        output=LogOutputSettings(
            outputs=(LogOutputTarget(transport="fs", destination=log_path),)
        ),
    )

    class _CaptureHandler:
        def __init__(self) -> None:
            self.events = []

        def __call__(self, event) -> None:
            self.events.append(event)

    handler = _CaptureHandler()
    logger = logging.getLogger("datapipeline.tests.logging_setup.heartbeat")
    token = set_current_execution_event_handler(handler)
    try:
        with execution_observer(make_execution_observer(logger)):
            with operation_scope("serve:dataset"):
                assert emit_operation_progress("write_output", 2_592_885, "rows")
    finally:
        reset_current_execution_event_handler(token)
        _flush_root_handlers()

    content = log_path.read_text(encoding="utf-8")
    heartbeat = (
        "Operation serve:dataset · write_output · running "
        "reported_at=3m00.0s rows=2592885"
    )
    assert content.count(heartbeat) == 1
    assert len(handler.events) == 3


def test_operation_and_output_logs_obey_warning_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr(observability.time, "perf_counter", lambda: 10.0)
    log_path = tmp_path / "warning.log"
    configure_root_logging(
        level=logging.WARNING,
        output=LogOutputSettings(
            outputs=(LogOutputTarget(transport="fs", destination=log_path),)
        ),
    )

    logger = logging.getLogger("datapipeline.tests.logging_setup.warning_result")
    _emit_materialize_outputs(logger)
    _flush_root_handlers()

    assert not log_path.exists()


def test_configure_root_logging_creates_file_on_first_record(tmp_path) -> None:
    log_path = tmp_path / "logs" / "lazy.log"
    configure_root_logging(
        level=logging.INFO,
        output=LogOutputSettings(
            outputs=(LogOutputTarget(transport="fs", destination=log_path),)
        ),
    )

    assert not log_path.exists()

    logging.getLogger("datapipeline.tests.logging_setup.lazy_file").info("first record")
    _flush_root_handlers()

    assert log_path.read_text(encoding="utf-8") == "first record\n"


def test_root_logging_scope_restores_command_outputs(tmp_path) -> None:
    command_log = tmp_path / "command.log"
    execution_log = tmp_path / "execution.log"
    configure_root_logging(
        level=logging.INFO,
        output=LogOutputSettings(
            outputs=(LogOutputTarget(transport="fs", destination=command_log),)
        ),
    )
    logger = logging.getLogger("datapipeline.tests.logging_setup.scope")
    root = logging.getLogger()
    command_handlers = tuple(root.handlers)
    command_level = root.level

    with root_logging_scope(
        logging.INFO,
        LogOutputSettings(
            outputs=(LogOutputTarget(transport="fs", destination=execution_log),)
        ),
    ):
        execution_handlers = tuple(root.handlers)
        logger.info("operation")

    assert tuple(root.handlers) == command_handlers
    assert root.level == command_level
    assert all(
        getattr(handler, "stream", None) is None for handler in execution_handlers
    )
    route_execution_event(CommandFinished("serve", "success", 1.5), logger)
    _flush_root_handlers()

    assert execution_log.read_text(encoding="utf-8") == "operation\n"
    assert command_log.read_text(encoding="utf-8") == (
        "Command serve finished status=success elapsed=1.5s\n"
    )


def test_root_logging_scope_restores_command_outputs_after_failure(tmp_path) -> None:
    command_log = tmp_path / "command-error.log"
    execution_log = tmp_path / "execution-error.log"
    configure_root_logging(
        level=logging.INFO,
        output=LogOutputSettings(
            outputs=(LogOutputTarget(transport="fs", destination=command_log),)
        ),
    )
    logger = logging.getLogger("datapipeline.tests.logging_setup.failed-scope")
    root = logging.getLogger()
    command_handlers = tuple(root.handlers)

    with pytest.raises(RuntimeError, match="failed"):
        with root_logging_scope(
            logging.INFO,
            LogOutputSettings(
                outputs=(LogOutputTarget(transport="fs", destination=execution_log),)
            ),
        ):
            logger.info("operation")
            raise RuntimeError("failed")

    assert tuple(root.handlers) == command_handlers
    route_execution_event(CommandFinished("serve", "error", 1.5), logger)
    _flush_root_handlers()

    assert execution_log.read_text(encoding="utf-8") == "operation\n"
    assert command_log.read_text(encoding="utf-8") == (
        "Command serve finished status=error elapsed=1.5s\n"
    )


def test_configure_root_logging_suppresses_terminal_execution_events_during_visuals(
    monkeypatch,
):
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    configure_root_logging(
        level=logging.INFO,
        output=LogOutputSettings(outputs=(LogOutputTarget(transport="stderr"),)),
    )

    logger = logging.getLogger("datapipeline.tests.logging_setup.stderr")
    token = set_current_execution_event_handler(lambda _event: None)
    try:
        logger.info(
            "Pipeline started name=demo",
            extra={"dp_event_kind": "pipeline_start"},
        )
        logger.info("plain log line")
    finally:
        reset_current_execution_event_handler(token)
        _flush_root_handlers()

    rendered = stream.getvalue()
    assert "plain log line" in rendered
    assert "Pipeline started name=demo" not in rendered


def test_visual_command_summary_is_rendered_and_logged_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stream = io.StringIO()
    log_path = tmp_path / "command.log"
    monkeypatch.setattr(sys, "stderr", stream)
    monkeypatch.setattr(
        "datapipeline.cli.visuals.rich.progress.rich_visuals_supported",
        lambda: True,
    )
    configure_root_logging(
        level=logging.INFO,
        output=LogOutputSettings(
            outputs=(
                LogOutputTarget(transport="stderr"),
                LogOutputTarget(transport="fs", destination=log_path),
            )
        ),
    )
    event = CommandFinished("serve", "success", 1.5)
    logger = logging.getLogger("datapipeline.tests.logging_setup.command_visual")

    with visual_summary(logging.INFO, enabled=True):
        route_execution_event(event, logger)
    _flush_root_handlers()

    message = "Command serve finished status=success elapsed=1.5s"
    assert stream.getvalue().count(message) == 1
    assert log_path.read_text(encoding="utf-8") == f"{message}\n"


def test_configure_root_logging_keeps_execution_events_in_file_during_visuals(
    tmp_path,
):
    log_path = tmp_path / "logs" / "app.log"
    configure_root_logging(
        level=logging.INFO,
        output=LogOutputSettings(
            outputs=(LogOutputTarget(transport="fs", destination=log_path),)
        ),
    )

    logger = logging.getLogger("datapipeline.tests.logging_setup.file")
    token = set_current_execution_event_handler(lambda _event: None)
    try:
        logger.info(
            "Pipeline started name=demo",
            extra={"dp_event_kind": "pipeline_start"},
        )
    finally:
        reset_current_execution_event_handler(token)
        _flush_root_handlers()

    content = log_path.read_text(encoding="utf-8")
    assert "Pipeline started name=demo" in content


def test_configure_root_logging_routes_plain_terminal_logs_through_rich_visuals(
    monkeypatch,
):
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    configure_root_logging(
        level=logging.INFO,
        output=LogOutputSettings(outputs=(LogOutputTarget(transport="stderr"),)),
    )

    class _RichHandler:
        def __init__(self) -> None:
            self.events = []

        def __call__(self, event) -> None:
            self.events.append(event)

    handler = _RichHandler()
    logger = logging.getLogger("datapipeline.tests.logging_setup.rich_proxy")
    event_token = set_current_execution_event_handler(handler)
    log_token = set_current_terminal_log_handler(handler)
    try:
        logger.warning("plain log line")
    finally:
        reset_current_terminal_log_handler(log_token)
        reset_current_execution_event_handler(event_token)
        _flush_root_handlers()

    rendered = stream.getvalue()
    assert "plain log line" not in rendered
    assert len(handler.events) == 1
    event = handler.events[0]
    assert isinstance(event, ExecutionMessage)
    assert event.message == "plain log line"
    assert event.log_level == logging.WARNING


def test_configure_root_logging_does_not_proxy_plain_logs_without_log_handler(
    monkeypatch,
):
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    configure_root_logging(
        level=logging.INFO,
        output=LogOutputSettings(outputs=(LogOutputTarget(transport="stderr"),)),
    )

    logger = logging.getLogger("datapipeline.tests.logging_setup.no_proxy")
    token = set_current_execution_event_handler(lambda _event: None)
    try:
        logger.warning("plain log line")
    finally:
        reset_current_execution_event_handler(token)
        _flush_root_handlers()

    rendered = stream.getvalue()
    assert "plain log line" in rendered


def test_parse_log_output_specs_parses_supported_targets(tmp_path):
    outputs = parse_log_output_specs(
        ["stderr", "execution:logs/serve.log", f"fs:{tmp_path / 'jerry.log'}"],
        resolve_global_path=Path,
    )

    assert [output.transport for output in outputs] == ["stderr", "fs", "fs"]
    assert [output.scope for output in outputs] == [
        "global",
        "execution",
        "global",
    ]
    assert outputs[1].destination == Path("logs/serve.log")
    assert outputs[2].destination == tmp_path / "jerry.log"


def test_parse_log_output_specs_rejects_invalid_values():
    with pytest.raises(ValueError, match="invalid --log-output value"):
        parse_log_output_specs(["file:./x.log"], resolve_global_path=Path)
