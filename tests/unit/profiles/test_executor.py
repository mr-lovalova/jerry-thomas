import logging
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest

from datapipeline.config.dataset.dataset import DatasetConfig, SampleConfig
from datapipeline.execution.settings import (
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    LogLevelDecision,
    LogOutputSettings,
    LogOutputTarget,
    ObservabilitySettings,
)
from datapipeline.execution.observability import (
    current_execution_observer,
    emit_execution_message,
    execution_observer,
)
from datapipeline.profiles.executor import execution_scope
from datapipeline.runtime import Runtime


def _log_output() -> LogOutputSettings:
    return LogOutputSettings(outputs=(LogOutputTarget(transport="stderr"),))


def _runtime() -> Runtime:
    return Runtime(
        project_yaml=Path("."),
        artifacts_root=Path("."),
        dataset=DatasetConfig(sample=SampleConfig(cadence="1h")),
    )


def test_execution_scope_configures_logging_and_runs_inside_visuals(monkeypatch):
    runtime = _runtime()
    runtime.observe_node_events = False
    calls = []
    inside_visual_context = False

    @contextmanager
    def root_logging_scope(level, output):
        calls.append(("logging", level, output))
        yield

    monkeypatch.setattr(
        "datapipeline.profiles.executor.root_logging_scope",
        root_logging_scope,
    )

    monkeypatch.setattr(
        "datapipeline.profiles.executor.rich_visuals_supported",
        lambda: True,
    )

    @contextmanager
    def visual_execution(level):
        nonlocal inside_visual_context
        calls.append(("visuals", level))
        inside_visual_context = True
        try:
            yield
        finally:
            inside_visual_context = False

    monkeypatch.setattr(
        "datapipeline.profiles.executor.visual_execution",
        visual_execution,
    )

    log_output = _log_output()
    with execution_scope(
        runtime,
        ObservabilitySettings(
            visuals="on",
            heartbeat_interval_seconds=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
            log_decision=LogLevelDecision(name="INFO", value=20),
            log_output=log_output,
        ),
    ):
        assert current_execution_observer() is not None
        assert runtime.observe_node_events
        calls.append(("work", inside_visual_context))

    assert calls == [
        ("logging", 20, log_output),
        ("visuals", 20),
        ("work", True),
    ]
    assert current_execution_observer() is None
    assert not runtime.observe_node_events


def test_execution_scope_routes_messages_to_file(tmp_path: Path) -> None:
    log_path = tmp_path / "execution.log"

    with execution_scope(
        _runtime(),
        ObservabilitySettings(
            visuals="off",
            heartbeat_interval_seconds=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
            log_decision=LogLevelDecision(name="DEBUG", value=logging.DEBUG),
            log_output=LogOutputSettings(
                outputs=(LogOutputTarget(transport="fs", destination=log_path),)
            ),
        ),
    ):
        assert emit_execution_message("Config:\n{}", logging.DEBUG)

    assert "Config:" in log_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("visuals", "rich_supported"),
    [("off", True), ("on", False)],
)
def test_execution_scope_uses_plain_context_when_visuals_are_unavailable(
    monkeypatch,
    visuals,
    rich_supported,
) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        "datapipeline.profiles.executor.root_logging_scope",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        "datapipeline.profiles.executor.rich_visuals_supported",
        lambda: rich_supported,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.executor.visual_execution",
        lambda _level: pytest.fail("Rich visuals must not start"),
    )

    with execution_scope(
        runtime,
        ObservabilitySettings(
            visuals=visuals,
            heartbeat_interval_seconds=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
            log_decision=LogLevelDecision(name="INFO", value=20),
            log_output=_log_output(),
        ),
    ):
        assert current_execution_observer() is not None
        assert not runtime.observe_node_events

    assert current_execution_observer() is None
    assert runtime.observe_node_events


def test_execution_scope_observes_nodes_for_debug_logging(monkeypatch) -> None:
    runtime = _runtime()
    runtime.observe_node_events = False
    monkeypatch.setattr(
        "datapipeline.profiles.executor.root_logging_scope",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        "datapipeline.profiles.executor.rich_visuals_supported",
        lambda: pytest.fail("visual support is irrelevant when visuals are off"),
    )

    with execution_scope(
        runtime,
        ObservabilitySettings(
            visuals="off",
            heartbeat_interval_seconds=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
            log_decision=LogLevelDecision(name="DEBUG", value=10),
            log_output=_log_output(),
        ),
    ):
        assert current_execution_observer() is not None
        assert runtime.observe_node_events

    assert current_execution_observer() is None
    assert not runtime.observe_node_events


def test_execution_scope_restores_observation_after_failure(monkeypatch) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        "datapipeline.profiles.executor.root_logging_scope",
        lambda *_args, **_kwargs: nullcontext(),
    )

    with pytest.raises(RuntimeError, match="failed"):
        with execution_scope(
            runtime,
            ObservabilitySettings(
                visuals="off",
                heartbeat_interval_seconds=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
                log_decision=LogLevelDecision(name="INFO", value=20),
                log_output=_log_output(),
            ),
        ):
            assert not runtime.observe_node_events
            raise RuntimeError("failed")

    assert current_execution_observer() is None
    assert runtime.observe_node_events


def test_execution_scope_restores_outer_state_after_visual_cleanup_failure(
    monkeypatch,
) -> None:
    runtime = _runtime()
    calls = []
    operation_active = False

    @contextmanager
    def logging_scope(*_args):
        calls.append("logging enter")
        try:
            yield
        finally:
            calls.append("logging exit")

    @contextmanager
    def observe_operation(_observer):
        nonlocal operation_active
        calls.append("operation enter")
        operation_active = True
        try:
            yield
        finally:
            operation_active = False
            calls.append("operation exit")

    @contextmanager
    def visuals(_level):
        calls.append("visuals enter")
        try:
            yield
        finally:
            assert operation_active
            calls.append("visuals exit")
            raise RuntimeError("visual cleanup failed")

    monkeypatch.setattr(
        "datapipeline.profiles.executor.root_logging_scope",
        logging_scope,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.executor.execution_observer",
        observe_operation,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.executor.rich_visuals_supported",
        lambda: True,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.executor.visual_execution",
        visuals,
    )

    with pytest.raises(RuntimeError, match="visual cleanup failed"):
        with execution_scope(
            runtime,
            ObservabilitySettings(
                visuals="on",
                heartbeat_interval_seconds=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
                log_decision=LogLevelDecision(name="INFO", value=20),
                log_output=_log_output(),
            ),
        ):
            calls.append("work")

    assert calls == [
        "logging enter",
        "operation enter",
        "visuals enter",
        "work",
        "visuals exit",
        "operation exit",
        "logging exit",
    ]
    assert runtime.observe_node_events


def test_execution_scope_restores_outer_execution_observer(monkeypatch) -> None:
    runtime = _runtime()
    runtime.observe_node_events = False

    def outer_observer(_event) -> None:
        pass

    def inner_observer(_event) -> None:
        pass

    monkeypatch.setattr(
        "datapipeline.profiles.executor.root_logging_scope",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        "datapipeline.profiles.executor.rich_visuals_supported",
        lambda: False,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.executor.make_execution_observer",
        lambda _logger: inner_observer,
    )

    with execution_observer(outer_observer):
        with execution_scope(
            runtime,
            ObservabilitySettings(
                visuals="on",
                heartbeat_interval_seconds=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
                log_decision=LogLevelDecision(name="INFO", value=20),
                log_output=_log_output(),
            ),
        ):
            assert current_execution_observer() is inner_observer
        assert current_execution_observer() is outer_observer

    assert current_execution_observer() is None
    assert not runtime.observe_node_events
