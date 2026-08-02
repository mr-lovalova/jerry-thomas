import logging
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

from jerrythomas.cli.logging_setup import root_logging_scope
from jerrythomas.cli.visuals.execution import make_execution_observer
from jerrythomas.cli.visuals.rich.progress import (
    rich_visuals_supported,
    visual_execution,
)
from jerrythomas.execution.observability import execution_observer
from jerrythomas.execution.settings import ObservabilitySettings
from jerrythomas.runtime import Runtime


@contextmanager
def execution_scope(
    runtime: Runtime,
    observability: ObservabilitySettings,
) -> Iterator[None]:
    level = observability.log_decision.value
    with root_logging_scope(level, observability.log_output):
        visuals_active = observability.visuals == "on" and rich_visuals_supported()
        visuals = visual_execution(level) if visuals_active else nullcontext()
        observer = make_execution_observer(logging.getLogger("jerrythomas.execution"))

        previous_observe_node_events = runtime.observe_node_events
        runtime.observe_node_events = visuals_active or level <= logging.DEBUG

        try:
            with execution_observer(observer), visuals:
                yield
        finally:
            runtime.observe_node_events = previous_observe_node_events
