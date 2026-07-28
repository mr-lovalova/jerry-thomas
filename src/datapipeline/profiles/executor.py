import logging
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

from datapipeline.cli.logging_setup import root_logging_scope
from datapipeline.cli.visuals.execution import make_execution_observer
from datapipeline.cli.visuals.rich.progress import (
    rich_visuals_supported,
    visual_execution,
)
from datapipeline.execution.observability import execution_observer
from datapipeline.execution.settings import ObservabilitySettings
from datapipeline.runtime import Runtime


@contextmanager
def execution_scope(
    runtime: Runtime,
    observability: ObservabilitySettings,
) -> Iterator[None]:
    level = observability.log_decision.value
    with root_logging_scope(level, observability.log_output):
        visuals_active = observability.visuals == "on" and rich_visuals_supported()
        visuals = visual_execution(level) if visuals_active else nullcontext()
        observer = make_execution_observer(logging.getLogger("datapipeline.execution"))

        previous_observe_node_events = runtime.observe_node_events
        runtime.observe_node_events = visuals_active or level <= logging.DEBUG

        try:
            with execution_observer(observer), visuals:
                yield
        finally:
            runtime.observe_node_events = previous_observe_node_events
