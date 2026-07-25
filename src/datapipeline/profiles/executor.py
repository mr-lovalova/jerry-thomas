import logging
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

from datapipeline.cli.logging_setup import root_logging_scope
from datapipeline.cli.visuals.execution import (
    make_operation_observer,
    make_pipeline_observer,
)
from datapipeline.cli.visuals.rich.progress import (
    rich_visuals_supported,
    visual_execution,
)
from datapipeline.execution.observability import operation_observer
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

        previous_pipeline_observer = runtime.pipeline_observer
        previous_observe_node_events = runtime.observe_node_events
        if previous_pipeline_observer is None:
            runtime.pipeline_observer = make_pipeline_observer(
                logging.getLogger("datapipeline.execution.observer")
            )
            runtime.observe_node_events = visuals_active or level <= logging.DEBUG

        try:
            observer = make_operation_observer(
                logging.getLogger("datapipeline.operation.observer")
            )
            with operation_observer(observer), visuals:
                yield
        finally:
            if previous_pipeline_observer is None:
                runtime.pipeline_observer = None
                runtime.observe_node_events = previous_observe_node_events
