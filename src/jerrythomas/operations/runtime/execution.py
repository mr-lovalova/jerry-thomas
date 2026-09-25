from dataclasses import dataclass

from jerrythomas.config.preview import PreviewStage
from jerrythomas.config.profiles.output import Format
from jerrythomas.config.tasks.base import PluginRuntimeTask, RuntimeTask
from jerrythomas.config.tasks.registry import CORE_OPERATION_MODELS
from jerrythomas.operations.persistence import (
    RuntimeOutput,
    RuntimeOutputBatch,
    RuntimeOutputItem,
)
from jerrythomas.plugins import RUNTIME_OPERATIONS_EP, load_entrypoint
from jerrythomas.runtime import Runtime


@dataclass(frozen=True, kw_only=True)
class OutputOptions:
    limit: int | None = None
    output_format: Format = "jsonl"
    throttle_ms: float | None = None
    preview: PreviewStage | None = None
    output_ids: tuple[str, ...] = ()


def run_output_operation(
    runtime: Runtime,
    task: RuntimeTask,
    options: OutputOptions,
) -> RuntimeOutputItem | RuntimeOutputBatch | None:
    if not isinstance(task, RuntimeTask):
        raise TypeError("Output operations require a RuntimeTask.")
    model = CORE_OPERATION_MODELS.get(task.entrypoint)
    if model is None:
        if task.entrypoint.startswith("core."):
            raise ValueError(f"Unknown built-in output entrypoint '{task.entrypoint}'.")
        model = PluginRuntimeTask
    if not isinstance(task, model):
        raise TypeError(
            f"Output entrypoint '{task.entrypoint}' requires {model.__name__}."
        )
    runner = load_entrypoint(RUNTIME_OPERATIONS_EP, task.entrypoint)
    result = runner(runtime, task, options)
    if isinstance(task, PluginRuntimeTask) and result is not None:
        if not isinstance(result, RuntimeOutput):
            raise TypeError(
                "Custom output operation must return RuntimeOutput or None."
            )
    if result is not None and not isinstance(
        result, RuntimeOutputItem | RuntimeOutputBatch
    ):
        raise TypeError("Output operation returned an unsupported output type.")
    return result
