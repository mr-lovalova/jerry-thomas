from itertools import islice

from jerrythomas.analysis.vector.matrix import MatrixBuilder, render_matrix_html
from jerrythomas.artifacts.registry import VECTOR_METADATA_SPEC
from jerrythomas.config.tasks.matrix import MatrixTask
from jerrythomas.operations.persistence import RuntimeOutput
from jerrythomas.operations.outputs.execution import OutputOptions
from jerrythomas.pipelines.dataset.pipeline import (
    run_dataset_pipeline,
    run_sample_pipeline,
)
from jerrythomas.pipelines.sample.keys import require_metadata_key_plan
from jerrythomas.runtime import Runtime


def run_matrix_operation(
    runtime: Runtime,
    task: MatrixTask,
    options: OutputOptions,
) -> RuntimeOutput:
    dataset = runtime.require_dataset()
    metadata = runtime.artifacts.load(VECTOR_METADATA_SPEC)
    schema = metadata.catalog
    key_plan = require_metadata_key_plan(
        schema.window,
        schema.sample,
        dataset.sample.cadence,
        dataset.sample.keys,
    )

    if task.options.stage == "postprocessed":
        samples = run_dataset_pipeline(runtime, schema, key_plan)
    else:
        samples = run_sample_pipeline(runtime, schema, key_plan)

    builder = MatrixBuilder(schema.features, schema.targets, task.options.max_cells)
    limited_samples = (
        islice(samples, options.limit) if options.limit is not None else samples
    )
    try:
        for sample in limited_samples:
            targets = sample.targets.values if sample.targets is not None else {}
            builder.add(sample.key, sample.features.values, targets)
    finally:
        close = getattr(samples, "close", None)
        if callable(close):
            close()
    matrix = builder.finish()

    return RuntimeOutput(
        rows=matrix.output_rows(),
        render_html=lambda: render_matrix_html(matrix),
    )
