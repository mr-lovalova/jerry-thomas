from itertools import islice

from datapipeline.analysis.vector.matrix import MatrixBuilder, render_matrix_html
from datapipeline.artifacts.registry import VECTOR_METADATA_SPEC
from datapipeline.config.tasks.matrix import MatrixTask
from datapipeline.operations.persistence import RuntimeOutput
from datapipeline.pipelines.dataset.pipeline import (
    run_dataset_pipeline,
    run_sample_pipeline,
)
from datapipeline.pipelines.sample.keys import require_metadata_key_plan
from datapipeline.runtime import Runtime


def run_matrix_operation(
    runtime: Runtime,
    task: MatrixTask,
    limit: int | None = None,
) -> RuntimeOutput:
    options = task.options
    dataset = runtime.dataset
    metadata = runtime.artifacts.load(VECTOR_METADATA_SPEC)
    schema = metadata.catalog
    key_plan = require_metadata_key_plan(
        schema.window,
        schema.sample,
        dataset.sample.cadence,
        dataset.sample.keys,
    )

    if options.stage == "postprocessed":
        samples = run_dataset_pipeline(runtime, schema, key_plan)
    else:
        samples = run_sample_pipeline(runtime, schema, key_plan)

    builder = MatrixBuilder(schema.features, schema.targets, options.max_cells)
    limited_samples = islice(samples, limit) if limit is not None else samples
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
