from itertools import islice

from datapipeline.analysis.vector.matrix import MatrixBuilder, render_matrix_html
from datapipeline.artifacts.registry import VECTOR_METADATA_SPEC
from datapipeline.config.tasks import MatrixTask
from datapipeline.execution.context import PipelineContext
from datapipeline.operations.persistence import RuntimeOutput
from datapipeline.pipelines.dataset.postprocess import build_postprocess_plan
from datapipeline.pipelines.sample.input import open_samples
from datapipeline.pipelines.sample.keys import require_metadata_key_plan
from datapipeline.runtime import Runtime


def run_matrix_operation(
    runtime: Runtime,
    task: MatrixTask,
    limit: int | None = None,
) -> RuntimeOutput:
    options = task.options
    dataset = runtime.dataset
    context = PipelineContext(runtime)
    metadata = context.require_artifact(VECTOR_METADATA_SPEC)
    schema = metadata.catalog
    key_plan = require_metadata_key_plan(
        schema.window,
        schema.sample,
        dataset.sample.cadence,
        dataset.sample.keys,
    )

    samples = open_samples(
        context,
        tuple(entry.id for entry in schema.features),
        dataset.sample.cadence,
        target_ids=tuple(entry.id for entry in schema.targets),
        sample_keys=dataset.sample.keys,
        key_plan=key_plan,
    )
    feature_entries = schema.features
    target_entries = schema.targets
    if options.stage == "postprocessed":
        plan = build_postprocess_plan(dataset.postprocess, schema)
        samples = plan.apply(samples)

    builder = MatrixBuilder(feature_entries, target_entries, options.max_cells)
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
