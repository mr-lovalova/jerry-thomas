from datetime import datetime, timezone

from datapipeline.artifacts.hydration import hydrate_runtime_artifacts_for_pipeline
from datapipeline.artifacts.specs import VECTOR_METADATA
from datapipeline.config.tasks import MetadataTask
from datapipeline.execution.context import PipelineContext
from datapipeline.operations.artifacts.metadata import materialize_metadata
from datapipeline.pipelines.dataset.postprocess import apply_postprocess
from datapipeline.pipelines.sample.input import open_samples
from datapipeline.services.project_definition import load_project_definition
from datapipeline.services.runtime_compiler import compile_runtime
from tests.series_helpers import register_series


def test_drop_with_metadata_and_partitioned_streams(copy_fixture):
    project_root = copy_fixture("drop_null_project")
    project = project_root / "project.yaml"
    definition = load_project_definition(project)
    runtime = compile_runtime(definition)
    hydrate_runtime_artifacts_for_pipeline(runtime, definition)
    dataset = definition.dataset
    context = PipelineContext(runtime)
    register_series(
        runtime,
        dataset.features,
        dataset.sample.cadence,
        targets=dataset.targets,
    )
    metadata = materialize_metadata(
        runtime,
        MetadataTask(id="metadata", output="metadata.json"),
    )
    runtime.artifacts.register(
        VECTOR_METADATA,
        relative_path=metadata.relative_path,
    )
    assembled_samples = open_samples(
        context,
        dataset.features,
        dataset.sample.cadence,
        target_configs=dataset.targets,
        rectangular=False,
    )
    samples = list(apply_postprocess(context, assembled_samples))

    # Source emits ticks every 2h; ensure_cadence fills 1h gaps with None.
    # Full feature coverage keeps only the original ticks.
    expected_hours = [0, 2, 4]
    assert len(samples) == len(expected_hours)
    for sample, hour in zip(samples, expected_hours):
        ts = sample.key[0]
        assert isinstance(ts, datetime)
        assert ts.tzinfo == timezone.utc
        assert ts.hour == hour
        assert sample.features.values["time_linear"] is not None
