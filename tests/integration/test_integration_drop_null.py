from datetime import datetime, timezone

from jerrythomas.artifacts.hydration import hydrate_runtime_artifacts_for_pipeline
from jerrythomas.artifacts.registry import VECTOR_METADATA_SPEC
from jerrythomas.artifacts.specs import VECTOR_METADATA
from jerrythomas.config.tasks.metadata import MetadataTask
from jerrythomas.operations.artifacts.metadata import build_metadata_artifact
from jerrythomas.pipelines.dataset.postprocess import build_postprocess_plan
from jerrythomas.pipelines.sample.input import open_samples
from jerrythomas.services.project_definition import load_project_definition
from jerrythomas.services.runtime_compiler import compile_runtime
from tests.series_helpers import register_series


def test_drop_with_metadata_and_partitioned_streams(copy_fixture):
    project_root = copy_fixture("drop_null_project")
    project = project_root / "project.yaml"
    definition = load_project_definition(project)
    runtime = compile_runtime(definition)
    hydrate_runtime_artifacts_for_pipeline(runtime, definition)
    dataset = definition.dataset
    register_series(
        runtime,
        dataset.features,
        dataset.sample.cadence,
        targets=dataset.targets,
    )
    metadata_task = MetadataTask(id="metadata", output="metadata.json")
    build_metadata_artifact(runtime, metadata_task)
    runtime.artifacts.register(
        VECTOR_METADATA,
        relative_path=metadata_task.output,
    )
    metadata_artifact = runtime.artifacts.load(VECTOR_METADATA_SPEC)
    schema = metadata_artifact.catalog
    assembled_samples = open_samples(
        runtime,
        [entry.id for entry in schema.features],
        target_ids=[entry.id for entry in schema.targets],
        key_plan=None,
    )
    samples = list(
        build_postprocess_plan(
            dataset.postprocess,
            schema,
        ).apply(assembled_samples)
    )

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
