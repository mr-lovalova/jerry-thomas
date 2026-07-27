import pytest

from datapipeline.artifacts.hydration import hydrate_runtime_artifacts_for_pipeline
from datapipeline.artifacts.registry import VECTOR_METADATA_SPEC
from datapipeline.artifacts.specs import (
    SCALER_STATISTICS,
    SERIES,
    VECTOR_METADATA,
)
from datapipeline.config.tasks.metadata import MetadataTask
from datapipeline.config.tasks.scaler import ScalerTask
from datapipeline.config.tasks.series import SeriesTask
from datapipeline.operations.artifacts.metadata import build_metadata_artifact
from datapipeline.operations.artifacts.scaler import build_scaler_artifact
from datapipeline.operations.artifacts.series import build_series_artifact
from datapipeline.pipelines.dataset.pipeline import run_scaled_dataset_pipeline
from datapipeline.services.project_definition import load_project_definition
from datapipeline.services.runtime_compiler import compile_runtime


def _dataset_samples(project_yaml):
    definition = load_project_definition(project_yaml)
    runtime = compile_runtime(definition)
    hydrate_runtime_artifacts_for_pipeline(runtime, definition)

    # Ensure artifacts are materialized for the test run.
    scaler_task = ScalerTask(id="scaler", output="scaler.json")
    build_scaler_artifact(runtime, scaler_task)
    runtime.artifacts.register(
        SCALER_STATISTICS,
        relative_path=scaler_task.output,
    )
    series_task = SeriesTask(id="series", output="series/manifest.json")
    build_series_artifact(runtime, series_task)
    runtime.artifacts.register(
        SERIES,
        relative_path=series_task.output,
    )
    metadata_task = MetadataTask(id="metadata", output="metadata.json")
    build_metadata_artifact(runtime, metadata_task)
    runtime.artifacts.register(
        VECTOR_METADATA,
        relative_path=metadata_task.output,
    )
    metadata = runtime.artifacts.load(VECTOR_METADATA_SPEC)
    return list(
        run_scaled_dataset_pipeline(
            runtime,
            schema=metadata.catalog,
            key_plan=None,
        )
    )


def test_incomplete_prices_project_samples(copy_fixture):
    project_root = copy_fixture("incomplete_prices_project")
    project = project_root / "project.yaml"
    samples = _dataset_samples(project)

    assert len(samples) == 8

    first = samples[0]
    assert first.key[0].hour == 3
    assert len(first.features.values) == 14  # 7 areas x 2 feature ids
    values = first.features.values
    assert values["spot_eur_scaled__@area:DK1"] == pytest.approx(
        -1.0020365384, rel=1e-6
    )
    assert values["spot_eur_scaled__@area:SYSTEM"] == pytest.approx(
        -1.3841396412, rel=1e-6
    )
    assert all(v is None for v in values["spot_eur_sequence__@area:DK1"])

    # Window stride keeps only a subset of buckets populated; others stay empty after postprocess.
    empty_window = samples[1].features.values["spot_eur_sequence__@area:DK1"]
    assert all(v is None for v in empty_window)

    populated = samples[2].features.values["spot_eur_sequence__@area:DK1"]
    assert populated == pytest.approx([37.669998, 39.700001, 40.59], rel=1e-6)


def test_incomplete_generation_project_alignment(copy_fixture):
    project_root = copy_fixture("incomplete_generation_project")
    project = project_root / "project.yaml"
    samples = _dataset_samples(project)

    assert len(samples) == 9

    expected_features = {
        "onshore_mwh_scaled__@municipality:400",
        "onshore_mwh_scaled__@municipality:550",
        "onshore_mwh_scaled__@municipality:849",
        "onshore_mwh_window__@municipality:400",
        "onshore_mwh_window__@municipality:550",
        "onshore_mwh_window__@municipality:849",
    }

    first = samples[0]
    assert set(first.features.keys()) == expected_features
    assert first.targets is not None
    assert first.targets.values["dk1_price"] == pytest.approx(39.700001, rel=1e-6)
    assert first.features.values[
        "onshore_mwh_scaled__@municipality:849"
    ] == pytest.approx(0.2560143735, rel=1e-6)
    assert first.features.values["onshore_mwh_window__@municipality:849"] == [
        None,
        None,
    ]

    window_sample = samples[3]
    assert window_sample.key[0].hour == 7
    window = window_sample.features.values["onshore_mwh_window__@municipality:849"]
    assert window == pytest.approx([2.880863, 2.351027], rel=1e-6)
