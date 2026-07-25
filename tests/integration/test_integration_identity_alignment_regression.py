import math
import shutil

import pytest

from datapipeline.artifacts.hydration import hydrate_runtime_artifacts_for_pipeline
from datapipeline.artifacts.models import FoldedMetadataLayout
from datapipeline.artifacts.registry import (
    SCALER_SPEC,
    VECTOR_METADATA_SPEC,
)
from datapipeline.artifacts.scaler import FoldedScalerArtifact
from datapipeline.artifacts.specs import SERIES
from datapipeline.services.runtime_compiler import compile_runtime
from datapipeline.artifacts.series import load_series_manifest
from tests.helpers.regression import read_jsonl, serve_dataset


def test_long_and_hybrid_identity_with_aligned_derived_stream(copy_fixture) -> None:
    project_root = copy_fixture("identity_alignment_project")
    request = serve_dataset(project_root)

    runtime = compile_runtime(request.definition)
    hydrated = hydrate_runtime_artifacts_for_pipeline(runtime, request.definition)
    assert set(hydrated) >= {"series", "scaler", "metadata"}

    manifest = load_series_manifest(runtime.artifacts.resolve_path(SERIES))
    assert manifest.sample_keys == ("ticker",)
    assert manifest.sample_key_types == ("string",)
    assert [(entry.id, entry.samples) for entry in manifest.features] == [
        ("price_scaled", 6),
        ("price_history", 4),
        ("price_mean_2", 6),
        ("price_lag_1", 6),
        ("price_lead_1", 6),
        ("pe_ratio", 3),
        ("fundamental", 6),
    ]

    scaler_artifact = runtime.artifacts.load(SCALER_SPEC)
    assert scaler_artifact.observations == 6
    assert set(scaler_artifact.statistics) == {"price_scaled"}
    price_statistics = scaler_artifact.statistics["price_scaled"]
    assert price_statistics.count == 6
    assert price_statistics.mean == pytest.approx(12.0)
    assert price_statistics.std == pytest.approx(math.sqrt(296 / 3))

    metadata_artifact = runtime.artifacts.load(VECTOR_METADATA_SPEC)
    assert metadata_artifact.layout.kind == "unsplit"
    catalog = metadata_artifact.catalog
    assert catalog.counts.feature_vectors == 6
    assert catalog.counts.target_vectors == 0
    assert catalog.sample is not None
    assert catalog.sample.keys == ["ticker"]
    assert [entry.key for entry in catalog.sample.domain] == [["A"], ["B"]]

    assert [(entry.id, entry.kind) for entry in catalog.features] == [
        ("price_scaled", "scalar"),
        ("price_history", "list"),
        ("price_mean_2", "scalar"),
        ("price_lag_1", "scalar"),
        ("price_lead_1", "scalar"),
        ("pe_ratio", "scalar"),
        ("fundamental__@metric:debt", "scalar"),
        ("fundamental__@metric:revenue", "scalar"),
    ]
    price_history = catalog.features[1]
    assert price_history.kind == "list"
    assert price_history.length == 2
    assert catalog.targets == ()

    dataset_path = request.serve_run_plans[0].paths.dataset_dir / "dataset.jsonl"
    samples = read_jsonl(dataset_path)
    price_std = math.sqrt(296 / 3)
    expected = [
        (1, "A", (2 - 12) / price_std, [None, None], 2.0, None, 4, 1.0, 50, 100),
        (
            1,
            "B",
            (10 - 12) / price_std,
            [None, None],
            10.0,
            None,
            20,
            None,
            80,
            200,
        ),
        (2, "A", (4 - 12) / price_std, [2, 4], 3.0, 2, 6, None, 55, 110),
        (
            2,
            "B",
            (20 - 12) / price_std,
            [10, 20],
            15.0,
            10,
            30,
            20.0,
            None,
            220,
        ),
        (3, "A", (6 - 12) / price_std, [4, 6], 5.0, 4, None, 2.0, 60, 120),
        (
            3,
            "B",
            (30 - 12) / price_std,
            [20, 30],
            25.0,
            20,
            None,
            None,
            96,
            240,
        ),
    ]
    assert len(samples) == len(expected)
    for sample, (
        day,
        ticker,
        price,
        history,
        mean_2,
        lag_1,
        lead_1,
        pe,
        debt,
        revenue,
    ) in zip(
        samples,
        expected,
        strict=True,
    ):
        assert sample["key"] == [f"2024-01-{day:02d} 00:00:00+00:00", ticker]
        assert list(sample["features"]["values"]) == [
            "price_scaled",
            "price_history",
            "price_mean_2",
            "price_lag_1",
            "price_lead_1",
            "pe_ratio",
            "fundamental__@metric:debt",
            "fundamental__@metric:revenue",
        ]
        assert sample["features"]["values"] == pytest.approx(
            {
                "price_scaled": price,
                "price_history": history,
                "price_mean_2": mean_2,
                "price_lag_1": lag_1,
                "price_lead_1": lead_1,
                "pe_ratio": pe,
                "fundamental__@metric:debt": debt,
                "fundamental__@metric:revenue": revenue,
            }
        )
        assert sample["targets"] is None


def test_validation_availability_does_not_change_hybrid_wide_training_contract(
    copy_fixture,
    tmp_path,
) -> None:
    project_root = copy_fixture("identity_alignment_project")
    changed_root = tmp_path / "identity_alignment_changed_validation"
    shutil.copytree(project_root, changed_root)
    dataset = """sample:
  cadence: 1d
  keys: [ticker]
features:
  - id: price
    stream: market.price
    field: value
    scale: true
  - id: fundamental
    stream: company.fundamental
    field: value
    scale: true
split:
  mode: time
  intervals:
    - {id: train, until: "2024-01-03T00:00:00Z"}
    - {id: validation}
  folds:
    - id: holdout
      train: [train]
      validation: [validation]
"""
    for root in (project_root, changed_root):
        (root / "dataset.yaml").write_text(dataset, encoding="utf-8")
        (root / "operations/metadata.yaml").write_text(
            "window_mode: intersection\n",
            encoding="utf-8",
        )

    fundamentals_path = project_root / "data" / "fundamentals.jsonl"
    training_gap = (
        "\n".join(
            line
            for line in fundamentals_path.read_text(encoding="utf-8").splitlines()
            if '"time":"2024-01-02T00:00:00Z"' not in line
        )
        + "\n"
    )
    fundamentals_path.write_text(training_gap, encoding="utf-8")
    (changed_root / "data/fundamentals.jsonl").write_text(
        "\n".join(
            line
            for line in training_gap.splitlines()
            if '"time":"2024-01-03T00:00:00Z"' not in line
        )
        + "\n",
        encoding="utf-8",
    )

    baseline_request = serve_dataset(project_root)
    changed_request = serve_dataset(changed_root)
    baseline_outputs = baseline_request.serve_run_plans[0].paths.dataset_dir
    changed_outputs = changed_request.serve_run_plans[0].paths.dataset_dir
    baseline_validation = read_jsonl(
        baseline_outputs / "dataset.holdout.validation.jsonl"
    )
    changed_validation = read_jsonl(
        changed_outputs / "dataset.holdout.validation.jsonl"
    )

    baseline_train_path = baseline_outputs / "dataset.holdout.train.jsonl"
    changed_train_path = changed_outputs / "dataset.holdout.train.jsonl"
    baseline_train = read_jsonl(baseline_train_path)
    assert [row["key"] for row in baseline_train] == [
        ["2024-01-01 00:00:00+00:00", "A"],
        ["2024-01-01 00:00:00+00:00", "B"],
    ]
    assert changed_train_path.read_bytes() == baseline_train_path.read_bytes()

    fundamental_ids = (
        "fundamental__@metric:debt",
        "fundamental__@metric:revenue",
    )
    assert [row["key"] for row in changed_validation] == [
        row["key"] for row in baseline_validation
    ]
    assert all(
        row["features"]["values"][feature_id] is not None
        for row in baseline_validation
        for feature_id in fundamental_ids
    )
    assert all(
        row["features"]["values"][feature_id] is None
        for row in changed_validation
        for feature_id in fundamental_ids
    )

    baseline_runtime = compile_runtime(baseline_request.definition)
    hydrate_runtime_artifacts_for_pipeline(
        baseline_runtime,
        baseline_request.definition,
    )
    changed_runtime = compile_runtime(changed_request.definition)
    hydrate_runtime_artifacts_for_pipeline(
        changed_runtime,
        changed_request.definition,
    )

    baseline_scaler = baseline_runtime.artifacts.load(SCALER_SPEC)
    changed_scaler = changed_runtime.artifacts.load(SCALER_SPEC)
    assert isinstance(baseline_scaler, FoldedScalerArtifact)
    assert isinstance(changed_scaler, FoldedScalerArtifact)
    assert changed_scaler.for_fold("holdout") == baseline_scaler.for_fold("holdout")

    baseline_metadata = baseline_runtime.artifacts.load(VECTOR_METADATA_SPEC)
    changed_metadata = changed_runtime.artifacts.load(VECTOR_METADATA_SPEC)
    assert isinstance(baseline_metadata.layout, FoldedMetadataLayout)
    assert isinstance(changed_metadata.layout, FoldedMetadataLayout)
    assert (
        changed_metadata.layout.folds[0].training_schema
        == baseline_metadata.layout.folds[0].training_schema
    )
