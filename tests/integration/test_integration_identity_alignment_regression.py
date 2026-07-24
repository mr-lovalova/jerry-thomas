import math
import shutil

import pytest

from datapipeline.artifacts.hydration import hydrate_runtime_artifacts_for_pipeline
from datapipeline.artifacts.registry import (
    SCALER_SPEC,
    VECTOR_METADATA_SPEC,
)
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
    assert metadata_artifact.counts.feature_vectors == 6
    assert metadata_artifact.counts.target_vectors == 0
    assert metadata_artifact.sample is not None
    assert metadata_artifact.sample.keys == ["ticker"]
    assert [entry.key for entry in metadata_artifact.sample.domain] == [["A"], ["B"]]

    assert [(entry.id, entry.kind) for entry in metadata_artifact.features] == [
        ("price_scaled", "scalar"),
        ("price_history", "list"),
        ("price_mean_2", "scalar"),
        ("price_lag_1", "scalar"),
        ("price_lead_1", "scalar"),
        ("pe_ratio", "scalar"),
        ("fundamental__@metric:debt", "scalar"),
        ("fundamental__@metric:revenue", "scalar"),
    ]
    price_history = metadata_artifact.features[1]
    assert price_history.kind == "list"
    assert price_history.length == 2
    assert metadata_artifact.targets == ()

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


def test_validation_values_do_not_change_hybrid_wide_training_output(
    copy_fixture,
    tmp_path,
) -> None:
    project_root = copy_fixture("identity_alignment_project")
    changed_root = tmp_path / "identity_alignment_changed_validation"
    shutil.copytree(project_root, changed_root)
    split = """

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
        dataset_path = root / "dataset.yaml"
        dataset_path.write_text(
            dataset_path.read_text(encoding="utf-8") + split,
            encoding="utf-8",
        )

    fundamentals_path = changed_root / "data" / "fundamentals.jsonl"
    fundamentals_path.write_text(
        fundamentals_path.read_text(encoding="utf-8").replace(
            '"ticker":"A","metric":"revenue","value":120',
            '"ticker":"A","metric":"revenue","value":null',
        ),
        encoding="utf-8",
    )

    baseline_request = serve_dataset(project_root)
    changed_request = serve_dataset(changed_root)
    baseline_outputs = baseline_request.serve_run_plans[0].paths.dataset_dir
    changed_outputs = changed_request.serve_run_plans[0].paths.dataset_dir
    baseline_train = read_jsonl(baseline_outputs / "dataset.holdout.train.jsonl")
    changed_train = read_jsonl(changed_outputs / "dataset.holdout.train.jsonl")
    baseline_validation = read_jsonl(
        baseline_outputs / "dataset.holdout.validation.jsonl"
    )
    changed_validation = read_jsonl(
        changed_outputs / "dataset.holdout.validation.jsonl"
    )

    assert changed_train == baseline_train
    assert changed_validation != baseline_validation
    assert [set(row["features"]["values"]) for row in changed_validation] == [
        set(row["features"]["values"]) for row in baseline_validation
    ]
