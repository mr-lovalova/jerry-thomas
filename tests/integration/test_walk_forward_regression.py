import json
import shutil
from pathlib import Path

import jerrythomas.pipelines.sample.input as sample_input

from jerrythomas.artifacts.scaler import FoldedScalerArtifact, load_scaler_artifact
from tests.helpers.regression import read_jsonl, serve_dataset


def _serve(project_root: Path) -> tuple[Path, FoldedScalerArtifact]:
    request = serve_dataset(project_root)

    run_paths = request.serve_run_plans[0].paths
    run_metadata = json.loads(run_paths.metadata_path.read_text(encoding="utf-8"))
    assert run_metadata["status"] == "success"
    assert run_metadata["finished_at"] is not None
    assert (run_paths.serve_root / "latest").resolve() == run_paths.run_root.resolve()

    scaler = load_scaler_artifact(project_root / "artifacts" / "build" / "scaler.json")
    assert isinstance(scaler, FoldedScalerArtifact)
    return run_paths.dataset_dir, scaler


def _read_output(directory: Path, output_id: str) -> list[dict[str, object]]:
    return read_jsonl(directory / f"dataset.{output_id}.jsonl")


def _sample(day: int, signal: float, outcome: float) -> dict[str, object]:
    return {
        "key": [f"2024-01-{day:02d} 00:00:00+00:00"],
        "features": {"values": {"signal": signal}},
        "targets": {"values": {"outcome": outcome}},
    }


def test_walk_forward_scaling_and_routed_outputs(copy_fixture, tmp_path: Path) -> None:
    project_root = copy_fixture("walk_forward_project")
    changed_root = tmp_path / "walk_forward_changed_validation"
    shutil.copytree(project_root, changed_root)

    output_dir, scaler = _serve(project_root)

    assert {path.name for path in output_dir.iterdir()} == {
        "dataset.fold_0.train.jsonl",
        "dataset.fold_0.validation.jsonl",
        "dataset.fold_0.test.jsonl",
        "dataset.fold_1.train.jsonl",
        "dataset.fold_1.validation.jsonl",
        "dataset.fold_1.test.jsonl",
    }
    assert scaler.model_dump(mode="json") == {
        "kind": "folded_scaler",
        "version": 4,
        "folds": {
            "fold_0": {
                "kind": "standard_scaler",
                "version": 4,
                "with_mean": True,
                "with_std": True,
                "epsilon": 1e-12,
                "observations": 4,
                "statistics": {
                    "signal": {"mean": 1.0, "std": 1.0, "count": 2},
                    "outcome": {"mean": 12.0, "std": 2.0, "count": 2},
                },
            },
            "fold_1": {
                "kind": "standard_scaler",
                "version": 4,
                "with_mean": True,
                "with_std": True,
                "epsilon": 1e-12,
                "observations": 10,
                "statistics": {
                    "signal": {
                        "mean": 6.0,
                        "std": 5.215361924162119,
                        "count": 5,
                    },
                    "outcome": {
                        "mean": 50.0,
                        "std": 44.23573216303761,
                        "count": 5,
                    },
                },
            },
        },
    }

    assert _read_output(output_dir, "fold_0.train") == [
        _sample(1, -1.0, -1.0),
        _sample(2, 1.0, 1.0),
    ]
    assert _read_output(output_dir, "fold_0.validation") == [_sample(4, 3.0, 3.0)]
    assert _read_output(output_dir, "fold_0.test") == [_sample(5, 4.0, 4.0)]
    assert _read_output(output_dir, "fold_1.train") == [
        _sample(1, -1.1504474832710556, -0.9042463647391171),
        _sample(2, -0.7669649888473704, -0.8138217282652054),
        _sample(4, -0.3834824944236852, -0.7233970917912936),
        _sample(6, 0.7669649888473704, 1.1303079559238964),
        _sample(7, 1.5339299776947408, 1.3111572288717197),
    ]
    assert _read_output(output_dir, "fold_1.validation") == [
        _sample(9, 1.917412472118426, 1.4015818653456316)
    ]
    assert _read_output(output_dir, "fold_1.test") == [
        _sample(10, 2.3008949665421112, 1.4920065018195432)
    ]

    signal_path = changed_root / "data" / "signal.csv"
    signal_text = signal_path.read_text(encoding="utf-8")
    signal_text = signal_text.replace(
        "2024-01-04T00:00:00Z,4", "2024-01-04T00:00:00Z,4000"
    ).replace("2024-01-05T00:00:00Z,5", "2024-01-05T00:00:00Z,5000")
    signal_path.write_text(signal_text, encoding="utf-8")

    outcome_path = changed_root / "data" / "outcome.csv"
    outcome_text = outcome_path.read_text(encoding="utf-8")
    outcome_text = outcome_text.replace(
        "2024-01-04T00:00:00Z,18", "2024-01-04T00:00:00Z,18000"
    ).replace("2024-01-05T00:00:00Z,20", "2024-01-05T00:00:00Z,20000")
    outcome_path.write_text(outcome_text, encoding="utf-8")

    changed_output_dir, changed_scaler = _serve(changed_root)

    assert changed_scaler.folds["fold_0"] == scaler.folds["fold_0"]
    assert _read_output(changed_output_dir, "fold_0.train") == _read_output(
        output_dir,
        "fold_0.train",
    )
    assert _read_output(changed_output_dir, "fold_0.validation") != _read_output(
        output_dir,
        "fold_0.validation",
    )
    assert changed_scaler.folds["fold_1"] != scaler.folds["fold_1"]
    assert _read_output(changed_output_dir, "fold_1.train") != _read_output(
        output_dir,
        "fold_1.train",
    )


def test_training_rows_do_not_depend_on_validation_availability(
    copy_fixture,
    tmp_path: Path,
) -> None:
    baseline_root = copy_fixture("walk_forward_project")
    changed_root = tmp_path / "walk_forward_missing_validation"
    shutil.copytree(baseline_root, changed_root)

    baseline_signal = "\n".join(
        [
            "time,value",
            "2024-01-01T00:00:00Z,0",
            "2024-01-04T00:00:00Z,4",
            "",
        ]
    )
    (baseline_root / "data" / "signal.csv").write_text(
        baseline_signal,
        encoding="utf-8",
    )
    (changed_root / "data" / "signal.csv").write_text(
        "time,value\n2024-01-01T00:00:00Z,0\n",
        encoding="utf-8",
    )

    baseline_output, _ = _serve(baseline_root)
    changed_output, _ = _serve(changed_root)
    baseline_train = _read_output(baseline_output, "fold_0.train")
    changed_train = _read_output(changed_output, "fold_0.train")

    assert [row["key"][0] for row in baseline_train] == [
        "2024-01-01 00:00:00+00:00",
    ]
    assert changed_train == baseline_train
    assert _read_output(changed_output, "fold_0.validation") != _read_output(
        baseline_output,
        "fold_0.validation",
    )


def test_target_horizon_removes_each_fold_role_tail(copy_fixture) -> None:
    project_root = copy_fixture("walk_forward_project")
    dataset_path = project_root / "dataset.yaml"
    dataset_path.write_text(
        dataset_path.read_text(encoding="utf-8").replace(
            "    horizon: 0s",
            "    horizon: 2d",
        ),
        encoding="utf-8",
    )

    output_dir, scaler = _serve(project_root)

    assert [row["key"][0] for row in _read_output(output_dir, "fold_0.train")] == [
        "2024-01-01 00:00:00+00:00"
    ]
    assert _read_output(output_dir, "fold_0.validation") == []
    assert [row["key"][0] for row in _read_output(output_dir, "fold_0.test")] == [
        "2024-01-05 00:00:00+00:00"
    ]
    assert [row["key"][0] for row in _read_output(output_dir, "fold_1.train")] == [
        "2024-01-01 00:00:00+00:00",
        "2024-01-02 00:00:00+00:00",
        "2024-01-04 00:00:00+00:00",
        "2024-01-06 00:00:00+00:00",
    ]
    assert _read_output(output_dir, "fold_1.validation") == []
    assert [row["key"][0] for row in _read_output(output_dir, "fold_1.test")] == [
        "2024-01-10 00:00:00+00:00"
    ]

    assert scaler.folds["fold_0"].statistics["signal"].count == 1
    assert scaler.folds["fold_0"].statistics["outcome"].count == 1
    assert scaler.folds["fold_1"].statistics["signal"].count == 4
    assert scaler.folds["fold_1"].statistics["outcome"].count == 4


def test_target_horizon_protects_future_derived_values(copy_fixture) -> None:
    project_root = copy_fixture("walk_forward_project")
    stream_path = project_root / "streams" / "regression.outcome.yaml"
    stream_path.write_text(
        stream_path.read_text(encoding="utf-8")
        + "\ntransforms:\n"
        + "  - { operation: forward_sum, field: value, window: 2, to: future_2 }\n",
        encoding="utf-8",
    )

    dataset_path = project_root / "dataset.yaml"
    dataset_path.write_text(
        dataset_path.read_text(encoding="utf-8").replace(
            "  - id: outcome\n"
            "    stream: regression.outcome\n"
            "    field: value\n"
            "    scale: true\n"
            "    horizon: 0s",
            "  - id: outcome\n"
            "    stream: regression.outcome\n"
            "    field: future_2\n"
            "    scale: false\n"
            "    horizon: 2d",
        ),
        encoding="utf-8",
    )

    output_dir, _ = _serve(project_root)

    fold_0_train = _read_output(output_dir, "fold_0.train")
    assert [row["key"][0] for row in fold_0_train] == ["2024-01-01 00:00:00+00:00"]
    assert fold_0_train[0]["targets"]["values"]["outcome"] == 1014.0
    assert _read_output(output_dir, "fold_0.validation") == []
    assert (
        _read_output(output_dir, "fold_0.test")[0]["targets"]["values"]["outcome"]
        == 208.0
    )

    fold_1_train = _read_output(output_dir, "fold_1.train")
    assert [row["key"][0] for row in fold_1_train] == [
        "2024-01-01 00:00:00+00:00",
        "2024-01-02 00:00:00+00:00",
        "2024-01-04 00:00:00+00:00",
        "2024-01-06 00:00:00+00:00",
    ]
    assert [row["targets"]["values"]["outcome"] for row in fold_1_train] == [
        1014.0,
        1018.0,
        120.0,
        10108.0,
    ]
    assert _read_output(output_dir, "fold_1.validation") == []


def test_walk_forward_serve_opens_series_once(
    copy_fixture,
    monkeypatch,
) -> None:
    project_root = copy_fixture("walk_forward_project")
    serve_dataset(project_root)

    original_open_series = sample_input.open_series
    calls = 0

    def count_open_series(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_open_series(*args, **kwargs)

    monkeypatch.setattr(sample_input, "open_series", count_open_series)

    serve_dataset(project_root, "AUTO")

    assert calls == 1


def test_shared_fold_scan_matches_separate_output_serves(
    copy_fixture,
    tmp_path: Path,
) -> None:
    shared_root = copy_fixture("walk_forward_project")
    separate_root = tmp_path / "walk_forward_separate_outputs"
    shutil.copytree(shared_root, separate_root)

    shared_output, _ = _serve(shared_root)
    profile_path = separate_root / "profiles" / "serve.dataset.yaml"
    profile = profile_path.read_text(encoding="utf-8")
    output_ids = (
        "fold_0.train",
        "fold_0.validation",
        "fold_0.test",
        "fold_1.train",
        "fold_1.validation",
        "fold_1.test",
    )

    for output_id in output_ids:
        profile_path.write_text(
            profile + f"include_outputs: [{output_id}]\n",
            encoding="utf-8",
        )
        request = serve_dataset(separate_root, "AUTO")
        separate_output = request.serve_run_plans[0].paths.dataset_dir
        filename = f"dataset.{output_id}.jsonl"

        assert {path.name for path in separate_output.iterdir()} == {filename}
        assert (separate_output / filename).read_bytes() == (
            shared_output / filename
        ).read_bytes()


def test_walk_forward_limit_applies_to_each_output(copy_fixture) -> None:
    project_root = copy_fixture("walk_forward_project")
    request = serve_dataset(project_root, limit=1)
    output_dir = request.serve_run_plans[0].paths.dataset_dir

    expected_days = {
        "fold_0.train": 1,
        "fold_0.validation": 4,
        "fold_0.test": 5,
        "fold_1.train": 1,
        "fold_1.validation": 9,
        "fold_1.test": 10,
    }
    for output_id, day in expected_days.items():
        rows = _read_output(output_dir, output_id)
        assert len(rows) == 1
        assert rows[0]["key"] == [f"2024-01-{day:02d} 00:00:00+00:00"]
