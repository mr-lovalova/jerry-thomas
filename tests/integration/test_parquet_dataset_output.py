from datetime import datetime
from pathlib import Path

import pyarrow.parquet as parquet

from jerrythomas.config.profiles.output import ServeOutputConfig
from tests.helpers.regression import read_jsonl, serve_dataset


def _flat_row(
    sample: dict[str, object],
    sample_keys: tuple[str, ...],
) -> dict[str, object]:
    key = sample["key"]
    assert isinstance(key, list)
    row: dict[str, object] = {
        "sample.time": datetime.fromisoformat(str(key[0])),
    }
    for field, value in zip(sample_keys, key[1:], strict=True):
        row[f"sample.{field}"] = value
    for namespace in ("features", "targets"):
        vector = sample[namespace]
        if vector is None:
            continue
        assert isinstance(vector, dict)
        values = vector["values"]
        assert isinstance(values, dict)
        for series_id, value in values.items():
            if isinstance(value, list):
                for index, element in enumerate(value):
                    row[f"{namespace}.{series_id}.{index}"] = element
            else:
                row[f"{namespace}.{series_id}"] = value
    return row


def _assert_parquet_matches_jsonl(
    project_root: Path,
    sample_keys: tuple[str, ...] = (),
) -> None:
    json_request = serve_dataset(project_root)
    json_path = json_request.serve_run_plans[0].paths.dataset_dir / "dataset.jsonl"
    expected = [_flat_row(sample, sample_keys) for sample in read_jsonl(json_path)]

    parquet_request = serve_dataset(
        project_root,
        "auto",
        ServeOutputConfig(
            transport="fs",
            format="parquet",
            directory=project_root / "parquet-output",
        ),
    )
    parquet_path = (
        parquet_request.serve_run_plans[0].paths.dataset_dir / "dataset.parquet"
    )

    table = parquet.read_table(parquet_path)
    assert table.column_names == list(expected[0])
    assert table.to_pylist() == expected


def test_parquet_dataset_matches_jsonl_dataset(copy_fixture) -> None:
    _assert_parquet_matches_jsonl(copy_fixture("regression_project"))


def test_parquet_dataset_preserves_named_sample_keys(copy_fixture) -> None:
    _assert_parquet_matches_jsonl(
        copy_fixture("identity_alignment_project"),
        ("ticker",),
    )


def test_parquet_dataset_matches_scaled_walk_forward_folds(copy_fixture) -> None:
    project_root = copy_fixture("walk_forward_project")
    json_request = serve_dataset(project_root)
    json_directory = json_request.serve_run_plans[0].paths.dataset_dir

    parquet_request = serve_dataset(
        project_root,
        "auto",
        ServeOutputConfig(
            transport="fs",
            format="parquet",
            directory=project_root / "parquet-output",
        ),
    )
    parquet_directory = parquet_request.serve_run_plans[0].paths.dataset_dir

    output_ids = (
        "fold_0.train",
        "fold_0.validation",
        "fold_0.test",
        "fold_1.train",
        "fold_1.validation",
        "fold_1.test",
    )
    assert {path.name for path in parquet_directory.iterdir()} == {
        f"dataset.{output_id}.parquet" for output_id in output_ids
    }

    for output_id in output_ids:
        expected = [
            _flat_row(sample, ())
            for sample in read_jsonl(json_directory / f"dataset.{output_id}.jsonl")
        ]
        table = parquet.read_table(parquet_directory / f"dataset.{output_id}.parquet")

        assert table.column_names == list(expected[0])
        assert table.to_pylist() == expected
