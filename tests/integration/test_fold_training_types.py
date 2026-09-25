import json
from pathlib import Path

import pytest
import yaml

from tests.helpers.regression import read_jsonl, serve_dataset


def _filled_fold_project(root: Path, fill_value: int | float) -> Path:
    for directory in ("sources", "streams", "profiles", "operations", "datasets"):
        (root / directory).mkdir()
    documents = {
        "project.yaml": {
            "schema_version": 7,
            "artifact_revision": 1,
            "paths": {
                "streams": "streams",
                "sources": "sources",
                "datasets": "datasets",
                "artifacts": "artifacts",
                "profiles": "profiles",
                "operations": "operations",
            },
        },
        "datasets/default.yaml": {
            "version": "v1",
            "sample": {"rounding": "ceil", "cadence": "1h"},
            "features": [
                {"id": "price", "stream": "prices", "field": "value"},
                {
                    "id": "price_window",
                    "stream": "prices",
                    "field": "value",
                    "sequence": {"size": 1},
                },
            ],
            "split": {
                "mode": "time",
                "intervals": [
                    {"id": "train", "until": "2024-01-01T03:00:00Z"},
                    {"id": "validation"},
                ],
                "folds": [
                    {
                        "id": "holdout",
                        "train": ["train"],
                        "validation": ["validation"],
                    }
                ],
            },
        },
        "sources/prices.yaml": {
            "id": "prices",
            "loader": {
                "transport": "fs",
                "path": "prices.jsonl",
                "reader": {"format": "jsonl"},
            },
            "parser": {"entrypoint": "core.temporal_record"},
        },
        "streams/prices.yaml": {
            "id": "prices",
            "from": {"source": "prices"},
            "map": {"entrypoint": "identity"},
            "transforms": [
                {"operation": "ensure_cadence", "cadence": "1h"},
                {"operation": "fill_missing", "field": "value", "value": fill_value},
            ],
        },
        "operations/dataset.yaml": {
            "kind": "output",
            "entrypoint": "core.dataset",
            "dataset": "default",
        },
        "profiles/serve.dataset.yaml": {
            "operation": "dataset",
            "output": {"transport": "fs", "format": "jsonl", "directory": "output"},
        },
    }
    for relative_path, document in documents.items():
        (root / relative_path).write_text(yaml.safe_dump(document), encoding="utf-8")
    (root / "prices.jsonl").write_text(
        "".join(
            json.dumps({"time": f"2024-01-01T0{hour}:00:00Z", "value": value}) + "\n"
            for hour, value in ((0, 1), (2, 2), (3, 3))
        ),
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize("fill_value", [0, 0.0], ids=["integer", "float"])
def test_filled_training_types_cover_validation(
    tmp_path: Path, fill_value: int | float
) -> None:
    project_root = _filled_fold_project(tmp_path, fill_value)

    request = serve_dataset(project_root)

    metadata = json.loads(
        (project_root / "artifacts/datasets/default/metadata.json").read_text(
            encoding="utf-8"
        )
    )
    training = metadata["layout"]["folds"][0]["training_schema"]["features"]
    expected_types = ["float", "int"] if isinstance(fill_value, float) else ["int"]
    assert training[0]["kind"] == "scalar"
    assert training[0]["value_types"] == expected_types
    assert training[1]["kind"] == "list"
    assert training[1]["element_types"] == expected_types
    assert training[1]["length"] == 1

    output = request.serve_run_plans[0].paths.dataset_dir
    train = read_jsonl(output / "dataset.holdout.train.jsonl")
    validation = read_jsonl(output / "dataset.holdout.validation.jsonl")
    assert train == [
        {
            "key": [f"2024-01-01 0{hour}:00:00+00:00"],
            "features": {"values": {"price": value, "price_window": [value]}},
            "targets": None,
        }
        for hour, value in enumerate((1, fill_value, 2))
    ]
    assert validation == [
        {
            "key": ["2024-01-01 03:00:00+00:00"],
            "features": {"values": {"price": 3, "price_window": [3]}},
            "targets": None,
        }
    ]


def test_fold_rejects_new_validation_types(tmp_path: Path) -> None:
    project_root = _filled_fold_project(tmp_path, 0)
    source = project_root / "prices.jsonl"
    records = read_jsonl(source)
    records[-1]["value"] = "three"
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match=r"fold 'holdout'.*feature series IDs, shapes, and value types: price, price_window",
    ):
        serve_dataset(project_root)
