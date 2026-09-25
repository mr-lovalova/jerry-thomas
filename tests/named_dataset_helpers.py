"""Small real project shared by named dataset integration tests."""

import json
from pathlib import Path

import yaml


def write_config(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def create_project(root: Path, *, datasets: bool = True) -> Path:
    paths = {
        "sources": "sources",
        "streams": "streams",
        "operations": "operations",
        "profiles": "profiles",
        "artifacts": "artifacts",
    }
    if datasets:
        paths["datasets"] = "datasets"
    write_config(
        root / "project.yaml",
        {"schema_version": 7, "artifact_revision": 1, "paths": paths},
    )
    write_config(
        root / "sources" / "rows.yaml",
        {
            "id": "rows",
            "parser": {"entrypoint": "core.temporal_record"},
            "loader": {
                "transport": "fs",
                "path": "rows.jsonl",
                "reader": {"format": "jsonl"},
            },
        },
    )
    (root / "rows.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "time": f"2024-01-0{day}T00:00:00Z",
                    "id_": "A",
                    "value": float(day),
                    "other": float(day * 10),
                }
            )
            + "\n"
            for day in (3, 1, 2)
        ),
        encoding="utf-8",
    )
    write_config(
        root / "streams" / "rows.yaml",
        {
            "id": "rows",
            "from": {"source": "rows"},
            "map": {"entrypoint": "identity"},
            "partition_by": ["id_"],
        },
    )
    write_config(
        root / "operations" / "raw.yaml",
        {
            "kind": "output",
            "entrypoint": "core.records",
            "stream": "rows",
        },
    )
    write_config(
        root / "profiles" / "materialize.raw.yaml",
        {
            "operation": "raw",
            "output": "exports/rows.jsonl.gz",
        },
    )
    if datasets:
        for dataset, version, field in (
            ("alpha", "v1", "value"),
            ("beta", "v2", "other"),
        ):
            body = {
                "version": version,
                "sample": {"cadence": "1d", "rounding": "exact", "keys": ["id_"]},
                "features": [
                    {"id": "x", "stream": "rows", "field": field, "scale": True}
                ],
            }
            if dataset == "alpha":
                body["split"] = {
                    "mode": "time",
                    "intervals": [
                        {"id": "train", "until": "2024-01-03T00:00:00Z"},
                        {"id": "test"},
                    ],
                    "folds": [{"id": "holdout", "train": ["train"], "test": ["test"]}],
                }
            write_config(root / "datasets" / f"{dataset}.yaml", body)
            write_config(
                root / "operations" / f"serve-{dataset}.yaml",
                {
                    "kind": "output",
                    "entrypoint": "core.dataset",
                    "dataset": dataset,
                },
            )
            write_config(
                root / "profiles" / f"serve.{dataset}.yaml",
                {"operation": f"serve-{dataset}"},
            )
            write_config(
                root / "profiles" / f"build.{dataset}.yaml",
                {"operation": f"dataset.{dataset}.scaler"},
            )
        write_config(
            root / "profiles" / "serve.defaults.yaml",
            {
                "output": {
                    "transport": "fs",
                    "format": "jsonl",
                    "directory": "outputs/${dataset_id}/${dataset_version}",
                },
            },
        )
    return root / "project.yaml"
