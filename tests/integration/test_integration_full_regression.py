import json
from math import log1p
from pathlib import Path

from tests.helpers.regression import read_jsonl, serve_dataset


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample(
    hour: int,
    linear: float,
    sine: list[float | None],
    north: float,
    south: float,
    adjusted_north: float,
    adjusted_south: float,
    slope_north: float | None,
    slope_south: float | None,
    power: float,
    future_power: float | None,
    log_power: float,
) -> dict[str, object]:
    return {
        "key": [f"2024-03-01 {hour:02d}:00:00+00:00"],
        "features": {
            "values": {
                "linear_scaled": linear,
                "sine_window": sine,
                "humidity_partitioned__@location:north": north,
                "humidity_partitioned__@location:south": south,
                "humidity_adjusted__@location:north": adjusted_north,
                "humidity_adjusted__@location:south": adjusted_south,
                "humidity_slope__@location:north": slope_north,
                "humidity_slope__@location:south": slope_south,
            }
        },
        "targets": {
            "values": {
                "power_target": power,
                "power_future_2": future_power,
                "power_log1p": log_power,
            }
        },
    }


def test_full_regression_project_through_serve(copy_fixture) -> None:
    project_root = copy_fixture("regression_project")
    request = serve_dataset(project_root)

    dataset_path = request.serve_run_plans[0].paths.dataset_dir / "dataset.jsonl"
    samples = read_jsonl(dataset_path)
    assert samples == [
        _sample(
            0,
            -1.4638501094227998,
            [0.0, 0.5],
            40.0,
            38.5,
            50.0,
            48.5,
            None,
            None,
            100.0,
            203.0,
            log1p(100.0),
        ),
        _sample(
            1,
            -0.8783100656536799,
            [1.0, 0.5],
            40.0,
            37.0,
            52.0,
            49.0,
            0.0,
            -0.75,
            102.0,
            206.0,
            log1p(102.0),
        ),
        _sample(
            2,
            -0.29277002188455997,
            [0.0, None],
            41.0,
            37.75,
            55.0,
            51.75,
            0.5,
            0.375,
            101.0,
            210.0,
            log1p(101.0),
        ),
        _sample(
            3,
            0.29277002188455997,
            [-0.5, -1.0],
            42.0,
            37.75,
            58.0,
            53.75,
            0.5,
            0.0,
            105.0,
            212.0,
            log1p(105.0),
        ),
        _sample(
            4,
            0.8783100656536799,
            [-0.5, 0.0],
            41.5,
            39.0,
            59.5,
            57.0,
            -0.25,
            0.625,
            105.0,
            None,
            log1p(105.0),
        ),
    ]

    build_root = project_root / "build" / "datasets" / "default"
    assert _read_json(build_root / "scaler.json") == {
        "epsilon": 1e-12,
        "kind": "standard_scaler",
        "observations": 6,
        "statistics": {
            "linear_scaled": {
                "count": 6,
                "mean": 15.0,
                "std": 3.415650255319866,
            }
        },
        "version": 4,
        "with_mean": True,
        "with_std": True,
    }
    assert _read_json(build_root / "metadata.json") == {
        "schema_version": 4,
        "catalog": {
            "features": [
                {
                    "base_id": "linear_scaled",
                    "first_observed": "2024-03-01T00:00:00Z",
                    "id": "linear_scaled",
                    "kind": "scalar",
                    "last_observed": "2024-03-01T05:00:00Z",
                    "null_count": 0,
                    "present_count": 6,
                    "value_types": ["float"],
                },
                {
                    "base_id": "sine_window",
                    "element_types": ["float", "null"],
                    "first_observed": "2024-03-01T00:00:00Z",
                    "id": "sine_window",
                    "kind": "list",
                    "last_observed": "2024-03-01T04:00:00Z",
                    "length": 2,
                    "null_count": 0,
                    "observed_elements": 9,
                    "present_count": 5,
                },
                {
                    "base_id": "humidity_partitioned",
                    "first_observed": "2024-03-01T00:00:00Z",
                    "id": "humidity_partitioned__@location:north",
                    "kind": "scalar",
                    "last_observed": "2024-03-01T05:00:00Z",
                    "null_count": 0,
                    "present_count": 6,
                    "value_types": ["float"],
                },
                {
                    "base_id": "humidity_partitioned",
                    "first_observed": "2024-03-01T00:00:00Z",
                    "id": "humidity_partitioned__@location:south",
                    "kind": "scalar",
                    "last_observed": "2024-03-01T05:00:00Z",
                    "null_count": 0,
                    "present_count": 6,
                    "value_types": ["float"],
                },
                {
                    "base_id": "humidity_adjusted",
                    "first_observed": "2024-03-01T00:00:00Z",
                    "id": "humidity_adjusted__@location:north",
                    "kind": "scalar",
                    "last_observed": "2024-03-01T05:00:00Z",
                    "null_count": 0,
                    "present_count": 6,
                    "value_types": ["float"],
                },
                {
                    "base_id": "humidity_adjusted",
                    "first_observed": "2024-03-01T00:00:00Z",
                    "id": "humidity_adjusted__@location:south",
                    "kind": "scalar",
                    "last_observed": "2024-03-01T05:00:00Z",
                    "null_count": 0,
                    "present_count": 6,
                    "value_types": ["float"],
                },
                {
                    "base_id": "humidity_slope",
                    "first_observed": "2024-03-01T00:00:00Z",
                    "id": "humidity_slope__@location:north",
                    "kind": "scalar",
                    "last_observed": "2024-03-01T05:00:00Z",
                    "null_count": 1,
                    "present_count": 6,
                    "value_types": ["float"],
                },
                {
                    "base_id": "humidity_slope",
                    "first_observed": "2024-03-01T00:00:00Z",
                    "id": "humidity_slope__@location:south",
                    "kind": "scalar",
                    "last_observed": "2024-03-01T05:00:00Z",
                    "null_count": 1,
                    "present_count": 6,
                    "value_types": ["float"],
                },
            ],
            "counts": {"feature_vectors": 6, "target_vectors": 6},
            "targets": [
                {
                    "base_id": "power_target",
                    "first_observed": "2024-03-01T00:00:00Z",
                    "id": "power_target",
                    "kind": "scalar",
                    "last_observed": "2024-03-01T05:00:00Z",
                    "null_count": 0,
                    "present_count": 6,
                    "value_types": ["float"],
                },
                {
                    "base_id": "power_future_2",
                    "first_observed": "2024-03-01T00:00:00Z",
                    "id": "power_future_2",
                    "kind": "scalar",
                    "last_observed": "2024-03-01T05:00:00Z",
                    "null_count": 2,
                    "present_count": 6,
                    "value_types": ["float"],
                },
                {
                    "base_id": "power_log1p",
                    "first_observed": "2024-03-01T00:00:00Z",
                    "id": "power_log1p",
                    "kind": "scalar",
                    "last_observed": "2024-03-01T05:00:00Z",
                    "null_count": 0,
                    "present_count": 6,
                    "value_types": ["float"],
                },
            ],
            "window": {
                "end": "2024-03-01T04:00:00Z",
                "mode": "intersection",
                "size": 5,
                "start": "2024-03-01T00:00:00Z",
            },
        },
        "layout": {"kind": "unsplit"},
    }

    manifest = _read_json(build_root / "series" / "manifest.json")
    assert isinstance(manifest, dict)
    assert {
        "cadence": manifest["cadence"],
        "rounding": manifest["rounding"],
        "format": manifest["format"],
        "sample_keys": manifest["sample_keys"],
        "version": manifest["version"],
        "features": [
            {"id": entry["id"], "samples": entry["samples"]}
            for entry in manifest["features"]
        ],
        "targets": [
            {"id": entry["id"], "samples": entry["samples"]}
            for entry in manifest["targets"]
        ],
    } == {
        "cadence": "1h",
        "format": "jsonl.gz",
        "sample_keys": [],
        "version": 13,
        "rounding": "floor",
        "features": [
            {"id": "linear_scaled", "samples": 6},
            {"id": "sine_window", "samples": 5},
            {"id": "humidity_partitioned", "samples": 6},
            {"id": "humidity_adjusted", "samples": 6},
            {"id": "humidity_slope", "samples": 6},
        ],
        "targets": [
            {"id": "power_target", "samples": 6},
            {"id": "power_future_2", "samples": 6},
            {"id": "power_log1p", "samples": 6},
        ],
    }
