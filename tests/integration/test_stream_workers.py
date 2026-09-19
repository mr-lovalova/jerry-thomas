import json
import tempfile
from pathlib import Path

import pytest
import yaml

import jerrythomas.pipelines.sort as sort_module
from jerrythomas.artifacts.series import (
    load_series_manifest,
    open_series,
    series_cache_root,
)
from jerrythomas.config.tasks.series import SeriesTask
from jerrythomas.execution.settings import CommandObservability
from jerrythomas.operations.artifacts.series_workers import StreamWorkerError
from jerrythomas.profiles.orchestration import run_profiles
from jerrythomas.profiles.request_builder import build_build_run_request
from tests.helpers.regression import serve_dataset


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _configure_workers(root: Path, command: str, workers: int) -> None:
    path = root / "profiles" / f"{command}.defaults.yaml"
    defaults = yaml.safe_load(path.read_text()) if path.exists() else {}
    defaults["execution"] = {"workers": workers, "sort_buffer_mb": 1}
    _write_yaml(path, defaults)


def _manifest_path(root: Path) -> Path:
    return root / "build" / SeriesTask().output


def _series_snapshot(root: Path) -> tuple[dict, bytes]:
    path = _manifest_path(root)
    manifest = load_series_manifest(path)
    return (
        manifest.model_dump(mode="json", exclude={"path"}),
        (path.parent / manifest.path).read_bytes(),
    )


def _build_series(root: Path, workers: int) -> None:
    _configure_workers(root, "build", workers)
    request = build_build_run_request(
        str(root / "project.yaml"),
        profile_name="series",
        force=True,
        command_observability=CommandObservability(visuals=False, log_level="CRITICAL"),
    )
    assert request is not None
    assert request.execution.workers == workers
    assert run_profiles(request) == ()


def _keyed_project(root: Path) -> Path:
    for directory in ("sources", "streams", "profiles", "data"):
        (root / directory).mkdir(parents=True)
    _write_yaml(
        root / "project.yaml",
        {
            "schema_version": 6,
            "artifact_revision": 1,
            "paths": {
                "sources": "sources",
                "streams": "streams",
                "profiles": "profiles",
                "dataset": "dataset.yaml",
                "artifacts": "build",
            },
        },
    )
    _write_yaml(
        root / "dataset.yaml",
        {
            "sample": {"rounding": "floor", "cadence": "1h", "keys": ["id_"]},
            "features": [
                {"id": "collected", "stream": "a", "field": "value", "collect": 2},
                {"id": "scalar", "stream": "b", "field": "value"},
                {"id": "empty", "stream": "empty", "field": "value"},
            ],
        },
    )
    _write_yaml(root / "profiles" / "build.series.yaml", {"operation": "series"})
    for stream in ("a", "b", "empty"):
        _write_yaml(
            root / "sources" / f"{stream}.yaml",
            {
                "id": stream,
                "parser": {"entrypoint": "core.temporal_record"},
                "loader": {
                    "transport": "fs",
                    "path": f"data/{stream}.jsonl",
                    "reader": {"format": "jsonl"},
                },
            },
        )
        _write_yaml(
            root / "streams" / f"{stream}.yaml",
            {
                "id": stream,
                "from": {"source": stream},
                "map": {"entrypoint": "identity"},
                "partition_by": ["id_"],
            },
        )
    late = "2024-01-01T01:00:00Z"
    early = "2024-01-01T00:00:00Z"
    _write_rows(
        root / "data" / "a.jsonl",
        [
            {"time": late, "id_": "A", "value": None},
            {"time": early, "id_": "B", "value": False},
            {"time": late, "id_": "A", "value": 2},
            {"time": early, "id_": "B", "value": 1.5},
        ],
    )
    _write_rows(
        root / "data" / "b.jsonl",
        [
            {"time": late, "id_": "A", "value": "present"},
            {"time": early, "id_": "B", "value": None},
        ],
    )
    _write_rows(root / "data" / "empty.jsonl", [])
    return root


@pytest.mark.parametrize("workers", [2, 3, 8])
def test_stream_workers_preserve_regression_dataset_with_source_spills(
    copy_fixture, tmp_path, monkeypatch, workers
) -> None:
    root = copy_fixture("regression_project")
    distribution = tmp_path / "stream_worker_test-1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Name: stream-worker-test\nVersion: 1.0\n", encoding="utf-8"
    )
    (distribution / "entry_points.txt").write_text(
        "[jerrythomas.combiners]\n"
        "combine_humidity_with_baseline = tests.combiners:combine_humidity_with_baseline\n",
        encoding="utf-8",
    )
    # Spawned interpreters discover the same importable plugin through metadata,
    # independently of the pytest-only registration in the parent process.
    monkeypatch.syspath_prepend(str(tmp_path))
    dataset_path = root / "dataset.yaml"
    dataset = yaml.safe_load(dataset_path.read_text())
    dataset["features"].append(
        {
            "id": "sine_collected",
            "stream": "metrics.sine",
            "field": "value",
            "collect": 2,
        }
    )
    _write_yaml(dataset_path, dataset)
    sine_path = root / "streams" / "metrics.sine.yaml"
    sine = yaml.safe_load(sine_path.read_text())
    # Include the final half-hour so every collected bucket has two observations.
    sine["preprocess"][1]["comparand"] = "2024-03-01T05:30:00Z"
    _write_yaml(sine_path, sine)

    linear_path = root / "data" / "linear_hourly.jsonl"
    records = [json.loads(line) for line in linear_path.read_text().splitlines()]
    for record in records:
        record["sort_test_padding"] = "x" * 300_000
    _write_rows(linear_path, records)

    spills = []
    original_write_run = sort_module._write_serialized_run

    def record_spill(*args, **kwargs):
        path = original_write_run(*args, **kwargs)
        spills.append(path)
        return path

    _configure_workers(root, "serve", 1)
    with monkeypatch.context() as patch:
        patch.setattr(sort_module, "_write_serialized_run", record_spill)
        sequential = serve_dataset(root)
    assert sequential.execution.workers == 1
    assert len(spills) >= 2
    expected_data = (
        sequential.serve_run_plans[0].paths.dataset_dir / "dataset.jsonl"
    ).read_bytes()
    expected_series = _series_snapshot(root)

    _configure_workers(root, "serve", workers)
    parallel = serve_dataset(root)
    assert parallel.execution.workers == workers
    assert (
        parallel.serve_run_plans[0].paths.dataset_dir / "dataset.jsonl"
    ).read_bytes() == expected_data
    assert _series_snapshot(root) == expected_series
    rows = [json.loads(line) for line in expected_data.splitlines()]
    assert any(row["features"]["values"]["sine_window"] == [0.0, None] for row in rows)
    assert all(
        row["features"]["values"]["sine_collected"]
        == row["features"]["values"]["sine_window"]
        for row in rows
    )
    assert rows[0]["features"]["values"]["humidity_adjusted__@location:north"] == 50.0
    assert rows[0]["features"]["values"]["humidity_slope__@location:north"] is None


def test_stream_workers_preserve_key_order_ties_and_empty_stream(tmp_path) -> None:
    root = _keyed_project(tmp_path / "project")
    _build_series(root, 1)
    expected = _series_snapshot(root)
    _build_series(root, 2)
    assert _series_snapshot(root) == expected

    path = _manifest_path(root)
    manifest = load_series_manifest(path)
    assert manifest.sample_key_types == ("string",)
    assert {entry.id: entry.samples for entry in manifest.features} == {
        "collected": 2,
        "scalar": 2,
        "empty": 0,
    }
    rows = list(open_series(path, manifest))
    assert [row.entity_key for row in rows] == [("B",), ("A",)]
    assert [row.features["collected"] for row in rows] == [[False, 1.5], [None, 2]]
    assert type(rows[0].features["collected"][0]) is bool
    assert type(rows[1].features["collected"][1]) is int
    assert [row.features["scalar"] for row in rows] == [None, "present"]


def test_worker_failure_preserves_published_series_and_cleans_temporary_files(
    tmp_path, monkeypatch
) -> None:
    root = _keyed_project(tmp_path / "project")
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary))
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    _build_series(root, 1)
    manifest_path = _manifest_path(root)
    manifest_bytes = manifest_path.read_bytes()
    expected_series = _series_snapshot(root)
    state_path = root / "build" / "_system" / "build" / "state.json"
    state_bytes = state_path.read_bytes()
    cache_root = series_cache_root(manifest_path)
    generations = set(cache_root.iterdir())

    source_path = root / "data" / "b.jsonl"
    records = [json.loads(line) for line in source_path.read_text().splitlines()]
    records[-1]["value"] = float("inf")
    _write_rows(source_path, records)
    stream_path = root / "streams" / "b.yaml"
    stream = yaml.safe_load(stream_path.read_text())
    stream["presorted"] = True
    _write_yaml(stream_path, stream)
    with pytest.raises(
        StreamWorkerError,
        match="Stream 'b' failed: ValueError: JSON contains non-standard constant Infinity",
    ):
        _build_series(root, 2)

    assert manifest_path.read_bytes() == manifest_bytes
    assert _series_snapshot(root) == expected_series
    assert state_path.read_bytes() == state_bytes
    assert set(cache_root.iterdir()) == generations
    assert not list(temporary.iterdir())


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("sequence_warmup", [False, True])
def test_stream_workers_reject_inconsistent_sample_key_types(
    tmp_path, workers, sequence_warmup
) -> None:
    root = _keyed_project(tmp_path / "project")
    if sequence_warmup:
        path = root / "dataset.yaml"
        dataset = yaml.safe_load(path.read_text())
        feature = dataset["features"][0]
        feature.pop("collect")
        feature["sequence"] = {"size": 3}
        _write_yaml(path, dataset)
    source_path = root / "data" / "b.jsonl"
    records = [json.loads(line) for line in source_path.read_text().splitlines()]
    for record in records:
        record["id_"] = 123
    _write_rows(source_path, records)
    with pytest.raises(TypeError, match="Sample key field 'id_'.*string.*integer"):
        _build_series(root, workers)
    assert not _manifest_path(root).exists()
