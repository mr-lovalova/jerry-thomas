import gzip
import io
import json
import pickle
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
from jerrythomas.execution.settings import CommandObservability
from jerrythomas.services.stream_workers import StreamWorkerError
from jerrythomas.profiles.orchestration import run_profiles
from jerrythomas.profiles.request_builder import build_build_run_request
from tests.helpers.regression import serve_dataset


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.name == "datasets":
        value = {"version": "v1", **value}
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _configure_workers(root: Path, command: str, workers: int) -> None:
    path = root / "profiles" / f"{command}.defaults.yaml"
    defaults = yaml.safe_load(path.read_text()) if path.exists() else {}
    defaults["execution"] = {"workers": workers, "sort_buffer_mb": 1}
    _write_yaml(path, defaults)


def _manifest_path(root: Path) -> Path:
    return root / "build" / "datasets/default/series/manifest.json"


def _series_snapshot(root: Path) -> tuple[dict, bytes]:
    path = _manifest_path(root)
    manifest = load_series_manifest(path)
    return (
        manifest.model_dump(mode="json", exclude={"path"}),
        (path.parent / manifest.path).read_bytes(),
    )


def _publication_snapshot(
    root: Path,
) -> tuple[bytes, tuple[dict, bytes], bytes, set[Path]]:
    manifest_path = _manifest_path(root)
    return (
        manifest_path.read_bytes(),
        _series_snapshot(root),
        (root / "build" / "_system" / "build" / "state.json").read_bytes(),
        set(series_cache_root(manifest_path).iterdir()),
    )


def _build_series(root: Path, workers: int) -> None:
    _configure_workers(root, "build", workers)
    request = build_build_run_request(
        str(root / "project.yaml"),
        profile_name="series",
        artifact_mode="rebuild",
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
            "schema_version": 7,
            "artifact_revision": 1,
            "paths": {
                "sources": "sources",
                "streams": "streams",
                "profiles": "profiles",
                "datasets": "datasets",
                "artifacts": "build",
            },
        },
    )
    _write_yaml(
        root / "datasets" / "default.yaml",
        {
            "sample": {"rounding": "floor", "cadence": "1h", "keys": ["id_"]},
            "features": [
                {"id": "collected", "stream": "a", "field": "value", "collect": 2},
                {"id": "scalar", "stream": "b", "field": "value"},
                {"id": "empty", "stream": "empty", "field": "value"},
            ],
        },
    )
    _write_yaml(
        root / "profiles" / "build.series.yaml", {"operation": "dataset.default.series"}
    )
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
    dataset_path = root / "datasets" / "default.yaml"
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
        with gzip.open(path, "rb") as source:
            first = pickle.load(source)
        if getattr(first, "sort_test_padding", None) is not None:
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
    assert [json.loads(line) for line in gzip.decompress(expected[1]).splitlines()] == [
        {
            "time": "2024-01-01T00:00:00Z",
            "entity_key": ["B"],
            "features": {"collected": [False, 1.5], "scalar": None},
            "targets": {},
            "placeholder_ids": [],
        },
        {
            "time": "2024-01-01T01:00:00Z",
            "entity_key": ["A"],
            "features": {"collected": [None, 2], "scalar": "present"},
            "targets": {},
            "placeholder_ids": [],
        },
    ]
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


def test_stream_workers_merge_spilled_projected_values(tmp_path, monkeypatch) -> None:
    root = _keyed_project(tmp_path / "project")
    path = root / "data" / "b.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["value"] = "selected value padding " * 60_000
    _write_rows(path, records)
    _build_series(root, 1)
    expected = _series_snapshot(root)
    merge_run_counts = []
    merge_runs = sort_module._merge_runs

    def record_merge_runs(paths, key):
        merge_run_counts.append(len(paths))
        return merge_runs(paths, key)

    monkeypatch.setattr(sort_module, "_merge_runs", record_merge_runs)
    _build_series(root, 2)

    # One run from a, at least two from b, none from the empty stream.
    assert max(merge_run_counts) >= 3
    assert _series_snapshot(root) == expected


def test_worker_failure_preserves_published_series_and_cleans_temporary_files(
    tmp_path, monkeypatch
) -> None:
    root = _keyed_project(tmp_path / "project")
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary))
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    _build_series(root, 1)
    expected = _publication_snapshot(root)

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

    assert _publication_snapshot(root) == expected
    assert not list(temporary.iterdir())


def test_inline_run_write_failure_preserves_published_series_and_cleans_temporary_files(
    tmp_path, monkeypatch
) -> None:
    root = _keyed_project(tmp_path / "project")
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary))
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    _build_series(root, 1)
    expected = _publication_snapshot(root)
    partial_runs = []

    def fail_write_run(directory, run_id, items):
        path = directory / f"run-{run_id}.pickle.gz"
        path.write_bytes(b"partial run")
        partial_runs.append(path)
        raise OSError("intentional sorted-run write failure")

    monkeypatch.setattr(sort_module, "_write_serialized_run", fail_write_run)
    with pytest.raises(OSError, match="intentional sorted-run write failure"):
        _build_series(root, 1)

    assert partial_runs
    assert all(not path.exists() for path in partial_runs)
    assert _publication_snapshot(root) == expected
    assert not list(temporary.iterdir())


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize(
    ("failure", "error", "message"),
    [
        ("missing-row", ValueError, "expected"),
        ("truncated", pickle.UnpicklingError, "truncated"),
        ("read", OSError, "intentional sorted-run read failure"),
    ],
)
def test_run_merge_failure_preserves_published_series_and_cleans_temporary_files(
    tmp_path, monkeypatch, workers, failure, error, message
) -> None:
    root = _keyed_project(tmp_path / "project")
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary))
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    _build_series(root, 1)
    expected = _publication_snapshot(root)
    read_run = sort_module._read_run
    injected = False

    def fail_read_run(path):
        nonlocal injected
        if not injected:
            injected = True
            if failure == "read":
                raise OSError("intentional sorted-run read failure")
            with gzip.open(path, "rb") as source:
                payload = source.read()
            if failure == "missing-row":
                serialized = io.BytesIO(payload)
                while serialized.tell() < len(payload):
                    last_row_start = serialized.tell()
                    pickle.load(serialized)
                # Losing a whole record still leaves a readable file. The
                # expected row count must prevent publishing incomplete output.
                payload = payload[:last_row_start]
            else:
                payload = payload[:-1]
            with gzip.open(path, "wb") as destination:
                destination.write(payload)
        return read_run(path)

    monkeypatch.setattr(sort_module, "_read_run", fail_read_run)
    with pytest.raises(error, match=message):
        _build_series(root, workers)

    assert injected
    assert _publication_snapshot(root) == expected
    assert not list(temporary.iterdir())


@pytest.mark.parametrize("workers", [1, 2])
def test_stream_warmup_preserves_sample_key_types_without_emitting_rows(
    tmp_path, workers
) -> None:
    root = _keyed_project(tmp_path / "project")
    path = root / "datasets" / "default.yaml"
    dataset = yaml.safe_load(path.read_text())
    for feature in dataset["features"]:
        feature.pop("collect", None)
        feature["sequence"] = {"size": 3}
    _write_yaml(path, dataset)

    _build_series(root, workers)

    manifest = load_series_manifest(_manifest_path(root))
    assert manifest.sample_key_types == ("string",)
    assert manifest.rows == 0
    assert all(entry.samples == 0 for entry in manifest.features)
    assert gzip.decompress(_series_snapshot(root)[1]) == b""


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("sequence_warmup", [False, True])
def test_stream_workers_reject_inconsistent_sample_key_types(
    tmp_path, workers, sequence_warmup
) -> None:
    root = _keyed_project(tmp_path / "project")
    if sequence_warmup:
        path = root / "datasets" / "default.yaml"
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
