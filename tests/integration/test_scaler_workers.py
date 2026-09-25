import json
import multiprocessing
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from jerrythomas.artifacts.scaler import (
    FoldedScalerArtifact,
    PositionalScalerStatistics,
    ScalerStatistics,
    StandardScalerArtifact,
    load_scaler_artifact,
)
from jerrythomas.config.dataset.series import ScalingConfig
from jerrythomas.config.execution import ExecutionConfig
from jerrythomas.config.tasks.scaler import ScalerTask
from jerrythomas.execution.events import PipelineFinished, PipelineStarted
from jerrythomas.execution.observability import (
    ExecutionEvent,
    ScopedExecutionEvent,
    execution_observer,
)
from jerrythomas.operations.artifacts.scaler import build_scaler_artifact
from jerrythomas.runtime import Runtime
from jerrythomas.services.project_definition import load_project_definition
from jerrythomas.services.runtime_compiler import compile_runtime
from jerrythomas.services.stream_workers import StreamWorkerError
from tests.helpers.regression import read_jsonl, serve_dataset


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.name == "datasets":
        value = {"version": "v1", **value}
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(day: int, value: object, **fields: object) -> dict:
    return {
        "time": f"2024-01-{day:02d}T00:00:00Z",
        "id_": "A",
        "value": value,
        **fields,
    }


def _runtime(
    root: Path,
    sources: dict[str, list[dict]],
    *,
    transforms: dict[str, list[dict]] | None = None,
    **dataset_fields: object,
) -> Runtime:
    for directory in ("sources", "streams", "data"):
        (root / directory).mkdir(parents=True)
    _write_yaml(
        root / "project.yaml",
        {
            "schema_version": 7,
            "artifact_revision": 1,
            "paths": {
                "sources": "sources",
                "streams": "streams",
                "datasets": "datasets",
                "artifacts": "build",
            },
        },
    )
    _write_yaml(
        root / "datasets" / "default.yaml",
        {
            "sample": {"rounding": "floor", "cadence": "1d", "keys": ["id_"]},
            "features": [
                {"id": stream, "stream": stream, "field": "value", "scale": True}
                for stream in sources
            ],
            **dataset_fields,
        },
    )
    for stream, rows in sources.items():
        _write_rows(root / "data" / f"{stream}.jsonl", rows)
        _write_yaml(
            root / "sources" / f"{stream}.yaml",
            {
                "id": stream,
                "parser": {"entrypoint": "core.record"},
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
                "map": {"entrypoint": "core.identity"},
                "partition_by": ["id_"],
                "transforms": (transforms or {}).get(stream, []),
            },
        )
    return compile_runtime(load_project_definition(root / "project.yaml"), "default")


def _build(runtime: Runtime, workers: int, task: ScalerTask) -> tuple[bytes, dict]:
    runtime.execution = ExecutionConfig(workers=workers, sort_buffer_mb=1)
    result = build_scaler_artifact(runtime, task)
    return (runtime.artifacts_root / task.output).read_bytes(), result.meta


@pytest.fixture(autouse=True)
def _clean_worker_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator:
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary))
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    children = {process.pid for process in multiprocessing.active_children()}
    yield
    assert {process.pid for process in multiprocessing.active_children()} == children
    assert list(temporary.iterdir()) == []


def test_scaler_workers_preserve_positions_nulls_and_filled_domain_rows(
    tmp_path,
) -> None:
    # Unselected padding forces source sorting to spill without affecting fitting.
    padding = "x" * 600_000
    runtime = _runtime(
        tmp_path / "project",
        {
            "a": [
                _row(3, [3.0, 9.0], padding=padding),
                _row(1, [1.0, None], padding=padding),
                _row(2, None, padding=padding),
                _row(4, [None, 15.0], padding=padding),
            ],
            "b": [_row(3, 30.0), _row(1, 10.0)],
            "empty": [],
        },
        transforms={
            "b": [
                {"operation": "ensure_cadence", "cadence": "1d"},
                {"operation": "forward_fill", "field": "value"},
            ],
        },
        features=[
            {
                "id": "a",
                "stream": "a",
                "field": "value",
                "scale": {"with_mean": False, "epsilon": 0.25},
            },
            {
                "id": "b",
                "stream": "b",
                "field": "value",
                "scale": {"with_std": False},
            },
            {"id": "empty", "stream": "empty", "field": "value", "scale": True},
        ],
    )
    task = ScalerTask(output="scaler.json")
    expected = _build(runtime, 1, task)
    for workers in (2, 8):
        assert _build(runtime, workers, task) == expected

    artifact = load_scaler_artifact(runtime.artifacts_root / task.output)
    assert isinstance(artifact, StandardScalerArtifact)
    assert artifact.scalers["a"].settings == ScalingConfig(
        with_mean=False, epsilon=0.25
    )
    assert artifact.scalers["b"].settings == ScalingConfig(with_std=False)
    assert artifact.observations == 7
    assert tuple(artifact.scalers) == ("a", "b")
    assert artifact.scalers["a"].statistics == PositionalScalerStatistics(
        positions=(
            ScalerStatistics(mean=2.0, std=1.0, count=2),
            ScalerStatistics(mean=12.0, std=3.0, count=2),
        ),
    )
    assert artifact.scalers["b"].statistics.count == 3
    assert artifact.scalers["b"].statistics.mean == pytest.approx(50 / 3)
    assert expected[1] == {"series": 2, "observations": 7}


def test_scaler_workers_preserve_two_folds_horizon_and_empty_partial_fold(
    tmp_path,
) -> None:
    runtime = _runtime(
        tmp_path / "project",
        {
            "a": [
                _row(13, 10_000.0, future=130.0),
                _row(11, 1_000.0, future=110.0),
                _row(10, 9.0, future=100.0),
                _row(3, 100.0, future=30.0),
                _row(2, 3.0, future=20.0),
                _row(1, 1.0, future=10.0),
            ],
            # This stream has no participating rows in the early fold. Its empty
            # partial result must not invalidate statistics from the other stream.
            "b": [_row(10, 20.0)],
            "empty": [],
        },
        features=[
            {
                "id": "a",
                "stream": "a",
                "field": "value",
                "scale": {"with_std": False},
            },
            {
                "id": "b",
                "stream": "b",
                "field": "value",
                "scale": {"with_mean": False, "epsilon": 0.5},
            },
            {"id": "empty", "stream": "empty", "field": "value", "scale": True},
        ],
        targets=[
            {
                "id": "target",
                "stream": "a",
                "field": "future",
                "horizon": "2d",
                "scale": {"with_mean": False, "epsilon": 2.0},
            }
        ],
        split={
            "mode": "time",
            "intervals": [
                {"id": "train_0", "until": "2024-01-05T00:00:00Z"},
                {"id": "validation_0", "until": "2024-01-09T00:00:00Z"},
                {"id": "train_1", "until": "2024-01-13T00:00:00Z"},
                {"id": "validation_1"},
            ],
            "folds": [
                {"id": "early", "train": ["train_0"], "validation": ["validation_0"]},
                {
                    "id": "expanded",
                    "train": ["train_0", "validation_0", "train_1"],
                    "validation": ["validation_1"],
                },
            ],
        },
    )
    task = ScalerTask(output="scaler.json")
    expected = _build(runtime, 1, task)
    for workers in (2, 8):
        assert _build(runtime, workers, task) == expected

    artifact = load_scaler_artifact(runtime.artifacts_root / task.output)
    assert isinstance(artifact, FoldedScalerArtifact)
    early = artifact.for_fold("early")
    assert tuple(early.scalers) == ("a", "target")
    assert early.scalers["a"].settings == ScalingConfig(with_std=False)
    assert early.scalers["target"].settings == ScalingConfig(
        with_mean=False, epsilon=2.0
    )
    assert early.scalers["a"].statistics == ScalerStatistics(mean=2.0, std=1.0, count=2)
    assert early.scalers["target"].statistics == ScalerStatistics(
        mean=15.0, std=5.0, count=2
    )
    expanded = artifact.for_fold("expanded")
    assert tuple(expanded.scalers) == ("a", "b", "target")
    assert expanded.scalers["a"].statistics.mean == 28.25
    assert expanded.scalers["a"].statistics.count == 4
    assert expanded.scalers["b"].statistics == ScalerStatistics(
        mean=20.0, std=0.5, count=1
    )
    assert expanded.scalers["target"].statistics.mean == 40.0
    assert expanded.scalers["target"].statistics.count == 4
    assert expected[1] == {"folds": 2, "observations": 13}


def test_mixed_feature_scaling_preserves_fold_outputs_across_workers(tmp_path) -> None:
    root = tmp_path / "project"
    rows = [
        _row(day, value, id_=partition, vector=[value, 2 * value], future=100 * value)
        for partition in ("A", "B")
        for day, value in ((4, 101.0), (2, 3.0), (1, 1.0), (3, 5.0))
    ]
    _runtime(
        root,
        {"a": rows, "b": rows},
        features=[
            {
                "id": "center",
                "stream": "a",
                "field": "value",
                "scale": {"with_std": False},
            },
            {
                "id": "divide",
                "stream": "a",
                "field": "value",
                "scale": {"with_mean": False, "epsilon": 4.0},
            },
            {"id": "raw", "stream": "a", "field": "value", "scale": False},
            {"id": "positions", "stream": "a", "field": "vector", "scale": True},
            {
                "id": "window",
                "stream": "a",
                "field": "value",
                "sequence": {"size": 2},
                "scale": {"with_std": False, "epsilon": 0.5},
            },
            {"id": "default", "stream": "b", "field": "value", "scale": True},
        ],
        targets=[
            {
                "id": "target",
                "stream": "a",
                "field": "future",
                "horizon": "0d",
                "scale": {"with_std": False, "epsilon": 0.01},
            }
        ],
        split={
            "mode": "time",
            "intervals": [
                {"id": "train", "until": "2024-01-04T00:00:00Z"},
                {"id": "validation"},
            ],
            "folds": [
                {"id": "holdout", "train": ["train"], "validation": ["validation"]}
            ],
        },
    )
    project = yaml.safe_load((root / "project.yaml").read_text(encoding="utf-8"))
    project["paths"].update({"profiles": "profiles", "operations": "operations"})
    _write_yaml(root / "project.yaml", project)
    _write_yaml(
        root / "operations" / "dataset.yaml",
        {"kind": "output", "entrypoint": "core.dataset", "dataset": "default"},
    )
    _write_yaml(
        root / "profiles" / "serve.dataset.yaml",
        {
            "operation": "dataset",
            "output": {"transport": "fs", "format": "jsonl", "directory": "output"},
        },
    )

    baseline = None
    for workers in (1, 2, 8):
        _write_yaml(
            root / "profiles" / "serve.defaults.yaml",
            {"execution": {"workers": workers, "sort_buffer_mb": 1}},
        )
        request = serve_dataset(root)
        output = request.serve_run_plans[0].paths.dataset_dir
        scaler_path = root / "build" / "datasets" / "default" / "scaler.json"
        outputs = {
            role: read_jsonl(output / f"dataset.holdout.{role}.jsonl")
            for role in ("train", "validation")
        }
        snapshot = scaler_path.read_bytes(), outputs
        if baseline is None:
            baseline = snapshot
        else:
            assert snapshot == baseline

    artifact = load_scaler_artifact(scaler_path)
    assert isinstance(artifact, FoldedScalerArtifact)
    scalers = artifact.for_fold("holdout").scalers
    assert "raw" not in scalers
    assert scalers["center"].statistics.mean == 3.0
    assert scalers["center"].statistics.count == 6
    assert scalers["divide"].statistics.std == 4.0
    assert scalers["target"].statistics.mean == 300.0
    # Sequence positions share statistics fitted on the original scalar series.
    assert scalers["window"].statistics == ScalerStatistics(
        mean=3.0, std=(8 / 3) ** 0.5, count=6
    )
    assert len(outputs["validation"]) == 2
    assert {row["key"][1] for row in outputs["validation"]} == {"A", "B"}
    for row in outputs["validation"]:
        values = row["features"]["values"]
        assert values["center"] == 98.0
        assert values["divide"] == 25.25
        assert values["raw"] == 101.0
        assert values["positions"] == pytest.approx([98 / (8 / 3) ** 0.5] * 2)
        assert values["window"] == [2.0, 98.0]
        assert values["default"] == pytest.approx(98 / (8 / 3) ** 0.5)
        assert row["targets"]["values"]["target"] == 9800.0


@pytest.mark.parametrize("workers", [1, 2])
def test_scaler_workers_validate_sample_key_types_across_streams(
    tmp_path, workers
) -> None:
    runtime = _runtime(
        tmp_path / "project",
        {"a": [_row(1, 1.0)], "b": [_row(1, 2.0, id_=123)]},
    )
    task = ScalerTask(output="scaler.json")
    events: list[ExecutionEvent] = []
    with execution_observer(events.append):
        with pytest.raises(TypeError, match="Sample key field 'id_'.*string.*integer"):
            _build(runtime, workers, task)
    assert not (runtime.artifacts_root / task.output).exists()
    assert [
        event.status
        for event in events
        if isinstance(event, PipelineFinished)
        and event.pipeline_name == "scaler:artifact"
    ] == ["error"]


@pytest.mark.parametrize("workers", [1, 2])
def test_scaler_workers_require_global_observations(tmp_path, workers) -> None:
    runtime = _runtime(tmp_path / "project", {"a": [], "b": []})
    task = ScalerTask(output="scaler.json")
    with pytest.raises(RuntimeError, match="produced no observations for dataset"):
        _build(runtime, workers, task)
    assert not (runtime.artifacts_root / task.output).exists()


@pytest.mark.parametrize("workers", [1, 2])
def test_scaler_workers_reject_series_seen_only_outside_training(
    tmp_path, workers
) -> None:
    runtime = _runtime(
        tmp_path / "project",
        {"a": [_row(1, 1.0)], "b": [_row(3, 2.0)]},
        split={
            "mode": "time",
            "intervals": [
                {"id": "train", "until": "2024-01-02T00:00:00Z"},
                {"id": "validation"},
            ],
            "folds": [{"id": "fold", "train": ["train"], "validation": ["validation"]}],
        },
    )
    task = ScalerTask(output="scaler.json")
    with pytest.raises(
        (RuntimeError, StreamWorkerError), match="no training observations.*b"
    ):
        _build(runtime, workers, task)
    assert not (runtime.artifacts_root / task.output).exists()


def test_scaler_workers_reject_unobserved_list_position_despite_other_stream(
    tmp_path,
) -> None:
    runtime = _runtime(
        tmp_path / "project", {"a": [_row(1, [None, None])], "b": [_row(1, 2.0)]}
    )
    task = ScalerTask(output="scaler.json")
    with pytest.raises(RuntimeError, match="no numeric observations.*positions: 0, 1"):
        _build(runtime, 2, task)
    assert not (runtime.artifacts_root / task.output).exists()


def test_scaler_worker_failure_preserves_published_artifact_and_cleans_spills(
    tmp_path,
) -> None:
    rows = [_row(day, float(day), padding="x" * 600_000) for day in (4, 3, 2, 1)]
    runtime = _runtime(tmp_path / "project", {"a": rows, "b": rows})
    task = ScalerTask(output="scaler.json")
    expected, _ = _build(runtime, 1, task)
    rows[-1]["value"] = "invalid numeric value"
    _write_rows(tmp_path / "project" / "data" / "b.jsonl", rows)

    with pytest.raises(StreamWorkerError, match="Stream 'b' failed:.*numeric or None"):
        _build(runtime, 2, task)

    assert (runtime.artifacts_root / task.output).read_bytes() == expected


def test_scaler_workers_report_each_stream_in_its_own_scope(tmp_path) -> None:
    runtime = _runtime(tmp_path / "project", {"a": [_row(1, 1.0)], "b": [_row(1, 2.0)]})
    events: list[ExecutionEvent] = []
    with execution_observer(events.append):
        _build(runtime, 2, ScalerTask(output="scaler.json"))

    starts = {
        event.event.pipeline_name: event.scope
        for event in events
        if isinstance(event, ScopedExecutionEvent)
        and isinstance(event.event, PipelineStarted)
        and event.event.pipeline_name in {"stream:a", "stream:b"}
    }
    assert set(starts) == {"stream:a", "stream:b"}
    assert {scope.label for scope in starts.values()} == {"Worker 1", "Worker 2"}
    assert len({scope.id for scope in starts.values()}) == 2
    for name, scope in starts.items():
        assert any(
            isinstance(event, ScopedExecutionEvent)
            and event.scope == scope
            and isinstance(event.event, PipelineFinished)
            and event.event.pipeline_name == name
            and event.event.status == "success"
            for event in events
        )


@pytest.mark.parametrize("workers,streams", [(1, ("a", "b")), (8, ("a",))])
def test_scaler_runs_inline_for_one_worker_or_one_stream(
    tmp_path, monkeypatch, workers, streams
) -> None:
    runtime = _runtime(
        tmp_path / "project", {stream: [_row(1, 1.0)] for stream in streams}
    )

    def fail_start(process) -> None:
        pytest.fail("One-worker or one-stream scaler builds must run inline")

    monkeypatch.setattr(multiprocessing.process.BaseProcess, "start", fail_start)
    _, meta = _build(runtime, workers, ScalerTask(output="scaler.json"))
    assert meta == {"series": len(streams), "observations": len(streams)}
