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
from jerrythomas.config.dataset.dataset import ScalingConfig
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
    )
    runtime.dataset = runtime.require_dataset().model_copy(
        update={"scaling": ScalingConfig(with_mean=False, epsilon=0.25)}
    )
    task = ScalerTask(output="scaler.json")
    expected = _build(runtime, 1, task)
    for workers in (2, 8):
        assert _build(runtime, workers, task) == expected

    artifact = load_scaler_artifact(runtime.artifacts_root / task.output)
    assert isinstance(artifact, StandardScalerArtifact)
    assert artifact.with_mean is False
    assert artifact.with_std is True
    assert artifact.epsilon == 0.25
    assert artifact.observations == 7
    assert tuple(artifact.statistics) == ("a", "b")
    assert artifact.statistics["a"] == PositionalScalerStatistics(
        positions=(
            ScalerStatistics(mean=2.0, std=1.0, count=2),
            ScalerStatistics(mean=12.0, std=3.0, count=2),
        ),
    )
    assert artifact.statistics["b"].count == 3
    assert artifact.statistics["b"].mean == pytest.approx(50 / 3)
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
        targets=[
            {
                "id": "target",
                "stream": "a",
                "field": "future",
                "horizon": "2d",
                "scale": True,
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
    assert tuple(early.statistics) == ("a", "target")
    assert early.statistics["a"] == ScalerStatistics(mean=2.0, std=1.0, count=2)
    assert early.statistics["target"] == ScalerStatistics(mean=15.0, std=5.0, count=2)
    expanded = artifact.for_fold("expanded")
    assert tuple(expanded.statistics) == ("a", "b", "target")
    assert expanded.statistics["a"].mean == 28.25
    assert expanded.statistics["a"].count == 4
    assert expanded.statistics["b"] == ScalerStatistics(mean=20.0, std=1e-12, count=1)
    assert expanded.statistics["target"].mean == 40.0
    assert expanded.statistics["target"].count == 4
    assert expected[1] == {"folds": 2, "observations": 13}


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
