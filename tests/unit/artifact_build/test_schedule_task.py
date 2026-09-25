import json
from datetime import datetime, timezone

import pytest

import jerrythomas.operations.artifacts.schedule as schedule_module
from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.execution import ExecutionConfig
from jerrythomas.config.tasks.schedule import ScheduleTask
from jerrythomas.config.transforms import WhereConfig
from jerrythomas.domain.record import TemporalRecord
from jerrythomas.operations.artifacts.schedule import build_schedule_artifact
from jerrythomas.runtime import (
    AlignedRuntimeStream,
    DerivedRuntimeStream,
    Runtime,
    SourceRuntimeStream,
)


class _Source:
    def __init__(self, rows):
        self.rows = rows

    def stream(self):
        return iter(self.rows)


def _ts(hour: int, minute: int = 0) -> datetime:
    return datetime(2024, 1, 1, hour=hour, minute=minute, tzinfo=timezone.utc)


def _record(
    hour: int,
    security_id: str | None = None,
    *,
    minute: int = 0,
) -> TemporalRecord:
    record = TemporalRecord(time=_ts(hour, minute))
    if security_id is not None:
        record.security_id = security_id
    return record


def _identity(records):
    yield from records


def _left_records(pairs):
    for left, _ in pairs:
        yield left


def _runtime(tmp_path, rows=None, partition_by=()) -> Runtime:
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(
        "schema_version: 7\nartifact_revision: 1\n", encoding="utf-8"
    )
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    runtime = Runtime(
        project_yaml=project_yaml,
        artifacts_root=artifacts_root,
        dataset=DatasetConfig(sample=SampleConfig(rounding="ceil", cadence="1h")),
        execution=ExecutionConfig(),
    )
    runtime.streams["source.stream"] = SourceRuntimeStream(
        source=_Source([_record(2), _record(0), _record(2)] if rows is None else rows),
        mapper=_identity,
        preprocess=(),
        transforms=(),
        partition_by=partition_by,
        presorted=False,
    )
    return runtime


def _stream_runtime(tmp_path, rows=None) -> Runtime:
    runtime = _runtime(
        tmp_path,
        [_record(0), _record(1)] if rows is None else rows,
    )
    runtime.streams["derived.stream"] = DerivedRuntimeStream(
        input_stream="source.stream",
        transforms=(WhereConfig(field="time", operator="lt", comparand=_ts(2)),),
        partition_by=(),
    )
    return runtime


def test_schedule_task_rejects_time_as_partition_field() -> None:
    with pytest.raises(ValueError, match="reserved field 'time'"):
        ScheduleTask(
            id="schedule",
            stream="source.stream",
            partition_by=["time"],
            output="build/schedule.jsonl",
        )


def test_build_schedule_artifact_writes_sorted_unique_rows(tmp_path) -> None:
    runtime = _runtime(tmp_path)

    task = ScheduleTask(
        id="schedule",
        entrypoint="core.schedule",
        stream="source.stream",
        partition_by=[],
        output="build/schedule.jsonl",
    )
    result = build_schedule_artifact(runtime, task)

    path = runtime.artifacts_root / task.output
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"time": "2024-01-01T00:00:00Z"},
        {"time": "2024-01-01T02:00:00Z"},
    ]
    assert result.meta == {
        "rows": 2,
        "stream": "source.stream",
        "partition_by": [],
    }


def test_build_schedule_artifact_writes_partitioned_rows(tmp_path) -> None:
    runtime = _runtime(
        tmp_path,
        [
            _record(0, "MSFT"),
            _record(0, "AAPL"),
            _record(1, "AAPL"),
            _record(1, "AAPL"),
        ],
    )

    task = ScheduleTask(
        id="schedule",
        entrypoint="core.schedule",
        stream="source.stream",
        partition_by=["security_id"],
        output="build/schedule.jsonl",
    )
    result = build_schedule_artifact(runtime, task)

    path = runtime.artifacts_root / task.output
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"time": "2024-01-01T00:00:00Z", "security_id": "AAPL"},
        {"time": "2024-01-01T01:00:00Z", "security_id": "AAPL"},
        {"time": "2024-01-01T00:00:00Z", "security_id": "MSFT"},
    ]
    assert result.meta == {
        "rows": 3,
        "stream": "source.stream",
        "partition_by": ["security_id"],
    }


def test_build_schedule_artifact_reuses_matching_stream_order(
    monkeypatch, tmp_path
) -> None:
    runtime = _runtime(
        tmp_path,
        [
            _record(0, "MSFT"),
            _record(0, "AAPL"),
            _record(1, "AAPL"),
            _record(1, "AAPL"),
        ],
        partition_by=("security_id",),
    )
    monkeypatch.setattr(
        schedule_module,
        "batch_sort",
        lambda *_args, **_kwargs: pytest.fail("ordered schedule was sorted again"),
    )

    task = ScheduleTask(
        id="schedule",
        entrypoint="core.schedule",
        stream="source.stream",
        partition_by=["security_id"],
        output="build/schedule.jsonl",
    )
    build_schedule_artifact(runtime, task)

    path = runtime.artifacts_root / task.output
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"time": "2024-01-01T00:00:00Z", "security_id": "AAPL"},
        {"time": "2024-01-01T01:00:00Z", "security_id": "AAPL"},
        {"time": "2024-01-01T00:00:00Z", "security_id": "MSFT"},
    ]


def test_build_schedule_artifact_reuses_aligned_stream_order(
    monkeypatch, tmp_path
) -> None:
    runtime = _runtime(
        tmp_path,
        [_record(1, "MSFT"), _record(0, "AAPL")],
        partition_by=("security_id",),
    )
    runtime.streams["right.stream"] = SourceRuntimeStream(
        source=_Source([_record(0, "AAPL"), _record(1, "MSFT")]),
        mapper=_identity,
        preprocess=(),
        transforms=(),
        partition_by=("security_id",),
        presorted=False,
    )
    runtime.streams["aligned.stream"] = AlignedRuntimeStream(
        inputs=("source.stream", "right.stream"),
        combine=_left_records,
        transforms=(),
        partition_by=("security_id",),
    )
    monkeypatch.setattr(
        schedule_module,
        "batch_sort",
        lambda *_args, **_kwargs: pytest.fail("ordered schedule was sorted again"),
    )

    task = ScheduleTask(
        id="schedule",
        entrypoint="core.schedule",
        stream="aligned.stream",
        partition_by=["security_id"],
        output="build/schedule.jsonl",
    )
    build_schedule_artifact(runtime, task)

    path = runtime.artifacts_root / task.output
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"time": "2024-01-01T00:00:00Z", "security_id": "AAPL"},
        {"time": "2024-01-01T01:00:00Z", "security_id": "MSFT"},
    ]


def test_build_schedule_artifact_rejects_broken_matching_order_atomically(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path, partition_by=("security_id",))
    destination = runtime.artifacts_root / "build/schedule.jsonl"
    destination.parent.mkdir(parents=True)
    destination.write_text("previous\n", encoding="utf-8")
    monkeypatch.setattr(
        schedule_module,
        "run_stream_pipeline",
        lambda *_args, **_kwargs: iter([_record(0, "MSFT"), _record(0, "AAPL")]),
    )
    monkeypatch.setattr(
        schedule_module,
        "batch_sort",
        lambda *_args, **_kwargs: pytest.fail("matching schedule was sorted"),
    )

    with pytest.raises(ValueError, match="violates canonical order"):
        build_schedule_artifact(
            runtime,
            ScheduleTask(
                id="schedule",
                entrypoint="core.schedule",
                stream="source.stream",
                partition_by=["security_id"],
                output="build/schedule.jsonl",
            ),
        )

    assert destination.read_text(encoding="utf-8") == "previous\n"
    assert list(destination.parent.iterdir()) == [destination]


def test_build_schedule_artifact_rejects_missing_partition_field(tmp_path) -> None:
    runtime = _runtime(tmp_path, [_record(0)])
    destination = runtime.artifacts_root / "build/schedule.jsonl"
    destination.parent.mkdir(parents=True)
    destination.write_text("previous\n", encoding="utf-8")

    with pytest.raises(KeyError, match="security_id"):
        build_schedule_artifact(
            runtime,
            ScheduleTask(
                id="schedule",
                entrypoint="core.schedule",
                stream="source.stream",
                partition_by=["security_id"],
                output="build/schedule.jsonl",
            ),
        )

    assert destination.read_text(encoding="utf-8") == "previous\n"
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (float("nan"), "missing partition_by field"),
        (float("inf"), "must not contain infinity"),
        (float("-inf"), "must not contain infinity"),
    ],
)
def test_build_schedule_artifact_rejects_non_finite_partition_values_atomically(
    tmp_path,
    value: float,
    message: str,
) -> None:
    record = _record(0, "placeholder")
    record.security_id = value
    runtime = _runtime(tmp_path, [record])

    with pytest.raises(ValueError, match=message):
        build_schedule_artifact(
            runtime,
            ScheduleTask(
                id="schedule",
                entrypoint="core.schedule",
                stream="source.stream",
                partition_by=["security_id"],
                output="build/schedule.jsonl",
            ),
        )

    assert not (runtime.artifacts_root / "build/schedule.jsonl").exists()


@pytest.mark.parametrize(
    "partition_values",
    [
        (False, 0),
        (1, 1.0),
    ],
)
def test_build_schedule_artifact_rejects_mixed_partition_types_atomically(
    tmp_path,
    partition_values: tuple[object, object],
) -> None:
    records = []
    for value in partition_values:
        record = _record(0)
        record.security_id = value
        records.append(record)
    runtime = _runtime(
        tmp_path,
        records,
    )

    with pytest.raises(TypeError, match="partition fields must use one exact type"):
        build_schedule_artifact(
            runtime,
            ScheduleTask(
                id="schedule",
                entrypoint="core.schedule",
                stream="source.stream",
                partition_by=["security_id"],
                output="build/schedule.jsonl",
            ),
        )

    assert not (runtime.artifacts_root / "build/schedule.jsonl").exists()


def test_build_schedule_artifact_uses_stream_transforms(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _stream_runtime(
        tmp_path,
        [_record(2, minute=30), _record(0, minute=30)],
    )
    monkeypatch.setattr(
        schedule_module,
        "batch_sort",
        lambda *_args, **_kwargs: pytest.fail("ordered schedule was sorted again"),
    )

    task = ScheduleTask(
        id="derived_schedule",
        entrypoint="core.schedule",
        stream="derived.stream",
        partition_by=[],
        output="build/derived_schedule.jsonl",
    )
    build_schedule_artifact(runtime, task)

    path = runtime.artifacts_root / task.output
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"time": "2024-01-01T00:30:00Z"},
    ]
