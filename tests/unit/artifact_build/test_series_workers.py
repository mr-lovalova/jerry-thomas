import logging
import multiprocessing
import os
import tempfile
import time
from collections.abc import Iterator
from datetime import timedelta
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from rich.console import Console
from rich.progress import Progress

import jerrythomas.operations.artifacts.series as series_module
import jerrythomas.operations.artifacts.series_workers as series_worker_module
import jerrythomas.services.stream_workers as worker_module
from jerrythomas.config.execution import ExecutionConfig
from jerrythomas.cli.visuals.rich.progress import (
    _ExecutionProgress,
    _RichExecutionRenderer,
)
from jerrythomas.domain.sample_key import SampleKeyContract
from jerrythomas.execution.events import (
    NodeProgress,
    NodeStarted,
    PipelineFinished,
    PipelineStarted,
)
from jerrythomas.execution.observability import (
    ExecutionEvent,
    ExecutionMessage,
    ScopedExecutionEvent,
    emit_execution_message,
    execution_observer,
)
from jerrythomas.operations.artifacts.series import _stream_plans
from jerrythomas.operations.artifacts.series_workers import (
    SeriesWorkerProgress,
    order_streams,
)
from jerrythomas.runtime import Runtime
from jerrythomas.services.project_definition import load_project_definition
from jerrythomas.services.runtime_compiler import compile_runtime
from jerrythomas.services.stream_workers import StreamWorkerError
from jerrythomas.services.temp_cleanup import sort_spill_directory
from jerrythomas.sources.loader import BaseDataLoader


class _LifecycleLoader(BaseDataLoader):
    """Importable plugin that keeps a worker-owned spill open while producing rows."""

    def __init__(self, mode: str, marker: str, barrier: str | None = None) -> None:
        self.mode = mode
        self.marker = Path(marker)
        self.barrier = None if barrier is None else Path(barrier)

    def load(self) -> Iterator[dict[str, Any]]:
        with sort_spill_directory() as spill:
            (spill / "run.pickle").write_bytes(b"unfinished worker spill")
            self.marker.write_text(str(spill), encoding="utf-8")
            deadline = time.monotonic() + 10
            while self.barrier is not None and not self.barrier.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Peer worker did not start")
                time.sleep(0.01)

            if self.mode == "crash":
                os._exit(17)
            if self.mode == "fail":
                raise ValueError("intentional later worker failure")
            if self.mode == "slow":
                time.sleep(10)
                raise TimeoutError("Slow peer exhausted test timeout")
            if self.mode == "finite":
                yield {"time": "2024-01-01T00:00:00Z", "id_": "A", "value": 1}
                return
            if self.mode == "observed":
                from jerrythomas.execution.pipeline import Input, Pipeline
                from jerrythomas.execution.runner import run_pipeline

                observed_runtime = SimpleNamespace(
                    observe_node_events=True, heartbeat_interval_seconds=0
                )
                pipeline = Pipeline(
                    name="shared:loader",
                    input=Input(
                        name="load",
                        open=lambda: iter(
                            [{"time": "2024-01-01T00:00:00Z", "id_": "A", "value": 1}]
                        ),
                    ),
                )
                yield from run_pipeline(observed_runtime, pipeline)
                return
            if self.mode == "messages":
                emit_execution_message("worker plugin message")
                logging.getLogger("tests.stream_workers").warning(
                    "worker standard warning"
                )
                yield {"time": "2024-01-01T00:00:00Z", "id_": "A", "value": 1}
                return
            projection_exhausted = False
            if self.mode == "sort_progress":
                import jerrythomas.pipelines.sort as sort_module

                write_run = sort_module._write_serialized_run

                def observed_final_write(directory, run_id, items):
                    if projection_exhausted:
                        sort_module._write_serialized_run = write_run
                        observed = self.marker.with_suffix(".sort-observed")
                        deadline = time.monotonic() + 10
                        while not observed.exists():
                            if time.monotonic() >= deadline:
                                raise TimeoutError("Parent did not observe final sort")
                            time.sleep(0.01)
                    return write_run(directory, run_id, items)

                sort_module._write_serialized_run = observed_final_write
            if self.mode == "write_failure":
                import jerrythomas.pipelines.sort as sort_module

                def fail_write_run(directory, run_id, items):
                    (directory / f"partial-{run_id}").write_bytes(b"partial worker run")
                    raise OSError("intentional sorted-run write failure")

                sort_module._write_serialized_run = fail_write_run

            # Selected values exceed the sort buffer and produce several runs.
            for index in range(12):
                yield {
                    "time": "2024-01-01T00:00:00Z",
                    "id_": "A",
                    "value": f"{index}:" + "x" * 600_000,
                }
            projection_exhausted = True


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


@pytest.fixture
def lifecycle_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    plugin_root = tmp_path / "plugins"
    distribution = plugin_root / "jerry_worker_test_fixture-0.1.dist-info"
    distribution.mkdir(parents=True)
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: jerry-worker-test-fixture\nVersion: 0.1\n",
        encoding="utf-8",
    )
    (distribution / "entry_points.txt").write_text(
        "[jerrythomas.loaders]\n"
        "test.stream.lifecycle = "
        "tests.unit.artifact_build.test_series_workers:_LifecycleLoader\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(plugin_root))
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))

    def build(
        modes: tuple[str, ...],
        barriers: tuple[int | None, ...] | None = None,
    ) -> tuple[Runtime, Path, tuple[Path, ...]]:
        root = tmp_path / "project"
        for name in ("sources", "streams"):
            (root / name).mkdir(parents=True)
        streams = ("first", "second", "third")[: len(modes)]
        markers = tuple(root / f"{stream}.started" for stream in streams)
        if barriers is None:
            barriers = (1, 0)
        _write_yaml(
            root / "project.yaml",
            {
                "schema_version": 6,
                "artifact_revision": 1,
                "paths": {
                    "sources": "sources",
                    "streams": "streams",
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
                    {"id": stream, "stream": stream, "field": "value"}
                    for stream in streams
                ],
            },
        )
        for index, (stream, mode, barrier) in enumerate(
            zip(streams, modes, barriers, strict=True)
        ):
            _write_yaml(
                root / "sources" / f"{stream}.yaml",
                {
                    "id": stream,
                    "freshness": "opaque",
                    "parser": {"entrypoint": "core.temporal_record"},
                    "loader": {
                        "entrypoint": "test.stream.lifecycle",
                        "args": {
                            "mode": mode,
                            "marker": str(markers[index]),
                            "barrier": (
                                None if barrier is None else str(markers[barrier])
                            ),
                        },
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
                    "presorted": True,
                },
            )
        runtime = compile_runtime(load_project_definition(root / "project.yaml"))
        runtime.execution = ExecutionConfig(workers=2, sort_buffer_mb=1)
        return runtime, temporary, markers

    return build


def _project(runtime: Runtime):
    return order_streams(
        runtime,
        _stream_plans(runtime.dataset.features, runtime.dataset.targets),
        SampleKeyContract(runtime.dataset.sample.keys),
        timedelta(hours=1),
        SeriesWorkerProgress(),
    )


def _child_pids() -> set[int | None]:
    return {process.pid for process in multiprocessing.active_children()}


@pytest.mark.parametrize("workers,stream_count", [(1, 2), (4, 1)])
def test_inline_sort_does_not_spawn_and_closing_output_removes_runs(
    lifecycle_project,
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
    stream_count: int,
) -> None:
    runtime, temporary, markers = lifecycle_project(
        ("finite",) * stream_count, barriers=(None,) * stream_count
    )
    runtime.execution = ExecutionConfig(workers=workers, sort_buffer_mb=1)

    def fail_start(process) -> None:
        pytest.fail("Inline sorting must not start a process")

    monkeypatch.setattr(multiprocessing.process.BaseProcess, "start", fail_start)
    projected = _project(runtime)
    try:
        assert next(projected).features[0].id == "first"
        assert list(temporary.rglob("*.pickle.gz"))
        for marker in markers:
            assert marker.exists()
            assert not Path(marker.read_text(encoding="utf-8")).exists()
    finally:
        projected.close()

    assert list(temporary.iterdir()) == []


def test_inline_write_failure_closes_input_and_preserves_original_error(
    lifecycle_project, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, temporary, markers = lifecycle_project(
        ("finite", "finite"), barriers=(None, None)
    )
    runtime.execution = ExecutionConfig(workers=1, sort_buffer_mb=1)
    project_stream = series_module._project_stream
    closed = []
    failure = OSError("intentional inline run write failure")

    def project(runtime, plan, sample_keys, cadence):
        try:
            yield from project_stream(runtime, plan, sample_keys, cadence)
        finally:
            closed.append(plan.stream_id)

    def fail_write(rows, buffer_bytes, key, directory, progress=None):
        next(rows)
        (directory / "partial-run").write_bytes(b"partial sorted run")
        raise failure

    monkeypatch.setattr(series_module, "_project_stream", project)
    monkeypatch.setattr(series_worker_module, "write_sort_runs", fail_write)
    projected = _project(runtime)
    try:
        with pytest.raises(OSError) as error:
            next(projected)
        assert error.value is failure
    finally:
        projected.close()

    assert closed == ["first"]
    assert markers[0].exists()
    assert not markers[1].exists()
    assert not Path(markers[0].read_text(encoding="utf-8")).exists()
    assert list(temporary.iterdir()) == []


def test_closing_merged_output_removes_completed_worker_runs(
    lifecycle_project,
) -> None:
    runtime, temporary, markers = lifecycle_project(("rows", "rows"))
    baseline = _child_pids()
    projected = _project(runtime)
    try:
        first = next(projected)
        assert first.features[0].id == "first"
        assert _child_pids() == baseline
        assert list(temporary.rglob("*.pickle.gz"))
        for marker in markers:
            spill = Path(marker.read_text(encoding="utf-8"))
            assert spill.is_relative_to(temporary)
            assert not spill.exists()
    finally:
        projected.close()

    assert _child_pids() == baseline
    assert list(temporary.iterdir()) == []


def test_completed_worker_frees_slot_before_earlier_stream_finishes(
    lifecycle_project,
) -> None:
    runtime, temporary, markers = lifecycle_project(
        ("finite", "finite", "finite"), barriers=(2, None, None)
    )
    baseline = _child_pids()
    events: list[ExecutionEvent] = []
    with execution_observer(events.append):
        rows = list(_project(runtime))

    assert all(marker.exists() for marker in markers)
    assert [row.features[0].id for row in rows] == ["first", "second", "third"]
    assert len({row.key for row in rows}) == 1
    scopes = {
        event.event.pipeline_name: event.scope
        for event in events
        if isinstance(event, ScopedExecutionEvent)
        and isinstance(event.event, PipelineStarted)
        and event.event.pipeline_name.startswith("prepare:")
    }
    assert scopes["prepare:first"].label == "Worker 1"
    assert scopes["prepare:second"].label == "Worker 2"
    assert scopes["prepare:third"].label == "Worker 2"
    assert len({scope.id for scope in scopes.values()}) == 3
    assert _child_pids() == baseline
    assert list(temporary.iterdir()) == []


def test_interrupt_while_workers_write_reaps_workers_and_removes_spills(
    lifecycle_project, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, temporary, markers = lifecycle_project(("slow", "slow"))
    baseline = _child_pids()
    check_workers = worker_module._check_workers

    def interrupt_after_workers_started(workers) -> None:
        check_workers(workers)
        if all(marker.exists() for marker in markers):
            raise KeyboardInterrupt("interrupt while workers write")

    monkeypatch.setattr(
        worker_module, "_check_workers", interrupt_after_workers_started
    )
    projected = _project(runtime)
    try:
        with pytest.raises(KeyboardInterrupt, match="interrupt while workers write"):
            next(projected)
    finally:
        projected.close()

    assert all(marker.exists() for marker in markers)
    assert _child_pids() == baseline
    assert list(temporary.iterdir()) == []


def test_abrupt_worker_exit_is_reported_and_reaps_waiting_peer(
    lifecycle_project,
) -> None:
    runtime, temporary, markers = lifecycle_project(("slow", "crash"))
    baseline = _child_pids()
    projected = _project(runtime)
    try:
        with pytest.raises(StreamWorkerError, match="second.*exit code 17"):
            next(projected)
    finally:
        projected.close()

    assert all(marker.exists() for marker in markers)
    assert _child_pids() == baseline
    assert list(temporary.iterdir()) == []


def test_later_worker_failure_interrupts_the_current_slow_stream(
    lifecycle_project,
) -> None:
    runtime, temporary, markers = lifecycle_project(("slow", "fail"))
    baseline = _child_pids()
    projected = _project(runtime)
    try:
        with pytest.raises(
            (ValueError, StreamWorkerError), match="intentional later worker failure"
        ):
            next(projected)
    finally:
        projected.close()

    assert all(marker.exists() for marker in markers)
    assert _child_pids() == baseline
    assert list(temporary.iterdir()) == []


def test_worker_run_write_failure_reaps_peer_and_removes_partial_output(
    lifecycle_project,
) -> None:
    runtime, temporary, markers = lifecycle_project(("slow", "write_failure"))
    baseline = _child_pids()
    projected = _project(runtime)
    try:
        with pytest.raises(
            StreamWorkerError, match="intentional sorted-run write failure"
        ):
            next(projected)
    finally:
        projected.close()

    assert all(marker.exists() for marker in markers)
    assert _child_pids() == baseline
    assert list(temporary.iterdir()) == []


def test_second_worker_start_failure_reaps_first_worker_and_removes_spills(
    lifecycle_project, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, temporary, _ = lifecycle_project(("rows", "rows"))
    baseline = _child_pids()
    get_context = multiprocessing.get_context
    context = get_context("spawn")
    created = 0

    def create_process(*args, **kwargs):
        nonlocal created
        process = context.Process(*args, **kwargs)
        created += 1
        if created == 2:

            def fail_start() -> None:
                assert len(_child_pids() - baseline) == 1
                raise OSError("intentional second worker start failure")

            monkeypatch.setattr(process, "start", fail_start)
        return process

    wrapped_context = SimpleNamespace(Pipe=context.Pipe, Process=create_process)
    monkeypatch.setattr(
        multiprocessing,
        "get_context",
        lambda method=None: (
            wrapped_context if method == "spawn" else get_context(method)
        ),
    )
    projected = _project(runtime)
    try:
        with pytest.raises(OSError, match="intentional second worker start failure"):
            next(projected)
    finally:
        projected.close()

    assert created == 2
    assert _child_pids() == baseline
    assert list(temporary.iterdir()) == []


def test_worker_messages_and_logs_reach_parent_observers(
    lifecycle_project, caplog: pytest.LogCaptureFixture
) -> None:
    runtime, temporary, _ = lifecycle_project(("messages", "finite"))
    baseline = _child_pids()
    events: list[ExecutionEvent] = []
    projected = _project(runtime)
    with execution_observer(events.append), caplog.at_level(logging.WARNING):
        try:
            next(projected)
        finally:
            projected.close()

    messages = [
        event
        for event in events
        if isinstance(event, ScopedExecutionEvent)
        and isinstance(event.event, ExecutionMessage)
    ]
    assert len(messages) == 1
    assert messages[0].scope.label == "Worker 1"
    assert messages[0].event == ExecutionMessage(message="worker plugin message")
    assert (
        "tests.stream_workers",
        logging.WARNING,
        "[Worker 1] worker standard warning",
    ) in caplog.record_tuples
    assert _child_pids() == baseline
    assert list(temporary.iterdir()) == []


def test_spawned_workers_render_same_pipeline_names_in_independent_scopes(
    lifecycle_project, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, temporary, _ = lifecycle_project(("observed", "observed"))
    runtime.heartbeat_interval_seconds = 0
    baseline = _child_pids()
    console = Console(file=StringIO(), force_terminal=False)
    progress = Progress(console=console, auto_refresh=False)
    renderer = _RichExecutionRenderer(
        logging.WARNING, console, _ExecutionProgress(progress, debug=False)
    )
    events: list[ExecutionEvent] = []
    close_worker = worker_module._close_worker
    finished_before_close: list[str] = []

    def observe(event: ExecutionEvent) -> None:
        events.append(event)
        renderer.render(event)

    def close_after_events(worker) -> None:
        if worker.complete is not None:
            assert any(
                isinstance(event, ScopedExecutionEvent)
                and event.scope == worker.scope
                and isinstance(event.event, PipelineFinished)
                and event.event.pipeline_name == f"prepare:{worker.stream_id}"
                and event.event.status == "success"
                for event in events
            )
            finished_before_close.append(worker.stream_id)
        close_worker(worker)

    monkeypatch.setattr(worker_module, "_close_worker", close_after_events)
    with execution_observer(observe):
        rows = list(_project(runtime))

    shared = [
        event
        for event in events
        if isinstance(event, ScopedExecutionEvent)
        and isinstance(event.event, PipelineStarted)
        and event.event.pipeline_name == "shared:loader"
    ]
    assert {event.scope.label for event in shared} == {"Worker 1", "Worker 2"}
    assert len({event.scope.id for event in shared}) == 2
    for started in shared:
        assert any(
            isinstance(event, ScopedExecutionEvent)
            and event.scope == started.scope
            and isinstance(event.event, NodeStarted)
            and event.event.pipeline_name == "shared:loader"
            for event in events
        )
    assert sorted(finished_before_close) == ["first", "second"]
    assert len(rows) == 2
    assert progress.tasks == []
    assert _child_pids() == baseline
    assert list(temporary.iterdir()) == []


def test_worker_observer_failure_reaps_workers_and_removes_spills(
    lifecycle_project,
) -> None:
    runtime, temporary, markers = lifecycle_project(("slow", "slow"))
    runtime.heartbeat_interval_seconds = 0.01
    baseline = _child_pids()
    failure = RuntimeError("intentional worker observer failure")

    def observe(event: ExecutionEvent) -> None:
        if (
            isinstance(event, ScopedExecutionEvent)
            and isinstance(event.event, NodeProgress)
            and all(marker.exists() for marker in markers)
        ):
            raise failure

    projected = _project(runtime)
    with execution_observer(observe):
        try:
            with pytest.raises(RuntimeError) as error:
                next(projected)
            assert error.value is failure
        finally:
            projected.close()

    assert all(marker.exists() for marker in markers)
    assert _child_pids() == baseline
    assert list(temporary.iterdir()) == []


def test_worker_final_sort_remains_visible_after_projection_finishes(
    lifecycle_project,
) -> None:
    runtime, temporary, markers = lifecycle_project(("sort_progress", "finite"))
    runtime.heartbeat_interval_seconds = 0.01
    baseline = _child_pids()
    console = Console(file=StringIO(), force_terminal=False)
    progress = Progress(console=console, auto_refresh=False)
    renderer = _RichExecutionRenderer(
        logging.WARNING, console, _ExecutionProgress(progress, debug=False)
    )
    projection_finished = False
    sort_snapshots = []

    def observe(event: ExecutionEvent) -> None:
        nonlocal projection_finished
        renderer.render(event)
        if not isinstance(event, ScopedExecutionEvent):
            return
        payload = event.event
        if isinstance(payload, PipelineFinished):
            if payload.pipeline_name == "series:first":
                projection_finished = True
        elif (
            projection_finished
            and isinstance(payload, NodeProgress)
            and payload.pipeline_name == "prepare:first"
            and payload.node_name == "order_series"
        ):
            matching = [
                task
                for task in progress.tasks
                if task.visible
                and task.description == "Worker 1 [prepare:first/order_series]"
            ]
            assert len(matching) == 1
            assert matching[0].completed == payload.progress.completed
            sort_snapshots.append(payload.progress)
            if payload.progress.phase == "spilling" and payload.progress.completed > 1:
                markers[0].with_suffix(".sort-observed").touch()

    with execution_observer(observe):
        rows = list(_project(runtime))

    assert len(rows) == 13
    assert projection_finished
    assert any(
        snapshot.phase == "spilling" and snapshot.completed > 1
        for snapshot in sort_snapshots
    )
    assert progress.tasks == []
    assert _child_pids() == baseline
    assert list(temporary.iterdir()) == []


def test_interrupt_after_finished_worker_close_still_reaps_active_peer(
    lifecycle_project, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, temporary, _ = lifecycle_project(("finite", "slow"))
    baseline = _child_pids()
    close_worker = worker_module._close_worker
    interrupted = False

    def interrupt_after_close(worker) -> None:
        nonlocal interrupted
        close_worker(worker)
        if worker.stream_id == "first" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("interrupt after finished worker close")

    monkeypatch.setattr(worker_module, "_close_worker", interrupt_after_close)
    projected = _project(runtime)
    try:
        with pytest.raises(KeyboardInterrupt, match="interrupt after finished worker"):
            list(projected)
    finally:
        projected.close()

    assert interrupted
    assert _child_pids() == baseline
    assert list(temporary.iterdir()) == []
