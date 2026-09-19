import logging
import multiprocessing
import os
import tempfile
import time
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import jerrythomas.operations.artifacts.series_workers as worker_module
from jerrythomas.config.execution import ExecutionConfig
from jerrythomas.domain.sample_key import SampleKeyContract
from jerrythomas.execution.observability import (
    ExecutionEvent,
    ExecutionMessage,
    emit_execution_message,
    execution_observer,
)
from jerrythomas.operations.artifacts.series import _stream_plans
from jerrythomas.operations.artifacts.series_workers import (
    StreamWorkerError,
    StreamWorkerProgress,
    project_streams_parallel,
)
from jerrythomas.runtime import Runtime
from jerrythomas.services.project_definition import load_project_definition
from jerrythomas.services.runtime_compiler import compile_runtime
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
            if self.mode == "messages":
                emit_execution_message("worker plugin message")
                logging.getLogger("tests.stream_workers").warning(
                    "worker standard warning"
                )

            # Each pair fills the IPC batch; remaining rows keep both workers
            # active when the parent closes after receiving the first record.
            for index in range(12):
                yield {
                    "time": "2024-01-01T00:00:00Z",
                    "id_": "A",
                    "value": f"{index}:" + "x" * 600_000,
                }


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

    def build(modes: tuple[str, str]) -> tuple[Runtime, Path, tuple[Path, Path]]:
        root = tmp_path / "project"
        for name in ("sources", "streams"):
            (root / name).mkdir(parents=True)
        markers = (root / "first.started", root / "second.started")
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
                    for stream in ("first", "second")
                ],
            },
        )
        for index, (stream, mode) in enumerate(zip(("first", "second"), modes)):
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
                            "barrier": str(markers[1 - index]),
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
    return project_streams_parallel(
        runtime,
        _stream_plans(runtime.dataset.features, runtime.dataset.targets),
        SampleKeyContract(runtime.dataset.sample.keys),
        timedelta(hours=1),
        StreamWorkerProgress(),
    )


def _child_pids() -> set[int | None]:
    return {process.pid for process in multiprocessing.active_children()}


def test_closing_projection_reaps_workers_and_removes_their_spills(
    lifecycle_project,
) -> None:
    runtime, temporary, markers = lifecycle_project(("rows", "rows"))
    baseline = _child_pids()
    projected = _project(runtime)
    try:
        first = next(projected)
        assert first.features[0].id == "first"
        assert len(_child_pids() - baseline) == 2
        for marker in markers:
            spill = Path(marker.read_text(encoding="utf-8"))
            assert spill.is_relative_to(temporary)
            assert (spill / "run.pickle").exists()
    finally:
        projected.close()

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
    runtime, temporary, _ = lifecycle_project(("messages", "rows"))
    baseline = _child_pids()
    events: list[ExecutionEvent] = []
    projected = _project(runtime)
    with execution_observer(events.append), caplog.at_level(logging.WARNING):
        try:
            next(projected)
        finally:
            projected.close()

    assert events == [ExecutionMessage(message="worker plugin message")]
    assert (
        "tests.stream_workers",
        logging.WARNING,
        "worker standard warning",
    ) in caplog.record_tuples
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
