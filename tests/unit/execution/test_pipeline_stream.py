import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from datapipeline.config.dataset.dataset import DatasetConfig, SampleConfig
from datapipeline.execution import runner as pipeline_runner
from datapipeline.execution.pipeline import Input, Pipeline, Stage
from datapipeline.execution.events import (
    NodeFinished,
    NodeProgress,
    NodeStarted,
    PipelineEvent,
    PipelineFinished,
    PipelineProgress,
    PipelineStarted,
    PipelineSummary,
    ProgressSnapshot,
)
from datapipeline.execution.observability import execution_observer
from datapipeline.execution.runner import run_pipeline
from datapipeline.runtime import Runtime


class _CollectingObserver:
    def __init__(self) -> None:
        self.pipeline_started: list[PipelineStarted] = []
        self.pipeline_summaries: list[PipelineSummary] = []
        self.node_started: list[NodeStarted] = []
        self.node_events: list[NodeFinished] = []
        self.progress_events: list[NodeProgress] = []
        self.pipeline_progress_events: list[PipelineProgress] = []
        self.pipeline_events: list[PipelineFinished] = []
        self._progress_condition = threading.Condition()

    def __call__(self, event: PipelineEvent) -> None:
        if isinstance(event, PipelineStarted):
            self.pipeline_started.append(event)
        elif isinstance(event, PipelineSummary):
            self.pipeline_summaries.append(event)
        elif isinstance(event, NodeStarted):
            self.node_started.append(event)
        elif isinstance(event, NodeFinished):
            self.node_events.append(event)
        elif isinstance(event, PipelineFinished):
            self.pipeline_events.append(event)
        elif isinstance(event, NodeProgress):
            with self._progress_condition:
                self.progress_events.append(event)
                self._progress_condition.notify_all()
        elif isinstance(event, PipelineProgress):
            with self._progress_condition:
                self.pipeline_progress_events.append(event)
                self._progress_condition.notify_all()

    def wait_for_progress(
        self,
        predicate: Callable[[NodeProgress], bool],
    ) -> bool:
        with self._progress_condition:
            return self._progress_condition.wait_for(
                lambda: any(predicate(event) for event in self.progress_events),
                timeout=1.0,
            )

    def wait_for_pipeline_progress(
        self,
        predicate: Callable[[PipelineProgress], bool],
    ) -> bool:
        with self._progress_condition:
            return self._progress_condition.wait_for(
                lambda: any(
                    predicate(event) for event in self.pipeline_progress_events
                ),
                timeout=1.0,
            )


class _ProcessingAndCleanupFailure(Iterator[int]):
    def __init__(self, processing_error: BaseException | None = None) -> None:
        self.processing_error = (
            processing_error
            if processing_error is not None
            else RuntimeError("processing failed")
        )
        self.closed = False

    def __next__(self) -> int:
        raise self.processing_error

    def close(self) -> None:
        self.closed = True
        raise OSError("cleanup failed")


def _runtime(tmp_path: Path) -> Runtime:
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(
        "schema_version: 4\nartifact_revision: 1\n", encoding="utf-8"
    )
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    return Runtime(
        project_yaml=project_yaml,
        artifacts_root=artifacts_root,
        dataset=DatasetConfig(sample=SampleConfig(cadence="1h")),
    )


def _events_by_name(
    observer: _CollectingObserver,
) -> dict[str, NodeFinished]:
    return {event.node_name: event for event in observer.node_events}


def test_pipeline_requires_unique_input_and_stage_names() -> None:
    with pytest.raises(ValueError, match="input and stage names must be unique"):
        Pipeline(
            name="duplicate",
            input=Input("records", lambda: ()),
            stages=(Stage("records", lambda records: records),),
        )


def test_pipeline_can_stop_at_an_ordered_stage() -> None:
    pipeline = Pipeline(
        name="linear",
        input=Input("source", lambda: ()),
        stages=(
            Stage("map", lambda records: records),
            Stage("filter", lambda records: records),
        ),
        summary="three stages",
    )

    assert len(pipeline.stages) == 2
    assert pipeline.input_only().input.name == "source"
    assert pipeline.input_only().stages == ()
    assert [stage.name for stage in pipeline.through_stage_count(1).stages] == ["map"]
    assert [stage.name for stage in pipeline.through_stage_named("filter").stages] == [
        "map",
        "filter",
    ]
    with pytest.raises(ValueError, match="no stage named 'missing'"):
        pipeline.through_stage_named("missing")
    with pytest.raises(ValueError, match="stage count -1 is out of range"):
        pipeline.through_stage_count(-1)


def test_pipeline_can_continue_under_a_new_identity() -> None:
    def progress(completed: int) -> ProgressSnapshot:
        return ProgressSnapshot(completed=completed)

    source = Input("source", lambda: (), progress=progress)
    map_stage = Stage("map", lambda records: records, progress=progress)
    filter_stage = Stage("filter", lambda records: records)
    upstream = Pipeline(
        name="upstream",
        input=source,
        stages=(map_stage,),
        summary="source summary",
    )

    continued = upstream.continue_as("downstream", (filter_stage,))

    assert continued.name == "downstream"
    assert continued.input == replace(source, name="upstream/source")
    assert continued.stages == (
        replace(map_stage, name="upstream/map"),
        filter_stage,
    )
    assert continued.input.open is source.open
    assert continued.input.progress is progress
    assert continued.stages[0].apply is map_stage.apply
    assert continued.stages[0].progress is progress
    assert continued.summary == "source summary"
    assert upstream.input == source
    assert upstream.stages == (map_stage,)

    nested = continued.continue_as("final", ())
    assert nested.input.name == "downstream/upstream/source"
    assert [stage.name for stage in nested.stages] == [
        "downstream/upstream/map",
        "downstream/filter",
    ]


def test_run_starts_lazily(tmp_path: Path) -> None:
    opened: list[str] = []
    observer = _CollectingObserver()

    def open_records() -> Iterator[int]:
        opened.append("source")
        yield 1

    stream = run_pipeline(
        _runtime(tmp_path),
        Pipeline(name="lazy", input=Input("source", open_records)),
        observer=observer,
    )

    assert opened == []
    assert observer.pipeline_started == []
    assert next(stream) == 1
    assert opened == ["source"]
    assert observer.pipeline_started == [PipelineStarted(pipeline_name="lazy")]
    stream.close()


def test_stages_emit_ordered_results_and_counts(tmp_path: Path) -> None:
    observer = _CollectingObserver()

    def odd(records: Iterable[int]) -> Iterator[int]:
        for value in records:
            if value % 2:
                yield value

    pipeline = Pipeline(
        name="numbers",
        summary="filter then scale",
        input=Input("source", lambda: [1, 2, 3]),
        stages=(
            Stage("odd", odd),
            Stage("scale", lambda records: (value * 10 for value in records)),
        ),
    )

    assert list(run_pipeline(_runtime(tmp_path), pipeline, observer=observer)) == [
        10,
        30,
    ]

    events = _events_by_name(observer)
    assert {
        name: (event.node_index, event.output_items, event.status)
        for name, event in events.items()
    } == {
        "source": (0, 3, "success"),
        "odd": (1, 2, "success"),
        "scale": (2, 2, "success"),
    }
    assert observer.pipeline_started == [PipelineStarted(pipeline_name="numbers")]
    assert observer.pipeline_summaries == [
        PipelineSummary(pipeline_name="numbers", summary="filter then scale")
    ]
    assert observer.pipeline_events[-1].output_items == 2
    assert observer.pipeline_events[-1].status == "success"


def test_custom_progress_belongs_to_the_reporting_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_runner, "_LIVE_PROGRESS_INTERVAL_SECONDS", 0.001)
    observer = _CollectingObserver()

    def annotate(records: Iterable[int]) -> Iterator[int]:
        for value in records:
            yield value
            assert observer.wait_for_progress(
                lambda event: (
                    event.node_name == "annotate"
                    and event.progress.completed == value
                    and event.progress.detail == "after input read"
                )
            )

    def progress(completed: int) -> ProgressSnapshot:
        return ProgressSnapshot(
            completed=completed,
            total=2,
            phase="mapping",
            detail="after input read",
        )

    pipeline = Pipeline(
        name="progress",
        input=Input("source", lambda: [1, 2]),
        stages=(Stage("annotate", annotate, progress=progress),),
    )

    assert list(run_pipeline(_runtime(tmp_path), pipeline, observer=observer)) == [1, 2]

    custom = [
        event
        for event in observer.progress_events
        if event.progress.detail == "after input read"
    ]
    assert [event.progress.completed for event in custom] == [1, 2]
    assert all(event.node_name == "annotate" for event in custom)
    assert all(event.node_index == 1 for event in custom)
    assert all(not event.heartbeat for event in custom)


def test_unobserved_pipeline_does_not_read_progress(tmp_path: Path) -> None:
    def fail_progress(_completed: int) -> ProgressSnapshot:
        raise AssertionError("unobserved pipeline sampled progress")

    pipeline = Pipeline(
        name="unobserved",
        input=Input("source", lambda: [1, 2], progress=fail_progress),
    )

    assert list(run_pipeline(_runtime(tmp_path), pipeline)) == [1, 2]


def test_pipeline_only_observation_skips_node_instrumentation(tmp_path: Path) -> None:
    observer = _CollectingObserver()
    runtime = _runtime(tmp_path)
    runtime.observe_node_events = False

    def fail_progress(_completed: int) -> ProgressSnapshot:
        raise AssertionError("pipeline-only observation sampled node progress")

    pipeline = Pipeline(
        name="pipeline-only",
        summary="two stages",
        input=Input("source", lambda: [1, 2], progress=fail_progress),
        stages=(Stage("double", lambda records: (value * 2 for value in records)),),
    )

    with execution_observer(observer):
        assert list(run_pipeline(runtime, pipeline)) == [2, 4]
    assert observer.pipeline_started == [PipelineStarted(pipeline_name="pipeline-only")]
    assert observer.pipeline_summaries == [
        PipelineSummary(pipeline_name="pipeline-only", summary="two stages")
    ]
    assert observer.pipeline_events[-1].output_items == 2
    assert observer.node_started == []
    assert observer.node_events == []
    assert observer.progress_events == []


def test_pipeline_only_observation_keeps_pipeline_heartbeats(tmp_path: Path) -> None:
    observer = _CollectingObserver()
    runtime = _runtime(tmp_path)
    runtime.observe_node_events = False
    runtime.heartbeat_interval_seconds = 0.01

    def source() -> Iterator[int]:
        assert observer.wait_for_pipeline_progress(
            lambda event: event.output_items == 0
        )
        yield 1

    pipeline = Pipeline(name="heartbeat", input=Input("source", source))

    with execution_observer(observer):
        assert list(run_pipeline(runtime, pipeline)) == [1]
    assert observer.pipeline_progress_events
    assert observer.progress_events == []


def test_pipeline_only_observation_without_heartbeats_skips_progress_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = _CollectingObserver()
    runtime = _runtime(tmp_path)
    runtime.observe_node_events = False
    runtime.heartbeat_interval_seconds = 0
    monkeypatch.setattr(
        pipeline_runner._RunProgress,
        "start",
        lambda _self: pytest.fail("progress thread started"),
    )
    pipeline = Pipeline(
        name="pipeline-only",
        input=Input("source", lambda: [1, 2]),
    )

    with execution_observer(observer):
        assert list(run_pipeline(runtime, pipeline)) == [1, 2]
    assert observer.pipeline_events[-1].output_items == 2
    assert observer.node_started == []
    assert observer.progress_events == []
    assert observer.pipeline_progress_events == []


def test_explicit_observer_includes_nodes_in_pipeline_only_runtime(
    tmp_path: Path,
) -> None:
    observer = _CollectingObserver()
    runtime = _runtime(tmp_path)
    runtime.observe_node_events = False
    pipeline = Pipeline(name="explicit", input=Input("source", lambda: [1]))

    assert list(run_pipeline(runtime, pipeline, observer=observer)) == [1]

    assert [event.node_name for event in observer.node_started] == ["source"]
    assert [event.node_name for event in observer.node_events] == ["source"]


def test_explicit_observer_overrides_scoped_observer(tmp_path: Path) -> None:
    scoped = _CollectingObserver()
    explicit = _CollectingObserver()
    pipeline = Pipeline(name="explicit", input=Input("source", lambda: [1]))

    with execution_observer(scoped):
        assert list(run_pipeline(_runtime(tmp_path), pipeline, observer=explicit)) == [
            1
        ]

    assert explicit.pipeline_started == [PipelineStarted(pipeline_name="explicit")]
    assert scoped.pipeline_started == []


def test_runtime_snapshots_node_observation_for_lazy_execution(tmp_path: Path) -> None:
    observer = _CollectingObserver()
    runtime = _runtime(tmp_path)
    runtime.observe_node_events = False
    pipeline = Pipeline(name="lazy-detail", input=Input("source", lambda: [1]))
    with execution_observer(observer):
        stream = run_pipeline(runtime, pipeline)

    runtime.observe_node_events = True

    assert list(stream) == [1]
    assert observer.node_started == []
    assert observer.node_events == []


def test_progress_reader_is_sampled_while_another_node_is_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_runner, "_LIVE_PROGRESS_INTERVAL_SECONDS", 0.001)
    observer = _CollectingObserver()

    def source_progress(completed: int) -> ProgressSnapshot:
        return ProgressSnapshot(completed=completed, detail="source resource")

    def slow_stage(records: Iterable[int]) -> Iterator[int]:
        for value in records:
            assert observer.wait_for_progress(
                lambda event: (
                    event.node_name == "source"
                    and event.progress.completed == value
                    and event.progress.detail == "source resource"
                )
            )
            yield value

    pipeline = Pipeline(
        name="nested-progress",
        input=Input("source", lambda: [1, 2], progress=source_progress),
        stages=(Stage("slow", slow_stage),),
    )

    assert list(run_pipeline(_runtime(tmp_path), pipeline, observer=observer)) == [1, 2]
    source_events = [
        event for event in observer.progress_events if event.node_name == "source"
    ]
    assert source_events
    assert all(not event.heartbeat for event in source_events)


def test_progress_failure_still_finishes_pipeline_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_runner, "_LIVE_PROGRESS_INTERVAL_SECONDS", 0.001)
    observer = _CollectingObserver()
    failed = threading.Event()

    def fail_progress(_completed: int) -> ProgressSnapshot:
        failed.set()
        raise ValueError("broken progress")

    def source() -> Iterator[int]:
        assert failed.wait(1.0)
        yield 1

    pipeline = Pipeline(
        name="progress-failure",
        input=Input("source", source, progress=fail_progress),
    )

    with pytest.raises(RuntimeError, match="Pipeline progress failed"):
        list(run_pipeline(_runtime(tmp_path), pipeline, observer=observer))

    assert observer.pipeline_events[-1].status == "error"
    assert observer.pipeline_events[-1].error_type == "RuntimeError"


def test_progress_failure_does_not_mask_pipeline_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_runner, "_LIVE_PROGRESS_INTERVAL_SECONDS", 0.001)
    observer = _CollectingObserver()
    progress_failed = threading.Event()

    def fail_progress(_completed: int) -> ProgressSnapshot:
        progress_failed.set()
        raise RuntimeError("progress failed")

    def fail_pipeline() -> Iterator[int]:
        assert progress_failed.wait(1.0)
        raise ValueError("pipeline failed")
        yield 1

    pipeline = Pipeline(
        name="two-failures",
        input=Input("source", fail_pipeline, progress=fail_progress),
    )

    with pytest.raises(ValueError, match="pipeline failed"):
        list(run_pipeline(_runtime(tmp_path), pipeline, observer=observer))

    assert observer.pipeline_events[-1].error_type == "ValueError"


def test_progress_stop_waits_without_an_arbitrary_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeouts: list[float | None] = []
    thread = threading.Thread()

    def record_timeout(timeout: float | None = None) -> None:
        timeouts.append(timeout)

    monkeypatch.setattr(thread, "join", record_timeout)
    progress = pipeline_runner._RunProgress(lambda _event: None, "slow", 1.0)
    progress._thread = thread

    progress.stop()

    assert timeouts == [None]


def test_finish_observer_failures_do_not_mask_pipeline_failure(
    tmp_path: Path,
) -> None:
    finished: list[type[PipelineEvent]] = []

    def observer(event: PipelineEvent) -> None:
        if isinstance(event, (NodeFinished, PipelineFinished)):
            finished.append(type(event))
            raise RuntimeError("observer failed")

    def fail() -> Iterator[int]:
        raise ValueError("source failed")
        yield

    pipeline = Pipeline(name="failure", input=Input("source", fail))

    with pytest.raises(ValueError, match="source failed"):
        list(run_pipeline(_runtime(tmp_path), pipeline, observer=observer))

    assert finished == [NodeFinished, PipelineFinished]


@pytest.mark.parametrize("event_type", [NodeFinished, PipelineFinished])
def test_finish_observer_failure_is_reported_after_success(
    tmp_path: Path,
    event_type: type[PipelineEvent],
) -> None:
    def observer(event: PipelineEvent) -> None:
        if isinstance(event, event_type):
            raise RuntimeError("observer failed")

    pipeline = Pipeline(name="success", input=Input("source", lambda: [1]))

    with pytest.raises(RuntimeError, match="observer failed"):
        list(run_pipeline(_runtime(tmp_path), pipeline, observer=observer))


def test_live_progress_samples_emitted_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_runner, "_LIVE_PROGRESS_INTERVAL_SECONDS", 0.001)
    observer = _CollectingObserver()
    runtime = _runtime(tmp_path)
    runtime.heartbeat_interval_seconds = 0

    def source() -> Iterator[int]:
        yield 1
        assert observer.wait_for_progress(
            lambda event: (
                event.node_name == "source"
                and event.progress.completed >= 1
                and not event.heartbeat
            )
        )
        yield 2

    pipeline = Pipeline(name="sampled", input=Input("source", source))

    assert list(run_pipeline(runtime, pipeline, observer=observer)) == [1, 2]


def test_heartbeat_reports_current_item_count(tmp_path: Path) -> None:
    observer = _CollectingObserver()
    runtime = _runtime(tmp_path)
    runtime.heartbeat_interval_seconds = 0.01

    def source() -> Iterator[int]:
        assert observer.wait_for_progress(
            lambda event: event.heartbeat and event.progress.completed == 0
        )
        yield 1
        assert observer.wait_for_progress(
            lambda event: event.heartbeat and event.progress.completed >= 1
        )
        yield 2

    pipeline = Pipeline(name="heartbeat", input=Input("source", source))

    assert list(run_pipeline(runtime, pipeline, observer=observer)) == [1, 2]
    heartbeats = [event for event in observer.progress_events if event.heartbeat]
    assert heartbeats[0].node_name == "source"
    assert heartbeats[0].progress.completed == 0
    assert heartbeats[-1].progress.completed == 1
    assert [event.output_items for event in observer.pipeline_progress_events] == [0, 1]


def test_heartbeat_interval_is_shared_across_pipeline_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    monkeypatch.setattr(pipeline_runner.time, "perf_counter", lambda: now)
    observer = _CollectingObserver()
    progress = pipeline_runner._RunProgress(observer, "pipeline", 10)
    source = pipeline_runner._NodeProgressContext("pipeline", "source", 0)
    output = pipeline_runner._NodeProgressContext("pipeline", "output", 1)
    source_state = progress.start_node(source, None)
    progress.start_node(output, None)
    progress.output_items = 7

    progress.active_node = source
    now = 10
    progress._emit_due_progress()

    progress.active_node = output
    now = 15
    progress._emit_due_progress()

    assert [
        event.node_name for event in observer.progress_events if event.heartbeat
    ] == ["source"]
    assert [
        (event.pipeline_name, event.output_items, event.elapsed_seconds)
        for event in observer.pipeline_progress_events
    ] == [("pipeline", 7, 10)]

    source_state.completed = 8
    progress.output_items = 8
    now = 20
    progress._emit_due_progress()

    assert [
        event.node_name for event in observer.progress_events if event.heartbeat
    ] == ["source", "output"]
    assert [event.output_items for event in observer.pipeline_progress_events] == [7, 8]


def test_live_progress_suppresses_unchanged_snapshots() -> None:
    observer = _CollectingObserver()
    progress = pipeline_runner._RunProgress(observer, "pipeline", 0)
    node = pipeline_runner._NodeProgressContext("pipeline", "source", 0)
    state = progress.start_node(node, None)
    progress.active_node = node

    progress._emit_due_progress()
    progress._emit_due_progress()
    state.completed = 1
    progress._emit_due_progress()

    assert [event.progress.completed for event in observer.progress_events] == [0, 1]


@pytest.mark.parametrize("observed", [False, True], ids=["unobserved", "observed"])
def test_partial_close_closes_stages_in_reverse_order(
    tmp_path: Path,
    observed: bool,
) -> None:
    closed: list[str] = []
    observer = _CollectingObserver() if observed else None

    def source() -> Iterator[int]:
        try:
            yield 1
            yield 2
        finally:
            closed.append("source")

    def closing_stage(
        records: Iterable[int],
        name: str,
    ) -> Iterator[int]:
        try:
            for record in records:
                yield record
        finally:
            closed.append(name)

    pipeline = Pipeline(
        name="closing",
        input=Input("source", source),
        stages=(
            Stage("first", lambda records: closing_stage(records, "first")),
            Stage("second", lambda records: closing_stage(records, "second")),
        ),
    )

    stream = run_pipeline(_runtime(tmp_path), pipeline, observer=observer)
    assert next(stream) == 1
    stream.close()

    assert closed == ["second", "first", "source"]
    if observer is None:
        return
    assert {
        event.node_name: (event.output_items, event.status)
        for event in observer.node_events
    } == {
        "source": (1, "success"),
        "first": (1, "success"),
        "second": (1, "success"),
    }
    assert observer.pipeline_events[-1].output_items == 1


@pytest.mark.parametrize("input_output", [True, False])
def test_none_pipeline_output_is_rejected(
    tmp_path: Path,
    input_output: bool,
) -> None:
    observer = _CollectingObserver()
    if input_output:
        pipeline = Pipeline(
            name="none",
            input=Input("broken", lambda: None),  # type: ignore[arg-type]
        )
    else:
        pipeline = Pipeline(
            name="none",
            input=Input("input", lambda: [1]),
            stages=(
                Stage("broken", lambda records: None),  # type: ignore[arg-type]
            ),
        )

    with pytest.raises(
        TypeError,
        match="Pipeline node 'broken' returned None; return an iterable",
    ):
        list(run_pipeline(_runtime(tmp_path), pipeline, observer=observer))

    event = observer.node_events[-1]
    assert event.node_name == "broken"
    assert event.status == "error"
    assert event.error_type == "TypeError"
    assert observer.pipeline_events[-1].status == "error"


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (RuntimeError("boom"), "boom"),
        (KeyboardInterrupt(), None),
        (SystemExit(2), "2"),
    ],
)
def test_stage_failures_reach_node_and_pipeline_events(
    tmp_path: Path,
    error: BaseException,
    message: str | None,
) -> None:
    observer = _CollectingObserver()

    def fail(records: Iterable[int]) -> Iterator[int]:
        for value in records:
            if value == 2:
                raise error
            yield value

    pipeline = Pipeline(
        name="failure",
        input=Input("source", lambda: [1, 2, 3]),
        stages=(Stage("fail", fail),),
    )

    with pytest.raises(type(error)):
        list(run_pipeline(_runtime(tmp_path), pipeline, observer=observer))

    node_event = _events_by_name(observer)["fail"]
    assert node_event.output_items == 1
    assert node_event.status == "error"
    assert node_event.error_type == type(error).__name__
    assert node_event.error_message == message
    pipeline_event = observer.pipeline_events[-1]
    assert pipeline_event.output_items == 1
    assert pipeline_event.status == "error"
    assert pipeline_event.error_type == type(error).__name__
    assert pipeline_event.error_message == message


@pytest.mark.parametrize(
    "processing_error",
    [RuntimeError("processing failed"), SystemExit(2)],
)
def test_node_cleanup_does_not_replace_processing_failure(
    tmp_path: Path,
    processing_error: BaseException,
) -> None:
    observer = _CollectingObserver()
    stream = _ProcessingAndCleanupFailure(processing_error)
    pipeline = Pipeline(
        name="failure",
        input=Input("source", lambda: stream),
    )

    with pytest.raises(type(processing_error), match=str(processing_error)):
        list(run_pipeline(_runtime(tmp_path), pipeline, observer=observer))

    node_event = _events_by_name(observer)["source"]
    assert node_event.status == "error"
    assert node_event.error_type == type(processing_error).__name__
    assert node_event.error_message == str(processing_error)
    pipeline_event = observer.pipeline_events[-1]
    assert pipeline_event.status == "error"
    assert pipeline_event.error_type == type(processing_error).__name__
    assert pipeline_event.error_message == str(processing_error)
    assert stream.closed


@pytest.mark.parametrize(
    "processing_error",
    [RuntimeError("processing failed"), SystemExit(2)],
)
def test_unobserved_cleanup_does_not_replace_processing_failure(
    tmp_path: Path,
    processing_error: BaseException,
) -> None:
    stream = _ProcessingAndCleanupFailure(processing_error)
    pipeline = Pipeline(
        name="failure",
        input=Input("source", lambda: stream),
    )

    with pytest.raises(type(processing_error), match=str(processing_error)):
        list(run_pipeline(_runtime(tmp_path), pipeline))

    assert stream.closed


def test_node_cleanup_failure_emits_failed_finish_event(tmp_path: Path) -> None:
    observer = _CollectingObserver()

    def source() -> Iterator[int]:
        try:
            yield 1
            yield 2
        finally:
            raise RuntimeError("close failed")

    pipeline = Pipeline(
        name="cleanup-failure",
        input=Input("source", source),
    )

    execution = run_pipeline(_runtime(tmp_path), pipeline, observer=observer)
    assert next(execution) == 1
    with pytest.raises(RuntimeError, match="close failed"):
        execution.close()

    node_event = _events_by_name(observer)["source"]
    assert node_event.output_items == 1
    assert node_event.status == "error"
    assert node_event.error_type == "RuntimeError"
    assert node_event.error_message == "close failed"
    assert observer.pipeline_events[-1].status == "error"


def test_unobserved_run_uses_the_fast_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_progress_start(_progress: object) -> None:
        raise AssertionError("unobserved pipelines must not start progress machinery")

    monkeypatch.setattr(
        pipeline_runner._RunProgress, "start", unexpected_progress_start
    )
    pipeline = Pipeline(
        name="fast",
        input=Input("source", lambda: [1, 2]),
        stages=(Stage("double", lambda records: (value * 2 for value in records)),),
    )

    assert list(run_pipeline(_runtime(tmp_path), pipeline)) == [2, 4]
