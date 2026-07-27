import threading
import time
from collections.abc import Iterable, Iterator
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any

from datapipeline.execution.events import (
    NodeFinished,
    NodeProgress,
    NodeStarted,
    PipelineFinished,
    PipelineProgress,
    PipelineStarted,
    PipelineSummary,
    ProgressSnapshot,
    RunStatus,
)
from datapipeline.execution.observer import PipelineObserver, ignore_pipeline_event
from datapipeline.execution.pipeline import Input, Pipeline, ProgressReader, Stage
from datapipeline.execution.settings import resolve_heartbeat_interval_seconds
from datapipeline.runtime import Runtime


_LIVE_PROGRESS_INTERVAL_SECONDS = 0.1
_NOOP_OBSERVER = ignore_pipeline_event


@dataclass(frozen=True)
class _NodeProgressContext:
    pipeline_name: str
    node_name: str
    node_index: int


@dataclass
class _NodeProgressState:
    started_at: float
    completed: int
    reader: ProgressReader | None
    last_live_progress: ProgressSnapshot | None = None


class _RunProgress:
    """Publish sampled progress without synchronizing every emitted item."""

    def __init__(
        self,
        observer: PipelineObserver,
        pipeline_name: str,
        interval_seconds: float,
    ) -> None:
        self._observer = observer
        self._pipeline_name = pipeline_name
        self._heartbeat_interval_seconds = interval_seconds
        self._started_at = time.perf_counter()
        self._last_heartbeat_at = self._started_at
        self._delivery_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_error: Exception | None = None
        self._states: dict[_NodeProgressContext, _NodeProgressState] = {}
        self.active_node: _NodeProgressContext | None = None
        self.output_items = 0

    def start(self) -> None:
        context = copy_context()
        self._thread = threading.Thread(
            target=lambda: context.run(self._run),
            name="datapipeline-pipeline-progress",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._thread_error is not None:
            raise RuntimeError("Pipeline progress failed") from self._thread_error

    def start_node(
        self,
        node: _NodeProgressContext,
        reader: ProgressReader | None,
    ) -> _NodeProgressState:
        now = time.perf_counter()
        state = _NodeProgressState(
            started_at=now,
            completed=0,
            reader=reader,
        )
        with self._state_lock:
            self._states[node] = state
        return state

    def clear(self, node: _NodeProgressContext) -> None:
        with self._delivery_lock:
            with self._state_lock:
                self._states.pop(node, None)
            if self.active_node == node:
                self.active_node = None

    def _run(self) -> None:
        try:
            while not self._stop.wait(_LIVE_PROGRESS_INTERVAL_SECONDS):
                self._emit_due_progress()
        except Exception as exc:
            self._thread_error = exc
            self._stop.set()

    def _emit_due_progress(self) -> None:
        active_node = self.active_node
        now = time.perf_counter()
        with self._delivery_lock:
            due = []
            with self._state_lock:
                heartbeat_due = (
                    self._heartbeat_interval_seconds > 0
                    and now - self._last_heartbeat_at
                    >= self._heartbeat_interval_seconds
                )
                if heartbeat_due:
                    self._last_heartbeat_at = now
                for node, state in self._states.items():
                    if node != active_node and state.reader is None:
                        continue
                    progress = self._snapshot(state)
                    heartbeat = heartbeat_due and node == active_node
                    if heartbeat:
                        state.last_live_progress = progress
                    elif progress != state.last_live_progress:
                        state.last_live_progress = progress
                    else:
                        continue
                    due.append((node, progress, now - state.started_at, heartbeat))
                output_items = self.output_items if heartbeat_due else 0
            for node, progress, elapsed, heartbeat in due:
                self._observer(
                    NodeProgress(
                        pipeline_name=node.pipeline_name,
                        node_name=node.node_name,
                        node_index=node.node_index,
                        progress=progress,
                        elapsed_seconds=elapsed,
                        heartbeat=heartbeat,
                    )
                )
            if heartbeat_due:
                self._observer(
                    PipelineProgress(
                        pipeline_name=self._pipeline_name,
                        output_items=output_items,
                        elapsed_seconds=now - self._started_at,
                    )
                )

    @staticmethod
    def _snapshot(state: _NodeProgressState) -> ProgressSnapshot:
        if state.reader is None:
            return ProgressSnapshot(completed=state.completed, unit="out")
        return state.reader(state.completed)


def _close_iterator(iterator: Iterable[Any]) -> None:
    closer = getattr(iterator, "close", None)
    if callable(closer):
        closer()


def _error_message(exc: BaseException) -> str | None:
    message = str(exc).strip()
    return message or None


def run_pipeline(
    runtime: Runtime,
    pipeline: Pipeline,
    *,
    observer: PipelineObserver | None = None,
) -> Iterator[Any]:
    """Run a pipeline input followed by its ordered stages."""

    active_observer = observer if observer is not None else runtime.pipeline_observer
    if active_observer is None or active_observer is _NOOP_OBSERVER:
        return _run_unobserved(pipeline)

    return _run_observed(
        pipeline,
        active_observer,
        observe_nodes=observer is not None or runtime.observe_node_events,
        heartbeat_interval_seconds=runtime.heartbeat_interval_seconds,
    )


def _run_unobserved(
    pipeline: Pipeline,
) -> Iterator[Any]:
    stream = _build_stream(pipeline, observer=None, progress=None)
    try:
        yield from stream
    finally:
        _close_iterator(stream)


def _run_observed(
    pipeline: Pipeline,
    observer: PipelineObserver,
    observe_nodes: bool,
    heartbeat_interval_seconds: float | None,
) -> Iterator[Any]:
    start_time = time.perf_counter()
    status: RunStatus = "success"
    error_type: str | None = None
    error_message: str | None = None
    stream: Iterable[Any] = ()
    iterator: Iterator[Any] = iter(())
    started = False
    heartbeat_interval = resolve_heartbeat_interval_seconds(
        heartbeat_interval_seconds
    )
    progress = _RunProgress(
        observer,
        pipeline.name,
        heartbeat_interval,
    )

    try:
        observer(
            PipelineStarted(
                pipeline_name=pipeline.name,
            )
        )
        if pipeline.summary:
            observer(
                PipelineSummary(
                    pipeline_name=pipeline.name,
                    summary=pipeline.summary,
                )
            )
        started = True
        if observe_nodes or heartbeat_interval > 0:
            progress.start()
        stream = _build_stream(
            pipeline,
            observer=observer if observe_nodes else None,
            progress=progress if observe_nodes else None,
        )
        iterator = iter(stream)
        while True:
            try:
                item = next(iterator)
            except StopIteration:
                break
            progress.output_items += 1
            yield item
    except GeneratorExit:
        raise
    except BaseException as exc:
        status = "error"
        error_type = type(exc).__name__
        error_message = _error_message(exc)
        raise
    finally:
        pipeline_failed = status == "error"
        close_error: BaseException | None = None
        try:
            _close_iterator(iterator)
            if iterator is not stream:
                _close_iterator(stream)
        except BaseException as exc:
            close_error = exc
            if not pipeline_failed:
                status = "error"
                error_type = type(exc).__name__
                error_message = _error_message(exc)

        progress_error: RuntimeError | None = None
        try:
            progress.stop()
        except RuntimeError as exc:
            progress_error = exc
            if not pipeline_failed and close_error is None:
                status = "error"
                error_type = type(exc).__name__
                error_message = _error_message(exc)

        if started:
            try:
                observer(
                    PipelineFinished(
                        pipeline_name=pipeline.name,
                        output_items=progress.output_items,
                        elapsed_seconds=time.perf_counter() - start_time,
                        status=status,
                        error_type=error_type,
                        error_message=error_message,
                    )
                )
            except BaseException:
                if (
                    not pipeline_failed
                    and close_error is None
                    and progress_error is None
                ):
                    raise
        if not pipeline_failed and close_error is not None:
            raise close_error
        if not pipeline_failed and progress_error is not None:
            raise progress_error


def _build_stream(
    pipeline: Pipeline,
    observer: PipelineObserver | None,
    progress: _RunProgress | None,
) -> Iterable[Any]:
    stream: Iterable[Any] = _wrap_node(
        pipeline.name,
        pipeline.input,
        0,
        upstream=None,
        observer=observer,
        progress=progress,
    )
    for index, stage in enumerate(pipeline.stages, start=1):
        stream = _wrap_node(
            pipeline.name,
            stage,
            index,
            upstream=stream,
            observer=observer,
            progress=progress,
        )
    return stream


def _wrap_node(
    pipeline_name: str,
    node: Input | Stage,
    node_index: int,
    upstream: Iterable[Any] | None,
    observer: PipelineObserver | None,
    progress: _RunProgress | None,
) -> Iterator[Any]:
    if observer is None or progress is None:
        return _unobserved_node(node, upstream)
    return _observed_node(
        pipeline_name,
        node,
        node_index,
        upstream,
        observer,
        progress,
    )


def _open_node(node: Input | Stage, upstream: Iterable[Any] | None) -> Iterable[Any]:
    if isinstance(node, Input):
        return node.open()
    if upstream is None:
        raise RuntimeError(f"Pipeline stage '{node.name}' has no input stream.")
    return node.apply(iter(upstream))


def _unobserved_node(
    node: Input | Stage,
    upstream: Iterable[Any] | None,
) -> Iterator[Any]:
    produced: Iterable[Any] = ()
    iterator: Iterator[Any] = iter(())
    processing_failed = False
    try:
        produced = _open_node(node, upstream)
        if produced is None:
            raise TypeError(
                f"Pipeline node '{node.name}' returned None; return an iterable."
            )
        iterator = iter(produced)
        yield from iterator
    except GeneratorExit:
        raise
    except BaseException:
        processing_failed = True
        raise
    finally:
        try:
            _close_iterator(iterator)
            if iterator is not produced:
                _close_iterator(produced)
            if upstream is not None and produced is not upstream:
                _close_iterator(upstream)
        except BaseException:
            if not processing_failed:
                raise


def _observed_node(
    pipeline_name: str,
    node: Input | Stage,
    node_index: int,
    upstream: Iterable[Any] | None,
    observer: PipelineObserver,
    progress: _RunProgress,
) -> Iterator[Any]:
    context = _NodeProgressContext(pipeline_name, node.name, node_index)
    start_time = time.perf_counter()
    output_items = 0
    status: RunStatus = "success"
    error_type: str | None = None
    error_message: str | None = None
    produced: Iterable[Any] = ()
    iterator: Iterator[Any] = iter(())
    started = False

    try:
        progress_state = progress.start_node(context, node.progress)
        observer(
            NodeStarted(
                pipeline_name=pipeline_name,
                node_name=node.name,
                node_index=node_index,
            )
        )
        started = True

        previous = progress.active_node
        progress.active_node = context
        try:
            produced = _open_node(node, upstream)
        finally:
            progress.active_node = previous
        if produced is None:
            raise TypeError(
                f"Pipeline node '{node.name}' returned None; return an iterable."
            )
        iterator = iter(produced)

        while True:
            previous = progress.active_node
            progress.active_node = context
            try:
                item = next(iterator)
            except StopIteration:
                break
            finally:
                progress.active_node = previous
            output_items += 1
            progress_state.completed = output_items
            yield item
    except GeneratorExit:
        raise
    except BaseException as exc:
        status = "error"
        error_type = type(exc).__name__
        error_message = _error_message(exc)
        raise
    finally:
        node_failed = status == "error"
        close_error: BaseException | None = None
        try:
            _close_iterator(iterator)
            if iterator is not produced:
                _close_iterator(produced)
            if upstream is not None and produced is not upstream:
                _close_iterator(upstream)
        except BaseException as exc:
            close_error = exc
            if not node_failed:
                status = "error"
                error_type = type(exc).__name__
                error_message = _error_message(exc)
        finally:
            progress.clear(context)

        if started:
            try:
                observer(
                    NodeFinished(
                        pipeline_name=pipeline_name,
                        node_name=node.name,
                        node_index=node_index,
                        output_items=output_items,
                        elapsed_seconds=time.perf_counter() - start_time,
                        status=status,
                        error_type=error_type,
                        error_message=error_message,
                    )
                )
            except BaseException:
                if not node_failed and close_error is None:
                    raise
        if not node_failed and close_error is not None:
            raise close_error
