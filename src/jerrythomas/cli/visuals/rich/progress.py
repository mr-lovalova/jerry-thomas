import logging
import sys
from contextlib import contextmanager
from datetime import timedelta

from rich.console import Console, RenderableType
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
)
from rich.rule import Rule
from rich.table import Column, Table
from rich.text import Text

from jerrythomas.cli.visuals.execution import (
    ExecutionEventFormatter,
)
from jerrythomas.cli.visuals.execution_context import (
    reset_current_execution_event_handler,
    reset_current_terminal_log_handler,
    set_current_execution_event_handler,
    set_current_terminal_log_handler,
)
from jerrythomas.execution.events import (
    NodeFinished,
    NodeProgress,
    NodeStarted,
    PipelineFinished,
    PipelineProgress,
    PipelineStarted,
    ProgressSnapshot,
)
from jerrythomas.execution.observability import (
    CommandFinished,
    ExecutionEvent,
    FileResult,
    OperationFinished,
    OperationProgress,
    OperationStarted,
)


class _ProgressRowColumn(ProgressColumn):
    def __init__(self, table_column: Column) -> None:
        super().__init__(table_column)
        self._bar = BarColumn(
            bar_width=20,
            style="grey30",
            complete_style="cyan",
            finished_style="cyan",
            pulse_style="cyan",
        )

    def render(self, task: Task) -> RenderableType:
        row = Table.grid(padding=(0, 1))
        row.add_column(overflow="ellipsis")
        row.add_column(min_width=7, no_wrap=True)
        cells: list[RenderableType] = [
            Text(task.description, no_wrap=True, overflow="ellipsis"),
            Text(str(timedelta(seconds=int(task.elapsed or 0)))),
        ]
        status = task.fields["status"]
        if task.total is not None:
            row.add_column(no_wrap=True)
            cells.append(self._bar.render(task))
        if status:
            if task.total is None:
                row.add_column(no_wrap=True)
                cells.append(Text("·", style="dim"))
            row.add_column(overflow="ellipsis")
            status = status.copy()
            status.no_wrap = True
            status.overflow = "ellipsis"
            cells.append(status)
        row.add_row(*cells)
        return row


class _ExecutionProgress:
    def __init__(self, progress: Progress, debug: bool) -> None:
        self._progress = progress
        self._debug = debug
        self._operation_name: str | None = None
        self._operation_task: TaskID | None = None
        self._pipeline_stack: list[str] = []
        self._pipeline_task: TaskID | None = None
        self._node_tasks: dict[tuple[str, int], TaskID] = {}
        self._open_nodes: list[tuple[str, int]] = []
        self._visible_node: tuple[str, int] | None = None

    def handle(self, event: ExecutionEvent) -> None:
        if isinstance(event, OperationStarted):
            self._start_operation(event)
        elif isinstance(event, OperationProgress):
            self._update_operation(event)
        elif isinstance(event, OperationFinished):
            self._finish_operation(event)
        elif isinstance(event, PipelineStarted):
            self._start_pipeline(event)
        elif isinstance(event, NodeStarted):
            self._start_node(event)
        elif isinstance(event, NodeProgress):
            self._update_node(event)
        elif isinstance(event, NodeFinished):
            self._finish_node(event)
        elif isinstance(event, PipelineFinished):
            self._finish_pipeline(event)
        else:
            raise TypeError(f"Unsupported progress event: {type(event).__name__}")

    def clear(self) -> None:
        self._clear_pipeline()
        if self._operation_task is not None:
            self._progress.remove_task(self._operation_task)
        self._operation_name = None
        self._operation_task = None
        self._progress.refresh()

    def _start_operation(self, event: OperationStarted) -> None:
        if self._operation_name is not None:
            raise RuntimeError("Cannot start overlapping operation progress")
        self._operation_name = event.name
        self._operation_task = self._progress.add_task(
            f"Operation {event.name}",
            total=None,
            status=Text(),
        )

    def _update_operation(self, event: OperationProgress) -> None:
        if self._operation_name is None or self._operation_task is None:
            raise RuntimeError("Cannot update operation progress before it starts")
        if self._operation_name != event.name:
            raise RuntimeError("Operation progress updated out of order")
        reported_at = timedelta(seconds=int(event.reported_at_seconds))
        status = Text.assemble(
            f"last report at {reported_at} · {event.step} · ",
            (f"{event.completed:,}", "cyan"),
            f" {event.unit}",
        )
        self._progress.update(
            self._operation_task,
            completed=event.completed,
            status=status,
        )

    def _finish_operation(self, event: OperationFinished) -> None:
        if self._operation_name is None or self._operation_task is None:
            raise RuntimeError("Cannot finish operation progress before it starts")
        if self._operation_name != event.name:
            raise RuntimeError("Operation progress finished out of order")
        self.clear()

    def _start_pipeline(self, event: PipelineStarted) -> None:
        if event.pipeline_name in self._pipeline_stack:
            raise RuntimeError("Cannot start duplicate pipeline progress")
        self._pipeline_stack.append(event.pipeline_name)
        if len(self._pipeline_stack) > 1:
            return
        self._pipeline_task = self._progress.add_task(
            f"[{event.pipeline_name}]",
            total=None,
            status=Text(),
        )

    def _finish_pipeline(self, event: PipelineFinished) -> None:
        if not self._pipeline_stack:
            raise RuntimeError("Cannot finish pipeline progress before it starts")
        if self._pipeline_stack[-1] != event.pipeline_name:
            raise RuntimeError("Pipeline progress finished out of order")
        self._pipeline_stack.pop()
        if self._pipeline_stack:
            return
        self._clear_pipeline()
        self._progress.refresh()

    def _start_node(self, event: NodeStarted) -> None:
        if event.pipeline_name not in self._pipeline_stack:
            raise RuntimeError("Cannot start node progress for an inactive pipeline")
        node = event.pipeline_name, event.node_index
        label = f"[{event.pipeline_name}/{event.node_name}]"
        status = Text.assemble(("0", "cyan"), " out")
        self._node_tasks[node] = self._progress.add_task(
            label,
            total=None,
            status=status,
            visible=self._debug,
        )
        self._open_nodes.append(node)
        if not self._debug:
            self._show_node(node)

    def _update_node(self, event: NodeProgress) -> None:
        if event.pipeline_name not in self._pipeline_stack:
            raise RuntimeError("Cannot update node progress for an inactive pipeline")
        node = event.pipeline_name, event.node_index
        self._update_task(
            self._node_tasks[node],
            completed=event.progress.completed,
            total=event.progress.total,
            status=_progress_status(event.progress),
        )
        if not self._debug and (
            event.pipeline_name == self._pipeline_stack[-1]
            and (
                event.heartbeat
                or event.progress.total is not None
                or event.progress.phase is not None
                or event.progress.detail is not None
                or event.progress.resource is not None
                or event.progress.unit not in ("items", "out")
                or self._open_nodes[-1] == node
            )
        ):
            self._show_node(node)

    def _finish_node(self, event: NodeFinished) -> None:
        if event.pipeline_name not in self._pipeline_stack:
            raise RuntimeError("Cannot finish node progress for an inactive pipeline")
        node = event.pipeline_name, event.node_index
        task_id = self._node_tasks.pop(node)
        self._open_nodes.remove(node)
        was_visible = self._visible_node == node
        if was_visible:
            self._visible_node = None
        self._progress.remove_task(task_id)
        if not self._debug and was_visible and self._open_nodes:
            self._show_node(self._open_nodes[-1])
        self._progress.refresh()

    def _show_node(self, node: tuple[str, int]) -> None:
        if self._visible_node == node:
            return
        if self._visible_node is not None:
            self._progress.update(
                self._node_tasks[self._visible_node],
                visible=False,
            )
        self._progress.update(self._node_tasks[node], visible=True)
        self._visible_node = node

    def _update_task(
        self,
        task_id: TaskID,
        completed: int,
        total: int | None,
        status: Text,
    ) -> None:
        task = next(task for task in self._progress.tasks if task.id == task_id)
        task.total = total
        self._progress.update(task_id, completed=completed, status=status)

    def _clear_pipeline(self) -> None:
        for task_id in self._node_tasks.values():
            self._progress.remove_task(task_id)
        if self._pipeline_task is not None:
            self._progress.remove_task(self._pipeline_task)
        self._pipeline_task = None
        self._pipeline_stack.clear()
        self._node_tasks.clear()
        self._open_nodes.clear()
        self._visible_node = None


def _progress_status(snapshot: ProgressSnapshot) -> Text:
    resource = snapshot.resource
    parts: list[Text] = []
    if snapshot.phase:
        parts.append(Text(snapshot.phase))
    if resource is not None:
        parts.append(
            Text.assemble(
                (f"{resource.index}/{resource.total}", "cyan"),
                f" {resource.label}",
            )
        )
    if snapshot.detail:
        parts.append(Text(snapshot.detail))
    count = f"{snapshot.completed:,}"
    if snapshot.total is not None:
        count = f"{count}/{snapshot.total:,}"
    parts.append(Text.assemble((count, "cyan"), f" {snapshot.unit}"))
    return Text(" · ").join(parts)


class _RichExecutionRenderer:
    def __init__(
        self,
        level: int,
        console,
        progress: _ExecutionProgress | None = None,
    ) -> None:
        self._level = int(level)
        self._console = console
        self._progress = progress

    def render(self, event: ExecutionEvent) -> None:
        if self._progress is not None and isinstance(event, PipelineProgress):
            return
        if self._progress is not None and isinstance(
            event,
            OperationStarted
            | OperationProgress
            | OperationFinished
            | PipelineStarted
            | NodeStarted
            | NodeProgress
            | NodeFinished
            | PipelineFinished,
        ):
            self._progress.handle(event)
            if isinstance(event, OperationStarted):
                self._console.print(Rule(Text(f"Operation {event.name}"), style="dim"))
                return
            if isinstance(
                event,
                OperationProgress | NodeStarted | NodeProgress,
            ):
                return
        if isinstance(event, NodeProgress):
            return
        event_level = ExecutionEventFormatter.level(event)
        if event_level < self._level:
            return
        if isinstance(event, FileResult):
            self._console.print(self._render_file_result(event))
            return
        text = self._render_event(event)
        self._console.print(text)

    @staticmethod
    def _render_file_result(event: FileResult) -> Table:
        table = Table.grid(padding=(0, 1))
        table.add_column(no_wrap=True)
        table.add_column(ratio=1, overflow="fold")
        result = Text(str(event.path))
        result.stylize(f"bright_blue link {event.path.resolve().as_uri()}")
        table.add_row(f"{event.label}:", result)
        return table

    def _render_event(self, event: ExecutionEvent) -> Text:
        level = ExecutionEventFormatter.level(event)
        text = Text(ExecutionEventFormatter.message(event))
        if isinstance(
            event,
            CommandFinished | PipelineFinished | NodeFinished | OperationFinished,
        ):
            status_style = "green" if event.status == "success" else "red"
            text.highlight_words([f"status={event.status}"], style=status_style)
        elif level >= logging.ERROR:
            text.stylize("red")
        elif level >= logging.WARNING:
            text.stylize("yellow")
        return text


def rich_visuals_supported() -> bool:
    if not sys.stderr.isatty():
        return False
    console = Console(file=sys.stderr, markup=False, highlight=False)
    return bool(
        console.is_terminal
        and console.is_interactive
        and not console.is_dumb_terminal
        and console.color_system is not None
    )


@contextmanager
def visual_summary(log_level: int, enabled: bool):
    if not enabled or not rich_visuals_supported():
        yield
        return
    console = Console(file=sys.stderr, markup=False, highlight=False)
    renderer = _RichExecutionRenderer(log_level, console)
    event_token = set_current_execution_event_handler(renderer.render)
    try:
        yield
    finally:
        reset_current_execution_event_handler(event_token)


@contextmanager
def visual_execution(log_level: int):
    console = Console(file=sys.stderr, markup=False, highlight=False)
    debug = log_level <= logging.DEBUG
    progress = Progress(
        _ProgressRowColumn(Column(no_wrap=True, overflow="ellipsis")),
        transient=True,
        console=console,
        refresh_per_second=10,
    )
    progress_renderer = _ExecutionProgress(progress, debug=debug)
    event_renderer = _RichExecutionRenderer(log_level, console, progress_renderer)
    event_token = set_current_execution_event_handler(event_renderer.render)
    proxy_token = set_current_terminal_log_handler(event_renderer.render)
    refresh_thread = None
    try:
        with progress:
            # Rich clears this reference when stopping without joining the thread.
            refresh_thread = progress.live._refresh_thread
            try:
                yield
            finally:
                progress_renderer.clear()
    finally:
        if refresh_thread is not None:
            # Do not let its FileProxy survive into interpreter shutdown.
            refresh_thread.join()
        reset_current_terminal_log_handler(proxy_token)
        reset_current_execution_event_handler(event_token)
