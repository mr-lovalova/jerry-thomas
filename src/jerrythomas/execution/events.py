from dataclasses import dataclass
from math import isfinite
from typing import Literal


RunStatus = Literal["success", "error"]


@dataclass(frozen=True, kw_only=True)
class PipelineStarted:
    pipeline_name: str


@dataclass(frozen=True, kw_only=True)
class PipelineSummary:
    pipeline_name: str
    summary: str


@dataclass(frozen=True, kw_only=True)
class NodeStarted:
    pipeline_name: str
    node_name: str
    node_index: int


@dataclass(frozen=True)
class ProgressResource:
    index: int
    total: int
    label: str


@dataclass(frozen=True)
class ProgressSnapshot:
    completed: int
    total: int | None = None
    unit: str = "items"
    phase: str | None = None
    detail: str | None = None
    resource: ProgressResource | None = None


def format_elapsed(seconds: float) -> str:
    if not isfinite(seconds) or seconds < 0:
        raise ValueError("Elapsed seconds must be finite and non-negative")

    milliseconds = round(seconds * 1_000)
    if milliseconds < 1_000:
        return f"{milliseconds}ms"

    tenths = round(seconds * 10)
    if tenths < 600:
        return f"{tenths / 10:.1f}s"

    hours, remainder = divmod(tenths, 36_000)
    minutes, remainder = divmod(remainder, 600)
    elapsed_seconds = remainder / 10
    if hours:
        return f"{hours}h{minutes:02d}m{elapsed_seconds:04.1f}s"
    return f"{minutes}m{elapsed_seconds:04.1f}s"


def format_node_progress(
    progress: ProgressSnapshot,
    elapsed_seconds: float,
) -> str:
    parts = [
        "running",
        f"elapsed={format_elapsed(elapsed_seconds)}",
        f"{progress.unit}={progress.completed}",
    ]
    if progress.total is not None:
        parts.append(f"total={progress.total}")
    if progress.phase:
        parts.append(f"phase={progress.phase}")
    if progress.detail:
        parts.append(f"detail={progress.detail}")
    if progress.resource is not None:
        resource = progress.resource
        parts.append(f"resource={resource.index}/{resource.total} {resource.label}")
    return " ".join(parts)


@dataclass(frozen=True, kw_only=True)
class NodeProgress:
    pipeline_name: str
    node_name: str
    node_index: int
    progress: ProgressSnapshot
    elapsed_seconds: float
    heartbeat: bool = False


@dataclass(frozen=True, kw_only=True)
class PipelineProgress:
    pipeline_name: str
    output_items: int
    elapsed_seconds: float


@dataclass(frozen=True, kw_only=True)
class NodeFinished:
    pipeline_name: str
    node_name: str
    node_index: int
    output_items: int
    elapsed_seconds: float
    status: RunStatus
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, kw_only=True)
class PipelineFinished:
    pipeline_name: str
    output_items: int
    elapsed_seconds: float
    status: RunStatus
    error_type: str | None = None
    error_message: str | None = None


PipelineEvent = (
    PipelineStarted
    | PipelineSummary
    | PipelineProgress
    | PipelineFinished
    | NodeStarted
    | NodeProgress
    | NodeFinished
)
