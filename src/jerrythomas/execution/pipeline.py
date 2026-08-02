from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Any, TypeAlias

from jerrythomas.execution.events import ProgressSnapshot


StageOp: TypeAlias = Callable[[Iterator[Any]], Iterable[Any]]
ProgressReader: TypeAlias = Callable[[int], ProgressSnapshot]


@dataclass(frozen=True)
class Input:
    """Open the first stream in a pipeline."""

    name: str
    open: Callable[[], Iterable[Any]]
    progress: ProgressReader | None = None


@dataclass(frozen=True)
class Stage:
    """Apply one ordered transformation to a stream."""

    name: str
    apply: StageOp
    progress: ProgressReader | None = None


@dataclass(frozen=True)
class Pipeline:
    """A linear, lazily evaluated record pipeline."""

    name: str
    input: Input
    stages: tuple[Stage, ...] = ()
    summary: str | None = None

    def __post_init__(self) -> None:
        names = [self.input.name, *(stage.name for stage in self.stages)]
        if len(names) != len(set(names)):
            raise ValueError(
                f"Pipeline '{self.name}' input and stage names must be unique."
            )

    def input_only(self) -> "Pipeline":
        return self.through_stage_count(0)

    def continue_as(
        self,
        pipeline_name: str,
        stages: tuple[Stage, ...],
    ) -> "Pipeline":
        inherited_stages = tuple(
            replace(stage, name=f"{self.name}/{stage.name}") for stage in self.stages
        )
        return Pipeline(
            name=pipeline_name,
            input=replace(self.input, name=f"{self.name}/{self.input.name}"),
            stages=(*inherited_stages, *stages),
            summary=self.summary,
        )

    def through_stage_named(self, stage_name: str) -> "Pipeline":
        for index, stage in enumerate(self.stages, start=1):
            if stage.name == stage_name:
                return self.through_stage_count(index)
        raise ValueError(f"Pipeline '{self.name}' has no stage named '{stage_name}'.")

    def through_stage_count(self, stage_count: int) -> "Pipeline":
        if not 0 <= stage_count <= len(self.stages):
            raise ValueError(
                f"Pipeline '{self.name}' stage count {stage_count} is out of range."
            )
        return Pipeline(
            name=self.name,
            input=self.input,
            stages=self.stages[:stage_count],
            summary=self.summary,
        )
