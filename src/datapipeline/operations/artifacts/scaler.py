from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from datapipeline.artifacts.scaler import (
    FoldedScalerArtifact,
    StandardScalerArtifact,
    save_scaler_artifact,
)
from datapipeline.artifacts.specs import dataset_requires_scaler
from datapipeline.config.dataset.series import SeriesConfig
from datapipeline.config.dataset.split import DatasetFold, TimeSplitConfig
from datapipeline.config.tasks.scaler import ScalerTask
from datapipeline.domain.series import SeriesRecord
from datapipeline.domain.sample_key import SampleKeyContract
from datapipeline.execution.context import PipelineContext
from datapipeline.operations.persistence import ArtifactOutput
from datapipeline.pipelines.dataset.split import TargetHorizonPolicy, build_labeler
from datapipeline.pipelines.series.projector import SeriesProjector
from datapipeline.pipelines.stream.pipeline import run_stream_pipeline
from datapipeline.runtime import Runtime, require_runtime_stream
from datapipeline.transforms.vector.scaler import ScalerAccumulator
from datapipeline.transforms.utils import record_establishes_domain
from datapipeline.utils.time import floor_time_to_cadence, parse_cadence


@dataclass(frozen=True)
class _ScalerInput:
    group_key: tuple
    records: tuple[SeriesRecord, ...]


@dataclass
class _FoldScalerState:
    accumulator: ScalerAccumulator
    expected_ids: set[str]
    training_ids: set[str]
    training_domain: set[tuple[tuple, str]]
    training_labels: frozenset[str]
    horizon_policy: TargetHorizonPolicy | None

    def observe(self, label: str, item: _ScalerInput) -> None:
        if self.horizon_policy is not None and not self.horizon_policy.allows(
            label,
            item.group_key,
        ):
            return
        genuine_ids = {
            record.id for record in item.records if record_establishes_domain(record)
        }
        self.expected_ids.update(genuine_ids)
        if label not in self.training_labels:
            return

        self.training_ids.update(genuine_ids)
        entity_key = item.group_key[1:]
        self.training_domain.update(
            (entity_key, series_id) for series_id in genuine_ids
        )
        for record in item.records:
            if (entity_key, record.id) in self.training_domain:
                self.accumulator.observe(record.id, record.value)


def build_scaler_artifact(
    runtime: Runtime,
    task_cfg: ScalerTask,
) -> ArtifactOutput | None:
    dataset = runtime.dataset
    if not dataset_requires_scaler(dataset):
        return None

    configs = tuple(config for config in dataset.series if config.scale)
    if dataset.split is None:
        standard = _fit_standard_scaler(runtime, configs, task_cfg)
        artifact: StandardScalerArtifact | FoldedScalerArtifact = standard
        meta = {
            "series": len(standard.statistics),
            "observations": standard.observations,
        }
    else:
        folded = _fit_folded_scaler(
            runtime,
            configs,
            dataset.split.folds,
            task_cfg,
        )
        artifact = folded
        meta = {
            "folds": len(folded.folds),
            "observations": sum(
                scaler.observations for scaler in folded.folds.values()
            ),
        }

    relative_path = Path(task_cfg.output)
    save_scaler_artifact(runtime.artifacts_root / relative_path, artifact)
    return ArtifactOutput(relative_path=str(relative_path), meta=meta)


def _fit_standard_scaler(
    runtime: Runtime,
    configs: Sequence[SeriesConfig],
    task: ScalerTask,
) -> StandardScalerArtifact:
    accumulator = _new_accumulator(task)
    expected_ids: set[str] = set()
    established_series: set[tuple[tuple, str]] = set()
    inputs = _iter_scaler_inputs(runtime, configs)
    try:
        for item in inputs:
            entity_key = item.group_key[1:]
            genuine_ids = {
                record.id
                for record in item.records
                if record_establishes_domain(record)
            }
            expected_ids.update(genuine_ids)
            established_series.update(
                (entity_key, series_id) for series_id in genuine_ids
            )
            for record in item.records:
                if (entity_key, record.id) in established_series:
                    accumulator.observe(record.id, record.value)
    finally:
        _close_iterator(inputs)
    return _finish_scaler(accumulator, expected_ids, "dataset")


def _fit_folded_scaler(
    runtime: Runtime,
    configs: Sequence[SeriesConfig],
    folds: Sequence[DatasetFold],
    task: ScalerTask,
) -> FoldedScalerArtifact:
    split = runtime.dataset.split
    assert split is not None

    target_horizon = runtime.dataset.max_target_horizon
    time_split = split if isinstance(split, TimeSplitConfig) else None
    states_by_label: dict[str, list[_FoldScalerState]] = defaultdict(list)
    states: dict[str, _FoldScalerState] = {}
    for fold in folds:
        state = _FoldScalerState(
            accumulator=_new_accumulator(task),
            expected_ids=set(),
            training_ids=set(),
            training_domain=set(),
            training_labels=frozenset(fold.train),
            horizon_policy=(
                TargetHorizonPolicy(time_split, fold, target_horizon)
                if time_split is not None and target_horizon > timedelta()
                else None
            ),
        )
        states[fold.id] = state
        for label in (*fold.train, *fold.validation, *fold.test):
            states_by_label[label].append(state)

    labeler = build_labeler(split)
    inputs = _iter_scaler_inputs(runtime, configs)
    try:
        for item in inputs:
            label = labeler.label(item.group_key)
            for state in states_by_label.get(label, ()):
                state.observe(label, item)
    finally:
        _close_iterator(inputs)

    return FoldedScalerArtifact(
        folds={
            fold.id: _finish_folded_scaler(
                states[fold.id],
                f"dataset fold {fold.id!r}",
            )
            for fold in folds
        }
    )


def _finish_folded_scaler(
    state: _FoldScalerState,
    scope: str,
) -> StandardScalerArtifact:
    untrained_ids = state.expected_ids - state.training_ids
    if untrained_ids:
        raise RuntimeError(
            f"Scaler fitting has no training observations for {scope} vector IDs: "
            + ", ".join(sorted(untrained_ids))
        )
    return _finish_scaler(state.accumulator, state.training_ids, scope)


def _finish_scaler(
    accumulator: ScalerAccumulator,
    expected_ids: set[str],
    scope: str,
) -> StandardScalerArtifact:
    if not expected_ids or accumulator.observations == 0:
        raise RuntimeError(f"Scaler fitting produced no observations for {scope}.")
    artifact = accumulator.artifact()
    missing = expected_ids - artifact.statistics.keys()
    if missing:
        raise RuntimeError(
            f"Scaler fitting has no training observations for {scope} vector IDs: "
            + ", ".join(sorted(missing))
        )
    statistics = {
        vector_id: artifact.statistics[vector_id] for vector_id in sorted(expected_ids)
    }
    return StandardScalerArtifact(
        with_mean=artifact.with_mean,
        with_std=artifact.with_std,
        epsilon=artifact.epsilon,
        observations=sum(entry.count for entry in statistics.values()),
        statistics=statistics,
    )


def _new_accumulator(task: ScalerTask) -> ScalerAccumulator:
    return ScalerAccumulator(task.with_mean, task.with_std, task.epsilon)


def _iter_scaler_inputs(
    runtime: Runtime,
    configs: Sequence[SeriesConfig],
) -> Iterator[_ScalerInput]:
    context = PipelineContext(runtime)
    cadence_step = parse_cadence(runtime.dataset.sample.cadence)
    sample_key_contract = SampleKeyContract(runtime.dataset.sample.keys)
    configs_by_stream: dict[str, list[SeriesConfig]] = defaultdict(list)
    for config in configs:
        configs_by_stream[config.stream].append(config)

    for stream_id, stream_configs in configs_by_stream.items():
        runtime_stream = require_runtime_stream(runtime, stream_id)
        projector = SeriesProjector(
            runtime_stream.partition_by,
            sample_key_contract,
        )
        records = run_stream_pipeline(context, stream_id)
        try:
            for record in records:
                series_records = tuple(projector.project(record, stream_configs))
                yield _ScalerInput(
                    group_key=(
                        floor_time_to_cadence(record.time, cadence_step),
                        *series_records[0].entity_key,
                    ),
                    records=series_records,
                )
        finally:
            _close_iterator(records)


def _close_iterator(items: Iterator[object]) -> None:
    closer = getattr(items, "close", None)
    if callable(closer):
        closer()
