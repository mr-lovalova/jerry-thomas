from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial
from pathlib import Path

from jerrythomas.artifacts.output import ArtifactOutput
from jerrythomas.artifacts.scaler import (
    FoldedScalerArtifact,
    StandardScalerArtifact,
    save_scaler_artifact,
)
from jerrythomas.config.dataset.series import SeriesConfig
from jerrythomas.config.dataset.split import DatasetFold, TimeSplitConfig
from jerrythomas.config.tasks.scaler import ScalerTask
from jerrythomas.domain.series import SeriesRecord
from jerrythomas.domain.sample_key import SampleKeyContract, SampleKeyValueType
from jerrythomas.execution.pipeline import Input, Pipeline
from jerrythomas.execution.runner import run_pipeline
from jerrythomas.pipelines.dataset.split import TargetHorizonPolicy, build_labeler
from jerrythomas.pipelines.series.projector import SeriesProjector
from jerrythomas.pipelines.stream.pipeline import run_stream_pipeline
from jerrythomas.runtime import Runtime, require_runtime_stream
from jerrythomas.services.stream_workers import StreamWorkerProgress, run_stream_jobs
from jerrythomas.services.temp_cleanup import sort_spill_directory
from jerrythomas.transforms.vector.scaler import ScalerAccumulator
from jerrythomas.transforms.utils import record_establishes_domain
from jerrythomas.utils.time import round_time_to_cadence, parse_cadence


@dataclass(frozen=True)
class _StreamPlan:
    stream_id: str
    configs: tuple[SeriesConfig, ...]


@dataclass
class _ScalerFit:
    accumulator: ScalerAccumulator
    expected_ids: set[str]
    training_ids: set[str]

    def extend(self, other: "_ScalerFit") -> None:
        self.accumulator.extend(other.accumulator)
        self.expected_ids.update(other.expected_ids)
        self.training_ids.update(other.training_ids)


@dataclass(frozen=True)
class _FittedStream:
    fits: dict[str | None, _ScalerFit]
    key_types: tuple[SampleKeyValueType | None, ...]


@dataclass(frozen=True)
class _ScalerInput:
    group_key: tuple
    records: tuple[SeriesRecord, ...]


@dataclass
class _ScalerState:
    fit: _ScalerFit
    training_domain: set[tuple[tuple, str]] = field(default_factory=set)
    training_labels: frozenset[str] | None = None
    horizon_policy: TargetHorizonPolicy | None = None

    def observe(self, item: _ScalerInput, label: str | None = None) -> None:
        if self.horizon_policy is not None:
            assert label is not None
            if not self.horizon_policy.allows(label, item.group_key):
                return
        fit = self.fit
        genuine_ids = {
            record.id for record in item.records if record_establishes_domain(record)
        }
        fit.expected_ids.update(genuine_ids)
        if self.training_labels is not None and label not in self.training_labels:
            return

        fit.training_ids.update(genuine_ids)
        entity_key = item.group_key[1:]
        self.training_domain.update(
            (entity_key, series_id) for series_id in genuine_ids
        )
        for record in item.records:
            if (entity_key, record.id) in self.training_domain:
                fit.accumulator.observe(record.id, record.value)


def build_scaler_artifact(
    runtime: Runtime,
    task_cfg: ScalerTask,
) -> ArtifactOutput:
    progress = StreamWorkerProgress()
    (artifact,) = run_pipeline(
        runtime,
        Pipeline(
            name="scaler:artifact",
            input=Input(
                name="fit_scaler",
                open=partial(_fit_scaler, runtime, progress),
                progress=progress.snapshot,
            ),
        ),
    )
    if isinstance(artifact, StandardScalerArtifact):
        meta = {
            "series": len(artifact.statistics),
            "observations": artifact.observations,
        }
    else:
        meta = {
            "folds": len(artifact.folds),
            "observations": sum(
                scaler.observations for scaler in artifact.folds.values()
            ),
        }
    relative_path = Path(task_cfg.output)
    save_scaler_artifact(runtime.artifacts_root / relative_path, artifact)
    return ArtifactOutput(meta=meta)


def _fit_scaler(
    runtime: Runtime,
    progress: StreamWorkerProgress,
) -> Iterator[StandardScalerArtifact | FoldedScalerArtifact]:
    dataset = runtime.require_dataset()
    configs_by_stream: dict[str, list[SeriesConfig]] = defaultdict(list)
    for config in dataset.series:
        if config.scale:
            configs_by_stream[config.stream].append(config)
    plans = tuple(
        _StreamPlan(stream_id, tuple(configs))
        for stream_id, configs in configs_by_stream.items()
    )
    fold_ids = (
        tuple(fold.id for fold in dataset.split.folds)
        if dataset.split is not None
        else (None,)
    )
    fits = {
        fold_id: _ScalerFit(_new_accumulator(runtime), set(), set())
        for fold_id in fold_ids
    }
    sample_keys = SampleKeyContract(dataset.sample.keys)
    with sort_spill_directory() as temp_root:
        results = run_stream_jobs(runtime, plans, _fit_stream, temp_root, progress)
        results.reverse()
        while results:
            result = results.pop()
            sample_keys.merge_types(result.key_types)
            for fold_id, fit in result.fits.items():
                fits[fold_id].extend(fit)

    if dataset.split is None:
        fit = fits[None]
        yield _finish_scaler(fit.accumulator, fit.expected_ids, "dataset")
    else:
        yield FoldedScalerArtifact(
            folds={
                fold.id: _finish_folded_scaler(
                    fits[fold.id], f"dataset fold {fold.id!r}"
                )
                for fold in dataset.split.folds
            }
        )


def _fit_stream(
    runtime: Runtime,
    plan: _StreamPlan,
    _temp_root: Path,
) -> _FittedStream:
    dataset = runtime.require_dataset()
    sample_keys = SampleKeyContract(dataset.sample.keys)
    fits: dict[str | None, _ScalerFit]
    split = dataset.split
    if split is None:
        fits = {None: _fit_standard_scaler(runtime, plan, sample_keys)}
    else:
        fits = _fit_folded_scaler(runtime, plan, split.folds, sample_keys)
    return _FittedStream(fits, sample_keys.inferred_types)


def _fit_standard_scaler(
    runtime: Runtime,
    plan: _StreamPlan,
    sample_keys: SampleKeyContract,
) -> _ScalerFit:
    state = _ScalerState(_ScalerFit(_new_accumulator(runtime), set(), set()))
    inputs = _iter_scaler_inputs(runtime, plan, sample_keys)
    try:
        for item in inputs:
            state.observe(item)
    finally:
        _close_iterator(inputs)
    return state.fit


def _fit_folded_scaler(
    runtime: Runtime,
    plan: _StreamPlan,
    folds: Sequence[DatasetFold],
    sample_keys: SampleKeyContract,
) -> dict[str | None, _ScalerFit]:
    dataset = runtime.require_dataset()
    split = dataset.split
    assert split is not None

    target_horizon = dataset.max_target_horizon
    time_split = split if isinstance(split, TimeSplitConfig) else None
    states_by_label: dict[str, list[_ScalerState]] = defaultdict(list)
    states: dict[str, _ScalerState] = {}
    for fold in folds:
        state = _ScalerState(
            fit=_ScalerFit(_new_accumulator(runtime), set(), set()),
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
    inputs = _iter_scaler_inputs(runtime, plan, sample_keys)
    try:
        for item in inputs:
            label = labeler.label(item.group_key)
            for state in states_by_label.get(label, ()):
                state.observe(item, label)
    finally:
        _close_iterator(inputs)

    return {fold.id: states[fold.id].fit for fold in folds}


def _finish_folded_scaler(
    fit: _ScalerFit,
    scope: str,
) -> StandardScalerArtifact:
    untrained_ids = fit.expected_ids - fit.training_ids
    if untrained_ids:
        raise RuntimeError(
            f"Scaler fitting has no training observations for {scope} vector IDs: "
            + ", ".join(sorted(untrained_ids))
        )
    return _finish_scaler(fit.accumulator, fit.training_ids, scope)


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


def _new_accumulator(runtime: Runtime) -> ScalerAccumulator:
    policy = runtime.require_dataset().scaling
    return ScalerAccumulator(policy.with_mean, policy.with_std, policy.epsilon)


def _iter_scaler_inputs(
    runtime: Runtime,
    plan: _StreamPlan,
    sample_keys: SampleKeyContract,
) -> Iterator[_ScalerInput]:
    sample = runtime.require_dataset().sample
    cadence_step = parse_cadence(sample.cadence)
    runtime_stream = require_runtime_stream(runtime, plan.stream_id)
    projector = SeriesProjector(
        runtime_stream.partition_by,
        sample_keys,
        plan.configs,
    )
    records = run_stream_pipeline(runtime, plan.stream_id)
    try:
        for record in records:
            series_records = tuple(projector.project(record))
            yield _ScalerInput(
                group_key=(
                    round_time_to_cadence(record.time, cadence_step, sample.rounding),
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
