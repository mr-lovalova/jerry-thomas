import logging
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from jerrythomas.artifacts.output import ArtifactOutput
from jerrythomas.artifacts.registry import SERIES_SPEC
from jerrythomas.artifacts.scaler import (
    FoldedScalerArtifact,
    StandardScalerArtifact,
    save_scaler_artifact,
)
from jerrythomas.config.dataset.dataset import DatasetConfig
from jerrythomas.config.dataset.series import SeriesConfig
from jerrythomas.config.tasks.scaler import ScalerTask
from jerrythomas.domain.sample_key import SampleKeyContract, SampleKeyValueType
from jerrythomas.execution.pipeline import Input, Pipeline
from jerrythomas.execution.runner import run_pipeline
from jerrythomas.operations.artifacts.scaler_fit import (
    ScalerFits,
    StreamScalerFitter,
    finish_scaler_artifact,
    merge_fits,
    read_scaler_fits,
    scaler_fit_identity,
)
from jerrythomas.pipelines.series.projector import SeriesProjector
from jerrythomas.pipelines.stream.pipeline import run_stream_pipeline
from jerrythomas.runtime import Runtime, require_runtime_stream
from jerrythomas.services.stream_workers import StreamWorkerProgress, run_stream_jobs
from jerrythomas.services.temp_cleanup import sort_spill_directory
from jerrythomas.utils.time import parse_cadence, round_time_to_cadence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StreamPlan:
    stream_id: str
    configs: tuple[SeriesConfig, ...]


@dataclass(frozen=True)
class _FittedStream:
    fits: ScalerFits
    key_types: tuple[SampleKeyValueType | None, ...]


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
            "series": len(artifact.scalers),
            "observations": artifact.observations,
        }
    else:
        meta = {
            "folds": len(artifact.folds),
            "observations": sum(
                scaler.observations for scaler in artifact.folds.values()
            ),
        }
    relative_path = Path(task_cfg.path)
    save_scaler_artifact(runtime.artifacts_root / relative_path, artifact)
    return ArtifactOutput(meta=meta)


def _fit_scaler(
    runtime: Runtime,
    progress: StreamWorkerProgress,
) -> Iterator[StandardScalerArtifact | FoldedScalerArtifact]:
    dataset = runtime.require_dataset()
    fits = _series_pass_fits(runtime, dataset)
    if fits is None:
        fits = _fit_from_streams(runtime, dataset, progress)
    yield finish_scaler_artifact(dataset, fits)


def _series_pass_fits(runtime: Runtime, dataset: DatasetConfig) -> ScalerFits | None:
    """Return fits stored by the series build when they match the dataset settings."""
    if not runtime.artifacts.has(SERIES_SPEC):
        return None
    manifest = runtime.artifacts.load(SERIES_SPEC)
    if manifest.scaler_fits is None:
        return None
    path = runtime.artifacts.resolve_path(SERIES_SPEC).parent / manifest.scaler_fits
    try:
        identity, fits = read_scaler_fits(path)
    except (OSError, ValueError) as exc:
        # Refitting is always correct, only slower.
        logger.debug("Cannot reuse series scaler fits (%s); refitting.", exc)
        return None
    if identity != scaler_fit_identity(dataset):
        logger.debug(
            "Series scaler fits were computed for other dataset settings; refitting."
        )
        return None
    return fits


def _fit_from_streams(
    runtime: Runtime,
    dataset: DatasetConfig,
    progress: StreamWorkerProgress,
) -> ScalerFits:
    configs_by_stream: dict[str, list[SeriesConfig]] = defaultdict(list)
    for config in dataset.series:
        if config.scale is not None:
            configs_by_stream[config.stream].append(config)
    plans = tuple(
        _StreamPlan(stream_id, tuple(configs))
        for stream_id, configs in configs_by_stream.items()
    )
    sample_keys = SampleKeyContract(dataset.sample.keys)
    # Worker jobs spill source sorts into their job directories under temp_root.
    with sort_spill_directory() as temp_root:
        results = run_stream_jobs(runtime, plans, _fit_stream, temp_root, progress)
    for result in results:
        sample_keys.merge_types(result.key_types)
    return merge_fits(dataset, (result.fits for result in results))


def _fit_stream(runtime: Runtime, plan: _StreamPlan, _job_dir: Path) -> _FittedStream:
    dataset = runtime.require_dataset()
    sample = dataset.sample
    cadence = parse_cadence(sample.cadence)
    sample_keys = SampleKeyContract(sample.keys)
    projector = SeriesProjector(
        require_runtime_stream(runtime, plan.stream_id).partition_by,
        sample_keys,
        plan.configs,
    )
    fitter = StreamScalerFitter(dataset)
    records = run_stream_pipeline(runtime, plan.stream_id)
    try:
        for record in records:
            series_records = tuple(projector.project(record))
            fitter.observe(
                (
                    round_time_to_cadence(record.time, cadence, sample.rounding),
                    *series_records[0].entity_key,
                ),
                series_records,
            )
    finally:
        close = getattr(records, "close", None)
        if callable(close):
            close()
    return _FittedStream(fitter.fits(), sample_keys.inferred_types)
