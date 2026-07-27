import logging
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from itertools import groupby
from pathlib import Path
from typing import Any
from uuid import uuid4

from datapipeline.artifacts.series import (
    SERIES_MANIFEST_VERSION,
    SeriesEntry,
    SeriesManifest,
    SeriesRow,
    write_series_rows,
)
from datapipeline.config.dataset.series import SeriesConfig
from datapipeline.config.tasks.series import SeriesTask
from datapipeline.domain.sample_key import SampleKeyContract
from datapipeline.domain.series import SeriesSequence
from datapipeline.domain.series_id import base_id
from datapipeline.execution.pipeline import Input, Pipeline, Stage
from datapipeline.execution.runner import run_pipeline
from datapipeline.io.json_file import write_json_object
from datapipeline.operations.persistence import ArtifactOutput
from datapipeline.pipelines.series.projector import SeriesProjector
from datapipeline.pipelines.series.stages import SeriesSequencer
from datapipeline.pipelines.sort import SortProgress, batch_sort
from datapipeline.pipelines.stream.pipeline import build_stream_pipeline
from datapipeline.runtime import Runtime, require_runtime_stream
from datapipeline.services.path_policy import resolve_artifact_output_path
from datapipeline.transforms.utils import record_establishes_domain
from datapipeline.utils.time import floor_time_to_cadence, parse_cadence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StreamPlan:
    stream_id: str
    features: tuple[SeriesConfig, ...]
    targets: tuple[SeriesConfig, ...]


@dataclass(frozen=True)
class _ProjectedScalar:
    id: str
    value: Any
    establishes_domain: bool


@dataclass(frozen=True)
class _ProjectedSequence:
    id: str
    values: list[Any]
    establishes_domain: bool


_ProjectedValue = _ProjectedScalar | _ProjectedSequence


@dataclass(frozen=True)
class _ProjectedRow:
    key: tuple[Any, ...]
    time: datetime
    features: tuple[_ProjectedValue, ...]
    targets: tuple[_ProjectedValue, ...]


def build_series_artifact(
    runtime: Runtime,
    task_cfg: SeriesTask,
) -> ArtifactOutput:
    dataset = runtime.dataset
    relative_path = Path(task_cfg.output)
    destination = resolve_artifact_output_path(relative_path, runtime.artifacts_root)
    cache_root = destination.parent / f"{destination.stem}.data"
    if cache_root.is_symlink():
        raise ValueError("Series data directory must not be a symlink.")

    generation = uuid4().hex
    staging_root = cache_root / f".staging-{generation}"
    generation_root = cache_root / generation
    data_path = staging_root / "series.jsonl.gz"
    sample_keys = SampleKeyContract(dataset.sample.keys)
    feature_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()

    try:
        staging_root.mkdir(parents=True)
        projected = _ordered_projected_rows(
            runtime,
            _stream_plans(dataset.features, dataset.targets),
            sample_keys,
            parse_cadence(dataset.sample.cadence),
        )
        try:
            rows = _group_series_rows(
                projected,
                dataset.features,
                dataset.targets,
                feature_counts,
                target_counts,
            )
            written = write_series_rows(data_path, rows)
        finally:
            _close_iterator(projected)

        relative_data_path = Path(cache_root.name) / generation / data_path.name
        manifest = SeriesManifest(
            version=SERIES_MANIFEST_VERSION,
            cadence=dataset.sample.cadence,
            sample_keys=tuple(dataset.sample.keys),
            sample_key_types=sample_keys.types,
            path=str(relative_data_path),
            rows=written.rows,
            sha256=written.sha256,
            features=tuple(
                SeriesEntry(id=config.id, samples=feature_counts[config.id])
                for config in dataset.features
            ),
            targets=tuple(
                SeriesEntry(id=config.id, samples=target_counts[config.id])
                for config in dataset.targets
            ),
        )
        staging_root.rename(generation_root)
    except BaseException:
        _remove_failed_generation(staging_root)
        raise

    try:
        write_json_object(destination, manifest.model_dump(mode="json"))
    except BaseException:
        _remove_failed_generation(generation_root)
        raise

    companion_path = relative_path.parent / manifest.path
    return ArtifactOutput(
        companion_paths=(str(companion_path),),
        meta={
            "features": len(manifest.features),
            "targets": len(manifest.targets),
            "rows": manifest.rows,
            "format": manifest.format,
        },
    )


def _stream_plans(
    features: Sequence[SeriesConfig],
    targets: Sequence[SeriesConfig],
) -> tuple[_StreamPlan, ...]:
    feature_configs: dict[str, list[SeriesConfig]] = defaultdict(list)
    target_configs: dict[str, list[SeriesConfig]] = defaultdict(list)
    stream_ids: list[str] = []
    seen: set[str] = set()

    for config in (*features, *targets):
        if config.stream not in seen:
            seen.add(config.stream)
            stream_ids.append(config.stream)
    for config in features:
        feature_configs[config.stream].append(config)
    for config in targets:
        target_configs[config.stream].append(config)

    return tuple(
        _StreamPlan(
            stream_id=stream_id,
            features=tuple(feature_configs[stream_id]),
            targets=tuple(target_configs[stream_id]),
        )
        for stream_id in stream_ids
    )


def _ordered_projected_rows(
    runtime: Runtime,
    plans: Sequence[_StreamPlan],
    sample_keys: SampleKeyContract,
    cadence: timedelta,
) -> Iterator[_ProjectedRow]:
    sort_progress = SortProgress()
    pipeline = Pipeline(
        name="series:artifact",
        input=Input(
            name="project_streams",
            open=partial(
                _project_streams,
                runtime,
                plans,
                sample_keys,
                cadence,
            ),
        ),
        stages=(
            Stage(
                name="order_series",
                apply=partial(
                    batch_sort,
                    buffer_bytes=runtime.execution.sort_buffer_bytes,
                    key=lambda row: (row.key, row.time),
                    progress=sort_progress,
                ),
                progress=sort_progress.snapshot,
            ),
        ),
    )
    return run_pipeline(runtime, pipeline)


def _project_streams(
    runtime: Runtime,
    plans: Sequence[_StreamPlan],
    sample_keys: SampleKeyContract,
    cadence: timedelta,
) -> Iterator[_ProjectedRow]:
    for plan in plans:
        yield from _project_stream(runtime, plan, sample_keys, cadence)


def _project_stream(
    runtime: Runtime,
    plan: _StreamPlan,
    sample_keys: SampleKeyContract,
    cadence: timedelta,
) -> Iterator[_ProjectedRow]:
    stream = require_runtime_stream(runtime, plan.stream_id)
    configs = (*plan.features, *plan.targets)
    feature_ids = {config.id for config in plan.features}
    projector = SeriesProjector(stream.partition_by, sample_keys, configs)
    sequencers = {
        config.id: SeriesSequencer(config.sequence)
        for config in configs
        if config.sequence is not None
    }

    def project(records: Iterator[Any]) -> Iterator[_ProjectedRow]:
        for record in records:
            features: list[_ProjectedValue] = []
            targets: list[_ProjectedValue] = []
            row_key: tuple[Any, ...] | None = None
            row_time: datetime | None = None
            sample_time: datetime | None = None

            for config, projected in zip(
                configs,
                projector.project(record),
                strict=True,
            ):
                sequencer = sequencers.get(config.id)
                result = projected if sequencer is None else sequencer.append(projected)
                if result is None:
                    continue

                if sample_time is None:
                    sample_time = floor_time_to_cadence(result.time, cadence)
                key = (sample_time, *result.entity_key)
                if row_key is None:
                    row_key = key
                    row_time = result.time
                elif key != row_key or result.time != row_time:
                    raise RuntimeError(
                        f"Stream '{plan.stream_id}' projected one record into different "
                        "sample keys."
                    )

                value: _ProjectedValue
                if isinstance(result, SeriesSequence):
                    value = _ProjectedSequence(
                        result.id,
                        result.values,
                        record_establishes_domain(result),
                    )
                else:
                    value = _ProjectedScalar(
                        result.id,
                        result.value,
                        record_establishes_domain(result),
                    )
                if config.id in feature_ids:
                    features.append(value)
                else:
                    targets.append(value)

            if row_key is not None and row_time is not None:
                yield _ProjectedRow(
                    key=row_key,
                    time=row_time,
                    features=tuple(features),
                    targets=tuple(targets),
                )

    record_pipeline = build_stream_pipeline(runtime, plan.stream_id)
    pipeline = Pipeline(
        name=f"series:{plan.stream_id}",
        input=record_pipeline.input,
        stages=(
            *record_pipeline.stages,
            Stage(name="project_series", apply=project),
        ),
        summary=record_pipeline.summary,
    )
    projected = run_pipeline(runtime, pipeline)
    try:
        yield from projected
    finally:
        _close_iterator(projected)


def _group_series_rows(
    projected: Iterator[_ProjectedRow],
    feature_configs: Sequence[SeriesConfig],
    target_configs: Sequence[SeriesConfig],
    feature_counts: Counter[str],
    target_counts: Counter[str],
) -> Iterator[SeriesRow]:
    feature_order = {config.id: index for index, config in enumerate(feature_configs)}
    target_order = {config.id: index for index, config in enumerate(target_configs)}
    collection_sizes = {
        config.id: config.collect
        for config in (*feature_configs, *target_configs)
        if config.collect is not None
    }

    for key, group in groupby(projected, key=lambda row: row.key):
        feature_records: list[_ProjectedValue] = []
        target_records: list[_ProjectedValue] = []
        for row in group:
            feature_records.extend(row.features)
            target_records.extend(row.targets)

        features, feature_placeholders = _assemble_values(
            feature_records,
            feature_order,
            collection_sizes,
        )
        targets, target_placeholders = _assemble_values(
            target_records,
            target_order,
            collection_sizes,
        )
        feature_counts.update({base_id(series_id) for series_id in features})
        target_counts.update({base_id(series_id) for series_id in targets})
        yield SeriesRow(
            time=key[0],
            entity_key=tuple(key[1:]),
            features=features,
            targets=targets,
            placeholder_ids=feature_placeholders | target_placeholders,
        )


def _assemble_values(
    records: Iterable[_ProjectedValue],
    config_order: dict[str, int],
    collection_sizes: dict[str, int],
) -> tuple[dict[str, Any], frozenset[str]]:
    values_by_id: dict[str, Any] = {}
    collections: dict[str, list[Any]] = {}
    established: set[str] = set()
    for record in records:
        if isinstance(record, _ProjectedSequence):
            if record.id in values_by_id:
                raise ValueError(
                    f"Series {record.id!r} emits multiple sequences in one "
                    "sample cadence bucket; increase sequence stride or use a "
                    "finer dataset sample cadence."
                )
            values_by_id[record.id] = list(record.values)
        elif base_id(record.id) in collection_sizes:
            if isinstance(record.value, list):
                raise TypeError(f"Series {record.id!r} collect requires scalar values.")
            collections.setdefault(record.id, []).append(record.value)
        else:
            if record.id in values_by_id:
                raise ValueError(
                    f"Series {record.id!r} emits multiple values in one sample "
                    "cadence bucket; configure collect explicitly or use a finer "
                    "dataset sample cadence."
                )
            values_by_id[record.id] = record.value
        if record.establishes_domain:
            established.add(record.id)

    for series_id, values in collections.items():
        required = collection_sizes[base_id(series_id)]
        if len(values) != required:
            raise ValueError(
                f"Series {series_id!r} collect requires {required} values in each "
                f"populated sample cadence bucket; got {len(values)}."
            )
        values_by_id[series_id] = values

    ordered_ids = sorted(
        values_by_id,
        key=lambda series_id: (config_order[base_id(series_id)], series_id),
    )
    assembled = {series_id: values_by_id[series_id] for series_id in ordered_ids}
    return assembled, frozenset(values_by_id.keys() - established)


def _remove_failed_generation(generation_root: Path) -> None:
    try:
        shutil.rmtree(generation_root)
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning(
            "Failed to remove incomplete series generation %s",
            generation_root,
            exc_info=True,
        )


def _close_iterator(items: Iterator[object]) -> None:
    closer = getattr(items, "close", None)
    if callable(closer):
        closer()
