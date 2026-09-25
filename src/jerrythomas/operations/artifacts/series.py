import logging
import shutil
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import timedelta
from functools import partial
from itertools import groupby
from pathlib import Path
from typing import Any
from uuid import uuid4

from jerrythomas.artifacts.output import ArtifactOutput
from jerrythomas.artifacts.series import (
    SERIES_MANIFEST_VERSION,
    SeriesEntry,
    SeriesManifest,
    SeriesRow,
    series_cache_root,
    write_series_rows,
)
from jerrythomas.artifacts.specs import dataset_requires_scaler
from jerrythomas.config.dataset.series import SeriesConfig
from jerrythomas.config.tasks.series import SeriesTask
from jerrythomas.domain.sample_key import SampleKeyContract
from jerrythomas.domain.series_id import base_id
from jerrythomas.execution.pipeline import Input, Pipeline
from jerrythomas.execution.runner import run_pipeline
from jerrythomas.io.json_file import write_json_object
from jerrythomas.operations.artifacts.scaler_fit import (
    ScalerFits,
    scaler_fit_identity,
    write_scaler_fits,
)
from jerrythomas.operations.artifacts.series_projection import (
    ProjectedRow,
    ProjectedSequence,
    ProjectedValue,
    StreamPlan,
    stream_plans,
)
from jerrythomas.operations.artifacts.series_workers import (
    SeriesWorkerProgress,
    order_streams,
)
from jerrythomas.runtime import Runtime
from jerrythomas.services.path_policy import resolve_artifact_output_path
from jerrythomas.utils.time import parse_cadence

logger = logging.getLogger(__name__)

SCALER_FITS_FILENAME = "scaler_fits.json"


def build_series_artifact(
    runtime: Runtime,
    task_cfg: SeriesTask,
) -> ArtifactOutput:
    dataset = runtime.require_dataset()
    relative_path = Path(task_cfg.path)
    destination = resolve_artifact_output_path(relative_path, runtime.artifacts_root)
    cache_root = series_cache_root(destination)
    if cache_root.is_symlink():
        raise ValueError("Series data directory must not be a symlink.")

    generation = uuid4().hex
    staging_root = cache_root / f".staging-{generation}"
    generation_root = cache_root / generation
    data_path = staging_root / "series.jsonl.gz"
    sample_keys = SampleKeyContract(dataset.sample.keys)
    feature_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    # The scaler reuses fits from this pass instead of reading its streams again.
    scaler_fits: list[ScalerFits] = []
    fit_scaler = dataset_requires_scaler(dataset)

    try:
        staging_root.mkdir(parents=True)
        projected = _ordered_projected_rows(
            runtime,
            stream_plans(dataset.features, dataset.targets),
            sample_keys,
            parse_cadence(dataset.sample.cadence),
            scaler_fits.append if fit_scaler else None,
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

        fits_path: Path | None = None
        if scaler_fits:
            (fits,) = scaler_fits
            fits_path = staging_root / SCALER_FITS_FILENAME
            write_scaler_fits(fits_path, scaler_fit_identity(dataset), fits)

        generation_path = Path(cache_root.name) / generation
        manifest = SeriesManifest(
            version=SERIES_MANIFEST_VERSION,
            cadence=dataset.sample.cadence,
            rounding=dataset.sample.rounding,
            sample_keys=tuple(dataset.sample.keys),
            sample_key_types=sample_keys.types,
            path=str(generation_path / data_path.name),
            scaler_fits=(
                str(generation_path / fits_path.name) if fits_path is not None else None
            ),
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

    companions = (manifest.path, manifest.scaler_fits)
    return ArtifactOutput(
        companion_paths=tuple(
            str(relative_path.parent / companion)
            for companion in companions
            if companion is not None
        ),
        meta={
            "features": len(manifest.features),
            "targets": len(manifest.targets),
            "rows": manifest.rows,
            "format": manifest.format,
        },
    )


def _ordered_projected_rows(
    runtime: Runtime,
    plans: Sequence[StreamPlan],
    sample_keys: SampleKeyContract,
    cadence: timedelta,
    on_scaler_fits: Callable[[ScalerFits], None] | None,
) -> Iterator[ProjectedRow]:
    progress = SeriesWorkerProgress()
    return run_pipeline(
        runtime,
        Pipeline(
            name="series:artifact",
            input=Input(
                name="order_series",
                open=partial(
                    order_streams,
                    runtime,
                    plans,
                    sample_keys,
                    cadence,
                    progress,
                    on_scaler_fits,
                ),
                progress=progress.snapshot,
            ),
        ),
    )


def _group_series_rows(
    projected: Iterator[ProjectedRow],
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
        feature_records: list[ProjectedValue] = []
        target_records: list[ProjectedValue] = []
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
    records: Iterable[ProjectedValue],
    config_order: dict[str, int],
    collection_sizes: dict[str, int],
) -> tuple[dict[str, Any], frozenset[str]]:
    values_by_id: dict[str, Any] = {}
    collections: dict[str, list[Any]] = {}
    established: set[str] = set()
    for record in records:
        if isinstance(record, ProjectedSequence):
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
