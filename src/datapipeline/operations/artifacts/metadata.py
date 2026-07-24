from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path

from datapipeline.artifacts.models import (
    SampleDomainEntry,
    SampleMetadata,
    VectorMetadata,
    VectorMetadataCounts,
    VECTOR_METADATA_VERSION,
    Window,
    WindowMode,
)
from datapipeline.artifacts.series import SeriesRow, load_series_manifest, open_series
from datapipeline.artifacts.specs import SERIES
from datapipeline.config.dataset.dataset import DatasetConfig
from datapipeline.config.dataset.split import DatasetFold, TimeSplitConfig
from datapipeline.config.tasks import MetadataTask
from datapipeline.domain.series_id import SERIES_ID_SEPARATOR, base_id
from datapipeline.execution.observability import OperationProgressTracker
from datapipeline.execution.runner import resolve_heartbeat_interval_seconds
from datapipeline.operations.persistence import ArtifactOutput
from datapipeline.pipelines.dataset.split import TargetHorizonPolicy, build_labeler
from datapipeline.runtime import Runtime, require_runtime_stream
from datapipeline.utils.json_artifact import write_json_artifact
from datapipeline.utils.time import count_cadence_buckets, parse_cadence

from .utils import (
    metadata_entries_from_stats,
    VectorMetadataCollector,
    VectorMetadataStats,
)


ObservedRange = tuple[datetime, datetime]


def _requires_fold_schema_validation(runtime: Runtime) -> bool:
    sample_keys = set(runtime.dataset.sample.keys)
    for config in runtime.dataset.series:
        stream = require_runtime_stream(runtime, config.stream)
        if any(field not in sample_keys for field in stream.partition_by):
            return True
    return False


def _observe_partitioned_values(
    values: Mapping[str, object],
    observed_at: datetime,
    stats: dict[str, VectorMetadataStats],
) -> None:
    for series_id, value in values.items():
        if SERIES_ID_SEPARATOR not in series_id:
            continue
        entry = stats.get(series_id)
        if entry is None:
            entry = stats[series_id] = VectorMetadataStats(
                id=series_id,
                base_id=base_id(series_id),
            )
        entry.observe(value, observed_at)


def _same_vector_schema(
    complete: VectorMetadataStats,
    training: VectorMetadataStats,
) -> bool:
    if complete.kind != training.kind:
        return False
    if complete.kind == "list":
        return (
            complete.list_length == training.list_length
            and complete.element_types - {"null"}
            == training.element_types - {"null"}
        )
    return complete.scalar_types == training.scalar_types


def _incompatible_series_ids(
    complete: Mapping[str, VectorMetadataStats],
    training: Mapping[str, VectorMetadataStats],
) -> list[str]:
    incompatible = []
    for series_id, complete_stats in complete.items():
        training_stats = training.get(series_id)
        if training_stats is None or not _same_vector_schema(
            complete_stats,
            training_stats,
        ):
            incompatible.append(series_id)
    return sorted(incompatible)


class _FoldTrainingSchemas:
    def __init__(self, dataset: DatasetConfig) -> None:
        split = dataset.split
        assert split is not None

        self._folds = tuple(split.folds)
        self._labeler = build_labeler(split)
        self._folds_by_label: dict[str, list[DatasetFold]] = defaultdict(list)
        self._feature_stats: dict[str, dict[str, VectorMetadataStats]] = {
            fold.id: {} for fold in self._folds
        }
        self._target_stats: dict[str, dict[str, VectorMetadataStats]] = {
            fold.id: {} for fold in self._folds
        }
        self._horizon_policies: dict[str, TargetHorizonPolicy] = {}

        for fold in self._folds:
            for label in fold.train:
                self._folds_by_label[label].append(fold)

        horizon = dataset.max_target_horizon
        if isinstance(split, TimeSplitConfig) and horizon > timedelta():
            self._horizon_policies = {
                fold.id: TargetHorizonPolicy(split, fold, horizon)
                for fold in self._folds
            }

    def observe(self, row: SeriesRow) -> None:
        label = self._labeler.label(row.key)
        for fold in self._folds_by_label.get(label, ()):
            horizon_policy = self._horizon_policies.get(fold.id)
            if horizon_policy is not None and not horizon_policy.allows(
                label,
                row.key,
            ):
                continue
            _observe_partitioned_values(
                row.features,
                row.time,
                self._feature_stats[fold.id],
            )
            _observe_partitioned_values(
                row.targets,
                row.time,
                self._target_stats[fold.id],
            )

    def validate(
        self,
        feature_stats: Sequence[VectorMetadataStats],
        target_stats: Sequence[VectorMetadataStats],
    ) -> None:
        partitioned_features = {
            entry.id: entry
            for entry in feature_stats
            if SERIES_ID_SEPARATOR in entry.id
        }
        partitioned_targets = {
            entry.id: entry for entry in target_stats if SERIES_ID_SEPARATOR in entry.id
        }
        for fold in self._folds:
            invalid_features = _incompatible_series_ids(
                partitioned_features,
                self._feature_stats[fold.id],
            )
            if invalid_features:
                raise RuntimeError(
                    f"Dataset fold {fold.id!r} training data does not establish "
                    "partition-derived feature series IDs: "
                    + ", ".join(invalid_features)
                    + ". Put the partition fields in sample.keys for long output "
                    "or ensure every wide series ID, shape, and value type is "
                    "present in training."
                )

            invalid_targets = _incompatible_series_ids(
                partitioned_targets,
                self._target_stats[fold.id],
            )
            if invalid_targets:
                raise RuntimeError(
                    f"Dataset fold {fold.id!r} training data does not establish "
                    "partition-derived target series IDs: "
                    + ", ".join(invalid_targets)
                    + ". Put the partition fields in sample.keys for long output "
                    "or ensure every wide series ID, shape, and value type is "
                    "present in training."
                )


def _collapsed_ranges(
    grouped: dict[str, list[ObservedRange]],
) -> list[ObservedRange]:
    return [
        (
            min(start for start, _ in values),
            max(end for _, end in values),
        )
        for values in grouped.values()
    ]


def _base_ranges(entries: Sequence[VectorMetadataStats]) -> list[ObservedRange]:
    grouped: dict[str, list[ObservedRange]] = defaultdict(list)
    for entry in entries:
        start = entry.first_observed
        end = entry.last_observed
        if start is None or end is None:
            continue
        grouped[entry.base_id].append((start, end))
    return _collapsed_ranges(grouped)


def _partition_ranges(
    entries: Sequence[VectorMetadataStats],
) -> list[ObservedRange]:
    grouped: dict[str, list[ObservedRange]] = defaultdict(list)
    for entry in entries:
        start = entry.first_observed
        end = entry.last_observed
        if start is None or end is None:
            continue
        grouped[entry.id].append((start, end))
    return _collapsed_ranges(grouped)


def _range_union(
    ranges: Sequence[ObservedRange],
) -> tuple[datetime | None, datetime | None]:
    if not ranges:
        return None, None
    start = min(r[0] for r in ranges)
    end = max(r[1] for r in ranges)
    return start, end


def _range_intersection(
    ranges: Sequence[ObservedRange],
) -> tuple[datetime | None, datetime | None]:
    if not ranges:
        return None, None
    start = max(r[0] for r in ranges)
    end = min(r[1] for r in ranges)
    if start > end:
        return None, None
    return start, end


def _window_bounds_from_stats(
    feature_stats: Sequence[VectorMetadataStats],
    target_stats: Sequence[VectorMetadataStats],
    mode: WindowMode,
) -> tuple[datetime | None, datetime | None]:
    base_ranges = _base_ranges(feature_stats) + _base_ranges(target_stats)
    partition_ranges = _partition_ranges(feature_stats) + _partition_ranges(
        target_stats
    )

    if mode == "union":
        return _range_union(base_ranges if base_ranges else partition_ranges)
    if mode == "intersection":
        return _range_intersection(base_ranges)
    if mode == "strict":
        return _range_intersection(partition_ranges)
    raise ValueError(f"Unsupported metadata window mode {mode!r}.")


def _merge_sample_domains(
    feature_domain: dict[tuple, tuple[datetime, datetime]],
    target_domain: dict[tuple, tuple[datetime, datetime]],
    mode: WindowMode,
) -> dict[tuple, tuple[datetime, datetime]]:
    if mode not in {"union", "intersection", "strict"}:
        raise ValueError(f"Unsupported metadata window mode {mode!r}.")
    if not target_domain:
        return dict(feature_domain)
    if mode in {"intersection", "strict"}:
        merged: dict[tuple, tuple[datetime, datetime]] = {}
        for key, feature_range in feature_domain.items():
            target_range = target_domain.get(key)
            if target_range is None:
                continue
            start = max(feature_range[0], target_range[0])
            end = min(feature_range[1], target_range[1])
            if start <= end:
                merged[key] = (start, end)
        return merged

    merged = dict(feature_domain)
    for key, target_range in target_domain.items():
        existing_range = merged.get(key)
        if existing_range is None:
            merged[key] = target_range
            continue
        merged[key] = (
            min(existing_range[0], target_range[0]),
            max(existing_range[1], target_range[1]),
        )
    return merged


def _sample_metadata(
    cadence: str,
    sample_keys: list[str],
    domain: dict[tuple, tuple[datetime, datetime]],
) -> SampleMetadata:
    return SampleMetadata(
        cadence=cadence,
        keys=sample_keys,
        domain=[
            SampleDomainEntry(key=list(key), start=start, end=end)
            for key, (start, end) in sorted(domain.items())
        ],
    )


def materialize_metadata(
    runtime: Runtime,
    task_cfg: MetadataTask,
) -> ArtifactOutput:
    dataset = runtime.dataset
    artifact = runtime.artifacts.optional(SERIES)
    if artifact is None:
        raise RuntimeError("Series artifact is required before metadata.")
    manifest_path = artifact.resolve(runtime.artifacts.root)
    manifest = load_series_manifest(manifest_path)
    if manifest.cadence != dataset.sample.cadence or manifest.sample_keys != tuple(
        dataset.sample.keys
    ):
        raise RuntimeError(
            "Series artifact sample configuration does not match the dataset."
        )
    feature_collector = VectorMetadataCollector(
        dataset.features,
        dataset.sample.keys,
    )
    target_collector = VectorMetadataCollector(
        dataset.targets,
        dataset.sample.keys,
    )
    fold_schemas = (
        _FoldTrainingSchemas(dataset)
        if dataset.split is not None and _requires_fold_schema_validation(runtime)
        else None
    )
    progress = OperationProgressTracker(
        "scan_series",
        "samples",
        resolve_heartbeat_interval_seconds(runtime.heartbeat_interval_seconds),
    )
    rows = open_series(manifest_path, manifest)
    try:
        for row in rows:
            feature_collector.observe(row.key, row.features)
            target_collector.observe(row.key, row.targets)
            if fold_schemas is not None:
                fold_schemas.observe(row)
            progress.advance()
    finally:
        closer = getattr(rows, "close", None)
        if callable(closer):
            closer()

    feature_stats, feature_vectors, feature_domain = feature_collector.result()
    target_stats, target_vectors, target_domain = target_collector.result()
    if fold_schemas is not None:
        fold_schemas.validate(feature_stats, target_stats)
    feature_meta = metadata_entries_from_stats(feature_stats)
    target_meta = metadata_entries_from_stats(target_stats)

    window_obj: Window | None = None
    computed_start, computed_end = _window_bounds_from_stats(
        feature_stats,
        target_stats,
        mode=task_cfg.window_mode,
    )
    start = computed_start
    end = computed_end
    if start is not None and end is not None:
        size = count_cadence_buckets(start, end, parse_cadence(dataset.sample.cadence))
        window_obj = Window(start=start, end=end, mode=task_cfg.window_mode, size=size)
    sample_domain = _merge_sample_domains(
        feature_domain,
        target_domain,
        task_cfg.window_mode,
    )
    sample_meta = None
    if dataset.sample.keys:
        sample_meta = _sample_metadata(
            dataset.sample.cadence,
            dataset.sample.keys,
            sample_domain,
        )

    doc = VectorMetadata(
        schema_version=VECTOR_METADATA_VERSION,
        features=feature_meta,
        targets=target_meta,
        counts=VectorMetadataCounts(
            feature_vectors=feature_vectors,
            target_vectors=target_vectors,
        ),
        window=window_obj,
        sample=sample_meta,
    )

    relative_path = Path(task_cfg.output)
    destination = (runtime.artifacts_root / relative_path).resolve()
    write_json_artifact(
        destination,
        doc.model_dump(mode="json", exclude_none=True),
    )

    meta: dict[str, object] = {
        "features": len(feature_meta),
        "targets": len(target_meta),
    }
    return ArtifactOutput(relative_path=str(relative_path), meta=meta)
