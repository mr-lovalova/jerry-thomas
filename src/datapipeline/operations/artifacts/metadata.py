from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

from datapipeline.artifacts.models import (
    FoldedMetadataLayout,
    FoldOutputMetadata,
    SampleDomainEntry,
    SampleMetadata,
    UnsplitMetadataLayout,
    VectorMetadata,
    VectorMetadataCatalog,
    VectorMetadataCounts,
    VectorMetadataFold,
    VectorSchema,
    VECTOR_METADATA_VERSION,
    Window,
    WindowMode,
)
from datapipeline.artifacts.series import SeriesRow, load_series_manifest, open_series
from datapipeline.artifacts.specs import SERIES
from datapipeline.config.dataset.dataset import DatasetConfig
from datapipeline.config.dataset.series import SeriesConfig
from datapipeline.config.dataset.split import (
    FOLD_ROLES,
    DatasetFold,
    FoldRole,
    TimeSplitConfig,
)
from datapipeline.config.tasks import MetadataTask
from datapipeline.domain.series_id import base_id
from datapipeline.execution.observability import OperationProgressTracker
from datapipeline.execution.settings import resolve_heartbeat_interval_seconds
from datapipeline.operations.persistence import ArtifactOutput
from datapipeline.pipelines.dataset.split import TargetHorizonPolicy, build_labeler
from datapipeline.runtime import Runtime
from datapipeline.utils.json_artifact import write_json_artifact
from datapipeline.utils.time import (
    count_cadence_buckets,
    floor_time_to_cadence,
    parse_cadence,
    parse_datetime,
)

from .utils import (
    metadata_entries_from_stats,
    VectorMetadataStats,
)


ObservedRange = tuple[datetime, datetime]


def _observe_values(
    values: Mapping[str, object],
    observed_at: datetime,
    stats: dict[str, VectorMetadataStats],
) -> None:
    for series_id, value in values.items():
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
            and complete.element_types - {"null"} == training.element_types - {"null"}
        )
    return complete.scalar_types == training.scalar_types


def _incompatible_series_ids(
    selected: Mapping[str, VectorMetadataStats],
    training: Mapping[str, VectorMetadataStats],
) -> list[str]:
    incompatible = []
    for series_id, selected_stats in selected.items():
        training_stats = training.get(series_id)
        if training_stats is None or not _same_vector_schema(
            selected_stats,
            training_stats,
        ):
            incompatible.append(series_id)
    return sorted(incompatible)


class _VectorStats:
    def __init__(self, configs: Sequence[SeriesConfig]) -> None:
        self._config_order = {
            config.id: position for position, config in enumerate(configs)
        }
        self.stats: dict[str, VectorMetadataStats] = {}
        self.vectors = 0

    def observe(self, values: Mapping[str, object], observed_at: datetime) -> None:
        if not values:
            return
        self.vectors += 1
        _observe_values(values, observed_at, self.stats)

    def ordered(self) -> list[VectorMetadataStats]:
        return sorted(
            self.stats.values(),
            key=lambda entry: (
                self._config_order[entry.base_id],
                entry.id,
            ),
        )

    def missing_config_ids(self) -> list[str]:
        observed = {entry.base_id for entry in self.stats.values()}
        return sorted(self._config_order.keys() - observed)

    def unexpected_config_ids(self) -> list[str]:
        observed = {entry.base_id for entry in self.stats.values()}
        return sorted(observed - self._config_order.keys())


def _update_domain(
    domain: dict[tuple, ObservedRange],
    entity_key: tuple,
    observed_at: datetime,
) -> None:
    current = domain.get(entity_key)
    if current is None:
        domain[entity_key] = observed_at, observed_at
        return
    domain[entity_key] = (
        min(current[0], observed_at),
        max(
            current[1],
            observed_at,
        ),
    )


class _OutputDomains:
    def __init__(
        self,
        feature_configs: Sequence[SeriesConfig],
        target_configs: Sequence[SeriesConfig],
    ) -> None:
        self._feature_rows: dict[tuple, ObservedRange] = {}
        self._target_rows: dict[tuple, ObservedRange] = {}
        self.features = _VectorStats(feature_configs)
        self.targets = _VectorStats(target_configs)

    def observe(
        self,
        row: SeriesRow,
        feature_domain_established: bool,
        target_domain_established: bool,
        established_feature_ids: set[str],
        established_target_ids: set[str],
    ) -> None:
        entity_key = row.entity_key
        if row.features and feature_domain_established:
            _update_domain(self._feature_rows, entity_key, row.time)
            self.features.observe(
                {
                    series_id: value
                    for series_id, value in row.features.items()
                    if series_id in established_feature_ids
                },
                row.time,
            )
        if row.targets and target_domain_established:
            _update_domain(self._target_rows, entity_key, row.time)
            self.targets.observe(
                {
                    series_id: value
                    for series_id, value in row.targets.items()
                    if series_id in established_target_ids
                },
                row.time,
            )

    def merged(self, mode: WindowMode) -> dict[tuple, ObservedRange]:
        return _merge_sample_domains(
            self._feature_rows,
            self._target_rows,
            mode,
        )

    def window_bounds(
        self,
        schema: VectorSchema,
        mode: WindowMode,
    ) -> tuple[datetime | None, datetime | None]:
        features = [
            self.features.stats[entry.id]
            for entry in schema.features
            if entry.id in self.features.stats
        ]
        targets = [
            self.targets.stats[entry.id]
            for entry in schema.targets
            if entry.id in self.targets.stats
        ]
        return _window_bounds_from_stats(features, targets, mode)


class _FoldCollector:
    def __init__(self, dataset: DatasetConfig, fold: DatasetFold) -> None:
        split = dataset.split
        assert split is not None
        self.fold = fold
        self.role_by_label = {
            label: role for role in FOLD_ROLES for label in getattr(fold, role)
        }
        self.features = _VectorStats(dataset.features)
        self.targets = _VectorStats(dataset.targets)
        self.output_domains = {
            role: _OutputDomains(dataset.features, dataset.targets)
            for role in FOLD_ROLES
            if getattr(fold, role)
        }
        self._feature_domains: dict[FoldRole, set[tuple]] = {
            role: set() for role in self.output_domains
        }
        self._target_domains: dict[FoldRole, set[tuple]] = {
            role: set() for role in self.output_domains
        }
        self._feature_ids: dict[FoldRole, set[str]] = {
            role: set() for role in self.output_domains
        }
        self._target_ids: dict[FoldRole, set[str]] = {
            role: set() for role in self.output_domains
        }
        horizon = dataset.max_target_horizon
        self.horizon_policy = (
            TargetHorizonPolicy(split, fold, horizon)
            if isinstance(split, TimeSplitConfig) and horizon > timedelta()
            else None
        )

    def observe(
        self,
        row: SeriesRow,
        label: str,
    ) -> None:
        role = self.role_by_label.get(label)
        if role is None:
            return
        if self.horizon_policy is not None and not self.horizon_policy.allows(
            label,
            row.key,
        ):
            return

        features = {
            series_id: value
            for series_id, value in row.features.items()
            if series_id not in row.placeholder_ids
        }
        targets = {
            series_id: value
            for series_id, value in row.targets.items()
            if series_id not in row.placeholder_ids
        }
        role_index = FOLD_ROLES.index(role)
        if features:
            for output_role in FOLD_ROLES[role_index:]:
                domain = self._feature_domains.get(output_role)
                if domain is not None:
                    domain.add(row.entity_key)
                    self._feature_ids[output_role].update(features)
        if targets:
            for output_role in FOLD_ROLES[role_index:]:
                domain = self._target_domains.get(output_role)
                if domain is not None:
                    domain.add(row.entity_key)
                    self._target_ids[output_role].update(targets)

        self.features.observe(features, row.time)
        self.targets.observe(targets, row.time)
        self.output_domains[role].observe(
            row,
            row.entity_key in self._feature_domains[role],
            row.entity_key in self._target_domains[role],
            self._feature_ids[role],
            self._target_ids[role],
        )

    def schema(self) -> VectorSchema:
        training = self.output_domains["train"]
        missing_features = training.features.missing_config_ids()
        if missing_features:
            raise RuntimeError(
                f"Dataset fold {self.fold.id!r} training data does not establish "
                "configured feature series: " + ", ".join(missing_features) + "."
            )
        missing_targets = training.targets.missing_config_ids()
        if missing_targets:
            raise RuntimeError(
                f"Dataset fold {self.fold.id!r} training data does not establish "
                "configured target series: " + ", ".join(missing_targets) + "."
            )

        invalid_features = _incompatible_series_ids(
            self.features.stats,
            training.features.stats,
        )
        if invalid_features:
            raise RuntimeError(
                f"Dataset fold {self.fold.id!r} training data does not establish "
                "feature series IDs, shapes, and value types: "
                + ", ".join(invalid_features)
                + "."
            )
        invalid_targets = _incompatible_series_ids(
            self.targets.stats,
            training.targets.stats,
        )
        if invalid_targets:
            raise RuntimeError(
                f"Dataset fold {self.fold.id!r} training data does not establish "
                "target series IDs, shapes, and value types: "
                + ", ".join(invalid_targets)
                + "."
            )

        return VectorSchema(
            features=metadata_entries_from_stats(training.features.ordered()),
            targets=metadata_entries_from_stats(training.targets.ordered()),
            counts=VectorMetadataCounts(
                feature_vectors=training.features.vectors,
                target_vectors=training.targets.vectors,
            ),
        )


class _FoldMetadataCollectors:
    def __init__(self, dataset: DatasetConfig) -> None:
        split = dataset.split
        assert split is not None
        self.labeler = build_labeler(split)
        self.collectors = tuple(_FoldCollector(dataset, fold) for fold in split.folds)
        self.by_label: dict[str, list[_FoldCollector]] = defaultdict(list)
        for collector in self.collectors:
            for label in collector.role_by_label:
                self.by_label[label].append(collector)

    def observe(self, row: SeriesRow) -> None:
        label = self.labeler.label(row.key)
        for collector in self.by_label.get(label, ()):
            collector.observe(row, label)


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


def _window(
    start: datetime | None,
    end: datetime | None,
    cadence: str,
    mode: WindowMode,
) -> Window | None:
    if start is None or end is None:
        return None
    return Window(
        start=start,
        end=end,
        mode=mode,
        size=count_cadence_buckets(start, end, parse_cadence(cadence)),
    )


def _ceil_time_to_cadence(value: datetime, cadence: timedelta) -> datetime:
    floored = floor_time_to_cadence(value, cadence)
    return floored if floored == value else floored + cadence


def _time_role_bounds(
    split: TimeSplitConfig,
    labels: Sequence[str],
    cadence: str,
    observed_start: datetime,
    observed_end: datetime,
) -> ObservedRange:
    positions = {
        interval.id: position for position, interval in enumerate(split.intervals)
    }
    first_position = min(positions[label] for label in labels)
    last_position = max(positions[label] for label in labels)
    step = parse_cadence(cadence)

    if first_position == 0:
        start = observed_start
    else:
        previous = split.intervals[first_position - 1]
        assert previous.until is not None
        start = max(
            observed_start,
            _ceil_time_to_cadence(
                parse_datetime(previous.until).astimezone(timezone.utc),
                step,
            ),
        )

    last = split.intervals[last_position]
    if last.until is None:
        end = observed_end
    else:
        end = min(
            observed_end,
            floor_time_to_cadence(
                parse_datetime(last.until).astimezone(timezone.utc)
                - timedelta(microseconds=1),
                step,
            ),
        )
    return start, end


def _fold_output_metadata(
    dataset: DatasetConfig,
    task_cfg: MetadataTask,
    collector: _FoldCollector,
    schema: VectorSchema,
    role: FoldRole,
) -> FoldOutputMetadata:
    labels = tuple(getattr(collector.fold, role))
    output_domain = collector.output_domains[role]
    domain = output_domain.merged(task_cfg.window_mode)
    observed_start, observed_end = output_domain.window_bounds(
        schema,
        task_cfg.window_mode,
    )
    window = None
    if observed_start is not None and observed_end is not None:
        start, end = observed_start, observed_end
        split = dataset.split
        if isinstance(split, TimeSplitConfig):
            start, end = _time_role_bounds(
                split,
                labels,
                dataset.sample.cadence,
                observed_start,
                observed_end,
            )
        window = _window(
            start,
            end,
            dataset.sample.cadence,
            task_cfg.window_mode,
        )

    sample = None
    if dataset.sample.keys:
        sample = _sample_metadata(
            dataset.sample.cadence,
            dataset.sample.keys,
            domain,
        )
    return FoldOutputMetadata(
        role=role,
        labels=labels,
        window=window,
        sample=sample,
    )


def _folded_layout(
    dataset: DatasetConfig,
    task_cfg: MetadataTask,
    collectors: _FoldMetadataCollectors,
) -> FoldedMetadataLayout:
    folds: list[VectorMetadataFold] = []
    for collector in collectors.collectors:
        schema = collector.schema()
        outputs = tuple(
            _fold_output_metadata(dataset, task_cfg, collector, schema, role)
            for role in FOLD_ROLES
            if getattr(collector.fold, role)
        )
        folds.append(
            VectorMetadataFold(
                id=collector.fold.id,
                training_schema=schema,
                outputs=outputs,
            )
        )
    return FoldedMetadataLayout(
        kind="folded",
        folds=tuple(folds),
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
    catalog_domains = _OutputDomains(dataset.features, dataset.targets)
    feature_domain: set[tuple] = set()
    target_domain: set[tuple] = set()
    established_feature_ids: set[str] = set()
    established_target_ids: set[str] = set()
    fold_collectors = (
        None if dataset.split is None else _FoldMetadataCollectors(dataset)
    )
    progress = OperationProgressTracker(
        "scan_series",
        "samples",
        resolve_heartbeat_interval_seconds(runtime.heartbeat_interval_seconds),
    )
    rows = open_series(manifest_path, manifest)
    try:
        for row in rows:
            features = {
                series_id: value
                for series_id, value in row.features.items()
                if series_id not in row.placeholder_ids
            }
            targets = {
                series_id: value
                for series_id, value in row.targets.items()
                if series_id not in row.placeholder_ids
            }
            if features:
                feature_domain.add(row.entity_key)
                established_feature_ids.update(features)
            if targets:
                target_domain.add(row.entity_key)
                established_target_ids.update(targets)
            catalog_domains.observe(
                row,
                row.entity_key in feature_domain,
                row.entity_key in target_domain,
                established_feature_ids,
                established_target_ids,
            )
            if fold_collectors is not None:
                fold_collectors.observe(row)
            progress.advance()
    finally:
        closer = getattr(rows, "close", None)
        if callable(closer):
            closer()

    unexpected_ids = (
        catalog_domains.features.unexpected_config_ids()
        + catalog_domains.targets.unexpected_config_ids()
    )
    if unexpected_ids:
        raise RuntimeError(
            "Vector metadata contains IDs outside the configured vectors: "
            f"{sorted(unexpected_ids)!r}."
        )
    missing_ids = (
        catalog_domains.features.missing_config_ids()
        + catalog_domains.targets.missing_config_ids()
    )
    if missing_ids:
        raise RuntimeError(
            "Configured vectors produced no metadata: "
            f"{sorted(missing_ids)!r}. Check upstream source data and credentials."
        )

    feature_stats = catalog_domains.features.ordered()
    target_stats = catalog_domains.targets.ordered()
    feature_meta = metadata_entries_from_stats(feature_stats)
    target_meta = metadata_entries_from_stats(target_stats)

    catalog_schema = VectorSchema(
        features=feature_meta,
        targets=target_meta,
        counts=VectorMetadataCounts(
            feature_vectors=catalog_domains.features.vectors,
            target_vectors=catalog_domains.targets.vectors,
        ),
    )
    computed_start, computed_end = catalog_domains.window_bounds(
        catalog_schema,
        task_cfg.window_mode,
    )
    window_obj = _window(
        computed_start,
        computed_end,
        dataset.sample.cadence,
        task_cfg.window_mode,
    )
    sample_domain = catalog_domains.merged(task_cfg.window_mode)
    sample_meta = None
    if dataset.sample.keys:
        sample_meta = _sample_metadata(
            dataset.sample.cadence,
            dataset.sample.keys,
            sample_domain,
        )

    catalog = VectorMetadataCatalog(
        features=feature_meta,
        targets=target_meta,
        counts=VectorMetadataCounts(
            feature_vectors=catalog_domains.features.vectors,
            target_vectors=catalog_domains.targets.vectors,
        ),
        window=window_obj,
        sample=sample_meta,
    )
    layout = (
        UnsplitMetadataLayout(kind="unsplit")
        if fold_collectors is None
        else _folded_layout(dataset, task_cfg, fold_collectors)
    )
    doc = VectorMetadata(
        schema_version=VECTOR_METADATA_VERSION,
        catalog=catalog,
        layout=layout,
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
