"""Exact scaler moments fitted from projected stream records.

The series build fits these as a by-product of its single pass over every stream
and stores them beside the series data. The scaler build reuses them when they
were fitted for the current dataset settings and otherwise refits from streams.
Both paths feed records through ``StreamScalerFitter``, so they agree exactly.
"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from jerrythomas.artifacts.scaler import FoldedScalerArtifact, StandardScalerArtifact
from jerrythomas.artifacts.specs import scaler_fit_inputs
from jerrythomas.config.dataset.dataset import DatasetConfig
from jerrythomas.config.dataset.series import ScalingConfig
from jerrythomas.config.dataset.split import TimeSplitConfig
from jerrythomas.domain.series import SeriesRecord
from jerrythomas.io.json_file import write_json_object
from jerrythomas.pipelines.dataset.split import TargetHorizonPolicy, build_labeler
from jerrythomas.transforms.utils import record_establishes_domain
from jerrythomas.transforms.vector.scaler import ScalerAccumulator

# Fold ID of each fitted scaler; None is the single scaler of an unsplit dataset.
FoldId = str | None
ScalerFits = dict[FoldId, "ScalerFit"]

# Increment when stored fits change meaning without a dataset configuration change.
SCALER_FIT_FORMAT = 1


@dataclass
class ScalerFit:
    accumulator: ScalerAccumulator = field(default_factory=ScalerAccumulator)
    expected_ids: set[str] = field(default_factory=set)
    training_ids: set[str] = field(default_factory=set)

    def extend(self, other: "ScalerFit") -> None:
        self.accumulator.extend(other.accumulator)
        self.expected_ids.update(other.expected_ids)
        self.training_ids.update(other.training_ids)


def fold_ids(dataset: DatasetConfig) -> tuple[FoldId, ...]:
    if dataset.split is None:
        return (None,)
    return tuple(fold.id for fold in dataset.split.folds)


def scaler_fit_identity(dataset: DatasetConfig) -> str:
    payload = json.dumps(
        {"format": SCALER_FIT_FORMAT, "inputs": scaler_fit_inputs(dataset)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class _FoldState:
    fit: ScalerFit = field(default_factory=ScalerFit)
    training_domain: set[tuple[tuple, str]] = field(default_factory=set)
    training_labels: frozenset[str] | None = None
    horizon_policy: TargetHorizonPolicy | None = None

    def observe(
        self,
        group_key: tuple,
        records: Sequence[SeriesRecord],
        label: str | None,
    ) -> None:
        if self.horizon_policy is not None:
            assert label is not None
            if not self.horizon_policy.allows(label, group_key):
                return
        fit = self.fit
        genuine_ids = {
            record.id for record in records if record_establishes_domain(record)
        }
        fit.expected_ids.update(genuine_ids)
        if self.training_labels is not None and label not in self.training_labels:
            return

        fit.training_ids.update(genuine_ids)
        entity_key = group_key[1:]
        self.training_domain.update(
            (entity_key, series_id) for series_id in genuine_ids
        )
        # A series contributes once a genuine training record established it.
        for record in records:
            if (entity_key, record.id) in self.training_domain:
                fit.accumulator.observe(record.id, record.value)


class StreamScalerFitter:
    """Fit every fold's scaler from one stream's records, observed in stream order."""

    def __init__(self, dataset: DatasetConfig) -> None:
        self.abandoned = False
        split = dataset.split
        if split is None:
            self._labeler = None
            self._states: dict[FoldId, _FoldState] = {None: _FoldState()}
            self._states_by_label: dict[str, list[_FoldState]] = {}
            return

        target_horizon = dataset.max_target_horizon
        time_split = split if isinstance(split, TimeSplitConfig) else None
        self._labeler = build_labeler(split)
        self._states = {}
        self._states_by_label = defaultdict(list)
        for fold in split.folds:
            state = _FoldState(
                training_labels=frozenset(fold.train),
                horizon_policy=(
                    TargetHorizonPolicy(time_split, fold, target_horizon)
                    if time_split is not None and target_horizon > timedelta()
                    else None
                ),
            )
            self._states[fold.id] = state
            for label in (*fold.train, *fold.validation, *fold.test):
                self._states_by_label[label].append(state)

    def observe(self, group_key: tuple, records: Sequence[SeriesRecord]) -> None:
        """Observe one stream record projected into its scaled series."""
        if self._labeler is None:
            self._states[None].observe(group_key, records, None)
            return
        label = self._labeler.label(group_key)
        for state in self._states_by_label.get(label, ()):
            state.observe(group_key, records, label)

    def observe_or_abandon(
        self, group_key: tuple, records: Sequence[SeriesRecord]
    ) -> None:
        """Observe during the series pass, where a fitting error must not fail series.

        After an error this stream has no fits; the scaler then refits from
        streams and reports the error itself.
        """
        if self.abandoned:
            return
        try:
            self.observe(group_key, records)
        except Exception:
            self.abandoned = True

    def fits(self) -> ScalerFits:
        if self.abandoned:
            raise RuntimeError("Scaler fitting was abandoned for this stream.")
        return {fold_id: state.fit for fold_id, state in self._states.items()}


def merge_fits(dataset: DatasetConfig, stream_fits: Iterable[ScalerFits]) -> ScalerFits:
    """Combine per-stream fits; streams own disjoint series, so moments stay exact."""
    merged: ScalerFits = {fold_id: ScalerFit() for fold_id in fold_ids(dataset)}
    for fits in stream_fits:
        for fold_id, fit in fits.items():
            merged[fold_id].extend(fit)
    return merged


def finish_scaler_artifact(
    dataset: DatasetConfig,
    fits: Mapping[FoldId, ScalerFit],
) -> StandardScalerArtifact | FoldedScalerArtifact:
    settings = {
        config.id: config.scale for config in dataset.series if config.scale is not None
    }
    if dataset.split is None:
        fit = fits[None]
        return _finish_scaler(fit.accumulator, fit.expected_ids, "dataset", settings)
    return FoldedScalerArtifact(
        folds={
            fold.id: _finish_fold(fits[fold.id], f"dataset fold {fold.id!r}", settings)
            for fold in dataset.split.folds
        }
    )


def _finish_fold(
    fit: ScalerFit,
    scope: str,
    settings: Mapping[str, ScalingConfig],
) -> StandardScalerArtifact:
    untrained_ids = fit.expected_ids - fit.training_ids
    if untrained_ids:
        raise RuntimeError(
            f"Scaler fitting has no training observations for {scope} vector IDs: "
            + ", ".join(sorted(untrained_ids))
        )
    return _finish_scaler(fit.accumulator, fit.training_ids, scope, settings)


def _finish_scaler(
    accumulator: ScalerAccumulator,
    expected_ids: set[str],
    scope: str,
    settings: Mapping[str, ScalingConfig],
) -> StandardScalerArtifact:
    if not expected_ids or accumulator.observations == 0:
        raise RuntimeError(f"Scaler fitting produced no observations for {scope}.")
    artifact = accumulator.artifact(settings)
    missing = expected_ids - artifact.scalers.keys()
    if missing:
        raise RuntimeError(
            f"Scaler fitting has no training observations for {scope} vector IDs: "
            + ", ".join(sorted(missing))
        )
    scalers = {
        vector_id: artifact.scalers[vector_id] for vector_id in sorted(expected_ids)
    }
    return StandardScalerArtifact(
        observations=sum(entry.statistics.count for entry in scalers.values()),
        scalers=scalers,
    )


def write_scaler_fits(
    path: Path, identity: str, fits: Mapping[FoldId, ScalerFit]
) -> None:
    write_json_object(
        path,
        {
            "format": SCALER_FIT_FORMAT,
            "identity": identity,
            "folds": [
                {
                    "fold": fold_id,
                    "expected_ids": sorted(fit.expected_ids),
                    "training_ids": sorted(fit.training_ids),
                    "accumulator": fit.accumulator.to_json(),
                }
                for fold_id, fit in fits.items()
            ],
        },
    )


def read_scaler_fits(path: Path) -> tuple[str, ScalerFits]:
    """Return the stored fit identity and fits; raise ValueError when malformed."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != SCALER_FIT_FORMAT:
        raise ValueError(f"Unsupported scaler fit format in {path}.")
    identity = data.get("identity")
    folds = data.get("folds")
    if not isinstance(identity, str) or not isinstance(folds, list):
        raise ValueError(f"Malformed scaler fits in {path}.")
    fits: ScalerFits = {}
    for entry in folds:
        if not isinstance(entry, dict):
            raise ValueError(f"Malformed scaler fits in {path}.")
        fold_id = entry.get("fold")
        if fold_id is not None and not isinstance(fold_id, str):
            raise ValueError(f"Malformed scaler fit fold in {path}.")
        fits[fold_id] = ScalerFit(
            accumulator=ScalerAccumulator.from_json(entry.get("accumulator")),
            expected_ids=_series_ids(entry.get("expected_ids"), path),
            training_ids=_series_ids(entry.get("training_ids"), path),
        )
    return identity, fits


def _series_ids(values: object, path: Path) -> set[str]:
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ValueError(f"Malformed scaler fit IDs in {path}.")
    return set(values)
