from collections.abc import Collection, Iterator, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from datapipeline.artifacts.series import (
    SeriesManifest,
    SeriesRow,
    load_series_manifest,
    open_series,
)
from datapipeline.artifacts.specs import SERIES
from datapipeline.domain.sample import Sample
from datapipeline.domain.series_id import base_id
from datapipeline.domain.vector import Vector
from datapipeline.execution.context import PipelineContext
from datapipeline.execution.events import ProgressSnapshot
from datapipeline.execution.pipeline import Input
from datapipeline.pipelines.sample.keys import RectangularKeyPlan


def open_samples(
    context: PipelineContext,
    feature_ids: Collection[str],
    group_by_cadence: str,
    target_ids: Collection[str] = (),
    sample_keys: Sequence[str] = (),
    key_plan: RectangularKeyPlan | None = None,
) -> Iterator[Sample]:
    selected_feature_ids = frozenset(feature_ids)
    selected_target_ids = frozenset(target_ids)
    sample_key_fields = tuple(sample_keys)
    if not selected_feature_ids and not selected_target_ids:
        return iter(())

    manifest_path, manifest = _require_series(
        context,
        group_by_cadence,
        sample_key_fields,
    )
    return _samples_from_series(
        manifest_path,
        manifest,
        selected_feature_ids,
        selected_target_ids,
        key_plan,
    )


def build_sample_input(
    context: PipelineContext,
    feature_ids: Collection[str],
    target_ids: Collection[str] = (),
    key_plan: RectangularKeyPlan | None = None,
) -> Input:
    sample = context.runtime.dataset.sample
    selected_feature_ids = frozenset(feature_ids)
    selected_target_ids = frozenset(target_ids)
    progress = None
    if key_plan is not None:
        progress = partial(
            ProgressSnapshot,
            total=key_plan.total,
            unit="samples",
        )
    return Input(
        name="assemble_samples",
        open=partial(
            open_samples,
            context,
            selected_feature_ids,
            sample.cadence,
            selected_target_ids,
            tuple(sample.keys),
            key_plan,
        ),
        progress=progress,
    )


def _require_series(
    context: PipelineContext,
    group_by_cadence: str,
    sample_keys: Sequence[str],
) -> tuple[Path, SeriesManifest]:
    artifact = context.runtime.artifacts.optional(SERIES)
    if artifact is None:
        raise RuntimeError(
            "Series artifact is required before sample assembly. "
            "Run `jerry build --profile series` or use "
            "`--artifact-mode AUTO|FORCE`."
        )

    manifest_path = artifact.resolve(context.runtime.artifacts.root)
    manifest = load_series_manifest(manifest_path)
    if manifest.cadence != group_by_cadence:
        raise RuntimeError(
            "Series artifact cadence does not match requested pipeline cadence: "
            f"{manifest.cadence!r} != {group_by_cadence!r}."
        )
    if manifest.sample_keys != tuple(sample_keys):
        raise RuntimeError(
            "Series artifact sample keys do not match requested pipeline sample keys."
        )
    return manifest_path, manifest


def _samples_from_series(
    manifest_path: Path,
    manifest: SeriesManifest,
    feature_ids: frozenset[str],
    target_ids: frozenset[str],
    key_plan: RectangularKeyPlan | None,
) -> Iterator[Sample]:
    feature_base_ids = {base_id(series_id) for series_id in feature_ids}
    available_feature_ids = {entry.id for entry in manifest.features}
    missing_features = sorted(feature_base_ids - available_feature_ids)
    if missing_features:
        raise RuntimeError(
            "Series artifact does not contain configured feature ids: "
            + ", ".join(missing_features)
        )

    target_base_ids = {base_id(series_id) for series_id in target_ids}
    available_target_ids = {entry.id for entry in manifest.targets}
    missing_targets = sorted(target_base_ids - available_target_ids)
    if missing_targets:
        raise RuntimeError(
            "Series artifact does not contain configured target ids: "
            + ", ".join(missing_targets)
        )

    rows = open_series(manifest_path, manifest)
    selected = _select_rows(
        rows,
        feature_ids,
        target_ids,
    )
    if key_plan is None:
        return _sparse_samples(selected)
    return _rectangular_samples(
        selected,
        key_plan.keys(),
        include_targets=bool(target_ids),
    )


def _select_rows(
    rows: Iterator[SeriesRow],
    feature_ids: frozenset[str],
    target_ids: frozenset[str],
) -> Iterator[tuple[tuple, dict[str, Any], dict[str, Any]]]:
    try:
        for row in rows:
            yield (
                row.key,
                _select_values(row.features, feature_ids),
                _select_values(row.targets, target_ids),
            )
    finally:
        _close_iterator(rows)


def _select_values(
    values: dict[str, Any],
    selected_ids: frozenset[str],
) -> dict[str, Any]:
    if values.keys() <= selected_ids:
        return values
    return {
        series_id: value
        for series_id, value in values.items()
        if series_id in selected_ids
    }


def _sparse_samples(
    rows: Iterator[tuple[tuple, dict[str, Any], dict[str, Any]]],
) -> Iterator[Sample]:
    try:
        for key, features, targets in rows:
            if not features:
                continue
            yield Sample(
                key=key,
                features=Vector(features),
                targets=Vector(targets) if targets else None,
            )
    finally:
        _close_iterator(rows)


def _rectangular_samples(
    rows: Iterator[tuple[tuple, dict[str, Any], dict[str, Any]]],
    keys: Iterator[tuple],
    include_targets: bool,
) -> Iterator[Sample]:
    try:
        current = next(rows, None)
        for key in keys:
            while current is not None and current[0] < key:
                current = next(rows, None)
            if current is not None and current[0] == key:
                _, features, targets = current
                current = next(rows, None)
            else:
                features = {}
                targets = {}
            yield Sample(
                key=key,
                features=Vector(features),
                targets=Vector(targets) if include_targets else None,
            )
    finally:
        _close_iterator(rows)
        _close_iterator(keys)


def _close_iterator(items: Iterator[object]) -> None:
    closer = getattr(items, "close", None)
    if callable(closer):
        closer()
