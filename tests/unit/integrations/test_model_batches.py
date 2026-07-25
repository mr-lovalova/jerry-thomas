from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

import datapipeline.integrations.ml as ml
from datapipeline.artifacts.models import (
    FoldedMetadataLayout,
    FoldOutputMetadata,
    ListVectorMetadataEntry,
    ScalarVectorMetadataEntry,
    VectorMetadata,
    VectorMetadataCatalog,
    VectorMetadataCounts,
    VectorMetadataEntry,
    VectorMetadataFold,
    VectorSchema,
)
from datapipeline.artifacts.specs import VECTOR_METADATA
from datapipeline.config.dataset.dataset import DatasetConfig, SampleConfig
from datapipeline.config.dataset.series import SeriesConfig, TargetSeriesConfig
from datapipeline.config.dataset.split import (
    DatasetFold,
    TimeInterval,
    TimeSplitConfig,
)
from datapipeline.domain.sample import Sample
from datapipeline.domain.vector import Vector
from datapipeline.pipelines.dataset.pipeline import FoldOutputPlan
from datapipeline.runtime import Runtime


def test_iter_samples_closes_dataset_pipeline_after_partial_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed = False

    def sample_stream() -> Iterator[Sample]:
        nonlocal closed
        try:
            yield _sample(("first",), {"value": 1.0})
            yield _sample(("second",), {"value": 2.0})
        finally:
            closed = True

    source = ml._SampleSource(
        _runtime(tmp_path, _dataset()),
        None,
        _schema((_scalar("value"),)),
        None,
    )
    _use_source(monkeypatch, source)
    monkeypatch.setattr(
        ml,
        "run_dataset_pipeline",
        lambda *_args, **_kwargs: sample_stream(),
    )

    samples = ml.iter_samples("project.yaml")
    assert next(samples) == _sample(("first",), {"value": 1.0})
    samples.close()

    assert closed


def test_iter_samples_runs_selected_fold_and_hydrates_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _dataset(split=_split(), scale=True)
    runtime = _runtime(tmp_path, dataset)
    _register_folded_metadata(runtime)
    definition = SimpleNamespace(dataset=dataset)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(ml, "load_project_definition", lambda _path: definition)
    monkeypatch.setattr(ml, "compile_runtime", lambda _definition: runtime)
    monkeypatch.setattr(
        ml,
        "hydrate_runtime_artifacts_for_pipeline",
        lambda actual_runtime, actual_definition: calls.append(
            ("hydrate", (actual_runtime, actual_definition))
        ),
    )

    def run_fold(
        _context: object,
        output: FoldOutputPlan,
    ) -> Iterator[Sample]:
        calls.append(
            (
                "fold",
                (
                    output.fold_id,
                    tuple(output.outputs),
                    tuple(entry.id for entry in output.schema.features),
                ),
            )
        )
        yield _sample(("selected",), {"value": -1.0})

    monkeypatch.setattr(ml, "run_fold_dataset_pipeline", run_fold)

    assert list(ml.iter_samples("project.yaml", output_id="walk_1.train")) == [
        _sample(("selected",), {"value": -1.0})
    ]
    assert calls == [
        ("hydrate", (runtime, definition)),
        ("fold", ("walk_1", ("walk_1.train",), ("value",))),
    ]


def test_iter_samples_uses_scaler_for_unsplit_scaled_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _dataset(scale=True)
    source = ml._SampleSource(
        _runtime(tmp_path, dataset),
        None,
        _schema((_scalar("value"),)),
        None,
    )
    _use_source(monkeypatch, source)
    calls: list[str] = []

    def run_scaled(*_args: object, **_kwargs: object) -> Iterator[Sample]:
        calls.append("scaled")
        yield _sample(("sample",), {"value": 0.0})

    monkeypatch.setattr(ml, "run_scaled_dataset_pipeline", run_scaled)

    assert list(ml.iter_samples("project.yaml")) == [
        _sample(("sample",), {"value": 0.0})
    ]
    assert calls == ["scaled"]


def test_iter_samples_validates_split_output_before_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_definition = SimpleNamespace(dataset=_dataset(split=_split()))
    monkeypatch.setattr(
        ml,
        "load_project_definition",
        lambda _path: split_definition,
    )

    with pytest.raises(
        ValueError,
        match=(
            "pass output_id with one of: "
            "walk_0.train, walk_0.validation, "
            "walk_1.train, walk_1.validation"
        ),
    ):
        ml.iter_samples("project.yaml")

    with pytest.raises(
        ValueError,
        match=(
            "Dataset output 'walk_2.train' is not defined; choose one of: "
            "walk_0.train, walk_0.validation, "
            "walk_1.train, walk_1.validation"
        ),
    ):
        ml.iter_samples("project.yaml", output_id="walk_2.train")

    unsplit_definition = SimpleNamespace(dataset=_dataset())
    monkeypatch.setattr(
        ml,
        "load_project_definition",
        lambda _path: unsplit_definition,
    )
    with pytest.raises(
        ValueError,
        match="output_id is only valid when dataset.split is configured",
    ):
        ml.iter_samples("project.yaml", output_id="walk_0.train")


def test_model_batches_are_lazy_bounded_and_metadata_ordered(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    consumed = 0
    closed = False
    samples = (
        _sample(
            (f"sample-{index}",),
            {
                "history": [index + 0.25, index + 0.5],
                "price": index,
            },
            {"label": index + 1},
        )
        for index in range(5)
    )

    def sample_stream() -> Iterator[Sample]:
        nonlocal consumed, closed
        try:
            for sample in samples:
                consumed += 1
                yield sample
        finally:
            closed = True

    feature_entries = (_scalar("price"), _sequence("history", 2))
    target_entries = (_scalar("label"),)
    _install_batch_pipeline(
        monkeypatch,
        tmp_path,
        sample_stream,
        feature_entries,
        target_entries,
    )

    batches = ml.iter_model_batches(
        "project.yaml",
        batch_size=2,
        limit=3,
        dtype="float64",
    )
    first = next(batches)

    assert consumed == 2
    assert not closed
    assert first.keys == (("sample-0",), ("sample-1",))
    assert first.feature_columns == ("price", "history[0]", "history[1]")
    assert first.target_columns == ("label",)
    assert first.features.dtype == np.float64
    assert first.targets is not None
    assert first.targets.dtype == np.float64
    np.testing.assert_array_equal(
        first.features,
        [[0.0, 0.25, 0.5], [1.0, 1.25, 1.5]],
    )
    np.testing.assert_array_equal(first.targets, [[1.0], [2.0]])

    [final] = list(batches)
    assert final.keys == (("sample-2",),)
    np.testing.assert_array_equal(final.features, [[2.0, 2.25, 2.5]])
    np.testing.assert_array_equal(final.targets, [[3.0]])
    assert consumed == 3
    assert closed


def test_model_batches_preserve_feature_and_target_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_batch_pipeline(
        monkeypatch,
        tmp_path,
        lambda: iter([_sample(("row",), {"feature": 1.0}, {"target": 2.0})]),
        (_scalar("feature"),),
        (_scalar("target"),),
    )

    [batch] = ml.iter_model_batches(
        "project.yaml",
        batch_size=1,
        dtype="float32",
    )

    assert batch.feature_columns == ("feature",)
    assert batch.target_columns == ("target",)
    assert batch.targets is not None
    np.testing.assert_array_equal(batch.features, [[1.0]])
    np.testing.assert_array_equal(batch.targets, [[2.0]])


def test_model_batches_use_none_for_feature_only_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_batch_pipeline(
        monkeypatch,
        tmp_path,
        lambda: iter([_sample(("row",), {"value": 1})]),
        (_scalar("value"),),
    )

    [batch] = ml.iter_model_batches(
        "project.yaml",
        batch_size=4,
        dtype="float32",
    )

    assert batch.features.dtype == np.float32
    assert batch.targets is None
    assert batch.target_columns == ()


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (None, ValueError, r"history\[1\].*missing.*sample \('row',\)"),
        (True, TypeError, r"history\[1\].*sample \('row',\).*numeric"),
        ("bad", TypeError, r"history\[1\].*sample \('row',\).*numeric"),
        (float("inf"), ValueError, r"history\[1\].*sample \('row',\).*finite"),
    ],
)
def test_model_batches_reject_invalid_values_with_column_and_sample(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    closed = False

    def sample_stream() -> Iterator[Sample]:
        nonlocal closed
        try:
            yield _sample(("row",), {"history": [1.0, value]})
        finally:
            closed = True

    _install_batch_pipeline(
        monkeypatch,
        tmp_path,
        sample_stream,
        (_sequence("history", 2),),
    )

    with pytest.raises(error, match=message):
        list(
            ml.iter_model_batches(
                "project.yaml",
                batch_size=1,
                dtype="float64",
            )
        )
    assert closed


def test_model_batches_reject_flattened_column_collisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_batch_pipeline(
        monkeypatch,
        tmp_path,
        lambda: iter(()),
        (_scalar("history[0]"), _sequence("history", 1)),
    )

    with pytest.raises(ValueError, match="features produce duplicate model columns"):
        list(ml.iter_model_batches("project.yaml"))


def test_model_batches_reject_values_not_representable_by_dtype(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_batch_pipeline(
        monkeypatch,
        tmp_path,
        lambda: iter([_sample(("row",), {"value": 1.0e100})]),
        (_scalar("value"),),
    )

    with pytest.raises(
        ValueError,
        match=r"column 'value'.*sample \('row',\).*represented as float32",
    ):
        list(
            ml.iter_model_batches(
                "project.yaml",
                batch_size=1,
                dtype="float32",
            )
        )


def test_closing_model_batches_closes_dataset_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed = False

    def sample_stream() -> Iterator[Sample]:
        nonlocal closed
        try:
            yield _sample(("first",), {"value": 1.0})
            yield _sample(("second",), {"value": 2.0})
        finally:
            closed = True

    _install_batch_pipeline(
        monkeypatch,
        tmp_path,
        sample_stream,
        (_scalar("value"),),
    )

    batches = ml.iter_model_batches(
        "project.yaml",
        batch_size=1,
        dtype="float32",
    )
    next(batches)
    batches.close()

    assert closed


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"batch_size": 0, "dtype": "float32"},
            "batch_size must be a positive integer",
        ),
        (
            {"batch_size": True, "dtype": "float32"},
            "batch_size must be a positive integer",
        ),
        (
            {"batch_size": 1.5, "dtype": "float32"},
            "batch_size must be a positive integer",
        ),
        (
            {"batch_size": float("nan"), "dtype": "float32"},
            "batch_size must be a positive integer",
        ),
        ({"batch_size": 1, "dtype": "float16"}, "dtype must be"),
    ],
)
def test_model_batches_validate_batch_contract(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        list(ml.iter_model_batches("project.yaml", **kwargs))


def _install_batch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_stream: Callable[[], Iterator[Sample]],
    feature_entries: Sequence[VectorMetadataEntry],
    target_entries: Sequence[VectorMetadataEntry] = (),
) -> None:
    dataset = _dataset(with_targets=bool(target_entries))
    schema = _schema(tuple(feature_entries), tuple(target_entries))
    source = ml._SampleSource(_runtime(tmp_path, dataset), None, schema, None)
    _use_source(monkeypatch, source)

    def run_dataset(*_args: object, **_kwargs: object) -> Iterator[Sample]:
        yield from sample_stream()

    monkeypatch.setattr(ml, "run_dataset_pipeline", run_dataset)


def _use_source(
    monkeypatch: pytest.MonkeyPatch,
    source: ml._SampleSource,
) -> None:
    monkeypatch.setattr(
        ml._SampleSource,
        "from_project",
        classmethod(lambda _cls, _project, _output: source),
    )


def _sample(
    key: tuple[Any, ...],
    features: dict[str, Any],
    targets: dict[str, Any] | None = None,
) -> Sample:
    return Sample(
        key=key,
        features=Vector(features),
        targets=None if targets is None else Vector(targets),
    )


def _scalar(identifier: str) -> VectorMetadataEntry:
    return ScalarVectorMetadataEntry(
        id=identifier,
        base_id=identifier,
        kind="scalar",
        present_count=5,
        null_count=0,
    )


def _sequence(identifier: str, length: int) -> VectorMetadataEntry:
    return ListVectorMetadataEntry(
        id=identifier,
        base_id=identifier,
        kind="list",
        present_count=5,
        null_count=0,
        length=length,
        observed_elements=5 * length,
    )


def _schema(
    features: tuple[VectorMetadataEntry, ...],
    targets: tuple[VectorMetadataEntry, ...] = (),
) -> VectorSchema:
    return VectorSchema(
        features=features,
        targets=targets,
        counts=VectorMetadataCounts(
            feature_vectors=5 if features else 0,
            target_vectors=5 if targets else 0,
        ),
    )


def _register_folded_metadata(runtime: Runtime) -> None:
    schema = _schema((_scalar("value"),))
    metadata = VectorMetadata(
        schema_version=4,
        catalog=VectorMetadataCatalog.model_validate(schema.model_dump()),
        layout=FoldedMetadataLayout(
            kind="folded",
            folds=(
                VectorMetadataFold(
                    id="walk_0",
                    training_schema=schema,
                    outputs=(
                        FoldOutputMetadata(role="train", labels=("train_0",)),
                        FoldOutputMetadata(
                            role="validation",
                            labels=("validation_0",),
                        ),
                    ),
                ),
                VectorMetadataFold(
                    id="walk_1",
                    training_schema=schema,
                    outputs=(
                        FoldOutputMetadata(
                            role="train",
                            labels=("train_0", "validation_0", "train_1"),
                        ),
                        FoldOutputMetadata(
                            role="validation",
                            labels=("validation_1",),
                        ),
                    ),
                ),
            ),
        ),
    )
    runtime.artifacts_root.mkdir(parents=True, exist_ok=True)
    path = runtime.artifacts_root / "metadata.json"
    path.write_text(metadata.model_dump_json(), encoding="utf-8")
    runtime.artifacts.register(VECTOR_METADATA, path.name)


def _dataset(
    *,
    split: TimeSplitConfig | None = None,
    scale: bool = False,
    with_targets: bool = False,
) -> DatasetConfig:
    return DatasetConfig(
        sample=SampleConfig(cadence="1d"),
        features=[
            SeriesConfig(
                id="value",
                stream="records",
                field="value",
                scale=scale,
            )
        ],
        targets=(
            [
                TargetSeriesConfig(
                    id="label",
                    stream="labels",
                    field="label",
                    horizon="0s",
                )
            ]
            if with_targets
            else []
        ),
        split=split,
    )


def _split() -> TimeSplitConfig:
    return TimeSplitConfig(
        intervals=[
            TimeInterval(id="train_0", until="2024-01-02T00:00:00Z"),
            TimeInterval(id="validation_0", until="2024-01-03T00:00:00Z"),
            TimeInterval(id="train_1", until="2024-01-04T00:00:00Z"),
            TimeInterval(id="validation_1"),
        ],
        folds=[
            DatasetFold(
                id="walk_0",
                train=["train_0"],
                validation=["validation_0"],
            ),
            DatasetFold(
                id="walk_1",
                train=["train_0", "validation_0", "train_1"],
                validation=["validation_1"],
            ),
        ],
    )


def _runtime(tmp_path: Path, dataset: DatasetConfig) -> Runtime:
    return Runtime(
        project_yaml=tmp_path / "project.yaml",
        artifacts_root=tmp_path / "artifacts",
        dataset=dataset,
    )
