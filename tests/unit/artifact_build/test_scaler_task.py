from dataclasses import replace
from datetime import datetime, timezone

import pytest

from jerrythomas.artifacts.scaler import (
    FoldedScalerArtifact,
    PositionalScalerStatistics,
    ScalerStatistics,
    StandardScalerArtifact,
    load_scaler_artifact,
)
from jerrythomas.config.dataset.dataset import (
    DatasetConfig,
    SampleConfig,
)
from jerrythomas.config.dataset.series import (
    ScalingConfig,
    SeriesConfig,
    SequenceConfig,
    TargetSeriesConfig,
)
from jerrythomas.config.dataset.split import (
    DatasetFold,
    HashSplitConfig,
    TimeInterval,
    TimeSplitConfig,
)
from jerrythomas.config.tasks.scaler import ScalerTask
from jerrythomas.config.transforms import EnsureCadenceConfig, ForwardFillConfig
from jerrythomas.domain.record import TemporalRecord
from jerrythomas.operations.artifacts.scaler import build_scaler_artifact
from jerrythomas.runtime import Runtime, SourceRuntimeStream
from jerrythomas.transforms.utils import set_record_domain_anchor


def _time(day: int) -> datetime:
    return datetime(2024, 1, day, tzinfo=timezone.utc)


def _record(day: int, value: object, other: object | None = None) -> TemporalRecord:
    record = TemporalRecord(time=_time(day))
    record.value = value
    record.other = other
    return record


class _CountingSource:
    def __init__(self, rows=()) -> None:
        self.rows = tuple(rows)
        self.opens = 0
        self.closes = 0

    def stream(self):
        self.opens += 1
        try:
            yield from self.rows
        finally:
            self.closes += 1


def _identity(records):
    return records


def _runtime(
    tmp_path,
    dataset: DatasetConfig | None = None,
    rows=(),
) -> Runtime:
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(
        "schema_version: 7\nartifact_revision: 1\n", encoding="utf-8"
    )
    if dataset is None:
        dataset = DatasetConfig(sample=SampleConfig(rounding="ceil", cadence="1h"))
    runtime = Runtime(
        project_yaml=project_yaml,
        artifacts_root=artifacts_root,
        dataset=dataset,
    )
    runtime.streams["stream"] = SourceRuntimeStream(
        source=_CountingSource(rows),
        mapper=_identity,
        preprocess=(),
        transforms=(),
        partition_by=(),
        presorted=False,
    )
    return runtime


def _dataset(
    *,
    cadence: str = "1h",
    sample_keys: tuple[str, ...] = (),
    scale: bool | ScalingConfig = True,
    sequence: SequenceConfig | None = None,
    split: HashSplitConfig | TimeSplitConfig | None = None,
) -> DatasetConfig:
    return DatasetConfig(
        sample=SampleConfig(rounding="ceil", cadence=cadence, keys=list(sample_keys)),
        features=[
            SeriesConfig(
                id="x",
                stream="stream",
                field="value",
                scale=scale,
                sequence=sequence,
            )
        ],
        split=split,
    )


def test_export_standard_scaler_uses_all_scalar_observations(
    tmp_path,
) -> None:
    runtime = _runtime(
        tmp_path,
        _dataset(),
        rows=[_record(1, 1.0), _record(3, 3.0)],
    )

    task = ScalerTask(path="scaler.json")
    result = build_scaler_artifact(runtime, task)

    artifact = load_scaler_artifact(runtime.artifacts_root / task.path)
    assert isinstance(artifact, StandardScalerArtifact)
    assert artifact.version == 5
    assert artifact.observations == 2
    assert artifact.scalers["x"].statistics.mean == 2.0
    assert result.meta == {
        "series": 1,
        "observations": 2,
    }


def test_export_standard_scaler_fits_intrinsic_list_positions(
    tmp_path,
) -> None:
    runtime = _runtime(
        tmp_path,
        _dataset(),
        rows=[
            _record(1, [1.0, None]),
            _record(3, [3.0, 5.0]),
        ],
    )

    task = ScalerTask(path="scaler.json")
    result = build_scaler_artifact(runtime, task)

    artifact = load_scaler_artifact(runtime.artifacts_root / task.path)
    assert isinstance(artifact, StandardScalerArtifact)
    statistics = artifact.scalers["x"].statistics
    assert isinstance(statistics, PositionalScalerStatistics)
    first, second = statistics.positions
    assert first.count == 2
    assert first.mean == 2.0
    assert first.std == 1.0
    assert second.count == 1
    assert second.mean == 5.0
    assert second.std == pytest.approx(1e-12)
    assert statistics.count == 3
    assert artifact.observations == 3
    assert result.meta == {
        "series": 1,
        "observations": 3,
    }


def test_folded_scaler_fits_list_positions_from_training_rows_only(
    tmp_path,
) -> None:
    runtime = _runtime(
        tmp_path,
        _dataset(
            cadence="1d",
            split=TimeSplitConfig(
                intervals=[
                    TimeInterval(
                        id="train",
                        until="2024-01-02T00:00:00Z",
                    ),
                    TimeInterval(id="validation"),
                ],
                folds=[
                    DatasetFold(
                        id="fold",
                        train=["train"],
                        validation=["validation"],
                    )
                ],
            ),
        ),
        rows=[
            _record(1, [1.0, 100.0]),
            _record(2, [1_000.0, 10_000.0]),
        ],
    )

    task = ScalerTask(path="scaler.json")
    build_scaler_artifact(runtime, task)

    artifact = load_scaler_artifact(runtime.artifacts_root / task.path)
    assert isinstance(artifact, FoldedScalerArtifact)
    statistics = artifact.for_fold("fold").scalers["x"].statistics
    assert isinstance(statistics, PositionalScalerStatistics)
    first, second = statistics.positions
    assert first == ScalerStatistics(mean=1.0, std=1e-12, count=1)
    assert second == ScalerStatistics(mean=100.0, std=1e-12, count=1)


def test_export_standard_scaler_persists_feature_scaling(
    tmp_path,
) -> None:
    settings = ScalingConfig(with_mean=False, with_std=False, epsilon=0.5)
    runtime = _runtime(tmp_path, _dataset(scale=settings), rows=[_record(1, 4.0)])
    task = ScalerTask(path="scaler.json")
    build_scaler_artifact(runtime, task)

    artifact = load_scaler_artifact(runtime.artifacts_root / task.path)
    assert isinstance(artifact, StandardScalerArtifact)
    assert artifact.scalers["x"].settings == settings
    assert artifact.scalers["x"].statistics.std == 0.5


@pytest.mark.parametrize("placeholder_value", [None, 100.0])
def test_standard_scaler_excludes_placeholder_only_wide_ids(
    monkeypatch,
    tmp_path,
    placeholder_value,
) -> None:
    runtime = _runtime(tmp_path, _dataset())
    runtime.streams["stream"] = replace(
        runtime.streams["stream"],
        partition_by=("bucket",),
    )
    genuine = _record(1, 1.0)
    genuine.bucket = "known"
    placeholder = _record(1, placeholder_value)
    placeholder.bucket = "future"
    set_record_domain_anchor(placeholder, False)
    monkeypatch.setattr(
        "jerrythomas.operations.artifacts.scaler.run_stream_pipeline",
        lambda *_args: iter((genuine, placeholder)),
    )

    task = ScalerTask(path="scaler.json")
    build_scaler_artifact(runtime, task)

    artifact = load_scaler_artifact(runtime.artifacts_root / task.path)
    assert isinstance(artifact, StandardScalerArtifact)
    assert artifact.observations == 1
    assert tuple(artifact.scalers) == ("x__@bucket:known",)


@pytest.mark.parametrize(
    "split",
    [
        pytest.param(None, id="standard"),
        pytest.param(
            TimeSplitConfig(
                intervals=[TimeInterval(id="train")],
                folds=[DatasetFold(id="fold", train=["train"])],
            ),
            id="folded",
        ),
    ],
)
def test_scaler_excludes_leading_placeholders_for_later_entities(
    monkeypatch,
    tmp_path,
    split,
) -> None:
    runtime = _runtime(
        tmp_path,
        _dataset(sample_keys=("bucket",), split=split),
    )
    runtime.streams["stream"] = replace(
        runtime.streams["stream"],
        partition_by=("bucket",),
    )
    genuine_a = _record(1, 1.0)
    genuine_a.bucket = "A"
    leading_placeholder_b = _record(1, 100.0)
    leading_placeholder_b.bucket = "B"
    set_record_domain_anchor(leading_placeholder_b, False)
    genuine_b = _record(2, 2.0)
    genuine_b.bucket = "B"
    trailing_placeholder_b = _record(3, 2.0)
    trailing_placeholder_b.bucket = "B"
    set_record_domain_anchor(trailing_placeholder_b, False)
    monkeypatch.setattr(
        "jerrythomas.operations.artifacts.scaler.run_stream_pipeline",
        lambda *_args: iter(
            (
                genuine_a,
                leading_placeholder_b,
                genuine_b,
                trailing_placeholder_b,
            )
        ),
    )

    task = ScalerTask(path="scaler.json")
    build_scaler_artifact(runtime, task)

    artifact = load_scaler_artifact(runtime.artifacts_root / task.path)
    scaler = artifact.for_fold("fold") if split is not None else artifact
    assert isinstance(scaler, StandardScalerArtifact)
    assert scaler.observations == 3
    assert scaler.scalers["x"].statistics.count == 3
    assert scaler.scalers["x"].statistics.mean == pytest.approx(5 / 3)


def test_scaler_fitting_observes_scalars_before_sequence(tmp_path) -> None:
    runtime = _runtime(
        tmp_path,
        _dataset(sequence=SequenceConfig(size=3)),
        rows=[_record(1, 1.0), _record(2, 3.0)],
    )

    task = ScalerTask(path="scaler.json")
    build_scaler_artifact(runtime, task)

    artifact = load_scaler_artifact(runtime.artifacts_root / task.path)
    assert isinstance(artifact, StandardScalerArtifact)
    assert artifact.observations == 2
    assert artifact.scalers["x"].statistics.mean == 2.0


def test_target_horizon_trims_pre_sequence_feature_scaler_origins(tmp_path) -> None:
    dataset = DatasetConfig(
        sample=SampleConfig(rounding="ceil", cadence="1d"),
        features=[
            SeriesConfig(
                id="x",
                stream="stream",
                field="value",
                scale=True,
                sequence=SequenceConfig(size=3),
            )
        ],
        targets=[
            TargetSeriesConfig(
                id="target",
                stream="stream",
                field="other",
                horizon="2d",
            )
        ],
        split=TimeSplitConfig(
            intervals=[
                TimeInterval(id="train", until="2024-01-05T00:00:00Z"),
                TimeInterval(id="validation"),
            ],
            folds=[
                DatasetFold(
                    id="fold",
                    train=["train"],
                    validation=["validation"],
                )
            ],
        ),
    )
    runtime = _runtime(
        tmp_path,
        dataset,
        rows=[
            _record(1, 1.0, other=10.0),
            _record(2, 3.0, other=20.0),
            _record(3, 100.0, other=30.0),
            _record(4, 200.0, other=40.0),
        ],
    )

    task = ScalerTask(path="scaler.json")
    build_scaler_artifact(runtime, task)

    artifact = load_scaler_artifact(runtime.artifacts_root / task.path)
    assert isinstance(artifact, FoldedScalerArtifact)
    scaler = artifact.for_fold("fold")
    assert scaler.observations == 2
    assert scaler.scalers["x"].statistics.mean == 2.0


def test_export_folded_scaler_uses_dataset_owned_expanding_train_roles(
    tmp_path,
) -> None:
    runtime = _runtime(
        tmp_path,
        _dataset(
            cadence="1d",
            split=TimeSplitConfig(
                intervals=[
                    TimeInterval(
                        id="train_0",
                        until="2024-01-02T00:00:00Z",
                    ),
                    TimeInterval(
                        id="validation_0",
                        until="2024-01-03T00:00:00Z",
                    ),
                    TimeInterval(
                        id="train_1",
                        until="2024-01-04T00:00:00Z",
                    ),
                    TimeInterval(id="validation_1"),
                ],
                folds=[
                    DatasetFold(
                        id="fold_0",
                        train=["train_0"],
                        validation=["validation_0"],
                    ),
                    DatasetFold(
                        id="fold_1",
                        train=["train_0", "validation_0", "train_1"],
                        validation=["validation_1"],
                    ),
                ],
            ),
        ),
        rows=[
            _record(1, 1.0),
            _record(2, 3.0),
            _record(3, 5.0),
            _record(4, 100.0),
        ],
    )

    task = ScalerTask(path="scaler.json")
    result = build_scaler_artifact(runtime, task)

    artifact = load_scaler_artifact(runtime.artifacts_root / task.path)
    assert isinstance(artifact, FoldedScalerArtifact)
    assert artifact.version == 5
    assert artifact.for_fold("fold_0").scalers["x"].statistics.mean == 1.0
    assert artifact.for_fold("fold_0").observations == 1
    assert artifact.for_fold("fold_1").scalers["x"].statistics.mean == 3.0
    assert artifact.for_fold("fold_1").observations == 3
    assert result.meta == {
        "folds": 2,
        "observations": 4,
    }


@pytest.mark.parametrize(
    ("rounding", "mean", "count"),
    [("floor", 5.5, 2), ("ceil", 1.0, 1), ("exact", None, None)],
)
def test_folded_scaler_honors_sample_rounding(tmp_path, rounding, mean, count) -> None:
    train_record = TemporalRecord(
        time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    train_record.value = 1.0
    validation_record = TemporalRecord(
        time=datetime(2024, 1, 1, 13, tzinfo=timezone.utc),
    )
    validation_record.value = 10.0
    runtime = _runtime(
        tmp_path,
        _dataset(
            cadence="1d",
            sequence=SequenceConfig(size=2),
            split=TimeSplitConfig(
                intervals=[
                    TimeInterval(
                        id="train",
                        until="2024-01-02T00:00:00Z",
                    ),
                    TimeInterval(id="validation"),
                ],
                folds=[
                    DatasetFold(
                        id="fold",
                        train=["train"],
                        validation=["validation"],
                    )
                ],
            ),
        ),
        rows=[train_record, validation_record],
    )

    task = ScalerTask(path="scaler.json")
    runtime.dataset = runtime.dataset.model_copy(
        update={
            "sample": SampleConfig(cadence="1d", rounding=rounding),
        }
    )
    if rounding == "exact":
        with pytest.raises(ValueError, match="not aligned.*rounding='exact'"):
            build_scaler_artifact(runtime, task)
        return
    build_scaler_artifact(runtime, task)

    artifact = load_scaler_artifact(runtime.artifacts_root / task.path)
    assert isinstance(artifact, FoldedScalerArtifact)
    scaler = artifact.for_fold("fold")
    assert scaler.scalers["x"].statistics.mean == mean
    assert scaler.observations == count


def test_scaler_opens_a_shared_stream_once_for_all_scaled_fields(tmp_path) -> None:
    dataset = DatasetConfig(
        sample=SampleConfig(rounding="ceil", cadence="1h"),
        features=[
            SeriesConfig(
                id="value",
                stream="stream",
                field="value",
                scale=True,
            ),
            SeriesConfig(
                id="other",
                stream="stream",
                field="other",
                scale=True,
            ),
        ],
    )
    runtime = _runtime(
        tmp_path,
        dataset,
        rows=[_record(1, 1.0, 10.0), _record(2, 3.0, 30.0)],
    )
    source = runtime.streams["stream"].source

    task = ScalerTask(path="scaler.json")
    build_scaler_artifact(runtime, task)

    artifact = load_scaler_artifact(runtime.artifacts_root / task.path)
    assert isinstance(artifact, StandardScalerArtifact)
    assert artifact.scalers["value"].statistics.mean == 2.0
    assert artifact.scalers["other"].statistics.mean == 20.0
    assert source.opens == 1
    assert source.closes == 1


def test_grouped_scaler_preserves_global_scalar_statistics_across_sample_keys(
    tmp_path,
) -> None:
    dataset = DatasetConfig(
        sample=SampleConfig(rounding="ceil", cadence="1h", keys=["security_id"]),
        features=[
            SeriesConfig(
                id="value",
                stream="stream",
                field="value",
                scale=True,
            )
        ],
    )
    records = [
        _record(2, 30.0),
        _record(1, 1.0),
        _record(2, 3.0),
        _record(1, 10.0),
    ]
    for record, security_id in zip(records, ["B", "A", "A", "B"]):
        record.security_id = security_id
    runtime = _runtime(tmp_path, dataset, rows=records)
    runtime.streams["stream"] = replace(
        runtime.streams["stream"],
        partition_by=("security_id",),
    )

    task = ScalerTask(path="scaler.json")
    build_scaler_artifact(runtime, task)

    artifact = load_scaler_artifact(runtime.artifacts_root / task.path)
    assert isinstance(artifact, StandardScalerArtifact)
    statistics = artifact.scalers["value"].statistics
    assert statistics.count == 4
    assert statistics.mean == pytest.approx(11.0)
    assert statistics.std == pytest.approx(131.5**0.5)


def test_scaler_validates_excluded_split_records_before_filtering(tmp_path) -> None:
    dataset = DatasetConfig(
        sample=SampleConfig(rounding="ceil", cadence="1h"),
        features=[
            SeriesConfig(
                id="other",
                stream="stream",
                field="other",
                scale=True,
            )
        ],
        split=TimeSplitConfig(
            intervals=[
                TimeInterval(id="train", until="2024-01-02T00:00:00Z"),
                TimeInterval(id="test"),
            ],
            folds=[DatasetFold(id="fold", train=["train"], test=["test"])],
        ),
    )
    included = _record(1, 0.0, other=1.0)
    excluded = TemporalRecord(time=_time(3))
    excluded.value = 3.0
    runtime = _runtime(tmp_path, dataset, rows=[included, excluded])
    source = runtime.streams["stream"].source
    assert isinstance(source, _CountingSource)

    with pytest.raises(KeyError, match="Record field 'other'"):
        build_scaler_artifact(
            runtime,
            ScalerTask(path="scaler.json"),
        )

    assert source.opens == 1
    assert source.closes == 1


def test_scaler_closes_shared_stream_after_invalid_value(tmp_path) -> None:
    runtime = _runtime(tmp_path, _dataset(), rows=[_record(1, "not numeric")])
    source = runtime.streams["stream"].source
    assert isinstance(source, _CountingSource)

    with pytest.raises(TypeError, match="numeric or None"):
        build_scaler_artifact(
            runtime,
            ScalerTask(path="scaler.json"),
        )

    assert source.opens == 1
    assert source.closes == 1


def test_folded_scaler_requires_training_statistics_for_all_output_vectors(
    tmp_path,
) -> None:
    train = _record(1, 1.0)
    train.bucket = "A"
    validation = _record(3, 3.0)
    validation.bucket = "B"
    runtime = _runtime(
        tmp_path,
        _dataset(
            split=TimeSplitConfig(
                intervals=[
                    TimeInterval(
                        id="train",
                        until="2024-01-02T00:00:00Z",
                    ),
                    TimeInterval(id="validation"),
                ],
                folds=[
                    DatasetFold(
                        id="fold",
                        train=["train"],
                        validation=["validation"],
                    )
                ],
            )
        ),
        rows=[train, validation],
    )
    runtime.streams["stream"] = replace(
        runtime.streams["stream"],
        partition_by=("bucket",),
    )

    with pytest.raises(
        RuntimeError,
        match=r"no training observations.*x__@bucket:B",
    ):
        build_scaler_artifact(
            runtime,
            ScalerTask(path="scaler.json"),
        )


def test_folded_scaler_includes_filled_training_placeholders(
    tmp_path,
) -> None:
    runtime = _runtime(
        tmp_path,
        _dataset(
            cadence="1d",
            split=TimeSplitConfig(
                intervals=[TimeInterval(id="train")],
                folds=[DatasetFold(id="fold", train=["train"])],
            ),
        ),
        rows=[_record(1, 1.0), _record(3, 3.0)],
    )
    runtime.streams["stream"] = replace(
        runtime.streams["stream"],
        transforms=(
            EnsureCadenceConfig(cadence="1d"),
            ForwardFillConfig(field="value"),
        ),
    )

    task = ScalerTask(path="scaler.json")
    build_scaler_artifact(runtime, task)

    artifact = load_scaler_artifact(runtime.artifacts_root / task.path)
    assert isinstance(artifact, FoldedScalerArtifact)
    scaler = artifact.for_fold("fold")
    assert scaler.observations == 3
    assert scaler.scalers["x"].statistics.count == 3
    assert scaler.scalers["x"].statistics.mean == pytest.approx(5 / 3)


def test_folded_scaler_excludes_placeholder_only_wide_ids(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime(
        tmp_path,
        _dataset(
            cadence="1d",
            split=TimeSplitConfig(
                intervals=[TimeInterval(id="train")],
                folds=[DatasetFold(id="fold", train=["train"])],
            ),
        ),
    )
    runtime.streams["stream"] = replace(
        runtime.streams["stream"],
        partition_by=("bucket",),
    )
    genuine = _record(1, 1.0)
    genuine.bucket = "known"
    placeholder = _record(1, 100.0)
    placeholder.bucket = "future"
    set_record_domain_anchor(placeholder, False)
    monkeypatch.setattr(
        "jerrythomas.operations.artifacts.scaler.run_stream_pipeline",
        lambda *_args: iter((genuine, placeholder)),
    )

    task = ScalerTask(path="scaler.json")
    build_scaler_artifact(runtime, task)

    artifact = load_scaler_artifact(runtime.artifacts_root / task.path)
    assert isinstance(artifact, FoldedScalerArtifact)
    scaler = artifact.for_fold("fold")
    assert scaler.observations == 1
    assert tuple(scaler.scalers) == ("x__@bucket:known",)


def test_export_folded_scaler_supports_hash_splits(tmp_path) -> None:
    runtime = _runtime(
        tmp_path,
        _dataset(
            split=HashSplitConfig(
                ratios={"train": 1.0},
                folds=[DatasetFold(id="fold", train=["train"])],
            )
        ),
        rows=[_record(1, 2.0), _record(2, 4.0)],
    )

    task = ScalerTask(path="scaler.json")
    build_scaler_artifact(runtime, task)

    artifact = load_scaler_artifact(runtime.artifacts_root / task.path)
    assert isinstance(artifact, FoldedScalerArtifact)
    assert artifact.for_fold("fold").scalers["x"].statistics.mean == 3.0
