import gzip
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pyarrow.parquet as parquet
import pytest

from datapipeline.artifacts.models import (
    FoldedMetadataLayout,
    FoldOutputMetadata,
    ScalarVectorMetadataEntry,
    UnsplitMetadataLayout,
    VectorMetadata,
    VectorMetadataCatalog,
    VectorMetadataCounts,
    VectorMetadataFold,
    VectorSchema,
    Window,
)
from datapipeline.config.dataset.dataset import DatasetConfig, SampleConfig
from datapipeline.config.dataset.series import SeriesConfig, TargetSeriesConfig
from datapipeline.config.dataset.split import DatasetFold, TimeInterval, TimeSplitConfig
from datapipeline.config.preview import PreviewStage
from datapipeline.domain.sample import Sample
from datapipeline.domain.vector import Vector
from datapipeline.execution.pipeline import Input, Pipeline, Stage
from datapipeline.io.dataset_table import DatasetTable
from datapipeline.io.output import OutputTarget
from datapipeline.operations.persistence import (
    DatasetTableOutput,
    RoutedDatasetTableOutput,
    RoutedRuntimeOutput,
    persist_runtime_result,
)
from datapipeline.operations.runtime.dataset import run_dataset_operation

START = datetime(2020, 1, 1, tzinfo=timezone.utc)
BOUNDARY = datetime(2021, 1, 1, tzinfo=timezone.utc)
END = datetime(2022, 1, 1, tzinfo=timezone.utc)


def _runtime(streams=None):
    runtime = SimpleNamespace(
        pipeline_observer=None,
        observe_node_events=True,
        heartbeat_interval_seconds=None,
        output_ids=(),
        streams=streams or {},
    )
    runtime.dataset = _dataset()
    return runtime


def _schema() -> VectorSchema:
    return VectorSchema(
        features=(
            ScalarVectorMetadataEntry(
                id="price",
                base_id="price",
                kind="scalar",
                present_count=1,
                null_count=0,
                value_types=("float",),
            ),
        ),
        counts=VectorMetadataCounts(feature_vectors=1, target_vectors=0),
    )


def _window(start: datetime, end: datetime) -> Window:
    return Window(start=start, end=end, mode="union", size=1)


def _metadata(split: bool = False) -> VectorMetadata:
    schema = _schema()
    catalog = VectorMetadataCatalog(
        features=schema.features,
        targets=schema.targets,
        counts=schema.counts,
        window=_window(START, END),
    )
    layout = (
        FoldedMetadataLayout(
            kind="folded",
            folds=(
                VectorMetadataFold(
                    id="holdout",
                    training_schema=schema,
                    outputs=(
                        FoldOutputMetadata(
                            role="train",
                            labels=("train",),
                            window=_window(START, BOUNDARY),
                        ),
                        FoldOutputMetadata(
                            role="validation",
                            labels=("val",),
                            window=_window(BOUNDARY, END),
                        ),
                    ),
                ),
            ),
        )
        if split
        else UnsplitMetadataLayout(kind="unsplit")
    )
    return VectorMetadata(
        schema_version=4,
        catalog=catalog,
        layout=layout,
    )


@pytest.fixture(autouse=True)
def _register_metadata(monkeypatch) -> None:
    def require_artifact(context, spec):
        assert spec.key == "metadata"
        return _metadata(split=context.runtime.dataset.split is not None)

    monkeypatch.setattr(
        "datapipeline.execution.context.PipelineContext.require_artifact",
        require_artifact,
    )


def _dataset(
    targets: list[TargetSeriesConfig] | None = None,
    split: TimeSplitConfig | None = None,
) -> DatasetConfig:
    return DatasetConfig(
        features=[SeriesConfig(id="price", stream="prices", field="value")],
        targets=[] if targets is None else targets,
        sample=SampleConfig(cadence="1d"),
        split=split,
    )


def _preview_dataset(stream: str) -> DatasetConfig:
    return DatasetConfig(
        features=[SeriesConfig(id="price", stream=stream, field="value")],
        sample=SampleConfig(cadence="1d"),
    )


def _target():
    return OutputTarget(
        transport="stdout",
        format="jsonl",
        view="raw",
        encoding=None,
        destination=None,
        run="run-paths",
    )


def _fs_target(destination, compression=None):
    return OutputTarget(
        transport="fs",
        format="jsonl",
        view="raw",
        encoding="utf-8",
        destination=destination,
        compression=compression,
        run=None,
    )


def _parquet_target(destination):
    return OutputTarget(
        transport="fs",
        format="parquet",
        view="flat",
        encoding=None,
        destination=destination,
        run=None,
    )


def _parquet_table() -> DatasetTable:
    return DatasetTable(
        (),
        (),
        (
            ScalarVectorMetadataEntry(
                id="price",
                base_id="price",
                kind="scalar",
                present_count=1,
                null_count=0,
                value_types=("float",),
            ),
        ),
        (),
    )


def _serve(
    runtime,
    dataset,
    target,
    preview: PreviewStage | None,
):
    runtime.dataset = dataset
    return run_dataset_operation(
        runtime=runtime,
        limit=None,
        target=target,
        throttle_ms=None,
        preview=preview,
    )


def _sample_preview_pipeline():
    return Pipeline(
        name="dataset",
        input=Input(
            name="assemble_samples",
            open=lambda: iter(["sample"]),
        ),
        stages=(
            Stage(
                name="conform_features",
                apply=lambda stream: (f"post:{item}" for item in stream),
            ),
        ),
    )


def _record_preview_pipeline():
    return Pipeline(
        name="stream:prices",
        input=Input(
            name="open_source",
            open=lambda: iter(["source"]),
        ),
        stages=(
            Stage(
                name="map_records",
                apply=lambda rows: (f"mapped:{row}" for row in rows),
            ),
            Stage(
                name="floor_time",
                apply=lambda rows: (f"transformed:{row}" for row in rows),
            ),
            Stage(
                name="ensure_record_order",
                apply=lambda rows: (f"records:{row}" for row in rows),
            ),
        ),
    )


def test_dataset_operation_reraises_keyboard_interrupt_and_marks_run_failed(
    monkeypatch,
):
    runtime = _runtime()
    dataset = _dataset()
    runtime.dataset = dataset
    target = _target()

    monkeypatch.setattr(
        "datapipeline.operations.runtime.dataset.run_dataset_pipeline",
        lambda *args, **kwargs: _samples(),
    )

    def _samples():
        raise KeyboardInterrupt()
        yield None

    result = run_dataset_operation(
        runtime=runtime,
        limit=None,
        target=target,
        throttle_ms=None,
        preview=None,
    )

    with pytest.raises(KeyboardInterrupt):
        persist_runtime_result(
            result,
            target=target,
            logger=logging.getLogger(__name__),
        )


def test_dataset_operation_returns_parquet_dataset_output(monkeypatch, tmp_path):
    runtime = _runtime()
    dataset = _dataset()
    monkeypatch.setattr(
        "datapipeline.operations.runtime.dataset.run_dataset_pipeline",
        lambda *args, **kwargs: iter(()),
    )
    monkeypatch.setattr(
        "datapipeline.operations.runtime.dataset._served_dataset_table",
        lambda *args: _parquet_table(),
    )

    result = _serve(
        runtime,
        dataset,
        _parquet_target(tmp_path / "dataset.parquet"),
        preview=None,
    )

    assert isinstance(result.outputs[0], DatasetTableOutput)


def test_dataset_operation_returns_split_fanout_output(monkeypatch, tmp_path):
    runtime = SimpleNamespace(
        pipeline_observer=None,
        observe_node_events=True,
        heartbeat_interval_seconds=None,
        output_ids=("holdout.train", "holdout.validation"),
    )
    dataset = _dataset(
        split=TimeSplitConfig(
            intervals=[
                TimeInterval(id="train", until="2021-01-01T00:00:00Z"),
                TimeInterval(id="val"),
            ],
            folds=[
                DatasetFold(
                    id="holdout",
                    train=["train"],
                    validation=["val"],
                )
            ],
        )
    )
    runtime.dataset = dataset
    target = _fs_target(tmp_path / "dataset.jsonl")
    samples = [
        Sample(key="2020-01-01T00:00:00Z", features=Vector(values={"x": 1})),
        Sample(key="2022-01-01T00:00:00Z", features=Vector(values={"x": 2})),
    ]

    monkeypatch.setattr(
        "datapipeline.operations.runtime.dataset.run_fold_outputs_pipeline",
        lambda *args, **kwargs: iter(
            (
                ("holdout.train", samples[0]),
                ("holdout.validation", samples[1]),
            )
        ),
    )

    result = _serve(runtime, dataset, target, preview=None)

    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert isinstance(output, RoutedRuntimeOutput)
    assert (
        output.targets["holdout.train"].destination
        == tmp_path / "dataset.holdout.train.jsonl"
    )
    assert (
        output.targets["holdout.validation"].destination
        == tmp_path / "dataset.holdout.validation.jsonl"
    )
    assert list(output.rows) == [
        ("holdout.train", samples[0]),
        ("holdout.validation", samples[1]),
    ]


def test_dataset_operation_returns_parquet_split_outputs(monkeypatch, tmp_path):
    runtime = SimpleNamespace(
        pipeline_observer=None,
        observe_node_events=True,
        heartbeat_interval_seconds=None,
        output_ids=("holdout.train", "holdout.validation"),
    )
    dataset = _dataset(
        split=TimeSplitConfig(
            intervals=[
                TimeInterval(id="train", until="2021-01-01T00:00:00Z"),
                TimeInterval(id="val"),
            ],
            folds=[
                DatasetFold(
                    id="holdout",
                    train=["train"],
                    validation=["val"],
                )
            ],
        )
    )
    runtime.dataset = dataset
    monkeypatch.setattr(
        "datapipeline.operations.runtime.dataset.run_fold_outputs_pipeline",
        lambda *args, **kwargs: iter(()),
    )
    monkeypatch.setattr(
        "datapipeline.operations.runtime.dataset._served_dataset_table",
        lambda *args: _parquet_table(),
    )

    result = _serve(
        runtime,
        dataset,
        _parquet_target(tmp_path / "dataset.parquet"),
        preview=None,
    )

    output = result.outputs[0]
    assert isinstance(output, RoutedDatasetTableOutput)
    assert set(output.tables) == {
        "holdout.train",
        "holdout.validation",
    }
    assert {
        target.destination.name
        for target in output.targets.values()
        if target.destination is not None
    } == {
        "dataset.holdout.train.parquet",
        "dataset.holdout.validation.parquet",
    }


def test_samples_preview_stops_before_postprocess(monkeypatch):
    runtime = _runtime()
    dataset = _dataset(
        targets=[
            TargetSeriesConfig(
                id="target",
                stream="targets",
                field="value",
                horizon="0s",
            )
        ]
    )
    target = _target()
    monkeypatch.setattr(
        "datapipeline.operations.runtime.dataset.build_dataset_pipeline",
        lambda *args, **kwargs: _sample_preview_pipeline(),
    )

    result = _serve(runtime, dataset, target, preview="samples")

    assert len(result.outputs) == 1
    assert result.outputs[0].target == target
    assert list(result.outputs[0].rows) == ["sample"]


def test_samples_preview_writes_schema_aware_parquet(monkeypatch, tmp_path):
    sample = Sample(
        key=(datetime(2024, 1, 1, tzinfo=timezone.utc),),
        features=Vector(values={"price": 10.0}),
    )
    pipeline = Pipeline(
        name="dataset",
        input=Input(name="assemble_samples", open=lambda: iter((sample,))),
    )
    monkeypatch.setattr(
        "datapipeline.operations.runtime.dataset.build_dataset_pipeline",
        lambda *args, **kwargs: pipeline,
    )
    monkeypatch.setattr(
        "datapipeline.operations.runtime.dataset._assembled_dataset_table",
        lambda *args: _parquet_table(),
    )
    target = _parquet_target(tmp_path / "samples.parquet")

    result = _serve(_runtime(), _dataset(), target, preview="samples")

    assert isinstance(result.outputs[0], DatasetTableOutput)
    persist_runtime_result(
        result,
        target=target,
        logger=logging.getLogger(__name__),
    )
    table = parquet.read_table(target.destination)
    assert table.column_names == ["sample.time", "features.price"]
    assert table.column("features.price").to_pylist() == [10.0]


def test_parquet_preview_rejects_non_dataset_stage_before_opening_stream(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "datapipeline.operations.runtime.dataset.build_stream_pipeline",
        lambda *args, **kwargs: pytest.fail("record preview must not be opened"),
    )

    with pytest.raises(ValueError, match="supports only.*samples.*postprocess"):
        _serve(
            _runtime({"prices": object()}),
            _preview_dataset("prices"),
            _parquet_target(tmp_path / "records.parquet"),
            preview="records",
        )


@pytest.mark.parametrize(
    ("preview", "expected"),
    [
        ("input", "source"),
        ("canonical", "mapped:source"),
        ("records", "records:transformed:mapped:source"),
    ],
)
def test_record_previews_stop_at_the_named_stage(monkeypatch, preview, expected):
    captured = {}

    def build_pipeline(context, stream_id):
        captured["stream_id"] = stream_id
        return _record_preview_pipeline()

    monkeypatch.setattr(
        "datapipeline.operations.runtime.dataset.build_stream_pipeline",
        build_pipeline,
    )
    result = _serve(
        _runtime({"derived.prices": object()}),
        _preview_dataset("derived.prices"),
        _target(),
        preview=preview,
    )

    assert captured == {"stream_id": "derived.prices"}
    assert list(result.outputs[0].rows) == [expected]


def test_series_preview_returns_processed_series(monkeypatch):
    monkeypatch.setattr(
        "datapipeline.operations.runtime.dataset.run_series_pipeline",
        lambda *args, **kwargs: iter(["series"]),
    )

    result = _serve(
        _runtime(),
        _preview_dataset("derived.prices"),
        _target(),
        preview="series",
    )

    assert list(result.outputs[0].rows) == ["series"]


def test_series_preview_rejects_duplicate_resolved_destinations(
    monkeypatch,
    tmp_path,
) -> None:
    dataset = DatasetConfig(
        features=[
            SeriesConfig(id="a/b", stream="first", field="value"),
            SeriesConfig(id="a?b", stream="second", field="value"),
        ],
        sample=SampleConfig(cadence="1d"),
    )
    monkeypatch.setattr(
        "datapipeline.operations.runtime.dataset.run_series_pipeline",
        lambda *args, **kwargs: pytest.fail(
            "preview streams must not be opened before destinations are validated"
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"Preview outputs 'a/b' and 'a\?b' resolve to the same destination",
    ):
        _serve(
            _runtime(),
            dataset,
            _fs_target(tmp_path / "preview.jsonl"),
            preview="series",
        )

    assert not list(tmp_path.iterdir())


def test_postprocess_preview_runs_postprocess(monkeypatch):
    runtime = _runtime()
    dataset = _dataset()
    target = _target()

    monkeypatch.setattr(
        "datapipeline.operations.runtime.dataset.build_dataset_pipeline",
        lambda *args, **kwargs: _sample_preview_pipeline(),
    )

    result = _serve(runtime, dataset, target, preview="postprocess")

    assert list(result.outputs[0].rows) == ["post:sample"]


@pytest.mark.parametrize(
    ("preview", "expected", "output_id"),
    [
        ("input", "source", "derived.prices"),
        ("canonical", "mapped:source", "derived.prices"),
        ("records", "records:transformed:mapped:source", "derived.prices"),
        ("series", "series", "price"),
        ("samples", "sample", None),
        ("postprocess", "post:sample", None),
    ],
)
def test_all_preview_stages_write_gzip_through_the_shared_output_path(
    monkeypatch,
    tmp_path,
    preview,
    expected,
    output_id,
) -> None:
    monkeypatch.setattr(
        "datapipeline.operations.runtime.dataset.build_stream_pipeline",
        lambda *args, **kwargs: _record_preview_pipeline(),
    )
    monkeypatch.setattr(
        "datapipeline.operations.runtime.dataset.run_series_pipeline",
        lambda *args, **kwargs: iter(["series"]),
    )
    monkeypatch.setattr(
        "datapipeline.operations.runtime.dataset.build_dataset_pipeline",
        lambda *args, **kwargs: _sample_preview_pipeline(),
    )

    target = _fs_target(tmp_path / "preview.jsonl.gz", compression="gzip")
    result = _serve(
        _runtime({"derived.prices": object()}),
        _preview_dataset("derived.prices"),
        target,
        preview=preview,
    )
    persist_runtime_result(
        result,
        target=target,
        logger=logging.getLogger(__name__),
    )

    destination = (
        target.for_output(output_id).destination
        if output_id is not None
        else target.destination
    )
    assert destination is not None
    with gzip.open(destination, "rt", encoding="utf-8") as stream:
        assert [json.loads(line) for line in stream] == [expected]


def test_preview_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="Unsupported preview stage"):
        _serve(
            _runtime(),
            _preview_dataset("prices"),
            _target(),
            preview="unknown",  # type: ignore[arg-type]
        )
