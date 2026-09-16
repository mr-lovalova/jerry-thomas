import gzip
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pyarrow.parquet as parquet
import pytest

from jerrythomas.artifacts.models import (
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
from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.dataset.series import SeriesConfig, TargetSeriesConfig
from jerrythomas.config.dataset.split import DatasetFold, TimeInterval, TimeSplitConfig
from jerrythomas.config.preview import PreviewStage
from jerrythomas.domain.sample import Sample
from jerrythomas.domain.vector import Vector
from jerrythomas.io.dataset_table import DatasetTable
from jerrythomas.io.output import OutputTarget
from jerrythomas.operations.persistence import (
    DatasetTableOutput,
    RoutedDatasetTableOutput,
    RoutedRuntimeOutput,
    RuntimeOutput,
    persist_runtime_result,
)
from jerrythomas.operations.runtime.dataset import run_dataset_operation

START = datetime(2020, 1, 1, tzinfo=timezone.utc)
BOUNDARY = datetime(2021, 1, 1, tzinfo=timezone.utc)
END = datetime(2022, 1, 1, tzinfo=timezone.utc)


class _CloseTrackingIterator:
    def __init__(self, *items):
        self._items = iter(items)
        self.close_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._items)

    def close(self):
        self.close_calls += 1


def _runtime(streams=None):
    runtime = SimpleNamespace(
        observe_node_events=True,
        heartbeat_interval_seconds=None,
        streams=streams or {},
    )
    runtime.dataset = _dataset()

    def load_artifact(spec):
        assert spec.key == "metadata"
        return _metadata(split=runtime.dataset.split is not None)

    runtime.artifacts = SimpleNamespace(load=load_artifact)
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


def _dataset(
    targets: list[TargetSeriesConfig] | None = None,
    split: TimeSplitConfig | None = None,
) -> DatasetConfig:
    return DatasetConfig(
        features=[SeriesConfig(id="price", stream="prices", field="value")],
        targets=[] if targets is None else targets,
        sample=SampleConfig(rounding="ceil", cadence="1d"),
        split=split,
    )


def _preview_dataset(stream: str) -> DatasetConfig:
    return DatasetConfig(
        features=[SeriesConfig(id="price", stream=stream, field="value")],
        sample=SampleConfig(rounding="ceil", cadence="1d"),
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
    output_ids: tuple[str, ...] = (),
):
    runtime.dataset = dataset
    return run_dataset_operation(
        runtime=runtime,
        output_ids=output_ids,
        limit=None,
        output_format=target.format,
        throttle_ms=None,
        preview=preview,
    )


def test_dataset_operation_reraises_keyboard_interrupt_and_marks_run_failed(
    monkeypatch,
):
    runtime = _runtime()
    dataset = _dataset()
    runtime.dataset = dataset
    target = _target()

    monkeypatch.setattr(
        "jerrythomas.operations.runtime.dataset.run_dataset_pipeline",
        lambda *args, **kwargs: _samples(),
    )

    def _samples():
        raise KeyboardInterrupt()
        yield None

    result = run_dataset_operation(
        runtime=runtime,
        output_ids=(),
        limit=None,
        output_format=target.format,
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
        "jerrythomas.operations.runtime.dataset.run_dataset_pipeline",
        lambda *args, **kwargs: iter(()),
    )
    monkeypatch.setattr(
        "jerrythomas.operations.runtime.dataset._served_dataset_table",
        lambda *args: _parquet_table(),
    )

    result = _serve(
        runtime,
        dataset,
        _parquet_target(tmp_path / "dataset.parquet"),
        preview=None,
    )

    assert isinstance(result, DatasetTableOutput)


def test_dataset_operation_returns_split_fanout_output(monkeypatch, tmp_path):
    runtime = _runtime()
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
        "jerrythomas.operations.runtime.dataset.run_fold_outputs_pipeline",
        lambda *args, **kwargs: iter(
            (
                ("holdout.train", samples[0]),
                ("holdout.validation", samples[1]),
            )
        ),
    )

    result = _serve(
        runtime,
        dataset,
        target,
        preview=None,
        output_ids=("holdout.train", "holdout.validation"),
    )

    assert isinstance(result, RoutedRuntimeOutput)
    assert list(result.rows) == [
        ("holdout.train", samples[0]),
        ("holdout.validation", samples[1]),
    ]


def test_dataset_operation_returns_parquet_split_outputs(monkeypatch, tmp_path):
    runtime = _runtime()
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
        "jerrythomas.operations.runtime.dataset.run_fold_outputs_pipeline",
        lambda *args, **kwargs: iter(()),
    )
    monkeypatch.setattr(
        "jerrythomas.operations.runtime.dataset._served_dataset_table",
        lambda *args: _parquet_table(),
    )

    result = _serve(
        runtime,
        dataset,
        _parquet_target(tmp_path / "dataset.parquet"),
        preview=None,
        output_ids=("holdout.train", "holdout.validation"),
    )

    assert isinstance(result, RoutedDatasetTableOutput)
    assert set(result.tables) == {
        "holdout.train",
        "holdout.validation",
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
        "jerrythomas.operations.runtime.dataset.run_sample_pipeline",
        lambda *args, **kwargs: iter(["sample"]),
    )

    result = _serve(runtime, dataset, target, preview="samples")

    assert isinstance(result, RuntimeOutput)
    assert list(result.rows) == ["sample"]


@pytest.mark.parametrize("throttle_ms", [None, 1])
def test_limited_preview_closes_sample_pipeline_once(monkeypatch, throttle_ms):
    stream = _CloseTrackingIterator("first", "second")
    monkeypatch.setattr(
        "jerrythomas.operations.runtime.dataset.run_sample_pipeline",
        lambda *args, **kwargs: stream,
    )

    result = run_dataset_operation(
        runtime=_runtime(),
        output_ids=(),
        limit=1,
        output_format="jsonl",
        throttle_ms=throttle_ms,
        preview="samples",
    )

    assert isinstance(result, RuntimeOutput)
    assert list(result.rows) == ["first"]
    assert stream.close_calls == 1


def test_samples_preview_writes_schema_aware_parquet(monkeypatch, tmp_path):
    sample = Sample(
        key=(datetime(2024, 1, 1, tzinfo=timezone.utc),),
        features=Vector(values={"price": 10.0}),
    )
    monkeypatch.setattr(
        "jerrythomas.operations.runtime.dataset.run_sample_pipeline",
        lambda *args, **kwargs: iter((sample,)),
    )
    monkeypatch.setattr(
        "jerrythomas.operations.runtime.dataset._dataset_table",
        lambda *args: _parquet_table(),
    )
    target = _parquet_target(tmp_path / "samples.parquet")

    result = _serve(_runtime(), _dataset(), target, preview="samples")

    assert isinstance(result, DatasetTableOutput)
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
        "jerrythomas.operations.runtime.dataset.run_stream_preview_pipeline",
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
def test_record_previews_use_stream_preview_pipeline(monkeypatch, preview, expected):
    captured = {}

    def run_preview(runtime, stream_id, selected_preview):
        captured["stream_id"] = stream_id
        captured["preview"] = selected_preview
        return iter((expected,))

    monkeypatch.setattr(
        "jerrythomas.operations.runtime.dataset.run_stream_preview_pipeline",
        run_preview,
    )
    result = _serve(
        _runtime({"derived.prices": object()}),
        _preview_dataset("derived.prices"),
        _target(),
        preview=preview,
    )

    assert captured == {"stream_id": "derived.prices", "preview": preview}
    assert list(result.outputs["derived.prices"].rows) == [expected]


def test_series_preview_returns_processed_series(monkeypatch):
    monkeypatch.setattr(
        "jerrythomas.operations.runtime.dataset.run_series_pipeline",
        lambda *args, **kwargs: iter(["series"]),
    )

    result = _serve(
        _runtime(),
        _preview_dataset("derived.prices"),
        _target(),
        preview="series",
    )

    assert list(result.outputs["price"].rows) == ["series"]


def test_postprocess_preview_runs_postprocess(monkeypatch):
    runtime = _runtime()
    dataset = _dataset()
    target = _target()

    monkeypatch.setattr(
        "jerrythomas.operations.runtime.dataset.run_dataset_pipeline",
        lambda *args, **kwargs: iter(["post:sample"]),
    )

    result = _serve(runtime, dataset, target, preview="postprocess")

    assert isinstance(result, RuntimeOutput)
    assert list(result.rows) == ["post:sample"]


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
        "jerrythomas.operations.runtime.dataset.run_stream_preview_pipeline",
        lambda _context, _stream_id, selected_preview: iter(
            (
                {
                    "input": "source",
                    "canonical": "mapped:source",
                    "records": "records:transformed:mapped:source",
                }[selected_preview],
            )
        ),
    )
    monkeypatch.setattr(
        "jerrythomas.operations.runtime.dataset.run_series_pipeline",
        lambda *args, **kwargs: iter(["series"]),
    )
    monkeypatch.setattr(
        "jerrythomas.operations.runtime.dataset.run_sample_pipeline",
        lambda _context, _schema, _key_plan: iter(("sample",)),
    )
    monkeypatch.setattr(
        "jerrythomas.operations.runtime.dataset.run_dataset_pipeline",
        lambda _context, _schema, _key_plan: iter(("post:sample",)),
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
        output_ids=(output_id,) if output_id is not None else (),
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
