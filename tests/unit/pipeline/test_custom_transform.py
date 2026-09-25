from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import pytest

from jerrythomas.artifacts.specs import SERIES
from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.dataset.series import SeriesConfig
from jerrythomas.config.dataset.split import (
    DatasetFold,
    HashSplitConfig,
    TimeInterval,
    TimeSplitConfig,
)
from jerrythomas.config.interpolation import MissingInterpolation
from jerrythomas.config.streams import SourceStreamConfig, StreamsConfig
from jerrythomas.config.tasks.metadata import MetadataTask
from jerrythomas.config.tasks.series import SeriesTask
from jerrythomas.config.transforms import (
    CustomTransformConfig,
    EnsureCadenceConfig,
    WhereConfig,
)
from jerrythomas.domain.record import TemporalRecord
from jerrythomas.operations.artifacts.metadata import build_metadata_artifact
from jerrythomas.operations.artifacts.series import build_series_artifact
from jerrythomas.pipelines.stream.pipeline import run_stream_pipeline
from jerrythomas.pipelines.stream.stages import build_transform_stages
from jerrythomas.runtime import Runtime, SourceRuntimeStream
from jerrythomas.services.dataset import validate_dataset_streams
from jerrythomas.services.project_definition import load_project_definition
from jerrythomas.services.runtime_compiler import compile_runtime
from jerrythomas.services.streams.validation import validate_stream_configs
from jerrythomas.transforms.scoped import PartitionScopedTransform
from jerrythomas.transforms.utils import clone_record


def test_custom_transform_config_defaults() -> None:
    config = CustomTransformConfig.model_validate(
        {"entrypoint": "my_plugin.transform:Factory"}
    )

    assert config.operation == "custom"
    assert config.args == {}
    assert config.writes == ()


def test_custom_transform_config_requires_entrypoint() -> None:
    with pytest.raises(ValueError, match="entrypoint"):
        CustomTransformConfig.model_validate({})


def test_custom_transform_writes_accept_yaml_lists() -> None:
    config = CustomTransformConfig.model_validate(
        {"entrypoint": "x:y", "writes": ["a", "b"]}
    )

    assert config.writes == ("a", "b")


class _Record:
    def __init__(self, time: datetime, partition: str, value: float) -> None:
        self.time = time
        self.partition = partition
        self.value = value


def _record(hour: int, partition: str = "A", value: float = 0.0) -> _Record:
    return _Record(datetime(2025, 1, 1, hour, tzinfo=UTC), partition, value)


class _RunningTotal(PartitionScopedTransform):
    """Stateful plugin transform: emits the running total as a new field."""

    def process_partition(self, records: Iterator) -> Iterator:
        total = 0.0
        for record in records:
            total += getattr(record, self.args["field"])
            record.total = total
            yield record


def _runtime(tmp_path: Path) -> Runtime:
    return Runtime(
        project_yaml=tmp_path,
        artifacts_root=tmp_path,
        dataset=None,
    )


def test_custom_transform_stage_scopes_state_per_partition(monkeypatch) -> None:
    monkeypatch.setattr(
        "jerrythomas.pipelines.stream.stages.load_entrypoint",
        lambda group, name: _RunningTotal,
    )
    config = CustomTransformConfig.model_validate(
        {
            "entrypoint": "my_plugin.transform:RunningTotal",
            "args": {"field": "value"},
        }
    )
    [stage] = build_transform_stages(
        _runtime(Path(".")),
        (config,),
        ("partition",),
    )

    rows = list(
        stage.apply(
            iter(
                [
                    _record(1, "A", 1.0),
                    _record(2, "A", 2.0),
                    _record(3, "B", 10.0),
                    _record(4, "B", 20.0),
                ]
            )
        )
    )

    assert [row.total for row in rows] == [1.0, 3.0, 10.0, 30.0]


def test_custom_transform_clone_keeps_placeholders_out_of_training_domain(
    monkeypatch,
    tmp_path: Path,
) -> None:
    @dataclass
    class PriceRecord(TemporalRecord):
        entity: str
        value: float | None

    class Prices:
        def stream(self) -> Iterator[PriceRecord]:
            for entity, hour, value in (
                ("BASE", 1, 10.0),
                ("BASE", 3, 30.0),
                ("HOLDOUT", 0, 1.0),
                ("HOLDOUT", 3, 3.0),
            ):
                yield PriceRecord(datetime(2024, 1, 1, hour, tzinfo=UTC), entity, value)

    class DoubleValues(PartitionScopedTransform):
        def process_partition(self, records):
            for record in records:
                value = None if record.value is None else record.value * 2
                yield clone_record(record, value=value)

    monkeypatch.setattr(
        "jerrythomas.pipelines.stream.stages.load_entrypoint",
        lambda group, name: DoubleValues,
    )
    runtime = Runtime(
        project_yaml=tmp_path / "project.yaml",
        artifacts_root=tmp_path,
        dataset=DatasetConfig(
            sample=SampleConfig(rounding="ceil", cadence="1h", keys=["entity"]),
            features=[SeriesConfig(id="price", stream="prices", field="value")],
            split=TimeSplitConfig(
                intervals=[
                    TimeInterval(id="train", until="2024-01-01T03:00:00Z"),
                    TimeInterval(id="validation"),
                ],
                folds=[
                    DatasetFold(id="fold", train=["train"], validation=["validation"])
                ],
            ),
        ),
    )
    runtime.streams["prices"] = SourceRuntimeStream(
        source=Prices(),
        mapper=lambda records: records,
        preprocess=(),
        partition_by=("entity",),
        presorted=True,
        transforms=(
            EnsureCadenceConfig(cadence="1h"),
            WhereConfig(field="time", operator="ge", comparand="2024-01-01T01:00:00Z"),
            CustomTransformConfig(entrypoint="double_values", writes=("value",)),
        ),
    )

    series_task = SeriesTask()
    result = build_series_artifact(runtime, series_task)
    runtime.artifacts.register(SERIES, series_task.path, meta=result.meta)
    build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))

    payload = json.loads((tmp_path / "metadata.json").read_text())
    fold = payload["layout"]["folds"][0]
    training = next(output for output in fold["outputs"] if output["role"] == "train")
    assert training["sample"]["domain"] == [
        {
            "key": ["BASE"],
            "start": "2024-01-01T01:00:00Z",
            "end": "2024-01-01T02:00:00Z",
        }
    ]
    validation = next(
        output for output in fold["outputs"] if output["role"] == "validation"
    )
    assert [entry["key"] for entry in validation["sample"]["domain"]] == [
        ["BASE"],
        ["HOLDOUT"],
    ]


def test_custom_transform_stage_resolves_interpolated_args(monkeypatch) -> None:
    captured: dict = {}

    class _Capturing(PartitionScopedTransform):
        def __init__(self, args, partition_by):
            super().__init__(args, partition_by)
            captured.update(args)

        def process_partition(self, records):
            return iter(())

    def factory(args, partition_by):
        instance = _Capturing(args, partition_by)
        args.clear()
        return instance

    monkeypatch.setattr(
        "jerrythomas.pipelines.stream.stages.load_entrypoint",
        lambda group, name: factory,
    )
    config = CustomTransformConfig.model_validate(
        {
            "entrypoint": "my_plugin.transform:Factory",
            "args": {"field": MissingInterpolation("window")},
        }
    )
    build_transform_stages(_runtime(Path(".")), (config,), ("partition",))

    assert captured == {"field": None}
    assert config.args == {"field": None}
    assert config.model_dump(mode="json")["args"] == {"field": None}


def test_custom_transform_stage_rejects_objects_without_apply(monkeypatch) -> None:
    class _NoApply:
        def __init__(self, args, partition_by):
            pass

    monkeypatch.setattr(
        "jerrythomas.pipelines.stream.stages.load_entrypoint",
        lambda group, name: _NoApply,
    )
    config = CustomTransformConfig.model_validate({"entrypoint": "my_plugin:x"})

    with pytest.raises(TypeError, match="must return an object with an apply"):
        build_transform_stages(_runtime(Path(".")), (config,), ("partition",))


def test_duplicate_custom_transforms_get_distinct_stage_names(monkeypatch) -> None:
    monkeypatch.setattr(
        "jerrythomas.pipelines.stream.stages.load_entrypoint",
        lambda group, name: _RunningTotal,
    )
    stages = build_transform_stages(
        _runtime(Path(".")),
        (
            CustomTransformConfig(entrypoint="a:a"),
            CustomTransformConfig(entrypoint="b:b"),
        ),
        ("partition",),
    )

    assert [stage.name for stage in stages] == ["custom_1", "custom_2"]


def _raw_source():
    from jerrythomas.config.sources import SourceConfig

    return SourceConfig.model_validate(
        {
            "id": "raw",
            "parser": {"entrypoint": "core.record"},
            "loader": {
                "transport": "fs",
                "path": "data/raw.jsonl",
                "reader": {"format": "jsonl"},
            },
        }
    )


def test_stream_validation_rejects_writes_to_canonical_fields() -> None:
    source = SourceStreamConfig.model_validate(
        {
            "id": "prices",
            "from": {"source": "raw"},
            "map": {"entrypoint": "core.identity"},
            "partition_by": ["ticker"],
        }
    )
    sources = {"raw": _raw_source()}
    streams = StreamsConfig(streams={"prices": source})

    benign = source.model_copy(
        update={
            "transforms": (
                CustomTransformConfig.model_validate(
                    {"entrypoint": "x:y", "writes": ("derived_value",)}
                ),
            )
        }
    )
    validate_stream_configs(sources, {**streams.streams, "prices": benign})

    hostile = source.model_copy(
        update={
            "transforms": (
                CustomTransformConfig.model_validate(
                    {"entrypoint": "x:y", "writes": ("time",)}
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="cannot write canonical order field 'time'"):
        validate_stream_configs(sources, {**streams.streams, "prices": hostile})


def _hash_split() -> HashSplitConfig:
    return HashSplitConfig(
        ratios={"train": 1.0},
        folds=[DatasetFold(id="default", train=["train"])],
    )


def test_hash_splits_reject_custom_transforms() -> None:
    stream = SourceStreamConfig.model_validate(
        {
            "id": "prices",
            "from": {"source": "raw"},
            "map": {"entrypoint": "core.identity"},
            "partition_by": ["ticker"],
            "transforms": [{"operation": "custom", "entrypoint": "x:y"}],
        }
    )
    dataset = DatasetConfig(
        sample=SampleConfig(rounding="ceil", cadence="1d", keys=[]),
        features=[SeriesConfig(stream="prices", id="close", field="close")],
        split=_hash_split(),
    )

    with pytest.raises(
        ValueError,
        match="hash splits cannot be used with cross-timestamp streams: prices",
    ):
        validate_dataset_streams(dataset, StreamsConfig(streams={"prices": stream}))


def test_custom_transform_runs_end_to_end_from_yaml(tmp_path, monkeypatch) -> None:
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(
        "\n".join(
            [
                "schema_version: 7",
                "artifact_revision: 1",
                "paths:",
                "  streams: streams",
                "  sources: sources",
                "  datasets: datasets",
                "  artifacts: artifacts",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets/default.yaml").write_text(
        "version: v1\nsample:\n  rounding: ceil\n  cadence: 1h\n", encoding="utf-8"
    )
    (tmp_path / "streams").mkdir()
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "events.source.yaml").write_text(
        """\
id: events.source
parser:
  entrypoint: core.record
loader:
  transport: fs
  path: data/events.jsonl
  reader:
    format: jsonl
""",
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "events.jsonl").write_text(
        "\n".join(
            [
                '{"time":"2025-01-01T01:00:00Z","ticker":"A","kind":"fill","value":10}',
                '{"time":"2025-01-01T02:00:00Z","ticker":"A","kind":"price","value":2}',
                '{"time":"2025-01-01T03:00:00Z","ticker":"B","kind":"fill","value":100}',
                '{"time":"2025-01-01T04:00:00Z","ticker":"B","kind":"price","value":4}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "streams" / "events.yaml").write_text(
        """\
id: events
from: {source: events.source}
partition_by: [ticker]
map: {entrypoint: core.identity}
""",
        encoding="utf-8",
    )
    (tmp_path / "streams" / "enriched.yaml").write_text(
        """\
id: enriched
from: {stream: events}
transforms:
  - operation: custom
    entrypoint: my_plugin.transform:FillThenPrice
""",
        encoding="utf-8",
    )

    class FillThenPrice(PartitionScopedTransform):
        """Carries the latest fill value forward onto price records."""

        def process_partition(self, records):
            latest_fill = None
            for record in records:
                if record.kind == "fill":
                    latest_fill = record.value
                    continue
                record.fill_value = latest_fill
                yield record

    monkeypatch.setattr(
        "jerrythomas.pipelines.stream.stages.load_entrypoint",
        lambda group, name: FillThenPrice,
    )

    runtime = compile_runtime(
        load_project_definition(project_yaml), dataset_id="default"
    )
    records = list(run_stream_pipeline(runtime, "enriched"))

    assert [(r.ticker, r.value, r.fill_value) for r in records] == [
        ("A", 2, 10),
        ("B", 4, 100),
    ]


def test_scoped_transform_rejects_missing_process_partition() -> None:
    class _Broken(PartitionScopedTransform):
        pass

    record = TemporalRecord(time=datetime(2025, 1, 1, tzinfo=UTC))
    with pytest.raises(NotImplementedError, match="_Broken must implement"):
        list(_Broken({}, ()).apply(iter([record])))


def test_scoped_transform_tolerates_time_offsets() -> None:
    """Partition grouping keys on partition fields only, not wall-clock deltas."""

    class _Echo(PartitionScopedTransform):
        def process_partition(self, records):
            yield from records

    base = datetime(2025, 1, 1, tzinfo=UTC)
    gap = timedelta(days=400)
    first = TemporalRecord(time=base)
    second = TemporalRecord(time=base + gap)

    assert list(_Echo({}, ()).apply(iter([first, second]))) == [first, second]
