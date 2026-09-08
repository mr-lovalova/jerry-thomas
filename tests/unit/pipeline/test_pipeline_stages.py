import json
from io import StringIO
import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from rich.console import Console
from rich.progress import Progress

import jerrythomas.operations.artifacts.series as series_operation
import jerrythomas.pipelines.stream.cross_section as cross_section_pipeline
import jerrythomas.pipelines.stream.order as stream_order
import jerrythomas.pipelines.stream.stages as stream_stages
from jerrythomas.artifacts.models import SampleDomainEntry, VectorMetadataCatalog
from jerrythomas.artifacts.registry import VECTOR_METADATA_SPEC
from jerrythomas.artifacts.specs import (
    SERIES,
    VECTOR_METADATA,
)
from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.dataset.postprocess import PostprocessConfig
from jerrythomas.config.dataset.series import (
    SeriesConfig,
    SequenceConfig,
    TargetSeriesConfig,
)
from jerrythomas.config.dataset.split import (
    DatasetFold,
    TimeInterval,
    TimeSplitConfig,
)
from jerrythomas.config.cross_section import OlsResidualConfig, RankScoreConfig
from jerrythomas.config.execution import ExecutionConfig
from jerrythomas.config.tasks.metadata import MetadataTask
from jerrythomas.config.tasks.series import SeriesTask
from jerrythomas.config.transforms import (
    EwmMeanConfig,
    EnsureCadenceConfig,
    EnsureScheduleConfig,
    FloorTimeConfig,
    LagConfig,
    PreprocessConfig,
    RollingQuantileConfig,
    TransformConfig,
)
from jerrythomas.cli.visuals.rich.progress import (
    _ExecutionProgress,
    _RichExecutionRenderer,
)
from jerrythomas.domain.series import SeriesRecord, SeriesSequence
from jerrythomas.domain.record import TemporalRecord
from jerrythomas.domain.sample import Sample
from jerrythomas.domain.vector import Vector
from jerrythomas.execution.events import (
    NodeStarted,
    PipelineEvent,
    PipelineStarted,
    ProgressSnapshot,
)
from jerrythomas.execution.observability import execution_observer
from jerrythomas.execution.pipeline import Input
from jerrythomas.execution.runner import run_pipeline
from jerrythomas.operations.artifacts.metadata import build_metadata_artifact
from jerrythomas.operations.artifacts.series import build_series_artifact
from jerrythomas.parsers.identity import IdentityParser
from jerrythomas.pipelines.dataset.postprocess import build_postprocess_plan
from jerrythomas.pipelines.dataset.pipeline import (
    build_dataset_pipeline,
    run_dataset_pipeline,
    run_sample_pipeline,
)
from jerrythomas.pipelines.series.pipeline import (
    build_series_pipeline,
    run_series_pipeline,
)
from jerrythomas.pipelines.stream.pipeline import (
    build_stream_pipeline,
    run_stream_preview_pipeline,
    run_stream_pipeline,
)
from jerrythomas.pipelines.sample import input as sample_input
from jerrythomas.pipelines.sample.keys import (
    sample_domain_key_plan,
    window_key_plan,
)
from jerrythomas.pipelines.sample.input import build_sample_input, open_samples
from jerrythomas.runtime import (
    AlignedRuntimeStream,
    AsOfRuntimeStream,
    BroadcastRuntimeStream,
    CrossSectionRuntimeStream,
    DerivedRuntimeStream,
    RecordStage,
    Runtime,
    SourceRuntimeStream,
)
from jerrythomas.sources.adapters.fs import FsFileTransport, FsGlobTransport
from jerrythomas.sources.loader import DataLoader
from jerrythomas.sources.decoders import JsonLinesDecoder
from jerrythomas.sources.source import Source
from jerrythomas.artifacts.series import (
    SeriesRow,
    load_series_manifest,
    open_series,
    prune_series_cache,
    read_series_rows,
)
from tests.series_helpers import register_series


def _ts(hour: int, minute: int = 0) -> datetime:
    return datetime(2024, 1, 1, hour=hour, minute=minute, tzinfo=timezone.utc)


class _MissingOffsetTimezone(tzinfo):
    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        return None


def test_window_key_plan_counts_inclusive_cadence_buckets() -> None:
    plan = window_key_plan(_ts(0, 10), _ts(2, 50), "1h")

    assert plan is not None
    keys = tuple(plan.keys())
    assert keys == ((_ts(0),), (_ts(1),), (_ts(2),))
    assert plan.total == len(keys) == 3


@pytest.mark.parametrize(
    ("start", "end", "cadence"),
    [
        (None, _ts(1), "1h"),
        (_ts(0), None, "1h"),
        (_ts(0), _ts(1), None),
    ],
)
def test_window_key_plan_requires_complete_bounds(
    start: datetime | None,
    end: datetime | None,
    cadence: str | None,
) -> None:
    assert window_key_plan(start, end, cadence) is None


def test_sample_domain_key_plan_counts_clipped_composite_keys() -> None:
    domain = [
        SampleDomainEntry(key=["MSFT"], start=_ts(0, 30), end=_ts(2, 59)),
        SampleDomainEntry(key=["AAPL"], start=_ts(2, 15), end=_ts(4, 30)),
        SampleDomainEntry(key=["GOOG"], start=_ts(4), end=_ts(4, 30)),
    ]

    plan = sample_domain_key_plan(
        _ts(1, 15),
        _ts(3, 45),
        "1h",
        ["security_id"],
        domain,
    )

    assert plan is not None
    keys = tuple(plan.keys())
    assert keys == (
        (_ts(1), "MSFT"),
        (_ts(2), "AAPL"),
        (_ts(2), "MSFT"),
        (_ts(3), "AAPL"),
    )
    assert plan.total == len(keys) == 4


def test_sample_domain_key_plan_includes_a_single_cadence_bucket() -> None:
    plan = sample_domain_key_plan(
        _ts(0),
        _ts(2),
        "1h",
        ["security_id"],
        [
            SampleDomainEntry(
                key=["AAPL"],
                start=_ts(1, 5),
                end=_ts(1, 55),
            )
        ],
    )

    assert plan is not None
    assert tuple(plan.keys()) == ((_ts(1), "AAPL"),)
    assert plan.total == 1


def test_sample_domain_key_plan_counts_fall_back_lattice_keys() -> None:
    timezone_ny = ZoneInfo("America/New_York")
    plan = sample_domain_key_plan(
        datetime(2024, 11, 3, 0, 7, tzinfo=timezone_ny),
        datetime(2024, 11, 3, 6, 48, tzinfo=timezone_ny),
        "2h",
        ["security_id"],
        [
            SampleDomainEntry(
                key=["AAPL"],
                start=datetime(2024, 11, 3, 3, 8, tzinfo=timezone_ny),
                end=datetime(2024, 11, 3, 6, 48, tzinfo=timezone_ny),
            )
        ],
    )

    assert plan is not None
    keys = tuple(plan.keys())
    assert [(key[0].isoformat(), key[1]) for key in keys] == [
        ("2024-11-03T04:00:00-05:00", "AAPL")
    ]
    assert plan.total == len(keys) == 1


def test_sample_domain_key_plan_counts_spring_forward_lattice_keys() -> None:
    timezone_ny = ZoneInfo("America/New_York")
    plan = sample_domain_key_plan(
        datetime(2024, 3, 10, 0, 7, tzinfo=timezone_ny),
        datetime(2024, 3, 10, 4, 48, tzinfo=timezone_ny),
        "2h",
        ["security_id"],
        [
            SampleDomainEntry(
                key=["AAPL"],
                start=datetime(2024, 3, 10, 4, 8, tzinfo=timezone_ny),
                end=datetime(2024, 3, 10, 4, 57, tzinfo=timezone_ny),
            )
        ],
    )

    assert plan is not None
    keys = tuple(plan.keys())
    assert keys == ()
    assert plan.total == len(keys) == 0


def test_sample_domain_key_plan_omits_total_for_mixed_time_lattices() -> None:
    timezone_ny = ZoneInfo("America/New_York")
    fixed_offset_end = datetime.fromisoformat("2024-11-03T04:00:00-05:00")
    plan = sample_domain_key_plan(
        datetime(2024, 11, 3, 0, tzinfo=timezone_ny),
        fixed_offset_end,
        "2h",
        ["security_id"],
        [
            SampleDomainEntry(
                key=["AAPL"],
                start=datetime(2024, 11, 3, 0, tzinfo=timezone_ny),
                end=fixed_offset_end,
            )
        ],
    )

    assert plan is not None
    keys = tuple(plan.keys())
    assert [(key[0].isoformat(), key[1]) for key in keys] == [
        ("2024-11-03T00:00:00-04:00", "AAPL"),
        ("2024-11-03T02:00:00-05:00", "AAPL"),
    ]
    assert plan.total is None


def test_window_keys_rejects_invalid_cadence() -> None:
    with pytest.raises(ValueError, match="Unsupported cadence"):
        window_key_plan(_ts(0), _ts(1), "0m")


def test_sample_domain_window_keys_rejects_invalid_cadence() -> None:
    with pytest.raises(ValueError, match="Unsupported cadence"):
        sample_domain_key_plan(
            _ts(0),
            _ts(1),
            "0m",
            ["security_id"],
            [SampleDomainEntry(key=["AAPL"], start=_ts(0), end=_ts(1))],
        )


def test_sample_domain_window_keys_rejects_mismatched_key_width() -> None:
    with pytest.raises(ValueError, match="key length"):
        sample_domain_key_plan(
            _ts(0),
            _ts(1),
            "1h",
            ["security_id"],
            [SampleDomainEntry(key=[], start=_ts(0), end=_ts(1))],
        )


class _StubSource:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.opens = 0
        self.closes = 0

    def stream(self):
        self.opens += 1
        try:
            yield from self._rows
        finally:
            self.closes += 1


class _PipelineStarts:
    def __init__(self) -> None:
        self.starts: list[str] = []
        self.nodes: list[str] = []

    def __call__(self, event: PipelineEvent) -> None:
        if isinstance(event, PipelineStarted):
            self.starts.append(event.pipeline_name)
        elif isinstance(event, NodeStarted):
            self.nodes.append(event.node_name)


def _mapper(rows):
    for row in rows:
        rec = TemporalRecord(time=row["time"])
        for key, value in row.items():
            if key == "time":
                continue
            setattr(rec, key, value)
        yield rec


def _sample_payload(samples):
    return [
        (
            item.key,
            dict(item.features.values),
            None if item.targets is None else dict(item.targets.values),
        )
        for item in samples
    ]


def _runtime_with_rows(
    tmp_path: Path,
    rows: list[dict],
    *,
    stream_id: str = "stream",
    preprocess: list[PreprocessConfig] | None = None,
    transforms: list[TransformConfig] | None = None,
    partition_by: tuple[str, ...] = (),
) -> Runtime:
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(
        "schema_version: 6\nartifact_revision: 1\n", encoding="utf-8"
    )
    runtime = Runtime(
        project_yaml=project_yaml,
        artifacts_root=artifacts_root,
        dataset=DatasetConfig(sample=SampleConfig(cadence="1h")),
        execution=ExecutionConfig(),
    )

    runtime.streams[stream_id] = SourceRuntimeStream(
        source=_StubSource(rows),
        mapper=_mapper,
        preprocess=tuple(preprocess or ()),
        partition_by=partition_by,
        presorted=False,
        transforms=tuple(transforms or ()),
    )
    return runtime


def _set_source_mapper(runtime: Runtime, mapper: RecordStage) -> None:
    source_stream = runtime.streams["stream"]
    assert isinstance(source_stream, SourceRuntimeStream)
    runtime.streams["stream"] = replace(
        source_stream,
        mapper=mapper,
        presorted=True,
    )


def _register_price_metadata(runtime: Runtime) -> VectorMetadataCatalog:
    metadata_path = runtime.artifacts_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "catalog": {
                    "counts": {"feature_vectors": 1, "target_vectors": 0},
                    "features": [
                        {
                            "id": "price",
                            "base_id": "price",
                            "kind": "scalar",
                            "present_count": 1,
                            "null_count": 0,
                        }
                    ],
                    "targets": [],
                },
                "layout": {"kind": "unsplit"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    runtime.artifacts.register(VECTOR_METADATA, "metadata.json")
    return runtime.artifacts.load(VECTOR_METADATA_SPEC).catalog


def test_source_pipeline_carries_source_summary(tmp_path: Path) -> None:
    runtime = _runtime_with_rows(tmp_path, [], stream_id="prices")
    runtime.streams["prices"] = replace(
        runtime.streams["prices"],
        source=Source(
            DataLoader(
                FsFileTransport("/tmp/prices.jsonl"),
                JsonLinesDecoder(),
            ),
            IdentityParser(),
        ),
    )

    pipeline = build_stream_pipeline(runtime, "prices")

    assert pipeline.name == "stream:prices"
    assert pipeline.summary == "transport=fs.file file=prices.jsonl"
    assert pipeline.input.name == "open_source"
    assert [stage.name for stage in pipeline.stages] == [
        "map_records",
        "ensure_record_order",
    ]
    assert pipeline.input.progress is not None
    assert [stage.progress is not None for stage in pipeline.stages] == [False, True]


def test_stream_pipeline_carries_source_summary(tmp_path: Path) -> None:
    runtime = _runtime_with_rows(tmp_path, [], stream_id="source")
    (tmp_path / "AAPL.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "MSFT.jsonl").write_text("", encoding="utf-8")
    source = Source(
        DataLoader(
            FsGlobTransport(str(tmp_path / "*.jsonl")),
            JsonLinesDecoder(),
        ),
        IdentityParser(),
    )
    runtime.streams["source"] = replace(
        runtime.streams["source"],
        source=source,
    )
    runtime.streams["derived"] = DerivedRuntimeStream(
        input_stream="source",
        partition_by=(),
        transforms=(),
    )

    pipeline = build_stream_pipeline(runtime, "derived")

    assert pipeline.name == "stream:derived"
    assert pipeline.summary == (
        "transport=fs.glob count=2 first=AAPL.jsonl last=MSFT.jsonl"
    )
    assert pipeline.input.name == "stream:source/open_source"
    assert [stage.name for stage in pipeline.stages] == [
        "stream:source/map_records",
        "stream:source/ensure_record_order",
    ]
    assert pipeline.input.progress is not None
    assert [stage.progress is not None for stage in pipeline.stages] == [False, True]


def test_derived_stream_reuses_upstream_order_without_a_mapper(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(1), "symbol": "B"},
            {"time": _ts(2), "symbol": "A"},
            {"time": _ts(0), "symbol": "A"},
        ],
        partition_by=("symbol",),
    )
    runtime.streams["derived"] = DerivedRuntimeStream(
        input_stream="stream",
        partition_by=("symbol",),
        transforms=(),
    )

    pipeline = build_stream_pipeline(runtime, "derived")

    assert pipeline.input.name == "stream:stream/open_source"
    assert [stage.name for stage in pipeline.stages] == [
        "stream:stream/map_records",
        "stream:stream/ensure_record_order",
    ]
    records = list(run_pipeline(runtime, pipeline))
    assert [(record.symbol, record.time.hour) for record in records] == [
        ("A", 0),
        ("A", 2),
        ("B", 1),
    ]


def test_cross_section_groups_by_time_and_restores_canonical_order(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0), "symbol": "A", "value": 1.0},
            {"time": _ts(1), "symbol": "A", "value": 3.0},
            {"time": _ts(0), "symbol": "B", "value": 2.0},
            {"time": _ts(1), "symbol": "B", "value": 4.0},
        ],
        partition_by=("symbol",),
    )
    runtime.streams["ranked"] = CrossSectionRuntimeStream(
        input_stream="stream",
        partition_by=("symbol",),
        cross_section=(RankScoreConfig(field="value", to="rank", min_samples=2),),
        transforms=(),
    )

    records = list(run_stream_pipeline(runtime, "ranked"))

    assert [(record.symbol, record.time.hour, record.rank) for record in records] == [
        ("A", 0, -0.5),
        ("A", 1, -0.5),
        ("B", 0, 0.5),
        ("B", 1, 0.5),
    ]


def test_cross_section_operations_run_in_configured_order(tmp_path: Path) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {
                "time": _ts(0),
                "symbol": symbol,
                "signal": signal,
                "control": control,
            }
            for symbol, signal, control in (
                ("A", 10.0, 1.0),
                ("B", 30.0, 2.0),
                ("C", 20.0, 4.0),
                ("D", 40.0, 8.0),
            )
        ],
        partition_by=("symbol",),
    )
    runtime.streams["neutralized"] = CrossSectionRuntimeStream(
        input_stream="stream",
        partition_by=("symbol",),
        cross_section=(
            RankScoreConfig(field="signal", to="signal_rank", min_samples=4),
            OlsResidualConfig(
                y="signal_rank",
                x=("control",),
                to="residual",
                min_samples=4,
            ),
        ),
        transforms=(),
    )

    records = list(run_stream_pipeline(runtime, "neutralized"))

    assert all(record.residual is not None for record in records)
    assert sum(record.residual for record in records) == pytest.approx(0.0)


def test_cross_section_spill_matches_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(hour), "symbol": symbol, "value": value}
            for symbol, values in (("A", (1.0, 4.0)), ("B", (2.0, 5.0)))
            for hour, value in enumerate(values)
        ],
        partition_by=("symbol",),
    )
    runtime.streams["ranked"] = CrossSectionRuntimeStream(
        input_stream="stream",
        partition_by=("symbol",),
        cross_section=(RankScoreConfig(field="value", to="rank", min_samples=2),),
        transforms=(),
    )
    expected = list(run_stream_pipeline(runtime, "ranked"))
    normal_batch_sort = cross_section_pipeline.batch_sort

    def spilling_batch_sort(items, buffer_bytes, key, spill_dir=None, progress=None):
        return normal_batch_sort(
            items,
            buffer_bytes=1,
            key=key,
            spill_dir=spill_dir,
            progress=progress,
        )

    monkeypatch.setattr(cross_section_pipeline, "batch_sort", spilling_batch_sort)
    monkeypatch.setattr(stream_order, "batch_sort", spilling_batch_sort)

    actual = list(run_stream_pipeline(runtime, "ranked"))

    assert actual == expected


def test_cross_section_previews_and_partition_local_lag(tmp_path: Path) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0), "symbol": "A", "value": 1.0},
            {"time": _ts(1), "symbol": "A", "value": 3.0},
            {"time": _ts(0), "symbol": "B", "value": 2.0},
            {"time": _ts(1), "symbol": "B", "value": 4.0},
        ],
        partition_by=("symbol",),
    )
    runtime.streams["ranked"] = CrossSectionRuntimeStream(
        input_stream="stream",
        partition_by=("symbol",),
        cross_section=(RankScoreConfig(field="value", to="rank", min_samples=2),),
        transforms=(LagConfig(field="rank", periods=1, to="previous_rank"),),
    )

    input_records = list(run_stream_preview_pipeline(runtime, "ranked", "input"))
    canonical = list(run_stream_preview_pipeline(runtime, "ranked", "canonical"))
    records = list(run_stream_preview_pipeline(runtime, "ranked", "records"))

    assert [(record.symbol, record.time.hour) for record in input_records] == [
        ("A", 0),
        ("A", 1),
        ("B", 0),
        ("B", 1),
    ]
    assert all(not hasattr(record, "rank") for record in input_records)
    assert [record.rank for record in canonical] == [-0.5, -0.5, 0.5, 0.5]
    assert all(not hasattr(record, "previous_rank") for record in canonical)
    assert [record.previous_rank for record in records] == [None, -0.5, None, 0.5]


def test_cross_section_rejects_duplicate_partition_at_one_time(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0), "symbol": "A", "value": 1.0},
            {"time": _ts(0), "symbol": "A", "value": 2.0},
        ],
        partition_by=("symbol",),
    )
    runtime.streams["ranked"] = CrossSectionRuntimeStream(
        input_stream="stream",
        partition_by=("symbol",),
        cross_section=(RankScoreConfig(field="value", to="rank", min_samples=2),),
        transforms=(),
    )

    with pytest.raises(ValueError, match="duplicate partition.*A"):
        list(run_stream_pipeline(runtime, "ranked"))


@pytest.mark.parametrize("preview", ["input", "canonical"])
def test_derived_record_previews_use_records_before_derived_transforms(
    tmp_path: Path,
    preview,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0), "value": 1.0},
            {"time": _ts(2), "value": 2.0},
        ],
    )
    runtime.streams["derived"] = DerivedRuntimeStream(
        input_stream="stream",
        partition_by=(),
        transforms=(EnsureCadenceConfig(cadence="1h"),),
    )

    pipeline = build_stream_pipeline(runtime, "derived")
    assert pipeline.input.name == "stream:stream/open_source"
    assert [stage.name for stage in pipeline.stages] == [
        "stream:stream/map_records",
        "stream:stream/ensure_record_order",
        "ensure_cadence",
    ]

    records = run_stream_preview_pipeline(
        runtime,
        "derived",
        preview,
    )

    assert [record.time.hour for record in records] == [0, 2]


def test_aligned_pipeline_closes_inputs_after_partial_read(tmp_path: Path) -> None:
    runtime = _runtime_with_rows(tmp_path, [])
    left = _StubSource([{"time": _ts(0)}, {"time": _ts(1)}])
    right = _StubSource([{"time": _ts(0)}, {"time": _ts(1)}])

    def take_left(rows):
        for left_record, _ in rows:
            yield left_record

    runtime.streams = {
        "left": SourceRuntimeStream(
            source=left,
            mapper=_mapper,
            preprocess=(),
            partition_by=(),
            presorted=True,
            transforms=(),
        ),
        "right": SourceRuntimeStream(
            source=right,
            mapper=_mapper,
            preprocess=(),
            partition_by=(),
            presorted=True,
            transforms=(),
        ),
        "combined": AlignedRuntimeStream(
            inputs=("left", "right"),
            combine=take_left,
            partition_by=(),
            transforms=(),
        ),
    }

    records = run_stream_pipeline(runtime, "combined")
    assert next(records).time == _ts(0)
    records.close()

    assert left.closes == 1
    assert right.closes == 1


def test_broadcast_pipeline_reuses_exact_input_across_partitions(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(tmp_path, [])
    primary = _StubSource(
        [
            {"time": _ts(0), "id_": "A", "value": 1.0},
            {"time": _ts(1), "id_": "A", "value": 2.0},
            {"time": _ts(0), "id_": "B", "value": 3.0},
            {"time": _ts(1), "id_": "B", "value": 4.0},
        ]
    )
    broadcast = _StubSource(
        [
            {"time": _ts(0), "value": 10.0},
            {"time": _ts(1), "value": 20.0},
        ]
    )

    def attach_broadcast(rows):
        for primary_record, broadcast_record in rows:
            record = TemporalRecord(time=primary_record.time)
            record.id_ = primary_record.id_
            record.value = primary_record.value
            record.broadcast_value = broadcast_record.value
            yield record

    runtime.streams = {
        "primary": SourceRuntimeStream(
            source=primary,
            mapper=_mapper,
            preprocess=(),
            partition_by=("id_",),
            presorted=True,
            transforms=(),
        ),
        "broadcast": SourceRuntimeStream(
            source=broadcast,
            mapper=_mapper,
            preprocess=(),
            partition_by=(),
            presorted=True,
            transforms=(),
        ),
        "enriched": BroadcastRuntimeStream(
            input_stream="primary",
            broadcast_stream="broadcast",
            combine=attach_broadcast,
            partition_by=("id_",),
            transforms=(),
        ),
    }

    pipeline = build_stream_pipeline(runtime, "enriched")
    records = list(run_pipeline(runtime, pipeline))

    assert pipeline.summary == "primary=primary,broadcast=broadcast"
    assert pipeline.input.name == "broadcast_inputs"
    assert [stage.name for stage in pipeline.stages] == ["combine_records"]
    assert [
        (record.id_, record.time.hour, record.value, record.broadcast_value)
        for record in records
    ] == [
        ("A", 0, 1.0, 10.0),
        ("A", 1, 2.0, 20.0),
        ("B", 0, 3.0, 10.0),
        ("B", 1, 4.0, 20.0),
    ]
    assert primary.closes == 1
    assert broadcast.closes == 1


def test_as_of_pipeline_validates_lookup_tail_after_primary_finishes(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(tmp_path, [])
    primary = _StubSource([{"time": _ts(3), "id_": "A"}])
    lookup = _StubSource(
        [
            {"time": _ts(1), "id_": "A"},
            {"time": _ts(4), "id_": "A"},
            {"time": _ts(2), "id_": "A"},
        ]
    )

    def take_primary(rows):
        for primary_record, _ in rows:
            yield primary_record

    runtime.streams = {
        "primary": SourceRuntimeStream(
            source=primary,
            mapper=_mapper,
            preprocess=(),
            partition_by=("id_",),
            presorted=True,
            transforms=(),
        ),
        "lookup": SourceRuntimeStream(
            source=lookup,
            mapper=_mapper,
            preprocess=(),
            partition_by=("id_",),
            presorted=True,
            transforms=(),
        ),
        "enriched": AsOfRuntimeStream(
            input_stream="primary",
            lookup_stream="lookup",
            combine=take_primary,
            partition_by=("id_",),
            max_age=None,
            require_match=True,
            transforms=(),
        ),
    }

    with pytest.raises(ValueError, match="violates presorted order"):
        list(run_stream_pipeline(runtime, "enriched"))

    assert primary.closes == 1
    assert lookup.closes == 1


def test_broadcast_pipeline_closes_inputs_after_partial_read(tmp_path: Path) -> None:
    runtime = _runtime_with_rows(tmp_path, [])
    primary = _StubSource(
        [
            {"time": _ts(0), "id_": "A"},
            {"time": _ts(1), "id_": "A"},
        ]
    )
    broadcast = _StubSource(
        [
            {"time": _ts(0)},
            {"time": _ts(1)},
        ]
    )

    def take_primary(rows):
        for primary_record, _ in rows:
            yield primary_record

    runtime.streams = {
        "primary": SourceRuntimeStream(
            source=primary,
            mapper=_mapper,
            preprocess=(),
            partition_by=("id_",),
            presorted=True,
            transforms=(),
        ),
        "broadcast": SourceRuntimeStream(
            source=broadcast,
            mapper=_mapper,
            preprocess=(),
            partition_by=(),
            presorted=True,
            transforms=(),
        ),
        "enriched": BroadcastRuntimeStream(
            input_stream="primary",
            broadcast_stream="broadcast",
            combine=take_primary,
            partition_by=("id_",),
            transforms=(),
        ),
    }

    records = run_stream_pipeline(runtime, "enriched")
    assert next(records).time == _ts(0)
    records.close()

    assert primary.closes == 1
    assert broadcast.closes == 1


def test_broadcast_record_previews_preserve_stage_boundaries(tmp_path: Path) -> None:
    runtime = _runtime_with_rows(tmp_path, [])

    def attach_broadcast(rows):
        for primary_record, broadcast_record in rows:
            record = TemporalRecord(time=primary_record.time)
            record.id_ = primary_record.id_
            record.value = primary_record.value
            record.broadcast_value = broadcast_record.value
            yield record

    runtime.streams = {
        "primary": SourceRuntimeStream(
            source=_StubSource(
                [
                    {"time": _ts(0), "id_": "A", "value": 1.0},
                    {"time": _ts(2), "id_": "A", "value": 2.0},
                ]
            ),
            mapper=_mapper,
            preprocess=(),
            partition_by=("id_",),
            presorted=True,
            transforms=(),
        ),
        "broadcast": SourceRuntimeStream(
            source=_StubSource(
                [
                    {"time": _ts(0), "value": 10.0},
                    {"time": _ts(2), "value": 20.0},
                ]
            ),
            mapper=_mapper,
            preprocess=(),
            partition_by=(),
            presorted=True,
            transforms=(),
        ),
        "enriched": BroadcastRuntimeStream(
            input_stream="primary",
            broadcast_stream="broadcast",
            combine=attach_broadcast,
            partition_by=("id_",),
            transforms=(EnsureCadenceConfig(cadence="1h"),),
        ),
    }

    input_rows = list(run_stream_preview_pipeline(runtime, "enriched", "input"))
    canonical = list(run_stream_preview_pipeline(runtime, "enriched", "canonical"))
    records = list(run_stream_preview_pipeline(runtime, "enriched", "records"))

    assert [
        (primary.time.hour, broadcast.time.hour) for primary, broadcast in input_rows
    ] == [(0, 0), (2, 2)]
    assert [record.time.hour for record in canonical] == [0, 2]
    assert [record.time.hour for record in records] == [0, 1, 2]


def test_source_pipeline_runs_map_then_preprocess(
    tmp_path: Path,
) -> None:
    rows = [
        {"time": _ts(0, 30), "value": 1.0},
        {"time": _ts(0, 10), "value": 2.0},
    ]
    runtime = _runtime_with_rows(
        tmp_path,
        rows,
        preprocess=[FloorTimeConfig(cadence="1h")],
    )
    ctx = runtime

    input_rows = list(run_stream_preview_pipeline(ctx, "stream", "input"))
    assert input_rows == rows

    mapped = list(run_stream_preview_pipeline(ctx, "stream", "canonical"))
    assert all(isinstance(record, TemporalRecord) for record in mapped)
    assert [record.time for record in mapped] == [rows[0]["time"], rows[1]["time"]]

    preprocessed = list(run_stream_preview_pipeline(ctx, "stream", "records"))
    assert all(record.time.minute == 0 for record in preprocessed)


@pytest.mark.parametrize(
    ("invalid_time", "error_type", "message"),
    [
        (
            "2024-01-01T00:00:00Z",
            TypeError,
            "Mapped record 1 time must be a datetime; got str",
        ),
        (
            datetime(2024, 1, 1),
            ValueError,
            "Mapped record 1 time must be timezone-aware",
        ),
        (
            datetime(2024, 1, 1, tzinfo=_MissingOffsetTimezone()),
            ValueError,
            "Mapped record 1 time must be timezone-aware",
        ),
    ],
)
def test_source_pipeline_rejects_invalid_mapped_time(
    tmp_path: Path,
    invalid_time,
    error_type,
    message: str,
) -> None:
    runtime = _runtime_with_rows(tmp_path, [{"raw": 1}])

    def map_record(rows):
        for _ in rows:
            yield SimpleNamespace(time=invalid_time)

    _set_source_mapper(runtime, map_record)

    with pytest.raises(error_type, match=message):
        list(run_stream_pipeline(runtime, "stream"))


def test_source_pipeline_normalizes_aware_mapped_time_to_utc(
    tmp_path: Path,
) -> None:
    mapped_time = datetime(2024, 1, 1, tzinfo=timezone(timedelta(hours=5, minutes=45)))
    runtime = _runtime_with_rows(tmp_path, [{"raw": 1}])

    def map_record(rows):
        for _ in rows:
            yield SimpleNamespace(time=mapped_time)

    _set_source_mapper(runtime, map_record)

    [record] = list(run_stream_pipeline(runtime, "stream"))

    assert record.time == datetime(2023, 12, 31, 18, 15, tzinfo=timezone.utc)
    assert record.time.tzinfo is timezone.utc


def test_source_pipeline_keeps_fall_back_instants_distinct_in_utc(
    tmp_path: Path,
) -> None:
    timezone_ny = ZoneInfo("America/New_York")
    mapped_times = (
        datetime(2024, 11, 3, 1, 30, tzinfo=timezone_ny, fold=0),
        datetime(2024, 11, 3, 1, 30, tzinfo=timezone_ny, fold=1),
    )
    runtime = _runtime_with_rows(tmp_path, [{"raw": 1}, {"raw": 2}])

    def map_record(rows):
        for row, mapped_time in zip(rows, mapped_times, strict=True):
            yield SimpleNamespace(time=mapped_time, raw=row["raw"])

    _set_source_mapper(runtime, map_record)

    records = list(run_stream_pipeline(runtime, "stream"))

    assert [record.time for record in records] == [
        datetime(2024, 11, 3, 5, 30, tzinfo=timezone.utc),
        datetime(2024, 11, 3, 6, 30, tzinfo=timezone.utc),
    ]


def test_as_of_pipeline_does_not_use_future_fall_back_record(
    tmp_path: Path,
) -> None:
    timezone_ny = ZoneInfo("America/New_York")
    primary_time = datetime(2024, 11, 3, 1, 30, tzinfo=timezone_ny, fold=0)
    future_lookup_time = datetime(
        2024,
        11,
        3,
        1,
        30,
        tzinfo=timezone_ny,
        fold=1,
    )
    runtime = _runtime_with_rows(tmp_path, [])

    def map_source_record(rows):
        for row in rows:
            yield SimpleNamespace(**row)

    def mark_lookup_match(rows):
        for primary_record, lookup_record in rows:
            primary_record.lookup_matched = lookup_record is not None
            yield primary_record

    runtime.streams = {
        "primary": SourceRuntimeStream(
            source=_StubSource(
                [{"time": primary_time, "id_": "A"}],
            ),
            mapper=map_source_record,
            preprocess=(),
            partition_by=("id_",),
            presorted=True,
            transforms=(),
        ),
        "lookup": SourceRuntimeStream(
            source=_StubSource(
                [{"time": future_lookup_time, "id_": "A"}],
            ),
            mapper=map_source_record,
            preprocess=(),
            partition_by=("id_",),
            presorted=True,
            transforms=(),
        ),
        "enriched": AsOfRuntimeStream(
            input_stream="primary",
            lookup_stream="lookup",
            combine=mark_lookup_match,
            partition_by=("id_",),
            max_age=timedelta(minutes=30),
            require_match=False,
            transforms=(),
        ),
    }

    [record] = list(run_stream_pipeline(runtime, "enriched"))

    assert record.time == datetime(2024, 11, 3, 5, 30, tzinfo=timezone.utc)
    assert record.lookup_matched is False


def test_source_pipeline_closes_mapper_after_partial_read(tmp_path: Path) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [{"time": _ts(0)}, {"time": _ts(1)}],
    )
    mapper_streams = []
    closed = False

    def map_record(rows):
        def mapped_records():
            nonlocal closed
            try:
                yield from _mapper(rows)
            finally:
                closed = True

        stream = mapped_records()
        mapper_streams.append(stream)
        return stream

    _set_source_mapper(runtime, map_record)

    records = run_stream_pipeline(runtime, "stream")
    next(records)
    records.close()

    assert mapper_streams
    assert closed is True


def test_source_pipeline_surfaces_mapper_close_errors(tmp_path: Path) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [{"time": _ts(0)}, {"time": _ts(1)}],
    )

    def map_record(rows):
        try:
            yield from _mapper(rows)
        finally:
            raise RuntimeError("mapper close failed")

    _set_source_mapper(runtime, map_record)
    records = run_stream_pipeline(runtime, "stream")
    next(records)

    with pytest.raises(RuntimeError, match="mapper close failed"):
        records.close()


def test_pipeline_builders_expose_structure(tmp_path: Path) -> None:
    runtime = _runtime_with_rows(tmp_path, [{"time": _ts(0), "value": 1.0}])
    _register_price_metadata(runtime)
    cfg = SeriesConfig(stream="stream", id="price", field="value")

    stream_pipeline = build_stream_pipeline(runtime, "stream")
    series_pipeline = build_series_pipeline(runtime, cfg)

    assert stream_pipeline.input.name == "open_source"
    assert stream_pipeline.stages[0].name == "map_records"
    assert series_pipeline.input.name == "stream:stream/open_source"
    assert [stage.name for stage in series_pipeline.stages] == [
        "stream:stream/map_records",
        "stream:stream/ensure_record_order",
        "project_series",
    ]
    assert series_pipeline.summary is None
    assert isinstance(series_pipeline.input, Input)


def test_source_pipeline_orders_by_partition_and_time(tmp_path: Path) -> None:
    rows = [
        {"time": _ts(1), "value": 10.0, "symbol": "B"},
        {"time": _ts(0), "value": 5.0, "symbol": "A"},
        {"time": _ts(2), "value": 6.0, "symbol": "A"},
    ]
    runtime = _runtime_with_rows(tmp_path, rows, partition_by=("symbol",))
    ctx = runtime

    ordered = list(run_pipeline(ctx, build_stream_pipeline(ctx, "stream")))
    assert [(rec.symbol, rec.time.hour) for rec in ordered] == [
        ("A", 0),
        ("A", 2),
        ("B", 1),
    ]


def test_stream_pipeline_applies_stream_transforms(tmp_path: Path) -> None:
    rows = [
        {"time": _ts(0), "value": 1.0},
        {"time": _ts(2), "value": 2.0},
    ]
    runtime = _runtime_with_rows(
        tmp_path,
        rows,
        transforms=[EnsureCadenceConfig(cadence="1h")],
    )
    ctx = runtime

    transformed = list(run_stream_pipeline(ctx, "stream"))
    assert [(rec.time.hour, rec.value) for rec in transformed] == [
        (0, 1.0),
        (1, None),
        (2, 2.0),
    ]


def test_stream_pipeline_applies_rolling_quantile(tmp_path: Path) -> None:
    rows = [
        {"time": _ts(hour), "value": value}
        for hour, value in enumerate([0.0, 10.0, 20.0, 30.0])
    ]
    runtime = _runtime_with_rows(
        tmp_path,
        rows,
        transforms=[
            RollingQuantileConfig(
                field="value",
                window=4,
                min_samples=1,
                quantile=0.25,
                to="q25",
            )
        ],
    )

    transformed = list(run_stream_pipeline(runtime, "stream"))

    assert [record.q25 for record in transformed] == [0.0, 2.5, 5.0, 7.5]


def test_stream_pipeline_applies_ewm_mean(tmp_path: Path) -> None:
    rows = [
        {"time": _ts(hour), "value": value}
        for hour, value in enumerate([10.0, 20.0, 30.0])
    ]
    runtime = _runtime_with_rows(
        tmp_path,
        rows,
        transforms=[EwmMeanConfig(field="value", alpha=0.5, to="smoothed")],
    )

    transformed = list(run_stream_pipeline(runtime, "stream"))

    assert [record.smoothed for record in transformed] == [10.0, 15.0, 22.5]


def test_ensure_cadence_placeholders_do_not_copy_payload_fields(
    tmp_path: Path,
) -> None:
    rows = [
        {"time": _ts(0), "symbol": "A", "value": 1.0, "volume": 100},
        {"time": _ts(2), "symbol": "A", "value": 2.0, "volume": 200},
    ]
    runtime = _runtime_with_rows(
        tmp_path,
        rows,
        partition_by=("symbol",),
        transforms=[EnsureCadenceConfig(cadence="1h")],
    )
    ctx = runtime

    placeholder = list(run_stream_pipeline(ctx, "stream"))[1]

    assert placeholder.time == _ts(1)
    assert placeholder.symbol == "A"
    assert placeholder.value is None
    assert placeholder.volume is None


def test_ensure_schedule_uses_stream_partition_and_caches_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rows = [
        {"time": _ts(1), "symbol": "A", "value": 1.0},
    ]
    runtime = _runtime_with_rows(
        tmp_path,
        rows,
        partition_by=("symbol",),
        transforms=[EnsureScheduleConfig(schedule="schedule")],
    )
    artifact_path = runtime.artifacts_root / "schedule.jsonl"
    artifact_path.write_text(
        "\n".join(
            [
                json.dumps({"time": _ts(0).isoformat(), "symbol": "A"}),
                json.dumps({"time": _ts(1).isoformat(), "symbol": "A"}),
                json.dumps({"time": _ts(2).isoformat(), "symbol": "A"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runtime.artifacts.register(
        "schedule",
        artifact_path.name,
        meta={"partition_by": ["symbol"]},
    )
    read_count = 0
    original_read_schedule = stream_stages.read_schedule

    def counted_read_schedule(path, partition_by):
        nonlocal read_count
        read_count += 1
        return original_read_schedule(path, partition_by)

    monkeypatch.setattr(stream_stages, "read_schedule", counted_read_schedule)
    ctx = runtime

    first = list(run_stream_pipeline(ctx, "stream"))
    second = list(run_stream_pipeline(ctx, "stream"))

    assert [(rec.symbol, rec.time.hour, rec.value) for rec in first] == [
        ("A", 0, None),
        ("A", 1, 1.0),
        ("A", 2, None),
    ]
    assert [(rec.symbol, rec.time.hour, rec.value) for rec in second] == [
        ("A", 0, None),
        ("A", 1, 1.0),
        ("A", 2, None),
    ]
    assert read_count == 1


def test_series_pipeline_wraps_record_values(tmp_path: Path) -> None:
    rows = [{"time": _ts(0), "value": 3.0, "symbol": "X"}]
    runtime = _runtime_with_rows(
        tmp_path,
        rows,
        partition_by=("symbol",),
    )
    ctx = runtime
    cfg = SeriesConfig(
        stream="stream",
        id="price",
        field="value",
    )

    preview_pipeline = build_series_pipeline(ctx, cfg)
    records = list(
        run_pipeline(
            ctx,
            preview_pipeline.through_stage_named("project_series"),
        )
    )
    assert len(records) == 1
    record = records[0]
    assert record.value == 3.0
    assert record.id == "price__@symbol:X"


def test_series_preview_keeps_collection_values_unassembled(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0), "value": 1.0},
            {"time": _ts(0, 30), "value": 2.0},
        ],
    )
    config = SeriesConfig(
        stream="stream",
        id="price",
        field="value",
        collect=2,
    )

    records = list(
        run_series_pipeline(
            runtime,
            config,
        )
    )

    assert all(isinstance(record, SeriesRecord) for record in records)
    assert [record.value for record in records] == [1.0, 2.0]


def test_unpartitioned_series_pipeline_preserves_time_order(tmp_path: Path) -> None:
    rows = [
        {"time": _ts(2), "value": 3.0},
        {"time": _ts(0), "value": 1.0},
        {"time": _ts(1), "value": 2.0},
    ]
    runtime = _runtime_with_rows(tmp_path, rows)

    records = list(
        run_series_pipeline(
            runtime,
            SeriesConfig(stream="stream", id="price", field="value"),
        )
    )

    assert [(record.time.hour, record.value) for record in records] == [
        (0, 1.0),
        (1, 2.0),
        (2, 3.0),
    ]


def test_partitioned_series_pipeline_orders_across_partitions(tmp_path: Path) -> None:
    rows = [
        {"time": _ts(0), "value": 1.0, "symbol": "A"},
        {"time": _ts(2), "value": 3.0, "symbol": "A"},
        {"time": _ts(1), "value": 2.0, "symbol": "B"},
    ]
    runtime = _runtime_with_rows(
        tmp_path,
        rows,
        partition_by=("symbol",),
    )

    records = list(
        run_series_pipeline(
            runtime,
            SeriesConfig(stream="stream", id="price", field="value"),
        )
    )

    assert [(record.time.hour, record.id) for record in records] == [
        (0, "price__@symbol:A"),
        (1, "price__@symbol:B"),
        (2, "price__@symbol:A"),
    ]


def test_series_pipeline_builds_sequences(tmp_path: Path) -> None:
    rows = [
        {"time": _ts(0), "value": 1.0},
        {"time": _ts(1), "value": 2.0},
        {"time": _ts(2), "value": 3.0},
        {"time": _ts(3), "value": 4.0},
    ]
    runtime = _runtime_with_rows(tmp_path, rows)
    ctx = runtime
    cfg = SeriesConfig(
        stream="stream",
        id="price",
        field="value",
        sequence={"size": 2, "stride": 2},
    )

    preview_pipeline = build_series_pipeline(ctx, cfg)
    sequences = list(
        run_pipeline(
            ctx,
            preview_pipeline.through_stage_named("sequence_series"),
        )
    )
    assert len(sequences) == 2
    assert isinstance(sequences[0], SeriesSequence)
    assert sequences[0].time == _ts(1)
    assert sequences[0].values == [1.0, 2.0]
    assert sequences[1].time == _ts(3)
    assert sequences[1].values == [3.0, 4.0]


def test_series_pipeline_keeps_scaled_sequence_inputs_raw(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0), "value": 1.0},
            {"time": _ts(1), "value": 11.0},
        ],
    )
    config = SeriesConfig(
        stream="stream",
        id="price",
        field="value",
        scale=True,
        sequence=SequenceConfig(size=2),
    )

    [sequence] = run_series_pipeline(
        runtime,
        config,
    )

    assert isinstance(sequence, SeriesSequence)
    assert sequence.time == _ts(1)
    assert sequence.values == [1.0, 11.0]


def test_postprocess_rejects_targets_absent_from_metadata(tmp_path: Path) -> None:
    runtime = _runtime_with_rows(tmp_path, [])
    schema = _register_price_metadata(runtime)
    samples = [
        Sample(key=(_ts(0),), features=Vector(values={"price": 1.0})),
        Sample(
            key=(_ts(1),),
            features=Vector(values={"price": 2.0}),
            targets=Vector(values={"return": 0.1}),
        ),
    ]

    with pytest.raises(RuntimeError, match="no target entries"):
        list(
            build_postprocess_plan(
                runtime.dataset.postprocess,
                schema,
            ).apply(iter(samples))
        )


def test_dataset_pipeline_matches_sample_and_postprocess_chain(tmp_path: Path) -> None:
    rows = [
        {"time": _ts(0), "value": None},
        {"time": _ts(1), "value": 2.0},
    ]
    runtime = _runtime_with_rows(tmp_path, rows)
    runtime.dataset = runtime.dataset.model_copy(
        update={
            "postprocess": PostprocessConfig.model_validate(
                {"features": {"threshold": 1.0}}
            )
        }
    )
    schema = _register_price_metadata(runtime)
    ctx = runtime
    cfg = SeriesConfig(stream="stream", id="price", field="value")
    register_series(runtime, [cfg], "1h")

    pipeline = build_dataset_pipeline(
        ctx,
        schema,
        None,
    )
    assert isinstance(pipeline.input, Input)
    assert pipeline.input.progress is None

    samples_preview = list(run_sample_pipeline(ctx, schema, None))
    postprocess_preview = list(run_dataset_pipeline(ctx, schema, None))
    dataset_out = list(run_dataset_pipeline(ctx, schema, None))

    manual = open_samples(ctx, [cfg.id])
    manual_out = list(
        build_postprocess_plan(runtime.dataset.postprocess, schema).apply(manual)
    )

    assert [sample.features.values for sample in samples_preview] == [
        {"price": None},
        {"price": 2.0},
    ]
    assert postprocess_preview == dataset_out == manual_out
    assert [sample.features.values for sample in postprocess_preview] == [
        {"price": 2.0}
    ]


def test_rectangular_dataset_source_reuses_its_key_plan(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(tmp_path, [])
    schema = _register_price_metadata(runtime)
    cfg = SeriesConfig(stream="stream", id="price", field="value")
    feature_configs = [cfg]
    register_series(runtime, feature_configs, "1h")

    key_plan = window_key_plan(_ts(0, 10), _ts(2, 50), "1h")
    assert key_plan is not None
    pipeline = build_dataset_pipeline(
        runtime,
        schema,
        key_plan,
    )

    assert isinstance(pipeline.input, Input)
    assert pipeline.input.progress is not None
    samples = list(pipeline.input.open())
    assert [sample.key for sample in samples] == [
        (_ts(0),),
        (_ts(1),),
        (_ts(2),),
    ]
    assert pipeline.input.progress(len(samples)) == ProgressSnapshot(
        completed=3,
        total=len(samples),
        unit="samples",
    )


def test_sample_input_snapshots_selected_ids(tmp_path: Path) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [{"time": _ts(0), "value": 1.0}],
    )
    config = SeriesConfig(stream="stream", id="price", field="value")
    register_series(runtime, [config], "1h")
    feature_ids = [config.id]

    sample_input = build_sample_input(runtime, feature_ids)
    feature_ids.clear()

    assert list(sample_input.open()) == list(open_samples(runtime, [config.id]))


def test_rectangular_features_and_targets_share_every_planned_key(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0), "value": 1.0, "target": 10.0},
            {"time": _ts(2), "value": 3.0, "target": 30.0},
        ],
    )
    feature = SeriesConfig(stream="stream", id="price", field="value")
    target = TargetSeriesConfig(
        stream="stream",
        id="return",
        field="target",
        horizon="0s",
    )
    register_series(runtime, [feature], "1h", targets=[target])
    key_plan = window_key_plan(_ts(0), _ts(2), "1h")
    assert key_plan is not None

    samples = list(
        open_samples(
            runtime,
            [feature.id],
            target_ids=[target.id],
            key_plan=key_plan,
        )
    )

    assert [sample.key for sample in samples] == [
        (_ts(0),),
        (_ts(1),),
        (_ts(2),),
    ]
    assert samples[1].features.values == {}
    assert samples[1].targets is not None
    assert samples[1].targets.values == {}


def test_series_artifact_feeds_serve_pipeline(tmp_path: Path) -> None:
    rows = [
        {"time": _ts(1), "id_": "B", "value": 2.0, "other": 20.0},
        {"time": _ts(0), "id_": "A", "value": 1.0, "other": 10.0},
        {"time": _ts(0), "id_": "B", "value": 3.0, "other": 30.0},
    ]
    runtime = _runtime_with_rows(
        tmp_path,
        rows,
        stream_id="prices",
        partition_by=("id_",),
    )
    configs = [
        SeriesConfig(
            stream="prices",
            id="value_feature",
            field="value",
        ),
        SeriesConfig(
            stream="prices",
            id="other_feature",
            field="other",
        ),
    ]
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h", keys=["id_"]),
        features=configs,
    )
    source = runtime.streams["prices"].source
    assert isinstance(source, _StubSource)

    unrelated = runtime.artifacts_root / "build/series/features/keep.txt"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("keep", encoding="utf-8")

    task = SeriesTask()
    result = build_series_artifact(runtime, task)
    runtime.artifacts.register(
        SERIES,
        relative_path=task.output,
        meta=result.meta,
    )
    cached = _sample_payload(
        open_samples(
            runtime,
            [config.id for config in configs],
        )
    )

    assert result.meta == {
        "features": 2,
        "targets": 0,
        "rows": 3,
        "format": "jsonl.gz",
    }
    manifest_path = runtime.artifacts_root / task.output
    manifest = load_series_manifest(manifest_path)
    data_path = Path(manifest.path)
    assert data_path.parts[0] == "manifest.data"
    assert data_path.parts[2:] == ("series.jsonl.gz",)
    assert [(entry.id, entry.samples) for entry in manifest.features] == [
        ("value_feature", 3),
        ("other_feature", 3),
    ]
    assert result.companion_paths == (str(Path("build/series") / data_path),)
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert cached == [
        (
            (_ts(0), "A"),
            {"value_feature": 1.0, "other_feature": 10.0},
            None,
        ),
        (
            (_ts(0), "B"),
            {"value_feature": 3.0, "other_feature": 30.0},
            None,
        ),
        (
            (_ts(1), "B"),
            {"value_feature": 2.0, "other_feature": 20.0},
            None,
        ),
    ]
    assert source.opens == 1
    assert source.closes == 1


def test_metadata_rejects_projected_wide_id_outside_fold_training(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0), "bucket": "known", "value": 1.0},
            {"time": _ts(2), "bucket": "future", "value": 2.0},
        ],
        partition_by=("bucket",),
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(
                id="metric",
                stream="stream",
                field="value",
            )
        ],
        split=TimeSplitConfig(
            intervals=[
                TimeInterval(id="train", until="2024-01-01T02:00:00Z"),
                TimeInterval(id="validation"),
            ],
            folds=[
                DatasetFold(
                    id="holdout",
                    train=["train"],
                    validation=["validation"],
                )
            ],
        ),
    )

    task = SeriesTask()
    result = build_series_artifact(runtime, task)
    runtime.artifacts.register(SERIES, task.output, meta=result.meta)
    manifest_path = runtime.artifacts_root / task.output
    manifest = load_series_manifest(manifest_path)

    assert [tuple(row.features) for row in open_series(manifest_path, manifest)] == [
        ("metric__@bucket:known",),
        ("metric__@bucket:future",),
    ]
    with pytest.raises(
        RuntimeError,
        match=r"fold 'holdout'.*feature series IDs.*metric__@bucket:future",
    ):
        build_metadata_artifact(runtime, MetadataTask(output="metadata.json"))


def test_series_shared_stream_matches_independent_series_pipelines(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rows = [
        {
            "time": _ts(0),
            "exchange": "X",
            "symbol": "A",
            "value": 1.0,
            "volume": 10.0,
        },
        {
            "time": _ts(1),
            "exchange": "X",
            "symbol": "A",
            "value": 3.0,
            "volume": 30.0,
        },
        {
            "time": _ts(0),
            "exchange": "X",
            "symbol": "B",
            "value": 10.0,
            "volume": 100.0,
        },
        {
            "time": _ts(1),
            "exchange": "X",
            "symbol": "B",
            "value": 14.0,
            "volume": 140.0,
        },
    ]
    runtime = _runtime_with_rows(
        tmp_path,
        rows,
        partition_by=("exchange", "symbol"),
    )
    runtime.streams["stream"] = replace(runtime.streams["stream"], presorted=True)
    price = SeriesConfig(
        stream="stream",
        id="price",
        field="value",
        scale=True,
        sequence=SequenceConfig(size=2),
    )
    volume = TargetSeriesConfig(
        stream="stream",
        id="volume",
        field="volume",
        horizon="0s",
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h", keys=["exchange"]),
        features=[price],
        targets=[volume],
    )
    expected_price = list(
        run_series_pipeline(
            runtime,
            price,
        )
    )
    expected_volume = list(
        run_series_pipeline(
            runtime,
            volume,
        )
    )

    for row in rows:
        row["unpickleable"] = lambda: None

    source = runtime.streams["stream"].source
    assert isinstance(source, _StubSource)
    source.opens = 0
    source.closes = 0
    normal_batch_sort = series_operation.batch_sort
    normal_floor_time = series_operation.floor_time_to_cadence
    floor_calls = 0

    def spilling_batch_sort(items, buffer_bytes, key, progress=None):
        return normal_batch_sort(
            items,
            buffer_bytes=1,
            key=key,
            progress=progress,
        )

    def count_floor_time(timestamp, cadence):
        nonlocal floor_calls
        floor_calls += 1
        return normal_floor_time(timestamp, cadence)

    monkeypatch.setattr(
        series_operation,
        "batch_sort",
        spilling_batch_sort,
    )
    monkeypatch.setattr(
        series_operation,
        "floor_time_to_cadence",
        count_floor_time,
    )

    task = SeriesTask()
    build_series_artifact(runtime, task)
    manifest_path = runtime.artifacts_root / task.output
    manifest = load_series_manifest(manifest_path)
    actual_rows = list(open_series(manifest_path, manifest))

    expected_features: dict[tuple, dict[str, object]] = {}
    for item in expected_price:
        assert isinstance(item, SeriesSequence)
        expected_features.setdefault(
            (item.time, *item.entity_key),
            {},
        )[item.id] = item.values
    expected_targets: dict[tuple, dict[str, object]] = {}
    for item in expected_volume:
        assert not isinstance(item, SeriesSequence)
        expected_targets.setdefault(
            (item.time, *item.entity_key),
            {},
        )[item.id] = item.value
    assert [row.key for row in actual_rows] == sorted(
        expected_features.keys() | expected_targets.keys()
    )
    for row in actual_rows:
        assert row.features == expected_features.get(row.key, {})
        assert row.targets == expected_targets.get(row.key, {})
    assert source.opens == 1
    assert source.closes == 1
    assert floor_calls == len(rows)


def test_series_store_sequence_values_unscaled(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0), "value": 1.0},
            {"time": _ts(1), "value": 11.0},
        ],
    )
    config = SeriesConfig(
        stream="stream",
        id="price",
        field="value",
        scale=True,
        sequence=SequenceConfig(size=2),
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[config],
    )
    task = SeriesTask()
    build_series_artifact(runtime, task)
    manifest_path = runtime.artifacts_root / task.output
    manifest = load_series_manifest(manifest_path)
    [row] = open_series(manifest_path, manifest)

    assert row.time == _ts(1)
    assert row.features == {"price": [1.0, 11.0]}
    assert row.targets == {}


def test_series_artifact_rejects_multiple_sequences_in_one_sample_bucket(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0, 0), "value": 1.0},
            {"time": _ts(0, 15), "value": 2.0},
            {"time": _ts(0, 30), "value": 3.0},
        ],
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(
                stream="stream",
                id="history",
                field="value",
                sequence=SequenceConfig(size=2, stride=1),
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"history.*multiple sequences.*stride.*sample cadence",
    ):
        build_series_artifact(runtime, SeriesTask())


def test_series_artifact_rejects_duplicate_scalars_within_sample_bucket(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0, 0), "value": 1.0},
            {"time": _ts(0, 15), "value": 2.0},
            {"time": _ts(0, 30), "value": 3.0},
        ],
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(
                stream="stream",
                id="price",
                field="value",
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"price.*multiple values.*collect.*sample cadence",
    ):
        build_series_artifact(runtime, SeriesTask())


def test_series_artifact_collects_an_explicit_fixed_size_bucket(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0, 30), "value": 3.0},
            {"time": _ts(1, 15), "value": 5.0},
            {"time": _ts(0, 0), "value": 1.0},
            {"time": _ts(1, 30), "value": 6.0},
            {"time": _ts(0, 15), "value": 2.0},
            {"time": _ts(1, 0), "value": 4.0},
        ],
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(
                stream="stream",
                id="price",
                field="value",
                collect=3,
            )
        ],
    )

    task = SeriesTask()
    build_series_artifact(runtime, task)
    manifest_path = runtime.artifacts_root / task.output
    manifest = load_series_manifest(manifest_path)
    rows = list(open_series(manifest_path, manifest))

    assert [(row.time, row.features) for row in rows] == [
        (_ts(0), {"price": [1.0, 2.0, 3.0]}),
        (_ts(1), {"price": [4.0, 5.0, 6.0]}),
    ]


def test_series_artifact_collect_size_one_remains_a_list(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [{"time": _ts(0), "value": None}],
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(
                stream="stream",
                id="price",
                field="value",
                collect=1,
            )
        ],
    )

    task = SeriesTask()
    build_series_artifact(runtime, task)
    manifest_path = runtime.artifacts_root / task.output
    [row] = open_series(manifest_path)

    assert row.features == {"price": [None]}


def test_collected_series_produces_fixed_list_metadata(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0), "value": 1.0},
            {"time": _ts(0, 30), "value": 2.0},
        ],
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(
                stream="stream",
                id="price",
                field="value",
                collect=2,
            )
        ],
    )
    series_task = SeriesTask()
    series = build_series_artifact(runtime, series_task)
    runtime.artifacts.register(SERIES, series_task.output, meta=series.meta)

    metadata_task = MetadataTask(output="metadata.json")
    build_metadata_artifact(runtime, metadata_task)
    payload = json.loads(
        (runtime.artifacts_root / metadata_task.output).read_text(encoding="utf-8")
    )

    [entry] = payload["catalog"]["features"]
    assert entry["id"] == "price"
    assert entry["kind"] == "list"
    assert entry["length"] == 2
    assert entry["element_types"] == ["float"]


@pytest.mark.parametrize("count", [2, 4])
def test_series_artifact_requires_the_declared_collection_size(
    tmp_path: Path,
    count: int,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [{"time": _ts(0, index * 10), "value": float(index)} for index in range(count)],
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(
                stream="stream",
                id="price",
                field="value",
                collect=3,
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match=rf"price.*collect requires 3 values.*got {count}",
    ):
        build_series_artifact(runtime, SeriesTask())


def test_series_artifact_preserves_a_list_valued_scalar(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [{"time": _ts(0), "value": [1.0, 2.0]}],
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(
                stream="stream",
                id="embedding",
                field="value",
            )
        ],
    )

    task = SeriesTask()
    build_series_artifact(runtime, task)
    manifest_path = runtime.artifacts_root / task.output
    manifest = load_series_manifest(manifest_path)
    [row] = open_series(manifest_path, manifest)

    assert row.features == {"embedding": [1.0, 2.0]}


def test_series_artifact_rejects_collecting_list_valued_records(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0), "value": [1.0, 2.0]},
            {"time": _ts(0, 30), "value": [3.0, 4.0]},
        ],
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(
                stream="stream",
                id="embedding",
                field="value",
                collect=2,
            )
        ],
    )

    with pytest.raises(TypeError, match=r"embedding.*collect requires scalar"):
        build_series_artifact(runtime, SeriesTask())


def test_series_artifact_collects_each_wide_series_independently(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0), "symbol": "AAPL", "value": 1.0},
            {"time": _ts(0), "symbol": "MSFT", "value": 10.0},
            {"time": _ts(1), "symbol": "AAPL", "value": 2.0},
            {"time": _ts(1), "symbol": "MSFT", "value": 20.0},
        ],
        partition_by=("symbol",),
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1d"),
        features=[
            SeriesConfig(
                stream="stream",
                id="price",
                field="value",
                collect=2,
            )
        ],
    )

    task = SeriesTask()
    build_series_artifact(runtime, task)
    manifest_path = runtime.artifacts_root / task.output
    [row] = open_series(manifest_path)

    assert row.features == {
        "price__@symbol:AAPL": [1.0, 2.0],
        "price__@symbol:MSFT": [10.0, 20.0],
    }


def test_series_artifact_collects_targets(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0), "price": 10.0, "return": 0.1},
            {"time": _ts(0, 30), "price": 11.0, "return": 0.2},
        ],
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(
                stream="stream",
                id="price",
                field="price",
                collect=2,
            )
        ],
        targets=[
            TargetSeriesConfig(
                stream="stream",
                id="return",
                field="return",
                horizon="1h",
                collect=2,
            )
        ],
    )

    task = SeriesTask()
    build_series_artifact(runtime, task)
    manifest_path = runtime.artifacts_root / task.output
    [row] = open_series(manifest_path)

    assert row.features == {"price": [10.0, 11.0]}
    assert row.targets == {"return": [0.1, 0.2]}


def test_series_artifact_collect_counts_and_tracks_cadence_placeholders(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0), "value": 1.0},
            {"time": _ts(2, 30), "value": 2.0},
        ],
        transforms=[EnsureCadenceConfig(cadence="30m")],
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(
                stream="stream",
                id="price",
                field="value",
                collect=2,
            )
        ],
    )

    task = SeriesTask()
    build_series_artifact(runtime, task)
    manifest_path = runtime.artifacts_root / task.output
    rows = list(open_series(manifest_path))

    assert [row.features for row in rows] == [
        {"price": [1.0, None]},
        {"price": [None, None]},
        {"price": [None, 2.0]},
    ]
    assert [row.placeholder_ids for row in rows] == [
        frozenset(),
        frozenset({"price"}),
        frozenset(),
    ]


def test_series_manifest_counts_empty_series_from_a_shared_stream(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0), "value": 1.0},
            {"time": _ts(1), "value": 2.0},
        ],
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(
                stream="stream",
                id="window",
                field="value",
                sequence=SequenceConfig(size=3),
            ),
            SeriesConfig(
                stream="stream",
                id="value",
                field="value",
            ),
        ],
    )

    task = SeriesTask()
    build_series_artifact(runtime, task)
    manifest_path = runtime.artifacts_root / task.output
    manifest = load_series_manifest(manifest_path)

    assert [(entry.id, entry.samples) for entry in manifest.features] == [
        ("window", 0),
        ("value", 2),
    ]
    assert all(
        "window" not in row.features for row in open_series(manifest_path, manifest)
    )


def test_series_record_sort_is_part_of_the_observed_stream_pipeline(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [{"time": _ts(0), "value": 1.0}],
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[SeriesConfig(stream="stream", id="value", field="value")],
    )
    observer = _PipelineStarts()
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    progress = Progress(console=console, auto_refresh=False)
    rich_progress = _ExecutionProgress(progress, debug=False)
    rich_renderer = _RichExecutionRenderer(logging.INFO, console, rich_progress)

    def observe(event: PipelineEvent) -> None:
        observer(event)
        rich_renderer.render(event)

    task = SeriesTask()
    with execution_observer(observe):
        build_series_artifact(runtime, task)
    manifest_path = runtime.artifacts_root / task.output
    manifest = load_series_manifest(manifest_path)
    [row] = open_series(manifest_path, manifest)

    assert observer.starts == ["series:artifact", "series:stream"]
    assert "project_series" in observer.nodes
    assert "order_series" in observer.nodes
    assert row.features == {"value": 1.0}
    assert progress.tasks == []
    assert "[series:stream] started" in output.getvalue()
    assert "[series:stream] finished status=success" in output.getvalue()


def test_series_closes_shared_stream_after_feature_error(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [{"time": _ts(0), "value": 1.0}],
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(stream="stream", id="value", field="value"),
            SeriesConfig(stream="stream", id="missing", field="missing"),
        ],
    )
    source = runtime.streams["stream"].source
    assert isinstance(source, _StubSource)

    with pytest.raises(KeyError, match="Record field 'missing'"):
        build_series_artifact(runtime, SeriesTask())

    assert source.opens == 1
    assert source.closes == 1
    assert not (runtime.artifacts_root / "build/series/manifest.json").exists()


def test_failed_series_rebuild_preserves_previous_generation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [
            {"time": _ts(0), "value": 1.0, "other": 10.0},
            {"time": _ts(1), "value": 2.0, "other": 20.0},
        ],
        stream_id="prices",
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(stream="prices", id="value", field="value"),
            SeriesConfig(stream="prices", id="other", field="other"),
        ],
    )
    task = SeriesTask()
    build_series_artifact(runtime, task)
    manifest_path = runtime.artifacts_root / task.output
    previous_manifest = manifest_path.read_bytes()
    previous = load_series_manifest(manifest_path)
    previous_data = manifest_path.parent / previous.path
    previous_generation = Path(previous.path).parts[1]

    def fail_data_write(path, rows):
        for _ in rows:
            break
        raise RuntimeError("series data failed")

    monkeypatch.setattr(
        series_operation,
        "write_series_rows",
        fail_data_write,
    )

    with pytest.raises(RuntimeError, match="series data failed"):
        build_series_artifact(runtime, task)

    assert manifest_path.read_bytes() == previous_manifest
    assert previous_data.is_file()
    assert list(read_series_rows(previous_data)) == list(
        open_series(manifest_path, previous)
    )
    cache_root = manifest_path.parent / "manifest.data"
    assert [path.name for path in cache_root.iterdir()] == [previous_generation]


def test_failed_series_manifest_commit_removes_new_generation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [{"time": _ts(0), "value": 1.0}],
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[SeriesConfig(stream="stream", id="value", field="value")],
    )
    task = SeriesTask()
    build_series_artifact(runtime, task)
    manifest_path = runtime.artifacts_root / task.output
    previous_manifest = manifest_path.read_bytes()
    previous = load_series_manifest(manifest_path)
    previous_path = manifest_path.parent / previous.path

    def fail_manifest(*_args, **_kwargs):
        raise OSError("manifest commit failed")

    monkeypatch.setattr(
        series_operation,
        "write_json_object",
        fail_manifest,
    )

    with pytest.raises(OSError, match="manifest commit failed"):
        build_series_artifact(runtime, task)

    assert manifest_path.read_bytes() == previous_manifest
    previous_generation = previous_path.parent
    assert set(previous_generation.parent.iterdir()) == {previous_generation}


def test_identical_series_rebuild_publishes_a_new_generation(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [{"time": _ts(0), "value": 1.0}],
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[SeriesConfig(stream="stream", id="value", field="value")],
    )
    task = SeriesTask()

    build_series_artifact(runtime, task)
    manifest_path = runtime.artifacts_root / task.output
    first_manifest = load_series_manifest(manifest_path)
    first_path = manifest_path.parent / first_manifest.path

    build_series_artifact(runtime, task)
    second_manifest = load_series_manifest(manifest_path)
    second_path = manifest_path.parent / second_manifest.path

    assert second_path != first_path
    assert first_path.is_file()
    assert len(list(open_series(manifest_path, second_manifest))) == 1


def test_changed_series_rebuild_retains_previous_generation(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [{"time": _ts(0), "value": 1.0}],
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[SeriesConfig(stream="stream", id="value", field="value")],
    )
    task = SeriesTask()

    build_series_artifact(runtime, task)
    manifest_path = runtime.artifacts_root / task.output
    first_manifest = load_series_manifest(manifest_path)
    first_path = manifest_path.parent / first_manifest.path
    first_rows = list(open_series(manifest_path, first_manifest))

    source = runtime.streams["stream"].source
    assert isinstance(source, _StubSource)
    source._rows.append({"time": _ts(1), "value": 2.0})
    build_series_artifact(runtime, task)

    second_manifest = load_series_manifest(manifest_path)
    second_path = manifest_path.parent / second_manifest.path
    assert second_path != first_path
    assert first_path.is_file()
    assert list(read_series_rows(first_path)) == first_rows
    assert len(list(open_series(manifest_path, second_manifest))) == 2

    assert prune_series_cache(
        manifest_path,
        runtime.artifacts_root,
    ) == (first_path.parent,)
    assert not first_path.exists()
    assert second_path.is_file()


def test_series_rebuild_replaces_a_corrupt_generation(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        [{"time": _ts(0), "value": 1.0}],
    )
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[SeriesConfig(stream="stream", id="value", field="value")],
    )
    task = SeriesTask()

    build_series_artifact(runtime, task)
    manifest_path = runtime.artifacts_root / task.output
    manifest = load_series_manifest(manifest_path)
    data_path = manifest_path.parent / manifest.path
    data_path.write_bytes(b"corrupt")

    build_series_artifact(runtime, task)

    rebuilt = load_series_manifest(manifest_path)
    rebuilt_path = manifest_path.parent / rebuilt.path
    assert rebuilt_path != data_path
    assert len(list(open_series(manifest_path, rebuilt))) == 1
    removed = prune_series_cache(manifest_path, runtime.artifacts_root)
    assert removed == (data_path.parent,)


def test_series_rejects_symlinked_output_before_mutation(
    tmp_path: Path,
) -> None:
    artifacts_root = tmp_path / "artifacts"
    redirected = artifacts_root / "redirected"
    redirected.mkdir(parents=True)
    victim = redirected / "manifest.json"
    victim.write_text("keep", encoding="utf-8")
    (artifacts_root / "build").symlink_to(redirected, target_is_directory=True)
    runtime = SimpleNamespace(
        project_yaml=tmp_path / "project.yaml",
        artifacts_root=artifacts_root,
        dataset=DatasetConfig(sample=SampleConfig(cadence="1h")),
        streams={},
    )

    with pytest.raises(ValueError, match="must not resolve through a symlink"):
        build_series_artifact(
            runtime,
            SeriesTask(output="build/manifest.json"),
        )

    assert victim.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "key_plan",
    [None, window_key_plan(_ts(0), _ts(1), "1h")],
)
def test_sample_input_requires_series_artifact(
    tmp_path: Path,
    key_plan,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        rows=[{"time": _ts(0), "value": 1.0}],
    )
    cfg = SeriesConfig(stream="stream", id="price", field="value")

    with pytest.raises(RuntimeError, match="Series artifact is required"):
        list(open_samples(runtime, [cfg.id], key_plan=key_plan))


def test_cached_sample_input_rejects_manifest_cadence_mismatch(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        rows=[{"time": _ts(0), "value": 1.0}],
    )
    cfg = SeriesConfig(stream="stream", id="price", field="value")
    register_series(runtime, [cfg], "1h")
    manifest = runtime.artifacts_root / "build/series/manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["cadence"] = "1d"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="cadence does not match"):
        list(
            open_samples(
                runtime,
                [cfg.id],
            )
        )


def test_cached_sample_input_verifies_manifest_rows(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_rows(
        tmp_path,
        rows=[{"time": _ts(0), "value": 1.0}],
    )
    cfg = SeriesConfig(stream="stream", id="price", field="value")
    register_series(runtime, [cfg], "1h")
    manifest = runtime.artifacts_root / "build/series/manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["rows"] = 2
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="declares 2 rows but contains 1"):
        list(
            open_samples(
                runtime,
                [cfg.id],
            )
        )


def test_cached_sample_input_reads_requested_feature_subset(
    tmp_path: Path,
) -> None:
    rows = [
        {"time": _ts(0), "value": 1.0, "other": 10.0},
        {"time": _ts(1), "value": 2.0, "other": 20.0},
    ]
    runtime = _runtime_with_rows(tmp_path, rows)
    value_cfg = SeriesConfig(stream="stream", id="price", field="value")
    other_cfg = SeriesConfig(stream="stream", id="other", field="other")
    register_series(runtime, [value_cfg, other_cfg], "1h")

    samples = list(
        open_samples(
            runtime,
            [value_cfg.id],
        )
    )

    assert _sample_payload(samples) == [
        ((_ts(0),), {"price": 1.0}, None),
        ((_ts(1),), {"price": 2.0}, None),
    ]


def test_cached_series_rows_close_reader_when_stopped_early(
    tmp_path: Path,
    monkeypatch,
) -> None:
    configs = [
        SeriesConfig(stream="stream", id="a", field="value"),
        SeriesConfig(stream="stream", id="b", field="value"),
    ]
    runtime = _runtime_with_rows(
        tmp_path,
        rows=[
            {"time": _ts(0), "value": 1.0},
            {"time": _ts(1), "value": 2.0},
        ],
    )
    register_series(runtime, configs, "1h")
    closed = False
    opens = 0
    rows = [
        SeriesRow(
            time=_ts(0),
            entity_key=(),
            features={"a": 1.0, "b": 1.0},
            targets={},
            placeholder_ids=frozenset(),
        ),
        SeriesRow(
            time=_ts(1),
            entity_key=(),
            features={"a": 2.0, "b": 2.0},
            targets={},
            placeholder_ids=frozenset(),
        ),
    ]

    class _ClosingRows:
        def __init__(self) -> None:
            self.items = iter(rows)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.items)

        def close(self) -> None:
            nonlocal closed
            closed = True

    def _open_rows(manifest_path, manifest):
        nonlocal opens
        opens += 1
        assert manifest_path.name == "manifest.json"
        assert manifest.rows == 2
        return _ClosingRows()

    monkeypatch.setattr(
        "jerrythomas.pipelines.sample.input.open_series",
        _open_rows,
    )

    samples = open_samples(
        runtime,
        [config.id for config in configs],
    )
    first = next(samples)
    assert first.features.values == {"a": 1.0, "b": 1.0}
    assert first.features.values is rows[0].features
    assert opens == 1
    assert not closed

    samples.close()
    assert closed


def test_cached_sample_input_opens_one_reader_for_many_series(
    tmp_path: Path,
    monkeypatch,
) -> None:
    feature_count = 80
    row = {"time": _ts(0)}
    configs: list[SeriesConfig] = []
    for index in range(feature_count):
        field = f"value_{index}"
        row[field] = float(index)
        configs.append(
            SeriesConfig(
                stream="stream",
                id=f"feature_{index}",
                field=field,
            )
        )

    runtime = _runtime_with_rows(tmp_path, rows=[row])
    register_series(runtime, configs, "1h")
    original_open_series = sample_input.open_series
    opens = 0

    def count_opened_reader(manifest_path, manifest):
        nonlocal opens
        opens += 1
        return original_open_series(manifest_path, manifest)

    monkeypatch.setattr(sample_input, "open_series", count_opened_reader)

    samples = list(
        open_samples(
            runtime,
            [config.id for config in configs],
        )
    )

    assert opens == 1
    assert len(samples) == 1
    assert len(samples[0].features.values) == feature_count
