from datetime import timedelta
from math import log, log1p

from datapipeline.domain.record import TemporalRecord
from datapipeline.execution.events import PipelineEvent, PipelineStarted
from datapipeline.pipelines.stream.pipeline import (
    build_stream_pipeline,
    run_stream_preview_pipeline,
    run_stream_pipeline,
)
from datapipeline.plugins import COMBINERS_EP
from datapipeline.runtime import (
    AsOfRuntimeStream,
    BroadcastAsOfRuntimeStream,
    BroadcastRuntimeStream,
    DerivedRuntimeStream,
    SourceRuntimeStream,
)
from datapipeline.services.project_definition import load_project_definition
from datapipeline.services.runtime_compiler import compile_runtime
from datapipeline.sources.source import Source


class _PipelineObserver:
    def __init__(self) -> None:
        self.started: list[str] = []

    def __call__(self, event: PipelineEvent) -> None:
        if isinstance(event, PipelineStarted):
            self.started.append(event.pipeline_name)


def _write_test_project(tmp_path):
    sources_dir = tmp_path / "sources"
    streams_dir = tmp_path / "streams"
    data_dir = tmp_path / "data"
    for directory in (sources_dir, streams_dir, data_dir):
        directory.mkdir()

    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(
        """\
schema_version: 4
artifact_revision: 1
name: runtime-compiler-test
paths:
  sources: sources
  streams: streams
  dataset: dataset.yaml
  artifacts: artifacts
""",
        encoding="utf-8",
    )
    (tmp_path / "dataset.yaml").write_text(
        "sample:\n  cadence: 1h\nfeatures: []\ntargets: []\n",
        encoding="utf-8",
    )
    return project_yaml, sources_dir, streams_dir, data_dir


def test_streams_sharing_a_source_compile_distinct_runtime_objects(tmp_path) -> None:
    project_yaml, sources_dir, streams_dir, _ = _write_test_project(tmp_path)
    (sources_dir / "shared.yaml").write_text(
        """\
id: shared
parser:
  entrypoint: core.temporal_record
loader:
  transport: fs
  path: data.jsonl
  reader:
    format: jsonl
""",
        encoding="utf-8",
    )
    for stream_id in ("left", "right"):
        (streams_dir / f"{stream_id}.yaml").write_text(
            f"""\
id: {stream_id}
from:
  source: shared
map:
  entrypoint: identity
""",
            encoding="utf-8",
        )

    runtime = compile_runtime(load_project_definition(project_yaml))
    left = runtime.streams["left"]
    right = runtime.streams["right"]

    assert isinstance(left, SourceRuntimeStream)
    assert isinstance(right, SourceRuntimeStream)
    assert isinstance(left.source, Source)
    assert isinstance(right.source, Source)
    assert left.source is not right.source
    assert left.source.loader is not right.source.loader
    assert left.source.parser is not right.source.parser


def test_yaml_derived_stream_runs_with_inherited_partition(tmp_path) -> None:
    project_yaml, sources_dir, streams_dir, data_dir = _write_test_project(tmp_path)
    (data_dir / "prices.jsonl").write_text(
        "\n".join(
            [
                '{"time":"2025-01-02T00:00:00Z","ticker":"A","value":2}',
                '{"time":"2025-01-01T00:00:00Z","ticker":"A","value":1}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (sources_dir / "prices.yaml").write_text(
        """\
id: prices.source
parser:
  entrypoint: core.temporal_record
loader:
  transport: fs
  path: data/prices.jsonl
  reader:
    format: jsonl
""",
        encoding="utf-8",
    )
    (streams_dir / "prices.yaml").write_text(
        """\
id: prices
from:
  source: prices.source
partition_by: [ticker]
map:
  entrypoint: identity
""",
        encoding="utf-8",
    )
    (streams_dir / "filtered.yaml").write_text(
        """\
id: filtered
from:
  stream: prices
transforms:
  - {operation: where, field: value, operator: gt, comparand: 1}
""",
        encoding="utf-8",
    )

    runtime = compile_runtime(load_project_definition(project_yaml))
    derived = runtime.streams["filtered"]

    assert isinstance(derived, DerivedRuntimeStream)
    assert derived.partition_by == ("ticker",)
    records = list(run_stream_pipeline(runtime, "filtered"))
    assert [(record.time.day, record.ticker, record.value) for record in records] == [
        (2, "A", 2)
    ]


def test_yaml_broadcast_stream_reuses_exact_input_across_partitions(
    tmp_path,
    monkeypatch,
) -> None:
    project_yaml, sources_dir, streams_dir, data_dir = _write_test_project(tmp_path)
    rows_by_input = {
        "measurements": [
            '{"time":"2025-01-02T00:00:00Z","station":"A","value":2}',
            '{"time":"2025-01-01T00:00:00Z","station":"B","value":3}',
            '{"time":"2025-01-01T00:00:00Z","station":"A","value":1}',
            '{"time":"2025-01-02T00:00:00Z","station":"B","value":4}',
        ],
        "reference": [
            '{"time":"2025-01-02T00:00:00Z","value":20}',
            '{"time":"2025-01-01T00:00:00Z","value":10}',
        ],
    }
    for input_name, rows in rows_by_input.items():
        (data_dir / f"{input_name}.jsonl").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        (sources_dir / f"{input_name}.yaml").write_text(
            f"""\
id: {input_name}.source
parser:
  entrypoint: core.temporal_record
loader:
  transport: fs
  path: data/{input_name}.jsonl
  reader:
    format: jsonl
""",
            encoding="utf-8",
        )
    (streams_dir / "measurements.yaml").write_text(
        """\
id: measurements
from: {source: measurements.source}
partition_by: [station]
map: {entrypoint: identity}
""",
        encoding="utf-8",
    )
    (streams_dir / "reference.yaml").write_text(
        """\
id: reference
from: {source: reference.source}
map: {entrypoint: identity}
""",
        encoding="utf-8",
    )
    (streams_dir / "enriched.yaml").write_text(
        """\
id: enriched
from:
  stream: measurements
  broadcast: reference
combine:
  entrypoint: attach_reference
  args: {offset: 2}
transforms:
  - {operation: derive, left: value, operator: mul, right_value: 2, to: doubled}
  - {operation: log, field: reference_value, to: log_reference}
  - {operation: log1p, field: measurement_value, to: log_measurement}
  - {operation: rolling_slope, x: reference_value, y: measurement_value, window: 2, to: slope}
  - {operation: forward_sum, field: measurement_value, window: 1, to: future_measurement}
""",
        encoding="utf-8",
    )

    def attach_reference(measurement, reference, offset):
        record = TemporalRecord(time=measurement.time)
        record.station = measurement.station
        record.measurement_value = measurement.value
        record.reference_value = reference.value
        record.value = measurement.value + reference.value + offset
        return record

    monkeypatch.setattr(
        "datapipeline.services.streams.combine.load_entrypoint",
        lambda group, entrypoint: attach_reference,
    )

    runtime = compile_runtime(load_project_definition(project_yaml))
    enriched = runtime.streams["enriched"]

    assert isinstance(enriched, BroadcastRuntimeStream)
    assert enriched.input_stream == "measurements"
    assert enriched.broadcast_stream == "reference"
    assert enriched.partition_by == ("station",)
    assert len(enriched.transforms) == 5

    records = list(run_stream_pipeline(runtime, "enriched"))
    assert [
        (
            record.station,
            record.time.day,
            record.value,
            record.doubled,
            record.log_reference,
            record.log_measurement,
            record.slope,
            record.future_measurement,
        )
        for record in records
    ] == [
        ("A", 1, 13, 26, log(10), log1p(1), None, 2.0),
        ("A", 2, 24, 48, log(20), log1p(2), 0.1, None),
        ("B", 1, 15, 30, log(10), log1p(3), None, 4.0),
        ("B", 2, 26, 52, log(20), log1p(4), 0.1, None),
    ]


def test_yaml_as_of_streams_compile_and_run(
    tmp_path,
    monkeypatch,
) -> None:
    project_yaml, sources_dir, streams_dir, data_dir = _write_test_project(tmp_path)
    rows_by_input = {
        "prices": [
            '{"time":"2025-01-04T00:00:00Z","ticker":"B","value":40}',
            '{"time":"2025-01-02T00:00:00Z","ticker":"A","value":2}',
            '{"time":"2025-01-04T00:00:00Z","ticker":"A","value":4}',
            '{"time":"2025-01-02T00:00:00Z","ticker":"B","value":20}',
        ],
        "reports": [
            '{"time":"2025-01-01T00:00:00Z","ticker":"B","value":1000}',
            '{"time":"2025-01-04T00:00:00Z","ticker":"A","value":400}',
            '{"time":"2025-01-01T00:00:00Z","ticker":"A","value":100}',
        ],
        "factors": [
            '{"time":"2025-01-03T00:00:00Z","value":30}',
            '{"time":"2025-01-01T00:00:00Z","value":10}',
        ],
    }
    for input_name, rows in rows_by_input.items():
        (data_dir / f"{input_name}.jsonl").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        (sources_dir / f"{input_name}.yaml").write_text(
            f"""\
id: {input_name}.source
parser:
  entrypoint: core.temporal_record
loader:
  transport: fs
  path: data/{input_name}.jsonl
  reader:
    format: jsonl
""",
            encoding="utf-8",
        )
        partition = "partition_by: [ticker]\n" if input_name != "factors" else ""
        (streams_dir / f"{input_name}.yaml").write_text(
            f"""\
id: {input_name}
from: {{source: {input_name}.source}}
{partition}map: {{entrypoint: identity}}
""",
            encoding="utf-8",
        )

    (streams_dir / "reported.yaml").write_text(
        """\
id: reported
from:
  stream: prices
  as_of: reports
max_age: 1d
require_match: false
combine:
  entrypoint: attach_report
transforms:
  - {operation: derive, left: price, operator: mul, right_value: 2, to: doubled}
""",
        encoding="utf-8",
    )
    (streams_dir / "factor_adjusted.yaml").write_text(
        """\
id: factor_adjusted
from:
  stream: prices
  broadcast_as_of: factors
max_age: 2d
combine:
  entrypoint: attach_factor
""",
        encoding="utf-8",
    )

    def attach_report(price, report):
        record = TemporalRecord(time=price.time)
        record.ticker = price.ticker
        record.price = price.value
        record.report = None if report is None else report.value
        return record

    def attach_factor(price, factor):
        record = TemporalRecord(time=price.time)
        record.ticker = price.ticker
        record.price = price.value
        record.factor = factor.value
        return record

    combiners = {
        "attach_report": attach_report,
        "attach_factor": attach_factor,
    }
    monkeypatch.setattr(
        "datapipeline.services.streams.combine.load_entrypoint",
        lambda group, entrypoint: combiners[entrypoint],
    )

    runtime = compile_runtime(load_project_definition(project_yaml))
    reported = runtime.streams["reported"]
    factor_adjusted = runtime.streams["factor_adjusted"]

    assert isinstance(reported, AsOfRuntimeStream)
    assert reported.partition_by == ("ticker",)
    assert reported.max_age == timedelta(days=1)
    assert reported.require_match is False
    assert isinstance(factor_adjusted, BroadcastAsOfRuntimeStream)
    assert factor_adjusted.partition_by == ("ticker",)
    assert factor_adjusted.max_age == timedelta(days=2)
    assert factor_adjusted.require_match is True

    reported_canonical = list(
        run_stream_preview_pipeline(runtime, "reported", "canonical")
    )
    assert [record.report for record in reported_canonical] == [100, 400, 1000, None]

    reported_records = list(run_stream_pipeline(runtime, "reported"))
    assert [
        (record.ticker, record.time.day, record.price, record.report, record.doubled)
        for record in reported_records
    ] == [
        ("A", 2, 2, 100, 4),
        ("A", 4, 4, 400, 8),
        ("B", 2, 20, 1000, 40),
        ("B", 4, 40, None, 80),
    ]

    factor_records = list(
        run_stream_preview_pipeline(runtime, "factor_adjusted", "canonical")
    )
    assert [
        (record.ticker, record.time.day, record.price, record.factor)
        for record in factor_records
    ] == [
        ("A", 2, 2, 10),
        ("A", 4, 4, 30),
        ("B", 2, 20, 10),
        ("B", 4, 40, 30),
    ]


def test_yaml_aligned_stream_runs_with_inherited_partition_and_combiner(
    tmp_path,
    monkeypatch,
) -> None:
    project_yaml, sources_dir, streams_dir, data_dir = _write_test_project(tmp_path)

    records_by_input = {
        "left": [
            '{"time":"2025-01-02T00:00:00Z","ticker":"A","value":2}',
            '{"time":"2025-01-01T00:00:00Z","ticker":"A","value":1}',
        ],
        "right": [
            '{"time":"2025-01-01T00:00:00Z","ticker":"A","value":10}',
            '{"time":"2025-01-02T00:00:00Z","ticker":"A","value":20}',
        ],
    }
    for input_name, rows in records_by_input.items():
        (data_dir / f"{input_name}.jsonl").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        (sources_dir / f"{input_name}.yaml").write_text(
            f"""\
id: {input_name}.source
parser:
  entrypoint: core.temporal_record
loader:
  transport: fs
  path: data/{input_name}.jsonl
  reader:
    format: jsonl
""",
            encoding="utf-8",
        )
        (streams_dir / f"{input_name}.yaml").write_text(
            f"""\
id: {input_name}
from:
  source: {input_name}.source
partition_by: [ticker]
map:
  entrypoint: identity
""",
            encoding="utf-8",
        )

    (streams_dir / "combined.yaml").write_text(
        """\
id: combined
from:
  align:
    - left
    - right
combine:
  entrypoint: combine
  args:
    offset: 5
""",
        encoding="utf-8",
    )

    def combine(left, right, offset):
        record = TemporalRecord(time=left.time)
        record.ticker = left.ticker
        record.value = left.value * 100 + right.value + offset
        return record

    def load_mapper(group, entrypoint):
        assert group == COMBINERS_EP
        assert entrypoint == "combine"
        return combine

    monkeypatch.setattr(
        "datapipeline.services.streams.combine.load_entrypoint",
        load_mapper,
    )

    runtime = compile_runtime(load_project_definition(project_yaml))
    pipeline = build_stream_pipeline(runtime, "combined")
    assert pipeline.name == "stream:combined"
    assert pipeline.input.name == "align_inputs"
    assert [stage.name for stage in pipeline.stages] == ["combine_records"]
    assert pipeline.input.progress is None
    assert [stage.progress is not None for stage in pipeline.stages] == [False]
    input_rows = list(run_stream_preview_pipeline(runtime, "combined", "input"))
    assert [[record.value for record in row] for row in input_rows] == [
        [1, 10],
        [2, 20],
    ]
    canonical = list(run_stream_preview_pipeline(runtime, "combined", "canonical"))
    assert [record.value for record in canonical] == [115, 225]

    observer = _PipelineObserver()
    runtime.pipeline_observer = observer

    assert runtime.streams["combined"].partition_by == ("ticker",)
    records = list(run_stream_pipeline(runtime, "combined"))
    assert [(record.time.day, record.ticker, record.value) for record in records] == [
        (1, "A", 115),
        (2, "A", 225),
    ]
    assert observer.started == ["stream:combined"]
