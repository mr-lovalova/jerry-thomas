from datetime import datetime, timedelta, timezone
from math import log, log1p

import pytest

from jerrythomas.artifacts.fingerprints import calculate_artifact_hashes
from jerrythomas.config.cross_section import OlsResidualConfig, RankScoreConfig
from jerrythomas.config.transforms import DedupeConfig
from jerrythomas.domain.record import TemporalRecord
from jerrythomas.execution.events import PipelineEvent, PipelineStarted
from jerrythomas.execution.observability import execution_observer
from jerrythomas.pipelines.stream.pipeline import (
    build_stream_pipeline,
    run_stream_preview_pipeline,
    run_stream_pipeline,
)
from jerrythomas.plugins import COMBINERS_EP, LOADERS_EP, MAPPERS_EP, PARSERS_EP
from jerrythomas.runtime import (
    AsOfRuntimeStream,
    BroadcastAsOfRuntimeStream,
    BroadcastRuntimeStream,
    CrossSectionRuntimeStream,
    DerivedRuntimeStream,
    SourceRuntimeStream,
)
from jerrythomas.services.project_definition import load_project_definition
from jerrythomas.services.runtime_compiler import compile_runtime
from jerrythomas.sources.loader import BaseDataLoader
from jerrythomas.sources.parser import DataParser
from jerrythomas.sources.source import Source


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
    for directory in (sources_dir, streams_dir, data_dir, tmp_path / "datasets"):
        directory.mkdir()

    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(
        """\
schema_version: 7
artifact_revision: 1
name: runtime-compiler-test
paths:
  sources: sources
  streams: streams
  datasets: datasets
  artifacts: artifacts
""",
        encoding="utf-8",
    )
    (tmp_path / "datasets" / "default.yaml").write_text(
        "version: v1\nsample:\n  rounding: ceil\n  cadence: 1h\nfeatures: []\ntargets: []\n",
        encoding="utf-8",
    )
    return project_yaml, sources_dir, streams_dir, data_dir


def test_streams_sharing_a_source_compile_distinct_runtime_objects(tmp_path) -> None:
    project_yaml, sources_dir, streams_dir, _ = _write_test_project(tmp_path)
    (sources_dir / "shared.yaml").write_text(
        """\
id: shared
parser:
  entrypoint: core.record
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
  entrypoint: core.identity
""",
            encoding="utf-8",
        )

    runtime = compile_runtime(load_project_definition(project_yaml), "default")
    left = runtime.streams["left"]
    right = runtime.streams["right"]

    assert isinstance(left, SourceRuntimeStream)
    assert isinstance(right, SourceRuntimeStream)
    assert isinstance(left.source, Source)
    assert isinstance(right.source, Source)
    assert left.source is not right.source
    assert left.source.loader is not right.source.loader
    assert left.source.parser is not right.source.parser


def test_compiled_runtimes_isolate_mutable_transform_configuration(tmp_path) -> None:
    project_yaml, sources_dir, streams_dir, data_dir = _write_test_project(tmp_path)
    (tmp_path / "datasets" / "default.yaml").write_text(
        "version: v1\nsample: {rounding: ceil, cadence: 1h}\n"
        "features: [{id: price, stream: selected, field: value}]\n",
        encoding="utf-8",
    )
    (sources_dir / "prices.yaml").write_text(
        "id: prices\n"
        "parser: {entrypoint: core.record}\n"
        "loader:\n"
        "  transport: fs\n"
        "  path: data/prices.jsonl\n"
        "  reader: {format: jsonl}\n",
        encoding="utf-8",
    )
    (streams_dir / "prices.yaml").write_text(
        "id: prices\n"
        "from: {source: prices}\n"
        "map: {entrypoint: core.identity}\n"
        "preprocess:\n"
        "  - {operation: where, field: value, operator: in, comparand: [1, 2]}\n"
        "transforms:\n"
        "  - {operation: where, field: value, operator: in, comparand: [1]}\n",
        encoding="utf-8",
    )
    (streams_dir / "selected.yaml").write_text(
        "id: selected\n"
        "from: {stream: prices}\n"
        "transforms:\n"
        "  - {operation: where, field: value, operator: in, comparand: [1, 2]}\n",
        encoding="utf-8",
    )
    (data_dir / "prices.jsonl").write_text(
        '{"time":"2024-01-01T00:00:00Z","value":1}\n'
        '{"time":"2024-01-01T01:00:00Z","value":2}\n'
        '{"time":"2024-01-01T02:00:00Z","value":3}\n',
        encoding="utf-8",
    )
    definition = load_project_definition(project_yaml)
    snapshot = definition.streams.model_dump(mode="json")
    first = compile_runtime(definition, "default")
    second = compile_runtime(definition, "default")
    assert [record.value for record in run_stream_pipeline(second, "selected")] == [1]

    first.streams["prices"].preprocess[0].comparand.append(3)
    first.streams["prices"].transforms[0].comparand.extend([2, 3])
    first.streams["selected"].transforms[0].comparand.append(3)

    assert [record.value for record in run_stream_pipeline(first, "selected")] == [
        1,
        2,
        3,
    ]
    assert [record.value for record in run_stream_pipeline(second, "selected")] == [1]
    assert definition.streams.model_dump(mode="json") == snapshot
    assert (
        calculate_artifact_hashes(
            definition.project,
            definition.datasets,
            definition.streams,
            definition.artifact_graph,
        )
        == definition.artifact_hashes
    )


def test_compiled_runtimes_isolate_nested_plugin_arguments(
    tmp_path, monkeypatch
) -> None:
    project_yaml, sources_dir, streams_dir, data_dir = _write_test_project(tmp_path)
    (tmp_path / "datasets" / "default.yaml").write_text(
        "version: v1\nsample: {rounding: ceil, cadence: 1h}\n"
        "features: [{id: total, stream: combined, field: value}]\n",
        encoding="utf-8",
    )
    (sources_dir / "numbers.yaml").write_text(
        "id: numbers\n"
        "freshness: opaque\n"
        "loader:\n"
        "  entrypoint: numbers\n"
        "  args: {settings: {values: [1]}}\n"
        "parser:\n"
        "  entrypoint: price\n"
        "  args: {settings: {offsets: []}}\n",
        encoding="utf-8",
    )
    (sources_dir / "lookup.yaml").write_text(
        "id: lookup\n"
        "parser: {entrypoint: core.record}\n"
        "loader:\n"
        "  transport: fs\n"
        "  path: data/lookup.jsonl\n"
        "  reader: {format: jsonl}\n",
        encoding="utf-8",
    )
    (streams_dir / "prices.yaml").write_text(
        "id: prices\n"
        "from: {source: numbers}\n"
        "map:\n"
        "  entrypoint: offset_prices\n"
        "  args: {settings: {offsets: []}}\n",
        encoding="utf-8",
    )
    (streams_dir / "lookup.yaml").write_text(
        "id: lookup\nfrom: {source: lookup}\nmap: {entrypoint: core.identity}\n",
        encoding="utf-8",
    )
    (streams_dir / "combined.yaml").write_text(
        "id: combined\n"
        "from: {stream: prices}\n"
        "join: {kind: align, streams: [lookup]}\n"
        "combine:\n"
        "  entrypoint: add_prices\n"
        "  args: {settings: {offsets: []}}\n",
        encoding="utf-8",
    )
    (data_dir / "lookup.jsonl").write_text(
        '{"time":"2024-01-01T00:00:00Z","value":3}\n'
        '{"time":"2024-01-01T01:00:00Z","value":4}\n',
        encoding="utf-8",
    )

    class NumbersLoader(BaseDataLoader):
        def __init__(self, settings):
            settings["values"].append(2)
            self.values = settings["values"]

        def load(self):
            for hour, value in enumerate(self.values):
                yield datetime(2024, 1, 1, hour, tzinfo=timezone.utc), value

    class PriceParser(DataParser):
        def __init__(self, settings):
            settings["offsets"].append(10)
            self.offsets = settings["offsets"]

        def parse(self, raw):
            time, value = raw
            record = TemporalRecord(time=time)
            record.value = value + sum(self.offsets)
            return record

    def offset_prices(records, settings):
        settings["offsets"].append(20)
        for record in records:
            record.value += sum(settings["offsets"])
            yield record

    def add_prices(price, lookup, settings):
        settings["offsets"].append(30)
        record = TemporalRecord(time=price.time)
        record.value = price.value + lookup.value + sum(settings["offsets"])
        return record

    from jerrythomas.services.streams import source as source_factories

    load_entrypoint = source_factories.load_entrypoint
    plugins = {
        (LOADERS_EP, "numbers"): NumbersLoader,
        (PARSERS_EP, "price"): PriceParser,
        (MAPPERS_EP, "offset_prices"): offset_prices,
        (COMBINERS_EP, "add_prices"): add_prices,
    }

    def source_plugin(group, entrypoint):
        plugin = plugins.get((group, entrypoint))
        return plugin if plugin is not None else load_entrypoint(group, entrypoint)

    monkeypatch.setattr(source_factories, "load_entrypoint", source_plugin)
    monkeypatch.setattr(
        "jerrythomas.services.streams.combine.load_entrypoint",
        lambda group, entrypoint: plugins[group, entrypoint],
    )
    definition = load_project_definition(project_yaml)
    snapshot = definition.streams.model_dump(mode="json")

    first = compile_runtime(definition, "default")
    assert definition.streams.model_dump(mode="json") == snapshot
    second = compile_runtime(definition, "default")
    assert definition.streams.model_dump(mode="json") == snapshot
    expected = [
        (datetime(2024, 1, 1, 0, tzinfo=timezone.utc), 64),
        (datetime(2024, 1, 1, 1, tzinfo=timezone.utc), 96),
    ]
    assert [
        (record.time, record.value) for record in run_stream_pipeline(first, "combined")
    ] == expected
    assert [
        (record.time, record.value)
        for record in run_stream_pipeline(second, "combined")
    ] == expected
    assert definition.streams.model_dump(mode="json") == snapshot
    assert (
        calculate_artifact_hashes(
            definition.project,
            definition.datasets,
            definition.streams,
            definition.artifact_graph,
        )
        == definition.artifact_hashes
    )


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
  entrypoint: core.record
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
  entrypoint: core.identity
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

    runtime = compile_runtime(load_project_definition(project_yaml), "default")
    derived = runtime.streams["filtered"]

    assert isinstance(derived, DerivedRuntimeStream)
    assert derived.partition_by == ("ticker",)
    records = list(run_stream_pipeline(runtime, "filtered"))
    assert [(record.time.day, record.ticker, record.value) for record in records] == [
        (2, "A", 2)
    ]


def test_yaml_cross_section_stream_compiles_typed_contract(tmp_path) -> None:
    project_yaml, sources_dir, streams_dir, _ = _write_test_project(tmp_path)
    (sources_dir / "signals.yaml").write_text(
        """\
id: signals.source
parser:
  entrypoint: core.record
loader:
  transport: fs
  path: data/signals.jsonl
  reader:
    format: jsonl
""",
        encoding="utf-8",
    )
    (streams_dir / "signals.yaml").write_text(
        """\
id: signals
from:
  source: signals.source
partition_by: [ticker]
map:
  entrypoint: core.identity
""",
        encoding="utf-8",
    )
    (streams_dir / "neutralized.yaml").write_text(
        """\
id: neutralized
from:
  stream: signals
cross_section:
  - operation: rank_score
    field: signal
    to: signal_rank
    min_samples: 30
  - operation: ols_residual
    y: signal_rank
    x: [liquidity_rank, volatility_rank]
    to: signal_residual
    min_samples: 30
transforms:
  - operation: dedupe
""",
        encoding="utf-8",
    )

    runtime = compile_runtime(load_project_definition(project_yaml), "default")
    stream = runtime.streams["neutralized"]

    assert isinstance(stream, CrossSectionRuntimeStream)
    assert stream.input_stream == "signals"
    assert stream.partition_by == ("ticker",)
    assert stream.cross_section == (
        RankScoreConfig(
            field="signal",
            to="signal_rank",
            min_samples=30,
        ),
        OlsResidualConfig(
            y="signal_rank",
            x=("liquidity_rank", "volatility_rank"),
            to="signal_residual",
            min_samples=30,
        ),
    )
    assert stream.transforms == (DedupeConfig(),)


def test_exact_global_factors_feed_partitioned_rolling_ols_with_gap(
    tmp_path,
    monkeypatch,
) -> None:
    project_yaml, sources_dir, streams_dir, data_dir = _write_test_project(tmp_path)
    positions = tuple(range(1, 26))
    factor_values = {
        "spy": list(positions),
        "hyg": [position**2 for position in positions],
        "lqd": [position**3 for position in positions],
    }
    rows_by_input = {
        factor: [
            f'{{"time":"2025-01-{position:02d}T00:00:00Z","value":{value}}}'
            for position, value in zip(positions, values, strict=True)
        ]
        for factor, values in factor_values.items()
    }
    rows_by_input["stocks"] = [
        (
            f'{{"time":"2025-01-{position:02d}T00:00:00Z",'
            f'"ticker":"{ticker}","stock_return":'
            f"{intercept + 2 * position - 3 * position**2 + 0.5 * position**3}"
            "}"
        )
        for ticker, intercept in (("A", 10), ("B", -5))
        for position in reversed(positions)
    ]

    for input_name, rows in rows_by_input.items():
        (data_dir / f"{input_name}.jsonl").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        (sources_dir / f"{input_name}.yaml").write_text(
            f"""\
id: {input_name}.source
parser:
  entrypoint: core.record
loader:
  transport: fs
  path: data/{input_name}.jsonl
  reader:
    format: jsonl
""",
            encoding="utf-8",
        )

    (streams_dir / "stocks.yaml").write_text(
        """\
id: stocks
from: {source: stocks.source}
partition_by: [ticker]
map: {entrypoint: core.identity}
""",
        encoding="utf-8",
    )
    for factor in factor_values:
        (streams_dir / f"{factor}.yaml").write_text(
            f"""\
id: {factor}
from: {{source: {factor}.source}}
map: {{entrypoint: core.identity}}
""",
            encoding="utf-8",
        )
    (streams_dir / "factors.yaml").write_text(
        """\
id: factors
from:
  stream: spy
join:
  kind: align
  streams: [hyg, lqd]
combine:
  entrypoint: combine_factors
""",
        encoding="utf-8",
    )
    (streams_dir / "enriched.yaml").write_text(
        """\
id: enriched
from:
  stream: stocks
join:
  kind: broadcast
  with: factors
combine:
  entrypoint: attach_factors
transforms:
  - operation: rolling_ols
    y: stock_return
    x: [spy_return, hyg_return, lqd_return]
    window: 4
    coefficient: hyg_return
    to: hyg_beta_raw
  - operation: lag
    field: hyg_beta_raw
    periods: 21
    to: hyg_beta
""",
        encoding="utf-8",
    )

    def combine_factors(spy, hyg, lqd):
        record = TemporalRecord(time=spy.time)
        record.spy_return = spy.value
        record.hyg_return = hyg.value
        record.lqd_return = lqd.value
        return record

    def attach_factors(stock, factors):
        record = TemporalRecord(time=stock.time)
        record.ticker = stock.ticker
        record.stock_return = stock.stock_return
        record.spy_return = factors.spy_return
        record.hyg_return = factors.hyg_return
        record.lqd_return = factors.lqd_return
        return record

    combiners = {
        "combine_factors": combine_factors,
        "attach_factors": attach_factors,
    }
    monkeypatch.setattr(
        "jerrythomas.services.streams.combine.load_entrypoint",
        lambda group, entrypoint: combiners[entrypoint],
    )

    runtime = compile_runtime(load_project_definition(project_yaml), "default")
    records = list(run_stream_pipeline(runtime, "enriched"))

    assert [record.ticker for record in records] == ["A"] * 25 + ["B"] * 25
    for offset in (0, 25):
        partition = records[offset : offset + 25]
        assert [record.hyg_beta for record in partition[:24]] == [None] * 24
        assert partition[24].hyg_beta == pytest.approx(-3.0)


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
  entrypoint: core.record
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
map: {entrypoint: core.identity}
""",
        encoding="utf-8",
    )
    (streams_dir / "reference.yaml").write_text(
        """\
id: reference
from: {source: reference.source}
map: {entrypoint: core.identity}
""",
        encoding="utf-8",
    )
    (streams_dir / "enriched.yaml").write_text(
        """\
id: enriched
from:
  stream: measurements
join:
  kind: broadcast
  with: reference
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
        "jerrythomas.services.streams.combine.load_entrypoint",
        lambda group, entrypoint: attach_reference,
    )

    runtime = compile_runtime(load_project_definition(project_yaml), "default")
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
  entrypoint: core.record
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
{partition}map: {{entrypoint: core.identity}}
""",
            encoding="utf-8",
        )

    (streams_dir / "reported.yaml").write_text(
        """\
id: reported
from:
  stream: prices
join:
  kind: as_of
  lookup: reports
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
join:
  kind: broadcast_as_of
  lookup: factors
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
        "jerrythomas.services.streams.combine.load_entrypoint",
        lambda group, entrypoint: combiners[entrypoint],
    )

    runtime = compile_runtime(load_project_definition(project_yaml), "default")
    reported = runtime.streams["reported"]
    factor_adjusted = runtime.streams["factor_adjusted"]

    assert isinstance(reported, AsOfRuntimeStream)
    assert reported.partition_by == ("ticker",)
    assert reported.max_age == timedelta(days=1)
    assert reported.require_match is False
    assert reported.direction == "backward"
    assert isinstance(factor_adjusted, BroadcastAsOfRuntimeStream)
    assert factor_adjusted.partition_by == ("ticker",)
    assert factor_adjusted.max_age == timedelta(days=2)
    assert factor_adjusted.require_match is True
    assert factor_adjusted.direction == "backward"

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


def test_yaml_aggregate_exact_optional_alignment_and_literal_fill(
    tmp_path,
    monkeypatch,
) -> None:
    project_yaml, sources_dir, streams_dir, data_dir = _write_test_project(tmp_path)
    rows_by_input = {
        "prices": [
            '{"time":"2025-01-01T00:00:00Z","ticker":"A","value":10}',
            '{"time":"2025-01-02T00:00:00Z","ticker":"A","value":20}',
            '{"time":"2025-01-03T00:00:00Z","ticker":"A","value":30}',
            '{"time":"2025-01-01T00:00:00Z","ticker":"B","value":40}',
            '{"time":"2025-01-02T00:00:00Z","ticker":"B","value":50}',
        ],
        "events": [
            '{"time":"2025-01-03T00:00:00Z","ticker":"A","value":4}',
            '{"time":"2025-01-01T00:00:00Z","ticker":"A","value":1.5}',
            '{"time":"2025-01-01T00:00:00Z","ticker":"A","value":2}',
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
  entrypoint: core.record
loader:
  transport: fs
  path: data/{input_name}.jsonl
  reader:
    format: jsonl
""",
            encoding="utf-8",
        )

    (streams_dir / "prices.yaml").write_text(
        """\
id: prices
from: {source: prices.source}
map: {entrypoint: core.identity}
partition_by: [ticker]
""",
        encoding="utf-8",
    )
    (streams_dir / "events.yaml").write_text(
        """\
id: events
from: {source: events.source}
map: {entrypoint: core.identity}
partition_by: [ticker]
transforms:
  - {operation: aggregate_sum, field: value, count_to: event_count}
""",
        encoding="utf-8",
    )
    (streams_dir / "enriched.yaml").write_text(
        """\
id: enriched
from:
  stream: prices
join:
  kind: as_of
  lookup: events
  max_age: 0s
  require_match: false
combine:
  entrypoint: attach_events
transforms:
  - {operation: fill_missing, field: event_value, value: 0}
  - {operation: fill_missing, field: event_count, value: 0}
""",
        encoding="utf-8",
    )

    def attach_events(price, event):
        record = TemporalRecord(time=price.time)
        record.ticker = price.ticker
        record.price = price.value
        record.event_value = None if event is None else event.value
        record.event_count = None if event is None else event.event_count
        return record

    monkeypatch.setattr(
        "jerrythomas.services.streams.combine.load_entrypoint",
        lambda group, entrypoint: attach_events,
    )

    runtime = compile_runtime(load_project_definition(project_yaml), "default")
    enriched = runtime.streams["enriched"]

    assert isinstance(enriched, AsOfRuntimeStream)
    assert enriched.max_age == timedelta(0)
    assert enriched.require_match is False

    output = list(run_stream_pipeline(runtime, "enriched"))

    assert [
        (
            record.time.day,
            record.price,
            record.event_value,
            record.event_count,
        )
        for record in output
    ] == [
        (1, 10, 3.5, 2),
        (2, 20, 0, 0),
        (3, 30, 4.0, 1),
        (1, 40, 0, 0),
        (2, 50, 0, 0),
    ]

    (data_dir / "events.jsonl").write_text("", encoding="utf-8")
    empty_output = list(run_stream_pipeline(runtime, "enriched"))

    assert len(empty_output) == 5
    assert [record.ticker for record in empty_output] == ["A", "A", "A", "B", "B"]
    assert all(record.event_value == 0 for record in empty_output)
    assert all(record.event_count == 0 for record in empty_output)


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
  entrypoint: core.record
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
  entrypoint: core.identity
""",
            encoding="utf-8",
        )

    (streams_dir / "combined.yaml").write_text(
        """\
id: combined
from:
  stream: left
join:
  kind: align
  streams:
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
        "jerrythomas.services.streams.combine.load_entrypoint",
        load_mapper,
    )

    runtime = compile_runtime(load_project_definition(project_yaml), "default")
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

    assert runtime.streams["combined"].partition_by == ("ticker",)
    with execution_observer(observer):
        records = list(run_stream_pipeline(runtime, "combined"))
    assert [(record.time.day, record.ticker, record.value) for record in records] == [
        (1, "A", 115),
        (2, "A", 225),
    ]
    assert observer.started == ["stream:combined"]


def test_yaml_forward_as_of_stream_compiles_and_runs(tmp_path, monkeypatch) -> None:
    project_yaml, sources_dir, streams_dir, data_dir = _write_test_project(tmp_path)
    rows_by_input = {
        "prices": [
            '{"time":"2025-01-02T00:00:00Z","ticker":"A","value":2}',
            '{"time":"2025-01-04T00:00:00Z","ticker":"A","value":4}',
        ],
        "reports": [
            '{"time":"2025-01-05T00:00:00Z","ticker":"A","value":500}',
            '{"time":"2025-01-07T00:00:00Z","ticker":"A","value":700}',
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
  entrypoint: core.record
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
from: {{source: {input_name}.source}}
partition_by: [ticker]
map: {{entrypoint: core.identity}}
""",
            encoding="utf-8",
        )
    (streams_dir / "next_report.yaml").write_text(
        """\
id: next_report
from:
  stream: prices
join:
  kind: as_of
  lookup: reports
  direction: forward
  require_match: false
combine:
  entrypoint: attach_next_report
""",
        encoding="utf-8",
    )

    def attach_next_report(price, report):
        record = TemporalRecord(time=price.time)
        record.ticker = price.ticker
        record.price = price.value
        record.next_report = None if report is None else report.value
        return record

    monkeypatch.setattr(
        "jerrythomas.services.streams.combine.load_entrypoint",
        lambda group, entrypoint: attach_next_report,
    )

    runtime = compile_runtime(load_project_definition(project_yaml), "default")
    stream = runtime.streams["next_report"]
    assert isinstance(stream, AsOfRuntimeStream)
    assert stream.direction == "forward"

    records = list(run_stream_preview_pipeline(runtime, "next_report", "canonical"))
    assert [(record.price, record.next_report) for record in records] == [
        (2, 500),
        (4, 500),
    ]


@pytest.mark.parametrize("presorted", [None, False, True])
@pytest.mark.parametrize("partition_by", [[], ["ticker"], ["venue", "ticker"]])
def test_source_ordering_declaration_reaches_runtime(
    tmp_path, presorted, partition_by
) -> None:
    import json

    project_yaml, sources_dir, streams_dir, data_dir = _write_test_project(tmp_path)
    (sources_dir / "prices.yaml").write_text(
        "id: prices\n"
        "parser: {entrypoint: core.record}\n"
        "loader:\n"
        "  transport: fs\n"
        "  path: data/prices.jsonl\n"
        "  reader: {format: jsonl}\n",
        encoding="utf-8",
    )
    declaration = "" if presorted is None else f"presorted: {str(presorted).lower()}\n"
    (streams_dir / "prices.yaml").write_text(
        "id: prices\n"
        "from: {source: prices}\n"
        "map: {entrypoint: core.identity}\n"
        f"partition_by: {json.dumps(partition_by)}\n" + declaration,
        encoding="utf-8",
    )
    (data_dir / "prices.jsonl").write_text(
        '{"time":"2024-01-02T00:00:00Z","venue":"X","ticker":"A","value":2}\n'
        '{"time":"2024-01-01T00:00:00Z","venue":"X","ticker":"A","value":1}\n',
        encoding="utf-8",
    )
    runtime = compile_runtime(load_project_definition(project_yaml), "default")
    records = run_stream_pipeline(runtime, "prices")
    if presorted:
        with pytest.raises(ValueError, match="violates presorted order"):
            list(records)
    else:
        assert [record.value for record in records] == [1, 2]
