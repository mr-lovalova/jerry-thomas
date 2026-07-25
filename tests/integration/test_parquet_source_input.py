import json
from datetime import datetime

import pyarrow as arrow
import pyarrow.compute as compute
import pyarrow.parquet as parquet

from datapipeline.config.profiles.output import ServeOutputConfig
from tests.helpers.regression import read_jsonl, serve_dataset


def test_parquet_source_matches_jsonl_source_through_dataset_pipeline(
    copy_fixture,
) -> None:
    project_root = copy_fixture("regression_project")
    baseline_request = serve_dataset(project_root)
    baseline_path = (
        baseline_request.serve_run_plans[0].paths.dataset_dir / "dataset.jsonl"
    )
    expected = read_jsonl(baseline_path)

    jsonl_path = project_root / "data" / "linear_hourly.jsonl"
    rows = [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["time"] = datetime.fromisoformat(row["time"])
    parquet_path = jsonl_path.with_suffix(".parquet")
    parquet.write_table(arrow.Table.from_pylist(rows), parquet_path)

    source_path = project_root / "sources" / "metrics.linear.yaml"
    source_path.write_text(
        source_path.read_text(encoding="utf-8")
        .replace("format: jsonl", "format: parquet")
        .replace("data/linear_hourly.jsonl", "data/linear_hourly.parquet"),
        encoding="utf-8",
    )

    parquet_request = serve_dataset(project_root, "FORCE")
    result_path = parquet_request.serve_run_plans[0].paths.dataset_dir / "dataset.jsonl"
    assert read_jsonl(result_path) == expected


def test_samples_parquet_can_be_derived_and_reingested(copy_fixture) -> None:
    project_root = copy_fixture("identity_alignment_project")
    export_request = serve_dataset(
        project_root,
        "FORCE",
        ServeOutputConfig(
            transport="fs",
            format="parquet",
            directory=project_root / "research-export",
        ),
        preview="samples",
    )
    samples_path = (
        export_request.serve_run_plans[0].paths.dataset_dir / "dataset.parquet"
    )
    samples = parquet.read_table(
        samples_path,
        columns=[
            "sample.time",
            "sample.ticker",
            "features.price_mean_2",
        ],
    )

    research_directory = project_root / "research"
    research_directory.mkdir()
    derived_path = research_directory / "price_mean_signal.parquet"
    parquet.write_table(
        arrow.table(
            {
                "time": samples["sample.time"],
                "ticker": samples["sample.ticker"],
                "value": compute.multiply(
                    samples["features.price_mean_2"],
                    10.0,
                ),
            }
        ),
        derived_path,
    )

    (project_root / "sources" / "research.price-mean-signal.yaml").write_text(
        """\
id: research.price-mean-signal
parser:
  entrypoint: core.temporal_record
loader:
  transport: fs
  path: research/price_mean_signal.parquet
  reader:
    format: parquet
""",
        encoding="utf-8",
    )
    (project_root / "streams" / "research.price-mean-signal.yaml").write_text(
        """\
id: research.price-mean-signal
from:
  source: research.price-mean-signal
map:
  entrypoint: identity
partition_by: [ticker]
""",
        encoding="utf-8",
    )
    (project_root / "dataset.yaml").write_text(
        """\
sample:
  cadence: 1d
  keys: [ticker]
features:
  - id: researched_signal
    stream: research.price-mean-signal
    field: value
""",
        encoding="utf-8",
    )

    result_request = serve_dataset(project_root, "FORCE")
    result_path = result_request.serve_run_plans[0].paths.dataset_dir / "dataset.jsonl"
    rows = read_jsonl(result_path)

    assert [
        (row["key"], row["features"]["values"]["researched_signal"]) for row in rows
    ] == [
        (["2024-01-01 00:00:00+00:00", "A"], 20.0),
        (["2024-01-01 00:00:00+00:00", "B"], 100.0),
        (["2024-01-02 00:00:00+00:00", "A"], 30.0),
        (["2024-01-02 00:00:00+00:00", "B"], 150.0),
        (["2024-01-03 00:00:00+00:00", "A"], 50.0),
        (["2024-01-03 00:00:00+00:00", "B"], 250.0),
    ]
