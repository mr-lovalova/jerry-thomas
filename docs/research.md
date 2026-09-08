# Research Workflow

Jerry can export assembled samples as a schema-aware Parquet table for research
in Polars, NumPy, or another analytical tool. A derived table can then return as
an ordinary Jerry source. This keeps experimentation outside the streaming
runtime while preserving Jerry's validation and reproducible dataset build.

```mermaid
flowchart LR
    A["Jerry canonical pipeline"] --> B["Schema-aware samples Parquet"]
    B --> C["Polars or NumPy research"]
    C --> D["Sorted derived Parquet"]
    D --> E["Jerry source validation and serving"]
```

## 1. Export assembled samples

Install the optional Parquet support and the research tool separately:

```bash
python -m pip install "jerry-thomas[parquet]" polars
```

Export the `samples` preview from one serve profile:

```bash
jerry serve \
  --project path/to/project.yaml \
  --profile dataset \
  --preview samples \
  --output-transport fs \
  --output-format parquet \
  --output-directory research/export
```

Jerry prints the concrete run-scoped output path. The table uses explicit
namespaces such as `sample.time`, `sample.ticker`, `features.adv_20`, and
`targets.forward_return`.

`samples` is the useful research boundary: series have been assembled into
rows, but postprocess filtering, split routing, and fold-specific scaling have
not run. Use `--preview postprocess` when the research input should include the
configured postprocess policy.

### Consume completed runs from Python

In v11, `run_profiles()` returns a tuple of `ServeRunResult` objects after
execution and publication succeed:

```python
from jerrythomas.profiles.orchestration import run_profiles
from jerrythomas.profiles.request_builder import build_runtime_run_request

request = build_runtime_run_request(
    "serve",
    "path/to/project.yaml",
    profile_name="dataset",
)
results = () if request is None else run_profiles(request)
for result in results:
    print(result.paths.run_id, result.paths.run_root)
    for output_path in result.outputs:
        print(output_path)
```

`ServeRunResult` is an immutable dataclass in `jerrythomas.profiles.models`:

- `paths` is the existing `RunPaths` object, including the run ID, run root,
  dataset directory, and metadata path.
- `outputs` is a tuple of filesystem paths actually written, in profile order
  and then output order. A completed empty file is included; an operation that
  returns no output contributes no files.
- `preview` records the preview stage, or `None` for a full run.

Profiles sharing a managed output directory contribute to one result. Separate
run directories produce separate results in plan order. Preview runs return
results without updating `latest`. Build, materialize, inspect, and stdout-only
execution return an empty tuple because they do not create managed serve runs.
Execution and publication failures raise instead of returning partial results.

For subprocess callers, use `jerry serve --result-json`; see
[completed run results](cli.md#completed-run-results). This avoids discovering
runs by comparing directories or selecting the newest timestamp.

## 2. Derive a series with Polars

This example ranks `adv_20` across tickers at each timestamp and writes one
canonical record series:

```python
from pathlib import Path
import sys

import polars as pl


samples_path = Path(sys.argv[1])
output_path = Path("research/adv_rank.parquet")
output_path.parent.mkdir(parents=True, exist_ok=True)

(
    pl.scan_parquet(samples_path)
    .select(
        pl.col("sample.time").alias("time"),
        pl.col("sample.ticker").alias("ticker"),
        pl.col("features.adv_20")
        .rank(method="average")
        .over("sample.time")
        .alias("value"),
    )
    .sort(["ticker", "time"])
    .sink_parquet(output_path)
)
```

Run it with the Parquet path printed by `jerry serve`:

```bash
python research/derive_adv_rank.py /absolute/path/to/dataset.parquet
```

Prefer native Polars expressions over Python row callbacks. Use NumPy,
scikit-learn, or statsmodels alongside Polars for algorithms that are naturally
matrix-based, such as PCA or regression.

## 3. Reingest the derived table

Declare the local Parquet file as a normal source:

```yaml
# sources/research.adv-rank.yaml
id: research.adv-rank
parser:
  entrypoint: core.temporal_record
loader:
  transport: fs
  path: research/adv_rank.parquet
  reader:
    format: parquet
```

Expose its canonical records as a stream:

```yaml
# streams/research.adv-rank.yaml
id: research.adv-rank
from:
  source: research.adv-rank
map:
  entrypoint: identity
partition_by: [ticker]
presorted: true
```

Then reference the field from `dataset.yaml`:

```yaml
features:
  - id: adv_rank
    stream: research.adv-rank
    field: value
```

The Polars script sorts by the stream's canonical order, so `presorted: true` lets
Jerry validate that order in one pass and skip external sorting. If the producer
cannot guarantee canonical order, omit `presorted` or set it to `false`; Jerry will sort the stream.
Never declare `presorted: true` based only on an assumption.

The research producer remains responsible for feature semantics and leakage.
Jerry still validates timestamps, stream ordering, sample identity, metadata,
and downstream dataset contracts when the derived series is reingested.
