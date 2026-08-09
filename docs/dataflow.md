# Data Flow (YAML Reference Chain)

This page shows how config files link together from workspace selection to final output files.
The goal is to make the reference chain explicit and easy to debug.

## End-to-end Reference Chain

```text
jerry.yaml: default_dataset
  -> datasets.<alias> = <path/to/project.yaml>
    -> project.yaml: paths.sources / paths.streams / paths.dataset
      -> sources/*.yaml: id
        -> streams/*.yaml: from.source|from.stream|from.broadcast|
                           from.as_of|from.broadcast_as_of|from.align, id
          -> dataset.yaml: stream: <streams.id>, field: <record_field>
            -> jerry serve
              -> runs/<run_id>/dataset/<profile>.jsonl|csv|parquet|...
              -> runs/<run_id>/dataset/<profile>.<fold>.<role>.jsonl|csv|parquet|... when dataset.split is configured
```

## 1) Workspace selects dataset project

`jerry.yaml` picks which `project.yaml` to run when `--dataset`/`--project` is omitted.

```yaml
plugin_root: demo
datasets:
  demo: demo/demo/project.yaml
default_dataset: demo
```

Expected behavior:
- `jerry serve` resolves to `datasets.demo`.
- Relative paths here are resolved from the workspace root (directory containing `jerry.yaml`).

## 2) Project maps to config folders/files

`project.yaml` is the root map for all dataset config.

```yaml
schema_version: 5
artifact_revision: 1
paths:
  sources: ./sources
  streams: ./streams
  dataset: dataset.yaml
  artifacts: ../artifacts/${project_name}
globals:
  cadence: 1d
```

Expected behavior:
- `schema_version` declares the `project.yaml` format Jerry must support.
- `artifact_revision` controls artifact cache invalidation independently of the
  project format.
- All relative `paths.*` values are resolved relative to this `project.yaml`.

## 3) Source id links source YAML to a stream

A source file declares the raw source id plus loader/parser wiring.

```yaml
# sources/sandbox.ohlcv.yaml
id: "sandbox.ohlcv"
parser:
  entrypoint: "sandbox_ohlcv_dto_parser"
loader:
  transport: fs
  path: data/*.jsonl
  reader:
    format: jsonl
```

Expected behavior:
- Stream `from.source: sandbox.ohlcv` resolves to this source spec.
- For fs loaders, relative `path` is normalized via runtime path policy.
- Standard glob characters in an fs `path` select matching files.
- Gzip CSV/JSONL inputs set `compression: gzip` explicitly; a `.gz` suffix
  alone does not enable decompression.
- Local Parquet inputs set `reader.format: parquet`. Jerry reads files and
  sorted globs in bounded row batches; the configured parser still owns domain
  conversion. Parquet sources do not use text encoding or external compression
  options.

## 4) Stream id links canonical records to dataset

Source-backed streams load and map raw source records before applying validated
preprocess and ordered transforms.

```yaml
# streams/equity.ohlcv.yaml
id: equity.ohlcv
from:
  source: sandbox.ohlcv
map:
  entrypoint: map_sandbox_ohlcv_dto_to_equity
preprocess:
  - { operation: floor_time, cadence: 1d }
transforms:
  - operation: dedupe
```

Expected behavior:
- `from.source` must match a `sources/*.yaml:id`.
- `id` is what `dataset.yaml` references under `stream`.

Derived streams consume existing stream ids:

```yaml
# streams/equity.daily_liquid.yaml
id: equity.daily_liquid
from:
  stream: equity.ohlcv
transforms:
  - operation: dedupe
```

The derived stream inherits `partition_by` and canonical ordering from
`equity.ohlcv`.

Cross-sectional streams compare partitions at one exact timestamp, then
restore canonical stream order:

```yaml
id: equity.signal.ranked
from:
  stream: equity.signal.raw
cross_section:
  - { operation: rank_score, field: signal, to: signal_rank, min_samples: 30 }
```

Their input must be partitioned. The temporary time-major and restored
partition-major sorts both use the configured bounded sort buffer. Hash-split
datasets cannot select cross-sectional streams; use a time split or no split.

Broadcast streams attach one unpartitioned temporal stream to every partition
of a primary stream at an exact timestamp:

```yaml
# streams/equity.price_with_factors.yaml
id: equity.price_with_factors
from:
  stream: equity.price.daily
  broadcast: market.factors.daily
combine:
  entrypoint: combine_price_and_factors
  args: {}
```

The primary must be partitioned, while the broadcast input must have
`partition_by: []`. Jerry establishes canonical stream order before indexing
the finite broadcast input in memory. Duplicate keys, violated ordering
assertions, and missing primary timestamps fail; broadcast timestamps the
primary does not use are ignored. Matching is exact: there is no implicit as-of
or fill behavior. Combiner inputs are read-only, and a broadcast record is
reused across primary partitions at its timestamp.

As-of streams use the latest record available at or before the primary time.
The ordinary form requires matching partition identities:

```yaml
id: equity.price_with_fundamentals
from:
  stream: equity.price.daily
  as_of: equity.fundamentals.reported
max_age: 180d
require_match: true
combine:
  entrypoint: combine_price_and_fundamentals
```

The `broadcast_as_of` form attaches one unpartitioned lookup history to every
primary partition. Exact timestamps are eligible; future lookups never are.
`max_age` is an optional inclusive, non-negative bound. `max_age: 0s` permits
only exact timestamps. With `require_match: false`, the combiner receives
`None` when no eligible lookup exists. Lookup `time` must represent when the
data became available.

Aligned streams intersect their inputs by partition and time. Input order is
also combine argument order:

```yaml
# streams/equity.price_to_earnings.yaml
id: equity.price_to_earnings
from:
  align:
    - equity.price.daily
    - equity.earnings.daily
combine:
  entrypoint: combine_price_to_earnings
  args: {}
```

## 5) Dataset selects fields from stream ids

Dataset config chooses which streams become features/targets and which record field is used as value.

```yaml
sample:
  cadence: ${cadence}
  window_mode: intersection
features:
  - id: closing_price
    stream: equity.ohlcv
    field: close
  - id: opening_price
    stream: equity.ohlcv
    field: open
```

Expected behavior:
- `stream` must match a stream `id`.
- `field` must exist on emitted records.
- Every `sample.keys` field must belong to each referenced stream's resolved
  `partition_by`.
- Partition fields in `sample.keys` identify rows. Remaining partition fields
  suffix series IDs in partition order, producing long, wide, or hybrid output
  without a separate format setting. For split datasets, each fold's eligible
  training rows must establish every generated ID's shape and value types.

## 6) Serve writes run-scoped outputs

Run command:

```bash
jerry serve --output-transport fs --output-format jsonl --output-directory outputs
```

Output layout:

```text
outputs/
  runs/<run_id>/
    dataset/
      dataset.test.jsonl
      dataset.train.jsonl
      dataset.val.jsonl
```

Expected behavior:
- Relative output directory resolves from workspace root.
- Output format extension follows `--output-format` or configured format.

## Quick Debug Checklist

1. Dataset not found:
- Verify `jerry.yaml` `default_dataset` and `datasets.<alias>`.

2. Unknown stream/source ids:
- Verify `streams/*.yaml:from.source` matches `sources/*.yaml:id`.
- Verify `dataset.yaml:stream` matches an id in `streams/`.

3. Empty output:
- Check source loader `path/url`.
- Check parser and map/combine output with
  `jerry serve --preview input|canonical|records`.

4. Wrong output location:
- Check workspace root and `--output-directory` value.
