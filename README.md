# Datapipeline Runtime

Jerry Thomas is a time-series data pipeline runtime. It reads source data,
maps it into ordered record streams, applies declarative transforms, and serves
datasets for analysis or model training.

The runtime is iterator-first: streams are processed on demand, with explicit
sorting, artifacts, observability, built-in transforms, and plugin entry points
for custom loaders, parsers, mappers, and stream combiners.

> **Core assumptions**
>
> - Every record carries a timezone-aware `time` attribute. Time-zone awareness
>   is a quality gate for correct sample assembly.
> - Samples are grouped by `sample.cadence`, plus optional
>   `sample.keys` such as `security_id`.
> - `partition_by` is the complete identity of an independent record series.
>   Dataset `sample.keys` select which of those fields identify rows; remaining
>   partition fields are appended to series IDs.

## Why You Might Use It

- Materialize canonical time-series datasets from disparate sources.
- Preview and debug each stage of the pipeline without writing ad-hoc scripts.
- Export schema-aware Parquet tables for fast Pandas, Polars, and Arrow research,
  then reingest derived Parquet series through Jerry's normal validation.
- Enforce coverage/quality gates and publish metadata and scaler statistics for
  downstream ML teams.
- Extend the runtime with entry-point driven plugins for domain-specific I/O,
  mapping, combining, and custom operations.
- Consume final samples or bounded, metadata-ordered NumPy batches directly
  from Python.

---

## Quick Start

From zero to a served dataset:

![Jerry demo](docs/assets/demo.gif)

```bash
python -m pip install -U jerry-thomas
jerry demo init
cd demo
python -m pip install -e .
jerry serve --dataset demo --limit 3
```

The generated demo is a self-contained workspace. Scaffold commands do not
modify a parent `jerry.yaml`; add a dataset alias explicitly when you want to
integrate a generated project into another workspace.

### Create Your Own Plugin + First Stream

```bash
jerry plugin init my-datapipeline --out lib/
cd lib/my-datapipeline

# Scaffold source YAML, DTO/parser, domain, mapper, and stream.
jerry inflow create

# Reinstall after commands that update entry points (pyproject.toml).
python -m pip install -e .

# Fill in the generated source and mapping templates, then serve.
jerry serve --limit 3
```

Import paths use the normalized package name: `my_datapipeline` for a
`my-datapipeline` distribution. Reinstall the plugin after scaffolding or
manually changing entry points.

## Preview a Pipeline

`jerry serve --preview <stage>` stops at one stable boundary: `input`,
`canonical`, `records`, `series`, `samples`, or `postprocess`. Preview bypasses
split output so the selected stage can be inspected directly. Sequence
construction runs before dataset routing and should remain causal; scaling is
applied only when a full serve selects a fold output.

See the [CLI reference](docs/cli.md#preview-stages) for the exact value emitted
at each boundary and [Artifacts](docs/artifacts.md#splitting--serving) for split
and scaler behavior.

## CLI Cheat Sheet

Profile commands run enabled profiles by default. `--profile <name>` selects
that profile explicitly, including one configured with `enabled: false`.

- `jerry demo init`: create the standalone demo plugin.
- `jerry plugin init <name> --out lib/`: create a plugin workspace.
- `jerry inflow create`: scaffold one source-backed stream end to end.
- `jerry serve`: stream enabled serve profiles.
- `jerry build`: build the series artifact, scaler statistics, and
  metadata.
- `jerry inspect`: run coverage, matrix, or custom inspection profiles.
- `jerry materialize`: write configured streams to durable `.jsonl` or
  gzip-compressed `.jsonl.gz` files.
- `jerry clean [--yes] [--older-than <age>]`: lists or removes stale sort spill directories. It does not delete materialized outputs.

Use `jerry <command> --help` for current flags and the
[CLI reference](docs/cli.md) for command behavior.

## MLOps & Reproducibility

- `jerry build` builds deterministic series, scaler, and metadata
  artifacts. Builds are keyed by configuration and local-source
  snapshots, and skip work when nothing changed unless you pass `--force`.
- Filesystem serve output is run-scoped under
  `<output-directory>/runs/<run_id>/dataset/`. Normal profiles write
  `<profile>.<ext>`; split profiles write
  `<profile>.<fold-id>.<role>.<ext>`.
- Versioning: tag the project config + plugin code in Git and pair with a data
  versioning tool like DVC for raw sources. With those inputs pinned, interim
  datasets and artifacts can be regenerated instead of stored.

## Concepts

### Workspace (`jerry.yaml`)

- `datasets`: dataset aliases → `project.yaml` paths (relative to `jerry.yaml`).
- `default_dataset`: which dataset project commands use when you omit `--dataset/--project`.
- `plugin_root`: where scaffolding commands write Python code (`src/<package>/...`) and where they look for `pyproject.toml`.

### Plugin Package

These live under `lib/<plugin>/src/<package>/`:

- `dtos/*.py`: DTO models (raw source shapes).
- `parsers/*.py`: raw -> DTO parsers (referenced by source YAML via entry point).
- `domains/<domain>/model.py`: domain record models.
- `mappers/*.py`: iterator mappings from parsed values to domain records.
- `combiners/*.py`: functions combining exact, as-of, or aligned domain records.
- `loaders/*.py`: optional custom loaders for inputs beyond built-in filesystem
  and HTTP transports.
- `pyproject.toml`: entry points for loaders, parsers, mappers, and combiners
  (rerun `pip install -e lib/<plugin>` after changes).

### Source to Domain Record

- Built-in filesystem and HTTP transports read input through the configured
  `reader`. Custom loaders handle other protocols.
- A parser converts each row into a source-shaped DTO and may drop invalid rows.
- A mapper converts DTOs into canonical domain records shared by downstream
  streams. Every record has a timezone-aware `time` field.
- Custom loaders are for behavior such as pagination, authentication, or
  proprietary protocols. See [Extending the runtime](docs/extending.md).

### Transforms (Preprocess -> Ordered Stream -> Series -> Sample)

- **Preprocess transforms** run on mapped domain records before ordering. Each transform operates on one record at a time. Configure source-backed streams under `preprocess:`.
- **Ordered transforms** run after ordering (dedupe, cadence enforcement, lag/lead, rolling, derive, fills). These operate across a sequence of records for a partition because they depend on sorted partition/time order and cadence. Configure streams under `transforms:`.
- **Series shaping** runs after stream regularization. `sequence` shapes the
  per-series payload for vectorization; `scale` marks feature or target
  vectors that receive the selected dataset fold's scaler during full serving.
- **Postprocess policies** select assembled vector columns and filter samples by coverage. Configure them under `postprocess:` in `dataset.yaml`.
- Transform lists contain flat, validated built-in operations. Each item has an
  `operation` discriminator and that operation's fields. See the
  [transform guide](docs/transforms/index.md) for the supported operations.

### Glossary

- **Source alias**: `sources/*.yaml:id` (referenced by source-backed streams under `from.source`).
- **Stream id**: `streams/*.yaml:id` (referenced by `dataset.yaml` under `stream:`).
- **Sample key**: sample identity: floored time plus optional `dataset.sample.keys`.
- **Partition**: complete identity of an independent record series, declared by
  stream `partition_by` and used as the state boundary for history-based
  transforms.
- **Series ID fields**: partition fields not present in `dataset.sample.keys`;
  these are appended to series IDs in partition order.
- **Group**: sample cadence set by `dataset.sample.cadence`.
- **Preview stage**: stable semantic boundary selected with
  `jerry serve --preview <stage>`.
- **Sort spill**: ordered stages sort pickle-serializable values in bounded
  serialized buffers and spill temporary runs when the next value would exceed
  the configured buffer.

## Documentation

- [Configuration](docs/config.md): config layout, precedence, and YAML reference.
- [Data flow](docs/dataflow.md): the YAML reference chain from workspace to output.
- [CLI](docs/cli.md): command behavior beyond `--help`.
- [Transforms](docs/transforms/index.md): preprocess, ordered, series, and postprocess stages.
- [Artifacts](docs/artifacts.md): dependencies, freshness, splitting, and serving.
- [Python integrations](docs/python.md): final-sample and bounded model-batch
  iterators.
- [Research workflow](docs/research.md): export samples to Parquet, derive a
  series with Polars, and reingest it into Jerry.
- [Extending](docs/extending.md): plugin entry points and contracts.
- [Architecture](docs/architecture.md): runtime and pipeline design.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).
