# Configuration

For upgrades from v10, see [Migrating to v11](migrations/v11.md).

### Dataset Project (YAML Config)

These live under the dataset “project root” directory (the folder containing `project.yaml`):

- `project.yaml`: paths + globals (single source of truth).
- `sources/*.yaml`: raw sources (loader + parser wiring).
- `streams/*.yaml`: source-backed, derived, exact/as-of fan-in, or aligned
  canonical streams.
- `dataset.yaml`: sample, feature/target, split, and postprocess policy.
- `profiles/serve.<name>.yaml`: serve profiles.
- `profiles/build.<name>.yaml`: build profiles.
- `profiles/inspect.<name>.yaml`: inspect profiles.
- `profiles/materialize.<name>.yaml`: durable stream-output profiles.
- `operations/*.yaml`: optional core-operation overrides and custom operations.

### Configuration & Resolution Order

Defaults are layered so you can set global preferences once, keep dataset/run
files focused on per-project behavior, and still override anything from the CLI.
For `jerry serve`, `jerry inspect`, `jerry build`, and `jerry materialize`,
options are merged in the following order (highest precedence first):

1. **CLI flags** – anything you pass on the command line always wins.
2. **Project profile files** – profile configs under `project.paths.profiles`
   supply per-command settings and selection policy. The filename determines
   the command and profile name.
3. **Per-kind profile defaults** – optional `profiles/<kind>.defaults.yaml`.
4. **Built-in defaults** – runtime hard-coded defaults.

## YAML Config Reference

All dataset configuration is rooted at a single `project.yaml` file. Other YAML files are discovered via `project.paths.*` (relative to `project.yaml` unless absolute).

### `project.yaml`

```yaml
schema_version: 6
artifact_revision: 1
name: default
paths:
  streams: ./streams
  sources: ./sources
  dataset: dataset.yaml
  artifacts: ../artifacts/${project_name}
  # operations: ./operations # optional; core operations need no declarations
  profiles: ./profiles
globals:
  start_time: 2021-01-01T00:00:00Z
  end_time: 2023-01-03T23:00:00Z
  vendor_api_key: ${env:VENDOR_API_KEY}
  raw_root: ${env:RAW_ROOT}
```

- `schema_version` identifies the `project.yaml` format. Jerry validates this
  compatibility marker before loading the rest of the project configuration.
- `name` provides a stable identifier you can reuse inside config files via `${project_name}`.
- `artifact_revision` is a positive integer that versions the semantics of
  generated artifacts. Increment it whenever parser, mapper, combine,
  transform, or custom artifact code changes what an artifact means without a
  corresponding config change. It is required so every project explicitly owns
  this cache contract.
- `paths.*` are resolved relative to the project file unless absolute; they also support `${var}` interpolation.
- `paths.sources` and `paths.streams` may be either one path
  or a list of paths. When a list is used, discovery scans every directory in
  order and rejects duplicate source or stream ids.
- Project, dataset, config-root, and YAML-file aliases are resolved to their
  canonical paths. Recursive discovery rejects linked directories inside a
  config root instead of silently skipping or following them.
- `globals` provide values for `${var}` interpolation across YAML files. Datetime
  values are normalized to strict UTC `YYYY-MM-DDTHH:MM:SSZ`.
- `${path:../shared/releases}` produces an absolute path relative to `project.yaml`,
  including in custom-loader arguments. It is independent of the working directory,
  accepts literal paths only, and does not require the destination to exist.
- External references use `${env:NAME}`. Resolution checks the process
  environment first and then an optional project-root `.env` file.
- New scaffolded dataset projects include a `.env.example` next to `project.yaml`.
- `paths.operations` optionally points to `operations/*.yaml`. Omit it when the
  built-in operations are sufficient. When configured, the directory must exist.
  Core artifact operations are `scaler`, `series`, `metadata`, and
  `coverage_stats`; core runtime operations are `dataset`, `coverage`, and
  `matrix`.
  Files override core settings or declare custom operations.
- `paths.profiles` points to profile specs grouped by type:
  `profiles/serve.<name>.yaml`, `profiles/build.<name>.yaml`,
  `profiles/inspect.<name>.yaml`, and `profiles/materialize.<name>.yaml`.
  Optional defaults files may also be declared once per kind:
  `profiles/serve.defaults.yaml`, `profiles/build.defaults.yaml`,
  `profiles/inspect.defaults.yaml`, and `profiles/materialize.defaults.yaml`.
  Each concrete file contains one mapping. Its `<kind>.<name>.yaml` filename is
  the profile identity; `cmd` and `name` are not repeated in the YAML body.
  When multiple profiles exist, `--profile <name>` selects one by filename.
  `enabled: false` excludes a profile from the default batch, but an explicit
  `--profile <name>` still selects it.
- Dataset split label names are free-form: match the keys declared in
  `dataset.yaml:split.ratios` (hash) or the IDs declared in
  `dataset.yaml:split.intervals` (time).
- Dataset output IDs are `<fold-id>.<role>`, where role is `train`,
  `validation`, or `test`.

### Serve Profiles (`profiles/serve.<name>.yaml`)

```yaml
# profiles/serve.dataset.yaml
operation: dataset # core runtime operation
# include_outputs: [holdout.train, holdout.test] # optional fold outputs
output:
  transport: fs # stdout | fs; configured split output requires fs
  format: jsonl # jsonl | csv | parquet | pickle
  directory: runs
  # view: raw # optional; csv/parquet default to flat; jsonl/pickle default to raw
  # encoding: utf-8 # fs jsonl/csv only
  # compression: gzip # optional; fs jsonl/csv only
limit: 100 # cap samples per output
throttle_ms: null # milliseconds to sleep between emitted samples
# Optional overrides:
# observability:
#   visuals: true      # true | false
#   heartbeat_interval_seconds: 60 # 0 disables heartbeat log records
#   logging:
#     level: INFO    # CRITICAL | ERROR | WARNING | INFO | DEBUG
#     outputs:
#       - transport: STDERR # STDERR | STDOUT | FS
#       - transport: FS
#         scope: EXECUTION  # GLOBAL | EXECUTION
#         path: logs/serve.log # optional; default logs/serve.<profile>.log
```

- Each serve profile is a flat file under `profiles/` with the `serve.` prefix.
- `include_outputs`, `output`, `limit`, `throttle_ms`, and `observability` may
  also be set in `serve.defaults.yaml`. Profile values override those defaults,
  and CLI flags still win where available (see
  _Configuration & Resolution Order_). Filesystem runs write under
  `<directory>/runs/<run_id>/dataset/`; normal outputs use `<profile>.<ext>` and
  split outputs use `<profile>.<fold-id>.<role>.<ext>`. When neither a profile
  nor `output.filename` supplies the base name, it defaults to `dataset`.
- `output.encoding` is supported for fs `jsonl`/`csv` outputs (default `utf-8`); it is invalid for `stdout`, `parquet`, and `pickle`.
- `output.compression: gzip` writes fs `jsonl`/`csv` outputs with a `.gz`
  suffix. This applies unchanged to full dataset, split, inspect, and preview
  outputs; Jerry does not infer compression from a filename.
- Selected data and log destinations are validated before any filesystem log
  is activated. Data and logs cannot both use stdout. Data files must be unique
  and cannot overlap another selected file path. Distinct log paths
  cannot be nested; multiple profiles may intentionally share the same log
  file.
- Global log files must stay outside the artifacts root and managed serve
  `runs`/`latest` paths. Use `scope: EXECUTION` for logs owned by a command
  execution.
- Dataset Parquet output is filesystem-only and schema-aware. It uses bounded
  Zstandard-compressed row groups and the stable columns `sample.time`,
  `sample.<key>`, `features.<id>`, and `targets.<id>`; fixed-length lists
  expand to numbered columns. It is supported for full datasets and the
  `samples` and `postprocess` preview stages. Install
  `jerry-thomas[parquet]` to enable it.
- Output values use `None` as the canonical missing value. A transient floating
  `NaN` is emitted as null (`None` in pickle and an empty CSV cell); positive
  and negative infinity fail the output operation.
- A full dataset serve with `dataset.yaml:split` writes one output file per
  configured fold role, using profile-qualified filenames such as
  `dataset.holdout.train.jsonl`. `include_outputs` optionally narrows that set
  using the same `<fold-id>.<role>` IDs. Split fanout requires filesystem
  output. When set, `output.filename` becomes the base name, producing files
  such as `dataset.holdout.train.jsonl`.
- Preview bypasses automatic split fanout. Record stages emit once per unique
  referenced stream, `series` emits once per configured feature or target, and
  `samples`/`postprocess` emit one combined output. A profile cannot combine
  explicit `include_outputs` with preview.
- Before any selected serve profile runs, Jerry unions their artifact
  requirements and prepares the union once according to `artifact_mode`.
  `auto` builds missing or stale artifacts, `rebuild` rebuilds the required
  closure, and `require_current` requires every artifact to be current. The mode is
  command-wide and belongs in `serve.defaults.yaml`, not an individual profile.
  Its precedence is CLI `--artifact-mode`, then `serve.defaults.yaml`, then
  `auto`.
- Visuals: set `observability.visuals: true/false` in the profile or use `--visuals` / `--no-visuals`. Strings such as `ON` and `OFF` are rejected.
- Pipeline heartbeat: set `observability.heartbeat_interval_seconds` or use
  `--heartbeat-interval`; `0` disables logged heartbeats, not live
  progress when visuals are enabled. The built-in interval is 60 seconds.
- The shared artifact prerequisite phase uses the command defaults and any CLI
  `--heartbeat-interval` override. A concrete profile heartbeat setting starts
  applying when that profile itself runs; Jerry does not select one profile's
  setting for shared prerequisite work.
- Add additional `serve.<name>.yaml` files under `profiles/`
  for distinct serve policies; `jerry serve` runs each enabled profile unless
  you pass `--profile <name>`.
- Use `profiles/serve.defaults.yaml` for common serve defaults shared across all
  serve profiles in a project. CLI flags and concrete profiles still take
  precedence for profile-level settings.

### Materialize Profiles (`profiles/materialize.<name>.yaml`)

```yaml
# profiles/materialize.adv.20.yaml
order: 10
stream: adv.20
output: ${data_root}/features/liquidity/adv/20.jsonl
overwrite: true
# observability:
#   visuals: true
#   heartbeat_interval_seconds: 60
```

- `jerry materialize` runs all enabled materialize profiles in `order`;
  `--profile` selects one. Unlike serve and inspect profiles, these profiles
  identify a stream directly and do not reference an operation.
- CLI `--output` overrides the selected profile and requires `--profile`; the
  profile remains the source of the stream identity.
- Relative profile outputs resolve from `project.yaml`; relative CLI `--output`
  values resolve from the workspace root, or the current directory without a
  workspace. A `.jsonl` path writes plain JSONL; `.jsonl.gz` writes gzip JSONL.
  The concrete path is the complete materialize output contract.
- Before execution, Jerry validates every selected stream and checks the full
  destination set for duplicates and existing files. No profile writes until
  the whole selected batch passes this preflight.
- `overwrite: false` is the built-in default. `--overwrite` and
  `--no-overwrite` override every selected profile; shared defaults belong in
  `profiles/materialize.defaults.yaml`.
- Output paths must be outside `project.paths.artifacts`; that directory is
  reserved for managed artifacts and build state.
- `artifact_mode` is command-wide and belongs only in
  `profiles/materialize.defaults.yaml`. `auto` prepares missing or stale stream
  prerequisites, `rebuild` rebuilds them, and `require_current` requires them to be current.
  CLI `--artifact-mode` takes precedence; the built-in mode is `auto`.

### Build Profiles (`profiles/build.<name>.yaml`)

```yaml
# profiles/build.metadata.yaml
operation: metadata # required; artifact operation ID to execute
artifact_mode: auto # auto | rebuild | require_current
# enabled: true # optional; profile-level switch
# Optional overrides:
# observability:
#   visuals: false
#   heartbeat_interval_seconds: 60 # 0 disables heartbeat log records
#   logging:
#     level: INFO
#     outputs:
#       - transport: FS
#         scope: GLOBAL
#         path: ./logs/build.log
```

- Build profiles are orchestration profiles; they do not replace operation definitions.
- `operation` selects the artifact operation ID for that profile (`metadata`,
  `scaler`, `coverage_stats`, ...). Selected build profiles must reference
  distinct operations.
- Build profile `observability.logging.outputs[].path` values are resolved relative to the dataset project root (`project.yaml` directory).
- `jerry build` runs enabled build profiles when they exist;
  `jerry build --profile <name>` selects one profile.
- Build profile `order` controls only the order of selected build profiles. The
  dependency graph orders the internal artifact jobs required by each operation
  and never reorders profiles. If selected profiles include both a dependency
  and its dependent, the dependency profile must come first.
- Precedence for build settings: CLI > `build.<name>.yaml` > `build.defaults.yaml` > built-ins.

### Profile Defaults (`profiles/<kind>.defaults.yaml`)

- Defaults files are optional and non-executable.
- Their `<kind>.defaults.yaml` filename determines the command. The YAML body
  contains only non-routing defaults for that kind.
- They must not include execution identity fields such as `name`, `operation`,
  `enabled`, or `order`.
- `execution` is command-wide and is not accepted in concrete profiles.
  Serve, inspect, and materialize `artifact_mode` is likewise defaults-only.
- Defaults-level `observability` configures the shared prerequisite phase as
  well as providing profile defaults. Concrete observability overrides apply
  only to that profile.
- Profile-level setting precedence is CLI > concrete profile >
  `<kind>.defaults.yaml` > built-ins. Command-wide `artifact_mode` precedence is
  CLI > `<command>.defaults.yaml` > `auto`.
- A concrete profile's `output` block replaces the defaults' complete output
  block. To inherit it, omit `output`; to replace it, provide a valid complete
  block, including `transport`, `format`, and a filesystem `directory` where
  required. `output: {format: csv}` alone is not a partial profile override.
- `observability` and its nested `logging` mapping merge by field; lists such as
  logging `outputs` are replaced. CLI `--output-*` flags override individual
  output fields after profile defaults are applied. An omitted encoding stays
  unspecified through that merge; filesystem text output then defaults to
  UTF-8. An explicitly configured encoding remains subject to the selected
  format's validation.

Sorting and worker concurrency are execution policies, not stream definitions.
Configure them in each command's defaults file:

```yaml
# profiles/serve.defaults.yaml
artifact_mode: auto
execution:
  sort_buffer_mb: 128
  workers: 1
```

`sort_buffer_mb` is the soft serialized-payload target for each active sort
buffer, interpreted as MiB (1024² bytes). When another item would exceed a
non-empty buffer, that buffer is sorted and spilled as a temporary run. One item
larger than the target occupies a buffer by itself. Python and sort-key overhead
are additional, so this is not a process memory limit. Sorted items must be
pickle-serializable. The built-in default is `128`. Build, materialize, serve,
and inspect resolve their execution settings independently.

`workers` is a positive integer, defaulting to `1`. During series artifact
building, larger values allow up to that many stream pipelines to run in separate
processes, including their source sorting, transforms, and final sample ordering.
This also applies when serve builds a missing or stale series artifact. Workers
write temporary sorted runs, which the parent merges before artifact assembly.
Equal-key rows retain their original stream and record order. Parallel builds
write runs even when a stream fits in memory. A single selected stream uses the
sequential path. Other operations remain sequential.
Each worker constructs fresh plugin instances from the resolved configuration;
plugins must be installed and available to child processes. Increasing workers
can increase memory and disk use: `sort_buffer_mb` still applies to each active
sort, and a stream can contain multiple sorts. Start with `workers: 2` and measure
your workload before increasing it.
Parallel builds currently show aggregate progress; individual worker stages are
not displayed.

### Operations (`operations/*.yaml`)

```yaml
# operations/coverage.yaml — optional core override
options:
  threshold: 0.8

# operations/matrix.yaml — optional bounded matrix override
options:
  stage: assembled
  max_cells: 250000

# operations/coverage_stats.yaml — optional summary-stage override
stage: assembled

# operations/schedule.yaml — explicit expected-timestamp artifact
kind: artifact
entrypoint: core.artifact.schedule
stream: exchange.sessions
partition_by: []
output: build/schedule.jsonl

# operations/custom_report.yaml — custom operation
kind: runtime
entrypoint: my_plugin.report
requires: [custom_artifact]
options: {}
```

- Stable core operations are registered by Jerry and need no YAML declarations.
- A schedule is source-specific and therefore remains an explicit artifact
  operation rather than a core override. Its required `partition_by` must match
  each consuming stream; an empty list means one global schedule. Streams
  reference it with
  `{ operation: ensure_schedule, schedule: schedule }`. See
  [Artifacts](artifacts.md) for completion behavior and migration details.
- Each file contains one mapping, and its filename supplies the operation ID.
  Do not repeat `id`. Core overrides also omit `kind` and `entrypoint`.
- Custom operations declare `kind: artifact|runtime` and an `entrypoint`.
- Runtime operations are executable units; profiles reference them via
  `operation`.
- Core operations use reserved `core.runtime.*` identifiers and call their typed
  implementations directly. A custom runtime operation's `entrypoint` must
  resolve in the `jerrythomas.operations.runtime` entry-point group.
- `requires` declares additional prerequisite artifact operation IDs for custom or
  built-in runtime operations. Each referenced artifact and its dependency chain must
  have available producer operations.
- Custom artifact operations accept `kind`, `entrypoint`, and `output`, with
  their ID supplied by the filename. They do not accept `options` or `requires`;
  their cache hashes conservatively cover the complete dataset and stream catalog.
- Built-in runtime operation options are entrypoint-specific:
  - The `dataset` operation uses `core.runtime.dataset` internally and accepts
    no operation options. Limit, preview, throttle, output, and visuals can be
    set by the serve profile or CLI. Dataset split output comes from
    `dataset.yaml`; `include_outputs` can narrow it. Preview, throttle, and split
    output are not accepted by other runtime operations.
  - `core.runtime.coverage`: optional `threshold` between `0` and `1`
    (default: `0.95`). Results are ordered from lowest to highest coverage.
    Coverage reads a completed `coverage_stats` artifact, so it does not accept
    `--limit`.
  - `core.runtime.matrix`: optional `stage: assembled|postprocessed` (default:
    `postprocessed`) and positive `max_cells` (default: `1000000`). The bound
    counts scalar cells and individual list elements. `jerry inspect --limit N`
    caps the samples inspected after the selected stage. Its output format and
    destination come from the inspect profile or CLI.
- Unknown keys on built-in runtime operations are rejected. Custom plugin
  runtime operations retain their plugin-defined `options` mapping.
- A custom runtime entry point has one positional contract: `(runtime, task,
  limit)`. It returns one `RuntimeOutput`, or `None`; shared persistence applies
  the profile output. Runtime results cannot choose output paths.

Jerry 7 names the built-in dataset runtime consistently. Explicit
`core.runtime.pipeline` references become `core.runtime.dataset`; Python users
replace `OperationTask` with `RuntimeTask`, `PipelineTask` with `DatasetTask`,
and `run_pipeline_operation` with `run_dataset_operation`. Normal
`operation: dataset` profiles require no change, and this rename does not
invalidate artifacts.

Jerry 8 gives custom runtime plugins their own task type. Plugin implementations
replace `RuntimeTask` with `PluginRuntimeTask`; operation YAML is unchanged.

### Workspace Routing (`jerry.yaml`)

Create an optional `jerry.yaml` in the directory where you run the CLI to share settings across commands. The CLI walks up from the current working directory to find the first `jerry.yaml`.

```yaml
plugin_root: lib/my-datapipeline # active plugin workspace (relative to this file)

# Dataset aliases for --dataset; values may be dirs (auto-append project.yaml).
datasets:
  your-dataset: lib/my-datapipeline/your-dataset/project.yaml
default_dataset: your-dataset
```

`jerry.yaml` sits near the root of your workspace and is only used for workspace routing (`plugin_root`, dataset aliases, default dataset).
- Command/runtime settings belong in profile files under `profiles/`.
- Execution-scoped logs (`scope: EXECUTION`) are configured on profiles or via
  CLI `--log-output execution[:<relative-path>]`.

### `<project_root>/sources/<alias>.yaml`

Each file defines a loader/parser pair exposed under `<alias>`. Files may live in nested
subdirectories under `<project_root>/sources/`; discovery is recursive.

```yaml
# Source identifier (commonly `provider.dataset`). Streams reference this under `from.source`.
id: stooq.ohlcv
parser:
  # Parser entry point name (registered in your plugin’s pyproject.toml).
  entrypoint: stooq.ohlcv
loader:
  transport: http
  url: "https://stooq.com/q/d/l/?s=aapl.us&i=d"
  headers:
    Authorization: "Bearer ${env:STOOQ_API_KEY}"
  reader:
    format: csv
```

- `id`: the source alias; referenced by source-backed streams under `from.source`.
- `parser.entrypoint`: which parser to use; `parser.args` are optional.
- `loader.transport` says where built-in input comes from (`fs` or `http`).
  Custom loaders instead declare `loader.entrypoint` and optional `loader.args`.
- `loader.reader.format` says how that input becomes raw rows: `csv`, `json`,
  `jsonl`, or local `parquet`. The parser then converts each raw row to the
  source value consumed by streams. Text readers accept `encoding`;
  CSV also accepts `delimiter` and `error_prefixes`, while JSON accepts
  `array_field`.
- `inputs.files`: optional project-relative regular files or glob patterns used
  to track custom-loader inputs for artifact freshness. Built-in filesystem
  paths are tracked automatically. The list is order-insensitive;
  duplicate entries are rejected.
- `freshness` defaults to `tracked`. Sources whose inputs Jerry cannot see —
  custom loaders and HTTP loaders without `inputs.files` — must either declare
  those files or set `freshness: opaque` to acknowledge that input changes will
  not invalidate artifacts. Filesystem sources are always tracked and reject
  `freshness: opaque`.
- A filesystem `path` containing standard glob characters (`*`, `?`, `[`) loads
  every matching file in sorted order; a path without them loads one file.
- Filesystem CSV and JSONL sources may set `compression: gzip`. Compression is
  explicit and is not inferred from a `.gz` suffix.
- Filesystem Parquet sources use bounded row batches and preserve native scalar,
  list, null, date, and datetime values for the parser. They do not accept text
  decoding options or external compression.
- Built-in CSV decoding requires a non-empty, unique header and the same number
  of fields in every record. Built-in JSON and JSONL decoding reject duplicate
  object keys, non-standard numeric constants, and numbers outside the finite
  float range instead of silently changing the data.
- Local freshness snapshots include glob membership, file paths, sizes, and
  filesystem modification metadata. HTTP response bodies and headers are not
  fingerprinted: use `--artifact-mode rebuild` when a stable URL can return new
  data, and increment `artifact_revision` when that change must invalidate
  other workspaces. The same rule applies to sources marked
  `freshness: opaque` and to other source changes that cannot be declared
  through `inputs.files`.
- Keep secrets and machine-local paths out of source files. Prefer `${env:...}`
  directly or route them through `project.yaml.globals` aliases like `${raw_root}`.

### Source-backed Streams

A source-backed stream loads parsed source values, maps them to domain records,
preprocesses individual records, establishes canonical order, and then applies
ordered transforms. It is the only stream kind that declares `map`,
`preprocess`, `partition_by`, or `presorted`.

```yaml
# <project_root>/streams/equity.ohlcv.yaml
id: equity.ohlcv # stream identifier (domain.dataset[.variant])
from:
  source: stooq.ohlcv # references sources/<alias>.yaml:id

map:
  entrypoint: equity.ohlcv
  args: {}

partition_by: [security_id]

preprocess:
  - { operation: where, field: time, operator: ge, comparand: "${start_time}" }
  - { operation: where, field: time, operator: lt, comparand: "${end_time}" }

transforms:
  - operation: resample
    period: { kind: fixed, every: 10m }
    start: "${start_time}"
    end: "${end_time}"
    aggregations:
      close: { field: close, statistic: last }
```

The mapper receives the parsed source iterator and returns canonical domain
records. Jerry normalizes their timezone-aware `time` fields to UTC before
`preprocess`, ordering, and downstream transforms.

### Derived Streams

A derived stream adds ordered transforms to one existing stream. It inherits
the upstream partition identity and canonical order; it does not repeat or
override source mapping and ordering policy.

```yaml
id: equity.ohlcv.liquid
from:
  stream: equity.ohlcv
transforms:
  - { operation: ensure_cadence, cadence: 10m }
  - { operation: collapse, keep: last }
  - { operation: fill, field: close, statistic: median, window: 6, min_samples: 2 }
  - { operation: forward_fill, field: close, to: close_asof }
```

- `preprocess`: built-in per-record transforms applied before ordering on a
  source-backed stream.
- `transforms`: built-in operations applied after canonical ordering on every
  stream kind.
- Each item is a flat mapping with an `operation` discriminator and that
  operation's fields. Missing or unknown operations, unknown fields, and
  invalid values are rejected. See [Transforms](transforms/index.md).
- Ordered transforms cannot write `time` or a `partition_by` field. Mapping
  runs before canonical ordering; combines and ordered transforms preserve it.
- `partition_by` is a source-backed stream field and must be a YAML list when
  present. Scalar shorthand is not accepted; blank and duplicate fields are
  rejected.
- `partition_by`: complete identity of an independent record series, used by
  ordering and history-based transforms. The runtime appends the reserved
  `time` field to the canonical sort key, so it must not appear here. Derived
  and cross-sectional streams inherit it from their input; fan-in streams
  inherit it from their partitioned input.
- `presorted`: source-backed stream boolean, default `false`. With `true`,
  records entering the ordering stage must already follow `[*partition_by, time]`
  order (or just `time` when unpartitioned). Jerry validates that order while
  streaming and fails on a violation. With `false`, mapped and preprocessed
  records are externally sorted. Derived streams reuse upstream canonical order.
- Series identity is derived at the dataset boundary. Every `sample.keys`
  field must occur in the resolved `partition_by` of every referenced stream.
  Partition fields absent from `sample.keys` suffix the configured feature or
  target ID in partition order (for example, `temp__@station_id:XYZ`). Putting
  every partition field in `sample.keys` produces long/entity-keyed output;
  putting none there produces wide output; using a subset produces a hybrid.
  For split datasets, each fold's eligible training rows must establish every
  generated wide or hybrid series ID, including its scalar/list shape and value
  types. Metadata building fails rather than let validation or test data alter
  the training schema.
  Dataset feature and target IDs cannot contain the reserved `__` separator.
  Generated suffixes escape strings and tag non-string scalar values so
  different component tuples cannot produce the same series ID.

### Cross-Sectional Streams

A cross-sectional stream compares partitioned records at each exact timestamp.
It inherits the upstream partition identity, temporarily orders records by
`[time, *partition_by]`, applies its operations in YAML order, and restores
canonical `[*partition_by, time]` order before ordinary transforms run.

```yaml
id: equity.signal.neutralized
from:
  stream: equity.signal.inputs
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
  - { operation: lag, field: signal_residual, to: signal_residual_lag_1, periods: 1 }
```

The input must have a non-empty `partition_by`, and each timestamp may contain
at most one record per partition. Matching is exact: Jerry does not floor,
fill, or infer sessions. `rank_score` uses normalized average-tie ranks;
`ols_residual` uses complete-case OLS with an intercept and requires
`jerry-thomas[numerical]`. Both require an explicit `min_samples`. Use an
upstream `where` transform to define the eligible population. Hash-split
datasets cannot select cross-sectional streams because peer records could land
in different output roles. See
[Cross-sectional operations](transforms/cross_section.md) for the complete
missing-value and numerical contracts.

### Broadcast Streams

A broadcast stream attaches one unpartitioned record to every partitioned
primary record at the same timestamp. The primary stream supplies partition
identity and output order; the broadcast stream supplies shared temporal data.

```yaml
id: equity.price_with_factors
from:
  stream: equity.price.daily
join:
  kind: broadcast
  with: market.factors.daily
combine:
  entrypoint: combine_price_and_factors
  args: {}

# Optional transforms run after combining.
# transforms: [...]
```

Notes:

- `from.stream` is the primary input and must resolve to a non-empty
  `partition_by`. The broadcast stream inherits that partition identity.
- `join.with` must resolve to an empty `partition_by`.
- To attach several global series, align them into one unpartitioned stream,
  then use that stream as the broadcast input.
- Matching is exact timestamp equality. Broadcast streams do not perform
  as-of matching, filling, tolerance matching, or many-to-many expansion.
- The primary input must contain at most one record per `(partition, time)`
  key. The broadcast input must contain at most one record per timestamp.
  Source-backed streams establish canonical order before broadcasting;
  duplicate keys or a violated `presorted` assertion still fail.
- Every primary timestamp must have a broadcast record. A missing match fails
  the stream; broadcast timestamps unused by the primary are ignored.
- Jerry fully indexes the finite broadcast input before reading the primary.
  Memory use is proportional to the number of broadcast records.
- Combine signature is `combine(primary_record, broadcast_record, **args)`.
  Inputs are read-only. The same broadcast record object is reused for every
  primary partition at its timestamp. The combiner returns one record or
  `None` to skip that primary record.
- The broadcast stream outputs records; its own `transforms` apply afterward.

### As-of Streams

An as-of stream attaches the latest lookup record available at or before each
primary record. Use `as_of` when both inputs have the same partition identity:

```yaml
id: equity.price_with_fundamentals
from:
  stream: equity.price.daily
join:
  kind: as_of
  lookup: equity.fundamentals.reported
  max_age: 180d
  require_match: true
combine:
  entrypoint: combine_price_and_fundamentals
  args: {}
```

Use `kind: broadcast_as_of` when one global lookup history applies to every
primary partition:

```yaml
id: equity.return_with_market_factor
from:
  stream: equity.return.daily
join:
  kind: broadcast_as_of
  lookup: market.factors.published
  max_age: 7d
combine:
  entrypoint: combine_return_and_factor
  args: {}
```

Notes:

- `direction` defaults to `backward`: the selected lookup has the greatest
  `time` satisfying `lookup.time <= primary.time`. With
  `direction: forward`, the selected lookup has the smallest `time`
  satisfying `lookup.time >= primary.time` — for example, the first monthly
  session at or after a filing becomes available. Exact timestamps are
  eligible in both directions; there is no nearest match.
- Lookup `time` is its availability time. Keep an effective or reporting period
  in a separate record field when it differs.
- `max_age` is optional, inclusive, and non-negative. Positive values such as
  `30min`, `12h`, or `180d` permit earlier lookup records under backward
  direction and later lookup records under forward direction. Use `0s` for
  exact timestamp matching only.
- `require_match` defaults to `true`. With `false`, an unmatched lookup is
  passed to the combiner as `None`. Together, `max_age: 0s` and
  `require_match: false` provide exact optional left alignment while preserving
  every primary record unless the combiner explicitly drops it.
- `join.lookup` in an `as_of` join must have the same `partition_by` as the
  primary. Both may be unpartitioned. Matching streams in canonical order uses
  constant memory.
- `broadcast_as_of` lookups must be unpartitioned, while their primary must be
  partitioned. Jerry indexes the finite lookup history in memory so it can be
  reused when time restarts for each primary partition.
- Primary and lookup canonical keys must be unique and ordered.
- Combine signature is `combine(primary_record, lookup_record, **args)`, where
  `lookup_record` may be `None` only when `require_match: false`. The combiner
  must preserve the primary time and partition and may return `None` to drop it.
- The stream's own `transforms` run after combining.

### Aligned Streams (Engineered Domains)

Aligned streams intersect two or more input streams with the same
partition fields and timestamp, then call a combine function with the matching
records in the configured order. In this example, both inputs use
`partition_by: [station_id]`, which the aligned stream inherits.

```yaml
# <project_root>/streams/air_density.processed.yaml
id: air_density.processed
from:
  stream: pressure.processed
join:
  kind: align
  streams:
    - temp_dry.processed
combine:
  entrypoint: combine_air_density
  args: {}

# Optional policies run after combining like any stream.
# transforms: [...]
```

Dataset stays minimal — features only reference the aligned stream:

```yaml
# dataset.yaml
sample:
  rounding: ceil
  cadence: 1h
  keys: [station_id]
features:
  - id: air_density
    stream: air_density.processed
    field: air_density
```

Notes:

- `from.stream` names the first combine input; `join.streams` supplies at least
  one more input. Their declared order defines positional combine arguments.
- `combine` is required and cannot be replaced by the iterator-level `map`.
- `combine.entrypoint` resolves from the `jerrythomas.combiners` plugin group.
- Jerry normalizes timezone-aware combiner output timestamps to UTC.
- Inputs must use the same `partition_by`; the aligned stream inherits it.
- Alignment validates and merges the already ordered inputs in one pass. Each
  source-backed stream establishes canonical `[*partition_by, time]` order
  before its ordered transforms, which preserve that order. `execution.sort_buffer_mb`
  applies to that upstream ordering stage, not to alignment.
- Each input must contain at most one record per `(partition, time)` key. Only
  keys present in every input are combined. Alignment stops when any input is
  exhausted and does not scan irrelevant tails of the remaining inputs.
- Combine signature is `combine(first_record, second_record, ..., **args)` and it
  returns one record or `None` to skip that key.
- The aligned stream outputs records; its own `transforms` apply afterward.

### `dataset.yaml`

Defines which canonical streams become features and targets and how samples are grouped.

```yaml
sample:
  rounding: ceil
  cadence: 1d
  keys: [security_id]
  window_mode: intersection

features:
  - id: close
    stream: equity.ohlcv
    field: close
    scale: true
    sequence: { size: 6, stride: 1 }
  - id: intraday_volume
    stream: equity.volume.hourly
    field: volume
    collect: 24

targets:
  - id: forward_return_1d
    stream: equity.ohlcv
    field: forward_return_1d
    horizon: 1d

split:
  mode: time # hash | time
  intervals:
    - id: train
      until: "2022-01-01T00:00:00Z"
    - id: validation
      until: "2023-01-01T00:00:00Z"
    - id: test
  folds:
    - id: holdout
      train: [train]
      validation: [validation]
      test: [test]

postprocess:
  features:
    threshold: 0.95
```

- `sample.cadence` controls the time bucket for samples (must match
  `^\\d+(m|min|h|d)$`, e.g. `10m`, `60min`, `1h`, `1d`).
- `sample.rounding` is required: `floor`, `ceil`, or `exact`. On an hourly UTC
  grid, `floor` assigns `[10:00, 11:00)` to sample 10:00; `ceil` assigns
  `(10:00, 11:00]` to sample 11:00. `exact` rejects observations between grid
  ticks, including sequence warmup records. Exact-grid timestamps stay unchanged
  in every mode. Stream timestamps are preserved. Features, targets, collection,
  sequence endpoint assignment, preview grouping, and scaler training share this
  contract. No mode silently drops observations or fills later samples.
- `sample.keys` optionally adds record fields to the sample key. For example,
  `keys: [security_id]` emits one sample per `(time, security_id)`. Every sample
  key must belong to the resolved `partition_by` of every referenced stream.
- `sample.window_mode` selects the rectangular sample domain. `union` spans
  every base series, `intersection` (the default) intersects base-series
  ranges, and `strict` intersects every expanded partition series. It controls
  domain bounds, not internal missing values.
- `stream` references the canonical stream that supplies the
  feature or target records.
- Each sample-key field must contain non-null JSON scalar values of one stable
  type. Floating-point keys must be finite; booleans, integers, and floats are
  distinct key types and cannot be mixed within one field.
- Stateful ordered transforms such as `lag`, `lead`, `ewm_mean`, `rolling`,
  `rolling_slope`, `rolling_ols`, `forward_sum`, `fill`, and `ensure_cadence`
  use stream `partition_by` as their entity partition. Define
  `partition_by: [security_id]` on the source-backed stream when transform state
  must stay per security; downstream streams inherit it.
- `partition_by` is the complete series identity. `sample.keys` select which
  partition fields identify output rows; remaining partition fields suffix
  series IDs such as `close__@security_id:AAPL`. This supports long, wide, and
  hybrid layouts without a separate format or series-identity setting. In a
  split dataset, each fold's eligible training rows must establish every
  resulting wide or hybrid ID's shape and value types.
- `field` selects the record attribute used as the feature/target value.
- Every target requires `horizon`. It is the conservative maximum elapsed
  time between the sample key and the latest observation used to compute that
  target. Use `0s` for a contemporaneous target. Jerry does not infer this
  contract from `lead`, `forward_sum`, or custom transforms.
- Features do not accept a horizon: they must be observable by the intended
  prediction cutoff. With `floor`, a sample timestamp can label a bucket that
  completes one cadence later; it does not assert availability at the bucket
  start. With `ceil` or `exact`, the sample tick can serve as the cutoff when
  source timestamps represent availability. Align targets, their horizons, and
  split boundaries to the chosen convention. Rounding alone does not establish
  when source data or future-derived values actually became available.
- `None` is the canonical missing series value. A floating `NaN` produced by
  a parser, mapper, or transform is converted to `None` when the field is
  projected; positive and negative infinity are rejected. Identity fields
  reject every non-finite float rather than treating it as missing. Series
  values must be scalars or non-empty flat lists of scalars; mappings, tuples,
  empty lists, and nested lists are rejected.
- Every `id` must be unique across both `features` and `targets`. Scaler and
  metadata operations plus postprocess policies use this shared vector-ID
  space.
- `scale: true` scales assembled scalar or fixed-length list values with the
  managed `build/scaler.json` artifact when a dataset output is produced.
  Fitting options belong to the scaler operation and cannot be overridden per
  vector. `None` and transient `NaN` remain missing; other nonnumeric values
  and infinity fail. Intrinsically list-valued fields are fitted independently
  by position. Lists created from scalar `sequence` or `collect` inputs use the
  scalar series' shared statistics at every position.
- `sequence` emits `SeriesSequence` windows and accepts `size` plus
  optional `stride` (default `1`). Regularize cadence with ordered transforms
  before series projection when contiguous ticks are required. The resolved
  stream partition keeps every independent series in one contiguous ordered
  group. Sequence inputs must be scalar; Jerry does not implicitly create or
  flatten nested list values.
- `collect` requires exactly `size` ordered scalar values assigned to each sample
  by `sample.rounding`. For hourly samples, `floor` collects 10:00 and 10:30 into
  10:00; `ceil` collects 10:30 and 11:00 into 11:00. With `exact`, records must
  share an exact grid timestamp. Zero values leave that series absent. `None` and
  cadence placeholders count as positions; underfilled and overfilled buckets
  fail. Collection validates cardinality, not temporal spacing, so regularize
  the stream first when positions must follow a fixed grid.
- `sequence` and `collect` are mutually exclusive. Without either policy, each
  concrete series ID may emit at most one value per sample bucket.
- Feature and target series support `scale`, `sequence`, and `collect`; targets
  additionally require `horizon`. A collected target's horizon must cover its
  latest supporting observation. Series configuration does not accept
  arbitrary transform entry-point clauses.
- `split` first assigns each sample one primitive label. Hash splits assign from
  the complete sample key and require `ratios`. Time splits require ordered
  `intervals` with unique IDs and strictly increasing endpoints. Every interval
  except the last has an exclusive `until` timestamp; the last interval omits
  `until` and is the open-ended tail.
- Every split defines one or more `folds`. Each fold has an `id`, at least one
  training label, and optional validation and test labels. Its published output
  IDs are `<fold-id>.train`, `<fold-id>.validation`, and `<fold-id>.test` for
  the nonempty roles. Labels omitted from every fold are purge/embargo
  intervals and are not published.
- For time folds with future targets, Jerry removes each role's trailing
  samples unless `sample time + maximum target horizon` is strictly before the
  next nonempty role starts. Equality is excluded. Omitted purge/embargo
  intervals may absorb that support window. The final nonempty role has no
  later fold boundary.
- Fold labels must exist as hash-ratio labels or time-interval IDs and cannot
  belong to two roles within the same fold. A fold ID names the output plan; it
  does not need to match any interval ID. Time-fold roles must be chronological.
  The same primitive label may be reused in another fold, enabling expanding
  walk-forward plans:

  ```yaml
  split:
    mode: time
    intervals:
      - id: period_0
        until: "2020-01-01T00:00:00Z"
      - id: period_1
        until: "2021-01-01T00:00:00Z"
      - id: period_2
        until: "2022-01-01T00:00:00Z"
      - id: period_3
        until: "2023-01-01T00:00:00Z"
      - id: period_4
    folds:
      - id: fold_0
        train: [period_0]
        validation: [period_1]
      - id: fold_1
        train: [period_0, period_1, period_2]
        validation: [period_3]
        test: [period_4]
  ```

- A simple hash holdout uses the same fold contract:

  ```yaml
  split:
    mode: hash
    seed: 42
    ratios: { train: 0.8, validation: 0.1, test: 0.1 }
    folds:
      - id: holdout
        train: [train]
        validation: [validation]
        test: [test]
  ```

- Hash-ratio mappings are canonicalized by label, so YAML key order does not
  change sample assignment. Hash splits reject sequences, positive target
  horizons, selected cross-sectional streams, and selected stream dependencies
  that derive records across timestamps. Cross-timestamp dependencies include
  lag, lead, rolling, temporal filling and completion, and as-of joins without
  `max_age: 0s`. Use a time split for temporal datasets.
- `postprocess.features` and `postprocess.targets` filter
  complete rows after typed conformance.
- `ids` is optional. Sample filters default to every declared ID.
  Empty, duplicate, or unknown IDs are errors.
- Execution order is fixed: typed conformance, then feature and target sample
  filters.
- Postprocess never selects columns from observed coverage. Use coverage reports
  to evaluate the declared series explicitly.
- Postprocess does not mutate values. Configure missing-value repair on the
  ordered record stream before feature extraction.

### Scaler Operation Override

The core scaler already has the defaults below. Create
`operations/scaler.yaml` only to change them:

```yaml
output: build/scaler.json
with_mean: true
with_std: true
epsilon: 1.0e-12
```

- An unsplit dataset stores one standard scaler fitted from all samples. A split
  dataset stores one scaler per dataset fold, fitted from that fold's `train`
  labels. Fold definitions belong only in `dataset.yaml`; the scaler operation
  does not duplicate split policy.
- `with_mean`, `with_std`, and positive finite `epsilon` are build-time options
  stored in the artifact and used unchanged at runtime.
- Every train, validation, and test output in a fold uses that fold's scaler.
  If a primitive label is reused across folds, its values are scaled separately
  for each output. Sequence windows are constructed before scaling, so every
  value in one output sequence uses the same fold scaler.
- `build/metadata.json` (from the `metadata` operation) is the canonical typed
  vector contract. It records feature/target identifiers (including
  partitions), scalar/list kinds and lengths, present/null counts, inferred
  value types, per-partition timestamps, and the dataset window. Mixed
  scalar/list values, empty lists, and varying list lengths fail metadata
  generation. Configure `sample.window_mode` with
  `union|intersection|strict` (default `intersection`) to control how
  start/end bounds are derived. `union` spans every series, `intersection`
  intersects base-series ranges, and `strict` intersects every partition.
- Artifact operation execution order comes from the typed dependency graph. Runtime
  commands prepare the union of all selected profiles' requirements once;
  explicit build profiles remain separate artifact roots.
- Profile `order` is authoritative for profile execution. The dependency graph
  orders internal artifact jobs but never changes the order of serve, inspect,
  or build profiles.
- Profiles select operations by ID through `operation`.
- Build profiles reference artifact operations. Serve and inspect profiles
  reference runtime operations. The built-in `dataset` operation uses
  `core.runtime.dataset`; custom runtime entry points are also supported.
- Observability defaults (visuals/logging outputs) belong in profile files (`serve.<name>.yaml`, `build.<name>.yaml`, `inspect.<name>.yaml`) or per-kind defaults (`<kind>.defaults.yaml`).

---

### Versioning & Reproducibility

- Jerry outputs are deterministic given a fixed config, plugin code, and source snapshot.
- A command uses the validated configuration and environment snapshot loaded at
  startup. Local source files and glob inventories are rechecked around every
  artifact operation; if they change, the command fails rather than registering
  an artifact from a mixed source generation as current. Configuration edits are
  observed by the next command.
- Core artifact hashes cover each artifact's effective typed dependency and
  source closure. Plugin artifact operations conservatively cover the complete
  dataset and stream catalog because they do not declare inputs. Hashing does
  not inspect Python plugin source. Increment `project.yaml`'s
  `artifact_revision` when parser, mapper, combine, transform, or custom
  artifact code changes artifact semantics without changing config.
- `--artifact-mode rebuild` rebuilds artifacts only in the current workspace. It
  is useful for a local refresh, but it does not communicate a semantic change
  to other workspaces; commit an incremented `artifact_revision` for that.
- `jerry serve` selects profiles by name and is reproducible when inputs and config are unchanged.
- A git tag on the workspace (plus the plugin repo) can represent a dataset “version” you can always rebuild.
- This pairs well with DVC: let DVC track raw inputs, and regenerate derived datasets from the tagged Jerry config when needed.
- Still use DVC for outputs when rebuilds are too expensive, transforms are non-deterministic, or sources are not snapshot-stable.

---
