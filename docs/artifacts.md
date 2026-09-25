# Artifacts

Core artifact operations are generated for each dataset as
`dataset.<id>.<kind>`, where kind is `scaler`, `series`, `metadata`, or
`coverage_stats`. Their registry keys are those operation IDs. Explicit operation
IDs come from YAML filenames, such as `operations/schedule.yaml`. Default outputs
are relative to `paths.artifacts`:

- `datasets/<id>/series/manifest.json` plus one compressed, globally ordered companion
  under `datasets/<id>/series/manifest.data/`: durable sparse sample inputs consumed
  by dataset assembly. Each row stores its sample key once with all available
  feature and target values. A value must be `None`, `bool`, `int`, `float`,
  `str`, or a non-empty flat list containing those scalar types. Mappings,
  tuples, nested lists, and other Python objects fail the build instead of
  being flattened or converted to strings. Sample-key components accept the
  same scalar types except `None`. Each successful build publishes a new
  immutable generation. The manifest records its row counts and content
  digest.
  Project commands hold one artifact-workspace lock; generations no longer
  referenced by the manifest are pruned only after the locked command finishes.
- `datasets/<id>/scaler.json`: managed scaler statistics. Unsplit datasets store one
  standard scaler; split datasets store one scaler fitted from each fold's
  training labels. Each scaler stores settings and statistics together for
  every concrete series.
- `datasets/<id>/metadata.json`: the typed feature/target contract used during
  postprocess, including identifiers, scalar/list kinds, fixed list lengths,
  coverage counts, value types, sample domain, and resolved dataset window.
  Unsplit datasets use the global catalog directly. Split datasets additionally
  store one training-owned schema and one output-domain contract per fold.
- `datasets/<id>/coverage_stats.json`: bounded assembled or postprocessed availability
  counters used by coverage inspection. It never stores per-sample status maps.
- Schedule operation outputs: named expected-timestamp sets used by
  `ensure_schedule` transforms. Their output paths are operation-defined.

The dependency graph is explicit: schedule artifacts referenced by configured
dataset streams feed the scaler and series artifacts; series feeds metadata;
`coverage_stats` depends on metadata. Matrix inspection reads series directly
and enforces a configured cell bound instead of expanding the `coverage_stats`
artifact. Nested schedule artifacts are rejected because that dependency is not
yet representable safely.

A schedule operation is explicit:

```yaml
# operations/schedule.yaml
product: schedule
stream: exchange.sessions
partition_by: []
output: build/schedule.jsonl
```

`partition_by` is required. An empty list declares one global schedule; named
fields declare one schedule per partition. A stream opts into that schedule
where timestamp completion belongs:

```yaml
transforms:
  - { operation: ensure_schedule, schedule: schedule }
```

Without `ensure_schedule`, a stream keeps its observed timestamps unchanged.
A schedule defines expected timestamps, not fields or value shapes; those
contracts belong to `metadata`, not to a schema-like schedule artifact. The
schedule partition fields must exactly match the consuming stream. Completion
preserves source records outside the schedule and inserts placeholders only for
partitions present in the source. A placeholder keeps its timestamp and
partition identity while clearing payload fields.

Coverage treats null values as uncovered. Base and scalar-column coverage is
the number of non-null samples divided by total samples. List-column coverage
is observed non-null elements divided by `total_samples * list_length`, so
absent samples and null list elements share the same explicit denominator.

Build state lives at `artifacts/_system/build/state.json`. An entry is current
only when its own artifact hash, declared output path, primary file,
companion-file fingerprints, and dependency chain are all current. Each core
artifact hash covers its typed configuration and dependency closure, including
only streams and local source snapshots that can feed that artifact. Unrelated
operations, streams, and sources therefore do not invalidate it. Every hash
includes `project.yaml:artifact_revision`; runtime-operation settings do not
enter it. Plugin artifact operations receive the full runtime but cannot declare
inputs, so their hashes conservatively cover the bound dataset and complete stream
catalog. Runtime hydration registers only current artifacts;
orphaned, missing, altered, stale, and incomplete chains are left unavailable.
Commands targeting the same artifacts root cannot overlap; a second command
fails before reading or mutating managed artifacts.

Artifact fingerprints store size, modification time, change time, and SHA-256.
When only change time (`ctime`) differs, Jerry checks the content checksum before
invalidating the artifact. Permission and other metadata-only changes therefore
do not invalidate unchanged bytes. Unchanged timestamps use the fast stat check;
files with changed `ctime` are rehashed on each validation without updating state.
Build-state version 9 requires one rebuild of older managed artifacts to record
checksums; exported materializations are unaffected.

Jerry 10.0.2 advances the core artifact-cache revision to isolate mutable plugin
configuration between runtimes. It includes the numerical corrections from
10.0.1. `auto` rebuilds required artifacts created under the previous revision.
Project schemas and stored artifact formats remain unchanged.

Jerry 8 uses `schedule` consistently for the expected-timestamp artifact:

- `operations/model_grid.yaml` becomes `operations/schedule.yaml`.
- `core.artifact.ticks` becomes `core.artifact.schedule`.
- `grid_by` becomes the required `partition_by`; use `partition_by: []` for
  one global schedule.
- `build/model_grid.jsonl` becomes `build/schedule.jsonl`.
- Stream transforms replace
  `{ operation: ensure_ticks, artifact: model_grid }` with
  `{ operation: ensure_schedule, schedule: schedule }`.
- Python integrations replace `TicksTask` with `ScheduleTask`, `TickGrid` with
  `Schedule`, `read_tick_grid` with `read_schedule`, `EnsureTicksConfig` with
  `EnsureScheduleConfig`, and `EnsureTicksTransform` with
  `EnsureScheduleTransform`.

The old operation and transform names are removed. Reinstall an editable
checkout after upgrading so its entry-point metadata exposes
`core.artifact.schedule`, then rebuild the renamed artifact and its dependents.

Jerry 8 requires every target in `datasets/default.yaml` to declare `horizon`. Add
`horizon: 0s` for contemporaneous targets. Future-derived targets must declare
a conservative wall-clock duration that covers their latest supporting
observation. Positive horizons cannot use a hash split. With a time split,
Jerry automatically removes unsafe role tails and fits folded scalers over the
same eligible sample origins; an unsplit dataset has no role boundary to
enforce. A horizon change does not change the grouped series values, but it does
change folded scaler eligibility and the fold contracts stored in metadata.

Jerry 8 series manifests use format version 11. Version 9 introduced source
record versus cadence-placeholder provenance; version 10 replaced implicit
scalar-to-list aggregation with explicit fixed-size `collect`; version 11
enforces scalar or non-empty flat-list values. The series format version
participates in artifact fingerprints, so `auto` rebuilds series, metadata, and
dependent coverage artifacts from earlier versions. Projects that relied on
multiple scalar values in one sample bucket must declare `collect: N` or use a
finer `sample.cadence`. Structured values must be projected into explicit
scalar or flat-list series. Otherwise the rebuild fails instead of silently
changing shape. `require_current` requires those artifacts to be rebuilt first. Scaler
artifacts remain current because collection shapes already-fitted scalar
observations.

Metadata format version 4 stores the global catalog plus the explicit unsplit
or folded layout. The artifact cache generation is also incremented for the new
folded-scaler semantics and UTC record canonicalization. `auto` therefore
rebuilds stale v7 and pre-canonicalization artifacts; `require_current` requires a v8 build
first.

Jerry 7 renames the v6 `variable_records` artifact to `series`:

- `operations/variable_records.yaml` becomes `operations/series.yaml`.
- `profiles/build.variable_records.yaml` becomes `profiles/build.series.yaml`,
  and its `operation` value becomes `series`.
- `requires: [variable_records]` becomes `requires: [series]`.
- `core.artifact.variable_records` becomes `core.artifact.series`.
- `build/variable_records/manifest.json` becomes
  `build/series/manifest.json`.
- `--preview variables` becomes `--preview series`.
- Python imports replace `VariableRecordsTask` with `SeriesTask`.

The old build-state entry and `build/variable_records/` directory are ignored;
`auto` builds the new artifact and its dependents. They may be deleted manually
after the migration. Reinstall an editable checkout after upgrading so its
entry-point metadata exposes the renamed core artifacts.

Jerry 7 beta series manifests used one gzip file per configured series. Jerry 7
manifests use format version 8 and one grouped companion, avoiding an
open-file-per-series limit and repeated key parsing. The artifact fingerprint
includes the format version, so `auto` rebuilds beta series, metadata, and
coverage artifacts. YAML configuration and final dataset output are unchanged.

Jerry 7 metadata supports the three distinct window modes `union`,
`intersection`, and `strict`. The former `relaxed` mode was identical to
`union`. Project schema 5 moves this policy from metadata operation overrides
to `datasets/default.yaml:sample.window_mode`; use `union` when migrating a former
`relaxed` value. Metadata format version 3 records the narrower contract, so
`auto` rebuilds older metadata and dependent coverage artifacts.

Jerry 7 also renames the raw availability-counter artifact from `stats` to
`coverage_stats`:

- `operations/stats.yaml` becomes `operations/coverage_stats.yaml`.
- `profiles/build.stats.yaml` becomes `profiles/build.coverage_stats.yaml`, and
  its `operation` value becomes `coverage_stats`.
- Custom dependencies use `requires: [coverage_stats]`.
- `core.artifact.stats` becomes `core.artifact.coverage_stats`.
- `build/stats.json` becomes `build/coverage_stats.json`.

Python integrations likewise use `CoverageStatsTask`,
`CoverageStatsArtifact`, `COVERAGE_STATS`, and `COVERAGE_STATS_SPEC`; the old
names are removed.

The runtime `coverage` operation and `inspect.coverage.yaml` profiles are
unchanged. The old build-state entry and `build/stats.json` are ignored; `auto`
builds the renamed artifact, while `require_current` requires it to exist first. The old
file may be deleted manually after the migration.

Jerry 6 also removes the separate `schema` artifact because `metadata` now owns
the complete typed feature/target contract. Delete build profiles whose
operation is `schema` (including the generated `build.schema.yaml`), remove
`operations/schema.yaml` overrides, and replace `schema` entries in `requires`
with `metadata`. Update scaffolded plugin dependencies from
`jerry-thomas>=5.0.7` to `jerry-thomas>=6.0.0`. The old build-state entry and
schema output (by default `build/schema.json`) are ignored and may be deleted
after the migration.

The v7 Python layer replaces the following types under the former
`datapipeline` namespace (Jerry 9 uses `jerrythomas`):
`datapipeline.config.dataset.variable.VariableConfig` with
`datapipeline.config.dataset.series.SeriesConfig`, and replaces
`datapipeline.domain.variable.VariableRecord` / `VariableSequence` with
`datapipeline.domain.series.SeriesRecord` / `SeriesSequence`. Preview stage
`variables` is now `series`. Series ID encoding and final sample values are
unchanged; the internal artifact rows are grouped by sample key.

Serve, inspect, and materialize use one command-wide `artifact_mode` for their
prerequisite phase. Its precedence is CLI `--artifact-mode`, then the matching
`<command>.defaults.yaml`, then the built-in `auto`:

- `auto`: build missing/stale requirements and reuse current dependencies.
- `rebuild`: rebuild the selected dependency closure.
- `require_current`: do not build; fail if a selected runtime requirement is unavailable.

Before any selected serve or inspect profile runs, Jerry unions their artifact
requirements and prepares that union once. The graph orders internal artifact
jobs; profile `order` remains authoritative for the subsequent runtime actions.
Concrete profiles do not carry individual artifact modes. Custom artifact
dependencies must have a configured producer; core artifact producers are
always available.

Before any selected materialize profile runs, Jerry similarly unions the
artifact requirements of its selected streams and prepares them once.
For built-in transforms, these are schedule IDs referenced by `ensure_schedule`
operations on the selected or upstream streams. Dependencies
hidden inside plugin code are not inferred. Materialize profiles do not carry
individual artifact modes.

The shared prerequisite phase has its own visual and logging envelope. Its
observability precedence is CLI, then `<command>.defaults.yaml`, then built-ins;
settings on a concrete profile apply only while that profile runs. A bare
execution-scoped log target writes the shared phase to
`logs/<command>.artifacts.log`.

Runtime operations may add `requires: [artifact_operation_id, ...]` when they
need artifacts beyond the built-in operation requirements. These IDs enter the
same dependency closure and must have declared, active producers. A CLI or
command-default heartbeat applies to the shared prerequisite build; heartbeat
values defined on individual profiles apply only when those profiles run.

Build profiles remain explicit roots for `jerry build` and retain their own
`mode`. They are not consulted by `serve`, `inspect`, or `materialize`.

---

## Splitting & Serving

`datasets/default.yaml:split` has two explicit responsibilities:

- `mode: hash` assigns one deterministic label from the complete sample key.
- `mode: time` assigns one interval ID from ordered, exclusive `until` timestamps.
- `folds` groups hash labels or time-interval IDs into named train, validation,
  and test outputs. Entries may be reused across folds, which supports expanding
  walk-forward training windows. Entries omitted from every fold are
  purge/embargo intervals and are not published.
- Output IDs are `<fold-id>.<role>`, such as `fold_1.train`. A full serve
  publishes every configured fold output; profile `include_outputs` can narrow
  that set. Without a dataset split, serve emits one combined stream.
- Preview bypasses split fanout. Record stages emit once per unique referenced
  stream, `series` emits once per configured feature or target, and sample
  stages emit one combined output.
- Split datasets fit one scaler from each fold's `train` labels. Every output in
  a fold uses that fold's scaler.
- Every target declares a conservative elapsed `horizon`. Time folds remove a
  role's trailing samples when their maximum target support reaches or crosses
  the next nonempty role. Omitted intervals can provide an explicit
  purge/embargo gap. Fold scalers use the same eligible sample origins.
- Hash splits cannot be combined with sequenced features or targets because
  overlapping windows could share observations across hash partitions. Use a
  time split for sequence datasets.
- Hash splits likewise reject positive target horizons because temporal support
  cannot be isolated by a key hash.
- Selected stream dependencies that derive records across timestamps are also
  rejected. This includes lag, lead, rolling, temporal filling and completion,
  and as-of joins unless `max_age: 0s` makes matching exact.

Canonical series artifacts remain unscaled and independent of fold
selection. The scaler artifact records the fold-specific statistics needed at
runtime.

---
