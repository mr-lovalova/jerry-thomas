# Pipeline Architecture

Each command reads and validates the project, dataset, source, stream,
and operation YAML once into a `ProjectDefinition`. That definition is the
command's configuration snapshot. Planning, validation, artifact hydration,
and execution do not re-read configuration or environment values. Changes are
picked up by the next command. Its per-artifact hashes cover each producer's
typed dependency and source closure plus the project's artifact revision.
Mutable `Runtime`
instances are compiled from the definition without reading configuration files.
Each runtime owns a deep copy of the dataset and source/stream configuration,
including nested plugin arguments. Runtime changes cannot alter the definition
or another compiled runtime.

## Runtime streams

Every canonical stream ID has exactly one entry in `Runtime.streams`:
`SourceRuntimeStream`, `DerivedRuntimeStream`, `CrossSectionRuntimeStream`,
`BroadcastRuntimeStream`, `AsOfRuntimeStream`, `BroadcastAsOfRuntimeStream`, or
`AlignedRuntimeStream`.

A source-backed stream owns an external source, mapper, preprocess operations,
partition identity, and ordering policy. A derived stream names one upstream
stream and adds ordered transforms. A broadcast stream attaches an
unpartitioned temporal input to a partitioned primary input. As-of streams
attach the latest eligible same-partition or global lookup. An aligned stream
intersects two or more inputs with the same partition identity. Every fan-in
stream owns a prepared combine stage and inherits the primary partition
identity; only source-backed streams declare it. Single-input streams are
flattened, while fan-in streams use the explicit boundaries described below.
The strict config models keep source mapping and fan-in behavior separate.

Dataset `sample.keys` select the partition fields represented in row identity.
The remaining partition fields deterministically suffix series IDs in declared
order. This derives long, wide, and hybrid layouts without a separate stream
series-identity setting. An unsplit dataset uses the global metadata catalog as
its schema. A split dataset stores one training-owned schema per fold and uses
that schema for every output role in the fold. Genuine IDs observed in a fold's
validation or test data must therefore already be established by the fold's
eligible training rows; cadence placeholders do not establish columns.

Configured loader, parser, map, and combine entry points are resolved while
compiling a runtime from the definition. The resulting callables are stored on the runtime stream. There are
no parallel source, mapper, transform, or debug registries to keep synchronized.

```mermaid
flowchart LR
  config["project configuration"] --> definition["ProjectDefinition<br/>validated snapshot"]
  definition --> runtime["Runtime.streams<br/>id -> compiled stream"]
  runtime --> records["linear record pipeline"]
  records --> projection["feature/target projection"] --> inputs["series"]
  inputs --> samples["sample assembly"] --> postprocess["postprocess stages"]
  postprocess --> output["samples / output"]
```

## Stream pipelines

A source-backed stream is one source followed by explicit stages:

```text
stream:<id>
  input: open_source
  stage: map_records
  <one stage per configured preprocess transform>
  stage: ensure_record_order
  <one stage per configured ordered transform>
```

A single-input derived stream flattens its upstream pipeline into the same run. The
upstream names are qualified, so debug output identifies both the root stream
and the stage that produced a record:

```text
stream:<id>
  input: stream:<upstream>/open_source
  stage: stream:<upstream>/map_records
  stage: stream:<upstream>/...
  <one stage per configured ordered transform>
```

The derived stream reuses upstream canonical order directly. It does not add
identity mapping or sorting stages.

A cross-sectional stream is a single-input boundary with an explicit order
change:

```text
stream:<id>
  <qualified upstream input and stages>
  stage: order_cross_sections       # [time, *partition_by]
  stage: apply_cross_section        # one exact timestamp at a time
  stage: ensure_record_order        # [*partition_by, time]
  <one stage per configured ordinary transform>
```

Both sorts use the bounded external sorter. The timestamp group is the only
in-memory population. Restoring canonical order keeps every existing series,
artifact, preview, and history-based transform contract unchanged.

Aligned streams are a symmetric fan-in boundary. Their pipeline input is
`align_inputs`; it owns opening, validating, merging, and closing the configured
ordered inputs.
Those input pipelines run internally without starting competing visual pipelines.
The aligned pipeline remains the single observable boundary: `align_inputs` reports
its current progress, then `combine_records` applies the configured combine
function before ordered transforms.

Broadcast streams are the asymmetric fan-in boundary. Their `broadcast_inputs`
pipeline input fully indexes one finite, unpartitioned input by time,
then pairs each partitioned primary record with the exact timestamp match. It
rejects missing, duplicate, and unordered keys, but ignores indexed timestamps
that the primary never uses. Index memory is proportional to the number of
broadcast records. `combine_records` receives read-only inputs; the indexed
broadcast record object is reused across primary partitions at that timestamp.

As-of streams are the backward-looking fan-in boundaries. `as_of_inputs`
advances two equally partitioned, canonically ordered streams together and
retains only the latest eligible lookup. `broadcast_as_of_inputs` indexes one
finite, unpartitioned lookup history, then performs a binary search for each
partitioned primary record. Both reject duplicates and ordering violations.
The former uses constant matching memory; the latter uses memory proportional
to the lookup history so it can reuse time when primary partitions restart.

The runner accepts one explicit input followed by ordered stages. It owns lazy
iteration, closing, output counts, timings, and sampled progress. It has no
generic fan-out, keyword-input, nested-parent, or nested-pipeline
machinery.

Preprocess and ordered transform configuration is a strict discriminated union. Each
entry names its built-in operation directly:

```yaml
preprocess:
  - operation: floor_time
    cadence: 1d

transforms:
  - operation: rolling
    field: close
    window: 20
    statistic: mean
```

Pydantic validates operation-specific fields and rejects extras before the pipeline
is built. Pipeline construction uses explicit type dispatch to create the transform.
Built-in transforms use their explicit typed constructors. The `custom` branch
resolves a registered transform factory with the fixed `(args, partition_by)`
contract; it does not inspect signatures or inject arbitrary keywords.

Sequence construction remains a series-pipeline stage rather than preprocess
or an ordered stream transform. Scaling is applied later, after a dataset fold
is selected, so every train, validation, and test output uses its fold's fitted
scaler. This keeps numerical scaling separate from dataset shaping and split
policy.

## Sample postprocess

`dataset.yaml:postprocess` is validated into `PostprocessConfig`, with separate
typed policies for feature and target sample filtering.

The dataset pipeline has one fixed postprocess order:

```text
dataset
  assemble_samples
  conform_features
  conform_targets (or reject_undeclared_targets)
  optional filter_samples_by_features
  optional filter_samples_by_targets
```

Configuration can enable and parameterize row filtering, but cannot reorder
phases, select columns from observed coverage, or mutate vector values. Vector
metadata is loaded once at the boundary where its validated contract is needed.

## Preview boundaries

`jerry serve --preview` selects a semantic boundary rather than an execution
position: `input`, `canonical`, `records`, `series`, `samples`, or `postprocess`.
The meaning stays stable when optional pipeline stages are added or removed. See the
README's **Preview stages** section for the output behavior of each boundary.
