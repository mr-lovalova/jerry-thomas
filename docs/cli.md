# CLI Reference

All commands live under the `jerry` entry point (`src/datapipeline/cli/app.py`).
Pass `--help` on any command for flags.
All commands that take a project accept either `--project <path/to/project.yaml>` or `--dataset <alias>` (from `jerry.yaml datasets:`).

Profile commands run enabled profiles by default. `--profile <name>` selects
that profile explicitly, including one configured with `enabled: false`.

With `--visuals on` in an interactive terminal, runtime commands show one live
pipeline row with the active input or stage. `--log-level DEBUG` expands that view
to one row per active input or stage. File-backed sources include the current file
and position (`2/17`). Determinate bars appear only when the item total is known; otherwise
the timer and current activity remain visible. Visuals are independent of log
filtering.
The final command summary uses the same Rich styling when any selected command
phase has visuals enabled.

### Preview Stages

- `jerry serve --project <project.yaml> --preview <stage> --limit N [--log-level LEVEL] [--visuals on|off] [--heartbeat-interval SECONDS]`
  - `input`: values entering the selected stream: parsed source values,
    completed upstream records, broadcast input pairs, or aligned input tuples.
  - `canonical`: domain records after source mapping or fan-in combining;
    derived streams pass their input through unchanged at this boundary.
  - `records`: records after configured transforms and ordering.
  - `series`: ordered feature/target records after sequence construction and
    before `collect` assembly. Values remain unscaled because scaling is
    selected by the full dataset output's fold.
  - `samples`: assembled samples before postprocess.
  - `postprocess`: samples after the configured postprocess pipeline.
    Preview stages remain unscaled; omit `--preview` to apply each selected
    fold's scaler.
  - Record stages emit once per unique configured record stream, `series`
    emits once per feature/target, and sample stages emit one combined stream.
  - Omit `--preview` to run the full pipeline and output persistence.
  - Use `--log-level DEBUG` for full debug output; the CLI default is `INFO`.
  - Before runtime execution, Jerry combines the artifact requirements of all
    selected profiles and prepares that union once. The artifact graph orders
    those internal jobs; it never changes profile order.
- `jerry serve --project <project.yaml> --output-transport <stdout|fs> --output-format <jsonl|csv|parquet|pickle> [--output-view flat|raw] [--output-encoding <codec>] [--output-compression gzip] --limit N [--artifact-mode AUTO|FORCE|OFF] [--log-level LEVEL] [--visuals on|off] [--heartbeat-interval SECONDS] [--profile name]`
  - Conforms samples to the declared schema and applies configured row filters before emitting. A configured dataset split routes a full dataset serve to one fs output per fold role, named `<profile-or-filename>.<fold-id>.<role>.<ext>`; profile `include_outputs` can narrow the set using IDs such as `fold_0.train`. Preview emits one combined stage and cannot be combined with explicit `include_outputs`. `--limit` applies separately to each output.
  - Use `--output-transport fs --output-format jsonl --output-directory build/serve` (or `csv`, `parquet`, `pickle`) to write outputs under `<output-directory>/runs/<run_id>/dataset/`.
  - `--output-view` controls payload shape:
    - `flat`: key + kind + flattened fields for JSONL/CSV; Parquet uses the
      explicit dataset-table columns documented below
    - `raw`: key + kind + raw object
  - If `--output-view` is omitted: `csv` and `parquet` use `flat`; `jsonl` and `pickle` use
    `raw`.
  - `csv` and `parquet` support only `flat`; `pickle` supports only `raw`.
  - Parquet is available for full dataset output and `samples`/`postprocess`
    previews. It uses an explicit dataset schema, bounded row groups, and
    internal Zstandard compression. Install `jerry-thomas[parquet]` first.
  - `--output-encoding` applies to fs `jsonl`/`csv` outputs (default `utf-8`).
  - `--output-compression gzip` applies to fs `jsonl`/`csv` outputs, including
    every preview stage and routed split output. Compression is never inferred
    from the filename.
  - Set `--log-level DEBUG` (or set `observability.logging.level: DEBUG` in the serve profile) to increase log detail while previewing a stage.
  - The built-in heartbeat interval is 60 seconds. Set `--heartbeat-interval 0` to disable logged pipeline heartbeats. Live progress remains enabled when visuals are on. The CLI value also controls the shared artifact prerequisite phase; concrete profile `observability.heartbeat_interval_seconds` begins applying only when that profile runs.
  - When multiple serve profiles exist, add `--profile <name>` to select a
    single profile; otherwise every enabled profile is executed in its exact
    configured order.
  - `artifact_mode` controls the command-level prerequisite phase: `AUTO`
    builds missing or stale artifacts, `FORCE` rebuilds the required closure,
    and `OFF` only accepts artifacts that are already current. The CLI
    `--artifact-mode` override applies to the whole command.
  - Artifact mode precedence is CLI `--artifact-mode`, then
    `serve.defaults.yaml`, then the built-in `AUTO`. It is command-wide and is
    not configured on individual serve profiles.
  - Argument precedence follows the order described under _Configuration & Resolution Order_.

### Build & Quality

- `jerry inspect --project <project.yaml> [--profile <name>] [--artifact-mode AUTO|FORCE|OFF] [--visuals on|off] [--heartbeat-interval SECONDS]`
  - Runs inspect profiles declared as `profiles/inspect.<name>.yaml`.
  - Without `--profile`, executes all enabled inspect profiles.
  - Use `--profile coverage` or `--profile matrix` to execute one profile.
  - Like `serve`, prepares the union of selected profiles' artifact requirements
    once, then executes the profiles in their exact configured order.
  - Profile `operation` values map to core or custom runtime operations. Core coverage and
    matrix operations require no YAML declarations.
  - `--limit N` caps samples for the matrix operation and is passed to custom
    runtime operations. Coverage is artifact-based and rejects `--limit`.
  - `--output-compression gzip` is available for filesystem JSONL and CSV
    inspection outputs.
  - Artifact mode precedence is CLI `--artifact-mode`, then
    `inspect.defaults.yaml`, then the built-in `AUTO`. It is command-wide and
    is not configured on individual inspect profiles.
- `jerry inspect --project <project.yaml> --profile matrix`
  - Typical matrix profile run. Matrix output format/path is controlled by the
    inspect profile and output flags. The matrix operation is bounded by its
    `max_cells` option and can inspect assembled or postprocessed samples.
    `--limit N` caps samples after that stage; `max_cells` remains the separate
    bound on scalar cells and individual list elements.
- `jerry build --project <project.yaml> [--profile <name>] [--force] [--visuals on|off] [--heartbeat-interval SECONDS]`
  - Regenerates core or custom artifacts when that artifact's hash changes.
  - If build profiles are defined, enabled profiles run by default; use
    `--profile` to select one profile.
  - Each build profile executes its configured artifact `operation`; selected
    profiles must reference distinct operations.
  - Build profiles remain explicit artifact roots and execute in their configured
    profile order. The graph orders only the internal dependency jobs needed by
    each root; it never reorders the profiles. A selected dependency profile
    must be ordered before a selected dependent profile.
- `jerry materialize [--profile <name>] [--output <path.jsonl|path.jsonl.gz>] [--overwrite|--no-overwrite] [--artifact-mode AUTO|FORCE|OFF] [--visuals on|off] [--heartbeat-interval SECONDS]`
  - Runs every enabled `profiles/materialize.<name>.yaml` file in configured
    order, or one profile selected by `--profile`.
  - Checks every selected output before the first profile starts writing.
  - Collects the selected streams' artifact requirements and prepares their
    union once. `--artifact-mode` overrides `materialize.defaults.yaml`; the
    built-in mode is `AUTO`.
  - Shared prerequisite visuals and logs use CLI settings, then
    `materialize.defaults.yaml`; concrete profile overrides begin afterward.
  - A CLI overwrite choice applies to every selected profile. Without it, each
    profile uses its own `overwrite` setting or `materialize.defaults.yaml`.
  - `--output` overrides one selected profile and therefore requires
    `--profile`.
  - The concrete output suffix selects compression: `.jsonl` writes plain
    JSONL and `.jsonl.gz` writes gzip JSONL.
- `jerry clean [--yes] [--older-than <age>]`
  - Lists stale sort spill directories by default.
  - Add `--yes` to remove them.

### Scaffolding & Reference

- `jerry plugin create <package> --out <dir>`
  - Generates a self-contained plugin workspace (pyproject, package skeleton,
    local `jerry.yaml`, and config templates).
- `jerry demo create`
  - Generates a standalone demo workspace at `./demo/`. Run it from that
    directory; the command does not modify a parent `jerry.yaml`.
- `jerry inflow create`
  - Wizard to scaffold a complete source-backed stream (source YAML + parser/DTO + mapper + stream).
- `jerry stream create [--identity]`
  - Writes a source-backed, broadcast, or aligned stream configuration. The
    wizard selects a mapper for source-backed streams and a combiner for fan-in
    streams; it does not create Python code.
    Use `jerry inflow create` when the source, DTO, parser, domain, or mapper
    must also be scaffolded.
  - Broadcast streams select one partitioned primary stream, one unpartitioned
    temporal stream, and a registered combine entry point. Matching is exact by
    timestamp.
  - Aligned streams require at least two existing input streams and the name of
    a combine entry point already registered in `datapipeline.combiners`. The
    combine function receives one matching record from each input in the
    selected order and returns one record or `None`.
  - `--identity` applies only to source-backed streams.
- `jerry source create <provider>.<dataset> --transport fs|http|synthetic --format csv|json|jsonl|parquet`
  - Creates a source YAML only (no Python code).
  - Parquet is supported only for local `fs` files and globs. It yields row
    mappings with native Parquet values to the configured parser; text encoding,
    external gzip compression, and HTTP Parquet are not supported.
- `jerry domain create <name>`
  - Adds a `domains/<name>/` package with a `model.py` stub.
- `jerry list sources|domains`
  - Introspect configured source aliases or domain packages.

---
