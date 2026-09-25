# CLI Reference

All commands live under the `jerry` entry point (`src/jerrythomas/cli/app.py`).
Use `jerry <command> [subcommand] [options]`; pass `--help` on any command
for flags. Long option names must be written in full. Logging flags work before
or after command names, including after nested commands such as `list datasets`.
A logging option supplied at a deeper command level overrides that option at an
earlier level; repeat `--log-output` at the same level to select multiple targets.
`jerry env` reports the Python/Jerry environment and installed plugin providers:
package versions, entry-point groups/names, Python targets, and editable source
locations when available. It reads package metadata without importing plugins;
the list describes the installed environment, not a project's selected plugins.
Project commands accept `--project <alias|folder|project.yaml>`; aliases come from
`jerry.yaml projects:`. Omit it to use `default_project`.

Profile commands run enabled profiles by default. `--profile <name>` selects
that profile explicitly, including one configured with `enabled: false`.
The profile selects an operation; that operation selects its dataset or stream.
There is no CLI dataset selector.

`jerry list datasets --project <alias>` prints dataset IDs and versions.
`jerry list streams --project <alias>` prints stream IDs.
`jerry list profiles --project <alias>` includes each profile’s operation, binding,
and enabled state.

Profile commands validate every selected data and log destination before
activating filesystem logging. Data and logs cannot both use stdout, and
data files cannot overlap another selected file path. Distinct log files cannot
be nested, although profiles may intentionally share one exact log path. Global
logs must also stay outside the artifacts root and managed serve `runs`/`latest`
paths; use execution-scoped logging for managed command logs.

### CLI Overrides

For serve and inspect, each `--output-*` flag addresses one leaf of the profile's
`output` block; absent flags inherit from the profile. For example,
`jerry serve --output-directory /tmp/out` keeps the profile's transport, format,
encoding, and compression and writes to `/tmp/out` instead. The combined
configuration must still be a valid output; invalid combinations are rejected
with the same rules that govern profile files.

`--log-level` overrides the profile's `observability.logging.level`. When
`--log-output` targets are provided they replace the profile's
`observability.logging.outputs` list; omit the flags to use the configured
outputs unchanged.

With `--visuals` in an interactive terminal, runtime commands show one live
pipeline row with the active input or stage. `--log-level DEBUG` expands that view
to one row per active input or stage. File-backed sources include the current file
and position (`2/17`). Determinate bars appear only when the item total is known; otherwise
the timer and current activity remain visible. Visuals are independent of log
filtering.
The final command summary uses the same Rich styling when any selected command
phase has visuals enabled.

### Completed Run Results

```bash
jerry serve --project project.yaml --profile dataset --result-json
```

`--result-json` writes one JSON object to stdout after execution and publication
succeed. Data outputs must use filesystem transport and logs must use stderr or
filesystem transport. Conflicts are rejected before execution. Normal terminal
progress remains on stderr.

Both commands emit the same versioned structure:

```json
{
  "schema_version": 3,
  "runs": [
    {
      "receipt": "/research/interim/volatility.jsonl.gz.run.json",
      "schema_version": 4,
      "command": "export",
      "dataset_id": null,
      "dataset_version": null,
      "run_id": "2026-09-15T10-00-00-000000Z",
      "started_at": "2026-09-15T10:00:00+00:00",
      "finished_at": "2026-09-15T10:01:00+00:00",
      "status": "success",
      "notes": null,
      "preview": null,
      "split": null,
      "recipe": {"path": "volatility.jsonl.gz.recipe.json", "sha256": "0000000000000000000000000000000000000000000000000000000000000000"},
      "outputs": [{
        "profile": "volatility", "operation": "volatility",
        "stream": "equity.volatility", "output_id": null,
        "path": "volatility.jsonl.gz", "format": "jsonl", "view": "raw",
        "encoding": "utf-8", "compression": "gzip", "row_count": 1000,
        "fold": null, "size_bytes": 12345,
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
      }]
    }
  ]
}
```

Each entry contains the saved receipt fields plus its absolute `receipt` path.
Dataset-bound runs record `dataset_id` and `dataset_version`; stream runs use null.
Output paths are relative to the receipt's directory. Serve emits one entry per
run directory; export emits one per output, in profile order. No enabled
profiles produces `{"schema_version": 3, "runs": []}`. Failure emits no result
object; callers must check the exit code. Saved receipts are automatic even
without `--result-json`. See [saved runs](research.md#read-a-saved-run).

### Preview Stages

- `jerry serve --project <project.yaml> --preview <stage> --limit N [--log-level LEVEL] [--visuals | --no-visuals] [--heartbeat-interval SECONDS]`
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
- `jerry serve --project <project.yaml> [--profile name] --output-transport <stdout|fs> --output-format <jsonl|csv|parquet|pickle> [--output-view flat|raw] [--output-encoding <codec>] [--output-compression gzip] --limit N [--artifact-mode auto|rebuild|require_current] [--log-level LEVEL] [--visuals | --no-visuals] [--heartbeat-interval SECONDS]`
  - Conforms samples to the declared schema and applies configured row filters before emitting. A configured dataset split routes a full dataset serve to one fs output per fold role, named `<profile-or-filename>.<fold-id>.<role>.<ext>`; profile `include_outputs` can narrow the set using IDs such as `fold_0.train`. Record previews emit once per unique referenced stream, `series` once per configured feature or target, and sample previews once for the combined stage. Preview cannot be combined with explicit `include_outputs`. `--limit` applies separately to each output.
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
  - `artifact_mode` controls the command-level prerequisite phase: `auto`
    builds missing or stale artifacts, `rebuild` rebuilds the required closure,
    and `require_current` only accepts artifacts that are already current. The CLI
    `--artifact-mode` override applies to the whole command.
  - Artifact mode precedence is CLI `--artifact-mode`, then
    `serve.defaults.yaml`, then the built-in `auto`. It is command-wide and is
    not configured on individual serve profiles.
  - Argument precedence follows the order described under _Configuration & Resolution Order_.

### Build & Quality

- `jerry inspect --project <project.yaml> [--profile <name>] [--artifact-mode auto|rebuild|require_current] [--visuals | --no-visuals] [--heartbeat-interval SECONDS]`
  - Runs inspect profiles declared as `profiles/inspect.<name>.yaml`.
  - Without `--profile`, executes all enabled inspect profiles.
  - Use `--profile coverage` or `--profile matrix` to execute one profile.
  - Like `serve`, prepares the union of selected profiles' artifact requirements
    once, then executes the profiles in their exact configured order.
  - Profiles select explicitly declared output operations. Coverage and matrix
    operations bind a named dataset.
  - `--limit N` caps samples for the matrix operation and is passed to custom
    output operations. Coverage is artifact-based and rejects `--limit`.
  - `--output-compression gzip` is available for filesystem JSONL and CSV
    inspection outputs.
  - Artifact mode precedence is CLI `--artifact-mode`, then
    `inspect.defaults.yaml`, then the built-in `auto`. It is command-wide and
    is not configured on individual inspect profiles.
- `jerry inspect --project <project.yaml> --profile matrix`
  - Typical matrix profile run. Matrix output format/path is controlled by the
    inspect profile and output flags. The matrix operation is bounded by its
    `max_cells` option and can inspect assembled or postprocessed samples.
    `--limit N` caps samples after that stage; `max_cells` remains the separate
    bound on scalar cells and individual list elements.
- `jerry build --project <project.yaml> [--profile <name>] [--artifact-mode auto|rebuild|require_current] [--visuals | --no-visuals] [--heartbeat-interval SECONDS]`
  - Builds missing or stale managed artifacts with `auto`; `rebuild` rebuilds
    the selected dependency closure and `require_current` checks it without
    building. CLI `--artifact-mode` overrides each selected build profile's
    policy, which inherits from `build.defaults.yaml` and then `auto`.
  - If build profiles are defined, enabled profiles run by default; use
    `--profile` to select one profile.
  - Each build profile executes its configured artifact `operation`; selected
    profiles must reference distinct operations.
  - Build profiles remain explicit artifact roots and execute in their configured
    profile order. The graph orders only the internal dependency jobs needed by
    each root; it never reorders the profiles. A selected dependency profile
    must be ordered before a selected dependent profile.
- `jerry export --project <alias|folder|project.yaml> [--profile <name>] [--output-file <path.jsonl|path.jsonl.gz>] [--result-json] [--overwrite|--no-overwrite] [--artifact-mode auto|rebuild|require_current] [--visuals | --no-visuals] [--heartbeat-interval SECONDS]`
  - Runs every enabled `profiles/export.<name>.yaml` file in configured
    order, or one profile selected by `--profile`.
  - Each profile selects a `kind: output`, `entrypoint: core.records` operation.
  - Checks every selected output before the first profile starts writing.
  - Collects the selected streams' artifact requirements and prepares their
    union once. `--artifact-mode` overrides `export.defaults.yaml`; the
    built-in mode is `auto`.
  - Shared prerequisite visuals and logs use CLI settings, then
    `export.defaults.yaml`; concrete profile overrides begin afterward.
  - A CLI overwrite choice applies to every selected profile. Without it, each
    profile uses its own `overwrite` setting or `export.defaults.yaml`.
  - `--output-file` overrides one selected profile and therefore requires
    `--profile`.
  - The concrete output suffix selects compression: `.jsonl` writes plain
    JSONL and `.jsonl.gz` writes gzip JSONL.
  - Every output gets a `<filename>.run.json` receipt automatically. A hidden
    `.<filename>.lock` file coordinates publication across projects and stays
    in place for reuse.
  - `--result-json` prints the completed receipts described above. Stdout logging
    is rejected. Profiles commit independently; later failures retain earlier
    completed outputs and receipts.
  - `--overwrite` replaces the receipt with a running state before changing data.
    Failures leave an unreadable running/failed receipt, never an old successful
    receipt describing replacement data. A failed overwrite may preserve the old
    data file, but its receipt is invalidated.
- `jerry clean [--yes] [--older-than <age>]`
  - Lists stale sort spill directories by default.
  - Add `--yes` to remove them.

### Scaffolding & Reference

`source create`, `stream create`, and `inflow create` accept the same `--project`
alias, folder, or YAML path as runtime commands. Omit it to use the workspace
default or the plugin scaffold project. The selected project owns the YAML
outputs; Python scaffolding still belongs to `plugin_root`.

- `jerry plugin create <package> --out <dir>`
  - Generates a self-contained plugin workspace (pyproject, package skeleton,
    local `jerry.yaml`, and config templates).
- `jerry demo create`
  - Generates a standalone demo workspace at `./demo/`. Run it from that
    directory; the command does not modify a parent `jerry.yaml`.
- `jerry inflow create [--project <alias|folder|project.yaml>]`
  - Wizard to scaffold a complete source-backed stream (source YAML + parser/DTO + mapper + stream).
- `jerry stream create [--project <alias|folder|project.yaml>] [--identity]`
  - Writes a source-backed, broadcast, or aligned stream configuration. The
    wizard selects a mapper for source-backed streams and a combiner for fan-in
    streams; it does not create Python code.
    Use `jerry inflow create` when the source, DTO, parser, domain, or mapper
    must also be scaffolded.
  - Broadcast streams select one partitioned primary stream, one unpartitioned
    temporal stream, and a registered combine entry point. Matching is exact by
    timestamp.
  - Aligned streams require at least two existing input streams and the name of
    a combine entry point already registered in `jerrythomas.combiners`. The
    combine function receives one matching record from each input in the
    selected order and returns one record or `None`.
  - `--identity` applies only to source-backed streams.
- `jerry source create <provider>.<dataset> [--project <alias|folder|project.yaml>] --transport fs|http|synthetic --format csv|json|jsonl|parquet`
  - Creates a source YAML only (no Python code).
  - Parquet is supported only for local `fs` files and globs. It yields row
    mappings with native Parquet values to the configured parser; text encoding,
    external gzip compression, and HTTP Parquet are not supported.
- `jerry domain create <name>`
  - Adds a `domains/<name>/` package with a `model.py` stub.
- `jerry list sources|domains`
  - Introspect configured source aliases or domain packages.

---
