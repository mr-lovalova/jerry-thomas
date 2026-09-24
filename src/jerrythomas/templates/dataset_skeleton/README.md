# Jerry Thomas plugin

Minimal plugin skeleton for the Jerry Thomas runtime.

## Quick start

```bash
python -m pip install -e .

# One-stop wizard: source YAML + DTO/parser + domain + mapper + stream.
jerry inflow create

# Complete the generated files described below, then register the new entry points.
python -m pip install -e .

jerry serve --project your-project --limit 3
```

## After scaffolding: what you must edit

- `your-project/sources/*.yaml`
  - Replace placeholders (`path`/`url`, headers/params, delimiter, etc.)
  - Prefer `${env:NAME}` for secrets or machine-local paths instead of literal values
- `your-project/streams/*.yaml`
  - Map sources into canonical records, derive streams, broadcast shared
    temporal records, attach point-in-time lookups, or align multiple streams.
- `your-project/.env.example`
  - Copy to `.env` next to `project.yaml` for local project secrets and paths
- `your-project/datasets/default.yaml`
  - Ensure `stream:` points at the stream id you created.
  - Select a `field:` for each feature/target (record attribute to use as value).
  - Ensure `sample.cadence` matches `^\d+(m|min|h|d)$` (e.g. `10m`, `1h`, `1d`). The scaffold fills it from the `${cadence}` project global.
  - Choose `sample.window_mode`: `union`, `intersection` (default), or `strict`.
  - Adjust the split ratios and `folds`. Fold output IDs use
    `<fold-id>.<role>`, such as `holdout.train`.

Reinstall the plugin after adding or editing entry points in `pyproject.toml`:

```bash
python -m pip install -e .
```

## Folder layout

YAML config (project root):

- `your-project/`
  - `project.yaml` (paths and globals)
  - `sources/*.yaml` (raw source definitions)
  - `streams/*.yaml` (source-backed, derived, exact/as-of fan-in, and aligned streams)
  - `datasets/default.yaml` (features, targets, split, and postprocess policy)
  - `profiles/{serve,build,inspect,materialize}.<name>.yaml` (profiles; optional overrides)
  - `profiles/{serve,build,inspect,materialize}.defaults.yaml` (optional per-kind defaults)
  - `operations/*.yaml` (explicit operations bound to datasets or streams)

Profile sequencing:

- Each concrete profile is one mapping in `<command>.<name>.yaml`; the filename
  supplies both command and name. Defaults use `<command>.defaults.yaml`.
- Profiles execute by `order` (ascending); unset falls back to filename order.
- Every concrete profile references an `operation`. Operations select the dataset
  or stream; profiles select output and execution settings.
- Dataset artifact operations are generated as `dataset.<id>.<kind>`. Runtime
  operations are declared under `operations/`.
- Before selected serve or inspect profiles run, their artifact requirements are
  combined and prepared once according to `artifact_mode: auto|rebuild|require_current`.
- The dependency graph orders only internal artifact jobs. It never changes
  profile order; build profiles remain explicit artifact roots.
- Selected build profiles must reference distinct operations.
- A selected dependency build profile must precede a selected dependent profile.
- Use multiple ordered profiles only when you want separate named build/runtime
  steps or different per-profile output and observability settings.

Python plugin code:

- `src/<package>/`
  - `dtos/` (DTO models)
  - `parsers/` (raw -> DTO)
  - `domains/<domain>/model.py` (domain record models)
  - `mappers/` (iterator mappings from parsed values to domain records)
  - `combiners/` (exact/as-of/aligned record combine functions)
  - `loaders/` (optional custom loaders)

## Learn more

- Preview stages and split/build timing: the Jerry Thomas runtime `README.md` ("Preview stages (serve --preview)").
- Deep dives: runtime `docs/config.md`, `docs/transforms/`, `docs/artifacts.md`, `docs/extending.md`, `docs/architecture.md`.
