# Jerry Thomas plugin

Minimal plugin skeleton for the Jerry Thomas (datapipeline) runtime.

## Quick start

```bash
python -m pip install -e .

# One-stop wizard: source YAML + DTO/parser + domain + mapper + stream.
jerry inflow create

# Complete the generated files described below, then register the new entry points.
python -m pip install -e .

jerry serve --dataset your-dataset --limit 3
```

## After scaffolding: what you must edit

- `your-dataset/sources/*.yaml`
  - Replace placeholders (`path`/`url`, headers/params, delimiter, etc.)
  - Prefer `${env:NAME}` for secrets or machine-local paths instead of literal values
- `your-dataset/streams/*.yaml`
  - Map sources into canonical records, derive streams, broadcast shared
    temporal records, attach point-in-time lookups, or align multiple streams.
- `your-dataset/.env.example`
  - Copy to `.env` next to `project.yaml` for local dataset-specific secrets and paths
- `your-dataset/dataset.yaml`
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

YAML config (dataset project root):

- `your-dataset/`
  - `project.yaml` (paths and globals)
  - `sources/*.yaml` (raw source definitions)
  - `streams/*.yaml` (source-backed, derived, exact/as-of fan-in, and aligned streams)
  - `dataset.yaml` (features, targets, split, and postprocess policy)
  - `profiles/{serve,build,inspect,materialize}.<name>.yaml` (profiles; optional overrides)
  - `profiles/{serve,build,inspect,materialize}.defaults.yaml` (optional per-kind defaults)
  - `operations/*.yaml` (optional core overrides and custom operations)

Profile sequencing:

- Each concrete profile is one mapping in `<command>.<name>.yaml`; the filename
  supplies both command and name. Defaults use `<command>.defaults.yaml`.
- Profiles execute by `order` (ascending); unset falls back to filename order.
- Materialize profiles name a stream and an exact durable JSONL output; they do
  not reference an operation.
- Build profiles reference artifact operations; serve and inspect profiles
  reference runtime operations through `operation`. Core operations are
  available without YAML declarations.
- Before selected serve or inspect profiles run, their artifact requirements are
  combined and prepared once according to `artifact_mode: AUTO|FORCE|OFF`.
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
