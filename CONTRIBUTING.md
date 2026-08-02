# Contributing

This repo ships the Jerry Thomas runtime (`jerry-thomas`) and its CLI (`jerry`).

## Development Setup

```bash
python -m pip install -e .[dev]
python -m pytest
ruff check .
```

Data-path changes should follow the regression and mutation checks in
[docs/testing.md](docs/testing.md).

When iterating on configs and transforms, `jerry serve --preview <stage>` is the
fastest way to validate a semantic pipeline boundary.

## Scaffolding Internals (Internal API)

The CLI scaffolding commands are implemented under `src/jerrythomas/services/scaffold/`.
These functions are considered *internal* (they may change without a deprecation
policy); prefer using the CLI unless you are working on the runtime itself.

Key entrypoints:

- Plugin/demo
  - `jerrythomas.services.scaffold.plugin.scaffold_plugin` (used by `jerry plugin create`)
  - `jerrythomas.services.scaffold.demo.scaffold_demo` (used by `jerry demo create`)

- YAML config
  - `jerrythomas.services.scaffold.source_yaml.create_source_yaml` (used by `jerry source create` and `jerry inflow create`)

- Python stubs + entry points
  - `jerrythomas.services.scaffold.dto.create_dto` (used by `jerry dto create`)
  - `jerrythomas.services.scaffold.parser.create_parser` (used by `jerry parser create`)
  - `jerrythomas.services.scaffold.mapper.create_mapper` (used by `jerry mapper create`)
  - `jerrythomas.services.scaffold.loader.create_loader` (used by `jerry loader create`)
  - `jerrythomas.services.scaffold.domain.create_domain` (used by `jerry domain create`)

- Composition (wizard)
  - `jerrythomas.services.scaffold.stream_plan.execute_stream_plan` (used by `jerry inflow create`)

Entry points are injected/updated in plugin `pyproject.toml` via:

- `jerrythomas.services.scaffold.entrypoints.register_entry_point`
