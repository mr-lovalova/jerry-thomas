# Extending the Runtime

## Migrating plugins to Jerry 9

The distribution remains `jerry-thomas` and the command remains `jerry`, but
the Python package and plugin entry-point groups now use `jerrythomas` instead
of `datapipeline`. Update plugin imports and group names, then reinstall the
plugin so its installed entry-point metadata is refreshed. Jerry 9 does not
provide a compatibility package or discover the former groups.

### Entry Points

Register custom components in your plugin’s `pyproject.toml`:

```toml
[project.entry-points."jerrythomas.loaders"]
demo.csv_loader = "my_datapipeline.loaders.csv:CsvLoader"

[project.entry-points."jerrythomas.parsers"]
demo.weather_parser = "my_datapipeline.parsers.weather:WeatherParser"

[project.entry-points."jerrythomas.mappers"]
time.ticks = "my_datapipeline.mappers.synthetic.ticks:map"

[project.entry-points."jerrythomas.combiners"]
air_density = "my_datapipeline.combiners.air_density:combine_air_density"

[project.entry-points."jerrythomas.operations.runtime"]
demo.report = "my_datapipeline.operations:run_report"
```

Each extension follows the contract of its entry-point group. A stream `map`
receives an iterator and returns an iterable. An aligned stream `combine`
receives one matching record from each configured input. A broadcast stream
`combine` receives the partitioned primary record followed by its exact-time
unpartitioned record. An as-of combiner receives the primary followed by the
latest eligible lookup; the lookup can be `None` when `require_match: false`.
All combiners return one record or `None`. Combiner inputs are read-only;
indexed broadcast records may be reused across primary partitions. Combiners
belong to `jerrythomas.combiners`, not the iterator-oriented
`jerrythomas.mappers` group. Mapper and combiner outputs must have
timezone-aware timestamps; Jerry normalizes them to UTC before downstream
processing.

A custom runtime operation receives exactly three positional arguments:

```python
from jerrythomas.config.tasks.base import PluginRuntimeTask
from jerrythomas.operations.persistence import (
    RoutedRuntimeOutput,
    RuntimeOutput,
    RuntimeOutputBatch,
)
from jerrythomas.runtime import Runtime


def run_report(
    runtime: Runtime,
    task: PluginRuntimeTask,
    limit: int | None,
) -> RuntimeOutput | RoutedRuntimeOutput | RuntimeOutputBatch | None: ...
```

`runtime` is the compiled `Runtime`, `task` is the configured
`PluginRuntimeTask`,
and `limit` is the CLI cap or `None`. Return `RuntimeOutput`,
`RoutedRuntimeOutput`, `RuntimeOutputBatch`, or `None`. Jerry persists the result
using the profile output. A routed output yields `(output_id, row)` pairs, where
each output ID is present in its `targets` mapping. Omit a row from the iterable
to drop it; an unknown output ID fails the operation. Dataset split routing,
preview, throttle, and `include_outputs` belong to the built-in dataset operation
and are not passed to plugins.

Preprocess and ordered transforms are validated built-in operations rather than
plugin entry points. Series shaping and postprocess policies are fixed pipeline
stages. See [Transforms](transforms/index.md) for their explicit configuration.

---
