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

[project.entry-points."jerrythomas.transforms"]
issuer_states = "my_datapipeline.transforms:IssuerStatesTransform"

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
from jerrythomas.operations.persistence import RuntimeOutput
from jerrythomas.runtime import Runtime


def run_report(
    runtime: Runtime,
    task: PluginRuntimeTask,
    limit: int | None,
) -> RuntimeOutput | None: ...
```

`runtime` is the compiled `Runtime`, `task` is the configured
`PluginRuntimeTask`,
and `limit` is the CLI cap or `None`. Return one `RuntimeOutput`, or `None` when
there is nothing to persist. The profile owns its output destination; runtime
results cannot select paths. Dataset split routing, preview, throttle, and
`include_outputs` belong to the built-in dataset operation and are not passed to
plugins.

Jerry 10 removes result-owned `target`/`targets` fields and custom
`RuntimeOutputBatch`/`RoutedRuntimeOutput` returns. Custom runtime operations
that used those Python APIs must return one `RuntimeOutput`; built-in dataset
fanout remains available through dataset profiles.

### Custom Stream Transforms

The `{operation: custom}` transform runs plugin code on one stream. The
entry point is a factory receiving `(args, partition_by)` and returning an
object with an `apply(records_iterator) -> records_iterator` method. Extend
`PartitionScopedTransform` from `jerrythomas.transforms.scoped` to receive
each partition's records in canonical order with state that resets between
partitions:

```python
from jerrythomas.transforms.scoped import PartitionScopedTransform


class IssuerStatesTransform(PartitionScopedTransform):
    def process_partition(self, records):
        state = None
        for record in records:
            ...
            yield enriched
```

```yaml
transforms:
  - operation: custom
    entrypoint: my_datapipeline.transforms:IssuerStatesTransform
    args: {window: 5d}
    writes: [ocf_ratio] # optional; declares outputs for validation
```

Custom transforms are stateful across time within a partition, so datasets
using hash splits reject streams that contain them. Declare every field the
transform may add via `writes`; Jerry rejects declarations that collide with
`time` or the stream's partition fields.

Preprocess and ordered built-in transforms remain validated core operations.
Series shaping and postprocess policies are fixed pipeline stages. See
[Transforms](transforms/index.md) for their explicit configuration.

---
