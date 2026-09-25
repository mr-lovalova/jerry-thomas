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
"demo.csv_loader" = "my_datapipeline.loaders.csv:CsvLoader"

[project.entry-points."jerrythomas.parsers"]
"demo.weather_parser" = "my_datapipeline.parsers.weather:WeatherParser"

[project.entry-points."jerrythomas.mappers"]
"time.ticks" = "my_datapipeline.mappers.synthetic.ticks:map"

[project.entry-points."jerrythomas.combiners"]
air_density = "my_datapipeline.combiners.air_density:combine_air_density"

[project.entry-points."jerrythomas.transforms"]
previous_value = "my_datapipeline.transforms:PreviousValueTransform"

[project.entry-points."jerrythomas.operations.runtime"]
"demo.report" = "my_datapipeline.operations:run_report"
```

YAML `entrypoint` values name a registered entry in the corresponding group,
such as `previous_value`. The `module:object` target belongs in `pyproject.toml`.
Quote entry names containing dots so TOML treats each as one key.

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

Built-in operation YAML selects a `product`. Custom operations instead declare
`kind: runtime|artifact` and a registered `entrypoint`, without `product`.

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

Custom artifact operations use the `jerrythomas.operations.build` group. Their
configuration accepts `kind: artifact`, `entrypoint`, and `output`; the filename
supplies the operation ID. They do not accept `options` or `requires`. Their
cache hashes cover the bound dataset and complete stream catalog. Runtime plugins
can declare prerequisite artifacts with `requires` and plugin settings with
`options`; those capabilities do not extend to custom artifact operations.

### Custom Stream Transforms

The `{operation: custom}` transform runs plugin code on one stream. The
entry point is a factory receiving `(args, partition_by)` and returning an
object with an `apply(records_iterator) -> records_iterator` method. Extend
`PartitionScopedTransform` from `jerrythomas.transforms.scoped` to receive
each partition's records in canonical order with state that resets between
partitions:

```python
from jerrythomas.transforms.scoped import PartitionScopedTransform
from jerrythomas.transforms.utils import clone_record


class PreviousValueTransform(PartitionScopedTransform):
    def process_partition(self, records):
        previous = None
        for record in records:
            yield clone_record(record, previous_value=previous)
            previous = record.value
```

```yaml
transforms:
  - operation: custom
    entrypoint: previous_value
    args: {}
    writes: [previous_value] # declares the added field for validation
```

Use `clone_record(record, **updates)` to copy or enrich an existing record. It
preserves the record class, dynamically added fields, and the provenance that
distinguishes genuine observations from generated cadence placeholders. It is
a shallow copy: unchanged nested values remain shared. Returning an unchanged
input record is also valid. Cloning does not rerun a custom constructor or
subclass validation; a `time` update does run Jerry's UTC timestamp validation.

`dataclasses.replace()` drops dynamic fields and resets Jerry's private
provenance field; constructing a new domain object from an existing record also
loses that provenance. Neither is suitable for copying records inside a custom
transform. Domain classes with custom constructors remain supported by
`clone_record`; transform authors do not need to manipulate private metadata.
When correcting an existing plugin to use this copying contract, increment
`project.yaml:artifact_revision` and rebuild its artifacts.

Custom transforms are stateful across time within a partition, so datasets
using hash splits reject streams that contain them. Declare every field the
transform may add via `writes`; Jerry rejects declarations that collide with
`time` or the stream's partition fields.

Preprocess and ordered built-in transforms remain validated core operations.
Series shaping and postprocess policies are fixed pipeline stages. See
[Transforms](transforms/index.md) for their explicit configuration.

---
