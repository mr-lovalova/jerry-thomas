# Transforms

Transforms are split by the stage where they run:

- [Preprocess transforms](preprocess.md): one record at a time, before ordering.
- [Ordered transforms](stream.md): ordered record streams, usually with
  per-partition history.
- [Series shaping](series.md): feature/target payload shaping before vector
  assembly.
- [Postprocess policies](postprocess.md): sample filtering before split and
  output persistence.

Preprocess and ordered transforms are explicit built-in operations. Their config is
validated before pipeline execution; arbitrary transform entry points are not
loaded at runtime.

## Configuration Shape

Each transform is one flat mapping. `operation` identifies the
built-in operation and its configuration fields are siblings:

```yaml
transforms:
  - operation: dedupe
  - operation: rolling
    field: close
    to: close_mean_20
    window: 20
    statistic: mean
  - operation: rolling_slope
    x: market_return
    y: stock_return
    to: beta_252
    window: 252
  - operation: forward_sum
    field: excess_return
    to: future_excess_return_21
    window: 21
  - operation: log1p
    field: simple_return
    to: log_return
```

Missing or unknown operations, unknown fields, and invalid field values are
rejected while loading the project. Postprocess instead has a fixed structural
shape with separate feature and target sample filters. See
[Postprocess policies](postprocess.md) for its complete shape.
