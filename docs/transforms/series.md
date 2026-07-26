# Series Shaping

`dataset.yaml` exposes three distinct policies. `sequence` and `collect` shape
series values for sample assembly. `scale` marks assembled values for output
scaling with the selected dataset fold.

## Built-In Policies

- `sequence`: emit sliding windows as `SeriesSequence` payloads.
- `collect`: require a fixed number of ordered scalar values inside each
  populated `sample.cadence` bucket.
- `scale`: standardize a feature or target with the scaler fitted from the selected
  dataset fold's training labels.

```yaml
features:
  - id: temperature
    stream: weather.hourly
    field: temp_c
    scale: true

  - id: monthly_return
    stream: equity.monthly_returns
    field: return_1m
    sequence:
      size: 12
      stride: 1

  - id: intraday_return
    stream: equity.returns.hourly
    field: value
    collect: 24
```

`scale` is a boolean. It does not alter canonical series artifacts or
preview output. During a full dataset serve, every scalar or fixed-length list
value in one fold output is scaled with that fold's scaler. `with_mean`,
`with_std`, and `epsilon` are configured once on the scaler build operation and
recorded in the managed artifact; individual features cannot override them.
`None` is the canonical missing value. A transient floating `NaN` is converted
to `None` when the field is projected; other nonnumeric values and infinity are
rejected. Intrinsically list-valued fields are fitted independently by position.
Lists produced from scalar `sequence` or `collect` inputs apply the same scalar
series statistics to every position.

`sequence` accepts strictly positive integer `size` and optional `stride`
(default `1`). It creates independent windows per series ID and entity from
the source stream's record order. The stream's complete `partition_by` identity
keeps each series contiguous and sequence memory bounded to one window. Dataset
`sample.keys` select the partition fields represented in each row; remaining
partition fields are appended to the series ID. Cadence regularization belongs
in that stream's `transforms:` when it is required. Sequence inputs must be
scalar; nested list values are rejected rather than implicitly flattened.

`collect` is a strictly positive integer. Zero records leave the series absent
from a sample. Every populated bucket must contain exactly that many values;
`None` values and cadence placeholders count as positions.
Too few or too many values fail the series build. Values retain chronological
order, and collection is applied independently to each concrete partitioned
series ID and sample key.

Collection does not infer cadence or repair timestamps. Use `ensure_cadence` or
`ensure_schedule` upstream when positions must represent a regular grid. A single
source value that is already a list remains valid without `collect`, but
collecting list-valued records is rejected because it would create nested
vectors. `sequence` and `collect` are mutually exclusive. Without either,
emitting a second value for the same concrete series ID and sample bucket is an
error rather than an implicit schema change.

The `series` preview remains before sample assembly and therefore displays
individual collected inputs. The `samples` and `postprocess` previews, and full
dataset output, display the assembled list. A collected target's `horizon`
must conservatively cover the latest observation represented in its bucket;
Jerry does not infer that duration.
