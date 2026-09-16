# Ordered Transforms

Ordered transforms run after canonical ordering. Configure them under
`transforms:` in a stream file.

Transforms that depend on history operate within a partition. A source-backed
stream declares `partition_by` as the complete identity of an independent
series, such as `[security_id]` or `[security_id, metric]`. Derived, broadcast,
as-of, aligned, and cross-sectional streams inherit that identity. Dataset
`sample.keys` select which partition fields identify output rows. Remaining
partition fields suffix series IDs in their declared order.

Operations that compare different partitions at one timestamp use a dedicated
cross-sectional stream. See [Cross-sectional operations](cross_section.md).

## Field-Writing Transforms

Most single-field transforms accept `field` and optional `to`. If `to` is
omitted, the transform writes back to `field`. `log`, `log1p`, `forward_sum`,
and multi-input transforms such as `derive`, `rolling_slope`, and `rolling_ols`
require `to`. Transforms cannot write `time` or a resolved `partition_by` field
because those fields define canonical record order. Identity changes belong in
a map or combine function, before the ordering stage.

```yaml
transforms:
  - { operation: rolling, field: dollar_volume, to: adv20, window: 20, statistic: mean }
```

## Built-In Transforms

- `ensure_cadence`: insert placeholder records at a fixed duration within each
  partition.
- `ensure_schedule`: complete records against a resolved schedule artifact.
- `resample`: aggregate fixed-duration or calendar-month periods within each
  partition, publishing completed periods at their end. See below.
- `where`: filter ordered records using the record `where` operator language.
- `lag` / `lead`: copy a prior or future field value into `to` by `periods`
  within each partition.
- `forward_sum`: write the sum of exactly the next `window` partition records
  to the required `to` field. The current record is excluded. A complete window
  containing `None` or `NaN` produces `None`, as do the final `window` records;
  partial sums are never emitted. Nonnumeric or infinite values fail.
- `log`: write the natural logarithm of `field` to `to`; values must be greater
  than zero.
- `log1p`: write the natural logarithm of one plus `field` to `to`; values must
  be greater than `-1`. This operation preserves precision for returns near
  zero and is not implemented as `log(1 + value)`.

Both logarithms preserve `None` and `NaN` as missing and reject other
nonnumeric or infinite values.
- `derive`: write `to` from binary arithmetic on `left` and exactly one of
  `right_field` or `right_value`. Operators: `add`, `sub`, `mul`, `div`.
- `aggregate_sum`: emit one record for each adjacent canonical
  `(*partition_by, time)` key, replacing `field` with its sum. An optional
  `count_to` field receives the number of input records. Values must be finite
  integers or floats; missing values fail rather than being treated as zero.
  Integer-only groups retain exact integer sums. Groups containing floats sum
  every binary value exactly and round once at the end, so equivalent input
  ordering cannot change the summed value. Integer contributions that cannot be
  represented exactly as floats fail instead of losing precision. Other fields
  come from the first record in the group. Put `dedupe` first when exact
  duplicate events should not count twice.
- `collapse`: keep the `first` or `last` adjacent record for each partition and
  timestamp.
- `dedupe`: drop exact duplicate records from an already sorted stream.
- `fill`: impute missing values from rolling history using an explicit `mean`
  or `median` statistic.
- `fill_missing`: replace `None` or `NaN` in `field` with an explicit finite
  scalar `value`. Falsey values such as `0`, `false`, and empty strings remain
  unchanged. Use optional `to` to preserve the source field.
- `forward_fill`: carry the last known value within each partition.
- `ewm_mean`: compute an unadjusted exponentially weighted mean using the
  recursive update `mean = mean + alpha * (value - mean)`. The first valid
  observation initializes the mean. `alpha` must be greater than zero and no
  greater than one. `min_samples` counts valid observations and defaults to
  one. Missing observations do not change the state; after warmup they emit
  the current mean. Weighting is observation-based, not elapsed-time-based.
- `ewm_std`: compute an exponentially weighted standard deviation with the
  same `alpha`, `min_samples`, and missing-observation semantics as
  `ewm_mean`. After warmup, missing observations emit the current standard
  deviation. The first two valid observations are required before a value is
  produced because dispersion over fewer samples is undefined. Weighting is
  observation-based, not elapsed-time-based.
- `rolling`: compute `mean`, `median`, `stdev`, `pstdev`, `max`, `min`, or
  `sum` over
  a rolling window. Missing ticks occupy a window position but do not count
  toward `min_samples`, which defaults to `window`. Values must be finite;
  `None` and `NaN` are treated as missing.
- `rolling_quantile`: compute a quantile between `0` and `1` using linear
  interpolation at position `quantile * (n - 1)` in the ordered complete
  values. It uses the same window, partition, missing-value, and `min_samples`
  semantics as `rolling`.
- `rolling_slope`: compute the least-squares slope of `y` on `x` over a strict
  rolling window. `x`, `y`, `to`, and `window` are required, and `window` must
  be at least two. The current record is included. A missing pair clears the
  window (`None` and `NaN` are missing); nonnumeric or infinite inputs and zero
  `x` variance fail explicitly.
- `rolling_ols`: fit `y` on two or more ordered predictors in `x`, with an
  intercept, over a strict rolling window and write one selected `coefficient`
  to `to`. The window must contain at least one more record than the number of
  predictors. Missing values clear the window; nonnumeric, infinite, or
  rank-deficient windows fail explicitly. Install `jerry-thomas[numerical]`
  to use this NumPy-backed transform.

Sparse events can be aggregated, attached to a complete primary stream with an
exact optional as-of match, and then filled explicitly:

```yaml
# Sparse event stream
transforms:
  - { operation: aggregate_sum, field: signed_amount, count_to: event_count }

# Complete primary stream enriched by those events
from:
  stream: eligible_sessions
join:
  kind: as_of
  lookup: session_events
  max_age: 0s
  require_match: false
combine:
  entrypoint: attach_session_events
transforms:
  - { operation: fill_missing, field: signed_amount, value: 0 }
  - { operation: fill_missing, field: event_count, value: 0 }
```

The combiner receives `None` for an unmatched event record and must emit the
configured fields as missing. `fill_missing` then makes the absence policy
explicit without creating rows outside the primary stream.

`rolling_slope` needs consecutive records, not merely consecutive values. Put
`ensure_schedule` or `ensure_cadence` first when absent timestamps must reset the
window. To exclude the current record, lag both inputs explicitly:

```yaml
transforms:
  - { operation: lag, field: stock_return, periods: 1, to: stock_return_lag_1 }
  - { operation: lag, field: market_return, periods: 1, to: market_return_lag_1 }
  - { operation: rolling_slope, x: market_return_lag_1, y: stock_return_lag_1, window: 252, to: beta }
```

Use exact broadcast to attach global context fields before multivariate OLS.
An availability gap remains an explicit `lag` after the regression:

```yaml
# The stream already combines stock_return with exact-date global factors.
transforms:
  - operation: rolling_ols
    y: stock_return
    x: [spy_return, hyg_return, lqd_return]
    window: 252
    coefficient: hyg_return
    to: hyg_beta_raw
  - operation: lag
    field: hyg_beta_raw
    periods: 21
    to: hyg_beta
```

Both rolling regression operations count records. Use `ensure_schedule` first
when their windows and subsequent lags must count scheduled sessions. In the
example, the coefficient at time `t` comes from the regression window that
ended 21 scheduled records earlier.

Rolling quantiles use the same record-counted window and partition semantics:

```yaml
transforms:
  - operation: rolling_quantile
    field: return
    window: 252
    min_samples: 126
    quantile: 0.9
    to: return_q90
```

Exponentially weighted means use constant memory and have no fixed window:

```yaml
transforms:
  - { operation: ewm_mean, field: return, alpha: 0.1, min_samples: 20, to: smoothed_return }
```

`forward_sum` also counts records rather than inferred sessions. Use
`ensure_schedule` first for an explicit session schedule, or `ensure_cadence`
for a fixed-duration series:

```yaml
transforms:
  - { operation: ensure_schedule, schedule: schedule }
  - { operation: forward_sum, field: market_excess_return, window: 21, to: future_market_excess_return_21 }
```

When a future-derived field is selected as a dataset target, declare its
conservative wall-clock support under that target's `horizon`. The transform
describes how values are calculated; the target horizon separately controls
fold-boundary eligibility. Jerry deliberately does not guess one from the
other.

```yaml
transforms:
  - { operation: lag, field: close, to: close_lag_21, periods: 21 }
  - { operation: lag, field: close, to: close_lag_189, periods: 189 }
  - { operation: derive, left: close_lag_21, operator: div, right_field: close_lag_189, to: close_ratio }
  - { operation: derive, left: close_ratio, operator: sub, right_value: 1, to: momentum_189_21 }
```

```yaml
transforms:
  - { operation: ensure_schedule, schedule: schedule }
  - { operation: forward_fill, field: gross_margin }
```

## Resampling

Use the same operation for elapsed-time buckets and calendar months:

```yaml
transforms:
  - operation: resample
    period: { kind: calendar, unit: month, timezone: America/New_York }
    start: "2024-01-01T00:00:00-05:00"
    end: "2025-01-01T00:00:00-05:00"
    aggregations:
      close: { field: adjusted_close, statistic: last }
      volume: { field: volume, statistic: sum }
  - { operation: lag, field: close, periods: 1, to: previous_month_close }
```

For fixed durations, replace `period` with `{ kind: fixed, every: 1d }`.
Fixed buckets align to the UTC epoch. Calendar months start at local midnight
on the first, follow the named timezone's clock changes, and emit UTC timestamps.
A calendar month is not a fixed number of days or observations.

Buckets include their start and exclude their end. The output's `time` is the
bucket end: January's close becomes available at February 1, not January 1.
Each output contains only the partition keys and declared aggregate fields.
Supported statistics are `first`, `last`, `sum`, `mean`, `min`, `max`, and `count`.
`first` and `last` preserve missing endpoint values. Numeric statistics ignore
`None` and `NaN` but reject other nonnumeric or infinite values. `count` counts
nonmissing field values. Sums retain integer precision and use the same exact
floating-point accumulation rules as `aggregate_sum`.

Optional `start` and `end` declare input coverage, with `end` exclusive; records
outside those bounds are excluded. Only fully covered buckets are emitted.
Without `start`, coverage begins at the first record in each partition, so a
first record inside a month leaves that month incomplete. Without `end`, a later
record must reach the next boundary to complete a bucket; EOF alone does not
complete it. Declaring coverage asserts that the source covers that interval;
Jerry cannot detect omitted source observations.

Empty covered buckets emit null aggregates (`count: 0`), keeping consecutive
months visible to lags and rolling windows. They do not establish sample-domain
coverage. Bounds also allow leading and trailing empty buckets for observed
partitions; they do not create partitions absent from the input. Missing values
are never filled implicitly. Dataset cadence and target horizons remain fixed
durations; this operation changes the stream's observation frequency.

Dataset projection uses the required `sample.rounding` contract. With `ceil`, a
New York month ending at `05:00 UTC` enters the next midnight's daily sample;
`floor` labels it with the preceding midnight and `exact` rejects that off-grid
timestamp. Choose a convention consistent with the prediction cutoff. Projection
does not carry values forward; use an as-of join when daily observations should
reuse the latest completed month.
