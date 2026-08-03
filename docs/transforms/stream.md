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
- `collapse`: keep the `first` or `last` adjacent record for each partition and
  timestamp.
- `dedupe`: drop exact duplicate records from an already sorted stream.
- `fill`: impute missing values from rolling history using an explicit `mean`
  or `median` statistic.
- `forward_fill`: carry the last known value within each partition.
- `rolling`: compute `mean`, `median`, `stdev`, `pstdev`, `max`, or `min` over
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
