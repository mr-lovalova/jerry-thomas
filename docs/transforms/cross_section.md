# Cross-Sectional Operations

Cross-sectional operations compare partitioned records at one exact timestamp.
They belong under `cross_section:` on a cross-sectional stream, not under the
ordinary `transforms:` list.

```yaml
id: equity.signal.neutralized
from:
  stream: equity.signal.inputs
cross_section:
  - operation: rank_score
    field: signal
    to: signal_rank
    min_samples: 30
  - operation: ols_residual
    y: signal_rank
    x: [liquidity_rank, volatility_rank]
    to: signal_residual
    min_samples: 30

# Optional ordered transforms run after canonical order is restored.
transforms:
  - { operation: lag, field: signal_residual, to: signal_residual_lag_1, periods: 1 }
```

The input must have a non-empty `partition_by`. Jerry groups records by exact
UTC timestamp, applies operations in YAML order, and then restores canonical
`[*partition_by, time]` order before ordinary transforms run. It never floors,
fills, or performs an as-of match implicitly. Duplicate partition keys at one
timestamp fail.

Use an upstream `where` transform to define the eligible population. Records
removed there do not contribute to the calculation and are absent from this
derived stream. When another selected series establishes those sample keys,
dataset assembly represents the missing derived values as `None`.

## `rank_score`

`rank_score` ranks complete numeric values in ascending order. Ties receive
their average rank. The score is normalized to `[-0.5, 0.5]`:

```text
(average_rank - 1) / (complete_count - 1) - 0.5
```

An all-equal eligible population receives `0.0`. `None` and `NaN` inputs remain
missing and do not enter the rank population. If fewer than `min_samples`
complete records remain, the operation writes `None` to every record at that
timestamp. `min_samples` is required and must be at least two.

## `ols_residual`

`ols_residual` fits one cross-sectional linear model with an intercept and
writes the residual for every complete row. `y` names the dependent field and
`x` contains one or more predictor fields. The implementation centers `y` and
every predictor before solving the least-squares system, which is equivalent
to fitting an intercept.

Rows containing `None` or `NaN` in `y` or any predictor are excluded from the
fit and receive `None`. If fewer than `min_samples` complete rows remain, every
row receives `None`. The design matrix must have full predictor rank;
rank-deficient input fails instead of silently choosing an unstable model.
`min_samples` must be at least `len(x) + 1`.

This operation requires the optional NumPy dependency:

```bash
python -m pip install 'jerry-thomas[numerical]'
```

## Split safety

Cross-sectional outputs depend on peer records at the same timestamp. Jerry
therefore rejects hash-split datasets whose selected stream closure includes a
cross-sectional stream. Time splits are valid because every record at one
timestamp belongs to the same temporal interval.
